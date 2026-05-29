"""Rigorous historical negative sampling benchmark on real LastFM data.

DyGLib's actual algorithm (from utils/utils.py historical_sample):
  1. np.logical_and mask over ALL edges to find edges before current_batch_start_time
  2. Python set construction of unique (src, dst) pairs from filtered arrays
  3. Subtract current batch edges
  4. Random sample from remaining

TGEngine algorithm:
  HistoricalNegPool built by reservoir sampling during training.
  At eval time: O(B) GPU gather from per-node pool.

Measurement:
  - Iterate through the EVALUATION batches in chronological order
  - At each batch, time both methods
  - Show how DyGLib cost grows as history size increases
  - Show TGEngine cost stays constant
"""

import time
import numpy as np
import pandas as pd
import torch

DATA_PATH = "/mnt/home/gyq/CodeBase/Graph/DG_Data/lastfm/ml_lastfm.csv"
FEAT_PATH = "/mnt/home/gyq/CodeBase/Graph/DG_Data/lastfm/ml_lastfm.npy"

BATCH_SIZE = 200
POOL_SIZE  = 512
DEVICE     = "cuda"


# ---------------------------------------------------------------------------
# DyGLib's actual historical_sample() — verbatim logic from utils/utils.py
# ---------------------------------------------------------------------------

class DyGLibHistoricalSampler:
    """Faithful reimplementation of DyGLib NegativeEdgeSampler(historical).

    Stores all edges as flat numpy arrays (exactly as DyGLib does).
    Per-batch cost: O(E) numpy boolean mask + O(|hist|) Python set construction.
    """

    def __init__(self, src_all, dst_all, ts_all):
        self.src = src_all          # full numpy array, all edges
        self.dst = dst_all
        self.ts  = ts_all
        self.earliest_time = float(ts_all.min())

    def _unique_edges_before(self, t_end):
        """numpy mask + Python set — DyGLib's exact approach."""
        mask = self.ts < t_end                          # numpy boolean, O(E)
        return set(zip(self.src[mask].tolist(),          # Python set, O(|result|)
                       self.dst[mask].tolist()))

    def sample(self, batch_src, batch_dst, t_start, size):
        """Sample historical negatives for one batch."""
        hist_edges = self._unique_edges_before(t_start)
        cur_edges  = set(zip(batch_src.tolist(), batch_dst.tolist()))
        candidates = list(hist_edges - cur_edges)

        if not candidates:
            return np.random.randint(0, int(self.dst.max()) + 1, size=size)

        idx = np.random.randint(0, len(candidates), size=size)
        return np.array([candidates[i][1] for i in idx])


# ---------------------------------------------------------------------------
# TGEngine historical sampler — HistoricalNegPool
# ---------------------------------------------------------------------------

def build_tge_pool(src_train, dst_train, num_nodes, pool_size, device):
    from tgengine.pipeline.negatives import HistoricalNegPool
    pool = HistoricalNegPool(num_nodes=num_nodes, pool_size=pool_size, device=device)
    dev = torch.device(device)
    chunk = 10_000
    n = len(src_train)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        pool.update(
            torch.from_numpy(src_train[start:end]).to(dev),
            torch.from_numpy(dst_train[start:end]).to(dev),
        )
    return pool


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*70}")
    print("Historical Negative Sampling — Real LastFM, Full History")
    print(f"  DyGLib: numpy mask O(E) + Python set  vs  TGE: O(B) GPU gather")
    print(f"{'='*70}")

    # Load full dataset
    print("\nLoading LastFM...")
    df   = pd.read_csv(DATA_PATH)
    src  = df["u"].values.astype(np.int32)
    dst  = df["i"].values.astype(np.int32)
    ts   = df["ts"].values.astype(np.float64)
    n    = len(src)
    num_nodes = max(src.max(), dst.max()) + 1

    # Chronological split: 70% train, 10% val, 20% test
    n_train = int(n * 0.70)
    n_val   = int(n * 0.80)
    print(f"  Total: {n:,}  Train: {n_train:,}  Val: {n_val-n_train:,}  "
          f"Test: {n-n_val:,}  Nodes: {num_nodes}")

    src_train = src[:n_train]
    dst_train = dst[:n_train]

    # DyGLib sampler holds ALL edges (train+val for test eval)
    dygl = DyGLibHistoricalSampler(src[:n_val], dst[:n_val], ts[:n_val])
    print(f"  DyGLib stores {len(dygl.src):,} edges in numpy arrays in RAM")

    # TGEngine pool built only from training edges
    print(f"\nBuilding TGEngine reservoir pool (pool_size={POOL_SIZE})...")
    pool = build_tge_pool(src_train, dst_train, num_nodes, POOL_SIZE, DEVICE)
    dev  = torch.device(DEVICE)
    valid = (pool._pool != pool.PADDING).float().sum(dim=1)
    print(f"  Pool coverage: avg {valid.mean().item():.1f}/{POOL_SIZE} slots per node")

    # Eval batches: iterate over test set in chronological order
    test_src = src[n_val:]
    test_dst = dst[n_val:]
    test_ts  = ts[n_val:]
    n_test   = len(test_src)
    n_batches = n_test // BATCH_SIZE
    print(f"\nEval batches: {n_batches} batches of {BATCH_SIZE} from test set")

    # We benchmark at 5 checkpoints spread across the test set
    check_points = [int(n_batches * frac) for frac in [0.1, 0.3, 0.5, 0.7, 0.9]]
    # Measure 10 consecutive batches around each checkpoint and average
    MEAS_WINDOW = 10

    print(f"\n{'Batch':>8}  {'History size':>14}  {'DyGLib (ms)':>13}  "
          f"{'TGE (ms)':>10}  {'Speedup':>9}")
    print("-" * 62)

    results = []
    for cp in check_points:
        i = cp
        # DyGLib: measure MEAS_WINDOW consecutive batches
        dygl_times = []
        tge_times  = []

        for j in range(MEAS_WINDOW):
            bi = i + j
            if bi >= n_batches:
                break
            b_src = test_src[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
            b_dst = test_dst[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
            b_t   = float(test_ts[bi*BATCH_SIZE])

            # DyGLib timing
            t0 = time.perf_counter()
            dygl.sample(b_src, b_dst, b_t, BATCH_SIZE)
            dygl_times.append((time.perf_counter() - t0) * 1000)

            # TGE timing
            b_src_gpu = torch.from_numpy(b_src).to(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pool.sample(b_src_gpu)
            torch.cuda.synchronize()
            tge_times.append((time.perf_counter() - t0) * 1000)

        t_dygl = np.mean(dygl_times)
        t_tge  = np.mean(tge_times)
        # History size at this checkpoint = edges before this batch's timestamp
        t_batch = float(test_ts[i*BATCH_SIZE])
        hist_size = (ts < t_batch).sum()
        speedup = t_dygl / t_tge

        print(f"{i:>8}  {hist_size:>14,}  {t_dygl:>13.2f}  "
              f"{t_tge:>10.3f}  {speedup:>8.1f}x")
        results.append((i, hist_size, t_dygl, t_tge, speedup))

    print(f"\n{'='*70}")
    print("Summary:")
    min_speedup = min(r[4] for r in results)
    max_speedup = max(r[4] for r in results)
    avg_speedup = np.mean([r[4] for r in results])
    avg_dygl    = np.mean([r[2] for r in results])
    avg_tge     = np.mean([r[3] for r in results])
    print(f"  DyGLib avg: {avg_dygl:.1f}ms/batch  (grows with history size)")
    print(f"  TGE avg:    {avg_tge:.3f}ms/batch  (constant)")
    print(f"  Speedup: min={min_speedup:.0f}x  max={max_speedup:.0f}x  avg={avg_speedup:.0f}x")
    print(f"\nNote: DyGLib scans numpy array of {len(dygl.src):,} edges at every batch.")
    print(f"      TGE queries GPU pool of {num_nodes}x{POOL_SIZE} bounded entries.")


if __name__ == "__main__":
    run()
