"""Input-output equivalence check: DyGLib vs TGEngine historical negative sampling.

For two algorithms to be semantically equivalent, given the same input (same src nodes,
same historical edges, same current batch), they must produce samples from the
same distribution. We check:

  1. Support equivalence: does TGE ever sample a dst that DyGLib would NEVER produce?
     (i.e., a dst not in src's history) — these are INVALID negatives
  2. Coverage: what fraction of DyGLib's reachable set is covered by TGE's pool?
     (uncovered items can never be sampled by TGE even in infinite trials)
  3. Current-batch exclusion: DyGLib explicitly removes current batch edges.
     Does TGE?
  4. Distribution uniformity: within the covered support, are both uniform?
"""
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

DATA_PATH = "/mnt/home/gyq/CodeBase/Graph/DG_Data/lastfm/ml_lastfm.csv"
POOL_SIZE  = 512
DEVICE     = "cuda"
BATCH_SIZE = 200
N_TRIALS   = 2000   # sample each src this many times to estimate distribution


# ---- Load data ----
df  = pd.read_csv(DATA_PATH)
src = df["u"].values.astype(np.int32)
dst = df["i"].values.astype(np.int32)
ts  = df["ts"].values.astype(np.float64)
n   = len(src)
num_nodes = max(src.max(), dst.max()) + 1

n_train = int(n * 0.70)
n_val   = int(n * 0.80)

# ---- Build ground truth: per-node full history ----
# "historical dst for node s at eval time" = all dsts that appeared as
# s's dst in train+val edges (what DyGLib's array contains)
full_history = defaultdict(set)
for s, d in zip(src[:n_val], dst[:n_val]):
    full_history[s].add(d)

# ---- Build TGE pool from training data only ----
print("Building TGE pool from training data...")
from tgengine.pipeline.negatives import HistoricalNegPool
pool = HistoricalNegPool(num_nodes=num_nodes, pool_size=POOL_SIZE, device=DEVICE)
dev  = torch.device(DEVICE)
chunk = 10_000
for start in range(0, n_train, chunk):
    end = min(start + chunk, n_train)
    pool.update(
        torch.from_numpy(src[start:end]).to(dev),
        torch.from_numpy(dst[start:end]).to(dev),
    )

# ---- DyGLib sampler ----
class DyGLibSampler:
    def __init__(self, src_arr, dst_arr, ts_arr):
        self.src = src_arr
        self.dst = dst_arr
        self.ts  = ts_arr

    def sample_once(self, batch_src, batch_dst, t_start):
        mask      = self.ts < t_start
        hist_set  = set(zip(self.src[mask].tolist(), self.dst[mask].tolist()))
        cur_set   = set(zip(batch_src.tolist(), batch_dst.tolist()))
        candidates = list(hist_set - cur_set)
        if not candidates:
            return None
        idx = np.random.randint(0, len(candidates), size=len(batch_src))
        return np.array([candidates[i][1] for i in idx])

dygl = DyGLibSampler(src[:n_val], dst[:n_val], ts[:n_val])

# ---- Pick one representative eval batch ----
# Use batch near 50% of test set
mid_test = n_val + (n - n_val) // 2
b_src = src[mid_test:mid_test + BATCH_SIZE]
b_dst = dst[mid_test:mid_test + BATCH_SIZE]
t_start = float(ts[mid_test])

# Unique src nodes in this batch (to analyze per-src)
unique_srcs = list(set(b_src.tolist()))
print(f"\nEval batch at t={t_start:.0f}: {BATCH_SIZE} edges, "
      f"{len(unique_srcs)} unique src nodes")

# ---- Check 1: TGE pool support validity ----
# For each src in pool, are all pool entries valid historical dsts?
print("\n[1] Support validity: does TGE ever sample dsts outside src's true history?")
invalid_count = 0
total_pool_entries = 0
for s in unique_srcs:
    pool_row = pool._pool[s]
    valid_mask = pool_row != pool.PADDING
    pool_dsts = set(pool_row[valid_mask].tolist())
    true_dsts = full_history.get(s, set())
    invalid = pool_dsts - true_dsts
    invalid_count += len(invalid)
    total_pool_entries += len(pool_dsts)
    if invalid:
        print(f"  src={s}: {len(invalid)} INVALID pool entries "
              f"(not in full_history). Examples: {list(invalid)[:3]}")

if invalid_count == 0:
    print(f"  PASS — all {total_pool_entries} pool entries for batch srcs "
          f"are valid historical dsts")
else:
    print(f"  FAIL — {invalid_count}/{total_pool_entries} pool entries "
          f"are NOT in src's history")

# ---- Check 2: Coverage — what fraction of DyGLib's reachable set is in pool? ----
print("\n[2] Coverage: fraction of DyGLib's reachable set covered by TGE pool")
# DyGLib's reachable set for a src = full_history[src] - current_batch_dsts_for_src
covered_total = 0
reachable_total = 0
per_src_recall = []

for s in unique_srcs:
    # DyGLib can sample from: historical dsts that are not (s, current_batch_dst)
    batch_dsts_for_s = set(d for bs, bd in zip(b_src, b_dst) if bs == s for d in [bd])
    dygl_reachable = full_history.get(s, set()) - batch_dsts_for_s

    # TGE pool entries for this src
    pool_row = pool._pool[s]
    pool_dsts = set(pool_row[pool_row != pool.PADDING].tolist())

    covered = len(pool_dsts & dygl_reachable)
    reachable = len(dygl_reachable)
    if reachable > 0:
        per_src_recall.append(covered / reachable)
        covered_total  += covered
        reachable_total += reachable

per_src_recall = np.array(per_src_recall)
overall_coverage = covered_total / reachable_total if reachable_total > 0 else 0
print(f"  Overall: {covered_total}/{reachable_total} = {100*overall_coverage:.1f}% covered")
print(f"  Per-src recall: mean={per_src_recall.mean():.3f}, "
      f"median={np.median(per_src_recall):.3f}, "
      f"min={per_src_recall.min():.3f}, "
      f"nodes with <50% coverage: {(per_src_recall < 0.5).sum()}")

# ---- Check 3: Current-batch exclusion ----
print("\n[3] Current-batch exclusion: does TGE return positive edges (src->dst in batch)?")
b_src_gpu = torch.from_numpy(b_src).to(dev)
batch_pos_pairs = set(zip(b_src.tolist(), b_dst.tolist()))

tge_violations = 0
n_check = 500
for _ in range(n_check):
    neg = pool.sample(b_src_gpu).cpu().numpy()
    for s, n_dst in zip(b_src.tolist(), neg.tolist()):
        if (s, n_dst) in batch_pos_pairs:
            tge_violations += 1

total_samples = n_check * BATCH_SIZE
print(f"  Sampled {total_samples:,} negatives, found {tge_violations} "
      f"that match a current batch positive edge")
print(f"  Violation rate: {tge_violations/total_samples:.5f} "
      f"({'NONE' if tge_violations==0 else 'EXISTS'})")

# ---- Check 4: Distribution uniformity comparison ----
print(f"\n[4] Distribution comparison (N={N_TRIALS} samples per src)")
# Pick the src node with most history for a meaningful comparison
analysis_src = max(unique_srcs, key=lambda s: len(full_history.get(s, set())))
true_hist = full_history.get(analysis_src, set())
print(f"  Analysis node: src={analysis_src}, "
      f"|true_history|={len(true_hist)}, "
      f"|pool|={(pool._pool[analysis_src] != pool.PADDING).sum().item()}")

# DyGLib distribution (N_TRIALS samples)
dygl_samples = []
single_src = np.array([analysis_src])
single_dst = np.array([0])   # dummy
for _ in range(N_TRIALS):
    r = dygl.sample_once(single_src, single_dst, t_start)
    if r is not None:
        dygl_samples.append(int(r[0]))
dygl_counts = defaultdict(int)
for v in dygl_samples: dygl_counts[v] += 1

# TGE distribution (N_TRIALS samples)
query = torch.tensor([analysis_src], device=dev)
tge_samples = [pool.sample(query).item() for _ in range(N_TRIALS)]
tge_counts = defaultdict(int)
for v in tge_samples: tge_counts[v] += 1

dygl_support = set(dygl_counts.keys())
tge_support  = set(tge_counts.keys())

print(f"  DyGLib sampled {len(dygl_support)} unique dsts in {N_TRIALS} trials")
print(f"  TGE    sampled {len(tge_support)} unique dsts in {N_TRIALS} trials")
print(f"  Overlap: {len(dygl_support & tge_support)} dsts appear in both")
print(f"  DyGLib-only: {len(dygl_support - tge_support)} dsts "
      f"(TGE can never reach these)")
print(f"  TGE-only: {len(tge_support - dygl_support)} dsts "
      f"(TGE samples these but DyGLib never did in {N_TRIALS} trials)")

# ---- Summary ----
print(f"\n{'='*60}")
print("Semantic equivalence verdict:")
verdict_valid = (invalid_count == 0)
verdict_coverage = overall_coverage
verdict_exclusion = (tge_violations == 0)
print(f"  [{'PASS' if verdict_valid else 'FAIL'}] TGE pool entries are valid "
      f"historical dsts (no hallucinated negatives)")
print(f"  [INFO] TGE covers {100*verdict_coverage:.1f}% of DyGLib's reachable set")
print(f"  [{'PASS' if verdict_exclusion else 'WARN'}] TGE batch-positive exclusion: "
      f"{'no violations' if verdict_exclusion else f'{tge_violations} violations'}")
dygl_only_frac = len(dygl_support - tge_support) / max(len(dygl_support), 1)
print(f"  [INFO] {100*dygl_only_frac:.1f}% of DyGLib's sampled dsts are "
      f"unreachable by TGE (pool coverage gap)")
print(f"\nConclusion: algorithms are {'EQUIVALENT' if overall_coverage > 0.99 else 'APPROXIMATE'}.")
if overall_coverage < 1.0:
    print(f"  TGE is a {100*overall_coverage:.1f}%-coverage approximation of DyGLib's "
          f"full-history semantics.\n  The approximation quality depends on pool_size "
          f"relative to node degree.")
