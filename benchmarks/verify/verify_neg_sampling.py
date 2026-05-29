"""Cross-validate TGEngine vs DyGLib negative sampling output (exact match).

Requires DyGLib source at /mnt/home/gyq/CodeBase/Graph/DG_Data/../exp_sourcecode
(or adjust DYGLIB_PATH). Run on scnu only.
"""
import sys
import numpy as np
import torch

DYGLIB_PATH = "/mnt/home/gyq/CodeBase/Graph/DyGLib_ref"
sys.path.insert(0, DYGLIB_PATH)

from utils.utils import NegativeEdgeSampler

sys.path.insert(0, ".")
from tgengine.pipeline.negatives import DyGLibHistoricalNegative, DyGLibInductiveNegative


def make_data():
    rng = np.random.RandomState(123)
    n = 50
    src = rng.randint(0, 10, n).astype(np.int64)
    dst = rng.randint(0, 10, n).astype(np.int64)
    times = np.sort(rng.uniform(0, 100, n))
    return src, dst, times


def test_historical():
    src, dst, times = make_data()
    seed = 42

    tge = DyGLibHistoricalNegative(src, dst, times, seed=seed)
    dyg = NegativeEdgeSampler(src, dst, times, negative_sample_strategy="historical", seed=seed)

    # Test on several batches in the second half of the data
    for batch_start in range(30, 45, 4):
        batch_end = min(batch_start + 4, len(src))
        b_src = src[batch_start:batch_end]
        b_dst = dst[batch_start:batch_end]
        b_times = times[batch_start:batch_end]
        size = len(b_src)

        dyg.reset_random_state()
        _, dyg_neg_dst = dyg.sample(
            size=size, batch_src_node_ids=b_src, batch_dst_node_ids=b_dst,
            current_batch_start_time=float(b_times[0]),
            current_batch_end_time=float(b_times[-1]),
        )

        tge._rng = np.random.RandomState(seed)
        tge_neg_dst = tge.sample(
            src=torch.from_numpy(b_src), dst=torch.from_numpy(b_dst),
            time=torch.from_numpy(b_times).float(), graph=None,
        ).numpy()

        print(f"Batch {batch_start}: DyGLib={dyg_neg_dst}, TGEngine={tge_neg_dst}")
        assert np.array_equal(tge_neg_dst, dyg_neg_dst), \
            f"Historical mismatch at batch {batch_start}"

    print("Historical: ALL MATCHED")


def test_inductive():
    src, dst, times = make_data()
    seed = 42
    last_observed_time = float(times[34])  # training ends at edge 34

    tge = DyGLibInductiveNegative(src, dst, times, last_observed_time=last_observed_time, seed=seed)
    dyg = NegativeEdgeSampler(src, dst, times, last_observed_time=last_observed_time,
                              negative_sample_strategy="inductive", seed=seed)

    for batch_start in range(35, 48, 4):
        batch_end = min(batch_start + 4, len(src))
        b_src = src[batch_start:batch_end]
        b_dst = dst[batch_start:batch_end]
        b_times = times[batch_start:batch_end]
        size = len(b_src)

        dyg.reset_random_state()
        _, dyg_neg_dst = dyg.sample(
            size=size, batch_src_node_ids=b_src, batch_dst_node_ids=b_dst,
            current_batch_start_time=float(b_times[0]),
            current_batch_end_time=float(b_times[-1]),
        )

        tge._rng = np.random.RandomState(seed)
        tge_neg_dst = tge.sample(
            src=torch.from_numpy(b_src), dst=torch.from_numpy(b_dst),
            time=torch.from_numpy(b_times).float(), graph=None,
        ).numpy()

        print(f"Batch {batch_start}: DyGLib={dyg_neg_dst}, TGEngine={tge_neg_dst}")
        assert np.array_equal(tge_neg_dst, dyg_neg_dst), \
            f"Inductive mismatch at batch {batch_start}"

    print("Inductive: ALL MATCHED")


if __name__ == "__main__":
    test_historical()
    print()
    test_inductive()
    print("\nAll cross-validation passed!")
