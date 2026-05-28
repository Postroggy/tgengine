"""Plot batch size sweep results as 4-panel figure.

Usage:
    python benchmarks/bench_bs_sweep_plot.py bench_bs_sweep.json
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot(results: list[dict], out_path: str = "bench_bs_sweep.png"):
    valid = [r for r in results if not r.get("oom")]
    if not valid:
        print("No valid results to plot")
        return

    bs = np.array([r["batch_size"] for r in valid])
    epoch_s = np.array([r["avg_epoch_sec"] for r in valid])
    total_s = np.array([r["total_wall_sec"] for r in valid])
    ap = np.array([r["best_test_ap"] for r in valid])
    gpu = np.array([r["avg_gpu_util_pct"] for r in valid])
    e_per_s = np.array([r["throughput_edges_per_sec"] for r in valid])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UCI — DyGFormer Batch Size Sweep (max 50 ep, early stop pat=5)", fontsize=13)

    # Panel 1: Epoch time
    ax = axes[0, 0]
    ax.plot(bs, epoch_s, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Avg Epoch Time (s)")
    ax.set_title("Epoch Time vs Batch Size")
    ax.grid(True, alpha=0.3)

    # Panel 2: Training time
    ax = axes[0, 1]
    ax.plot(bs, total_s, "o-", color="firebrick", markersize=4)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Total Training Time vs Batch Size")
    ax.grid(True, alpha=0.3)

    # Panel 3: Test AP
    ax = axes[1, 0]
    ax.plot(bs, ap, "o-", color="forestgreen", markersize=4)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Test AP")
    ax.set_title("Accuracy vs Batch Size")
    ax.grid(True, alpha=0.3)
    # Mark best
    best_idx = np.argmax(ap)
    ax.axvline(bs[best_idx], color="forestgreen", linestyle="--", alpha=0.5)
    ax.annotate(f"BS={bs[best_idx]}\nAP={ap[best_idx]:.4f}",
                xy=(bs[best_idx], ap[best_idx]),
                xytext=(bs[best_idx] + 500, ap[best_idx] - 0.01),
                fontsize=8, color="forestgreen")

    # Panel 4: GPU utilization
    ax = axes[1, 1]
    ax.plot(bs, gpu, "o-", color="darkorange", markersize=4)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Avg GPU Utilization (%)")
    ax.set_title("GPU Utilization (SM active %) vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'BS':>8} {'Ep':>4} {'TotWall':>8} {'Epoch':>8} {'e/s':>9} {'AP':>8} {'GPU%':>7}")
    print("-" * 58)
    for r in valid:
        print(f"{r['batch_size']:>8} {r['n_epochs']:>4} {r['total_wall_sec']:>7.1f}s "
              f"{r['avg_epoch_sec']:>7.1f}s {r['throughput_edges_per_sec']:>9,.0f} "
              f"{r['best_test_ap']:>8.4f} {r['avg_gpu_util_pct']:>6.1f}%")

    best = max(valid, key=lambda r: r["best_test_ap"])
    best_gpu = max(valid, key=lambda r: r["avg_gpu_util_pct"])
    print(f"\nBest AP:    BS={best['batch_size']} → AP={best['best_test_ap']:.4f}")
    print(f"Best GPU%:  BS={best_gpu['batch_size']} → GPU={best_gpu['avg_gpu_util_pct']:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bench_bs_sweep_plot.py <results.json> [output.png]")
        sys.exit(1)
    data = load(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", ".png")
    plot(data, out)
