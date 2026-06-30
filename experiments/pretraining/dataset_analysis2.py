"""Analyze temporal signals for cross-domain pretraining.

Key question: is "past interaction predicts future" a domain-agnostic signal?
"""
import sys, os
import numpy as np
import torch
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.expanduser("~/CodeBase/Graph/tgengine"))
from tgengine.core.dataset import load_dataset

DATASETS = ["uci", "wikipedia", "reddit", "lastfm", "enron", "mooc", "BitcoinAlpha"]
DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


def analyze_temporal(name):
    ds = load_dataset(name, DATA_ROOT)
    src, dst, t = ds.src.numpy(), ds.dst.numpy(), ds.time.numpy()
    train_end = ds.train_end

    # Signal 1: repeated interaction (edge appeared in train → appears in val/test?)
    train_edges = set()
    for s, d in zip(src[:train_end], dst[:train_end]):
        train_edges.add((int(s), int(d)))

    val_pos_seen = sum(1 for s, d in zip(src[train_end:], dst[train_end:]) if (int(s), int(d)) in train_edges)
    val_total = len(src) - train_end
    repeat_rate = val_pos_seen / val_total

    # AUC of "seen in train" as binary classifier
    np.random.seed(42)
    n_val = len(src) - train_end
    sample = min(5000, n_val)
    idx = np.random.choice(n_val, sample, replace=False)
    pos_seen = np.array([1.0 if (int(src[train_end+i]), int(dst[train_end+i])) in train_edges else 0.0 for i in idx])
    # Random negatives
    neg_seen = np.array([1.0 if (int(src[train_end+i]), int(np.random.choice(dst[train_end:]))) in train_edges else 0.0 for i in idx])
    labels = np.concatenate([np.ones(sample), np.zeros(sample)])
    scores = np.concatenate([pos_seen, neg_seen])
    try:
        repeat_auc = roc_auc_score(labels, scores)
    except:
        repeat_auc = 0.5

    # Signal 2: src-dst interaction count (how many times interacted in train)
    edge_count = Counter()
    for s, d in zip(src[:train_end], dst[:train_end]):
        edge_count[(int(s), int(d))] += 1
    pos_counts = np.array([edge_count.get((int(src[train_end+i]), int(dst[train_end+i])), 0) for i in idx])
    neg_counts = np.array([edge_count.get((int(src[train_end+i]), int(np.random.choice(dst[train_end:]))), 0) for i in idx])
    try:
        count_auc = roc_auc_score(labels, np.concatenate([pos_counts, neg_counts]))
    except:
        count_auc = 0.5

    # Signal 3: src activity (degree in train)
    src_deg = Counter(src[:train_end].tolist())
    pos_src_deg = np.array([src_deg.get(int(src[train_end+i]), 0) for i in idx])
    neg_src_deg = np.array([src_deg.get(int(src[train_end+i]), 0) for i in idx])
    try:
        srcdeg_auc = roc_auc_score(labels, np.concatenate([pos_src_deg, neg_src_deg]))
    except:
        srcdeg_auc = 0.5

    # Signal 4: dst popularity (degree in train)
    dst_deg = Counter(dst[:train_end].tolist())
    pos_dst_deg = np.array([dst_deg.get(int(dst[train_end+i]), 0) for i in idx])
    neg_dst_deg = np.array([dst_deg.get(int(np.random.choice(dst[train_end:])), 0) for i in idx])
    try:
        dstdeg_auc = roc_auc_score(labels, np.concatenate([pos_dst_deg, neg_dst_deg]))
    except:
        dstdeg_auc = 0.5

    # Signal 5: recency (time since last interaction)
    last_time = {}
    train_t = t[:train_end]
    for i in range(train_end):
        last_time[(int(src[i]), int(dst[i]))] = train_t[i]
    val_t = t[train_end:]
    pos_recency = np.array([val_t[idx[i]] - last_time.get((int(src[train_end+idx[i]]), int(dst[train_end+idx[i]])), val_t[idx[i]]) for i in range(sample)])
    neg_recency = np.array([val_t[idx[i]] - last_time.get((int(src[train_end+idx[i]]), int(np.random.choice(dst[train_end:]))), val_t[idx[i]]) for i in range(sample)])
    # More recent (smaller gap) → more likely positive, so negate
    try:
        recency_auc = roc_auc_score(labels, -np.concatenate([pos_recency, neg_recency]))
    except:
        recency_auc = 0.5

    print(f"\n{'='*60}")
    print(f"  {name}  ({ds.num_nodes:,} nodes, {ds.num_edges:,} edges)")
    print(f"{'='*60}")
    print(f"  Repeated edge in val:    {repeat_rate*100:>6.1f}%  (edge seen in train)")
    print(f"  --- Temporal signal AUC (pos vs random neg) ---")
    print(f"  Repeated (binary):       {repeat_auc:.4f}")
    print(f"  Interaction count:       {count_auc:.4f}")
    print(f"  Src degree:              {srcdeg_auc:.4f}")
    print(f"  Dst degree:              {dstdeg_auc:.4f}")
    print(f"  Recency (last Δt):       {recency_auc:.4f}")

    return {"name": name, "repeat_auc": repeat_auc, "count_auc": count_auc,
            "srcdeg_auc": srcdeg_auc, "dstdeg_auc": dstdeg_auc, "recency_auc": recency_auc}


if __name__ == "__main__":
    results = []
    for name in DATASETS:
        try:
            results.append(analyze_temporal(name))
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")

    print(f"\n\n{'='*70}")
    print("  CROSS-DOMAIN TEMPORAL SIGNAL AUC SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Repeat':>8} {'Count':>8} {'SrcDeg':>8} {'DstDeg':>8} {'Recency':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<15} {r['repeat_auc']:>8.4f} {r['count_auc']:>8.4f} {r['srcdeg_auc']:>8.4f} {r['dstdeg_auc']:>8.4f} {r['recency_auc']:>8.4f}")

    # Check consistency
    print(f"\n  Signal consistency across datasets (std):")
    for sig in ["repeat_auc", "count_auc", "srcdeg_auc", "dstdeg_auc", "recency_auc"]:
        vals = [r[sig] for r in results]
        print(f"    {sig:<15}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
