"""Cross-domain CTDG dataset analysis for foundation model pretraining design.

Key questions:
1. How heterogeneous are our datasets (features, structure, scale)?
2. Is co-occurrence a domain-agnostic signal? (DyGFormer's assumption)
3. What structural features transfer across domains?
"""
import sys, os
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/CodeBase/Graph/tgengine"))
from tgengine.core.dataset import load_dataset

DATASETS = [
    "uci", "wikipedia", "reddit", "lastfm",
    "enron", "mooc", "BitcoinAlpha", "tgbl_wiki",
]
DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"


def analyze_cooccurrence(src, dst, time, train_end):
    """Analyze co-occurrence as a link prediction signal.

    For val/test edges (src,dst), count how many common neighbors src and dst
    share in the training graph. Higher co-occurrence → more likely positive.
    """
    # Build adjacency from training edges only
    train_src = src[:train_end].numpy()
    train_dst = dst[:train_end].numpy()

    # Per-node neighbor sets (sample if too large)
    nbrs = defaultdict(set)
    for s, d in zip(train_src, train_dst):
        nbrs[s].add(d)
        nbrs[d].add(s)

    # Sample val/test edges for co-occurrence analysis
    val_src = src[train_end:].numpy()
    val_dst = dst[train_end:].numpy()
    n_val = len(val_src)
    sample_size = min(5000, n_val)
    idx = np.random.RandomState(42).choice(n_val, sample_size, replace=False)

    # Positive co-occurrence distribution
    pos_cooc = []
    for i in idx:
        s, d = int(val_src[i]), int(val_dst[i])
        ns, nd = nbrs.get(s, set()), nbrs.get(d, set())
        pos_cooc.append(len(ns & nd))

    # Negative: random src-dst pairs from val
    neg_cooc = []
    for i in idx:
        s = int(val_src[i])
        d = int(np.random.choice(val_dst))
        ns, nd = nbrs.get(s, set()), nbrs.get(d, set())
        neg_cooc.append(len(ns & nd))

    pos_cooc = np.array(pos_cooc)
    neg_cooc = np.array(neg_cooc)
    return pos_cooc, neg_cooc


def analyze(name):
    ds = load_dataset(name, DATA_ROOT)
    src, dst, t = ds.src, ds.dst, ds.time
    n, m = ds.num_nodes, ds.num_edges
    d_edge = ds.edge_feat_dim
    d_node = ds.node_feat_dim

    # Degree distribution
    all_nodes = torch.cat([src, dst])
    degree = torch.bincount(all_nodes, minlength=n)
    deg_np = degree.numpy()

    # Repeated interactions (same (src,dst) pair)
    edges_set = defaultdict(int)
    for s, d in zip(src.numpy()[:ds.train_end], dst.numpy()[:ds.train_end]):
        edges_set[(int(s), int(d))] += 1
    repeat_counts = np.array(list(edges_set.values()))
    repeat_frac = (repeat_counts > 1).mean()

    # Reciprocal edges (A->B and B->A both exist)
    edge_pairs = set((int(s), int(d)) for s, d in zip(src.numpy()[:ds.train_end], dst.numpy()[:ds.train_end]))
    recip = sum(1 for (s, d) in edge_pairs if (d, s) in edge_pairs)
    recip_frac = recip / max(len(edge_pairs), 1)

    # Co-occurrence analysis
    pos_cooc, neg_cooc = analyze_cooccurrence(src, dst, t, ds.train_end)

    # Co-occurrence discriminative power
    pos_mean = pos_cooc.mean()
    neg_mean = neg_cooc.mean()
    # AUC of co-occurrence as binary classifier
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(pos_cooc)), np.zeros(len(neg_cooc))])
    scores = np.concatenate([pos_cooc, neg_cooc])
    try:
        cooc_auc = roc_auc_score(labels, scores)
    except:
        cooc_auc = 0.5

    # Temporal patterns
    t_np = t.numpy()
    t_span = t_np[-1] - t_np[0]
    dt = np.diff(np.sort(t_np[:ds.train_end]))
    dt = dt[dt > 0]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Nodes:        {n:>10,}")
    print(f"  Edges:        {m:>10,}  (train={ds.train_size:,} val={ds.val_size:,} test={ds.test_size:,})")
    print(f"  d_edge:       {d_edge:>10}   d_node: {d_node}")
    print(f"  Time span:    {t_span:>10.1f}")
    print(f"  Avg degree:   {deg_np.mean():>10.1f}   median: {np.median(deg_np):.0f}")
    print(f"  Repeated %:   {repeat_frac*100:>10.1f}  (same src-dst pair seen >1x)")
    print(f"  Reciprocal %: {recip_frac*100:>10.1f}  (A→B and B→A both exist)")
    print(f"  Δt median:    {np.median(dt):>10.1f}  (inter-event gap)")
    print(f"  --- Co-occurrence (shared neighbors) ---")
    print(f"  Pos mean:     {pos_mean:>10.2f}   Neg mean: {neg_mean:.2f}")
    print(f"  Co-oc AUC:    {cooc_auc:>10.4f}   (discriminative power)")

    return {
        "name": name, "n": n, "m": m, "d_edge": d_edge, "d_node": d_node,
        "avg_deg": deg_np.mean(), "repeat_frac": repeat_frac,
        "recip_frac": recip_frac, "cooc_auc": cooc_auc,
        "pos_cooc_mean": pos_mean, "neg_cooc_mean": neg_mean,
    }


if __name__ == "__main__":
    results = []
    for name in DATASETS:
        try:
            r = analyze(name)
            results.append(r)
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
            import traceback; traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("  CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Nodes':>10} {'Edges':>10} {'d_edge':>7} {'AvgDeg':>7} {'Repeat%':>8} {'Recip%':>7} {'CoocAUC':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['n']:>10,} {r['m']:>10,} {r['d_edge']:>7} {r['avg_deg']:>7.1f} {r['repeat_frac']*100:>8.1f} {r['recip_frac']*100:>7.1f} {r['cooc_auc']:>8.4f}")
