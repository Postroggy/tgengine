"""Evaluation speed benchmark: TGEngine vs DyGLib on Reddit.

Compares ThreeWayEval (random + historical + inductive) speed.
Uses random model weights — we only care about speed, not accuracy.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
BS = 200
K = 64
PATCH_SIZE = 2


def load_reddit():
    import pandas as pd
    df = pd.read_csv(Path(DATA_ROOT) / "reddit" / "ml_reddit.csv")
    feats = np.load(Path(DATA_ROOT) / "reddit" / "ml_reddit.npy")
    src = df.iloc[:, 1].values.astype(np.int64)
    dst = df.iloc[:, 2].values.astype(np.int64)
    timestamps = df.iloc[:, 3].values.astype(np.float64)
    num_nodes = max(src.max(), dst.max()) + 1
    num_edges = len(src)
    d_edge = feats.shape[1]
    train_end = int(num_edges * 0.7)
    val_end = int(num_edges * 0.85)
    return src, dst, timestamps, feats, num_nodes, num_edges, d_edge, train_end, val_end


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    src, dst, timestamps, feats, num_nodes, num_edges, d_edge, train_end, val_end = load_reddit()
    test_edges = num_edges - val_end
    test_steps = (test_edges + BS - 1) // BS
    print(f"Reddit: {num_edges:,} edges, {num_nodes:,} nodes, d_edge={d_edge}")
    print(f"Train: {train_end:,}, Val: {val_end - train_end:,}, Test: {test_edges:,}")
    print(f"Test steps: {test_steps} (BS={BS})")
    print()

    src_t = torch.from_numpy(src).long().to(device)
    dst_t = torch.from_numpy(dst).long().to(device)
    time_t = torch.from_numpy(timestamps).float().to(device)
    feat_t = torch.from_numpy(feats).float().to(device)

    # Identify inductive nodes (appear in test but not train)
    train_nodes = set(src[:train_end].tolist()) | set(dst[:train_end].tolist())
    test_nodes = set(src[val_end:].tolist()) | set(dst[val_end:].tolist())
    inductive_nodes = torch.tensor(sorted(test_nodes - train_nodes), device=device)
    print(f"Inductive nodes: {len(inductive_nodes)}")
    print()

    # =========================================================================
    # TGEngine Evaluation
    # =========================================================================
    print("=" * 70)
    print("TGEngine ThreeWayEval (random + historical + inductive)")
    print("=" * 70)

    from tgengine.core.temporal_graph import TemporalGraph
    from tgengine.core.batch import RawBatch
    from tgengine.models.dygformer import DyGFormer
    from tgengine.pipeline import DataPipeline
    from tgengine.engine import ThreeWayEval

    # Build graph with ALL edges (train+val+test) for eval — matches DyGLib's full_neighbor_sampler
    graph = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph.advance(src_t[:val_end], dst_t[:val_end], time_t[:val_end], feat_t[:val_end])
    graph.freeze_csr()

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2,
                      patch_size=PATCH_SIZE, K=K, num_nodes=num_nodes).to(device)
    model.eval()
    pipeline = DataPipeline(model.gather_spec, graph)

    # Build eval batches for test set
    eval_batches = []
    for start in range(val_end, num_edges, BS):
        end = min(start + BS, num_edges)
        eval_batches.append(RawBatch(
            src=src_t[start:end], dst=dst_t[start:end],
            time=time_t[start:end], edge_feat=feat_t[start:end],
            neg=None, edge_indices=torch.arange(start, end, device=device),
        ))

    evaluator = ThreeWayEval(num_nodes, inductive_nodes, device=device)

    # Warmup (first few batches)
    with torch.no_grad():
        for batch in eval_batches[:3]:
            raw = RawBatch(src=batch.src, dst=batch.dst, time=batch.time,
                           edge_feat=batch.edge_feat,
                           neg=torch.randint(0, num_nodes, (len(batch.src),), device=device))
            prepared = pipeline.prepare(raw)
            model(prepared)
    torch.cuda.synchronize()

    # Time the full ThreeWayEval
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    results = evaluator.evaluate(model, pipeline, eval_batches, graph)
    torch.cuda.synchronize()
    tge_eval_time = time.perf_counter() - t_start

    print(f"\n  Results: {results}")
    print(f"  Total eval time: {tge_eval_time:.1f}s")
    print(f"  Per-step (avg across 3 passes): {tge_eval_time / (test_steps * 3) * 1000:.2f} ms")
    print(f"  Per-pass: {tge_eval_time / 3:.1f}s")

    del model, pipeline, graph, evaluator
    torch.cuda.empty_cache()

    # =========================================================================
    # DyGLib-style Evaluation
    # =========================================================================
    print()
    print("=" * 70)
    print("DyGLib-style ThreeWayEval (CPU sampling)")
    print("=" * 70)

    from tgengine.core.batch import PreparedBatch, NeighborData
    from tgengine.pipeline.negatives import RandomNegative, HistoricalNegative, InductiveNegative

    # Build DyGLib adjacency list (all edges up to val_end for eval)
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(val_end):
        adj_list[src[i]].append((dst[i], i, timestamps[i]))
        adj_list[dst[i]].append((src[i], i, timestamps[i]))

    nodes_neighbor_ids = []
    nodes_neighbor_times = []
    nodes_edge_feats = []
    for per_node in adj_list:
        sorted_n = sorted(per_node, key=lambda x: x[2])
        nodes_neighbor_ids.append(np.array([x[0] for x in sorted_n], dtype=np.int64))
        nodes_neighbor_times.append(np.array([x[2] for x in sorted_n], dtype=np.float64))
        nodes_edge_feats.append(np.array([feats[x[1]] for x in sorted_n], dtype=np.float32))

    def dyglib_get_neighbors(node_ids_np, times_np, k):
        B = len(node_ids_np)
        out_ids = np.zeros((B, k), dtype=np.int64)
        out_times = np.zeros((B, k), dtype=np.float64)
        out_feats = np.zeros((B, k, d_edge), dtype=np.float32)
        out_mask = np.zeros((B, k), dtype=bool)
        for idx in range(B):
            nid = int(node_ids_np[idx])
            t = times_np[idx]
            i = np.searchsorted(nodes_neighbor_times[nid], t)
            start = max(0, i - k)
            nbr_slice = nodes_neighbor_ids[nid][start:i]
            time_slice = nodes_neighbor_times[nid][start:i]
            feat_slice = nodes_edge_feats[nid][start:i]
            n = len(nbr_slice)
            if n > 0:
                out_ids[idx, k - n:] = nbr_slice
                out_times[idx, k - n:] = time_slice
                out_feats[idx, k - n:] = feat_slice
                out_mask[idx, k - n:] = True
        return out_ids, out_times, out_feats, out_mask

    # DyGLib negative strategies
    neg_strategies = {
        "random": RandomNegative(num_nodes),
        "historical": HistoricalNegative(num_nodes, device=device),
        "inductive": InductiveNegative(inductive_nodes),
    }

    # We need a graph for historical neg sampling
    graph2 = TemporalGraph(num_nodes, edge_feat_dim=d_edge, device=device)
    graph2.advance(src_t[:val_end], dst_t[:val_end], time_t[:val_end], feat_t[:val_end])
    graph2.freeze_csr()

    model = DyGFormer(d_edge=d_edge, d_model=172, n_layers=2, n_heads=2,
                      patch_size=PATCH_SIZE, K=K, num_nodes=num_nodes).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for batch in eval_batches[:3]:
            b = len(batch.src)
            batch_src_np = src[val_end:val_end + b]
            batch_time_np = timestamps[val_end:val_end + b]
            batch_dst_np = dst[val_end:val_end + b]
            neg_np = np.random.randint(0, num_nodes, b)
            s_ids, s_times, s_feats, s_mask = dyglib_get_neighbors(batch_src_np, batch_time_np, K)
            d_ids, d_times, d_feats, d_mask = dyglib_get_neighbors(batch_dst_np, batch_time_np, K)
            n_ids, n_times, n_feats, n_mask = dyglib_get_neighbors(neg_np, batch_time_np, K)
            prepared = PreparedBatch(
                src=batch.src, dst=batch.dst,
                neg=torch.from_numpy(neg_np).long().to(device), time=batch.time,
                src_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(s_ids).long().to(device),
                    timestamps=torch.from_numpy(s_times).float().to(device),
                    edge_feats=torch.from_numpy(s_feats).float().to(device),
                    mask=torch.from_numpy(s_mask).to(device)),
                dst_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(d_ids).long().to(device),
                    timestamps=torch.from_numpy(d_times).float().to(device),
                    edge_feats=torch.from_numpy(d_feats).float().to(device),
                    mask=torch.from_numpy(d_mask).to(device)),
                neg_neighbors=NeighborData(
                    neighbor_ids=torch.from_numpy(n_ids).long().to(device),
                    timestamps=torch.from_numpy(n_times).float().to(device),
                    edge_feats=torch.from_numpy(n_feats).float().to(device),
                    mask=torch.from_numpy(n_mask).to(device)),
            )
            model(prepared)
    torch.cuda.synchronize()

    # Full DyGLib-style eval with 3 neg types
    from sklearn.metrics import average_precision_score

    dyg_results = {}
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    t_per_pass = {}
    for neg_name, strategy in neg_strategies.items():
        pass_start = time.perf_counter()
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for batch in eval_batches:
                b = len(batch.src)
                # Sample negatives (same as TGEngine)
                neg = strategy.sample(batch.src, batch.dst, batch.time, graph2, batch.edge_indices)

                # CPU neighbor sampling (DyGLib style)
                batch_src_np = batch.src.cpu().numpy()
                batch_dst_np = batch.dst.cpu().numpy()
                batch_time_np = batch.time.cpu().numpy()
                neg_np = neg.cpu().numpy()

                s_ids, s_times, s_feats, s_mask = dyglib_get_neighbors(batch_src_np, batch_time_np, K)
                d_ids, d_times, d_feats, d_mask = dyglib_get_neighbors(batch_dst_np, batch_time_np, K)
                n_ids, n_times, n_feats, n_mask = dyglib_get_neighbors(neg_np, batch_time_np, K)

                prepared = PreparedBatch(
                    src=batch.src, dst=batch.dst, neg=neg, time=batch.time,
                    src_neighbors=NeighborData(
                        neighbor_ids=torch.from_numpy(s_ids).long().to(device),
                        timestamps=torch.from_numpy(s_times).float().to(device),
                        edge_feats=torch.from_numpy(s_feats).float().to(device),
                        mask=torch.from_numpy(s_mask).to(device)),
                    dst_neighbors=NeighborData(
                        neighbor_ids=torch.from_numpy(d_ids).long().to(device),
                        timestamps=torch.from_numpy(d_times).float().to(device),
                        edge_feats=torch.from_numpy(d_feats).float().to(device),
                        mask=torch.from_numpy(d_mask).to(device)),
                    neg_neighbors=NeighborData(
                        neighbor_ids=torch.from_numpy(n_ids).long().to(device),
                        timestamps=torch.from_numpy(n_times).float().to(device),
                        edge_feats=torch.from_numpy(n_feats).float().to(device),
                        mask=torch.from_numpy(n_mask).to(device)),
                )
                output = model(prepared)
                all_pos_scores.append(output.pos_score)
                all_neg_scores.append(output.neg_score)

        pos = torch.cat(all_pos_scores).sigmoid().cpu().numpy()
        neg_s = torch.cat(all_neg_scores).sigmoid().cpu().numpy()
        predicts = np.concatenate([pos, neg_s])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_s))])
        ap = float(average_precision_score(y_true=labels, y_score=predicts))
        dyg_results[f"ap_{neg_name}"] = ap
        t_per_pass[neg_name] = time.perf_counter() - pass_start

    torch.cuda.synchronize()
    dyg_eval_time = time.perf_counter() - t_start

    print(f"\n  Results: {dyg_results}")
    print(f"  Total eval time: {dyg_eval_time:.1f}s")
    print(f"  Per-pass breakdown:")
    for name, t in t_per_pass.items():
        print(f"    {name}: {t:.1f}s")
    print(f"  Per-step (avg across 3 passes): {dyg_eval_time / (test_steps * 3) * 1000:.2f} ms")

    del model, graph2
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("EVALUATION SPEED SUMMARY — Reddit ThreeWayEval")
    print("=" * 70)
    print(f"  Test set: {test_edges:,} edges, {test_steps} steps/pass, 3 passes")
    print(f"  Config: BS={BS}, K={K}, patch_size={PATCH_SIZE}")
    print()
    print(f"  {'Framework':<15} {'Total':>8} {'Per-pass':>10} {'Per-step':>10} {'Speedup':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    tge_per_step = tge_eval_time / (test_steps * 3) * 1000
    dyg_per_step = dyg_eval_time / (test_steps * 3) * 1000
    print(f"  {'TGEngine':<15} {tge_eval_time:>6.1f}s {tge_eval_time/3:>8.1f}s {tge_per_step:>8.2f}ms {'—':>10}")
    print(f"  {'DyGLib':<15} {dyg_eval_time:>6.1f}s {dyg_eval_time/3:>8.1f}s {dyg_per_step:>8.2f}ms {dyg_eval_time/tge_eval_time:>9.2f}x")
    print()
    print(f"  TGEngine eval speedup: {dyg_eval_time/tge_eval_time:.2f}x")


if __name__ == "__main__":
    main()
