"""Diagnose CSR query correctness during training."""
import sys
sys.path.insert(0, ".")
import torch
from tgengine.core.dataset import load_dataset
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.batch import RawBatch
from tgengine.pipeline import DataPipeline
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.utils import seed_everything

DATA_ROOT = "/mnt/home/gyq/CodeBase/Graph/DG_Data"
seed_everything(2020)

ds = load_dataset("uci", DATA_ROOT)
device = "cuda"
K = 31

# Build graph like Engine does
graph = TemporalGraph(ds.num_nodes, edge_feat_dim=ds.edge_feat_dim, device=device)
train_batches = ds.get_batches("train", 200, device)

print(f"Preloading {len(train_batches)} train batches...")
for rb in train_batches:
    graph.advance(rb.src, rb.dst, rb.time, rb.edge_feat)
graph.freeze_csr()

print(f"CSR stats:")
print(f"  offsets shape: {graph._offsets.shape}")
print(f"  nbr_ids shape: {graph._nbr_ids.shape}")
print(f"  nbr_times shape: {graph._nbr_times.shape}")
print(f"  nbr_feats shape: {graph._nbr_feats.shape}")
print(f"  Total directed edges: {len(graph._nbr_ids)}")
print(f"  Expected: {ds.train_end * 2} = {ds.train_end} edges x 2 directions")

# Check a few training batch queries
spec = GatherSpec(neighbors=NeighborSpec(k=K, strategy="recency"))
pipeline = DataPipeline(spec, graph)

for i in [0, 50, 100, 150, 171]:
    rb = train_batches[i]
    result = graph.recent(rb.src[:5], rb.time[:5], K)
    valid_counts = result.mask.sum(dim=1)
    print(f"\nBatch {i}: time range [{rb.time.min():.1f}, {rb.time.max():.1f}]")
    print(f"  First 5 src nodes: {rb.src[:5].tolist()}")
    print(f"  Valid neighbor counts: {valid_counts.tolist()}")
    if result.mask.any():
        # Check time ordering: most recent should be last
        for j in range(min(5, rb.src.shape[0])):
            if result.mask[j].any():
                valid_times = result.timestamps[j][result.mask[j]]
                if len(valid_times) > 1:
                    is_sorted = (valid_times[1:] >= valid_times[:-1]).all()
                    print(f"  Node {rb.src[j].item()}: {len(valid_times)} nbrs, "
                          f"times [{valid_times[0]:.1f}..{valid_times[-1]:.1f}], "
                          f"sorted={is_sorted.item()}")

# Check edge feats are not all zeros
sample = graph.recent(train_batches[100].src[:3], train_batches[100].time[:3], K)
feat_norms = sample.edge_feats[sample.mask].norm(dim=-1)
print(f"\nEdge feat norms (sample): min={feat_norms.min():.4f}, max={feat_norms.max():.4f}, mean={feat_norms.mean():.4f}")
if feat_norms.max() < 0.001:
    print("WARNING: Edge features appear to be all zeros!")
