"""Quick benchmark: co_occurrence fusion vs redundant extra call."""
import torch
import time
from tgengine.core.temporal_graph import TemporalGraph
from tgengine.core.batch import RawBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.pipeline import DataPipeline

device = "cuda"
N, B, k = 10000, 200, 32
graph = TemporalGraph(N, buffer_size=32, edge_feat_dim=172, device=device)
graph.advance(
    torch.randint(0, N, (50000,), device=device),
    torch.randint(0, N, (50000,), device=device),
    torch.arange(50000, dtype=torch.float64, device=device),
    torch.randn(50000, 172, device=device),
)

spec_co = GatherSpec(neighbors=NeighborSpec(k=k, strategy="recency"), co_occurrence=True)
pipeline = DataPipeline(spec_co, graph)

src = torch.randint(0, N, (B,), device=device)
dst = torch.randint(0, N, (B,), device=device)
neg = torch.randint(0, N, (B,), device=device)
t = torch.full((B,), 60000., dtype=torch.float64, device=device)
raw = RawBatch(src=src, dst=dst, time=t, neg=neg)


def run_fused():
    return pipeline.prepare(raw)


def run_old():
    b = pipeline.prepare(raw)
    graph.co_neighbors(src, dst, t, k=k)
    return b


def timed(fn, n=200):
    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


t_fused = timed(run_fused)
t_old = timed(run_old)

print(f"prepare() with co_occur FUSED : {t_fused:.3f}ms")
print(f"prepare() + extra co_neighbors: {t_old:.3f}ms")
print(f"Speedup from co_occur fusion  : {t_old/t_fused:.2f}x")
