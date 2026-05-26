# TGEngine

High-performance framework for Continuous-Time Dynamic Graph (CTDG) learning.

## Core Design

- **GatherSpec**: Models declare what data they need (static, optimizable)
- **DataPipeline**: Fuses all graph operations into a single optimized pass
- **TemporalGraph**: GPU-resident circular buffer storage
- **Engine**: Manages training, evaluation, and temporal state lifecycle

## Quick Start

```python
from tgengine import TemporalModel, GatherSpec, NeighborSpec, PreparedBatch, ModelOutput
from tgengine.nn import Time2Vec, MambaSeqEncoder, BilinearDecoder
from tgengine.engine import Engine, TrainConfig

class MyModel(TemporalModel):
    gather_spec = GatherSpec(neighbors=NeighborSpec(k=32))

    def __init__(self):
        super().__init__()
        self.time_enc = Time2Vec(172)
        self.encoder = MambaSeqEncoder(172, n_layers=2)
        self.decoder = BilinearDecoder(172)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self.encoder(self.time_enc(batch.src_neighbors), batch.src_neighbors.mask)
        dst_emb = self.encoder(self.time_enc(batch.dst_neighbors), batch.dst_neighbors.mask)
        ...
```

See `ARCHITECTURE.md` for full design documentation.
