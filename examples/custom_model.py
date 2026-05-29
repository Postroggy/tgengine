"""Example: define and train a custom model with TGEngine.

Shows how to build a model from components in ~40 lines,
then train it with the standard Engine.

Usage:
    python examples/custom_model.py --dataset wikipedia --data_root datasets
"""

import argparse
import torch
import torch.nn.functional as F

from tgengine import (
    TemporalModel, GatherSpec, NeighborSpec, PreparedBatch, ModelOutput,
    Engine, TrainConfig, APEval, RandomNegative,
    TemporalGraph, load_dataset,
)
from tgengine.nn import Time2Vec, TransformerSeqEncoder


class SimpleTemporalGNN(TemporalModel):
    """A minimal temporal GNN: Time2Vec + Transformer + dot-product scoring."""

    gather_spec = GatherSpec(neighbors=NeighborSpec(k=20))

    def __init__(self, d_edge: int = 172, d_model: int = 128):
        super().__init__()
        self.time_enc = Time2Vec(d_model=d_edge)
        self.encoder = TransformerSeqEncoder(d_in=d_edge, n_layers=1, n_heads=2)
        self.proj = torch.nn.Linear(d_edge, d_model)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        src_emb = self.proj(self.encoder(self.time_enc(batch.src_neighbors), batch.src_neighbors.mask))
        dst_emb = self.proj(self.encoder(self.time_enc(batch.dst_neighbors), batch.dst_neighbors.mask))
        neg_emb = self.proj(self.encoder(self.time_enc(batch.neg_neighbors), batch.neg_neighbors.mask))

        pos_score = (src_emb * dst_emb).sum(dim=-1)
        neg_score = (src_emb * neg_emb).sum(dim=-1)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        return ModelOutput(pos_score=pos_score, neg_score=neg_score, loss=loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, dataset_path=args.data_root)
    graph = TemporalGraph(dataset.num_nodes, buffer_size=20,
                          edge_feat_dim=dataset.edge_feat_dim, device=args.device)

    model = SimpleTemporalGNN(d_edge=dataset.edge_feat_dim)

    engine = Engine(
        model=model,
        graph=graph,
        train_batches=dataset.get_batches("train", 200, device=args.device),
        val_batches=dataset.get_batches("val", 200, device=args.device),
        test_batches=dataset.get_batches("test", 200, device=args.device),
        neg_strategy=RandomNegative(dataset.num_nodes),
        eval_protocol=APEval(),
        config=TrainConfig(epochs=args.epochs, lr=1e-4, device=args.device),
        inductive_edges=dataset.inductive_edges,
    )

    results = engine.train()
    print(f"\nTest AP: {results.get('ap', 'N/A')}")


if __name__ == "__main__":
    main()
