"""Example: Minimal training script.

Shows the complete workflow: load data → build model → train → evaluate.
"""

from tgengine import GatherSpec, TemporalGraph
from tgengine.engine import APEval, Engine, TrainConfig
from tgengine.models.dygmamba import DyGMamba
from tgengine.pipeline.negatives import RandomNegative
from tgengine.utils import seed_everything


def main():
    seed_everything(42)

    config = TrainConfig(
        epochs=100,
        batch_size=200,
        lr=1e-4,
        patience=10,
        device="cuda",
    )

    # TODO: implement dataset loading
    # train_batches, val_batches, test_batches = load_dataset("wikipedia", config)
    # graph = TemporalGraph.from_dataset(...)
    # neg_strategy = RandomNegative(num_nodes=graph.num_nodes)

    # model = DyGMamba(d_model=172, d_edge=172, n_layers=2)
    # engine = Engine(
    #     model=model,
    #     graph=graph,
    #     train_batches=train_batches,
    #     val_batches=val_batches,
    #     test_batches=test_batches,
    #     neg_strategy=neg_strategy,
    #     eval_protocol=APEval(),
    #     config=config,
    # )
    # results = engine.train()
    # print(f"Test AP: {results['ap']:.4f}")


if __name__ == "__main__":
    main()
