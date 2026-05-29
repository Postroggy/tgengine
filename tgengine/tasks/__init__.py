"""Task heads for TGEngine — plug on top of any TemporalModel embedding.

Every TaskHead:
  - receives node/edge embeddings from the model's backbone
  - computes task-specific loss and predictions
  - returns a partial ModelOutput (loss + task fields filled)

Usage pattern:
    class MyModel(TemporalModel):
        def __init__(self):
            ...
            self.task_head = NodeClassificationHead(d_model=172, num_classes=7)

        def forward(self, batch):
            src_emb = self._encode(batch)      # (B, d)
            out = self.task_head(src_emb, batch.node_labels)
            # out.loss, out.node_pred are filled; add link-pred loss if needed
            return out
"""

from .base import TaskHead
from .link_pred import LinkPredHead
from .node_cls import NodeClassificationHead, NodeBinaryClassificationHead
from .node_reg import NodeRegressionHead
from .edge_cls import EdgeClassificationHead, EdgeBinaryClassificationHead
from .edge_reg import EdgeRegressionHead
from .anomaly import AnomalyDetectionHead
from .multi import MultiTaskHead

__all__ = [
    "TaskHead",
    "LinkPredHead",
    "NodeClassificationHead",
    "NodeBinaryClassificationHead",
    "NodeRegressionHead",
    "EdgeClassificationHead",
    "EdgeBinaryClassificationHead",
    "EdgeRegressionHead",
    "AnomalyDetectionHead",
    "MultiTaskHead",
]
