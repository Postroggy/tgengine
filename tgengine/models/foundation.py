"""Foundation model for cross-domain continuous-time dynamic graphs.

Architecture (d_model=512, K=64, ~19M params):

    InputTokenizer ──→ [TimeAwareMambaBlock]×4 ──→ GCA ──→
    [TimeAwareMambaBlock]×4 ──→ GCA ──→ [TimeAwareMambaBlock]×2 ──→
    LayerNorm ──→ output (B, K, d_model)

Design principles:
  - Domain-agnostic: no node IDs, only structural + temporal features
  - Full sequence output: pretraining heads operate on (B, K, d_model)
  - Three pretraining tasks: MTM, NTP, LP BCE
  - TimeAwareMambaBlock: A(Δt) modulation for Ebbinghaus forgetting
  - GCA: periodic structural injection via cross-attention
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgengine.core.batch import NeighborData, PreparedBatch
from tgengine.core.gather_spec import GatherSpec, NeighborSpec
from tgengine.models.base import TemporalModel, EmbeddingBundle, ModelOutput
from tgengine.nn.input_tokenizer import InputTokenizer
from tgengine.nn.mamba_block import TimeAwareMambaBlock
from tgengine.nn.graph_cross_attention import GraphCrossAttention
from tgengine.nn.pretraining_heads import (
    MTMHead, NTPHead, LPHead, EMAEncoder, block_wise_mask,
)


class FoundationModel(TemporalModel):
    """Cross-domain CTDG foundation model.

    Encodes a node's K-neighbor history into a rich (B, K, d_model)
    representation using domain-agnostic features (no node IDs).

    Args:
        d_edge: input edge feature dimension (e.g. 172)
        d_model: hidden dimension (default 512)
        d_state: SSM state size (default 64)
        d_conv: conv1d kernel width (default 4)
        expand: Mamba inner expansion factor (default 2)
        K: number of neighbors (default 64)
        d_time: time encoding dimension (default 64, must be even)
        n_mamba_layers: total Mamba layers (default 10)
        gca_every: insert GCA every N Mamba layers (default 5)
        n_gca_heads: attention heads in GCA (default 4)
        gca_ff_mult: GCA feedforward multiplier (default 2)
        mtm_target_dim: reconstruction target dim (d_edge + d_pair)
    """

    def __init__(
        self,
        d_edge: int = 172,
        d_model: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        K: int = 64,
        d_time: int = 64,
        n_mamba_layers: int = 10,
        gca_every: int = 5,
        n_gca_heads: int = 4,
        gca_ff_mult: int = 2,
        mtm_target_dim: int = 174,  # d_edge(172) + d_pair(2)
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.d_time = d_time
        self.mtm_target_dim = mtm_target_dim

        # GatherSpec: K neighbors for src, dst, neg
        self.gather_spec = GatherSpec(
            neighbors=NeighborSpec(k=K, for_nodes=("src", "dst", "neg")),
        )

        # --- Input tokenizer ---
        self.input_tokenizer = InputTokenizer(
            d_edge=d_edge, d_model=d_model, d_time=d_time,
        )

        # --- Encoder blocks: Mamba + GCA interleaved ---
        self.blocks = nn.ModuleList()
        mamba_count = 0
        for i in range(n_mamba_layers):
            self.blocks.append(
                TimeAwareMambaBlock(
                    d_model=d_model, d_state=d_state,
                    d_conv=d_conv, expand=expand,
                )
            )
            mamba_count += 1
            # Insert GCA after every gca_every Mamba layers
            if mamba_count % gca_every == 0 and mamba_count < n_mamba_layers:
                self.blocks.append(
                    GraphCrossAttention(
                        d_model=d_model, d_kv=d_model,
                        n_heads=n_gca_heads, ff_mult=gca_ff_mult,
                    )
                )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # --- Pretraining heads ---
        self.mtm_head = MTMHead(d_model=d_model, d_target=mtm_target_dim)
        self.ntp_head = NTPHead(d_model=d_model, d_time=d_time)
        self.lp_head = LPHead(d_model=d_model)

        # EMA encoder for MTM targets (initialized later via init_ema)
        self._ema: Optional[EMAEncoder] = None

        # Learnable mask embeddings
        self.mtm_mask_emb = nn.Parameter(torch.zeros(d_model))
        self.ntp_mask_emb = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mtm_mask_emb, std=0.02)
        nn.init.normal_(self.ntp_mask_emb, std=0.02)

    def init_ema(self, momentum: float = 0.999):
        """Initialize EMA encoder for MTM targets."""
        self._ema = EMAEncoder(self, momentum=momentum)

    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder parameters."""
        if self._ema is not None:
            self._ema.update(self)

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode_one(
        self,
        nbr: NeighborData,
        query_time: Tensor,
        mtm_mask: Optional[Tensor] = None,
        ntp_mask_pos0: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Encode a single node's K-neighbor buffer.

        Args:
            nbr: NeighborData (B, K)
            query_time: (B,) query timestamps
            mtm_mask: (K,) bool, True = masked position for MTM
            ntp_mask_pos0: if True, mask position 0 for NTP

        Returns:
            h: (B, K, d_model) encoded sequence
            struct_token: (B, 1, d_model) structural token
        """
        # Tokenize
        tokens, struct_token = self.input_tokenizer(nbr, query_time)
        # tokens: (B, K, d_model), struct_token: (B, 1, d_model)

        # Apply MTM masking
        if mtm_mask is not None:
            mtm_mask_expanded = mtm_mask.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
            tokens = torch.where(
                mtm_mask_expanded.expand_as(tokens),
                self.mtm_mask_emb.view(1, 1, self.d_model).expand_as(tokens),
                tokens,
            )

        # Apply NTP masking (mask position 0)
        if ntp_mask_pos0:
            tokens[:, 0, :] = self.ntp_mask_emb

        # Run through encoder blocks
        h = tokens
        # Cast to float32: TemporalGraph stores timestamps as float64,
        # but TimeAwareMambaBlock's dt_time_proj uses float32 weights.
        dt = (query_time.unsqueeze(1) - nbr.timestamps).clamp(min=0).float()  # (B, K)

        for block in self.blocks:
            if isinstance(block, TimeAwareMambaBlock):
                h = block(h, dt=dt)
            elif isinstance(block, GraphCrossAttention):
                h = block(h, struct_token)  # cross-attend to struct

        h = self.final_norm(h)
        return h, struct_token

    def encode(self, batch: PreparedBatch) -> EmbeddingBundle:
        """Standard encode: returns pooled node vectors for LP.

        Compatible with the Engine's standard training pipeline.
        """
        src_h, _ = self.encode_one(batch.src_neighbors, batch.time)
        dst_h, _ = self.encode_one(batch.dst_neighbors, batch.time)
        neg_h, _ = self.encode_one(batch.neg_neighbors, batch.time)

        src_vec = self.lp_head.pool(src_h, batch.src_neighbors.mask)
        dst_vec = self.lp_head.pool(dst_h, batch.dst_neighbors.mask)
        neg_vec = self.lp_head.pool(neg_h, batch.neg_neighbors.mask)

        return EmbeddingBundle(src=src_vec, dst=dst_vec, neg=neg_vec)

    def forward(self, batch: PreparedBatch) -> ModelOutput:
        """Standard forward: LP BCE loss via Engine compatibility."""
        bundle = self.encode(batch)
        return self._link_pred_loss(bundle)

    # ------------------------------------------------------------------
    # Pretraining forward (all 3 tasks)
    # ------------------------------------------------------------------

    def pretrain_forward(
        self,
        batch: PreparedBatch,
        mtm_mask_ratio: float = 0.15,
        mtm_block_size: int = 4,
    ) -> dict[str, Tensor]:
        """Pretraining forward pass: compute MTM + NTP + LP losses.

        Args:
            batch: PreparedBatch with src/dst/neg neighbors
            mtm_mask_ratio: fraction of positions to mask for MTM
            mtm_block_size: block size for MTM masking

        Returns:
            dict with keys: 'loss', 'mtm_loss', 'ntp_loss', 'lp_loss',
            'pos_score', 'neg_score'
        """
        device = batch.src.device
        K = self.K

        # --- Generate MTM mask (same for all three encodings in a batch) ---
        mtm_mask = block_wise_mask(
            K, mask_ratio=mtm_mask_ratio, block_size=mtm_block_size, device=device,
        )  # (K,)

        # === MTM: Masked Token Modeling (on src neighbors) ===
        # Encode src with MTM masking
        src_h_masked, _ = self.encode_one(
            batch.src_neighbors, batch.time, mtm_mask=mtm_mask,
        )  # (B, K, d_model)

        # Reconstruct masked positions
        mtm_pred = self.mtm_head(src_h_masked)  # (B, K, d_target)

        # Generate EMA targets for masked positions
        if self._ema is not None:
            with torch.no_grad():
                src_h_clean_ema, _ = self._ema.encoder.encode_one(
                    batch.src_neighbors, batch.time,
                )  # (B, K, d_model)
                mtm_target_full = self._ema.encoder.mtm_head(src_h_clean_ema)
        else:
            # No EMA: use self-reconstruction (will be less stable)
            with torch.no_grad():
                src_h_clean_ema, _ = self.encode_one(batch.src_neighbors, batch.time)
                mtm_target_full = self.mtm_head(src_h_clean_ema)

        # MTM loss: only on masked positions
        mtm_mask_b = mtm_mask.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
        mtm_mask_b = mtm_mask_b & batch.src_neighbors.mask.unsqueeze(-1)  # (B, K, 1)
        mtm_count = mtm_mask_b.float().sum().clamp(min=1)
        mtm_loss = (
            ((mtm_pred - mtm_target_full.detach()) ** 2) * mtm_mask_b.float()
        ).sum() / mtm_count

        # === NTP + LP: share one clean src encoding (no masking) ===
        # NTP masks position 0 conceptually, but for efficiency we reuse the
        # clean src encoding and predict time from pooled representation.
        # (TGPM's NTP uses encoder output directly; masking pos 0 is optional.)
        src_h, _ = self.encode_one(batch.src_neighbors, batch.time)  # reused
        dst_h, _ = self.encode_one(batch.dst_neighbors, batch.time)
        neg_h, _ = self.encode_one(batch.neg_neighbors, batch.time)

        # NTP: predict next-event time encoding from pooled src
        ntp_pred = self.ntp_head(src_h, batch.src_neighbors.mask)  # (B, d_time)
        dt_0 = (batch.time - batch.src_neighbors.timestamps[:, 0]).clamp(min=0).float()  # (B,)
        ntp_target = self.input_tokenizer.time_enc(dt_0).detach()  # (B, d_time)
        has_nbr = batch.src_neighbors.mask.any(dim=1).float()  # (B,)
        ntp_loss = (
            ((ntp_pred - ntp_target) ** 2).sum(dim=-1) * has_nbr
        ).sum() / has_nbr.sum().clamp(min=1)

        # LP: pooled dot-product scoring
        src_vec = self.lp_head.pool(src_h, batch.src_neighbors.mask)
        dst_vec = self.lp_head.pool(dst_h, batch.dst_neighbors.mask)
        neg_vec = self.lp_head.pool(neg_h, batch.neg_neighbors.mask)

        pos_score = self.lp_head.score(src_vec, dst_vec)  # (B,)
        neg_score = self.lp_head.score(src_vec, neg_vec)  # (B,)

        lp_loss = (
            F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        )

        # === Total loss ===
        total_loss = mtm_loss + ntp_loss + lp_loss

        return {
            "loss": total_loss,
            "mtm_loss": mtm_loss.detach(),
            "ntp_loss": ntp_loss.detach(),
            "lp_loss": lp_loss.detach(),
            "pos_score": pos_score.detach(),
            "neg_score": neg_score.detach(),
        }
