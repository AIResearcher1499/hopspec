"""EAGLE-1-style draft model with hop-aware auxiliary embeddings.

    concat(token_embedding, target_feature) -> Linear
        -> (+ segment_embed + recency_embed)
        -> single pre-norm causal decoder block
        -> predicted_feature -> frozen target LM head -> logits

No RMSNorm before the head: verified against EAGLE's source — cnets1.py
(EAGLE-1) does `last_headout = head(last_hidden)`; the `lm_head(self.norm(...))`
form exists only in the EAGLE-3-style cnets.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hopspec.data.schema import NUM_RECENCY_BUCKETS, NUM_SEGMENT_TYPES


@dataclass
class HopSpecDraftConfig:
    target_hidden_size: int
    vocab_size: int
    draft_hidden_size: int = 512
    # Derive from the schema, never hardcode: a hardcoded 6 once made adding
    # a SegmentType produce an index-out-of-range 30 minutes into a GPU run.
    num_segment_types: int = NUM_SEGMENT_TYPES
    num_recency_buckets: int = NUM_RECENCY_BUCKETS
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0


class _CausalDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_multiplier: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ln_attn = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_multiplier),
            nn.SiLU(),
            nn.Linear(hidden_size * ffn_multiplier, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        h = self.ln_attn(x)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h)),
            is_causal=True,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq_len, hidden)
        x = x + self.dropout(self.o_proj(attn))
        x = x + self.dropout(self.ffn(self.ln_ffn(x)))
        return x


class HopSpecDraftModel(nn.Module):
    def __init__(self, config: HopSpecDraftConfig, token_embedding: nn.Embedding):
        super().__init__()
        if token_embedding.embedding_dim != config.target_hidden_size:
            raise ValueError(
                "token embedding dim must equal target_hidden_size "
                f"({token_embedding.embedding_dim} != {config.target_hidden_size})"
            )
        self.config = config
        self.token_embedding = token_embedding
        self.input_proj = nn.Linear(2 * config.target_hidden_size, config.draft_hidden_size)
        self.segment_embedding = nn.Embedding(config.num_segment_types, config.draft_hidden_size)
        self.recency_embedding = nn.Embedding(config.num_recency_buckets, config.draft_hidden_size)
        self.block = _CausalDecoderBlock(
            config.draft_hidden_size, config.num_heads, config.ffn_multiplier, config.dropout
        )
        self.output_proj = nn.Linear(config.draft_hidden_size, config.target_hidden_size)

    @classmethod
    def from_target_embedding(
        cls, config: HopSpecDraftConfig, target_embedding: nn.Embedding
    ) -> "HopSpecDraftModel":
        """Reuse and freeze the target's embedding table so draft and target
        representations stay compatible."""
        embedding = nn.Embedding(
            target_embedding.num_embeddings, target_embedding.embedding_dim
        )
        with torch.no_grad():
            embedding.weight.copy_(target_embedding.weight.float())
        embedding.weight.requires_grad_(False)
        return cls(config, embedding)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_features: torch.Tensor,
        segment_type_ids: torch.Tensor,
        recency_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, T], got {tuple(token_ids.shape)}")
        batch, seq_len = token_ids.shape
        if target_features.shape != (batch, seq_len, self.config.target_hidden_size):
            raise ValueError(
                f"target_features must be [B, T, {self.config.target_hidden_size}], "
                f"got {tuple(target_features.shape)}"
            )
        for name, tensor in (("segment_type_ids", segment_type_ids),
                             ("recency_bucket_ids", recency_bucket_ids)):
            if tensor.shape != (batch, seq_len):
                raise ValueError(f"{name} must be [B, T], got {tuple(tensor.shape)}")

        # Target features are bfloat16 when the target model is; the draft is
        # float32. Cast here rather than at each call site.
        features = target_features.float()
        tokens = self.token_embedding(token_ids).float()
        x = self.input_proj(torch.cat([tokens, features], dim=-1))
        x = x + self.segment_embedding(segment_type_ids)
        x = x + self.recency_embedding(recency_bucket_ids)
        x = self.block(x)
        return self.output_proj(x)

    @staticmethod
    def predict_logits(
        predicted_features: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Frozen target LM head. Cast the (possibly bfloat16) head weight
        here — both direction mismatches were real runtime crashes."""
        return predicted_features @ lm_head_weight.t().to(predicted_features.dtype)
