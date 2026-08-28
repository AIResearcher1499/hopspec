"""Hop-aware (recency-weighted) training losses.

EAGLE's actual objective, from eagle/train/main.py:

    vloss = SmoothL1Loss(reduction="none")(predict, target)  # next-feature regression
    ploss = -sum(target_p * log_softmax(head(predict)))      # soft KL on tokens
    loss  = 1.0 * vloss + 0.1 * ploss

EAGLE trains the draft primarily to regress the next hidden state; the token
term carries only 0.1 weight. Hard CE is retained here for the token term
instead of EAGLE's soft KL — a documented, deliberate simplification, since
soft KL needs the target's full [B, T, V] logits (hop_aware_kd_loss exists
for when they are available).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recency_weights(
    distances: torch.Tensor, alpha: float = 2.0, tau: float = 16.0
) -> torch.Tensor:
    """w(d) = 1 + alpha * exp(-d / tau). NO_PRIOR_HOP_DISTANCE underflows the
    exponential to 0, so pre-retrieval positions get weight 1."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 1.0 + alpha * torch.exp(-distances.float() / tau)


def _weighted_mean(per_token: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-8)


def hop_aware_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    distances: torch.Tensor,
    alpha: float = 2.0,
    tau: float = 16.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    per_token = F.cross_entropy(
        flat_logits, flat_targets, reduction="none", ignore_index=ignore_index
    )
    weights = recency_weights(distances.reshape(-1), alpha=alpha, tau=tau)
    weights = weights * (flat_targets != ignore_index).float()
    return _weighted_mean(per_token, weights)


def hop_aware_feature_loss(
    pred_feats: torch.Tensor,
    target_feats: torch.Tensor,
    distances: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 2.0,
    tau: float = 16.0,
) -> torch.Tensor:
    per_token = F.smooth_l1_loss(
        pred_feats, target_feats.to(pred_feats.dtype), reduction="none"
    ).mean(-1)
    weights = recency_weights(distances, alpha=alpha, tau=tau) * valid_mask.float()
    return _weighted_mean(per_token, weights)


def hop_aware_kd_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    distances: torch.Tensor,
    alpha: float = 2.0,
    tau: float = 16.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    log_p_draft = F.log_softmax(logits / temperature, dim=-1)
    log_p_target = F.log_softmax(target_logits.to(logits.dtype) / temperature, dim=-1)
    p_target = log_p_target.exp()
    per_token = (p_target * (log_p_target - log_p_draft)).sum(-1)
    weights = recency_weights(distances, alpha=alpha, tau=tau)
    return _weighted_mean(per_token, weights)
