import math

import pytest
import torch

from hopspec.data.schema import NO_PRIOR_HOP_DISTANCE
from hopspec.train.loss import (
    hop_aware_ce_loss,
    hop_aware_feature_loss,
    hop_aware_kd_loss,
    recency_weights,
)


# ---- recency_weights ----

def test_weight_at_distance_zero():
    w = recency_weights(torch.tensor([0]), alpha=2.0, tau=16.0)
    assert w.item() == pytest.approx(3.0)


def test_weight_formula():
    w = recency_weights(torch.tensor([16]), alpha=2.0, tau=16.0)
    assert w.item() == pytest.approx(1.0 + 2.0 * math.exp(-1.0))


def test_weight_no_prior_distance_is_one():
    w = recency_weights(torch.tensor([NO_PRIOR_HOP_DISTANCE]))
    assert w.item() == pytest.approx(1.0)


def test_alpha_zero_gives_uniform_weights():
    w = recency_weights(torch.tensor([0, 5, 100]), alpha=0.0)
    assert w.tolist() == [1.0, 1.0, 1.0]


def test_tau_must_be_positive():
    with pytest.raises(ValueError):
        recency_weights(torch.tensor([1]), tau=0.0)
    with pytest.raises(ValueError):
        recency_weights(torch.tensor([1]), tau=-1.0)


# ---- hop_aware_ce_loss ----

def test_ce_matches_hand_computation():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    targets = torch.tensor([[0, 1]])
    distances = torch.tensor([[0, 1000]])
    loss = hop_aware_ce_loss(logits, targets, distances, alpha=2.0, tau=16.0)
    per_token = -torch.log_softmax(logits[0], dim=-1)[[0, 1], [0, 1]]
    w = recency_weights(distances[0].float())
    expected = (per_token * w).sum() / w.sum()
    assert loss.item() == pytest.approx(expected.item())


def test_ce_ignores_ignore_index():
    logits = torch.randn(1, 3, 4)
    targets = torch.tensor([[1, -100, 2]])
    distances = torch.tensor([[0, 0, 0]])
    loss_with = hop_aware_ce_loss(logits, targets, distances)
    loss_without = hop_aware_ce_loss(
        logits[:, [0, 2]], targets[:, [0, 2]], distances[:, [0, 2]]
    )
    assert loss_with.item() == pytest.approx(loss_without.item())


def test_ce_all_masked_is_finite_zero():
    logits = torch.randn(1, 2, 4)
    targets = torch.full((1, 2), -100)
    distances = torch.zeros(1, 2, dtype=torch.long)
    loss = hop_aware_ce_loss(logits, targets, distances)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_ce_weighting_emphasizes_near_hop():
    # A wrong prediction near a hop should cost more than the same far away.
    logits = torch.tensor([[[0.0, 3.0]], [[0.0, 3.0]]])
    targets = torch.tensor([[0], [0]])
    near = hop_aware_ce_loss(logits[:1], targets[:1], torch.tensor([[0]]))
    far = hop_aware_ce_loss(logits[1:], targets[1:], torch.tensor([[1000]]))
    assert near.item() == pytest.approx(far.item())  # weighted MEAN of one element
    # With mixed positions in one batch, the near one dominates the mean.
    both_wrong_near = hop_aware_ce_loss(
        torch.tensor([[[0.0, 3.0], [3.0, 0.0]]]), torch.tensor([[0, 0]]),
        torch.tensor([[0, 1000]]),
    )
    both_wrong_far = hop_aware_ce_loss(
        torch.tensor([[[0.0, 3.0], [3.0, 0.0]]]), torch.tensor([[0, 0]]),
        torch.tensor([[1000, 0]]),
    )
    assert both_wrong_near.item() > both_wrong_far.item()


# ---- hop_aware_feature_loss ----

def test_feature_loss_zero_on_exact_prediction():
    feats = torch.randn(1, 3, 8)
    distances = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = hop_aware_feature_loss(feats, feats.clone(), distances, mask)
    assert loss.item() == 0.0


def test_feature_loss_matches_hand_computation():
    pred = torch.zeros(1, 2, 4)
    target = torch.ones(1, 2, 4) * 0.5  # |diff|=0.5 < 1 -> smooth_l1 = 0.5*d^2
    distances = torch.tensor([[0, 1000]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = hop_aware_feature_loss(pred, target, distances, mask, alpha=2.0, tau=16.0)
    per_token = 0.5 * 0.5**2
    w = recency_weights(distances.float())
    expected = (per_token * w).sum() / w.sum()
    assert loss.item() == pytest.approx(expected.item())


def test_feature_loss_excludes_masked_positions():
    pred = torch.zeros(1, 2, 4)
    target = torch.zeros(1, 2, 4)
    target[0, 1] = 100.0  # huge error, but masked out
    distances = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.tensor([[True, False]])
    loss = hop_aware_feature_loss(pred, target, distances, mask)
    assert loss.item() == 0.0


def test_feature_loss_all_masked_finite():
    pred = torch.randn(1, 2, 4)
    target = torch.randn(1, 2, 4)
    distances = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    loss = hop_aware_feature_loss(pred, target, distances, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


# ---- hop_aware_kd_loss ----

def test_kd_loss_zero_when_identical():
    logits = torch.randn(1, 3, 5)
    distances = torch.zeros(1, 3, dtype=torch.long)
    loss = hop_aware_kd_loss(logits, logits.clone(), distances)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_positive_when_different():
    draft = torch.tensor([[[0.0, 0.0, 5.0]]])
    target = torch.tensor([[[5.0, 0.0, 0.0]]])
    distances = torch.zeros(1, 1, dtype=torch.long)
    assert hop_aware_kd_loss(draft, target, distances).item() > 0.0
