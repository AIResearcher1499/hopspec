import torch

from hopspec.data.schema import SegmentType
from hopspec.train.loss import hop_aware_ce_loss
from hopspec.train.target_policy import GeneratedTokensOnlyPolicy, LossTargetPolicy


def make_aligned():
    return {
        "target_token_ids": torch.tensor([[5, 6, -100, 7, 8]]),
        "segment_ids_target": torch.tensor([
            [
                int(SegmentType.QUESTION),
                int(SegmentType.THOUGHT),
                int(SegmentType.OTHER),          # padding position
                int(SegmentType.RETRIEVED_PASSAGE),
                int(SegmentType.TEMPLATE),
            ]
        ]),
    }


def test_base_policy_returns_targets_unchanged():
    aligned = make_aligned()
    assert torch.equal(
        LossTargetPolicy().loss_targets(aligned), aligned["target_token_ids"]
    )


def test_generated_policy_masks_prefill_segments():
    out = GeneratedTokensOnlyPolicy().loss_targets(make_aligned())
    assert out[0, 0].item() == -100  # QUESTION
    assert out[0, 3].item() == -100  # RETRIEVED_PASSAGE


def test_generated_policy_keeps_thought_and_template():
    out = GeneratedTokensOnlyPolicy().loss_targets(make_aligned())
    assert out[0, 1].item() == 6  # THOUGHT stays scored
    assert out[0, 4].item() == 8  # TEMPLATE stays scored: the model emits it


def test_generated_policy_extends_padding_mask():
    out = GeneratedTokensOnlyPolicy().loss_targets(make_aligned())
    assert out[0, 2].item() == -100  # padding stays masked


def test_generated_policy_does_not_mutate_input():
    aligned = make_aligned()
    original = aligned["target_token_ids"].clone()
    GeneratedTokensOnlyPolicy().loss_targets(aligned)
    assert torch.equal(aligned["target_token_ids"], original)


def test_all_prefill_batch_gives_finite_zero_loss():
    aligned = {
        "target_token_ids": torch.tensor([[1, 2]]),
        "segment_ids_target": torch.tensor(
            [[int(SegmentType.QUESTION), int(SegmentType.RETRIEVED_PASSAGE)]]
        ),
    }
    targets = GeneratedTokensOnlyPolicy().loss_targets(aligned)
    logits = torch.randn(1, 2, 4)
    distances = torch.zeros(1, 2, dtype=torch.long)
    loss = hop_aware_ce_loss(logits, targets, distances)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_prefill_segments_constant():
    assert GeneratedTokensOnlyPolicy.PREFILL_SEGMENTS == (
        SegmentType.QUESTION, SegmentType.RETRIEVED_PASSAGE,
    )
