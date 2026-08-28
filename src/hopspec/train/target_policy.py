"""Which positions the TOKEN loss supervises.

Keep the default `all`. `generated` was tested and REFUTED: masking prefill
made acceptance significantly worse in all 8 paired comparisons (−2.5 to −7.7
points, most p<1e-6). Capacity was not the binding constraint — data was:
masking removes ~86% of next-token supervision from an already small corpus.
The class is kept for a retest at much larger data scale.
"""

from __future__ import annotations

import torch

from hopspec.data.schema import SegmentType


class LossTargetPolicy:
    """Score all non-padding positions (padding is already -100)."""

    def loss_targets(self, aligned: dict[str, torch.Tensor]) -> torch.Tensor:
        return aligned["target_token_ids"]


class GeneratedTokensOnlyPolicy(LossTargetPolicy):
    """EXTENDS the base's masking rather than reimplementing it, so it cannot
    forget padding. TEMPLATE stays scored: the model really does emit
    "Thought:", and "exclude from measurement" is a different question from
    "exclude from training"."""

    PREFILL_SEGMENTS = (SegmentType.QUESTION, SegmentType.RETRIEVED_PASSAGE)

    def loss_targets(self, aligned: dict[str, torch.Tensor]) -> torch.Tensor:
        targets = super().loss_targets(aligned)  # reuse the padding mask
        segments = aligned["segment_ids_target"]
        is_prefill = torch.zeros_like(segments, dtype=torch.bool)
        for segment in self.PREFILL_SEGMENTS:
            is_prefill |= segments == int(segment)
        return targets.masked_fill(is_prefill, -100)  # add one rule
