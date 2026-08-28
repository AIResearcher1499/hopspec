import pytest

from hopspec.data.schema import NO_PRIOR_HOP_BUCKET_ID, NO_PRIOR_HOP_DISTANCE
from hopspec.infer.adaptive_length import HopAwareLengthPolicy
from hopspec.infer.speculative_generate import RecencyStateTracker, run_speculative_round


# ---- RecencyStateTracker ----

def test_tracker_starts_no_prior():
    tracker = RecencyStateTracker()
    assert tracker.distance == NO_PRIOR_HOP_DISTANCE
    assert tracker.bucket_id == NO_PRIOR_HOP_BUCKET_ID


def test_tracker_appending_before_any_hop_stays_no_prior():
    tracker = RecencyStateTracker()
    tracker.on_tokens_appended(10)
    assert tracker.distance == NO_PRIOR_HOP_DISTANCE


def test_tracker_hop_boundary_sets_zero():
    tracker = RecencyStateTracker()
    tracker.on_hop_boundary()
    assert tracker.distance == 0
    assert tracker.bucket_id == 0


def test_tracker_read_then_advance():
    tracker = RecencyStateTracker()
    tracker.on_hop_boundary()
    # The token AT the anchor has distance 0: read before advancing.
    assert tracker.bucket_id == 0
    tracker.on_tokens_appended(1)
    assert tracker.distance == 1
    tracker.on_tokens_appended(2)
    assert tracker.distance == 3
    assert tracker.bucket_id == 2


def test_tracker_second_hop_resets():
    tracker = RecencyStateTracker()
    tracker.on_hop_boundary()
    tracker.on_tokens_appended(30)
    tracker.on_hop_boundary()
    assert tracker.distance == 0


def test_tracker_rejects_negative():
    with pytest.raises(ValueError):
        RecencyStateTracker().on_tokens_appended(-1)


# ---- run_speculative_round ----

class ScriptedDraft:
    def __init__(self, tokens):
        self.tokens = tokens
        self.requested = []

    def propose(self, context_ids, num_tokens):
        self.requested.append(num_tokens)
        return self.tokens[:num_tokens]


class ScriptedTarget:
    def __init__(self, continuation):
        self.continuation = continuation

    def next_tokens(self, context_ids, proposed):
        return self.continuation[: len(proposed) + 1]


def test_round_reads_bucket_before_proposing():
    tracker = RecencyStateTracker()
    tracker.on_hop_boundary()  # bucket 0 -> gamma 1
    draft = ScriptedDraft([1, 2, 3, 4, 5, 6, 7, 8])
    target = ScriptedTarget([1, 2, 3, 4, 5, 6, 7, 8, 9])
    run_speculative_round([0], draft, target, tracker, HopAwareLengthPolicy())
    assert draft.requested == [1]


def test_round_full_acceptance_appends_bonus():
    tracker = RecencyStateTracker()  # no-prior -> gamma 8
    draft = ScriptedDraft([1, 2, 3, 4, 5, 6, 7, 8])
    target = ScriptedTarget([1, 2, 3, 4, 5, 6, 7, 8, 9])
    context, accepted = run_speculative_round(
        [0], draft, target, tracker, HopAwareLengthPolicy()
    )
    assert accepted == 8
    assert context == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 8 accepted + bonus


def test_round_partial_acceptance_takes_correction():
    tracker = RecencyStateTracker()
    draft = ScriptedDraft([1, 2, 99, 4, 5, 6, 7, 8])
    target = ScriptedTarget([1, 2, 3, 4, 5, 6, 7, 8, 9])
    context, accepted = run_speculative_round(
        [0], draft, target, tracker, HopAwareLengthPolicy()
    )
    assert accepted == 2
    assert context == [0, 1, 2, 3]  # 2 accepted + the target's correction


def test_round_advances_tracker_by_emitted_tokens():
    tracker = RecencyStateTracker()
    tracker.on_hop_boundary()
    draft = ScriptedDraft([1])
    target = ScriptedTarget([1, 2])
    _context, accepted = run_speculative_round(
        [0], draft, target, tracker, HopAwareLengthPolicy()
    )
    assert accepted == 1
    assert tracker.distance == 2  # 1 accepted + bonus token


def test_round_rejects_short_verifier_output():
    tracker = RecencyStateTracker()
    draft = ScriptedDraft([1, 2, 3, 4, 5, 6, 7, 8])

    class ShortTarget:
        def next_tokens(self, context_ids, proposed):
            return proposed  # missing the bonus slot

    with pytest.raises(ValueError):
        run_speculative_round([0], draft, ShortTarget(), tracker, HopAwareLengthPolicy())


def test_round_does_not_mutate_context():
    tracker = RecencyStateTracker()
    context = [0, 1]
    draft = ScriptedDraft([5, 6, 7, 8, 9, 10, 11, 12])
    target = ScriptedTarget([5, 6, 7, 8, 9, 10, 11, 12, 13])
    run_speculative_round(context, draft, target, tracker, HopAwareLengthPolicy())
    assert context == [0, 1]
