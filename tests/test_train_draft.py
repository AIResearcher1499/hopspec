import torch

from hopspec.data.schema import (
    NO_PRIOR_HOP_DISTANCE,
    NUM_RECENCY_BUCKETS,
    SegmentType,
)
from hopspec.train.train_draft import EAGLE_FEATURE_LAYER, collate, eagle_aligned_batch

from conftest import HIDDEN_SIZE, ROUTED_VOCAB_SIZE, TinyTargetModel


def make_batch_records():
    return [
        {
            "input_ids": [5, 6, 7, 8, 9],
            "segment_type_ids": [0, 1, 1, 2, 4],
            "recency_bucket_ids": [6, 6, 0, 1, 1],
            "recency_distances": [NO_PRIOR_HOP_DISTANCE, NO_PRIOR_HOP_DISTANCE, 0, 1, 2],
        },
        {
            "input_ids": [3, 4, 5],
            "segment_type_ids": [0, 1, 1],
            "recency_bucket_ids": [6, 6, 6],
            "recency_distances": [NO_PRIOR_HOP_DISTANCE] * 3,
        },
    ]


# ---- collate ----

def test_collate_pads_input_ids_with_pad_id():
    batch = collate(make_batch_records(), pad_id=99)
    assert batch["input_ids"][1].tolist() == [3, 4, 5, 99, 99]


def test_collate_attention_mask():
    batch = collate(make_batch_records(), pad_id=0)
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]


def test_collate_pads_buckets_with_far_bucket_never_zero():
    batch = collate(make_batch_records(), pad_id=0)
    padded = batch["recency_bucket_ids"][1, 3:].tolist()
    assert padded == [NUM_RECENCY_BUCKETS - 1] * 2
    assert 0 not in padded


def test_collate_pads_segments_with_other_never_question():
    batch = collate(make_batch_records(), pad_id=0)
    padded = batch["segment_type_ids"][1, 3:].tolist()
    assert padded == [int(SegmentType.OTHER)] * 2
    assert int(SegmentType.QUESTION) not in padded


def test_collate_pads_distances_with_no_prior():
    batch = collate(make_batch_records(), pad_id=0)
    assert batch["recency_distances"][1, 3:].tolist() == [NO_PRIOR_HOP_DISTANCE] * 2


# ---- eagle_aligned_batch ----

class PositionFeatureModel(torch.nn.Module):
    """hidden_states[layer][b, t, :] == layer * 1000 + t everywhere, so any
    off-by-one in feature slicing is directly visible."""

    def __init__(self, hidden_size=4, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, seq_len = input_ids.shape
        states = []
        for layer in range(self.num_layers + 1):
            positions = torch.arange(seq_len, dtype=torch.float32)
            state = (layer * 1000 + positions).view(1, seq_len, 1)
            states.append(state.expand(batch, seq_len, self.hidden_size).clone())
        from types import SimpleNamespace

        return SimpleNamespace(hidden_states=tuple(states))


def aligned(records=None, feature_layer=EAGLE_FEATURE_LAYER):
    batch = collate(records or make_batch_records(), pad_id=0)
    return eagle_aligned_batch(PositionFeatureModel(), batch, feature_layer=feature_layer)


def test_eagle_feature_layer_constant_is_minus_one():
    assert EAGLE_FEATURE_LAYER == -1


def test_all_tensors_have_length_t_minus_2():
    out = aligned()
    for key, tensor in out.items():
        assert tensor.shape[1] == 3, key  # T=5


def test_input_feature_is_q_minus_2():
    out = aligned()
    # Position value of target_feats at slot 0 must be feature index 0 (= q-2 for q=2).
    assert out["target_feats"][0, :, 0].tolist() == [2000.0, 2001.0, 2002.0]


def test_feature_targets_exactly_one_step_ahead_of_inputs():
    out = aligned()
    assert torch.equal(out["feature_targets"] - out["target_feats"],
                       torch.ones_like(out["target_feats"]))


def test_draft_token_is_q_minus_1():
    out = aligned()
    assert out["draft_token_ids"][0].tolist() == [6, 7, 8]


def test_labels_are_at_q():
    out = aligned()
    assert out["target_token_ids"][0].tolist() == [7, 8, 9]
    assert out["segment_ids_target"][0].tolist() == [1, 2, 4]
    assert out["recency_buckets"][0].tolist() == [0, 1, 1]
    assert out["recency_distances"][0].tolist() == [0, 1, 2]


def test_segment_input_is_at_q_minus_1():
    out = aligned()
    assert out["segment_ids_input"][0].tolist() == [1, 1, 2]


def test_padding_targets_become_minus_100():
    out = aligned()
    assert out["target_token_ids"][1].tolist() == [5, -100, -100]


def test_hop_index_minus_one_before_first_hop():
    out = aligned()
    # Row 1 never has a hop.
    assert out["hop_index"][1].tolist() == [-1, -1, -1]


def test_hop_index_counted_on_full_sequence():
    # Hop closes at absolute position 1 — inside the sliced-away prefix.
    records = [{
        "input_ids": [1, 2, 3, 4, 5],
        "segment_type_ids": [1, 1, 1, 1, 1],
        "recency_bucket_ids": [6, 0, 1, 0, 1],
        "recency_distances": [NO_PRIOR_HOP_DISTANCE, 0, 1, 0, 1],
    }]
    out = aligned(records)
    # Positions q=2,3,4: hop 0 governs q=2, hop 1 governs q=3,4.
    assert out["hop_index"][0].tolist() == [0, 1, 1]


def test_feature_layer_is_selectable():
    out = aligned(feature_layer=-2)
    assert out["target_feats"][0, 0, 0].item() == 1000.0


def test_short_sequence_raises():
    import pytest

    records = [{
        "input_ids": [1, 2],
        "segment_type_ids": [1, 1],
        "recency_bucket_ids": [6, 6],
        "recency_distances": [NO_PRIOR_HOP_DISTANCE] * 2,
    }]
    batch = collate(records, pad_id=0)
    with pytest.raises(ValueError):
        eagle_aligned_batch(PositionFeatureModel(), batch)


# =====================================================================
# chat feature layout — the deployed loop hands EAGLE features computed
# behind the assistant separator, not from a bare-document forward.
# Wrapper tokens live in the KV cache and NOWHERE else: never in a label
# array, never in the loss.
# =====================================================================

import pytest

from hopspec.infer.chained_eval import assistant_turns, token_regions
from hopspec.train.train_draft import chat_layout_features

CHAT_PREFIX = [1, 2, 3]
CHAT_SUFFIX = [4, 5]


def wrapped_full_forward(model, ids):
    tensor = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    output_hidden_states=True)
    return out.hidden_states[-1][0].float()


def test_raw_features_are_bit_identical_to_before(records, tiny_target):
    """The mode switch must not move a single existing number."""
    batch = collate(records[:2], pad_id=0)
    baseline = eagle_aligned_batch(tiny_target, batch)
    with_default = eagle_aligned_batch(tiny_target, batch, features=None)
    for key in baseline:
        assert torch.equal(baseline[key], with_default[key])


def test_features_override_replaces_only_the_forward(records, tiny_target):
    batch = collate(records[:2], pad_id=0)
    shape = (*batch["input_ids"].shape, HIDDEN_SIZE)
    override = torch.full(shape, 0.5)
    aligned = eagle_aligned_batch(tiny_target, batch, features=override)
    assert torch.allclose(aligned["target_feats"], torch.full_like(aligned["target_feats"], 0.5))
    # labels and token slices are untouched by the override
    baseline = eagle_aligned_batch(tiny_target, batch)
    for key in ("draft_token_ids", "segment_ids_input", "recency_buckets",
                "target_token_ids", "segment_ids_target", "attention_mask_target",
                "hop_index", "recency_distances"):
        assert torch.equal(baseline[key], aligned[key])


def test_features_override_rejects_a_wrapper_sized_tensor(records, tiny_target):
    """Passing features that include wrapper rows must fail loudly — silently
    accepting them would shift every label by len(prefix)."""
    batch = collate(records[:2], pad_id=0)
    batch_size, seq_len = batch["input_ids"].shape
    wrong = torch.zeros(batch_size, seq_len + len(CHAT_PREFIX), HIDDEN_SIZE)
    with pytest.raises(ValueError, match="document positions"):
        eagle_aligned_batch(tiny_target, batch, features=wrong)


def test_chat_features_are_document_sized_and_pad_free(routed_records, routed_tokenizer,
                                                       tiny_target):
    target = TinyTargetModel(vocab_size=ROUTED_VOCAB_SIZE)
    batch = collate(routed_records, pad_id=0)
    feats = chat_layout_features(target, batch, CHAT_PREFIX, CHAT_SUFFIX,
                                 routed_tokenizer.decode)
    assert feats.shape == (*batch["input_ids"].shape, HIDDEN_SIZE)
    # padding rows stay zero: no wrapper row can leak into a padded slot
    for row in range(feats.shape[0]):
        length = int(batch["attention_mask"][row].sum())
        assert torch.count_nonzero(feats[row, length:]) == 0


def test_chat_features_match_a_full_wrapped_forward(routed_records, routed_tokenizer):
    """The load-bearing test: each document position's feature must equal the
    one from an uncached forward over the sequence the deployed loop builds."""
    target = TinyTargetModel(vocab_size=ROUTED_VOCAB_SIZE)
    record = routed_records[0]
    batch = collate([record], pad_id=0)
    feats = chat_layout_features(target, batch, CHAT_PREFIX, CHAT_SUFFIX,
                                 routed_tokenizer.decode)[0]
    ids = record["input_ids"]
    segments = record["segment_type_ids"]

    user_layout = wrapped_full_forward(target, CHAT_PREFIX + ids)[len(CHAT_PREFIX):]
    expected = user_layout.clone()
    turns = assistant_turns(ids, segments, routed_tokenizer.decode)
    assert turns, "the fixture must contain at least one assistant turn"
    for turn, end in turns:
        wrapped = CHAT_PREFIX + ids[:turn] + CHAT_SUFFIX + ids[turn:end]
        rows = wrapped_full_forward(target, wrapped)
        expected[turn:end] = rows[len(CHAT_PREFIX) + turn + len(CHAT_SUFFIX):]
    assert torch.allclose(feats[:len(ids)], expected, atol=1e-4)


def test_chat_features_differ_from_raw_only_inside_the_turns(routed_records,
                                                             routed_tokenizer):
    target = TinyTargetModel(vocab_size=ROUTED_VOCAB_SIZE)
    record = routed_records[0]
    batch = collate([record], pad_id=0)
    chat = chat_layout_features(target, batch, CHAT_PREFIX, CHAT_SUFFIX,
                                routed_tokenizer.decode)[0]
    plain = chat_layout_features(target, batch, [], [], None)[0]
    ids = record["input_ids"]
    inside = set()
    for turn, end in assistant_turns(ids, record["segment_type_ids"],
                                     routed_tokenizer.decode):
        inside.update(range(turn, end))
    assert inside
    # an empty wrapper reproduces the bare-document forward
    assert torch.allclose(plain[:len(ids)], wrapped_full_forward(target, ids), atol=1e-4)


def test_assistant_turns_skip_the_question_prefix_region(routed_records,
                                                         routed_tokenizer):
    """The "Question: " region is committed verbatim by the replay (fewer than
    two document tokens precede it), so it opens no turn — training must agree
    or the two layouts diverge at the very first region."""
    record = routed_records[0]
    turns = assistant_turns(record["input_ids"], record["segment_type_ids"],
                            routed_tokenizer.decode)
    assert all(turn >= 2 for turn, _end in turns)
    first_generated = next(start for flag, start, _end
                           in token_regions(record["segment_type_ids"]) if not flag)
    assert first_generated == 0
    assert all(turn != first_generated for turn, _end in turns)


def test_assistant_turns_start_after_leading_prompt_whitespace(routed_records,
                                                               routed_tokenizer):
    record = routed_records[0]
    segments = record["segment_type_ids"]
    ids = record["input_ids"]
    turns = dict(assistant_turns(ids, segments, routed_tokenizer.decode))
    for flag, start, end in token_regions(segments):
        if flag or start < 2:
            continue
        lead = routed_tokenizer.decode(ids[start:start + 1])
        expected = start + (1 if lead.strip() == "" else 0)
        assert expected in turns


def test_training_features_equal_replay_features(routed_records, routed_tokenizer):
    """Train/serve parity, asserted directly.

    The whole point of --feature-mode chat: the features training builds for a
    generated step must be the ones chat replay hands the draft for that step.
    Two independent code paths, one layout — if they ever diverge, this fails.
    """
    from hopspec.infer.chained_eval import ChainedSpeculator

    target = TinyTargetModel(vocab_size=ROUTED_VOCAB_SIZE)
    record = routed_records[0]
    ids = record["input_ids"]
    segments = record["segment_type_ids"]
    buckets = record["recency_bucket_ids"]
    turn, end = assistant_turns(ids, segments, routed_tokenizer.decode)[0]

    speculator = ChainedSpeculator(
        None, target, device="cpu", prompt_prefix_ids=CHAT_PREFIX,
        prompt_suffix_ids=CHAT_SUFFIX, decode=routed_tokenizer.decode,
    )
    speculator.append_committed(ids[:turn], segments[:turn], buckets[:turn])
    speculator.open_turn()
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)
    speculator.commit(list(ids[:end]))

    training = chat_layout_features(
        target, collate([record], pad_id=0), CHAT_PREFIX, CHAT_SUFFIX,
        routed_tokenizer.decode,
    )[0]
    # inside the turn: assistant layout on both sides
    assert torch.allclose(speculator.feats[turn:end], training[turn:end], atol=1e-4)
    # before it: user layout on both sides
    assert torch.allclose(speculator.feats[:turn], training[:turn], atol=1e-4)
