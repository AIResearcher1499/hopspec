import pytest
import torch

from hopspec.data.schema import NUM_RECENCY_BUCKETS, NUM_SEGMENT_TYPES
from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel

from conftest import HIDDEN_SIZE, VOCAB_SIZE, TinyTargetModel


def make_model(tiny=None):
    tiny = tiny or TinyTargetModel()
    config = HopSpecDraftConfig(
        target_hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
        draft_hidden_size=32, num_heads=4,
    )
    return HopSpecDraftModel.from_target_embedding(config, tiny.get_input_embeddings()), tiny


def make_inputs(batch=2, seq=5):
    torch.manual_seed(1)
    return (
        torch.randint(0, VOCAB_SIZE, (batch, seq)),
        torch.randn(batch, seq, HIDDEN_SIZE),
        torch.randint(0, NUM_SEGMENT_TYPES, (batch, seq)),
        torch.randint(0, NUM_RECENCY_BUCKETS, (batch, seq)),
    )


def test_config_aux_sizes_derive_from_schema():
    config = HopSpecDraftConfig(target_hidden_size=8, vocab_size=10)
    assert config.num_segment_types == NUM_SEGMENT_TYPES
    assert config.num_recency_buckets == NUM_RECENCY_BUCKETS


def test_from_target_embedding_copies_and_freezes():
    model, tiny = make_model()
    assert not model.token_embedding.weight.requires_grad
    assert torch.equal(model.token_embedding.weight, tiny.get_input_embeddings().weight)


def test_frozen_embedding_is_a_copy_not_shared():
    model, tiny = make_model()
    with torch.no_grad():
        tiny.get_input_embeddings().weight.add_(1.0)
    assert not torch.equal(model.token_embedding.weight, tiny.get_input_embeddings().weight)


def test_forward_output_shape():
    model, _ = make_model()
    out = model(*make_inputs())
    assert out.shape == (2, 5, HIDDEN_SIZE)


def test_forward_is_causal():
    model, _ = make_model()
    model.eval()
    tokens, feats, segments, buckets = make_inputs(batch=1)
    out1 = model(tokens, feats, segments, buckets)
    tokens2 = tokens.clone()
    tokens2[0, -1] = (tokens2[0, -1] + 1) % VOCAB_SIZE
    out2 = model(tokens2, feats, segments, buckets)
    # Changing the last token must not change any earlier position.
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-6)


def test_aux_embeddings_affect_output():
    model, _ = make_model()
    model.eval()
    tokens, feats, segments, buckets = make_inputs(batch=1)
    out1 = model(tokens, feats, segments, buckets)
    out2 = model(tokens, feats, segments, (buckets + 1) % NUM_RECENCY_BUCKETS)
    assert not torch.allclose(out1, out2)


def test_forward_rejects_bad_feature_shape():
    model, _ = make_model()
    tokens, feats, segments, buckets = make_inputs()
    with pytest.raises(ValueError):
        model(tokens, feats[:, :, :-1], segments, buckets)


def test_forward_rejects_bad_label_shape():
    model, _ = make_model()
    tokens, feats, segments, buckets = make_inputs()
    with pytest.raises(ValueError):
        model(tokens, feats, segments[:, :-1], buckets)


def test_forward_casts_bfloat16_features():
    model, _ = make_model()
    tokens, feats, segments, buckets = make_inputs()
    out = model(tokens, feats.to(torch.bfloat16), segments, buckets)
    assert out.dtype == torch.float32


def test_predict_logits_casts_bfloat16_head():
    model, tiny = make_model()
    features = torch.randn(1, 3, HIDDEN_SIZE)
    head = tiny.get_output_embeddings().weight.to(torch.bfloat16)
    logits = HopSpecDraftModel.predict_logits(features, head)
    assert logits.dtype == torch.float32
    assert logits.shape == (1, 3, VOCAB_SIZE)


def test_predict_logits_matches_matmul():
    model, tiny = make_model()
    features = torch.randn(1, 3, HIDDEN_SIZE)
    weight = tiny.get_output_embeddings().weight
    logits = HopSpecDraftModel.predict_logits(features, weight)
    assert torch.allclose(logits, features @ weight.t(), atol=1e-6)


def test_no_norm_before_head():
    # EAGLE-1 applies the head to the raw predicted feature. Scaling the
    # feature must scale logits linearly — any normalization would break this.
    model, tiny = make_model()
    features = torch.randn(1, 2, HIDDEN_SIZE)
    weight = tiny.get_output_embeddings().weight
    assert torch.allclose(
        HopSpecDraftModel.predict_logits(features * 2, weight),
        HopSpecDraftModel.predict_logits(features, weight) * 2,
        atol=1e-5,
    )


def test_embedding_dim_mismatch_rejected():
    config = HopSpecDraftConfig(target_hidden_size=HIDDEN_SIZE + 1, vocab_size=VOCAB_SIZE)
    with pytest.raises(ValueError):
        HopSpecDraftModel.from_target_embedding(
            config, TinyTargetModel().get_input_embeddings()
        )
