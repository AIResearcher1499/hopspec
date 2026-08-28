"""Collation and the EAGLE-aligned batch — the training-time index contract.

The alignment below was verified against EAGLE's SOURCE CODE, not its paper.
eagle/train/main.py builds:

    input_ids_target = input_ids[:, 1:]        # token shifted one ahead
    target           = hidden_state[:, 1:, :]  # regression target shifted one ahead
    new_data["hidden_state_big"] = hidden_state    # input feature NOT shifted

i.e. (f_t, e_{t+1}) -> f_{t+1}. Writing q for the index of the token being
PREDICTED (q = t + 2), q runs 2..T-1 and every returned tensor has length T-2:

    target_feats      = feats[:, :-2]      f_{q-2}, the draft's feature input
    feature_targets   = feats[:, 1:-1]     f_{q-1}, EAGLE's regression target
    draft_token_ids   = ids[:, 1:-1]       e_{q-1}
    labels/masks                          at q

The two off-by-ones this replaces both produced published-looking numbers:
pairing (f_{q-1}, e_{q-1}) collapsed training into feature reconstruction
(acceptance ~0.83, no headroom); taking labels at q-1 silently dropped all
247 first-post-hop-token predictions from the measurement.
"""

from __future__ import annotations

import argparse

import torch

from hopspec.data.schema import (
    NO_PRIOR_HOP_DISTANCE,
    NUM_RECENCY_BUCKETS,
    SegmentType,
)

# EAGLE's feature is the LM-head input = HF's hidden_states[-1] (post final
# norm). hidden_states[-2] is a different object: on Qwen3-4B its norm is 4.4x
# larger and it reproduces the target's own next token 24% of the time versus
# 81.5%. Fixing this alone improved pooled acceptance by +3.7 points (p=2e-16).
EAGLE_FEATURE_LAYER = -1


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a list of records to a tensor batch.

    Never pad recency_bucket_ids with 0: bucket 0 is "right after a hop", the
    rarest and most important class. Padding with 0 once mislabeled 257,814
    padding positions as bucket 0 versus 1,387 genuine ones — 186x
    noise-to-signal on the one embedding row the method most needs.
    """
    max_len = max(len(record["input_ids"]) for record in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        rows = [
            list(record[key]) + [value] * (max_len - len(record[key]))
            for record in batch
        ]
        return torch.tensor(rows, dtype=torch.long)

    attention_mask = torch.tensor(
        [
            [1] * len(record["input_ids"]) + [0] * (max_len - len(record["input_ids"]))
            for record in batch
        ],
        dtype=torch.long,
    )
    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": attention_mask,
        "segment_type_ids": pad("segment_type_ids", int(SegmentType.OTHER)),
        "recency_bucket_ids": pad("recency_bucket_ids", NUM_RECENCY_BUCKETS - 1),
        "recency_distances": pad("recency_distances", NO_PRIOR_HOP_DISTANCE),
    }


def eagle_aligned_batch(
    target_model,
    batch: dict[str, torch.Tensor],
    feature_layer: int = EAGLE_FEATURE_LAYER,
    features: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """`features` overrides the target forward with a precomputed [B, T, H]
    tensor over DOCUMENT positions (see `chat_layout_features`). Labels and
    slicing are untouched by it, so no wrapper token can reach the loss or a
    label array. Default None keeps the original single-forward behaviour."""
    device = next(target_model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if input_ids.shape[1] < 3:
        raise ValueError("EAGLE alignment needs sequences of length >= 3")

    if features is None:
        with torch.no_grad():
            outputs = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        feats = outputs.hidden_states[feature_layer].float()
    else:
        if features.shape[:2] != input_ids.shape:
            raise ValueError(
                f"features must be [B, T, H] over document positions, got "
                f"{tuple(features.shape)} for input_ids {tuple(input_ids.shape)}"
            )
        feats = features.to(device).float()

    segment_type_ids = batch["segment_type_ids"].to(device)
    recency_bucket_ids = batch["recency_bucket_ids"].to(device)
    recency_distances = batch["recency_distances"].to(device)

    # hop_index must be computed on the FULL sequence and only then sliced:
    # each closed passage contributes exactly one distance == 0 position, and
    # counting in the already-sliced array would miss a hop closing at
    # absolute position 0 or 1 and shift every later index.
    hop_index_full = (recency_distances == 0).long().cumsum(dim=1) - 1

    target_token_ids = input_ids[:, 2:].masked_fill(attention_mask[:, 2:] == 0, -100)

    return {
        "target_feats": feats[:, :-2],
        "feature_targets": feats[:, 1:-1],
        "draft_token_ids": input_ids[:, 1:-1],
        # Deliberately q-1, not q: the model may only be fed things known at
        # inference. The recency bucket at q IS known (the tracker maintains
        # it); the segment type of a not-yet-generated token is not, while the
        # previous token's is — and that carries the "a passage just ended"
        # transition anyway.
        "segment_ids_input": segment_type_ids[:, 1:-1],
        "recency_buckets": recency_bucket_ids[:, 2:],
        "recency_distances": recency_distances[:, 2:],
        "hop_index": hop_index_full[:, 2:],
        "target_token_ids": target_token_ids,
        "segment_ids_target": segment_type_ids[:, 2:],
        "attention_mask_target": attention_mask[:, 2:],
    }


def chat_layout_features(
    target_model,
    batch: dict[str, torch.Tensor],
    prefix_ids: list[int],
    suffix_ids: list[int],
    decode,
    feature_layer: int = EAGLE_FEATURE_LAYER,
) -> torch.Tensor:
    """Target features in the layout the DEPLOYED loop produces.

    A position inside a generated step is computed behind the assistant
    separator; every other position is computed inside the user message. That
    is exactly what chat-mode replay hands the draft — and it is produced here
    by the SAME `CachedTargetRunner` and the SAME `assistant_turns` definition,
    so training features and serving features cannot drift apart.

    Returns [B, T, H] over DOCUMENT positions only, padding rows left at zero:
    the wrapper exists in the cache and nowhere else, so it can never enter a
    label array or the loss.

    Cost: the wrapper prefix once per record, plus one extra pass over each
    generated step (~7% of tokens on the pilot shard) — not a forward per step.
    """
    from hopspec.infer.chained_eval import CachedTargetRunner, assistant_turns

    device = next(target_model.parameters()).device
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    segment_type_ids = batch["segment_type_ids"]
    batch_size, seq_len = input_ids.shape
    hidden_size = target_model.config.hidden_size
    out = torch.zeros(batch_size, seq_len, hidden_size, device=device,
                      dtype=torch.float32)
    runner = CachedTargetRunner(target_model, str(device), feature_layer)

    for row in range(batch_size):
        length = int(attention_mask[row].sum())
        if length == 0:
            continue
        ids = input_ids[row, :length].tolist()
        segments = segment_type_ids[row, :length].tolist()
        feats = torch.zeros(length, hidden_size, device=device, dtype=torch.float32)
        runner.reset()
        if prefix_ids:
            runner.extend(list(prefix_ids))
        cursor = 0
        # No suffix means no assistant separator, i.e. the raw layout: there is
        # no turn to open and the whole document is one user-layout pass.
        turns = assistant_turns(ids, segments, decode) if suffix_ids else []
        for turn, end in turns:
            if turn > cursor:                       # user-layout run
                feats[cursor:turn] = runner.extend(ids[cursor:turn])[0]
                cursor = turn
            runner.extend(list(suffix_ids))         # open the assistant turn
            feats[turn:end] = runner.extend(ids[turn:end])[0]
            # Drop the separator and the step, then replay the step inside the
            # user message so later positions continue the user-layout chain —
            # the same close-turn/restore the replay loop performs.
            runner.rollback(len(prefix_ids) + turn)
            runner.extend(ids[turn:end])
            cursor = end
        if cursor < length:
            feats[cursor:] = runner.extend(ids[cursor:])[0]
        out[row, :length] = feats
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone draft-model training")
    parser.add_argument("--target-model-name", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=16.0)
    parser.add_argument("--feature-layer", type=int, default=EAGLE_FEATURE_LAYER)
    parser.add_argument("--v-w", type=float, default=1.0)
    parser.add_argument("--p-w", type=float, default=0.1)
    parser.add_argument("--hop-signal", action="store_true")
    parser.add_argument("--checkpoint-out", default=None)
    return parser


def train(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel
    from hopspec.train.ablation import load_records, train_draft_model

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, torch_dtype="auto", attn_implementation="sdpa"
    ).to(args.device)
    target_model.eval()

    records = load_records(args.trajectory_file)
    config = HopSpecDraftConfig(
        target_hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
    )
    draft_model = HopSpecDraftModel.from_target_embedding(
        config, target_model.get_input_embeddings()
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    train_draft_model(
        draft_model, target_model, records, pad_id, args.device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every, hop_signal_enabled=args.hop_signal,
        alpha=args.alpha, tau=args.tau, feature_layer=args.feature_layer,
        v_w=args.v_w, p_w=args.p_w,
    )
    if args.checkpoint_out:
        _torch.save(
            {"state_dict": draft_model.state_dict(), "config": vars(config)},
            args.checkpoint_out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
