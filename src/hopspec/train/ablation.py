"""Shared training/evaluation used by BOTH runner scripts (03 and 04), so the
only thing differing between arms is the switch under test."""

from __future__ import annotations

import json
import random
from typing import Iterable, Iterator

import torch

from hopspec.eval.diagnostic import decode_phase_mask, resolve_recency_buckets_for_model
from hopspec.train.loss import hop_aware_ce_loss, hop_aware_feature_loss
from hopspec.train.target_policy import LossTargetPolicy
from hopspec.train.train_draft import EAGLE_FEATURE_LAYER, collate, eagle_aligned_batch


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "question_id" not in record:
                raise ValueError(f"{path}:{line_number}: record has no question_id")
            records.append(record)
    return records


def split_records(
    records: list[dict], eval_fraction: float = 0.15, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Deterministic in-shard train/held-out split by question_id (distinct
    from the benchmark-level leakage split in question_split.py)."""
    question_ids = sorted({record["question_id"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(question_ids)
    num_eval = int(round(len(question_ids) * eval_fraction))
    heldout_ids = set(question_ids[:num_eval])
    train = [r for r in records if r["question_id"] not in heldout_ids]
    heldout = [r for r in records if r["question_id"] in heldout_ids]
    return train, heldout


def batches(records: list[dict], batch_size: int) -> Iterator[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _lm_head_weight(target_model) -> torch.Tensor:
    return target_model.get_output_embeddings().weight


def train_draft_model(
    draft_model,
    target_model,
    train_records: list[dict],
    pad_id: int,
    device: str,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
    log_every: int = 10,
    hop_signal_enabled: bool = True,
    alpha: float = 2.0,
    tau: float = 16.0,
    loss_target_policy: LossTargetPolicy | None = None,
    feature_layer: int = EAGLE_FEATURE_LAYER,
    v_w: float = 1.0,
    p_w: float = 0.1,
    feature_builder=None,
) -> None:
    policy = loss_target_policy if loss_target_policy is not None else LossTargetPolicy()
    draft_model.to(device)
    draft_model.train()
    lm_head_weight = _lm_head_weight(target_model)
    optimizer = torch.optim.AdamW(
        [p for p in draft_model.parameters() if p.requires_grad], lr=lr
    )
    step = 0
    for epoch in range(epochs):
        for raw_batch in batches(train_records, batch_size):
            batch = collate(raw_batch, pad_id)
            aligned = eagle_aligned_batch(
                target_model, batch, feature_layer=feature_layer,
                features=None if feature_builder is None
                else feature_builder(target_model, batch),
            )
            segments_in, buckets_in = resolve_recency_buckets_for_model(
                aligned["segment_ids_input"], aligned["recency_buckets"], hop_signal_enabled
            )
            predicted = draft_model(
                aligned["draft_token_ids"], aligned["target_feats"], segments_in, buckets_in
            )
            logits = draft_model.predict_logits(predicted, lm_head_weight)
            targets = policy.loss_targets(aligned)
            distances = aligned["recency_distances"]
            feature_loss = hop_aware_feature_loss(
                predicted, aligned["feature_targets"], distances,
                aligned["attention_mask_target"].bool(), alpha=alpha, tau=tau,
            )
            token_loss = hop_aware_ce_loss(logits, targets, distances, alpha=alpha, tau=tau)
            loss = v_w * feature_loss + p_w * token_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if log_every and step % log_every == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"(feat {feature_loss.item():.4f} tok {token_loss.item():.4f})"
                )


def evaluate_draft_model(
    draft_model,
    target_model,
    heldout_records: list[dict],
    pad_id: int,
    device: str,
    batch_size: int = 2,
    hop_signal_enabled: bool = True,
    feature_layer: int = EAGLE_FEATURE_LAYER,
    feature_builder=None,
) -> dict[str, list[int]]:
    """Returns RAW PER-POSITION COLUMNS, not a summary — re-bucketing, cohort
    filters and novelty filters must be offline operations so the expensive
    pass never has to be repeated for an analysis change.

    Deliberately does NOT accept a loss-target policy: evaluation has to be
    identical across all arms or the comparison is meaningless.
    """
    draft_model.to(device)
    draft_model.eval()
    lm_head_weight = _lm_head_weight(target_model)
    columns: dict[str, list[int]] = {
        "recency_distance": [],
        "recency_bucket": [],
        "hop_index": [],
        "target_token": [],
        "correct": [],
        "record_index": [],
    }
    record_base = 0
    with torch.no_grad():
        for raw_batch in batches(heldout_records, batch_size):
            batch = collate(raw_batch, pad_id)
            aligned = eagle_aligned_batch(
                target_model, batch, feature_layer=feature_layer,
                features=None if feature_builder is None
                else feature_builder(target_model, batch),
            )
            segments_in, buckets_in = resolve_recency_buckets_for_model(
                aligned["segment_ids_input"], aligned["recency_buckets"], hop_signal_enabled
            )
            predicted = draft_model(
                aligned["draft_token_ids"], aligned["target_feats"], segments_in, buckets_in
            )
            logits = draft_model.predict_logits(predicted, lm_head_weight)
            predicted_tokens = logits.argmax(dim=-1)
            correct = predicted_tokens == aligned["target_token_ids"]
            mask = decode_phase_mask(
                aligned["segment_ids_target"], aligned["attention_mask_target"]
            )
            for row in range(mask.shape[0]):
                positions = mask[row].nonzero(as_tuple=True)[0]
                # Always group by the REAL bucket, whatever the model saw.
                columns["recency_distance"].extend(
                    aligned["recency_distances"][row, positions].tolist()
                )
                columns["recency_bucket"].extend(
                    aligned["recency_buckets"][row, positions].tolist()
                )
                columns["hop_index"].extend(aligned["hop_index"][row, positions].tolist())
                columns["target_token"].extend(
                    aligned["target_token_ids"][row, positions].tolist()
                )
                columns["correct"].extend(
                    correct[row, positions].long().tolist()
                )
                columns["record_index"].extend([record_base + row] * len(positions))
            record_base += len(raw_batch)
    return columns
