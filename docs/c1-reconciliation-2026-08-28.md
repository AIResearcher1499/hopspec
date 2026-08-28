# C1 reconciliation: 0.0471 vs 0.141 — 2026-08-28

Answers the blocking question on confound C1 of
`docs/prereg-chained-chat-replay-2026-08-28.md`. Written before any
retraining work started.

**Verdict: the C1 probe was mis-conditioned. The 0.141 number is correct and
the raw-mode evaluation pipeline is not implicated. Re-measured correctly, the
C1 effect is nil — so G1 is not inflated by C1 and the "provisional" caveat as
originally written is withdrawn.**

## 1. What the C1 probe did, exactly

For each of the 18 held-out records it took the record's own
`input_ids`/`segment_type_ids`/`recency_bucket_ids`, forwarded the target over
the **unwrapped document** (`raw`) or over `chat_prefix + document` with each
generated step re-forwarded behind the assistant separator (`chat`), and fed
the draft the EAGLE-aligned slices

```
tokens   = ids[1:-1]      features = feats[:-2]
segments = segs[1:-1]     buckets  = recency_bucket_ids[2:]
targets  = ids[2:]        mask     = decode_phase_mask(segs[2:])
```

Those slices match `eagle_aligned_batch` exactly, and the positions are the
same 403 decode-phase held-out positions. **The alignment was right. The
conditioning was not.**

`evaluate_draft_model` does one more thing the probe omitted:

```python
segments_in, buckets_in = resolve_recency_buckets_for_model(
    aligned["segment_ids_input"], aligned["recency_buckets"], hop_signal_enabled
)
```

`data/ckpt_base_1p7b.pt` is the BASELINE arm, trained with
`hop_signal_enabled=False`, and in that arm `resolve_recency_buckets_for_model`
returns `zeros_like(segments)` and `full_like(buckets, 0)` — it zeroes **both**
aux embedding tables, not just the recency one. That checkpoint has therefore
only ever seen `segment=0, bucket=0`. The probe handed it the real segment
types and real recency buckets, i.e. aux embeddings it was never trained to
use. The 3x gap is a train/inference mismatch **the probe itself introduced**,
present in both of its arms.

## 2. Re-measurement

Same 403 positions, same checkpoint, same slices; the only change is applying
`resolve_recency_buckets_for_model(..., hop_signal_enabled=False)`.

| conditioning | raw features | chat features |
|---|---|---|
| omitted (the original C1 probe) | 19/403 = 0.0471 | 16/403 = 0.0397 |
| applied (what `evaluate_draft_model` does) | **57/403 = 0.1414** | **57/403 = 0.1414** |

Raw + conditioned reproduces `data/raw_base_1p7b.json` (n=403, 0.1414) to the
digit. **The 0.141 number and the pipeline that produced it are vindicated**,
so the escalation condition ("if the 0.141 number is the wrong one, stop —
that implicates the raw-mode evaluation pipeline") does not fire.

## 3. Is the new null real, or is `chat_feats` silently returning raw features?

Checked, because an identical count is exactly what a plumbing bug looks like
(spec §15: decode and look, do not trust an aggregate):

- the assistant-layout overwrite fires on **646 of 9,210** document positions
  (7.0%) — the generated steps, as intended;
- the two feature tensors genuinely differ: mean relative L2 **0.310**, mean
  cosine similarity **0.937**;
- the draft's predictions differ at **half** the measured positions — only
  200/403 agree.

So the features are different and the model reacts to them. It is the
*accuracy* that does not move. Both layouts are scored on byte-identical
positions, which is the case spec §10 reserves paired McNemar for:

| | chat correct | chat wrong |
|---|---|---|
| **raw correct** | 40 | 17 |
| **raw wrong** | 17 | 329 |

Exact two-sided McNemar on the 34 discordant pairs: **p = 1.000**. The layout
shift reshuffles *which* positions the draft gets right without changing *how
many*. For scale, the SE of a single rate at n=403 is 0.017.

## 4. How provisional is G1, now?

The prereg said C1 "weakens the neural arm and INFLATES this gate", so G1's
2.98x was to be read as provisional pending a chat-trained draft. **That
specific caveat is withdrawn**: the measured penalty is 0.0000 [McNemar
p=1.000], not the −0.020 the mis-conditioned probe suggested. On this
evidence G1's 2.98x is not inflated by the feature layout, and the routed
arm's margin over the neural arm stands as measured.

Three limits on that statement, none of which restore the original caveat:

1. **Regime.** This is a teacher-forced, single-step measurement; the chained
   replay lets the draft chain on its own predicted features. The null is
   measured where it can be measured cleanly and strongly suggests — but does
   not prove — that the chained neural arm is unaffected. Item 1's retrained
   checkpoint tests it directly.
2. **One weak checkpoint.** At 0.141 top-1 the draft may simply be too weak
   for a 0.31-relative-L2 feature perturbation to register. A stronger draft
   could be more layout-sensitive; the null should not be generalised to one.
3. **Retraining is still correct.** It is a validity fix — the deployed loop
   hands EAGLE chat-layout features, so that is what the draft should be
   trained on — and it stays scheduled before the scale-up. What changes is
   the *expectation*: item 1 should be expected to move the neural arm
   little, and a large move would itself be a finding worth explaining.

## 5. What this does not touch

The chained-replay arms were **correctly** conditioned: `ChainedSpeculator.propose`
calls `resolve_recency_buckets_for_model(segments, bucket_inputs,
self.hop_signal_enabled)` and every arm ran without `--hop-signal`. The bug
was confined to the standalone C1 side-probe. No arm result in
`docs/chained-chat-result-2026-08-28.md` changes; only that document's C1
paragraph and its "provisional" framing do, and both are annotated there.
