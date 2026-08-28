# Chained-replay round artifacts — what each one is measured under

Not a quarantine: nothing here is invalid. This file exists so that nobody
quotes a chained accepted-per-round number without knowing which measurement
produced it, and so that no one compares across the two (`hopspc.md` §15 — the
measurement definition has now changed once more).

Every round row written from 2026-08-28 onward carries a `replay_mode` field.
Rows written before that date have none; they are all `raw`.

## `raw` mode — the target sees the recorded document verbatim

| file | arm |
|---|---|
| `rounds_baseline_pilot.jsonl` | early pilot, baseline draft |
| `rounds_base_1p7b.jsonl` | §11.4 chained, baseline arm (hop signal off) |
| `rounds_hop_1p7b.jsonl` | §11.4 chained, hop-signal arm |
| `rounds_neural_1p7b.jsonl` | routed-drafting run, neural arm |
| `rounds_lookup_1p7b.jsonl` | routed-drafting run, scoped lookup |
| `rounds_scaffold_1p7b.jsonl` | routed-drafting run, scaffold FSM |
| `rounds_routed_1p7b.jsonl` | routed-drafting run, scaffold→lookup→neural |
| `rounds_routedchain_1p7b.jsonl` | as above, sources chained within a round |
| `rounds_entropy_1p7b.jsonl` | ReSpec-style entropy routing |
| `rounds_structent_1p7b.jsonl` | structure + entropy |

**Known defect, shared by all of them.** The agent never saw the document these
were replayed over: at collection every step went through
`apply_chat_template(system + user(context so far))`, and the system prompt is
what tells the model to open a ReAct step. Measured on the 18 held-out records:
the raw-mode target reproduces the recorded token at **2/40 = 0.05** of
step-opening positions versus **409/588 = 0.70** elsewhere.

Consequence: comparisons *among* raw-mode arms are internally consistent (they
all carry the same handicap) but no raw-mode number is an absolute statement
about the deployed loop, and any arm whose value sits at a step opening — the
scaffold FSM above all — is measured where the harness is blind.

## `chat` mode, 1000-record shard, re-run with `accepted_ids` (CURRENT REFERENCE)

Files `rounds_r2_*_1p7b.jsonl`, checkpoint `ckpt_base_chat_scale2.pt`, same
shard. The only set carrying `accepted_ids`, so the only one from which a true
template/content split can be computed (prereg amendment 2026-08-29b).
Supersedes `rounds_sc_*`.

## `chat` mode, 1000-record shard, first execution (SUPERSEDED)

`rounds_sc_*` carry no `accepted_ids`; their template/content columns label by
proposing source, not token type, and are relabelled accordingly. Retained
because both executions' gate outcomes are reported side by side.

Files `rounds_sc_*_1p7b.jsonl`, shard `shard_1p7b_scale.jsonl`
(md5 `3b50e71c5526176622fca24cb7449ac4`), checkpoints
`ckpt_{base,hop}_chat_scale.pt`, prereg `docs/prereg-scaleup-2026-08-29.md`,
results `docs/scaleup-result-2026-08-29.md`.

Collected on CUDA with `--batch-size 8`, so **not reproducible by a sequential
run** (3/5 trajectories matched `--batch-size 1` in the validation). 150
held-out records. This is the current reference measurement and may not be
compared with any set below.

## `chat` mode, 120-record shard, chat-trained draft (superseded)

Files `rounds_ct_*_1p7b.jsonl`, checkpoints `ckpt_{base,hop}_chat_1p7b.pt`
(each records `feature_mode: chat`), prereg
`docs/prereg-chat-trained-draft-2026-08-29.md`, results
`docs/chat-trained-draft-result-2026-08-29.md`.

The draft is trained on the same layout it is served — `--feature-mode chat`
computes target features through the same `assistant_turns` definition and the
same `CachedTargetRunner` the replay uses. This is the current reference
measurement. It may not be compared with either set below.

## `chat` mode, raw-trained draft (superseded)

Files `rounds_chat_*_1p7b.jsonl`, prereg
`docs/prereg-chained-chat-replay-2026-08-28.md`, results
`docs/chained-chat-result-2026-08-28.md`.

The harness is the deployed loop, but the draft was trained on raw-document
features. Superseded by the `rounds_ct_*` set; kept because the C1
reconciliation and the entropy-degeneracy finding are read off it.
