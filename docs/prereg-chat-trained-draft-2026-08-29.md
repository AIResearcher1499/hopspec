# Prereg: chained re-run with a chat-trained draft — 2026-08-29

**FROZEN once `data/rounds_ct_*_1p7b.jsonl` exists.** Amendments appended with
a date; no threshold edited after a number is seen (`hopspc.md` §15).

Written before the re-run. Supersedes
`docs/prereg-chained-chat-replay-2026-08-28.md` for the arms below; that
prereg's chat-mode measurement definition is unchanged and still governs.

## 1. What changed since the last prereg

- **The harness is fixed** (`--replay-mode chat`) and stays fixed. Nothing
  about the replay definition changes here.
- **The draft is now trained on the layout it is served.** `--feature-mode
  chat` computes target features exactly as chat replay does — user layout
  outside a generated step, assistant layout inside it — via the same
  `assistant_turns` definition and the same `CachedTargetRunner`, so training
  and serving cannot drift. Wrapper tokens live only in the KV cache: they
  enter no label array and no loss term. New checkpoints
  `data/ckpt_{base,hop}_chat_1p7b.pt` record `feature_mode` alongside the
  weights, and `07` warns when a checkpoint's mode differs from the replay
  mode.
- **C1 is withdrawn** (`docs/c1-reconciliation-2026-08-28.md`). The probe that
  produced it omitted `resolve_recency_buckets_for_model`. Re-measured, the
  feature layout costs the draft nothing: 57/403 both ways, McNemar p=1.000.
  Retraining is still the right thing — the deployed loop hands EAGLE
  chat-layout features — but it is a **validity fix, expected to move the
  neural arm little**, not a repair of a known deficit.

## 2. Arms

Same shard, same 15%/seed-0 split, **never re-split**, gamma=4,
`--replay-mode chat`, checkpoint `data/ckpt_base_chat_1p7b.pt` wherever the
neural draft is used. Outputs `data/rounds_ct_<arm>_1p7b.jsonl`.

**Primary — the gates are read off these:**

| id | `--draft-source` | flags |
|---|---|---|
| a | neural | — |
| b | lookup | — |
| c | scaffold | `--scaffold-verb fit` |
| d | **routed** | `--scaffold-verb fit` |
| e | entropy | `--tune-entropy 0.25,0.5,1.0,1.5,2.0,3.0 --tune-max-records 6` |

The verb bet is **promoted to the primary routed configuration**. Fitted on
the train split only, as before; the fit on `data/shard_1p7b.jsonl`'s 102
train records is `Search` (118 occurrences vs `Finish` 102) and the run
prints what it fitted. Shared-prefix routed becomes secondary.

**Secondary — reported, never used to decide a gate:**

| id | `--draft-source` | flags |
|---|---|---|
| c2 | scaffold | shared prefix |
| d2 | routed | shared prefix |
| d3 | routed | `--scaffold-verb fit --chain` |
| f | entropy | `--entropy-scaffold` + the same tuning grid |

## 3. Gates

Unit of analysis: one mean accepted-tokens-per-round per held-out record, 18
pairs. Exact two-sided sign test on non-zero differences plus a paired
bootstrap (20 000 resamples, seed 0). Rounds are not byte-identical across
arms, so McNemar over positions does not apply between arms.

- **G1 — routed (d) ≥ 1.5× neural (a).** PASS requires ratio ≥ 1.5 **and** the
  bootstrap interval on the mean difference to exclude 0. **This is the number
  that replaces the provisional 2.98×**, now measured against a draft trained
  on the layout it is served.
- **G2 — routed (d) ≥ entropy-routed (e).** PASS requires the mean difference
  ≥ 0; an interval containing 0 is a TIE.

## 4. Pre-registered expectations

Recorded now so nothing below can be claimed as a prediction after the fact.

1. **The entropy gate will again be inert.** Measured last run: median target
   entropy at round starts 0.000 nats, p75 0.014, 87.5% of rounds below the
   smallest grid threshold. If the tuning table is again flat across the grid
   and the gate again routes nearly everything to the lookup, then **G2 is
   read as "scaffold coverage versus none", not as "structure signal beats
   entropy signal"**, and it is reported in exactly those words. The arm is
   kept and its gate statistics reported either way: the degeneracy is a
   finding (novelty verdict, safe claim 5), not an arm to drop quietly.
2. **The chat-trained draft will move the neural arm little**, because C1 was
   withdrawn. A large move would be a finding requiring explanation, not a
   confirmation.
3. **Whatever separates d from e will again sit at step openings** (bucket 0
   and the pre-retrieval bucket 6), because the FSM is the only source e
   lacks and it can only fire there. Bucket 0 carries ~22 rounds: descriptive
   only, quoted with its round count, never reported as a test.

## 5. The C1-resolution measurement

"Does the chat-trained draft change the neural arm materially?" is answered by
a dedicated paired probe, not by comparing tables:

Both checkpoints — `ckpt_base_1p7b.pt` (raw-trained) and
`ckpt_base_chat_1p7b.pt` (chat-trained) — evaluated **in one run**, under
**chat-layout features**, on the **same 403 held-out decode-phase positions**,
with `resolve_recency_buckets_for_model` applied (the step the old probe
omitted). Positions are byte-identical, so spec §10's paired McNemar applies.

**"Material" is defined now: exact two-sided McNemar p < 0.05.** Anything
else is reported as no material change.

## 6. Reporting requirements

- The **TEMPLATE/content split for every arm** (novelty verdict, safe claim 6).
  Any margin carried by TEMPLATE tokens is a systems margin and is stated as
  one; no content-quality claim may be read off it.
- Per-bucket means always with round counts (spec §10).
- Per-record paired statistics as in §3.
- The teacher-forced summaries of both retrained arms
  (`summary_{base,hop}_chat_1p7b.json`) are reported descriptively. The
  hop-signal arm is **not** being retuned and its re-measurement under a fixed
  harness does not reopen the §11 negative; it is reported only so the
  corrected-layout number exists.

## 7. Forbidden comparisons

No number from this run may be compared with: any raw-mode table
(`chained-routed-result-2026-08-28.md`, `data/rounds_*_1p7b.jsonl`), the
stale-checkpoint chat run (`chained-chat-result-2026-08-28.md`,
`data/rounds_chat_*_1p7b.jsonl`), or the §11 findings. The checkpoint changed;
§5's probe is the only sanctioned way to ask what that changed.

## 8. Stopping rules

- A seam failure at any wrapper/document boundary aborts the run rather than
  shifting labels silently.
- If a confound outside §4 appears, stop and report before building around it.
- The 500–1000-trajectory collection is scheduled **only if G1 holds** with
  the chat-trained draft.
