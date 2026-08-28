# Prereg: chat-mode chained replay — 2026-08-28

**FROZEN once `data/rounds_chat_*_1p7b.jsonl` exists.** Amendments are appended
with a date; no threshold in this file may be edited after a number is seen
(repo rule, `hopspc.md` §15).

Written before any code for the change it describes.

## 1. Why: this is a validity fix, not an optimization

`replay_record` today feeds the target the recorded document verbatim. The
agent never saw that document. At collection every step went through
`HFTargetLLM.generate`, i.e.

```
apply_chat_template([system=SYSTEM_PROMPT, user=<context so far>],
                    add_generation_prompt=True, enable_thinking=False)
```

and the model generated one step inside a fresh assistant turn. The system
prompt is what tells it to write `Thought:` / `Action:` at all.

This is the same bug family as the spec §4 "worst bug in the project's
history" — target hidden states computed over a document that never existed —
relocated from `collect` into `replay`. Measured consequence in raw mode
(`docs/chained-routed-result-2026-08-28.md`): the replay target reproduces the
recorded token at **2/40 = 0.05** of step-opening positions versus
**409/588 = 0.70** elsewhere. Fixing it is required regardless of which draft
arm benefits.

## 2. What changes

A **mode**, not a replacement: `--replay-mode {raw,chat}`, default `raw`.

- `raw` must reproduce today's behaviour bit for bit. Every existing test stays
  pinned to it and every existing artifact stays reproducible.
- `chat` re-renders the wrapper at every step boundary, mirroring
  `HFTargetLLM.generate` exactly, including the
  `enable_thinking=False` → `TypeError` fallback for templates that reject the
  kwarg.

### 2.1 Definition of `chat` mode

Measured for `Qwen/Qwen3-1.7B` + `SYSTEM_PROMPT`: the wrapper splits into a
fixed **82-token prefix** (`<|im_start|>system … <|im_start|>user\n`) and a
fixed **9-token suffix** (`<|im_end|>\n<|im_start|>assistant\n<think>\n\n
</think>\n\n`).

Per record:

1. Prefill the target cache with the prefix. The prefix is NOT part of the
   document: it gets no segment label, no bucket, and no entry in `feats`.
2. Walk the regions exactly as today. Inserted regions (QUESTION,
   RETRIEVED_PASSAGE) are committed verbatim in **user layout**.
3. For each generated region:
   a. Commit any **leading whitespace-only tokens verbatim** — they were
      prompt, not generation (see §2.2).
   b. Open the assistant turn: extend the cache with the suffix.
   c. Run speculative rounds until the region's recorded length is produced,
      identically to raw mode.
   d. Close the turn: crop the cache back to `len(prefix) + turn_start`,
      dropping the suffix and every speculated token.
   e. Restore the recorded region tokens in user layout, as today, so every
      later region starts on the recorded rails.

Draft (EAGLE) features follow the same layout: positions committed inside a
turn carry assistant-layout features, every other position carries
user-layout features. That is what the deployed loop hands EAGLE.

### 2.2 The leading-whitespace rule, and why it exists

The user content at a step boundary is `decode(document[:region_start])`.
Checked over the whole 120-record shard against the context reconstructed
from each record's own steps: **3 of 260 boundaries disagree**, all the same
case — the `"\n"` closing the question did not merge into the question's last
token, so it is a separate TEMPLATE token that lands at the head of the
generated region although it was prompt. Committing leading whitespace-only
tokens as prompt makes all 260 boundaries agree. It is a no-op for the other
257.

Consequence: `chat` speculates over slightly fewer tokens than `raw` on those
records. That is correct, and it is one more reason the two modes' numbers are
not comparable.

### 2.3 Cache

The stable common prefix is cached, not reset per step: across steps the cache
holds `prefix + document[:turn_start]`, which only grows. Only the 9-token
suffix and the current step's tokens are recomputed per step. Correctness is
pinned by extending the existing cached-equals-uncached tests to the chat
layout, in addition to the existing "speculated tokens never persist" test,
which must pass in both modes.

## 3. What does NOT change

The proposers (`routed_draft.py`), the split (15%, seed 0 — **never
re-split**), gamma=4, the 18 held-out records, the fit-on-train-only rule for
the scaffold's opening literal and the entropy threshold, the round-row
schema, and the gate structure below.

## 4. Arms

Primary (the gate is read off these):

| id | `--draft-source` | extra flags |
|---|---|---|
| a | neural | — |
| b | lookup | — |
| c | scaffold | — |
| d | **routed** | — |
| e | entropy | `--tune-entropy 0.25,0.5,1.0,1.5,2.0,3.0 --tune-max-records 6` |

Secondary (reported, never used to decide the gate):

| id | `--draft-source` | extra flags |
|---|---|---|
| c2 | scaffold | `--scaffold-verb fit` |
| d2 | routed | `--chain` |
| d3 | routed | `--scaffold-verb fit` |
| f | entropy | `--entropy-scaffold` + the same tuning grid |

All with `--replay-mode chat`, `--gamma 4`, checkpoint
`data/ckpt_base_1p7b.pt` (hop signal off) wherever the neural draft is used.
Outputs: `data/rounds_chat_<arm>_1p7b.jsonl`.

## 5. Gates (structure unchanged from the plan)

Unit of analysis: **one mean accepted-tokens-per-round per held-out record**,
18 pairs. Rounds are not byte-identical across arms, so paired McNemar over
positions does not apply; the pairing is per record. Tests: exact two-sided
sign test on non-zero differences, plus a paired bootstrap (20 000 resamples,
seed 0) for a 95% interval on the mean difference.

- **G1 — routed ≥ 1.5× neural.** Read on the per-record means of arms d vs a.
  PASS requires the ratio ≥ 1.5 **and** the bootstrap interval on the mean
  difference to exclude 0.
- **G2 — routed ≥ entropy-routed.** Arms d vs e. PASS requires the mean
  difference ≥ 0. A bootstrap interval containing 0 is recorded as a TIE, not
  a pass and not a loss.

Only if both gates are stable do we schedule the 500–1000-trajectory
collection and retraining.

## 6. Pre-registered expectation

Structure routing and entropy routing route between the same scoped lookup and
the same neural draft; the only structural source the entropy arm lacks is the
scaffold FSM. The FSM can only fire at a step opening, and step openings are
exactly the positions raw mode could not verify. **So we predict that whatever
separation exists between arms d and e is concentrated in bucket 0** (distance
0, the first token after a closed passage), and that arms d and e are
indistinguishable in buckets 2–6.

Directional prediction, recorded now: arm d's bucket-0 mean accepted/round
exceeds arm e's. Bucket 0 carries roughly 22 rounds on 18 held-out records, so
this is **descriptive only** — it is not powered to be a test, and it will not
be reported as one. Per §15 it will be quoted with its round count.

## 7. Confounds pre-registered BEFORE the run

- **C1 — the draft checkpoint is trained for the wrong feature layout.**
  `data/ckpt_base_1p7b.pt` was trained on features from a raw-document forward
  (`eagle_aligned_batch` forwards `input_ids`). Chat mode hands the draft
  chat-layout features. Measured teacher-forced decode-phase top-1 on the 18
  held-out records, same draft, same positions, only the feature source
  changed: **0.0471 → 0.0273 (19 → 11 of 403)**.
  Chat mode is the correct measurement; the checkpoint is stale for it. The
  fix is retraining the draft on chat-layout features, which is scheduled
  after the gate.
  Direction of the bias, recorded now: this **weakens the neural arm**, which
  **inflates G1** (routed vs neural) and **leaves G2 unaffected** (arms d and
  e share the same neural fallthrough). Therefore **G1 in chat mode is
  provisional pending a chat-trained draft, and G2 is the gate that carries
  weight.**
- **C2 — bucket 0 is tiny** (~22 rounds). Descriptive only, always quoted with
  n. Spec §10: never a bucket acceptance number without its round count.
- **C3 — no cross-mode comparison, ever.** No number produced in `chat` mode
  may be compared with any table in
  `docs/chained-routed-result-2026-08-28.md`, with `data/rounds_*_1p7b.jsonl`
  from the raw runs, or with the §11 findings. The measurement definition
  changed; §15 forbids it. Raw-mode artifacts are annotated accordingly rather
  than deleted.

## 8. Stopping rules

- If chat mode surfaces a confound not listed in §7, stop and report before
  building around it.
- If `raw` mode stops reproducing today's numbers bit for bit, the mode switch
  is wrong: stop, do not reinterpret.
- If the chat-mode step-opening reproduction rate does not materially improve
  over raw mode's 0.05, the fix did not work; report that rather than reading
  the gates.

## 9. Analysis plan

Fixed before the run: `summarize_rounds` (per bucket), `summarize_rounds_by_source`
(per source, template vs content split), `record_means` (per record), then §5's
paired tests. Results go to `docs/chained-chat-result-2026-08-XX.md`. The
per-source table is reported for every arm, and the scaffold's accepted count
is always labelled TEMPLATE.

---

## Amendment 2026-08-28 (after the run, appended — nothing above was edited)

Two things were found during the run that §7 did not anticipate. Recorded here
so the record shows plainly what was predicted and what was not.

1. **The entropy comparator's gate is degenerate in chat mode — POST-HOC.**
   Measured over the 367 round starts of the entropy arm, the target's
   entropy is essentially zero (median 0.000 nats, p75 0.014, p95 0.631,
   max 1.618), so 87.5% of rounds fall below the smallest threshold in the
   pre-registered grid and 100% below 2.0. The tuning table is flat across
   0.25–3.0 for exactly this reason. The comparator therefore reduces to
   "always try the scoped lookup, fall back to the neural draft": its gate
   contributes nothing.
   Consequence for G2, which was pre-registered without knowing this: G2
   measures **scaffold versus no scaffold**, not structure-signal versus
   entropy-signal. It is reported that way. This does not invalidate the
   measurement; it reinterprets what the comparator is.

2. **Arm d's margin over arm e is entirely TEMPLATE — POST-HOC.** Routed
   accepts 82 TEMPLATE + 182 content tokens; entropy accepts 0 + 227. Routed
   accepts FEWER content tokens than the comparator. The systems quantity
   (tokens per target verification) is still what the gate is defined on and
   the gate stands, but no content claim may be read off it. §7 C2 warned
   about bucket 0's size; it did not warn about this.

Neither is built around: both are reported as the leading caveats on G2 in
`docs/chained-chat-result-2026-08-28.md`.

---

## Amendment 2026-08-29 (C1 was mis-measured — appended, nothing above edited)

§7's confound **C1 is withdrawn as stated**. Full working:
`docs/c1-reconciliation-2026-08-28.md`.

The C1 probe fed the baseline-arm draft the REAL segment ids and REAL recency
buckets, omitting `resolve_recency_buckets_for_model`, which for
`hop_signal_enabled=False` zeroes BOTH aux tables. That checkpoint has only
ever seen `segment=0, bucket=0`, so the probe imposed a train/inference
mismatch of its own on both of its arms. The quoted "0.0471 → 0.0273" is an
artifact of that mismatch.

Re-measured with the conditioning `evaluate_draft_model` applies, on the same
403 positions: raw-layout **57/403 = 0.1414** (reproducing
`data/raw_base_1p7b.json` exactly, so the raw evaluation pipeline is sound)
and chat-layout **57/403 = 0.1414**. Verified not to be a plumbing null: the
feature tensors differ (cosine 0.937, 7.0% of positions overwritten with
assistant-layout rows) and the draft's predictions differ at 203 of 403
positions, yet paired McNemar over the byte-identical positions gives 17 vs 17
discordant, **p = 1.000**.

Consequence for the gate readout in
`docs/chained-chat-result-2026-08-28.md`: **G1's 2.98x is not inflated by the
feature layout** and the "provisional pending a chat-trained draft" caveat is
withdrawn. G1's remaining limits are the ordinary ones — n=18 records, one
undertrained checkpoint, and the teacher-forced-versus-chained regime gap.

Retraining on chat-layout features remains scheduled and remains correct: the
deployed loop hands EAGLE chat-layout features. The expectation changes, not
the plan — it should move the neural arm little.
