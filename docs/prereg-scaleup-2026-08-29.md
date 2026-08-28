# Prereg: scale-up to 500–1000 trajectories — 2026-08-29

**FROZEN once `data/shard_1p7b_scale.jsonl` exists.** Amendments are appended
with a date; no threshold in this file may be edited after a number is seen
(`hopspc.md` §15). Written before any collection.

Supersedes `prereg-chat-trained-draft-2026-08-29.md` for the arms below; that
prereg's chat-mode measurement definition is unchanged and still governs.

## 1. What changes

- **Scale.** 1000 HotpotQA trajectories instead of 120, from the same
  persisted collect/eval split (`data/hotpot_split.json`, seed 0,
  eval_fraction 0.2 — **already written, never regenerate it**). The shard's
  own 15%/seed-0 train/held-out split is computed once on the new shard and
  never re-split.
- **Device.** Collected, trained and evaluated on a rented CUDA pod.
  `docs/device-bridge-2026-08-29.md` establishes that the MPS pilot is
  directionally comparable but **not byte-identical**, so nothing from this
  run may be pooled into an MPS table.
- **Batched collection.** `--batch-size 8`. Batched matmuls are not
  bit-identical to single-sequence ones, so this shard is not reproducible by
  a sequential run. That belongs in its provenance and is never an
  explanation for a result.
- **Verb bet is the primary routed configuration**, carried over from the
  previous prereg where it was promoted.

## 2. Preconditions, all of which must hold before any training

1. `06_validate_shard.py --shard data/shard_1p7b_scale.jsonl --split-file
   data/hotpot_split.json` exits zero. Run it **on the Mac** — it needs no GPU
   and paying for it is waste.
2. No question id appears twice (the `--resume` path appends; the duplicate-id
   check is the guard).
3. No collected id is in the eval pool of `hotpot_split.json`.
4. The 5-question batched-vs-sequential validation has been run on the pod and
   the number of exactly-matching trajectories reported.

If any fails, stop. Do not train on an unvalidated shard.

## 3. Arms

Same as the previous prereg. `--replay-mode chat`, gamma=4, checkpoint from
`03_run_diagnostic.py --feature-mode chat` on the new shard.

**Primary — the gates are read off these:**

| id | `--draft-source` | flags |
|---|---|---|
| a | neural | — |
| b | lookup | — |
| c | scaffold | `--scaffold-verb fit` |
| d | **routed** | `--scaffold-verb fit` |
| e | entropy | `--tune-entropy 0.25,0.5,1.0,1.5,2.0,3.0 --tune-max-records 6` |

**Secondary — reported, never used to decide a gate:** scaffold and routed
with the shared prefix, routed `--chain`, entropy `--entropy-scaffold`.

Everything fittable — the scaffold's opening literal, the action verb, the
entropy threshold — is fitted on the **train split only** and printed. The
fitted opening is expected to change: a shard collected from a
better-behaved agent may open steps with `Thought:` rather than `Action:`.

## 4. Gates

Unit of analysis: one mean accepted-tokens-per-round per held-out record
(~150 records). Exact two-sided sign test on non-zero differences plus a
paired bootstrap (20 000 resamples, seed 0). Rounds are not byte-identical
across arms, so McNemar over positions does not apply between them.

- **G1 — routed (d) ≥ 1.5× neural (a).** PASS requires ratio ≥ 1.5 **and** the
  bootstrap interval on the mean difference to exclude 0.
- **G2 — routed (d) ≥ entropy-routed (e).** PASS requires the mean difference
  ≥ 0; an interval containing 0 is a TIE.

## 5. Pre-registered hypotheses

Recorded now so none can be claimed as a prediction afterwards.

### H1 — the scaffold/neural trade flips as the draft strengthens

On the 120-record pilot the scaffold paid for itself: it displaced neural
rounds worth fewer content tokens than the TEMPLATE tokens it won. The draft
here is trained on ~8× more data, so its per-round yield should rise, making
each displaced round more expensive.

**The statistic, defined now, computed within this run only** (no comparison
to any earlier table):

```
net_trade = scaffold_accepted(d)
          - [ content_accepted(e) - content_accepted(d) ]
```

i.e. the TEMPLATE tokens the scaffold wins, minus the content tokens routed
gives up against the comparator that has no scaffold. Both arms open the same
lookup rounds, so the difference is the scaffold's displacement of the neural
draft.

**Prediction: `net_trade ≤ 0`** — the trade has flipped and the scaffold no
longer pays at this draft strength.

Either outcome is publishable and neither is a failure. `net_trade > 0` means
a grammar drafter is a durable component; `net_trade ≤ 0` means it is a crutch
for a weak draft, and the paper says so. What is not acceptable is discovering
the sign afterwards and narrating it as expected.

### H2 — the entropy gate is degenerate again

Twice measured: median target entropy at round starts 0.000 nats, and tuning
tables flat across 0.25–3.0. Now a first-class measurement rather than an
inference: every round row carries `gate_signal`, the entropy the router
gated on.

**Prediction: ≥ 80% of the entropy arm's round starts have `gate_signal` ≤
0.25 nats, and the tuning table varies by < 0.05 accepted/round across the
whole grid.** If both hold, **G2 is read as "scaffold coverage versus none",
not as "structure signal beats entropy signal"**, and is reported in exactly
those words. The arm is kept and its statistics reported either way — the
degeneracy is a finding, not an arm to drop quietly.

### H3 — separation stays at step openings

Whatever separates d from e sits in bucket 0 and the pre-retrieval bucket,
because the FSM is the only source e lacks and it can only fire at a step
opening. Bucket means are descriptive, always quoted with round counts, never
reported as a test.

## 6. Reporting requirements

- **TEMPLATE/content split for every arm.** Any margin carried by TEMPLATE
  tokens is a systems margin and is stated as one; no content-quality claim
  may be read off it.
- Per-bucket means with round counts (spec §10); no acceptance number without
  its bucket's majority-class rate.
- Tokens per target verification for every arm — the systems quantity.
- The teacher-forced summaries of both retrained arms. The hop-signal arm
  rides along **with no tuning of any kind**: it is re-measured for the record
  under a corrected harness and a larger shard, and that does **not** reopen
  the §11 negative.
- The shard's provenance: batch size, device, collection date, and that it is
  not reproducible by a sequential run.

## 7. Forbidden comparisons

No number from this run may be compared with: the raw-mode tables, the
stale-checkpoint chat run, the 120-record chat-trained run
(`chat-trained-draft-result-2026-08-29.md`), or the §11 findings. The shard,
the device and the checkpoint all changed at once. Comparisons live **within**
this run.

This is also why H1 is defined as a within-run statistic rather than as
"smaller than last time".

## 8. Stopping rules

- Any precondition in §2 fails → stop, do not train.
- A confound outside §5 appears → stop and report before building around it.
- The fitted scaffold opening covers < 50% of generated steps → report it and
  treat the scaffold arm as inapplicable to this shard rather than tuning the
  FSM to rescue it.
