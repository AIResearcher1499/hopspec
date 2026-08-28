# Result: scale-up to 1000 trajectories — 2026-08-29

Executes `docs/prereg-scaleup-2026-08-29.md`. Arms, gates, hypotheses and the
"material" thresholds were all fixed before collection.

**Forbidden comparisons (prereg §7).** Nothing here may be compared with the
raw-mode tables, the stale-checkpoint chat run, the 120-record chat-trained
run, or the §11 findings. The shard, the device and the checkpoint changed at
once. Comparisons live **within** this run.

## 0. Provenance

| | |
|---|---|
| shard | `data/shard_1p7b_scale.jsonl`, 1000 records, md5 `3b50e71c5526176622fca24cb7449ac4` |
| collected | RTX 5000 Ada (32 GB), `--batch-size 8 --resume`, ~18 min |
| **reproducibility** | **Not reproducible by a sequential run.** Batched matmuls are not bit-identical; measured on 5 questions, **3/5 trajectories match `--batch-size 1` exactly**. The divergences are in free `Finish[…]` answer text; the scaffold structure is identical. This is documented behaviour, never an explanation for a result |
| validated | `06_validate_shard.py` exit 0 on the Mac — all hard checks, 0 incomplete, no duplicate ids, no eval-pool leakage |
| shape | mean 520 tokens, p50 445, p95 1062, max 1598. Generated mix: TEMPLATE 31.1%, ANSWER 25.6%, TOOL_CALL 24.1%, THOUGHT 19.2% |
| draft | `ckpt_{base,hop}_chat_scale.pt`, `--feature-mode chat`, 850 train / 150 held-out (15%, seed 0, computed once) |
| fitted on train only | `step_opening='Action:'` (1646 vs 224 `Thought:`), `action_verb='Search'` (1020 vs 850) |

The §8 stopping rule — fitted opening covering < 50% of steps — did **not**
trigger: `Action:` covers 88%.

## 1. Headline: accepted tokens per round, 150 held-out records

Primary arms:

| arm | rounds | pooled | per-record | ratio vs neural | paired delta [95% boot] | sign p | TEMPLATE / content |
|---|---|---|---|---|---|---|---|
| neural | 4267 | 0.525 | 0.727 | 1.00x | — | — | 0 / 2240 |
| scoped lookup | 4839 | 0.348 | 0.312 | 0.43x | −0.415 [−0.525, −0.306] | <0.001 | 0 / 1683 |
| scaffold (verb bet) | 5518 | 0.178 | 0.377 | 0.52x | −0.350 [−0.381, −0.320] | <0.001 | 980 / 0 |
| **routed (verb bet)** | 3135 | **1.081** | 1.284 | **1.77x** | +0.557 [+0.487, +0.629] | <0.001 | 968 / 2422 |
| entropy-routed | 3244 | 1.012 | 1.150 | 1.58x | +0.423 [+0.354, +0.495] | <0.001 | 0 / 3284 |

Secondary (reported, never used to decide a gate):

| arm | rounds | pooled | per-record | ratio | TEMPLATE / content |
|---|---|---|---|---|---|
| scaffold, shared prefix | 5806 | 0.118 | 0.208 | 0.29x | 688 / 0 |
| routed, shared prefix | 3252 | 1.005 | 1.133 | 1.56x | 680 / 2589 |
| routed, sources chained | 3134 | 1.082 | 1.284 | 1.77x | 968 / 2423 |
| structure + entropy | 3256 | 1.003 | 1.127 | 1.55x | 680 / 2585 |

Every arm emits the same work (6498–6528 tokens) in a different number of
target verifications:

| arm | tokens per target forward |
|---|---|
| scaffold only | 1.178 |
| lookup only | 1.348 |
| neural only | 1.525 |
| entropy-routed | 2.012 |
| **routed, verb bet** | **2.081** |

Pairing is per record (150 pairs), exact sign test plus a paired bootstrap
(20 000 resamples, seed 0). Rounds are not byte-identical across arms, so
McNemar over positions does not apply between them.

## 2. Gates

- **G1 — routed ≥ 1.5× neural: PASS.** 1.77×, +0.557 [+0.487, +0.629].
- **G2 — routed ≥ entropy-routed: PASS.** +0.134 [+0.106, +0.163], p<0.001.

At 150 records both intervals are far clear of zero. This is the first
properly powered version of either gate.

## 3. The pre-registered hypotheses

### H1 — FALSIFIED. The scaffold still pays.

Predicted: `net_trade ≤ 0` — that a stronger draft would make each displaced
round expensive enough to flip the trade. Measured, using the statistic fixed
in the prereg:

```
net_trade = scaffold_accepted(routed) − [content(entropy) − content(routed)]
          = 968 − (3284 − 2422)
          = +106
```

**The prediction was wrong.** The draft did strengthen exactly as expected —
teacher-forced held-out accuracy 0.3022, per-token acceptance 0.13 in the
neural arm — and the scaffold still wins 968 TEMPLATE tokens against 862
content tokens surrendered. A grammar drafter is not a crutch for a weak
draft at this scale; it is additive.

**Caveat on the statistic, found while writing this up.** The prereg justified
`net_trade` with "both arms open the same lookup rounds, so the difference is
the scaffold's displacement of the neural draft". At this scale that premise
is only approximately true: routed opens 811 lookup rounds, the comparator 764
— a 47-round gap, where the pilot had none. Decomposing the content
difference by source removes the need for the assumption:

| source | routed | entropy | difference |
|---|---|---|---|
| lookup | 1583 | 1569 | +14 |
| neural | 839 | 1715 | **−876** |
| total content | 2422 | 3284 | −862 |

The deficit is essentially all neural displacement (−876), with the lookup
contribution nearly cancelling (+14). So the conclusion survives the imperfect
premise, and the decomposition — not the assumption — is what should be
quoted. Future preregs should define `net_trade` against the neural row
directly.

Recorded as a failed prediction, not narrated as a success. What it licenses
is narrow: at 1.7B, with this draft, on this shard, the trade is positive. The
mechanism that would flip it is unchanged and still plausible at a larger
draft, and the statistic is now defined so the next run can test it again.

### H2 — CONFIRMED. The entropy gate is inert, now measured rather than inferred.

`gate_signal` is logged on every round, so this is no longer read off a flat
table:

- **86.7%** of the entropy arm's 3244 round starts sit at ≤ 0.25 nats
  (predicted ≥ 80%); median **0.0000**, p75 0.0214, p95 0.667.
- The tuning table is **exactly flat**: 1.165 accepted/round at every one of
  0.25, 0.5, 1.0, 1.5, 2.0, 3.0 — variation 0.000, predicted < 0.05.

**G2 is therefore read as pre-registered: "scaffold coverage versus none",
not "structure signal beats entropy signal."** Both arms open effectively the
same lookup rounds; the gate contributes nothing because there is nothing to
gate on. A target-entropy routing signal is uninformative in this workload.

### H3 — CONFIRMED, with the same knock-on as before.

| bucket | neural | lookup | scaffold | routed | entropy |
|---|---|---|---|---|---|
| 0 (at the hop) | 0.911 (180) | 0.000 (180) | **2.194 (180)** | **2.194 (180)** | 0.911 (180) |
| 1 (1–2) | 2.796 (108) | 0.000 (360) | 0.000 (12) | 1.167 (6) | 2.796 (108) |
| 2 (3–8) | 0.405 (407) | 0.146 (732) | 0.000 (719) | 0.858 (431) | 0.692 (380) |
| 3 (9–24) | 0.346 (679) | 0.784 (487) | 0.002 (908) | 1.092 (415) | 1.132 (408) |
| 4 (25–48) | 0.355 (488) | 0.591 (416) | 0.020 (659) | 0.860 (351) | 0.850 (353) |
| 5 (49+) | 0.420 (300) | 0.688 (247) | 0.024 (421) | 1.090 (200) | 1.064 (202) |
| 6 (no prior hop) | 0.511 (2105) | 0.322 (2417) | 0.214 (2619) | 1.060 (1552) | 0.978 (1613) |

Routed and entropy are indistinguishable in buckets 3–5 (entropy is
marginally ahead in bucket 3) and separate at bucket 0, where only the FSM can
fire. Buckets 2 and 6 also separate; as before that is a knock-on — once the
scaffold changes what was emitted, the arms' later rounds are different
rounds. Buckets are not aligned across arms.

## 4. The hop signal, re-measured for the record

Not tuned in any way; re-run under the corrected harness on the larger shard
so the number exists. Byte-identical positions, so McNemar applies:

baseline **1468/4857 = 0.3022**, hop-signal **1487/4857 = 0.3062**,
discordant 139/158, exact **McNemar p = 0.2963**.

**No effect.** The §11 negative is not reopened by this and no retuning
follows from it.

## 5. What this licenses

Licensed: routing three structure-selected draft sources gives **2.081 tokens
per target verification against 1.525** for the neural draft alone (G1 1.77×,
interval clear of zero at n=150), and beats a ReSpec-style entropy comparator
by 0.134 [0.106, 0.163] accepted tokens per round.

Licensed, and now measured twice at two scales: **a target-entropy routing
signal has nothing to route on in this workload** — 86.7% of routing decisions
face a near-deterministic target, and the tuning table is exactly flat. Frame
it as "entropy is uninformative here", never as "structure beats entropy in
general".

NOT licensed: any content-quality claim. Routed's margin over the comparator
is TEMPLATE tokens; it accepts 862 fewer content tokens. The systems quantity
is the claim.

## 6. Cost

One RTX 5000 Ada pod, ~2.5 h at $0.49/hr ≈ **$1.25**. A5000 ($0.16) and A6000
($0.33) were supply-constrained on both community and secure at deploy time.

Peak VRAM during training was **28.2 GB of 32.7 GB**. A 24 GB card would very
likely have OOM'd on this shard's 1598-token tail — the earlier reasoning that
"24 GB is proven sufficient because the pilot ran on a 24 GB Mac" did not
survive the longer shard. Size future rentals at **32 GB or more**.

## 7. Next

1. The verb bet is now clearly the right primary (1.081 vs 1.005 pooled,
   2.081 vs 2.005 tokens per verification). Keep it.
2. H1 is open again at larger draft scale. The statistic is defined; re-test
   it rather than assuming either answer.
3. The larger target model (8B) is the remaining direction, and the only one
   whose cost is non-trivial (§8 of the execution plan).
