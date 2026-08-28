# Result: chat-mode chained replay — 2026-08-28

Executes `docs/prereg-chained-chat-replay-2026-08-28.md`. Read that first; its
gates, arms and analysis were fixed before this run, and its amendment records
what was found afterwards.

**No number in this file may be compared with any number in
`chained-routed-result-2026-08-28.md` or with any `data/rounds_*_1p7b.jsonl`
from the raw runs.** The measurement definition changed; `hopspc.md` §15
forbids it. See `data/README-chained-rounds.md`.

Setup: `Qwen/Qwen3-1.7B`, `data/shard_1p7b.jsonl` (120 records), the existing
15%/seed-0 split (18 held-out, 102 train — not re-split), draft checkpoint
`data/ckpt_base_1p7b.pt`, gamma=4, MPS, `--replay-mode chat`. Artifacts
`data/rounds_chat_<arm>_1p7b.jsonl`; every row carries `replay_mode`.

## 0. Did the harness fix work? Yes.

The wrapper splits into an 82-token prefix and a 9-token suffix; the seam was
verified at every step boundary of the probe record before the run (a merge
there would shift every label).

| | raw mode | chat mode |
|---|---|---|
| recorded step-opening token reproduced | 2/40 = **0.05** | 41/51 = **0.80** |
| scaffold FSM token acceptance | 4/102 = **0.039** | 82/94 = **0.872** |

The FSM was never wrong: measured against the recorded trajectory it was
97.6% accurate all along. Raw mode simply could not verify it, because it
prompted the target with a document the agent never saw. Prereg §8's stopping
rule ("if the step-opening rate does not materially improve, the fix did not
work") is satisfied.

## 1. Headline: accepted tokens per round, held-out, chat mode

Primary arms — the gates are read off these:

| arm | rounds | pooled | per-record mean | ratio vs neural | paired delta [95% boot] | sign p | accepted TEMPLATE / content |
|---|---|---|---|---|---|---|---|
| neural | 473 | 0.254 | 0.285 | 1.00x | — | — | 0 / 120 |
| scoped lookup | 449 | 0.321 | 0.299 | 1.05x | +0.014 [−0.130, +0.164] | 1.000 | 0 / 144 |
| scaffold only | 510 | 0.161 | 0.222 | 0.78x | −0.063 [−0.113, −0.017] | 0.118 | 82 / 0 |
| **routed** | 330 | **0.800** | 0.850 | **2.98x** | +0.565 [+0.446, +0.705] | <0.001 | 82 / 182 |
| entropy-routed | 367 | 0.619 | 0.612 | 2.15x | +0.327 [+0.205, +0.466] | <0.001 | 0 / 227 |

Secondary — reported, never used to decide a gate:

| arm | rounds | pooled | per-record | ratio | accepted TEMPLATE / content |
|---|---|---|---|---|---|
| scaffold, verb bet | 478 | 0.238 | 0.373 | 1.31x | 114 / 0 |
| routed, sources chained | 330 | 0.800 | 0.850 | 2.98x | 82 / 182 |
| **routed, verb bet** | 307 | **0.935** | 1.045 | 3.67x | 114 / 173 |
| structure + entropy | 330 | 0.800 | 0.850 | 2.98x | 82 / 182 |

Pairing is per record (18 pairs), exact two-sided sign test on non-zero
differences plus a paired bootstrap (20 000 resamples, seed 0). Rounds are not
byte-identical across arms, so McNemar over positions does not apply. n=18:
read the interval, not the p-value.

Every arm emits the same work — 592–594 tokens — in a different number of
target verifications: 1.254 tokens per verification for neural, 1.800 for
routed, 1.935 for routed with the verb bet.

## 2. Gates

- **G1 — routed ≥ 1.5× neural: PASS.** 2.98×, +0.565 [+0.446, +0.705].
  ~~Provisional, for the reason pre-registered as C1 (teacher-forced
  decode-phase top-1 0.0471 → 0.0273 under the feature-layout change).~~
  **CORRECTED 2026-08-29 — the C1 caveat is withdrawn.** That probe was
  mis-conditioned: it omitted `resolve_recency_buckets_for_model`, feeding the
  baseline-arm draft real segment ids and real recency buckets when that arm
  has only ever seen `segment=0, bucket=0`. Re-measured correctly on the same
  403 positions, both layouts score **57/403 = 0.1414** — raw-layout
  reproducing `data/raw_base_1p7b.json` exactly — with paired McNemar 17 vs 17
  discordant, **p = 1.000**. The feature layout reshuffles which positions the
  draft gets right, not how many. **G1 is therefore not inflated by C1.** Full
  working: `docs/c1-reconciliation-2026-08-28.md`. Remaining limits on G1 are
  the ordinary ones: n=18 records, one undertrained checkpoint, and the
  teacher-forced-versus-chained regime gap.
- **G2 — routed ≥ entropy-routed: PASS.** +0.238 [+0.173, +0.305], sign
  p<0.001. Unaffected by C1 — both arms share the same neural fallthrough.

**But G2 does not say what its name suggests. Two things, both post-hoc:**

1. **The entropy gate is degenerate here.** Target entropy at the 367 round
   starts: median 0.000 nats, p75 0.014, p95 0.631, max 1.618. 87.5% of rounds
   sit below the smallest threshold in the grid and 100% below 2.0 — which is
   why the tuning table was flat across 0.25–3.0. The comparator reduces to
   "always try the lookup, else the neural draft". Its gate contributes
   nothing, so G2 is measuring **scaffold vs no scaffold**, not structure
   signal vs entropy signal.
2. **Routed's whole margin is TEMPLATE.** Routed accepts 82 TEMPLATE + 182
   content; entropy accepts 0 + 227. Routed accepts *fewer content tokens*
   than the comparator — the scaffold takes rounds the lookup would otherwise
   have opened. The systems quantity (tokens per target verification) is what
   the gate is defined on and it stands, but there is no content claim here.

## 3. Per bucket (mean accepted/round, round counts in parens)

| bucket | neural | lookup | scaffold | routed | entropy |
|---|---|---|---|---|---|
| 0 (at the hop) | 0.000 (22) | 0.000 (22) | **2.000 (22)** | **2.000 (22)** | 0.000 (22) |
| 1 (1–2) | 1.000 (22) | 0.000 (44) | — | — | 1.000 (22) |
| 2 (3–8) | 0.301 (73) | 0.154 (91) | 0.000 (94) | 0.514 (70) | 0.514 (70) |
| 3 (9–24) | 0.208 (77) | 1.103 (39) | 0.000 (93) | 1.250 (36) | 1.250 (36) |
| 4 (25–48) | 0.231 (39) | 0.690 (29) | 0.000 (49) | 0.690 (29) | 0.690 (29) |
| 5 (49+) | 0.000 (3) | 0.500 (2) | 0.000 (3) | 0.500 (2) | 0.500 (2) |
| 6 (no prior hop) | 0.215 (237) | 0.297 (222) | 0.153 (249) | 0.690 (171) | 0.554 (186) |

**The pre-registered expectation (prereg §6) is confirmed, and cleanly.**
Routed and entropy are *identical* in buckets 2, 3, 4 and 5 — 0.514/0.514,
1.250/1.250, 0.690/0.690, 0.500/0.500 — and differ only at bucket 0
(2.000 vs 0.000) and bucket 6 (0.690 vs 0.554). Bucket 6 is the pre-retrieval
span, whose rounds include the trajectory's first step opening. So the entire
separation between structure routing and the comparator sits at step
openings, exactly where it was predicted to sit and nowhere else.

Bucket 0 carries 22 rounds. Per §10 that is descriptive, not a test, and it is
not reported as one.

## 4. Other findings

- **The verb bet pays.** Betting `Action: Search[` instead of stopping at the
  shared `Action:` prefix: the scaffold arm goes 0.161 → 0.238 and routed goes
  0.800 → 0.935. The full 4-token span is accepted on only 9/51 rounds versus
  41/51 for the 2-token prefix, but partial acceptance more than covers the
  misses (114 accepted tokens vs 82). The pilot's recorded-trajectory
  measurement (144 vs 80 tokens) predicted this correctly. Both are secondary
  arms: the gate was pre-registered on the shared-prefix configuration.
- **Chaining sources within a round is still a no-op**, for the second time.
  It proposes 122 more neural tokens and accepts exactly the same 264. At this
  checkpoint quality it is pure extra draft compute.
- **Structure + entropy is identical to routed** (0.800, same source mix), the
  expected consequence of a degenerate gate.
- **The scaffold alone is worse than the neural draft** (0.161 vs 0.254): it
  defers on 459 of 510 rounds and proposes nothing there. It only pays in
  combination, which is the argument for routing rather than for the FSM.
- **The lookup alone no longer separates from the neural draft** in this mode
  (+0.014, p=1.000), though it still dominates within buckets 3–4. Do not
  reach for the raw-mode table to explain this; the two are not comparable.

## 5. What this licenses

Licensed: in a correctly simulated ReAct loop, routing three structure-selected
draft sources gives 1.800 tokens per target verification against 1.254 for the
neural draft alone, and beats a ReSpec-style entropy comparator. The separation
from that comparator is entirely at step openings, as predicted in advance.

Also licensed, and arguably the more interesting result: **a target-entropy
routing signal has nothing to route on in this workload.** Correctly prompted,
the target is near-deterministic at the boundaries where a router must decide
(median entropy 0.000 nats). Structure is available there and free; entropy is
not merely equalled, it is uninformative.

NOT licensed: any content claim from G1 or G2. Routed's margin over the
comparator is TEMPLATE tokens, and it accepts fewer content tokens than the
comparator does. NOT licensed either: G1 as a
final number — not because of the withdrawn C1 caveat, but because n=18 and
the draft is undertrained.

## 6. Next, in order

1. **Retrain both draft arms on chat-layout features**, then re-run a–e. Still
   correct and still scheduled first — the deployed loop hands EAGLE
   chat-layout features, so that is what the draft should be trained on — but
   after the C1 correction it should be expected to move the neural arm
   *little*, and a large move would itself be a finding to explain.
2. Promote the verb bet to a primary arm in the next prereg — on this evidence
   it is the better default — and pre-register the fitted verb the same way.
3. Retire the entropy comparator as a *routing signal* baseline for this
   workload, or replace it with one that has something to route on. Report the
   degeneracy as a finding rather than quietly dropping the arm.
4. Only then: the 500–1000-trajectory collection and retraining.
