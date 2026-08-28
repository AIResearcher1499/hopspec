# Adversarial audit of `scaleup-result-2026-08-29.md` — 2026-08-29

Audited at commit `0d298b4` by recomputation from the committed artifacts
(`data/rounds_sc_*_1p7b.jsonl`, `data/raw_{base,hop}_chat_scale.json`,
`data/summary_base_chat_scale.json`). Nothing was re-run; the rounds files
are trusted to be the output of the documented commands. Scipy exact
binomial for the sign test; numpy `default_rng(0)`, 20 000 resamples for the
bootstrap — intervals may differ from the result doc's in the third decimal
for RNG reasons only.

Verdicts, ranked by severity. Where a claim survives it says so.

---

## 1. WRONG (definition) — the "TEMPLATE / content" column is labelled by proposing SOURCE, not by token TYPE

`token_sources` names which drafter proposed each token. The result doc's
"TEMPLATE / content" column is `scaffold-proposed / non-scaffold-proposed`.
That is not the template/content split the prereg (§6) and spec (§3) mean.

Demonstration from the rounds alone. Bucket 0 is the step-opening round
(distance 0 after a passage); the target's first tokens there are the ReAct
scaffold by construction (`Action:` opens 88% of steps per the doc's own
fit). In that bucket:

| arm | rounds | accepted, by source |
|---|---|---|
| neural | 180 | neural 164 |
| entropy | 180 | neural 164, lookup 0 |
| routed | 180 | scaffold 395 |

The entropy arm accepts **164 tokens at step openings through its neural
draft**, and every one of them is counted as "content" in `0 / 3284`. They
are scaffold tokens. So:

- "entropy: 0 TEMPLATE" is false; the true count is ≥ 164 (unknown beyond
  that — rows carry no segment ids, and the neural draft can also accept
  scaffold tokens inside later rounds, e.g. `]`/newline).
- "routed accepts 862 fewer content tokens" is an upper bound at best; the
  bucket-0 correction alone reduces it to ≤ 698.
- H1's `net_trade` inherits the mislabel. Reclassifying only the bucket-0
  tokens moves it from +106 to ≥ +270. The SIGN of H1 is therefore robust
  to this error; the MAGNITUDE quoted is not a measurement of what the
  prereg defined.

Fix: log the accepted token ids (or their decoded text) per round; classify
template vs content offline by matching the fitted scaffold literals, and
recompute every T/C figure. Until then, the column must be relabelled
"scaffold-proposed / other-proposed" wherever it appears.

## 2. OVERSTATED — H1 "the scaffold still pays … a grammar drafter is additive" is configuration-dependent and small

Primary config (verb bet), recomputed: `net_trade = 968 − (3284 − 2422) =
+106`. Per-record paired bootstrap on the sum: **[+81, +132]**, 90 records
positive / 8 negative — so the point estimate is not noise (note the prereg
defined no uncertainty for this statistic at all; this is the first
interval it has had).

But under the secondary shared-prefix config:

| config | scaffold-proposed | other-proposed | `net_trade` vs entropy |
|---|---|---|---|
| routed, verb bet (primary) | 968 | 2422 | **+106** |
| routed, shared prefix | 680 | 2589 | **−15** |
| entropy + shared scaffold | 680 | 2585 | −19 |

The prereg's own prediction (`net_trade ≤ 0`) is CONFIRMED for the
shared-prefix scaffold and FALSIFIED only for the verb bet. Since the
prereg fixed the verb bet as primary before the run, the verdict
"FALSIFIED" is legitimate as pre-registered. The generalising sentence "a
grammar drafter is not a crutch for a weak draft at this scale; it is
additive" is not: +106 is ~3% of routed's accepted tokens, and the
shared-prefix FSM does not pay.

The mechanism paragraph is also muddled. The entropy arm's neural draft
already wins **0.911 accepted/round at step openings** (164 tokens over 180
rounds). The scaffold's incremental value at bucket 0 is therefore
2.194 − 0.911 = **1.28/round ≈ 231 tokens**, not 2.194/round ≈ 395. The
"−876 neural" row in the decomposition is mostly the neural draft losing
step-opening rounds it was partially winning, plus a selection effect:
inside routed the neural draft is left the hard rounds (yield 0.444/round)
while inside the entropy arm it keeps the easy openings (0.692/round).

## 3. OVERSTATED — G2 PASS is carried entirely by the verb bet; "scaffold coverage versus none" is imprecise

Recomputed, per-record pairing, 150 records:

| comparison | delta [95% boot] | sign test | records negative |
|---|---|---|---|
| routed (verb) − entropy | **+0.134 [+0.106, +0.163]** | 92/100, p=3e-19 | 8 |
| routed (shared prefix) − entropy | −0.017 [−0.042, +0.006] | 27/63, p=0.31 | 36 |
| entropy + scaffold (shared) − entropy | −0.022 [−0.046, +0.000] | 25/60, p=0.25 | 35 |
| routed (verb) − routed (shared) | +0.151 [+0.120, +0.185] | 100/101 | 1 |

G2 as pre-registered (primary = verb bet) is PASS. But a shared-prefix
scaffold does NOT beat the comparator — it ties or slightly loses. The
correct reading of G2 is "**verb-bet** scaffold coverage versus none",
and §2 of the result doc must state the shared-prefix tie, not leave it in
a secondary table. The like-for-like SIGNAL test — same three sources,
structure routing vs entropy routing (routed-shared vs entropy+scaffold) —
is −0.017 vs −0.022: **indistinguishable**. That is the honest sentence
about the routing signal and it should be the one quoted.

## 4. CONFIRMED with a gap — H2 numbers hold; "the gate contributes nothing" is partly UNVERIFIABLE

Recomputed on the entropy arm's 3244 rounds: `gate_signal` ≤ 0.25 nats at
**86.7%**, median **0.0000**, p75 0.0214, p95 0.667, max 3.865. Prediction
(≥ 80%) CONFIRMED.

However the implied threshold in use is **0.25** (max `gate_signal` among
lookup-opened rounds = 0.247), and **433 rounds (13.3%) were sent to the
neural draft without trying the lookup**. Whether the lookup would have hit
on them is not measured on held-out; the "exactly flat tuning table" that
underwrites "the gate contributes nothing" comes from 6 TRAIN records and
is not in any committed artifact → UNVERIFIABLE. The doc's "both arms open
effectively the same lookup rounds" is 811 vs 764; a 47-round gap is what
a gate that blocks some lookup hits would produce. The gate is inert in the
*tuning* sense; it is not demonstrably inert in *behaviour*. Say that.

On fairness (audit question 4): the comparator does implement ReSpec's
mechanism (low entropy → retrieval draft), so it is not a strawman as a
*mechanism*. It is a strawman as a *source set* — it lacks the FSM — which
is why the like-for-like row in §3 above is the only fair signal test.

## 5. Prereg §6 / spec §10 not followed — bucket acceptance numbers without majority-class rates

The H3 table quotes 35 bucket-level acceptance numbers with no
distinct-token count or majority-class rate. The rounds artifacts cannot
supply them (no target tokens per round). Prereg §6 bullet 2 requires them
for every acceptance number; it was not followed for the chained table.

The teacher-forced summary does carry them and is fine: buckets 2–6 are
non-degenerate (majority 0.037–0.171); bucket 1 is degenerate (n=6,
majority 0.833) and is correctly not leaned on.

Spec §15's constant-predictor rule applies with force at bucket 0: the FSM
*is* a constant predictor (`Action: Search[`), and it scores 2.194/4 = 55%
per-token acceptance there. This is exactly the §3 "template
predictability" effect that the project excludes from *measurement* and now
correctly counts as *systems work*. The doc should say so in those words;
at present bucket 0 is "descriptive" without naming what it is descriptive
of.

## 6. CONFIRMED — G1

Recomputed: routed − neural = **+0.557 [+0.487, +0.629]**, 136/137
non-zero differences positive, sign p = 2e-39, one record negative. Ratio
of per-record means 1.766 (doc: 1.77×); ratio of pooled means 2.06 — the
doc uses the more conservative one. Tokens per verification 2.081 vs 1.525
reproduce; every arm emits 6494–6528 tokens ("same work") reproduces.

On the pairing question (audit item 1): per-record accepted/round and
per-record tokens/verification give the identical delta (+0.134 for G2)
because `emitted = accepted + 1` per round, so the unit is coherent with
the systems claim. 50 records tie exactly on G2 and the sign test drops
them correctly. Records are the right exchangeable unit when rounds are not
aligned; I found nothing it hides.

## 7. CONFIRMED — hop signal

Raw columns align element-for-element on `recency_distance`,
`recency_bucket`, `hop_index`, `target_token`, `record_index`. Baseline
1468/4857 = 0.3022, hop 1487/4857 = 0.3062, discordant 158 (base wrong/hop
right) vs 139, exact McNemar **p = 0.2963**. The doc writes the discordant
pair as "139/158" — same numbers, opposite order; harmless. No effect; §11
stands.

## 8. UNVERIFIABLE (shard and checkpoints not committed)

Shard shape (mean 520 / max 1598 tokens), generated-token mix, scaffold fit
(`Action:` 1646 vs 224; `Search` 1020 vs 850), `06_validate_shard.py` exit
0, batched-vs-sequential 3/5, the tuning table, peak VRAM, cost. Send the
shard if you want the fit and the validate outcome checked; the rest needs
the checkpoints and a pod.

## 9. Minor

- H3 is a hypothesis with no threshold; calling it "CONFIRMED" is soft.
  Bucket 1 shows the misalignment plainly (routed 6 rounds vs neural 108).
- The result doc obeys its own forbidden-comparison rule: no number in it
  is set against the raw-mode, stale-checkpoint, 120-record or §11 tables.

---

## What must change before any of this is quoted

1. Add accepted token ids to every round row; recompute a true
   template/content split by scaffold-literal match; relabel every existing
   T/C column as "scaffold-proposed / other-proposed" in the meantime.
2. Move the shared-prefix G2 tie and the shared-prefix `net_trade = −15`
   into §2/§3 of the result doc; retract "a grammar drafter … is additive";
   state H1 as "positive only with the verb bet, +106 ≈ 3%, [81, 132]".
3. Report the like-for-like signal test (routed-shared vs entropy+scaffold,
   −0.017 vs −0.022) as the routing-signal result.
4. Report the 433 gate-to-neural rounds and mark "gate contributes
   nothing" as inferred from a 6-record train tuning, not measured on
   held-out.
5. Either add per-bucket majority-class rates to the chained table (needs
   token ids) or state that the requirement cannot be met from rounds and
   why.
