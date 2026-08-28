# Result: chained re-run with a chat-trained draft — 2026-08-29

Executes `docs/prereg-chat-trained-draft-2026-08-29.md`. Read it first: arms,
gates, the "material change" threshold and every expectation below were fixed
before the run.

**Forbidden comparisons (prereg §7).** Nothing here may be compared with the
raw-mode tables (`chained-routed-result-2026-08-28.md`), the stale-checkpoint
chat run (`chained-chat-result-2026-08-28.md`), or the §11 findings. The
checkpoint changed; §5's probe below is the only sanctioned way to ask what
that changed.

Setup: `Qwen/Qwen3-1.7B`, `data/shard_1p7b.jsonl`, the existing 15%/seed-0
split (18 held-out, 102 train — not re-split), `--replay-mode chat`, gamma=4,
MPS, draft `data/ckpt_base_chat_1p7b.pt` (`feature_mode: chat`, recorded in
the checkpoint and verified by `07` at load: *"checkpoint feature_mode=chat
matches the replay mode"*). Fitted on the train split only and printed by the
run: `step_opening='Action:'`, `action_verb='Search'`.

## 0. C1 is resolved

C1 as originally written was a mis-measurement, not a confound
(`docs/c1-reconciliation-2026-08-28.md`). The pre-registered replacement
(prereg §5) evaluates **both** checkpoints in one run, under chat-layout
features, on the same 403 held-out decode-phase positions, with
`resolve_recency_buckets_for_model` applied:

| draft | decode-phase top-1 |
|---|---|
| raw-trained (`ckpt_base_1p7b.pt`) | 56/403 = 0.1390 |
| chat-trained (`ckpt_base_chat_1p7b.pt`) | 68/403 = 0.1687 |

Exact two-sided McNemar: **p = 0.0652 → no material change** by the
pre-registered p<0.05 threshold. Pre-registered expectation 2 ("the
chat-trained draft will move the neural arm little") holds in the regime the
prereg fixed for it.

Limit, recorded in the prereg before the run: this is teacher-forced and
single-step, while the chained replay lets the draft chain on its own
predicted features. The single-source ordering in *this* run is
**neural > lookup > scaffold**. Whether that ordering differs from the
stale-checkpoint run is not a comparison this prereg permits. If it matters
for the paper, it needs its own prereg replaying both checkpoints chained in
one run — recommended, not done here.

## 1. Headline: accepted tokens per round, held-out, chat mode

Primary arms — the gates are read off these:

| arm | rounds | pooled | per-record | ratio vs neural | paired delta [95% boot] | sign p | TEMPLATE / content |
|---|---|---|---|---|---|---|---|
| neural | 449 | 0.323 | 0.492 | 1.00x | — | — | 0 / 145 |
| scoped lookup | 449 | 0.321 | 0.299 | 0.61x | −0.193 [−0.488, +0.066] | 0.804 | 0 / 144 |
| scaffold (verb bet) | 478 | 0.238 | 0.373 | 0.76x | −0.119 [−0.179, −0.065] | 0.007 | 114 / 0 |
| **routed (verb bet)** | 299 | **0.993** | 1.193 | **2.42x** | +0.701 [+0.553, +0.867] | <0.001 | 114 / 183 |
| entropy-routed | 336 | 0.771 | 0.878 | 1.79x | +0.387 [+0.244, +0.545] | <0.001 | 0 / 259 |

Secondary — reported, never used to decide a gate:

| arm | rounds | pooled | per-record | ratio | TEMPLATE / content |
|---|---|---|---|---|---|
| scaffold, shared prefix | 510 | 0.161 | 0.222 | 0.45x | 82 / 0 |
| routed, shared prefix | 315 | 0.886 | 1.046 | 2.13x | 82 / 197 |
| routed, sources chained | 299 | 0.993 | 1.193 | 2.42x | 114 / 183 |
| structure + entropy | 315 | 0.886 | 1.046 | 2.13x | 82 / 197 |

Pairing is per record (18 pairs), exact two-sided sign test plus a paired
bootstrap (20 000 resamples, seed 0). Rounds are not byte-identical across
arms, so McNemar over positions does not apply between them. n=18: read the
interval, not the p-value.

Every arm emits the same work (592–596 tokens) in a different number of target
verifications:

| arm | tokens per target forward |
|---|---|
| scaffold only | 1.238 |
| lookup only | 1.321 |
| neural only | 1.323 |
| entropy-routed | 1.771 |
| routed, shared prefix | 1.886 |
| **routed, verb bet** | **1.993** |

## 2. Gates

- **G1 — routed ≥ 1.5× neural: PASS.** 2.42×, +0.701 [+0.553, +0.867],
  sign p<0.001. **This replaces the provisional 2.98×.** It is measured
  against a draft trained on the layout it is served, and the C1 caveat that
  made the earlier figure provisional was withdrawn before this run.
- **G2 — routed ≥ entropy-routed: PASS.** +0.314 [+0.235, +0.402], p<0.001.

**G2 must be read as pre-registered: "scaffold coverage versus none."** The
entropy gate is inert again, exactly as expectation 1 predicted. Its tuning
table is flat — 1.124 accepted/round at thresholds 0.25 and 0.5, 1.146 at
1.0 through 3.0, a difference of one round in 97 — so the comparator is
"always try the scoped lookup, else the neural draft". Both arms open the same
73 lookup rounds. This is a finding (novelty verdict, safe claim 5), not an
arm to drop: **a target-entropy routing signal has nothing to route on in
this workload.**

## 3. Per bucket (mean accepted/round, round counts in parens)

| bucket | neural | lookup | scaffold | routed | entropy |
|---|---|---|---|---|---|
| 0 (at the hop) | 0.182 (22) | 0.000 (22) | **2.318 (22)** | **2.318 (22)** | 0.182 (22) |
| 1 (1–2) | 1.900 (20) | 0.000 (44) | — | — | 1.900 (20) |
| 2 (3–8) | 0.300 (60) | 0.154 (91) | 0.000 (87) | 0.772 (57) | 0.561 (57) |
| 3 (9–24) | 0.120 (83) | 1.103 (39) | 0.000 (93) | 1.189 (37) | 1.189 (37) |
| 4 (25–48) | 0.116 (43) | 0.690 (29) | 0.000 (49) | 0.690 (29) | 0.690 (29) |
| 5 (49+) | 0.000 (3) | 0.500 (2) | 0.000 (3) | 0.500 (2) | 0.500 (2) |
| 6 (no prior hop) | 0.321 (218) | 0.297 (222) | 0.281 (224) | **0.901 (152)** | 0.710 (169) |

Expectation 3 said the separation between routed and the comparator would sit
at step openings. **Confirmed where the FSM actually fires:** of the routed
arm's 51 scaffold-opened rounds, 45 are in bucket 0 (22) and bucket 6 (23) —
bucket 6 being the pre-retrieval span that contains the trajectory's first
step opening — with 6 stragglers in buckets 3–5. Routed and entropy are
identical in buckets 3, 4 and 5.

**Partial deviation, stated as such:** bucket 2 also separates (0.772 vs
0.561) although no scaffold round lands there. That is a knock-on effect, not
a source effect — the arms' sequences diverge once the scaffold changes what
was emitted earlier, so their bucket-2 rounds are not the same rounds. Buckets
are not aligned across arms; equal round counts are coincidence.

Bucket 0 carries 22 rounds: descriptive only, never reported as a test.

## 4. Routed's content deficit persists, and here is the mechanism

Routed accepts 114 TEMPLATE + 183 content = 297 tokens. The comparator accepts
0 + 259. **Routed accepts 76 fewer content tokens** and still wins the systems
metric. The bookkeeping says exactly why:

| | rounds | opened by scaffold | by neural | by lookup | tokens emitted |
|---|---|---|---|---|---|
| routed | 299 | 51 | 175 | 73 | 596 |
| entropy | 336 | 0 | 263 | 73 | 595 |

Both arms open the same 73 lookup rounds. The scaffold's 51 rounds displace
**88 neural rounds** (263 → 175), and those displaced rounds were worth about
76 accepted content tokens (neural accepts 123 in the comparator, 47 in
routed). In exchange the scaffold contributes 114 TEMPLATE tokens over its 51
rounds at 0.63 acceptance.

So the scaffold is not adding acceptance on top of the content sources; it is
**buying cheap deterministic tokens with rounds the neural draft would have
spent on content**, and the trade is net-positive only because a scaffold
round is nearly free and the neural draft's per-round yield is low (47/175).
Two consequences, both load-bearing for the paper:

1. The margin is a **systems** margin — 1.993 vs 1.771 tokens per target
   forward — and carries **no content-quality claim** (novelty verdict, safe
   claim 6). Stated in exactly those words wherever it is quoted.
2. The trade would reverse if the neural draft got materially stronger. A
   scaffold source is worth most when the model draft is weak. That is a
   prediction the scale-up can falsify.

## 5. Other findings

- **The verb bet is the better primary**, as promoted: routed 0.993 with the
  bet against 0.886 with the shared prefix, and 1.993 against 1.886 tokens per
  target forward. Its per-token acceptance is lower (114/180 = 0.63 against
  82/94 = 0.87) because it proposes four tokens where the prefix proposes two,
  but it accepts more tokens in total.
- **Chaining sources within a round is still a no-op** — third run in a row.
  Identical rounds, identical 297 accepted tokens; only the count of proposed
  neural tokens moves (700 → 735). Pure extra draft compute.
- **Structure + entropy is identical to shared-prefix routed** (0.886, same
  source mix), the expected consequence of an inert gate.
- **The scaffold alone is the weakest arm** (0.238, and −0.119 against neural
  with the interval excluding 0): it defers on most rounds. It pays only in
  combination — the argument is for routing, not for the FSM.
- **The hop signal still does nothing**, re-measured under the corrected
  feature layout: baseline 68/403 = 0.1687, hop-signal 70/403 = 0.1737,
  positions aligned, discordant 7/9, exact McNemar **p = 0.804**. Reported
  descriptively only. This is not a retune and it does not reopen §11; it
  exists so the corrected-layout number is on record.

## 6. What this licenses

Licensed: with a draft trained on the layout it is served, routing three
structure-selected sources in a correctly simulated ReAct loop gives **1.993
tokens per target verification against 1.323** for the neural draft alone
(G1 2.42×, interval clear of zero), and beats a ReSpec-style entropy
comparator by 0.314 [0.235, 0.402] accepted tokens per round.

Licensed, and the sturdier claim: **entropy routing is uninformative in this
workload.** Two independent runs, tuning tables flat across an order of
magnitude of thresholds. Frame it as "entropy has nothing to route on here",
never as "structure beats entropy in general".

NOT licensed: any content-quality claim. Routed accepts fewer content tokens
than the comparator, by 76, and §4 gives the mechanism. NOT licensed either:
G1 as a final number — n=18 records on one undertrained draft.

## 7. Next

**G1 holds with the chat-trained draft, so the scale-up is scheduled.**

1. Collect 500–1000 trajectories with the existing pipeline
   (`01_generate_trajectories.py`), launched detached with `PPID=1` per spec
   §1, then `06_validate_shard.py` before spending any GPU time on it.
2. Retrain both arms with `--feature-mode chat` on the new shard; keep the
   split seed and re-fit the scaffold literal and verb on the train split
   only. Expect the fitted opening to change with a better-behaved agent.
3. Re-run the arms of this prereg on the new shard. Pre-register before
   running; the gates carry over unchanged.
4. Separately: replay both checkpoints chained in one run, to answer the
   checkpoint question in the regime that matters (see §0).
