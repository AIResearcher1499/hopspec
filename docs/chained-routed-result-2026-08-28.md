# Result: structure-routed drafting in chained replay — 2026-08-28

> **ANNOTATION 2026-08-28 (added after the harness fix).** Every number in this
> file was produced in **`--replay-mode raw`**, which feeds the target the
> recorded document verbatim. The agent never saw that document: at collection
> each step went through `apply_chat_template(system + user(context so far))`.
> The measured consequence is in "The confound that dominates the scaffold arm"
> below — the raw-mode target reproduces the recorded step-opening token only
> 2/40 = 0.05 of the time, versus 0.70 elsewhere.
>
> Every arm here shares that handicap, so the comparisons **between the arms in
> this file** remain internally consistent and are not withdrawn. What they are
> not is an absolute claim about the deployed agent loop, and the scaffold arm
> in particular is measured at a position the harness could not verify.
>
> `chat` mode (prereg `prereg-chained-chat-replay-2026-08-28.md`) fixes this.
> Per `hopspc.md` §15 **no number from chat mode may ever be compared with a
> number in this file**: the measurement definition changed.

Implementation of `plan-structure-routed-drafting-2026-08-28.md`, and the
small-scale run it specifies (§5). Read `hopspc.md` §15 before quoting
anything here. **These numbers are from the 120-record 1.7B pilot with an
undertrained draft checkpoint; they are direction, not effect size, and must
never be compared against a number from a different run.**

## What was run

Target `Qwen/Qwen3-1.7B`, shard `data/shard_1p7b.jsonl` (120 records), the
existing 15%/seed-0 split (18 held-out, 102 train — NOT re-split), draft
checkpoint `data/ckpt_base_1p7b.pt` (baseline arm, hop signal off), gamma=4,
MPS. One command per arm:

```
scripts/07_chained_eval.py --target-model-name Qwen/Qwen3-1.7B \
  --trajectory-file data/shard_1p7b.jsonl --device mps --gamma 4 \
  [--checkpoint data/ckpt_base_1p7b.pt] --draft-source <arm> \
  --rounds-out data/rounds_<arm>_1p7b.jsonl
```

Raw rounds: `data/rounds_{neural,lookup,scaffold,routed,routedchain,entropy,
structent}_1p7b.jsonl`. Every row carries `source` and `token_sources`, so
the template/content split is recoverable offline.

Two fitted quantities, both fitted on the TRAIN split only and printed by the
script: the scaffold's opening literal, and the entropy comparator's
threshold (grid 0.25–3.0, chosen 1.5).

## Headline: accepted tokens per round, held-out

| arm | rounds | pooled | per-record mean | ratio vs neural | paired delta [95% boot] | sign p |
|---|---|---|---|---|---|---|
| neural (§11.4 arm) | 492 | 0.220 | 0.250 | 1.00x | — | — |
| scaffold only | 588 | 0.007 | 0.010 | 0.04x | −0.240 [−0.299, −0.186] | <0.001 |
| scoped lookup only | 449 | 0.405 | 0.474 | 1.90x | +0.224 [+0.082, +0.373] | 0.008 |
| **routed** (scaffold→lookup→neural) | 393 | **0.608** | 0.688 | 2.75x | +0.438 [+0.310, +0.579] | <0.001 |
| routed, chained sources | 393 | 0.608 | 0.688 | 2.75x | +0.438 [+0.310, +0.579] | <0.001 |
| entropy-routed (ReSpec-style) | 394 | 0.612 | 0.701 | 2.80x | +0.451 [+0.316, +0.596] | <0.001 |
| structure + entropy | 393 | **0.616** | 0.705 | 2.82x | +0.455 [+0.322, +0.599] | <0.001 |

Rounds are NOT byte-identical across arms (the sequences diverge as soon as
two sources propose differently), so paired McNemar over positions does not
apply between arms. The pairing above is per RECORD: one mean per held-out
record, 18 pairs, exact sign test plus a paired bootstrap over records. 18 is
a small n — read the interval, not the p-value.

## Gate verdict (plan §5)

- **routed ≥ 1.5x neural: PASS.** 2.75x, +0.438 accepted/round, interval well
  clear of zero.
- **routed ≥ entropy-routed: NOT MET, and it is a tie, not a loss.**
  −0.013 [−0.048, +0.019], sign test p=1.000. The plan's contingency applies:
  the combination was measured before concluding, and it is the best arm
  (0.616) — but by 0.004 over entropy alone, which is noise at this n.

So on this pilot the honest claim is: **routing wins, the routing SIGNAL does
not distinguish itself.** Structure and entropy buy the same thing here.

## Per bucket (mean accepted/round; rounds in parens)

| bucket | neural | lookup | scaffold | routed | entropy |
|---|---|---|---|---|---|
| 0 (at the hop) | 0.045 (22) | 0.000 (22) | **0.182 (22)** | **0.182 (22)** | 0.045 (22) |
| 1 (1–2) | 0.222 (36) | 0.250 (40) | 0.000 (40) | 0.343 (35) | 0.382 (34) |
| 2 (3–8) | 0.289 (76) | 1.098 (51) | 0.000 (94) | 1.283 (46) | 1.292 (48) |
| 3 (9–24) | 0.179 (78) | 0.476 (63) | 0.000 (93) | 0.661 (56) | 0.661 (56) |
| 4 (25–48) | 0.167 (42) | 0.116 (43) | 0.000 (49) | 0.171 (41) | 0.200 (40) |
| 5 (49+) | 0.500 (2) | 0.000 (3) | 0.000 (3) | 0.500 (2) | 0.500 (2) |
| 6 (no prior hop) | 0.233 (236) | 0.357 (227) | 0.000 (287) | 0.623 (191) | 0.620 (192) |

The one place the sources are not interchangeable is **bucket 0**: the lookup
scores 0.000 over 22 rounds there (P5 predicted this — the scaffold follows
novel passage text, so no n-gram match exists), and only the scaffold scores.
That is the mechanism the paper's spine claims. **It rests on 2 successful
rounds out of 22, 4 accepted tokens.** Do not quote it as an effect.

## The confound that dominates the scaffold arm

The scaffold FSM proposes 102 tokens and gets 4 accepted (0.039). That is not
the FSM being wrong — it is the replay harness's target not being the agent's
target:

- Against the RECORDED trajectory (what the agent actually wrote), the fitted
  FSM's proposals are **80/82 = 0.976** token-accurate on held-out; full-span
  hit 40/41. Betting on the action verb as well proposes far more
  (244 tokens) at 0.590, i.e. 144 accepted vs 80 — worth measuring properly.
- Against the REPLAY target, the target's greedy continuation reproduces the
  recorded token at only **2/40 = 0.05** of step-opening positions, versus
  **409/588 = 0.70** at every other generated position.

Cause: collection ran each step through `apply_chat_template` with
`SYSTEM_PROMPT` (`HFTargetLLM.generate`), and the system prompt is what tells
the model to write `Action:`. `replay_record` feeds the raw accumulated
context with no template and no system prompt, so at each step boundary the
replay target has no reason to open a ReAct step at all; it writes `Answer`
or `-`. Once the recorded `Action` token is teacher-forced back in, the rest
of the scaffold (`:`, ` Search`, `[`, …) is predicted correctly — the
handicap is specific to the first token of each step.

A static system-prompt prefix does NOT fix it: measured over 8 held-out
records, step openings go 0/18 → 3/18 (0.167) and other generated positions
0.706 → 0.814. The collection re-rendered the chat template per STEP, which a
single long sequence cannot reproduce. Fixing it properly means re-rendering
per step inside the replay — a change to the measurement machinery, which
this plan explicitly froze, and which would invalidate comparison with every
existing §11.4 number.

**Consequence for reading the table above: the scaffold and routed arms are
measured with their highest-value proposal position handicapped ~14x.** Every
arm shares the handicap, so the comparisons are internally consistent; the
absolute rates are not a systems claim about a deployed agent loop.

## Other findings

- **Chaining sources within a round changes nothing here.** It fires (57 of
  393 rounds get a mixed-source proposal, 115 extra neural tokens proposed)
  and accepts exactly the same 239 tokens: every token chained onto a
  scaffold or lookup span was rejected. At this checkpoint quality, chaining
  is pure extra draft compute. Single-source-per-round is the default.
- **The routed win is a CONTENT win, not template inflation.** Of 239
  accepted tokens in the routed arm, 4 are TEMPLATE (scaffold) and 235 are
  content. The §6 trap ("accepted-per-round counts easy template tokens")
  does not apply to this result — the opposite is true, template acceptance
  is missing because of the confound above.
- **The comparator is a strong one.** The entropy arm routes between the same
  SCOPED lookup and the same neural draft, so the head-to-head isolates the
  routing signal, not the provenance scope. Both routers beat their own best
  single source (0.405), so routing itself is what pays.
- Target forwards per emitted token: neural emits 1.220 tokens per
  verification, routed 1.608 — 1.32x fewer target forwards, ignoring draft
  cost. Wall-clock is explicitly out of scope until the acceptance gate is
  settled.

## What this does and does not license

Licensed: routing between a scoped lookup and a neural draft is worth
2.75–2.8x the neural draft alone on this pilot, and lookup-vs-neural
complementarity along the hop axis (P2/P5) reproduces in the realized
chained measurement — bucket 2 goes 0.289 → 1.28.

NOT licensed: "structure routing beats entropy routing". It does not, here.
It ties, and the one bucket where structure is uniquely right (bucket 0)
carries 4 accepted tokens.

## Next, in order

1. **Fix the replay prompt before anything else.** Per-step chat-template
   re-rendering in `replay_record`. Until then the scaffold source cannot be
   evaluated and the routed/entropy head-to-head is decided entirely by the
   two sources that do not depend on step openings. This is a measurement
   change: give it a new prereg and never compare its output to the numbers
   above.
2. Re-run a–f after the fix. Only then is the gate meaningful.
3. `--scaffold-verb fit` deserves a run: against the recorded trajectory it
   accepts 144 tokens vs the shared prefix's 80.
4. Scaling (500–1000 trajectories, retrain both draft arms) is what the plan
   schedules next, but it is second to the harness fix: more data will not
   move a 5% step-opening reproduction rate.
