# Next-direction research notes — 2026-08-28

Where to take HopSpec after the negative result on hop-aware draft
embeddings (spec §11, replicated here on Qwen3-1.7B / 120 trajectories /
teacher-forced AND chained regimes). All probes below are $0: offline over
`data/shard_1p7b.jsonl` plus one forward pass for entropies.

## Probe results (all reproducible from the shard)

**P1 — Generated-token composition.** Of 4,654 generated tokens: TEMPLATE
35.4%, ANSWER 29.2%, TOOL_CALL 25.5%, THOUGHT 10.0%. A third of everything
the agent emits is deterministic ReAct scaffolding.

**P2 — Prompt-lookup (3→2-gram, model-free) vs the trained EAGLE draft, on
decode-phase tokens by recency bucket:**

| bucket | PLD acc | PLD coverage | EAGLE draft acc (baseline arm) |
|---|---|---|---|
| 2 (3–8 after hop) | 0.247 | 0.332 | 0.153 |
| 3 (9–24) | 0.605 | 0.831 | 0.122 |
| 4 (25–48) | 0.503 | 0.706 | 0.196 |
| 5 (49+) | 0.801 | 0.901 | — (n=2) |
| no-hop | 0.346 | 0.451 | 0.131 |

(Caveat: PLD measured over the full shard, EAGLE over the 18 held-out
records with a 102-record training run — direction, not effect size.)
The two sources are complementary along the hop axis: lookup dominates far
from the hop (copying retrieved facts), is weakest right after it (formulaic
openings with nothing yet to copy) — exactly where a model draft is at its
best.

**P3 — Copy provenance.** Of 581 correct PLD predictions: 53.9% copy from
the QUESTION, 34.6% from a RETRIEVED_PASSAGE, only 11.5% from generated
text. A lookup scoped to inserted content keeps ~88% of the value with a
far smaller search space.

**P4 — Structure vs entropy.** Within EVERY target-entropy tercile,
settled positions (d>8) accept PLD ~2x near-hop and ~1.4x no-hop
(low-entropy: 0.843 vs 0.468 vs 0.589; mid: 0.598 vs 0.280 vs 0.353).
Hop recency is NOT subsumed by entropy — and unlike entropy it is free and
known BEFORE proposing (the tracker maintains it; entropy of the next
target step is not available until verification).

## Landscape (checked 2026-08)

- SuffixDecoding (arXiv 2411.04975, NeurIPS'25 spotlight): model-free
  suffix trees over past generations; large wins on agentic workloads;
  beats EAGLE-2/3 there. Owns "model-free for agents".
- SAM-Decoding (arXiv 2411.10666): suffix-automaton retrieval draft with
  heuristic fallback to EAGLE-2. Owns the basic hybrid mechanism.
- RASD (arXiv 2503.03434, ACL'25 Findings): retrieval tree fused with the
  model draft tree. Same territory.
- ReSpec (arXiv 2511.01282): adaptive routing between retrieval- and
  model-drafts via target-entropy trigger + EMA-scored match positions.
  Owns "principled routing" — but the signal is entropy, not structure.

Plain "lookup + EAGLE hybrid" is taken. "Routing" is taken with an
entropy signal. Untaken: **routing and scoping by agent structure** —
segment type, hop recency, and passage provenance, all free at inference
inside the agent loop.

## Ranked directions

1. **Structure-routed drafting for agentic RAG** (the pivot this data
   supports). Three draft sources, routed by structure the loop already
   knows: (a) a deterministic scaffold FSM proposes TEMPLATE spans
   (~35% of generated tokens, near-free acceptance); (b) a lookup draft
   scoped to question + retrieved passages (P3) for settled/TOOL_CALL/
   ANSWER regions; (c) the neural draft for near-hop/THOUGHT content.
   The paper's spine is the diagnostic: hop recency does not modulate
   neural-draft quality (the §11 negative, kept honest) but strongly
   predicts WHICH source wins (P2/P4). Head-to-head baseline: ReSpec-style
   entropy routing; the claim to test is structure ≥ entropy at zero
   signal cost, or structure + entropy > either.
   Risk: crowded space; the defensible core is the measurement, the
   routing win must be shown against SAM-D/ReSpec-class baselines.

2. **The clean science question** (separate, smaller): break the
   within-sentence confound by injecting a passage MID-SPAN and comparing
   same ordinal positions with/without the context shift. Answers the
   original hypothesis properly; publishable as analysis regardless of
   direction. Cheap on the Mac at 1.7B.

3. **Not recommended:** scaling the current design (more data, bigger
   target) — it will only tighten the same negative; and retuning
   hop-embeddings variants — that is retuning after a NO-GO.

## P5 — Oracle-union gate (held-out only, alignment verified) — GO

403 decode-phase positions, joined element-for-element with the baseline
arm's raw columns:

| bucket | n | neural | PLD full | PLD scoped | UNION | overlap |
|---|---|---|---|---|---|---|
| 2 | 59 | 0.153 | 0.169 | 0.119 | 0.254 | 0.017 |
| 3 | 90 | 0.122 | 0.578 | 0.578 | 0.622 | 0.078 |
| 4 | 46 | 0.196 | 0.435 | 0.457 | 0.565 | 0.087 |
| no-hop | 206 | 0.131 | 0.398 | 0.398 | 0.451 | 0.078 |
| pooled | 403 | 0.141 | 0.409 | 0.404 | **0.476** | 0.069 |

- Sources are NEARLY DISJOINT (overlap 0.069): union = 3.4x the neural
  draft, +6.7pt over PLD alone. Routing/merging has real headroom.
- Scoping the lookup to question+passages costs ~nothing (0.404 vs
  0.409) — P3's provenance claim holds on held-out.
- Surprise: PLD CANNOT predict TEMPLATE tokens (0.092 on 207 template
  positions) — the scaffold follows novel passage text, so no n-gram
  match exists. An FSM is genuinely load-bearing for the 35% template
  share, not a nice-to-have; none of the published lookup methods covers
  it.
- Caveats: teacher-forced per-token numbers; draft undertrained (102
  records); realized speedup requires the routed chained measurement.

## Immediate $0/cheap next steps

- [x] Head-to-head on held-out only: scoped-PLD vs neural draft vs oracle
      union, per bucket (upper bound for routing). → P5 above, GO.
- [ ] Scaffold FSM prototype: regex-level proposer for TEMPLATE spans,
      measure realized acceptance in chained replay (07 machinery).
- [ ] Add a `--draft-source` switch to the chained evaluator to measure
      routed acceptance per round.
- [ ] Mid-span injection pilot (direction 2), 20 trajectories.
