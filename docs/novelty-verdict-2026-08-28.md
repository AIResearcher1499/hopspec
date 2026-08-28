# Novelty verdict and claim discipline — 2026-08-28

Digest of an external deep-research sweep (Grok, two passes: a broad
Partial sweep, then a full-text read of the flagged gaps on the same day).
This file is the claim police for the paper: check every draft claim
against it. Companion to `next-direction-research-2026-08-28.md` and
`chained-routed-result-2026-08-28.md`.

## Verdict in one line

The slot is occupied on more sides than the first pass suggested, but not
killed. Token-level EXCLUSIVE routing exists (WhiFlash — AR vs diffusion,
by entropy/hidden state). Agent-STRUCTURE-aware drafting exists (AgentSpec
— app-sent semantic-block tags, used to ISOLATE the lookup cache). Still
unclaimed: routing lookup vs neural vs scaffold-FSM **source selection**
by hop recency / ReAct segment / question-vs-passage provenance, measured
**inside a multi-hop RAG loop that retrieves mid-generation with per-step
chat re-rendering** — no public benchmark covers that loop.

The one-sentence delta for the paper: *AgentSpec is structure-aware
retrieval SCOPING; WhiFlash is confidence-based paradigm ROUTING; HopSpec
is structure-based SOURCE routing with a grammar drafter for the
post-retrieval n-gram blind spot, in the loop none of them evaluate.*

## Safe claims (defensible with our data)

1. In ReAct multi-hop RAG, lookup and a small neural draft fail in
   different places; lookup is structurally blind on scaffold tokens that
   follow a novel retrieved passage (no n-gram match can exist).
2. Hop recency, segment type, and copy provenance are available BEFORE a
   draft is proposed, unlike target entropy (which needs a verification
   first). Structure matches entropy routing at zero signal cost.
3. A deterministic scaffold FSM is the natural cover for (1); no published
   lookup method covers it.
4. No public benchmark evaluates SD in a multi-hop RAG loop whose context
   is rewritten by retrieval mid-generation with per-step chat-template
   re-rendering. Our chat-mode replay harness is that benchmark.
5. (chat-mode result, 2026-08-28) In the correctly prompted loop, a
   target-entropy routing signal is DEGENERATE at routing decision points
   (median 0.000 nats at round starts; 87.5% of rounds below every grid
   threshold) — it has nothing to route on, while structure is free and
   separates exactly at step openings, as pre-registered. Frame as "entropy
   is uninformative in this workload", never as "structure beats entropy
   in general" (WhiFlash shows entropy routing works elsewhere).
6. The routed-vs-comparator margin is a SYSTEMS margin (1.800 vs 1.254
   tokens per target verification), carried by TEMPLATE tokens; routed
   accepts fewer content tokens than the comparator. No content-quality
   claim exists.

## Forbidden claims

- "First retrieval–neural hybrid" (SAM-Decoding, RASD, CopySpec, Token
  Recycling, SuffixDecoding; Graft for fusion).
- "First grammar/FSM speculative drafting" (ToolSpec is an FSM drafter for
  tool schemas; OoO-Spec has a schema FSM + sidecar).
- "First agentic SD evaluation" (SuffixDecoding: SWE-Bench/AgenticSQL;
  AgentSpec: Reflexion/DeepResearch/GAIA; WhiFlash: SWE/Tau/AgentBench).
- "First token-level exclusive routing" (WhiFlash routes EAGLE-3 vs block
  diffusion per token, EMNLP 2026).
- "First use of agent structure for drafting" (AgentSpec's semantic-block
  isolation IS agent structure — used for cache scoping, not source
  selection; its ablation credits structure isolation over budget
  allocation, so do not minimize it).
- "Structure routing beats entropy routing" — it tied (0.608 vs 0.612;
  structure+entropy 0.616; n=18 records). WhiFlash makes this ban
  stronger: token-level entropy routing demonstrably works and is
  published.

## Must-cite and the distinction sentence for each

| Work | What it is | Our delta |
|---|---|---|
| ReSpec (arXiv 2511.01282) | entropy+EMA routing between SAM lookup and neural | signal is post-hoc entropy; ours is pre-proposal structure; we tie at zero cost |
| SAM-Decoding (2411.10666) | suffix-automaton draft, match-length fallback | match length ≠ agent structure; their hybrid gain is small and can reverse |
| RASD (2503.03434, ACL'25 F) | fuses retrieval tree into EAGLE tree | fusion, not routing; always pays the neural draft |
| SuffixDecoding (2411.04975, NeurIPS'25 spot.) | suffix trees over past agent traces | repetition across requests, not hop structure within one trajectory |
| ToolSpec (2604.13519) | FSM drafts deterministic tool-call schema | schema after `<tool_call>`; ours is ReAct scaffold after retrieval + provenance routing |
| OoO-Spec (2608.00814) | schema FSM + 0.6B sidecar | no hop recency, no provenance |
| AgentSpec (2608.24004, EMNLP'26 per abs comment) | app-sent semantic-block tags isolate per-block lookup caches (rejection ~26% vs NGram >85%); batched serving eval | structure-aware retrieval SCOPING, model-free, no source switching; also the warning: batched speedups collapse to ≤1x — do not promise deployment gains |
| WhiFlash (2606.07710, EMNLP'26) | token-level exclusive routing EAGLE-3 vs block diffusion, by target entropy or an MLP on target hidden states | closest routing paper; signal is entropy/hidden-state (post-target), paradigms are AR-vs-diffusion, no retrieval, no grammar drafter, not a RAG loop |
| AsymSpec (2608.26004, EMNLP'26 Main) | drafter reads full context, verifier reads compressed; delta-fusion logits; GAIA ReAct CodeAgent | orthogonal (not lossless SD); one-line cite |
| Graft (2605.20104) | prunes the neural tree, grafts retrieved tokens into the gaps | fusion, RASD family, not routing |
| Spec-Bench "RAG" | single-prompt DPR slice, 80 instances | not a loop; the field's RAG evidence is this thin — our harness is the fix |

Reference speedups reviewers will hold us to (single-prompt unless noted):
ReSpec ~2.5–2.8x RAG; EAGLE-2 ~2.3x overall; SuffixDecoding 2.5–5.3x on
agent traces (bs=1); AgentSpec shows ~1x for everything under batching.

## Reframed contribution (post-sweep)

Lead with the measurement, not the router:

1. **Benchmark/harness**: chained SD evaluation inside a real ReAct
   multi-hop RAG loop (chat-mode replay, per-step re-rendering) — the
   setting no existing benchmark covers (§Safe-4).
2. **Findings**: source disjointness along the hop axis; the scaffold
   blind spot; the §11 negative (hop signal doesn't help the neural draft)
   reframed as "structure predicts WHICH source wins, not how to improve
   one source"; structure == entropy at zero cost.
3. **System**: scaffold FSM + scoped lookup + neural with structural
   routing as the reference implementation.

Venue fit (per sweep): MLSys, ACL/EMNLP Findings or industry track,
ES-FoMo — measurement-study frame. NOT a "new algorithm beats X" frame.

## Sweep coverage — second pass (same day) closed the first pass's gaps

Resolved by full-text reads:

- WhiFlash: read in full — see the must-cite table. EMNLP'26 (author
  announcement, 21 Aug 2026).
- AgentSpec: read in full — the first pass's "just batched serving" was
  WRONG; it has structure-isolated drafting. EMNLP'26 per its own arXiv
  comment; camera-ready not yet independently confirmed.
- ReSpec: still preprint-only (v1, no venue, no v2). Cite as arXiv.
- AsymSpec and Graft: caught in the second pass, both distinguished above.
- X/Twitter: no research discussion of ReAct-hop routing found.

Still open before submission:

- Guidance-style interleaved template generation not checked in primary
  source.
- AgentSpec camera-ready confirmation.
- Re-run the whole sweep near submission; the area moves monthly (both
  closest papers appeared within four weeks of this sweep).
