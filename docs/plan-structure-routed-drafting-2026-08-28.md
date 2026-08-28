# Plan: structure-routed drafting for agentic RAG — 2026-08-28

Implementation plan for the pivot decided in
`next-direction-research-2026-08-28.md`. Written to be executed by a fresh
agent without this conversation's context. Read that notes file and the
repo spec `hopspc.md` first — especially §14/§15 (testing and working
rules); every rule there applies to new code too.

## 0. What is being built, and the claim under test

Three draft sources, routed by structure the agent loop already knows
(segment type, hop recency, passage provenance), measured in the chained
speculative evaluator (`scripts/07_chained_eval.py` machinery):

- **S1 — Scaffold FSM**: deterministic proposer for ReAct TEMPLATE spans.
  35.4% of generated tokens are template; prompt-lookup CANNOT cover them
  (P5: 0.092 acc on template positions) because the scaffold follows novel
  passage text. Highest-value, zero-cost source.
- **S2 — Scoped lookup (PLD)**: n-gram suffix match restricted to
  QUESTION + RETRIEVED_PASSAGE token positions. Held-out acc 0.404 pooled
  (≈ full-context 0.409), dominates far-from-hop content tokens.
- **S3 — Neural draft**: the existing HopSpecDraftModel (baseline arm,
  hop signal off). Weak pooled (0.141 on the undertrained pilot) but the
  only source with signal right after a hop (bucket 2) and on THOUGHT.

Oracle union on held-out = 0.476 pooled vs 0.409 best single source,
overlap 0.069 → routing headroom is real (P5, GO). The deliverable is the
REALIZED accepted-tokens-per-round of the routed system vs each single
source and vs an entropy-routing baseline (ReSpec-style).

## 1. Files to create / touch

```
src/hopspec/infer/routed_draft.py     S1 + S2 + router (new)
src/hopspec/infer/chained_eval.py     accept a pluggable DraftProposer
scripts/07_chained_eval.py            --draft-source {neural,lookup,scaffold,routed,entropy}
tests/test_routed_draft.py            new
tests/test_chained_eval.py            extend
```

Do NOT touch train/, eval/, data/ semantics. The measurement machinery is
frozen; this work only adds proposers.

## 2. `routed_draft.py` — design

### 2.1 ScaffoldFSM

State machine over the ReAct grammar the pipeline itself enforces
(`agent_pipeline.py` truncates to one Thought + one Action per step):

```
after passage-newline        -> propose tokenization of "Thought:"
inside THOUGHT content       -> no proposal (defer)
after thought-ending newline -> propose "Action: Search[" or "Action: Finish["
inside TOOL_CALL/ANSWER      -> no proposal (defer)
after "]"                    -> propose "\n"
```

Implementation notes:
- Work in TOKEN space, not characters: pre-tokenize the scaffold literals
  with the target tokenizer ONCE at init (`"Thought:"`, `" Action"`, etc.).
  Beware BPE leading-space attachment (spec §4): the token after
  "Thought:" is `" The"`-style content — the FSM must stop at the literal
  boundary, never swallow the next content token's leading space.
- The FSM does not know Search-vs-Finish. Propose the shared prefix
  `"Action: "` deterministically; stop there (defer to S3 for the verb),
  OR propose "Search[" (the majority continuation) and accept the miss on
  finish steps. Start with shared-prefix-only; measure both if cheap.
- State is derived from the committed tail (last tokens + their segment
  ids), which `ChainedSpeculator` already tracks. Pure function
  `next_span(committed_ids, committed_segments) -> list[int] | None`.

### 2.2 ScopedLookup

- Inputs: committed token ids + the set of positions whose segment is
  QUESTION or RETRIEVED_PASSAGE (scope), maintained incrementally.
- Longest-suffix n-gram match (n = 3 then 2, as in probes P2/P5) where the
  MATCH POSITION must be in scope; on hit, propose the following k tokens
  from the source span (k = remaining gamma), stopping at a scope
  boundary.
- Keep it simple first (linear scan is fine at these lengths); a suffix
  automaton is an optimization, not a requirement.

### 2.3 Router

`RoutedProposer` implements the existing `DraftProposer` protocol
(`propose(context_ids, num_tokens)`) so `run_speculative_round` and the
replay loop need no changes. Per call:

1. If ScaffoldFSM has a span → propose it (truncate to gamma).
2. Else if ScopedLookup hits → propose its continuation.
3. Else → neural draft (S3) for the remaining budget.

Sources may CHAIN within one proposal (scaffold span, then lookup for the
rest) — implement single-source-per-round first, chaining second; measure
the difference.

Log per round which source proposed (add a `source` field to the round
rows) — the per-source acceptance table is a paper figure.

### 2.4 Entropy-routing baseline

For the head-to-head: a router that picks lookup vs neural by the target's
last-token entropy (threshold tuned on train split only). The entropy of
the last committed position is already available in
`ChainedSpeculator.last_logits` — no extra forward. This is the
ReSpec-style comparator; structure-routing must beat or match it at zero
signal cost for the paper's claim.

## 3. Evaluator changes

- `ChainedSpeculator` currently is both proposer and verifier. Split:
  keep it as verifier/state-holder; let `replay_record` take an optional
  `proposer` (default = the speculator itself, preserving today's
  behavior and tests).
- `scripts/07_chained_eval.py`: `--draft-source` flag; `routed` and
  `entropy` need the checkpoint AND the lookup/FSM (built from the
  record's own labels for scope — labels are available at replay time by
  construction).
- Round rows gain `source`. `summarize_rounds` gains a per-source
  breakdown.

## 4. Tests (CPU, no network — non-negotiable)

1. FSM: after a passage-final newline, proposes exactly the "Thought:"
   tokenization; never proposes past a literal boundary; never emits a
   token owning a content token's leading space (regression: spec §4).
2. ScopedLookup: hit inside scope accepted, identical n-gram OUTSIDE
   scope ignored; proposal stops at scope end; deterministic.
3. Router: precedence order; falls through to neural when both miss;
   proposals always ≤ gamma; source labels correct.
4. replay with `--draft-source lookup` on conftest records produces
   rounds and restores the recorded rails (reuse existing replay tests
   parametrized over sources).
5. Entropy baseline: threshold respected; uses cached logits (assert no
   extra target forward — count runner.extend calls).

## 5. Experiment protocol (Mac, MPS, Qwen3-1.7B)

Data: existing `data/shard_1p7b.jsonl` (120 records, validated) and the
existing checkpoints `data/ckpt_base_1p7b.pt`. Same 15% held-out split,
seed 0 — NEVER re-split.

Runs (gamma=4, all 18 held-out records, `--rounds-out` for each):
  a. neural (already done: `data/rounds_base_1p7b.jsonl`)
  b. lookup           c. scaffold-only
  d. routed           e. entropy-routed
Report accepted-tokens-per-round per bucket AND per source, plus pooled.
Paired comparison per round is not byte-identical across sources (rounds
diverge) — compare per-record means with a paired test across the 18
records, and say so honestly.

Success gate: routed ≥ 1.5x neural-only mean accepted/round AND
routed ≥ entropy-routed. If routed loses to entropy-routed, the paper
claim shifts to "structure + entropy" — measure the combination before
concluding.

Then scale: collect 500–1000 trajectories (same script, overnight,
`nohup … & disown`, confirm PPID=1 — spec §1), retrain both draft arms,
rerun a–e. Only after the small run's gate passes.

## 6. Known traps (each cost real time already)

- Template tokens are EXCLUDED from `decode_phase_mask` measurement but
  ARE proposed and accepted in chained replay — accepted-per-round counts
  them. That is correct for a systems claim (they are real generated
  tokens) but report the split (template vs content acceptance) so the
  two claims never blur. This is the §3 "TEMPLATE exists" lesson in a new
  coat.
- `DynamicCache.crop` takes a NEGATIVE offset in transformers ≥ 5.16.
- Speculated tokens must never persist in the KV cache
  (`next_tokens` rolls back — keep the test that pins this).
- Mac runs: the collect step is hours; launch detached or it dies with
  the shell.
- Numbers from the 120-record pilot are direction-only; never compare a
  new run's absolute rates against them (spec §15).

## 7. Out of scope (explicitly)

- Retuning hop-embedding variants (NO-GO discipline).
- Tree/multi-candidate drafts, suffix automata, wall-clock benchmarks —
  after the acceptance-rate gate passes, not before.
- The mid-span injection experiment (science question) is a separate
  workstream; do not entangle it with this systems build.
