# HopSpec — full rebuild specification

You are rebuilding a research codebase from scratch. This document is the complete
specification: read it end to end before writing any code. It is written to be
sufficient on its own — you should not need the original repository.

The project has been through fifteen debugging cycles, and roughly half of this
document is a record of specific mistakes that were made, the wrong results they
produced, and the invariants that now prevent them. **Those sections are not
background colour. They are the requirements.** Every one of them describes a bug
that produced confident, plausible, completely wrong numbers.

---

## 0. What you are building, and why

**HopSpec: hop-aware speculative decoding for agentic multi-hop RAG.**

In agentic multi-hop RAG (ReAct/Search-R1 style) an agent alternates
`Thought → Action: Search[query] → Observation: <retrieved passage>` for several
hops before answering. A speculative-decoding draft model trained the usual way
(EAGLE-style, on ordinary text) has never been trained to expect its context to
be rewritten mid-generation by a freshly retrieved passage.

**Hypothesis.** Token-acceptance rate drops right after each such hop boundary and
recovers as the model "settles into" the new context. If so, two additions should
help:

1. **Hop-boundary signal** (training time): auxiliary embeddings telling the draft
   model what kind of span it is in and how long ago the context last shifted.
2. **Adaptive speculation length** (inference time): propose fewer tokens right
   after a hop, more once settled — composable with EAGLE-2's confidence-based
   dynamic tree, not a replacement for it.

**Empirical status after building this: the hypothesis is not supported.** Section
11 gives the results and the reasoning. You are rebuilding a codebase whose main
finding is a well-established negative plus the measurement machinery that makes
that negative trustworthy. Build it so the negative stays trustworthy.

---

## 1. Environment and conventions

- Python ≥3.10. Use `uv`, not raw pip/venv: `uv venv .venv` then
  `uv pip install -e ".[dense-retrieval]" --python .venv/bin/python`.
- **Do not use `uv venv --system-site-packages`.** It caused real ABI breakage
  (system `pandas`/`numpy` vs. venv numpy broke `datasets`; system `torchvision`
  built against an older torch crashed `transformers` on import with
  `RuntimeError: operator torchvision::nms does not exist`).
- Tests must run on CPU with no model download and no network.
- `pyproject.toml`: setuptools backend, `packages.find` with `where = ["src"]`.
  Dependencies: `torch>=2.5`, `pytest>=8.0`, `pyyaml>=5.4`, `transformers>=4.44`,
  `accelerate>=1.0`, `datasets>=3.0`, `sentencepiece`, `huggingface_hub`. Optional
  extra `dense-retrieval = ["sentence-transformers", "faiss-cpu"]`.
- Target size: ~2,700 lines under `src/`, ~1,000 under `scripts/`, ~2,200 under
  `tests/`, 169 tests. If your test count is far lower, you have skipped the
  regression tests that are the point of this document.

### Long-running jobs

GPU jobs here run 30 minutes to 2 hours. Launch them detached
(`nohup … > log 2>&1 < /dev/null & disown`) and confirm `PPID=1`. A plain
background job dies with its parent shell; that once destroyed 1.5 hours of index
building because the script had no checkpointing.

---

## 2. Package layout

```
src/hopspec/
  data/
    schema.py            SegmentType, RECENCY_BUCKETS, TrajectoryStep, Trajectory, LabeledSequence
    agent_pipeline.py    LLM protocol, MockLLM, HFTargetLLM, run_react_trajectory
    segment_labeling.py  OffsetTokenizer, label_trajectory, assign_tokens_to_steps
    collect.py           benchmark loading + trajectory_to_record + collect_shard + CLI
    question_split.py    deterministic collect/eval question-id split (leakage guard)
    retriever.py         Document, BaseRetriever, InMemoryBM25Retriever, DenseRetriever
    wiki_dpr.py          psgs_w100 TSV streaming
    gold_titles.py       per-benchmark supporting-fact title extractors
    pilot_corpus.py      gold-title ∪ random-background corpus split
    reservoir_sampling.py Algorithm R
  model/draft_model.py   HopSpecDraftConfig, HopSpecDraftModel
  train/
    loss.py              recency_weights, hop_aware_ce_loss, hop_aware_feature_loss, hop_aware_kd_loss
    train_draft.py       collate, eagle_aligned_batch, standalone train CLI
    target_policy.py     LossTargetPolicy, GeneratedTokensOnlyPolicy
    ablation.py          train_draft_model, evaluate_draft_model (shared by both runner scripts)
  infer/
    adaptive_length.py   HopAwareLengthPolicy
    speculative_generate.py RecencyStateTracker, run_speculative_round
  eval/
    metrics.py           acceptance_rate_by_bucket
    diagnostic.py        decode_phase_mask, resolve_recency_buckets_for_model, bucket_token_*
    analysis.py          BucketSummary, novel_hop_flags, select, summarize, rebucket, format_table
scripts/                 00_smoke_test … 06_validate_shard (see §9)
tests/                   one file per module
```

---

## 3. `data/schema.py`

```python
class SegmentType(IntEnum):
    QUESTION = 0
    THOUGHT = 1
    TOOL_CALL = 2
    RETRIEVED_PASSAGE = 3
    ANSWER = 4
    OTHER = 5      # padding and malformed steps
    TEMPLATE = 6   # fixed ReAct scaffolding

NUM_SEGMENT_TYPES = len(SegmentType)   # 7

RECENCY_BUCKETS = ((0,0), (1,2), (3,8), (9,24), (25,48), (49,None))
NO_PRIOR_HOP_BUCKET_ID = len(RECENCY_BUCKETS)      # 6
NUM_RECENCY_BUCKETS = len(RECENCY_BUCKETS) + 1     # 7
NO_PRIOR_HOP_DISTANCE = 10**9

def recency_bucket_id(distance: int) -> int:
    # raises ValueError on negative
    # NO_PRIOR_HOP_DISTANCE -> NO_PRIOR_HOP_BUCKET_ID, never the farthest distance bucket
```

`TrajectoryStep(segment_type, text, hop_index=None)`,
`Trajectory(question, steps, final_answer, is_complete, context)`,
`LabeledSequence(input_ids, segment_type_ids, recency_bucket_ids, recency_distances,
hop_boundary_positions)` with a `__post_init__` that rejects unequal lengths.

### Why `TEMPLATE` exists — required, not cosmetic

Scaffolding (`"Thought:"`, `"Action: Search["`, `"]"`, inter-step newlines) is
emitted by the model, so it must be **trained on**, but it is template-determined,
so it must be **excluded from measurement**. Without a separate type, the first
token after every hop is always literally `"Thought"` and your post-hop
"acceptance" number measures template predictability. This produced a spurious
"+50 point improvement" that survived until someone asked what a constant
predictor scores.

### Why `NO_PRIOR_HOP_BUCKET_ID` exists — required

The obvious design pins pre-first-retrieval tokens to the farthest distance
bucket and calls them "settled". **Do not.** Measured on real data that made the
farthest bucket **99.6% pre-retrieval positions**, so "just after a hop vs.
settled after a hop" silently became "after retrieval vs. before any retrieval
ever" — regimes differing in context length and trajectory stage, not distance.

### Why these bucket boundaries

They were fitted to the measured distribution, not chosen a priori. Post-hop
generated spans run **p50=27, p95=74, max=152** tokens. An earlier `(9,32),(33,128),
(129,None)` tail put 0.2% of post-hop tokens in the far bucket (unreachable) and
85% in two middle buckets. The current boundaries give roughly 2/12/32/35/19%.

---

## 4. `data/agent_pipeline.py`

### The ReAct loop

System prompt instructs exactly one `Thought:` + one `Action: Search[...]` or
`Action: Finish[...]` per turn, and never to write an `Observation:` itself.

```python
_SEARCH_RE  = re.compile(r"Action:\s*Search\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_FINISH_RE  = re.compile(r"Action:\s*Finish\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=Action:|$)", re.IGNORECASE | re.DOTALL)
_THOUGHT_LABEL_RE = re.compile(r"\s*Thought:", re.IGNORECASE)   # NO trailing \s* — see below
```

`_truncate_to_first_action(text)` keeps only up to the end of the first Search or
Finish match. Real models ramble past their action and hallucinate their own
`Observation:`; truncating stops that entering the context. `HFTargetLLM` applies
it; `MockLLM` returns clean steps and does not need it.

`HFTargetLLM.__init__(model_name, device="cuda", max_new_tokens=200)`. Build the
prompt with `apply_chat_template(messages, add_generation_prompt=True,
enable_thinking=False)`, falling back to no `enable_thinking` on `TypeError`
(templates that reject the kwarg). Strip any leftover `<think>…</think>`. Generate
with `do_sample=False`.

`_format_passage(docs)` → `"Observation: " + "\n".join("- " + d.text)`.

### THE CENTRAL INVARIANT

```python
"".join(step.text for step in trajectory.steps) == trajectory.context
```

Every step's `text` is the **verbatim slice** of the agent's context it
contributed — template prefixes and trailing newlines included. `run_react_trajectory`
builds `context` as:

```
"Question: {q}\n" + (raw_step + "\n") + (passage + "\n") + (raw_step + "\n") + …
```

and appends steps that reproduce exactly those substrings.

**The bug this prevents, in full, because it is the worst one in the project's
history.** The original code stored `thought_match.group(1).strip()` and a
re-rendered `f"Action: Search[{query}]"`, dropping the `"Thought: "` prefix and
every newline. Labeling then concatenated step texts with no separator, so a hop
boundary decoded as:

```
'...a chance in education and toThe observation indicates that...'
```

— the passage running straight into the next thought with no delimiter, whereas
the real context had `Observation: ...\nThought: The observation indicates ...`.
Consequences: the target model's hidden states were computed over a document that
never existed; the baseline could not possibly know where a passage ended, so its
post-hop acceptance was 4.9%; and that artifact was reported as the project's
confirming result. **Assert the invariant in tests for 2-hop, malformed-step,
incomplete, and finish trajectories.**

### `_split_generated_step(rendered) -> list[TrajectoryStep]`

Span-slicing, never regex-extract-and-reformat. `"".join(s.text) == rendered`
always. For `"Thought: I need X.\nAction: Search[q]\n"`:

| span | type |
|---|---|
| `"Thought:"` | TEMPLATE |
| `" I need X."` | THOUGHT |
| `"\n"` | TEMPLATE |
| `"Action: Search["` | TEMPLATE |
| `"q"` | TOOL_CALL |
| `"]\n"` | TEMPLATE |

For `Finish[...]` the payload is `ANSWER` instead of `TOOL_CALL`. The answer is a
**sub-span of the tool call**, not a separate appended step — that is where it
actually occurs in the context; appending a second copy fabricates text.

If neither action matched, emit the whole thing as THOUGHT (if a `Thought:` label
is present) or OTHER.

**`_THOUGHT_LABEL_RE` must not consume trailing whitespace.** BPE attaches a
leading space to the next word (`" The"` is one token). A TEMPLATE span that
swallowed the space would own that token's first character, and offset-based
attribution would then classify the first *content* token as scaffolding —
silently excluding the single most important token in the project from every
measurement.

---

## 5. `data/segment_labeling.py`

```python
OffsetTokenizer = Callable[[str], tuple[list[int], list[tuple[int, int]]]]

def hf_offset_tokenizer(hf_tokenizer) -> OffsetTokenizer   # uses return_offsets_mapping=True
def trajectory_text(trajectory) -> str                     # "".join(step.text)
def assign_tokens_to_steps(trajectory, offsets) -> list[int]
def label_trajectory(trajectory, tokenize: OffsetTokenizer) -> LabeledSequence
```

**Tokenize the whole trajectory in ONE call and map tokens back to steps by
character offset.** Per-step tokenization then concatenating ids cannot reproduce
the tokenization of the joined string, because subword merges cross step
boundaries — and the discrepancies land exactly at step boundaries, which is where
the measurement looks. A token straddling a boundary is attributed to the step its
**first character** falls in.

### Recency distance definition

Distance counts tokens since the **END of the most recently CLOSED**
`RETRIEVED_PASSAGE` step — since the model resumed generating — not since its
start. Anchoring at the start puts the fine-grained near-zero buckets entirely
inside mechanically-inserted passage content that is never a speculation target;
the first diagnostic run had 100% of measured positions in the single farthest
bucket because of this.

Implementation: iterate steps in order; for each step assign
`token_index - last_anchor` (or `NO_PRIOR_HOP_DISTANCE`); **after** a
`RETRIEVED_PASSAGE` step's tokens are accounted for, set
`last_anchor = last_token_index + 1`. `hop_boundary_positions` records where each
passage *starts* (unchanged, used for bookkeeping).

Passage-internal tokens keep counting from the previous hop's close. That differs
from the online tracker (§8), which never sees them; the two agree at every
position that is actually speculated on, which is the only place the number is used.

### Unicode caveat — expected, not a bug

For NFD source text (`"e"` + combining acute) `decode(input_ids) != text`, because
the tokenizer normalizes to NFC. Measured: ~3% of real HotpotQA records, and in
every case `NFC(decode(ids)) == NFC(text)`. Labels are unaffected: offsets index
the original string and stay monotonic (the combining mark falls in a gap between
token spans), step boundaries sit at word edges so they never split a base
character from its accent, and recomputed `segment_type_ids` match exactly.
Compare round-trips under NFC.

---

## 6. `model/draft_model.py`

EAGLE-1 style:

```
concat(token_embedding, target_feature) → Linear → (+ segment_embed + recency_embed)
  → single pre-norm causal decoder block → predicted_feature → frozen target LM head → logits
```

```python
@dataclass
class HopSpecDraftConfig:
    target_hidden_size: int
    vocab_size: int
    draft_hidden_size: int = 512
    num_segment_types: int = NUM_SEGMENT_TYPES      # derive from schema, never hardcode
    num_recency_buckets: int = NUM_RECENCY_BUCKETS  # ditto
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0
```

Hardcoding these once made adding a `SegmentType` produce an index-out-of-range
30 minutes into a GPU run.

`HopSpecDraftModel.from_target_embedding(config, target_embedding)` reuses the
target's embedding table and freezes it, so draft and target representations stay
compatible. `forward(token_ids, target_features, segment_type_ids,
recency_bucket_ids)` validates shapes, builds a causal mask, returns predicted
features. `predict_logits(predicted_features, lm_head_weight)` returns
`predicted_features @ lm_head_weight.t().to(predicted_features.dtype)`.

**No RMSNorm before the head.** Verified against EAGLE's source: `cnets1.py`
(EAGLE-1) does `last_headout = head(last_hidden)`. The `lm_head(self.norm(...))`
form exists only in the EAGLE-3-style `cnets.py`.

**Dtype:** the target model is usually bfloat16 while the draft is float32. Cast
extracted features to `.float()`, and cast `lm_head_weight` inside `predict_logits`
rather than at each call site. Both mismatches were real runtime crashes.

---

## 7. `train/` — the EAGLE contract

### 7.1 `collate(batch, pad_id)`

Pads `input_ids` with `pad_id`, `attention_mask` with 0, `recency_distances` with
`10**9`, `recency_bucket_ids` with `NUM_RECENCY_BUCKETS - 1`, `segment_type_ids`
with `SegmentType.OTHER`.

**Never pad `recency_bucket_ids` with 0.** Bucket 0 is "right after a hop", the
rarest and most important class. Measured on a real 850-trajectory training set:
**257,814 padding positions mislabeled bucket-0 versus 1,387 genuine ones — a 186x
noise-to-signal ratio** landing on the one embedding row the method most needs.
Invisible to the disabled-signal baseline (which overrides every bucket anyway),
so it only corrupted the arm under test.

### 7.2 `eagle_aligned_batch(target_model, batch, feature_layer=-1)`

**Verified against EAGLE's source code, not its paper.** The paper's "second-to-top
layer" wording is ambiguous and a fetched summary of the repo got it backwards.
The decisive lines, `eagle/model/ea_model.py`:

```python
outputs = self.base_model.model(...)
orig = self.base_model.lm_head(outputs[0])
hidden_states = outputs[0]          # the draft model gets the SAME tensor
```

and `modeling_qwen3_kv.py` applies `self.norm(hidden_states)` before returning it
as `last_hidden_state`. So EAGLE's feature is the LM-head input = HF's
`hidden_states[-1]`.

Using `hidden_states[-2]` instead is a different object: on Qwen3-4B its norm is
4.4x larger (640 vs 146) and pushed through the LM head it reproduces the target's
own next token 24% of the time versus 81.5%. Fixing this alone improved pooled
acceptance by **+3.7 points (p=2e-16)**.

Write `q` for the index of the token being **predicted**; `q` runs `2 … T-1`, so
every returned tensor has length `T-2`:

| key | value | meaning |
|---|---|---|
| `target_feats` | `feats[:, :-2]` | `f_{q-2}` — the draft's feature input |
| `feature_targets` | `feats[:, 1:-1]` | `f_{q-1}` — EAGLE's regression target |
| `draft_token_ids` | `input_ids[:, 1:-1]` | `e_{q-1}` |
| `segment_ids_input` | `segment_type_ids[:, 1:-1]` | segment at `q-1` |
| `recency_buckets` | `recency_bucket_ids[:, 2:]` | bucket at `q` |
| `recency_distances` | `recency_distances[:, 2:]` | distance at `q` |
| `hop_index` | see below | which hop governs `q`, −1 before the first |
| `target_token_ids` | `input_ids[:, 2:]`, padding → −100 | token at `q` |
| `segment_ids_target`, `attention_mask_target` | labels at `q` | for the eval mask |

Confirmed against `eagle/train/main.py`, which builds exactly this shift:

```python
input_ids_target = input_ids[:, 1:]        # token shifted one ahead
target           = hidden_state[:, 1:, :]  # regression target shifted one ahead
new_data["hidden_state_big"] = hidden_state    # input feature NOT shifted
```

i.e. `(f_t, e_{t+1}) → f_{t+1}`; substituting `q = t+2` gives the table above.

**Two off-by-one bugs this replaces, both of which produced published-looking
numbers:**

- *Degenerate input pairing.* The old code passed `(f_{q-1}, e_{q-1})` — the same
  index. `f_{q-1}` already nearly determines token `q`, so the token embedding was
  redundant and the task collapsed into "reconstruct the target's last layer",
  inflating acceptance to ~0.83 and leaving no headroom for any auxiliary signal.
- *Labels describing the wrong token.* Mask, grouping, aux embeddings and loss
  weight were all taken at `q-1` while the prediction was at `q`. Verified
  consequence: **all 247 predictions of the first post-hop token were silently
  dropped from the measurement** (their label position was `RETRIEVED_PASSAGE`, so
  `decode_phase_mask` excluded them), and the numbers reported as "bucket 0" were
  the *second* post-hop token.

`hop_index` must be computed on the **full** sequence and only then sliced:
`(batch["recency_distances"] == 0).cumsum(dim=1) - 1`. Each closed passage
contributes exactly one `distance == 0` position. Counting in the already-sliced
array would miss a hop closing at absolute position 0 or 1 and shift every later
index. Real data has a ~434-token margin; the margin is not the guarantee.

`segment_ids_input` deliberately uses `q-1`, not `q`: the model may only be fed
things known at inference. The recency bucket at `q` is known (the tracker
maintains it and `run_speculative_round` reads it before proposing); the segment
type of a not-yet-generated token is not, while the previous token's is — and that
is what carries the "a passage just ended" transition anyway.

### 7.3 `train/loss.py`

```python
def recency_weights(d, alpha=2.0, tau=16.0):   # 1 + alpha * exp(-d / tau), tau > 0
def hop_aware_ce_loss(logits, targets, distances, alpha=2.0, tau=16.0, ignore_index=-100)
def hop_aware_feature_loss(pred_feats, target_feats, distances, valid_mask, alpha=2.0, tau=16.0)
def hop_aware_kd_loss(logits, target_logits, distances, alpha=2.0, tau=16.0, temperature=1.0)
```

All are **weighted means**: `(per_token * weights).sum() / weights.sum().clamp_min(1e-8)`.
`hop_aware_ce_loss` zeroes weights where `targets == ignore_index`.
`hop_aware_feature_loss` uses `smooth_l1_loss(..., reduction="none").mean(-1)`.

EAGLE's actual objective, from `eagle/train/main.py`:

```python
vloss = SmoothL1Loss(reduction="none")(predict, target)   # next-feature regression
ploss = -sum(target_p * log_softmax(head(predict)))       # soft KL on tokens
loss  = 1.0 * vloss + 0.1 * ploss
```

EAGLE trains the draft **primarily to regress the next hidden state**; the token
term carries only 0.1 weight. Supervising tokens alone — which this project did
originally — omits the signal carrying ~91% of EAGLE's loss weight. Use
`loss = v_w * feature_loss + p_w * token_loss` with defaults `v_w=1.0, p_w=0.1`.
Apply `w(d)` to **both** terms: the method's claim is that near-hop positions
deserve more weight, which should hold for whichever objective is learning.

(Hard CE is retained for the token term instead of EAGLE's soft KL — a documented,
deliberate simplification, since soft KL needs the target's full `[B,T,V]` logits.)

### 7.4 `train/target_policy.py`

```python
class LossTargetPolicy:                       # score all non-padding positions
    def loss_targets(self, aligned): return aligned["target_token_ids"]

class GeneratedTokensOnlyPolicy(LossTargetPolicy):
    PREFILL_SEGMENTS = (SegmentType.QUESTION, SegmentType.RETRIEVED_PASSAGE)
    def loss_targets(self, aligned):
        targets = super().loss_targets(aligned)      # reuse the padding mask
        return targets.masked_fill(is_prefill, -100)  # add one rule
```

The subclass **extends** the base's masking rather than reimplementing it, so it
cannot forget padding. `TEMPLATE` stays scored: the model really does emit
`"Thought:"`, and "exclude from measurement" is a different question from
"exclude from training".

**Keep the default `all`.** `generated` was tested and **refuted**: masking prefill
made acceptance significantly *worse* in all 8 paired comparisons (−2.5 to −7.7
points, most p<1e-6). The motivating reasoning — "82% of loss weight trains on
tokens never speculated on, so a 1-block draft model wastes its capacity" — was
plausible and wrong: capacity was not the binding constraint, **data** was. Masking
removes ~86% of next-token supervision from an already small corpus. Keep the class
for a retest at much larger data scale.

### 7.5 `train/ablation.py`

Both runner scripts share this so the only thing differing between arms is the
switch under test.

```python
def train_draft_model(draft_model, target_model, train_records, pad_id, device,
                      epochs, batch_size, lr, log_every, hop_signal_enabled,
                      alpha, tau, loss_target_policy=None,
                      feature_layer=-1, v_w=1.0, p_w=0.1) -> None

def evaluate_draft_model(draft_model, target_model, heldout_records, pad_id,
                         device, batch_size, hop_signal_enabled,
                         feature_layer=-1) -> dict[str, list[int]]
```

**`evaluate_draft_model` returns RAW PER-POSITION COLUMNS, not a summary:**
`recency_distance`, `recency_bucket`, `hop_index`, `target_token`, `correct`,
`record_index` — one entry per measured position.

This is a hard requirement. The measurement definition changed four times during
development, and each change previously forced a fresh GPU run just to re-slice the
same predictions. With raw columns saved (`--raw-out`), re-bucketing, cohort
filters and novelty filters are offline operations. **The expensive pass must never
have to be repeated for an analysis change.**

`evaluate_draft_model` must **not** accept or use the loss-target policy —
evaluation has to be identical across all arms or the comparison is meaningless.

Also here: `load_records` (rejects records without `question_id`), `split_records`
(deterministic in-shard train/held-out split, distinct from the benchmark-level
leakage split in §12), `batches`.

---

## 8. `infer/`

`RecencyStateTracker`: `on_hop_boundary()` (call once the passage has fully
landed, before generation resumes) sets distance 0; `on_tokens_appended(n)` for
generated tokens only; `distance`, `bucket_id` properties; starts at
`NO_PRIOR_HOP_DISTANCE`. Read-then-advance: the token *at* the anchor has distance
0, so read `bucket_id` before advancing.

`HopAwareLengthPolicy`: lookup `recency_bucket_id → gamma`, defaults
`{0:1, 1:2, 2:3, 3:5, 4:6, 5:8, NO_PRIOR_HOP_BUCKET_ID:8}`, clipped to
`[gamma_min, gamma_max] = [1, 8]`; constructor rejects missing bucket entries;
`is_monotonic()` sanity check. Hand-set placeholders — tune before trusting.

`run_speculative_round(context_ids, draft, target, tracker, policy)` reads
`tracker.bucket_id` **before** proposing, proposes `gamma` tokens, accepts the
prefix up to the first rejection, advances the tracker. Positioned as composable
with EAGLE-2's dynamic tree (it sets the tree's target length), not a replacement.

**Still unbuilt, and it is the decisive experiment** — see §11.

---

## 9. `scripts/`

| script | purpose |
|---|---|
| `00_smoke_test.py` | 5 HotpotQA questions through the real pipeline end to end |
| `01_build_dense_index.py` | encode passages → FAISS; `--shard-index/--num-shards` for multi-GPU; **no checkpointing, saves only at the end** |
| `01_generate_trajectories.py` | thin wrapper over `collect.main` |
| `02_build_pilot_corpus.py` | gold-title ∪ random background corpus |
| `02_train_draft_model.py` | thin wrapper over `train_draft.train` |
| `03_merge_dense_shards.py` | merge sharded indices |
| `03_run_diagnostic.py` | train the BASELINE arm (hop signal off, alpha=0) and measure |
| `04_train_hop_signal_model.py` | train the HOP-SIGNAL arm (signal on, alpha=2.0) and measure; saves a checkpoint |
| `05_relabel_shard.py` | re-label a collected shard offline from its stored `steps` |
| `06_validate_shard.py` | audit a shard before spending GPU time on it |

Shared flags on 03/04: `--target-model-name --trajectory-file --eval-fraction
--split-seed --device --batch-size --epochs --lr --log-every --out --raw-out
--loss-targets{all,generated} --feature-layer --v-w --p-w --min-hop-span`.
04 adds `--alpha --tau --checkpoint-out`.

Both print **three views** every run — all positions, novel hops only, matched
cohort — each with `n`, acceptance, distinct-token count, majority-class rate, and
a verdict column.

`05_relabel_shard.py` **asserts that `input_ids` do not change** and aborts
(deleting its output) if they do: the token sequence is a function of the text
alone, so if it moved, either the stored steps do not reproduce the original
context or the tokenizer changed.

`06_validate_shard.py` exits non-zero on any hard-check failure. Every check
corresponds to a defect that actually shipped: step/context invariant, NFC
round-trip, array lengths, `bucket == f(distance)`, duplicate ids, empty steps,
eval-pool leakage, passages ending in newline, the token after a closed passage
being `'Thought'`, malformed steps, empty passages, `"Observation:"` leaking into
thoughts, repeat-hop rate, and a per-bucket measurement-viability table.

### Memory

Load target models with `attn_implementation="sdpa"` and default `--batch-size 2`.
Eager attention materializes a full `[T,T]` matrix per head per layer and OOM'd a
49 GB A6000 on ~2,600-token trajectories at batch size 4.

---

## 10. `eval/` — measurement, and how it goes wrong

### `decode_phase_mask(segment_type_ids, attention_mask)`

True only at non-padding positions whose segment is in
`DECODE_SEGMENT_TYPES = (THOUGHT, TOOL_CALL, ANSWER)`. `QUESTION` and
`RETRIEVED_PASSAGE` are prefill-only — inserted, never generated, never a
speculation target. `TEMPLATE` is generated but excluded as template-determined.

The first version counted every position. Because `"Observation: "` is a
mechanically inserted literal, that reported ~100% acceptance at buckets 0/1 and
looked like a finding.

### `resolve_recency_buckets_for_model(segment_ids, real_buckets, hop_signal_enabled)`

Returns the real buckets, or a constant `DISABLED_RECENCY_BUCKET = 0` everywhere.
Affects only what the model **sees**; results are always grouped by the real
bucket regardless.

### `bucket_token_stats` / `MAJORITY_DEGENERACY_THRESHOLD`

**Report the per-bucket distinct-token count and majority-class rate next to every
acceptance number, always.** A bucket flagged degenerate if
`distinct <= 10 or majority_rate >= 0.5`.

This single diagnostic caught the project's worst false positive. An apparent
"+50 point improvement" at bucket 0 dissolved when it turned out the bucket had 6
distinct target tokens with `' The'` at 64% — a constant predictor scored 0.6397
and the "improved" model scored 0.5547, i.e. **worse than a constant**.

The count-only threshold is a leak on its own: at 1000 trajectories that same
bucket had 18 distinct tokens (passing a naive count check) while one token was
still 64% of answers. **Majority rate is the load-bearing signal.**

### `eval/analysis.py` — offline summarizers over the raw columns

- `novel_hop_flags(records)` — per hop, did the retrieval bring text not already
  in context? **17-18% of hops re-retrieve a passage already present** (a stuck
  agent repeating its query; strongly tied to the failure mode — 84/96 incomplete
  trajectories have repeat hops versus 19/516 complete ones). Those boundaries
  shift nothing and cannot test the hypothesis.
- `hop_span_lengths` / `select(min_hop_span=…)` — matched cohort. Far distance
  buckets are only reachable by hops with long generated spans, so comparing
  near-hop against far-hop across all hops compares different populations.
  Restricting to hops that reach the far bucket makes every bucket come from the
  same hops.
- `select(exclude_no_prior=True)`, `summarize`, `rebucket` (recompute buckets from
  stored raw distances — no GPU pass), `format_table`.

### Statistics

All arms are evaluated on byte-identical positions, so **use paired McNemar**, not
unpaired CIs — it removes the "underpowered" objection. Verify the alignment
(`record_index`, `recency_distance`, `recency_bucket`, `hop_index`, `target_token`
must match element-for-element; only `correct` may differ) before trusting it.
Many comparisons have been run; correct for multiplicity and do not read isolated
p≈0.05 as a discovery.

---

## 11. Findings — what the rebuilt system should reproduce

Setup: 1,000 HotpotQA trajectories, `Qwen/Qwen3-4B`, 850/150 split, 3 epochs,
batch size 2, lr 1e-4, matched cohort.

**1. There is no post-hop dip. The observed pattern is its opposite.** Acceptance
is *highest* right after a hop (0.56 at distance 3-8) and declines with distance
(0.37 at 49+).

**2. That pattern is a within-sentence effect, not a hop effect.** Re-slicing the
raw data by ordinal position within the generated span:

| position in span | 0-2 | 3-8 | 9-24 | 25-48 | 49+ |
|---|---|---|---|---|---|
| after a hop | 0.706 | 0.441 | 0.372 | 0.381 | 0.323 |
| **before ANY retrieval** | **1.000** | **0.527** | 0.309 | 0.270 | 0.370 |

Spans with no preceding retrieval show the same, steeper curve. "Distance since
last hop" was effectively measuring "how far into the sentence we are" — the two
are nearly collinear because every generated span begins right after a hop.
Formulaic openings are easy; later tokens paraphrasing specific retrieved facts
are hard.

**3. The hop signal does nothing**, on a draft model that is demonstrably correct
(the EAGLE fixes improved the baseline by +3.7 points, p=2e-16):

| bucket | n | baseline | hop-signal | delta | p |
|---|---|---|---|---|---|
| 2 | 678 | 0.5634 | 0.5546 | −0.009 | 0.561 |
| 3 | 1808 | 0.4049 | 0.4110 | +0.006 | 0.413 |
| 4 | 2562 | 0.4528 | 0.4575 | +0.005 | 0.448 |
| 5 | 2213 | 0.3656 | 0.3656 | +0.000 | 1.000 |
| pooled | 7261 | 0.4246 | 0.4269 | +0.002 | 0.515 |

Three generations of draft model, no effect in any bucket, view, or test.

**4. The one untested regime — build this if you continue.** Every measurement is
single-step and teacher-forced: the draft is handed the target's own `f_{q-2}`,
which already encodes the retrieved passage. That is the regime where a draft
model is *most* helped and *least* likely to show hop degradation. Real
speculative decoding chains several tokens per round off the draft's **own**
predicted features, and that is where a context shift should bite. Wire
`run_speculative_round` to real models and measure accepted-tokens-per-round
against distance from the hop. Until then, "no effect" means "no effect in the
easiest regime".

Note also that the within-sentence confound is **structural to this ReAct
pipeline** — every generated span starts right after a hop — so answering the
question at all may require a different experimental design.

---

## 12. Leakage guard — `data/question_split.py`

The pilot corpus guarantees gold-passage coverage only for the benchmarks'
**validation** splits, so trajectory collection draws from validation too — the
same pool the eval plan names. Without a guard, training and eval questions come
from the same set.

```python
split_question_ids(ids, eval_fraction=0.2, seed=0) -> (collect_ids, eval_ids)
get_or_create_split(ids, split_path, ...)   # compute once, persist, then load UNCHANGED
```

Sort ids before shuffling so the split does not depend on dataset iteration order
(which varies across `datasets` versions and mirrors). `get_or_create_split` must
return a persisted split unchanged even if the input id set later shifts — a split
must never silently move a question between pools after collection has started.
80/20; Bamboogle is exempt (no supporting-facts annotation, eval-only).

`collect.py` must call it and iterate **only** `collect_ids`.

---

## 13. Corpus and retrieval (build only if you need real retrieval)

`DenseRetriever` over `BAAI/bge-base-en-v1.5` + FAISS `IndexFlatIP`, with
`build`/`save`/`load`/`merge`. Pilot corpus: gold-title passages from
HotpotQA/2WikiMultihopQA/MuSiQue validation splits ∪ 1.5M reservoir-sampled
background from the Dec-2018 Wikipedia DPR dump (`psgs_w100`), ≈1.9M passages.
Load the retriever on **CPU** so both GPUs stay free for the LLMs.

Benchmark mirrors: `hotpotqa/hotpot_qa` (config `distractor`, id field `id`),
`voidful/2WikiMultihopQA` (id field `_id`), `dgslibisey/MuSiQue` (id field `id`).
The legacy `hotpot_qa`, `facebook/wiki_dpr` and original 2Wiki mirrors are
loading-script datasets no longer supported by modern `datasets`.

**`wiki_dpr.py`'s TSV parser must NOT use `csv.QUOTE_NONE`.** The file uses
standard `QUOTE_MINIMAL`; forcing `QUOTE_NONE` left literal quote characters
attached to any title needing quoting, which broke gold-title matching by ~10x
(HotpotQA match rate 7.9% → 84.5% after the fix). Keep a regression test with a
title containing a comma, plus one that legitimately starts with a quote
character (`'"Hello, World!" program'`).

---

## 14. Testing requirements

169 tests, all CPU-only, no network. Beyond ordinary unit tests, these are
mandatory because each pins a bug that shipped:

1. `"".join(step.text) == context` for 2-hop, malformed, incomplete, and finish
   trajectories.
2. Labels come from a single joint tokenization — a token straddling a step
   boundary yields one token, attributed by first character.
3. `collate` pads buckets with the far bucket and segments with `OTHER`, never 0 /
   `QUESTION`.
4. `eagle_aligned_batch`: feature at `q-2`, token at `q-1`, labels at `q`,
   `feature_targets` exactly one step ahead of `target_feats` (use
   position-identifying features so an off-by-one is visible), padding → −100,
   `hop_index` counted on the full sequence, `hop_index == -1` before the first hop.
5. `EAGLE_FEATURE_LAYER == -1`, and the layer is selectable for comparison.
6. `decode_phase_mask` drops `TEMPLATE` and keeps `THOUGHT`.
7. The first post-hop token is what gets labeled bucket 0.
8. `bucket_token_stats` flags a high-majority bucket as degenerate **even with
   many distinct tokens**.
9. `GeneratedTokensOnlyPolicy` extends rather than replaces padding masking, keeps
   `TEMPLATE`, does not mutate its input, and an all-prefill batch gives finite
   zero loss.
10. `question_split` is disjoint, exhaustive, order-independent, and immutable
    once persisted.
11. Feature loss matches a hand computation, excludes masked positions, is zero on
    exact prediction, finite when everything is masked.
12. **End-to-end**: run the real `train_draft_model` + `evaluate_draft_model` on
    CPU with a stand-in target model and records from the real collection path,
    for both `hop_signal_enabled` values; assert the model updates, the raw
    columns are non-empty and equal length, and every label the data can produce
    is within the aux embedding tables. Nothing else exercises training against
    real labeled records; a schema change breaking it would otherwise surface only
    30 minutes into a GPU run.

---

## 15. Working rules

- **Never quote an acceptance number without its bucket's majority-class rate.**
- **Never compare against a number from an earlier run.** The measurement
  definition changed four times; absolute rates moved by tens of points for
  reasons unrelated to the method.
- **Ask what a constant predictor scores** before believing any large effect.
- **Decode the actual token sequence and look at it.** Two of the worst bugs were
  invisible in aggregate statistics and obvious on one line of decoded text.
- **Read reference implementations, not papers.** The paper's wording was
  ambiguous and a fetched summary of the repository stated the opposite of what
  its own quoted code showed.
- **Validate a shard before spending GPU time on it** (`06_validate_shard.py`).
- **Quarantine invalidated artifacts** into `data/_invalidated/` with a README
  explaining why each class is unusable, so nobody cites them later.
