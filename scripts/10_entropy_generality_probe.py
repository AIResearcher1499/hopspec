"""Is entropy degeneracy general to agentic loops, or specific to ReAct-RAG?

Executes docs/prereg-entropy-generality-probe-2026-08-29.md. Measures the
target's next-token entropy at STRUCTURAL BOUNDARY positions versus CONTENT
positions, within the same generation, and reports everything §6 requires.

Self-generated mode is the primary quantity and the only one the gate is read
from: the degeneracy claim is about a model's own generation. Teacher-forced
mode is secondary and its absolute level is confounded, so only the ratio and
the boundary median are reported from it.

CPU/MPS is enough for 1.7B at these sizes; this probe never rents a GPU.
"""

from __future__ import annotations

import argparse
import json
import math


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True,
                        choices=["function_call", "code_agent"])
    parser.add_argument("--mode", required=True, choices=["self", "forced"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--out", required=True)
    return parser


# ---- data: each workload yields (system_prompt, user_text, reference_text) ----

def _tool_schemas(system_text: str) -> list[dict]:
    """Glaive stores its tool schemas as JSON objects inside the system text.
    Brace-match them out rather than regexing, so a nested object survives."""
    schemas, depth, start = [], 0, None
    for index, char in enumerate(system_text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    schema = json.loads(system_text[start:index + 1])
                except json.JSONDecodeError:
                    schema = None
                if isinstance(schema, dict) and "name" in schema:
                    schemas.append(schema)
                start = None
    return schemas


def load_function_call(n: int):
    """Glaive's real tool schemas, prompted in Qwen's own tool-calling format
    (prereg amendment 2026-08-29): glaive's system text never states the
    output syntax, so a model not fine-tuned on it answers in prose."""
    from datasets import load_dataset

    stream = load_dataset("glaiveai/glaive-function-calling-v2",
                          split="train", streaming=True)
    out = []
    for row in stream:
        chat = row.get("chat") or ""
        system = (row.get("system") or "").replace("SYSTEM:", "", 1).strip()
        if "ASSISTANT:" not in chat or "<functioncall>" not in chat:
            continue
        schemas = _tool_schemas(system)
        if not schemas:
            continue
        user = chat.split("ASSISTANT:")[0].replace("USER:", "", 1).strip()
        reference = chat.split("ASSISTANT:", 1)[1].split("USER:")[0].strip()
        if not user or not reference:
            continue
        out.append({"tools": schemas, "system": None, "user": user,
                    "reference": reference})
        if len(out) >= n:
            break
    return out


def load_code_agent(n: int):
    from datasets import load_dataset

    stream = load_dataset("nebius/SWE-agent-trajectories", split="train",
                          streaming=True)
    out = []
    for row in stream:
        # role is 'ai', text is under 'text', system under 'system_prompt'
        traj = row.get("trajectory") or []
        if not traj:
            continue
        system = next((m.get("system_prompt") or "" for m in traj
                       if m.get("system_prompt")), "")
        user = next((m.get("text") or "" for m in traj
                     if m.get("role") == "user" and (m.get("text") or "").strip()), "")
        assistant = next((m.get("text") or "" for m in traj
                          if m.get("role") == "ai" and (m.get("text") or "").strip()), "")
        if not (system and user and assistant):
            continue
        out.append({"tools": None, "system": system[:8000], "user": user[:8000],
                    "reference": assistant})
        if len(out) >= n:
            break
    return out


LOADERS = {"function_call": load_function_call, "code_agent": load_code_agent}


# ---- statistics ----

def entropy_from_logits(row, torch) -> float:
    log_probs = torch.log_softmax(row.float(), dim=-1)
    return float(-(log_probs.exp() * log_probs).sum())


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    import numpy as np

    a = np.asarray(values)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "frac_le_0.25": float((a <= 0.25).mean()),
    }


def main() -> int:
    args = build_parser().parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hopspec.probe.agent_grammars import (
        BOUNDARY, CONTENT, GRAMMARS, label_positions, label_tokens,
    )

    grammar = GRAMMARS[args.workload]
    examples = LOADERS[args.workload](args.n)
    print(f"{len(examples)} examples for {args.workload} ({args.mode} mode)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", attn_implementation="sdpa"
    ).to(args.device)
    model.eval()

    entropies = {BOUNDARY: [], CONTENT: []}
    tokens_at = {BOUNDARY: [], CONTENT: []}
    parsed = attempted = 0

    for example in examples:
        attempted += 1
        system, user = example["system"], example["user"]
        reference, tools = example["reference"], example["tools"]
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        kwargs = {"tools": tools} if tools else {}
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False, **kwargs)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs)
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(args.device)

        if args.mode == "self":
            with torch.no_grad():
                out = model.generate(
                    **prompt_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, output_scores=True,
                    return_dict_in_generate=True,
                )
            gen_ids = out.sequences[0, prompt_ids["input_ids"].shape[1]:].tolist()
            scores = out.scores
        else:
            body = tokenizer(reference, add_special_tokens=False)["input_ids"]
            if not body:
                continue
            full = torch.cat(
                [prompt_ids["input_ids"],
                 torch.tensor([body], device=args.device)], dim=1)
            with torch.no_grad():
                logits = model(input_ids=full).logits[0]
            start = prompt_ids["input_ids"].shape[1]
            # row at position-1 predicts the token AT that position
            scores = [logits[start + i - 1] for i in range(len(body))]
            gen_ids = body

        # Build the text from per-token decodes so offsets are exact by
        # construction — no re-tokenisation mismatch is possible.
        pieces = [tokenizer.decode([t]) for t in gen_ids]
        text = "".join(pieces)
        offsets, cursor = [], 0
        for piece in pieces:
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)

        try:
            kinds = label_positions(text, grammar)
        except ValueError:
            continue
        token_kinds = label_tokens(offsets, kinds)
        if BOUNDARY not in token_kinds:
            continue                      # nothing to measure in this sample
        parsed += 1

        for index, (kind, token) in enumerate(zip(token_kinds, gen_ids)):
            row = scores[index] if index < len(scores) else None
            if row is None:
                break
            value = entropy_from_logits(row[0] if row.dim() == 2 else row, torch)
            if not math.isfinite(value):
                continue
            entropies[kind].append(value)
            tokens_at[kind].append(int(token))

    from collections import Counter

    boundary_tokens = Counter(tokens_at[BOUNDARY])
    total_boundary = sum(boundary_tokens.values())
    majority = (boundary_tokens.most_common(1)[0][1] / total_boundary
                if total_boundary else 0.0)
    result = {
        "workload": args.workload, "mode": args.mode, "model": args.model,
        "examples_attempted": attempted, "examples_parsed": parsed,
        "parse_rate": parsed / attempted if attempted else 0.0,
        "boundary": describe(entropies[BOUNDARY]),
        "content": describe(entropies[CONTENT]),
        # spec §10: a low boundary entropy that merely restates "one token is
        # most boundaries" is not a finding. Say which it is.
        "boundary_distinct_tokens": len(boundary_tokens),
        "boundary_majority_rate": majority,
        "boundary_constant_predictor_score": majority,
        "boundary_top_tokens": [
            [tokenizer.decode([t]), c] for t, c in boundary_tokens.most_common(6)
        ],
    }
    b, c = result["boundary"], result["content"]
    result["median_ratio"] = (
        None if not c.get("n") or c.get("median", 0) == 0
        else b.get("median", 0) / c["median"]
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "boundary_top_tokens"}, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
