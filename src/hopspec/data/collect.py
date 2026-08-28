"""Benchmark loading, trajectory -> record conversion, shard collection, CLI."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Iterable

from hopspec.data.agent_pipeline import (
    LLM,
    run_react_trajectories_batched,
    run_react_trajectory,
)
from hopspec.data.question_split import get_or_create_split
from hopspec.data.retriever import BaseRetriever
from hopspec.data.schema import Trajectory
from hopspec.data.segment_labeling import OffsetTokenizer, label_trajectory

# Mirrors that still work with modern `datasets` (the legacy `hotpot_qa`,
# `facebook/wiki_dpr` and original 2Wiki mirrors are loading-script datasets
# no longer supported).
BENCHMARKS = {
    "hotpotqa": {"path": "hotpotqa/hotpot_qa", "config": "distractor", "id_field": "id"},
    "2wikimultihopqa": {"path": "voidful/2WikiMultihopQA", "config": None, "id_field": "_id"},
    "musique": {"path": "dgslibisey/MuSiQue", "config": None, "id_field": "id"},
}

# No supporting-facts annotation -> no gold-passage coverage in the pilot
# corpus -> eval-only, exempt from the collect/eval split.
EVAL_ONLY_BENCHMARKS = ("bamboogle",)


def load_benchmark(name: str, split: str = "validation"):
    from datasets import load_dataset

    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    if spec["config"] is not None:
        return load_dataset(spec["path"], spec["config"], split=split)
    return load_dataset(spec["path"], split=split)


def benchmark_questions(name: str, dataset) -> list[tuple[str, str]]:
    id_field = BENCHMARKS[name]["id_field"]
    return [(example[id_field], example["question"]) for example in dataset]


def trajectory_to_record(
    trajectory: Trajectory, question_id: str, tokenize: OffsetTokenizer
) -> dict:
    labeled = label_trajectory(trajectory, tokenize)
    return {
        "question_id": question_id,
        "question": trajectory.question,
        "final_answer": trajectory.final_answer,
        "is_complete": trajectory.is_complete,
        "context": trajectory.context,
        "steps": [
            {
                "segment_type": int(step.segment_type),
                "text": step.text,
                "hop_index": step.hop_index,
            }
            for step in trajectory.steps
        ],
        "input_ids": labeled.input_ids,
        "segment_type_ids": labeled.segment_type_ids,
        "recency_bucket_ids": labeled.recency_bucket_ids,
        "recency_distances": labeled.recency_distances,
        "hop_boundary_positions": labeled.hop_boundary_positions,
    }


def collected_question_ids(out_path: str) -> set[str]:
    """question_ids already in a shard, for --resume.

    Tolerates a truncated final line: a killed collect can leave one, and the
    whole point of resuming is to survive a kill. A record without a
    question_id is a corrupt line, not a resumable one, and is ignored — the
    question is simply re-collected and `06_validate_shard.py`'s duplicate-id
    check stays the guard against getting that wrong.
    """
    if not os.path.exists(out_path):
        return set()
    ids: set[str] = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail of a killed run
            question_id = record.get("question_id")
            if question_id is not None:
                ids.add(question_id)
    return ids


def collect_shard(
    questions: Iterable[tuple[str, str]],
    llm: LLM,
    retriever: BaseRetriever,
    tokenize: OffsetTokenizer,
    out_path: str,
    max_hops: int = 4,
    num_docs: int = 3,
    on_record: Callable[[dict], None] | None = None,
    resume: bool = False,
    batch_size: int = 1,
) -> int:
    """Run trajectories and append records to a jsonl shard. Returns count.

    The file is opened for APPEND, so a killed run leaves valid partial output
    — but re-running would collect everything again and duplicate every id.
    `resume=True` skips question_ids already present. Returns the number of
    trajectories actually collected on THIS call, not the file's total.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    already = collected_question_ids(out_path) if resume else set()
    todo = [(qid, q) for qid, q in questions if qid not in already]
    count = 0
    with open(out_path, "a", encoding="utf-8") as f:

        def emit(trajectory, question_id: str) -> None:
            nonlocal count
            record = trajectory_to_record(trajectory, question_id, tokenize)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if on_record is not None:
                on_record(record)
            count += 1

        if batch_size == 1:
            # The original path, untouched. Everything published so far came
            # off it, so it stays the default and stays byte-identical.
            for question_id, question in todo:
                emit(run_react_trajectory(question, llm, retriever,
                                          max_hops=max_hops, num_docs=num_docs),
                     question_id)
        else:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                trajectories = run_react_trajectories_batched(
                    [question for _qid, question in chunk], llm, retriever,
                    max_hops=max_hops, num_docs=num_docs,
                )
                for (question_id, _question), trajectory in zip(chunk, trajectories):
                    emit(trajectory, question_id)
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect agentic RAG trajectories")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--split-file", required=True,
                        help="persisted collect/eval question-id split (leakage guard)")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--target-model-name", required=True)
    parser.add_argument("--retriever", choices=["dense", "bm25-distractor"],
                        default="dense",
                        help="bm25-distractor: in-memory BM25 over the benchmark's "
                             "own context paragraphs (hotpotqa only; no index needed "
                             "— for local/laptop pilot runs)")
    parser.add_argument("--index-dir", default=None,
                        help="saved DenseRetriever directory (required for --retriever dense)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--num-docs", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="generate this many trajectories in lockstep, one "
                             "generate call per round. Default 1 keeps the "
                             "original sequential path. Batched matmuls are not "
                             "bit-identical to single-sequence ones, so a "
                             "batched shard differs slightly from a sequential "
                             "one — record that in the shard's provenance.")
    parser.add_argument("--resume", action="store_true",
                        help="skip question_ids already in --out. The output is "
                             "APPENDED to, so without this a restarted run "
                             "re-collects everything and duplicates every id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from hopspec.data.agent_pipeline import HFTargetLLM
    from hopspec.data.retriever import DenseRetriever
    from hopspec.data.segment_labeling import hf_offset_tokenizer

    dataset = load_benchmark(args.benchmark, split=args.split)
    questions = benchmark_questions(args.benchmark, dataset)
    by_id = dict(questions)

    collect_ids, _eval_ids = get_or_create_split(
        by_id.keys(), args.split_file,
        eval_fraction=args.eval_fraction, seed=args.split_seed,
    )
    # Iterate ONLY collect_ids — eval questions must never enter training data.
    todo = [(qid, by_id[qid]) for qid in collect_ids if qid in by_id]
    if args.max_questions is not None:
        todo = todo[: args.max_questions]

    llm = HFTargetLLM(args.target_model_name, device=args.device,
                      max_new_tokens=args.max_new_tokens)
    if args.retriever == "bm25-distractor":
        if args.benchmark != "hotpotqa":
            raise SystemExit("--retriever bm25-distractor requires --benchmark hotpotqa")
        from hopspec.data.retriever import Document, InMemoryBM25Retriever

        documents = []
        for example in dataset:
            for title, sentences in zip(
                example["context"]["title"], example["context"]["sentences"]
            ):
                documents.append(Document(
                    doc_id=f"{example['id']}:{title}", title=title,
                    text=" ".join(sentences),
                ))
        retriever = InMemoryBM25Retriever(documents)
    else:
        if not args.index_dir:
            raise SystemExit("--index-dir is required for --retriever dense")
        retriever = DenseRetriever.load(args.index_dir, device="cpu")
    tokenize = hf_offset_tokenizer(llm.tokenizer)

    count = collect_shard(
        todo, llm, retriever, tokenize, args.out,
        max_hops=args.max_hops, num_docs=args.num_docs, resume=args.resume,
        batch_size=args.batch_size,
    )
    total = len(collected_question_ids(args.out))
    print(f"collected {count} trajectories this run -> {args.out} "
          f"({total} unique question ids in the shard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
