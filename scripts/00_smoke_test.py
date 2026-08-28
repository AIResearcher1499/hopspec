"""Run 5 HotpotQA questions through the real pipeline end to end.

Uses the distractor config's own context paragraphs as a BM25 corpus so no
dense index is needed. Verifies the step/context invariant on every
trajectory before printing a summary.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    from datasets import load_dataset

    from hopspec.data.agent_pipeline import HFTargetLLM, run_react_trajectory
    from hopspec.data.collect import trajectory_to_record
    from hopspec.data.retriever import Document, InMemoryBM25Retriever
    from hopspec.data.segment_labeling import hf_offset_tokenizer

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = list(dataset.select(range(args.num_questions)))

    documents = []
    for example in examples:
        for title, sentences in zip(
            example["context"]["title"], example["context"]["sentences"]
        ):
            documents.append(
                Document(doc_id=f"{example['id']}:{title}", title=title,
                         text=" ".join(sentences))
            )
    retriever = InMemoryBM25Retriever(documents)

    llm = HFTargetLLM(args.target_model_name, device=args.device,
                      max_new_tokens=args.max_new_tokens)
    tokenize = hf_offset_tokenizer(llm.tokenizer)

    for example in examples:
        trajectory = run_react_trajectory(
            example["question"], llm, retriever, max_hops=args.max_hops
        )
        joined = "".join(step.text for step in trajectory.steps)
        assert joined == trajectory.context, "step/context invariant violated"
        record = trajectory_to_record(trajectory, example["id"], tokenize)
        hops = len(record["hop_boundary_positions"])
        print(
            f"[{example['id']}] complete={trajectory.is_complete} hops={hops} "
            f"tokens={len(record['input_ids'])} answer={trajectory.final_answer!r} "
            f"(gold={example['answer']!r})"
        )
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
