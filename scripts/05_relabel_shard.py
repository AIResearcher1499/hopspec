"""Re-label a collected shard offline from its stored `steps`.

Asserts that input_ids do not change and aborts (deleting its output) if they
do: the token sequence is a function of the text alone, so if it moved,
either the stored steps do not reproduce the original context or the
tokenizer changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-model-name", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from hopspec.data.schema import SegmentType, Trajectory, TrajectoryStep
    from hopspec.data.segment_labeling import hf_offset_tokenizer, label_trajectory

    tokenize = hf_offset_tokenizer(AutoTokenizer.from_pretrained(args.target_model_name))

    try:
        with open(args.in_path, encoding="utf-8") as fin, \
                open(args.out, "w", encoding="utf-8") as fout:
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                trajectory = Trajectory(
                    question=record["question"],
                    steps=[
                        TrajectoryStep(
                            SegmentType(step["segment_type"]), step["text"],
                            step.get("hop_index"),
                        )
                        for step in record["steps"]
                    ],
                    final_answer=record.get("final_answer"),
                    is_complete=record.get("is_complete", False),
                    context=record["context"],
                )
                labeled = label_trajectory(trajectory, tokenize)
                if labeled.input_ids != record["input_ids"]:
                    raise AssertionError(
                        f"{args.in_path}:{line_number}: input_ids changed on relabel "
                        "— stored steps do not reproduce the original context, or "
                        "the tokenizer changed"
                    )
                record["segment_type_ids"] = labeled.segment_type_ids
                record["recency_bucket_ids"] = labeled.recency_bucket_ids
                record["recency_distances"] = labeled.recency_distances
                record["hop_boundary_positions"] = labeled.hop_boundary_positions
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    except BaseException:
        if os.path.exists(args.out):
            os.remove(args.out)
        raise
    print(f"relabeled shard -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
