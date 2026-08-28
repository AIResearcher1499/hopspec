"""Encode corpus passages into a FAISS shard.

NOTE: no checkpointing — the index is saved only at the end. Launch detached
(`nohup ... > log 2>&1 < /dev/null & disown`) and confirm PPID=1; a plain
background job dies with its parent shell and once destroyed 1.5 hours of
index building.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        help="jsonl of {doc_id, title, text} (see 02_build_pilot_corpus)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    from hopspec.data.retriever import DenseRetriever, Document

    documents = []
    with open(args.corpus, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % args.num_shards != args.shard_index:
                continue
            documents.append(Document(**json.loads(line)))
    print(f"shard {args.shard_index}/{args.num_shards}: {len(documents)} passages")

    retriever = DenseRetriever(model_name=args.model_name, device=args.device)
    retriever.build(documents, batch_size=args.batch_size)
    retriever.save(args.out_dir)
    print(f"saved index -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
