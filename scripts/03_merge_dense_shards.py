"""Merge sharded dense indices into one."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from hopspec.data.retriever import DenseRetriever

    merged = DenseRetriever.merge(args.shard_dirs, args.out_dir)
    print(f"merged {len(args.shard_dirs)} shards, {len(merged.documents)} passages "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
