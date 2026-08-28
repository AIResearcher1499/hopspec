# Invalidated / unusable artifacts

Nothing here may be cited. Each entry says why.

## `shard_1p7b_scale.PARTIAL.jsonl` (2026-08-29)

70 records of an interrupted 1000-trajectory HotpotQA collection
(`--retriever bm25-distractor --split-file data/hotpot_split.json
--max-questions 1000`). The run was stopped by hand after ~2 minutes so the
collection could be moved to another machine.

**Unusable, and quarantined rather than deleted for one specific reason:**
`collect_shard` opens the output with `"a"` and appends. A partial shard left
at the live path `data/shard_1p7b_scale.jsonl` would be silently extended by
the next run — the resulting file would look like a complete collection while
containing a truncated prefix, and a copy of it onto the other machine would
produce duplicate question ids. That is the "an overwrite looks exactly like a
successful run" failure mode, in append form.

The scale-up collection must start from a path that does not exist.
`06_validate_shard.py` checks duplicate ids and would catch the duplication
case, but not the "stopped early and resumed by appending" case.
