#!/usr/bin/env bash
# Device bridge (docs/plan-runpod-execution-2026-08-29.md): re-run the two CHECKPOINT-FREE arms on the pod so
# their rounds can be compared against the MPS baselines committed in this
# repository. No weight transfer is needed — that is the point.
#
#   bash scripts/08_device_bridge.sh cuda
#
# Then copy data/rounds_<device>_{lookup,scaffold}_1p7b.jsonl to the Mac and run
# scripts/09_device_bridge_compare.py there: analysis never needs a GPU, and a
# rented pod is billed by the hour.
set -euo pipefail

DEVICE="${1:-cuda}"
# A rented pod usually installs into the image's own python; a .venv only
# exists on a dev machine. Defaulting blindly to .venv/bin/python cost a
# billed run that exited in 0.004s with "No such file or directory".
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi
"$PY" -c "import hopspec" 2>/dev/null || { echo "hopspec not importable by $PY"; exit 1; }
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SHARD="${SHARD:-data/shard_1p7b.jsonl}"

for f in "$SHARD" data/rounds_ct_lookup_1p7b.jsonl data/rounds_ct_scaffold_1p7b.jsonl; do
  [ -f "$f" ] || { echo "missing $f — run from the repository root"; exit 1; }
done

COMMON=(--target-model-name "$MODEL" --trajectory-file "$SHARD"
        --device "$DEVICE" --gamma 4 --replay-mode chat)

echo "### arm: lookup (model-free)"
"$PY" scripts/07_chained_eval.py "${COMMON[@]}" \
  --draft-source lookup \
  --rounds-out "data/rounds_${DEVICE}_lookup_1p7b.jsonl"

echo "### arm: scaffold, verb bet (model-free)"
"$PY" scripts/07_chained_eval.py "${COMMON[@]}" \
  --draft-source scaffold --scaffold-verb fit \
  --rounds-out "data/rounds_${DEVICE}_scaffold_1p7b.jsonl"

echo
echo "done. bring these back to the Mac:"
ls -la "data/rounds_${DEVICE}_lookup_1p7b.jsonl" "data/rounds_${DEVICE}_scaffold_1p7b.jsonl"
