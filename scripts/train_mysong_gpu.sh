#!/usr/bin/env bash
# Train the large MySong tagger on a GPU server and leave a CPU-loadable
# checkpoint beside it.  Run from the repository root.
#
#   bash scripts/train_mysong_gpu.sh
#
# Override any of these with environment variables:
#   EPOCHS=60 HIDDEN=256 DEVICE=cuda OUT=data/mysong/model/mysong_large.npz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-60}"
HIDDEN="${HIDDEN:-256}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-data/mysong/model/mysong_large.npz}"
BASE="${BASE:-data/mysong/corpus.jsonl}"
EXTRA="${EXTRA:-data/mysong/corpus_extra.jsonl}"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

echo "== MySong large-corpus training =="
echo "python: $PYTHON"
echo "device: $DEVICE"
echo "epochs: $EPOCHS, hidden: $HIDDEN"
echo "output: $OUT"

"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; set DEVICE=cpu for a CPU run or fix the driver")
print("gpu", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$BASE" ]]; then
  echo "missing $BASE; building OpenEWLD cache"
  "$PYTHON" -m mysong.cli corpus
fi

if [[ ! -f "$EXTRA" ]]; then
  echo "missing $EXTRA"
  echo "Run the download/build steps in GPU_MYSONG.md section 2 first." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
exec "$PYTHON" tools/train_mysong_tagger_large.py \
  --base "$BASE" \
  --extra "$EXTRA" \
  --out-model "$OUT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --hidden "$HIDDEN"
