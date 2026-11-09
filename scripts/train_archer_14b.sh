#!/usr/bin/env bash
# Durable 14B training driver for the Archer Cree model.
#
# Run this on the GPU server, never on the laptop. Once it is detached (tmux,
# nohup, or systemd -- see docs/server-training-runbook.md section 8) closing
# the laptop, losing Wi-Fi or ending the SSH session does not stop training.
#
# It runs the stages in order, writes one log per stage, and exits non-zero on
# the first failure so systemd or a shell loop can restart it. Finished stages
# are skipped by their marker in $CREE_WORK/run_state.json, and an interrupted
# stage resumes from its last checkpoint, so re-running the script is always
# safe and never restarts the run from zero by accident.
#
#   bash scripts/train_archer_14b.sh
#   CREE_PROFILE=big24 bash scripts/train_archer_14b.sh
#   CREE_STAGES="dpo eval report export" bash scripts/train_archer_14b.sh
set -u
set -o pipefail

ARCHER_ROOT="${ARCHER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="${CREE_WORK:-/data/cree_llm}"
HF="${HF_HOME:-/data/hf}"
PROFILE="${CREE_PROFILE:-big}"
# Which interpreter runs the pipeline. The repo-local venv is only one way to
# set this up: on a shared server the environment usually lives in the user's
# home ($HOME/cree-venv) or in a conda env that is already activated. Hardcoding
# one path made the driver exit 127 before it reached a single stage, so it now
# takes PYTHON when given and otherwise picks the first interpreter that exists.
PYTHON_BIN="${PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in "$ARCHER_ROOT/.venv/bin/python" "${VIRTUAL_ENV:-}/bin/python" \
                   "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "no python interpreter found. Activate your environment, or set" >&2
  echo "  PYTHON=/path/to/python bash scripts/train_archer_14b.sh" >&2
  exit 2
fi
LOGS="$WORK/logs"
STAGES="${CREE_STAGES:-corpus clean tokenizer cpt sft dpo eval report export}"

cd "$ARCHER_ROOT" || exit 2
export CREE_WORK="$WORK" HF_HOME="$HF"
mkdir -p "$WORK" "$LOGS" "$HF" || exit 2

# One driver per work folder. Without the lock, a second copy started after a
# reconnect would write the same checkpoints as the first and corrupt them.
exec 9>"$WORK/train.lock"
if ! flock -n 9; then
  echo "another training driver already holds $WORK/train.lock" >&2
  exit 3
fi

echo "[$(date -Is)] driver start: profile=$PROFILE work=$WORK stages=$STAGES"
echo "[$(date -Is)] python: $PYTHON_BIN"
for stage in $STAGES; do
  log="$LOGS/$stage.log"
  echo "[$(date -Is)] stage $stage -> $log"
  if "$PYTHON_BIN" train_cree_llm.py --profile "$PROFILE" --work "$WORK" \
      --to "$stage" --log-file "$log"; then
    echo "[$(date -Is)] stage $stage finished"
  else
    code=$?
    echo "[$(date -Is)] stage $stage failed with exit $code; rerun this script to resume" >&2
    exit "$code"
  fi
done
echo "[$(date -Is)] all stages finished"
