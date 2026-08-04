#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$HOME/rocko-tabfm-run"
REPO="$RUN/Rocko"
VENV="$RUN/.venv"
RESULTS="$RUN/results"
mkdir -p "$RUN"

finish() {
  status=$?
  trap - EXIT
  echo "TABFM_EXIT_STATUS=$status"
  if [ "$status" -eq 0 ]; then
    echo "TABFM_RUN_COMPLETE"
  else
    echo "TABFM_RUN_FAILED"
  fi
  exit "$status"
}
trap finish EXIT

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch origin research-main
  git -C "$REPO" checkout research-main
  git -C "$REPO" reset --hard origin/research-main
else
  rm -rf "$REPO"
  git clone --branch research-main --single-branch \
    https://github.com/MohammadESteitieh/Rocko.git "$REPO"
fi

uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" 'tabfm[jax,cuda]==1.0.1'

rm -rf "$RESULTS"
mkdir -p "$RESULTS"
cd "$REPO"

"$VENV/bin/python" experiments/tabfm_decoder/export_rs18_table.py \
  --query-sequence 24 --output-dir "$RESULTS"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$VENV/bin/python" experiments/tabfm_decoder/run_tabfm.py \
  "$RESULTS/rs18-sequence-24.context-query.csv" \
  "$RESULTS/rs18-sequence-24.truth.csv" \
  "$RESULTS/rs18-sequence-24.metadata.json" \
  --backend jax \
  --output "$RESULTS/rs18-sequence-24.predictions.csv" \
  --summary "$RESULTS/rs18-sequence-24.summary.json"

cat "$RESULTS/rs18-sequence-24.summary.json"
