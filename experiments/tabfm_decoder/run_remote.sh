#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$HOME/rocko-tabfm-run"
REPO="$RUN/Rocko"
PINNED_COMMIT="2f95468b2c9662571909535514a91568a385dea1"
export ROCKO_REMOTE_BOOTSTRAP_SHA256=$(
  sha256sum "$0" | awk '{print $1}'
)
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
else
  rm -rf "$REPO"
  git clone https://github.com/MohammadESteitieh/Rocko.git "$REPO"
fi
git -C "$REPO" checkout --detach "$PINNED_COMMIT"
git -C "$REPO" reset --hard "$PINNED_COMMIT"

uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" 'tabfm[jax,cuda]==1.0.1'

rm -rf "$RESULTS"
mkdir -p "$RESULTS"
cd "$REPO"

# Each 16 GiB RTX 4080 exhausted memory while restoring the pinned model.
# TabFM replicates rather than parameter-shards across GPUs, so use the host's
# 94 GiB RAM. This is slower but preserves the frozen model and experiment.
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
"$VENV/bin/python" -c 'import jax; print("JAX devices:", jax.devices())'

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$VENV/bin/python" experiments/tabfm_decoder/run_batch.py \
  --output-dir "$RESULTS"

cat "$RESULTS/batch-summary.json"
