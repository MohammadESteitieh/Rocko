#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$HOME/rocko-single-frame-validation"
REPO="$RUN/Rocko"
PINNED_COMMIT="ad8d2d13939938286669b4f9b0bb865596f55134"
export ROCKO_REMOTE_BOOTSTRAP_SHA256=$(
  sha256sum "$0" | awk '{print $1}'
)
VENV="$RUN/.venv"
RESULTS="$RUN/results"
mkdir -p "$RUN"

finish() {
  status=$?
  trap - EXIT
  echo "SINGLE_FRAME_EXIT_STATUS=$status"
  if [ "$status" -eq 0 ]; then
    echo "SINGLE_FRAME_RUN_COMPLETE"
  else
    echo "SINGLE_FRAME_RUN_FAILED"
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
uv pip install --python "$VENV/bin/python" \
  'numpy==2.5.1' 'scipy==1.18.0' 'tabfm[jax]==1.0.1'

rm -rf "$RESULTS"
mkdir -p "$RESULTS"
uv pip freeze --python "$VENV/bin/python" > "$RESULTS/environment.txt"
(cd "$RESULTS" && sha256sum environment.txt > environment.txt.sha256)
cd "$REPO"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$VENV/bin/python" experiments/tabfm_decoder/validate_single_frame_bundle.py \
  --output "$RESULTS/single-frame-validation.json"

cat "$RESULTS/single-frame-validation.json"
