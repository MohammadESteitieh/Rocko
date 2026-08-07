#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV="${ROCKO_DECODER_VENV:-$ROOT/.venv}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" \
  'numpy==2.5.1' 'scipy==1.18.0' 'tabfm[jax]==1.0.1'

"$VENV/bin/python" - <<'PY'
import jax, numpy, scipy, tabfm
print("Decoder environment ready")
print("JAX devices:", jax.devices())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("TabFM:", tabfm.__version__ if hasattr(tabfm, "__version__") else "1.0.1")
PY

echo "Run: CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $VENV/bin/python experiments/tabfm_decoder/decode_single_csv.py message.csv"
