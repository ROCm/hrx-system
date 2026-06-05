#!/usr/bin/env bash
# Create/refresh a Python venv with the LATEST NIGHTLY ROCm (TheRock) + PyTorch wheels.
#
# Source of truth for these commands:
#   https://github.com/ROCm/TheRock/blob/main/RELEASES.md
# Unified nightly index:  https://rocm.nightlies.amd.com/whl-multi-arch/
# Wheels are pulled with --no-cache-dir so we always fetch the freshest nightly.
#
# Usage:
#   nightly_rocm_requirements.sh [--venv PATH] [--device SPEC] [--python BIN] [--yes]
# Env overrides: HRX_VENV, ROCM_DEVICE, PYTHON
#
# Examples:
#   ./nightly_rocm_requirements.sh                          # prompts for device (default device-gfx942)
#   ./nightly_rocm_requirements.sh --device device-gfx942 --yes
set -euo pipefail

INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"
PYTHON="${PYTHON:-python3.12}"

# Default venv = sibling of the repo root (…/<parent>/hrx-venv), matching the existing layout.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${HRX_VENV:-$(dirname "$REPO_ROOT")/hrx-venv}"
DEVICE="${ROCM_DEVICE:-}"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)   VENV="$2";   shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --yes|-y) ASSUME_YES=1; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Prompt for the device spec unless one was supplied.
if [[ -z "$DEVICE" ]]; then
  read -rp "ROCm device spec [device-gfx942]: " DEVICE || true
  DEVICE="${DEVICE:-device-gfx942}"
fi

echo ">> venv:   $VENV"
echo ">> python: $PYTHON"
echo ">> device: $DEVICE"
echo ">> index:  $INDEX_URL"
if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -rp "Proceed? This DELETES and recreates the venv above. [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "aborted"; exit 1; }
fi

# Fresh venv (nuke + recreate).
rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --no-cache-dir --upgrade pip

# ROCm SDK: libraries + devel + the device package, from the nightly multi-arch index.
pip install --no-cache-dir --index-url "$INDEX_URL" "rocm[libraries,devel,${DEVICE}]"

# PyTorch stack from the same unified nightly index.
pip install --no-cache-dir --index-url "$INDEX_URL" \
    "torch[${DEVICE}]" "torchvision[${DEVICE}]" torchaudio

echo "== verify =="
rocm-sdk test || echo "WARNING: 'rocm-sdk test' reported issues"
echo "ROCM_HOME      (rocm-sdk path --root):  $(rocm-sdk path --root)"
echo "CMAKE prefix   (rocm-sdk path --cmake): $(rocm-sdk path --cmake)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| hip", getattr(torch.version, "hip", None),
      "| cuda-avail", torch.cuda.is_available())
PY
