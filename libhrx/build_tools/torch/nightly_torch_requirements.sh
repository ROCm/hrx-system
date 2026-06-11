#!/usr/bin/env bash
# OPTIONAL PyTorch layer for HRX downstream A/B testing.
#
# Installs nightly PyTorch (+ torchvision/torchaudio) into an ALREADY-ACTIVE
# ROCm venv created by ../nightly_rocm_requirements.sh. Kept separate from the
# core ROCm deps so the HRX build never pulls in torch — activate the ROCm venv
# first, then run this only when you need the torch-driven tests.
#
# Source of truth: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
# Unified nightly index: https://rocm.nightlies.amd.com/whl-multi-arch/
#
# Usage:
#   source <venv>/bin/activate
#   libhrx/build_tools/torch/nightly_torch_requirements.sh [--device SPEC]
# Env overrides: ROCM_DEVICE
set -euo pipefail

INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"
DEVICE="${ROCM_DEVICE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Require the core ROCm venv (rocm-sdk) to be active; torch layers on top of it.
[[ -n "${VIRTUAL_ENV:-}" ]] \
  || { echo "ERROR: no active venv. Run libhrx/build_tools/nightly_rocm_requirements.sh and activate it first." >&2; exit 1; }
command -v rocm-sdk >/dev/null \
  || { echo "ERROR: active venv has no rocm-sdk; this layers torch onto the ROCm venv." >&2; exit 1; }

if [[ -z "$DEVICE" ]]; then
  read -rp "ROCm device spec [device-gfx942]: " DEVICE || true
  DEVICE="${DEVICE:-device-gfx942}"
fi

echo ">> venv:   $VIRTUAL_ENV"
echo ">> device: $DEVICE"
echo ">> index:  $INDEX_URL"

# PyTorch stack from the same unified nightly index the ROCm SDK came from.
pip install --no-cache-dir --index-url "$INDEX_URL" \
    "torch[${DEVICE}]" "torchvision[${DEVICE}]" torchaudio

echo "== verify =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| hip", getattr(torch.version, "hip", None),
      "| cuda-avail", torch.cuda.is_available())
PY
