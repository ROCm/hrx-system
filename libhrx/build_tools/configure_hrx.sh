#!/usr/bin/env bash
# Configure the HRX HIP binding (libamdhip64.so) against the venv TheRock ROCm,
# via dev.py.
#
# ROCm is located through TheRock's `rocm-sdk` CLI so NOTHING points at
# /opt/rocm: CMAKE_PREFIX_PATH = $ROCM_HOME/lib/cmake (== `rocm-sdk path
# --cmake`). Override the SDK with ROCM_HOME=/path/to/sdk; a wrong/missing SDK
# hard-fails rather than falling back to a system /opt/rocm.
#
# dev.py owns the actual configure step: the `cmake -S/-B` invocation, build-dir
# tracking (set IREE_CMAKE_BUILD_DIR to override; default build/cmake), and the
# CMake File API query. This script only adds the ROCm/HRX-specific options
# dev.py can't know about, then delegates. Build separately with
# `./dev.py cmake build` (or the iree-cmake-build alias).
#
# Usage:  libhrx/build_tools/configure_hrx.sh [extra -D... cmake args]
#         # e.g. enable the CTS:  configure_hrx.sh -DLIBHRX_BUILD_CTS=ON
# Env:    ROCM_HOME, CHIP (default gfx942), CC/CXX, IREE_CMAKE_BUILD_DIR
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHIP="${CHIP:-gfx942}"

# Locate ROCm strictly from the (venv) TheRock SDK — never /opt/rocm.
if [[ -z "${ROCM_HOME:-}" ]]; then
  command -v rocm-sdk >/dev/null \
    || { echo "ERROR: rocm-sdk not on PATH; activate the venv or set ROCM_HOME=" >&2; exit 1; }
  ROCM_HOME="$(rocm-sdk path --root)"
fi
ROCM_CMAKE="$ROCM_HOME/lib/cmake"
[[ -d "$ROCM_CMAKE" ]] || { echo "ERROR: $ROCM_CMAKE not found (bad ROCM_HOME)" >&2; exit 1; }

echo ">> ROCM_HOME: $ROCM_HOME"
echo ">> chip:      $CHIP"
echo ">> build dir: ${IREE_CMAKE_BUILD_DIR:-build/cmake} (delegating to dev.py)"

# The ROCm find_package() calls are REQUIRED, so a wrong/missing SDK hard-fails
# instead of silently using a system ROCm. Everything below is HRX/ROCm-specific;
# general build plumbing (build dir, File API) belongs to dev.py.
exec "${PYTHON:-python3}" "$REPO_ROOT/dev.py" cmake configure \
  -DCMAKE_PREFIX_PATH="$ROCM_CMAKE" \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DLIBHRX_BUILD=ON \
  -DLIBHRX_BUILD_CUDA_BINDING=OFF \
  -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt \
  -DIREE_ROCM_TEST_TARGET_CHIP="$CHIP" \
  -DHRX_ENABLE_ZSTD=ON \
  "$@"
