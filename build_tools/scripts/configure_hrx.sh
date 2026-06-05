#!/usr/bin/env bash
# Configure + build the HRX HIP binding (libamdhip64.so) against the venv TheRock ROCm.
#
# ROCm is located via TheRock's `rocm-sdk` CLI, so NOTHING points at /opt/rocm:
#   CMAKE_PREFIX_PATH = $ROCM_HOME/lib/cmake   (== `rocm-sdk path --cmake`)
# Override with ROCM_HOME=/path/to/sdk. The ROCm find_package() calls are REQUIRED,
# so a wrong/missing SDK hard-fails instead of falling back to a system /opt/rocm.
#
# Option names follow the post-#27 "split CMake ownership" layout
# (runtime/project.cmake, libhrx/project.cmake). Default build is LEAN: the binding +
# amdgpu HAL only. Turn CTS/tests on for the test phase.
#
# Usage: configure_hrx.sh [--build-dir DIR] [--chip gfx942]
#                         [--cts on|off] [--tests on|off] [--passthrough on|off] [--no-build]
# Env:   ROCM_HOME, CC, CXX, JOBS
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/hrx"
CHIP="gfx942"
CTS="OFF"          # LIBHRX_BUILD_CTS
TESTS="OFF"        # IREE_BUILD_TESTS (IREE runtime unit tests + CTS targets)
PASSTHROUGH="OFF"  # LIBHRX_BUILD_PASSTHROUGH (deprecated dev tool)
DO_BUILD=1
up() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)   BUILD_DIR="$2"; shift 2;;
    --chip)        CHIP="$2"; shift 2;;
    --cts)         CTS="$(up "$2")"; shift 2;;
    --tests)       TESTS="$(up "$2")"; shift 2;;
    --passthrough) PASSTHROUGH="$(up "$2")"; shift 2;;
    --no-build)    DO_BUILD=0; shift;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Locate ROCm strictly from the (venv) TheRock SDK — never /opt/rocm.
if [[ -z "${ROCM_HOME:-}" ]]; then
  command -v rocm-sdk >/dev/null \
    || { echo "ERROR: rocm-sdk not on PATH; activate the venv or pass ROCM_HOME=" >&2; exit 1; }
  ROCM_HOME="$(rocm-sdk path --root)"
fi
ROCM_CMAKE="$ROCM_HOME/lib/cmake"
[[ -d "$ROCM_CMAKE" ]] || { echo "ERROR: $ROCM_CMAKE not found (bad ROCM_HOME)" >&2; exit 1; }

CC="${CC:-clang}"; CXX="${CXX:-clang++}"
JOBS="${JOBS:-$(nproc)}"
LAUNCHER=()
command -v ccache >/dev/null && LAUNCHER=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)

echo ">> repo:      $REPO_ROOT"
echo ">> build dir: $BUILD_DIR"
echo ">> ROCM_HOME: $ROCM_HOME"
echo ">> chip: $CHIP | cts: $CTS | iree-tests: $TESTS | passthrough: $PASSTHROUGH | jobs: $JOBS | cc: $CC"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$ROCM_CMAKE" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_ASM_COMPILER="$CC" \
  -DCMAKE_AR="$(command -v llvm-ar || command -v ar)" \
  -DCMAKE_RANLIB="$(command -v llvm-ranlib || command -v ranlib)" \
  "${LAUNCHER[@]}" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DLIBHRX_BUILD=ON \
  -DLIBHRX_BUILD_CUDA_BINDING=OFF \
  -DLIBHRX_BUILD_CTS="$CTS" \
  -DLIBHRX_BUILD_PASSTHROUGH="$PASSTHROUGH" \
  -DIREE_BUILD_TESTS="$TESTS" \
  -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt \
  -DIREE_ROCM_TEST_TARGET_CHIP="$CHIP"

if [[ "$DO_BUILD" -eq 1 ]]; then
  cmake --build "$BUILD_DIR" -j "$JOBS"
  echo "== binding artifacts =="
  ls -la "$BUILD_DIR"/libhrx/src/binding/hip/libamdhip64.so* 2>/dev/null \
    || echo "(libamdhip64.so not found — check build output above)"
fi
