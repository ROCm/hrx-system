# HRX System

> [!CAUTION]
> This is early-access runtime infrastructure. Expect API, packaging, and
> deployment details to move while the fork is brought up. This is not an
> official component within the ROCm stack.

The HRX System is a collection of minimal runtime components providing an
alternative implementation of HIP as it presently ships within ROCm. It
provides a common substrate for low latency, high performance integration
with AMD's GPU, NPU, and CPU products.

## What Is Included

- `runtime/`: core runtime components
- `libhrx/`: the public HRX C ABI, C++ helpers, HIP compatibility binding,
  streaming support, CTS, and debug passthrough tools.
- `build_tools/`: shared CMake/Bazel translation helpers, CI packaging
  scripts, sanitizer wiring, and common build overlays.

Project-specific build support lives with the project it serves. Runtime CMake
bootstrap and dependencies are under `runtime/project.cmake` and
`runtime/build_tools/`, while libhrx owns `libhrx/project.cmake` and
`libhrx/build_tools/`. Shared generator and build-system mechanics stay under
root `build_tools/`, with shared third-party CMake configuration beside the
matching Bazel overlays in `build_tools/third_party/`.

The public install exports `libhrx.so`, `hrx-info`, public HRX headers for the
native low-latency API, CMake package files, and the HRX HIP compatibility
layer as `lib/libamdhip64.so`.

## Building

Build-tree tests that use YAML execution manifests require the locked
development Python requirements in the interpreter CMake discovers. This is
currently just PyYAML:

```bash
python -m venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.lock.txt
```

Use a ROCm installation or unpacked ROCm build environment that provides HSA,
AQL profile headers, and a ROCm LLVM toolchain. The CI path uses ROCm clang and
forces libraries into `lib`.

```bash
cmake -S . -B build/hrx-system -GNinja \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_ASM_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_AR=/opt/rocm/llvm/bin/llvm-ar \
  -DCMAKE_RANLIB=/opt/rocm/llvm/bin/llvm-ranlib \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ROCM_TEST_TARGET_CHIP=gfx942

cmake --build build/hrx-system
cmake --install build/hrx-system --prefix build/hrx-install \
  --component HrxPublicDist
cmake --install build/hrx-system --prefix build/hrx-tests \
  --component HrxTestsDist
```

Useful options:

| Option | Default | Purpose |
|--------|---------|---------|
| `HRX_HERMETIC_BUILD` | OFF | Error instead of using libhrx FetchContent fallbacks. |
| `IREE_HERMETIC_BUILD` | `${HRX_HERMETIC_BUILD}` | Error instead of using IREE/shared FetchContent fallbacks. |
| `LIBHRX_BUILD` | ON | Build libhrx and compatibility targets. |
| `LIBHRX_BUILD_CTS` | `${IREE_BUILD_TESTS}` | Build libhrx CTS binaries. |
| `LIBHRX_BUILD_PASSTHROUGH` | ON | Build HIP passthrough/interception tools. |
| `IREE_BUILD_TESTS` | ON | Build IREE runtime tests and CTS targets. |
| `HRX_INSTALL_TESTS` | `${IREE_BUILD_TESTS}` | Install a relocatable CTest tree. |
| `IREE_HAL_DRIVER_AMDGPU` | ON | Build the AMDGPU runtime HAL driver. |
| `IREE_ROCM_TEST_TARGET_CHIP` | empty | Target chip for ROCm tests that compile device code. |

Hermetic build options are intended for release and distro-style builds where
all dependencies must be supplied by package discovery.

## Running Tests

The installed test tree is relocatable and can run outside the build directory:

```bash
ctest --test-dir build/hrx-tests/share/hrx-system/tests \
  --output-on-failure --parallel "$(nproc)"
```

Set `HRX_TEST_TMPDIR` to put per-test temporary directories somewhere writable:

```bash
HRX_TEST_TMPDIR=/tmp/hrx-system-tests \
  ctest --test-dir build/hrx-tests/share/hrx-system/tests \
  --output-on-failure --parallel "$(nproc)"
```

GPU tests are labeled with `requires-gpu-*`. CPU-only runners should exclude
them:

```bash
ctest --test-dir build/hrx-tests/share/hrx-system/tests \
  --output-on-failure -LE requires-gpu-
```

## Release Artifacts

Tagged releases publish one tarball per artifact, plus manifests and environment
helper scripts where useful:

| Artifact | Contents |
|----------|----------|
| `hrx-public-linux-x86_64` | Public HRX install: `libhrx.so`, `libamdhip64.so`, headers, CMake package files, and tools. |
| `hrx-public-deps-linux-x86_64` | Runtime ROCm dependency subset needed with the public install, including HSA and AQL profile libraries. |
| `hrx-tests-linux-x86_64` | Installed CTest tree, unit tests, CTS binaries, test data, and benchmarks. |
| `hrx-rocm-buildenv-linux-x86_64` | ROCm build environment used by CI, including ROCm LLVM. |

To recreate the runtime overlay used by CI, extract the public package and the
public dependency package into the same root:

```bash
mkdir -p /tmp/hrx-root
cp hrx-public-linux-x86_64-env.sh /tmp/hrx-root/
tar --use-compress-program=unzstd -xf hrx-public-linux-x86_64-<tag>.tar.zst \
  -C /tmp/hrx-root
tar --use-compress-program=unzstd -xf hrx-public-deps-linux-x86_64-<tag>.tar.zst \
  -C /tmp/hrx-root
source /tmp/hrx-root/hrx-public-linux-x86_64-env.sh
```

Extract `hrx-tests-linux-x86_64-<tag>.tar.zst` into a separate tests root and
run its installed CTest tree against the composed runtime environment.

## Running HIP Applications

The public artifact contains an HRX-provided `lib/libamdhip64.so`. Preload that
library to route a HIP application through HRX without relinking the
application:

```bash
export HRX_ROOT=/tmp/hrx-root
export LD_LIBRARY_PATH="${HRX_ROOT}/lib:${HRX_ROOT}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export HRX_GPU_DRIVER=amdgpu

LD_PRELOAD="${HRX_ROOT}/lib/libamdhip64.so" ./my_hip_application
```

This path is for running HIP-based applications on the HRX compatibility layer.
The passthrough/debug libraries under `libhrx/src/passthrough/` are separate
developer tools for comparing behavior against the real ROCm HIP runtime.

## More Documentation

- `docs/testing/installed_tests.md`: installed test tree details.
- `libhrx/src/passthrough/README.md`: HIP passthrough and tracing tools.
