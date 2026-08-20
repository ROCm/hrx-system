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

Use a ROCm installation or unpacked ROCm build environment that provides HSA,
AQL profile headers, and a ROCm LLVM toolchain. The CI path uses ROCm clang and
forces libraries into `lib`.

### Linux

The blessed local command path uses `dev.py` to keep setup, hook configuration,
and build-lane selection consistent:

```bash
python dev.py cmake setup
python dev.py cmake configure \
  -DIREE_ROCM_PATH=/opt/rocm \
  -DIREE_ROCM_DEPENDENCY_MODE=package \
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
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_AMDGPU_TARGETS='gfx942;gfx1151;gfx1201'

python dev.py cmake build
cmake --install build/cmake --prefix build/hrx-install \
  --component HrxPublicDist
cmake --install build/cmake --prefix build/hrx-tests \
  --component HrxTestsDist
```

The equivalent raw CMake command is still valid when embedding or debugging the
build without `dev.py`:

```bash
cmake -S . -B build/hrx-system -GNinja \
  -DIREE_ROCM_PATH=/opt/rocm \
  -DIREE_ROCM_DEPENDENCY_MODE=package \
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
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_AMDGPU_TARGETS='gfx942;gfx1151;gfx1201'

cmake --build build/hrx-system
cmake --install build/hrx-system --prefix build/hrx-install \
  --component HrxPublicDist
cmake --install build/hrx-system --prefix build/hrx-tests \
  --component HrxTestsDist
```

### Windows

Run the build from an x64 Developer PowerShell for Visual Studio 2022. Set
`THEROCK_ROOT` to a TheRock checkout whose `build/dist/rocm` tree includes the
ROCm LLVM toolchain, headers, and `hsa-runtime64.dll`. For example, configure
TheRock with `THEROCK_ENABLE_HSA_WINDOWS_SHARED_RUNTIME=ON` before building it.

The equivalent `dev.py` flow for a `gfx1201` build is:

```powershell
$env:THEROCK_ROOT = "D:\src\TheRock"
$rocmRoot = Join-Path $env:THEROCK_ROOT "build\dist\rocm"
$aqlProfileRoot = Join-Path $env:THEROCK_ROOT "build\profiler\aqlprofile\dist"
$cmakePrefixPath = "$aqlProfileRoot;$rocmRoot;$rocmRoot\lib\rocm_sysdeps"
$llvmBin = Join-Path $rocmRoot "lib\llvm\bin"

python dev.py cmake setup
python dev.py cmake configure --fresh `
  "-DCMAKE_PREFIX_PATH=$cmakePrefixPath" `
  "-DIREE_ROCM_PATH=$rocmRoot" `
  -DIREE_ROCM_DEPENDENCY_MODE=package `
  -DCMAKE_INSTALL_LIBDIR=lib `
  "-DCMAKE_C_COMPILER=$llvmBin\clang-cl.exe" `
  "-DCMAKE_CXX_COMPILER=$llvmBin\clang-cl.exe" `
  "-DCMAKE_AR=$llvmBin\llvm-lib.exe" `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DIREE_HAL_DRIVER_AMDGPU=ON `
  -DIREE_HAL_AMDGPU_TARGETS=gfx1201

python dev.py cmake build --parallel
python dev.py cmake test --output-on-failure
cmake --install build/cmake --prefix build/hrx-install `
  --component HrxPublicDist
cmake --install build/cmake --prefix build/hrx-tests `
  --component HrxTestsDist
```

Add the ROCm and HRX DLL directories to `PATH` before running the installed
tools or tests:

```powershell
$env:PATH = "$rocmRoot\bin;$PWD\build\hrx-install\bin;$env:PATH"
& .\build\hrx-install\bin\hrx-info.exe
ctest --test-dir build/hrx-tests/share/hrx-system/tests --output-on-failure
```

This default build loads `hsa-runtime64.dll` dynamically. A Windows ROCr
package that exports `hsa-runtime64::hsa-runtime64_static` can instead be linked
statically by changing the runtime selection and CRT mode during configure:

```powershell
-DIREE_HAL_AMDGPU_LIBHSA_STATIC=ON `
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
```

Useful options:

| Option | Default | Purpose |
|--------|---------|---------|
| `IREE_DEPENDENCY_MODE` | `pinned` | Controls dependency resolution: `pinned`, `package`, or `auto`. |
| `LIBHRX_BUILD` | ON | Build libhrx and compatibility targets. |
| `LIBHRX_BUILD_CTS` | `${IREE_BUILD_TESTS}` | Build libhrx CTS binaries. |
| `LIBHRX_BUILD_HIP_BINDING` | ON | Build the HIP runtime API compatibility DLL. |
| `LIBHRX_BUILD_PASSTHROUGH` | ON | Build HIP passthrough/interception tools. |
| `IREE_BUILD_TESTS` | ON | Build IREE runtime tests and CTS targets. |
| `HRX_INSTALL_TESTS` | `${IREE_BUILD_TESTS}` | Install a relocatable CTest tree. |
| `IREE_HAL_DRIVER_AMDGPU` | ON | Build the AMDGPU runtime HAL driver. |
| `IREE_HAL_AMDGPU_LIBHSA_STATIC` | OFF | Link an installed static HSA runtime instead of loading it dynamically. |
| `IREE_ROCM_DEPENDENCY_MODE` | empty | Overrides ROCm header dependency resolution: `pinned`, `package`, or `auto`; empty selects package mode when `IREE_ROCM_PATH` is set. |
| `IREE_ROCM_PATH` | empty | ROCm or TheRock SDK root used for package-mode ROCm headers and device tooling. |
| `IREE_HAL_AMDGPU_TARGETS` | checked-in AMDGPU generic set | AMDGPU target selectors supported by the runtime build. |

`IREE_DEPENDENCY_MODE=pinned` uses the checked-in source lock and is the
repository default. `package` requires embedding or distribution builds to
provide package targets and never falls back to source fetching. `auto` tries
package discovery first and falls back to pinned source dependencies.

## Local Presubmit

Set up the development tools once, install the lane-specific Git hook, and run
the local change-set checks before committing:

```bash
python dev.py bazel setup
python dev.py bazel hook --profile paranoid
python dev.py bazel precommit
```

Use the CMake lane when you are working from the package/install-test side:

```bash
python dev.py cmake setup
python dev.py cmake hook --profile default
python dev.py cmake precommit
```

`precommit` checks staged, unstaged, and untracked files. Use
`python dev.py <lane> precommit --base <git-ref>` to check the branch diff from
the merge base through `HEAD` plus local changes, or `--staged` for staged
files only. Add `--profile default`, `--profile paranoid`, or `--profile ci`
to select the check profile for a manual run. Re-run
`python dev.py <lane> hook --profile <profile>` to change the default profile
used by Git commits. Use `python dev.py <lane> presubmit` for the full-tree
CI-shaped check.

Mechanical fixups are explicit:

```bash
python dev.py bazel fix
python dev.py cmake fix
```

Test-bearing Git commit-hook profiles apply mechanical fixups before running
the same profile in non-mutating check mode. The hook validates commit scope,
not the full branch: files staged for commit plus files changed by `HEAD`, so
amended commits include the commit being replaced. See
`CONTRIBUTING.md` for contributor setup and `build_tools/lefthook/README.md`
for the hook architecture.

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

Tests that require runtime hardware resources are labeled with
`runtime-resource=*`. CPU-only runners should exclude them:

```bash
ctest --test-dir build/hrx-tests/share/hrx-system/tests \
  --output-on-failure -LE runtime-resource=
```

GPU runners use the AMD GPU resource label to run all GPU-dependent installed
tests across HRX and runtime projects:

```bash
ctest --test-dir build/hrx-tests/share/hrx-system/tests \
  --output-on-failure -L runtime-resource=amd-gpu
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
| `hrx-public-windows-x86_64` | Public HRX Windows install: `hrx.dll`, import library, public HRX headers, CMake package files, and `hrx-info`. |
| `hrx-public-deps-windows-x86_64` | Reserved public dependency overlay for the Windows package; the initial Windows package does not bundle ROCm runtime DLLs. |
| `hrx-tests-windows-x86_64` | Windows installed CTest tree, runtime/libhrx unit tests, and test data. |
| `hrx-rocm-buildenv-windows-x86_64` | ROCm Windows build environment used by CI, including ROCm LLVM. |

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
Windows release artifacts use `.zip` archives and include `.ps1` and `.cmd`
environment helper scripts next to the package archive.

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

- `BUILDING.md`: source-build, embedding, and build configuration options.
- `CONTRIBUTING.md`: contributor workflow and `dev.py` modes.
- `build_tools/lefthook/README.md`: Lefthook profiles and hook architecture.
- `docs/testing/installed_tests.md`: installed test tree details.
- `libhrx/src/passthrough/README.md`: HIP passthrough and tracing tools.
