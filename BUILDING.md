# Building

This repository has two supported build systems:

- Bazel is the source-tree build and the source of truth for generated CMake
  build graph structure.
- CMake is the package, install, and embedding build.

`dev.py` is a command router. It prepares local tools and delegates to Bazel,
CMake, CTest, and project scripts. Anything `dev.py` does must also be possible
with the underlying tools directly.

## Quick Start

Bazel source-tree build:

```bash
python dev.py bazel setup
python dev.py bazel configure
python dev.py bazel test
```

CMake package build with AMDGPU enabled:

```bash
python dev.py cmake setup
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
python dev.py cmake build
python dev.py cmake test
```

Install Git hooks for the build system you use for commits:

```bash
python dev.py bazel hook
```

or:

```bash
python dev.py cmake hook
```

## Command Shape

Put wrapper execution and tool-environment options before the build-system
command:

```bash
python dev.py --dry-run bazel build //runtime/...
python dev.py --system bazel configure
python dev.py --verbose cmake test -R hrx
```

Arguments after `<build-system> <command>` belong to the underlying tool:

```bash
python dev.py bazel build //runtime/... --config=presubmit
python dev.py cmake configure -DCMAKE_BUILD_TYPE=Debug
python dev.py cmake build hrx --parallel 8
python dev.py cmake test -R hrx
```

Generated aliases follow the same shape:

```bash
iree-bazel-build //runtime/... --config=presubmit
iree-bazel-cquery 'kind(cc_library, //runtime/...)'
iree-bazel-info execution_root
iree-bazel-run //runtime/src/iree/base:allocator_benchmark
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
iree-cmake-test -R hrx
```

The repo also checks in POSIX Bazel launchers under `build_tools/bin/` for
paths that need a stable source-tree wrapper:

```bash
build_tools/bin/iree-bazel-query 'rdeps(//runtime/..., //runtime/src/iree/base)'
build_tools/bin/iree-bazel-cquery --output=files //runtime/src/iree/base
build_tools/bin/iree-bazel-try -e 'int main() { return 0; }'
```

`iree-bazel-run`, `iree-bazel-try`, and `iree-bazel-fuzz` build first and then
run the resolved executable directly, so Bazel does not hold its server lock
while a benchmark, tool, or fuzzer is running.

## Bazel Sanitizers

Bazel has native sanitizer configs for source-tree development and CI:

| Config | Meaning |
| --- | --- |
| `--config=asan` | AddressSanitizer with use-after-scope checks. |
| `--config=ubsan` | UndefinedBehaviorSanitizer with the function and vptr checks disabled to match the runtime's type-erased dispatch and `-fno-rtti` C++ mode. |
| `--config=tsan` | ThreadSanitizer. |
| `--config=msan` | MemorySanitizer. MSAN builds are useful before the host dependency stack is fully instrumented; MSAN test failures can be dependency-instrumentation failures rather than runtime bugs. |
| `--config=fuzzer` | libFuzzer build mode with ASAN enabled. |

Examples:

```bash
python dev.py bazel test //runtime/... --config=asan
python dev.py bazel test //runtime/src/iree/async/... --config=tsan
```

## IREE CI Reproduction

IREE source-tree CI is run through `build_tools/devtools/ci.py` so GitHub
workflow failures have copyable local commands. Command names are
`iree-<build-system>-<target-group>[-<configuration>]`. Bazel jobs take explicit
target patterns. CMake jobs use generated CTest names and labels directly.

```bash
python build_tools/devtools/ci.py iree-bazel-cpu --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-cpu-sanitizers --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-sanitizers --target //runtime/... --keep-going

python build_tools/devtools/ci.py iree-cmake-cpu --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-sanitizers --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu-sanitizers --keep-going
```

The aggregate `*-sanitizers` commands batch sanitizer configurations for CI
scheduling. Individual sanitizer commands are the targeted reproduction form:

```bash
python build_tools/devtools/ci.py iree-bazel-cpu-asan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-tsan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-ubsan --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu-tsan --keep-going
```

Sanitizer CI tests ASAN, UBSAN, and TSAN. MSAN is build-only until the CI host
dependency stack is MSAN-instrumented enough for test execution to produce
runtime signal instead of dependency instrumentation noise.

## Shared Project Configuration

Shared project options use CMake-style `-DNAME=VALUE` spelling. These options
are a small published configuration API, not a universal compatibility layer
between Bazel and CMake.

| Option | Values | CMake | Bazel portable | Bazel native |
| --- | --- | --- | --- | --- |
| `IREE_HAL_DRIVER_AMDGPU` | `ON`, `OFF` | Builds the AMDGPU runtime HAL driver. | Adds or removes `amdgpu` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_HIP` | `ON`, `OFF` | Currently forced `OFF` in this reduced runtime tree. | Adds or removes `hip` from the runtime driver registry and recursive package scope for testing and development. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_LOCAL_SYNC` | `ON`, `OFF` | Builds the local-sync runtime HAL driver. | Adds or removes `local-sync` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_LOCAL_TASK` | `ON`, `OFF` | Builds the local-task runtime HAL driver. | Adds or removes `local-task` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_NULL` | `ON`, `OFF` | Builds the null runtime HAL driver. | Adds or removes `null` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_ROCM_PATH` | path | Prepends the ROCm or TheRock SDK root to `CMAKE_PREFIX_PATH` and uses it for AMDGPU device tooling. | Writes `--repo_env=IREE_ROCM_PATH=<path>`. | `--repo_env=IREE_ROCM_PATH=<path>` |

The Bazel native driver flag is a complete list. Include every driver you want
enabled:

```bash
python dev.py bazel configure \
  --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null \
  --repo_env=IREE_ROCM_PATH=/opt/rocm
```

The portable spelling is shorter for common cases:

```bash
python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm
```

When AMDGPU is enabled, Bazel configuration also accepts `IREE_ROCM_PATH` from
the inherited environment. This keeps CI reproduction commands independent of
machine-local SDK paths:

```bash
IREE_ROCM_PATH=/opt/rocm python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
```

The Bazel HIP HAL driver is an opt-in testing/development path, not part of the
default libhrx build path. It is ROCm-backed, so `-DIREE_HAL_DRIVER_HIP=ON`
also requires `IREE_ROCM_PATH` through `-DIREE_ROCM_PATH=...`,
`--repo_env=IREE_ROCM_PATH=...`, or the inherited environment.

Other Bazel-native overrides belong in `.bazelrc.local`.

## Project Availability

CMake has project availability options because it configures a package build
tree. Bazel project availability is currently expressed by target selection.

| Option | Default | Lane | Meaning |
| --- | --- | --- | --- |
| `LIBHRX_BUILD` | `ON` | CMake | Builds libhrx and HRX compatibility targets. `ON` requires AMDGPU support; CMake enables `IREE_HAL_DRIVER_AMDGPU` by default when libhrx is built. |
| `LIBHRX_BUILD_CTS` | `${IREE_BUILD_TESTS}` | CMake | Builds libhrx CTS binaries. |
| `LIBHRX_BUILD_PASSTHROUGH` | `ON` | CMake | Builds HIP passthrough and interception developer tools. |
| `IREE_BUILD_TESTS` | `ON` | CMake | Builds runtime tests and CTS targets. |
| `HRX_INSTALL_TESTS` | `${IREE_BUILD_TESTS}` | CMake | Installs a relocatable CTest tree. |

## Dependency Resolution

`IREE_DEPENDENCY_MODE` is the CMake dependency policy:

| Value | Meaning |
| --- | --- |
| `pinned` | Uses checked-in source locks. This is the repository default. |
| `package` | Requires dependencies to be provided as packages or parent-project targets. |
| `auto` | Tries packages first, then falls back to pinned source dependencies. |

Bazel dependency resolution is controlled by `MODULE.bazel`,
`MODULE.bazel.lock`, and local `.bazelrc.local` overrides.

## Raw Tool Equivalents

The `dev.py` commands are intentionally thin. These pairs are equivalent in
normal local checkouts:

```bash
python dev.py bazel build //runtime/...
bazel build //runtime/...
```

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
```

```bash
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
cmake -S . -B ../builds/$(basename "$PWD") -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
```

```bash
python dev.py cmake build hrx
cmake --build ../builds/$(basename "$PWD") --target hrx
```

```bash
python dev.py cmake test -R hrx
ctest --test-dir ../builds/$(basename "$PWD") --output-on-failure -R hrx
```
