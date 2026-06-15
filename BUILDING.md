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
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON
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
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON
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
python build_tools/devtools/ci.py iree-bazel-vulkan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-vulkan-sanitizers --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-sanitizers --target //runtime/... --keep-going

python build_tools/devtools/ci.py iree-cmake-cpu --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-sanitizers --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan-sanitizers --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu-sanitizers --keep-going
```

The aggregate `*-sanitizers` commands batch sanitizer configurations for CI
scheduling. Individual sanitizer commands are the targeted reproduction form:

```bash
python build_tools/devtools/ci.py iree-bazel-cpu-asan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-vulkan-asan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-tsan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-ubsan --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan-ubsan --keep-going
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
| `IREE_HAL_DRIVER_HIP` | `ON`, `OFF` | Builds the HIP runtime HAL driver. | Adds or removes `hip` from the runtime driver registry and recursive package scope for testing and development. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_LOCAL_SYNC` | `ON`, `OFF` | Builds the local-sync runtime HAL driver. | Adds or removes `local-sync` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_LOCAL_TASK` | `ON`, `OFF` | Builds the local-task runtime HAL driver. | Adds or removes `local-task` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_NULL` | `ON`, `OFF` | Builds the null runtime HAL driver. | Adds or removes `null` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_VULKAN` | `ON`, `OFF` | Builds the Vulkan runtime HAL driver. | Adds or removes `vulkan` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_WEBGPU` | `ON`, `OFF` | Builds the WebGPU runtime HAL driver. | Adds or removes `webgpu` from the runtime driver registry and recursive package scope for WebGPU development. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_DEPENDENCY_MODE` | `pinned`, `package`, `auto` | Selects locked source archives, package discovery, or package-then-pinned dependency resolution. | Writes `--repo_env=IREE_DEPENDENCY_MODE=<mode>`. | `--repo_env=IREE_DEPENDENCY_MODE=<mode>` |
| `IREE_ROCM_DEPENDENCY_MODE` | `pinned`, `package`, `auto` | Overrides dependency resolution for ROCm header facades; empty uses package mode when `IREE_ROCM_PATH` is set and otherwise inherits `IREE_DEPENDENCY_MODE`. | Writes `--repo_env=IREE_ROCM_DEPENDENCY_MODE=<mode>`. | `--repo_env=IREE_ROCM_DEPENDENCY_MODE=<mode>` |
| `IREE_ROCM_PATH` | path | Prepends the ROCm or TheRock SDK root to `CMAKE_PREFIX_PATH`, uses it for AMDGPU device tooling, and selects ROCm package header mode by default. | Writes `--repo_env=IREE_ROCM_PATH=<path>` and `--repo_env=IREE_ROCM_DEPENDENCY_MODE=package` unless explicitly overridden. | `--repo_env=IREE_ROCM_PATH=<path>` |

The Bazel native driver flag is a complete list. Include every driver you want
enabled:

```bash
python dev.py bazel configure \
  --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null \
  --repo_env=IREE_ROCM_PATH=/opt/rocm \
  --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned
```

The portable spelling is shorter for common cases:

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
```

Pinned mode is the default. It lets AMDGPU host-side code compile without a
ROCm/TheRock root; Bazel writes `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none` when no
ROCm path is configured:

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
```

Package mode intentionally tests a configured ROCm/TheRock root. Setting
`IREE_ROCM_PATH` selects ROCm package header mode by default:

```bash
python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm
```

When a ROCm path is only needed for device tooling and the headers should still
come from pinned sources, set `IREE_ROCM_DEPENDENCY_MODE=pinned` explicitly.

When a ROCm path is needed, Bazel configuration also accepts `IREE_ROCM_PATH`
from the inherited environment. This keeps CI reproduction commands independent
of machine-local SDK paths:

```bash
IREE_ROCM_PATH=/opt/rocm python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned
```

The Bazel HIP HAL driver is an opt-in testing/development path, not part of the
default libhrx build path. It uses pinned HIP API headers by default and only
requires `IREE_ROCM_PATH` when package mode or ROCm device/runtime tooling is
required.

Loom target options describe product compiler capability: `LOOM_TARGET_AMDGPU=ON`
means Loom can compile for AMDGPU, including the target architecture metadata and
production artifact emission needed by that backend. Runtime execution remains a
separate concern controlled by Loom execution support and the runtime
`IREE_HAL_DRIVER_*` options.

The default dependency-satisfied Loom target set is
`amdgpu,iree_vm,spirv,x86`. AMDGPU and SPIR-V target compilation use pinned
source dependencies by default and do not enable the matching runtime HAL
drivers. WebAssembly remains opt-in until the WASI SDK repository is available
in this checkout. The default execution substrate set is `iree_hal,iree_vm`;
backend execution providers still require a matching runtime HAL driver such as
`IREE_HAL_DRIVER_VULKAN` or
`IREE_HAL_DRIVER_AMDGPU`.

CMake exposes `LOOM_TARGET_DEFAULTS` and `LOOM_EXECUTE_DEFAULTS` to set the
default value for dependency-satisfied target and execution options before the
individual `LOOM_TARGET_*` and `LOOM_EXECUTE_*` overrides are evaluated. Bazel
configuration writes complete native lists instead, so portable `-D...=OFF`
options remove entries from the default set.

AMDGPU has an additional compiler target selector list. `LOOM_TARGET_AMDGPU=ON`
selects the product capability; `LOOM_TARGET_AMDGPU_TARGETS` selects which
descriptor-backed AMDGPU processors are compiled into that capability. The
default selector is `loom_defaults`, which expands to every descriptor-backed
processor Loom currently supports:

| Descriptor set | Exact processors |
| --- | --- |
| `amdgpu.cdna3.core` | `gfx940`, `gfx941`, `gfx942` |
| `amdgpu.cdna4.core` | `gfx950` |
| `amdgpu.rdna3.core` | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153` |
| `amdgpu.rdna3_5.core` | `gfx1170`, `gfx1171`, `gfx1172` |
| `amdgpu.rdna4.core` | `gfx1200`, `gfx1201` |
| `amdgpu.rdna4.gfx125x.core` | `gfx1250`, `gfx1251` |

The accepted Loom AMDGPU selector vocabulary is the intersection of the shared
AMDGPU target map and Loom's descriptor-backed compiler support. It accepts:

- Source selectors: `loom_defaults`, `iree_hal`.
- Exact processors listed in the descriptor-set table above.
- Generic code-object selectors: `gfx9-4-generic`, `gfx11-generic`,
  `gfx12-generic`, `gfx12-5-generic`.
- Fully covered family selectors: `gfx94X-all`, `gfx94X-dcgpu`,
  `gfx950-all`, `gfx950-dcgpu`, `gfx110X-all`, `gfx110X-dgpu`,
  `gfx110X-igpu`, `gfx115X-all`, `gfx115X-igpu`, `gfx117X-all`,
  `gfx120X-all`, `gfx125X-all`.

Older shared selectors such as `gfx9-generic`, `gfx90a`, `gfx908`,
`gfx10-1-generic`, and `gfx10-3-generic` are still valid for runtime-side
AMDGPU tooling, but they are not Loom compiler targets until matching Loom
descriptor sets exist. The `iree_hal` source selector narrows Loom AMDGPU
support to the descriptor-backed subset requested by the runtime
`IREE_HAL_AMDGPU_TARGETS` setting. That is useful for executable-cache builds
that want Loom linked with exactly the runtime HAL target horizon, while normal
compiler and `loom-compile` builds should usually keep `loom_defaults`.

| Option | Values | CMake | Bazel portable | Bazel native |
| --- | --- | --- | --- | --- |
| `LOOM_TARGET_AMDGPU` | `ON`, `OFF` | Builds Loom AMDGPU target support and production AMDGPU emission. | Adds or removes `amdgpu` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_AMDGPU_TARGETS` | AMDGPU selectors | Selects descriptor-backed AMDGPU processors compiled into Loom AMDGPU target support. | Not exposed as a portable `-D` option. | `--//loom/config/target/amdgpu:targets=<complete-selector-list>` |
| `LOOM_TARGET_IREE_VM` | `ON`, `OFF` | Builds Loom IREE VM target support and production IREE VM emission. | Adds or removes `iree_vm` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_SPIRV` | `ON`, `OFF` | Builds Loom SPIR-V target support and production SPIR-V emission. | Adds or removes `spirv` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_WASM` | `ON`, `OFF` | Builds Loom WebAssembly target support and production Wasm emission. | Adds or removes `wasm` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_X86` | `ON`, `OFF` | Builds Loom x86 target support. | Adds or removes `x86` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_EMIT_LLVMIR` | `ON`, `OFF` | Builds LLVM IR debug/developer emission for enabled target archs. | Adds or removes `llvmir` from the explicit Loom emitter set. | `--//loom/config/emit:enable=<complete-emitter-list>` |
| `LOOM_EXECUTE_IREE_HAL` | `ON`, `OFF` | Builds Loom execution providers that run through IREE HAL when a matching runtime HAL driver is enabled. | Adds or removes `iree_hal` from the Loom execute substrate set. | `--//loom/config/execute:enable=<complete-execute-list>` |
| `LOOM_EXECUTE_IREE_VM` | `ON`, `OFF` | Builds Loom execution providers that run through the IREE VM substrate. | Adds or removes `iree_vm` from the Loom execute substrate set. | `--//loom/config/execute:enable=<complete-execute-list>` |

The native Loom target flag is a complete list. The default target set is
`amdgpu,iree_vm,spirv,x86`, and the default execution substrate set is
`iree_hal,iree_vm`:

```bash
python dev.py bazel configure \
  --//loom/config/target:enable=amdgpu,iree_vm,spirv,x86
```

AMDGPU compiler target selection is also a complete list. Bazel uses
comma-separated list values; CMake uses normal semicolon-separated CMake lists:

```bash
python dev.py bazel configure \
  -DLOOM_TARGET_AMDGPU=ON \
  --//loom/config/target/amdgpu:targets=gfx942,gfx120X-all,gfx12-5-generic

python dev.py cmake configure \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS='gfx942;gfx120X-all;gfx12-5-generic'
```

Use the shared runtime selector only when Loom should intentionally match the
runtime HAL target horizon:

```bash
python dev.py cmake configure \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=iree_hal \
  -DIREE_HAL_AMDGPU_TARGETS='gfx942;gfx1201'
```

The portable spelling can disable a default target without exposing the
internal target-architecture and emitter slices:

```bash
python dev.py bazel configure -DLOOM_TARGET_SPIRV=OFF
```

LLVM IR emission is a debug/developer artifact path. It is explicit even when a
native target such as AMDGPU or x86 is enabled:

```bash
python dev.py bazel configure \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_EMIT_LLVMIR=ON
```

Execution options describe the runtime substrate available to Loom tools, not a
target backend by themselves. For example, AMDGPU execution needs the AMDGPU
Loom target, the IREE HAL execution substrate, and the AMDGPU runtime HAL
driver:

```bash
python dev.py bazel configure \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_EXECUTE_IREE_HAL=ON \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm
```

CPU-only broad compiler validation should use the dedicated Loom AMDGPU compile
slices. These do not enable the AMDGPU runtime HAL driver or require matching
hardware:

```bash
python build_tools/devtools/ci.py iree-bazel-loom-amdgpu
python build_tools/devtools/ci.py iree-cmake-loom-amdgpu
```

The raw `//loom/config/target/arch:enable=...`,
`//loom/config/emit:enable=...`, and `//loom/config/execute:enable=...` values
are advanced source-embedding and CI-audit surfaces. They exist to build narrow
slices deliberately; the published portable API is the `LOOM_TARGET_*` product
target set plus explicit debug emitters and execution substrates.

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

`IREE_DEPENDENCY_MODE` is the shared dependency policy:

| Value | Meaning |
| --- | --- |
| `pinned` | Uses checked-in source locks. This is the repository default. |
| `package` | Requires dependencies to be provided as packages or parent-project targets. |
| `auto` | Tries packages first, then falls back to pinned source dependencies. |

CMake consumes this as a cache variable. Bazel consumes the same values through
`--repo_env=IREE_DEPENDENCY_MODE=...`; `dev.py bazel configure` writes that
repo environment into `.bazelrc.configured`. ROCm header facades also support
`IREE_ROCM_DEPENDENCY_MODE`. When it is empty, `IREE_ROCM_PATH` selects ROCm
package mode; without a ROCm path it inherits the global mode. libhrx TheRock
validation should set `IREE_ROCM_PATH` while leaving ordinary source
dependencies pinned.

## Raw Tool Equivalents

The `dev.py` commands are intentionally thin. These pairs are equivalent in
normal local checkouts:

The CMake raw equivalents below use the default `build/cmake` tree; replace it
with the path selected by `--cmake-build-dir` or `IREE_CMAKE_BUILD_DIR` when
using a different tree.

```bash
python dev.py bazel build //runtime/...
bazel build //runtime/...
```

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=pinned
python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=pinned
```

```bash
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
cmake -S . -B build/cmake -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
```

```bash
python dev.py cmake build hrx
cmake --build build/cmake --target hrx
```

```bash
python dev.py cmake test -R hrx
ctest --test-dir build/cmake --output-on-failure -R hrx
```
