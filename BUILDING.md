# Building

This repository has two supported build systems:

- Bazel is the source-tree build and the source of truth for generated CMake
  build graph structure.
- CMake is the package, install, and embedding build.

`dev.py` is a command router. It prepares local tools and delegates to Bazel,
CMake, CTest, and project scripts. Anything `dev.py` does must also be possible
with the underlying tools directly.

Build-system integration follows the same ownership boundary. Tests that
configure or drive CMake are registered only in CTest; the Bazel build and test
graphs neither invoke CMake nor require it to be installed. Bazel-to-CMake
conversion is an explicit source-generation and presubmit step.

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

After setup, generated aliases follow the same shape. They are the stable
spelling for docs, scripts, and agent instructions:

```bash
iree-bazel-build //runtime/... --config=presubmit
iree-bazel-cquery 'kind(cc_library, //runtime/...)'
iree-bazel-info execution_root
iree-bazel-run //runtime/src/iree/base:allocator_benchmark
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-cmake-test -R hrx
```

CMake build and run commands also accept a unique executable output name, such
as `loom-compile`. If two packages emit the same filename, use the qualified
generated alias so the selection remains explicit.

`iree-cmake-test` asks CTest for the exact selected records, builds the
concrete CMake roots declared by those records, and then runs the same
selection. A filtered test run therefore needs no separate matching
`iree-cmake-build` invocation, and source-only selections perform no build.

PATH aliases are also the stable spelling for launcher-backed commands:

```bash
iree-bazel-query 'rdeps(//runtime/..., //runtime/src/iree/base)'
iree-bazel-cquery --output=files //runtime/src/iree/base
iree-bazel-try -e 'int main() { return 0; }'
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

IREE source-tree CI is run through the repo-local CI command script so GitHub
workflow failures have copyable local commands. This is the script-backed
surface; ordinary build/test docs use the `iree-bazel-*` and `iree-cmake-*`
PATH aliases above. Command names are
`iree-<build-system>-<target-group>[-<configuration>]`. Bazel jobs take explicit
target patterns. CMake jobs use generated CTest names and labels directly.

The CI compiler matrix assigns each compiler a deliberate role. Host compiler
selection is explicit through `CC`, `CXX`, and `AR`; fetching ROCm never changes
the host compiler through `PATH`. AMDGPU device actions receive the ROCm LLVM
root independently through build configuration.

| Workflow surface | Host compiler | AMDGPU device compiler | Coverage intent |
| --- | --- | --- | --- |
| Presubmit | Fetched ROCm Clang 23 | None | Runs repository policy checks and clang-tidy with the newest supported LLVM APIs. |
| IREE Bazel/CMake CPU and importers | Ubuntu Clang 18 | None | Primary source build, test, sanitizer, and importer coverage. |
| IREE Bazel/CMake Vulkan | Fetched ROCm Clang 23 | None | Vulkan source build and execution coverage on self-hosted runners that do not currently provision a generic Clang toolchain. |
| IREE Bazel/CMake AMDGPU | Fetched ROCm Clang 23 | Fetched ROCm Clang 23 | Compiles and runs AMDGPU host and device code in the ROCm toolchain environment. |
| IREE Bazel repository build | GCC 13 system toolchain | Fetched ROCm Clang 23 | Builds every supported Linux HAL driver, Loom target/importer, and build-compatible target under `//...`; it does not duplicate test execution. |
| libHRX Bazel | Fetched ROCm Clang 23 | Fetched ROCm Clang 23 | Validates the source HRX product against its shipping ROCm compiler environment. |
| Installed CMake/package CI | Fetched ROCm Clang 23 | Fetched ROCm Clang 23 | Builds, installs, packages, and tests the composed HRX distribution. |

The repository-wide GCC lane intentionally uses the complete `//...` pattern,
not a hand-maintained project list or exclusions. Platform-incompatible targets
remain incompatible through their declared Bazel constraints; CUDA and Metal
join this lane when their Linux Bazel dependency surfaces are enabled. The
lane does not override GCC's linker selection; Bazel uses the GNU binutils
provided by the system toolchain. The copyable build-shape command is:

```bash
CC=gcc CXX=g++ AR=ar \
  python build_tools/devtools/ci.py iree-bazel-repository-build --keep-going
```

```bash
python build_tools/devtools/ci.py iree-bazel-cpu --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-cpu-sanitizers --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-vulkan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu --amdgpu-target gfx942 --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-asan --amdgpu-target gfx942 --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-tsan --amdgpu-target gfx942 --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-ubsan --amdgpu-target gfx942 --keep-going

python build_tools/devtools/ci.py iree-cmake-cpu --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-sanitizers --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan-sanitizers --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu --amdgpu-target gfx942 --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu-sanitizers --amdgpu-target gfx942 --keep-going
```

AMDGPU commands default to `gfx942`. `--amdgpu-target` accepts an exact target
or family selector and applies it to both the runtime HAL target set and Loom's
`iree_hal`-derived compiler target set. Bazel AMDGPU commands build both source
trees, then run the union of tests that require the AMDGPU HAL at build time or
an AMD GPU at execution time.

AMDGPU Bazel sanitizer configurations are separate CI jobs so they build and
test independently. Aggregate CPU Bazel and CMake commands remain available as
local batch commands. Individual sanitizer commands are the targeted
reproduction form:

```bash
python build_tools/devtools/ci.py iree-bazel-cpu-asan --target //runtime/... --keep-going
python build_tools/devtools/ci.py iree-bazel-amdgpu-tsan --amdgpu-target gfx942 --keep-going
python build_tools/devtools/ci.py iree-cmake-cpu-ubsan --keep-going
python build_tools/devtools/ci.py iree-cmake-vulkan-ubsan --keep-going
python build_tools/devtools/ci.py iree-cmake-amdgpu-tsan --amdgpu-target gfx942 --keep-going
```

AMDGPU Bazel CI tests ASAN, UBSAN, and TSAN. It does not publish an MSAN lane:
the CI host dependency stack is not MSAN-instrumented enough for execution to
produce useful runtime signal. CPU Bazel and CMake retain their explicit
build-only MSAN configurations.

## Shared Project Configuration

Shared project options use CMake-style `-DNAME=VALUE` spelling. These options
are a small published configuration API, not a universal compatibility layer
between Bazel and CMake.

| Option | Values | CMake | Bazel portable | Bazel native |
| --- | --- | --- | --- | --- |
| `IREE_HAL_DRIVER_AMDGPU` | `ON`, `OFF` | Builds the AMDGPU runtime HAL driver. | Adds or removes `amdgpu` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
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

| Descriptor set | Processor targets |
| --- | --- |
| `amdgpu.cdna3.core` | `gfx940`, `gfx941`, `gfx942` |
| `amdgpu.cdna4.core` | `gfx950` |
| `amdgpu.gfx9_4.generic.core` | `gfx9-4-generic` |
| `amdgpu.rdna3.core` | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103` |
| `amdgpu.rdna3_5.core` | `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153` |
| `amdgpu.gfx11.generic.core` | `gfx11-generic` |
| `amdgpu.rdna4m.core` | `gfx1170`, `gfx1171`, `gfx1172` |
| `amdgpu.rdna4.core` | `gfx1200`, `gfx1201` |
| `amdgpu.gfx12.generic.core` | `gfx12-generic` |
| `amdgpu.rdna4.gfx125x.core` | `gfx1250`, `gfx1251` |
| `amdgpu.gfx12_5.generic.core` | `gfx12-5-generic` |

Every generic processor is a compiler target in its own right with a distinct
descriptor and encoding contract. A generic contract may share immutable
generated storage with an exact-family contract when their current contents
match, but it never selects that exact contract as an implementation alias.
`gfx11-generic` is the common GFX11 surface validated against both the RDNA 3
and RDNA 3.5 ISA descriptions, and its code objects cover `gfx1100`-`gfx1103`
and `gfx1150`-`gfx1153`. The `gfx1170`-`gfx1172` targets remain exact-only
because the pinned device toolchain does not yet expose LLVM's distinct
`gfx11-7-generic` code-object target.
`gfx9-4-generic` is the common CDNA 3/CDNA 4 surface for `gfx940`, `gfx941`,
`gfx942`, and `gfx950`; its instruction, matrix, resource, scheduling, ABI,
limit, and occupancy facts are portable member intersections.

The accepted Loom AMDGPU selector vocabulary is the intersection of the shared
AMDGPU target map and Loom's descriptor-backed compiler support. It accepts:

- Source selectors: `loom_defaults`, `iree_hal`.
- Exact processors listed in the descriptor-set table above.
- Generic compiler targets: `gfx9-4-generic`, `gfx11-generic`,
  `gfx12-generic`, `gfx12-5-generic`.
- Fully covered family selectors: `gfx94X-all`, `gfx94X-dcgpu`,
  `gfx950-all`, `gfx950-dcgpu`, `gfx110X-all`, `gfx110X-dgpu`,
  `gfx110X-igpu`, `gfx115X-all`, `gfx115X-igpu`, `gfx117X-all`,
  `gfx120X-all`, `gfx125X-all`.

Older shared selectors such as `gfx9-generic`, `gfx90a`, `gfx908`,
`gfx10-1-generic`, and
`gfx10-3-generic` are still valid for runtime-side AMDGPU tooling, but they are
not Loom compiler targets until matching Loom descriptor sets exist. The
`iree_hal` source selector narrows Loom AMDGPU support to the descriptor-backed
subset requested by the runtime `IREE_HAL_AMDGPU_TARGETS` setting. That is
useful for runtime-integrated JIT builds that want Loom linked with exactly the
runtime HAL target horizon, while normal compiler and `loom-compile` builds
should usually keep `loom_defaults`.

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

## Optional Local NativeLink Execution

NativeLink can provide one shared Bazel action cache and execution limit across
multiple local worktrees. The repository provides an inert named Bazel config
and a loopback-only local server configuration; ordinary builds remain
unchanged until `--config=nativelink` is selected. See the
[local NativeLink guide](build_tools/nativelink/README.md) for installation,
startup, verification, capacity, and trust-boundary details.

## Loom Importers

Importer frontends are optional dependency lanes. Start with the importer-local
docs instead of expanding the root build surface:

```bash
ls loom/py/loom/importers
```

The entry point is `loom/py/loom/importers/README.md`. It explains importer
build selection, managed importer Python environments, and the current
TileLang/MLIR split. Managed importer environments are selected with
`python dev.py importers setup <name>` and consumed by Bazel/CMake through
`--importer-env <name>`.

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
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
cmake -S . -B build/cmake -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm -DIREE_ROCM_DEPENDENCY_MODE=package
```

```bash
python dev.py cmake build hrx
cmake --build build/cmake --target hrx
```

```bash
python dev.py cmake test -R hrx
ctest --test-dir build/cmake -R hrx --show-only=json-v1
cmake --build build/cmake --target selected-root-a selected-root-b
ctest --test-dir build/cmake --output-on-failure -R hrx
```

In the raw pipeline, `selected-root-a selected-root-b` stands for the stable
union produced by joining the selected CTest names with the validated
`iree_ctest_build_targets.json` catalog generated beside the CTest files.

## Platform-Specific Host Builds

### Windows

Windows builds require an x64 MSVC ABI environment even when `clang-cl` is the
host compiler. Install Python 3.12, Visual Studio 2022 Build Tools with the x64
C++ tools and a Windows SDK, and Ninja. Install LLVM separately when building
with `clang-cl`. The CI CMake version is 3.31.6; using that version locally
removes an otherwise unhelpful source of generator differences.

Start from an x64 Visual Studio developer shell so `INCLUDE`, `LIB`, the SDK
tools, and the MSVC linker are available. Git for Windows also ships a Unix
program named `link.exe`, so compiler activation order is load-bearing:

```powershell
where.exe cl
where.exe link
```

The first `link.exe` must be the MSVC linker, not Git's `usr\bin\link.exe`.
Create the repository tool environment and add the CI-pinned CMake plus Ninja:

```powershell
python dev.py cmake setup --venv
.\.venv\Scripts\python.exe -m pip install --upgrade cmake==3.31.6 ninja
python dev.py cmake doctor
```

Keep Windows build trees short and keep one tree per compiler. The `C:\b` CMake
trees below remain within the legacy Win32 path limit and do not require the
machine-wide `LongPathsEnabled` policy. Bazel has a different host contract:
its managed Python runfiles exceed the legacy limit and use symbolic links, so
Windows Bazel hosts require `LongPathsEnabled` plus Developer Mode or an
equivalent symbolic-link policy. Provision both policies in the base image for
CI runners that cannot elevate during a job. `python dev.py bazel configure`
and `python dev.py bazel doctor` diagnose those capabilities.

Windows Firewall displays an interactive approval prompt when a newly built
executable begins listening for inbound connections. Approving one executable
is not durable because build output paths change across configurations and
rebuilds. On an unattended development or CI host, disable listening
notifications for every network profile from a normal PowerShell session with
one elevated command:

```powershell
Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList '-NoProfile','-Command','Set-NetFirewallProfile -Profile Domain,Private,Public -NotifyOnListen False'
```

This leaves Windows Firewall enabled and does not add an inbound allow rule;
remote connections remain subject to the active firewall policy. Verify the
effective setting with:

```powershell
Get-NetFirewallProfile | Select-Object Name, Enabled, NotifyOnListen
```

Restore the interactive notifications with the same command and
`-NotifyOnListen True`.

Windows Bazel builds use `clang-cl` by default and require `BAZEL_LLVM` to name
the LLVM installation root. The MSVC ABI tools and SDK still come from the
active Visual Studio developer environment. Configure once, then build the
normal Loom tool surface:

```powershell
$env:BAZEL_LLVM = 'C:\Program Files\LLVM'
python dev.py bazel setup --venv
python dev.py bazel configure
python dev.py bazel build `
  //loom/src/loom/tools/iree-run-loom `
  //loom/src/loom/tools/loom-compile `
  //loom/binding/c:loomc
```

Use the explicit MSVC lane when checking both host compilers. It clears the
clang-cl execution-platform selection while preserving the same configured
feature and dependency graph:

```powershell
python dev.py bazel build `
  //loom/src/loom/tools/iree-run-loom `
  //loom/src/loom/tools/loom-compile `
  //loom/binding/c:loomc `
  --config=windows-msvc
```

The following is the host-only Loom baseline: it builds the VM and x86 target
paths without requiring ROCm, Vulkan, WebGPU, or libHRX.

```powershell
$baseOptions = @(
  '-GNinja'
  '-DCMAKE_BUILD_TYPE=RelWithDebInfo'
  '-DIREE_BUILD_TESTS=ON'
  '-DIREE_BUILD_BENCHMARKS=ON'
  '-DLIBHRX_BUILD=OFF'
  '-DIREE_DEPENDENCY_MODE=pinned'
  '-DIREE_HAL_DRIVER_AMDGPU=OFF'
  '-DIREE_HAL_DRIVER_VULKAN=OFF'
  '-DIREE_HAL_DRIVER_WEBGPU=OFF'
  '-DLOOM_TARGET_AMDGPU=OFF'
  '-DLOOM_TARGET_SPIRV=OFF'
  '-DLOOM_TARGET_WASM=OFF'
)

$llvmBin = 'C:\Program Files\LLVM\bin'
$env:PATH = "$llvmBin;$env:PATH"
$env:CC = "$llvmBin\clang-cl.exe"
$env:CXX = "$llvmBin\clang-cl.exe"
$env:AR = "$llvmBin\llvm-lib.exe"
python dev.py --cmake-build-dir C:\b\hrx-clang cmake configure @baseOptions
python dev.py --cmake-build-dir C:\b\hrx-clang cmake build `
  loom-compile iree-run-loom loom-check loom-format loom-opt loom-link `
  iree-test-loom iree-benchmark-loom --parallel 8
python dev.py --cmake-build-dir C:\b\hrx-clang cmake test `
  -R '^loom/tools/(.*execution_test|loom-check/test/.*)$' -j 8
```

Reset the compiler selection for the distinct MSVC tree. The separate build
directory, rather than shell state, keeps compiler identities from leaking
across configurations:

```powershell
$env:CC = 'cl.exe'
$env:CXX = 'cl.exe'
$env:AR = 'lib.exe'
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake configure @baseOptions
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake build `
  loom-compile iree-run-loom loom-check loom-format loom-opt loom-link `
  iree-test-loom iree-benchmark-loom --parallel 8
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake test `
  -R '^loom/tools/(.*execution_test|loom-check/test/.*)$' -j 8
```

Repository-wide Loom hygiene has a broader compiler-capability contract than
the host-only smoke: `loom-format` verifies every tracked standalone module
with the AMDGPU, IREE VM, LLVM IR, SPIR-V, and x86 target descriptors. A CMake
tree used for `cmake precommit` therefore needs AMDGPU and SPIR-V target support
even when their HAL drivers remain disabled:

```powershell
python dev.py --cmake-build-dir C:\b\hrx-clang-presubmit cmake configure `
  @baseOptions -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_SPIRV=ON
python dev.py --cmake-build-dir C:\b\hrx-clang-presubmit cmake precommit
```
