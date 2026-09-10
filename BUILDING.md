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

Developer environment setup selects Python 3.12, shared by the tested tool
and importer environments. Install that interpreter before setup. Setup rejects
an incompatible populated environment; move it aside or choose a new
`--tool-root` before retrying.

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

### Bazel target selection

Bazel builds for the host by default. Destination and compiler choices are
independent:

| Build host | Destination/compiler | Configs |
| --- | --- | --- |
| Linux | Windows x86-64, clang-cl | `--config=windows-x86_64` |
| Windows x64 | Native, clang-cl (default) | None |
| Windows x64 | Native, MSVC | `--config=windows-msvc` |
| Linux or macOS | macOS arm64 | `--config=macos-arm64` |
| Linux or macOS | macOS x86-64 | `--config=macos-x86_64` |
| macOS | Native, Xcode/Command Line Tools | None |

Destination configs set only `--platforms`. Native builds infer their
destination from the host, so adding a matching destination config is optional.
For example, a native MSVC build needs only `--config=windows-msvc`; adding
`--config=windows-x86_64` produces the same selection in either flag order.
`windows-clang-cl` explicitly selects the default Windows compiler. MSVC requires
a Windows build host; Linux-to-Windows builds use clang-cl. macOS destinations
use Xcode/Command Line Tools on a Mac and LLVM on Linux.

Build-time generators retain the host toolchain and remain executable on the
build host. Native builds do not require either cross SDK installation. SDK
acquisition is separate from Bazel configuration; the sections below describe
the inputs for each destination.

Linux cross builds use stable execution-root-relative compiler and SDK paths.
Compiler and linker actions declare separate file sets, including LLVM's
resolved non-glibc shared libraries; the host supplies its Linux loader and
glibc. Native Windows retains upstream `local_config_cc` discovery and its
installation paths. Shared remote caching and execution additionally require
a specified execution environment; these local configurations do not define one.

Cross-built executables run on their destination OS. Transfer the executable,
dependent libraries, debug artifacts, and consumer runfiles to that host.
Linux `bazel run` and `bazel test` cannot execute Windows or macOS binaries.
Starlark test wrappers require a matching test execution platform even during
build analysis; build their source binary targets when only producing artifacts.
For example,
`//runtime/src/iree/hal/drivers/task/executable/elf:elf_module_test_binary`
is the artifact target for the `:elf_module_test` wrapper. Destination selection
does not configure a remote executor or test runner.

### Windows targets

The selected C/C++ toolchain also supplies the MASM assembler and sanitizer
runtimes. Native Windows uses the tools discovered in its Visual Studio/LLVM
environment. Linux cross builds use clang-cl, lld-link, and llvm-ml from the
selected Linux LLVM installation.

#### Cross-compilation from Linux

Provide a Linux LLVM installation and a Windows SDK/MSVC sysroot. The Bazel
repository reads `LLVM_ROOT` and `WINSDK_ROOT` from the environment or explicit
`--repo_env` options; acquisition is independent of the build configuration.
For example, [xwin](https://github.com/Jake-Shadle/xwin) can prepare a sysroot
with the required versioned MSVC/Windows Kits layout and case-correction
symlinks:

```bash
xwin --accept-license --arch x86_64 splat \
  --use-winsysroot-style --preserve-ms-arch-notation \
  --include-debug-libs --output /path/to/windows-sysroot
export LLVM_ROOT=/path/to/llvm
export WINSDK_ROOT=/path/to/windows-sysroot
```

The build consumes headers and libraries; it does not require a particular
installer or a runtime-DLL packaging layout.

```bash
iree-bazel-build --config=windows-x86_64 //tools:iree-dump-cpuinfo
iree-bazel-build --config=windows-x86_64 -c dbg \
  //runtime/src/iree/base/testing:dynamic_library_test
iree-bazel-build --config=windows-x86_64 -c opt //tools:iree-dump-cpuinfo
```

Debug and fastbuild outputs include PDBs. Cross-built DLLs declare exports with
`__declspec(dllexport)` or a `win_def_file`; the cross toolchain disables Bazel's
Windows-only automatic export extractor. Native Windows retains that capability.
The default CRT is dynamic (`/MD`, or `/MDd` in debug). Execution requires the
corresponding Windows runtime DLLs, installed on the Windows machine or deployed
beside the executable. `--features=static_link_msvcrt` selects `/MT` or `/MTd`
when a consumer requires a static CRT.

For cross-built AddressSanitizer targets, add `--config=asan` and set
`WINDOWS_COMPILER_RT_ROOT` (or `--repo_env=WINDOWS_COMPILER_RT_ROOT=...`) to the
`lib/clang/<version>/lib/windows` directory from a Windows LLVM distribution
matching the Linux compiler. It must contain:

- `clang_rt.asan_dynamic-x86_64.dll`
- `clang_rt.asan_dynamic-x86_64.lib`
- `clang_rt.asan_dynamic_runtime_thunk-x86_64.lib`
- `clang_rt.asan_static_runtime_thunk-x86_64.lib`

The SDK's `VC/Tools/MSVC/<version>/lib/x64` directory must also contain
`stl_asan.lib` for instrumented C++ standard library containers. Microsoft ships
it in the matching Visual Studio ASAN component; xwin's desktop CRT package may
omit it. Native Windows obtains sanitizer artifacts from the selected LLVM or
Visual Studio installation automatically. Both dynamic and static CRT builds
deploy the sanitizer DLL beside each instrumented executable. Bazel supplies
these runtimes to library and executable link actions, including third-party
`cc_library`, `cc_binary`, and `cc_test` targets.

### macOS targets

Native builds use the Xcode or Command Line Tools selected by `xcode-select`
or `DEVELOPER_DIR`. They ignore the Linux cross-toolchain environment variables.
Linux builds use Clang, `ld64.lld`, and `llvm-libtool-darwin` from `LLVM_ROOT`.
Both paths support C, C++, Objective-C, Objective-C++, static libraries, and
Mach-O executables and dylibs.
The default minimum deployment version is macOS 11.0, independent of the SDK
version; the `macos_toolchain_repository` declaration owns that policy.

Export an SDK on a Mac with Xcode or Command Line Tools installed:

```bash
python3 build_tools/macos/export_sdk.py --output macos-sdk.tar.gz
```

The archive preserves framework aliases and includes the matching libc++
headers, including Xcode versions that store them outside the SDK. Transfer
and unpack it into a versioned local installation directory on Linux, then set:

```bash
export LLVM_ROOT=/path/to/llvm
export MACOS_SDK_ROOT=/path/to/macos-sdk/MacOSX.sdk
iree-bazel-build --config=macos-arm64 //tools:iree-dump-cpuinfo
iree-bazel-build --config=macos-x86_64 -c opt \
  //runtime/src/iree/base/testing:dynamic_library_test
```

These paths can also be supplied with `--repo_env=LLVM_ROOT=...` and
`--repo_env=MACOS_SDK_ROOT=...`.

Framework consumers use ordinary dependencies such as
`@iree_macos_toolchain//:Foundation` and `@iree_macos_toolchain//:Metal`.
These provide public headers and framework link interfaces for the selected SDK.
`objc_library` from `@rules_cc//cc:objc_library.bzl` handles `.m` and `.mm`
sources, including ARC; its `CcInfo` also works with normal C/C++ rules.
Framework dependencies carry the SDK files as well as link flags, so consumers
do not need separate `sdk_frameworks` entries. The repository currently exposes
CoreFoundation, CoreGraphics, Foundation, and Metal. Adding another framework
to its explicit framework list inventories that framework's dependency closure.

Compiler inputs include system headers and Clang resource headers. Framework
headers enter through their dependencies; linker inputs include the SDK stubs
actually opened by the selected linker, including re-exports. The full SDK and
LLVM installation are not compiler or linker inputs. Both native and cross
macOS actions use paths relative to the execution root, and the tool projection
includes non-system loader libraries.

The integration test embeds MSL with a host-built generator, compiles C and both
Objective-C language modes, and dispatches and verifies a Metal compute kernel:

```bash
# On Linux, build and transfer the executable to a Mac for execution.
iree-bazel-build --config=macos-arm64 //build_tools/macos/tests:metal_test

# On a Mac with a Metal device, build and execute natively.
iree-bazel-test //build_tools/macos/tests:metal_test
```

Metal compiles this source through its runtime API; no offline Apple shader
compiler is required. Command-line programs using `MTLCreateSystemDefaultDevice`
also link CoreGraphics, as required by
[Apple's API contract](https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice%28%29?language=objc).
Metal compute execution works over SSH without a desktop login.

The toolchain adds `@loader_path` to the runtime library search path so deployed
dylibs can sit beside their consumer. Preserve the library names in the
consumer's load commands (`llvm-otool -L`), including Bazel's `_solib` names.
Debug builds retain object debug information; LLVM's `dsymutil` can collect it
into a `.dSYM` bundle before deployment.

Native AddressSanitizer uses the selected Xcode runtime with `--config=asan`.
For Linux cross builds, `MACOS_COMPILER_RT_ROOT` can supply the matching LLVM
`lib/clang/<version>/lib/darwin` directory, including
`libclang_rt.asan_osx_dynamic.dylib`. The runtime must match the compiler and
support the destination architecture; the SDK itself does not provide it.
Missing sanitizer artifacts fail during analysis. Bazel includes the runtime
in native test runfiles; deploy the dylib beside cross-built executables.

### Wrapper arguments

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
remain incompatible through their declared Bazel constraints. The lane does
not override GCC's linker selection; Bazel uses the GNU binutils provided by
the system toolchain. The copyable build-shape command is:

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
| `IREE_HAL_DRIVER_TASK` | `ON`, `OFF` | Builds the task runtime HAL driver. | Adds or removes `task` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_VULKAN` | `ON`, `OFF` | Builds the Vulkan runtime HAL driver. | Adds or removes `vulkan` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_WEBGPU` | `ON`, `OFF` | Builds the WebGPU runtime HAL driver. | Adds or removes `webgpu` from the runtime driver registry and recursive package scope for WebGPU development. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_DEPENDENCY_MODE` | `pinned`, `package`, `auto` | Selects locked source archives, package discovery, or package-then-pinned dependency resolution. | Writes `--repo_env=IREE_DEPENDENCY_MODE=<mode>`. | `--repo_env=IREE_DEPENDENCY_MODE=<mode>` |
| `IREE_ROCM_DEPENDENCY_MODE` | `pinned`, `package`, `auto` | Overrides dependency resolution for ROCm header facades; empty uses package mode when `IREE_ROCM_PATH` is set and otherwise inherits `IREE_DEPENDENCY_MODE`. | Writes `--repo_env=IREE_ROCM_DEPENDENCY_MODE=<mode>`. | `--repo_env=IREE_ROCM_DEPENDENCY_MODE=<mode>` |
| `IREE_ROCM_PATH` | path | Prepends the ROCm or TheRock SDK root to `CMAKE_PREFIX_PATH`, uses it for AMDGPU device tooling, and selects ROCm package header mode by default. | Writes `--repo_env=IREE_ROCM_PATH=<path>` and `--repo_env=IREE_ROCM_DEPENDENCY_MODE=package` unless explicitly overridden. | `--repo_env=IREE_ROCM_PATH=<path>` |

The Bazel native driver flag is a complete list. Include every driver you want
enabled:

```bash
python dev.py bazel configure \
  --//runtime/config/hal:drivers=amdgpu,task \
  --repo_env=IREE_ROCM_PATH=/opt/rocm \
  --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned
```

The portable spelling is shorter for common cases:

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
```

### External HAL drivers

External HAL drivers remain ordinary static C/C++ libraries. Each driver
publishes one C header and registration function:

```c
iree_status_t acme_hal_driver_module_register(
    iree_hal_driver_registry_t* registry);
```

The root module resolves HRX and any independently distributed driver modules
through normal Bzlmod dependencies or overrides:

```starlark
# MODULE.bazel
bazel_dep(name = "hrx", version = "<version>")
bazel_dep(name = "acme_hal", version = "<version>")
```

The driver module packages its library with registration metadata:

```starlark
load("@hrx//runtime/build_tools/bazel:hal_driver.bzl", "iree_hal_driver_module")

iree_hal_driver_module(
    name = "module",
    registration = ":registration",
    registration_header = "acme/hal/registration.h",
    registration_function = "acme_hal_driver_module_register",
    visibility = ["//visibility:public"],
)
```

The root application explicitly composes the modules it wants. Dependencies do
not register drivers merely because they appear in the transitive Bzlmod graph:

```starlark
load("@hrx//runtime/build_tools/bazel:hal_driver.bzl", "iree_hal_driver_registry")

iree_hal_driver_registry(
    name = "external_drivers",
    modules = [
        "@acme_hal//:module",
    ],
)
```

Select that aggregate for every binary using HRX's available-driver registry:

```text
build --@hrx//runtime/config/hal:external_driver_registry=//hal:external_drivers
```

The generated registry directly references each registration function, keeping
the selected static libraries live without constructors, linker sections, or
`alwayslink`. Modules register in list order after the built-in drivers. Since
the HAL registry resolves factories in most-recently-added order, later external
modules may deliberately override earlier modules or built-in driver names.

CMake projects declare the same target, header, and function contract before
adding HRX. A source directory is evaluated only when its name appears in
`IREE_EXTERNAL_HAL_DRIVERS`:

```cmake
include("${HRX_SOURCE_DIR}/runtime/build_tools/cmake/iree_external_hal_driver.cmake")
iree_register_external_hal_driver(
  NAME acme
  TARGET acme_hal_registration
  HEADER acme/hal/registration.h
  REGISTER acme_hal_driver_module_register
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acme_hal"
)
set(IREE_EXTERNAL_HAL_DRIVERS acme CACHE STRING "")
add_subdirectory("${HRX_SOURCE_DIR}" hrx)
```

Selecting an unknown driver, omitting its target, or failing to configure its
dependencies is a configuration error rather than an optional fallback.

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

Loom target options describe product compiler capability: `LOOM_TARGET_AMDGPU=ON`
means Loom can compile for AMDGPU, including the target architecture metadata and
canonical artifact formats registered for that target family. Runtime execution
remains a separate concern controlled by Loom execution support and the runtime
`IREE_HAL_DRIVER_*` options.

The default dependency-satisfied Loom target set is
`amdgpu,llvmir,spirv,x86`. AMDGPU and SPIR-V target compilation use pinned
source dependencies by default and do not enable the matching runtime HAL
drivers. WebAssembly remains opt-in until the WASI SDK repository is available
in this checkout. The default execution substrate set is `iree_hal`; HAL
execution providers still require a matching runtime HAL driver such as
`IREE_HAL_DRIVER_VULKAN` or `IREE_HAL_DRIVER_AMDGPU`.

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
| `LOOM_TARGET_SPIRV` | `ON`, `OFF` | Builds Loom SPIR-V target support and production SPIR-V emission. | Adds or removes `spirv` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_WASM` | `ON`, `OFF` | Builds Loom WebAssembly target support and production Wasm emission. | Adds or removes `wasm` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_X86` | `ON`, `OFF` | Builds Loom x86 target support. | Adds or removes `x86` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_EMIT_LLVMIR` | `ON`, `OFF` | Builds LLVM IR debug/developer emission for enabled target archs. | Adds or removes `llvmir` from the explicit Loom emitter set. | `--//loom/config/emit:enable=<complete-emitter-list>` |
| `LOOM_EXECUTE_IREE_HAL` | `ON`, `OFF` | Builds Loom execution providers that run through IREE HAL when a matching runtime HAL driver is enabled. | Adds or removes `iree_hal` from the Loom execute substrate set. | `--//loom/config/execute:enable=<complete-execute-list>` |

The native Loom target flag is a complete list. The default target set is
`amdgpu,llvmir,spirv,x86`, and the default execution substrate set is
`iree_hal`:

```bash
python dev.py bazel configure \
  --//loom/config/target:enable=amdgpu,llvmir,spirv,x86
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
compilation target by themselves. For example, AMDGPU execution needs the
AMDGPU Loom target, the IREE HAL execution substrate, and the AMDGPU runtime
HAL driver:

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
C++ tools and a Windows SDK, and Git for Windows. Bazel uses the Bash supplied
by Git for Windows; the developer wrapper discovers it automatically. The
managed developer environment installs Ninja and the CI-pinned CMake 3.31.6.
Install LLVM separately when building with `clang-cl`.

Start from an x64 Visual Studio developer shell so `INCLUDE`, `LIB`, the SDK
tools, and the MSVC linker are available. Git for Windows also ships a Unix
program named `link.exe`, so compiler activation order is load-bearing:

```powershell
where.exe cl
where.exe link
```

The first `link.exe` must be the MSVC linker, not Git's `usr\bin\link.exe`.
Create the repository tool environment and check it:

```powershell
python dev.py cmake setup --venv
python dev.py cmake doctor
```

Setup, configure, build, test, and doctor also work from an exported source
snapshot. Setup and doctor omit Git-specific checks when `.git` is absent;
installing hooks and running Git-based presubmit require a checkout. This lets
a Windows build host consume source synchronized from another machine while
Git operations remain on the source machine.

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

Use `--config=windows-msvc` when checking the MSVC compiler. It changes the
Windows compiler choice while preserving the configured feature and dependency
graph. `--config=windows-clang-cl` explicitly selects the default compiler.
Neither native command needs `--config=windows-x86_64`:

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
