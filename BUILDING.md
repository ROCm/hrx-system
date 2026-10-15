# Building

Bazel is the source-tree build system and the source of truth for generated
CMake build files. CMake supports package installation and embedding in other
projects. `dev.py` prepares the developer tools and delegates to either build
system; both can also be used directly.

- [Prerequisites and quick start](#quick-start)
- [Build and test commands](#build-and-test-commands)
- [Build configuration](#build-configuration)
- [Platform builds and cross-compilation](#platform-builds-and-cross-compilation)
- [Shared project configuration](#shared-project-configuration)
- [Installation and embedding](#installation-and-embedding)
- [Troubleshooting](#troubleshooting)

## Quick Start

Install Python 3.12 with `venv` support and a C/C++ compiler, linker, and system
headers. The base build uses C11 and C++20. Linux supports Clang and GCC;
macOS uses Xcode or Command Line Tools. Windows requires the [MSVC ABI tools
and Windows SDK](#native-windows), including when compiling with clang-cl.
The optional [C/C++ importer](loom/src/loom/import/cxx/README.md) additionally
requires a C++23 compiler and standard library with `std::expected` support.

Run the following commands from the repository root. Setup creates a managed
`.venv` with the pinned build tools; it does not install a native compiler or
platform SDK. These examples build Loom and exercise CPU tests without
requiring GPU or NPU hardware.

### Bazel

```bash
python dev.py bazel setup
python dev.py bazel configure
python dev.py bazel build //loom/src/loom/tools/loom-compile
python dev.py bazel test //runtime/src/iree/base/... --config=asan
```

Bazel does not require CMake. Generated CMake files are checked in, so neither
build requires a manual Bazel-to-CMake conversion.

### CMake

```bash
python dev.py cmake setup
python dev.py cmake configure -DLIBHRX_BUILD=OFF
python dev.py cmake build loom-compile
python dev.py cmake test -R '^iree/base/'
```

`LIBHRX_BUILD=OFF` selects a runtime/compiler build. Building the HRX GPU
runtime instead uses `LIBHRX_BUILD=ON` and its AMDGPU dependencies; see
[ROCm configuration](#rocm) and the [HRX package build](README.md#building).

The default build tree is `build/cmake`. Select another with
`python dev.py --cmake-build-dir /path/to/build cmake configure ...`; subsequent
wrapper commands remember that tree. New trees default to Ninja and a Release
build. Existing trees retain their configuration, and explicit `-G`,
`CMAKE_GENERATOR`, and preset selections are honored. System-tool builds need
CMake 3.26 or newer; [shared Loom tools](#shared-loom-tool-bundles) need 3.27.

### Tool environment

Activate `.venv` to put the managed tools and `iree-bazel-*` / `iree-cmake-*`
aliases on `PATH`:

```bash
source .venv/bin/activate
```

In PowerShell, use `.venv\Scripts\Activate.ps1`. Activation is optional when
invoking `python dev.py`, which locates the managed environment itself.

An embedding project can use `python dev.py bazel setup --system` to check
already-installed tools, or `--tool-root /path/to/tools` to keep a managed
environment outside the source tree. The same options work for CMake.
[Contributing](CONTRIBUTING.md) covers Git hooks and development checks;
source archive builds do not need Git hooks.

## Build and Test Commands

Put wrapper execution and tool-environment options before the build-system
command:

```bash
python dev.py --dry-run bazel build //runtime/...
python dev.py --system bazel configure
python dev.py --verbose cmake test -R hrx
```

Put build-system options and target names after `<build-system> <command>`:

```bash
python dev.py bazel build //runtime/...
python dev.py cmake configure -DCMAKE_BUILD_TYPE=Debug
python dev.py cmake build hrx --parallel 8
python dev.py cmake test -R hrx
```

After setup and environment activation, aliases use the same arguments:

```bash
iree-bazel-build //runtime/...
iree-bazel-cquery 'kind(cc_library, //runtime/...)'
iree-bazel-info execution_root
iree-bazel-run //runtime/src/iree/base:allocator_benchmark
iree-cmake-configure -DIREE_HAL_DRIVER_AMDGPU=ON
iree-cmake-test -R hrx
```

CMake build and run commands also accept a unique executable output name, such
as `loom-compile`. If two packages emit the same filename, use the qualified
generated alias so the selection remains explicit.

`iree-cmake-test` builds the selected tests and their dependencies before
running them. A filtered test run therefore needs no separate matching
`iree-cmake-build` invocation; source-only tests need no build.

`iree-bazel-run`, `iree-bazel-try`, and `iree-bazel-fuzz` build first and then
run the resolved executable directly, so Bazel does not hold its server lock
while a benchmark, tool, or fuzzer is running.

`iree-bazel-build` and `iree-bazel-test` without targets select `//runtime/...`
and `//libhrx/...`. Name `//loom/...` or another package explicitly to include it.
`iree-bazel-test` builds the selected tests before running them. Hardware tests
need the matching driver, device access, and runtime libraries; compiling a
backend does not provide those execution dependencies.

Bazel outputs are available under `bazel-bin/`; test logs are under
`bazel-testlogs/`. CMake places outputs in the selected build tree. Use
`iree-bazel-run TARGET -- ARGS...` or `iree-cmake-run TARGET -- ARGS...` to
build and run a tool with its runtime dependencies.

### Using Bazel and CMake directly

The wrappers select the tool environment and forward build arguments. With the
same tools on `PATH`, the basic build commands are equivalent:

```bash
python dev.py bazel build //runtime/src/iree/base/...
bazel build //runtime/src/iree/base/...

python dev.py cmake configure -DLIBHRX_BUILD=OFF
cmake -S . -B build/cmake -GNinja -DLIBHRX_BUILD=OFF

python dev.py cmake build loom-compile
cmake --build build/cmake --target loom_tools_loom-compile_loom-compile
```

The CMake wrapper resolves unique executable names such as `loom-compile` to
their generated target names. Raw CTest runs tests that are already built;
`iree-cmake-test` also builds the selected tests and their dependencies. After
a full `cmake --build build/cmake`, run them directly with:

```bash
ctest --test-dir build/cmake --output-on-failure -R '^iree/base/'
```

## Build Configuration

### Optimized builds

Ordinary Bazel commands use `fastbuild` without link-time optimization. This is
the intended mode for edit/build/test iteration. Use the checked-in optimized
profile when producing release artifacts or performance measurements:

```bash
iree-bazel-build --config=opt //loom/src/loom/tools/loom-compile
```

`--config=opt` selects optimized compilation for target and host tools and
ThinLTO for target links. Windows ThinLTO requires clang-cl. Native MSVC users
can select `-c opt` for optimization without ThinLTO.

The profile excludes `-march=native`, which would select the build machine's
CPU and could make the output unusable on another host. Select an explicit
architecture when producing binaries for a particular deployment CPU.

CMake uses its native optimization settings:

```bash
iree-cmake-configure -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

This enables interprocedural optimization for targets throughout the build,
except components with explicit exclusions such as the mimalloc adapter.

### Sanitizers

Bazel selects sanitizer instrumentation with named configurations:

| Config | Meaning |
| --- | --- |
| `--config=asan` | AddressSanitizer with use-after-scope checks. |
| `--config=ubsan` | UndefinedBehaviorSanitizer with the function and vptr checks disabled to match the runtime's type-erased dispatch and `-fno-rtti` C++ mode. |
| `--config=tsan` | ThreadSanitizer. |
| `--config=msan` | MemorySanitizer; execution also requires instrumented native dependencies. |
| `--config=fuzzer` | libFuzzer build mode with ASAN enabled. |

Examples:

```bash
python dev.py bazel test //runtime/... --config=asan
python dev.py bazel test //runtime/src/iree/async/... --config=tsan
```

CMake exposes `IREE_ENABLE_ASAN`, `IREE_ENABLE_UBSAN`, `IREE_ENABLE_TSAN`,
and `IREE_ENABLE_MSAN`. Use a separate build tree for each configuration:

```bash
python dev.py --cmake-build-dir build/asan cmake configure \
  -DLIBHRX_BUILD=OFF -DIREE_ENABLE_ASAN=ON
python dev.py --cmake-build-dir build/asan cmake test -R '^iree/base/'
```

Availability depends on the selected compiler and platform. Native Windows
clang-cl supports ASAN; cross-built sanitizer runtimes are described in the
[Windows](#windows-from-linux) and [macOS](#macos) sections.

Tests and tools inherit sanitizer suppressions from the runtime libraries they
use, including combined sets when they use multiple APIs. Bazel run/test, the
`iree-bazel-*` launch wrappers, and CTest supply the required files and
environment. Launching a raw executable outside these wrappers requires setting
up that sanitizer environment separately.

### Shared Loom tool bundles

Loom tools link statically by default. Release bundles can share the compiler
and runtime used by `loom-compile`, `loom-opt`, and `loom-import-cxx`:

```bash
iree-bazel-build --config=opt --//loom/config:shared_tools \
  //loom/src/loom/tools/loom-compile \
  //loom/src/loom/tools/loom-opt \
  //loom/src/loom/tools/loom-serve
iree-cmake-configure -DLOOM_LINK_SHARED_TOOLS=ON
```

The shared mode packages one `libloomc` and one `libiree` alongside the compiler
tools. `loom-serve` keeps its compiler and runtime statically linked in either
mode, retaining whole-program optimization and standalone deployment.
The IREE companion includes the configured HAL drivers. Deploy all files from
the same build; these companion libraries have no independent ABI version.
Shared compiler tools trade some cross-library optimization for a smaller
bundle. Local development and benchmarks retain the default static mode.

CMake shared tools require CMake 3.27 or newer and static component libraries
(`BUILD_SHARED_LIBS=OFF`). This option controls final tool composition separately
from the component sharing used for in-tree development. Each selected tool's
install component includes both libraries, with relative loader paths for
`bin/` and `lib/` installs or a single-directory bundle. The installed LoomC SDK
exports the same configured bindings in either mode.
Windows CMake builds place the shared tools and both DLLs together under the
build tree's `bin/` directory so the tools also run directly from the build tree.

### Python runtime symbols

Bazel's Linux x86-64 glibc Python runtime omits interpreter C debug information
by default. Python source debugging and tracebacks are unaffected. To debug
inside CPython, select the matching runtime with native debug symbols:

```bash
iree-bazel-build --config=python-runtime-debug //loom/py/loom/tools:loom-compile-report
```

This composes with `--config=asan` and `--config=opt` without changing project
optimization or C/C++ debug information.

### Default allocator

IREE uses libc for `iree_allocator_system()` by default. Native builds can
select the repository-pinned mimalloc provider for the shared default:

```bash
iree-bazel-build --config=mimalloc //runtime/src/iree/base:allocator_mimalloc_test
iree-cmake-configure -DIREE_ALLOCATOR_SYSTEM=mimalloc
```

Explicit allocator arguments retain their caller-selected implementation. The
mimalloc configuration composes with ASAN, MSAN, TSAN, and UBSAN. Sanitizer
instrumentation owns the process allocation boundary in those builds, so the
effective shared default remains libc and global C++ allocation is not replaced.

## Platform Builds and Cross-Compilation

### Destination and compiler selection

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

Destination configs select `--platforms` independently of the compiler. Native
builds infer their destination from the host, so adding a matching destination
config is optional.
For example, a native MSVC build needs only `--config=windows-msvc`; adding
`--config=windows-x86_64` produces the same selection in either flag order.
`windows-clang-cl` explicitly selects the default Windows compiler. MSVC requires
a Windows build host; Linux-to-Windows builds use clang-cl. macOS destinations
use Xcode/Command Line Tools on a Mac and LLVM on Linux.

Build-time generators retain the host toolchain and remain executable on the
build host. Native builds do not require either cross SDK installation. SDK
acquisition is separate from Bazel configuration; the sections below describe
the inputs for each destination.

Cross-built executables run on their destination OS. Transfer the executable,
dependent libraries, debug artifacts, and runfiles to that host. The Windows
and macOS destination presets allow `bazel build` to produce test artifacts
without a destination executor. `bazel test` still requires a matching execution
platform; a Linux host cannot execute Windows or macOS binaries itself.

Cross-compilation does not configure remote execution. Shared remote caching
and execution additionally need a compatible execution environment for the
selected host tools.

### Native Windows

Windows builds require an x64 MSVC ABI environment even when `clang-cl` is the
host compiler. Install Python 3.12, Visual Studio 2022 Build Tools with the x64
C++ tools and a Windows SDK, and Git for Windows. Bazel uses the Bash supplied
by Git for Windows; the developer wrapper discovers it automatically. The
managed developer environment installs Ninja and CMake. Install LLVM separately
when building with `clang-cl`.

Start from an x64 Visual Studio developer shell so `INCLUDE`, `LIB`, the SDK
tools, and the MSVC linker are available. Git for Windows also ships a Unix
program named `link.exe`, so check which executable comes first on `PATH`:

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

Keep source and build paths short, and use a separate CMake tree for each
compiler. Bazel's Python runfiles require symbolic links: enable Windows
Developer Mode or grant the account the Create symbolic links user right.
`python dev.py bazel configure` and `python dev.py bazel doctor` check that
the current process can create one. The Windows long-path policy can help tools
that support it, but does not remove path limits from every tool.

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

Use `--config=windows-msvc` to compile with MSVC. It changes the Windows
compiler choice while preserving the configured features and dependencies.
`--config=windows-clang-cl` explicitly selects the default compiler.
Neither native command needs `--config=windows-x86_64`:

```powershell
python dev.py bazel build `
  //loom/src/loom/tools/iree-run-loom `
  //loom/src/loom/tools/loom-compile `
  //loom/binding/c:loomc `
  --config=windows-msvc
```

For a fresh CMake build limited to Loom's VM and x86 compiler targets, use:

```powershell
$baseOptions = @(
  '-GNinja'
  '-DCMAKE_BUILD_TYPE=RelWithDebInfo'
  '-DLIBHRX_BUILD=OFF'
  '-DLOOM_TARGET_DEFAULTS=OFF'
  '-DLOOM_TARGET_VM=ON'
  '-DLOOM_TARGET_X86=ON'
)

$llvmBin = 'C:\Program Files\LLVM\bin'
$env:PATH = "$llvmBin;$env:PATH"
$env:CC = "$llvmBin\clang-cl.exe"
$env:CXX = "$llvmBin\clang-cl.exe"
$env:AR = "$llvmBin\llvm-lib.exe"
python dev.py --cmake-build-dir C:\b\hrx-clang cmake configure @baseOptions
python dev.py --cmake-build-dir C:\b\hrx-clang cmake build loom-compile --parallel 8
python dev.py --cmake-build-dir C:\b\hrx-clang cmake test -R '^iree/base/' -j 8
```

To use MSVC, select its tools and configure a separate build tree. CMake caches
the compiler selection, so changing shell variables does not switch an already
configured tree:

```powershell
$env:CC = 'cl.exe'
$env:CXX = 'cl.exe'
$env:AR = 'lib.exe'
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake configure @baseOptions
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake build loom-compile --parallel 8
python dev.py --cmake-build-dir C:\b\hrx-msvc cmake test -R '^iree/base/' -j 8
```

### Windows from Linux

Provide a Linux LLVM installation with clang-cl, lld-link, llvm-ml, and the
x86-64 Windows compiler runtime, plus a Windows SDK/MSVC sysroot. The LLVM
resource directory must contain
`lib/windows/clang_rt.builtins-x86_64.lib`; the compiler can emit calls to
these helpers for ordinary C and C++ operations. The Bazel repository reads
`LLVM_ROOT` and `WINSDK_ROOT` from the environment or explicit `--repo_env`
options; acquisition is independent of the build configuration.
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
need the sanitizer DLL beside each instrumented executable.

### macOS

Native builds use the Xcode or Command Line Tools selected by `xcode-select`
or `DEVELOPER_DIR`. They ignore the Linux cross-toolchain environment variables.
Linux builds use Clang, `ld64.lld`, and `llvm-libtool-darwin` from `LLVM_ROOT`.
Both paths support C, C++, Objective-C, Objective-C++, static libraries, and
Mach-O executables and dylibs.
The default minimum deployment version is macOS 11.0, independent of the SDK
version.

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

The toolchain exposes framework dependencies such as
`@iree_macos_toolchain//:Foundation` and `@iree_macos_toolchain//:Metal`, plus
CoreFoundation and CoreGraphics. These supply both headers and link interfaces.
Use `objc_library` from `@rules_cc//cc:objc_library.bzl` for `.m` and `.mm`
sources, including ARC.

Metal programs can compile MSL through the runtime API, so they do not require
an offline Apple shader compiler. To build the Metal compute example on Linux
and transfer it to a Mac:

```bash
iree-bazel-build --config=macos-arm64 //build_tools/macos/tests:metal_test
```

On a Mac with a Metal device, build and run it with:

```bash
iree-bazel-test //build_tools/macos/tests:metal_test
```

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

## Shared Project Configuration

Shared project options use CMake-style `-DNAME=VALUE` spelling with
`iree-bazel-configure` or `iree-cmake-configure`. Only the options documented as
portable have a Bazel translation.

Bazel configure regenerates `.bazelrc.configured` from defaults and the supplied
options: include all desired overrides in the same invocation. CMake updates
its existing cache. Bazel-native overrides not handled by configure belong in
`.bazelrc.local` or on an individual build command.

### Runtime libraries and drivers

| Option | Values | CMake | Bazel portable | Bazel native |
| --- | --- | --- | --- | --- |
| `AMDF_BUILD` | `ON`, `OFF` | Enables the portable libamdf shared/static libraries, examples, and tests. Defaults to `ON` when the AMD HAL driver requires it and `OFF` otherwise. | Enables or disables the libamdf package scope. The portable AMD HAL selection enables it automatically. | `--//libamdf/config:enabled=<bool>` |
| `AMDF_FAMILY_RDNA` | `ON`, `OFF` | Admits RDNA implementation packages to libamdf. | Adds or removes `rdna` from the libamdf implementation-family scope. | `--//libamdf/config:families=<complete-family-list>` |
| `AMDF_FAMILY_CDNA` | `ON`, `OFF` | Admits CDNA implementation packages to libamdf. | Adds or removes `cdna` from the libamdf implementation-family scope. | `--//libamdf/config:families=<complete-family-list>` |
| `AMDF_FAMILY_XDNA` | `ON`, `OFF` | Admits XDNA implementation packages to libamdf. | Adds or removes `xdna` from the libamdf implementation-family scope. | `--//libamdf/config:families=<complete-family-list>` |
| `IREE_ENABLE_VULKAN` | `ON`, `OFF` | Enables Vulkan API clients independently of HAL drivers. Defaults to `ON`; the Vulkan HAL also enables API availability. | Selects Vulkan API clients with the same default and HAL implication. | `--//build_tools/vulkan/config:enabled=<bool>` |
| `IREE_ENABLE_D3D12` | `ON`, `OFF` | Enables D3D12 API clients for Windows targets. Defaults to `ON`. | Selects D3D12 API clients for Windows targets, including cross-compilation. Defaults to `ON`. | `--//build_tools/d3d12/config:enabled=<bool>` |
| `IREE_NET_RDMA` | `ON`, `OFF` | Enables native RDMA resources and registration on Linux. Defaults to `OFF`. | Enables the same native RDMA packages; no host carrier is implicitly registered. Defaults to `OFF`. | `--//runtime/config/net:rdma=<bool>` |
| `IREE_HAL_DRIVER_AMDGPU` | `ON`, `OFF` | Builds the AMDGPU runtime HAL driver. | Adds or removes `amdgpu` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_TASK` | `ON`, `OFF` | Builds the task runtime HAL driver. | Adds or removes `task` from the runtime driver registry. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_VULKAN` | `ON`, `OFF` | Builds the Vulkan runtime HAL driver. | Adds or removes `vulkan` from the runtime driver registry and recursive package scope. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_WEBGPU` | `ON`, `OFF` | Builds the WebGPU runtime HAL driver. | Adds or removes `webgpu` from the runtime driver registry and recursive package scope for WebGPU development. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
| `IREE_HAL_DRIVER_AMD` | `ON`, `OFF` | Builds the shared AMD runtime HAL driver using libamdf. Native families are selected independently. | Adds or removes `amd` from the runtime driver registry build and enables libamdf when selected. Device schemes remain `rdna` and `xdna`. | `--//runtime/config/hal:drivers=<complete-driver-list>` |
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

libamdf can also be built independently of the legacy AMDGPU HAL and ROCm
dependency stack:

```bash
python dev.py bazel configure -DAMDF_BUILD=ON
iree-bazel-build //libamdf:all

iree-cmake-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF -DLOOM_BUILD=OFF
iree-cmake-build libamdf/all
```

The [libamdf test guide](libamdf/cts/README.md) describes device execution
requirements. Kernel-based tests also need Loom enabled to compile their test
programs; the library itself does not.

The native family setting is a complete list. For example, an XDNA-only source
configuration uses `--//libamdf/config:families=xdna`; the portable equivalent
sets `AMDF_FAMILY_RDNA=OFF` and `AMDF_FAMILY_CDNA=OFF`.

Vulkan and D3D12 API selections control client compilation separately from
device execution. They do not enable a HAL driver or shader compiler. Vulkan
HAL selection also makes Vulkan available, so excluding all Vulkan clients
requires disabling both requests. Running API tests also requires a loader,
driver, and compatible device.
See the [Vulkan client guide](build_tools/vulkan/README.md) and
[D3D12 client guide](build_tools/d3d12/README.md) for details.

### Dependency resolution

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
package mode; without a ROCm path it inherits the global mode. To consume a
ROCm or TheRock SDK while keeping other source dependencies pinned, set
`IREE_ROCM_PATH` and leave `IREE_DEPENDENCY_MODE=pinned`.

### ROCm

Pinned mode is the default. It lets AMDGPU host-side code compile without a
ROCm/TheRock root; Bazel writes `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none` when no
ROCm path is configured:

```bash
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON
```

Package mode consumes headers from a configured ROCm/TheRock root. Setting
`IREE_ROCM_PATH` selects ROCm package header mode by default:

```bash
python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_PATH=/opt/rocm
```

When a ROCm path is only needed for device tooling and the headers should still
come from pinned sources, set `IREE_ROCM_DEPENDENCY_MODE=pinned` explicitly.

Bazel configuration also accepts `IREE_ROCM_PATH` from the environment when
the AMDGPU HAL is enabled:

```bash
IREE_ROCM_PATH=/opt/rocm python dev.py bazel configure \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned
```

Host compiler selection is independent of the ROCm device toolchain. Set
`CC`, `CXX`, and `AR` before configuring to choose the host compiler. A
successful host build with pinned headers does not install ROCr; executing
ROCr-backed AMDGPU programs still needs the runtime and access to a GPU.
See the [AMDGPU build tools guide](build_tools/amdgpu/README.md) for device
compiler selection and runtime target selectors.

### Loom compiler targets and execution

Loom target options select compiler backends: `LOOM_TARGET_AMDGPU=ON` enables
compilation for AMDGPU. Executing the compiled programs is controlled separately
by Loom execution support and the runtime `IREE_HAL_DRIVER_*` options.

The default dependency-satisfied Loom target set is
`amdgpu,spirv,vm,xdna,x86`. VM compilation and function execution provide the
standard evaluator for `check.case` and `check.benchmark`. AMDGPU and SPIR-V
target compilation use pinned source dependencies by default and do not enable
the matching runtime HAL drivers. WebAssembly is opt-in. The default execution
substrate is `iree_hal`; execution still needs a
matching runtime HAL driver such as `IREE_HAL_DRIVER_VULKAN` or
`IREE_HAL_DRIVER_AMDGPU`.

CMake exposes `LOOM_TARGET_DEFAULTS` and `LOOM_EXECUTE_DEFAULTS` to set the
default value for dependency-satisfied target and execution options before the
individual `LOOM_TARGET_*` and `LOOM_EXECUTE_*` overrides are evaluated. Bazel
configuration writes complete native lists instead, so portable `-D...=OFF`
options remove entries from the default set.

AMDGPU has an additional compiler target selector list. `LOOM_TARGET_AMDGPU=ON`
enables the backend; `LOOM_TARGET_AMDGPU_TARGETS` selects which AMDGPU
processors it supports. The default selector is `loom_defaults`, which includes
all of the following:

| Architecture family | Processor targets |
| --- | --- |
| CDNA 2 | `gfx90a` |
| CDNA 3 | `gfx940`, `gfx941`, `gfx942` |
| CDNA 4 | `gfx950` |
| Generic CDNA 3/4 | `gfx9-4-generic` |
| RDNA 3 | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103` |
| RDNA 3.5 | `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153` |
| Generic GFX11 | `gfx11-generic` |
| RDNA 4m | `gfx1170`, `gfx1171`, `gfx1172` |
| RDNA 4 | `gfx1200`, `gfx1201` |
| Generic GFX12 | `gfx12-generic` |
| GFX12.5 | `gfx1250`, `gfx1251` |
| Generic GFX12.5 | `gfx12-5-generic` |

Generic targets compile code for a compatible group of processors. Exact
processors allow target-specific features. `gfx1170`–`gfx1172` require exact
target selection; no generic selector covers them in the pinned toolchain.

The Loom AMDGPU selector accepts:

- Source selectors: `loom_defaults`, `iree_hal`.
- Exact processors listed in the table above.
- Generic compiler targets: `gfx9-4-generic`, `gfx11-generic`,
  `gfx12-generic`, `gfx12-5-generic`.
- Fully covered family selectors: `gfx90a-dcgpu`, `gfx94X-all`, `gfx94X-dcgpu`,
  `gfx950-all`, `gfx950-dcgpu`, `gfx110X-all`, `gfx110X-dgpu`,
  `gfx110X-igpu`, `gfx115X-all`, `gfx115X-igpu`, `gfx117X-all`,
  `gfx120X-all`, `gfx125X-all`.

Runtime AMDGPU selectors such as `gfx9-generic`, `gfx908`,
`gfx10-1-generic`, and `gfx10-3-generic` are not supported by the Loom compiler.
The `iree_hal` selector narrows Loom to the supported subset requested by
`IREE_HAL_AMDGPU_TARGETS`. Use it for a JIT build targeting the same processors
as the runtime; `loom_defaults` keeps the compiler's full target set.

| Option | Values | CMake | Bazel portable | Bazel native |
| --- | --- | --- | --- | --- |
| `LOOM_TARGET_AMDGPU` | `ON`, `OFF` | Builds Loom AMDGPU target support and production AMDGPU emission. | Adds or removes `amdgpu` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_AMDGPU_TARGETS` | AMDGPU selectors | Selects descriptor-backed AMDGPU processors compiled into Loom AMDGPU target support. | Not exposed as a portable `-D` option. | `--//loom/config/target/amdgpu:targets=<complete-selector-list>` |
| `LOOM_TARGET_SPIRV` | `ON`, `OFF` | Builds Loom SPIR-V target support and production SPIR-V emission. | Adds or removes `spirv` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_VM` | `ON`, `OFF` | Builds Loom VM compilation and function execution. | Adds or removes `vm` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_WASM` | `ON`, `OFF` | Builds Loom WebAssembly target support and production Wasm emission. | Adds or removes `wasm` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_XDNA` | `ON`, `OFF` | Builds Loom XDNA target support and production XDNA emission. | Adds or removes `xdna` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_TARGET_X86` | `ON`, `OFF` | Builds Loom x86 target support. | Adds or removes `x86` from the Loom target product set. | `--//loom/config/target:enable=<complete-target-list>` |
| `LOOM_EXECUTE_IREE_HAL` | `ON`, `OFF` | Builds Loom execution providers that run through IREE HAL when a matching runtime HAL driver is enabled. | Adds or removes `iree_hal` from the Loom execute substrate set. | `--//loom/config/execute:enable=<complete-execute-list>` |

The native Loom target flag is a complete list. The default target set is
`amdgpu,spirv,vm,xdna,x86`, and the default execution substrate set is
`iree_hal`:

```bash
python dev.py bazel configure \
  --//loom/config/target:enable=amdgpu,spirv,vm,xdna,x86
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

To select Loom targets from the runtime's processor set:

```bash
python dev.py cmake configure \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=iree_hal \
  -DIREE_HAL_AMDGPU_TARGETS='gfx942;gfx1201'
```

To disable one of the default compiler backends:

```bash
python dev.py bazel configure -DLOOM_TARGET_SPIRV=OFF
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
  --//runtime/src/iree/hal/drivers/amdgpu:targets=gfx942 \
  --//loom/config/target/amdgpu:targets=gfx942 \
  -DIREE_ROCM_PATH=/opt/rocm
```

Compiling for AMDGPU does not require a GPU. Enable the runtime HAL only when
execution support is needed. Advanced embedders can select architecture,
emitter, and execution slices separately through `//loom/config/target/arch`,
`//loom/config/emit`, and `//loom/config/execute`; ordinary builds use the
`LOOM_TARGET_*` options above.

### Loom importers

Importers are optional. The native [C/C++ importer](loom/src/loom/import/cxx/README.md)
uses `--config=loom-importer-cxx` with Bazel or `-DLOOM_IMPORT_CXX=ON` with
CMake. The [Python importer guide](loom/py/loom/importers/README.md) covers
TileLang and MLIR, including their separate managed Python environments:

```bash
python dev.py importers setup mlir
iree-bazel-test --importer-env mlir //loom/py/loom/importers/mlir:mlir_test
```

### Serving image inputs

Loom serving image input is enabled by default on Linux, macOS, and Windows.
PAM images have a built-in decoder with no third-party dependencies. PNG, JPEG,
GIF, and WebP have independent providers, all enabled by default. A minimal
build can retain image input with only PAM, select optional codecs, or disable
image input entirely:

```bash
iree-bazel-configure -DLOOM_SERVE_IMAGE_CODECS=
iree-cmake-configure -DLOOM_SERVE_IMAGE_CODECS=

iree-bazel-configure '-DLOOM_SERVE_IMAGE_CODECS=png;jpeg'
iree-cmake-configure '-DLOOM_SERVE_IMAGE_CODECS=png;jpeg'

iree-bazel-configure -DLOOM_SERVE_IMAGE_INPUT=OFF
iree-cmake-configure -DLOOM_SERVE_IMAGE_INPUT=OFF
```

The native Bazel options are `--//loom/config/serve:image_input=false` and
`--//loom/config/serve:image_codecs=png,jpeg`; an empty codec list retains PAM.
Only selected optional providers enter the source-fetch and link graphs.
Disabled formats return an explicit error; other serving modalities do not
depend on codec availability. Other operating systems exclude image providers
by default.

### Serving audio inputs

WAV, MP3, and FLAC transcription inputs are enabled by default on Linux, macOS,
and Windows. They use the pinned miniaudio decoder; no system audio library is
required.

One codec list selects a subset or removes the dependency entirely:

```bash
iree-bazel-configure '-DLOOM_SERVE_AUDIO_CODECS=wav;flac'
iree-cmake-configure '-DLOOM_SERVE_AUDIO_CODECS=wav;flac'

iree-bazel-configure -DLOOM_SERVE_AUDIO_CODECS=
iree-cmake-configure -DLOOM_SERVE_AUDIO_CODECS=
```

The native Bazel spelling is `--//loom/config/serve:audio_codecs=wav,flac`.
An empty list excludes miniaudio from the server's configured dependency graph;
raw PCM16LE transcription and PCM speech synthesis remain available. Disabled
encoded formats return an explicit error. This option selects input decoding;
it does not add compressed speech output formats. Other operating systems
exclude encoded-audio providers by default.

### CMake project selection

CMake has project availability options because it configures a package build
tree. Bazel project availability is currently expressed by target selection.

| Option | Default | Build system | Meaning |
| --- | --- | --- | --- |
| `LIBHRX_BUILD` | `ON` | CMake | Builds libhrx and HRX compatibility targets. `ON` requires AMDGPU support; CMake enables `IREE_HAL_DRIVER_AMDGPU` by default when libhrx is built. |
| `LIBHRX_BUILD_CTS` | `ON` when libhrx and tests are enabled | CMake | Builds libhrx CTS binaries. |
| `LIBHRX_BUILD_PASSTHROUGH` | `ON` on Linux, `OFF` elsewhere | CMake | Builds HIP passthrough and interception developer tools. |
| `LOOM_BUILD` | `ON` | CMake | Builds Loom libraries, tools, bindings, and tests. |
| `IREE_BUILD_TESTS` | `ON` | CMake | Builds runtime tests and CTS targets. |
| `HRX_INSTALL_TESTS` | `${IREE_BUILD_TESTS}` | CMake | Installs a relocatable CTest tree. |

## Installation and Embedding

CMake installs into the prefix selected by `cmake --install`. Build the desired
products first. With the managed environment activated, a full install uses:

```bash
python dev.py cmake build
cmake --install build/cmake --prefix build/install
```

Replace `build/cmake` with the configured build tree. HRX's `HrxPublicDist` and
`HrxTestsDist` install components allow separate runtime and test packages;
see the [HRX package instructions](README.md#building). Public HRX packaging
uses `BUILD_SHARED_LIBS=OFF` so private runtime components are incorporated into
`libhrx`. Shared Loom tool bundles have their own
[linkage option](#shared-loom-tool-bundles).

An embedding CMake project can set project/dependency options before
`add_subdirectory`. Bazel consumers use normal Bzlmod dependencies or overrides.
External HAL drivers can be composed into either build as described below.

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

## Troubleshooting

Check the tools visible to the wrapper and print a command plan without running
it:

```bash
python dev.py bazel doctor
python dev.py cmake doctor
python dev.py --dry-run bazel build //loom/src/loom/tools/loom-compile
```

- **Tool environment cannot be reused:** setup requires Python 3.12. Preserve
  the old environment and choose a new `--tool-root` if it contains a different
  interpreter or incomplete installation.
- **Compiler or SDK is missing:** setup installs build utilities, not a native
  compiler. Check the selected compiler's system headers and SDK, and the
  [platform-specific requirements](#platform-builds-and-cross-compilation).
- **A requested Bazel target is incompatible:** check the enabled projects,
  compiler targets, and destination platform. Building a backend and running
  tests on its hardware are separate selections.
- **A Windows link selects the wrong linker:** in an x64 Visual Studio developer
  shell, `where.exe link` must list MSVC's linker before Git's `link.exe`.
- **A cross-built executable cannot start:** run it on the destination OS and
  deploy its runtime libraries and runfiles. ASAN builds also need the matching
  sanitizer runtime.

For a shared local Bazel cache and executor, see the optional
[NativeLink setup](build_tools/nativelink/README.md). Remote execution needs
compatible host tools and declared runtime dependencies; selecting a destination
config alone does not provide an executor.
