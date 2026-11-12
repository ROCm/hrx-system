# libhrx

libhrx exposes a native C API, C++ helpers, and a HIP compatibility binding over
the shared [runtime and HAL](../runtime/README.md). It provides device groups,
memory pools, queues, streams, executable loading, and synchronization for
applications using those APIs. Loom's compiler and serving components use the
shared runtime directly and do not depend on libhrx.

This is experimental runtime infrastructure. API, packaging, and deployment
contracts are still changing. It is not an official component of the ROCm stack.

## APIs and implementation

| Entry point | Contents |
| --- | --- |
| [Public headers](include/) | Native C API and C++ ownership helpers |
| [Native implementation](src/libhrx/) | Runtime and device groups, memory, execution, and synchronization |
| [HIP binding](src/binding/hip/) | Linux HIP runtime compatibility library, exported as `libamdhip64.so` |
| [Shared adapter facilities](src/binding/common/) | Execution, graph, and resource machinery used by API bindings |
| [hrx-info](tools/hrx_info.c) | Runtime and device inspection |
| [Conformance tests](cts/README.md) | Public API coverage, device selection, and focused checks |
| [Passthrough tools](src/passthrough/README.md) | HIP interception and comparison against the ROCm implementation |

Native AMD GPU and NPU execution uses [libamdf](../libamdf/README.md) through
the HAL. It needs compatible operating-system drivers; it does not require
ROCr or a ROCm SDK installation. The optional `legacy-amdgpu` HAL uses ROCr.
Compiling HIP device programs or exercising ROCm-specific tooling has its own
toolchain requirements, described in the
[AMDGPU build guide](../build_tools/amdgpu/README.md).

## Build and install

The [build guide](../BUILDING.md) owns host prerequisites, platform setup,
dependency selection, and build configuration. Run commands from the repository
root. This Linux example builds the native library, information tool, and HIP
binding, then installs the public package:

```bash
python dev.py cmake setup
python dev.py cmake configure \
  -DLIBHRX_BUILD=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_LIBDIR=lib
python dev.py cmake build hrx hrx-info libhrx_src_binding_hip_amdhip64
cmake --install build/cmake --prefix build/hrx-install \
  --component HrxPublicDist
```

Use the configured build directory if it differs from `build/cmake`.
The package contains `libhrx`, public headers, `hrx-info`, CMake package files,
and the HIP binding when enabled on Linux. Native Windows builds provide
`hrx.dll`, its import library, the public headers, and `hrx-info`; see
[Windows build setup](../BUILDING.md#native-windows).

| Option | Default | Purpose |
| --- | --- | --- |
| `LIBHRX_BUILD` | `ON` | Build libhrx and its API bindings. |
| `LIBHRX_BUILD_CTS` | `ON` when libhrx and tests are enabled | Include the conformance tests. |
| `LIBHRX_BUILD_HIP_BINDING` | `ON` | Include the Linux HIP compatibility library. |
| `LIBHRX_BUILD_PASSTHROUGH` | `ON` on Linux, `OFF` elsewhere | Include the Linux HIP interception tools. |
| `HRX_PUBLIC_DIST_COMPONENT` | `HrxPublicDist` | Name the public SDK install component. |

`BUILD_SHARED_LIBS=OFF` incorporates the private runtime libraries into libhrx
and is required for the public package. `BUILD_SHARED_LIBS=ON` supports in-tree
development with shared runtime components. Compiler, driver, target, and
dependency options are documented in
[shared project configuration](../BUILDING.md#shared-project-configuration).

An installed package exports the `hrx::hrx` CMake target:

```cmake
find_package(hrx CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE hrx::hrx)
```

Add the installation prefix to `CMAKE_PREFIX_PATH`. The
[package examples](build_tools/cmake/package_smoke/) exercise the installed C
and C++ interfaces.

## Run HIP applications

On Linux, the public package includes `lib/libamdhip64.so`. Preload it to route
a dynamically linked HIP application through HRX:

```bash
export HRX_ROOT="$PWD/build/hrx-install"
export LD_LIBRARY_PATH="${HRX_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LD_PRELOAD="${HRX_ROOT}/lib/libamdhip64.so" ./my_hip_application
```

The default `HRX_GPU_DRIVER=amdgpu` selects the native AMD GPU HAL.
`HRX_GPU_DRIVER=rdna` or `HRX_GPU_DRIVER=cdna` restricts enumeration to one
architecture family. `legacy-amdgpu` explicitly selects a build containing the
ROCr-backed HAL. The [CTS guide](cts/README.md#selecting-the-runtime-and-device)
describes device selection and how to identify the implementation under test.

HIP's automatic scheduling mode favors active waiting for low-latency stream
synchronization. `hipDeviceScheduleYield` performs a bounded observation period
before blocking; `hipDeviceScheduleBlockingSync` blocks directly. These flags
govern host synchronization. Event synchronization independently follows
`hipEventBlockingSync`.

## Release artifacts

The [Linux](../.github/workflows/build_core_linux.yml) and
[Windows](../.github/workflows/build_core_windows.yml) release workflows assemble
public packages, dependency overlays, and build environments. Tagged releases attach
archives, manifests, and environment helpers to
[GitHub Releases](https://github.com/ROCm/hrx-system/releases).

| Artifact | Contents |
| --- | --- |
| `hrx-public-linux-x86_64` | `libhrx.so`, `libamdhip64.so`, public headers, CMake package files, and tools. |
| `hrx-public-deps-linux-x86_64` | ROCm dependency overlay used by the release checks, including HSA and AQL profile libraries. |
| `hrx-rocm-buildenv-linux-x86_64` | The CI ROCm build environment, including ROCm LLVM. |
| `hrx-public-windows-x86_64` | `hrx.dll`, its import library, public headers, CMake package files, and `hrx-info`. |
| `hrx-public-deps-windows-x86_64` | The Windows dependency overlay produced by the release recipe. |
| `hrx-rocm-buildenv-windows-x86_64` | The CI Windows ROCm build environment. |

The dependency overlays describe that release's packaged environment; they are
not prerequisites for the native libamdf execution path. To reproduce a Linux
release environment, extract its public and dependency archives into one
directory and source the matching helper:

```bash
mkdir -p /tmp/hrx-root
cp hrx-public-linux-x86_64-env.sh /tmp/hrx-root/
tar --use-compress-program=unzstd -xf 'hrx-public-linux-x86_64-<tag>.tar.zst' \
  -C /tmp/hrx-root
tar --use-compress-program=unzstd -xf 'hrx-public-deps-linux-x86_64-<tag>.tar.zst' \
  -C /tmp/hrx-root
source /tmp/hrx-root/hrx-public-linux-x86_64-env.sh
```

Replace `<tag>` with the release's archive suffix. Windows uses `.zip` archives
with `.ps1` and `.cmd` environment helpers.

## Development and testing

The [CTS guide](cts/README.md) owns public API qualification and focused test
commands. Tests requiring GPUs or NPUs declare their device requirements;
compiling a driver does not provide that hardware. The
[build guide](../BUILDING.md#build-and-test-commands) describes both test runners,
and [Contributing](../CONTRIBUTING.md) covers hooks, change-set checks, and PRs.
