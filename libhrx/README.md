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
ROCr or a ROCm SDK installation.
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

## Copy, fill, and update geometry

The native API carries a complete rectangle or volume through queues, streams,
and graphs. `hrx_transfer_region_t` describes useful bytes per row, row and slice
counts, and independent source and destination strides. All offsets and strides
are in bytes. For example, this records a 32-row copy between differently padded
buffers:

```c
#include <hrx_runtime.h>

hrx_status_t copy_tile(hrx_stream_t stream, hrx_buffer_t source,
                       size_t source_offset, hrx_buffer_t target,
                       size_t target_offset) {
  const hrx_transfer_region_t region = {
      .row_length = 1024,
      .row_count = 32,
      .slice_count = 1,
      .source_strides = {.row_stride = 1088},
      .target_strides = {.row_stride = 1152},
  };
  return hrx_stream_copy_buffer(stream, source, source_offset, target,
                                target_offset, &region);
}
```

Only the 1,024 useful bytes in each row are copied; destination padding is
preserved. Each buffer must cover its complete strided span from the supplied
offset. Zero strides are literal zero displacements, not a request for packed
storage. A dimension with a count of one ignores its stride. Copies have
`memcpy` overlap rules for the useful bytes.

Geometry is captured before the call returns; buffer contents are consumed when
the operation executes. Streams order their operations and batch recorded work
until `hrx_stream_flush` or `hrx_stream_synchronize`. Direct `hrx_queue_copy`
calls take the same geometry with explicit wait/signal semaphores; call order
alone does not establish dependencies. The `_linear` helpers describe a single
row through these same APIs.

`hrx_fill_region_t` describes the destination of a fill. Its origin, row length,
and used strides must align to the 1-, 2-, or 4-byte pattern. Stream updates
capture up to 65,536 useful source bytes before returning, without storing row
padding. Larger host transfers use `hrx_stream_copy_h2d` or
`hrx_stream_copy_d2h`; their host storage remains borrowed until completion.
An upload source remains live and unmodified, and a download destination remains
live and inaccessible until that completion. Applications retain the buffers,
their backing storage, and their pools for all queued uses.

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
architecture family. The
[CTS guide](cts/README.md#selecting-the-runtime-and-device) describes device
selection and how to identify the implementation under test.

HIP's automatic scheduling mode favors active waiting for low-latency stream
synchronization. `hipDeviceScheduleYield` performs a bounded observation period
before blocking; `hipDeviceScheduleBlockingSync` blocks directly. These flags
govern host synchronization. Event synchronization independently follows
`hipEventBlockingSync`.

### Host transfers

For recurring host/device traffic, `hipHostRegister` registers application-owned
host storage, while `hipHostMalloc` allocates registered host storage. Reusing
these ranges amortizes registration and allows copies to use the device-visible
backing directly. The application keeps the storage and registration live until
all queued uses complete. Application-owned storage is unregistered with
`hipHostUnregister` before it is freed; `hipHostFree` releases host allocations
created by HIP.

Ordinary pageable host memory is also supported. A copy does not turn that
memory into a persistent implicit GPU registration. Transfers may use staging
or a qualified CPU mapping of the device allocation. Large pageable uploads and
pageable downloads can block inside an `Async` call; small uploads may capture
the source bytes before returning. Event or stream completion establishes when
the application can reuse the source or read the destination. The
[transfer measurements](cts/README.md#transfer-measurements) separate pageable
and explicitly registered memory so that these different costs remain visible.

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
