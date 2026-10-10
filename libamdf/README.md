# libamdf

libamdf is a small, portable C library for native AMD GPU and NPU access. It
provides the memory, capabilities, and execution resources for a unified
CPU/GPU/NPU fabric, with executable loading, command construction, scheduling,
and memory policy owned by the calling runtime, typically a hardware
abstraction layer (HAL).

The goal is one program across three execution domains. A CPU handles control,
a GPU prepares data and performs parallel computation, and an NPU executes
streaming and matrix workloads. Shared backing, explicit visibility, and
caller-owned lifetimes let those engines cooperate without turning each
boundary into a tensor copy or a separate runtime session.

libamdf targets RDNA and CDNA GPUs and XDNA NPUs on Linux, and RDNA GPUs and
XDNA NPUs on Windows. A common API exposes topology, locality, access, and
native queue capabilities so a HAL selects mechanisms by what they provide.

## What a common fabric enables

A compact model occupies one shared backing. GPU shaders prepare bounded
windows of weights in the layout needed by NPU workers, overlapping preparation
with computation. NPU results feed subsequent GPU or NPU work directly.
Expanded working storage follows the size of the pipeline window rather than
the size of another complete model copy.

On the NPU, tensor and channel state remain in SRAM as tile programs change
roles. Device-side transfers prefetch code while useful work continues.
Program lifetime follows the active computation; data lifetime follows the
values still needed by the pipeline. The compiler chooses between a larger
resident executable and a sequence of streamed programs.

Channels connect those stages at the granularity of useful data. Consumers
start from ready shards, independent readers retain their own uses, and storage
returns to a producer after its last reader finishes. Dynamic work enters
queues alongside data readiness. The CPU participates in application control
without acting as a payload relay or the scheduler of every internal stage.

[Loom](../loom/README.md) brings these pieces together in one language and
pipeline model. Arithmetic, communication, engine selection, and resource
reuse are part of the same program. libamdf supplies its native foundation
and remains independently usable by other compilers and runtimes.

[One fabric for CPU, GPU, and NPU programs](docs/fabric.md) develops these
scenarios and the architecture that makes them possible.

## A small boundary beneath the runtime

libamdf establishes backing and access through scope-taking memory operations.
The caller holds a memory handle and uses explicit ranges and addresses.
Discovery is passive, device creation is explicit, and allocation establishes
the requested consumer access before publishing the resource. The HAL owns
suballocation and retires memory before destroying its required native owners.

Execution uses caller-prepared native commands. A user queue exposes queue
state and doorbells; a kernel-mediated queue supplies the required platform
transport. GPU packet construction and XDNA executable loading, relocation,
and scheduling stay above the library. XDNA data memory is resident for its
allocation lifetime, without a per-invocation list of indirect data buffers.

Kernel-mediated queues admit a configurable window of pending submissions,
defaulting to 128 for XDNA and 4096 for GPUs. Callers can publish independent
immutable command ranges without waiting between them, then use synchronous
waits or nonwaiting checked status refresh to reclaim completed work. XDNA
queues can notify a caller-owned native event for an accepted point, allowing
one persistent event-loop registration to serve the queue. Notifications prompt
a status refresh; they are not a second completion or retirement mechanism.

This is the device-access foundation for a runtime that owns its execution
model. It replaces the need to adopt XRT or ROCr for native access while
keeping the compiler, loader, and scheduler with their caller. Family-selective
builds keep dependencies aligned with the hardware an application uses.
Static, shared-linked, and runtime-loaded clients share one C ABI.

## Native providers

The library provides passive discovery, scope-based allocation, registration
and external-memory sharing, explicit host visibility, and native queues.
It does not replace the existing IREE AMDGPU HAL or implement XRT/ROCr API
compatibility. The runtime can adopt this foundation without adopting either
runtime's execution model.

| Execution domain | Native provider | Foundation |
| --- | --- | --- |
| CPU | Linux and Windows host memory | CPU-only backing and explicit host views into shared memory; no synthetic CPU device. |
| RDNA / CDNA on Linux | KFD and DRM | System and local memory, caller-page registration, peer topology and shared GPU addresses, native user queues. |
| RDNA on Windows | WDDM / KMT and the private WKMI bridge | System and local memory, host registration, same-adapter D3D12 buffer import/re-export, native kernel-published PM4 and SDMA ranges. |
| XDNA on Linux | Modern amdxdna DRM | NPU4/NPU5/NPU6 support, resident data, context-private instruction storage and kernel-mediated instruction submission. |
| XDNA on Windows | MCDM / KMT | NPU4/NPU5/NPU6 support, resident data, context-private instruction storage and native transaction-interpreter submission. |

Capability queries describe the implemented platform, target and native driver
combination. Native interface support determines available operations; a driver
package version does not select an implementation path. CDNA targets Linux.

## Working with the API

The public entry point is [amdf_query_api](include/amdf/api.h), which returns an
immutable, versioned C table. Optional [GPU](include/amdf/gpu.h) and
[XDNA](include/amdf/xdna.h) tables expose family-specific services. Loading the
library and querying its tables do not discover or activate hardware.

A caller follows an explicit resource lifecycle:

1. Create an instance, enumerate endpoints, and open metadata snapshots. Inspect
   PCI identity, hardware family and passive topology before activating a device.
2. Create the devices selected for the workload. Query native XDNA geometry,
   instruction/context capabilities and achieved memory contracts. Allocate
   from a scope with the intended live consumers.
3. Map host views and obtain device addresses. Create queues and any required
   XDNA contexts; allocate private instruction backing from its owning context.
4. Publish data and caller-prepared commands, reuse backing and addresses, and
   observe explicit completion. The runtime supplies scheduling and last-use
   tracking.
5. Release resources in dependency order: mappings and memory before their
   required devices or contexts, then endpoints and the instance.

The [enumeration example](examples/enumerate.c) shows ABI negotiation and
passive selection without activating a device.

Focused design documents describe the contracts:

- [Discovery and activation](docs/discovery.md): passive identity and live capabilities,
  queue families and native driver ownership.
- [Memory fabric](docs/memory.md): scopes, shared backing, addresses, visibility
  and caller-owned lifetimes.
- [Performance contracts](docs/performance.md): method-level preparation,
  allocation, locking, native-call and steady-state cost guarantees.
- [GPU execution](docs/gpu.md): mapped publication, kernel-mediated submission,
  AQL fixed scratch, native signals and execution lifetimes.
- [GPU command reference](../docs/reference/amd/gpu/README.md): PM4, SDMA and AQL commands,
  native publication, memory visibility and execution ordering.
- [GPU timing and counters](../docs/reference/amd/gpu/observability.md): clock domains,
  timestamp completion, native profiling ownership and measurement boundaries.
- [XDNA execution](docs/xdna.md): instruction storage, submission and
  the native Linux and Windows requirements.
- [XDNA timing, counters, and trace](../docs/reference/amd/xdna/observability.md): tile timers,
  event selection, trace transport, and native firmware instrumentation.

## Native drivers

libamdf requires the native interfaces used by its providers and accepts
compatible newer drivers without a release allowlist. Hardware capabilities and
native interface support determine which operations are available; driver
package versions do not select implementation paths. The
[XDNA native requirements](docs/xdna.md#native-requirements) describe the NPU
interface floor, including Windows discovery of direct and metadata partition
admission. The baseline Windows metadata interface supports drivers that do not
populate the private adapter query. Drivers that lack required execution
interfaces need an update.

Install the complete driver package for the hardware, including its firmware.
libamdf accesses native drivers directly and does not require the XRT, ROCr, or
Ryzen AI application runtimes. A vendor's driver package may install XRT tools
or other dependencies as part of its supported installation procedure.

### Linux

The GPU provider uses `amdgpu`/KFD; the NPU provider uses `amdxdna`. Update the
distribution's kernel and firmware packages together, then reboot into the new
kernel. For example, Ubuntu 24.04 provides a rolling hardware-enablement kernel:

```bash
sudo apt update
sudo apt install --install-recommends linux-generic-hwe-24.04 linux-firmware
sudo reboot
```

When the distribution's `amdxdna` lacks the required interfaces, AMD's
[Linux NPU installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html#install-npu-drivers)
provide the current driver bundle for supported platforms. Follow that bundle's
package installation instructions, including its dependencies. The
[amd/xdna-driver project](https://github.com/amd/xdna-driver) also documents
building and installing its DKMS driver and firmware for supported
distributions. Use the matching kernel headers and the distribution's module
signing procedure when Secure Boot is enabled. A loaded `amdxdna` module alone
does not establish that every required native operation is available.

The NPU must be visible and accessible through `/dev/accel`; GPU access through
`/dev/kfd` and `/dev/dri` is separate. Containers need the corresponding host
device nodes and permissions as well as an updated host driver.

### Windows

Obtain the current NPU package for the processor from AMD's
[NPU driver installation page](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
or the computer manufacturer's support page. For AMD's standalone package,
extract the complete ZIP, open an administrator terminal in the extracted
directory, and run:

```powershell
.\npu_sw_installer.exe
```

Complete the installer and restart Windows if requested. Device Manager lists
the NPU under **Compute accelerators**; check that it reports a working device
and the newly installed driver. The NPU package is separate from the Radeon GPU
driver, which is available through
[AMD Drivers and Support](https://www.amd.com/en/support/download/drivers.html)
or the computer manufacturer. Keep each device's complete signed package
together so its kernel driver, firmware, and companion files agree.

After building libamdf, the enumeration example reports passive endpoint
identity and capabilities. Updating a compatible native driver does not require
adding its release number to libamdf.

## Building and embedding

The build produces two link modes from one implementation:

- `//libamdf:amdf` and `amdf::amdf` consume the shared library.
- `//libamdf:amdf_static` and `amdf::amdf_static` consume the static library.
- `//libamdf:amdf_shared_artifact` names the loadable DLL or shared object for
  clients that resolve `amdf_query_api` at runtime.

Windows GPU distributions install the private WKMI bridge beside `amdf.dll`.
It is neither a public link input nor part of the libamdf ABI.

`AMDF_BUILD` controls the CMake subtree and defaults to `OFF` unless the XDNA
HAL driver requires it. The Bazel equivalent is
`--//libamdf/config:enabled`. RDNA, CDNA, and XDNA package admission is selected
with the `AMDF_FAMILY_*` CMake options or the
`--//libamdf/config:families=...` Bazel setting. Bazel implementation packages
select individual members through `//libamdf/config/family:rdna`, `:cdna`, and
`:xdna` without interpreting the setting themselves.

Linux builds use pinned userspace driver protocol headers instead of depending
on the host's installed DRM/KFD/XDNA header versions. These are private build
inputs: libamdf does not link libdrm, and installed clients need only the public
amdf headers. Runtime driver capabilities are still queried independently.

Build libamdf without the legacy AMDGPU HAL:

```bash
iree-bazel-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=OFF
iree-bazel-build //libamdf:amdf //libamdf:amdf_static
iree-bazel-run //libamdf/examples:enumerate
```

The equivalent CMake build is:

```bash
iree-cmake-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF -DLOOM_BUILD=OFF
iree-cmake-build amdf amdf_static
```

The [benchmark guide](benchmarks/README.md) separates memory lifecycle,
publication-only, warm submission and completed execution measurements.

Installing the repository exports `amdf::amdf` and `amdf::amdf_static` through
`find_package(amdf CONFIG REQUIRED)` and installs the public headers beneath
`include/amdf/`.
