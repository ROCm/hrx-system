# AMD HAL ownership

The AMD driver separates discovery and memory from execution engines:

```text
driver.{c,h}                     Scheme selection and device creation.
registration/driver_module.*     Registry entries sharing one driver identity.
fabric/
  fabric.{c,h}                   Native instance and passive discovery.
  memory.{c,h}                   Native backing and optional host mappings.
  buffer.{c,h}                   HAL views over prepared backing and bindings.
  file_transfer.{c,h}            Async file I/O through host execution bindings.
  slab_provider.{c,h}            Backing acquisition through the slab interface.
  memory_backend.{c,h}           Complete pool qualification and captured facts.
  memory_transition.{c,h}        Native visibility recipes translated to HAL.
xdna/                           Native NPU contexts, programs, and execution.
gpu/                            Native GPU images, PM4 execution, and retirement.
```

`fabric/` depends on libamdf core and HAL memory interfaces. It has no execution
engine dependency. XDNA owns its extension, execution context, queue resources,
private firmware scope, and firmware address interpretation. GPU owns its
extension, native context, command storage, and completion resources, with the
same dependency direction.

## Discovery lifetime and device selection

The registry retains one lazily created AMD driver per canonical factory ID.
Several URI schemes can name that identity without creating separate native
instances. The original scheme reaches enumeration and device creation, so each
family has its own ordinal namespace. Independent registries and direct driver
construction provide separate instances. The registry uses PROCESS native
lifetime to support Linux KFD host registration. Its primary VM can survive
fabric destruction and prevent a fresh acquisition, so applications retain one
registry/driver for devices using that native process context. Direct driver
construction with INSTANCE bounds native reclamation to the fabric at the cost
of services unsupported by secondary KFD contexts, including host registration.

The `xdna` and `rdna` schemes share that driver identity. Both `rdna:0` and
`rdna://0` select the first RDNA endpoint; `xdna:0?columns=4` configures an NPU
context. Enumeration returns `id/<32 hexadecimal digits>` paths for exact native
identity selection. Such identities can expire on reset, removal, or reboot.
Selection checks the requested architecture, so ordinals exclude other GPU
architectures and NPUs.

GPU scheme filtering opens query-only endpoints and uses
`amdf_gpu_endpoint_architecture`; it creates no device, address space, queue,
or event. Queue admission then selects an advertised native protocol and
publication mode. RDNA supports PM4 USER host-producer and KERNEL submission
families with GFX11/GFX12 launch and cache encodings. Both use the same bounded
command recording, dependency, and retirement paths. Other native services fail
explicitly at construction or operation admission.

`rdna://0?queue_count=2` provisions two queues sharing the device's execution
workers. `queue_family` selects an exact native family ordinal instead of the
first supported family. `host_numa_node` selects CPU placement for the workers
and shared proactor; GPU-local memory node IDs are not CPU NUMA node IDs.

Each materialized device retains the fabric independently. Releasing the
registry and driver leaves those devices operational. The fabric owns the native
instance and its discovery snapshot; engine contexts own their native children.

## Shader diagnostics and sanitizers

The generic URI parameters request runtime services at device creation:

```text
rdna://0?feedback
rdna://0?asan&asan_report_policy=fail-device
rdna://0?tsan&tsan_report_policy=report-only
```

Programmatic callers set the corresponding `runtime_features` in
`iree_hal_device_create_params_t`. ASAN and TSAN also provision feedback.
Executables use the existing compiler-generated configuration globals and report
layouts; requesting a runtime service does not instrument an ordinary HSACO.
Loading an executable requiring an unprovisioned service fails explicitly.

Reports reach the supplied `event_sink`, with executable and source-site identity
retained through delivery. The default sink discards events; applications can
select `iree_hal_device_event_sink_stderr()` or provide their own callback. The
existing completion worker drains the shared channel. There is no additional
feedback thread. Assertion reports preserve the existing diagnostic behavior;
they do not automatically fail submissions.

ASAN defaults to `report-only`; TSAN defaults to `fail-device`. Both policies are
selectable through the URI parameters above. A fail-device report propagates
`ABORTED` for ASAN or `DATA_LOSS` for TSAN and rejects subsequent work. Failure
notification does not cancel already-issued native work or certify its resource
retirement. Pools and caller-owned backing must still outlive their users.

ASAN maintains sparse address shadows, allocation redzones, and freed-backing
quarantine. TSAN maintains per-queue workgroup shadows and immutable executable
placements, including when a recorded command buffer runs on different queues.
Shadow-slot reuse inserts execution joins before prior users can be overwritten.
Sanitizer shadows use local storage when available and qualified system storage
on devices without a separate local heap. Ordinary devices allocate none of this
sanitizer state.

The shared producer requires native shader-triggered events. The current Windows
provider exposes submission completion but lacks that shader event producer, so
feedback/ASAN/TSAN device creation is rejected there. Enabling those services
requires implementing and qualifying the native shader producer; ordinary
Windows PM4 execution does not depend on it.

## Memory construction

An engine captures its native API, instance, device, storage scopes, family
ordinals, address interfaces, minimum native permissions, and private host-access
needs in `iree_hal_amd_fabric_memory_backend_options_t`. The backend borrows these
facts and existing progress services from the device. Each storage choice names
its native scope, locality properties, and any enforced physical affinity. Pool
construction qualifies every participating family together against each eligible
scope and captures the selected placement, access set, binding layout, and
visibility table. Allocation and retirement use the selected storage owner's
existing progress services.

Placement is independent of the access-family set and its order. A pool can
require backing on one physical GPU while exposing that allocation only to a
peer's families or CPU families. Construction qualifies any private native owner
attachment without publishing extra HAL scopes or bindings.
Required placement selects the corresponding physical node in the sealed group;
an unavailable route yields no candidate. Preferred placement can select another
qualified route and reports the achieved placement. Automatic selection prefers
qualified device-local backing when the requested access and host preferences
permit it. SYSTEM scopes make no enforced host NUMA placement promise.

One allocation can publish several address interfaces for several native
devices. A host mapping is acquired only when public or engine-private access
requires it. XDNA currently requires a private read/write host view for its
transfer implementation. Its native memory also requires read/write device
permissions even when the HAL contract grants narrower logical usage.

Pools remain explicit. A sealed device group dominates its pools and their
borrowed native owners. Slabs own backing; buffer views borrow backing and
binding tables through the existing release-callback contract. The slab cache,
suballocation layers, and explicit trim policy retain their existing ownership.
No buffer adds a fabric reference or discovers native addresses at submission.

Owned allocation, host registration and native import/export classes are
qualified together against the selected scope. Imports preserve caller-owned
storage and its physical placement; owned-backing placement does not relocate
an imported allocation. Unqualified native transports and uncached device-access
requirements yield no candidate. Queue visibility uses libamdf's exact
queue-family recipes; program-side visibility remains unqualified by that native
query.

## File transfers

XDNA imports host-memory files and platform file handles through the shared HAL
file implementations. Platform handles used by queues are opened with
`IREE_IO_FILE_MODE_ASYNC`. Queue reads and writes use the existing public API:

```c
IREE_RETURN_IF_ERROR(iree_hal_queue_read(
    queue, prerequisites, loaded, file, file_offset, buffer, buffer_offset,
    length, barriers, IREE_HAL_READ_FLAG_NONE));
// Device work waits for loaded and signals computed.
IREE_RETURN_IF_ERROR(iree_hal_queue_write(
    queue, computed, saved, buffer, buffer_offset, file, file_offset, length,
    barriers, IREE_HAL_WRITE_FLAG_NONE));
```

The fabric transfer owns the exact byte range and continues partial native I/O.
The execution engine owns semaphore dependencies, before/after barriers, and
terminal signaling. A failed transfer fails its dependent signals; it can leave
partial output and does not fail independent queue work. Write completion means
the operating system has consumed the bytes, without promising disk durability.

XDNA resolves its private host execution binding after dependencies complete,
so file operations can target queue-allocated buffers that have not committed at
capture time. Public host mapping permission is unnecessary. The existing host
service acquires source bytes before writes and publishes destination bytes
after reads. Disk I/O runs on the imported file's proactor; its completion
returns to the queue owner before final publication. Neither the proactor nor
the host service blocks waiting for disk I/O. Files on another proactor use the
same completion path, with each progress owner kept live by its caller.

Memory files with device-visible storage already use HAL queue copies. Other
memory files copy on the host service. This implementation allocates no staging
memory and creates no additional threads. Device-only backing without a host
execution binding needs an engine-specific copy/staging route and is rejected by
the direct host path. File and buffer references remain retained through the
operation; callers still own backing retirement and the device lifetime.
