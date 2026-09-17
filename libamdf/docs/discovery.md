# Discovery and activation

A runtime should be able to reject an unsuitable accelerator without powering
it up or allocating its execution resources. libamdf separates selecting an
endpoint from creating a device and querying its native resource capabilities.
Linux and Windows use the same public contracts.

## From library to workload

| Boundary | Result | Native cost and ownership |
| --- | --- | --- |
| `amdf_query_api`, `query_extension` | Immutable versioned tables for the services compiled into the library. | No allocation, system call, discovery, dependent-library loading or device activation. Tables remain valid for the library lifetime. |
| `instance_create` | Explicit lifetime root and native lifetime policy. | Host bookkeeping; shared driver connections are acquired when their devices are explicitly created. |
| `endpoint_enumerate`, `endpoint_open` | Selectable AMD endpoints and cached passive metadata. | Bounded discovery and metadata queries, including OS metadata handles where needed. No execution device, VM, workload context, paging queue or device allocation. |
| Family `device_create` | The selected live device and its achieved capabilities. | Native interface qualification and device-specific execution or memory state. Shared driver connections belong to the instance. |
| Memory, context and queue creation | Resources requested by the workload. | Explicit native allocation, mapping, residency and scheduling-context setup at their owning boundaries. |

An endpoint is a metadata snapshot, not a dormant execution device. The
[enumeration example](../examples/enumerate.c) negotiates optional family tables,
enumerates summaries, opens endpoints and prints their capabilities without
creating execution resources. A missing extension means the family was not
compiled into that library; it does not mean no such hardware is installed.

## Identity before activation, resources after activation

Endpoint information supplies identity and passive topology. XDNA family
information identifies its architecture and compiler target; its native array
geometry, context admission and instruction limits come from the live device
query after explicit activation. GPU family information includes the hardware
facts available through passive native metadata. Queue families describe native
submission mechanisms; memory profiles require the live devices that will
consume the storage. Returned records are complete: zero describes
an absent capability, not a field waiting for an expensive query.

For example, a HAL selecting shared CPU/GPU/NPU storage first filters hardware
by identity and architecture, then explicitly creates the selected devices.
It calls `memory_scope_query_device_profile` with their complete access set to
obtain native allocation, registration and sharing capabilities. The caller
can reject an unsuitable contract before acquiring memory. The query reserves
no memory and does not guarantee allocation success.

CPU participation uses system-memory scopes and host mappings. There is no
synthetic CPU endpoint or device to activate. Native-private scopes, such as an
XDNA context's instruction aperture, are obtained from that explicitly created
owner. Discovering a physical memory location does not implicitly create the
device required to allocate there.

## Matching other native APIs

`endpoint_query_info` also returns a typed `native_identity` captured during
passive discovery. It identifies an actual native device when multiple
endpoints have identical PCI product IDs. The opaque `id` continues to select
libamdf endpoints; the diagnostic name is intended for display.

For a Linux GPU, the identity contains its DRM render-node major and minor
numbers. A Vulkan consumer can compare these with
`VkPhysicalDeviceDrmPropertiesEXT.renderMajor` and `renderMinor` when `hasRender`
is true. An XDNA endpoint reports its accelerator character-device numbers in
the same native namespace.

On Windows, the identity contains the adapter LUID bits and its KMT
physical-adapter index. DXGI can select the logical adapter directly with
`EnumAdapterByLuid`. Linked adapters retain their separate physical indices;
matching the logical adapter alone does not select a physical node. The LUID
bits encode `(uint64_t)(uint32_t)HighPart << 32 | LowPart` without exposing a
Windows ABI type in the public headers.

Native identities correlate contemporaneous providers on the same machine.
They are not persistent across reboot or device removal. A match establishes
identity; the caller still qualifies the requested transport, memory geometry
and access before constructing shared backing. Identity queries acquire no
execution resources and expose no native handles to retain or release.

## Native metadata boundaries

Linux discovery reads cached sysfs identity, topology and heap metadata without
opening a render, KFD or accelerator execution file. Explicit device creation
qualifies the native interface and obtains actual memory limits before VM
initialization. GPU address ranges and integrated-versus-discrete placement come
from the native device query, without PCI or GFX memory tables.

Windows opens KMT adapters for metadata queries. Those handles and the graphics
kernel's process bookkeeping are distinct from the driver process context and
GPU address domain acquired by device creation. GPU qualification uses a private
`amdf_wkmi_bridge.dll` companion, which contains the pinned binary-only WKMI C++
and CRT ABI behind a versioned C table. Successful qualification releases the
temporary WKMI adapter state and unloads the bridge before endpoint open returns.
Subsequent GPU information queries copy cached identity, ASIC revision, compute
geometry, LDS limits and XCC topology. Memory limits are queried during explicit
device creation and retained by the live device.

Hardware profiles and installed native interfaces answer different questions.
XDNA target identity selects architecture encodings and firmware bootstrap,
while native activation supplies actual array geometry. Neither selects a
driver interface by its package release number. Device creation qualifies
that ABI before constructing native contexts. Failed family qualification does
not publish a partial family record; core endpoint identity remains queryable.

## Queue families describe mechanisms

Each endpoint reports a dense immutable set of queue families. A family names
its accepted command representation independently from publication:

- PM4, SDMA, AQL or XDNA identifies what the caller prepares.
- User publication means directly updating mapped queue state and doorbells.
- Kernel publication means a bounded library call transports prepared commands
  through the native platform interface.

CPU translation from AQL to PM4 is not a publication mode. The library exposes
the native mechanism so a runtime can choose its own command representation
and scheduling strategy.

An advertised command/publication pair promises a matching queue implementation
for that endpoint, subject to normal live admission and resource availability.
Missing families describe the provider's services, not necessarily every
capability of the silicon. Windows GPU PM4/SDMA families require the loaded KMT
and WKMI surfaces to provide hardware-scheduled queues; the XDNA family requires
the KMT operations needed by its native path.

`endpoint_query_queue_family_info` copies cached records. It allocates nothing,
performs no system call, and neither initializes nor waits for a device. A HAL
can select by representation and publication mode instead of branching on the
operating system.

## Lifetime remains explicit

The instance owns shared driver connections, not memory allocations, execution
objects or a registry of descendants. A memory resource owns its backing and
native access state and borrows the devices named in its construction request.
Private resources also borrow their specific context or device. The caller
releases dependent resources before those owners; closing a device does not
silently keep it alive through its memory.

The instance's PROCESS or INSTANCE native lifetime policy selects a supported
kernel reclamation boundary. It does not change public handle ownership or
promise a separate process address space. The [memory design](memory.md)
describes the native lifetime choices and their platform consequences.
