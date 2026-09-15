# Discovery and activation

A runtime should be able to reject an unsuitable accelerator without powering
it up or allocating its execution resources. libamdf separates selecting an
endpoint from creating a device, and separates expected capabilities from the
resources actually obtained. Linux and Windows use the same public contracts.

## From library to workload

| Boundary | Result | Native cost and ownership |
| --- | --- | --- |
| `amdf_query_api`, `query_extension` | Immutable versioned tables for the services compiled into the library. | No allocation, system call, discovery, dependent-library loading or device activation. Tables remain valid for the library lifetime. |
| `instance_create` | Explicit lifetime root and native lifetime policy. | Host bookkeeping; shared driver connections are acquired when their devices are explicitly created. |
| `endpoint_enumerate`, `endpoint_open` | Selectable AMD endpoints and complete cached metadata. | Bounded discovery and metadata queries, including OS metadata handles where needed. No execution device, VM, workload context, paging queue or device allocation. |
| Family `device_create` | The selected live device and its achieved capabilities. | Native interface qualification and device-specific execution or memory state. Shared driver connections belong to the instance. |
| Memory, context and queue creation | Resources requested by the workload. | Explicit native allocation, mapping, residency and scheduling-context setup at their owning boundaries. |

An endpoint is a metadata snapshot, not a dormant execution device. The
[enumeration example](../examples/enumerate.c) negotiates optional family tables,
enumerates summaries, opens endpoints and prints their capabilities without
creating execution resources. A missing extension means the family was not
compiled into that library; it does not mean no such hardware is installed.

## Complete information before activation

Endpoint information supplies identity and topology. Family information adds
hardware facts such as GPU compute geometry or XDNA array geometry, instruction
format and limits. Queue-family and scope-profile queries describe the native
services that can be requested. Returned records are complete: zero describes
an absent capability, not a field waiting for an expensive query.

For example, a HAL selecting shared CPU/GPU/NPU storage can enumerate scopes and
call `memory_scope_query_profile` with the proposed endpoints. It can reject an
unsupported registration or consumer combination before initializing any
accelerator. After creating the selected devices, it calls
`memory_scope_query_device_profile` to obtain the achieved contract on the live
connections. Those results can refine expected limits without rewriting the
endpoint snapshot. The caller decides whether the achieved contract meets its
requirements; neither query reserves memory or guarantees allocation success.

CPU participation uses system-memory scopes and host mappings. There is no
synthetic CPU endpoint or device to activate. Native-private scopes, such as an
XDNA context's instruction aperture, are obtained from that explicitly created
owner. Discovering a physical memory location does not implicitly create the
device required to allocate there.

## Native metadata boundaries

Linux discovery reads cached sysfs identity, topology and heap metadata without
opening a render, KFD or accelerator execution file. Expected GPU memory limits
come from explicit ISA and package descriptions; unknown identities do not
produce guessed capabilities. Explicit device creation qualifies the native
interface and obtains actual memory limits before VM initialization.

Windows opens KMT adapters for metadata queries. Those handles and the graphics
kernel's process bookkeeping are distinct from the driver process context and
GPU address domain acquired by device creation. GPU qualification uses a private
`amdf_wkmi_bridge.dll` companion, which contains the pinned binary-only WKMI C++
and CRT ABI behind a versioned C table. Successful qualification releases the
temporary WKMI adapter state and unloads the bridge before endpoint open returns.
Subsequent GPU information queries copy cached identity, ASIC revision, compute
geometry, LDS limits, XCC topology and complete memory profiles.

Hardware profiles and installed native interfaces answer different questions.
An XDNA profile describes the target's array and address interpretations; it
does not identify a Windows driver-private wire ABI. Device creation qualifies
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
