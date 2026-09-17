# Memory fabric

libamdf obtains storage, establishes access to it, and describes the rules for
using it. Allocation policy, suballocation, scheduling, executable loading, and
deciding when storage can be reused belong to the caller, typically a hardware
abstraction layer (HAL).

This document describes the memory contracts and their caller scenarios.
[Public headers](../include/amdf/memory.h) specify the C ABI; the
[implementation overview](../README.md#implementation-and-qualification)
summarizes the native providers and qualification scope.

## Fabric discovery

A libamdf instance is the application's explicit lifetime root. Through it, the
application discovers GPU and NPU endpoints, system-memory locations, GPU-local
memory, and their supported access relationships. The CPU participates through
host mappings, without a synthetic CPU endpoint or execution device.

The instance selects a native lifetime policy before any devices are initialized.
`AMDF_NATIVE_LIFETIME_PROCESS` is the default: kernel-owned state may survive
instance destruction until process exit. `AMDF_NATIVE_LIFETIME_INSTANCE` requires
native state to be reclaimable by instance teardown. Either policy permits
earlier release and leaves public handle ownership unchanged. The policy is
neither an address-space isolation request nor automatic integration with other
native clients.

Linux GPU access uses KFD's primary context for process lifetime and a secondary
context for instance lifetime. Primary contexts support caller-page registration;
secondary contexts currently do not. Capability queries reflect the instance's
policy before device creation. Explicit GPU device creation prepares one shared
KFD connection on the instance and acquires each GPU's VM with a stable render
file. Execution devices borrow those connections. Closing and recreating devices
within that instance reuses the exact binding under either policy; closing a
device does not retain the execution object or any of its memory. Under process
lifetime the primary VM binding can survive instance destruction, so a new
instance or another native client with a different render file can encounter
`EBUSY`. Applications coordinate their primary KFD use. Windows uses explicitly
reclaimable native objects under either policy; instance lifetime does not create
a separate process GPU address space.

Discovery supplies identity and native metadata for hardware selection without
creating execution devices, queues, firmware contexts, address spaces or device
allocations. Explicit device creation obtains native placement, alignment,
address limits and sharing capabilities. Queries over the resulting live devices
return complete contracts before memory construction; explicit requirements
remain binding.

The fabric does not imply that every engine can access every allocation. It
presents resources and relationships together so callers can reason about them
without assembling separate platform-specific memory systems.

## Scopes and profiles

A memory scope answers: what storage can be obtained here, under what access and
lifetime rules? It is not itself an allocation, pool, device, or address space.

| Scope | Storage and access | Discovery |
| --- | --- | --- |
| Fabric system memory | System backing usable by a supported set of CPU, GPU, and NPU consumers. | Available before device initialization. |
| Physically local memory | Storage in an identified location, such as one GPU's VRAM or HBM. Peer and host access depend on capabilities. | Enumerated through the live storage device. |
| Native-private memory | Storage whose meaning or accessibility depends on a particular device or context. | Retrieved from that live owner. |

Discovering a scope makes its storage contract available for selection. It does
not initialize the devices needed to obtain that storage or establish access.
The caller explicitly initializes those devices before device-dependent memory
construction. CPU-only system-memory allocation requires no accelerator.

Physical locality identifies where backing lives; it does not make the public
allocation interface an operation on one originating device. Shared access is
established through the scope-taking operation, including for local memory when
supported. A context-private aperture has the narrower access and lifetime rules
of its actual owner. The same scope-taking memory operations serve all three
cases.

Within a scope, memory profiles describe supported combinations of placement,
construction method, access, alignment, host mapping, and sharing. A profile is
a valid combination, not a collection of independently composable flags that
can accidentally describe an impossible request.

`instance_enumerate_memory_scopes` returns system scopes, while
`device_enumerate_memory_scopes` returns physical-local scopes available through
the live device. Physical scopes retain their endpoint identity, so separate
devices for the same endpoint refer to the same storage location. These
are borrowed descriptors, not additional objects to destroy. Enumeration with
zero capacity and null storage returns the required count with
`BUFFER_TOO_SMALL` when any scopes exist.

For XDNA instruction storage, `context_enumerate_memory_scopes` returns a private
scope borrowed from the explicitly created context. A HAL queries that scope
with the context's device and EXECUTE access, allocates its instruction backing
with `memory_create`, maps and publishes the bytes, then submits a memory range.
The allocation profile supplies its size and alignment requirements, including
when the native path admits only one full aperture for that context. The HAL
suballocates the usable range; libamdf owns no slab allocator. Native bootstrap
storage is excluded from the allocation's public address and usable extent.
The HAL retires work and destroys its mappings and memory before destroying the
context. No allocation retains that execution owner.

`memory_scope_query_device_profile` takes the live consumers and their access
requirements. Its backing profile and per-consumer capability array describe
one jointly supported request, in caller order. The query returns complete
records without reserving resources or guaranteeing allocation success. An
empty consumer array describes CPU-only storage and requires no accelerator.

For example, a HAL creates its selected GPU and NPU, then queries their joint
profile before allocating caller-owned storage. That profile supplies the
page-cover granularity and supported alignment used for registration. If the
contract cannot meet the HAL's requirements, it can reject the combination
before acquiring memory.

`memory_create` and `memory_import` take the same scope and explicitly initialized
devices in the same order. They qualify the live contract and establish every
requested property before publishing the resource. An empty consumer array
requests CPU-only storage; it does not mean access for every discovered device.
The resulting memory owns its immutable access array, not a mutable membership
list.

## Backing and resource handles

An `amdf_memory_t` represents one backing store and the access established for
it. Several engines can reach that backing without a separate public memory
object for each engine.

There are three ways to obtain a memory resource:

- Allocation obtains new backing owned by libamdf.
- Registration accepts caller-owned storage. Libamdf owns the registrations,
  pins, and mappings; the caller owns the original storage and keeps it alive.
- Import acquires the documented ownership or reference to externally supplied
  backing and establishes the requested access.

Registration and import are acquisition mechanisms, not physical locations.
Imported memory can be system memory or GPU-local memory.

A construction request specifies the scope and profile, size, alignment,
placement requirements, and consumers and access it needs. Device consumers
identify devices the application has already initialized. Scope capabilities
describe supported access combinations; the request selects the access that
must be established for this allocation. For example:

> Allocate system memory that the CPU can map, this GPU can read and write,
> and this NPU's DMA engine can read.

Success establishes the entire requested contract. A second copy, weaker
access, or an unprepared consumer is not a successful substitute. Native
handles, registrations, peer mappings, and backing references are construction
details owned by the resource. External import and export serve interoperability;
ordinary sharing within the fabric does not require the application to assemble
an export/import chain.

Linux GPU system allocations and caller-page registrations use one KFD backing
handle mapped into the complete requested GPU group at a common virtual address.
The native allocation fixes read, write, and execute permissions, so this path
requires the same exact permissions for every participating GPU. Scope queries
qualify that combination before construction. The memory resource owns the fixed
mapping set; releasing it unmaps every participating VM before releasing the
native backing handle. Other engines join allocated backing through their
supported external-memory transport. For registration, other engines register
the same original caller pages, not an exported allocation or a copy. Both paths
use the same scope-based call.
Registration preserves the requested subpage offset in each GPU address and
exposes only the logical range; the caller retains the original storage until
all native mappings are released.

Registration also supplies `registered_host_cacheability`, established from the
caller's allocation or mapping contract and held unchanged for the registration's
lifetime. The profile reports the accepted class. A raw pointer is insufficient
to establish cacheability; unknown or incompatible classes fail admission.
Current registration profiles require ordinary write-back pages.

Windows GPU/XDNA sharing uses the same REGISTER request with both devices in
the consumer list. Each native adapter attaches the original caller pages;
there is no payload copy or exported-handle chain. The joint profile requires
a 64 KiB aligned host pointer and a length that is a multiple of 64 KiB.
Device addresses are resolved independently and guarantee 4 KiB alignment;
host allocation alignment does not imply the same device-address alignment.
Normal release destroys the native attachments without freeing caller storage.
This registration capability does not imply joint CREATE or XDNA import/export
support. Queue visibility recipes and completion edges remain explicit.

GPU-local sharing is relative to the chosen placement. Selecting GPU A's local
scope and naming consumers B and A allocates in A's VRAM or HBM, with access
records remaining in B,A order. The joint profile qualifies B's ability to reach
that backing; B need not offer a local heap itself. Selecting B's local scope
asks a different question and can produce a different result.

On Linux, KFD admits local peer access within an enabled xGMI hive or through
its directed PCIe peer links. PCIe admission accounts for the backing GPU's
visible BAR, consumer DMA addressability, and platform peer-routing support.
Hive membership does not imply PCIe reachability, and a PCIe edge does not imply
the reverse edge. Device creation retains the driver's topology for subsequent
queries. Unsupported consumer sets are rejected before allocating backing.
Supported local groups use the same fixed
mapping owner, common GPU address, exact permissions and ordered release as
system groups, without host staging or an application export/import chain.

Resource operations identify memory with its opaque handle and an explicit
range: conceptually `{memory, offset, length}`. Host-mapping operations similarly
use their mapping handle. Libamdf does not reverse-map arbitrary physical or
virtual addresses to allocations through a global registry, hash table, or
interval search. The caller already has the handle identifying the resource.
An opaque handle does not require an integer handle table; its representation
can directly identify the library-owned object.

Host registration receives a pointer and length at the acquisition boundary;
subsequent operations use the resulting handle. Address queries return pointers
or numeric addresses for CPU and device use, not identifiers to feed back into
an allocation lookup. Native imports likewise carry an explicit transport and
ownership contract rather than asking libamdf to infer an owner from an address.

A HAL can allocate a large slab, retain its handle, and suballocate using offsets.
An interior range is explicit and needs no search for an original allocation
base. Libamdf needs no object for every tensor, argument block, or executable
subrange.

`memory_query_info` reports the requested logical range and the complete native
payload extent separately. Native rounding and any private backing prefix count
toward `native_allocation_byte_length`; driver and library metadata are separate.
Several attachments to one backing can each report that complete extent. When
the provider can establish physical identity, `physical_backing_id` identifies
the shared backing; an all-zero identity leaves that relationship unknown.

A pool accounts for the backing it owns and the logical ranges it allocates.
Native process residency charges and device-wide usage have their own scopes:
shared imports can incur separate residency charges, and exported backing can
outlive the original owner's accounting entry. Those mutable observations cannot
establish the pool's owned capacity or whether an allocation will succeed.
The profile's construction limits describe accepted dimensions, while the live
resource query describes the extent actually acquired.

## Windows foreign buffers

Windows GPU local-memory scopes expose an IMPORT profile for
`AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE`. Selecting that profile with the full
live consumer set fixes both admission and same-format re-export before any
native preparation. The accepted resource is a shared, same-adapter committed
D3D12 buffer in a DEFAULT heap, with one uncompressed linear native allocation.
Heap handles, textures, fences, cross-adapter resources, legacy KMT shares and
Vulkan opaque handles have different contracts and are rejected. Existing
libamdf allocation profiles do not acquire exportability retroactively.

The producer creates its buffer with `D3D12_HEAP_FLAG_SHARED` and obtains an NT
handle with `ID3D12Device::CreateSharedHandle`. After completing foreign work
and returning the resource to COMMON, it supplies the exact logical range:

```c
amdf_external_memory_t external = {
    .type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE,
    .payload.native_handle = shared_resource_handle,
    .source_byte_offset = slab_byte_offset,
    .byte_length = allocation_byte_length,
    .release = release_owned_windows_handle,
};
amdf_memory_device_access_t access = {
    .device = gpu,
    .requirements = {
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
    },
};
amdf_memory_import_info_t import_info = {
    .type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
    .structure_size = sizeof(import_info),
    .memory_profile_ordinal = import_profile.ordinal,
    .access_count = 1,
    .accesses = &access,
    .required_flags = AMDF_MEMORY_FLAG_SHAREABLE,
};
amdf_status_t status = api->memory_import(local_scope, &import_info,
                                          &external, &memory);
```

Here `import_profile` is the local scope's queried IMPORT/EXPORT profile and
`release_owned_windows_handle` closes the owned handle. On success, `external`
is empty and its callback has run; the memory owns independent native
references. Failure leaves `external` and the output memory slot unchanged.
The native provider validates the typed D3D12 resource and its logical width,
queries the driver's allocation extent, and establishes GPU mapping and
residency during import. These costs occur once during cold construction;
subsequent address queries read the prepared access array.

The logical range can start at any buffer byte. Returned GPU addresses identify
that logical byte zero. `native_allocation_byte_length` can exceed the buffer's
width, but padding outside the buffer is not importable storage. Re-export with
D3D12_RESOURCE returns an independent same-access NT handle naming the whole
resource, together with the exact accumulated logical offset and length.
The handle can outlive the libamdf memory object and be opened with
`ID3D12Device::OpenSharedHandle`. It does not keep a separately managed slab
suballocation live. The caller preserves that allocation until every user has
retired and releases each exported transport explicitly.

The prepared GPU queue contract supplies global release-to-system and
acquire-from-system cache operations. The interop caller also supplies D3D12
resource transitions and completion edges. A native cache operation does not
replace a foreign API's ownership transition, and import/export perform neither
synchronization nor data copies. Prospective queries use the same import type,
consumer requirements and queue family as actual memory queries. Physical
identity remains unknown when the native interface cannot establish it; equal
handle values or caller-provided identity words are not alias evidence.

## Addresses and their consumers

An address is meaningful to a particular consumer in a particular address domain:
the environment in which the numeric address is interpreted. One backing store
can have a CPU mapping, a GPU virtual address, an XDNA DMA address, and a different
address interpretation required by firmware.

Different numeric addresses can identify the same bytes. Equal numeric addresses
do not prove shared backing. An address result describes its domain, intended
consumer, accessible range, and granted access. The caller can cache the base
and use offsets within that range. Querying it is a metadata operation, not a
hidden mapping or residency operation.

Ordinary unified GPU addressing gives participating GPUs a common pointer once
their access is established. Native per-GPU page tables do not require the
application to manage different pointers. Scopes select obtainable storage;
there is no separate public ordinary address-space object to create or select
for memory or queues. Allocation establishes access for the requested live
devices; their queues use those established mappings. Creating a queue does
not require another public memory attachment or allocate a new copy.

The address-query kind identifies the interface that consumes the number:

```c
api->memory_query_address(memory, gpu_access, AMDF_MEMORY_ADDRESS_GPU,
                          &gpu_address);
api->memory_query_address(memory, xdna_access, AMDF_MEMORY_ADDRESS_XDNA_DMA,
                          &dma_address);
api->memory_query_address(memory, xdna_access, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                          &firmware_address);
```

Each call returns a status; the example omits propagation. Access ordinals index
the memory's immutable consumer records directly. Profiles report supported
address kinds, and each access info reports the kinds established for the
whole logical range. Unsupported kinds fail without modifying the output. The
GPU kind is for ordinary GPU addressing. The XDNA DMA kind includes the native
translation needed by shim DMA descriptors; the firmware kind is for native
firmware arguments or private instruction storage. Passing a DMA address to a
firmware interface that applies that translation itself would translate twice.
CPU pointers come from explicit host mappings, not this device-address query.

Genuinely context-private memory obtains its ownership qualification from its
scope. Equal firmware instruction addresses in different private scopes can
name different storage without changing the ordinary shared-data pointer model.

## Native resources and activation

The instance owns shared native driver connections. Explicit device creation
initializes the participating device and establishes the device-specific native
state needed by its supported operations. Instance creation and passive scope
discovery do not initialize every available accelerator.

The instance has no allocation registry or deferred memory-release list. Memory
bookkeeping belongs to the individual resource, including when construction
fails; sharing a driver connection does not make the instance a memory owner.

An `amdf_memory_t` owns its backing references, allocation handles, registrations,
and established access. A platform can require native device handles, virtual
memory state, or paging queues for those operations. Those dependencies are
concrete implementation state, not a second public allocation owner or a reason
to create a hidden public device during memory construction.

For example, Windows allocation APIs use native device handles, and GPU address
mapping requires synchronization with a paging fence. Those obligations belong
beneath the portable contract; they do not require the application's first
execution queue to own shared memory. See the [Windows allocation API] and
[GPU virtual-address management].

Devices required to obtain or access the allocation are explicit lifetime
dependencies. The caller keeps them alive until the dependent memory is
released. Instance ownership of shared connections does not retain a destroyed
public device or extend the validity of its dependent memory.

Memory construction performs the native allocation and access setup required
by its scope and requested consumers. It does not implicitly initialize a
participating device. CPU-only system memory does not activate a GPU or NPU;
ordinary data allocation does not create workload contexts or load a program.

## Readiness, visibility, and ordering

Readiness means the requested backing, mappings, pinning, and residency
requirements have been established. Ordinary memory construction returns ready
resources. Native setup costs occur at that resource boundary, not once per
dispatch. There is no per-invocation indirect BO list, first-launch registration,
hidden pinning submission, or scan of the application's pointer graph.

Visibility describes how one participant observes another's writes. Shared
backing does not automatically imply coherent caches or mutually supported
atomics. Libamdf exposes directional rules for the actual producer, consumer,
memory, and access mechanism. These identify required CPU or engine-side cache
operations, or establish that no cache maintenance is needed. Atomic support
includes the relevant operation, width, and participating domains, not merely
a property saying that memory supports atomics.

A CPU mapping reports its CPU cacheability and the flush/invalidate operations
available through that mapping. Those are mechanisms, not requirements for
every consumer. Ordinary write-back pages can be coherent with a GPU and still
need flushing before XDNA reads them. An explicit cache-control request performs
the advertised operation even when one device is host-coherent. A CPU view has
no device reset epoch; reset validity belongs to each device's access.

`memory_query_pair_info` describes the required transitions between two sites.
A DEVICE site selects `{memory, access_ordinal, queue_family_ordinal}`; a HOST
site selects an `amdf_host_mapping_t`. That mapping supplies its own backing,
range and cache behavior, without another memory argument or a synthetic CPU
device. Two sites on one memory handle share backing even without an external
physical identity. Distinct imported handles require qualified matching
identities. Pair queries only read metadata; they neither perform the reported
cache operations nor supply ordering. Native host API requirements remain
native operations, including Windows allocation-cache publication.

`memory_scope_query_pair_info` qualifies those recipes before any backing or
host view exists. Its `amdf_memory_profile_pair_query_t` fixes the scope-local
profile, backing requirements, complete caller-ordered device/access set and
registration cache class. A prospective DEVICE site names an access ordinal and
exact queue family; a HOST site declares view permissions and requires
`HOST_VISIBLE`. Successful constructions with those same inputs can reuse the
answer across sizes and addresses. The query shares native selection and policy
with construction and actual mapping descriptions, without retaining a plan,
reserving resources or performing native allocation, mapping or synchronization.

A pool can therefore query its producer/consumer matrix once, record visibility
operations before individual buffers exist, and replay using each allocation's
actual operands. `HOST_DIRECT` operates on the actual pointer and covered range;
`HOST_API` requires the actual host-mapping handle; `QUEUE` belongs to the exact
queried family. Queue qualification does not qualify shader `PROGRAM` operations
or cross-device atomics. Unsupported sites return `UNSUPPORTED`, not a no-op.
Range granularity covers absolute host cache-line boundaries, including any
neighboring bytes touched by an unaligned subrange. Independent transitions
require caller ownership of those boundary lines.

Import qualification additionally fixes the external type and provenance.
Linux native-owned write-back system backing advertises an `OPAQUE_FD` contract
whose matching provenance qualifies direct host cache maintenance through an
XDNA import. Its independent native backing reference survives destruction of
the exporting resource. The caller preserves the exporter-provided typed value;
relabeling a foreign descriptor does not establish that contract. Portable
`DMA_BUF_FD` remains available for external device interoperability, but its
exporter CPU-access protocol is unqualified. Such host mappings report UNKNOWN
cacheability and operations; their host pair queries and cache-control
operations return `UNSUPPORTED`. Device-only qualification remains independent
of host policy.

Ordering establishes when the consumer may act. Cache coherence does not create
a producer-to-consumer dependency. The HAL or application supplies that
dependency through its synchronization and scheduling mechanisms. Libamdf
exposes native mechanisms and requirements without inferring dependencies from
pointers or generating a scheduling policy. The HAL constructs engine commands;
the library does not need a packet builder to describe a cache requirement.

## Executable storage

Storage intended for execution is `amdf_memory_t` with the appropriate
executable-use contract. Its scope and profile describe which consumer can use
it and any native restrictions. A buffer containing machine code is not
automatically executable memory.

A catalog of tile images in DRAM, read by DMA and streamed into tile instruction
memory, is ordinary readable backing from the DMA consumer's perspective.
Storage passed directly to a firmware execution interface can require a
different scope and address interpretation. Libamdf exposes storage and access
supported by the native interface, not an assumed allocator for every tile-local
resource.

Libamdf supplies storage, addresses, access guarantees, and native resource
mechanisms. The caller supplies application bytes and understands their meaning.
Application ELF loading, relocations, tile-control tables, replacement programs,
and streaming schedules belong above the library.

Native interpreter admission belongs to libamdf. The Windows XDNA provider
constructs its minimal PDI/CDO container directly in reserved private storage
before exposing the usable instruction range. Its single NOP admits the
interpreter without installing a tile program or assigning application DMA,
locks, routes, or tile data. Admission runs once for the context's private
backing, independently of application dispatch or program replacement. The
provider publishes only the encoded bytes and retains storage until accepted
native preparation retires, including when an observation fails. Modern Linux
DRM submits native ELF instruction ranges without this bootstrap container.

## Lifetime and failure

Device-dependent memory is constructed after the required devices have been
initialized and is released before those devices are destroyed. Private memory
also depends on the device or context identified by its scope. These are caller
preconditions: libamdf neither retains those owners through memory nor tracks
their memory descendants to diagnose premature destruction. Destroying a
required owner while dependent memory remains live is application misuse.

Ordinary memory can precede workload queues and remain usable after a queue is
destroyed, while its required devices remain live. A context-private allocation
has its scope's narrower lifetime. The lifetime of a queue, a context, and a
device are distinct contracts, not interchangeable meanings of an execution
object.

An external import acquires the reference or ownership defined by its transport.
Its backing can remain valid after an independent source reference is released.
That is not continued validity of the source memory handle or permission to
destroy a device on which the surviving import itself depends.

The caller establishes last use before destroying or reusing backing. Under
user-mode scheduling, completion of a kernel submission does not necessarily
mean that no tile, queue, DMA engine, or subsequent program can follow a pointer
into the allocation. Libamdf cannot discover references embedded in opaque
commands or device-side queues. Destruction performs no implicit wait for all
possible uses, and a successful destruction is not proof of device retirement.

Caller lifetime obligations do not remove public range and capability
validation or native error reporting. Constructors own rollback of the resources
they acquire before publication. They publish complete outputs only on success;
failure leaves caller output storage unchanged and creates no caller cleanup
obligation. A cleanup error does not transfer memory ownership to the instance.

Native error handling follows the particular operation's resource-consumption
and progress contract. An interrupted operation with documented resumable
progress differs from a terminal release failure. An error alone does not
establish that retaining an object and retrying later is safe or useful, and
reporting failure does not imply that a native resource has been released.

Final memory release consumes the library object even when native teardown
fails. Construction rollback follows the same single-attempt rule. Both report
the native error and free their bookkeeping. Unreleased native resources and
any library-owned backing or address reservation they still depend on are
leaked, not transferred to a retry list. A driver failure
cannot be repaired by retaining more libamdf objects. Native residue is left to
native teardown; leaked host reservations can remain until process exit. The
native lifetime policy describes supported reclamation boundaries, not a
guarantee of reclamation when the native release operation itself fails.

Host mappings are lightweight views of their memory's persistent native mapping.
Destroying a valid view consumes its metadata and succeeds; actual native unmap
belongs to final memory release. Neither views nor queue commands register a
memory-use count, and premature memory release is undefined behavior.

Registered source storage remains caller-owned. A failed native detach does not
prove the source may be recycled, even though its libamdf handle was consumed.
A caller releases its source storage only after successful native teardown; on
failure it reports the error and leaves that storage unreclaimed through the
remaining native lifetime. That can require process exit. There is no surviving
memory handle to retry and no library object that will perform later cleanup.
An independent imported or exported transport reference has its own native
lifetime; a failed import release does not invalidate other owners' references.

Normal execution teardown differs from physical loss. Destroying a GPU queue
need not invalidate shared system memory. Losing the GPU holding an HBM
allocation can lose that backing; there is no invisible surviving copy. A failed
projection does not automatically imply that the shared backing or another
participant's projection was lost.

## Caller scenarios

A CPU/GPU/NPU pipeline follows one resource lifecycle:

1. Create an instance and discover endpoint identities and system scopes.
2. Explicitly initialize the selected GPU and NPU devices.
3. Query their joint system-memory contract for the required access and address
   kinds, then qualify the needed directional pairs with
   `memory_scope_query_pair_info`. Retain the answers with that fixed contract.
4. Allocate backing once from the selected scope, requesting access for those
   live devices. Obtain a CPU mapping and cache the established device addresses.
5. Create workload queues and any required contexts. Retrieve private scopes from
   their live owners and allocate private executable storage where needed.
6. Run the pipeline with explicit visibility operations and dependencies, reusing
   backing and cached addresses across submissions. Retiring and destroying a
   workload queue does not require rebuilding ordinary memory access.
7. Establish final device-side last use and release resources in dependency
   order: commands and queues before memory they borrow, host mappings before
   their memory, and private memory before its owning context or device.
8. Destroy the now-unused devices, close endpoints, and destroy the instance.

A discrete-GPU pipeline can instead select local VRAM or HBM and require access
from particular peer GPUs. If a required peer or NPU cannot access that placement,
construction fails. The HAL can explicitly choose separate system memory and a
scheduled transfer; libamdf does not manufacture shared backing with copies.

An application registering existing host storage keeps that storage alive
through registration and every device use. An application importing external
memory follows the transport's ownership contract. Both then operate with
handles and ranges through the same scope-based interface.

The memory ABI establishes native address mappings as part of construction. It
does not expose separate virtual-address reservation, alias/remap operations or
live budget accounting. Those are distinct services, not hidden side effects
of command submission or additional responsibilities of a memory scope.

Libamdf owns the native machinery that makes memory usable. The caller owns what
the bytes mean, how work moves through them, and when they can be reused.

[Windows allocation API]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtcreateallocation
[GPU virtual-address management]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/per-process-gpu-virtual-address-spaces
