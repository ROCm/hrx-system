# Performance contracts

The public methods specify cost as well as results and ownership. These are
implementation requirements, including for a port or another native provider,
not observations about a warmed-up benchmark. The same contract applies to the
first call and to failure paths. A function that returns the correct value after
acquiring a new lock or lazily allocating state can still violate its contract.

[Base methods](../include/amdf/api.h), [GPU methods](../include/amdf/gpu.h) and
[XDNA methods](../include/amdf/xdna.h) carry their guarantees immediately above
each declaration. Native provider seams carry the corresponding obligations.
The bounds describe libamdf-owned work. Calls explicitly entering a driver
remain subject to native implementation cost and OS scheduling; the API does
not promise a wall-clock deadline for kernel submission.

Eager queue creation does not pin the device awake. An idle XDNA device can
runtime-suspend while its public context, queue and memory remain live. A later
native submission can synchronously resume hardware, firmware and contexts
before accepting work. This native power-management cost is distinct from
libamdf's prohibited lazy initialization. First use after native suspension
and warm publication are separate measurement scenarios; a warm result says
nothing about wake-up latency. Sustained XDNA results use an explicit
[held-power measurement policy](xdna.md#power-policy-and-measurement), while
deployment-representative measurements retain the deployment's actual policy.

## Preparation and steady-state use

| Path | Work that belongs there | Excluded from that path |
| --- | --- | --- |
| API-table negotiation | Read static version and extension tables. | Allocation, locks, once-initialization guards, dependency loading, discovery and system calls. |
| Immutable object and address queries | Validate the public request and copy or directly index retained facts. | Locks, allocation, lazy initialization, ownership-counter updates and native queries. |
| Scope-profile and visibility planning | Qualify a proposed consumer set and exact producer/consumer pair, using temporary host storage. | Device activation, native allocation, mapping and execution. This is not an allocation-free per-dispatch query. |
| Device, memory, context and queue creation | Acquire the native resources, address mappings, residency, packet storage and completion objects required by the requested resource. | Deferring that resource's preparation to a metadata query or its first publication. |
| Reservation creation and host-completed mapping updates | Reserve addresses, prepare publication transport, validate explicit range vectors, publish native mappings and wait for their translation readiness. | Discovering old users, retaining source objects, keeping an installed-mapping ledger, or deferring publication to the next payload access. |
| Queue-ordered mapping publication | Validate and translate range vectors using temporary descriptor storage, claim existing queue capacity, and publish native mapping/dependency operations. | Waiting for execution in libamdf, creating a worker, retaining backing, or introducing a second progress sequence. |
| Kernel command publication | Claim publication, check pending capacity, resolve the caller-owned command range, fill required transport fields and publish natively. Capacity pressure can reclaim completed credits; XDNA also consumes completed command results before packet reuse. | Library locks, allocation, lazy setup, command parsing/copying, indirect-buffer scans and completion waits. |
| Read-only kernel status query | Read established retirement and cached terminal state without mutating either. | Native completion checks, command-result consumption, retirement, library locks, allocation, lazy setup, system calls and active polling. |
| Explicit kernel progress refresh | Observe available native progress and consume completed command results without waiting for a particular submission. Native observation can require nonblocking system calls; result inspection scales with the newly completed prefix. | Allocation, resource creation, lazy setup, sleep and active polling. |
| Explicit waiting | Refresh native progress, inspect command results and establish retirement; query clocks, poll within the requested budget, yield and enter native waits. | Allocation or first-wait resource creation. Wait-event serialization consumes the same deadline. |

An address query indexes the memory's established access record and address
kind. Its cost does not grow with the allocation size, number of live
allocations or other consumers. Pair queries compose two concrete sites' retained
facts; they do not recover sharing relationships by traversing allocation state.
Immutable facts need no reader lock or ownership-count increment: the caller
already holds the owning handle alive through the operation.

`memory_scope_query_pair_info` belongs to pool or graph setup. Its temporary
planning storage scales with the declared consumer set. The result is sufficient
for each successfully admitted backing with matching construction inputs, so
recording and submission can reuse it without querying individual bindings.

Mapping-source access is explicit allocation/import preparation. Linux prepares
reusable GEM handles once for the requested native consumers; Windows uses the
ordinary resource's existing native allocation handles. Unchanged allocations
pay no additional native source-preparation cost. Mapping updates may allocate
temporary range vectors and serialize cold native VM mutations. The Linux lock
extends through publication and compute-PASID invalidation, not through
application execution. Neither reservation size nor installed mapping count
adds work to address queries or queue publication.

`kernel_queue_update_memory` is a cold mapping operation, not the allocation-free
command-publication path. It shares the queue's publisher and progress counters
but may allocate range descriptors. Windows bounds mapping admission to a
128-point unretired prefix, independently of the configured command capacity.
Unsupported providers return `UNSUPPORTED`; the runtime schedules the explicit
host-completed operation on its existing executor instead of paying a hidden
host wait inside native queue publication.

## Native storage and borrowed views

Queue capacity controls preparation cost as well as admission. The native
providers establish all packet and response storage before returning a queue;
they do not allocate another object when a submission claims a slot.

| Provider | Native storage for a pending capacity of N |
| --- | --- |
| Linux XDNA DRM | N page-sized command BOs and mappings. The submission ABI names a BO without a byte offset; deferred kernel processing reads and updates that BO through completion. Independently pending packets therefore require distinct BO identities. |
| Windows XDNA MCDM | One fixed-stride command allocation and one compact response allocation. Commands use allocation-relative addresses; each response occupies an eight-byte cell. The response allocation has a distinct native role. |
| Windows GPU WDDM | One prepared native queue with reusable submission metadata and a wait event. Pending capacity needs fence counters, not N native command allocations. |

These counts exclude context initialization and caller-owned instruction/data
memory. At XDNA's default capacity of 128, Linux command backing occupies
512 KiB across 128 BOs on a system with 4 KiB pages. Windows XDNA allocates
1 MiB for its current 8192-byte transport stride and one 4 KiB page for
responses. Explicitly requesting 4096 slots costs 16 MiB on Linux or 32 MiB
plus 32 KiB on Windows. Fixed backing removes per-slot native allocation calls;
it does not make the byte cost independent of capacity. Transport storage
remains live until checked retirement permits reuse, even when the publication
call has returned.

Host-visible memory already owns its persistent native mapping. Creating a
public host view allocates its one metadata record and borrows that mapping;
the GPU and XDNA providers allocate no second native view record. Creating a
host producer view of a Linux GPU user queue likewise borrows the queue's
established ring, control and doorbell mappings. Releasing a view releases no
backing; the caller releases views before destroying their owner. Cache
operations translate the view range to the backing once, then apply its
established recipe.

Memory construction also establishes initial cache readiness for a device
writer. A noncoherent provider may release dirty allocation or registration
cache lines once, with work proportional to the constructed range. Neither
creating a public host view nor submitting the first command defers or repeats
that preparation. Caller writes after construction still need the explicit
visibility recipe; import retains the external producer's ownership protocol.

## Synchronization is path-specific

Kernel publication and retirement use atomic ownership state. A competing
publisher receives `BUSY` instead of waiting for the queue slot. Queue-slot
updates can contend, so “no library lock” is not “wait-free” or “no atomic
operations.” Memory lifetime is a caller precondition; publication and
retirement perform no memory-user tracking or reference-count updates. The
runtime controls sharing and scheduling above this boundary. A wrapper that
adds a mutex to every handle or a reference-count operation to every metadata
read changes the steady-state contract.

GPU and XDNA kernel queues have a configurable pending-submission capacity,
defaulting to 4096 for GPUs and 128 for XDNA. Work scheduled within persistent
programs does not consume additional kernel submission slots. The native
publication claim lasts only through the driver call; it does not serialize
submissions against execution completion. Accepted and checked-retired fence
points account for the pending window without a per-submission allocation,
command copy, or memory-retention list. When the
window fills, submission refreshes native progress without waiting and reclaims
completed credits before returning `BUSY` if capacity is still unavailable.
GPU queues need only the fence counters. XDNA additionally retains preallocated
packet/result slots and consumes each completed result before reusing its slot;
reclamation cost is proportional to the completed prefix. A caller can pipeline
commands without intermediate host waits. Native resource exhaustion can reject
work before the configured admission bound is reached.

Windows uses its mapped progress fence. Linux XDNA uses a native timeline query
when reclaiming capacity; its ambiguous first point additionally needs a
zero-time check of an exact fence snapshot. The snapshot object is created with
the queue. A one-time transfer captures the first accepted fence before either
waiting on it or publishing a second command; this preserves identity without
allocating another object or waiting for completion. Concurrent capture returns
`BUSY` to a publisher instead of delaying publication behind the observer.

This capacity counts unfinished submissions, unlike a user queue's byte-sized
command ring. Native ring consumption permits those bytes to be reused but
does not establish completion of the work they described.

The wait path is deliberately different. Windows native waits reuse an event
prepared during queue creation and serialize access to it. That serialization
belongs to waiting, not submission or status sampling. Linux DRM progress
without a mapped fence is refreshed by `kernel_queue_refresh_status`, explicit
waits, including a zero-time wait, or submission under capacity pressure.
`kernel_queue_query_status` returns established retirement without entering
the driver. Refresh is a separate synchronization operation: it checks newly
completed command results and advances retirement, but success does not mean
that every accepted command has completed.

GPU user-queue status has another contract: it may query native queue or VM
fault state. It is not interchangeable with the syscall-free kernel-queue
status query. Direct user publication itself uses caller-owned mapped ring
state and doorbells without a libamdf submission call.

Host cache control also has an explicit cost. It executes the mapping's already
selected recipe over the intersected cache lines or native allocation spans.
A HOST_API recipe can enter the driver. Submission may publish its small
queue-owned transport packet, but does not publish the caller's instruction
stream or every indirectly referenced data allocation on their behalf.

## Measurement boundaries

The [benchmarks](../benchmarks/README.md) measure separate memory lifecycle,
publication-only and completed-operation boundaries. A completed-execution
number cannot establish allocation-free publication, and a warm timing cannot
establish the absence of lazy first-use work. The method contracts above apply
to every path, independently of the workload or measurement interval.
