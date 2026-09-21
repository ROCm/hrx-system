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

## Preparation and steady-state use

| Path | Work that belongs there | Excluded from that path |
| --- | --- | --- |
| API-table negotiation | Read static version and extension tables. | Allocation, locks, once-initialization guards, dependency loading, discovery and system calls. |
| Immutable object and address queries | Validate the public request and copy or directly index retained facts. | Locks, allocation, lazy initialization, ownership-counter updates and native queries. |
| Scope-profile and visibility planning | Qualify a proposed consumer set and exact producer/consumer pair, using temporary host storage. | Device activation, native allocation, mapping and execution. This is not an allocation-free per-dispatch query. |
| Device, memory, context and queue creation | Acquire the native resources, address mappings, residency, packet storage and completion objects required by the requested resource. | Deferring that resource's preparation to a metadata query or its first publication. |
| Kernel publication | Claim publication, check pending capacity, resolve the caller-owned command range, fill required transport fields and publish natively. Capacity pressure can reclaim completed credits; XDNA also consumes completed command results before packet reuse. | Library locks, allocation, lazy setup, command parsing/copying, indirect-buffer scans and completion waits. |
| Kernel progress observation | Read established retirement and cached terminal state without mutating either. | Native completion checks, command-result consumption, retirement, library locks, allocation, lazy setup, system calls and active polling. |
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
defaulting to 4096. The native publication claim lasts only through the driver
call; it does not serialize submissions against execution completion. Accepted and
checked-retired fence points account for the pending window without a
per-submission allocation, command copy, or memory-retention list. When the
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
without a mapped fence is refreshed by explicit waits, including a zero-time
wait, or by submission under capacity pressure; the status method returns
established retirement rather than hiding an ioctl.

GPU user-queue status has another contract: it may query native queue or VM
fault state. It is not interchangeable with the syscall-free kernel-queue
status query. Direct user publication itself uses caller-owned mapped ring
state and doorbells without a libamdf submission call.

Host cache control also has an explicit cost. It executes the mapping's already
selected recipe over the intersected cache lines or native allocation spans.
A HOST_API recipe can enter the driver. Submission may publish its small
queue-owned transport packet, but does not publish the caller's instruction
stream or every indirectly referenced data allocation on their behalf.

## Checking an implementation

Functional conformance and performance evidence answer different questions.
CTS checks public behavior and native execution. Source review follows each
hot entry point through its provider, including failure and first-use paths,
to establish where allocation, synchronization and driver calls occur.
The [benchmarks](../benchmarks/README.md) measure separate memory lifecycle,
publication-only and completed-operation boundaries. A completed-execution
number cannot establish allocation-free publication, and a warm timing cannot
establish the absence of lazy first-use work.
