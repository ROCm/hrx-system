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
| Scope-profile planning | Qualify a proposed consumer set, using temporary host storage. | Device activation, native allocation, mapping and execution. This is not an allocation-free per-dispatch query. |
| Device, memory, context and queue creation | Acquire the native resources, address mappings, residency, packet storage and completion objects required by the requested resource. | Deferring that resource's preparation to a metadata query or its first publication. |
| Kernel publication | Claim a queue slot, borrow command memory, fill required transport fields and publish natively. | Library locks, allocation, lazy setup, command parsing/copying, indirect-buffer scans and completion waits. |
| Kernel progress observation | Read mapped or cached progress and retire completed command borrows. | Library locks, allocation, lazy setup, system calls and active polling. |
| Explicit waiting | Query clocks, poll within the requested budget, yield and enter native waits. | Allocation or first-wait resource creation. Wait-event serialization consumes the same deadline. |

An address query indexes the memory's established access record and address
kind. Its cost does not grow with the allocation size, number of live
allocations or other consumers. Pair queries compose two concrete sites' retained
facts; they do not recover sharing relationships by traversing allocation state.
Immutable facts need no reader lock or ownership-count increment: the caller
already holds the owning handle alive through the operation.

## Synchronization is path-specific

Kernel publication and retirement use atomic ownership state. A competing
publisher receives `BUSY` instead of waiting for the queue slot. Command-memory
borrow counters can contend, so “no library lock” is not “wait-free” or “no
atomic operations.” The runtime controls sharing and scheduling above this
boundary. A wrapper that adds a mutex to every handle or a reference-count
operation to every metadata read changes the steady-state contract.

The wait path is deliberately different. Windows native waits reuse an event
prepared during queue creation and serialize access to it. That serialization
belongs to waiting, not submission or status sampling. Linux DRM progress
without a mapped fence is refreshed by explicit waits, including a zero-time
wait; the status method returns cached progress rather than hiding an ioctl.

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
