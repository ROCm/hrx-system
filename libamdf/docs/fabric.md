# One fabric for CPU, GPU, and NPU programs

libamdf gives compilers and runtimes a common native foundation for AMD GPUs
and NPUs: memory, access, hardware capabilities, and execution resources. It
makes those resources available through a small, portable C ABI. The calling
runtime owns the program.

That boundary enables a different way to build heterogeneous software. A
program is a flow of computation and data across CPU, GPU, and NPU resources.
Memory belongs to that flow. An engine consumes a region, publishes a result,
and releases storage to its next use. Choosing another engine changes where
work happens, not who owns the program or how the application manages its data.

[Loom](../../loom/README.md) expresses this composition through one language and
pipeline model. Its compiler describes arithmetic, communication, placement,
and reuse together; its hardware abstraction layer (HAL) realizes the program
using libamdf's native resources. The same foundation serves other compilers
and runtimes that own their execution model.

## One model, shared across engines

Consider a language model stored in a compact quantized representation. The
weights occupy one backing allocation shared by their consumers. The CPU
handles request state and control, the GPU prepares blocks of weights and
activations, and the NPU performs matrix computation. Results flow into GPU
normalization, reductions, attention, or subsequent NPU stages.

Each engine sees the representation useful for its part of the computation.
A GPU shader reads compact weights, decodes a block, and writes the layout
consumed by an NPU worker group. That block occupies a bounded staging window,
not a second expanded copy of the entire model. The NPU consumes the window
while the GPU prepares another. Persistent weights stay compact; expanded
working storage follows the size of the pipeline window.

The return path uses the same fabric. NPU-produced output becomes GPU input
through shared backing and an explicit visibility transition. The CPU handles
the application's control decisions without relaying tensor payloads between
engines. A residual tensor, recurrent state, or attention cache retains its
identity across the operations that use it.

This makes engine selection a scheduling decision inside one program. The
compiler places dense arithmetic, irregular operations, data preparation, and
control where each is effective. It optimizes the boundaries between them
alongside the work within them.

## Overlap preparation, transfer, and computation

A bounded channel connects a producer to a consumer. For example, two input
windows let a GPU prepare the next matrix block while the NPU consumes the
current block. The producer publishes a ready window; the consumer returns it
after its last read. Backpressure follows the availability of those windows.

The schedule separates three useful events: source bytes have been read,
destination bytes are ready, and the last consumer has finished with them.
A DMA transfer releases its use of the source when its read completes.
Computation starts when the destination is ready. Destination storage returns
to the producer after its last reader finishes. Those boundaries expose overlap
that a single whole-stage completion would hide.

The same pattern works within the NPU array. Transfers refill local windows
while workers compute from other windows. Neighboring workers exchange partial
results through on-chip storage. A reducer consumes those partials directly,
keeping intermediate traffic close to the arithmetic. The compiler chooses
window sizes and placement together with the computation's service rate and
the available memory and transfer resources.

For inference, this connects GPU weight preparation, transfers into NPU SRAM,
matrix computation, and downstream consumption into one flowing pipeline.
Preparing the next block is useful work performed concurrently with the
current block, rather than an extra serialized phase before each invocation.

## Keep the data, change the program

A tensor's useful lifetime often spans several different programs. NPU SRAM
holds activations, accumulators, channel state, and intermediate results while
successive instruction images operate on them.

Consider a sequence that projects an activation, applies a nonlinearity, and
combines the result with a residual. The runtime places the activation and
residual in persistent local storage. Tile programs take turns using the
compute resources, retaining the data needed by later roles. Instruction
memory follows the active role; data memory follows the values still live in
the pipeline.

Code itself participates in the dataflow. A program catalog resides in memory,
and device-side transfers bring the next program into instruction storage.
Prefetch overlaps that transfer with useful computation. An explicit handoff
retires the outgoing instruction use and enters the next program while tensor
and channel storage remain in place.

A larger compiled region keeps several stages in one executable. A streamed
region reuses instruction storage across roles. Both implement the same
pipeline semantics. The compiler and runtime choose the boundary according to
code size, storage pressure, reuse, and the schedule.

This changes the role of loading. Cold setup establishes the array resources
and initial work; it is not the application's scheduling language. Executable
loading, relocation, and any PDI construction are runtime responsibilities.
libamdf supplies the memory addresses and native activation mechanism that
make those choices possible.

## Consume results as they become available

Pipeline composition extends inside an operation. A projection produces output
shards, and downstream workers consume each shard as it becomes ready. A
reduction combines worker-local partials. A nested attention pipeline feeds its
projection directly. The useful boundary is the region of data that the next
operation needs, rather than a host-visible completion for the entire tensor.

Fanout makes ownership especially important. Suppose an activation feeds both
a normalization stage and a residual path. Those are independent readers.
Normalization finishing releases its own use; the residual path keeps the
backing live until its use also ends. A compiler that owns the complete
dataflow carries both lifetimes into the schedule.

The HAL then realizes those lifetimes with channel credits, device-side
dependencies, and storage reuse. Each consumer advances on its own readiness
and completion edges, while shared backing stays protected until its last
reader releases it. Intermediate values remain in local storage or shared
memory through the chain of operations that use them.

This is how an application composes larger computations without turning every
source-level operation into a full tensor materialization, a global barrier,
or a host dispatch.

## Make dynamic control part of the pipeline

The same model handles work selected at runtime. In an autoregressive model,
the next iteration consumes updated state. A routed model selects its next
workers from computed routing decisions. A request scheduler admits another
request into a running service.

Queues and channels carry those decisions alongside data readiness. Resident
workers consume the next entry, reuse retained storage, and transition to the
next program. A cold worker enters through native activation. A running worker
continues through the device-side protocol. The compiler and HAL own both paths
as implementations of the same logical work.

The CPU participates where application control belongs: admitting requests,
handling external input, or making a host-side decision. Device-side
dependencies carry execution forward between those decisions. A host decision
exchanges the control data it needs; it does not require the CPU to shuttle
the tensors used by the next engine.

Independent queues have their own control state and executable instances.
Sharing immutable code or model storage is separate from sharing a schedule.
That separation supports concurrent pipelines while leaving placement,
admission, and fairness with the runtime that understands the workload.

## Place storage across shared DRAM and local HBM

On an integrated system, CPU, GPU, and NPU consumers use common system backing.
The runtime establishes their access once and follows the appropriate
publication and visibility rules at each handoff. Addresses and cache behavior
are explicit properties of the access contract.

On a multi-GPU system, the same memory model expresses physical locality.
A model is partitioned across local HBM, workers consume local shards, and
peer access or explicit transfers connect stages. Host staging storage is
another scope in that fabric. The schedule accounts for the bandwidth and
latency of each route and keeps heavily reused data near its consumers.

A scope describes where storage comes from and how it is used. A memory handle
identifies the backing and its established consumers. That gives the runtime
one ownership model for shared system memory, local memory, and private
execution storage. Topology and access capabilities determine placement;
the application's dataflow remains the same.

This separation also makes the programming model extensible. More local
storage changes retention and tiling choices. More compute changes
partitioning. New peer or queue capabilities change how dependencies are
realized. Those are target decisions within a common program representation.

## A precise division of responsibilities

The layer boundary keeps the programming model independent of native platform
plumbing.

| Layer | Responsibility |
| --- | --- |
| Application and compiler | Arithmetic, dataflow, engine selection, layouts, communication, and resource lifetimes. |
| HAL | Executable loading, command construction, scheduling, dependency realization, and suballocation. |
| libamdf | Passive discovery, capabilities and limits, backing and access, native device/context/queue resources, and required kernel-mediated submission. |
| Kernel and firmware | Protection, admission, native scheduling, and hardware services. |

Discovery supplies the hardware and access facts needed to select resources
before activating devices. The runtime explicitly creates the devices it will
use, obtains memory from scopes, and establishes the requested access. Prepared
memory remains mapped and resident for its required lifetime.

Subsequent operations use a caller-held memory handle and range. Address
queries return established metadata. There is no reverse lookup from an
arbitrary pointer, hidden device retention, or global allocation registry.
The runtime owns suballocation and decides when a range is reusable; libamdf
owns the native allocation and mapping state associated with its handle.

Queue families identify both their native command representation and their
publication mechanism. User queues expose native queue state and doorbells.
Kernel-mediated queues accept prepared commands through the platform's
required transport. The caller constructs GPU AQL or PM4 packets and XDNA
instruction streams. libamdf keeps that representation intact.

For XDNA, submission identifies an instruction-memory range. Data access is
established when memory is acquired, with no indirect data-buffer list attached
to each invocation. Executable preparation and relocation happen at the
lifetime chosen by the runtime. Changing input data does not require the
driver library to reinterpret or patch a program.

Linux and Windows implementations expose these resources through the same
public contracts. The HAL selects by capabilities and limits; the platform
provider handles native file descriptors, handles, mappings, and transport.

## Small enough to embed, open enough to evolve

A runtime that already owns executable loading, memory policy, and scheduling
needs native access underneath those facilities. libamdf provides that access
directly, replacing the need to bring XRT or ROCr into the execution path for
their device-access services. Its integration boundary is a native C ABI,
rather than an emulation of their application-facing runtime APIs.

The library's size follows from that responsibility. Compiler toolchains,
executable loaders, packet builders, tensor management, suballocators, and
schedulers live with their callers. Builds select the device families they
need, so an XDNA-only application does not acquire GPU-only dependencies.
Static, shared-linked, and runtime-loaded clients use the same API.

The hot path follows the same principle: establish resources once, prepare
commands at their useful lifetime, publish the work, and observe the completion
needed by the caller. Resource discovery and registration stay outside that
path. Native publication carries prepared work instead of reconstructing the
runtime's dataflow from buffers or arguments.

That access also shortens the authoring loop. A compiler developer changes a
layout, channel protocol, tile program, or reuse schedule in the program
itself. A runtime developer changes admission, code residency, or queue
composition above the driver boundary. Neither change requires teaching a
second runtime about a new execution model.

With one Loom language and pipeline model, CPU, GPU, and NPU code become
parts of the same optimization problem. The compiler sees the arithmetic,
the bytes moved, the storage retained, and the dependencies that permit
overlap. libamdf keeps the native foundation small while giving that compiler
and runtime control over the complete program.

The [memory design](memory.md) describes the scope and ownership contracts.
The [XDNA execution design](xdna.md) follows native instruction storage and
submission. The [README](../README.md) covers the library surface, embedding,
and build configuration.
