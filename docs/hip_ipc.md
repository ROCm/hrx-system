# HIP IPC memory and events

The HIP compatibility binding supports sharing AMDGPU allocations and events
between Linux processes through the standard HIP IPC entry points:

- `hipIpcGetMemHandle`, `hipIpcOpenMemHandle`, and
  `hipIpcCloseMemHandle`;
- `hipIpcGetEventHandle` and `hipIpcOpenEventHandle`.

These functions and their opaque 64-byte handle types are already part of the
HIP ABI. The implementation fills in that existing ABI without adding HRX
entry points or changing the generic HAL ABI. AMDGPU builds report event IPC
through `hipDeviceProp_t::ipcEventSupported`; other driver configurations keep
the HIP ABI but report IPC as unsupported.

## Review map

| Layer | Main files | Responsibility |
| --- | --- | --- |
| HIP binding | `libhrx/src/binding/hip/api.c`, `ipc.c` | Validates the public entry points, encodes the HIP wire layouts, maps errors, and adapts HIP events to binding-common transactions |
| Binding common | `libhrx/src/binding/common/ipc_memory.{c,h}`, `ipc_event.{c,h}` | Owns the process-wide memory-import registry and context-retirement gate; defines the private event transaction interface |
| AMDGPU driver | `runtime/src/iree/hal/drivers/amdgpu/ipc_memory.{c,h}`, `ipc_event.{c,h}`, `ipc_event_monitor.{c,h}` | Owns ROCr memory tokens, attach and detach, GPU-agent access, ROCr IPC signals, canonical event carriers, and asynchronous event completion |
| Generic HAL | Existing buffers, semaphores, queues, and callbacks | Supplies portable primitives; gains no IPC handle, vtable method, or backend obligation |

The AMDGPU layer exposes typed, backend-internal memory and event operations.
The binding-common event interface stages record and wait transactions without
exposing ROCr types. IPC does not belong in a generic HAL vtable: allocation
tokens, HSA memory pools, GPU-agent access, and IPC signals are ROCr ownership
contracts rather than capabilities every HAL driver can implement. The typed
carriers convert them back into existing HAL buffers and semaphores at the
backend boundary.

## Wire compatibility

Applications must treat both HIP IPC handles as opaque bytes. HRX uses the
native ROCm layouts at the HIP boundary:

| Handle | Bytes | Contents |
| --- | --- | --- |
| Memory | 0-31 | Opaque ROCr memory token |
| Memory | 32-39 | Exact requested ROCr shareable allocation extent |
| Memory | 40-47 | Exported view offset from the allocation base |
| Memory | 48-51 | Exporting process ID |
| Memory | 52-55 | Exporting HIP device ordinal |
| Memory | 56-63 | Zeroed reserved bytes |
| Event | 0-3 | Native event discriminator, value 1 |
| Event | 4-7 | Exporting process ID |
| Event | 8-39 | Opaque ROCr IPC signal token |
| Event | 40-63 | Zeroed reserved bytes |

Only the ROCr type-1 event representation is supported. HRX exports type 1 and
rejects other event discriminators, including the shared-memory type-0 form.
An imported event can be re-exported: the ROCr token stays unchanged and the
new exporting process ID is encoded. Imported memory is not exportable again.

The layouts are native-endian, same-host values for compatible 64-bit
processes. They are not a network or cross-architecture serialization format.
The exporting and importing processes must use a consistent visible-device
ordinal namespace.

## Memory ownership

`hipIpcGetMemHandle` accepts the exact base pointer of a live, plain
`hipMalloc` allocation. Host, managed, virtual-memory, memory-pool, imported,
freed, and interior pointers are rejected. The wire extent and the public HAL
buffer and HIP pointer ranges are the exact size requested from `hipMalloc`.
Any page or pool-granule rounding used for physical backing remains private and
does not expand the exported or imported addressable range.

Imports have process-wide identity but context-local bindings:

1. The binding-common registry keys a mapping by ROCr token and view offset and
   verifies its extent and exporter identity on duplicate opens.
2. The first open creates one anchor HAL buffer that owns the ROCr attachment.
3. Duplicate opens increment a process-wide open count. A compatible importing
   device receives a context-local alias over the same attachment; aliases do
   not own detach.
4. `hipIpcCloseMemHandle` consumes one reference identified by the exact
   imported pointer. The final close releases every alias before the anchor,
   causing exactly one detach.

Close synchronizes each context that can have already-submitted work using the
mapping before changing ownership. Context destruction and reset first close
IPC-import admission, wait for admitted opens, and transactionally revoke that
context's bindings. If synchronization fails, counts and mappings remain
unchanged and admission reopens so the live context can be retried. Mappings
owned by other contexts remain live.

All handles from one exporter process and device use one importing context per
local device. `hipIpcOpenMemHandle` accepts only
`hipIpcMemLazyEnablePeerAccess`; the AMDGPU carrier grants access to the
selected importing GPU and visible peers that can access the exporter's
device-local pool. The exporter must keep the source allocation live while an
importer may use its handle.

## Event ownership and transactions

An exportable event is created with `hipEventInterprocess` and
`hipEventDisableTiming`. Its HIP adapter is installed at creation, but the ROCr
signal carrier is lazy: the first accepted record or an export creates it. A
first record that the queue rejects does not publish its speculative carrier.

Each process maintains one canonical carrier per live ROCr signal token. A
local source owns the canonical carrier; duplicate imports and a source process
opening a foreign re-export retain that carrier instead of attaching the token
again. Registry lookup and the last non-anchor release are serialized under the
registry mutex. Attach and signal destruction run outside the mutex while an
attaching or detaching state keeps the token reserved, and the registry anchor
remains until detach has completed and the entry is unlinked.

Destroying a HIP event does not wait for accepted work. Committed monitor
operations retain the carrier and its device until they settle; the signal is
destroyed only after the final process-local carrier reference is released.

The signal is binary: 1 is pending and 0 is complete. Record and wait use a
prepare, commit, and abort protocol around queue acceptance:

1. Record preparation allocates and retains all state and registers the local
   producer semaphore before submission. Commit rearms the ROCr signal only
   after the queue accepts the record; abort leaves the previous generation
   unchanged.
2. Wait reservation allocates a destination-device proxy semaphore before
   submission. Arming is allocation-free and binds the proxy to the generation
   visible at that queue position. Commit transfers an accepted wait to the
   monitor; abort releases an unaccepted wait.
3. A failure after queue acceptance fails the local proxy so accepted work is
   not left blocked. Normal producer completion, including a local producer
   failure, resolves the binary signal because the ROCr token has no
   cross-process error channel; the producer's local stream retains its error.

This ordering prevents rejected queue operations from publishing a generation
or consuming transport state. Ordinary events can be recorded only in their
exact creating context. IPC events additionally allow standard
`hipEventRecord` on another context backed by the same device. An accepted
non-capture record stores its ordering in the event point and releases the
recording stream; only capture-time records retain a stream for capture
metadata. The `hipExecutionCtxRecordEvent` and `hipExecutionCtxWaitEvent`
extension APIs do not accept IPC events.

ROCr IPC signals do not carry a cross-process generation number. Applications
that rerecord from multiple processes must externally order each rerecord after
all consumers have passed their waits for the prior generation.

## Capture behavior

IPC event record and wait operations are unsupported during stream capture.
When streams have joined one capture graph, rejection snapshots the
process-wide registered contexts and their streams, then invalidates every
active participant in that exact capture session before returning
`hipErrorStreamCaptureUnsupported`. If the snapshot cannot be allocated, no
participant is changed and the operation returns the corresponding allocation
error. Direct graph event nodes and event-node updates also reject IPC events.

## Monitor and shutdown

Committed event bridges use one lazy, process-wide, low-priority monitor
thread. Local producer records are callback-driven. External signal waits use
per-operation exponential polling from 100 microseconds up to 1 millisecond.
The scheduler uses fixed FIFO buckets and gives a due finite-poll operation a
service opportunity after at most four immediate callback operations, so
neither class can starve under sustained traffic. All fallible allocation and
monitor admission happens before queue acceptance.

Runtime shutdown closes monitor admission, cancels blocked preparation and
pending local proxies, drains ownership handoffs, and joins the worker before
backing contexts and devices are released. If accepted producer work is still
pending, shutdown marks that generation abandoned and leaves the interoperable
signal at 1; it does not fabricate completion or expose unfinished producer
writes. Later reuse of the abandoned generation fails promptly. A completion
already observable when shutdown begins is still published. A later IPC
operation may start a fresh monitor generation.

`hipHALDeinit` requires process-wide quiescence: no other HIP API call may be
active or begin until it returns. Sequential shutdown and reinitialization are
supported. Concurrent runtime-wide API admission during deinitialization is
not part of the event monitor contract.

The generic HAL dispatch surface and ordinary HRX queues do not participate in
this monitor. Ordinary events keep a null IPC adapter and take only
predicted-cold adapter checks on query, synchronize, record, and stream wait;
they create no ROCr IPC signal or proxy semaphore. Monitor scheduling is kept
off the core HRX worker paths and runs at low priority.

## Supported scope

- IPC is available only in Linux AMDGPU builds using ROCr.
- Same-process imports are rejected. Imported virtual addresses are
  process-local and need not equal the exporter's address.
- Imported memory is released by exact-pointer `hipIpcCloseMemHandle`, not
  `hipFree`; context teardown also revokes that context's remaining opens.
- Memory-pool IPC, VMM shareable handles, external-memory APIs, and external
  semaphore APIs are separate facilities and are not implemented here.
- IPC event timing is unsupported. A transferable event token remains valid
  only while at least one process keeps the referenced ROCr signal live.
- A producer that exits or shuts down with an abandoned pending generation
  cannot communicate an error through the binary signal; remote consumers may
  remain pending.

## Validation

Validation is split along the same ownership boundaries:

- HIP codec tests check both 64-byte layouts, malformed and same-process
  handles, opaque token handling, and error mapping.
- AMDGPU tests exercise exact 1-byte and 257-byte memory extents, export and
  attach ownership, access policy, canonical event import and release,
  transactional generations, monitor backoff and bounded fairness, shutdown,
  and restart.
- Binding-common tests exercise duplicate memory opens, aliases, concurrent
  close and context retirement, rollback, and event record and wait
  transactions.
- HIP shared-library tests exercise exact imported address ranges and the first
  byte outside each range, duplicate open, re-export, record, wait, joined
  capture invalidation, shutdown, and cross-process lifecycle behavior.
- Bidirectional memory interoperability tests use the selected stock ROCm HIP
  runtime and HRX shared libraries as separate exporter and importer roles.
  Native IPC event wire and cross-process behavior are validated between HRX
  processes; stock event interoperability is not asserted.

Cross-process tests initialize the runtime independently in each process and
exchange only opaque handles over explicit control channels. Interoperability
results apply to the selected libraries and configuration rather than every
ROCm release or post-initialization `fork()` behavior.
