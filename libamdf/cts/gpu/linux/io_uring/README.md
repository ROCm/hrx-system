# GPU initiated Linux file transfers

The [causal round trips](file_io_test.cc) and [concurrent gathers](file_gather_test.cc)
demonstrate one GPU invocation constructing native io_uring requests, consuming
kernel completions, and computing on the returned bytes. libamdf supplies
registered memory and a native PM4 queue. The [staged-file path](file-staged.md)
adds GPU-generated SDMA and independently dispatched AQL readers. The caller
uses Linux syscalls directly; the running test has no HAL, IREE async,
liburing, or shader-compiler dependency. Loom compiles the
[GPU programs](../../kernels/README.md) during the build.

This is a bounded ownership and visibility witness for model-table gathers and
block-cache writeback/reload. It establishes the native boundary, not a storage
API or a throughput result. One-credit cases isolate native visibility and
causal progress. Three-credit cases add a key-based demand join, independent
ready work, a retained final reader, scattered writeback/reload, and error
drain. Both have one SQ publisher and one CQ consumer. The logical readers run
inside one invocation. The staged-file cases separately prove native completion
and last-reader reuse across independent multi-workgroup dispatches. These
integer ownership witnesses do not implement matmul, cancellation or a
production cache scheduler.

## One credit causal workload

The file has a power-of-two bank of immutable input blocks followed by an
equally sized output bank. Each block occupies one host page. A GPU-owned cause
selects an input block. The GPU reads it, transforms every 32-bit word modulo
2^32, writes a permuted output block, and reads that output into a different
payload window. The actual reload's first word becomes the next cause. The
host supplies only the initial seed. Each successor is derived from returned
bytes rather than a host-authored request list.

The round-trip cases perform 33 rounds and 99 requests through an eight-entry
SQ and sixteen-entry CQ. Repeated physical slot reuse is intentional. Separate
read, write, and reload windows have complete guard pages between them. A
transcript retains each round's selected blocks, cause, and every reload word.
After native completion and queue retirement, independent CPU expectations
check the transcript, all payload and guard words, unchanged arguments, ring
positions, recent CQ identities, and the complete file. A final completion word
alone is insufficient.

The source includes four distinct cases:

- `BufferedCausalReadWriteReload` exercises kernel-copy visibility and the
  complete device-owned request/response chain.
- `DirectCausalReadWriteReload` uses `O_DIRECT`, checks the filesystem's
  `STATX_DIOALIGN` contract, and requires aligned registered buffers and file
  extents. A tmpfs file or unavailable alignment contract produces an explicit
  skip, never a buffered substitute.
- `PartialReadThenEofRetiresWithoutConsumingIncompletePayload` reads a half-block
  file. The GPU advances the file and buffer offsets after the positive short
  completion, observes EOF on the remainder, and terminates with `-ENODATA`
  without transforming incomplete input or writing output.
- `InvalidFixedFileRetiresWithTheNativeError` selects an absent fixed-file
  entry. The GPU observes `-EBADF`; all payloads, records beyond the summary,
  and file bytes remain unchanged.

## Concurrent gathers and final reader release

[file_gather.loom](../../kernels/file_gather.loom) begins by publishing three
independent input reads before consuming any CQE. A fourth requester derives
its key from its own seed and joins a matching live source slot. The lookup
increments that source's reader count and supplies the slot identity used by
the delayed consumer; it does not publish another read. This is a bounded
three-entry key table, not a scalable cache or hashing implementation.

The joined source has two consumers with different arithmetic results. The
first transforms, writes, and reloads immediately. The second remains pending
while both independent streams perform 33 causal round trips. Their first
trips can advance immediately; subsequent reuse waits until the held source
has demonstrably retained its final reader. The remaining 64 peer reloads
record that live reference. Only after both peers finish does the delayed
consumer read the source and release it. A third consumer of that stream then
reuses the released source slot for a new response-derived input generation.

Nine payload windows form input, write, and reload banks, with a complete guard
page between windows. Writes use the reversed slot order; reloads use a rotated
slot order in a different bank. The 69 consumer results occupy distinct,
permuted file blocks. A successful case has 68 unique input reads plus 69 writes
and 69 reloads: 206 logical transfers through an eight-entry SQ and sixteen-entry
CQ, with additional submissions if an operation completes short.
Every consumer's full reloaded payload is compared with independent CPU
arithmetic, alongside the complete file, its exact length, every final pool
word, guards, state, arguments, and retained native CQEs.

Native `user_data` carries the owner slot and submission ticket. The event loop
processes whichever completion arrives next; it does not wait for an earlier
ticket to complete. The request journal records the CQ position at consumption
and the consumed-CQ frontier at publication. These observations establish that
each slot's reuse follows its own completion, peers recycle while the shared
source remains retained, and the delayed reader follows both peers' final
reloads. The oracle accepts any valid CQ order and records actual reordering.
Linux's [I/O model][model] explicitly permits out-of-order completion.

All three held-slot identities run in both buffered and direct modes. Five
additional cases cover absent fixed-file failure during read, write, and reload,
a positive short read followed by EOF, and exhaustion of the caller's bounded
request journal. The first observed failure freezes SQ publication. Accepted
I/Os are all consumed; successful pending operations may still modify their
destinations, but no subsequent arithmetic or successor work begins. The
oracle includes those post-failure bytes and any accepted
writes when checking the final file. Abandoned logical reader references are
released only after drain. The four-entry journal case stops with exactly two
accepted requests to retire; exhaustion returns `-EOVERFLOW` without writing
past the supplied journal extent. Writeback is not transactional: successful
peer writes remain in the file after another operation fails.

The reference has three I/O credits and conservatively keeps each stream's
input/write/reload chain on one credit. It proves independent progress and
final-reader ownership, not optimal queue depth or independently scheduled
compute. Increasing the pool or adding matmul workers requires a separate
admission/release protocol and matched performance measurements.

## Address spaces and native ownership

Cold setup creates ordinary anonymous write-back pages and registers their GPU
access with libamdf. The same caller-owned pages back the native SQEs, shared
SQ/CQ control, and registered I/O payloads. GPU addresses returned by libamdf
are distinct arguments from the CPU virtual address used by the Linux fixed
buffer table. Storage-device DMA addresses belong to the kernel. No identity
between these three address spaces is assumed.

`IORING_SETUP_NO_MMAP` lets the caller supply the SQE and ring storage;
`IORING_SETUP_NO_SQARRAY` removes the submission-index array. Returned UAPI
offsets locate ring state. Compile-time checks bind the shader's 64-byte SQE
and 16-byte CQE field layout to the build's Linux headers. The
[typed shader arguments](../../kernels/file_exchange.h) have independent
[compiled-product checks](../../kernels/resident_kernel_test.cc).

The build uses pinned Linux protocol headers, not the build host's header
version. Runtime admission independently probes a fixed, disabled caller-owned
ring. An absent syscall, unsupported base ring flags, or a policy denial skips
the case before workload I/O; allocation and other resource failures remain
failures. The probe submits no work and creates no polling thread. Each
workload's actual setup also checks policy admission for its selected path:
SQPOLL permission is independent of ordinary ring creation. A denial skips that
path before GPU submission, while host-submission cases retain their own
admission. Other setup errors, registration failures, and enablement failures
remain test failures.
Direct-storage cases additionally query the filesystem's `STATX_DIOALIGN`
contract; an unavailable query or absent alignment contract skips those cases
without substituting buffered I/O. No runtime decision compares kernel release
strings, so backported support is exercised as well.

The [shared native fixture](file_io_fixture.h) owns cold registration, idle
wakes, and queue-first teardown. The ring starts disabled. The host registers
one private, unlinked regular file and the payload mapping, restricts the ring
to fixed-file `READ_FIXED`
and `WRITE_FIXED`, and then enables it. These restrictions limit accepted I/O
operations; they are not a sandbox for untrusted GPU programs. No raw device
namespace or unrelated application file is accessed.

## Publication and progress

The GPU fills each complete SQE before a system-release store advances the SQ
tail. This is a contiguous publication boundary, unlike AQL's independent
packet-header publication. The one-credit witness reuses a slot only after
its completion. In the three-credit witness, published-minus-consumed requests
never exceed three, below SQ capacity. Every completed request has had its SQE
consumed; therefore the outstanding count also bounds unconsumed SQ occupancy
even when CQEs arrive out of order.

The GPU polls the CQ tail atomically, then performs a system-acquire fence
before reading the CQE result or payload. It returns the CQ entry after reading
its metadata. CQ space and payload credits are different lifetimes: returning
a CQ entry does not authorize overwriting data still used by computation.
The single GPU owner completes each arithmetic reader before decrementing its
source reference. A new input requires an unowned source credit; a positive
short retry retains the same credit and advances both file and buffer offsets.
EOF and negative results stop issuance and drain every accepted request.

`SQPOLL` provides a kernel CPU thread that submits I/O and processes its
completion work. It removes the userspace submit/reap relay, not CPU execution
from the storage stack. An idle poller needs `IORING_ENTER_SQ_WAKEUP`. The host
control loop observes only SQ flags/positions and the final GPU completion;
it never constructs an SQE, consumes a CQE, changes a ring position, or repairs
payload visibility while the GPU runs. The test waits for the real
`IORING_SQ_NEED_WAKEUP` state before dispatch so this progress path is exercised.
The one-millisecond poller idle policy is not a timeout on valid I/O.

The host wake loop is deliberately a correctness service, not a qualified
low-CPU notification mechanism. Multiple wake calls can occur before the
poller clears its flag. XML records the count; ASAN test timing and wake counts
are not throughput or efficiency measurements.

The [io_uring setup specification][setup] defines the ring flags and wake
protocol. Linux's [SQPOLL implementation][sqpoll] owns poller progress, and
[ring-memory implementation][memmap] owns the supplied pages' kernel lifetime.
The shader's release/acquire edges additionally depend on the selected GPU
and registered-memory contract; the CPU ring protocol alone cannot prove them.

## Completion and release

The finite shader consumes every submitted I/O completion before exiting.
An outer PM4 system barrier precedes its host-visible completion word. Native
queue retirement precedes host verification and destruction; checked GPU queue
and memory release precede closing the ring/file descriptors and unmapping
caller pages. A native release failure retains storage that may still be
reachable. An oracle failure does not skip orderly teardown.

Direct I/O demonstrates the selected filesystem's aligned direct path into
registered system memory. It does not establish peer-to-peer access to VRAM,
the absence of every kernel/driver bounce buffer, or GPU-owned NVMe hardware
queues. The kernel retains filesystem, block-layer, DMA-mapping, and protection
ownership. Write completion and successful reload also do not establish power
failure durability: this program issues no storage flush or checkpoint commit.

## Build and qualification

The Linux x86-64 corpus is `//libamdf/cts/gpu/linux/io_uring:file_io_dynamic`; the
`_instance` invocation independently checks lifetime capability discovery.
Caller-page registration requires the advertised host-registration capability.
On the current Linux KFD path, it is available with process lifetime; instance
lifetime reports that absence instead of substituting another backing route.
The dispatch requires a qualified RDNA PM4 compute/cache profile and its exact
compiled shader product.

For a hardware runner with the required direct-I/O filesystem:

```sh
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/linux/io_uring:file_io_dynamic \
  --test_arg=--amdf_require_test=GpuFileIoTest.BufferedCausalReadWriteReload \
  --test_arg=--amdf_require_test=GpuFileIoTest.DirectCausalReadWriteReload \
  --test_arg=--amdf_require_test=GpuFileIoTest.PartialReadThenEofRetiresWithoutConsumingIncompletePayload \
  --test_arg=--amdf_require_test=GpuFileIoTest.InvalidFixedFileRetiresWithTheNativeError
```

The concurrent matrix uses the same target with these required case names:

```sh
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/linux/io_uring:file_io_dynamic \
  --test_arg=--amdf_require_test=GpuFileGatherTest.BufferedHeldFirst \
  --test_arg=--amdf_require_test=GpuFileGatherTest.BufferedHeldMiddle \
  --test_arg=--amdf_require_test=GpuFileGatherTest.BufferedHeldLast \
  --test_arg=--amdf_require_test=GpuFileGatherTest.DirectHeldFirst \
  --test_arg=--amdf_require_test=GpuFileGatherTest.DirectHeldMiddle \
  --test_arg=--amdf_require_test=GpuFileGatherTest.DirectHeldLast \
  --test_arg=--amdf_require_test=GpuFileGatherTest.InvalidFixedFileStopsIssuanceAndDrainsAcceptedIo \
  --test_arg=--amdf_require_test=GpuFileGatherTest.FailedWriteDrainsWithoutPublishingAConsumerResult \
  --test_arg=--amdf_require_test=GpuFileGatherTest.FailedReloadDrainsWithoutPublishingAConsumerResult \
  --test_arg=--amdf_require_test=GpuFileGatherTest.PartialReadThenEofDrainsWithoutConsumingIncompleteInput \
  --test_arg=--amdf_require_test=GpuFileGatherTest.FullJournalStopsPublicationAndDrainsTwoAcceptedRequests
```

A required case cannot pass by being absent or skipped. XML records physical
GPU identity, kernel release, filesystem type, direct-I/O alignment, setup
flags/features, compiled image identity, wake calls, request/round counts, and
terminal status. A generic GPU resource tag does not by itself establish the
filesystem or io_uring services needed by this corpus.
Concurrent cases additionally record peak outstanding requests, unique input
reads, deduplicated demand, peer reloads with a held reference, observed CQ
reordering, and the number of accepted requests drained after failure.

The generated CMake executable is
`libamdf_cts_gpu_linux_io_uring_file_io_dynamic_bin`; its CTest names are
`libamdf/cts/gpu/linux/io_uring/file_io_dynamic` and the `_instance` variant.
The ordinary kernel-product test checks the same authored fixtures without
activating a GPU.

Production integration still needs the caller's actual request identity,
cache admission/eviction and final-reader lifecycle, plus a matched host-issued
baseline before a performance claim. The finite fixture does not establish
fairness under sustained arrivals, per-request cancellation, or application-level
cache ownership. Independent multi-workgroup consumers have their separate
[staged-file ownership cases](file-staged.md). Checkpointing
further needs consistent tensor versions and a separate durability/publication
boundary. Those gates remain distinct from the native ownership proof.

## Matched host relay

The host-relay cases run the same compiled GPU programs and full numerical,
file, guard, lifetime and error-drain oracles through an additional control
handoff. The GPU publishes SQEs into a separate registered ring. One host owner
copies those records into the native SQ and copies native CQEs back to the
GPU-facing CQ. Acquire/release publication carries payload visibility across
that handoff; payloads themselves use the same registered I/O windows and are
never copied by the relay. Source decisions and arithmetic remain on the GPU.

This comparator retains one persistent dispatch, fixed files/buffers and
SQPOLL. It introduces neither a dispatch per request nor an application-payload
staging copy. Both paths still require idle poller wakes. Fifteen relay cases
cover the same one-credit, concurrent, held-reader and failure families as the
native path. This establishes a correctness baseline for isolating control
handoff costs; ordinary CTS execution does not measure those costs.

The `host_wait` comparator uses the same control relay without SQPOLL. Its
single CPU owner [submits available requests and waits][enter] for one actual
completion with `IORING_ENTER_GETEVENTS`. The native CQ belongs exclusively to
that CPU: the GPU consumes the separate relayed CQ and cannot remove the native
wait condition. Completion publication precedes the next host service delay.
With no outstanding I/O the host still polls GPU admission; this path removes
the kernel polling thread, not the need to observe new device requests.
The `host_poll` comparator uses the same ordinary native ring and relay, but
passes `min_complete=0` to each GETEVENTS service call. One userspace CPU
owner submits and runs kernel task work without sleeping for I/O; there is no
SQPOLL thread. This separates the cost of a relay from the cost of a second
polling CPU or a completion sleep. All three host strategies and `device_wait`
cover the complete fifteen-case correctness family. Additional direct-I/O cases
qualify 8, 64, 256 and 512 native SQ entries across all five paths; backing
scales with the actual SQE extent and returned CQ layout.

### Sleeping progress with device owned completions

`device_wait` retains the native SQ publisher and CQ consumer on the GPU,
without SQPOLL or a userspace control-record relay. One CPU thread creates,
enables and services the ring with `SINGLE_ISSUER`, `DEFER_TASKRUN` and
`TASKRUN_FLAG`. A registered nonblocking eventfd notifies that thread when
kernel completion work becomes runnable. Registration precedes the restricted
ring's enable operation; the descriptor survives until ring closure.

The helper drains the notification counter, submits available SQEs and runs
deferred work with nonblocking `GETEVENTS`, then checks the kernel-owned CQ
tail. A changed tail establishes progress even if the GPU has already consumed
every new CQE. Pending `IORING_SQ_TASKRUN` repeats service instead of sleeping:
the kernel can retain work after a bounded pass without another empty-to-ready
notification. Otherwise, accepted requests with no CQE permit an indefinite
eventfd wait. A completion racing that wait leaves either observable tail
progress, pending work, or a readable eventfd. The helper never uses unread CQ
occupancy as a wait condition and never advances CQ head.

This relies on [Linux deferred work][taskwork] notifying eventfd when the local
work list first becomes nonempty, before that work necessarily produces CQEs.
The [eventfd path][eventfd] distinguishes work notification from CQ publication.
Using `io_uring_enter(min_complete=1)` directly would be wrong here: the GPU can
consume its awaited CQE before the helper sleeps, leaving no future wake.

The finite programs issue exactly one CQE per accepted request and keep
outstanding work below ring capacity. Those contracts make SQ head versus CQ
tail an accepted-I/O predicate. They do not generalize to suppressed or multishot
completions. With no outstanding I/O the helper returns to poll GPU admission
and final completion. New GPU submissions while it sleeps can wait behind an
older I/O; eventfd does not itself notify GPU SQ publication. This strategy
removes continuous polling during I/O waits, not all CPU work or admission
latency. XML reports syscall service and eventfd wait counts; neither proves an
efficiency improvement without a matched workload measurement.

### CPU execution model

Every transport uses the GPU to generate requests and consume data. IRQ and
possible I/O-worker execution are not eliminated by any of these paths.

| Transport | Userspace CPU owner | SQPOLL CPU owner | Request dependency |
| --- | --- | --- | --- |
| `device` | Busy or paced idle-wake helper | One polling thread | No per-request userspace relay while SQPOLL is awake |
| `device_wait` | Syscall service and eventfd waits; polls GPU-only gaps | None | No SQE/CQE relay; new demand can wait behind an older I/O |
| `host_relay` | Busy or paced SQE/CQE relay | One polling thread | Both record handoffs require userspace service |
| `host_poll` | Busy or paced relay and nonblocking syscall service | None | Both handoffs require userspace service; no I/O sleep |
| `host_wait` | Relay and syscall service; sleeps for a pending completion | None | Both handoffs require userspace service; new demand can wait behind the sleep |

The busy `device` and `host_relay` configurations therefore have two CPU
polling loops, not zero or one. The shader's completion loop runs on the GPU.
A paced wake helper removes continuous userspace spinning, not SQPOLL's CPU
cost. None of these descriptions claims that every loop remains scheduled
100% of elapsed time. The host thread ID, its start/end CPUs, and SQPOLL's
thread ID and last reported CPU accompany each measured sample. IDs identify
owners; start/end CPU observations do not establish fixed affinity or exclude
intervening migration.

## Completion driven latency comparison

[file_latency_test.cc](file_latency_test.cc) runs one through four causal streams
with the same [GPU program](../../kernels/file_latency.loom) on all four transports.
Each completion immediately admits its stream's successor. File keys depend on
three words from the previous response; the host has no prepared request list.
Read/write/reload cases write disjoint scattered output blocks and reload into
separate guarded windows. Payload consumption is a three-word probe, not a
matmul. The CPU oracle checks every probe and dependency, every final payload
word, untouched guards and state, terminal native positions, and the whole file.
Error cases establish stop/drain on missing fixed files and partial reads ending
at EOF. Direct I/O can modify the unreported tail inside a requested EOF block:
Linux's [iomap completion][iomap] trims the reported count to file length after
transfer. Only returned bytes are meaningful; guards outside the entire
submitted destination remain protected and incomplete inputs are never consumed.

The six profiles cover dependent 4 KiB lookups, four independent 4 KiB streams,
64 KiB block round trips, four 4 MiB block streams, and 4 KiB arrivals separated
by 200 microseconds or 2 milliseconds. The buffered files are warm page-cache
controls. `O_DIRECT` exercises the filesystem's direct path; neither mode
establishes cold controller caches or a production model's access distribution.
The largest private file is 128 MiB. No model, raw device or existing file is
modified. File initialization and between-sample writeback flushing occur
outside timing; timed write completion is not a durability boundary.

Ordinary CTS performs eight rounds per stream without printing performance
samples. An optimized run sets `AMDF_IO_REPETITIONS` (1 through 31) and requires
an existing `BENCHMARK_LOCK_LEASE_ID`. That environment value is a launch
precondition, not proof of machine isolation; the runner's broker and host-health
record supply that evidence. The same executable and library serve all paths.
One warm-up pass precedes rotated and reversed four-path epochs; eight epochs
balance every path across each position in both directions, not all twenty-four
permutations. Paths within an epoch use the same
seed, geometry and fixed consumer count. Separate native rings retain their
own backing and registrations across retired dispatches. Fixed counts are 1024
dependent lookups, 256 consumers per independent stream or 64 KiB chain, eight
4 MiB consumers per stream, and 128/64 sparse consumers respectively.

`AMDF_IO_IDLE_MS` selects the native SQPOLL idle policy (default 1 millisecond).
`AMDF_IO_SERVICE_US` selects a requested delay between host service passes;
omitting it selects busy polling between service calls. All transports use the
same selected policy; `host_wait` additionally sleeps for actual I/O completion.
Scheduling and timer slack can exceed a requested polling interval. Wake calls
are coalesced by published SQ tail, and SQPOLL samples begin with an observed
sleeping poller. No policy value is a timeout on valid asynchronous work.
`poller_idle_ms` identifies the shared comparison setting; `idle_ms` and
`setup_flags` describe the actual ring, with zero idle time and no SQPOLL flag
for `host_wait`.

Each `AMDF_IO_SAMPLE` JSON line reports the following distinct intervals:

| Measurement | Boundary |
| --- | --- |
| `request_ticks` | First GPU SQ-tail publication through completion acquisition, payload probes and result stores; positive short retries retain the initial timestamp. |
| `device_ticks` | Initial device request preparation through final device consumption, including programmed arrival gaps. |
| `wall_ns` | Host queue publication through observed final GPU completion. |
| `host_cpu_ns` | The service thread's CPU time over the host interval. |
| `sqpoll_cpu_us` | Kernel `SqTotalTime` delta sampled around that interval. |
| `sqpoll_tail_cpu_us` | Additional poller CPU time through its observed return to sleep. |
| `submit_calls` | Non-SQPOLL submission/task-work service syscalls, including interrupted calls; `host_waits_for_io` distinguishes blocking from polling. |
| `wake_calls` | SQPOLL wake syscalls, coalesced by the published tail. |

Kernel accounting samples bracket rather than exactly coincide with the host
interval. The two CPU owners do not double count one another, but their sum is
not whole-machine CPU consumption: interrupt, worker and other kernel threads
are not attributed by these counters. Poller tail cost is reported separately,
not silently excluded from a resource-cost conclusion. The non-SQPOLL path has
no poller, so both poller CPU counters are zero; its calling-thread kernel work
is included in `host_cpu_ns`. Kernels without `SqTotalTime`, or fdinfo samples
without an observable SQPOLL owner, produce JSON `null` for the affected CPU
intervals. The I/O workload and its correctness oracles still run; unavailable
accounting cannot support a CPU-cost comparison. Malformed fields and failed
fdinfo reads remain failures. None of these counters measures joules.

Clock conversion uses the nominal `AMDGPU_INFO_DEV_INFO.gpu_counter_freq`, not
GPU operating MHz or KFD's host-clock frequency. Full-width `GET_REALTIME`
samples at the shader's start and end bound the interval in the same clock
domain as its compact request timestamps. The interval must fit the records'
32-bit width; crossing a low-word wrap is valid, running beyond that width or
observing a backwards clock is not. Only the two endpoint samples use the
wide instruction; per-request samples remain 32-bit. No shared epoch with
`AMDGPU_INFO_TIMESTAMP`, host/device clock calibration, or measured clock rate
is claimed. The
[clock reference](../../../../../docs/reference/amd/gpu/observability.md)
describes those domains. Request timestamp and transcript overhead is shared
by all paths; results characterize this instrumented native boundary, not a
complete serving engine or contention with independent compute workers.

The runnable target is `:file_io_dynamic_bin`; the ordinary `:file_io_dynamic`
target supplies ASAN correctness. Optimized builds use the repository benchmark
flags and an explicit runner ISA. A measurement invocation supplies the built
library with `--amdf_library`, selects the physical GPU, and filters
`FileModes/GpuFileLatencyTest.*`. The file's XML retains device, filesystem,
kernel and compiled-image identity. Performance results require the matched
JSON samples plus artifact digests, compiler flags, broker lease, machine
state and the exact policy settings; a successful CTS run alone is insufficient.

## Scheduled demand and bounded backing

`FileModes/GpuFileDemandTest.*` uses [file_demand.loom](../../kernels/file_demand.loom)
to measure demand latency with 32, 64, 128 or 256 logical arrivals in each burst.
Four bursts form an epoch. Their offered times are fixed before execution and
do not wait for I/O completion or free backing. The primary interval is
`ready - arrival`: scheduled GPU demand through the consumer's first returned
payload probe. `admitted - arrival` exposes queueing before a credit is acquired;
`released - ready` includes deliberately retained readers. Physical read, write
and reload intervals remain separate from logical request latency.

| Profile | Physical operation | Offered work and ownership |
| --- | --- | --- |
| `lookup32/64/128/256` | 4 KiB read | Matching burst and credit counts, four bursts 1 ms apart. |
| `lookup_backlog` | 4 KiB read | 256-demand bursts, 32 credits, every seventh reader retained for 200 µs. |
| `lookup_staggered` | 4 KiB read | 32-demand bursts 40 µs apart, 64 credits, independent GPU arithmetic. |
| `expert_tiles` | 256 KiB read | 256-demand bursts over 16 keys, 32 credits, every seventh reader retained for 2 ms. |
| `expert_pressure` | 256 KiB read | 256-demand bursts over 64 keys, eight credits, retained readers and independent arithmetic. |
| `kv_blocks` | 64 KiB read/write/reload | 256-demand bursts, 64 credits, scattered output blocks and a separate reload bank. |

The GPU hashes each offered key seed and looks up a direct key-to-credit map.
Matching immutable demands join an existing read, including already completed
backing still retained by a reader. A ready queue rotates held readers so they
cannot stop other ready consumers or completion processing. Only the last
reader returns a credit to the free list. KV chains retain exclusive ownership
through read, write and reload; they do not deduplicate mutable blocks.
One coordinator owns these structures in 53,312 bytes of workgroup-local
storage: a summary, up to 256 credits, a 1,024-entry key map, and eight private
words for each of at most 1,024 logical readers. The reader state holds links,
seed, retention interval, admission/readiness/release timestamps and flags.
None is a CPU/GPU mailbox. Keeping private ownership state local separates its
accesses from the native ring's system-scope visibility operations. Immutable
readers obtain physical timestamps from their live credit; exclusive KV
readers already own their phase journal. Neither path reads the global output
journal to recover those facts.

The SQ/CQ, payloads, and result journal remain globally visible. Final private
state is exported for host verification only after all accepted I/O and
readers retire; dispatch wall time includes initialization and export. The
scheduled clock starts after table initialization and ends before export;
reader state is initialized on admission inside timing. This measures a
finite storage-channel witness, not independently dispatched model matmuls
or a selected cache API. Its large LDS reservation and finite journal bound
are explicit costs of this single-workgroup witness.

All five measured configurations use identical GPU programs, offered manifests
and payloads: native SQPOLL with busy or requested 50 µs host service, busy
relayed SQPOLL, ordinary host polling, and ordinary host completion waits.
The paced policy changes only the native idle-wake helper. Ten measured epochs
balance all configurations over each order position in both directions after
one warm-up epoch. `AMDF_IO_REPETITIONS` selects the fixed measured count under
the same benchmark-lease requirement as the causal cases. These scenarios
fix SQPOLL idle at 1 ms and their own five-policy matrix; the causal cases'
`AMDF_IO_SERVICE_US` and `AMDF_IO_IDLE_MS` overrides do not alter them.

Each `AMDF_IO_DEMAND` line contains all requests, with column names and times
relative to the same GPU reference-clock origin. Reported `burst` is offered
concurrency, not observed device queue depth. `peak_outstanding`, `peak_credits`,
physical request count and duplicate count show what actually happened.
Outstanding means submitted but not yet GPU-reaped; it includes completed CQEs,
so it is not an NVMe hardware queue-depth counter. Timing
can change the overlap of duplicate readers and therefore physical read count.
The comparison preserves identical logical work rather than assuming equal
physical work. Thread IDs, CPU time, syscall counts and service policies retain
the execution model described above. CPU time is neither request latency nor
an energy measurement.

The optional second GPU queue performs checkable integer arithmetic until the
I/O owner publishes its terminal stop. Its startup completes outside timing;
there is no CPU dispatch per request. This establishes simultaneous progress,
not representative matmul saturation. The host verifies every consumer result,
physical phase order, previous-generation release, final backing and file
contents, guards and native ring retirement. Absent files and a partial read
ending at EOF exercise stop/drain without consuming incomplete payloads.
The largest scheduled-demand file is 40 MiB; registered payload is at most
16.3 MiB, referenced by three retained rings. Writes are not durability claims.

[taskwork]: https://github.com/torvalds/linux/blob/v7.0/io_uring/tw.c
[eventfd]: https://github.com/torvalds/linux/blob/v7.0/io_uring/eventfd.c
[model]: https://github.com/axboe/liburing/blob/master/man/io_uring.7
[setup]: https://github.com/axboe/liburing/blob/master/man/io_uring_setup.2
[enter]: https://github.com/axboe/liburing/blob/master/man/io_uring_enter.2
[sqpoll]: https://github.com/torvalds/linux/blob/master/io_uring/sqpoll.c
[memmap]: https://github.com/torvalds/linux/blob/master/io_uring/memmap.c
[iomap]: https://github.com/torvalds/linux/blob/master/fs/iomap/direct-io.c
