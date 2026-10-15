# XDNA HAL

The `xdna` driver runs native Loom array programs through libamdf. It owns
device discovery, native contexts, executable storage, buffers, direct and
recorded queue operations, and completion. The [image loader](image/README.md)
admits `.xdna` ELFs and applies their declared storage and binding contracts.

## Running a Loom scenario

Enable the shared AMD HAL driver. This selects libamdf automatically; the default
Loom product already includes XDNA emission and the VM reference target:

```sh
python dev.py bazel configure -DIREE_HAL_DRIVER_AMD=ON
python dev.py bazel run //loom/src/loom/tools/iree-test-loom -- \
  loom/src/loom/tooling/target/amd/xdna/test/hal_execution.loom --device=xdna
```

The scenario compiles a finite multiply pipeline for the selected device and
compares its outputs with an independent VM implementation. Trial inputs vary;
tensor views exercise nonzero binding offsets, shared input storage, and output
guards. The device's exact target profile selects the compiler's array profile.

`xdna://0?columns=1` selects discovery ordinal zero and a one-column native
context. The default is one column. The context width must equal the compiled
array's width; a wider context does not implicitly pad the program. Endpoint
identity, firmware ABI, and geometry are checked when an executable is loaded.

The [native execution requirements](../../../../../../../libamdf/docs/xdna.md#native-requirements)
must be available before HAL device creation succeeds. On Linux, libamdf
qualifies the required direct instruction transport with a no-effect transaction
when creating the first queue for a context. Unsupported transport produces an
explicit update-driver diagnostic during device creation, before loading or
submitting application work. Install the NPU driver and its matching firmware;
discovering the device or successfully creating a native context is insufficient.
This startup check runs once per context and adds no steady-state submission
work. Optional profiling services have separate capability requirements.

The normal tool accepts finite `pipeline.def<kernel>` subjects with buffer
launch bindings and no leading specialization arguments. Specialization
arguments need a compiler preparation step that binds the scenario's
configuration values before emission; the adapter reports `UNIMPLEMENTED` for
that form. Runtime scalar bindings and invocation results are absent from the
current native image ABI.

## Execution and ownership

A device provisions one dispatch/transfer queue and one time-sliced native
context. Each accepted operation captures its transient arguments and retains
its resources. Semaphore dependencies express ordering. A consumer submitted
before its producer waits without occupying a native execution slot.

Queue-owned fixed-block arenas capture operation metadata, placed semaphore
timepoints, transfer descriptors, and dispatch bindings. UPDATE data is split
across reusable payload blocks and copied through one target mapping. The pools
grow for a new workload shape and then reuse those blocks; repeating a warmed
shape and depth performs no host allocation or free.

Each dispatch first attempts a nonwaiting claim on native publication. When the
persistent observer is active, the pending ring has capacity, and every wait is
already reached or proved by the exact accepted frontier, the calling thread
prepares and submits the command directly, then commits its accepted frontier
and pending-ring entry before returning. Full capacity, unsatisfied waits, and
claim contention use the private publisher instead. Inexact causal state
disables future FIFO proofs but does not defer an operation whose waits are
already reached. libamdf reports its own full publication window as `BUSY`; the
HAL retains the prepared invocation and retries only after checked native
progress. The underlying OS submission can still wait for device wake or native
credits, so a direct caller can pay that latency. The private publisher isolates
that cost on every queued route.

The shared proactor owns queued causal admission, deferred native acceptance
commits, and unattended checked retirement. Exact blocking waits on a local
submitted signal instead wait for its libamdf point and publish retirement on
the calling thread, including before a standalone device has been assigned a
topology frontier. Accepted invocations occupy a bounded ring sized to the
native queue's prepared capacity. Each pending invocation owns exclusive
mutable command and binding storage; checked retirement returns that storage
for reuse. Immutable backing is shared when its static address references also
point to shared allocations. Binding updates publish only their patched ranges.
Every dispatch executes invocation zero to establish the array state; time
slicing does not promise resident tile state across submissions.

Eligible host operations run on an independent private worker and never advance
the native queue's completion frontier. Neither mapped transfer work nor native
submission can block the shared proactor. Private services return queued results
through placed proactor operations; queue-local synchronization serializes them
with caller-owned direct acceptance and retirement. A dependent dispatch whose
producer is still entering the native queue retries after that producer's
acceptance; other unresolved waits register their ordinary semaphore timepoints
immediately. The public queue call always captures its arguments before either
direct or queued publication takes ownership.

Direct fill, update, copy, upload, download, and queue barriers are supported.
Execution/access dependencies and global system visibility need no additional
XDNA command. Ranged memory effects return `UNIMPLEMENTED` because this queue
family has no native range-maintenance action and cannot promote a prepared
range without weakening its contract. Transfers use mapped native storage with
the required cache operations. Allocated buffers report the provider's actual
coherence properties. Loom correctness scenarios keep canonical host data in
coherent heap storage and stage device copies; buffer offsets and aliases
survive upload and readback.

Queue notifications are wake hints. One cold-registered persistent proactor
source consumes each event, refreshes the native queue's checked retirement and
terminal outcome, and requests a new one-shot native notification for the
oldest remaining point. No per-completion source registration is required. An
exact host waiter marks its retirement claim before entering
libamdf. A concurrent wake records its consumed one-shot hint atomically and
returns without contending on the caller's publication locks. A
completion-drain mutex preserves FIFO publication between direct callers and
unattended notification processing, and is never held across a native wait.
Ordinary retired work releases payload resources before signaling HAL
semaphores. An operation-held device reference keeps the queue's capture pools
alive through inline signal callbacks, then the arenas are returned before the
final device release.

Final device release places shutdown on the proactor owner. It joins the two
idle services, destroys the native queue, and begins terminal source
unregistration outside the source callback. The unregistration receipt places a
second operation before releasing the event, queue, context, and proactor entry.
If observation fails without proving retirement, the queue reports the error,
fails completion edges, and retains the unsafe-to-release ownership graph.
Teardown failures similarly report and retain ownership without aborting the
application or maintaining a live-object registry.

The shared AMD memory backend supplies pools, prepared external-memory access
and stable reservations according to native capabilities. File operations and
host calls participate in ordinary semaphore dependencies. Atomic queue
operations return explicit unsupported statuses.

## Reusable command buffers

Command buffers record array dispatches, fill/update/copy operations, and
barriers. Records retain direct resources and capture update bytes. Indirect
binding tables are captured separately for each execution, so the same recording
can run concurrently with different buffers. Replay frames pack the captured
table, dispatch binding scratch, and bounded native descriptor/invocation arrays
into one allocation with `IREE_STRUCT_LAYOUT`. Completed frames return to a
recording-owned cache. Stable shape and concurrency reuse these frames and the
existing prepared invocation cache without allocating or repatching unchanged
bindings.

One queue execution owns the external wait/signal lists and a cursor into the
immutable recording. Its publisher fills the free native ring window, resuming
the remaining commands after checked retirement returns capacity. There is no
recorded-command count limit or host wait between adjacent array dispatches.
Consecutive dispatches use one libamdf command-array submission and one checked
native completion point. Recording retains dispatch-run lengths so publication
can select a batch without rescanning the commands. Batches shorter than six
dispatches use ordinary packets to avoid native command-list setup latency.
Host transfer phases run after the preceding native prefix retires and resume
native publication after their cache work completes. This preserves explicit
barriers while using the same queue services as direct operations.

The native queue advertises its array capacity. Linux accepts up to 64 commands
per array and shares the existing packet pool between singleton packets and
chain roots with children. Windows Direct-protocol queues accept up to 36
commands per list and reuse seven 4 KiB cells in the instruction allocation's
reserved prefix. Windows metadata protocols retain their single-command
capability. Neither provider allocates native backing on submission. Aggregate
packet/list credits can fill before the public pending-submission bound; `BUSY`
retains the prepared array until checked retirement returns capacity.

The submitted semaphore frontier becomes visible only after the recording's
terminal native tail is accepted. Rejection of a later array retains the
accepted prefix until retirement, then reports failure. Every invocation in an
accepted array remains owned through its root's fence and result check. Native
acceptance does not promise that all commands succeed. Loss of retirement
observation retains the native-reachable graph as for direct dispatch.

## Benchmarking and profiling

### Host work in Tracy

Build the normal tool with the Tracy provider, then start a matching collector
before running it:

```sh
python dev.py bazel build --config=opt \
  --//runtime/config/base/tracing:provider=tracy \
  //loom/src/loom/tools/iree-benchmark-loom
tracy-capture -a 127.0.0.1 -p 18086 -o xdna.tracy -m 5
```

In another terminal, run the just-built executable:

```sh
TRACY_PORT=18086 TRACY_NO_EXIT=1 \
  bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom \
  loom/src/loom/tooling/target/amd/xdna/test/hal_execution.loom \
  --device=xdna --benchmark=@serial_array_latency --batch-size=6 \
  --iterations=8 --warmup-iterations=2 --profile-final-batch=false
```

The same build option enables these zones in `loom-serve`. The
[runtime tracing guide](../../../../base/tracing/README.md) describes tool
versions and capture lifetimes; the
[serving observation workflow](../../../../../../../loom/docs/src/serving/workflows/observe-model-server.md#inspect-host-work-in-tracy)
connects these backend stages to request preparation and output processing.

XDNA zones use the `iree_hal_amd_xdna_` prefix. The table abbreviates that
prefix; numeric annotations appear in the listed order.

| Zone | Work and annotations |
| --- | --- |
| `operation_try_direct_publish` | Admission decision only. Text distinguishes caller-direct publication, queue-state or publisher contention, unavailable native credits/window, and an unproven wait frontier. |
| `operation_admit` | Deferred admission on the queue owner. Text identifies waiting for producer acceptance or a semaphore; the scope ends after registering that interest. |
| `command_buffer_acquire_execution`, `execution_prepare_batch` | Capturing a replay frame and preparing its next native array. Cold replay growth is marked; batch preparation reports its command count. |
| `function_prepare` | Address resolution and exclusive invocation selection, with function name and executable ID. `bindings unchanged` means no dynamic patch or cache publication. Nested `invocation_allocate` and `patch_bindings` expose cold growth and changed bindings. |
| `native_submit` | The libamdf call, annotated with command count, native submission point, and native status. OS wake or credit latency can be inside this call; it does not measure execution on the array. |
| `queue_commit_publication` | Publishing the accepted frontier and pending ownership; the value is the accepted command count. |
| `queue_apply_checked_status`, `operation_complete` | Checked retirement, resource release, and completion signaling. Checked status reports the retired native point and number of commands removed from the pending ring. `wait_retire` starts after a caller's native wait returns; `queue_native_ready` handles unattended notification progress. |
| `memory_finish_acquire`, `operation_prepare_alloca_storage`, `operation_execute_dealloca` | Pool readiness, backing attachment, and release, with request/buffer counts. Outcome text distinguishes satisfied or absent reuse frontiers, reuse-frontier waiting, and capacity/budget waiting. An absent frontier can describe cached storage whose prior use completed; it does not imply a new native allocation. Native backing work remains under the allocator worker's slab-batch zone. |
| `operation_execute_host` | Mapped transfers, host calls, and file preparation/publication, with operation kind and completed payload count. Asynchronous file I/O is outside this scope. |

Worker notification sleeps and native completion waits have no XDNA parent
scope. Every CPU zone ends on the thread that began it; callbacks publish their
own active work. The gap between submission and completion includes device
execution and notification delivery, and is not a measured NPU duration.
A resident model compiled into one native invocation appears as one dispatch;
these host scopes do not expose its internal layers. Physical allocation events
belong to the shared memory provider, so transient wrappers and pool reuse do
not count the same backing again. Tracing changes timing; performance comparisons
use separate optimized runs with tracing disabled.

### HAL profiling

The ordinary benchmark tool uses the same XDNA provider as correctness tests:

```sh
python dev.py bazel run --config=opt \
  //loom/src/loom/tools/iree-benchmark-loom -- \
  loom/src/loom/tooling/target/amd/xdna/test/hal_execution.loom \
  --device=xdna --benchmark=@serial_array_latency --batch-size=64 \
  --profile-data=queue-events,host-execution,counters
```

`QUEUE_EVENTS` describes public submissions, dependency deferral, payload
counts, and first execution readiness. `HOST_EXECUTION_EVENTS` measures native
command preparation/publication and mapped host work. Native spans finish when
publication returns; they exclude NPU execution and completion waits. Callback
spans cover the callback body until it returns. File spans cover preparation and
final cache publication, excluding asynchronous file I/O.
`LIGHTWEIGHT_STATISTICS` selects these two host families.

`COUNTER_SAMPLES` records `invocation_cycles`: the partition-shim timer delta
inside the compiler-declared invocation interval. Generated pipelines bracket
worker activation through completion waits; array programs bracket their control
phase after resident activation. Authored configuration declares the interval
with `measure.begin` / `measure.end`. The interval cannot contain a timer reset,
and ordinary executable commands contain no result-producing instructions.
Missing interval metadata fails the capture while ordinary execution continues.
These are cycle counts, not device-wide busy cycles or elapsed nanoseconds: no
shared epoch or stable operating frequency is implied. Device dispatch timelines
and clock correlations remain unsupported and requests fail explicitly.

Counter captures include executable/function metadata for attribution;
`EXECUTABLE_METADATA` also works alone without native instrumentation. Use the
benchmark's reported `artifact_path` with `iree-profile counter --format=jsonl`
to inspect per-function aggregates, or add `--counter_samples` for individual
invocations. The benchmark's dispatch-time summary does not convert these cycle
counters into a duration distribution. The counter name `invocation_cycles` may
be requested explicitly, and capture filters select functions, command buffers,
command indices, physical devices, and queues.

Profiling begin/end follows the HAL idle-device contract. Begin allocates one
packed host session with 4096 records per enabled host stream; completed records
are appended without allocation or sink calls. Flush can run with work in flight
and sends stable host snapshots outside queue and recording locks. Full host
rings report dropped records through `TRUNCATED` chunks. Failed host sink writes
retain the snapshot for retry.

Native capture reserves a finite result destination and a placed attribution
ledger for at most 4096 commands, reduced by native capacity. Attachment occurs
once at begin; collection and removal occur at idle end. Flush neither collects
native results nor recovers this budget. Selected prepared commands reserve
capacity once, including across native BUSY retries; only accepted commands enter
the result ledger. Capacity exhaustion stops adding timer instructions and
preserves workload execution and all retained samples. End emits the retained
data followed by a `TRUNCATED` chunk with the number of accepted selected
dispatches that could not be captured; overflow alone succeeds. Unmatched
dispatches, BUSY retries, and rejected commands do not count as dropped samples.
A rejected command conservatively consumes its reservation. Metadata-only
capture reports truncation without a count because the missing executable
identities were not retained. Missing or extra native records fail the capture
before cycle samples are published.

The first measured use of each exclusive invocation allocates a reusable command
copy containing two fixed timer instructions. Unchanged warm bindings reuse the
copy without allocation, patching, or cache publication. Changed bindings patch
only their admitted address fields. Native arrays and queue scheduling remain
unchanged; there are no per-command result controls, waits, or extra thread hops.
End retains executable metadata until serialization, then releases session
resources even if a sink fails. An ambiguous native result-control failure
retains the result backing, arena, context, and native parents with a diagnostic.
Disabled profiling performs no clock reads or profile allocation.

## Verification

Native correctness suites declare an XDNA hardware requirement:

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amd \
  //runtime/src/iree/hal/drivers/amd/xdna/cts:native_test \
  //runtime/src/iree/hal/drivers/amd/fabric/cts/xdna:virtual_buffer_test \
  //loom/src/loom/tooling/target/amd/xdna:hal_execution_test
```

The native suite exercises submission capture, rebinding, guarded buffer views,
consumer-before-producer dependencies, direct transfers, file I/O, failure
propagation, first-device-writer visibility and allocation properties. Its
reservation case builds a pool from queried source requirements, replaces
retained backing at stable NPU addresses, dispatches immediately after the
mapping dependency and checks final detach. The fabric suite exercises composed
CPU/GPU/NPU replacement through independently queried mapping groups. Each uses
the source routes and queue capabilities advertised by the selected devices.
The Loom suite exercises the public tool, compiler, HAL, physical execution,
readback, and numerical oracle together.

## Queue benchmarks

[`cts:queue_benchmark`](cts/queue_benchmark.cc) measures the production HAL queue
with prepared native programs and buffers. Each case creates its device, pool,
executables and semaphores outside timing and reuses them across iterations.
Warmup precedes measurement. Every dispatch includes the program's independent
array initialization; native context lifetime does not promise that tile state
survives time slicing.

The names below follow the `XdnaQueueBenchmark/` prefix:

| Scenario | Timed interval |
| --- | --- |
| `ReadySubmitOnly` | One HAL dispatch with no unresolved waits. Checked completion is outside timing. |
| `ReadyEndToEnd` | One ready HAL dispatch and its completion wait. |
| `UnresolvedSubmitOnly` | Acceptance of a dispatch waiting on an unreached host gate. Gate signaling and completion are outside timing. |
| `StreamingSubmitOnly` | Acceptance of multiply dispatches chained by semaphore dependencies. The terminal wait is outside timing. |
| `StreamingEndToEnd` | The same dispatch chain and its terminal wait. |
| `SwitchingSubmitOnly` | Acceptance of dependent dispatches alternating multiply and add executables. The terminal wait is outside timing. |
| `SwitchingEndToEnd` | The same alternating chain and its terminal wait. |
| `RecordedStreaming*`, `RecordedSwitching*` | The corresponding batch submitted through one reusable command buffer. |
| `MeasuredStreaming*`, `MeasuredRecordedStreaming*` | The corresponding streaming batch with native invocation counters enabled. |

These are HAL operation costs, including capture, causal admission and native
publication or deferral. Completed rows also include checked retirement and
HAL semaphore publication. They are not isolated libamdf entry-point timings.
Inputs are prepared before measurement and no row includes host readback, so
end-to-end here means dispatch through HAL completion, not an entire
host-to-host application pipeline. Numerical output and memory lifetime are
checked by the native CTS above.

Streaming and switching times are per batch; the benchmark's numeric argument
gives its dispatch count. Divide elapsed time by that count for per-dispatch
comparisons. Their item rate already counts individual dispatches. Ready and
unresolved rows measure one dispatch per iteration.

Measured variants begin capture before warmup and collect results after timing.
They use one- and eight-dispatch batches with 32 fixed iterations so the warmup
and measured commands fit the finite native result budget. Warmup materializes
the optional command copies. Timed submissions reuse those copies, and result
collection validates the native protocol before discarding the serialized
records. Compare them with matching unmeasured cases at the same iteration count
to isolate warm capture costs; neither case includes session setup or teardown.
An unavailable native result service or insufficient permission skips the
measured cases; execution, collection, and protocol failures remain errors.

```sh
iree-bazel-build --config=opt \
  --//runtime/config/hal:drivers=task,amd \
  //runtime/src/iree/hal/drivers/amd/xdna/cts:queue_benchmark
bazel-bin/runtime/src/iree/hal/drivers/amd/xdna/cts/queue_benchmark \
  --benchmark_min_time=200x --benchmark_repetitions=7 \
  --benchmark_out=xdna-queue.json --benchmark_out_format=json
```

Fixed counts bound submissions even when timing excludes completion. Sustained
measurements hold runtime power active and exclude competing hardware use;
deployment measurements retain normal idle/wake policy. The
[libamdf measurement contract](../../../../../../../libamdf/docs/xdna.md#power-policy-and-measurement)
defines that distinction. Report the build identity, power policy, repetition
spread and raw results with any comparison.

`XdnaQueueMemoryBenchmark/` measures `queue_alloca` → recorded execution →
`queue_dealloca` for one transient output, with resident inputs and a reusable
recording with an indirect output binding. `Cached*` returns and reacquires
backing through an explicit one-entry slab cache. `Fresh*` retains no idle
backing and exercises native allocation on the memory worker. `CapacityWait*` first occupies the
pool's one-output budget, submits the dependent chain, and then releases the
held reservation through `queue_dealloca`; that extra release is timed, while
preparing the held reservation is not. This measures recovery after capacity
returns, with no artificial delay or polling loop.

All six rows reuse a single semaphore across iterations. `SubmitOnly` excludes
the terminal completion wait; `EndToEnd` includes it. Each iteration is one
complete output lifetime, with numerical checks before and after measurement.
`native_slabs_per_iteration` counts actual backing-pool acquisitions, including
the pressure setup, so cached reuse and native growth remain distinguishable.
The rows use 128 fixed iterations and stay outside the ordinary smoke filter:

```sh
bazel-bin/runtime/src/iree/hal/drivers/amd/xdna/cts/queue_benchmark \
  --benchmark_filter=XdnaQueueMemoryBenchmark/ --benchmark_repetitions=5 \
  --benchmark_out=xdna-memory.json --benchmark_out_format=json
```

The resource-tagged smoke target exercises four representative dispatch and
profiling cases without collecting performance evidence:

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amd \
  //runtime/src/iree/hal/drivers/amd/xdna/cts:queue_benchmark_test
```
