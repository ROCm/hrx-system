# Source-defined autoregressive serving

`//experimental/loom_serve/text:server` and `:generate` are model-independent
executables. They have no dependency on a model package or embedded source
catalog. `--model=/path/to/package` is required. Model files are JIT inputs,
not build inputs to these binaries.

The [tiny source fixture](../models/fixtures/text/README.md) exercises retained
rows, shared pages and HTTP continuations with a generated 76-byte checkpoint.
The [Qwen package](../models/qwen/README.md) is the real-weight implementation
of this contract. Its numerical kernels, stage composition, storage geometry,
checkpoint policy, tokenizer framing and chat policy live in Loom source.
Its remaining C programs are test-only numerical/replay observers.
The [image runner](../image/model.h) provides a separate source-defined
diffusion contract over the same JIT, weight and execution infrastructure.

## First output and first port

From the repository root:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/text:server -- \
  --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --rows=4 --context_capacity=32768 --pool_capacity=65536 \
  --prefill_capacity=512 --draft_depth=3
```

The real Qwen sources have been exercised on gfx1100 and gfx1151. Repository
build/execution policy still applies; a successful build on another host is
not a GPU check.
The [reproduction packet](../docs/README.md) supplies assets, request examples
and the generated-tool/retained-session witnesses.

A new text model supplies `prepare.loom`, `control.loom`, `sources.txt`,
the providers named by that catalog, and its weight-policy source. Catalog
paths are relative to the package; shared motif paths must travel with it.
There is no native registration list. Editing the package changes the next
process without rebuilding the server.

The first numerical slice is one real-weight block through its actual command
root and source catalog, then a complete retained continuation. After that,
the same source runs under HTTP admission and mixed scheduling. Copying the
Qwen package is a starting scaffold, not a model conversion: its dimensions,
layer graph, layouts and checkpoint names must be replaced with the target
model's actual contract.

## Execution contract

This is a bounded packed-autoregressive runner, with optional three-token
speculation. It is not an arbitrary model-graph interpreter.

* One residency shares a VM process, cached commands and weights. Sessions are
  data rows. Calls to the shared invocation are serialized. Private command
  workspace comes from the device execution pool, shared across models.
* The host chooses ready spans and a cached token/span shape. Source creates
  device packets and chooses commands. Attention and recurrent state layout
  are source concerns; the host owns physical page IDs and completion credit.
* Known input, one-token decode and speculative verification can share an
  epoch. The runner currently supports 1–16 resident rows and a logical
  context ceiling of 262144 positions. Token capacity may reach that ceiling;
  the source package owns any tighter kernel or packet limit. The current
  Qwen package supports up to 512 input tokens per epoch and rejects larger
  shapes in its bootstrap. This is not a generic serving limit.
  Physical pool capacity is independent, up to 4194304 positions, divisible
  by the source-declared page size.
* Drafting is optional. `--draft_depth` specifies up to seven proposals after
  each known anchor; zero disables drafting. The source validates its supported
  depths and implements proposal, target verification, accepted-state publication
  and draft catch-up. Qwen's MTP path uses depth three.
* `--continuation_epochs=2` pre-issues a second device-fed cohort. The host
  still joins each cohort; this is not an autonomous persistent device loop.
* Pooled execution reserves stable device addresses at startup. HTTP admission
  backs each request's page-rounded completion high-water, including speculative
  slack and its recurrent writer, while pinning the required parameters. This
  is the turn's requested extent, not the session's entire context ceiling.
  Shared explicit prefixes count once; excess work queues or refuses before SSE.
  Explicit model-level suspend/resume is described below; HTTP admission does
  not yet select sessions for offload. Fixed backing remains an explicit
  comparison and device-sanitizer configuration.

The automatic shape catalog crosses token classes up to `--prefill_capacity`
with span classes up to `--rows`, including exact odd endpoints. Repeated
`--epoch=tokens:spans` replaces that catalog. Source may reject unsupported
specializations; the runner never substitutes another model or stale code.
Input staging is allocated once for the largest configured cohorts and
source-declared dense token views. It is shared by the serialized model
invocation; rows retain only their pending prediction, not padded host inputs.

### Native ownership boundaries

The [service](service.c) owns HTTP claims, queued admission, session phases and
one asynchronous completion handoff. Neither device readiness callbacks nor
CPU release callbacks destroy model state; they wake this serialized owner.

The [checkpoint registry](checkpoint_registry.h) owns named endpoint records,
retained canonical transcripts and completed snapshot-byte accounting. The
service publishes a name only after successful pin readiness and tracks queued
borrowers before permitting deletion. Failed device readiness leaves native
handles with model retirement, rather than releasing possibly active storage.

The [epoch planner](epoch.h) receives snapshots of already-credited rows and a
selected packing shape. It produces retained spans, output limits and an
optional device-fed continuation without allocation, synchronization or HTTP
access. Packed completion consumes the model program's authoritative progress;
isolated controls publish only the span that actually completed. Telemetry
observes that progress without deciding request advancement. This separation
lets packing and continuation choices be tested directly while the source
fixtures verify real HTTP output, retained history, cancellation and removal.

## Bootstrap and storage

The cold entry in `prepare.loom` is:

```text
prepare(prefill:i64, context:i64, pool:i64, rows:i64, pins:i64, draft_depth:i32,
        shapes:buffer, shape_count:i64, weights:buffer)
  -> (prefill_capacity:i64, shared_stage_count:i32, state:buffer,
      terminal_spelling:buffer, allocations:buffer, row_views:buffer,
      target_origins:buffer, draft_origins:buffer, geometry:buffer)
```

`shapes` contains little-endian i64 `{token_capacity, span_capacity}`
pairs. The source declares command roots, configuration overrides and
checkpoint bindings through the [preparation capability](../runtime/preparation.h).
The existing `input.require` capability accepts source-specific configuration
checks during bootstrap; tokenization is available only in the warm program.
The first `shared_stage_count` stages have identical target parameter
placement and six persistent bindings. All stages use one checkpoint domain,
with at most seven explicit dynamic bindings plus private workspace when present.
The runner inserts the reflected transient binding; the model VM never receives
it. Allocation, command execution and deallocation have explicit queue edges,
so draft/verify/catch-up stages can reuse the same backing without host waits.
Auxiliary roots may share existing tensors or add draft tensors.

`state` is opaque to C and survives cold-VM teardown. The terminal spelling
is looked up in the supplied tokenizer. The current token protocol has one
terminal token ID. Geometry is not inferred from checkpoint filenames.

Native storage records are little-endian i64. Initial origin-table payloads
are opaque; the model source owns their record format:

| Result | Records and meaning |
| --- | --- |
| `allocations` | Eleven `{byte_length, alignment, initial_zero_length}` records, in the order below |
| `row_views` | `rows * 4` pairs `{offset, length}` into the state arena: control, dense attention, input IDs, progress |
| `target_origins` | Source-defined payload uploaded to the target row table |
| `draft_origins` | Source-defined payload uploaded to the draft row table when enabled |
| `geometry` | Header `{page_tokens, page_map_byte_origin, draft_carry_stride, feedback_split, recurrent_origin, recurrent_stride, recurrent_length, recurrent_count}`, then zero or more cache-region records described below |

The eleven allocation slots are residual, state arena, packed metadata,
target origins/page map, packed input IDs, ordinary selected IDs, draft carry,
committed metadata, verification feedback, draft cache, and draft origins/page
map. The first two are present; packed slots 2–5 exist with epoch shapes;
draft slots 6–10 exist with drafting. Workspace size and alignment come from command
reflection, independently of this list.

Each enabled origin payload must fit its table allocation. In pooled mode,
the payload precedes `page_map_byte_origin`; the following row-major U32 map
holds `rows * ceil(context / page_tokens)` physical block IDs and must fit
the same allocation. Native code validates those byte boundaries without
interpreting the model's origin records. Target and draft payload formats may
differ. An empty payload needs no initial upload; the source establishes any
contents its kernels consume. Payloads for disabled allocation slots are not
uploaded.

Recurrent slots occupy `recurrent_length` bytes at
`recurrent_origin + slot * recurrent_stride` in the state arena. They are
disjoint from each other and all row-private views, and fit its initialized
private extent. With `pins=0`, at least `rows` slots are available. Opt-in
checkpoints require at least `2*rows+pins` slots so every shared reader and
private writer can survive an entire queued cohort. This is virtual headroom
on elastic backing, not eagerly committed physical memory. A row acquires a
slot on activation or restore and releases it after retired reset or successful
suspension; an idle row owns none. Slot identity is independent of row identity.
Views are created once, and elastic physical backing is committed on demand.
Rebinding uses the source encoder below, not native device-record offsets.

Pooled sources append cache regions, each five little-endian i64 values:
`{allocation_slot, byte_origin, plane_count, plane_stride, block_bytes}`.
A physical block ID selects `block_bytes` consecutive bytes in every plane:
`byte_origin + plane * plane_stride + block_id * block_bytes`. Regions describe
actual kernel addressing; native code has no layer count or KV element type.
Dense sources have no regions. Region extents must fit their allocation and
each plane's block range must fit its stride.

CLI execution defaults to `--pool_backing=elastic`: pooled source allocations
containing cache regions reserve stable addresses and commit physical slabs as
rows/pages grow. Private row views initialize on first use. `--slab_bytes`
selects coarse physical granularity (2 MiB by default; zero queries the
allocator recommendation), independent of logical token-page size. The explicit
`--pool_backing=fixed` comparison path backs all source state at startup and
supports device address sanitization, which excludes user VMM. Unsupported VMM
is an error, not a silent change of storage policy. Host ASAN remains usable.
Elastic backing also reserves stable parameter roots in the same physical
allocation domain. Heartbeat `elastic_state` and `elastic_parameters` account
mutable state and weights independently; transient workspace is separate.

The caller creates one `loom_serve_device_t` and passes it to each text or
image model constructor. That owner holds the common physical pool, queues,
execution timelines, and profiling session. It outlives all models; one host
owner serializes their calls and residency transitions. `--memory_bytes` bounds
physical parameter/state commitment across those models, not virtual address
space or shared execution workspace. Creation compiles, indexes, and
records without loading parameter payloads. Explicit activation can warm a
model before accepting requests; otherwise its first inference loads weights.

`loom_serve_text_model_deactivate` retires consuming work and releases parameter
backing without discarding retained rows or rebuilding commands. The next
inference call activates on demand; `loom_serve_text_model_activate` can warm
explicitly before admission. Activation reuses the checkpoint provider and
preparation commands, streaming original file bytes into the same final roots
and preparing them in place. It neither JITs nor holds a second weight copy.
An already active model does no loading/preparation. Fixed backing rejects
deactivation explicitly. The `models/qwen:epoch_check --reload_weights` witness
checks real target/MTP continuation across compaction and parameter eviction.
`--shared_residency` additionally alternates two independently retained models
on one device. A physical budget below their combined weight extent exercises
reuse of returned backing rather than simultaneous residency.

`loom_serve_text_model_trim` returns empty physical slabs, then compacts owned
blocks when the complete temporary destination union fits the budget. Otherwise
the live sparse map stays usable without turning maintenance into an execution
failure. Source regions drive bounded device-copy batches;
KV never reads back to the host. Only consumed pages and resident checkpoint
pages carry payload to copy, including partial tails. Unwritten completion
credit still relocates with fully backed destinations but no content transfer.
Copies retire before host and device maps change, and map uploads retire before
old backing is unmapped. Live recurrent state and partially occupied slabs stay
backed. Reset rows release their private backing at this maintenance cut and
initialize again on reuse. The operation does not evict live history, rebuild
commands, or change virtual addresses.
Its result separates relocated blocks, copied bytes, and physically released
bytes. The real-weight `models/qwen:epoch_check --trim` witness compares an
uncompacted continuation, including MTP, then checks full trim and regrowth.
The HTTP service reclaims evicted/cancelled rows and unused completion credit
at admission/completed-cohort boundaries. `state_trim` JSONL records expose
copied/released bytes and elapsed maintenance time. Active and retained idle
histories survive; normal epochs without ownership loss skip this maintenance.

`loom_serve_text_row_suspend` captures a completed row's private state, MTP carry
and logical-order target/draft blocks into one host-owned DRAM image before
returning its physical IDs. Its position, selected-but-unconsumed token and
metrics remain in the row. Trim can then unmap its private views and cache
backing. `loom_serve_text_row_try_resume` assigns fresh IDs, admits the union of
their missing physical slabs, restores the image, and publishes both maps
before making the row runnable. Capacity denial reports `resumed=false` without
discarding the image or partially committing destination slabs. Execution or
platform failure remains terminal. The serialized owner excludes work on that
row throughout each cold transition.

Snapshots contain no physical block identities and need no weights, JIT or
command rebuilding. Reset discards a suspended image without accessing unmapped
private state. `loom_serve_text_row_suspended_bytes` reports host payload bytes
separately from physical device commitment; UMA copies do not create extra
machine RAM. Transfers use bounded descriptor batches and the HAL's staging
pool, whose host allocation is separate from the device pool budget. The
`models/qwen:epoch_check --suspend_rows --pool_capacity=2048 --draft_depth=3` witness
overwrites the old blocks, denies resume under logical capacity pressure, then
compares resumed target/MTP output with uninterrupted execution. It also covers
unselected prefill state and resetting a suspended row. File-backed images and
automatic eviction require additional policy consumers; these explicit DRAM
operations do not implement them.

### Explicit shared endpoints

`--checkpoint_capacity=N` opts into native `row_try_pin`, `row_try_restore` and
`checkpoint_release`. Zero, the default, reserves no endpoint records or extra
recurrent slots. A pin retains the exact consumed frontier and pending-token
validity/value, with shared references to logical-order KV and recurrent state.
Several branches of one prefix therefore share the same immutable anchor.
MTP adds one source-sized carry entry per pin after the ordinary row entries
in allocation six; its small device-to-device copy does not clone recurrence.
The common pool metadata adds one host-side 32-bit reference count per physical
ID, including when checkpoints are disabled. No extra recurrent device reserve
or pin records are created in that default mode.

`checkpoint_suspend` moves that endpoint's recurrence, logical KV and carry to
an owned DRAM image. The originating row can be reset and reused; no active row
slot belongs to the image. Shared live branches remain resident. A later
`row_try_restore` warms the endpoint into fresh pool IDs and restores any row
of the same model. It admits both owners' physical ranges before replacing the
selected row, so denial preserves the old continuation and the cold image.
`checkpoint_suspended_bytes` exposes host payload separately from device backing;
warm restore or release frees it. The real-model `epoch_check
--suspend_checkpoints --checkpoint_capacity=1 --pool_capacity=2048` witness
covers row/ID reuse, live readers, denied restore, target/MTP continuation and
zero final mutable commitment. It requires elastic backing. The HTTP controls
below expose explicit suspension; no automatic eviction policy or disk image
format is implied.

A bare `row_try_restore` initially owns no private recurrent destination. Its
first advance acquires one and the model kernels read the anchor directly. The
source fork frontier switches subsequent queued epochs to the writer without a host
wait. A shared partial KV tail gets one private page copied across all declared
target/draft planes before append. Other prefix pages remain shared. Old readers
and COW sources retire at the existing completed-cohort boundary, never during
cohort assembly. Ordinary decode examines the current state and last page;
it does not walk or retain the whole prefix per token.

Request-level callers use `row_try_reserve` with a fresh, continued or checkpoint
frontier and the turn's completion high-water. It acquires exact page IDs and a
private recurrent writer, admits all their slabs beside pinned parameters,
and only then replaces the selected row. Epochs consume those already-backed
pages without growing physical commitment. `row_release_reservation` returns
unused pages and the parameter pin while retaining the consumed frontier; even
a fork cancelled before its first epoch remains valid. Reset also releases a
reservation. The source/kernel ABI is unchanged. This reserve policy commits
the full requested turn at admission; demand growth across turns remains
independent of the model's maximum context, and overcommit is not enabled.

Pin capacity refusal returns a null handle without changing the row. Restore
admits missing private slabs before replacing a continuation and returns
`restored=false` on ordinary physical pressure. Effort metrics are not rewound.
Compaction updates both live and pinned maps, moving each unique physical block
once. Trim preserves pinned anchors even without a live row. An unadvanced
fork's suspension captures its actual shared state. The last release makes
storage reusable; no automatic soft history retains a released endpoint.

HTTP maintenance submits compaction on the model timeline and yields the owner
until readiness. Source and destination backing remain owned through the copy;
only completion commits block reuse and physical release. Admission resumes
against that completed accounting. Cancellation resets an already-retired row
without another device wait. Native allocation and mapping calls remain cold
platform operations, outside the steady-state epoch path.

The bounded admission selector retains compact encoded input independently of
execution rows while capacity is unavailable. Only its current attempt borrows
an idle row or endpoint; parked inputs own no device reservation. Cancellation
or rejection relinquishes that choice; admission consumes the same completion
extent. The `prepared`
JSONL event reports source/tokenizer duration separately from admission queue
time, along with retained, appended and reserved token counts. Only invalid
request arguments or context overflow become client errors; allocation and
other preparation failures remain terminal service errors.

Reservation admission also yields through activation, cold restore and map
publication. A successful reservation becomes an active HTTP row only after
model readiness. Cancellation after submission completes that ownership first,
then returns its credit and resets the selected row. A no-new-work capacity
denial consumes the already-completed semaphore/value pair immediately; it
does not register a self-waking retry. Remaining pressure waits for transport
or model progress, with prepared input retained unchanged.

These handles are serialized model API objects. The HTTP service uses them for
explicit pins, idle completed histories and optional consumed-token prefixes.

`session.h` owns reusable host token/decoder storage, canonical continuation
text and bounded output packets. It performs source-directed token decoding and
SSE formatting without sending transport data or releasing native reservations.
The service owns those claim/admission policies and serializes session CPU work
against the model's shared VM invocation. Scratch builders are shared by the
exclusive model owner, not multiplied per retained session.

### Automatic token-prefix reuse

The shared server's `prefix_cache_capacity` reserves an independent number of
evictable endpoints; zero disables capture. An endpoint contains immutable KV
pages, a recurrent reader and exact consumed token IDs, not rendered transcript
bytes. Compatible fresh named or anonymous requests can reuse it while its
originating request is still generating. A continuing row reserves its private
writer and partial tail before capture becomes visible, preserving the request's
already-promised completion capacity.

The initial policy captures natural intermediate prefill frontiers of at least
one KV page. An existing ancestor suppresses redundant descendant captures.
Admission selects the longest retained strict prefix; at least one known input
remains for ordinary token selection. A prompt equal to an endpoint therefore
replays unless another shorter endpoint matches. Completed same-session history
and explicit checkpoint requests retain their exact pending-token semantics.
No epoch splitting, batch hold or change to device-fed continuation is imposed.

`prefix_cache.c` owns a bounded exact-token table with admission-recency LRU.
The service lends an entry only through reservation readiness, then the new row
owns its independent references. Automatic entries yield before idle completed
histories under pressure. If the selected prefix itself prevents admission,
the preserved full input replays after releasing that optional endpoint. Explicit
pins never silently degrade. Physical trim still follows the existing model
timeline; failed device readiness leaves accepted handles to model retirement.

Client `prompt_tokens_details.cached_tokens` and `admit.retained_tokens` report
actual reuse. `text_memory.prefix_cache` reports capacity and retained endpoints;
`prefix_capture.readiness_ns` measures the host readiness handoff separately from
epoch time. Capture consumes a recurrent snapshot per distinct endpoint, so its
capacity is a memory/workload choice, not a free additional session count.

Source request rendering/tokenization and completed-token response formatting
run as finite jobs on the shared host task executor. The HTTP claim, input
array, decoder and model VM remain exclusively borrowed until the job's release
callback publishes readiness. New headers can queue or reject while that job
runs, but no control operation, reservation release or subsequent command can
interleave its source invocation. Bad input becomes a client rejection after
release; cancellation and shutdown likewise retain storage until release.
Pure prompt epochs with no selected output need no response job. Device and
pool bookkeeping stays on the router's serialized task-owner turns. The router
returns workers between bounded turns, allowing shared CPU work to progress
even under continuous request arrivals. Heartbeat deadlines use the host
proactor and wake that owner to report copied statistics even while model work
is pending. Shutdown requests timer cancellation on the poll owner and joins
both terminal completion and the cancellation receipt. Reporting creates no
per-model thread and never counts as activity that can relieve memory pressure.

Input preparation can overlap an already-submitted epoch: the selected request
borrows one idle row's host input/history while device work owns other rows and its
transfer buffers. CPU release and device readiness have independent completion
slots. The owner consumes CPU release before publishing device results or
invoking the model again, preserving the single-VM-call contract regardless of
which operation finishes first. The prepared request acquires device capacity
only at the normal retired epoch boundary. This does not force smaller cohorts
or change the policy of collecting ready admissions before dispatch. Output
translation still completes before the next epoch submission.

### Admission under KV pressure

Text deployments default `admission_overtakes` to `rows`: when the oldest
request cannot reserve its full completion, smaller fitting requests can pass
it up to that many times. Zero gives strict FIFO; another nonnegative value
changes the budget independently of execution-row count. The single-model
server exposes the same setting as `--admission_overtakes` (`-1` selects rows).
New arrivals, failed probes and cancelled tails cannot replenish the oldest
request's budget. Once spent, later requests wait until the oldest admits or
leaves. This bounds overtakes, not elapsed waiting time: a slow output consumer
can retain an active reservation indefinitely.

Selection policy is separate from token preparation and native reservation.
Only a selected attempt borrows an idle row/cache through CPU/native readiness.
Parked requests own compact prepared token IDs, bounded by pending request and
context limits, with no extra device pin or recurrent snapshot. Unchanged
capacity retries reuse those IDs. If ordinary cache eviction removes a parked
request's history, its suffix is replaced by full replay before reservation;
an immutable host-history reference never authorizes use of missing device state.

`admit` events report `queue_rank`, `overtakes`, and `overtake_budget` alongside
queue time, exact completion reservation and actual cached/appended token counts.
These expose the throughput/fairness tradeoff on the same workload without
changing kernels, completion reservations or cache ownership.

The native real-model witness is
`models/qwen:epoch_check --checkpoints --checkpoint_capacity=1
--pool_capacity=65536 --context_capacity=1024 --prefill_capacity=128
--epoch=64:4 --epoch=128:8 --draft_depth=3`, with normal model/weights/tokenizer flags.

### Idle sessions beyond execution capacity

A routed text model can set `session_cache_capacity` to retain idle conversations
beyond its `rows` execution slots. The default is zero. With 16 rows and a cache
capacity of 16, up to 32 named conversations can retain their endpoints when the
shared token/physical memory budget fits them. `context_capacity` still bounds
each conversation; it does not reserve a full context slab per session.

Only recycling an idle named row pays endpoint capture. Its pooled KV and
recurrent state remain shared; source-defined draft carry is copied on the model
timeline. Normal resident continuations keep their existing path. Full-cache
exchange uses one temporary endpoint while the incoming history is admitted;
this is not another execution row. `X-Loom-Session` resumes either resident or
cached history after canonical validation. A different history replays. Invalid
input and denied admission preserve the selected incoming continuation.

This is an explicit memory/throughput tradeoff: recurrent endpoints can be much
larger than their KV prefix. Source initialization includes cache capacity plus
one exchange endpoint separately from `checkpoint_capacity`; fixed backing
allocates the declared extents, while elastic backing commits touched storage.
LRU automatic histories yield to admission pressure. Explicit pins remain
protected. `text_memory.session_cache` reports configured `capacity` and owned
`endpoints`, including at most one temporary exchange. Cached-token usage in the
HTTP response and `admit.retained_tokens` report actual reuse, not assumed hits.

### Named HTTP checkpoints

With nonzero `--checkpoint_capacity`, an empty-body
`POST /v1/checkpoints/NAME` carrying `X-Loom-Session: SESSION` pins that idle,
completed session. The `201` JSON receipt reports its consumed token position.
The routed server uses the equivalent catalog `checkpoint_capacity` field.
Detached sessions are pinnable too: each explicit name retains the same immutable
endpoint without copying KV, recurrence or carry. Automatic and explicit owners
release independently; the last owner returns endpoint storage. Suspension/wake
changes the shared endpoint's placement, not its logical contents.
Names use 1–64 ASCII letters, digits, underscores, dots or hyphens. They are
process/model-local identifiers, not content hashes or authentication tokens.
An absent/evicted session returns `404`; a busy session or duplicate name
returns `409`; exhausted pin capacity returns `503`. Pins never replace one
another implicitly.

Pin and suspend requests retain their HTTP claim through asynchronous model
readiness. The name becomes visible only after pin completion; suspension keeps
the source blocks until the host capture completes. A bounded control slot
orders these operations after the current model work and ahead of stalled
generation admission. Other models and transport processing remain runnable.
Disconnecting after submission does not roll back a successfully completed
named operation. Shutdown drains accepted work without submitting new work;
failed readiness leaves native ownership to outer model destruction.

A normal chat completion can set `X-Loom-Checkpoint: NAME` alongside its
destination `X-Loom-Session`. It still sends full history. The source-rendered
history must extend the checkpoint's canonical transcript; mismatches return
`400` before changing that session. The service restores the exact consumed
frontier and pending raw token, then encodes the appended suffix with the model
source's existing framing. Another destination session forks the endpoint;
the same session rewinds it. An unknown explicit name returns `404`, never a
silently fabricated cache hit. Omitting the header retains normal best-effort
session reuse and full replay on a miss.

With elastic backing, `POST /v1/checkpoints/NAME/suspend` and an empty body
parks that endpoint in DRAM. The `200` receipt reports `host_snapshot_bytes`;
repeating the operation retains the same image. Live branches keep their
device references, and the originating session is not reset. Once those rows
yield to idle eviction, their pages can be physically trimmed while the
independent endpoint remains resumable. The next `X-Loom-Checkpoint` request
warms it into new pool IDs and frees the image. Queued requests retain the same
canonical endpoint through this transition. Fixed backing rejects suspension
with `400`; unknown names return `404`. These are process-local, opt-in images,
not durable storage, and on UMA their DRAM payload is not additional RAM.

`DELETE /v1/checkpoints/NAME` releases the explicit pin and returns `200`.
Queued admissions borrow the record, so deletion returns `409` until they are
admitted or cancelled. Active branches already own independent references and
do not prevent deletion. A client can therefore pin before disposable tool
output, generate a summary, restore with summary-only history, then release
the pin without keeping hidden history. These endpoints mark completed turns;
they do not invent intermediate system/message snapshots inside a traversal.

Admission counts unique resident pages once plus assigned active completion
pages, including partial-tail detachment and speculative credit. The JSONL
`pool.reserved_tokens` counts owned positions beyond consumed frontiers;
`pool.resident_tokens` is unique consumed residency including explicit pins.
Both have physical backing and their sum stays within logical capacity.
Idle cache can be evicted under pressure. Pins are never automatically evicted;
if pins make a request impossible even without active work, it returns `503`
with a release/increase-capacity diagnostic instead of waiting indefinitely.
Rejected replacement preserves the selected continuation. Other idle session
caches may be reclaimed while admission evaluates a request, even if it cannot
ultimately fit; their independent explicit pins remain usable.
Cold request admission includes the missing prefix, private writer, future
pages and shared partial tail. It can tentatively retain the selected row's
private IDs for reuse, but changes no contents until the entire physical union
is admitted. Denial releases only tentative references, preserving both owners.
Physical restore refusal also returns `503` without replacing the destination
continuation. Heartbeat `checkpoint_host_bytes` counts each native cold payload
once even when multiple automatic/explicit owners retain it, separately from the
device pool. Admission events report `restored_bytes` when
a cold endpoint is warmed. Native metadata does not constitute disk persistence.

`models/qwen:check_checkpoints` exercises the actual HTTP summary/rewind,
concurrent shared-prefix and cold suspend/reuse/resume flows, comparing output
with independent full replay. It also checks preservation after a refused wake,
queued borrowing, host-image accounting and fixed-backing rejection.
It accepts the built `--server`, external `--model`, `--weights`, `--tokenizer`
and a new `--output` evidence directory. It runs on a separately excluded GPU;
it is not a synthetic model test or a performance benchmark.

Each row's control/input/progress views are present. Dense
attention is present only without a physical pool; pooled kernels find it
through source origins. An active row also borrows its independently owned
recurrent slot, a zero-initialized state region with no hardcoded Gated
DeltaNet layout. The current materializer requires a nonempty recurrent slot
even for an attention-only model. Such a port must explicitly account for that
resettable span in its storage plan;
the contract does not allocate a Qwen-sized state behind the model's back.
This exact contract is implemented by [model.c](model.c) and exercised with
the real [bootstrap](../models/qwen/prepare.loom) and its
[layout checks](../models/qwen/prepare_test.cc).

Initialization uploads and returned source buffers remain owned through
queue retirement. A failed partial creation drains accepted work before
releasing backing.

## Warm numerical entries

The signatures and complete source implementation are in
[control.loom](../models/qwen/control.loom). The host-to-source semantic plan
is independent of the model's device packet format:

* On ownership changes, `encode_state(bindings:buffer, count:i32,
  origins:buffer)` consumes `count` little-endian i64 records
  `{row, read_byte_origin, write_byte_origin, fork_position}` from a sixteen-
  record bank. It updates the retained opaque target-origin payload in place,
  preserving other rows and model-specific fields such as KV origins. Native
  code orders its upload before subsequent work on the existing timeline.
  Steady-state epochs do not invoke this encoder. Exclusive slots supply equal
  origins; the first advance of a shared slot supplies its anchor and writer.
* A plan has 32 little-endian i32 records of eight fields:
  `{length, position, row, input_begin, input_count, flags, output_credit,
  first_span}`. The first cohort occupies records 0–15 and the continuation
  starts at 16. Flags are SELECT=1 and PROPOSE=2.
* `encode_epoch` receives that plan, contiguous known token IDs, both span
  counts and the terminal ID. It fills the opaque metadata/input upload
  buffers and returns ordinary output count.
* `epoch` receives opaque state, shape/selection/speculation facts and the
  twelve retained device buffers. It invokes `runner.execute_N` and
  `runner.feedback` without waiting. Private workspace is allocated by the
  execution domain around each command; escaping state has separate bindings.
* After retirement, `publish_epoch` translates opaque feedback into
  `{consumed, known, outputs, verifications, tokens[16]}` i32 records in
  original span order. Only accepted state contributes to consumed progress.
* The isolated `step` receives state, a prefill/decode selector and six
  buffers: residual, control, recurrent, attention, IDs and progress.
  Control holds `{input_count, absolute_position, terminal_id}`.
  Selected ID is at input slot zero; progress slot zero is selected count
  and slot seven is the terminal flag.

The last selected token has not yet been consumed. Retained HTTP appends
preserve that original ID separately from canonical text. Re-tokenizing a
canonical checkpoint and treating it as already consumed corrupts that
invariant when tools or whitespace were normalized.

## Chat policy entries

All text policy lives in the same warm program as numerical execution:

| Export | Inputs → result |
| --- | --- |
| `model_name` | none → UTF-8 identity buffer, retained by the native policy |
| `chat_begin` | original request JSON, raw tools array → prefix buffer, opaque i64 rendering state |
| `chat_message` | original message JSON, raw role spelling, decoded content, decoded reasoning, rendered calls, state, ordinal, presence bits → fragment buffer, next state |
| `chat_end` | state → final generation-prefix buffer |
| `render_tool` | name, decoded argument JSON, call ordinal → canonical buffer and used byte length |
| `parse_tools` | raw schemas, generated suffix → JSON array of function-call records |
| `prepare_input` | text, format:i32, boundary:i32, capacity:i64 → little-endian i32 IDs and complete count:i64 |
| `text_end` | accumulated response, previous published extent:i64, phase:i32 → safe byte extent:i64 |
| `complete_text` | canonical prompt, final ordinary content, canonical calls → owned canonical checkpoint buffer |

Unmarked types above are buffers; rendering state, ordinal and presence bits
are i64. Presence bits are reasoning=1, tool_calls=2 and tool_call_id=4.
Raw JSON remains available so a policy can inspect names, IDs or fields not
represented in the decoded text. `chat_begin` receives the whole request;
a policy requiring whole-history traversal can render there and return empty
per-message fragments. C does not interpret its rendering state.

The current HTTP protocol is streaming greedy text, with up to 256 messages
and 16 function tools. Text content may be a string, null, or an array of text
parts. Unsupported sampling, strict constrained output, stored responses and
image parts fail explicitly. `model_options` is an opaque JSON object for
model policy; `enable_thinking` is forwarded too. The Qwen policy rejects
thinking and nonempty model options; another source policy may implement them.

Tool parsing returns records of the form
`{"type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}`.
The host assigns IDs and serializes SSE; source owns generated grammar and
canonical separators. `tools.parse_xml` is an optional bulk codec selected
explicitly by the Qwen program, not the default interpretation of text.
The independent [plain-text/JSON policy](../models/qwen/testdata/portable_chat.loom)
is tested through the same native caller without linking that codec.

Qwen uses the generic [`json.compact`](../runtime/json.h) capability before
rendering tool arguments. Clients commonly parse and reserialize generated
arguments; insignificant spacing inside nested objects or arrays then changes
without changing the tool call. Compaction makes those histories share the
same checkpoint key while preserving string contents, number and escape
spelling, and member order. The key validates continuation of the original
generated tokens, not equality with a fresh tokenization. Native positions and
the exact selected-but-unconsumed token remain separately owned by the session.

Input format is rendered=0 or single-user=1; boundary is fresh=0, open=1 or
terminal=2. Open/terminal refers to the separately retained pending token.
The input capability returns a bounded token prefix; complete encoding needs
the source's capacity-plus-one check. Output phase is streaming=0 or complete=1.
Published extents are monotonic and within the accumulated response.

Request fragments borrow input bytes only during invocation and are copied
before references release. Completed checkpoint arguments instead own their
backing, so the returned buffer may alias any argument safely. Cancellation,
replacement and shutdown release the retained result. No session owns a VM.

## Checks that matter for a port

The CPU-only request inspector JITs the actual chat/input/completion entries
without loading model weights or creating a device:

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve/text:request_check -- \
  --model=/path/to/source/package --tokenizer=/path/to/tokenizer.json \
  --request=/path/to/request.json --capacity=8192
```

The request file uses the same body as the HTTP endpoint, for example
`{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"Hello."}]}`
with the package's actual public name. Standard output is one JSON record with
`model`, the complete rendered `prompt`, and its `tokens`. These can be compared
directly with the model's reference tokenizer/template before any numerical
work. An input exceeding capacity fails, rather than returning a truncated
prompt as success.

Adding `--response=/path/to/raw-generated-text.txt` also reports `text_end`,
the canonical `checkpoint`, and the structured `tool_calls` delta. The file is
raw model text, not a serialized OpenAI response. This exercises the same source
completion and tool parsing used by HTTP; the observer neither generates tokens
nor claims to test GPU cache retention. Invalid input produces a nonzero exit
with diagnostics on stderr and no successful JSON record.

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve/...
```

The serving suite covers shared native mechanisms; model-local checks cover
actual source policy, parameter layout and packet semantics. A new model adds
its own real-weight numerical witness and retained HTTP continuation, including
a malformed intervening request and an independently paced second row.
The Qwen checks demonstrate this path, not another model's numerical accuracy.

The [authoring guide](../docs/MODEL_AUTHORING.md) covers motifs, source catalogs,
JIT specialization and compiler reports. The [performance guide](../docs/PERFORMANCE.md)
separates controlled throughput from instrumentation and correctness runs.
