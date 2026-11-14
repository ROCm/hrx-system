# Packed retained-token serving

This package executes source-defined text models with retained sessions.
Model programs supply numerical stages, parameter preparation, storage geometry,
tokenizer framing and chat policy. Native code supplies execution ownership,
admission, packing, cache lifetime, and HTTP or native output; it contains no
learned model or model catalog.

The `sequence` deployment profile reuses this ownership for known-token stages.
It has no tokenizer/chat dependency and never implicitly selects or feeds back a
token. [Sequence protocol](sequence.h) owns its envelope and exact frontier;
[captures](capture.h) own source-defined output independently of reusable model
residuals, native rows, and weights. Its HTTP route is `/v1/sequences`.

## Ownership

The [model](model.h) owns one prepared source program, cached command set,
parameter residency and execution timeline. Sessions are rows over that shared
model, not separate VM instances. [Source preparation](source.h) and
[stage binding](stages.h) separate host compilation from native device ownership.
The shared [program runtime](../../runtime/program/program.h) accepts text or
bytecode source inputs through the same compilation path.

The [service](service.h) is the serialized owner of request receipts, admission,
and epoch completion. Finite CPU preparation/output tasks borrow its model and
session storage. Device readiness and CPU release callbacks publish completion
and wake the owner; they do not mutate sessions or destroy the model. Input
preparation can overlap an accepted epoch on other rows. No service creates a
private worker thread or polls device completion in a busy loop.

The [epoch planner](epoch.h) receives already-credited rows and a selected
packing shape. It produces bounded spans and optional device-fed continuation
without allocation or HTTP access. The model program's completed progress is
authoritative; the service commits that progress only after readiness.

## Executable choices

The [variant pool](../../runtime/program/variant_pool.h) retains independently
compiled choices and their diagnostics. Construction publishes readiness, not
execution policy. Explicit consumer borrows protect entries from removal;
compilation and failure records count against the configured cache capacity.
IDs are process-lifetime identities and never transfer across registrations.

The text [reconfiguration policy](reconfiguration.h) stages native providers
against the original model's parameter and state contract. It captures provider
locators before returning and compiles on the shared task pool while inference
continues. Both source text and portable bytecode providers use this path.
Preparation declarations, control VM, tokenizer, shape catalog and retained
state stay fixed. Reflection checks parameter binding layout; the author remains
responsible for semantic compatibility of retained bytes.

Selecting a ready choice queues a preferred-policy change with an expected
current ID. The owner retires accepted CPU and device work before rebinding
native stages. A failed construction or binding leaves the previous choice
usable; shutdown cancels queued selections and joins accepted compilation before
the model can retire. Selection neither reads parameters again nor copies KV
state. It is a cold table-binding operation, separate from ordinary epoch
packing and from request-local executable selection.

The pool does not permanently pin the original entry. The text model separately
retains its bootstrap stage metadata and original commands for parameter
binding and restoration. Dropping a cached entry releases that entry's ownership,
not resources still held by the model or its selected native module.

## Admission and packing

[Packing](packing.h) treats token capacity and independently advancing span
capacity as separate dimensions. Known prompt tokens, decode anchors and full
speculative verifier spans can share one traversal. The policy evaluates cached
shapes against the same readiness set and rotating priority, then chooses the
most useful tokens with stable tie-breaking. It is an occupancy policy, not a
claim that every device has the same execution-cost curve.

[Admission](admission.h) applies bounded backfilling to a stable request queue.
A parked request retains prepared input and may own projected image results,
but holds no language execution row or completion reservation. Only the selected
attempt borrows a row and history. The native storage layer admits its
page-rounded completion high-water and private recurrent writer, including
speculative credit, before generation.
Insufficient capacity queues or rejects the request before successful output
headers. Slow output readers backpressure their own row without preventing
credited peers from advancing.

Resident identities are independent of epoch shape. The initial `rows` value
sizes a directory, capped by the optional per-deployment `max_sessions`.
That ceiling bounds resident owners, including preparation and output loans,
not the number of participants selected for one epoch. No separate active
counter mirrors the ownership already represented by those records. Finished
owners become reusable; idle history caches retain their separate limits.
When all residents have borrowers and the directory is below its ceiling,
[capacity growth](service_capacity.h) evaluates source `resize_state` and
allocates host session records on the shared task pool at a retired model
boundary. This worker does not touch shared memory-pool accounting. After CPU
release, the serialized server owner submits native growth without waiting for
device completion. Existing session records and state IDs remain stable; new
directory entries publish only after the captured device frontier. Prepared
media continuations defer growth until their address-dependent plans retire.
The same model, JIT artifacts, and parameters serve the enlarged directory.

Source returns new control state and the five ordinary storage records. Private
views, recurrent slots, and cache planes are append-stable. Virtual reservations
retain existing physical slabs; fixed backing allocates a replacement and copies
retained bytes. The small carry checkpoint bank follows resident entries and
moves independently. No state copy or directory allocation occurs per epoch.
Actual request memory admission remains separate from directory capacity.

## Image projection

The optional [finite component](../finite/component.h) owns image preparation
and parameter residency independently of the language model. Text-only requests
and exact retained media-prefix hits leave it dormant. The service resolves
prefix reuse before offering missing image occurrences to the
[projector](media_projector.h), and projects them before reserving language
completion storage. The projector batches compatible images from one or several
queued requests without waiting for hypothetical arrivals or changing admission
priority.

One bounded preparation window realizes compressed inputs directly into
[RGB/RGBA source packets](../../media/image_request.h). Source programs own
orientation, compositing, resize, normalization, projected row counts, and
position geometry. The finite source's work-unit limit bounds a cohort; its
upload capacity is the decoded-window byte target, with one larger checked
image permitted alone. A window drains before another begins decoding.

Live requests own [device results](../finite/result.h), not host feature
copies or persistent projection-cache entries. The result retains pooled output
storage and producer readiness independently of the projector's model and
weights. [Media epochs](media_epoch.h) import that readiness and gather device
rows into source-defined text input packets. Only control records and metadata
cross from the host; projected features never return to it. Consumed results
are released after consumer readiness, while queue captures protect actual
device use on failure.

Cancellation waits for an accepted codec job's request-body borrow before freeing
the request. Later cancellation detaches the request without renumbering images
in a shared cohort or canceling peers. Stable pending-request storage preserves
those asynchronous borrows as the admission queue changes.

## Retained state and checkpoints

[Storage geometry](storage.h) describes byte ranges, pages and recurrent slots
without native assumptions about attention heads, layers or cache element
types. [State ownership](state.h) uses the shared logical block pool and physical
memory facilities. Session context ceilings and total pool capacity are separate
limits; short sessions do not reserve their entire context ceiling.

[Session cache](session_cache.h) entries retain completed named histories
independently of execution rows. [Prefix cache](prefix_cache.h) entries share
exact consumed-token frontiers between requests. Automatic entries can yield to
admission pressure; explicit pins remain protected until released.

The cold [retention forecast](retention.h) separates that policy from native
ownership. The service marks only references its ordinary eviction path could
release; every other reference remains protected. The model computes a necessary
storage bound from source geometry and retained logical IDs. Fixed ranges use
their actual backed union; movable KV and a retryable request's new slots may
reuse any legal placement. Overlapping source domains do not double-charge
aliases. Prepared continuations retain their exact commitment ranges, while
unaccepted declarations alone consume no physical capacity.

Both logical admission and physical reclamation use this bound before dropping
history. It can prove eviction futile, but never grants backing or edits a plan.
The ordinary allocator remains authoritative. Completed maintenance prioritizes
free slots in surviving backing so retries can use newly freed IDs without
requesting another physical slab. This reorders only free IDs; it does not move
live recurrent state or change a prepared continuation. Fitting admissions and
token execution do not run retention queries. Cold scratch scales with grown
ownership metadata and backing, not the declared virtual token capacity.

The [checkpoint registry](checkpoint_registry.h) owns named immutable endpoints
and canonical transcripts. The [checkpoint request](checkpoint_request.h) owns
one control claim through accepted capture or suspension. The service resolves
live-session and queued-borrow facts explicitly. Endpoint publication happens
after successful readiness, including when a caller closes during accepted
work. Failed readiness leaves native retirement with the model owner. Release
and completed suspension request asynchronous pool trim.

## Protocol and observation

[Chat policy](chat.h) calls the model's source VM to render input, classify output
and retain canonical conversation state. [Session output](session.h) owns
incremental decoding and source classification. A request's [consumer
receipt](consumer.h) preserves input storage and owns bounded output delivery
independently of the model row. The [HTTP adapter](http.h) encodes JSON/SSE in
the existing CPU output task, then publishes its staged packet on the serving
owner. An empty stage permits another epoch while the previous copied send is
in flight; a full stage behind a stalled reader pauses only that row.

Pending requests move their receipt into the admitted row without changing
input storage. Rejection, cancellation, and completion release it only after
accepted CPU/device borrowers and request-local views retire. HTTP rejection
uses independent scratch and can overlap another row's output task. Translation
failure discards the incomplete staged response, invalidates canonical reuse,
and publishes the terminal protocol error.

Generated function calls form one completed transaction. Source parsing and
canonical history formatting must both succeed before the owned
[tool-call table](tool_calls.h) is published. It retains decoded names,
argument documents, and stable request-derived IDs independently of VM or
request storage. HTTP encoders consume those records directly, adding delta
indexes for streaming output. A later invalid record leaves no partial call
set, and retained session/checkpoint histories do not keep this output table.

## Native text consumers

An in-process serving owner uses `loom_serve_registry_try_open_text` with a
deployment name and a [native request](stream.h). The request contains the same
supported chat JSON document as HTTP, in an owning string builder, plus optional
session and checkpoint selectors. Acceptance moves the builder and copies the
selectors; pressure or immediate failure leaves all caller ownership intact.
Source validation, image preparation, canonical history, and packed model
execution use the existing pending queue and shared workers. Constructing a
large conversation or image payload can happen before entering the owner turn;
admission never copies the complete JSON body.

With `stream:true`, `loom_serve_text_stream_peek` borrows an ordered block of
content and reasoning spans. `loom_serve_text_stream_block_count` and
`loom_serve_text_stream_block_at` traverse those spans without parsing output
JSON. Consuming the block returns delivery credit. The CPU output task may
prepare the next block while the first is unread, then only this request waits
for credit. An empty final block also requires consumption. With `stream:false`,
the handle retains the complete result without incremental spans.

After execution retires, `loom_serve_text_stream_result` exposes the complete
content, reasoning, validated tool calls, finish reason, request identity, and
committed token usage. Those allocations move from the completed source
transaction; they do not retain a model row or its source VM. Canonical history
remains separately owned by the session and can resume through HTTP or native
admission. The final result remains valid through model removal, same-name
replacement, and registry destruction until the caller closes the handle.

The caller serializes handle operations with registry advancement. Closing
immediately cancels output interest, while accepted CPU/device borrowers retain
execution ownership until retirement. Source rejection and generation failure
publish an owned terminal status. A stop discards undelivered publication but
preserves previously borrowed bytes until consume or close. Holding a final
result does not prevent graceful model removal.

## Native checkpoint controls

Native callers use `loom_serve_registry_try_open_text_checkpoint` with a
[checkpoint request](checkpoint_control.h) to pin, suspend, or release retained
history. These operations share the existing model control slot and readiness
boundary with HTTP. Acceptance copies the identifiers and returns a control
handle; an OK result with no handle means preparation or queue pressure. The
owner retries after readiness or other progress, without blocking a worker.

For example, after session `conversation` completes with reusable history, the
owner can submit this request:

```c
const loom_serve_text_checkpoint_request_t request = {
    .action = LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
    .name = IREE_SV("before-branch"),
    .session = IREE_SV("conversation"),
};
loom_serve_text_checkpoint_control_t* control = NULL;
iree_status_t status = loom_serve_registry_try_open_text_checkpoint(
    registry, IREE_SV("chat"), &request, &control);
```

The caller handles `status`, retains an accepted `control` across owner turns,
and checks `loom_serve_text_checkpoint_control_is_complete` before inspecting
its status and result. Successful results report the consumed token position
and host snapshot size. They remain valid through model removal until
`loom_serve_text_checkpoint_control_close` consumes the caller handle.

A subsequent text request can select `checkpoint = IREE_SV("before-branch")`
and a different session name. It still supplies the full conversation extending
the checkpoint's canonical history. An incompatible history fails explicitly;
an interrupted or omitted generated suffix cannot silently become part of that
history. The selected state remains owned by its original model registration.

Closing a control only relinquishes that receipt. It does not release a named
checkpoint or undo an accepted pin or suspension. A separate `RELEASE` request
with the checkpoint name and an empty session releases the retained state.
Release remains available while inference admission is drained, even with held
inference requests. An unprepared registration reports `NOT_FOUND` without
model preparation. Pin and suspend continue to obey inference admission.
Accepted operations retain their storage through actual readiness, including
when the caller closes before completion.
Suspension requires elastic retained storage; fixed backing returns
`FAILED_PRECONDITION` through the native control's status.

## Source request policy

Source `request_defaults(options)` validates deployment policy once. Each
`request_options(defaults, request, nonce)` call validates overrides and returns
an owned opaque buffer retained by the chat through completion. Both inputs
have owned backing, so source may return or retain them without borrowing HTTP
storage. The receipt serial supplies a distinct nonce when source needs an
implicit seed. Native code does not decode the packet or implement sampling.

Known-input sequences use `prepare_request(defaults, request, nonce, tokens)`
instead of the chat-only `request_options` role. Its immutable parsed token
buffer covers the complete append, independent of scheduled epoch sizes.
Source validates vocabulary and other input constraints before reservation,
retained-state mutation, or response streaming. Rejection uses the existing
request-local diagnostic path and preserves other requests and retained
frontiers. The token buffer replaces the compact prepared-input allocation;
source can retain it without borrowing scratch storage. Epoch execution errors
remain terminal because shared encoding and submission are not transactions.

After row admission, `bind_request(control_state, row, options)` binds that
policy to the source-owned row table. `encode_epoch` receives the epoch plans,
known input IDs, optional projected-row records, and packet capacities, then
writes the source's control metadata and input records. It returns
`(output_count, transfer_plan, byte_offset, byte_length)`. The plan view contains
little-endian u64 records
`{source_binding, source_offset, target_binding, target_offset, length}`.
Source and target binding tables each name metadata at ordinal zero and the
input packet at ordinal one. Independent offsets allow compact source bytes to
populate sparse destination storage. Capacity describes available storage, not
transfer size; bytes outside these ranges are untouched. Source initializes
every byte its device consumers will read.

The shared [transfer plan](../../runtime/program/transfer_plan.h) boundary checks
all bindings and extents before submission. Nonempty ranges are ordered by
target binding and offset, with no overlapping targets. Source ranges may
repeat. The records describe independent placements, not execution order, and
contain no physical addresses or transport handles.

The local adapter binds checked placements to HAL uploads. The plan view is
consumed during submission and can reuse source control state without allocating
a new buffer. Packet bytes remain immutable until accepted queue accesses
retire, including partial submission failures. Transport source return,
destination visibility, and final destination-consumer retirement are distinct;
neither an early error nor successful placement alone authorizes target reuse.
Media producer waits
and device-feature gathers follow these host control uploads without uploading
the feature storage. The
[source control example](../../fixture/text/control.loom) implements this
boundary, with upload ownership in [epoch_upload.h](epoch_upload.h) and media
record layouts in [media_epoch.h](media_epoch.h).
The serialized model owner excludes outstanding
execution while changing those packets. Device selection returns only selected
IDs through the existing completion path; logits and RNG state stay on device.

[Telemetry](telemetry.h) observes committed work, while
[heartbeats](heartbeat.h) report copied state independently of device readiness.
These observers do not decide admission, advancement or retirement. Unit tests
exercise the pure packing and epoch contracts; actual source-model HTTP tests
exercise retained continuation, cancellation, cache ownership and final output.

## Two-bank groups

A deployment can select `group_mode: "two_bank"` and the registered
`plan_provider: "two_bank_v1"` in its text definition. Portable model cards
put the same defaults in a schema-version-2 preset's `execution` object.
`group_mode: "single"` preserves ordinary epochs. Missing group exports or an
unknown provider fail preparation before the model becomes ready. Existing
version-1 cards continue using their ordinary execution contract.

The source implements [text_prefill_group_v1](prefill_group.h): a semantic
stage table, group encoding and one final publication. Each layer's prefix
publishes the shared causal state that B needs; bank-private residuals,
packets, outputs and workspaces remain live through every declared consumer.
The source reserves both positions while committed row state stays at the last
successful boundary. Completion publishes both known-input counts together.
There is no intermediate device wait or host completion of A.

The builtin `sequential_v1` provider consumes the same source contract as a
comparison baseline. Providers are explicitly registered on the host before
model preparation and retained by each selected model. Preparation runs once;
a pure selection callback receives absolute position, shapes and live token
counts at each candidate chunk boundary and can choose ordinary packing. Group construction
copies upload bytes and bindings; partial enqueue failure ends model execution
and keeps accepted resources through actual queue retirement. Readiness failure
alone never releases borrowed host payloads.

Descriptor version 2 declares eight phases per shape and layer through the
same group export. Prefix prepares activation input, primary and secondary FFN
branches produce disjoint output columns, and FFN merge joins those columns
before activation and down-input packing. Primary and secondary down branches
join at down merge before residual/norm/capture. Secondary producers require
their explicit merge stages. Optional phases use `UINT32_MAX`.
An eight-phase shape may omit every stage to keep that shape on ordinary
execution. Both banks must have complete grouped implementations before
selection; a partially omitted shape is invalid.

The builtin plan emits cross-lane predecessor edges for these branches and
preserves the same-layer A-prefix to B-prefix state dependency. B's prefix
and FFN branches are submitted ahead of A's merge, permitting overlap on
independent lanes. Each bank reserves a separate aligned temporary region per
lane. Persistent intermediate results belong in source-declared bank storage.
Version 1 descriptors keep their existing three phases and ordered submission.

Descriptor version 3 declares twelve phases: prefix, gate primary/secondary/
consume, up primary/secondary/consume, activation, down primary/secondary/
consume, and residual. Its trailing records name each resident producer's
consumer and bank-relative request/response ranges. The model bridge publishes
RUN before dispatch and signals independent readiness only after all sixteen
payload generation words agree. The emitter waits on that readiness at the
declared consumer, sends ACK after consumption, and waits for the producer's
terminal retirement before downstream buffer reuse. Cancellation publishes
ABORT and final ACK, fails readiness, and drops bridge mappings before model
retirement observes the accepted native captures. Each live producer needs
distinct control ranges. The shared pool must qualify uncached or host-coherent
access for each resident consumer and persistent host mappings. A DMA producer
can use its ordinary access route with explicit cache transitions; it need not
advertise the consumer's uncached mapping property. Versions 1 and 2 keep their
existing completion edges.

Descriptor version 4 declares one continuous resident lifetime separately from
the twelve-phase table. The trailer contains one record with the lifetime
stage, generation count, and request/response offsets in bank A's features.
Seven-resource GPU leaves receive that shared features root after their six
ordinary resources. The source publishes generations and polls responses in
device memory, consumes each payload, and publishes the final ACK. Serving
submits the resident command once after packet uploads and joins its actual
terminal after the final GPU leaf, before feedback or buffer reuse. The host
bridge owns admission, timeout and cancellation; aborted groups cannot publish
progress. Descriptor replacement preserves lifetime and exchange geometry.

A thirteen-resource pair command binds bank A's six ordinary roots, bank B's
six ordinary roots, and bank A's shared features root, in that order. It records
both banks' GPU work in one command buffer. The emitter executes it at bank A's
visit and reuses that node at bank B's visit; bank-local residual and drafter
leaves retain their ordinary bindings. Command-private workspace follows the
thirteen roots. This lets source programs express complete interleaved GPU
schedules while the version-4 resident lifetime runs on its independent lane.
The encoder may select other prepared stages in its descriptor before enqueue;
serving validates that table again after encoding, retaining unchanged plans.

The startup compatibility variable `LOOM_SERVE_TWO_BANK=1` applies only to a
single startup deployment, when a provider ID declares group-capable source
and no explicit group mode is set. Its value is captured at host creation and
translated into the model's immutable selection after card resolution.
Dynamic registration never inherits this compatibility choice. Requests never
read environment variables. Native callers use the typed mode. Registration by shared-library symbol and preload interception
are unsupported.

Initial pairing covers two contiguous known-input spans of one row. Media,
proposals, device-fed continuation and ready peer work use ordinary packing.
Only B may select. Resource ABI 2 allows source-owned draft-cache warm stages;
proposal/verification stages remain outside groups. The model source must
include every warm/capture consumer before bank reuse.

Native variant selection on a group model happens at a drained boundary: with
no group or ordinary invocation in flight, the source plan export and provider
preparation run again against the candidate's stages, outside the execution
lock. Bank workspaces are replaced in one allocation transaction after the
bank working-set check when the reflected requirement changes. The plan, leaf
descriptors and workspaces are published only after the native rebinding
succeeds; any earlier failure leaves the current selection usable. Resident
growth re-exports the plan from the resized control state on its cold
preparation path and publishes it with that state. Bank packet geometry is part
of the compiled cohort, so a resize that would change it is rejected for every
model by the source storage contract.

Group nodes are checked against the lane's queues at append (usage, access,
memory type, flags, fill alignment, copy overlap), so such faults reject the
group before any node is accepted.

This path uses one existing device and its queues. It changes scheduling and
storage ownership without introducing approximate arithmetic or NPU transport.
The shared tests run on CPU and AMDGPU leaves; real-model golden, DFlash and
teacher-forced numerical qualification remain separate release checks.

## Device placement

Runtime model-card v2 program sets may map exact preparation-stage roots to
logical names in the application's device inventory.

```json
{
  "device_placement": {
    "ffn_gate": "npu",
    "ffn_up": "npu",
    "*": "gpu"
  }
}
```

Place this object beside `inputs` and `libraries` in the selected program set.
An exact root overrides `*`. A direct card uses `*` as its primary device when
present; an explicit deployment device takes precedence. Omission uses the
deployment's primary device.
Every explicit root must occur in preparation and every device name must resolve
in the owning group. Native callers may supply the equivalent
`loom_serve_text_options_t.device_placement` table. Each command root and its
reachable kernels compile on one device. A multi-device model retains separate
catalogs, profiles, executable caches and execution timelines.

For continuous hybrid execution, a v2 program set may also declare
`"exchange_pool": "shared"`. The deployment explicitly binds its primary pool
and configures the named exchange pool for both devices. GPU weights, state and
private workspace use the primary pool; secondary parameter domains and the
continuous handshake use the exchange pool. This permits cached GPU storage
with an uncached shared exchange. Keep primary and secondary parameters in
separate checkpoint scopes. Omitting this binding retains the single-pool
contract. Missing pool names or unsupported access scopes fail before serving.

The selected model pool must admit all consuming devices. A deployment pool can
set `"uncached": true` to require uncached access from every participating
device family (host execution families require host coherence). With
`"uncached_devices": ["gpu"]`, that requirement applies only to the named
nonempty subset of pool devices; other devices retain ordinary access with
cache transitions. Persistent host mapping remains required. HAL qualifies
the complete scope before allocating backing, and resident transport rejects a
consumer without qualified live access. Fixed backing is needed when the joint
scope cannot bind and unbind elastic slabs. Both ordinary VM
stage execution and two-bank groups return a joined primary work value that
covers foreign producers. Partial acceptance prevents replay and preserves
all accepted consumers through model retirement. Code variants for multi-device
models currently reject until per-lane variant compilation is implemented.

Placed text stages retain the existing command parameter-reflection and
six/twelve-resource group ABI requirements. A model provider must author
compatible numerical roots, packing, panels and merge stages. Device placement
alone does not convert a GPU FFN into an XDNA implementation. Resident polling
services additionally require a jointly owned protocol and a bounded
startup/abort/drain contract before serving can enable them.

Parameter roots with compatible translated layouts share storage. Independently
packed roots from the same checkpoint own separately accounted copies when no
existing layout fits; subsequent matching roots reuse that placement.
