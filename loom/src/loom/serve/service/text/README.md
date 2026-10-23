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
