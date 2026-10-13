# Packed autoregressive serving

This package executes source-defined text models with retained sessions.
Model programs supply numerical stages, parameter preparation, storage geometry,
tokenizer framing and chat policy. Native code supplies execution ownership,
admission, packing, cache lifetime and HTTP output; it contains no learned model
or model catalog.

## Ownership

The [model](model.h) owns one prepared source program, cached command set,
parameter residency and execution timeline. Sessions are rows over that shared
model, not separate VM instances. [Source preparation](source.h) and
[stage binding](stages.h) separate host compilation from native device ownership.
The shared [program runtime](../../runtime/program/program.h) accepts text or
bytecode source inputs through the same compilation path.

The [service](service.h) is the serialized owner of HTTP claims, admission and
epoch completion. Finite CPU preparation/output tasks borrow its model and
session storage. Device readiness and CPU release callbacks publish completion
and wake the owner; they do not mutate sessions or destroy the model. Input
preparation can overlap an accepted epoch on other rows. No service creates a
private worker thread or polls device completion in a busy loop.

The [epoch planner](epoch.h) receives already-credited rows and a selected
packing shape. It produces bounded spans and optional device-fed continuation
without allocation or HTTP access. The model program's completed progress is
authoritative; the service commits that progress only after readiness.

## Admission and packing

[Packing](packing.h) treats token capacity and independently advancing span
capacity as separate dimensions. Known prompt tokens, decode anchors and full
speculative verifier spans can share one traversal. The policy evaluates cached
shapes against the same readiness set and rotating priority, then chooses the
most useful tokens with stable tie-breaking. It is an occupancy policy, not a
claim that every device has the same execution-cost curve.

[Admission](admission.h) applies bounded backfilling to a host-only request
queue. A parked request retains compact prepared tokens, not an execution row
or device reservation. Only the selected attempt borrows a row and history.
The native storage layer admits its page-rounded completion high-water and
private recurrent writer, including speculative credit, before generation.
Insufficient capacity queues or rejects the request before successful output
headers. Slow output readers backpressure their own row without preventing
credited peers from advancing.

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
after successful readiness, even if the client disconnected; HTTP cancellation
does not imply cancellation of already accepted device work. Failed readiness
leaves native retirement with the model owner. Release and completed suspension
request asynchronous pool trim.

## Protocol and observation

[Chat policy](chat.h) calls the model's source VM to render input, classify output
and retain canonical conversation state. [Session output](session.h) owns
incremental decoding, bounded JSON/SSE packets and output credit. Shared error
[responses](response.h) consume diagnostic status at the wire boundary without
borrowing scratch used by a CPU output task.

[Telemetry](telemetry.h) observes committed work, while
[heartbeats](heartbeat.h) report copied state independently of device readiness.
These observers do not decide admission, advancement or retirement. Unit tests
exercise the pure packing and epoch contracts; actual source-model HTTP tests
exercise retained continuation, cancellation, cache ownership and final output.
