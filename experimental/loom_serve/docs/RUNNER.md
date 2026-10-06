# Runner ownership and scheduling

The runner connects source-authored math to explicit queue submission. Its VM
is a model program, not a container for each chat session. One host owner
multiplexes rows through shared code, weights, and workspace. Transport and
heartbeat progress are independent of the model owner's GPU wait.

The shared `control.loom` has isolated and packed entry points. Source owns
prefill/decode selection and the proposal, target/verify, feedback, and catch-up
sequence. The native model adapter publishes validated descriptors, invokes
that program once per epoch, joins both timelines, and commits host progress.
Cold stage selection and dynamic configuration come from `prepare.loom` through
the shared declaration module. Source also produces retained storage geometry
and initial row tables; the native materializer owns their allocation and
retirement. Source also constructs device descriptors, publishes feedback and
owns chat policy. The shared text adapter carries semantic spans, opaque source
state and canonical checkpoints without naming a model.

## Boundaries and the information each owns

| Component | Receives and owns | Does not infer |
| --- | --- | --- |
| [`device`](../runtime/device.h) | HAL device URI, async I/O services, device/group, exact queues, execution timelines, physical pool and profiling | Compiler target support, logical model state or request lifetimes |
| [`jit`](../runtime/jit.h) | Source catalog, configuration, live device profile; compiled command images and native entries | Session identity, cache lifetime, chat semantics |
| [`command`](../runtime/command.h) | Compiler-produced parameter/binding requirements, executable reflection; reusable HAL command recording | Model graph from buffer contents or filenames |
| [`execution`](../runtime/execution.h) | Exact dispatch/transfer queues and explicit work/feedback timelines | Ordering from FIFO submission or alias inspection |
| [`module`](../runtime/module.h) | Indexed prepared commands, typed execute imports and registered host feedback spans | Per-session VM state, model stages, or a general HAL instruction set |
| [`program`](../runtime/program.h) | Source-JIT bytecode, linked libraries, one process and serialized invocation | Model geometry or the lifetime of asynchronously borrowed host payloads |
| [`preparation`](../runtime/preparation.h) | Cold source declarations of commands, configuration and checkpoint domains, copied independently of the bootstrap process | Model geometry, stage semantics, or GPU allocation |
| [`text_model`](../text/model.h) | Source-declared storage materialization, physical page ownership, scratch, source packet invocation, numerical progress | HTTP or tool semantics |
| [`weights`](../runtime/weights.h) | Shared parameter residency, source policy queries, cached preparers, file-read/preparation readiness | Tensor naming rules, model geometry, or ordering from submission order |
| [`packing`](../scheduling/packing.h) | Trusted ready span lengths, indivisible minima, shapes, rotating priority | Token values, model identity, attention state, measured kernel cost |
| [`text_schedule`](../text/schedule.h) | Default model shape catalog and completion-reservation extent | HTTP output credit or committed model progress |
| [`text_service`](../text/service.h) | Validated chat, session keys, canonical history, output credit, scheduling policy | Kernel layout decisions |
| [`http_server`](../http/server.h) | Bounded HTTP framing and copied response bytes over IREE TCP carriers | Model sessions or sampling |

These are runner-private seams, not a proposed public serving ABI. The
compiler/runtime handoff is the command product: its parameter placement,
entry requirements, launch structure, and transient requirements are consumed
directly, rather than reconstructed in host code.

## Cold residency

`text_initialize` first invokes the source bootstrap, retains its opaque control
state, and loads the tokenizer with the source-declared terminal marker. It then
borrows the caller's device owner, creates its JIT, compiles the declared stages,
checks their layout agreement, indexes parameter plans, records commands, creates the warm VM
program/native capabilities, and allocates retained rows and MTP state. Each
stage can have different kernel choices while binding the same model storage.
No session gets another copy of the weights or code.

Allocation lengths, initial zero extents, arena views and target/draft origins
come from the cold source result. Native code retains those result buffers after
the bootstrap process is gone and through the initialization frontier. Successful
initialization releases them after one join; partial failure leaves them owned
until teardown drains accepted work. Workspace extent/alignment comes directly
from compiler reflection, not a model formula or a source-side estimate.

`loom_serve_device_create` establishes one runtime domain from a HAL device
URI and explicit backing/budget options. Model components borrow its device,
group, exact queues, execution object and physical pool; the helper does not
contain a tokenizer, stage catalog or model policy. Text and image constructors
can share this owner. One host owner serializes calls and residency changes.
The application destroys models before their device owner; profiling spans the
whole device lifetime rather than opening overlapping per-model sessions.
The current JIT still selects AMDGPU explicitly. A different device URI alone
does not provide a compiler backend or model kernels for that device.

The model catalog includes `config.loom` for fixed specialization bounds.
The source bootstrap supplies run-dependent overrides to the public compiler
configuration interface; reusable device helpers still take explicit operands.

The source JIT shares immutable compiler state and uses the standard loomc task
pool for concurrent native requests. Each worker owns reusable scratch; each
stage call drains its task queue before returning a complete stage or failure.
This startup join governs compiler/result lifetime, not GPU inference ordering.
The pool is independent of session count and idle during warm serving.

The packed server constructs a bounded Cartesian catalog of token classes and
independent span classes, including exact terminal sizes for odd residency
counts. Explicit `--epoch` lists replace it for experiments. Shapes share one
maximum workspace; proposal storage follows resident count and verification
capture follows each compiled span capacity. Shape changes reuse backing;
elastic row/page growth commits new physical slabs only as capacity is needed.

`text_prepare_weights` delegates cold planning to `loom_serve_weights_create`.
It resolves all target/MTP parameter sharing before I/O, so each unique tensor
is loaded once per activation. Each call selects explicit reflected roots from one checkpoint
domain. Distinct domains, such as an immutable base and its LoRA adapter, load
separately and cannot collide through equal tensor names. Recorded commands
can bind roots from both domains. The model's [`weights.loom`](../models/qwen/weights.loom)
export `prepare_weight(buffer key) -> (buffer command_root, i64 byte_length)`
selects each tensor's transformation. Empty root and zero length mean unchanged
file bytes. Otherwise the loader checks the exact reflected size and JITs each
distinct root once. This query owns Qwen's tensor-name and shape rules; the
native loader has none. Its pure cold VM process is destroyed before the shared
inference process is created, and no session receives VM state.

Creation leaves parameter payloads unread. First inference or explicit
`model_activate` streams them into stable virtual roots. At a retired cut,
`model_deactivate` unmaps their physical slabs while preserving commands,
checkpoint/preparation plans and live mutable state. Re-activation repeats
in-place preparation from original file bytes; an already active model does
no loading. These are explicit lifecycle operations, not automatic LRU
admission. All elastic model reservations charge the same device physical
budget; `--memory_bytes` excludes workspace until it uses a shared queue pool.

Qwen selects a preparation command that permutes FFN gate/up Q5 blocks
in place into eight-channel groups; all other tensors retain their checkpoint
encoding. The inference commands consume this final layout directly, including
decode, narrow and wide packed work, and MTP. No session, shape, or preparer
owns a second weight image.

Four independent loading lanes each have read-ready and prepared-ready
timelines. A lane's next read waits for its previous preparation, while other
lanes can overlap. Consecutive unchanged tensors share a readiness group;
each transformed tensor has its own read-to-dispatch edge. The loader makes
one terminal host join, not a wait after every tensor. Failed readiness
abandons the model; accepted queue operations retain resources until retirement.
The [model guide](../models/qwen/README.md#online-weight-residency) defines
the layout, allocation strategy, and startup profiling recipe.

## One real packed epoch

Before scheduling, `text_enqueue` validates a bounded request without assigning
a device row. `text_admit_pending` selects an idle row for the oldest request,
prepares its actual retained append, and reserves page-rounded completion and
speculative capacity. Active guarantees must fit before any idle cache is
displaced. Physical pages are assigned as epochs grow; idle rows are reclaimed
in LRU order only when admission needs their pages. A completed response gives
back unused credit before network output finishes draining.

Pending requests borrow HTTP payloads until admission, rejection, cancellation,
or shutdown. Connection and body-byte budgets belong to transport; pending
request count belongs to the service. A queued request has no recurrent slot,
page map, or VM instance. Admission is retried on credit/row changes, not on
every decode step. Current policy reserves rather than overcommits capacity.

`text_prepare_ready` first resolves output backpressure and cancellation. A
prompt row contributes its remaining known tokens with minimum one. A decoding
row contributes one pending token, or four reserved inputs for an indivisible
MTP verifier when context and output credit permit it.

The shared `loom_serve_pack_shapes` evaluates the same readiness against each
cached shape. It admits each selected row's minimum, fills remaining token
capacity from longer spans, and advances rotating priority only for the winning
plan. The objective currently maximizes useful tokens, then participating
spans, then prefers smaller token/span capacities. It is not a latency-constrained
cost model. The packer allocates nothing, has no fixed row count, and receives
no model configuration. The bounded text adapter owns its catalog bounds and computes indivisible
verifier lengths before calling it. Model programs implement that semantic
contract with their own numerical layout and stage composition.

`text_execute_epoch` constructs spans with stable resident row indices. Packed
activation offsets are temporary; a row's KV and recurrent state do not move
when its position in the batch changes. Prompt chunks request an output only
at their final input token. Ordinary decode consumes the previously selected
token and selects its successor. A selected token is not yet part of retained
model state until a subsequent invocation consumes it.

`loom_serve_text_model_epoch` validates the whole external span request before
submission, asks source to build immutable device descriptors, uploads input/control,
calls the source VM sequence, and obtains completed output records. Dense
projections share a flat token matrix; attention and recurrent kernels use
span boundaries and row origins to preserve independent histories. The output
head works on requested output rows rather than every prompt token.

The service commits host progress after successful execution, decodes selected
tokens, and copies SSE output into bounded transport storage. A slow client
backpressures its row rather than forcing an unbounded output queue. These
steps repeat under one model owner; another session has no VM instantiation.

## MTP and bounded device-fed cohorts

With `--mtp --mtp_depth=3`, admitted speculative rows reserve the anchor plus
three proposals. The proposal command compacts those rows and executes three
draft rounds, routing samples directly into the next round and verifier input.
The target verifies them together with ordinary known spans. Greedy acceptance
ends at mismatch, EOS, or output credit.

Speculative GDN transitions remain captured until the accepted prefix is
published; attention beyond the committed position remains unreachable.
Catch-up advances private MTP KV/carry using accepted target information.
Proposal, target verification, commit, and catch-up share the work timeline
without an intermediate host readback. By default the host receives the
completed epoch's output/progress records and then chooses the next epoch.
Depth zero with `--mtp` measures a warm draft state without proposal; omitting
`--mtp` measures target-only residency.

`--continuation_epochs=2` grants two device-fed epochs per scheduling turn.
The host pre-issues a second mixed plan from the admitted row cohort, including
remaining prompt chunks and prompt tails ready to begin generation. After
first-epoch catch-up, a source command compacts survivors, advances actual
accepted positions, reduces output credit, copies queued known inputs and
routes pending predictions directly into speculative anchors. A cached shape
bounds the second plan; prompt-only plans skip drafting. Each known chunk
executes once. EOS,
credit and context gates run on device before further state mutation. Distinct
result banks preserve both feedback lifetimes; the host joins once and folds
tagged records into per-session output. KV pages cover the bounded speculative
high-water mark before submission and retire after both branches complete.

This is a bounded continuation experiment, not the continuous admission/output
ring. Transport and new arrivals are observed between cohorts. Fixed commands
still execute padded stateless work if the second cohort becomes empty; the
default remains one pending controlled endpoint measurements. Metrics count
device epochs separately from host scheduling turns and record prompt work
carried into the continuation. Request input offsets advance only by known
consumption, not by speculative inputs generated after the prompt tail.

## Queues, failure, and reclaim

`runner.execute_N(i32 stage, N hal.buffer)` selects one retained command by its
residency-local index and checks its exact binding count. The complete arity
family through the configured capacity is available even when an optional
source branch has no loaded stages. The native call enqueues and returns an
accepted submission value; it does not wait for GPU completion or yield the VM.
That value belongs to one execution object and named timeline, not a global
event namespace.

`runner.feedback(i32 slot, hal.buffer source, i64 offset, i64 length)` downloads
into host storage registered during model creation. The module copies the span
descriptors, not their payload. Their owner retains the backing until accepted
feedback retires, including failure cleanup. A VM-local buffer or a pointer
encoded in a byte buffer is not a valid substitute for that lifetime contract.
Host destination bounds and source-controlled indices are checked at the native
boundary; HAL validates device ranges. Source places this call between target
and catch-up, so the latter does not depend on the host download.

The execution object serializes commands/input transfers with explicit semaphore
edges so workspace reuse is safe even on non-FIFO queues. Feedback branches
wait on their producer and previous feedback without advancing the work
frontier. Independent later work can proceed; a feedback source still cannot be
overwritten until its transfer retires. The model adapter currently waits for
its epoch's result before rescheduling, even though the lower execution API
supports this branch.

A synchronous enqueue rejection advances neither frontier. An accepted failure
is propagated through completion/drain. Host upload and download storage stays
alive until accepted work retires, including teardown after an error. Final
model destruction joins both branches before releasing retained resources.
It ends profiling before releasing command metadata, then releases VM, commands,
buffers and JIT before destroying the device owner. The owner releases its
execution/group/device before its async services. Destroying the owner is not
an implicit drain; it cannot determine which host payloads the model borrowed.
The JIT and weight integration tests use this same owner with actual queues.

## Extension boundaries

| Desired behavior | Concrete boundary that changes |
| --- | --- |
| More rows or wider epochs | Host fixed arrays, descriptor capacities, authored views, scratch sizing, shape selection, and full-sized correctness/performance qualification |
| Online shape insertion | Stage publication and immutable command-table lifetime; cached code and in-flight bindings must remain valid |
| Overcommitted pooled sessions | Replace full-completion admission guarantees with explicit held/offloaded residency and a policy for restoring older sessions; kernels still consume only resident pages |
| Shared prefix cache | Add shared ownership, partial-tail copy-on-write, recurrent snapshots, and retirement to the private-page lifecycle |
| Continuous device-owned continuation | Extend the bounded two-epoch handoff to admission/completion rings with credit, cancellation and independently retired output slots |
| Additional prepared weight formats | Model-specific in-place ownership or bounded scratch, all consuming kernel variants, shared target/auxiliary placement, and startup/inference qualification |
| NPU/GPU or collective execution | Target packages, actual queue/device domains, shared-memory/coherency contracts, and cross-device completion/ownership |

The present stage boundary accepts either kernels composed into commands or a
different implementation with the same model buffer contract. A resident
pipeline within a stage can replace its internals without teaching HTTP about
tiles, weights, or device roles. Cross-stage persistence additionally needs an
explicit channel and lifetime protocol; it is not implied by the existing VM
import's `i64` return value.
