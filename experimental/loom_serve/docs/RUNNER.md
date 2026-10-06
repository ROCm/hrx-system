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
the shared declaration module. State geometry and descriptor construction still
live in the Qwen adapter; source-owned startup does not make those remaining
contracts model-independent.

## Boundaries and the information each owns

| Component | Receives and owns | Does not infer |
| --- | --- | --- |
| [`device`](../runtime/device.h) | HAL device URI, async I/O services, device/group, exact queues and execution timelines | Compiler target support, model resources or request lifetimes |
| [`jit`](../runtime/jit.h) | Source catalog, configuration, live device profile; compiled command images and native entries | Session identity, cache lifetime, chat semantics |
| [`command`](../runtime/command.h) | Compiler-produced parameter/binding requirements, executable reflection; reusable HAL command recording | Model graph from buffer contents or filenames |
| [`execution`](../runtime/execution.h) | Exact dispatch/transfer queues and explicit work/feedback timelines | Ordering from FIFO submission or alias inspection |
| [`module`](../runtime/module.h) | Indexed prepared commands, typed execute imports and registered host feedback spans | Per-session VM state, model stages, or a general HAL instruction set |
| [`program`](../runtime/program.h) | Source-JIT bytecode, linked libraries, one process and serialized invocation | Model geometry or the lifetime of asynchronously borrowed host payloads |
| [`preparation`](../runtime/preparation.h) | Cold source declarations of commands, configuration and checkpoint domains, copied independently of the bootstrap process | Model geometry, stage semantics, or GPU allocation |
| [`qwen_model`](../models/qwen/model.h) | Weight interpretation, row/state layout, scratch, descriptor construction, numerical progress | HTTP or tool semantics |
| [`weights`](../runtime/weights.h) | Shared parameter residency, source policy queries, cached preparers, file-read/preparation readiness | Tensor naming rules, model geometry, or ordering from submission order |
| [`packing`](../scheduling/packing.h) | Trusted ready span lengths, indivisible minima, shapes, rotating priority | Token values, model identity, attention state, measured kernel cost |
| [`qwen_schedule`](../models/qwen/schedule.h) | Default model shape catalog and completion-reservation extent | HTTP output credit or committed model progress |
| [`qwen_service`](../models/qwen/service.h) | Validated chat, session keys, canonical history, output credit, scheduling policy | Kernel layout decisions |
| [`http_server`](../http/server.h) | Bounded HTTP framing and copied response bytes over IREE TCP carriers | Model sessions or sampling |

These are runner-private seams, not a proposed public serving ABI. The
compiler/runtime handoff is the command product: its parameter placement,
entry requirements, launch structure, and transient requirements are consumed
directly, rather than reconstructed in host code.

## Cold residency

`qwen_initialize` first invokes the source bootstrap, retains its opaque control
state, and loads the tokenizer with the source-declared terminal marker. It then
creates a shared device owner and JIT, compiles the declared stages, checks their
layout agreement, loads shared weights, records commands, creates the warm VM
program/native capabilities, and allocates retained rows and MTP state. Each
stage can have different kernel choices while binding the same model storage.
No session gets another copy of the weights or code.

`loom_serve_device_create` establishes one runtime domain from a HAL device
URI. Model components borrow its device, group, exact queues and execution
object; the helper does not contain a tokenizer, stage catalog or model policy.
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
capture follows each compiled span capacity. Neither a shape change nor a row's
page growth allocates more device backing during steady-state execution.

`qwen_load_weights` delegates cold residency to `loom_serve_weights_load`.
It resolves all target/MTP parameter sharing before I/O, so each unique tensor
is loaded once. Each call selects explicit reflected roots from one checkpoint
domain. Distinct domains, such as an immutable base and its LoRA adapter, load
separately and cannot collide through equal tensor names. Recorded commands
can bind roots from both domains. The model's [`weights.loom`](../models/qwen/weights.loom)
export `prepare_weight(buffer key) -> (buffer command_root, i64 byte_length)`
selects each tensor's transformation. Empty root and zero length mean unchanged
file bytes. Otherwise the loader checks the exact reflected size and JITs each
distinct root once. This query owns Qwen's tensor-name and shape rules; the
native loader has none. Its pure cold VM process is destroyed before the shared
inference process is created, and no session receives VM state.

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

Before scheduling, `qwen_enqueue` validates a bounded request without assigning
a device row. `qwen_admit_pending` selects an idle row for the oldest request,
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

`qwen_prepare_ready` first resolves output backpressure and cancellation. A
prompt row contributes its remaining known tokens with minimum one. A decoding
row contributes one pending token, or four reserved inputs for an indivisible
MTP verifier when context and output credit permit it.

The shared `loom_serve_pack_shapes` evaluates the same readiness against each
cached shape. It admits each selected row's minimum, fills remaining token
capacity from longer spans, and advances rotating priority only for the winning
plan. The objective currently maximizes useful tokens, then participating
spans, then prefers smaller token/span capacities. It is not a latency-constrained
cost model. The packer allocates nothing, has no fixed row count, and receives
no model configuration. The Qwen adapter owns its catalog bounds and computes
indivisible verifier lengths before calling it; other sequence models can
reuse packing without adopting those Qwen constraints.

`qwen_execute_epoch` constructs spans with stable resident row indices. Packed
activation offsets are temporary; a row's KV and recurrent state do not move
when its position in the batch changes. Prompt chunks request an output only
at their final input token. Ordinary decode consumes the previously selected
token and selects its successor. A selected token is not yet part of retained
model state until a subsequent invocation consumes it.

`loom_serve_qwen_model_epoch` validates the whole external span request before
submission, builds the immutable device descriptors, uploads input/control,
calls the source VM sequence, and obtains completed output records. Dense
projections share a flat token matrix; attention and recurrent kernels use
span boundaries and row origins to preserve independent histories. The output
head works on requested output rows rather than every prompt token.

The service commits host progress after successful execution, decodes selected
tokens, and copies SSE output into bounded transport storage. A slow client
backpressures its row rather than forcing an unbounded output queue. These
steps repeat under one model owner; another session has no VM instantiation.

## MTP is a device chain inside that epoch boundary

With `--mtp --mtp_depth=3`, admitted speculative rows reserve the anchor plus
three proposals. The proposal command compacts those rows and executes three
draft rounds, routing samples directly into the next round and verifier input.
The target verifies them together with ordinary known spans. Greedy acceptance
ends at mismatch, EOS, or output credit.

Speculative GDN transitions remain captured until the accepted prefix is
published; attention beyond the committed position remains unreachable.
Catch-up advances private MTP KV/carry using accepted target information.
Proposal, target verification, commit, and catch-up share the work timeline
without an intermediate host readback. The host receives the completed epoch's
output/progress records and then chooses the next epoch. Thus this is not yet
an autonomous device decode loop. Depth zero with `--mtp` measures a warm draft
state without proposal; omitting `--mtp` measures target-only residency.

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
| Device-owned continuation | Admission/completion rings with credit and cancellation; row progress and token routing leave the host epoch wait without recycling in-flight buffers |
| Additional prepared weight formats | Model-specific in-place ownership or bounded scratch, all consuming kernel variants, shared target/auxiliary placement, and startup/inference qualification |
| NPU/GPU or collective execution | Target packages, actual queue/device domains, shared-memory/coherency contracts, and cross-device completion/ownership |

The present stage boundary accepts either kernels composed into commands or a
different implementation with the same model buffer contract. A resident
pipeline within a stage can replace its internals without teaching HTTP about
tiles, weights, or device roles. Cross-stage persistence additionally needs an
explicit channel and lifetime protocol; it is not implied by the existing VM
import's `i64` return value.
