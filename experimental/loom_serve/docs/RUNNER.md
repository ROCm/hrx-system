# Runner ownership and scheduling

The runner connects source-authored math to explicit queue submission. Its VM
is a model program, not a container for each chat session. One host owner
multiplexes rows through shared code, weights, and workspace. Transport and
heartbeat progress are independent of the model owner's GPU wait.

## Boundaries and the information each owns

| Component | Receives and owns | Does not infer |
| --- | --- | --- |
| [`jit`](../jit.h) | Source catalog, configuration, live device profile; compiled command images and native entries | Session identity, cache lifetime, chat semantics |
| [`command`](../command.h) | Compiler-produced parameter/binding requirements, executable reflection; reusable HAL command recording | Model graph from buffer contents or filenames |
| [`execution`](../execution.h) | Exact dispatch/transfer queues and explicit work/feedback timelines | Ordering from FIFO submission or alias inspection |
| [`module`](../module.h) | Named prepared stages; typed VM imports accepting buffers and returning a submission value | Per-session VM state or a general HAL instruction set |
| [`qwen_model`](../qwen_model.h) | Weight interpretation, row/state layout, scratch, descriptor construction, numerical progress | HTTP or tool semantics |
| [`qwen_weights`](../qwen_weights.h) | Shared parameter residency, model-specific preparation, file-read/preparation readiness | Session state or queue ordering from submission order |
| [`qwen_schedule`](../qwen_schedule.h) | Trusted ready span lengths, indivisible minima, shapes, rotating priority | Tokens, attention state, measured kernel cost |
| [`qwen_service`](../qwen_service.h) | Validated chat, session keys, canonical history, output credit, scheduling policy | Kernel layout decisions |
| [`http_server`](../http_server.h) | Bounded HTTP framing and copied response bytes over IREE TCP carriers | Model sessions or sampling |

These are runner-private seams, not a proposed public serving ABI. The
compiler/runtime handoff is the command product: its parameter placement,
entry requirements, launch structure, and transient requirements are consumed
directly, rather than reconstructed in host code.

## Cold residency

`qwen_initialize` creates the live device and JIT, loads the tokenizer, compiles
isolated and packed stages, checks their layout agreement, loads shared weights,
records commands, creates the VM program/native exports, and allocates retained
rows and MTP state. Each stage can have different kernel choices while binding
the same model storage. No session gets another copy of the weights or code.

`qwen_load_weights` delegates cold residency to `loom_serve_qwen_weights_load`.
It resolves all target/MTP parameter sharing before I/O, so each unique tensor
is loaded once. A source-JIT preparation command permutes FFN gate/up Q5 blocks
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
The [model guide](../models/qwen38/README.md#online-weight-residency) defines
the layout, allocation strategy, and startup profiling recipe.

## One real packed epoch

`qwen_prepare_ready` first resolves output backpressure and cancellation. A
prompt row contributes its remaining known tokens with minimum one. A decoding
row contributes one pending token, or four reserved inputs for an indivisible
MTP verifier when context and output credit permit it.

`loom_serve_qwen_schedule_shapes` evaluates the same readiness against each
cached shape. It admits each selected row's minimum, fills remaining token
capacity from longer spans, and advances rotating priority only for the winning
plan. The objective currently maximizes useful tokens and breaks ties with
smaller token/span capacities. It is not a latency-constrained cost model.

`qwen_execute_epoch` constructs spans with stable resident row indices. Packed
activation offsets are temporary; a row's KV and recurrent state do not move
when its position in the batch changes. Prompt chunks request an output only
at their final input token. Ordinary decode consumes the previously selected
token and selects its successor. A selected token is not yet part of retained
model state until a subsequent invocation consumes it.

`loom_serve_qwen_model_epoch` validates the whole external span request before
submission, builds the immutable device descriptors, uploads input/control,
submits the prepared stage, and obtains completed output records. Dense
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

The native import enqueues a prepared command and returns an accepted submission
value. It does not wait for GPU completion or yield the VM on every dispatch.
That value belongs to one execution object and named timeline, not a global
event namespace.

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
The small control tests exercise these contracts with actual queues.

## Extension boundaries

| Desired behavior | Concrete boundary that changes |
| --- | --- |
| More rows or wider epochs | Host fixed arrays, descriptor capacities, authored views, scratch sizing, shape selection, and full-sized correctness/performance qualification |
| Online shape insertion | Stage publication and immutable native export lifetime; cached code and in-flight bindings must remain valid |
| Shared block-pool/prefix cache | Replace contiguous row origins with a page-map contract; explicit shared ownership, partial-tail copy-on-write, and recurrent snapshots |
| Device-owned continuation | Admission/completion rings with credit and cancellation; row progress and token routing leave the host epoch wait without recycling in-flight buffers |
| Additional prepared weight formats | Model-specific in-place ownership or bounded scratch, all consuming kernel variants, shared target/auxiliary placement, and startup/inference qualification |
| NPU/GPU or collective execution | Target packages, actual queue/device domains, shared-memory/coherency contracts, and cross-device completion/ownership |

The present stage boundary accepts either kernels composed into commands or a
different implementation with the same model buffer contract. A resident
pipeline within a stage can replace its internals without teaching HTTP about
tiles, weights, or device roles. Cross-stage persistence additionally needs an
explicit channel and lifetime protocol; it is not implied by the existing VM
import's `i64` return value.
