# Source-defined finite vector models

This experimental runner hosts computations whose final result is one or more
small F32 vectors: an embedding is a final vector, while a decision model can
produce question probabilities. The native runner owns source JIT, shared
parameter residency, command submission and result lifetime. The source owns
tokenization/task framing, model shapes, pooling, normalization and learned
heads. It does not require a decode loop, KV reservation or VM process per
request. The implemented HTTP consumer is [embeddings](../embedding/protocol.h).

The [source ABI](model.h) is implemented by the
[deterministic embedding package](../models/fixtures/embedding/README.md).
It uses the same task executor, parameter cache and queue workspace as text and
image deployments. Source packages are loaded from directories at startup;
replacing one does not require relinking the server.

## Cold and warm source programs

`prepare.loom` exports `prepare(asset_root, options)`. Both inputs are owned VM
byte buffers; `options` contains the catalog's model-owned JSON object. The
existing `prepare` imports declare command roots, specialization configuration,
checkpoint domains and preparation policies. The result contains:

* A positive additive batch-unit limit.
* Maximum upload, physical device-output and compact host-feedback byte counts.
* A tokenizer asset path relative to `asset_root`, or an empty string.
* A nonempty source name and an opaque source state buffer.

Device kernels and command programs come from the ordinary `sources.txt`
catalog. Every retained stage uses the same fixed parameter layout. The cold
loader validates this before sharing checkpoint storage. It indexes weights
without activating them; admission later streams them into their final storage.

`control.loom` exports three functions, resolved once in one shared VM process:

```
prepare_request(state, input) -> (prepared_bytes, units, compatibility_tag)
prepare_batch(state, stage_tags, ranges, prepared_bytes)
    -> (stage_index, upload_payload, result_plan)
run(stage_index, feedback_bytes, input_buffer, output_buffer)
```

The native protocol provides ordered, indexed input records. Its layout is
declared in [embedding/protocol.h](../embedding/protocol.h) or
[decision/protocol.h](../decision/protocol.h). Source preparation
validates model-specific limits through `input.require`; an invalid input is a
request rejection, not a failed device. The state is immutable across request
preparation. Returned buffers own their storage or retain any aliased input.

`units` is a positive additive cost within the declared batch limit. It bounds
the source's actual work and storage; the native scheduler does not assume that
it means tokens. Equal compatibility tags indicate requests the source can
pack together. The scheduler starts with the oldest prepared request and fills
its cohort from compatible requests already waiting. It never delays a cohort
to collect hypothetical arrivals.

Batch preparation receives LE i64 `[offset, length]` pairs indexing concatenated
prepared bytes in stable request order. It receives stage tags from cold
declarations, so command-bucket choice stays in source. A denied residency
admission retains the prepared cohort without repeating source preparation.

## Compact results and physical storage

The result plan contains little-endian i64 records:

| Record | Fields |
| --- | --- |
| Header | physical output bytes, compact feedback bytes, request count, row count |
| Per request | first row, row count, input-token usage |
| Per row | byte offset in feedback, F32 element count |

Rows are contiguous in original request order. Their physical output locations
can be scattered or reordered. The native model boundary checks this external
plan once against storage capacities and protocol row counts; the encoder then
uses trusted row views. Physical output and live feedback extents are separate:
commands may need padded binding storage without downloading the padding.

The current command ABI has two dynamic roots (input/output) and a
compiler-reflected private workspace. `run` uses the runner's `execute_2` and
`feedback` imports. Normalization, dimensional projection and other neural math
execute in kernels. The CPU formats final vectors and computes the decision
protocol's answer/confidence statistics from final probabilities. Nonfinite values or
a result width inconsistent with the request are model failures, not an excuse
for native truncation or normalization.

Input/output roots borrow the device's shared queue pool only while executing.
The input retires after work. Output retirement imports the feedback frontier
onto the work timeline before `queue_dealloca`, so reuse follows the final
reader rather than queue submission order. Private command workspace can retire
earlier. The source payload survives upload; compact host feedback survives
until the CPU encoding process's release callback. The transport copies final
responses and does not retain a model's output allocation for a slow peer.

One reset client never edits an already submitted cohort. Its output is
discarded while the other requests keep their original row associations.
Device errors publish status for outer-owner wind-down; callbacks never destroy
queues or pools, and failed semaphore readiness is not a resource-retirement
proof.

## Evidence and extension boundary

[service.h](service.h) owns the common asynchronous cohort lifecycle.
[protocol.h](protocol.h) supplies immutable native modality wiring for parsing
and encoding. Each accepted job contains its concrete request storage in one
allocation; model programs are not protocol callbacks. Embeddings and decisions
share preparation, admission, device readiness, encoding lifetime and retirement.

The fixture runs actual source JIT, two GPU kernels, private workspace, compact
feedback and HTTP encoding with one/four workers. Its weighted source variant
adds a checkpoint-backed head without changing native code and exercises
admission behind another pinned residency. These are ABI and ownership tests,
not model-quality or throughput measurements. The decision source fixture adds
variable question/option counts, exact typed answers and a mixed-model HTTP gate
through the same service.

A real embedding port must still supply its source package, checkpoint policy,
input/task policy, numerical oracle and shape coverage. Target-family selection
and the required queue/command operations must be supported by the selected
HAL; registering an endpoint does not establish NPU execution support. This
runner is not yet a general tensor graph or multimodal workflow interface.
