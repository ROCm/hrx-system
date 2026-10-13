# Source-defined finite vector models

This experimental runner hosts computations whose final result is one or more
small F32 vectors: an embedding is a final vector, while a decision model can
produce question probabilities. The native runner owns source JIT, shared
parameter residency, stage submission and result lifetime. The source owns
tokenization/task framing, model shapes, pooling, normalization and learned
heads. It does not require a decode loop, KV reservation or VM process per
request. The HTTP consumers are [embeddings](../embedding/protocol.h) and
[SystemOne decisions](../decision/README.md).

The [source ABI](model.h) is implemented by the
[deterministic embedding package](../models/fixtures/embedding/README.md).
It uses the same task executor, parameter cache and queue workspace as text and
image deployments. Model documents name source programs and assets at registration;
replacing one does not require relinking the server. The
[weight-free multimodal catalog](../models/fixtures/README.md) exercises both
alongside an image model on the same listener.

## Cold and warm source programs

`prepare.loom` exports `prepare(asset_root, options)`. Both inputs are owned VM
byte buffers; `options` contains the catalog's model-owned JSON object. The
existing `prepare` imports declare command roots with `prepare.stage` or native
entries with `prepare.dispatch`, plus specialization configuration. Command
stages can also declare checkpoint domains and preparation policies. The result
contains:

* A positive additive batch-unit limit.
* Maximum upload, physical device-output and compact host-feedback byte counts.
* A tokenizer asset path relative to `asset_root`, or an empty string.
* A nonempty source name and an opaque source state buffer.

Device kernels and command programs come from the model document's
`definition.sources` array. Weighted command stages use the same fixed parameter layout. The cold
loader validates this before sharing checkpoint storage. It indexes weights
without activating them; admission later streams them into their final storage.
Direct stages currently accept only input/output buffers, without checkpoint
domains. Native argument reflection cannot supply checkpoint tensor names,
placement or lengths; weighted native entries require a source-owned placement
contract before they can use this loader.

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
declarations, so stage selection stays in source. A denied residency
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
stages may need padded binding storage without downloading the padding.

Both stage kinds have two dynamic roots (input/output). Commands additionally
have compiler-reflected private workspace. Direct native entries instead use
their compiled launch-config companion and HAL logical-parameter reflection,
submitting through `queue_dispatch` without a command buffer or invented host
workspace. A resident array program uses this same path; source owns its
device-local storage. Both kinds use the runner's `execute_2` and `feedback`
imports with the same model timelines and retirement rules.

Normalization, dimensional projection and other neural math execute in kernels.
The CPU formats final vectors and computes the decision protocol's
answer/confidence statistics from final probabilities. Nonfinite values or a
result width inconsistent with the request are model failures, not an excuse
for native truncation or normalization.

Input/output roots have 16-byte-aligned bases and borrow the device's shared
queue pool only while executing. Interior record alignment remains source-owned.
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

The embedding fixture's [direct variant](../models/fixtures/embedding/variants/direct.loom)
fuses the same ragged computation into one native dispatch on CPU, AMDGPU and
Vulkan. Its HTTP checks cover compact row ordering, repeated requests,
cancellation and retirement with one/four workers. Its
[XDNA variant](../models/fixtures/embedding/variants/xdna.loom) executes the same
HTTP contract with a resident array program and DMA/channel packet transport.
These prove direct-stage ownership, including untouched first output, not
learned embedding quality or checkpoint-backed resident stages.

A real embedding port must still supply its source package, checkpoint policy,
input/task policy, numerical oracle and shape coverage. Target-family selection
and the required queue/command operations must be supported by the selected
HAL; registering an endpoint does not establish NPU execution support. This
runner is not yet a general tensor graph or multimodal workflow interface.
