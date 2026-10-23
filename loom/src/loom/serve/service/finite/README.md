# Source-defined finite vector models

This service hosts computations whose result is one or more F32 vectors:
an embedding is a final vector, a decision model produces question
probabilities, and a model component can publish device-resident features.
The native runner owns source JIT, shared
parameter residency, stage submission and result lifetime. The source owns
tokenization/task framing, model shapes, pooling, normalization and learned
heads. It does not require a decode loop, KV reservation or VM process per
request. The HTTP consumers are [embeddings](embedding/protocol.h) and
[SystemOne decisions](decision/README.md).

The [source ABI](model.h) is implemented by the
[deterministic embedding package](../../fixture/embedding/README.md).
It uses the same task executor, parameter cache and queue workspace as text and
image deployments. Model documents name source programs and assets at registration;
replacing one does not require relinking the server. The
[weight-free multimodal catalog](../../fixture/README.md) exercises both
alongside an image model on the same listener.

## Cold and warm source programs

`prepare.loom` exports `prepare(asset_root, options)`. Both inputs are owned VM
byte buffers; `options` contains the catalog's model-owned JSON object. The
existing `prepare` imports declare command roots with `prepare.stage` or native
entries with `prepare.dispatch`, plus specialization configuration. Command
stages can also declare checkpoint domains and preparation policies. The result
contains:

* A positive additive batch-unit limit.
* Maximum upload, physical device-output and live result byte counts.
* A tokenizer asset path relative to `asset_root`, or an empty string.
* A nonempty source name and an opaque source state buffer.

Device kernels and command programs come from the model document's
`definition.sources` array. Command stages that bind checkpoint parameters
share the same fixed parameter layout. The cold loader validates this before
sharing checkpoint storage. It indexes weights without activating them;
admission later streams them into their final storage.
Direct stages currently accept only input/output buffers, without checkpoint
domains. Native argument reflection cannot supply checkpoint tensor names,
placement or lengths; native entries that consume checkpoint parameters require
a source-owned placement contract before they can use this loader.

`control.loom` exports three functions, resolved once in one shared VM process:

```
prepare_request(state, input) -> (prepared_bytes, units, compatibility_tag)
prepare_batch(state, stage_tags, ranges, prepared_bytes)
    -> (stage_index, upload_payload, result_plan)
run(stage_index, input_buffer, output_buffer)
```

The native protocol provides ordered, indexed input records. Its layout is
declared in [embedding/protocol.h](embedding/protocol.h) or
[decision/protocol.h](decision/protocol.h). Source preparation
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

During `prepare_batch`, the same `prepare` capability can declare additional
command/direct stages and bind their configuration or workspace limit. Its
ordinals start after the immutable cold declarations and live for this invocation.
Command stages can select or reorder retained checkpoint domains, or prepare
new domains on demand. Identity includes the source generation, ordered
checkpoint files, preparation policy, and packed placement; equal tensor names
or shapes never substitute for selected contents. Every request stage declares
the same binding map. Omitting declarations preserves the original layout.
Cold declarations remain immutable, and both stage kinds preserve the two
input/output roots. The complete working set includes cold roots because the
source can still invoke those ordinals.

The finite owner captures the selected source generation with the batch. It
resolves exact declaration matches before parameter admission. Choices with the
same stage kinds, roots, and configuration share immutable compiled products;
different checkpoint selections receive their own fixed recordings without
recompilation. Workspace limits are checked independently for each declaration.
Other misses compile asynchronously. Both the finite HTTP scheduler and embedded
text projector use this owner. Source-package uploads can retire after indexing; every choice
shares its generation's immutable source index. The original source namespace
is retained for preparation policies; filesystem inputs remain immutable while
registered. Cancellation joins accepted work before releasing its borrows.

### Forecasting a composed request

A finite component may produce several cohorts whose outputs must remain live
for a downstream consumer. An image projector feeding text is one example:
admitting each projection separately cannot establish that the complete set
of features and the language continuation will fit.

Cold preparation can select a host control entry with
`prepare.memory_forecast(root)`. Its signature is:

```
forecast_memory(state, ranges, descriptors)
    -> (private_peak_bytes, retained_bytes, expanded_bytes)
```

The three inputs are buffers, and the three results are nonnegative i64 byte
estimates for the complete logical request, not just the next producer cohort:

| Estimate | Live storage represented |
| --- | --- |
| `private_peak_bytes` | Peak producer parameters, activation staging, input, and private command workspace, excluding retained output. |
| `retained_bytes` | All output roots carried into the consumer, including padding and other rows sharing those roots. |
| `expanded_bytes` | Gather storage for every output row with 16-byte row alignment, including separate copies of aliased rows. |

The consumer supplies descriptors and the same LE i64 offset/length range
table used by batch preparation. For image-to-text composition, each descriptor
contains the image request header and source-authored metadata, but no pixel
plane. The pixel offset and length describe the eventual execution packet.
The source therefore has its image geometry and resize decisions before RGB
decoding, request specialization, or weight activation. Source state remains
immutable, and this call submits no device work.

Under a finite serving memory limit, the text consumer requires this forecast.
It checks the producer peak plus retained outputs, and separately the language
continuation plus retained outputs and gather storage. A request that cannot
fit alone receives a capacity error before either model activates. These are
trusted source estimates, not limits checked against subsequent compiled
programs. This preflight does not reserve capacity against concurrent requests
or establish that a producer or consumer has retired.

### Retaining specialized programs and checkpoints

`specialization_cache` limits idle choices by count and retained payload bytes.
Zero disables idle retention, not specialization. One active choice can exceed
the idle budget and retire afterward. LRU eviction first releases any runner
table borrowing the victim, then its portable stages, and finally unreferenced
native keys/code. Code shared by another retained choice survives and is charged
only once. Accounting includes requested declaration/table storage and native artifact bytes, not
fixed source/index capacity, compiler scratch, or driver-private allocations.
Management source selection affects future batches without changing a captured
or executing batch.

Prepared checkpoints have independent count and packed-byte retention limits.
The model retains up to 16 entries and 1 GiB of final packed factors; one active
request may exceed the idle limit and retire afterward. These are logical
retention limits, not total host-memory limits. Source indexes, preparation
recipes, and native recording metadata are separate. The shared physical pool
governs actual backing and admits cold plus selected groups atomically. A code
eviction does not discard prepared checkpoints. Parameter eviction releases
dependent fixed recordings before their banks, with idle owner turns joining
accepted retirement. Management pinning and drain include cached banks, and
cumulative activity survives logical eviction. No dense merged base copy is
created.

## Compact results and physical storage

The result plan contains little-endian i64 records:

| Record | Fields |
| --- | --- |
| Header | physical output bytes, live result bytes, request count, row count |
| Per request | first row, row count, input-token usage |
| Per row | byte offset in device output, F32 element count |

Rows are contiguous in original request order. Their physical output locations
can be scattered or reordered. The native model boundary checks this external
plan once against storage capacities and protocol row counts, then copies the
checked request and row metadata into an independent result. Subsequent source
invocations can reuse their plan buffers without changing retained row views.
Physical output and live result extents are separate: stages may need padded
binding storage that a host consumer does not download.

Both stage kinds have two dynamic roots (input/output). Commands additionally
bind private workspace when the compiler reflects a nonzero requirement.
Direct native entries use their compiled launch-config companion and HAL
logical-parameter reflection,
submitting through `queue_dispatch` without a command buffer or invented host
workspace. A resident array program uses this same path; source owns its
device-local storage. Both kinds use the runner's `execute_2` import and publish
device results. The source has no host-feedback destination in this ABI.

Normalization, dimensional projection and other neural math execute in kernels.
The CPU formats final vectors and computes the decision protocol's
answer/confidence statistics from final probabilities. Nonfinite values or a
result width inconsistent with the request are model failures, not an excuse
for native truncation or normalization.

Input/output roots have 16-byte-aligned bases and use the device's shared queue
pool. Interior record alignment remains source-owned. The input and private
command workspace retire after producer work. A result retains the output
allocation independently of the producer's retirement-tracked views, allowing
another cohort, parameter eviction, or model destruction while consumers keep
the result. Device consumers import its accepted readiness frontier and use
checked buffer ranges directly; no host completion wait or payload download is
required to submit dependent work. The shared workspace retains each root's
source pool through its final queue capture.

HTTP consumers explicitly request an exact host snapshot. The download capture
retains that snapshot through actual queue release, even when readiness fails
or every client cancels. A successful snapshot survives until the CPU encoding
process's release callback. The transport copies final responses and does not
retain device output or snapshot storage for a slow peer. Results and snapshots
are request/cohort ownership, not an implicit persistent cache.

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
feedback and HTTP encoding with one/four workers. Its checkpoint variant loads
a scalar scale factor, doubles it on first device activation, and applies it to
the output vectors. It also exercises admission behind another pinned residency.
These are ABI and ownership tests, not model-quality or throughput measurements.
The decision source fixture adds variable question/option counts, exact typed
answers and a mixed-model HTTP gate through the same service.

The embedding fixture's [direct variant](../../fixture/embedding/variants/direct.loom)
fuses the same ragged computation into one native dispatch on CPU, AMDGPU and
Vulkan. Its HTTP checks cover compact row ordering, repeated requests,
cancellation and retirement with one/four workers. Its
[XDNA variant](../../../tooling/target/amd/xdna/serve/testdata/resident_embedding.loom) executes the same
HTTP contract with a resident array program and DMA/channel packet transport.
The [resident checkpoint fixture](../../../tooling/target/amd/xdna/serve/testdata/resident_checkpoint.loom)
loads the same scalar checkpoint, prepares it on device, and binds it to a
staged resident pipeline through a command stage with no global workspace. Its
HTTP checks cover changed inputs, batching, cancellation, and retained
preparation across reuse.
These fixtures qualify source and resource ownership; their deterministic
outputs have no learned embedding meaning.

A real embedding port must still supply its source package, checkpoint policy,
input/task policy, numerical oracle and shape coverage. Target-family selection
and the required queue/command operations must be supported by the selected
HAL; registering an endpoint does not establish NPU execution support. This
runner is not yet a general tensor graph or multimodal workflow interface.
