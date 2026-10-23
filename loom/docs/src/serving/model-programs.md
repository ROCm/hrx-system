# Author a model package for loom-serve

A serving package combines a model's numerical programs with its input,
parameter and state contracts. The server supplies shared workers, JIT,
device queues, memory ownership, request scheduling and HTTP encoding. Adding
a model that fits an existing modality changes the package, not the server's C
code or executable.

The [first-output walkthrough](workflows/serve-models.md) runs a complete
arithmetic package before weights or neural accuracy enter the picture. That
is a useful starting point for a port: changing its math preserves a working
path through registration, execution, output and retirement.

## Select the execution contract

The important distinction is the shape of state and output, not the model name:

| Contract | Source responsibilities | Starting package |
| --- | --- | --- |
| Retained text | Token framing, paged state geometry, mixed prompt/decode math, progress, sampling and chat policy. | [Tiny text model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/text) |
| Staged image | Prompt preparation, parameter domains, numerical stages, invocation continuation and final RGB. | [Image model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/image) |
| Finite embeddings | Input preparation, cohort cost/compatibility, pooling/projection and final vectors. | [Embedding model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/embedding) |
| Finite decisions | Structured question framing and compact normalized answer distributions. | [Decision model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/decision) |
| Streaming speech | Voice conditioning, independent request state, compatible next-work batches and continuous waveform blocks. | [Speech model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/speech) |
| Continuing recognition | Independent input clocks, compatible next-work batches and revisable transcript segments. | [Recognition model](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/recognition) |

These packages use actual source VM programs, native kernels and production
HTTP services, not canned native replies. They establish the serving contract,
not learned quality or competitive neural performance. Each learned port needs
its own numerical oracle and representative workload.

Image-conditioned text composes retained text with an optional finite image
projector, described below. Audio input or an arbitrary workflow graph needs
its own input/output ownership contract beyond these HTTP profiles, not an
embedding vector with overloaded fields. The [HTTP reference](http-profiles.md)
defines the existing boundary.

## Keep the package complete and relocatable

A typical package contains:

```text
model.json           deployment, source roles, assets and capacities
prepare.loom         cold declarations and parameter interpretation
control.loom         warm request and execution policy
config.loom          model/target specialization inputs
kernel.loom          numerical providers and reusable helpers
tokenizer.json       tokenizer asset, when needed
weights.safetensors  checkpoint selected by preparation policy
```

These filenames are conventions, not hardcoded server paths.
The [model document](model-documents.md) names the
actual files. GGUF and safetensors inputs follow the selected preparation
policy; a file extension or model name does not convert an arbitrary checkpoint.
Its keys, encodings and layouts must agree with the authored source.

All program roles and numerical providers accept `.loom` or `.loombc`.
Bytecode is portable IR and still JITs for the selected device. References in a
file document resolve from its directory, including references to shared motif
libraries elsewhere. JSON-only HTTP registration supplies its absolute
server-local base. A multipart source package instead carries programs and
metadata in a separate logical namespace while parameter files stay server-local.
The [live development workflow](workflows/live-model-development.md)
shows upload, native selection and retained-state comparisons.

`definition.sources` supplies shared VM and native providers. Roots and
referenced symbols determine dependency closure through the
normal Loom linker. Reusable helpers take SSA values or template arguments;
model-specific configuration belongs at program/kernel boundaries. This keeps
a motif usable without importing another model's global configuration namespace.

Preparation and control VM files each link against that catalog as primary
inputs. Ordinary function declarations and template applications resolve to
exported library definitions; library-private helpers stay private. This lets
preparation, control and numerical commands share layout calculations without
duplicating them across files.

Explicit `func.decl import("module")` declarations remain capability imports,
bound to native modules when the VM program loads. The cold program names
numerical roots; the warm program invokes the resulting prepared stages.
Shared source helpers do not turn device kernels into direct VM calls.

## Separate preparation from execution

The cold `prepare` VM program receives modality inputs and the source-owned
`options` object. It validates architecture constraints and declares stages,
configuration and checkpoint domains. Shared preparation capabilities include:

| Import in module `prepare` | Meaning |
| --- | --- |
| `stage(root, tag)` | Declare a command-program root and a source-defined selection tag. |
| `dispatch(root, tag)` | Declare a buffer-only native kernel root with its compiled launch configuration. |
| `config(stage, key, value)` | Bind a public Loom configuration value in textual syntax. |
| `config_i64(stage, key, value)` | Bind a signed integer configuration value. |
| `workspace_limit(stage, bytes)` | Bound command workspace, including compatible live replacements. |
| `memory_forecast(root)` | Select a finite component's host control entry for complete-request memory estimates. |
| `source(directory, path)` | Declare the first shard of a checkpoint and return its source ordinal. |
| `source_append(source, directory, path)` | Append a shard to that checkpoint's ordered index. |
| `source_scope(scope)` | Select a model card's named, typed parameter files as one shared checkpoint source. |
| `parameter(stage, binding, input, source, policy)` | Bind a checkpoint to one policy-local input of a reflected fixed parameter binding. |

A split checkpoint uses `source` for its first file and `source_append` for
each additional shard. The files populate one index in declaration order.
The same source can supply several resident bindings or stages without repeating
the file list. Sources identify bytes, independently of resident placement or
preparation policy; a source declaration by itself loads no parameter payload.
Cold preparation opens each declared source once and shares its index and
imported-file cache across consuming bindings. Request choices retain the cold
prefix and index new sources on the preparation worker; warm execution does not
reopen checkpoint paths. A missing or invalid checkpoint fails preparation
before its parameter placement or activation.

Each fixed binding declares its policy inputs once, densely from zero. A policy
can combine independent checkpoints containing the same keys: its `weights`
imports use those local input ordinals, not model-wide source ordinals. Stages
naming the same ordered file lists and policy share a resident domain. The
loader accepts explicit GGUF, safetensors, and other supported parameter files
without inferring filenames.

Request preparation can declare additional sources after the immutable cold
source prefix, or bind that prefix to new stages. It cannot append shards to a
cold source. Each request choice owns its declarations independently; selecting
another choice does not change the checkpoint identity of accepted work.

The exact signatures and ownership are in the
[preparation contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/runtime/program/preparation.h).
Declarations copy their strings; opaque returned state must own its storage or
retain its inputs. Preparation is not a place to submit device work.

The server resolves these declarations, compiles through its shared compiler
task queue, and retains the selected stages for the model. Configuration can
specialize stage shapes or layouts without giving every session its own program.
Compatible requests share the prepared code and parameters. Warm work does not
invoke a compiler subprocess or redo unchanged model preparation.

The warm `control` VM imports the runner-local `runner.execute_N` and
`runner.feedback` capabilities. `execute_N` submits a prepared stage with its
buffers; `feedback` transfers a bounded final result into a registered host slot.
Both enqueue work without waiting. The native model owner schedules completion
on work/feedback timelines and preserves buffers until their actual last use.
The [runner-module contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/runtime/program/module.h)
defines arity, buffer types and lifetime. This is a coarse serving capability,
not a source program's ownership of the entire device or process scheduler.

Independent work can use explicit dependencies. `runner.fence(work)` returns a
`hal.fence` for a previously accepted work value. It captures that producer,
not later submissions or the separate feedback branch. `runner.join(ready)`
orders subsequent model work after the supplied fence without waiting on the
CPU or synchronizing other models. A null fence represents no external work.
Completion describes readiness; storage ownership still extends through the
last accepted consumer, including when a dependency fails.

### Read checkpoint bytes on demand

Large lookup tables and paged experts can remain in checkpoint files instead
of occupying a resident parameter binding. The warm `parameters` capability
reads demanded byte ranges from the same sources declared during preparation:

```text
parameters.gather(source, key, wait, descriptors, descriptor_offset, count, target)
    -> hal.fence
```

`source` is the selected invocation's source ordinal, and `key` is a host byte
buffer naming an entry in that checkpoint. Independent sources may use the
same key. `descriptors` and `target` are `hal.buffer` references. Starting at
`descriptor_offset`, each 24-byte record contains three little-endian `u64`
values: the entry-relative byte offset, destination-buffer byte offset, and
byte length. Destination ranges are disjoint; zero-length records require no
payload transfer. Hashing, deduplication, packing, and cache placement remain
model-program decisions, not tensor interpretations in the server.

A control program can enqueue independent reads before joining either one:

```text
producer = runner.execute_N(descriptor_stage, ...)
produced = runner.fence(producer)
first = parameters.gather(source_a, key_a, produced, ranges_a, 0, count_a, target_a)
second = parameters.gather(source_b, key_b, produced, ranges_b, 0, count_b, target_b)
runner.execute_N(independent_stage, ...)
runner.join(first)
runner.execute_N(first_consumer, ...)
runner.join(second)
runner.execute_N(second_consumer, ...)
```

The supplied fence covers descriptor production and destination availability,
including any earlier readers of reused storage. The returned fence orders
payload consumers and descriptor reuse. An empty gather preserves its incoming
dependency without touching storage. Changing a request's source choice or
removing its model does not redirect or release accepted reads: each read
retains its original provider and buffer captures until actual retirement.
Failed readiness is not permission to recycle those buffers.

This implementation asynchronously snapshots the descriptors to host memory,
then submits the ranges through the parameter provider and HAL file queues.
The VM call never waits for completion or synchronizes unrelated models. It
requires queue download and deferred host-call support. Checkpoint opening
uses the configured parameter-file mode; direct I/O requires appropriately
aligned file ranges and destinations and does not silently fall back to
buffered reads. A snapshot costs 24 bytes per descriptor plus one copied key;
checkpoint payloads are not materialized in a separate model-owned host array.

This is host-VM-initiated asynchronous I/O, not a persistent GPU request
service. A kernel that requests data while continuing other work needs a
device-side progress and credit protocol beyond this call. The shared source
identity and byte-range contract do not impose a model-wide barrier on that
separate execution mechanism.

### Tokenize request inputs

Request preparation can use the `input` capability to encode text and decode
token spans with the deployment's tokenizer. `encode(text, flags, capacity)`
returns a zero-padded little-endian I32 ID buffer and live token count; it can
stop at the requested capacity. `decode(ids, flags, byte_capacity)` consumes
the complete ID buffer and returns a zero-padded UTF-8 buffer and live byte
length. An insufficient decoding capacity fails instead of returning partial
text. Pass only live IDs, excluding encoding padding. Each call starts fresh
tokenizer state, including leading-space handling. Encoding flags are
`ADD_SPECIAL_TOKENS = 4` and `NO_SPECIAL_TOKEN_MATCHING = 8`; decoding supports
`SKIP_SPECIAL_TOKENS = 1`, with zero retaining special tokens. The source owns
text normalization, sentence boundaries, and capacity admission.

## Trace one finite request end to end

The embedding package is the smallest complete example. Its cold entry is
`prepare(asset_root, options)`. It returns a batch-unit limit, upload/output/
result capacities, an optional tokenizer asset, a source name and opaque state.
Its control program exports:

```text
prepare_request(state, input) -> (prepared_bytes, units, compatibility_tag)
prepare_batch(state, stage_tags, ranges, prepared_bytes)
    -> (stage_index, upload_payload, result_plan)
run(stage_index, input_buffer, output_buffer)
```

The HTTP parser provides indexed protocol records, not a source-private JSON
guess. `prepare_request` performs model framing and validation. `units` is a
positive additive bound on real work/storage; it need not mean tokens. Equal
compatibility tags mean requests can share a cohort. Source state is immutable
across request preparation, and returned buffers retain any aliased input.

The native scheduler starts with the oldest prepared request and fills the
cohort from compatible waiting requests. It does not wait for hypothetical
arrivals. Batch preparation receives stable request-order ranges over concatenated
prepared bytes, selects a declared stage and maps physical output back to requests.

Finite `prepare_batch` can also declare stages with the `prepare` capability.
It uses the same `stage`, `dispatch`, `config`, `config_i64`, and
`workspace_limit` imports as cold preparation. Returned ordinals follow the
cold stages and are valid for that invocation. This lets a package specialize
an actual image geometry or cohort extent without enumerating a size catalog
at registration. A request can also declare sources and parameter bindings;
omitting parameter declarations preserves the cold layout. New source ordinals
follow the immutable cold prefix and belong to that exact request choice.

The server reuses an exact declaration set within its selected source generation
or compiles it asynchronously before parameter admission. Other models keep
running on the shared workers. Cancellation joins accepted compilation before
releasing its source and invocation, and cached choices are bounded by the
deployment's [`specialization_cache`](model-documents.md). Neither specialization
nor image-to-text composition requires downloading hidden features to the host.

The result plan describes physical output bytes separately from the live result
prefix, then each request's rows and each row's offset/width. Results retain
their device storage independently of the producing model. Device consumers
import the producer's readiness and read those rows without a host roundtrip.
HTTP consumers explicitly download a compact snapshot for encoding; device
padding and activation storage need not be downloaded. Projection and
normalization happen in the numerical program. Decision encoding similarly
consumes final probabilities, not hidden activations.

The example deliberately reverses physical output rows. Correct client indices
therefore prove that result mapping works rather than merely agreeing with a
convenient device layout. Its independent arithmetic oracle also makes ragged
packing and cohort association observable. See the
[finite-source contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/finite/README.md)
for exact record layouts and ownership.

## Retained text and staged images

Text uses one prepared model and cached stage set. Sessions are rows and storage
references, not independent VM processes. Its source declares page/recurrent
geometry and consumes packed spans that can mix prompt tokens, decode anchors
and speculative verifier work. Token capacity and independently advancing span
capacity are separate. The source reports completed progress; native ownership
commits it after readiness.

Shared prefixes and explicit checkpoints keep immutable endpoint references.
A branch needs a private recurrent writer without copying the entire KV prefix.
The tiny text package exercises a partial-page frontier, an unconsumed pending
token, fork/rewind, cancellation and full-replay equivalence. Those boundaries
matter more than getting one unretained prompt to emit plausible text. The
[text service](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/README.md)
links the storage, epoch and chat-policy declarations beside their owners.

Cold image preparation selects stages and parameter domains for default geometry.
Warm source receives the requested height and width alongside prompt, seed, and
decoded adapter ID/scale records. It resolves IDs against its registrations,
validates geometry, and returns opaque request bytes with work and compatibility
facts. Compatible requests form one cohort; source batch preparation selects a
cold stage or declares stages through the same invocation-only `prepare`
capability as finite cohorts. Exact declaration sets reuse code within the
selected source generation; misses compile before parameter admission. Explicit
checkpoint declarations select retained banks or prepare independent banks on
demand; omitted declarations preserve the cold parameter layout. Stages within
one cohort share a parameter binding map.

Image control then drives command parts to final RGB. It returns a continuation
cursor; another invocation occurs only after the previous accepted frontiers
retire. Compiler-reflected workspace remains live across barrier-aligned parts.
RGB and compiler-aligned workspace share one cohort-sized pool allocation,
retained through final feedback. Each request keeps a stable output slice even
if a peer is cancelled. Neither storage nor pixel geometry belongs to the
executable cache. Final output is F32 CHW RGB in `[-1,1]` before native image
encoding uses each request's dimensions.
The [image source contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/image/model.h)
defines entry signatures, alignment and cancellation.

## Known-token stages

A `sequence` deployment uses retained rows, packed epochs, block pools, and
checkpoints without the generative chat contract. It consumes only supplied
known IDs and publishes source-defined opaque output. It can implement an
encoder or another retained stage; a source may also run a head and include its
result bytes explicitly, without asking the server to sample or feed them back.

Preparation retains the text model's nine-result ABI. The terminal-token marker
may be null. Control exports `epoch`, `encode_epoch`, `publish_epoch`,
`encode_state`, `bind_request`, `request_defaults`, `request_options`, and
`capture_epoch`. Isolated `step`, tokenizer/chat roots, and drafting are not
required. Every submitted span has selection/proposal flags zero, and completed
progress contains no native output-token selection.

`encode_epoch` writes metadata and input packets, then returns
`(output_count, transfer_plan, byte_offset, byte_length)`. The plan view contains
little-endian u64 records
`{source_binding, source_offset, target_binding, target_offset, length}`.
The source and target tables each bind metadata at ordinal zero and input at
ordinal one. Source and destination offsets are independent: compact host data
can populate a sparse device layout without transferring its holes.
Only listed bytes are uploaded:
preparing capacity for a large packed epoch does not make a one-token epoch
transfer that entire capacity. Source initializes all bytes its numerical
stages read. Nonempty records are ordered by target binding and offset, with
no overlapping targets. Repeated source ranges are permitted. Records describe
independent placements, not an ordered sequence of copies.

The plan names logical storage, not pointers, registrations, or completion
handles. The execution owner binds physical resources and establishes readiness.
The plan can reuse retained control state and is consumed during submission;
packet bytes stay immutable until accepted transfers retire, including failure.
Source reuse, destination visibility, and destination reuse after consumption
are separate ownership boundaries. See the
[byte-placement contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/runtime/program/transfer_plan.h)
for the encoding and validation rules.
The same contract applies to generative text and media epochs. See
[epoch uploads](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/epoch_upload.h)
for local HAL binding and payload ownership.

`capture_epoch(control_state, shape_index, span_count, plan)` returns a buffer
of little-endian i64 triples: `[span_ordinal, residual_byte_offset, byte_length]`.
Ranges are grouped in submitted span order, with at least one nonempty range
per span. Multiple ranges concatenate in source order. The plan is checked and
copied before row mutation; output bytes are copied after the producer and
before the next epoch can overwrite the shared residual. See
[capture ownership](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/capture.h)
for device views, readiness, and explicit host materialization.

Captured output owns its pooled storage and execution domain independently of
the model, its weights, and retained rows. A downstream device consumer imports
readiness and retains its view through queue retirement. HTTP uses a compact
host snapshot and the [sequence profile](http-profiles.md#known-token-sequences).
Row IDs, request-relative input offsets, and packed activation offsets are
different coordinates: the source must derive capture ranges from its actual
packed layout, not treat a row or input offset as a physical output address.

## Stream speech with independent requests

A speech package can share weights and numerical execution among clients while
retaining each request's voice, cache history, RNG progression and playback
position. Start with the [speech walkthrough](workflows/stream-speech.md) for a
complete HTTP caller and a package whose output is independently checkable.

Cold preparation declares numerical stages and parameter domains, plus resident
request capacity, additive batch capacity, input/state strides, conditioning
storage, bounded output/feedback storage and waveform geometry. Those declarations
separate persistent request storage from temporary batch geometry. A larger
model can change its vocoder, diffusion schedule, cache layout or block size
without teaching the HTTP service its architecture.

The warm source exports these operations:

| Operation | Source-owned result |
| --- | --- |
| `prepare_request` | Validate text, voice, instructions, seed and speed; return retained CPU state and immutable request bytes. |
| `prepare_work` | Describe one request's next operation with compact work bytes, positive additive units and a compatibility tag, without advancing it. |
| `prepare_batch` | Pack a compatible cohort and retain its plan, using explicit resident-row and byte-range records. |
| `generate` | Execute the plan through the prepared runner stages, optionally requesting bounded feedback. |
| `complete` | Interpret retired feedback for one original batch ordinal; publish playback position, sample range, frame count and EOF. |

The [model declaration](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/speech/model.h)
owns the exact signatures, scalar widths and buffer layouts. `prepare_batch`
receives the prepared stage-tag catalog separately from the packed request work;
the source chooses its numerical stage and packing representation.

Consider a request A already decoding when B arrives with another voice and a
longer prefix. B first runs its required conditioning and prefix work. At a
retired boundary, the service queries ready requests again. If A and B offer
equal compatibility tags and their total work fits, they can share the next
batch while retaining distinct cache positions. If their current phases need
different numerical work, they advance in separate cohorts. Compatibility is
a fact established by the package's kernels; the service cannot infer it from
two opaque generation calls or from matching output formats.

Five dynamic roots give numerical programs compact batch work, stable request
rows, shared conditioning, stable stream rows and packed output. Changing batch
membership changes the small work description and row mapping; it does not move
entire request histories. Compiler-private workspace belongs to the current
command. Persistent convolution tails, recurrent state, KV caches and solver
history belong to the request's stream row or explicitly owned conditioning.

One accepted batch retains its original row mapping through execution and
collection, even if a client cancels. `complete` runs for every original ordinal,
including phases with no audio or readback. A zero-frame result is independent
of EOF. Native serving checks the published range, converts samples to the wire
format and copies each result into that request's bounded output storage before
releasing shared feedback. A stalled client then prevents only its own next
operation from becoming eligible.

Shared conditioning has a publication boundary too. A voice entry becomes
reusable only after its producer retires, and its storage remains valid while
accepted readers use it. Per-request RNG state must produce the same sequence
regardless of neighboring requests or their scheduling. The model owns these
numerical invariants; the service owns admission, output credit and retirement.

Qualify a port with staggered requests of different lengths and voices, changing
batch membership, a stalled reader, cancellation during an accepted batch and
reuse of the released row. Compare each surviving stream with isolated execution
and compare the model's numerical stages with an independent reference. Batch
occupancy alone establishes neither a throughput improvement nor a playback
deadline.

## Image-conditioned text

A text definition's optional [image component](model-documents.md#image-input)
implements the finite contract. It shares the deployment's device and numerical
provider catalog, while owning separate preparation and parameter residency.
Ordinary text and exact retained image-prefix hits do not prepare the projector.
Explicit component warmup follows the same preparation boundary. Its parameter
pin policy is independent of the primary language model's policy; deactivation
does not discard retained text state.

The text control program declares image limits and receives ordered text/image
content records. It renders model framing and returns each image's projected
row count and opaque preprocessing metadata. Its input preparation maps those
occurrences into logical input spans; native code checks the returned ranges
against the inspected images. The
[ordered-media contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/media.h)
defines these records and retained identity.

On a media-prefix miss, codecs realize packed RGB8/RGBA8 pixels without
resizing or normalizing them. The projector receives an
[image request packet](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/media/image_request.h)
containing encoded/display geometry, orientation, source metadata and pixels.
Its device program owns preprocessing and projection. Compatible images from
the same request or different requests share finite cohorts; source work units
and compatibility tags govern packing, not a native assumption of one image
per invocation.

Projected rows remain on the device. The language epoch imports producer
readiness and gathers only the rows its current spans need. Source
`encode_epoch` maps those rows and ordinary tokens into its input packet and
owns the position information consumed by its numerical stages. The
[media epoch contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/text/media_epoch.h)
defines the host control records separately from device features. Projector
weights and completed features have independent lifetimes; live requests retain
features only until their language consumers finish.

With a finite serving memory limit, the projector's cold preparation selects a
control entry with `prepare.memory_forecast(root)`. The selected function has
this signature:

```text
forecast_memory(state, ranges, descriptors)
    -> (private_peak_bytes, retained_bytes, expanded_bytes)
```

The three inputs are buffers. The range table uses little-endian i64
offset/length pairs for descriptors containing image headers and source metadata,
without decoded pixel planes. The three results are nonnegative i64 byte
estimates for the complete logical request: the producer's private peak
(parameters, activation staging, input and command workspace), all retained
output roots, and gathered output rows with 16-byte row alignment. Retained
output is excluded from the private peak and includes padded/shared roots;
gather storage includes separate copies of aliased rows.

These trusted source estimates let the server reject a request that cannot fit
alone before pixel decoding or weight activation. It checks producer and
language-consumer phases separately. The forecast does not reserve future
capacity against concurrent requests; each phase still passes ordinary admission.

## Recognize continuing audio

A recognition package accepts a continuing sequence of normalized F32 PCM frames
and explicit end-of-input. Input packet boundaries do not pad audio or finalize
an utterance. Each request retains its own sample clock, convolution history,
encoder state, decoder cache and transcript revisions as batch membership changes.
A source can also publish speech activity probabilities, either alone or together
with transcripts. Native callers explicitly select supported output types.
The `recognition` deployment role requires transcript support and exposes HTTP
transcription. The `activity` role requires activity support and serves native
streams without an HTTP inference endpoint. Both use the same preparation,
batching, input credit, and retirement contracts; compound sources can share
their weights and execution across different output selections.

The [recognition model declaration](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/recognition/model.h)
owns entry signatures and binary layouts. Cold preparation declares stages,
parameter domains, sample geometry, bounded input/event storage, recording limits,
resident rows and additive batch capacity. Warm source operations are:

| Operation | Source-owned result |
| --- | --- |
| `prepare_request` | Validate language, prompt and temperature; receive the selected output types and retain independent request state and immutable conditioning. |
| `prepare_work` | Inspect a bounded PCM snapshot with its absolute sample position and EOF flag; offer compatible numerical work or zero units while awaiting input. |
| `prepare_batch` | Pack ready work using explicit resident-row and byte-range records, retaining original cohort order. |
| `generate` | Submit one source-selected numerical phase without waiting for device completion. |
| `complete` | Acknowledge consumed input and publish a bounded ordered packet of typed events, independently of request completion. |

Acknowledgement returns input credit only after actual execution retirement.
Source code owns any PCM needed beyond `prepare_work`; the snapshot itself is
borrowed only during that call. Every accepted cohort member is collected,
including cancelled requests, before shared feedback or request rows can be
reused. A stalled event consumer excludes only its own request from new
work. Source-defined compatibility permits requests at different sample or token
positions to share a batch when their next operations support it.

Cold preparation declares supported output types, maximum packet bytes, and
maximum events per completion. A request selects a nonempty subset; an unrequested
event is a source error. The completion buffer and its produced byte extent are
separate, allowing the source to reuse output capacity across different counts.
Native collection copies the packet into reusable request-owned storage before
another source call can overwrite it. Events remain independently observable,
even when several revise the same transcript segment. Consuming the entire packet
returns that request's output credit. Final events can outlive model removal.

An activity event covers a positive interval of actual input and supplies a finite
speech probability in `[0,1]`. When activity is requested, each completion's
observations partition its acknowledged input prefix in order, without gaps or
overlap. Source programs retain future context as unconsumed input until its
observation is established; intermediate numerical phases can acknowledge zero
frames. The declared input window includes that lookahead. This lets callers
retain bounded pre-roll while the source chooses its own numerical work.
Coverage is not an exact onset, offset, or word boundary. Turn thresholds and
hangover belong to the caller's policy; source code owns numerical state and
EOF handling.

Each transcript segment has a stable request-local identity, increasing revision,
covered sample interval and replacement UTF-8 text. Finalizing a segment makes
its text and metadata immutable. Request completion requires every published
segment to be finalized. A source can additionally provide speaker labels and
word alignment; an input-coverage interval alone does not establish where words
were spoken. Consumers preserve these distinctions instead of converting a
revisable hypothesis into an append-only text delta.

The [finite HTTP adapter](http-profiles.md#audio-transcription) decodes enabled
WAV, MP3, and FLAC containers into the same bounded F32 input used by raw PCM.
Encoded input supports channel conversion and filtered resampling on shared CPU
workers. Raw PCM also supports filtered resampling without an audio codec
dependency, with matching channel counts and an exact rational sample clock;
model feature extraction, numerical stages, and transcript policy remain source
responsibilities. Each upload owns its codec history independently of source
batch membership, and accepted decode work retains that state until retirement.
Packet and worker boundaries preserve filter history. Explicit input EOF flushes
lookahead without extending the recording's duration.

The adapter selects transcript output, consumes revisions, and returns the
completed document. Typed consumers can apply revisions as they arrive and
independently supply more PCM.
Qualify a model with isolated reference transcripts, staggered arrivals,
different recording lengths, a stalled reader,
accepted-work cancellation and row reuse. Compound-output sources also preserve
event ordering and earlier revisions within a single packet. Compare text and
sample clocks across changing cohorts, including a final partial model frame.

## Make parameters and temporaries visible

Command stages declare fixed checkpoint domains and preparation policies.
Stages sharing a weight bank must agree on reflected placement. Preparation
indexes checkpoint metadata; activation streams and transforms weight payloads
into final storage. Warm requests reuse those plans. Deactivation can release
reloadable backing without discarding compiled code or retained session state.

The embedding package's
[weighted variant](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/fixture/embedding/variants/weights.loom)
and [policy](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/fixture/embedding/variants/policy.loom)
demonstrate adding a checkpoint-backed head without native model code.
Comparing the checkpoint's actual shapes/types with the numerical oracle is
part of the port; current key/extent checks do not establish model identity.

Private command workspace comes from compiler reflection and queue-ordered
allocation. It is not a permanently resident per-session tensor arena.
Under elastic backing it draws from the same physical pool and serving budget
as parameters and model state. Stable workspace addresses can survive physical
trim without retaining the pages. Prepared working-set peaks drive admission;
the physical slab size determines the minimum commitment, not each small
logical allocation's size.
Escaping output remains live through its final device or transfer consumer;
copied host feedback survives CPU encoding. A slow network reader receives
copied response storage rather than retaining device workspace.

Finite direct-dispatch stages instead take input/output buffers and own their
device-local working storage. The resident XDNA example uses this path. These
direct stages have no checkpoint-domain binding in the current source contract;
weighted direct programs need an explicit source-owned parameter-placement
contract, not a guess from native argument reflection. The ordinary weighted
command example and unweighted resident example prove different boundaries.

## Establish correctness before throughput

Start with the selected package's deterministic output and replace one numerical
stage with the intended model operation. Checks can compare functions directly
with independent VM oracles; kernel checks specify an appropriate oracle rather
than copying candidate output into a golden. [Correctness workflows](../workflows/test-correctness.md)
cover functions, cases and scenarios.

A complete port then establishes final outputs through the same server binary
using an external package directory. The useful cases include ragged inputs,
different request sizes, cancellation during accepted device work, a failed
request followed by successful reuse, and unregister/re-register. Retained text
also compares append/fork/rewind against full replay. A compilation-only success
does not establish these ownership or numerical contracts.

[Agent-driven kernel development](../workflows/agent-driven-kernel-development.md)
and [compile reports](../workflows/compile-reports.md) provide the optimization
loop once correctness is measurable. Real model shapes, memory traffic and
independent error bounds guide layout, traversal, fusion and specialization.
Microkernel speed is one input; [server observations](workflows/observe-model-server.md)
and end-to-end useful throughput determine whether the change helps the workload.
