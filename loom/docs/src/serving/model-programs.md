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
| `parameter(stage, binding, directory, path, policy)` | Append a checkpoint file to a reflected fixed binding's parameter domain. |

A split checkpoint uses one `parameter` call per file with the same binding
and preparation policy. The files populate one parameter index in declaration
order, so their tensors share one residency plan. Stages naming the same ordered
file list and policy share that domain. File selection belongs to the model's
cold source; the loader accepts explicit GGUF, safetensors, and other supported
parameter files without inferring filenames.

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
at registration. Checkpoint domains remain the cold program's responsibility;
request stages must preserve their layout and cannot declare new parameters.

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
adapter strength. It validates the geometry, prepares opaque request bytes, and
selects a cold stage or declares request stages through the same invocation-only
`prepare` capability as finite cohorts. Exact declaration sets reuse code within
the selected source generation; misses compile before parameter admission.
Request stages share the cold parameter layout and cannot declare new parameters.

Image control then drives command parts to final RGB. It returns a continuation
cursor; another invocation occurs only after the previous accepted frontiers
retire. Compiler-reflected workspace remains live across barrier-aligned parts.
RGB and compiler-aligned workspace share one request-sized pool allocation,
retained through final feedback. Neither storage nor pixel geometry belongs to
the executable cache. Final output is F32 CHW RGB in `[-1,1]` before native image
encoding uses that request's dimensions.
The [image source contract](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/image/model.h)
defines entry signatures, alignment and cancellation.

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

## Recognize continuing audio

A recognition package accepts a continuing sequence of normalized F32 PCM frames
and explicit end-of-input. Input packet boundaries do not pad audio or finalize
an utterance. Each request retains its own sample clock, convolution history,
encoder state, decoder cache and transcript revisions as batch membership changes.

The [recognition model declaration](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/serve/service/recognition/model.h)
owns entry signatures and binary layouts. Cold preparation declares stages,
parameter domains, sample geometry, bounded input/event storage, recording limits,
resident rows and additive batch capacity. Warm source operations are:

| Operation | Source-owned result |
| --- | --- |
| `prepare_request` | Validate language, prompt and temperature; retain independent request state and immutable conditioning. |
| `prepare_work` | Inspect a bounded PCM snapshot with its absolute sample position and EOF flag; offer compatible numerical work or zero units while awaiting input. |
| `prepare_batch` | Pack ready work using explicit resident-row and byte-range records, retaining original cohort order. |
| `generate` | Submit one source-selected numerical phase without waiting for device completion. |
| `complete` | Acknowledge consumed input and optionally publish a transcript revision, independently of request completion. |

Acknowledgement returns input credit only after actual execution retirement.
Source code owns any PCM needed beyond `prepare_work`; the snapshot itself is
borrowed only during that call. Every accepted cohort member is collected,
including cancelled requests, before shared feedback or request rows can be
reused. A stalled transcript consumer excludes only its own request from new
work. Source-defined compatibility permits requests at different sample or token
positions to share a batch when their next operations support it.

Each transcript segment has a stable request-local identity, increasing revision,
covered sample interval and replacement UTF-8 text. Finalizing a segment makes
its text and metadata immutable. Request completion requires every published
segment to be finalized. A source can additionally provide speaker labels and
word alignment; an input-coverage interval alone does not establish where words
were spoken. Consumers preserve these distinctions instead of converting a
revisable hypothesis into an append-only text delta.

The [finite HTTP adapter](http-profiles.md#audio-transcription) decodes enabled
WAV, MP3, and FLAC containers into the same bounded F32 input used by raw PCM.
It performs channel conversion and filtered resampling on shared CPU workers;
model feature extraction, numerical stages, and transcript policy remain source
responsibilities. Each upload owns its codec history independently of source
batch membership, and accepted decode work retains that state until retirement.

The adapter consumes revisions and returns the completed document. Typed
consumers can apply revisions as they arrive and independently supply more PCM.
Qualify a model with isolated reference transcripts, staggered arrivals,
different recording lengths, a stalled reader,
accepted-work cancellation and row reuse. Compare text and sample clocks across
changing cohorts, including a final partial model frame.

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
