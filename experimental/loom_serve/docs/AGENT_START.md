# Start here: a working handoff, not a framework specification

The first outcome is an observed execution, not a new serving abstraction. This
branch has source-defined text, image, embedding and decision runners, a shared
HTTP catalog, Krea/Qwen examples, and reusable JIT/queue ownership code. The
[three-model walkthrough](../models/fixtures/README.md) establishes real
VM/GPU/HTTP output without a model download. The
[CPU first-output recipe](README.md#cpu-first-output-without-a-model-download)
instead runs source-JIT text/embedding/decision paths without a GPU or ROCr.
A port can then replace the matching source package, establish independent
numerical agreement, and return to the same serving boundary. No private working
history is required.

For deployment and administration rather than a new model port, the
[operator guide](OPERATING.md) starts with an empty shared server and exercises
dynamic registration, resource configuration, observation and removal.

## What changes without rebuilding the server

The server supplies shared execution and modality services. A model package
supplies the programs those services run, rather than a native model adapter.
Its [model document](MODEL_FORMAT.md) names the cold preparation program, warm
control program, native source providers and assets. Programs may be Loom text
or portable bytecode. Source symbol references establish dependencies between
providers; their order in the document is not execution order.

| Intended change | Owning input or implementation | How it takes effect |
| --- | --- | --- |
| Public model name, device placement, capacity or admission mode | Deployment JSON and [administration endpoints](OPERATING.md) | Register the deployment or change its supported lifecycle state; model source does not need an identity rewrite |
| Model geometry, tensor encoding, weight preparation or numerical kernels | Source preparation, weight policy and reachable kernels/commands | Register the updated package; the existing binary JITs its selected programs |
| Prompt framing, generated tool grammar or canonical history | Source [chat-policy entries](../text/README.md#chat-policy-entries) | Register the updated package within the native request/output contract |
| Stage composition, speculative proposal/verification or prepared shape selection | Source preparation and warm control | Source selects and submits prepared stages; native code still owns admission, ready-span packing and retirement |
| New HTTP fields, output media or scheduling policy | The corresponding native modality service, parser or scheduler | Update and rebuild the native owner, with a real request-to-output witness |

An existing registration retains its prepared programs. Replacing source files
does not mutate in-flight work; unregistering and registering again prepares
the new package. Registration can happen while other models continue serving.
The [operator walkthrough](OPERATING.md#first-output-starting-with-no-models)
exercises that boundary with deterministic source programs before a learned
model adds checkpoint and numerical requirements.

This is source-defined model execution, not an arbitrary workflow engine or a
promise that every serving policy already lives in Loom. For example, the
native text parser currently requires greedy sampling, while Qwen's source
policy separately rejects thinking mode. JSON and streamed SSE responses are
both supported. A different chat policy can interpret `model_options` without
native changes, but cannot expand the native sampling or output protocol by
itself. The [text contract](../text/README.md#chat-policy-entries) specifies
that boundary; backend support also depends on the actual programs in a package.

## Choose the existing execution contract

A port starts from its state and result lifetime, not its model name. The
compatible source package is the extension point; the native server supplies
transport, shared workers, device timelines, memory ownership and observation.

| Model behavior | Source contract and first-output witness | Scheduling/storage boundary |
| --- | --- | --- |
| Autoregressive text | [Text entries](../text/README.md), [tiny retained model](../models/fixtures/text/README.md) | Ready prompt/decode/verification spans share epochs; pages and recurrent state survive requests |
| Text to finite image | [Image entries](../image/model.h), [image fixture](../models/fixtures/image/README.md), then Krea | One prepared invocation per image residency; request storage retires after output |
| Embedding or structured decision | [Finite entries](../finite/README.md), [embedding](../models/fixtures/embedding/README.md) or [decision](../models/fixtures/decision/README.md) fixture | Source declares additive batch cost, compatibility and original result indices; compatible requests share a cohort |
| Resident native computation | [Finite direct-stage contract](../finite/README.md#compact-results-and-physical-storage), [resident variants](../models/fixtures/embedding/README.md#verification) | `prepare.dispatch` supplies a native entry submitted through `queue_dispatch`, with the same timeline/feedback retirement as command stages |

The resident XDNA fixture reaches real HTTP output without a command buffer or
host output initialization. It has no checkpoint parameters. Weighted command
stages have a separate working loader; a weighted resident port still needs a
source-owned placement contract connecting checkpoint names, lengths and native
bindings. Native argument reflection alone does not contain those facts.

One server owns an explicitly configured HAL device group. Each registration
currently selects one member, so different models can use different devices;
a model does not yet span multiple members. CPU, AMDGPU, Vulkan and XDNA fixture
evidence covers their authored paths, not every kernel on every backend. A
learned model brings its numerical and target coverage with it. Audio streams,
image-input conditioning and cross-device graphs do not fit these existing
protocols merely by changing a model-document field; their first real caller
must establish the missing input/output ownership flow.

## One bounded port assignment

An initial assignment can be this concrete:

> Establish the source-only JIT tests using this packet's README and record the
> actual target, build configuration, and passing case counts. Read the minimal
> JIT caller and the text source contract before the Qwen numerical package.
> Then implement the assigned model through one real-weight block and a retained
> greedy continuation,
> with an independent numerical reference. Keep model math and policy in its
> source package, reuse the text residency and queue/JIT ownership boundary,
> and expose unsupported compiler
> behavior with the smallest real reproducer. HTTP and performance expansion
> follow a correct retained continuation, not the other way around.

That is a new model implementation assignment. Changing Qwen's `--model` path
alone does not supply another architecture's math, storage or tokenizer policy.

The assignment records the model's checkpoint, configuration and tokenizer
artifacts before implementation. The [text contract](../text/README.md) is the
implementation map: source bootstrap, warm numerical entries, semantic spans
and chat policy. There is no C model adapter to copy and no native model
registry to extend. The [CPU request
inspector](../text/README.md#checks-that-matter-for-a-port) exposes the real
source-rendered prompt, token IDs and completed transcript before checkpoint
loading. This makes tokenizer/template differences visible without conflating
them with a numerical kernel failure.

For embedding or decision work, the [finite-vector contract](../finite/README.md)
is the existing source-owned batching path, not a text decode loop. Its
[embedding fixture](../models/fixtures/embedding/README.md) and
[decision fixture](../models/fixtures/decision/README.md) demonstrate request
preparation, compatible ragged cohorts, compact feedback and original result
indices. Pooling, normalization and learned heads belong in kernels; native
code parses and serializes the modality protocol. Real EmbeddingGemma2/CLEF
ports still need their own numerical/backend evidence; the resident fixture
above establishes NPU serving ownership only.

For image or audio work, the
[tensor-in/tensor-out ownership flow](MODEL_AUTHORING.md#image-and-audio-entry-points)
starts from the same small JIT caller without importing Qwen's chat, cache, or
stage-layout assumptions. The [Krea native CLI](../models/krea/README.md#generate-an-image-natively)
is a real prompt/seed-to-image caller with LoRA; its source commands and cold
request leaf provide a concrete second model to study. The
[retained image server](../models/krea/README.md#serve-images-over-http) adds
bounded HTTP admission and native PNG responses without Qwen scheduling
assumptions. Image batching, multiple-adapter composition and audio adapters
remain distinct implementation boundaries, not capabilities implied by the
shared transport.

## Establish a baseline before editing

The [README](README.md#fresh-machine-prerequisites) supplies bootstrap,
configuration, public model downloads, exact commands, and success output.
CPU source-service tests establish actual VM/x86 compilation, queue submission,
HTTP output and retained lifetimes. Small GPU tests exercise those boundaries
through their device backend. Full Qwen adds real checkpoint interpretation,
retained state, MTP, and multi-turn HTTP.
Each result establishes that boundary only. A machine without the required
device can complete the CPU gate but cannot certify the GPU gate.

The recorded baseline consists of the checkout commit and dirty diff, compiler
and build flags, execution device/ROCr identity, commands and exit statuses,
test cases actually executed, and model revisions/hashes when applicable.
Skipped hardware tests remain unqualified. A cached test result represents the
same declared inputs; a fresh-device qualification records a real execution.

The authoring loop is source-driven. Editing model `.loom` files changes the
next registration of that package without rebuilding the C executable; an
already-prepared registration keeps its cached code until removal. Editing
shared native mechanisms requires building their exact target again.
The [model document](MODEL_FORMAT.md) lists source providers and explicitly names
cold/warm VM roles; its source-declared weight policy is loaded separately.
Every role and provider can use `.loom` or `.loombc`. An added command/kernel library must be present in
that catalog; a generated HSACO or compiler subprocess is not part of this
deployment path.

## Read in this order

| Evidence | What to recover from it |
| --- | --- |
| [`testdata/jit/config.loom`](../runtime/testdata/jit/config.loom), [`stage.loom`](../runtime/testdata/jit/stage.loom), [`control.loom`](../runtime/testdata/jit/control.loom), [`jit_test.cc`](../runtime/jit_test.cc) | Source defaults and explicit overrides through one complete VM/GPU call, including compiler storage dying before command execution |
| [`jit.h`](../runtime/jit.h), [`jit.c`](../runtime/jit.c) | Public `loomc` embedding, source indexing, live device facts, native request ownership, reusable command recording |
| [`command.h`](../runtime/command.h), [`execution.h`](../runtime/execution.h), [`module.h`](../runtime/module.h), [`control_test.cc`](../runtime/control_test.cc) | Buffer borrowing/retention, exact queues and timelines, accepted work, feedback lifetime, drain after failure |
| [`program.h`](../runtime/program.h), model [`control.loom`](../models/qwen/control.loom) | One shared source-JIT process with isolated and packed entries; indexed command selection, proposal/verify/catch-up routing, and bounded feedback without intermediate host waits |
| [`preparation.h`](../runtime/preparation.h), model [`prepare.loom`](../models/qwen/prepare.loom) | Source-owned startup declarations, run-dependent specialization, checkpoint bindings and opaque control state surviving cold process teardown |
| [`model.h`](../text/model.h), [`model.c`](../text/model.c), [`epoch.loom`](../models/qwen/epoch.loom) | A concrete residency: weight placement, row origins, mutable state, scratch, descriptors, packed traversal and progress |
| [`weights.h`](../runtime/weights.h), [`weights.loom`](../models/qwen/weights.loom), [`weights_test.cc`](../runtime/weights_test.cc) | Model-owned VM policy selecting cached preparers, actual file bytes into shared final storage, and transformation exactly once per unique tensor |
| [`schedule.h`](../text/schedule.h), [`service.c`](../text/service.c) | Ready spans versus model rows, canonical history, output credit, admission and scheduling policy |

The detailed [authoring](MODEL_AUTHORING.md) and [runner](RUNNER.md) guides
explain the invariants behind those callers. Canonical Loom documentation wins
over guessed MLIR syntax or remembered compiler behavior. The
[authoring corpus](../../../loom/src/loom/test/corpus/authoring/README.md) gives
small checked examples for individual numerical mechanisms.

## The engineering loop this template is meant to transfer

One live experiment record carries the numerical/storage contract, observed
baseline, one proposed mechanism, expected evidence, and the next decision.
Measurements and code are evidence; an earlier explanation is a hypothesis
until its callers or results agree. A contradiction changes the record rather
than gaining a workaround beside it.

A new model first crosses the hard boundary: public checkpoint bytes through
the actual loader, JIT, command, and retained GPU state into independently
checked outputs. Matching isolated arithmetic is not enough if a second token
or interleaved row corrupts state. A convincing sentence is not an oracle.
The port expands after one real block and its cache updates agree, then after
the whole model's retained continuation agrees.

Performance starts from a controlled workload and a falsifiable bottleneck.
Compiler reports explain register pressure, spills, waits and memory access;
completed device work establishes speed. Changing one mechanism at a time
keeps regressions interpretable. A narrow kernel win returns to the retained
multi-agent workload before it becomes a serving claim. The
[performance guide](PERFORMANCE.md) supplies controls and measurement details.

The code is an experimentation platform. A model outside the text, image or
finite-vector contracts first proves a concrete ownership flow through the shared runtime.
Additional modality machinery earns its place through that caller, not by
anticipating every model family. One compiler/JIT residency, shared weights and code, data rows
instead of per-session VMs, explicit dependency edges, and no steady-state
device allocation are the intended invariants. HAL and the command-program ABI
are not extension slots for model bookkeeping. The shared HTTP scheduler returns
workers while awaiting model timelines; it consumes completed cohorts before
planning more work. That host scheduling boundary and the fixed row/shape bounds
are explicit in the runner guide, not hidden behind a promise of a device-owned
loop.

Small coherent commits keep a working boundary available while the next one is
investigated. A continuation note records what actually passed, which mechanism
failed, exact reproduction inputs, and the next concrete action. Completed
work stays reproducible from tracked sources; large generated artifacts and
downloaded weights remain separate from source history.

## Failure triage by boundary

| Observation | First discriminating evidence |
| --- | --- |
| Setup/configuration fails | Python 3.12, managed tool diagnostics, [BUILDING.md](../../../BUILDING.md); build-service failures belong to that service's repair path |
| Target is incompatible or tests skip | Enabled AMDGPU/VM compiler, HAL substrate and runtime driver, plus the actual test runner's GPU capability |
| `hsa_init`, loader, or device creation fails | ROCr dependency loading, device enumeration/permissions and available resources on the **execution** machine; model source has not run yet |
| JIT cannot resolve a symbol or configuration | Catalog entry, named root, source `config.def`, `config.decl` constraints and supplied overrides; stderr identifies the failing stage |
| Target contract rejects the GPU | Authored architecture/features and subgroup width versus the live device profile; Qwen's Q6 wave64 kernels declare `gfx11-generic` |
| Tensor missing, wrong size, or bad output at the first layer | Exact checkpoint hash, tensor name/shape/orientation/encoding, tokenizer IDs and reference intermediate values |
| Single row works; retained or packed work fails | Absolute row origins/byte offsets, cache publication, inactive-row masking, input/output lifetime and explicit semaphore edges |
| A GPU fault or unstable output | Smallest retained reproducer, target/profile, shape and source; `--kernel_sanitizer=access` instruments device accesses separately from host `--config=asan` |
| HTTP accepts requests but the witness fails | Client exception and server `admit`/`epoch`/`complete` records; `ready` proves initialization, not correct retained history |

An unsupported operation or compiler failure is useful transfer evidence when
it comes with exact source, command/configuration, device and first divergence.
Relaxing the numerical check, substituting stale code, or moving orchestration
outside the JIT would lose the property being tested. If the next step needs a
different device, checkpoint, or architecture decision, that concrete missing
input is the handoff back to the human.
