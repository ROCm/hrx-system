# Ideogram 4 Loom Prototype

This directory is the experimental home for a standalone Ideogram 4 inference
prototype built from Loom kernels and the IREE runtime HAL. The immediate goal
is correctness: reproduce a known-working `stable-diffusion.cpp` Ideogram 4
pipeline end to end, then port the kernel families and scheduler deliberately
until every tensor-producing stage has executable Loom coverage, golden
fixtures, and benchmark hooks.

The value of the project is not a one-off demo. The value is a readable C/HAL
model program that teaches us how Loom should eventually represent whole-model
scheduling. This code should be clear enough to serve as the hand-authored
reference for a future Loom scheduling IR.

## Source References

Primary public references:

- Ideogram 4 model card:
  https://huggingface.co/ideogram-ai/ideogram-4-fp8
- Ideogram 4 reference code and docs:
  https://github.com/ideogram-oss/ideogram4
- `stable-diffusion.cpp`:
  https://github.com/leejet/stable-diffusion.cpp
- Experimental Ideogram 4 LoRAs:
  https://huggingface.co/DeverStyle/Ideogram-4.0-Loras

The original model weights are gated and governed by the Ideogram 4
non-commercial license. Model weights, LoRA files, generated images, and large
tensor dumps are external artifacts that live outside the repository. Verify
redistribution terms from the model cards and upstream sources before adding
any reduced fixtures.

## Model Shape

Ideogram 4 is a 9.3B parameter flow-matching text-to-image model. Its core
image generator is a single-stream Diffusion Transformer where text tokens and
image latent tokens are concatenated and processed by the same transformer
blocks.

Important model facts for the first port:

- Qwen3-VL-8B-Instruct is used as a frozen text encoder in text-only mode.
- Hidden states from 13 Qwen3-VL layers are concatenated for conditioning:
  layers 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, and 35.
- The Ideogram 4 transformer has 34 layers.
- Transformer hidden size is 4608.
- Attention has 18 heads.
- MLP intermediate size is 12288 and uses SwiGLU.
- Timestep conditioning flows through AdaLN modulation with an AdaLN dimension
  of 512.
- Attention uses QK-RMSNorm and multimodal RoPE.
- Public diffusion image latents are 128-channel tensors. For the Flux2-format
  VAE used by Ideogram 4, those channels represent 2x2 patches over the
  decoder's 32-channel internal latent representation.
- The denoising loop uses Euler flow matching with asymmetric classifier-free
  guidance.
- Prompting is strongest with structured JSON captions. Plain text prompt
  expansion is a reference convenience, not a scheduling requirement for this
  prototype.

## Precision And Weight Strategy

The Python Ideogram 4 implementation is the correctness oracle for numeric
behavior. `stable-diffusion.cpp` remains useful for model discovery,
dispatch-shape comparison, and shader inspection, but any difference from the
Python implementation should be treated as an implementation choice or accident
until proven otherwise.

Dense BF16 is the first correctness lane. The Python model loads the DiT with a
BF16 compute dtype, casts the latent input, timestep embedding path, and LLM
conditioning into that dtype before the transformer body, and casts only the
final public velocity output back to F32. The first Loom kernels and C/HAL
sub-pipelines should therefore match BF16 activation behavior rather than an
F32-activation scalar implementation detail.

Ideogram's FP8 checkpoint is weight-only e4m3 storage. Its Python `Fp8Linear`
widens each FP8 weight row to the current activation dtype, applies the stored
per-output-channel scale in that dtype, and then runs the same linear math.
FP8 support in this prototype must preserve that semantic contract: compact
FP8 weights and row scales feed BF16-activation matmuls, not a separate FP8
activation pipeline.

The logical kernel contracts should be written around operation semantics, not
one storage format. For a linear layer, the durable contract is the input,
weight, bias or modulation inputs, output, accumulation behavior, and accepted
numeric tolerance. The first implementation may bind BF16 weights directly, but
the same operation family should have room for target-specialized weight
loaders and inner loops.

The active precision sequence is:

- BF16 weights and BF16-activation kernels first, using the Python BF16
  execution path as the tensor-golden reference.
- FP8-weight structural parity next, matching the Python path: official FP8
  e4m3 source weights plus F32 row scales are prepared into BF16 execution
  layouts and consumed by BF16-activation WMMA kernels.
- Direct FP8-weight kernels on RDNA3/gfx1100 after parity is established, where
  compact FP8 values and scale metadata may be decoded in the kernel when that
  wins over prepared BF16 layouts.
- Native FP8-weight kernels on CDNA3/gfx942, using the same logical operation
  contracts and goldens while specializing the implementation to the target's
  FP8 execution path.
- Later FP8 targets, including gfx12-class machines, should fit the same model:
  target-specific internals behind stable kernel I/O and comparison fixtures.

FP8 storage on gfx1100 is an important bandwidth and capacity experiment, but
the first usable product shape should stay structurally close to the Python
implementation. Prepared BF16 execution layouts give us a correctness and
schedule baseline before asking whether direct in-kernel FP8 decode or native
FP8 target paths are better. Compiler reports and benchmark artifacts should
make the chosen inner loop visible, including whether the generated code uses
WMMA/MFMA instructions for the relevant target.

## Product Goal

The target product surface is a C library plus thin applications around it:

- a load/prepare/execute API for Ideogram 4 model sessions;
- a CLI that can run a full JSON prompt to an image;
- a small server or embedding API once the library boundary is stable;
- streaming sinks for progress, intermediate previews, diagnostics, and final
  output;
- instrumentation modes that can dump tensors, Loom compiler artifacts, HAL
  execution metadata, and reference-comparison records at named stage
  boundaries.

The shared stage lifecycle should separate static model loading, deterministic
planning, prepared executable bundles, and asynchronous issue. That boundary is
the contract for the C API, diagnostics, memory planning, and stage benchmarks.
Stages are coarse user-visible scheduling units, not internal kernel or layer
fragments: each stage boundary implies queue submissions, memory scopes,
readiness edges, and benchmark/reporting semantics that should be notable when
compared against external references. Qwen3-VL prefill/forward, decode,
sampler loops, VAE decode, and full model slices are plausible stages;
condition epilogues, RMSNorms, linears, attention projections, and fused layer
fragments are kernel families authored inside those stages.

Kernel configuration should describe model and tensor facts such as token
counts, hidden sizes, dtype lanes, layout choices, and optional feature paths.
Launch geometry belongs in Loom launch-config regions. The C runtime evaluates
those regions through the `loomc` API during preparation and records the
resulting HAL dispatch configuration with the prepared executable. Stage
authoring code should not mirror workgroup-count or workgroup-size arithmetic;
it supplies operation configs and bindings, while the selected kernel owns its
launch shape.

The CPU should eventually own only prompt ingestion, tokenization, high-level
request control, and final result delivery. The diffusion loop, CFG,
schedule-dependent sampling math, random initialization, latent updates, VAE
decode preparation, and tensor diagnostics should run on the device whenever
that is technically possible. Early versions may keep the outer loop on the
host while issuing asynchronous HAL work through timeline semaphores.

## Program Shape

The C code is an ordinary model program with direct HAL scheduling and Loom
kernel dispatches. Repeated layers are expressed as loops over model,
parameter, and memory-plan tables. Reference traces are oracles for behavior
and configuration discovery, while the authored C program is the schedule
representation.

Correctness and coverage gate the work. Performance matters because the design
choices here affect future fusion and scheduling, and that optimization work is
grounded in tensor goldens, kernel checks, and benchmarkable stage boundaries.

## Repository Layout

The implementation should follow the repository's normal package style:
headers, C sources, tests, benchmarks, and package build files live together by
component. Do not split ID4 into central `src/`, `tests/`, and `benchmarks/`
directories. Each package owns its `BUILD.bazel`; CMake files are generated
from those Bazel targets.

The exact directories and file names should follow the invariants that emerge,
but the ownership split should remain stable:

- `pipeline/` owns generic stage lifecycle, plans, bundles, diagnostics,
  memory plans, parameter slabs, kernel cache, and shared stage benchmarks.
- `tooling/` owns reusable helpers for command-line tools, benchmarks, smoke
  tests, parameter index construction, and device-group setup. Runtime stage
  code should not depend on CLI-only helpers.
- `kernels/` owns hand-authored Loom kernel families and their source-local
  `check.case` and `check.benchmark` coverage. The internal folder structure is
  deliberately open until the kernel family boundaries are known.
- `stages/` owns concrete pipeline stage implementations such as Qwen3-VL
  prefill/forward, sampler loops, VAE decode, and full model slices. Stage
  packages depend on `pipeline/`, `tooling/` only where appropriate, and
  `kernels/`.
- `ideogram4/` owns the assembled model configuration and execution API that
  wires stages into the full Ideogram 4 pipeline.
- `binding/cli/` owns the thin one-shot command-line program using
  `iree/base/flags`.
- `binding/server/` is reserved for a future API server.
- `binding/c/` is reserved for a future stable C API boundary and is out of
  scope for the prototype infrastructure.
- `build_tools/` owns ID4-specific presubmit and developer workflow hooks.

## Build Shape

Bazel is the source of truth. Packages should add targets next to the code they
build, matching the runtime and Loom style:

- `cc_library` targets for each component, such as `//experimental/id4/pipeline:stage`;
- `cc_test` targets next to the component under test, such as
  `//experimental/id4/pipeline:stage_test`;
- benchmark targets next to the measured component, such as
  `//experimental/id4/pipeline:stage_benchmark`;
- `cc_binary` targets for thin bindings such as
  `//experimental/id4/binding/cli:id4`;
- Loom test and benchmark targets in `kernels/` that run `.loom` sources
  through `iree-test-loom` and `iree-benchmark-loom`;
- filegroups for `.loom` libraries and reduced fixtures where useful.

When a `BUILD.bazel` file changes, regenerate generated CMake metadata with the
repository Bazel-to-CMake tool. Generated `CMakeLists.txt` files should mirror
the Bazel package layout as the single checked-in build structure.

Use existing `loom_*` Bazel rules when they match the desired test shape. When
the existing rules are tied to older authoring workflows or cannot express a
folder-level `.loom` test suite cleanly, add focused rules that preserve the
testing strategy. Build infrastructure is part of the project when it removes
friction from repeatable correctness work.

The checked build must not depend on downloaded full model weights. Small,
purpose-built fixtures are allowed when they are license-compatible and useful
for automated correctness. Large model files, generated images, and full tensor
captures remain outside the repository.

## Reference Artifacts

Reference traces and reduced tensor fixtures are generated artifacts. The
checked-in repository surface should contain the tools, reduction plans,
schemas, small synthetic unit tests, and hand-authored `.loom` cases needed to
create and consume those artifacts, not the heavyweight payloads themselves.

The normal checked build should stay independent of model downloads, full
activation dumps, generated images, and multi-megabyte golden tensors. Optional
heavyweight validation may use explicit external artifact roots, for example:

- `ID4_REFERENCE_TRACE_ROOT` for raw reference traces;
- `ID4_REFERENCE_FIXTURE_ROOT` for reduced fixture directories generated from
  those traces.

Targets that consume external fixture roots should be opt-in and clearly
separate from presubmit. A missing external root is a configuration error for
that explicitly requested target, not a reason to search the source tree or
download artifacts implicitly.

Checked-in reduction plans live under `reference/plans/`. The fixture generator
is the canonical path from a raw reference trace to compact fixtures:

```bash
bazel run //experimental/id4/build_tools:generate_fixture -- \
  --fixture-id=<fixture-id> \
  --trace-root="${ID4_REFERENCE_TRACE_ROOT}" \
  --fixture-root="${ID4_REFERENCE_FIXTURE_ROOT}"
```

Generated fixture directories contain payload files plus `manifest.json` and
`inventory.json`. The manifest is the detailed provenance and checksum record.
The inventory is the compact stage/role/shape view intended for planning stage
tests, benchmark harnesses, and kernel-family triage. Legacy reduced fixtures
may use NPY payloads for simple dense f32 tensors, but generated non-f32
fixtures and captures use exact ID4 tensor payloads so BF16, FP8, scaled,
packed, and blocked storage can be represented without widening or descriptor
tricks.

Actual stage captures use the same manifest record shape with `role` values
such as `actual` or `output`. The fixture comparator matches `expected`
fixture records against actual capture records by semantic `stage` and `name`,
then applies the per-record tolerance from the fixture manifest:

```bash
bazel run //experimental/id4/tooling:fixture_compare -- \
  --fixture-dir="${ID4_REFERENCE_FIXTURE_ROOT}/<fixture-id>" \
  --actual-dir="<actual-capture-dir>" \
  --output="<comparison-report>.json"
```

Comparison reports are machine-readable JSON with manifest hashes, pass/fail
counts, shape and dtype mismatches, tolerance failures, maximum absolute and
relative errors, and first mismatch coordinates. They are meant for opt-in
regression runs and human diagnostics over external artifacts, not ordinary
presubmit.

Raw traces can also be inventoried without reducing payloads:

```bash
bazel run //experimental/id4/build_tools:trace_inventory -- \
  --trace-dir="${ID4_REFERENCE_TRACE_ROOT}/<trace-id>" \
  --output="<inventory-output>.json" \
  --hash-payloads
```

The raw-trace inventory records stage order, tensor shapes, dtypes, payload
sizes, operation-family groupings, optional payload hashes, and coverage facts.
It is the first pass for deciding which kernel families have enough oracle
material to become `.loom` cases and which seams still need reference
instrumentation.

`.loom` tests should keep their checked-in data tiny. Inlined values should be
synthetic or small representative extractions that exercise the operation
class. Larger oracle comparisons belong in explicit external fixture tests once
the relevant stage has a stable load/compare harness.

## Runtime Architecture

The runtime has four major pieces:

- model loading and immutable model metadata;
- Loom source/link/compile/emit management through `loomc`;
- HAL resource ownership, command-buffer recording, and timeline scheduling;
- request execution with progress and diagnostics sinks.

### Model Loading

The loader treats BF16 and FP8 model artifacts as parameter sources. It
extracts tensor metadata, validates required tensors, uploads weight data into
device-visible slabs, and produces typed model tables used by the scheduler.

The first quality reference configuration uses:

- main Ideogram 4 diffusion weights expanded to dense BF16;
- unconditional Ideogram 4 diffusion weights expanded to dense BF16;
- Qwen3-VL-8B-Instruct text encoder weights in dense BF16 or F16 form;
- FLUX-family KL VAE weights compatible with the reference pipeline;
- optional LoRA adapters.

Weights should land in one or more long-lived weight slabs. Command buffers
bind those slabs and use per-dispatch binding references with byte offsets for
individual tensors.

Weight loaders should preserve enough metadata for later compressed-weight
paths. For FP8 sources this includes the element format, per-tensor or
per-channel scale tensors, scale application order, and the exact BF16
expansion rule used by the reference. BF16 execution consumes the expanded
view; FP8 execution consumes the compact view plus scale metadata while
producing outputs compared against the same BF16 goldens.

### Loom Embedding

`id4` should use the public `loomc` API directly. The Loom command-line tools
remain important for authoring tests, corpus checks, and one-off diagnostics,
but the runtime integration must dogfood the C API:

- create immutable `loomc_source_t` handles for checked `.loom` libraries and
  any small generated specialization sources;
- build or reuse link indexes for kernel libraries;
- create reusable `loomc_compiler_t`, `loomc_pass_program_t`,
  `loomc_target_profile_t`, and `loomc_target_selection_t` handles;
- keep `loomc_workspace_t` scratch per worker or compile lane;
- request transformed module artifacts, bytecode, compile reports, executable
  artifacts, and artifact manifests according to runtime diagnostic flags;
- load emitted executable artifacts into the IREE HAL executable cache used by
  the active device.

The AMDGPU target is the product target. SPIR-V/Vulkan is a reference and
debugging aid because it gives a second backend, RADV tooling, and a useful ISA
comparison path. If the current `loomc` API cannot route an IREE HAL AMDGPU
device to an AMDGPU target profile or cannot hand an AMDGPU HSACO artifact to
the HAL executable cache, that integration gap is part of this project.

### HAL Scheduling

The scheduler should be written against the IREE HAL queue APIs:

- `iree_hal_device_queue_alloca` for transient slab allocation;
- `iree_hal_device_queue_dealloca` for asynchronous slab retirement;
- `iree_hal_device_queue_fill`, update, copy, read, and write for explicit
  transfer work;
- `iree_hal_device_queue_execute` for pre-recorded reusable command buffers;
- timeline semaphores for all ordering, readiness, and pipeline edges.

HAL queues are not FIFO. Every user-visible ordering edge must be expressed
through semaphores or explicit command-buffer barriers. Submission order alone
is not a correctness contract.

Command buffers should be reusable whenever their operation sequence is stable.
Recordings should use indirect buffer references into the queue-execute binding
table so the same command buffer can run against different slabs, offsets,
request instances, LoRA selections, and transient allocations.

### Memory Planning

A command buffer should normally see:

- one or more immutable weight slabs;
- one local transient slab allocated with `queue_alloca`;
- any cross-command-buffer transients that intentionally live past the local
  slab;
- small scalar/config buffers when dispatch parameters need device-visible
  storage.

The local transient slab is suballocated by a C memory plan. If a command
buffer touches hundreds of temporary tensors, it still queues one allocation
for the slab. Slab offsets become binding
reference offsets or dispatch constants.

The memory planner must make lifetimes and aliasing explicit. A tensor range
may be reused only when the producer-to-consumer semaphore graph proves the
previous value is dead. The plan should be inspectable and dumpable because bad
reuse bugs are otherwise brutal to diagnose.

### LoRA Handling

LoRA support should handle multiple adapters and strengths. The runtime needs
two paths:

- dynamic LoRA kernels where optional adapter math is controlled by Loom
  `config.*` facts and folds away when no adapter applies;
- prebaked LoRA weight materialization where selected adapters are applied into
  a derived weight slab for repeated use.

Prebaking is an API operation, not a hidden side effect. The user-visible model
session should be able to load read-only base weights, load or select adapters,
choose adapter strengths, bake a derived weight set, and execute requests
against that selection. Multiple baked selections may coexist when memory
allows.

## Pipeline Skeleton

The intended dataflow is:

1. Accept a structured JSON prompt and generation parameters.
2. Tokenize with the IREE tokenizer.
3. Run Qwen3-VL text encoding and extract the required hidden states.
4. Concatenate and project text features for the DiT.
5. Initialize latent noise on the device with a deterministic seed.
6. Generate or upload the Euler flow-matching schedule.
7. For each denoising step:
   - execute the conditional DiT pass;
   - execute the unconditional image-only DiT pass;
   - run CFG and any guidance math on the device;
   - update latents with the Euler step on the device;
   - stream progress or preview data when requested.
8. Transform diffusion latents into the VAE decode representation inside the
   VAE component.
9. Run VAE decode.
10. Clamp/convert pixels and deliver output through the sink.

The first implementation may keep the step loop on the host while issuing
asynchronous queue work. The design should keep the loop body pure enough that
a future `queue_execute_while`-style primitive could move the step control to
the device.

The CLI accepts the same generation policy that the C API carries into
planning. `--dry_run` should be able to tokenize a full prompt, build the
generation plan, and dump the structured plan without loading parameters or
issuing device work. VAE tiling is an explicit plan policy: disabled decode,
explicit latent tile size, relative latent tile size, and memory-budgeted tile
selection are distinct modes, and mode-specific fields must either be provided
where required or rejected when they would otherwise be ignored.

## Reference Oracle Plan

The reference oracle is a known-good `stable-diffusion.cpp` run using dense
BF16 Ideogram 4 diffusion weights, unconditional BF16 weights, dense
Qwen3-VL-8B-Instruct text encoder weights, VAE, and a full JSON prompt. The
BF16 path is the quality oracle for kernel and sub-pipeline goldens.
Vulkan/RADV is useful early because it provides a portable debugging backend
and ISA-dump workflow; AMDGPU HAL remains the product path.

The original Python implementation is also useful even when it is not the
fastest or most stable execution path. It exposes semantic structure, fusion
opportunities, and clean submodel boundaries that may be obscured by
`stable-diffusion.cpp` execution details. The Qwen3-VL text encoder is a good
example: it can be treated as a substantial sub-pipeline with a crisp
input/output golden boundary.

The reference checkout should be instrumented at these seams:

- model loading and tensor name/shape/type discovery;
- Qwen3-VL tokenization and hidden-state extraction;
- Ideogram 4 transformer block inputs and outputs;
- attention QKV, QK-RMSNorm, MRoPE, and attention output;
- MLP/SwiGLU;
- AdaLN modulation and final layer;
- patchify and unpatchify;
- conditional and unconditional DiT pass boundaries;
- CFG and Euler update;
- VAE decode;
- final pixel conversion.

Trace extraction should emit deterministic records with tensor names, shapes,
types, strides, FP8 scale metadata where present, parameter values, seeds,
schedule values, and enough binary tensor data to build reduced golden
fixtures. The trace is an oracle and inventory source, not the runtime schedule
representation.

## Kernel Porting Strategy

Each Loom kernel family should start with the smallest behavior class that can
be proved:

- write a hand-authored `.loom` source with meaningful names;
- add `check.case` records using reduced tensors from the reference oracle;
- add `check.benchmark` rows for the configurations the model actually uses;
- run host-only dry-run planning continuously;
- run AMDGPU correctness when a device is available;
- only then wire the kernel into the C/HAL scheduler.

Important expected kernel families include:

- BF16 and FP8 linear/matmul variants used by Qwen3-VL, DiT, and VAE;
- RMSNorm and LayerNorm;
- QKV projection and splitting;
- RoPE/MRoPE construction or application;
- attention, including flash-attention-shaped paths where possible;
- SwiGLU;
- AdaLN modulation and gated residual updates;
- tensor layout transforms, slices, concatenation, patchify, and unpatchify;
- CFG and adaptive/projected guidance math used by selected presets;
- schedule generation and Euler latent update;
- random latent initialization;
- VAE convolution/residual/normalization/decode pieces;
- pixel clamp, scale, and format conversion.

When a reference operation is host-side C++ today, still consider it a kernel
candidate if it manipulates tensors. The long-term target is device-owned tensor
state from initial latent creation through final pixel tensor production.

The kernel schedule should be allowed to improve on both `stable-diffusion.cpp`
and the Python reference. Reference traces establish semantics and provide
goldens; they are not a limit on fusion, tiling, command-buffer grouping, or
target specialization. When a fusion removes an intermediate tensor from the
production path, retain a diagnostic mode or reduced fixture strategy that can
still prove the fused result against the original stage boundaries.

For linear and attention-heavy regions, the first implementation target is
BF16 correctness. The next implementation target is an FP8-weight variant on
gfx1100 that software-expands compact weights to BF16 inside the kernel. The
third target is a native FP8-weight variant on gfx942. Those variants should
share source-level operation declarations and test fixtures where possible, with
target-specific implementation blocks and tolerances that name the exact
numeric mechanism under test.

## Correctness Strategy

Correctness is layered:

- tokenizer tests compare prompt token IDs against the reference tokenizer;
- `loom` `check.case` tests compare individual kernel families against reduced
  tensor goldens;
- C scheduler tests compare memory plans, binding tables, and semaphore graphs;
- C/HAL integration tests compare sub-pipeline outputs against oracle tensors;
- end-to-end tests compare deterministic final tensors or images under a fixed
  model/sample configuration.

Image-only comparisons are too weak for kernel porting. End-to-end images are
useful smoke tests, but every new kernel family needs tensor-level evidence at
the narrowest practical boundary.

Numeric tolerances should be explicit per fixture and per dtype. Any tolerance
that exists because the reference and Loom path intentionally use different
math must name the mechanism. Wider tolerance is not a substitute for
understanding a mismatch.

## Diagnostics

The runtime should have first-class diagnostics from the beginning. Expected
CLI/API controls include:

- a dump directory selected by the caller;
- compile dump modes for transformed Loom text, Loom bytecode, compile reports,
  executable artifacts, and artifact manifests;
- tensor dump filters by stage, layer, tensor name, and request index;
- HAL dump modes for command-buffer summaries, binding tables, slab plans,
  semaphore timelines, and queue submission records;
- HAL profiling modes for queue-level and dispatch-level timing through the
  IREE HAL profiling APIs;
- comparison modes that run a selected stage against oracle tensors and emit
  mismatch summaries plus optional full tensors;
- profile modes that retain HAL dispatch metadata and Loom compiler reports for
  later performance analysis.

`loomc` artifacts are in-memory results. File dumping should be implemented as
an `id4` diagnostic sink that walks `loomc_result_t`, prints structured
diagnostics, and writes selected `loomc_artifact_t` objects with stable
identifiers. Useful identifiers should include kernel family, specialization
key, target profile, artifact kind, and a short content hash.

At the emit stage, request artifact manifests and compile reports when the user
asks for detailed dumps. At the compile stage, request transformed module text,
module bytecode, and JSON reports when those are the evidence needed. These are
separate evidence channels and should stay separately configurable.

HAL profiling data is gap-finding evidence, not the primary benchmark result.
The runtime should make it easy for an API benchmark or smoke test to request a
full queue and dispatch profile dump, then correlate those timings with kernel
specialization keys, command-buffer records, slab plans, and Loom artifacts.
This makes profile output useful when deciding which kernels dominate or where
scheduling gaps sit, while keeping durable performance claims anchored in Loom
kernel benchmarks and C API benchmarks.

## Smoke Test

Maintain a checked-in smoke-test script now that the CLI can produce an image.
The first harness runs one fixed structured JSON prompt through the public CLI,
writes artifacts into one caller-selected output directory, and validates that
the resulting image has plausible contents rather than merely existing as a
file. The request parser canonicalizes JSON prompt payloads before tokenization,
so human-readable and compact spellings of the same request stay in the same
token bucket.

The smoke-test output directory should be self-contained:

- request configuration, prompt, seed, model identifiers, and command line;
- final image and any partial or progress images;
- selected tensor dumps and mismatch summaries;
- `loomc` diagnostics, transformed modules, bytecode, executable artifacts, and
  artifact manifests;
- HAL command-buffer summaries, binding tables, slab plans, semaphore timelines,
  queue submissions, and profiling records;
- benchmark-style timing summaries when requested.

The smoke test is for occasional human inspection and regression triage. It
should fail loud on missing model files, missing devices, invalid images, or
diagnostic write failures. It should not replace tensor-level tests, Loom
`check.case` coverage, or benchmark targets.

## Benchmarking

Benchmarking comes after correctness for each scope. The first benchmark suite
should answer three different questions:

- per-kernel timing through `iree-benchmark-loom`;
- sub-pipeline timing for text encoding, one DiT block, one denoising step, VAE
  decode, and pixel conversion;
- full prompt-to-image timing through the C API.

Benchmarks must distinguish submission overhead from end-to-end completion.
Submit-only timing is useful for HAL scheduling work; user-visible image
latency must include completion waits, decode, and output delivery.

HAL profiling can be enabled during API benchmarks to explain where time went:
queue waits, transfers, dispatches, command-buffer reuse, and synchronization
gaps. These profiles are diagnostic drill-down artifacts. The stable comparison
numbers remain the per-kernel Loom benchmarks and the end-to-end API benchmark.

## Initial Milestones

1. Establish BF16 reference execution with `stable-diffusion.cpp` and one fixed
   JSON prompt.
2. Add reference instrumentation that can dump tensor metadata, per-stage
   tensors, and operation configuration for the Ideogram 4 path.
3. Build a kernel inventory from the trace and group operations into reusable
   Loom kernel families.
4. Add the first BF16 `.loom` kernel family with reduced golden fixtures and
   benchmark rows.
5. Add target-specialized FP8-weight variants for that kernel family: software
   FP8-to-BF16 expansion on gfx1100 and native FP8 execution on gfx942, both
   compared against the BF16 goldens.
6. Add the C API skeleton, HAL device/session ownership, and `loomc` compile
   cache skeleton.
7. Add the memory-plan and command-buffer-recording layer for one sub-pipeline.
8. Widen from a single sub-pipeline to one full transformer block.
9. Widen to one denoising step with conditional, unconditional, CFG, and Euler
   update.
10. Widen to the full denoising loop.
11. Add VAE decode and final pixel output.
12. Add LoRA selection and prebaking.
13. Add the checked-in smoke-test script with a self-contained artifact dump
    directory and image-content validation.
14. Add end-to-end tests and benchmarks that run through the public C API,
    including optional HAL profiling dumps for benchmark diagnosis.

Every milestone should leave behind executable correctness evidence. If a
milestone exposes a Loom compiler, `loomc`, HAL, or AMDGPU backend gap, fix or
record the gap as a design requirement before building around it.
