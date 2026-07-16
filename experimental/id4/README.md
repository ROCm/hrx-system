# Ideogram 4 Loom Prototype

This directory contains a standalone Ideogram 4 inference implementation built
from Loom kernels and the IREE runtime HAL. It accepts dynamic structured
prompts, tokenizes with the IREE tokenizer, runs Qwen3-VL conditioning, executes
the full dual-branch diffusion loop, decodes through the VAE, and produces an
image through a C API and thin CLI.

The implementation is also a scheduling-language experiment. Its C model
program is the hand-authored reference for a future Loom host scheduling IR:
mathematical model structure, parameter representation, target-specialized
kernels, tensor lifetimes, asynchronous queues, and diagnostics remain explicit
and inspectable.

The measured result and reusable method are documented separately:

- [Build and run quickstart](docs/quickstart.md)
- [Ideogram 4 FP8 inference case study](docs/case_study.md)
- [Production-shaped Loom model porting guide](docs/porting_guide.md)

## Quick Start

The current full-model evidence is Linux on a Radeon Pro W7900. The runtime
discovers the selected AMDGPU target and uses the compatible Loom providers;
there is no architecture encoded in the model command. The direct-checkpoint
FP8 base path has a sampled 1024x1024 physical VRAM peak of about 25.8 GiB;
dynamic LoRAs have also run end to end, with their separate cost recorded in
the quickstart. Accept the gated
[Ideogram 4 FP8 license](https://huggingface.co/ideogram-ai/ideogram-4-fp8),
authenticate with `hf auth login`, and download the exact model artifacts used
by this prototype:

```bash
hf download ideogram-ai/ideogram-4-fp8 \
  transformer/diffusion_pytorch_model.safetensors \
  unconditional_transformer/diffusion_pytorch_model.safetensors \
  --revision ee79a7237b519f1402ceacf952f30c8a31ec5073 \
  --local-dir models/ideogram-4-fp8

hf download Qwen/Qwen3-VL-8B-Instruct-FP8 \
  tokenizer.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors \
  --revision 9cdc6310a8cb770ce18efaf4e9935334512aee45 \
  --local-dir models/qwen3-vl-8b-instruct-fp8

hf download black-forest-labs/FLUX.2-dev \
  ae.safetensors \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --local-dir models/flux2-dev
```

After the one-time AMDGPU setup in the [quickstart](docs/quickstart.md), build
the CLI:

```bash
python dev.py bazel build \
  -c opt \
  --features=thin_lto \
  --copt=-O3 \
  --cxxopt=-O3 \
  --host_copt=-O3 \
  --host_cxxopt=-O3 \
  --copt=-march=native \
  --cxxopt=-march=native \
  --host_copt=-march=native \
  --host_cxxopt=-march=native \
  //experimental/id4/binding/cli:id4
```

Generate a 1024x1024 binary PPM from the checked-in structured request:

```bash
bazel-bin/experimental/id4/binding/cli/id4 \
  --flagfile=experimental/id4/docs/ideogram4-fp8.flags \
  --device=amdgpu:// \
  --prompt_json_file=experimental/id4/docs/requests/long_1024.json \
  --output=ideogram4.ppm
```

The checked flagfile maps the downloaded checkpoint with ordinary
`--parameters=` flags and selects the measured compact-FP8 execution policy.
No model preparation or second weight copy is required. The CLI also accepts a
prompt directly with `--prompt`, and repeatable
`--lora`/`--lora_strength` flags compose one or more Ideogram 4 adapters. The
[quickstart](docs/quickstart.md) has complete commands and the current hardware,
output, first-run, and model-layout constraints.

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

The model weights are gated and governed by the Ideogram 4 non-commercial
license. Model weights, LoRA files, large tensor captures, and ordinary generated
images remain external artifacts. A reduced fixture or compact visual result
enters the repository only after an explicit redistribution and provenance
review.

## Model Contract

Ideogram 4 is a 9.3B parameter flow-matching text-to-image model. Its core is a
single-stream Diffusion Transformer: text and image tokens are concatenated and
processed by the same 34 transformer blocks.

The implementation models these public architecture facts:

- Qwen3-VL-8B-Instruct is a frozen text encoder used in text-only mode.
- Conditioning concatenates hidden states from Qwen layers 0, 3, 6, 9, 12, 15,
  18, 21, 24, 27, 30, 33, and 35.
- The DiT hidden size is 4608, with 18 attention heads and a 12288-wide SwiGLU
  intermediate.
- Attention uses QK-RMSNorm and multimodal rotary position encoding.
- AdaLN timestep conditioning has width 512.
- Public diffusion latents have 128 channels. The Flux-family VAE interprets
  them as 2x2 patches over its 32-channel internal latent representation.
- Sampling uses Euler flow matching with asymmetric classifier-free guidance.
- Structured JSON captions are the primary prompt format. Plain text expansion
  is an application convenience rather than part of the scheduler.

The Python implementation under its advertised sampler configuration is the
semantic oracle. `stable-diffusion.cpp` remains useful for implementation
discovery, Vulkan/RADV shader inspection, and a second execution comparison;
its dispatch schedule and numeric choices are not presumed canonical.

## Product Contract

The runtime surface is a C library with thin applications around it:

- a session API for load, plan, prepare, asynchronous issue, completion, and
  result access;
- a CLI that accepts a plain or structured prompt and produces an image;
- streaming progress and diagnostics sinks;
- dry-run planning and memory-policy selection without model loading;
- tensor taps, Loom compiler artifacts, HAL queue profiles, and result
  summaries for fault localization;
- a future server or stable C binding built over the same session API.

This is not a GGML backend, framework adapter, operation router, or trace replay
engine. Scheduling is ordinary C that expresses the model mathematically.
Repeated layers are loops, and parameter keys and kernel references live at the
single private site that authors each operation.

Host work is limited to request ingestion, tokenization, lifecycle control,
diagnostic delivery, and the outer diffusion loop. Noise initialization,
guidance, latent updates, VAE transforms, pixel conversion, and tensor taps run
on the device. The stage and issue APIs preserve a path to moving the remaining
outer loop into a future device-side queue control primitive.

## Program Lifecycle

Every coarse model stage follows the same lifecycle:

1. **Load** creates immutable stage configuration and shared runtime services.
2. **Plan** authors and seals a semantic tensor program for the actual request,
   computes parameter and transient layouts, selects kernel specializations,
   and produces inspectable JSON without JIT or device execution.
3. **Prepare** resolves the plan through an in-memory Loom library and `loomc`,
   loads parameters, prepares HAL executables, and records immutable reusable
   command buffers.
4. **Issue** binds request-specific subranges and queues allocation, execution,
   and retirement against explicit wait and signal semaphore lists.

Stages are user-visible scheduling units such as Qwen forward, one conditioned
or unconditioned DiT forward, sampler work, and VAE decode. A transformer layer,
linear, normalization, attention projection, or epilogue remains an internal
program operation. Stage boundaries are deliberately rare because each can
become a queue submission, synchronization edge, memory scope, and benchmark
boundary.

Planning and recording use the same program walker. Dry mode computes the same
lifetimes, kernel requests, layouts, and diagnostics without retaining a shadow
command-buffer representation. Record mode immediately emits the corresponding
HAL command-buffer operations.

Single-device stages normally prepare one command buffer. A device group is
available during planning so tensor parallel or heterogeneous placement can
produce per-device recordings and explicit interchange edges without changing
the semantic stage interface.

HAL queues are not FIFO. Every ordering edge is a timeline semaphore dependency
or an explicit command-buffer barrier. Wait lists also provide backpressure:
an allocation can depend on an earlier asynchronous retirement instead of
temporarily exceeding the memory budget.

## Kernels and JIT

The runtime embeds Loom through the public `loomc` API. Production execution
does not shell out and does not require compiler artifacts on disk. An in-memory
kernel library maps stable module paths such as `qwen3_vl/rmsnorm` to source or
linked archive contents. Diagnostic sinks may persist selected in-memory
artifacts when requested.

Kernel references contain only a module path and export name. Model code passes
semantic `config.*` values for shapes, dtypes, layouts, and optional features.
Workgroup count, workgroup size, and required local memory belong to Loom
launch-config regions; C does not mirror that arithmetic.

Target-neutral exports own stable operation contracts. Target-qualified
providers own schedules that materially differ by architecture. Config-shaped
vectors and encodings cover ordinary lane and tile variation without cloning a
source module for every dispatch shape. A separate provider remains appropriate
when synchronization, dataflow, or the algorithm genuinely differs.

Every active kernel family carries small nonzero `check.case` coverage and
representative `check.benchmark` rows. Correctness cases cover tails, scale
metadata, and reduction behavior. Compiler reports make WMMA/MFMA usage,
instruction mix, memory requests, wait behavior, register pressure, LDS,
private memory, spills, and occupancy inspectable.

## Precision and Parameters

BF16 establishes the tensor-golden lane. The official FP8 model then supplies
compact weights while preserving the Python compute contract:

- DiT weights use E4M3 storage with F32 per-output-channel scales.
- Qwen FP8 weights use F32 inverse scales over 128x128 source blocks.
- Activations entering the matrix paths are BF16.
- Matrix accumulation is F32.
- Public outputs retain the reference dtype at each stage boundary.

On `gfx1100`, the production contraction path keeps FP8 weights in compact RHS
tiles, software-decodes them to BF16 inside the cooperative consumer, applies
the scale, and executes BF16 WMMA. The same semantic operation can select native
FP8 matrix instructions on targets that provide them. A model-scale BF16 weight
expansion is not a valid hidden cache.

The IREE parameter index supplies checkpoint and baked archive tensors.
Planning treats checkpoint schema, gather batching, execution layout, residency
lifetime, and HAL bindings as independent decisions. Parameters are grouped by
device placement and semantic locality rather than lexicographic name or an
arbitrary allocation size.

Baked IRPA archives store immutable tensors in their final execution layouts.
Entries include physical dtype, shape, encoding, alignment, byte length, and a
source-schema fingerprint. Server startup validates the archive and gathers it
directly into resident slabs. Request-dependent tensors, such as full token
embedding tables, retain their logical source representation only where
dynamic slicing requires it.

Resident and bounded-streaming parameter policies are explicit closed choices.
Checkpoint-to-layout encoding and streaming refill use bounded device relay
windows so source and expanded model-scale duplicates never need to coexist.
Each policy exposes its parameter, staging, constant, boundary, and transient
high-water in the plan.

## Memory Planning

A prepared stage normally binds:

- semantically coherent immutable parameter slabs;
- one stage-local transient slab allocated through `queue_alloca`;
- cross-stage boundary tensors whose lifetime intentionally spans stages;
- a constant slab and optional diagnostic capture buffers.

Hundreds of transient tensors remain subranges of one local slab rather than
hundreds of HAL allocations. `acquire` returns uninitialized storage. `release`
states the last semantic use. Reuse occurs only when lifetime and epoch analysis
prove the prior value dead.

Barriers advance the planning epoch and represent real visibility. Independent
dispatches that touch disjoint ranges remain independent; authoring order does
not create a false dependency.

VAE tiling is a plan policy with explicit tile, relative-size, or memory-budget
modes. It bounds decoder intermediates without changing the VAE stage contract,
and the same component boundary can support alternate compatible VAEs used by
other image models.

## Correctness and Diagnostics

Correctness is layered so a failure narrows instead of merely producing a bad
image:

- tokenizer tests compare prompt token IDs;
- Loom cases validate individual mathematical families;
- host GTests validate program, lifetime, plan, and policy contracts;
- real-HAL integration tests compile checked Loom modules and execute complete
  stage lifecycles on the selected device;
- external tensor fixtures compare major stage and repeated-block taps;
- the smoke harness validates a deterministic full image and retains a complete
  diagnostic directory.

Planning tests do not freeze private kernel or layer lists. Fusion and provider
selection are implementation details. Preparation and issue tests never swap
fake executables into production caches: those boundaries exist to prove real
JIT, parameter, command-buffer, binding, queue, and completion behavior.

Reference traces and full tensor payloads remain external. Checked tools can
inventory traces, reduce exact BF16/FP8 fixtures, compare captured tensors, join
HAL profiles with plans, and build acceptance scorecards. Ordinary presubmit
uses small synthetic or reduced data and never downloads a model implicitly.

Diagnostics are part of the runtime contract:

- plan JSON describes tensors, lifetimes, slabs, kernels, configs, launch
  records, parameter load groups, and resource statistics;
- named taps copy selected device tensors without changing the ordinary path;
- `loomc` can expose transformed modules, bytecode, reports, executables, and
  structured diagnostics through an in-memory result;
- HAL profiling records queue, command-buffer, dispatch, transfer, and memory
  events and joins them back to semantic specialization keys;
- fault-localization modes may split regions or wait after each segment, and
  are explicitly separate from performance execution.

## LoRA Contract

The current dynamic adapter path accepts an ordered set of LoRA safetensors and
one issue-time strength per adapter. Each artifact is indexed and validated as
BF16 down/up projection pairs over any of six supported linear sites across the
conditioned-DiT blocks. Planning authors the corresponding low-rank math while
the unconditioned branch and the no-adapter program remain unchanged.

Topology is a plan-time fact because it changes program structure. Strengths
are issue-time values, so one prepared topology can be reused at different
strengths by a long-lived session. The one-shot CLI exposes this as repeatable
`--lora=<path>` and `--lora_strength=<f32>` flags; omitted strengths default to
1.0. Multiple files preserve command-line order and are fused additively into
each shared target.

Checkpoint sources execute that authored adapter math dynamically. When the
base conditioned DiT already comes from an execution-layout archive, the CLI
instead materializes the selected strengths into an in-memory resident
conditioned-DiT variant before preparing the generation. The derived slab is
immutable for its lifetime, the base archive remains unchanged and shareable,
and no model-sized LoRA artifact is written to disk.

The resident variant is distinct from the optional disk execution-layout cache
documented in the quickstart. The direct-checkpoint path remains the ordinary
user path because it avoids the extra model-sized archive and its measured
startup benefit is small for a one-shot generation.

## Build and Repository Shape

Bazel is the source of truth. Components use the repository's package-local
layout: headers, C sources, tests, benchmarks, `BUILD.bazel`, and generated
`CMakeLists.txt` files live together. Build metadata is regenerated from Bazel
when a package changes.

The ownership boundaries are:

- `pipeline/` for generic programs, plans, regions, parameters, kernel caching,
  stage lifecycle, and diagnostics;
- `stages/` for Qwen, DiT, sampler, and VAE programs;
- `ideogram4/` for request parsing and assembled generation sessions;
- `kernels/` for semantic Loom families, target providers, cases, and
  benchmarks;
- `tooling/` and `build_tools/` for reusable capture, comparison, profiling,
  smoke, and presubmit support;
- `binding/cli/` for the thin command-line application.

The checked build has no dependency on downloaded weights, generated images,
or large activation dumps. Full-model integration and performance runs are
explicit opt-in targets supplied with standard `--device=` and `--parameters=`
flags. Kernel tests, host planning tests, and production-path linting run in the
normal Bazel presubmit.

## Continuation

The current `gfx1100` path is measured against the official FP8 implementation
in the [case study](docs/case_study.md). The next durable extensions are native
FP8 target providers on `gfx942` and newer architectures, tensor-parallel and
heterogeneous device placement, persistent resident LoRA variant reuse,
persistent compiled-executable caching, and migration of the semantic C
program into Loom scheduling IR as its host and VM facilities come online.

Those extensions keep the existing contracts: dynamic request planning,
source-native compact parameters, coarse reusable stage command buffers,
explicit timeline scheduling, device-owned tensor state, narrow correctness
oracles, and measured target specialization.
