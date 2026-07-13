# Ideogram 4 FP8 Inference with Loom and the IREE HAL

This case study records a complete Ideogram 4 inference implementation authored
as Loom kernels and a standalone C program over the IREE HAL. The purpose was
not to reproduce a framework graph. It was to test whether a compact JIT kernel
compiler, explicit device scheduling, and model-aware memory planning could
match an optimized reference while retaining an authorable path to fusion and
target specialization.

The measured result on one Radeon Pro W7900 (`gfx1100`) is:

- **89.710 seconds** for a warm 1024x1024, 20-step generation in ID4;
- **93.494 seconds** for the official Ideogram 4 FP8 PyTorch implementation;
- **0.960x reference time**, or a 4.2% throughput lead;
- **25.381 GiB** combined resident and pooled device memory in ID4;
- **33.158 GiB allocated** and **36.262 GiB reserved** by PyTorch.

Both paths used the advertised `V4_DEFAULT_20` sampler, seed `20260625`, the
same structured request, source-native FP8 Qwen and DiT weights, BF16
activations, 269 Qwen tokens, and 4,096 image tokens. ID4 was faster in all six
measured prompt and resolution buckets. The strongest gains occurred at small
resolutions, where explicit scheduling and low framework overhead matter most;
the 1024 result is near parity with a measured lead, not a claim that every ID4
kernel is faster.

## Evidence Classes

The performance analysis uses five labels consistently:

- **Measured** values come directly from a request or kernel execution.
- **Derived** values are arithmetic over measured rows, such as ratios or
  profile shares.
- **Projected** values substitute a demonstrated same-shape kernel result into
  a measured request.
- **Theoretical** values apply a calibrated rate to work across a request.
- **Contextual** values explain scale but are not accepted bounds.

Only measured values are headline results. In particular, the demonstrated
attention schedule and the rocBLAS matrix rate suggest a 0.754x theoretical
envelope; ID4 has not measured that request time.

## Measurement Contract

The primary number is warm serving latency. Each implementation loaded the
model once, completed one full same-request warm generation, and then executed
three measured generations without reloading model state. The timed region
includes asynchronous issue and final device completion. It excludes process
startup, model loading, JIT compilation, parameter preparation, result
readback, and image serialization.

The ID4 row is the Google Benchmark mean across three iterations. The PyTorch
row is the median of three explicit invocations. This asymmetry is retained in
the table rather than hidden behind a common label. The runs used the same
physical GPU. ID4 used the IREE AMDGPU HAL with a ROCm 7.14 development runtime;
the reference used PyTorch `2.9.1+rocm7.11.0a20251218` and Triton
`3.5.1+rocm7.11.0a20251218`. ID4 was built with Bazel opt mode, ThinLTO,
`-O3`, and `-march=native`. ASLR was disabled and the benchmark host was held
under its performance lock.

The measured ID4 revision was `7a387df86d1dc76a3c1a707689462e0fd86849ae`.
The official Python source revision was
`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`; the checkpoint revision was
`ee79a7237b519f1402ceacf952f30c8a31ec5073`. Subsequent source contraction is
gated by compiler-report, ISA, correctness, and ABABA equivalence checks before
it replaces a measured provider.

A current-source confirmation at revision
`3b98db8291e73ac69c76bc2fe198d876a240a7e3` measured 90.046 seconds, or
0.963x PyTorch and 0.374% above the canonical ID4 row. Its generation plan
changes six scalar leaves from the canonical long-1024 plan, all of them the
module identity for two VAE dispatches moved to the final target-neutral
provider. Shapes, layouts, launches, dispatches, operation order, and memory
are unchanged. The sanitized
[confirmation record](evidence/current_head_long_1024_confirmation.json)
retains the binary, benchmark, request, and plan hashes.

### One-Shot Cost

A separate diagnostic CLI run retained every boundary excluded from the warm
benchmark. It used the same long 1024 request and produced the final image.

| Boundary | Duration |
| --- | ---: |
| CLI setup and archive opening | 335 ms |
| request planning | 21 ms |
| initial generation-bundle preparation call | 110 ms |
| uncached Loom and HAL executable preparation | 736 ms for 140 specializations |
| device copies into resident execution layouts | 957 ms for 25.39 GB |
| device execute stream | 88.362 s |
| final completion wait | 421 ms |
| decoded image readback | 17 ms |
| PPM serialization | 18 ms |
| full CLI command | **91.034 s** |

The one-shot issue interval includes lazy stage preparation in stage-serial
mode, so it is not a submit-only benchmark. Its purpose is to make JIT,
parameter movement, execution, and output costs visible; it does not replace
the clean warm-serving row.

The checked [warm-serving scorecard](evidence/warm_serving_scorecard.json)
contains the exact machine-readable timing, memory, token-capacity, plan, and
stage-profile values used below. It is compact derived evidence; raw benchmark
outputs, full HAL profiles, tensor captures, and compiler bundles remain
external. The companion
[performance scorecard](evidence/performance_scorecard.json) records the
profile Pareto, exact-shape kernel anchors, hardware calibration, evidence
classes, and hashes of its raw inputs.

## Scaling

| Request | Qwen tokens | Image tokens | PyTorch FP8 | ID4 | ID4 / PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| [short 128](requests/short_128.json) | 19 | 64 | 8.258 s | 3.062 s | **0.371x** |
| [structured 128](requests/structured_128.json) | 171 | 64 | 9.227 s | 3.952 s | **0.428x** |
| [long 128](requests/long_128.json) | 269 | 64 | 10.146 s | 5.050 s | **0.498x** |
| [long 256](requests/long_256.json) | 269 | 256 | 12.715 s | 7.494 s | **0.589x** |
| [long 512](requests/long_512.json) | 269 | 1,024 | 25.567 s | 20.123 s | **0.787x** |
| [long 1024](requests/long_1024.json) | 269 | 4,096 | 93.494 s | 89.710 s | **0.960x** |

This sweep matters for two reasons. First, the prompt path is dynamic: token
counts of 19, 171, and 269 produce different capacities and launch
configurations rather than selecting a fixed smoke-test trace. Second, image
tokens scale from 64 to 4,096. The narrowing advantage at large sizes exposes
matrix and attention schedule quality instead of masking it with host overhead.

## Memory

| Request | ID4 direct resident | ID4 pooled transient | ID4 combined | ID4 logical plan | PyTorch allocated | PyTorch reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short 128 | 23.655 GiB | 0.055 GiB | 23.710 GiB | 23.701 GiB | 26.670 GiB | 27.469 GiB |
| structured 128 | 23.717 GiB | 0.055 GiB | 23.771 GiB | 23.733 GiB | 26.718 GiB | 27.523 GiB |
| long 128 | 23.757 GiB | 0.058 GiB | 23.815 GiB | 23.757 GiB | 26.747 GiB | 27.523 GiB |
| long 256 | 23.759 GiB | 0.072 GiB | 23.831 GiB | 23.763 GiB | 26.844 GiB | 27.570 GiB |
| long 512 | 23.768 GiB | 0.188 GiB | 23.955 GiB | 23.898 GiB | 27.617 GiB | 29.516 GiB |
| long 1024 | 23.803 GiB | 1.578 GiB | **25.381 GiB** | 25.268 GiB | **33.158 GiB** | 36.262 GiB |

The ID4 direct-resident count is the device allocator's live byte count after
the warm generation. Queue `alloca` transient slabs are pooled beneath that
allocator surface and are measured independently by the HAL memory profile.
Their sum is the combined physical estimate. The logical plan high-water is an
independent semantic lifetime calculation, not another allocation to add. At
1024, the physical estimate and the plan differ by 0.113 GiB.

ID4 therefore uses 23.5% less memory than PyTorch's allocated peak and 29.9%
less than its reserved peak in the largest case. The execution-layout archives
retain compact FP8 linear weights. They do not cache a model-scale BF16
expansion. The serving plan holds 25.39 GB of unique parameter bytes in four
coarse stage layouts; request-dependent source tensors are retained only where
the model actually requires dynamic slicing.

## Correctness

The Python implementation is the semantic oracle. The production policy uses
the same official sampler preset and generation request, and operation-level
validation proceeds from exact tokenization and major-stage taps down to
reduced kernel cases. The final 1024 images reproduce the same three people,
clothing, storefront, crosswalk, pose, framing, and lighting structure.

![Official PyTorch reference on the left and ID4 output on the right.](assets/id4_long_1024_reference_then_actual.png)

*Official PyTorch reference on the left; ID4 on the right. Both images were
generated with Ideogram 4. The model is used for non-commercial research under
the [Ideogram 4 model agreement](https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md).*

The paired image comparison reports:

- luma SSIM: `0.749804`;
- mean RGB SSIM: `0.743529`;
- PSNR: `18.781 dB`;
- mean absolute channel difference: `18.650 / 255`.

These metrics describe one deterministic end-to-end sample, not general image
quality. The residual difference is local generative and numeric drift rather
than a collapsed scene, random noise, or a different prompt interpretation.
Tensor taps remain the stronger evidence for individual operations because
image similarity alone cannot localize a dtype, layout, scale, or reduction
error.

The checked [correctness scorecard](evidence/correctness_scorecard.json)
retains the oracle revisions and precision policy, two Qwen token lengths,
first-block DiT taps, selected sampler-trajectory steps, exact-latent VAE
decode, final-image metrics, and hashes of the raw comparisons. It makes the
scope of the claim explicit: operation-local agreement is tight, while
iterative numeric drift reaches 0.919409 cosine similarity in the 128x128
step-20 latent. The final images establish structural parity, not pixel
identity.

Correctness is gated at several scopes:

1. The IREE tokenizer is compared against the reference token sequence.
2. Loom `check.case` records exercise nonzero reduced kernels and numeric
   tolerances.
3. C planning tests validate tensor lifetimes, aliases, parameter layouts,
   launch specialization records, and semaphore-independent program structure.
4. Real-HAL integration tests prepare real Loom kernels, record reusable
   command buffers, issue them on the selected device, and compare results.
5. Major Qwen, DiT, sampler, and VAE taps compare production execution with
   reference tensors.
6. The fixed official request produces the final image comparison.

## Where Time Goes

The 1024 diagnostic profile contains 24,803 dispatches and 90.391 seconds of
summed dispatch time. It is attribution evidence rather than the clean serving
number.

| Family | Calls | Time | Profile share |
| --- | ---: | ---: | ---: |
| compact FP8 DiT contractions | 6,800 | 56.996 s | 63.05% |
| online DiT attention | 1,360 | 27.925 s | 30.89% |
| QKV normalization and rotary | 1,360 | 2.017 s | 2.23% |
| normalization, modulation, residual, and SwiGLU | 6,800 | 2.323 s | 2.57% |
| Qwen conditioning | 557 dispatches | 0.175 s | 0.19% |
| VAE decode | 85 dispatches | 0.432 s | 0.48% |

Contractions and attention account for 93.95% of dispatch time. The profile is
not a collection of minor launch overheads waiting to be fused away. Qwen and
VAE are already too small to move the 1024 headline materially.

Separate stage profiles give the following attribution:

| Stage | PyTorch event span | ID4 queue span |
| --- | ---: | ---: |
| Qwen / conditioning | 1,764 ms | 122 ms |
| conditioned DiT, 20 calls | 48,042 ms | 45,780 ms |
| unconditioned DiT, 20 calls | 42,958 ms | 43,139 ms |
| VAE decode | 627 ms | 433 ms |

These stage rows are not independent benchmark claims. The official Qwen event
includes neighboring conditioning work and the sampler scopes differ. The full
request is the apples-to-apples comparison. The rows identify the
unconditioned DiT as the remaining large-shape deficit.

## Kernel Anchors

The hot kernels were rerun at exact 1024 production shapes with rotating
device-local binding sets. Rotation prevents a repeatedly reused allocation
from making a cache-hot microbenchmark look like model execution.

| Kernel | Shape | ID4 p50 | Complete official path | ID4 / official |
| --- | --- | ---: | ---: | ---: |
| unconditional QKV | M4096xN13824xK4608 | 8.562 ms | 8.859 ms | **0.966x** |
| unconditional MLP down | M4096xN4608xK12288 | 7.807 ms | 7.974 ms | **0.979x** |
| unconditional attention | Q4096xKV4096xH18xD256 | 16.663 ms | 21.807 ms | **0.764x** |
| conditioned attention | Q4368xKV4365xH18xD256 | 16.341 ms | 26.643 ms | **0.613x** |

The matrix comparison includes PyTorch's FP8 widening, row scaling, and BF16
GEMM, not just the library GEMM. Pure rocBLAS BF16 GEMM measures 7.885 ms for
QKV and 7.113 ms for MLP down, so ID4 remains 8.6-9.8% behind that narrower
operation while beating the complete official FP8 weight path.

The accepted matrix kernel keeps E4M3 bytes in a compact RHS tile, performs a
software FP8-to-BF16 decode on `gfx1100`, applies the row scale, and feeds BF16
WMMA with F32 accumulation. It sustains 59.4-60.9 TFLOP/s with no spills. The
accepted attention kernel performs online BF16 QK/PV WMMA with F32 max and sum
accumulation and never materializes the full score matrix.

## Calibrated Headroom

Device calibration on the same HAL path measured 720.1 GB/s streaming read,
654.8 GB/s streaming write, 583.0 GB/s aggregate copy traffic, 2.247 trillion
F32 exponential values per second, and 66.18 TFLOP/s for the best exact-shape
rocBLAS BF16 matrix row.

| Evidence | Request time | ID4 / PyTorch | Interpretation |
| --- | ---: | ---: | --- |
| measured | 89.710 s | **0.960x** | accepted warm-serving result |
| projected | 82.633 s | **0.884x** | substitutes a measured historical spill-free attention schedule |
| theoretical | 70.540 s | **0.754x** | also applies the best exact-shape rocBLAS rate to every DiT contraction |
| target context | 70.120 s | **0.750x** | aspirational threshold, not a result |

The attention projection is grounded in a 12.440 ms same-shape Loom schedule
and is worth 7.078 seconds at request scale. The matrix quotient contributes at
most another 12.093 seconds under the strong assumption that every contraction
reaches the best observed library rate. Even that arithmetic misses 0.75x by
0.420 seconds. A 0.75x request is therefore near the calibrated edge for this
request, not an ordinary tuning milestone.

Compiler-issued global request counts are not physical HBM traffic. Cache reuse
can make those counts exceed board bandwidth, so they are used to compare
lowered schedules rather than to claim impossible memory rates.

## Runtime Architecture

The runtime is a C model program rather than a serialized framework graph.
Each stage authors its forward pass with typed tensor handles, parameters,
acquires, dispatches, barriers, taps, and exports. Repeated transformer layers
are ordinary C loops. Kernel module paths and parameter keys live at the point
where the model math is expressed instead of in public enums or parallel side
tables.

The lifecycle has four concrete boundaries:

1. **Load** creates the session, shared device group, tokenizer, in-memory Loom
   library, compiler cache, and immutable model configuration.
2. **Plan** tokenizes the actual request and runs the semantic C program in a
   dry mode. It selects operation policies, resolves dynamic capacities,
   computes tensor live ranges, packs local slabs, lays out parameter slabs,
   records kernel specializations, and emits inspectable JSON without JITing or
   touching model data.
3. **Prepare** resolves each stable module path through the in-memory kernel
   library, specializes its `config.*` values, obtains launch configuration
   from the Loom program, compiles through `loomc`, prepares HAL executables,
   loads or gathers parameter layouts, and records immutable reusable command
   buffers.
4. **Issue** binds request and slab subranges, queues transient slab allocation,
   executes the prepared command buffers, and queues deallocation. Caller wait
   and signal semaphore lists define every user-visible ordering edge; HAL queue
   submission order is never treated as FIFO.

Stages are coarse scheduling units: Qwen forward, conditioned DiT,
unconditioned DiT, sampler operations, and VAE decode. A layer, linear,
normalization, or attention projection is an internal program operation rather
than a stage. This keeps command-buffer reuse broad and prevents layer
boundaries from becoming queue submissions or memory-scope boundaries.

The parameter path is similarly explicit. Checkpoint tensors can be gathered
and encoded into final execution layouts, while baked IRPA archives store those
layouts for direct server startup. Archive entries carry physical dtype, shape,
encoding, alignment, byte length, and a source-schema fingerprint. Resident and
bounded-streaming policies are closed choices with different memory and
submission contracts; one does not silently fall back to the other.

The implementation surfaces are [the semantic program builder](../pipeline/program.h),
[the plan and statistics API](../pipeline/plan.h),
[the program preparation path](../pipeline/program_prepare.h),
[the kernel cache](../pipeline/kernel_cache.h),
[parameter layout archives](../pipeline/parameter_layout.h), and
[the assembled session API](../ideogram4/session.h).

## Diagnostics as a Product Surface

The same semantic program drives dry-run planning and HAL recording. That gives
diagnostics a stable vocabulary before a command buffer exists:

- tensor dtype, shape, byte range, lifetime, alias root, and slab offset;
- parameter source, physical encoding, execution-layout destination, and load
  group;
- kernel module path, export, config bindings, specialization key, and launch
  configuration;
- explicit barriers, region boundaries, binding slots, and timeline edges;
- named tensor taps lowered to device copies only when requested.

The `loomc` path retains selected transformed modules, bytecode, compile JSON,
code objects, disassembly-oriented reports, and structured diagnostics in
memory. A diagnostic sink may write those artifacts without putting file I/O on
the production compile path. HAL profiling joins queue and dispatch records
back to plan specialization keys, so a slow request can be reduced to an exact
kernel shape and compiler report.

## Designs Replaced Along the Way

Several early shapes were useful only because they exposed the wrong contract:

- A trace-shaped scheduler with per-kernel stage objects made fusion and layer
  loops expensive to author. The replacement is one mathematical C program per
  coarse stage.
- CPU stand-ins and fake executables made preparation tests pass without
  exercising the production contract. Planning remains host-testable;
  preparation and issue use real Loom kernels and real HAL devices.
- A model-scale source or BF16 expansion cache consumed roughly 12 GiB beyond
  compact FP8 residency. Baked final-layout archives and bounded relay windows
  removed the duplicate.
- Byte-count or lexicographic parameter partitioning destroyed locality and
  leaked HAL binding decisions into kernel ABIs. Parameter gathering,
  execution layout, residency, and ISA offset limits are now independent,
  semantic decisions.
- Fixed prompt lengths and copied launch arithmetic optimized a smoke shape.
  Request token capacities are dynamic, and launch count and size belong to the
  Loom launch-config region.
- Full-image VAE intermediates made 1024 decode unnecessarily expensive in
  memory. VAE tiling is an explicit policy and the planner reports its live-set
  consequence.
- False barriers serialized independent writes. Program barriers now represent
  real visibility and lifetime epochs, while independent dispatches remain
  eligible for concurrent execution.
- An artificial 16-token attention fixture sat outside the model's declared
  minimum configuration and selected a globally slower 16 KiB LDS accumulator
  schedule. Keeping reduced fixtures inside the production domain restored the
  512-byte register-carried recurrence; smaller compiler stress shapes remain
  valuable as standalone compiler cases rather than product schedule inputs.
- Obvious fusion was not assumed to be faster. Register pressure, duplicate
  arithmetic, and lost reuse can reverse the win; compiler reports and ABABA
  measurements decide which schedule remains active.

## Result and Remaining Frontier

The experiment demonstrates that a JIT-authored model can beat the official
PyTorch FP8 implementation across dynamic prompt and resolution buckets while
using materially less VRAM. It also demonstrates where the abstraction stops
being the differentiator: at 1024, two highly optimized kernel families own
almost the entire request.

The immediate performance frontier is narrow. The production online-attention
provider is again register-carried, spill-free, and structurally accepted; the
historical 12.440 ms row remains a projected opportunity rather than a missing
correctness contract. Matrix work is next: exact FP8 decode, WMMA, wait, load,
and store schedules remain 8.6-9.8% behind the narrower rocBLAS BF16 GEMMs at
the same shapes even though the complete ID4 FP8 consumers beat PyTorch's
expand-scale-GEMM paths. Minor kernels are below the threshold where local
improvements can change the headline.

The larger result is the method. Model math, dynamic shape policy, target
specialization, parameter representation, memory lifetime, diagnostics, and
performance evidence are separate but composable contracts. The companion
[porting guide](porting_guide.md) describes how to apply those contracts to a
new model family.
