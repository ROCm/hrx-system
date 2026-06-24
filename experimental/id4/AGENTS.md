# ID4 Agent Guide

This guide applies to work under `experimental/id4/`. Read
`experimental/id4/README.md` before editing code here; it is the durable design
contract for this prototype. Detailed stage lifecycle, preparation,
diagnostics, and benchmark harness drafts should become checked-in design docs
only after their invariants are stable enough to serve as public archaeology.

## Project Frame

This directory is a whole-model scheduling prototype for Ideogram 4 using Loom
kernels and the IREE HAL. The implementation is a clear C model program with
direct HAL scheduling, Loom kernels, typed model tables, reusable plans, and
reference traces used as semantic oracle material.

The north star is correctness with evidence. Performance work is valuable only
after the represented tensor behavior is locked down with narrow tests and
golden fixtures.

## Checked-In Text

Checked-in files in this tree must be public archaeology. Do not mention
machine-specific paths, downloaded model locations, scratch directories,
private workflow names, or issue-tracker records. External models, generated
images, and large tensor captures are inputs or diagnostics, not repository
content.

When referencing external projects, use public URLs and describe the stable
behavior being relied on. Preserve licensing and attribution for any borrowed
code. Prefer small clean C implementations of the mechanism over copying
reference implementation structure.

## C Runtime Shape

Keep scheduling in straightforward C around the IREE HAL APIs. Repeated model
structure should be represented as loops over typed parameter and memory-plan
tables. Do not introduce JSON, ad hoc metadata interpreters, or generated
schedule files as runtime machinery.

The public C API should own loading, preparation, execution, progress sinks,
diagnostics sinks, and teardown. The CLI and any server wrapper stay thin.

Stage boundaries are coarse scheduling contracts. A stage should correspond to
a user-visible or reference-comparable unit such as Qwen3-VL prefill/forward,
decode, a sampler loop, VAE decode, or a full model slice. Internal kernels,
layer fragments, condition epilogues, RMSNorms, linears, and attention
projections belong inside a stage; they get Loom `check.case` and
`check.benchmark` coverage, not standalone C stage lifecycles or HAL
integration tests.

Loom config values describe model, tensor, dtype, and shape facts. Launch
geometry is not public stage configuration and should not be passed as generic
config knobs. Put workgroup-count and workgroup-size arithmetic in the
`kernel.launch.config` region and mirror it in C only where the current runtime
cannot yet execute that region directly. That mirror must stay local to the
dispatch authoring code so future VM-backed launch evaluation can replace it
without changing stage APIs.

Every struct field added in C headers or runtime structs needs its own adjacent
field comment. Use full words in identifiers unless an external API forces a
short spelling.

## Build Infrastructure

Use package-local layout like `runtime/` and `loom/`: a component package owns
its `.h`, `.c`, `_test.cc`, `_benchmark.cc`, `BUILD.bazel`, and generated
`CMakeLists.txt` together. Do not create a central `src/`, `tests/`, or
`benchmarks/` tree for ID4. The first shared infrastructure package is
`experimental/id4/pipeline/`; concrete model stages belong under
`experimental/id4/stages/`; assembled Ideogram 4 orchestration belongs under
`experimental/id4/ideogram4/`; thin command-line binding code belongs under
`experimental/id4/binding/cli/`.

`experimental/id4/binding/c/` is reserved for a future stable C API boundary
and is out of scope for the prototype unless explicitly requested. Runtime
integration should expose ordinary component headers from the packages that own
the implementation.

Use existing Loom Bazel rules when they express the desired production-shaped
tests and benchmarks. If the available rules cannot run a directory of `.loom`
cases cleanly, or if they force old authoring conventions into this prototype,
add focused infrastructure instead of weakening the coverage.

Infrastructure changes outside `experimental/id4/` should be reviewable on
their own. Keep those commits clean and separable so they can be moved into the
main Loom, HAL, or infrastructure history after the experiment. Use the
subsystem tag that explains the change, such as `[Loom]`, `[HAL]`, or `[Infra]`,
so shared infrastructure work has clear ownership history.

## Loom Integration

Use `loomc` as the application embedding API. Runtime code should create and
reuse `loomc` contexts, sources, link indexes, compilers, pass programs, target
profiles, target selections, workspaces, results, and artifacts directly.

The Loom command-line tools remain valid for authoring, test generation, and
manual diagnostics, but the `id4` runtime must not shell out to compile kernels
on the execution path.

Diagnostic modes should expose the artifacts that `loomc` produces:
transformed module text, bytecode, compile JSON, executable artifacts, artifact
manifests, and structured diagnostics. File output is a caller-selected
diagnostic sink over in-memory results.

AMDGPU via the IREE HAL is the product target. SPIR-V/Vulkan is a debugging and
oracle target. If a missing `loomc` or HAL API prevents clean AMDGPU artifact
handoff, treat that as a first-class integration gap in the embedding boundary.

## Precision And Targets

BF16 is the first correctness lane for Ideogram 4 kernels and sub-pipelines.
Start new tensor-producing kernel families from dense BF16 goldens unless the
operation is inherently about FP8 storage or expansion.

Keep logical kernel contracts target-independent: operation inputs, outputs,
configuration facts, accumulation behavior, and tolerance model should remain
stable while implementation blocks specialize for the target. The planned
sequence after BF16 is FP8-weight execution on gfx1100 with software
FP8-to-BF16 expansion inside the kernel, then native FP8-weight execution on
gfx942 using the same semantic fixtures. Later gfx12-class FP8 paths should fit
behind the same operation boundary.

FP8 work should preserve the exact expansion rule, scale metadata, and numeric
comparison story used by the BF16 reference. A tolerance change needs a named
mechanism such as dtype conversion, scale application order, accumulation type,
or reduction order.

## HAL Discipline

HAL queue ordering must be expressed with semaphores and command-buffer
barriers. Submission order alone is not a correctness mechanism.

Use reusable command buffers where the operation sequence is stable. Prefer
indirect binding references into the `queue_execute` binding table so command
buffers can be reused across requests, slabs, offsets, and specializations.

Transient tensors inside a command-buffer scope should be suballocations from a
small number of `queue_alloca` slabs. Do not queue hundreds of independent HAL
allocations for tensors whose lifetimes are known to be local to one recorded
region.

Memory plans are correctness objects. They should make slab sizes, offsets,
lifetimes, aliasing, and cross-command-buffer transients inspectable.

Expose IREE HAL profiling as part of the diagnostic surface once dispatching is
in place. Queue-level and dispatch-level timing dumps are for finding gaps and
prioritizing optimization work; benchmark claims should still come from Loom
kernel benchmarks and C API benchmarks.

## Kernels And Tests

Each `.loom` file should carry meaningful `check.case` coverage and benchmark
rows for the configuration classes it represents. Start from reduced oracle
tensors, prove the behavior, then wire the kernel into C.

Checked-in `.loom` cases should use synthetic values or small representative
extractions that make the operation behavior obvious. Large activation dumps,
full weight slices, generated images, and multi-megabyte tensor goldens live in
external artifact roots, not in git.

NPY fixtures are an F32 interchange convenience, not the tensor artifact model.
BF16, FP8, scaled FP8, quantized, and packed or blocked tensors need an exact
artifact format that records logical dtype, physical storage, shape, layout,
padding, and scale metadata instead of widening through F32 or smuggling bits
through integer arrays.

Reference trace tooling is checked in; reference payloads are not. Use explicit
artifact roots such as `ID4_REFERENCE_TRACE_ROOT` for raw traces and
`ID4_REFERENCE_FIXTURE_ROOT` for reduced fixtures when adding opt-in
heavyweight validation. Default presubmit and ordinary build targets should not
depend on those roots.

When a kernel covers an operation family, keep it parameterized with Loom
configuration facts instead of cloning one source per dispatch shape. LoRA
optional math should be controlled by configuration so no-adapter paths can
fold away.

Tests should exercise production paths: `loom` tool tests for kernel behavior,
GTest coverage for C API and scheduler behavior, and HAL integration tests for
real command-buffer execution. Benchmarks should be added once the correctness
surface is stable.

Maintain a checked-in smoke-test script once the CLI exists. It should run a
fixed structured prompt, validate that the generated image has plausible
contents, and write a self-contained artifact directory containing final and
progress images, `loomc` outputs, tensor dumps, HAL schedules, HAL profiles, and
the exact request configuration. The smoke test is for human inspection and
triage; it is not a substitute for narrow tensor-level correctness tests.

Do not loosen tests to accommodate unclear math. If a tolerance needs to be
wider than expected, identify the dtype conversion, reduction order, FP8 scale
application, or reference mismatch that explains it.

Reference traces are semantic evidence and oracle material. They are not a
requirement to preserve the `stable-diffusion.cpp` dispatch schedule. Use the
Python reference and model structure to identify better fusion boundaries,
especially around Qwen3-VL, attention, linear, modulation, and guidance
regions, while keeping diagnostics capable of proving fused outputs against the
original stage boundaries.

## Diagnostics

Instrumentation is part of the product, not temporary scaffolding. New
scheduler layers should have dumpable summaries for tensor metadata, slab
plans, binding tables, command-buffer dispatches, semaphore timelines, and
Loom compiler artifacts.

Tensor dumping must be filterable by stage, layer, tensor name, and request.
The default path should stay quiet enough for normal execution, but the detailed
path should make a single bad tensor traceable back to source inputs, kernel
configuration, binding offsets, and reference oracle data.

## Bug Handoffs

When this work exposes a Loom, HAL, or runtime bug, file it in the repository's
native issue tracker and create a standalone handoff directory under
`.notes/loom/bugs/<issue-id>/`. In this worktree the issue identifier is usually
a bead ID; other repositories may use a different tracker.

Each handoff directory should be useful without any surrounding session context:
a `README.md` with the mechanism, expected behavior, observed behavior,
environment, reproduction steps, and current hypothesis; minimized reproducers;
golden inputs or outputs when licensing permits; relevant logs; and any
reduced `.loom` cases or C snippets needed to reproduce the failure. Treat the
directory as something that could be archived and handed to another agent in a
different worktree.
