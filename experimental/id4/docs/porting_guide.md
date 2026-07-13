# A Production-Shaped Loom Model Port

This guide distills the workflow used to bring Ideogram 4 from public reference
code to a dynamic Loom and IREE HAL implementation. It is intentionally about
method rather than the current ID4 kernel inventory. A new model should be able
to reuse the contracts while changing its mathematical program, stage
boundaries, parameter layouts, fusions, and target schedules.

## Start with the Product Contract

The first executable target is the application, not a pile of isolated
kernels. The minimum product contract is:

- arbitrary supported user input becomes tokens and shape-dependent model
  configuration;
- a deterministic dry run builds a complete inspectable plan without model
  data or device execution;
- preparation JITs real Loom kernels, loads real parameters, and records
  reusable HAL command buffers;
- issue accepts explicit wait and signal semaphores and leaves tensor state on
  the device until a declared output boundary;
- correctness can be observed at named tensor seams without changing ordinary
  execution;
- the same C API backs the CLI, integration tests, benchmarks, and a future
  server.

This contract prevents a common bring-up trap: a fixed-shape smoke path can
appear complete while token counts, launch geometry, parameter names, or
memory lifetimes remain hardcoded outside the model.

## 1. Establish the Oracle

The semantic oracle is the model author's implementation under its advertised
configuration. Alternate runtimes are useful for shader discovery, memory
behavior, and performance comparisons, but their differences are not silently
promoted to model semantics.

A useful reference capture has three layers:

1. **Request provenance** records model and checkpoint revisions, sampler,
   seed, prompt payload, tokenizer inputs, dtype policy, resolution, step count,
   software versions, and device target.
2. **Major taps** capture tokenizer output, text-encoder output, one block of
   each repeated architecture, each coarse stage boundary, sampler trajectory,
   VAE input/output, and final pixels.
3. **Kernel reductions** extract small nonzero slices that preserve tails,
   scale metadata, layouts, and reduction behavior without checking full model
   tensors into source control.

The tensor artifact format carries logical dtype, physical storage, shape,
strides or encoding, padding, and scale metadata. F32-only containers are not
an adequate interchange for BF16, FP8, blocked matrices, or packed fragments.
Widening a tensor merely to serialize it destroys the exact bits needed to
diagnose conversion and rounding errors.

Raw captures can be inventoried and reduced through the checked tooling:

```bash
bazel run //experimental/id4/build_tools:trace_inventory -- \
  --trace-dir=<raw-trace> \
  --output=<inventory.json> \
  --hash-payloads

bazel run //experimental/id4/build_tools:generate_fixture -- \
  --fixture-id=<fixture-id> \
  --trace-root=<raw-trace-root> \
  --fixture-root=<fixture-root>

bazel run //experimental/id4/tooling:fixture_compare -- \
  --fixture-dir=<fixture> \
  --actual-dir=<stage-capture> \
  --output=<comparison.json>
```

The capture is an oracle, not a schedule. Its operator boundaries establish
what must match and where taps are useful. They do not require Loom to preserve
the framework's dispatch count, intermediate tensors, or fusion choices.

## 2. Transcribe the Mathematical Program

Each coarse stage owns one C function that reads like its forward pass. Model
hyperparameters and parameter-key formatting remain private to the stage
implementation. Repeated layers are loops over ordinals and typed model
tables. Parameter strings, tensor names, and kernel module paths occur at the
single mathematical site that consumes them instead of being mirrored through
public enums and registration tables.

The ID4 semantic program supports imports, parameters, embedded constants,
uninitialized transient acquisition, subviews, fills, Loom dispatches,
barriers, diagnostic taps, exports, and explicit region cuts. The builder uses
an arena during append-only construction and seals into an immutable packed
program. Tensor shapes store a small fixed maximum rank inline, avoiding a heap
allocation per value.

[The Qwen program](../stages/qwen3_vl_program.c) and
[the DiT program](../stages/ideogram4_dit_program.c) show the intended shape:
helpers name mathematical motifs, while the top-level function remains a
readable layer loop.

Three authoring rules matter:

- `acquire` returns uninitialized storage. A consumer either overwrites the
  complete live range or an explicit producer initializes it. Zero-filled HAL
  allocations are never assumed.
- `release` states the last semantic use. Storage reuse occurs only after the
  lifetime and epoch graph proves the old value dead.
- `barrier` represents a real visibility edge and advances the planning epoch.
  Independent writes remain independent; the builder does not insert a barrier
  simply because dispatches were authored consecutively.

The same program walker supports dry planning and actual command-buffer
recording. There is no retained shadow command-buffer language. In record mode
the semantic operations immediately call the HAL command-buffer API; in dry
mode they produce the same lifetimes, layouts, kernel requests, and statistics
without a HAL command buffer.

## 3. Keep Stages Coarse

A stage boundary is a user-visible scheduling and lifetime boundary. It can
create a queue submission, completion edge, parameter residency transition,
and benchmark scope. That cost makes boundaries intentionally rare.

Good stage examples include a complete text-encoder forward pass, one
conditioned or unconditioned diffusion forward pass, a sampler phase, and VAE
decode. A linear, normalization, attention projection, transformer layer, or
condition epilogue is an internal operation inside a stage.

The prepared form for a single-device stage is normally one immutable reusable
command buffer. Tensor parallelism or heterogeneous placement may produce one
command buffer per participating device, connected by explicit semaphore and
interchange-buffer edges, without changing the semantic stage contract.

## 4. Separate Plan, Prepare, and Issue

The lifecycle separates decisions that answer different questions.

### Plan

Planning consumes the actual tokenized request, model configuration, device
group, and execution policy. It produces:

- boundary tensors and aliases;
- transient live ranges and slab offsets;
- parameter source tensors, physical encodings, destination spans, and load
  groups;
- stable Loom module paths, exports, config bindings, and specialization keys;
- region and barrier structure;
- diagnostic tap selections;
- logical memory high-water statistics.

Plans are immutable and JSON-formattable. A dry-run CLI can therefore answer
whether a request fits, which layouts and kernels it selects, and where memory
goes without downloading weights or compiling a kernel.

### Prepare

Preparation takes an exact plan and a closed parameter policy. It resolves
module paths from the in-memory kernel library, specializes and compiles through
`loomc`, obtains launch configuration from the Loom launch-config program,
prepares HAL executables, populates parameter slabs, and records reusable
command buffers.

Parameter loading is asynchronous. Caller wait semaphores gate the first load;
internal timeline points chain gathers, encoders, and command recording; caller
signal semaphores publish completion of the whole operation. JIT and
command-buffer recording can proceed on the host while parameter transfers run
on the device.

### Issue

Issue binds boundary, parameter, constant, transient, and diagnostic subranges
to the prepared command buffers. It queues `alloca`, execute, and `dealloca`
with explicit timeline dependencies. Wait lists are required because they
provide readiness and backpressure; a later allocation may need to wait for a
prior asynchronous retirement to avoid exceeding the memory budget.

The HAL queue is not FIFO. Every ordering assumption appears as a semaphore
edge or an in-command-buffer barrier. Submission order alone carries no
semantic meaning.

## 5. Treat Parameter Layout as a First-Class Program Output

Checkpoint layout, transfer batching, execution layout, residency lifetime,
and HAL bindings are orthogonal decisions.

The program names logical provider tensors and the transformation required by
the selected kernel. The plan packs final execution tensors into semantically
coherent slabs and emits load groups. One large slab may receive many gathers;
the number of transfers does not imply the number of allocations or kernel
bindings. Partitioning can follow device placement, phase locality, mutation,
or an ISA offset limit, but not an alphabetical walk over parameter names.

For server startup, baked IRPA archives contain final immutable execution
layouts. Each entry records its physical dtype, shape, encoding, alignment,
byte length, and source-schema fingerprint. A validated archive gathers
directly into its final resident slab without rerunning encoders. Full logical
source tensors are retained only for request-dependent access such as token
embedding row selection.

Checkpoint-to-archive construction uses bounded relay windows. A large source
can be gathered in chunks, encoded on the device, and written to its final
location without keeping both a model-scale source cache and an expanded target
cache resident. The same mechanism supports streaming execution when memory is
more valuable than refill latency.

FP8 weight policy is part of this contract. On `gfx1100`, ID4 stores E4M3 bytes
in compact RHS tiles, decodes to BF16 cooperatively inside the consumer, applies
the checkpoint scale, and executes BF16 WMMA with F32 accumulation. A target
with native FP8 matrix instructions may select a different provider behind the
same semantic operation and fixtures. A model-scale BF16 expansion is neither
required nor accepted as hidden cache state.

## 6. Author Semantic Kernel Families

A kernel reference consists of a stable module path and export name. Paths such
as `qwen3_vl/rmsnorm` map to source modules today and can map to linked bytecode
archives or a VFS later. The C program supplies semantic configuration such as
matrix dimensions, token counts, dtype policy, layouts, and optional features.
Workgroup count, workgroup size, and local-memory requirements belong to the
kernel launch-config region.

The source organization has two layers:

- a target-neutral exported kernel declares the operation ABI, semantic config,
  and launch contract;
- target-qualified `func.template` providers own schedules that materially
  differ by architecture.

Vector lengths and encodings can depend on config values, so lane-width clones
are not automatically separate modules. Distinct files remain appropriate when
the dataflow, synchronization, or algorithm differs enough that one source
would become harder to inspect than the variants. A large mixed module is split
by semantic family and provider before it exceeds what a reviewer or agent can
reason about as one unit.

Every active family carries:

- a concise comment for algorithm, layout, precision, launch, tail, and target
  assumptions;
- small nonzero `check.case` coverage, including tails and unusual scale values;
- representative `check.benchmark` rows for production shape buckets;
- compiler-report visibility for instruction mix, register pressure, LDS,
  private memory, spills, occupancy, memory requests, and wait behavior.

The layout is chosen from the consumer backward. A producer writing the
consumer's compact or transposed format removes an intermediate write, global
barrier, reread, and rewrite. Obvious pointwise initialization is fused into
the first consumer when the resulting register and arithmetic schedule remains
healthy. Coalescing, wide loads, and target matrix-fragment layout are
architectural concerns at authoring time, not cleanup after correctness.

## 7. Prove the Real Integration

Planning tests run without a device because the plan is a deterministic model
program output. They assert contracts such as valid lifetimes, no read before
initialization, legal aliases, closed option combinations, dynamic shape
selection, and stable resource accounting. Tests that assert an exact list of
private kernels or layer operations are change detectors and prevent fusion.

Preparation and issue tests are integration tests. They create devices through
the standard `--device=` provider, reuse the shared device across cases, compile
real checked-in Loom modules, load production-shaped parameter sources, record
real command buffers, submit them through the HAL, wait on the declared final
semaphore, and compare device results. Swapping fake executables into a cache
tests a different system and provides no confidence in this boundary.

Kernel-local math stays in `.loom` cases. A C/HAL integration test earns its
cost when it crosses the whole stage lifecycle or exercises an ownership,
synchronization, parameter-loading, or binding contract that kernel tooling
cannot observe.

## 8. Build Diagnostics Before Optimization

An effective diagnostic bundle answers four questions without rerunning the
model:

1. Which semantic tensor or parameter was this byte range?
2. Which specialization and launch configuration produced it?
3. Which queue, command buffer, binding table, and timeline edges made it ready?
4. Which compiler artifacts and reference tap explain the generated value or
   timing?

The production implementation therefore exposes plan JSON, named tensor taps,
parameter-layout metadata, memory-lifetime records, `loomc` artifacts, HAL queue
profiles, dispatch profiles, and final tensor summaries. Taps are planned
operations and become device copies only when selected. Diagnostic
region-per-dispatch and wait-after-segment modes deliberately change execution
and are labeled as fault-localization modes rather than performance paths.

The smoke harness creates one self-contained output directory with the request,
plan, final and progress images, tensor captures, compiler artifacts, queue
profile, summary, and validation result. That directory is a triage unit for a
human. It complements narrow numeric tests; it does not replace them.

## 9. Optimize from the Request Pareto

Optimization begins with a matched warm-serving baseline over realistic prompt
and resolution buckets. Initialization and compilation are measured separately
because a server amortizes them. Memory accounting includes resident allocator
bytes, pooled queue allocations, and an independent logical-plan high-water.

The iteration loop is:

1. Capture a full request profile and rank semantic kernel families by total
   time.
2. Extract the top family at exact production dimensions and use rotating
   device-local bindings.
3. Capture the reference framework's selected kernel, code object, generated
   source or solution metadata, and per-operation timing at the same shape.
4. Compare compiler reports and ISA: matrix instructions, global and LDS
   traffic, decode sequence, waits, register pressure, occupancy, spills, and
   epilogue structure.
5. Form a structural hypothesis, author the smallest meaningful variant, and
   gate correctness before timing.
6. Run candidate and baseline in interleaved ABABA order. Keep a candidate only
   when its timing or memory benefit survives alternation and its compiler
   report explains the mechanism.
7. Substitute demonstrated kernel gains into the request profile before paying
   for another full generation.
8. Rerun the full request once projected impact is material.

This process separates three comparisons that are otherwise easy to conflate.
A compact FP8 consumer may be slower than a pure BF16 GEMM, faster than the
reference's complete FP8-expand-scale-GEMM path, and still slower in the full
model because of cache rotation or neighboring work. All three are valid when
their operation boundaries are named.

Hardware calibration bounds the remaining work. Sustained read, write, copy,
special-function, and exact-shape matrix probes identify whether a family is
limited by bytes, math, reductions, register occupancy, or schedule quality.
Semantic binding bytes and compiler-issued requests remain diagnostic counts;
neither is automatically physical HBM traffic.

## 10. Make Compiler Gaps Standalone

A compiler failure is reduced to a `.loom` packet that does not require the
model application. The packet contains:

- a minimal source preserving the failing semantic and target facts;
- the exact tool command and target profile;
- expected and observed behavior;
- compiler diagnostics, report excerpts, and relevant generated IR or ISA;
- a known-good comparison when one exists;
- the production consequence stated without requiring ID4 knowledge.

Correctness failures and hangs take priority over performance tuning. Runtime
semantics are first checked for undefined buffer contents, missing final
semaphore waits, host-visible mapping in place of staged device buffers, false
FIFO assumptions, and missing barriers. Once the application contract is
sound, the reduced compiler reproducer becomes the handoff boundary.

Compiler feedback that proved broadly useful in this port included static
launch-config extraction, target facts with value ranges, dynamic edge-range
proofs, direct FP8 view loads, fragment liveness across loops, configurable loop
unroll schedules, barrier-aware fragment storage, and structured instruction,
resource, memory-request, and wait reports.

## 11. Contract after the Result Works

Bring-up naturally creates reference providers, rejected schedules, duplicate
shape variants, and diagnostic paths. Once correctness and performance gates
are stable, every embedded source receives one role: production selection,
active oracle, measured candidate, parameter encoder, or retired motif.
Unreachable modules leave production embedding and default tests. Correct
standalone motifs can remain in an explicit research corpus without competing
with the active implementation during search or maintenance.

Contraction preserves behavior through a sequence of narrow gates:

- source formatting and linking;
- reduced cases and production-shape integration tests;
- compiler resource reports and normalized ISA;
- ABABA timing at affected shapes;
- full model correctness, memory, and serving performance when the changed
  family is material to the request.

The endpoint is not one universal kernel file. It is the smallest readable set
of semantic families and measured target providers that reproduces the accepted
plans. That surface is what another model port should copy.

## Definition of a Reusable Port

A model port is ready to serve as a template when:

- arbitrary supported prompts and shapes flow through tokenization, planning,
  preparation, and issue without copied launch arithmetic or fixed tensor
  counts;
- all tensor-producing work between declared inputs and outputs runs on the
  device;
- stage boundaries correspond to user-visible pipeline units;
- compact static weights remain compact in resident memory;
- plans expose parameter, transient, boundary, and diagnostic lifetimes;
- prepared command buffers are immutable and reused;
- every queue order is explicit in semaphores or barriers;
- kernels have reduced correctness cases and production-shape benchmarks;
- major stage taps match the semantic oracle under a named precision policy;
- warm-serving latency and memory are measured against the reference across a
  shape matrix;
- the hot-kernel gap is explained by exact-shape compiler and reference
  artifacts rather than guesswork;
- diagnostics and standalone compiler reproducers make failures transferable.

The [ID4 case study](case_study.md) shows the measured result of applying this
workflow. The project [README](../README.md) remains the model-specific design
contract and source-reference map.
