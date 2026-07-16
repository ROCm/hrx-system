# Generalizing ID4 Target and Scheduling Specialization

The ID4 prototype establishes one complete, dynamically shaped model program on
`gfx1100`. The next step is not to clone that program for each GPU. It is to
make the model math, operation identities, and host API stable while the chosen
device controls kernel providers, launch policy, physical parameter layouts,
and eventually the schedule itself.

This document defines that boundary. The near-term work extends Loom's kernel
IR and embedding API enough to select high-fidelity implementations across
AMDGPU targets without changing model C. The longer work moves the semantic C
program into host scheduling IR so fusion, placement, memory planning, and
queue construction can specialize with the same target and request facts.

The [case study](case_study.md) records the measured `gfx1100` result. The
[porting guide](porting_guide.md) describes the model-authoring workflow. This
roadmap focuses only on the generalization pressure exposed by that work.

## Current Boundary

Several important decisions are already independent of the target:

- The C program expresses model math with semantic tensor handles, parameters,
  transient acquisition and release, dispatches, barriers, taps, and exports.
- Prompt length, image dimensions, precision policy, and model hyperparameters
  become configuration or workload values instead of copied launch geometry.
- A kernel is identified by a stable module path and export. Model C does not
  name code objects, compiler pipelines, or target instruction families.
- `kernel.launch.config` owns workgroup count and size arithmetic. The runtime
  evaluates it through `loomc`, including subgroup size and workgroup-local
  storage when the program proves them.
- A target-neutral `kernel.def` can call a semantic `func.apply`, while a
  target-qualified `func.template` supplies the implementation. Converting an
  existing kernel to this form does not change its C identity or dispatch site.
- The HAL device group, command buffers, indirect bindings, and timeline
  semaphores already separate model scheduling from one concrete driver.
- Ordinary runtime initialization selects the exact AMDGPU executable target
  from the primary HAL device specification and uses its processor to create
  the Loom target profile. The CLI and model policy contain no processor
  default.

The remaining target coupling is concentrated but consequential:

- The kernel and executable caches still describe only the primary device. A
  heterogeneous group needs a target selection and cache partition for each
  distinct placement class.
- Hot matrix and attention providers encode `gfx1100` wave, fragment, staging,
  and instruction assumptions in target-qualified Loom implementations.
- Baked parameter archives contain provider-specific physical layouts, but
  archive selection is not yet driven by a general target layout contract.
- The C program still chooses the cross-dispatch graph, fusion boundaries,
  transient representations, device placement, and stage queue structure.

The first three belong in target specialization. The last belongs in host
scheduling IR.

## Generalization Contract

The semantic model program should be identical for every supported target. A
dispatch requests an operation, shapes, dtypes, precision behavior, and logical
layouts. Target selection resolves that request to one implementation or fails
with a diagnostic. It does not silently change model semantics.

| Concern | Durable owner |
| --- | --- |
| Model equations, repeated layers, and stage boundaries | Semantic model program |
| Operation ABI, shape facts, dtype policy, and tolerance contract | Target-neutral kernel declaration |
| Matrix instruction, wave shape, lane mapping, staging, and swizzle | Target-qualified provider |
| Workgroup count, workgroup size, subgroup size, and local storage | Target-aware launch-config region |
| Compact weight packing and scale representation | Provider-declared execution-layout contract |
| Transient lifetimes, fusion, placement, and queue topology | Host scheduling program |
| Allocation, execution, ordering, and profiling | IREE HAL |

Target specialization may select a different instruction sequence, tile shape,
or compact weight encoding while preserving the declared mathematical and
numeric contract. A change from BF16 matrix inputs to native FP8 matrix inputs
is not automatically such an implementation detail: activation quantization,
scale granularity, and rounding can change the model. Those choices remain
explicit precision policy unless the provider proves equivalence within the
declared tolerance.

## Near-Term Target Specialization

### Device-Derived Target Selection

The selected HAL device exposes executable targets through its device
specification. ID4 now translates the exact AMDGPU target's processor into a
Loom target profile and uses that selection for compilation and launch
evaluation. The next step is to carry the complete executable-target identity
through execution-layout validation and persistent cache identity instead of
reducing it to a processor string after initialization.

For a homogeneous device group, each participating device normally shares the
same compiled artifacts and layout class. A heterogeneous group may hold a
different target selection and cache partition per device. The model program
still sees device ordinals and placement policy rather than processor names.

This removes the architecture flag from ordinary model execution. An explicit
target override remains useful for offline compilation and diagnostics, but it
is a deliberate override of discovered device facts rather than a default the
application must know.

### Foldable Target Queries

Launch and provider code needs a small vocabulary for target facts. These are
ordinary SSA-producing operations scoped to the enclosing selected target, not
symbol references and not compiler-report records. They participate in value
fact analysis and fold once the target is concrete.

Generic query families cover source-level concepts shared by targets:

```loom
%subgroup_size = target.subgroup.limit<default_size> : index
%max_workgroup_size = target.workgroup.limit<max_flat_size> : index
%local_memory_limit =
    target.workgroup.limit<max_local_memory_bytes> : index
```

Target-family queries cover behavior that is meaningful only within that
family:

```loom
%has_bf16_matrix = amdgpu.target.feature<matrix_bf16> : i1
%has_native_fp8 = amdgpu.target.feature<native_fp8_e4m3> : i1
%supports_wave32 = amdgpu.target.feature<wave32> : i1
```

An exact architecture query is an escape hatch for a hardware erratum or a
diagnostic. Provider selection should normally ask for capabilities. This lets
a later processor reuse a provider when it satisfies the same contract and
keeps authored IR from becoming a switch over product names.

Value facts make the queries useful before exact target selection. A subgroup
size can carry positive, bounded, and power-of-two facts; workgroup limits can
carry conservative ranges; feature queries can remain unknown booleans. Exact
selection sharpens those facts to constants and allows ordinary folding,
unrolling, layout analysis, and launch evaluation to remove untaken paths.

### Capability-Qualified Providers

A semantic `func.apply` should resolve against providers that declare required
capabilities, supported type and shape facts, and priority. Target names remain
valid on a provider when the implementation is truly processor-specific, but
the common path is a capability set such as:

- fixed wave32 plus BF16 WMMA;
- wave64 plus BF16 MFMA;
- native E4M3/BF16 matrix inputs with F32 accumulation;
- software E4M3 decode plus BF16 matrix inputs;
- a local-memory and register budget sufficient for a particular attention
  schedule.

Provider resolution is closed. Exactly one best implementation is selected for
the specialized call, or compilation fails with the unsatisfied requirements.
The application does not retry a different architecture path or maintain a
second target dispatch table.

Configuration values must remain visible through provider inlining. A semantic
provider that receives static model dimensions, dtype policy, and tile choices
should expose those values to folding just as if the implementation had read
the configuration directly. This is necessary for reusable matrix providers;
otherwise each model has to restate an implementation merely to recover
constants the compiler already knows.

### Target-Aware Launch Configuration

The existing launch evaluator already returns workgroup count, workgroup size,
subgroup size, and workgroup-local storage through `loomc`. Target queries let
the same launch region choose geometry from the selected target instead of
embedding wave size or local-memory assumptions.

Launch evaluation should be memoized by the semantic kernel, resolved config,
target selection, and workload arguments. The evaluator is an exact oracle,
not copied arithmetic in C, so a provider or launch-policy change cannot leave
the host and kernel disagreeing.

The launch contract remains independent of the HAL. Counts, local sizes,
subgroup size, and required local storage are properties of a compiled kernel.
The HAL consumes those properties when it prepares and dispatches an
executable, but Loom tooling and non-HAL embedders can evaluate them as well.

### Matrix and Layout Contracts

Matrix operations need semantic contracts above target fragments. A contract
describes input and accumulator types, logical matrix shape, transpose state,
scale semantics, and result layout. The provider owns the mapping to WMMA,
MFMA, native FP8 instructions, or a software-unpack schedule.

The same separation applies to static parameters. A provider declares the
physical execution layout required by its selected implementation. Preparation
then either:

- gathers a matching baked archive entry directly into its final slab;
- gathers source checkpoint data through a bounded encoder window and writes
  the final layout; or
- rejects an archive whose schema or source fingerprint does not match.

Execution-layout identity includes the semantic operation contract, target
provider class, physical dtype, shape, encoding, alignment, scale layout, and a
versioned schema. It does not become another HAL binding identity. Multiple
layouts may occupy subranges of one semantically coherent parameter slab.

### Bytecode Libraries and Cache Identity

Stable module paths should resolve through the same library interface whether
development supplies editable `.loom` sources, a package embeds source bytes,
or deployment supplies linked `.loombc` archives. Model C names the semantic
module and export; it does not change when source modules are linked together or
when target providers move into a separate archive.

A deployable library contains target-neutral declarations, reusable semantic
families, and the provider set intended for its supported targets. The JIT
links and compiles only the transitive providers selected by the model's
specializations. Detailed reports and diagnostic bundles remain optional
observations of that production compile rather than a second compilation path.

Executable cache identity includes bytecode or source identity, semantic
module and export, resolved config and workload facts, exact target selection,
compiler pipeline identity, and physical execution-layout schema. Persistent
caches may serialize the executable and launch contract, while interactive
development can invalidate entries when a VFS source changes. Neither case
requires the model program to know whether its kernel came from live source or
an archive.

## Cross-Target Bring-Up

The next two target classes exercise complementary parts of the design.

`gfx942` changes the matrix and wave regime. It has wave64/MFMA behavior and
native FP8 matrix capabilities that should replace `gfx1100` software decode
inside the same semantic FP8 contractions. The expected work is concentrated
in target providers, launch choices, and execution layouts. Model C and stage
structure should not change.

Newer `gfx12` devices retain a wave32-oriented programming shape but add their
own matrix and FP8 capabilities. They test whether provider selection is truly
capability-based instead of an indirect `gfx1100` clone.

Each target follows the same gates:

1. Derive the exact target from the HAL device and compile every active semantic
   kernel without model-C architecture branches.
2. Run reduced target-independent correctness cases through the selected
   providers.
3. Validate major stage taps against the same oracle and precision policy.
4. Build target-qualified execution-layout archives and verify their schema and
   residency accounting.
5. Measure warm stage and whole-request performance, memory, compiler resources,
   and hardware calibration on that target.
6. Contract duplicate providers only after the shared form preserves both
   targets' correctness and performance.

An explicitly selected scalar device provider can establish a temporary
correctness lane during target bring-up, but it is not evidence of target
support and is never a silent fallback. A supported target has measured
providers for the request-dominating operation families and an explained gap
to its hardware or library oracle.

## The Boundary of Kernel IR

Kernel specialization can hide instruction selection, vector width, unroll
schedule, local staging, launch geometry, and physical parameter layout. It
cannot cleanly hide a target choosing a different graph of dispatches. Three
decisions still force model-C changes today:

- fusing or splitting operations across kernel boundaries;
- changing transient representations shared by several dispatches;
- placing work across multiple devices and introducing exchange edges.

Encoding those decisions as more C target branches would make the current
prototype the permanent architecture. They instead define the scope of host
scheduling IR.

## Host Scheduling IR

The semantic C program is intentionally a prototype for this IR. Its useful
properties should survive the move: the code reads like a forward pass,
repeated layers are loops, tensor lifetimes are explicit, planning can run
without a device allocation, and HAL recording is a lowering rather than a
second handwritten model.

The initial scheduling dialect needs values and operations for:

- typed tensor imports, parameters, constants, subviews, and exports;
- uninitialized transient acquire and explicit last-use release;
- semantic Loom dispatches with read and write footprints;
- visibility barriers and lifetime epochs without implied FIFO ordering;
- named diagnostic taps;
- coarse stages and request-dependent loops;
- device-group placement and cross-device tensor exchange;
- parameter residency, execution-layout requirements, and bounded preparation;
- wait and signal timeline edges around preparation and issue.

A scheduling function consumes model configuration, request shapes, selected
devices, precision policy, and memory policy. Planning evaluates that function
to produce tensor lifetimes, slab layouts, parameter requirements, kernel
specializations, and a queue graph. Preparation compiles the selected kernels,
materializes target layouts, and records immutable HAL command buffers. Issue
binds request storage and submits the prepared graph with caller-owned timeline
edges.

Once represented in IR, target-aware transformations can change the dispatch
graph itself. They can select online or materialized attention, fuse producer
layouts into consumers, choose a tiled VAE under a memory budget, shard a
contraction across devices, or retain an intermediate when recomputation is
more expensive than memory. Those choices can be evaluated by plan statistics
before execution and lowered into a small number of reusable command buffers.

The IREE VM is the natural evaluator for dynamic planning and launch functions.
The shipping C surface remains responsible for session lifetime, tokenizer
inputs, parameter providers, device groups, outer request control, diagnostic
sinks, and HAL submission. It loads a compiled scheduling module instead of
embedding a model-specific scheduler.

## Migration Sequence

The migration should preserve one authoritative implementation at every step:

1. Treat the current C program operations and plan output as the executable
   behavioral specification for scheduling IR.
2. Define target-neutral scheduling operations with the same initialization,
   lifetime, aliasing, barrier, and semaphore contracts.
3. Lower one coarse stage from scheduling IR through the existing plan,
   preparation, and HAL recording machinery. Offline comparison may evaluate
   both forms, but production selects one form explicitly.
4. Move repeated model structure and request-dependent control into VM-executed
   scheduling functions.
5. Add target and memory-policy transforms that alter fusion, layout, and
   placement while retaining the semantic tap boundaries used by the oracle.
6. Remove the model-specific C builder after every stage is represented and the
   C API, diagnostics, memory, correctness, and warm-performance gates match.

This sequence avoids inventing a generic graph format disconnected from the
working runtime. The dialect is extracted from operations that already plan,
prepare, record, diagnose, and execute a complete model.

## Completion Criteria

The near-term target work is complete when:

- ordinary execution derives its Loom target from every selected HAL device;
- model C contains no processor-name or matrix-instruction selection;
- semantic kernel paths and exports remain stable across supported targets;
- launch geometry and local storage come only from Loom launch evaluation;
- provider resolution uses target facts and reports unsatisfied capabilities;
- execution-layout archives are selected and validated by target layout
  contracts;
- source and linked `.loombc` libraries resolve the same semantic module paths,
  with target and config facts represented in cache identity;
- `gfx1100`, `gfx942`, and a `gfx12` class run the same reduced cases and model
  taps under explicit precision policies;
- each target has warm performance, memory, compiler-resource, and calibration
  evidence for its request-dominating providers.

The scheduling work is complete when the model-specific C forward builders are
replaced by compiled scheduling functions, dry planning remains available,
target and memory policy can change the dispatch graph without C edits, and the
same prepared HAL lifecycle serves CLI, benchmark, and server callers.
