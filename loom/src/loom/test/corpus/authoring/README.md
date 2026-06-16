# Loom Authoring Corpus

This corpus is the checked reference shape for hand-authored model and kernel
Loom. The files are ordinary `.loom` programs a model-porting agent should be
able to read, copy from, and extend: helper decomposition, provider selection,
local transform intent, correctness policy, and benchmark rows live in source
without a Python script generating the source itself.

The examples are tested through production-facing tools:

- `iree-benchmark-loom --dry-run` proves `check.case` and `check.benchmark`
  planning without requiring a local GPU during host-only CI.
- `iree-benchmark-loom --device=amdgpu --measure=dispatch_complete` compiles,
  executes correctness samples, and benchmarks the same sources on AMDGPU test
  hosts.

The timing flags used by the Bazel AMDGPU smoke targets are harness policy. The
source files name workloads and correctness expectations; iteration counts,
warmups, profiling, compile-time measurement, soak runs, and quick proof runs
belong to `iree-benchmark-loom` flags or embedding APIs.

## Source Map

| Pattern | Checked source |
| --- | --- |
| Concrete helper call | `ffn_gate_up_swiglu_q6q8.loom` calls `@q6_signed_pack_dot4i` with `func.call` because it is exact bit manipulation. |
| Dynamic extent byte fill | `memset_i8.loom` keeps a 64-bit pattern argument and narrows it at the byte store, matching runtime-style ABI pressure without runtime-specific names. |
| Loaded row-index gather | `indexed_row_gather_f32.loom` guards loaded i32 row ids before `index.cast` and uses real dynamic view extents instead of sentinel shapes. |
| Template provider selection | `ffn_gate_up_swiglu_q6q8.loom` applies `model.q6q8.accumulate_part` so libraries can provide alternate packed-dot implementations. |
| Local unroll intent | `ffn_gate_up_swiglu_q6q8.loom` keeps block/part loops structured and marks the tiny trip-count loops with `unroll`. |
| Logical indexing | The examples use index/view math for logical rows, blocks, lanes, byte positions, and dense tensor coordinates. |
| Dynamic case parameters | `mlp_down_projection_residual_bf16.loom` names `rows` on a `check.param.choice` and threads it through shapes, launch geometry, and the kernel ABI. |
| Benchmark slices | `mlp_down_projection_residual_bf16.loom` has an anonymous full sweep plus named decode/full rows with assignment dictionaries. |
| HIP C++ porting motifs | `hip/README.md` maps HIP/CUDA kernel habits to Loom source spellings, proof commands, diagnostics, and authoring-level report workflows. |
| Packed field contracts | `hip/packed_field_contracts.loom` shows q2/q3/q4/q5/q6-style fields as explicit storage/decode/repack contracts instead of fake scalar element types. |
| HIP shared memory tile | `hip/shared_memory_tile.loom` stages a 64-lane i32 tile through workgroup memory, synchronizes, and reads a reversed lane so correctness depends on LDS traffic. |
| HIP shared memory transpose | `hip/shared_memory_transpose.loom` stages an 8x8 i32 tile through two workgroup allocations, synchronizes twice, and validates x/y cross-axis LDS traffic. |
| HIP vector-width shared memory | `hip/shared_memory_vector_tile.loom` stages one `vector<4xi32>` row per work item through workgroup memory and proves the 128-bit LDS path with compile evidence. |

## FFN q6/q8 Gate-Up SwiGLU

`ffn_gate_up_swiglu_q6q8.loom` models a q6_K-weight by q8_1 activation gate/up
fusion. The file keeps the model structure visible: launch topology, q8 input
views, q6 gate/up weight views, shared q8 load, gate/up accumulation, subgroup
reduction, SiLU, and final store.

The q6 sign-pack helper is a direct `func.call` because the call site wants that
specific bit helper. The q6/q8 part accumulator is a `func.template` provider
for the `model.q6q8.accumulate_part` contract and the kernel uses
`func.apply<model.q6q8.accumulate_part>`. Selection rewrites the apply to an
inline call to the selected provider, then normal callable inlining removes the
boundary before executable lowering. That is the intended library shape for
layout, target, or algorithm families: the model kernel asks for a contract,
while libraries own provider symbols and selection predicates.

The block/part loops are source-level transform intent, not source expansion.
They carry gate/up accumulators as loop results and request local unrolling only
where the q6_K trip counts are tiny and compile-time known. This keeps the
logical reduction visible to analysis while still producing the expanded low
code expected by the backend.

The `hip/` cookbook is the fast path for users arriving from hand-written HIP
C++, CUDA, or inline assembly. It is organized by source terms such as
`#pragma unroll`, `threadIdx`, `global_load_b128`, and `restrict`, then points
to the Loom spelling, the pass or lowering stage that consumes it, and the
structured diagnostic/report query that proves what happened.

The zero case is an execution smoke test: zero weights and activations make the
expected tensor simple, while the dispatch still exercises unpack, dot, scale,
reduction, SiLU, and store. Higher-fidelity math oracles belong in the external
fixture/reference layer when the expected values are too large or too expensive
to express inline.

### Quantized AMDGPU Command Flow

Start with the host-only planner when editing source shape, check parameters,
or benchmark rows:

```bash
python dev.py bazel run //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom -- \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --dry-run \
  --output=/tmp/loom-q6q8-plan.json
```

`--dry-run` parses, verifies, and plans `check.case`/`check.benchmark`
workloads. It does not compile, allocate device buffers, run correctness, or
measure timing, so it is the cheapest way to catch source and benchmark
selection mistakes. Inspect `summary`, `work_items`, `benchmarks`, `failures`,
and `failed_samples` in the output JSON.

Compile the same authored file to an AMDGPU HAL executable plus a native HSACO
sidecar when validating target lowering and packaging:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-compile:loom-compile -- \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=/tmp/loom-q6q8.vmfb \
  --emit-target-artifact=/tmp/loom-q6q8.hsaco \
  --artifact-manifest=summary \
  --emit-artifact-manifest=/tmp/loom-q6q8.manifest.json \
  --compile-report=summary \
  --compile-report-output=/tmp/loom-q6q8.compile-report.json
```

`--target=gfx1100` is invocation target selection. The source kernel stays
targetless, template providers are resolved against the effective target, and
target-sensitive passes see that same selected target. A successful summary
manifest for this kernel reports one `ffn_gate_up_swiglu_q6q8` function, four
parameters/bindings, zero constant bytes, workgroup size `[512,1,1]`, and
subgroup size `32`.

The artifact manifest describes the emitted artifact contract. The compile
report describes compiler evidence for the invocation: status, selected backend
and target bundle, schedule size, register pressure, instruction mix, spills,
emitted code bytes, and memory summaries. Useful first inspections are:

```bash
jq '{artifact, targets, functions}' /tmp/loom-q6q8.manifest.json
jq '{status, target_key, target_bundle, target_export, spills:.allocation.spill_count, code_bytes:.emission.code_byte_count, dots:.static_instruction_mix.dot_count}' \
  /tmp/loom-q6q8.compile-report.json
```

When provider selection, inlining, or math legalization is suspect, capture IR
snapshots around those boundaries:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-compile:loom-compile -- \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=/tmp/loom-q6q8.vmfb \
  --emit-target-artifact=/tmp/loom-q6q8.hsaco \
  --dump-ir-after=select-templates \
  --dump-ir-after=inline-callables \
  --dump-ir-after=legalize-math \
  --dump-ir-format=jsonl \
  --dump-ir-output=/tmp/loom-q6q8-trace/
```

The JSONL trace is the scriptable index. The adjacent `.loom` snapshots are the
human-readable IR. `select-templates` should remove residual
`func.apply<model.q6q8.accumulate_part>` sites, `inline-callables` should remove
the selected provider boundary, and `legalize-math` should rewrite semantic
SiLU before target-low emission.

For pass-pipeline debugging outside the full artifact path, `loom-opt` can emit
a structured pass report:

```bash
loom-opt --pass=select-templates --pass-report=json input.loom \
  --output=/tmp/selected.loom
```

The `select-templates` report statistics include `apply-sites`,
`selected-sites`, `fallback-selected-sites`, `no-provider-sites`,
`target-mismatch-sites`, `rejected-sites`, `missing-fact-sites`,
`ambiguous-sites`, and `materialization-blocked-sites`. That report answers the
first triage question quickly: whether provider selection picked a lower
priority fallback, failed before selection, or left unresolved applies because
more predicate facts are needed.

The same report includes one `template-selection` detail row per analyzed
`func.apply` site when pass reporting is enabled. The row records the enclosing
function, contract key, selected provider when present, effective target when
known, candidate counts, and an outcome such as `selected`,
`fallback_selected`, `target_mismatch`, `missing_facts`, or `ambiguous`.

```bash
loom-opt --pass=select-templates --pass-report=json input.loom \
  --output=/tmp/selected.loom \
  2>/tmp/pass-report.json

jq '.invocations[]
  | select(.pass == "select-templates")
  | .details[]
  | select(.category == "template-selection")' /tmp/pass-report.json

jq '.invocations[]
  | select(.pass == "select-templates")
  | .details[]
  | select(.outcome != "selected")' /tmp/pass-report.json
```

Full artifact runs should still use `loom-compile --dump-ir-*`, artifact
manifests, and compile reports for target emission evidence.

Standalone `loom-compile` emits the artifact, manifest, and compile report. To
keep target-owned assembly/listing text with benchmark evidence, run
`iree-benchmark-loom` with a debug or full artifact bundle on an AMDGPU-capable
build:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null \
  //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom -- \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --device=amdgpu \
  --measure=dispatch_complete \
  --iterations=1 \
  --warmup-iterations=0 \
  --batch-size=1 \
  --min-time-ms=0 \
  --max-batches=1 \
  --input-ring-count=1 \
  --artifact-bundle-dir=/tmp/loom-q6q8-run \
  --artifact-bundle-policy=debug \
  --artifact-manifest=summary \
  --output-format=jsonl
```

Debug/full bundles keep per-candidate compile reports under
`compile_reports/`, artifact manifests under `artifact_manifests/`, native
artifacts under `target_artifacts/`, target-owned listings under
`target_listings/`, and HAL executable artifacts under `hal_executables/`.
JSONL compile rows link those files with
`compile_report_path`, `artifact_manifest_path`, `target_artifact_path`,
`target_listing_path`, and `hal_executable_path`; benchmark rows expose the
same links under `benchmark_result`. Rows with embedded reports use the same
`compile_report` JSON tree as standalone `loom-compile`, so jq recipes can move
between tools without changing the report fields:

```bash
jq 'select(.row=="compile" and .compile_report) |
  {candidate_id,
   code:.compile_report.emission.code_byte_count,
   spills:.compile_report.allocation.spill_count,
   local:.compile_report.memory.local_bytes,
   pressure:.compile_report.schedule.register_pressure_peak_live_units}' \
  /tmp/loom-q6q8-run/results.jsonl

jq 'select(.row=="benchmark" and .benchmark_result.compile_report) |
  .benchmark_result |
  {benchmark,
   code:.compile_report.emission.code_byte_count,
   spills:.compile_report.allocation.spill_count,
   local:.compile_report.memory.local_bytes}' \
  /tmp/loom-q6q8-run/results.jsonl
```

For quick object-level disassembly of a standalone HSACO, use the LLVM object
tools on the emitted sidecar:

```bash
llvm-objdump -d --mcpu=gfx1100 /tmp/loom-q6q8.hsaco
```

Treat the evidence channels separately. Planner output answers "what would run?"
Compile reports answer "what did the compiler emit?" Artifact manifests answer
"what does this loader-ready artifact contain?" `dispatch_complete` benchmark
rows answer "how long did completed HAL dispatch work take?" The quick command
above intentionally uses one hot-reuse input ring and tiny iteration counts for
smoke coverage; serious timing should use larger batches, warmups, a stable
minimum duration, and enough input-ring bytes to avoid measuring only cache-hot
data reuse.

## AMDGPU Shared-Memory Feedback

The AMDGPU compile report can explain selected workgroup-memory packets and the
static LDS bank pattern visible from the source facts. This is structural
compiler feedback, not a runtime performance verdict: final tuning still needs
measurements and profiler counters when occupancy, cache behavior, scheduling,
or data-dependent control flow dominates. The useful shift is that a source
author can see the selected packet, stride facts, and bank-conflict
classification while the source layout is still in view.

Compile with detailed reports when investigating shared-memory layout,
padding, swizzling, vectorization, or imported kernel staging choices:

```bash
loom-compile loom/src/loom/test/corpus/authoring/hip/shared_memory_vector_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=/tmp/shared-memory-vector-tile.vmfb \
  --compile-report=json-details \
  --compile-report-output=/tmp/shared-memory-vector-tile.compile-report.json
```

The `source_low.memory_rows` array records one selected memory-packet row per
reported source memory operation:

```bash
jq '.source_low.memory_rows[]?
  | {function, source_op, operation, packet, vector_lanes,
     dynamic_stride_bytes, vector_lane_stride_bytes,
     bank_stride_words, bank_conflict_degree, bank_conflict_kind}' \
  /tmp/shared-memory-vector-tile.compile-report.json
```

The text form carries the same fields as `source_low_memory[...]` rows when a
greppable report is more convenient:

```bash
loom-compile loom/src/loom/test/corpus/authoring/hip/shared_memory_vector_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --output=/tmp/shared-memory-vector-tile.vmfb \
  --compile-report=text-details \
  --compile-report-output=/tmp/shared-memory-vector-tile.compile-report.txt

rg 'source_low_memory|bank_conflict|ds_' \
  /tmp/shared-memory-vector-tile.compile-report.txt
```

The classification key is a compact summary of the facts Loom could prove for
the selected AMDGPU LDS packet:

| Classification | Meaning |
| --- | --- |
| `bank-conflict-free` | Static facts prove conflict degree one for the selected target bank geometry. |
| `padded-bank-conflict-free` | The access is conflict-free and the stride exposes visible padding beyond the packet footprint. |
| `bank-conflict-risk` | Static facts imply a conflict degree greater than one; padding, swizzling, or a different staging shape deserves investigation. |
| `bank-pattern-unknown` | Loom selected a workgroup-memory packet but the current facts do not prove one lane-to-bank pattern. |

The report should be read before object-level profiling when the question is
whether the source layout and selected packet shape make sense. Runtime tools
such as `rocprof` and Nsight still decide whether the complete kernel is fast,
but they should not be the first place a source author learns that a static LDS
access pattern is structurally suspicious.

## Memset i8

`memset_i8.loom` is the minimal byte-fill reference for dynamic extent kernels.
It uses ordinary launch geometry and a guarded store for the tail workgroup. The
pattern is intentionally an `i64` launch argument even though the stored element
is `i8`; runtime and embedding ABIs often widen small scalar payloads, and the
authored kernel should express the narrowing with `scalar.trunci` instead of
requiring a source generator or target-specific ABI hook.

The case sweeps a partial workgroup, an exact workgroup boundary, and a
multi-workgroup tail. Its expected tensor uses the low byte of the wide pattern,
so host dry-run and AMDGPU execution both keep the 64-bit-to-byte path visible.
The named benchmark rows make those three shapes easy to select independently
when debugging launch geometry or store lowering.

## MLP Down-Projection Residual

`mlp_down_projection_residual_bf16.loom` keeps one down-projection kernel with a
residual add and a named `rows` parameter. The parameter drives the case tensor
shapes, the scalar kernel argument, dynamic buffer views, and dispatch geometry,
so one authored source covers both a two-row decode-shaped sample and the full
projection.

The anonymous `check.benchmark<@mlp_down_projection_residual_case>` sweeps all
case samples and receives generated benchmark names. The named benchmark rows
pin specific samples with assignment dictionaries:

```loom
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_decode {rows = 2}
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_full {rows = 3584}
```

The case uses deterministic iota inputs and zero projection weights for a
residual-preservation oracle. The AMDGPU dispatch test runs this file with
per-sample compilation so each selected row count becomes a compile-time fact
before launch geometry and memory legality are finalized.

## Authoring Rules

`func.call` is an exact symbol reference. It is the right spelling when the
caller wants one specific helper or declaration, as with
`@q6_signed_pack_dot4i` and `@bf16_dot32`.

`func.apply<K>` is a compile-time implementation demand. The key is a contract,
not a symbol. The selection pass resolves live applies against visible
`func.template` providers, prunes dead private providers, rewrites selected
sites to inline calls, and leaves normal inlining to splice the body. Good
contract names describe a reusable motif or layout operation rather than a
particular model brand.

`inline` and `noinline` describe callable-boundary intent. `inline` is useful
for private helpers that should disappear before target-sensitive lowering.
`noinline` preserves a real callable boundary and therefore needs a target-ready
symbol when target lowering reaches it. `hot` and `cold` are separate
temperature hints for cost models or profile feedback; they are not substitutes
for authored inline policy.

Facts are source facts, not global flags. Put reusable facts at the boundary
that owns them: `config.decl` and function/kernel signatures for shape choices,
`kernel.launch.config` for launch topology, kernel ABI buffer arguments for
global memory space, and checked samples for benchmark-specific values. Local
assumes are for facts discovered inside the body, such as guarded row IDs,
clamped values, or dynamic alignment proof. They should not reassert config
declarations, launch dimensions, or kernel buffer memory space.

Use `index` for logical coordinates, extents, and tensor/view indices. Use
`offset` for byte offsets and byte strides. Views should carry real extents
from the source contract; large sentinel shapes that only make proofs pass
destroy bounds-checking and sanitizer value because the compiler can no longer
see the real accessible range.

When a logical coordinate selects a packed byte window, `index.scale` is the
explicit boundary: it multiplies an `index` coordinate by an `offset` byte
stride and produces the `offset` value expected by `buffer.view`.

The authoring source linter keeps this reference surface aligned with those
rules. It rejects redundant kernel-buffer memory-space assumes, sentinel-sized
views, late `index.cast` byte-address conversions, and ggml-style `nb*` byte
strides typed as `index`, because agents copy examples before they read design
notes.

`check.case` owns correctness policy for a workload. It creates inputs, calls
the unit under test, and states expectations. `check.benchmark<@case>` selects
which case samples should be timed. Benchmark rows name workloads; the runner
chooses timing rigor, output format, profiling, compile-time measurement, and
batching.

Python, C, and C++ remain appropriate for oracle code, external comparison
harnesses, fixture extraction, and binary fixture preparation. They are outside
the authored Loom source contract unless they are producing data consumed by a
checked `.loom` case.

## Failure Signals

| Signal | Mechanism to inspect |
| --- | --- |
| Residual `func.apply` after final selection | No provider implemented the contract, every provider was rejected by signature or predicates, or multiple highest-priority providers tied. |
| Template ambiguity | Matching providers need distinct priorities, sharper predicates, or separate contract keys. |
| Unresolved unroll intent | The loop trip count or requested factor was not known where the unroller ran; add facts earlier or leave the loop structured. |
| Inline/noinline conflict | Caller and callee policy disagree about whether the boundary may survive lowering. Fix the authored policy instead of relying on pass ordering. |
| Targetless helper reached target lowering | The pipeline missed callable specialization or inlining for the selected call graph. Adding target attributes to every reusable helper is the wrong source shape. |
| Benchmark parameter mismatch | A benchmark assignment dictionary referenced a name that is not a named `check.param` in the selected case. |

## Library-Scale Pressure

The q6/q8 accumulator intentionally exposes a wide signature: today it is the
honest Loom spelling for the view bundle and derived lane/block coordinates the
kernel needs. If several kernels repeat the same bundle, the pressure is for a
real representation primitive or a sharper helper boundary, not a source
generator.

The benchmark style scales by absence. A model library can carry many
`check.benchmark<@case>` rows because each row is just a workload selection.
Repeated timing dictionaries, profiling flags, and per-row harness policy would
make the source noisy and would couple authored IR to one command-line tool.
