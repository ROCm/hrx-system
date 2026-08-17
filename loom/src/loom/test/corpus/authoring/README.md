# Loom Authoring Corpus

This corpus is the checked reference shape for hand-authored model and kernel
Loom. The files are ordinary `.loom` programs a model-porting agent should be
able to read, copy from, and extend: helper decomposition, provider selection,
local transform intent, correctness policy, and benchmark rows live in source
without a Python script generating the source itself.

Backend qualification matrices are not authoring examples. Exhaustive numeric
edges, target sweeps, and instruction-selection cases live with the backend or
integration contract they test even when the fixture itself is written in Loom.

The examples are tested through production-facing tools:

- `iree-test-loom --device=<device>` compiles and executes `check.case`
  correctness samples through the selected target provider.
- `iree-benchmark-loom --device=<device>` reuses the same checked cases for
  deliberate performance experiments with queue and device timing.

Target providers compose these source modules into physical execution suites;
the corpus itself does not choose a device. Iteration counts, warmups,
profiling, compile-time measurement, and soak runs belong to explicit
`iree-benchmark-loom` invocations or embedding APIs.

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
specific bit helper. The q6/q8 part accumulator is a `template.def` provider
for the `model.q6q8.accumulate_part` contract and the kernel uses
`template.apply<@model.q6q8.accumulate_part>`. Selection rewrites the apply to an
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
iree-benchmark-loom \
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
loom-compile \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/loom-q6q8.vmfb \
  --emit-target-artifact=/tmp/loom-q6q8.hsaco \
  --artifact-manifest=summary \
  --emit-artifact-manifest=/tmp/loom-q6q8.manifest.json \
  --compile-report=summary \
  --compile-report-output=/tmp/loom-q6q8.compile-report.json
```

`--target=gfx11-generic` specializes the kernel function for this compile
invocation; it does not establish a module-global target. Template providers
and target-sensitive passes resolve the durable target written onto that
function, while other functions in a multi-target module remain unchanged. A
successful summary manifest for this kernel reports one
`ffn_gate_up_swiglu_q6q8` function, four parameters/bindings, zero constant
bytes, workgroup size `[512,1,1]`, and subgroup size `32`.

The artifact manifest describes the emitted artifact contract. The compile
report describes compiler evidence for the invocation: status, selected backend
and target bundle, schedule size, register pressure, instruction mix, spills,
emitted code bytes, and memory summaries. Start with the bounded report views:

```bash
loom-compile-report show /tmp/loom-q6q8.compile-report.json
loom-compile-report suggest /tmp/loom-q6q8.compile-report.json
loom-compile-report \
  diff /tmp/baseline.compile-report.json \
       /tmp/loom-q6q8.compile-report.json --format=json
```

The JSON views are sparse so they can feed an autoresearch loop without
replaying the full report. Omitted metrics are unavailable, not zero. `diff`
requires exact schema, target, config, workload, and entry identity by default.
`--force` permits an explicitly observational comparison only when each report
contains one entry; it preserves every identity mismatch instead of pretending
the inputs describe one controlled experiment. To compare two target
specializations of the same source, config, workload, artifact family, and
target family, select the bounded target comparison explicitly:

```bash
loom-compile-report \
  diff /tmp/gfx1100.compile-report.json \
       /tmp/gfx1151.compile-report.json --comparison=target
```

That mode permits only target key, bundle, snapshot, and target configuration
identity to vary and renders both specializations. All other identity remains
strict. `suggest` uses only explicitly registered target providers and reports
unavailable target interpretation instead of guessing from bundle names.
Findings carry an evidence tier. The default output contains only target models
backed by public documentation or silicon calibration;
`--include-experimental` additionally admits exact compiler proofs against
hardware-unvalidated models. That opt-in is useful for pre-silicon search, but
hardware timing still decides whether a candidate is adopted.

Version-zero reports are ephemeral diagnostics co-versioned with the compiler.
Regenerate them after changing compiler versions. Use the full report and
manifest for deeper source and packet attribution:

```bash
jq '{artifact, targets, functions}' /tmp/loom-q6q8.manifest.json
jq '{status, target_key, target_bundle, target_export,
     planned_spills:.allocation.spill_count,
     materialized_spill_storage:.allocation.materialized_spill_storage_count,
     materialized_spill_stores:.allocation.materialized_spill_store_count,
     materialized_reloads:.allocation.materialized_reload_count,
     private:.memory.private_bytes,
     final_vgprs:.target_resources.vector.final.register_count,
     scheduled_vgpr_pressure:
       .target_resources.vector.scheduled_pressure.peak_live_units,
     code_bytes:.emission.code_byte_count,
     matrix:.static_instruction_mix.matrix_count,
     wmma:.static_instruction_mix.wmma_count,
     mfma:.static_instruction_mix.mfma_count,
     dots:.static_instruction_mix.dot_count}' \
  /tmp/loom-q6q8.compile-report.json
```

`target_resources.{scalar,vector}.final.register_count` records final target
metadata used for occupancy. The adjacent
`target_resources.{scalar,vector}.scheduled_pressure.peak_live_units` value is
scheduled virtual pressure before final allocation metadata, so the two numbers
can differ without implying that allocation contradicted itself.

Source-low selection summaries show which target rule or plan handled each
source operation. The descriptor key is target-specific, while
`descriptor_semantic_tag` gives the portable instruction family used for
high-level comparisons. Matrix-family checks should filter on semantic tags
instead of hardcoding one target mnemonic:

```bash
jq '.source_low.selection_summaries.rows[]?
  | {function, source_op, selection, plan_key, descriptor_key,
     descriptor_semantic_tag, selected_op_count, emitted_low_op_count}
  | with_entries(select(.value != null))' \
  /tmp/loom-q6q8.compile-report.json

jq '.source_low.selection_summaries.rows[]?
  | select((.descriptor_semantic_tag // "") | startswith("matrix."))
  | {function, source_op, plan_key, descriptor_key, descriptor_semantic_tag,
     selected_op_count, emitted_low_op_count}' \
  /tmp/loom-q6q8.compile-report.json
```

When provider selection, inlining, or math legalization is suspect, capture IR
snapshots around those boundaries:

```bash
loom-compile \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
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
`template.apply<@model.q6q8.accumulate_part>` sites, `inline-callables` should remove
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
`template.apply` site when pass reporting is enabled. The row records the enclosing
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
iree-benchmark-loom \
  loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom \
  --device=amdgpu \
  --measure=dispatch_complete \
  --iterations=1 \
  --warmup-iterations=0 \
  --batch-size=1 \
  --min-time-ms=0 \
  --max-batches=1 \
  --input-ring-count=1 \
  --profile-final-batch=true \
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
   planned_spills:.compile_report.allocation.spill_count,
   materialized_spill_storage:
     .compile_report.allocation.materialized_spill_storage_count,
   materialized_spill_stores:
     .compile_report.allocation.materialized_spill_store_count,
   materialized_reloads:.compile_report.allocation.materialized_reload_count,
   local:.compile_report.memory.local_bytes,
   private:.compile_report.memory.private_bytes,
   final_vgprs:.compile_report.target_resources.vector.final.register_count,
   scheduled_vgpr_pressure:
     .compile_report.target_resources.vector.scheduled_pressure.peak_live_units,
   pressure:.compile_report.schedule.register_pressure_peak_live_units}' \
  /tmp/loom-q6q8-run/results.jsonl

jq 'select(.row=="benchmark" and .benchmark_result.compile_report) |
  .benchmark_result |
  {benchmark,
   code:.compile_report.emission.code_byte_count,
   planned_spills:.compile_report.allocation.spill_count,
   materialized_spill_storage:
     .compile_report.allocation.materialized_spill_storage_count,
   materialized_spill_stores:
     .compile_report.allocation.materialized_spill_store_count,
   materialized_reloads:.compile_report.allocation.materialized_reload_count,
   local:.compile_report.memory.local_bytes,
   final_vgprs:.compile_report.target_resources.vector.final.register_count,
   scheduled_vgpr_pressure:
     .compile_report.target_resources.vector.scheduled_pressure.peak_live_units,
   private:.compile_report.memory.private_bytes}' \
  /tmp/loom-q6q8-run/results.jsonl
```

For quick object-level disassembly of a standalone HSACO, use the LLVM object
tools on the emitted sidecar:

```bash
llvm-objdump -d --mcpu=gfx11-generic /tmp/loom-q6q8.hsaco
```

Treat the evidence channels separately. Planner output answers "what would run?"
Compile reports answer "what did the compiler emit?" Artifact manifests answer
"what does this loader-ready artifact contain?"

`dispatch_complete` produces two distinct executions when
`--profile-final-batch=true`. `measurement.operation_timing_ns` measures the
warmed major batch from host submission through queue completion. The final
profile replay then executes the same candidate, configuration, invocation
plan, bindings, and dispatch multiplicity with profile metadata retained.
`profile_replay.measurement_relationship` identifies it as a distinct
execution, `profile_replay.dispatch_timing.duration_ns` contains aggregate
device timing, and
`profile_replay.dispatch_timing.dispatch_distribution.duration_ns` contains
exact per-dispatch p50, p90, and spread statistics when every replay sample can
be reconstructed. Profiling stays outside the major timing window because
instrumentation may perturb queue-completion timing.

Kernel comparisons against Vulkan, HIP, or another device profiler use
`profile_replay.dispatch_timing.dispatch_distribution.duration_ns.p50` on the
Loom side. `profile_replay.comparison` appears only when at least 16 complete,
comparable, homogeneous, non-overlapping samples describe one physical
dispatch per logical operation. The comparison is against equivalently
instrumented device timing, never against the ordinary host queue-completion
measurement. Batch shape still matters: match dispatch multiplicity and inspect
the distribution provenance before comparing results.

An isolated `--batch-size=1` benchmark submits through direct HAL
`queue_dispatch`; no command buffer is created. Larger batches use reusable
command buffers with execution barriers between dispatches so submission
overhead is amortized without allowing accidental overlap.

GPU kernel optimization uses a serialized multi-dispatch batch with independent
binding sets as its primary score. This sustains device clocks, amortizes host
submission, and gives the final profiled replay enough device samples to expose
variance. A one-dispatch direct submission is an isolated latency cross-check,
not a statistically strong kernel-throughput result.

The quick command above intentionally uses one hot-reuse input ring and tiny
iteration counts for smoke coverage; serious timing should use warmups, a
stable major measurement window, a representative input ring, and final-batch
profiling.

## AMDGPU Global And Descriptor Memory Feedback

The AMDGPU compile report separates raw global VMEM packets from descriptor
MUBUF packets. This matters when comparing a Loom route against HIP, RADV,
LLVM, or hand-written assembly: two sources can have the same semantic global
memory operation while selecting very different instruction families.

`static_instruction_mix.global_memory_count` remains the broad global-memory
aggregate. The narrower family counters identify the selected packet family:
`global_load_count` and `global_store_count` count raw global-address packets,
`buffer_load_count` and `buffer_store_count` count descriptor-backed MUBUF
packets, and `flat_memory_count` counts flat-memory packets.

```bash
jq '.static_instruction_mix
  | {global_memory_count,
     global_load_count, global_store_count,
     buffer_load_count, buffer_store_count,
     flat_memory_count}' \
  /tmp/kernel.compile-report.json
```

For per-operation attribution, detailed reports expose the selected packet and
memory-space facts in `.source_low.memory_rows[]`:

```bash
jq '.source_low.memory_rows[]?
  | {function, source_op, operation, memory_space, packet, address_form,
     vector_lanes}' \
  /tmp/kernel.compile-report.json
```

For traffic economics, `.source_low.memory` groups the same evidence by source
root, argument, and selected strategy:

```bash
jq '.source_low.memory
  | {read_bytes: .dispatch_issued.read_bytes,
     write_bytes: .dispatch_issued.write_bytes,
     roots: [.roots[]? | {function, source_root, memory_space,
                          read_bytes: .dispatch_issued.read_bytes,
                          write_bytes: .dispatch_issued.write_bytes}],
     strategies: [.strategies[]? | {function, operation, strategy,
                                    packet_count,
                                    read_bytes: .dispatch_issued.read_bytes,
                                    write_bytes:
                                      .dispatch_issued.write_bytes}]}' \
  /tmp/kernel.compile-report.json
```

On AMDGPU, a source-low buffer with global memory facts lowers through the raw
global-address family when the address form is legal. A buffer with descriptor
memory facts can lower through the MUBUF family when the descriptor, offset, and
range facts prove the packet requirements. The report counters make that choice
visible before object disassembly enters the debugging loop.

## AMDGPU Shared-Memory Feedback

The AMDGPU compile report explains selected workgroup-memory packets and the
source address facts retained at packet selection. When Loom has a named
instruction- and target-specific service model, summary reports retain compact
global and per-source-operation bank-service groups. Detailed rows additionally
report each exact structural LDS bank-service profile or the reason the source
address and active-lane facts were insufficient. These are bank service rounds,
not cycle predictions. Runtime measurements and profiler counters remain
necessary when occupancy, cache behavior, scheduling, or data-dependent control
flow dominates.

The bounded report tools preserve the same evidence without replaying lowering:

```bash
loom-compile-report show /tmp/kernel.compile-report.json
loom-compile-report \
  diff /tmp/baseline.compile-report.json \
       /tmp/candidate.compile-report.json
loom-compile-report \
  suggest /tmp/candidate.compile-report.json --include-experimental
```

`show` calls out conflicted and incomplete semantic groups. `diff` matches groups
by function, source operation, memory root, operation, packet, and strategy, so
an improvement in one access cannot hide a regression in another through global
netting. It also reports loss of exact proof coverage and changes to the model
revision or evidence class. `suggest` proposes bounded layout experiments only
when a group has complete exact proof, conflicts, and structural extra rounds.
The experiment searches padding or pitch, lane mapping, fragment layout, and
packet width, then rejects spill or occupancy regressions before hardware
timing.

Compile with detailed reports when investigating shared-memory layout,
padding, swizzling, vectorization, or imported kernel staging choices:

```bash
loom-compile loom/src/loom/test/corpus/authoring/hip/shared_memory_vector_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/shared-memory-vector-tile.vmfb \
  --compile-report=json-details \
  --compile-report-output=/tmp/shared-memory-vector-tile.compile-report.json
```

The `.source_low.memory_rows[]` array records one selected memory-packet row per
reported source memory operation:

```bash
jq '.source_low.memory_rows[]?
  | {function, source_op, operation, packet, vector_lanes,
     dynamic_stride_bytes, vector_lane_stride_bytes,
     bank_service}' \
  /tmp/shared-memory-vector-tile.compile-report.json
```

`bank_service` is absent when no model is registered for the selected target
and packet. A modeled packet records the immutable model revision and evidence
class independently from the address proof. `proof: "exact"` carries the
per-phase service rounds, uncontended baseline, extra rounds, and maximum
same-bank request multiplicity. `proof: "unknown"` carries a stable
`unknown_reason` instead of guessing through unsupported address shapes or
divergent active-lane control. Exact proof establishes the result within the
named model; the model's evidence class separately states whether its behavior
has been documented, calibrated on silicon, or only recovered from vendor
software.

The current gfx1250 model covers full-wave `ds_read_b128` and
`ds_write_b128` packets whose lane addresses are an aligned affine function of
`workitem.id<x>`. Its phase topology and request policy come from the named
rocRoller software model and are deliberately labeled
`vendor-software-model-unvalidated` until calibrated against silicon.

The text form carries the same fields as `source_low_memory[...]` rows when a
greppable report is more convenient:

```bash
loom-compile loom/src/loom/test/corpus/authoring/hip/shared_memory_vector_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/shared-memory-vector-tile.vmfb \
  --compile-report=text-details \
  --compile-report-output=/tmp/shared-memory-vector-tile.compile-report.txt

rg 'source_low_memory|dynamic_stride_bytes|ds_' \
  /tmp/shared-memory-vector-tile.compile-report.txt
```

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
residual add and a named `rows` parameter. The runtime parameter drives the case
tensor shapes, scalar kernel argument, and dynamic buffer views. A separately
bound row-capacity config controls reusable launch geometry, and the kernel
relates the runtime row count to that capacity with `index.assume`.

The anonymous `check.benchmark<@mlp_down_projection_residual_case>` sweeps all
case samples and receives generated benchmark names. The named benchmark rows
pin specific samples with assignment dictionaries:

```loom
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_decode {rows = 2}
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_full {rows = 3584}
```

The case uses deterministic iota inputs and zero projection weights for a
residual-preservation oracle. The AMDGPU dispatch test explicitly binds the row
capacity through `--config`, exactly as an embedding application does. Selected
sample values remain runtime inputs and do not alter the executable.

## Authoring Rules

An SSA value name carries the value's program role. A constant named
`%fivehundredtwelve` communicates no more than its literal and reads like
`int fivehundredtwelve = 512;` in C. The authored spelling is `%batch_size`
when 512 is the batch size, and `%c512` when the literal itself is the only
meaning. Type suffixes disambiguate otherwise identical bare literals when
needed, such as `%c0_i32` and `%c0_f32x4`.

Type, shape, sign, and duplicate markers do not turn a literal into a role.
Names such as `%i32_zero`, `%zero_scalar`, `%positive_zero`, `%zero_a`, and
`%f16_ones` still read like declarations of the number itself. They use a
semantic name when the use provides one, or the `%c<literal>` spelling when it
does not.

Literal equality does not imply semantic identity. Two constants whose value
is 16 can remain separate as `%channels_per_group` and
`%lane_partition_width`; merging them under `%c16` would discard information
the source already has.

`func.call` is an exact symbol reference. It is the right spelling when the
caller wants one specific helper or declaration, as with
`@q6_signed_pack_dot4i` and `@bf16_dot32`.

`template.apply<@K>` is a compile-time implementation demand. The key is a contract,
not a symbol. The selection pass resolves live applies against visible
`template.def` providers, prunes dead private providers, rewrites selected
sites to inline calls, and leaves normal inlining to splice the body. Good
contract names describe a reusable motif or layout operation rather than a
particular model brand.

`inline` and `noinline` describe callable-boundary intent. `inline` is useful
for private helpers that should disappear before target-sensitive lowering.
`noinline` preserves a real callable boundary and therefore needs a target-ready
symbol when target lowering reaches it. `hot` and `cold` are separate
temperature hints for cost models or profile feedback; they are not substitutes
for authored inline policy.

Facts belong at the boundary that owns them. Compile-time choices are declared
with `config.decl` and materialized by the application or test driver. Runtime
workload values stay in function and kernel signatures even when a particular
configuration constrains them. Use `index.assume` to relate a runtime value to a
configured capacity or to record facts established by guards, clamping, or
dynamic alignment. Exact shape choices should be read from config directly;
checked samples only supply runtime values and expected results. Launch topology
belongs in `kernel.launch.config`, and kernel ABI buffers already carry global
memory-space facts.

Use `index` for logical coordinates, extents, and tensor/view indices. Use
`offset` for byte offsets and byte strides. Views should carry real extents
from the source contract; large sentinel shapes that only make proofs pass
destroy bounds-checking and sanitizer value because the compiler can no longer
see the real accessible range.

When a logical coordinate selects a packed byte window, `index.scale` is the
explicit boundary: it multiplies an `index` coordinate by an `offset` byte
stride and produces the `offset` value expected by `buffer.view`.

FP8 checkpoint storage should carry both the payload dialect and the content
contract on the view storage schema. A plain `f8E4M3` or `f8E5M2` view only
carries the scalar FP8 type facts; NaN, zero, and subnormal payloads remain
possible unless storage says more, and IEEE-style `f8E5M2` can also represent
infinity. The schema `element_format` records the payload dialect, such as
`f8e4m3`, `f8e4m3fn`, or `f8e5m2`. Model weights that have been validated finite
should also set `rounding=finite_only` so loads and fragments publish
no-NaN/no-infinity facts while still preserving exact zero and subnormal
behavior. `finite_flush_subnormal` is only for storage whose physical payloads
have already been flushed or are otherwise guaranteed not to contain subnormal
values; it is a stronger content contract, not a request for the target decoder
to repair contradictory bytes at load time.

The distinction is visible in codegen on targets that construct FP8 fragments
through software packets. `finite_only` removes NaN and infinity repair, but
exact zero and subnormal payloads still require repair unless other value facts
prove they cannot appear. `finite_flush_subnormal` lets those targets skip the
subnormal side of that repair. Use the source-low memory report strategy key to
confirm the selected route, such as
`fp8_packed_bf16_decode_repair_zero_subnormal` versus
`fp8_packed_bf16_decode_repair_zero`, instead of inferring it from source text.

For weight-only FP8 linears, the portable source shape is often two explicit
phases: preserve FP8/BF8 checkpoint storage facts on the source view, then
materialize the packed rows into the BF16 layout consumed by the contraction.
Targets without a profitable native FP8 matrix path can select the BF16 GEMM
route, while targets with native FP8 support can specialize the same semantic
contract through target facts and report rows. Direct FP8 fragment loads remain
useful coverage and may be profitable on some targets, but they should be
selected because the report proves they are the best route, not because the
source hid storage materialization inside the matrix kernel.

```loom
%weight_layout = encoding.layout.strided [1, %input_size] : encoding<layout>
%weight_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=16, payload_registers=4, rounding=finite_only> : encoding<schema>
%weight_storage = encoding.define #encoding.storage {layout = %weight_layout : encoding<layout>, schema = %weight_schema : encoding<schema>} : encoding<storage>
%weight_view = buffer.view %weight_buffer[%base] : buffer -> view<[%input_size]x[%output_size]xf8E4M3, %weight_storage>
```

The Loom source linter applies constant naming to every checked `.loom` file
and to authored input sections in `.loom-test` files. Runner-owned expected
sections remain generated test output; only `loom-check --update` changes them.
The linter rejects English-spelled numeric constant names in favor of semantic
roles or the `%c<literal>` convention, because agents copy examples before they
read design notes. Loom project hygiene runs this linter in normal precommit and
CI flows, including root CI invocations that delegate project test suites to
separate workflows.

Additional authoring-corpus rules keep this reference surface aligned with the
boundary contract. They reject redundant kernel-buffer memory-space assumes,
sentinel-sized views, late `index.cast` byte-address conversions, and ggml-style
`nb*` byte strides typed as `index`.

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
| Residual `template.apply` after final selection | No provider implemented the contract, every provider was rejected by signature or predicates, or multiple highest-priority providers tied. |
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
