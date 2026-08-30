# HIP C++ to Loom Cookbook

Audience: a user or agent has a HIP/CUDA/inline-asm kernel and wants Loom source
that can match or beat it. The entries are grep/context friendly: tags first,
then the HIP habit, Loom spelling, proof command, query surface, and pressure
points.

Loom does not silently clone the HIP compiler. Clang/LLVM infer profitable
unrolling, vector widths, contraction, address forms, and target idioms from C++
source plus backend cost models. Loom source is a compiler IR: authoring states
the facts and local transform intent that must be true, target libraries own
provider alternatives, and reports/diagnostics explain why a pass did or did
not consume that intent.

Diagnostics and reports are the query interface. A useful recipe should leave a
user one command away from structured feedback, not reading tea leaves between
IR dumps.

Command blocks are written in the short form used after adding built Loom tools
to `PATH` and running from this directory. From the repository root, invoke the
same tool through Bazel and pass root-relative recipe paths:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-opt:loom-opt -- \
  loom/src/loom/test/corpus/authoring/hip/q8_block_unroll.loom \
  --pass=unroll-scf-for \
  --pass-report=json \
  --output=/tmp/q8-block-unrolled.loom
```

## Recipe Index

| Tags / HIP terms | Loom feature | Recipe |
| --- | --- | --- |
| `#pragma unroll`, `pragma-unroll`, `loop-expansion`, `for`, `q8`, `WG64` | `scf.for ... unroll`, `unroll-scf-for`, range-fact trip count inference | `q8_block_unroll.loom` |
| `blockIdx`, `threadIdx`, `lane`, `warp`, `wavefront` | `kernel.launch.config`, `kernel.workgroup.id`, `kernel.workitem.id`, `kernel.subgroup.*` | `q8_block_unroll.loom` |
| `threadIdx`, `lane_id`, `warp lane`, `wavefront lane`, `SGPR`, `VGPR`, `EXEC`, `scalarized`, `lane-varying`, `uniform`, `dot operand` | value distribution facts, `test.fact_subgroup_uniform`, `test.fact_workgroup_uniform`, `test.fact_lane_varying`, `test.fact_lane_predicate` | `lane_distribution.loom` |
| `__global__`, `restrict`, pointer casts, address arithmetic | kernel ABI `buffer`, `buffer.assume.noalias`, `buffer.view`, `index`/`offset` math | `q8_block_unroll.loom` |
| `global_load_b32`, packed bytes, `q8`, bitfield, unpack | `vector.load`, `vector.bitunpacks<8>`, `scalar.extf`, `vector.dotf` | `q8_block_unroll.loom` |
| `global_load_b32`, `global_load_b128`, `uint4`, adjacent scalar loads, coalescing | `vector.load -> vector<1xi32>` versus `vector.load -> vector<4xi32>` | `q8_load_width.loom` |
| `__shared__`, `__syncthreads`, LDS, tile staging, cross-lane exchange | `buffer.alloca ... memory_space = workgroup`, `buffer.view`, `kernel.barrier<workgroup>` | `shared_memory_tile.loom` |
| `__shared__`, `__syncthreads`, LDS, 2D tile, transpose, double buffering | two workgroup `buffer.alloca` tiles, x/y workitem ids, repeated `kernel.barrier<workgroup>` | `shared_memory_transpose.loom` |
| `__shared__`, `__syncthreads`, LDS, `uint4`, `int4`, `ds_store_b128`, `ds_load_b128` | `vector.load -> vector<4xi32>` staged through workgroup memory | `shared_memory_vector_tile.loom` |
| `q8`, `q4`, `u8`, `s8`, `u4`, `s4`, `v_dot4_i32_iu8`, `dp4a` | `vector.bitunpacku`, `vector.bitunpacks`, `vector.dot4i<u8s8>` | `q8_q4_signedness.loom` |
| `q2`, `q3`, `q4`, `q5`, `q6`, split high bits, lookup tables, offset-binary, packed-dot repack | `vector.bitunpacku`, `vector.bitfield.extractu`, `vector.bitfield.insert`, `vector.table.lookup`, `vector.dot8i4` | `packed_field_contracts.loom` |
| dynamic lower-bound unroll, missing facts | structured diagnostic, exact/range facts | `q8_hip_shaped_unroll_unresolved.loom` |
| wave size, template, specialization, fallback | targetless `template.apply`, typed `template.def ... requires`, `#target.subgroup.size` | `target_provider_selection.loom` |
| targetless template device function, `expf`, target math policy, inline | invocation specialization, then authoring expansion and math legalization | `template_math_legalization.loom` |
| `__cluster_dims__`, cluster multicast, async-to-LDS, `s_wait_asynccnt`, b128 | static `cluster_size`, `kernel.async.cluster.gather`, async groups and waits | `cluster_b128_multicast.loom` |

## Lane Distribution

Tags: `threadIdx`, `lane_id`, `warp lane`, `wavefront lane`, `SGPR`, `VGPR`,
`EXEC`, `scalarized`, `lane-varying`, `uniform`, `dot operand`,
`inline asm constraint`.

HIP habit:

```c++
int element_index = blockIdx.x * blockDim.x + threadIdx.x;
int lane = __builtin_amdgcn_mbcnt_lo(-1, 0);
bool lane0 = lane == 0;
int selected = lane0 ? lhs : rhs;
int dot_operand = lane + lhs;
```

Loom spelling:

```loom
%block = kernel.workgroup.id<x> : index
%thread = kernel.workitem.id<x> : index
%element_index = index.madd %block, %workgroup_size, %thread : index
%lane = kernel.subgroup.lane.id : index
%is_lane_zero = index.cmp eq, %lane, %zero_index : index
%selected = scf.select %is_lane_zero, %lhs, %rhs : i32
%lane_i32 = index.cast %lane : index to i32
%dot_operand = scalar.addi %lane_i32, %lhs : i32
```

The source-level contract is value distribution: `kernel.workitem.id` and
`kernel.subgroup.lane.id` are lane-varying roots; `kernel.workgroup.id`,
`kernel.workgroup.size`, constants, and scalar kernel arguments are uniform
until mixed with lane-varying data. Arithmetic depending on a lane-varying value
stays lane-varying. A lane-dependent `i1` is a lane predicate. Selecting between
uniform values with a lane predicate produces lane-varying data.

AMDGPU SGPR/VGPR placement is a target-lowering consequence, not a portable
source type. For HIP ports, the actionable source question is whether the value
that feeds an address, dot operand, or EXEC-controlled path still depends on a
lane-varying root at the point where source transforms run.

Proof command:

```bash
loom-opt lane_distribution.loom \
  --pass=canonicalize \
  --output=/tmp/lane-distribution.loom
```

Useful query:

```bash
rg 'is_(uniform|varying|lane)' /tmp/lane-distribution.loom
```

Expected signal:

```loom
%thread_is_varying = scalar.constant true : i1
%element_is_varying = scalar.constant true : i1
%predicate_is_lane = scalar.constant true : i1
%selected_is_varying = scalar.constant true : i1
%dot_operand_is_varying = scalar.constant true : i1
```

`test.fact_*` probes are an authoring/debug surface used in recipes and tests.
Production kernels leave those probes out. When a target has a stronger
hardware-specific requirement, such as rejecting an operand that target lowering
would place in a scalar register, that belongs in structured diagnostics or
reports at the target boundary.

## Q8 WG64 Local Unroll

Tags: `HIP #pragma unroll`, `CUDA #pragma unroll`, `scf.for`, `unroll`,
`unroll-scf-for`, `STRUCTURE/014`, `q8_0`, `f32`, `WG64`, `block_slot`,
`blocks_per_row`, `range-facts`, `dynamic-lower-bound`.

HIP habit:

```c++
#pragma unroll
for (int block = block_slot; block < blocks_per_row; block += block_step) {
  // load one packed Q8 word, unpack 4 bytes, scale, dot with RHS.
}
```

Loom spelling:

```loom
%block_slot = index.div %lane, %eight : index
%block_step = index.div %workgroup_size, %eight : index

%sum = scf.for %block_idx_raw = [%block_slot to %blocks_per_row step %block_step](%acc = %zero_f32 : f32) -> (f32) unroll {
  %block_idx = index.assume %block_idx_raw [range(%block_idx_raw, 0, 15)] : index
  ...
  scf.yield %next : f32
}
```

The `unroll` marker is full-unroll intent, not a heuristic. `unroll-scf-for`
consumes it when facts prove one exact trip count. Exact lower/upper/step facts
are enough. A dynamic lower bound is also enough when it has a finite integer
range and the first and last values in that range have the same trip count for
the exact positive step. In this Q8 WG64 pattern, `%lane` is in `[0,63]`,
`%block_slot = %lane / 8` is in `[0,7]`, `%blocks_per_row` is `16`, and
`%block_step` is `8`, so every lane executes two local blocks.

`kernel.launch.config` is the source-level contract for grid and workgroup
range facts. `index.assume` records facts that are not already present in the
IR; the body-level `range(%block_idx_raw, 0, 15)` records the valid packed-block
domain for later address and load reasoning.

Proof command:

```bash
loom-opt q8_block_unroll.loom \
  --pass=unroll-scf-for \
  --pass-report=json \
  --output=/tmp/q8-block-unrolled.loom \
  2>/tmp/q8-block-unroll-report.json
```

Useful queries:

```bash
jq '.invocations[]
  | select(.pass == "unroll-scf-for")
  | .statistics' /tmp/q8-block-unroll-report.json

jq '.invocations[]
  | select(.pass == "unroll-scf-for")
  | .details[]
  | select(.category == "scf-unroll")
  | {outcome, policy, trip_count, step, lower_bound_kind, lower_range_min, lower_range_max}' /tmp/q8-block-unroll-report.json

rg 'scf.for|vector.bitunpacks|vector.dotf' /tmp/q8-block-unrolled.loom
```

Expected signal:

```json
{"loops-unrolled":1}
{"outcome":"unrolled","policy":"bare","trip_count":2,"step":8,"lower_bound_kind":"dynamic","lower_range_min":0,"lower_range_max":7}
```

The transformed file should keep `vector.bitunpacks<8>` and `vector.dotf`, while
the local `scf.for` disappears.

## Q8 Load Width

Tags: `global_load_b32`, `global_load_b64`, `global_load_b128`, `uint4`,
`reinterpret_cast`, `packed q8`, `packed q4`, `coalescing`, `vectorized-load`,
`restrict`, `alignment`.

HIP habit:

```c++
const uint32_t *words = reinterpret_cast<const uint32_t *>(q8_bytes);
uint32_t w0 = words[0];
uint32_t w1 = words[1];
uint32_t w2 = words[2];
uint32_t w3 = words[3];

uint4 wide = *reinterpret_cast<const uint4 *>(words);
```

Loom spelling:

```loom
%input_words = buffer.view %input_noalias[%base] : buffer -> view<4xi32>

%w0 = vector.load %input_words[0] : view<4xi32> -> vector<1xi32>
%w1 = vector.load %input_words[1] : view<4xi32> -> vector<1xi32>
%w2 = vector.load %input_words[2] : view<4xi32> -> vector<1xi32>
%w3 = vector.load %input_words[3] : view<4xi32> -> vector<1xi32>

%wide = vector.load %input_words[0] : view<4xi32> -> vector<4xi32>
```

The scalar path and vector path are both correct source shapes. They are not
the same storage contract. Use scalar loads when each word is independent; use
the vector load when the source pattern expects one wide load. Alignment,
address-range, and alias facts still live next to the view root. Kernel ABI
buffers already carry global memory-space facts; spell `buffer.assume.noalias`,
`buffer.view`, and `index.assume ... [mul(...)]` for dynamic aligned offsets
when those facts are part of the source contract.

Proof command:

```bash
loom-opt q8_load_width.loom \
  --pass=canonicalize \
  --output=/tmp/q8-load-width.loom
```

Useful query:

```bash
rg 'vector.load .*vector<(1|4)xi32>' /tmp/q8-load-width.loom
```

Expected signal:

```loom
%w0 = vector.load %input_words[0] : view<4xi32> -> vector<1xi32>
%wide = vector.load %input_words[0] : view<4xi32> -> vector<4xi32>
```

## Shared Memory Tile

Tags: `__shared__`, `__syncthreads`, `LDS`, `workgroup memory`, `tile staging`,
`cross-lane exchange`, `ds_write`, `ds_read`, `s_barrier`.

HIP habit:

```c++
__shared__ int scratch[64];
int lane = threadIdx.x;
scratch[lane] = input[lane];
__syncthreads();
output[lane] = scratch[63 - lane];
```

Loom spelling:

```loom
%scratch = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer
%scratch_view = buffer.view %scratch[%base] : buffer -> view<64xi32>

%loaded = vector.load %input_view[%lane] : view<64xi32> -> vector<1xi32>
vector.store %loaded, %scratch_view[%lane] : vector<1xi32>, view<64xi32>
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
%reversed = vector.load %scratch_view[%reverse_lane] : view<64xi32> -> vector<1xi32>
vector.store %reversed, %output_view[%lane] : vector<1xi32>, view<64xi32>
```

The contract is the memory space and synchronization relationship.
`buffer.alloca` with `memory_space = workgroup` creates per-workgroup storage,
`buffer.view` gives it a typed indexing contract, and
`kernel.barrier<workgroup>` is the acquire/release point that makes writes by
one work item visible before another work item reads the tile. The reverse-lane
example intentionally uses distinct input values so the checked output depends
on cross-lane LDS traffic, not private register roundtripping.

Proof command:

```bash
iree-test-loom shared_memory_tile.loom --device=amdgpu
```

Target compile evidence:

```bash
loom-compile shared_memory_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/shared-memory-tile.hal \
  --emit-target-artifact=/tmp/shared-memory-tile.hsaco \
  --artifact-manifest=summary \
  --compile-report=summary \
  --compile-report-output=/tmp/shared-memory-tile.compile-report.json
```

Useful queries:

```bash
jq '{status, target_key, local:.entries.rows[0].local_memory_bytes, lds_ops:.static_instruction_mix.local_memory_count, barriers:.static_instruction_mix.barrier_count}' \
  /tmp/shared-memory-tile.compile-report.json

llvm-objdump -d --mcpu=gfx11-generic /tmp/shared-memory-tile.hsaco | rg 'ds_(read|write)|s_barrier'
```

Expected signal: `case_shared_memory_tile_reverse` passes.

For the 64-element i32 tile, the compile report records `local_memory_bytes` as
`256`, two local memory instructions, and one barrier. On AMDGPU targets, object
disassembly should show LDS read/write instructions and a workgroup barrier.

## Shared Memory Transpose

Tags: `__shared__`, `__syncthreads`, `LDS`, `workgroup memory`, `2D tile`,
`transpose`, `double buffering`, `threadIdx.x`, `threadIdx.y`.

HIP habit:

```c++
__shared__ int scratch_a[8][8];
__shared__ int scratch_b[8][8];
int row = threadIdx.y;
int column = threadIdx.x;

scratch_a[row][column] = input[row][column];
__syncthreads();
scratch_b[row][column] = scratch_a[column][row];
__syncthreads();
output[row][column] = scratch_b[column][row];
```

Loom spelling:

```loom
%row = kernel.workitem.id<y> : index
%column = kernel.workitem.id<x> : index
%scratch_a = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer
%scratch_b = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer

vector.store %loaded, %scratch_a_view[%row, %column] : vector<1xi32>, view<8x8xi32>
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
%transposed = vector.load %scratch_a_view[%column, %row] : view<8x8xi32> -> vector<1xi32>
vector.store %transposed, %scratch_b_view[%row, %column] : vector<1xi32>, view<8x8xi32>
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
%roundtrip = vector.load %scratch_b_view[%column, %row] : view<8x8xi32> -> vector<1xi32>
```

This recipe is a stronger LDS smoke test than a one-dimensional lane exchange:
it uses x/y workitem ids, two scratch allocations, two synchronization points,
and source-visible cross-axis addressing. The correctness oracle stays small:
transposing through LDS twice must reproduce the original row-major iota.

Proof command:

```bash
iree-test-loom shared_memory_transpose.loom --device=amdgpu
```

Target compile evidence:

```bash
loom-compile shared_memory_transpose.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/shared-memory-transpose.hal \
  --emit-target-artifact=/tmp/shared-memory-transpose.hsaco \
  --artifact-manifest=summary \
  --compile-report=summary \
  --compile-report-output=/tmp/shared-memory-transpose.compile-report.json
```

Useful queries:

```bash
jq '{status, target_key, local:.entries.rows[0].local_memory_bytes, lds_ops:.static_instruction_mix.local_memory_count, barriers:.static_instruction_mix.barrier_count}' \
  /tmp/shared-memory-transpose.compile-report.json

llvm-objdump -d --mcpu=gfx11-generic /tmp/shared-memory-transpose.hsaco | rg 'ds_(read|store)|s_barrier'
```

Expected signal: `case_shared_memory_tile_double_transpose` passes.

For the two 8x8 i32 tiles, the compile report records `local_memory_bytes` as
`512`, four local memory instructions, and two barriers. AMDGPU object
disassembly should show two LDS stores, two LDS reads, and two workgroup
barriers.

## Shared Memory Vector Tile

Tags: `__shared__`, `__syncthreads`, `LDS`, `workgroup memory`, `uint4`,
`int4`, `vectorized-load`, `ds_store_b128`, `ds_load_b128`.

HIP habit:

```c++
__shared__ int4 scratch[64];
int lane = threadIdx.x;

scratch[lane] = reinterpret_cast<const int4 *>(input)[lane];
__syncthreads();
reinterpret_cast<int4 *>(output)[lane] = scratch[lane];
```

Loom spelling:

```loom
%scratch = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer
%scratch_view = buffer.view %scratch[%base] : buffer -> view<64x4xi32>

%loaded = vector.load %input_view[%lane, 0] : view<64x4xi32> -> vector<4xi32>
vector.store %loaded, %scratch_view[%lane, 0] : vector<4xi32>, view<64x4xi32>
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
%roundtrip = vector.load %scratch_view[%lane, 0] : view<64x4xi32> -> vector<4xi32>
vector.store %roundtrip, %output_view[%lane, 0] : vector<4xi32>, view<64x4xi32>
```

This recipe covers the wide-row shared-memory path that HIP code often spells
with `uint4`, `int4`, or a reinterpret-cast vector load. The source contract is
the vector width and 16-byte alignment, not a target opcode request; AMDGPU
lowering is responsible for selecting the matching 128-bit LDS access when the
target supports it.

Proof command:

```bash
iree-test-loom shared_memory_vector_tile.loom --device=amdgpu
```

Target compile evidence:

```bash
loom-compile shared_memory_vector_tile.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/shared-memory-vector-tile.hal \
  --emit-target-artifact=/tmp/shared-memory-vector-tile.hsaco \
  --artifact-manifest=summary \
  --compile-report=summary \
  --compile-report-output=/tmp/shared-memory-vector-tile.compile-report.json
```

Useful queries:

```bash
jq '{status, target_key, local:.entries.rows[0].local_memory_bytes, lds_ops:.static_instruction_mix.local_memory_count, barriers:.static_instruction_mix.barrier_count}' \
  /tmp/shared-memory-vector-tile.compile-report.json

llvm-objdump -d --mcpu=gfx11-generic /tmp/shared-memory-vector-tile.hsaco | rg 'global_(load|store)_b128|ds_(store|load)_b128|s_barrier'
```

Expected signal: `case_shared_memory_vector_tile_roundtrip` passes.

For the 64 row by 4 i32 tile, the compile report records `local_memory_bytes` as
`1024`, two local memory instructions, and one barrier. AMDGPU object
disassembly should show `global_load_b128`, `ds_store_b128`, `s_barrier`,
`ds_load_b128`, and `global_store_b128`.

## Shared Memory Bank Feedback

Tags: `__shared__`, `LDS`, `bank conflict`, `padding`, `swizzle`,
`compile-report`, `source_low_memory`, `rocprof`, `Nsight`.

HIP shared-memory ports often arrive with layout decisions encoded as padding,
swizzling, reinterpret-cast vector widths, or inline LDS instruction choices.
In Loom those choices should become source layout and indexing contracts first;
the AMDGPU compile report then explains the selected LDS packet and visible
bank pattern.

The canonical workflow and classification table live in the authoring guide's
[AMDGPU shared-memory feedback](/loom/src/loom/test/corpus/authoring/README.md#amdgpu-shared-memory-feedback)
section. Use it after translating HIP `__shared__` storage into `buffer.alloca`
with `memory_space = workgroup`, after preserving intentional vector widths
with `vector.load`/`vector.store`, and before spending time in runtime
profilers. `rocprof` and Nsight still decide final performance; the compile
report answers whether the source layout and selected packet are structurally
reasonable before object-level profiling enters the loop.

## Packed Signedness

Tags: `q8`, `q4`, `packed-byte`, `packed-nibble`, `uint8_t`, `int8_t`,
`u8`, `s8`, `u4`, `s4`, `sign_extend`, `zero_extend`, `v_dot4_i32_iu8`,
`dp4a`.

HIP habit:

```c++
uint32_t q8_word = *reinterpret_cast<const uint32_t *>(q8_bytes);
uint8_t u8_code = byte(q8_word);
int8_t s8_code = byte(q8_word);

uint32_t q4_word = *reinterpret_cast<const uint32_t *>(q4_nibbles);
uint4_t u4_code = nibble(q4_word);
int4_t s4_code = signext(nibble(q4_word));

acc = __builtin_amdgcn_sdot4(/* unsigned lhs, signed rhs */);
```

Loom spelling:

```loom
%q8_word = vector.load %q8_words[0] : view<1xi32> -> vector<1xi32>
%q8_unsigned = vector.bitunpacku<8> %q8_word : vector<1xi32> -> vector<4xi32>
%q8_signed = vector.bitunpacks<8> %q8_word : vector<1xi32> -> vector<4xi32>

%q4_unsigned = vector.bitunpacku<4> %q4_word : vector<1xi32> -> vector<8xi32>
%q4_signed = vector.bitunpacks<4> %q4_word : vector<1xi32> -> vector<8xi32>

%dot = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<4xi8>, vector<4xi8>, vector<1xi32>
```

The packed word owns bit layout; the operation owns numeric interpretation.
For all-ones storage, unsigned q8 lanes are `255`, signed q8 lanes are `-1`,
unsigned q4 lanes are `15`, and signed q4 lanes are `-1`. Mixed dot spelling
follows operand order: `u8s8` means unsigned lhs bytes and signed rhs bytes.
The checked mixed-dot recipe stores its scalar result to a one-element output
tensor because the execution harness models kernel calls as in-place tensor
updates.

Proof command:

```bash
loom-opt q8_q4_signedness.loom \
  --output=/tmp/q8-q4-signedness.loom
```

Useful query:

```bash
rg 'bitunpack(u|s)<[48]>|dot4i<' /tmp/q8-q4-signedness.loom
```

Expected signal:

```loom
%q8_unsigned = vector.bitunpacku<8> ...
%q8_signed = vector.bitunpacks<8> ...
%q4_unsigned = vector.bitunpacku<4> ...
%q4_signed = vector.bitunpacks<4> ...
%dot = vector.dot4i<u8s8> ...
```

Diagnostic contract when the source omits enough facts:

```bash
loom-opt q8_hip_shaped_unroll_unresolved.loom \
  --pass=unroll-scf-for \
  --diagnostic-format=json \
  --output=/tmp/unused.loom 2>/tmp/unroll-diag.jsonl
```

Useful query:

```bash
jq 'select(.error_id == "ERR_STRUCTURE_014")
  | {diagnostic: .error_id,
     code: "\(.domain)/014",
     attr_name: .params.attr_name,
     expected_constraint: .params.expected_constraint}' /tmp/unroll-diag.jsonl
```

Expected signal:

```json
{"diagnostic":"ERR_STRUCTURE_014","code":"STRUCTURE/014","attr_name":"unroll","expected_constraint":"exact static trip count"}
```

That is actionable feedback, not a compiler mystery: add exact/range facts for
the lower bound, upper bound, and step; rewrite through an explicit local
ordinal when the dynamic loop really cannot prove a range-independent trip
count; or leave the loop structured.

## Packed Field Contracts

Tags: `q2`, `q3`, `q4`, `q5`, `q6`, `packed bits`, `split high bits`,
`lookup table`, `offset-binary`, `dot8i4`, `i4 dot`.

HIP habit:

```c++
// The exact helper varies by format, but the shape is usually:
//   load packed bytes or words
//   extract low fields
//   merge side high-bit streams
//   apply bias, table decode, or repack for a dot instruction
```

Loom spelling:

```loom
%codes = vector.bitunpacku<2> %packed : vector<1xi8> -> vector<4xi32>
%field = vector.bitfield.extractu %word {offset = 3, width = 3} : vector<1xi32> -> vector<1xi32>
%merged = vector.bitfield.insert %high4 into %low_codes {offset = 4, width = 1} : vector<4xi8>, vector<4xi8>
%decoded = vector.table.lookup %grid[%codes] : vector<16xi8>, vector<4xi8> -> vector<4xi8>
%dot = vector.dot8i4<s4u4> %signed_rows, %rhs, %acc : vector<2xi32>
```

The contract is the field interpretation, not the nickname of a model format.
Regular q2/q4 fields can use `vector.bitunpacku` when the packed payload
divides cleanly into lanes. Awkward widths such as q3 should keep explicit
offsets with `vector.bitfield.extractu` so the source states the field order
directly. Split formats such as q5/q6 should keep low fields and side high-bit
streams as separate SSA values until `vector.bitfield.insert` assembles the
logical code.

Offset-binary and table formats are different contracts. Offset-binary uses an
explicit bias or equivalent packed-bit repack, while table formats use
`vector.table.lookup` with the grid as ordinary SSA data. Choosing
`vector.dot8i4` is also explicit: it says the frontend/importer has already
regrouped the lanes into the packed integer-dot semantics a target can map to a
native instruction.

Proof command:

```bash
loom-opt packed_field_contracts.loom --output=/tmp/packed-field-contracts.loom
```

Useful query:

```bash
rg 'bitunpacku<2>|offset = 3, width = 3|bitfield.insert|table.lookup|dot8i4' \
  /tmp/packed-field-contracts.loom
```

Expected signal:

```loom
vector.bitunpacku<2>
vector.bitfield.extractu %packed {offset = 3, width = 3}
vector.bitfield.insert %high4 into %low_codes {offset = 4, width = 1}
vector.table.lookup %grid[%codes]
vector.dot8i4<s4u4>
```

## Target-Fact Provider Selection

Tags: `HIP template`, `CUDA template`, wave size, subgroup size,
specialization, fallback, targetless, `template.apply`, `template.def`, `requires`,
`where`, `#target.subgroup.size`, `priority`.

HIP habit:

```c++
template <int WarpSize>
int scale_i32(int value) {
  if constexpr (WarpSize == 64) return scale_i32_wave64(value);
  if constexpr (WarpSize == 32) return scale_i32_wave32(value);
  return scale_i32_fallback(value);
}
```

Loom spelling:

```loom
template.def<@hip.recipe.scale_i32> requires [#target.subgroup.size<64>] priority(20) @scale_i32_subgroup_64(%value: i32) -> (i32) { ... }
template.def<@hip.recipe.scale_i32> requires [#target.subgroup.size<32>] priority(20) @scale_i32_subgroup_32(%value: i32) -> (i32) { ... }
template.def<@hip.recipe.scale_i32> priority(1) @scale_i32_fallback(%value: i32) -> (i32) { ... }

kernel.def @selects_subgroup_provider() {
  %c1 = index.constant 1 : index
  %subgroup_size = target.subgroup.size : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%subgroup_size, %c1, %c1) : index
} launch(%input: buffer, %output: buffer) {
  ...
  %scaled = template.apply<@hip.recipe.scale_i32>(%value) : (i32) -> (i32)
  ...
  kernel.return
}
```

`template.apply<@contract>` is the call-site demand. `template.def<@contract>` rows
are providers. `requires [...]` states typed facts that the application site
must prove, while `where [...]` constrains the provider's formal SSA values.
Both are conjunctive, and `priority(...)` orders providers whose applicability
is proven. Nothing in this source names AMDGPU or SPIR-V: subgroup size is the
complete applicability requirement, so the kernel and all three providers
remain targetless. A live JIT supplies facts from its device; offline
compilation supplies the same structured profile with `--target`.
`target.subgroup.size` reads the selected fact as SSA for launch arithmetic;
the paired `#target.subgroup.size<...>` spelling constrains static provider
applicability. Both consume the same function-version context.
`target(@...)` belongs only on a provider whose implementation actually
requires that target identity.

Compile the same source for generic wave32 and wave64 target profiles:

```bash
loom-compile target_provider_selection.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=/tmp/target-provider-gfx11-generic.hal \
  --emit-target-artifact=/tmp/target-provider-gfx11-generic.hsaco \
  --dump-ir-after=select-templates \
  --dump-ir-format=jsonl \
  --dump-ir-output=/tmp/target-provider-gfx11-generic-trace.jsonl

loom-compile target_provider_selection.loom \
  --backend=amdgpu-hal \
  --target=gfx9-4-generic \
  --output=/tmp/target-provider-gfx9-4-generic.hal \
  --emit-target-artifact=/tmp/target-provider-gfx9-4-generic.hsaco \
  --dump-ir-after=select-templates \
  --dump-ir-format=jsonl \
  --dump-ir-output=/tmp/target-provider-gfx9-4-generic-trace.jsonl
```

Useful query:

```bash
jq -r 'select(.pass == "select-templates" and .changed) | .ir' \
  /tmp/target-provider-gfx11-generic-trace.jsonl \
  /tmp/target-provider-gfx9-4-generic-trace.jsonl
```

Expected signal:

```loom
%subgroup_size = index.constant 32 : index
func.call inline @scale_i32_subgroup_32
%subgroup_size = index.constant 64 : index
func.call inline @scale_i32_subgroup_64
```

Each trace folds the same query and retains exactly one applicable provider
before inlining. The emitted artifacts prove that normalized fact selection
and launch arithmetic compose with the full target pipeline, rather than only
a synthetic selection pass.

## Workgroup-Cluster B128 Multicast

Tags: `__cluster_dims__`, `workgroup cluster`, `cluster multicast`,
`async-to-LDS`, `global_load_lds`, `b128`, `s_wait_asynccnt`, `LDS`,
`recipient-owned LDS`.

HIP habit:

```c++
// Both workgroups in a 1x2x1 cluster execute the same collective transfer.
// Hardware may coalesce the matching requests and writes each recipient's LDS.
cluster_memcpy_async_multicast(destination, source, 16, /*participants=*/0x3);
cluster_async_wait();
__syncthreads();
```

Loom spelling:

```loom
kernel.launch.config workgroups(%one, %two, %one)
    workgroup_size(%thirty_two, %one, %one)
    cluster_size(%one, %two, %one) : index

%copy = kernel.async.cluster.gather %source to %destination using %participants
    {cache_scope = device, cache_temporal = regular}
    : view<16xi8> to view<16xi8>, i32 -> kernel.async.token
%group = kernel.async.group %copy : kernel.async.token -> kernel.async.group
kernel.async.wait %group {newer_groups = 0} : kernel.async.group
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
```

Every selected workgroup participates in the collective operation. The mask
names recipient workgroups; it does not elect one workgroup to issue on behalf
of the others. Each lane names the same logical 16-byte source and the
lane-corresponding destination in its workgroup-owned LDS. The async wait drains
the target's cluster-transfer counter before the ordinary workgroup barrier
makes the populated LDS visible to consumers.

`cluster_b128_multicast.loom` uses a nonuniform iota rather than a constant
payload. Both recipients independently read their own LDS allocation and check
all four words transferred for every lane. That shape catches wrong cluster
identity, wrong recipient remapping, missing destination offsets, truncated
b128 transfers, and accidental validation of only one participant.

Proof command:

```bash
loom-compile cluster_b128_multicast.loom \
  --backend=amdgpu-hal \
  --target=gfx1250 \
  --output=/tmp/cluster-b128-multicast.hal \
  --emit-target-artifact=/tmp/cluster-b128-multicast.hsaco \
  --artifact-manifest=summary \
  --emit-artifact-manifest=/tmp/cluster-b128-multicast.manifest.json \
  --compile-report=summary \
  --compile-report-output=/tmp/cluster-b128-multicast.compile-report.json
```

Useful queries:

```bash
llvm-objdump --disassemble --mcpu=gfx1250 \
  /tmp/cluster-b128-multicast.hsaco

jq '{target_key, workload, local_memory_bytes,
     explicit_action_count: .wait_plan.explicit_action_count,
     barrier_count: .static_instruction_mix.barrier_count}' \
  /tmp/cluster-b128-multicast.compile-report.json
```

Expected target signal:

```text
cluster_load_async_to_lds_b128 ... scope:SCOPE_DEV
s_wait_asynccnt 0
s_barrier_signal
s_barrier_wait
ds_load_b128
```

The compile report and final disassembly prove the selected launch shape,
transfer width, wait domain, barrier placement, and resource viability. Running
the check case through a compatible simulator additionally proves the modeled
recipient and data semantics. Neither is a bandwidth claim: request coalescing,
latency hiding, and the physical multicast ratio require counters and timing on
the target GPU.

## First Translation Questions

Tags: `blockIdx`, `threadIdx`, `workgroup`, `thread`, `lane`, `warp`,
`wavefront`.

Loom names the launch grid and the executing invocation explicitly:
`kernel.launch.config` declares workgroup count and workgroup size,
`kernel.workgroup.id<x/y/z>` reads block IDs, `kernel.workitem.id<x/y/z>` reads
thread IDs, and `kernel.subgroup.*` reads wave/subgroup IDs. Grid and workgroup
range facts come from the launch config. `index.assume` is for bounds or
relations that the launch shape cannot express directly.

Tags: `__global__`, `restrict`, pointer cast, typed pointer, `reinterpret_cast`.

Kernel ABI buffers start as `buffer` and already carry global memory-space
facts. State alias facts with `buffer.assume.noalias` when the source contract
has `restrict`, then form typed views with `buffer.view`. Logical element
coordinates stay in `index`; byte offsets and byte strides use `offset`.
`index.scale` is the explicit boundary from an element coordinate and byte
stride to the byte offset expected by `buffer.view`.

Tags: `global_load_b128`, vectorized load, coalescing, packed load.

At this source level, Loom expects the source to name the load width that
matters. Four nearby scalar loads are not a promise that the compiler will
coalesce them into the same target instruction. Use
`vector.load ... -> vector<4xi32>` when the storage pattern wants a 128-bit
load, and use smaller vectors when that is the intent.

Tags: `__shared__`, `__syncthreads`, `LDS`, `workgroup memory`.

Use `buffer.alloca` with `memory_space = workgroup`, form typed views with
`buffer.view`, and put a `kernel.barrier<workgroup>` between producer and
consumer phases when different work items communicate through the tile. The
compile report should record local memory bytes and local memory operations;
`source_low_memory` detail rows should explain selected LDS packet bank
behavior; target listings should show LDS instructions on AMDGPU.

Tags: `fma`, `contract`, `fast-math`, `v_fma`, `v_dot`, `dot4`.

Use Loom fast-math flags on arithmetic when contraction/reassociation is part
of the intended target shape. Then inspect compile reports and target listings
instead of assuming an operation selected the same instruction as HIP C++.

Tags: `template`, wave size, subgroup size, targetless specialization.

Use `template.apply<@contract>` at call sites and `template.def<@contract>`
providers for implementations. Normalized requirements such as subgroup size
use typed `requires` clauses without naming a backend or target. Keep a correct
lower-priority fallback for profiles that cannot prove a specialization, and
reserve `target(@...)` for implementations that truly require one target
identity. `target_provider_selection.loom` compiles the same targetless kernel
for wave32 and wave64 profiles. The authoring/linking corpus shows the full
library and bytecode flow.
