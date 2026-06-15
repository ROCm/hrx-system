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
| `threadIdx`, `lane_id`, `warp lane`, `wavefront lane`, `SGPR`, `VGPR`, `EXEC`, `scalarized`, `lane-varying`, `uniform`, `dot operand` | value distribution facts, `test.fact_uniform`, `test.fact_lane_varying`, `test.fact_lane_predicate` | `lane_distribution.loom` |
| `__global__`, `restrict`, pointer casts, address arithmetic | kernel ABI `buffer`, `buffer.assume.noalias`, `buffer.view`, `index`/`offset` math | `q8_block_unroll.loom` |
| `global_load_b32`, packed bytes, `q8`, bitfield, unpack | `vector.load`, `vector.bitunpacks<8>`, `scalar.extf`, `vector.dotf` | `q8_block_unroll.loom` |
| `global_load_b32`, `global_load_b128`, `uint4`, adjacent scalar loads, coalescing | `vector.load -> vector<1xi32>` versus `vector.load -> vector<4xi32>` | `q8_load_width.loom` |
| `q8`, `q4`, `u8`, `s8`, `u4`, `s4`, `v_dot4_i32_iu8`, `dp4a` | `vector.bitunpacku`, `vector.bitunpacks`, `vector.dot4i<u8s8>` | `q8_q4_signedness.loom` |
| `q2`, `q3`, `q4`, `q5`, `q6`, split high bits, lookup tables, offset-binary, packed-dot repack | `vector.bitunpacku`, `vector.bitfield.extractu`, `vector.bitfield.insert`, `vector.table.lookup`, `vector.dot8i4` | `packed_field_contracts.loom` |
| dynamic lower-bound unroll, missing facts | structured diagnostic, exact/range facts | `q8_hip_shaped_unroll_unresolved.loom` |
| `#if __gfx*__`, template, macro, arch-specialization, fallback | `func.apply`, `func.template`, `target(@...)`, `priority(...)` | `target_provider_selection.loom` |

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
%input_words = buffer.view %input_noalias[%base] : buffer -> view<4xi32, #dense>

%w0 = vector.load %input_words[0] : view<4xi32, #dense> -> vector<1xi32>
%w1 = vector.load %input_words[1] : view<4xi32, #dense> -> vector<1xi32>
%w2 = vector.load %input_words[2] : view<4xi32, #dense> -> vector<1xi32>
%w3 = vector.load %input_words[3] : view<4xi32, #dense> -> vector<1xi32>

%wide = vector.load %input_words[0] : view<4xi32, #dense> -> vector<4xi32>
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
%w0 = vector.load %input_words[0] : view<4xi32, #dense> -> vector<1xi32>
%wide = vector.load %input_words[0] : view<4xi32, #dense> -> vector<4xi32>
```

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
%q8_word = vector.load %q8_words[0] : view<1xi32, #dense> -> vector<1xi32>
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
iree-benchmark-loom packed_field_contracts.loom --dry-run
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

## Target Provider Selection

Tags: `HIP template`, `CUDA template`, `macro`, `#if __gfx1100__`,
`arch-specialization`, `gfx1100`, `gfx1200`, `fallback`, `func.apply`,
`func.template`, `target(@...)`, `priority`.

HIP habit:

```c++
#if __gfx1100__
  return scale_i32_gfx1100(value);
#elif __gfx1200__
  return scale_i32_gfx1200(value);
#else
  return scale_i32_fallback(value);
#endif
```

Loom spelling:

```loom
amdgpu.target<gfx1100> @gfx1100
amdgpu.target<gfx1200> @gfx1200

func.template<hip.recipe.scale_i32> target(@gfx1100) priority(20) @scale_i32_gfx1100(%value: i32) -> (i32) { ... }
func.template<hip.recipe.scale_i32> target(@gfx1200) priority(20) @scale_i32_gfx1200(%value: i32) -> (i32) { ... }
func.template<hip.recipe.scale_i32> priority(1) @scale_i32_fallback(%value: i32) -> (i32) { ... }

func.def public target(@gfx1100) @selects_gfx1100_provider(%value: i32) -> (i32) {
  %scaled = func.apply<hip.recipe.scale_i32>(%value) : (i32) -> (i32)
  func.return %scaled : i32
}
```

`func.apply<contract>` is the call-site demand. `func.template<contract>` rows
are providers. `target(@...)` filters applicability, and `priority(...)` orders
providers after signature, target, and predicate filtering. A targetless
provider is the generic fallback.

Proof command:

```bash
loom-opt target_provider_selection.loom \
  --pass=select-templates,inline-callables \
  --pass-report=json \
  --output=/tmp/target-provider-selected.loom \
  2>/tmp/template-selection-report.json
```

Useful query:

```bash
jq '.invocations[]
  | select(.pass == "select-templates")
  | .details[]
  | select(.category == "template-selection")
  | {outcome, contract, target, selected_provider, provider_count, target_applicable_count, best_exact_count}' /tmp/template-selection-report.json
```

Expected signal:

```json
{"outcome":"selected","contract":"hip.recipe.scale_i32","target":"gfx1100","selected_provider":"scale_i32_gfx1100","provider_count":3,"target_applicable_count":2,"best_exact_count":1}
```

The transformed file should contain the selected gfx1100 implementation body,
not `func.apply`, `@scale_i32_gfx1200`, or `@scale_i32_fallback`.

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

Tags: `fma`, `contract`, `fast-math`, `v_fma`, `v_dot`, `dot4`.

Use Loom fast-math flags on arithmetic when contraction/reassociation is part
of the intended target shape. Then inspect compile reports and target listings
instead of assuming an operation selected the same instruction as HIP C++.

Tags: `template`, `macro`, `gfx1100`, `gfx1200`, `arch-specialization`.

Use `func.apply<contract>` at call sites and `func.template<contract>`
providers for implementations. Target-specific providers use `target(@...)`
and priority; generic fallbacks stay correct for targets without a specialized
provider. `target_provider_selection.loom` shows the source-level selection
proof. The authoring/linking corpus shows the full library and bytecode flow.
