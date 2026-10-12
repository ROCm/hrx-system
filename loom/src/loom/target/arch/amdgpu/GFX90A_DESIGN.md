# Native gfx90a backend design

Status: core compiler implemented; hardware execution suites pass on an MI210.
The chemistry production kernel does not validate on the current baseline: its
advance kernel fails register allocation (BACKEND/005) on gfx90a and gfx942
alike. It last matched the HIP reference for 20 steps on an earlier baseline
with loop-invariant hoisting disabled. Matrix operations are out of scope for
gfx90a. Qualified on `bb0882812c` plus two local fixes, a whole-wave SGPR
spill fix and a scratch address fix. Upstream has since superseded both, with
#1594 and #1591 respectively, and the local fixes are dropped. Requalified on
`3e950c5806` with the hardware suites; chemistry cannot be rerun there because
the advance kernel does not compile. References checked on 2026-10-11. See [Implementation record](#implementation-record).

Add `amdgpu:gfx90a` to Loom's native AMDGPU compiler for AMD Instinct MI200
devices. Reuse the shared lowering, allocation, scheduling, and code-object
writer, with a distinct CDNA2 instruction descriptor set and processor profile.
The first deliverable is a usable scalar/vector compute backend, including
FP64 and automatic spilling, qualified against the chemistry workload. Matrix
lowering and performance qualification are subsequent deliverables.

The principal integration risk is scratch: gfx90a requires an absolute flat
scratch base initialized by the kernel, whereas gfx942 uses architected flat
scratch. A descriptor-set alias to CDNA3 cannot establish correct execution.
The CDNA2 manual section 10.8 and LLVM's processor and flat-scratch descriptions
agree on this distinction. [AMD ISA][amd-isa] [LLVM ABI][llvm-scratch]

## Scope and observable behavior

The completed core backend must compile target-free Loom kernels, including
C++ importer output, to native gfx90a assembly and HSACO artifacts using the
existing tool interface. This is the intended invocation after implementation:

```sh
loom-compile kernel.loom \
  --root=@kernel \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx90a \
  --output=kernel.hsaco
```

Use the existing target-ID normalization for `sramecc` and `xnack`; preserve
the selected feature states in artifact identity and ELF metadata. Initially
qualify the non-XNACK execution configuration available on the test device.
An accepted but untested feature configuration is not hardware qualification.
Do not advertise an XNACK configuration until replay-sensitive allocation and
scheduling have been checked for it; reject configurations whose correctness
requirements are not implemented.

Core support includes integer and FP32/FP64 arithmetic, comparisons and
conversions, existing applicable math recipes, global/buffer memory, LDS,
barriers, divergent control flow, private storage, and automatic register
spilling. Device-scope relaxed integer add and compare-exchange are required
for the chemistry integration. Additional atomic orders, scopes, and floating
types are enabled only with complete lowering and tests.

Matrix operations may initially report deliberate unsupported-operation
diagnostics. Unsupported instructions and memory contracts must fail before
native emission; they must never silently select a CDNA3 packet or weaken an
authored ordering requirement. Existing target-free source does not acquire
new gfx90a-specific annotations.

The implementation does not add an LLVM code-generation dependency, a new
runtime driver, or a new importer. LLVM is an independent validation tool.
Graphics instructions, kernel argument preloading, and thread-group splitting
are outside the initial contract. Keep preloading and splitting disabled until
their separate ABI and memory contracts are qualified.

## Existing implementation and missing pieces

The baseline already contains the following reusable pieces:

| Component | Existing support | Work required |
| --- | --- | --- |
| [Target catalog](../../../../../py/loom/target/arch/amdgpu/target_catalog.py) | Exact gfx90a identity and target-ID features | Retain exact identity; connect it to a supported compiler target |
| [Target facts](../../../../../py/loom/target/arch/amdgpu/target_info.py) | ELF machine `0x03f`, wave64 default, packed work-item ID marker, gfx90a matrix feature inventory | Complete processor properties, descriptor set, target row, and HSACO enablement |
| [Vendor ISA archive](../../../../../build_tools/amdgpu/isa_xml/archive.bzl) | Pinned archive includes `amdgpu_isa_cdna2.xml` | Consume it in descriptor and encoding generation |
| [CDNA descriptors](../../../../../py/loom/target/arch/amdgpu/descriptors/cdna.py) and [overlays](../../../../../py/loom/target/arch/amdgpu/descriptors/sets.py) | CDNA3/CDNA4 instruction families and register models | Add CDNA2 composition and audit every reused operand form |
| [Memory coherence lowering](lower/memory_coherence.c) | Current `CDNA` ordering row uses SC0/SC1 | Introduce a gfx90a ordering model and cache-policy encoding |
| [Kernel ABI](hal/kernel_abi.c) and [native descriptor writer](../../emit/native/amdgpu/descriptor.c) | Kernargs, launch inputs, descriptor fields, metadata | Integrate scratch launch inputs, special-register reservations, and prologue |
| [Native entry envelope](../../emit/native/amdgpu/kernel_entry.h) | Processor-selected immutable entry packets | Support entry materialization from the retained ABI and scratch plan |
| [Runtime identity catalog](../../../../../../runtime/src/iree/hal/drivers/amdgpu/target/identity_catalog.inl) | gfx90a recognition; runtime device binaries also exist | Test actual Loom artifacts through loading and dispatch |

The current gfx90a processor entry uses `gfx9_10_processor_info`, leaving the
descriptor-set key empty and HSACO emission disabled. It is absent from
`AMDGPU_TARGET_INFOS`. The local compiler probe rejects the target before
kernel lowering. Identity and matrix capability records therefore do not
establish compiler support.

## Reference contract and cross checks

Use AMD's *AMD Instinct MI200 Instruction Set Architecture Reference Guide*,
dated 4 February 2022. Page numbers below are printed page numbers; PDF page
indices differ. The XML archive is separately pinned to the 2026-03-05 release
in the repository; its CDNA2 document identifies architecture ID `1`, schema
`1.1.1`, and release date `2026-02-20`.

The manual owns instruction semantics and hazards; the pinned XML supplies
machine-readable packet facts. LLVM's AMDGPU usage guide supplies the AMDHSA
software ABI and a cross-check for memory lowering. LLVM MC tests encoding,
not execution correctness. If these disagree, retain the discrepancy and a
focused reproducer instead of choosing whichever source makes generation pass.

| Topic | AMD manual location | LLVM cross-check | Design consequence |
| --- | --- | --- | --- |
| Scratch | Sections 10.5 and 10.8, pp. 84–85: kernel-composed scratch base | [Processor properties][llvm-processors] and [Flat Scratch][llvm-scratch]: absolute versus architected scratch | Separate gfx90a descriptor flags and an ABI-aware scratch entry sequence |
| Launch coordinates | Chapter 3 describes architectural state; it is not the HSA launch ABI | [Initial kernel state][llvm-initial]: packed work-item IDs | Reuse packed-ID extraction with multidimensional execution tests |
| Register layout | Section 3.6, pp. 14–15 | [Kernel descriptor][llvm-descriptor] and [operand syntax][llvm-operands]: accumulator accounting and even-aligned vector/accumulator pairs | Test register boundaries, reservations, and descriptor accounting together |
| Cache policy and atomics | Sections 9.1.10 and 10.3.1, pp. 67–68 and 83 | [GFX90A memory sequences][llvm-memory] | Separate cache-policy bits from atomic return control; generate only supported order/scope combinations |
| Completion and hazards | Sections 4.4–4.5, pp. 22–24; 10.2, pp. 82–83 | GFX90A memory sequences and target-specific LLVM assembly | Qualify wait counters and software delays independently of instruction encodings |
| LDS and M0 | Sections 3.6.1 and 10.6 mention M0 bounds | [LLVM M0][llvm-m0] says ordinary GFX9+ LDS accesses need no M0 range setup | Record the discrepancy; follow the GFX9 ABI for ordinary LDS and test separately from explicit M0 operands |
| Floating point | Section 3.5, p. 13, and Chapter 6 | Kernel descriptor floating-point controls; instruction operand/modifier validation | Preserve authored math policy and verify exceptional inputs on hardware |

Two details require particular care. Generic FLAT operations have different
completion behavior from known GLOBAL/SCRATCH operations (manual section 10.2).
Also, ECC can affect the untouched half of a D16 destination (section 9.1.6).
The proposed tests must exercise both distinctions. [AMD ISA][amd-isa]

The current LLVM processor table limits kernarg preload availability within
gfx90a devices. Keeping it disabled avoids making processor identity stand in
for a device-specific capability. [LLVM processors][llvm-processors]

## Target facts and generated tables

Add `AMDGPU_DESCRIPTOR_SET_ISA_CDNA2` with the checked XML architecture name
and ID, plus a distinct `amdgpu.cdna2.core` descriptor-set record. Add a gfx90a
processor helper or explicit processor row with its complete ABI, instruction,
memory, scheduling, occupancy, and limit properties. Add a stable new target
enum value without renumbering existing targets.

Keep Python as the source of truth for these facts. Extend the checked-in C
types in `target_info_defs.h` only where existing representations cannot express
the new data. Generators emit data rows, not target-specific control flow.
Do not use numeric processor comparisons or processor-name tests in lowering.

Construct CDNA2 descriptors from validated reusable families. The existing
`_cdna_core_overlays` includes newer memory/cache and matrix choices, so calling
it unchanged is not sufficient. Factor shared families where their semantic
and packet contracts really match; compose CDNA2-specific memory and matrix
variants explicitly. Emit a generation-time coverage report listing each
required descriptor, XML instruction/form, and missing or excluded capability.

Validate packet fields, operand widths, literal handling, special-register
uses/defs, alignment, immediate ranges, spelling, and feature predicates.
Instruction-name overlap alone does not justify reuse. Unsupported newer
matrix formats, sparse operations, conversion forms, and cache instructions
must not enter the CDNA2 set through an inherited overlay.

Wire the set through target table generation, descriptors, encoding, contracts,
aliases, references, and build selection. Update source Bazel definitions and
regenerate their CMake counterparts using the repository workflow. Regenerate
Loom selector fragments through `loom/build_tools/amdgpu/target_config.py`.
Do not broaden gfx90a into the `gfx9-4-generic` artifact family.

## Kernel ABI and scratch lifecycle

Define a gfx90a kernel-descriptor profile using the existing GFX9 descriptor
format, with accumulator and floating-point controls selected independently
from scratch handling. In particular, do not enable
`ARCHITECTED_FLAT_SCRATCH`. Retain the packed work-item ID capability.

The proposed scratch implementation uses native SCRATCH instructions and the
absolute flat-scratch ABI. Keep buffer-based scratch out of the initial path
unless it is needed to satisfy an existing public operation contract.
For scratch-using kernels, request the flat-scratch initialization pair and
per-wave scratch offset, add them with 64-bit carry propagation, and install
the result in the reserved `FLAT_SCRATCH` pair. Derive their incoming positions
from the enabled launch-input sequence. [LLVM scratch ABI][llvm-scratch]

The kernel ABI owner must produce one retained entry plan containing:

- enabled launch inputs and their physical locations;
- scratch initialization input locations and reserved special registers;
- scratch use and final per-work-item private size;
- descriptor enable bits and SGPR accounting;
- entry instructions and their temporary/clobbered registers.

Scratch can first appear during register allocation. Reserve the potential
scratch ABI locations before allocation and finalize scratch use from the
existing allocation/storage result. Do not discover it by rescanning emitted
instructions. The implementation must establish a stable launch layout across
this transition: if enabling scratch changes incoming register positions, the
owning pipeline must invalidate and rebuild the affected ABI/allocation plan
before scheduling or emission. Silently moving launch inputs after allocation
is forbidden. The exact reservation strategy is a blocking design detail for
the first implementation milestone; measure its register cost on spill-free
kernels before accepting a permanent conservative reservation.

Extend the native entry-envelope API to materialize packets from that retained
plan. The existing processor-only static envelope cannot encode varying launch
input locations. Assembly and binary emission must consume the same finalized
entry plan and encoding machinery, including resource floors and text-fixup
displacement. Account for SCC and every temporary the entry sequence clobbers.
Emit scratch initialization before any body or spill access that depends on it.

Validate the no-scratch and scratch cases separately. For scratch kernels,
cross-check initialized inputs, special-register reservations, descriptor bits,
SGPR count, private size, and actual addresses observed by different lanes and
waves. Exercise automatic SGPR, VGPR, predicate-mask, and eventually AGPR
spilling. Explicit private arrays alone do not test automatic spill insertion.

The scope of this change stays within AMDGPU where possible. Any necessary
shared allocation or entry-planning interface must be target-independent,
consumed by all applicable existing targets, and replace the old path in the
same change. No second allocator, hidden IR marker, or backend-owned whole
function analysis is introduced.

## Memory ordering and scheduling

Introduce a distinct gfx90a memory-ordering selector and cache-policy encoding.
The present `LOOM_AMDGPU_MEMORY_ORDERING_MODEL_CDNA` explicitly models SC0/SC1;
renaming gfx90a's descriptor set does not adapt it. Reuse the existing memory
coherence interpreter after extending its data representation where needed.

For each exposed memory contract, record address space, operation, order,
scope, return-value requirement, cache controls, waits, invalidations, and
writebacks. Cross-check against LLVM's GFX90A table for the chosen execution
mode. Keep load/store cache policy and atomic return control as distinct
semantic inputs. A wait for completion cannot substitute for a missing cache
operation. Device memory, host-visible memory, and peer memory need separate
qualification; device-local integer atomic tests cannot qualify system-scope
floating-point atomics.

Use the existing descriptor resource/effect model and shared scheduler for
completion, source retention, and required instruction spacing. Audit every
enabled instruction against manual sections 4.4–4.5 and 10.2. Give generic
FLAT accesses conservative completion rules until a narrower rule is proven.
Cover scalar/vector dependency transitions, EXEC/VCC updates, wide store
payload retention, scratch address/data reuse, and MFMA dependencies when
matrix support is enabled.

Separate correctness constraints from performance estimates. Required delays
must be supported by an ISA rule or independently qualified behavior.
Throughput and latency estimates may start conservatively, with their provenance
recorded; do not import gfx942 matrix timings or LDS service models as measured
gfx90a facts. Use a gfx90a occupancy model and compare boundary cases against
[LLVM's occupancy tool][llvm-occupancy] and hardware resource metadata.

## Arithmetic and matrix lowering

Reuse existing source-to-low contracts by descriptor availability. Audit the
complete closure of FP64 add/multiply/FMA, comparison, conversion, division,
sqrt, and the exp/log/cbrt recipes used by chemistry. An available top-level
instruction does not establish availability of a recipe's helper instructions.

Preserve the caller's contraction, approximation, rounding, and denormal
requirements. Qualify strict recipes against independent high-precision
references, including signed zero, infinities, NaNs, subnormals, range-reduction
boundaries, overflow, and underflow. HIP output provides a comparison, not the
definition of acceptable numerical behavior. Ordinary relaxed math keeps its
existing policy and must not acquire strict-mode costs.

For the later matrix milestone, connect the existing gfx90a feature inventory
and `matrix_contracts.py` rows to real CDNA2 descriptors. Validate fragment
layouts, operand/accumulator banks, AGPR transfers, accumulator offsets, tuple
alignment, and source retention. Do not infer coverage from capability bits.
Require elementwise numerical checks and resource-boundary tests for each
enabled family before performance tuning.

## Implementation milestones

These are development gates, not a requirement to publish dormant registration
or table-only PRs. Keep dependent work together until the advertised behavior
works end to end; split review boundaries only at independently valid contracts.

1. **Core execution.** Add target/table/build integration and the minimum
   instruction closure, then complete the ABI, scratch lifecycle, hazards,
   and integer/FP32/FP64 memory kernels. Acceptance requires a loadable HSACO
   and correct native execution, including forced automatic spills.
2. **Chemistry and core qualification.** Complete required math and integer
   atomics; execute both chemistry roots with independent numeric and exact
   integer/status checks. Add divergence, LDS/barrier, memory ordering, feature
   identity, and resource-boundary tests. Publish core support only with an
   explicit supported-operation contract and precise diagnostics outside it.
3. **Matrix support.** Enable and qualify the gfx90a matrix families, including
   AGPR pressure and spilling. Preserve the core workload's correctness and
   compile-time/resource behavior.
4. **Performance qualification.** Tune measured occupancy, scheduling, memory
   forms, and spill behavior against HIP and relevant assembly baselines.
   Record numerical error and timing independently for every tuning decision.

## Validation and acceptance

| Layer | Required evidence | Claim established |
| --- | --- | --- |
| Python generation | Descriptor/XML consistency, target/profile validation, deterministic tables, selector checks | Internal data consistency |
| Source lowering | Common corpus instantiated for gfx90a; precise rejection tests; readable assembly round trips | Supported semantic lowering and diagnostics |
| LLVM oracle | Exact packet byte comparison and disassembly for the enabled forms; descriptor/metadata checks | Encoding and ABI representation agreement for a recorded toolchain |
| Native kernels | Scalar/vector math, multidimensional IDs, LDS, barriers, divergence, atomics, explicit and automatic private storage | Correctness on the tested device/configuration |
| Integration | Both chemistry roots; integer/status/counter equality; independently specified FP64 tolerances and difficult inputs | Workload correctness on that configuration |
| Performance | Runtime and compile time, memory use, register/private sizes, timing distributions and numerical error | Measured performance within the recorded error envelope |

Use the existing [LLVM MC oracle workflow](../../../../../docs/src/workflows/oracles/llvm-mc.md).
Qualify the oracle binaries and record their paths, hashes, versions, processor,
and feature flags. Compare the same instruction operands, modifiers, and literal
forms. Include negative tests for newer instructions and illegal register
alignments. Never treat an unknown-processor warning or target fallback as an
oracle pass.

Hardware tests must cover multiple waves and workgroups, partial active masks,
large offsets, wide loads/stores, repeated launches, and near-limit register/LDS
usage. Run scratch tests with distinct per-lane/per-wave values and guard
regions so address aliasing cannot pass through coincidentally equal data.
Test emitted assembly and direct HSACO paths for equivalent descriptors and
results. Run representative existing gfx942/gfx950 and RDNA regressions after
shared family changes.

Memory-order tests include producer/consumer publication within each advertised
scope, including across workgroups for device scope. They supplement the documented sequence audit;
stress tests alone cannot prove a weak-memory mapping. Record GPU model, target
ID, feature state, driver/runtime, launch dimensions, input seeds, source and
compiler revisions, artifacts, and full failure logs.

Acceptance requires all core gates, with matrix scope labeled separately. An
emitted ELF, a passing LLVM comparison, or CPU execution is not gfx90a runtime
evidence. This design records requirements; it does not claim hardware testing
or implementation completion.

## Outstanding decisions and risks

- **Late scratch activation:** settle launch-input reservations and allocation
  invalidation before landing core support. This is the largest architectural
  integration risk, particularly for kernels whose only private storage is
  introduced by spilling.
- **Manual and XML differences:** preserve the pinned XML identity and document
  any overlay correction with a minimal LLVM comparison and relevant manual
  section. The 2022 prose and 2026 XML are different revisions.
- **LDS M0 discrepancy:** test ordinary DS/FLAT LDS behavior against the GFX9
  ABI. Explicit M0-consuming instructions retain their own dependencies; the
  ordinary-LDS decision must not remove those dependencies.
- **Atomics coverage:** resolve instruction, return, type, and memory-domain
  support individually before enabling each contract. Do not generalize a
  statement about one atomic family to all global or LDS operations.
- **XNACK and ECC:** qualify replay and D16 behavior under each advertised
  feature state; retain conservative behavior where the artifact permits
  multiple states.
- **Hardware access:** core completion requires a gfx90a device. Offline work
  can close table, lowering, and encoding gates while execution remains open.

## Implementation record

This section records what the core milestone implements, the decisions taken
for the open design points, and the evidence collected offline and on an
MI210 (gfx90a:sramecc+:xnack-, ROCm 10.1).

### Target facts and descriptors

- `amdgpu:gfx90a` is target enum 25. It selects the `amdgpu.cdna2.core`
  descriptor set, built from the pinned CDNA2 XML (architecture ID 1). It
  enables HSACO emission, uses the GFX9 descriptor profile without
  `ARCHITECTED_FLAT_SCRATCH`, and uses the CDNA3-equivalent occupancy model.
  The model matches `llc -mcpu=gfx90a` at the 64/65/129/256-VGPR boundaries.
- The CDNA2 and CDNA3 XMLs share 2365 instruction encodings with identical
  opcodes and operands. Their memory packets differ only in field names
  (GLC/SLC/SCC versus SC0/NT/SC1, at the same bit positions) and in FLAT bit 13
  (LDS versus SVE).
- CDNA2 reuses the CDNA overlay composition with explicit parameters: GLC/SLC/SCC
  cache fields, GLC atomic return control, no SVE field, no offset-only (ST
  mode) scratch, integer-only global/flat atomics, and no LDS float add. It also
  omits FMAAK/FMAMK, FP8 conversions, LDS-direct loads, and matrix packets, and
  uses the CDNA2 cache-control set (`buffer_wbl2`, `buffer_invl2`,
  `buffer_wbinvl1_vol`).
- `_CDNA2_CDNA3_COVERAGE_EXCLUSIONS` in `descriptors/sets.py` is the coverage
  report. Every CDNA3 descriptor that CDNA2 omits matches exactly one row.
  Generation validates each row against the CDNA2 XML: `absent_from_isa`
  packets must not exist there, and `withheld` packets must exist. Today the
  table covers 31 absent, 39 withheld, and 46 out-of-scope matrix rows.
- The gfx90a processor row withholds the MFMA feature profile. Fragment and
  `vector.mma` operations fail with AMDGPU/021 and AMDGPU/023 before native
  emission.
- MI200 erratum `Shift64HighRegBug`: the VGPR shift-amount operand of
  `v_lshlrev_b64`, `v_lshrrev_b64`, and `v_ashrrev_i64` is even-aligned on
  CDNA2, so it can never occupy the last register of an eight-VGPR block.
- Explicit `xnack+` is rejected with AMDGPU/052
  (`AMDGPU_PROCESSOR_INFO_FLAG_REJECT_XNACK_ON`). The default `any` state and
  `xnack-`/`sramecc±` are accepted and preserved in the code-object identity.

### Kernel ABI and scratch lifecycle

The late-scratch decision uses a stable launch layout rather than invalidation:

- Every gfx90a HAL kernel requests the flat-scratch-init user SGPR pair, after
  the other enabled user SGPRs. The kernel ABI owner records it as
  `flat_scratch_init_sgpr`, so workgroup-ID live-ins never move when scratch
  first appears during allocation.
- The private-segment wave offset is the last AMDHSA system SGPR. The kernel
  record enables it only when the finalized storage layout has a private
  segment, so enabling it moves no other input.
- `loom_amdgpu_kernel_entry_plan_t` is retained on the kernel record. When a
  private segment exists, it materializes
  `s_add_u32 flat_scratch_lo, s[init], s[wave_offset]` and
  `s_addc_u32 flat_scratch_hi, s[init+1], 0` after any profile packets and
  before the body. Assembly text and binary emission consume the same envelope.
  The encodings match llvm-mc. The sequence clobbers only SCC and
  FLAT_SCRATCH, neither of which is live into the body.
- The SGPR descriptor accounting reserves VCC, XNACK_MASK, and FLAT_SCRATCH
  (6 extra) when the entry initializes flat scratch, as LLVM's
  `getNumExtraSGPRs` does. `.amdhsa_reserve_flat_scratch` follows the same
  decision.
- Measured cost of the permanent reservation on 71 scratch-free corpus kernels
  compared with gfx942: 65 have identical VGPR/SGPR granules. The other 6 use one
  fewer SGPR granule, because they reserve no FLAT_SCRATCH. The reservation
  never increased a granule, and its SGPRs are reusable after entry.
- Spill lowering uses VADDR scratch forms when a descriptor set has no
  offset-only form.

### Memory ordering

`LOOM_AMDGPU_MEMORY_ORDERING_MODEL_GFX90A` follows LLVM's GFX90A table for
non-TgSplit mode:

- Device and system observations set GLC. Stores and update atomics carry no
  coherence bits.
- System release issues `buffer_wbl2` and waits for `vmcnt(0)`.
- Device acquire issues `buffer_wbinvl1_vol`. System acquire issues
  `buffer_invl2` and then `buffer_wbinvl1_vol`. Each invalidation carries its
  own minimum scope (`invalidate_scopes`).
- The cache-policy encoding `gfx90a_glc_slc_scc` accepts only the regular
  device policy until SLC/SCC behavior is qualified.
- Generic FLAT accesses wait on both `vmcnt` and `lgkmcnt`.
- CDNA D16 loads are not treated as preserving the other half, because of the
  ECC behavior.

### Evidence collected offline

- Generator/Python suites, all AMDGPU CTest suites (826 tests), and new
  `.loom-test` fixtures covering the following:
  - atomic observations, integer atomics, and the float compare-exchange loop;
  - matrix rejection;
  - the scratch entry and a scratch-free kernel;
  - the shift erratum, XNACK rejection, FLAT completion, and occupancy
    boundaries.
- Encoding oracle: 71 source-lowering corpus kernels and both chemistry roots
  were compiled for gfx90a. Every distinct emitted instruction (941 in the
  corpus, 64,551 in the advance kernel, 1,145 in the prepare kernel)
  re-assembles byte-identically with `llvm-mc -mcpu=gfx90a`.
- Chemistry: both roots compile. The advance kernel's 45,744-byte private
  segment (including automatic spills), register granules, and 28,223 scratch
  accesses match the gfx942 build. The instruction streams differ only in the
  entry sequence, the workgroup-ID register, and the GLC spelling of the atomic
  return bit.

### Hardware evidence

On an AMD Instinct MI210:

- The chemistry comparison harness, built with `hipcc --offload-arch=gfx90a`,
  runs the HIP reference and Loom's kernels on the same 262,144-cell grid.
  Twenty full gridwide steps pass at rtol 2e-4; the final grid time matches the
  reference to every printed digit. Loom takes 455.8 ms per advance step
  against 183.5 ms for HIP (2.48x).
- The chemistry result above was produced on baseline `f436a5e57b` plus the
  local SGPR spill and scratch address fixes, with loop-invariant hoisting
  disabled locally. The harness dispatches
  through a HIP runtime, not the HAL `amdgpu` driver. Which HIP library it
  linked, ROCm's or HRX's, was not recorded.
- The hardware test suites run through the HAL `amdgpu` driver
  (`--device=amdgpu`), not HIP. The driver loads ROCm 10.1's
  `libhsa-runtime64` (ROCr) for agent discovery, memory pools, queue and
  signal creation, and code-object loading. It builds and submits the AQL
  packets itself. These results therefore cover Loom HSACOs passing ROCr's
  loader and executing under the driver's own dispatch path.
- On baseline `bb0882812c` plus the same two local fixes, 64 of 66
  `loom-executor=hardware` tests pass; the two failures are `repeated_wmma`,
  which requires matrix support. Of the 9 `requires-gpu-amd` tests, the
  generic corpus and execution tests pass; the 5 failures are corpus
  selections compiled for other architectures (gfx11, gfx1100, gfx9_4,
  gfx1250).
- On baseline `3e950c5806`, which carries upstream's #1594 and #1591 instead of
  the local fixes, a fresh Release build with ROCm 10.1 clang gives the same
  results: 64 of 66 `loom-executor=hardware` and 4 of 9 `requires-gpu-amd`
  tests pass, with the same failures.
- On baseline `3e950c5806`, the chemistry advance kernel fails register
  allocation (BACKEND/005) with hoisting enabled: peak 439 VGPRs on gfx90a and
  278 on gfx942, against 256.
- On baseline `bb0882812c`, the chemistry advance kernel fails register
  allocation (BACKEND/005, peak 558 VGPRs against 256) on gfx90a and gfx942,
  even with hoisting disabled, so chemistry does not validate there.

### Defects found on hardware

Three defects passed every offline check and failed only on the device:

- Kernarg/flat-scratch layout: requesting the flat-scratch-init pair without
  the kernarg pointer left the user SGPR count below the count the packet
  processor implies. The kernarg pointer is now reserved whenever flat-scratch
  init is requested.
- `scalar.divf` selected an FMAAK form that CDNA2 lacks. Division now
  materializes the constant one in a register.
- Whole-wave SGPR spill clobber: SGPR spill and reload traffic writes VGPR
  temporaries with EXEC widened to every lane. Lanes retired from a divergent
  loop still held live values in those registers, so chemistry diverged at
  step 15. This affects every AMDGPU target. The hardware results above used a
  local fix that pinned the temporaries in a two-VGPR window ordinary values
  never occupy. Upstream fixed the defect in #1594 ("Preserve inactive-lane
  state across wave-wide spills"), which reports every VGPR defined under
  widened EXEC as point-local interference, so allocation moves only a helper
  whose definition would overwrite retained storage. The local fix is dropped
  in favor of #1594.

Rebasing also exposed an AMDGPU scratch addressing defect on every target:
countdown-indexed private accesses put a negative dynamic term in the unsigned
scratch address register. Upstream fixed this in #1591 ("Keep scratch VADDR
offsets in domain"), which routes every scratch dynamic term through VADDR and
keeps the static offset in the immediate only when the remaining VADDR is
proven to be a valid unsigned 32-bit offset.

### Open items

- Confirm #1594 on chemistry: the hardware suites pass on `3e950c5806`, but the
  20-step chemistry comparison last ran with the dropped local SGPR spill fix.
  Once the advance kernel compiles again, rerun it on the MI210 to confirm #1594
  covers the step-15 divergence.
- The chemistry production kernel does not validate on the current baseline:
  the advance kernel fails register allocation (BACKEND/005) on gfx90a and
  gfx942. On `f436a5e57b`, bisection identified loop-invariant hoisting in the
  default source pipeline (#1548), and disabling it restored compilation. On
  `bb0882812c` the kernel fails even with hoisting disabled. A second
  bisection, with hoisting disabled, identifies the low-scheduler setup
  priority change in #1561 (`78964ed8fe`); its parent compiles. Requalifying
  chemistry needs a pressure bound for hoisting and a fix for the #1561
  regression.
- Hardware-suite selections for other architectures still run, and fail, on
  a gfx90a device; they need an architecture gate.
- Performance: Loom is 2.5x slower than HIP on chemistry.
- Float atomics, LDS float add, LDS-direct loads, SLC/SCC cache policies, and
  XNACK-on replay remain withheld until each is qualified.

## References

- [AMD CDNA2 ISA manual][amd-isa], 4 February 2022; sections cited above.
- [AMD GPU architecture documentation index][amd-index], current manual link.
- [Pinned AMD machine-readable ISA archive][amd-xml]; repository pin and SHA256
  are in `loom/build_tools/amdgpu/isa_xml/archive.bzl`.
- [LLVM AMDGPU processor table][llvm-processors], [kernel descriptor][llvm-descriptor],
  [initial kernel state][llvm-initial], [flat scratch][llvm-scratch],
  [M0][llvm-m0], and [GFX90A memory-model sequences][llvm-memory].
- [LLVM AMDGPU operand syntax][llvm-operands] and [occupancy calculator][llvm-occupancy].

[amd-isa]: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf
[amd-index]: https://rocmdocs.amd.com/en/latest/reference/gpu-arch/index.html
[amd-xml]: https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2026_03_05.zip
[llvm-processors]: https://llvm.org/docs/AMDGPUUsage.html#processors
[llvm-descriptor]: https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor
[llvm-initial]: https://llvm.org/docs/AMDGPUUsage.html#initial-kernel-execution-state
[llvm-scratch]: https://llvm.org/docs/AMDGPUUsage.html#flat-scratch
[llvm-m0]: https://llvm.org/docs/AMDGPUUsage.html#m0
[llvm-memory]: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-memory-model-code-sequences-gfx90a-table
[llvm-operands]: https://llvm.org/docs/AMDGPUOperandSyntax.html
[llvm-occupancy]: https://www.llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
