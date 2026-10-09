# AMD XDNA target

Loom compiles tile programs and their array transport into a native `.xdna`
executable. The product contains AIE2P instructions, initialized data, array
configuration, explicit storage and binding requirements, and native invocation
ranges. The [XDNA HAL](../../../../../../../runtime/src/iree/hal/drivers/amd/xdna/README.md)
loads and executes it through
[libamdf](../../../../../../../libamdf/docs/xdna.md); compilation does not invoke
the AIE SDK, LLVM, Python, or an external linker.

This directory owns device facts and AIE2P target mechanics. The shared Loom
compiler owns IR analysis, canonicalization, scheduling, register allocation,
and pass lifecycles. The target supplies descriptors, legal realizations,
physical constraints, and native encoding. Pipeline strands author complete
resident worker programs. Typed channels carry record ownership, and explicit
worker coordinates and memory pools select placement before native emission.

## Start here

| Task | Entry point |
| --- | --- |
| Write a streamed program or author AIE2P Low | [Source lowering examples](aie2p/test/source_low/) and [Low descriptor fixtures](aie2p/test/descriptors/) |
| Embed the compiler | [Public XDNA C binding](../../../../../../binding/c/include/loomc/target/amd/xdna.h) |
| Understand the executable contract | [Native image format and lifecycle](../../../../../../../runtime/src/iree/hal/drivers/amd/xdna/image/README.md) |
| Load and execute a compiled image | [XDNA HAL and normal Loom test workflow](../../../../../../../runtime/src/iree/hal/drivers/amd/xdna/README.md) |
| Compare compiler work across targets | [Shared benchmark workloads](../../../../../../binding/c/benchmark/kernels/README.md) |

## Target identity

The instruction-set contract and deployment profile answer different questions.
`amd.xdna.aie2p.core` describes a tile function; `amd.xdna.aie2p.array`
describes the workers and transport surrounding those functions. An exact
[device profile](device/profile.h) supplies the geometry, array family,
firmware protocol identity, partition constraints, and native memory alignment.

| Device | Exact compiler profile | PCI device/revision |
| --- | --- | --- |
| Strix NPU4 | `amd.xdna.strix.17f0_10` | `17f0:10` |
| Strix Halo NPU5 | `amd.xdna.strix_halo.17f0_11` | `17f0:11` |

Both profiles use the NPU2 array family and AIE2P cores. This does not make their
executable images interchangeable: admission compares the exact profile,
revision, firmware ABI, and context geometry. Tile memory, DMA engines, locks,
stream switches, register fields, and instruction limits come from owned
profile and family tables rather than assumptions about one attached device.

## Compilation path

```text
Pipeline strands + typed channels + explicit storage
                 |
      shared construction and worker preparation
                 |
      target channel/storage realization
                 |
    complete AIE2P core Low + configuration Low
          |                    |
   schedule / allocate    selected physical resources
          |                    |
   packetize / encode          |
          |                    |
   detached native leaf ------+
   + retained requirements    |
                       placement and tile linking
                              |
                  native array command generation
                              |
                  allocation/load/bind/invocation tables
                              |
                         .xdna ELF
```

The [pipeline realization](aie2p/pipeline/native.h) consumes shared construction,
worker and channel plans. Worker preparation preserves caller storage facts
through inlining and ordinary SCF transforms. Channel realization supplies local
views, ownership waits and publications, and strided DMA operations before memory
legalization. Invocation bindings retain their source argument ordinals and
required access spans. Consumers use indexed relations retained by their
producers.

The initial native path requires specialized singleton worker domains, explicit
tile storage, and bounded single-producer/single-consumer channels. Complete
worker images must fit the selected tile instruction budget. Unsupported
placement, transfer or code-size requirements are diagnosed before emission.
Invocation completion joins worker completion before releasing the owned
resources. Instruction-image replacement requires a further target realization;
channel storage and ownership are independent of instruction residency.

Hardware FIFO credits describe a completed prefix, not an arbitrary count of
finished records. If a strand reserves A then B and publishes B first, that
publication exposes no credit; publishing A exposes both. Read retirement uses
the same rule so returning B cannot let a producer overwrite a still-owned A.
Shared worker analysis retains these per-action credit counts through branches
and loop-carried accesses. An already acquired read can cross a loop argument
and use `channel.wait` to borrow its payload again without another admission.

Reclamation delay cannot introduce a new wait cycle. A schedule that releases B
and needs another channel admission before releasing A requires independently
reusable storage, rather than this aggregate-prefix ring. Likewise, control
joins requiring runtime completion state need a record-aware realization.
The native selector diagnoses those requirements before emitting the program;
they are not invalid generic channel semantics. The
[completion fixtures](aie2p/test/native_completion.loom-test) exercise reordered
publication, reclamation, loop rotation and branch-local completion.

For directly authored physical array programs, the
[array plan](aie2p/array/plan.h) owns worker placement, endpoint storage,
ring slots, locks, DMA channels and buffer descriptors, stream routes, and
completion resources. It selects external DMA, adjacent shared-memory windows,
or routed DMA according to the communication pattern and physical constraints.
[Resident scheduling](aie2p/array/resident.h) and
[program emission](aie2p/array/program.h) turn that plan into device behavior.
Emitters consume the selected resources without independently allocating a DMA
channel or recovering topology from native bytes.

For a core, common Low machinery handles dependence scheduling and physical
allocation. The target's [bundle planner](aie2p/emit/bundle_plan.h) applies
AIE2P slots, hazards, timing, and encoding constraints. The result fits the
selected tile's instruction and data budgets; code growth is a physical
resource cost, not just a compile-time statistic.

## Detached leaves and placement

[Leaf compilation](aie2p/emit/leaf_compile.h) returns a
[detached contribution](aie2p/emit/leaf_object.h): native sections, symbols,
fixups, and exact realization facts. It retains no compiler IR or temporary
planning references. Temporary scheduling and allocation storage returns to
the arena block pool before the leaf compiler returns.

The realization records code and data extents, function-storage requirements,
resource imports and their physical register bindings, capability bits, and
the union of physical register units written by the final native program.
That union includes implicit writes and generated moves across CFG paths.
`loom_aie2p_leaf_may_write_register` lets placement and linking prove that an
absent register unit is preserved without inspecting the leaf again.

Fixed-value constraints describe locations over SSA live intervals, including
values live through the exit. This provides an explicit boundary between
independent leaf compilation and its surrounding program. The
[tile linker](aie2p/emit/tile_link.h) binds placement-dependent storage and
resources using these retained facts. It does not deserialize a worker ELF or
recompile a leaf to discover its requirements.

## Component map

| Component | Responsibility |
| --- | --- |
| [device/](device/) | Exact deployment identity, firmware ABI, context limits, and partition admission. |
| [array/](array/) | Owned array-family topology, resource counts, memory windows, and register fields. |
| [aie2p/provider.c](aie2p/provider.c), [profile.h](aie2p/profile.h), [facts.h](aie2p/facts.h) | Registration and target capability/fact queries. |
| [aie2p/legalization.c](aie2p/legalization.c), [math_policy.h](aie2p/math_policy.h), [low_verify.h](aie2p/low_verify.h) | Supported High realizations, floating-point policy, and authored Low contracts. |
| [aie2p/contracts/](aie2p/contracts/), [lower/](aie2p/lower/) | Declarative legalization contracts and their C lowering mechanics. |
| [aie2p/descriptors/](aie2p/descriptors/), [machine/](aie2p/machine/), [encoding/](aie2p/encoding/) | Instruction forms, physical registers, operand constraints, native bit encodings, and packet formats. |
| [aie2p/pipeline/](aie2p/pipeline/) | Realization of complete strands, explicit memory and owned channels into core/configuration Low. |
| [aie2p/array/](aie2p/array/) | Topology, resource planning, routes, resident worker protocols, and native commands. |
| [aie2p/emit/](aie2p/emit/) | Leaf compilation, packet planning, detached objects, tile linking, reports, and final `.xdna` production. |
| [aie2p/test/](aie2p/test/) | Authored compiler regressions and descriptor/encoding coverage. |
| [Python target data](../../../../../../py/loom/target/arch/amd/xdna) | Source-owned hardware data, instruction records, and build-time validation. |

Python generators normalize the owned data into compact C tables consumed by
the shipping compiler. Pinned vendor extraction oracles support table
maintenance and differential checks; they are not a runtime compiler dependency.
Checked-in C headers define the interfaces and representation invariants.

## Executable production and residency

[Product emission](aie2p/emit/xdna_product.h) serializes the already selected
plan as load ranges and compact indexed tables. Identical file payloads can
initialize several destinations through exact source aliases. An allocation
requirement describes storage; each exported entry selects the allocations it
uses. Loading applies static address fixups, and binding applies external
buffer addresses to explicitly declared fields.

Invocation zero establishes the program's state. The image format can represent
a continuation (`0 -> 1 -> 1`) for a caller with an established native
state-retention contract. The finite XDNA HAL path uses the complete establishing
range for every independent submission (`0 -> 0`), because retaining a
time-sliced context does not preserve application tile state. Program shards
and role changes within an invocation are compiled device behavior, not a
mandatory host dispatch per tile function.

The format has no fixed ARRAY/CONTROL split, XRT argument-patching contract, or
precompiled PDI dependency. libamdf owns device/context establishment and family
bootstrap. Loom owns the executable protocol, and the caller owns its completion
frontier and backing lifetime. The [image specification](../../../../../../../runtime/src/iree/hal/drivers/amd/xdna/image/README.md)
defines this boundary precisely.

## Support and evidence

Low exposes instruction families beyond those selected automatically from High,
including integer and floating matrix forms, BFP operations, cascade transfers,
lookup memory, dimensional addressing, FIFO forms, and post-increment memory
access. The [descriptor fixtures](aie2p/test/descriptors/) identify the supported
forms and operand contracts. Hand-authored Low still obeys explicit hardware
state and ordering contracts; it does not authorize arbitrary instruction
permutation.

High support is determined by the legalizations and
[source-to-Low fixtures](aie2p/test/source_low/). An available Low instruction
does not imply a numerically conforming High lowering. For example, strict
BF16 multiplication requires a conforming realization; the compiler rejects
unsupported cases instead of silently changing floating-point behavior.

Compiler transformations and diagnostics are exercised through `.loom-test`
fixtures. C tests cover API and representation contracts. The
[native HAL CTS](../../../../../../../runtime/src/iree/hal/drivers/amd/xdna/cts/)
verifies loading, binding, repeated execution, numerical outputs, and resource
lifetimes through libamdf on matching hardware. The
[normal Loom HAL scenario](../../../../tooling/target/amd/xdna/test/hal_execution.loom)
adds device-driven target selection, compilation, staging, dispatch, readback,
and comparison with an independent VM oracle. The
[BF16 FFN benchmark](../../../../../../binding/c/benchmark/kernels/ffn/gate_up_quadratic_bf16_xdna.loom)
provides a small High-to-image workload comparable to the AMDGPU implementation;
its benchmark measures compilation, not device throughput.
