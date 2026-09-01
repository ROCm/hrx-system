# Kernels and launch configuration

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

A Loom kernel owns the computation performed on a device and the mapping from
a logical workload to the physical launch that performs it. Keeping both
contracts beside the kernel lets an embedding evaluate launch geometry without
copying scheduling arithmetic into host code.

In this chapter, you will learn:

- how the launch-configuration region and device body divide responsibility;
- why workload arguments and launch arguments are separate signatures;
- how workgroups, workitems, subgroups, and clusters map onto a dispatch;
- how over-dispatched kernels prove safe tail access;
- when target facts belong in launch configuration; and
- how callers launch a declared kernel and evaluate its compiled configuration.

## A kernel owns two signatures

[`kernel.def`](../reference/dialects/kernel/ops/def.md) contains a launch
configuration region followed by a device body:

```loom
kernel.def @fill(%element_count: index) {
  %unit = index.constant 1 : index
  %workgroup_size = index.constant 256 : index
  %rounding = index.constant 255 : index
  %rounded_count = index.add %element_count, %rounding : index
  %workgroups = index.div %rounded_count, %workgroup_size : index
  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
} launch(%element_count: index, %output: buffer, %value: f32) {
  // Device body.
  kernel.return
}
```

The values after `@fill` are the **workload signature**. They execute in the
compiled host launch-config program and compute the workgroup grid, workgroup
size, and optional cluster size. The values after `launch` are the **device
signature** carried through the selected target ABI.

| Question | Owning signature |
| --- | --- |
| How much logical work is requested? | Workload arguments. |
| How many workgroups and workitems should execute? | Launch-configuration region. |
| Which scalars and buffers does each device invocation receive? | Launch arguments. |
| How are those values physically passed? | Selected target ABI lowering. |

The signatures are deliberately independent. `%element_count` appears in both
because the host needs it to calculate the grid and the device needs it to
guard the tail. A value needed only for launch arithmetic does not consume a
device argument; a buffer needed only by the device does not become a workload
argument.

## Launch configuration is pure executable logic

The launch-configuration region is a pure function of its workload arguments
and immutable compilation inputs such as configuration and target facts. It
cannot read or write memory, call an opaque effectful function, observe
non-deterministic state, or perform a convergent operation. Kernel verification
rejects such effects at the `kernel.def` that owns the contract.

Purity keeps execution placement separate from meaning. The same residual
calculation may run in a host VM companion, remain as ordinary pure calls in a
command program, or execute on a device to produce an indirect dispatch tuple.
None of those products needs a different source representation or permission
to observe additional state.

## Workload is not workgroup count

[`kernel.launch.config`](../reference/dialects/kernel/ops/launch-config.md)
terminates the host region with three physical dimensions:

- `workgroups(x, y, z)` is the number of workgroups in the dispatch;
- `workgroup_size(x, y, z)` is the required invocation shape of each
  workgroup; and
- optional `cluster_size(x, y, z)` groups neighboring workgroups for targets
  that support cooperative clusters.

An element count is a workload. The ceiling division that maps it to
workgroups is a scheduling decision owned by the kernel. For 1009 elements and
a workgroup size of 256, the region returns four workgroups; the application
does not need to know that policy.

The compiled launch-config result also contains the selected subgroup size and
the total workgroup-local storage required by the specialized kernel. These are
compiler products derived from the same source and target facts, not extra
numbers that an embedding reconstructs.

## Read coordinates in the device body

The device body queries coordinates at the level it needs:

```loom
%workgroup = kernel.workgroup.id<x> : index
%lane = kernel.workitem.id<x> : index
%workgroup_size = kernel.workgroup.size<x> : index
%element_index = index.madd %workgroup, %workgroup_size, %lane : index
```

[`kernel.workitem.dispatch.id`](../reference/dialects/kernel/ops/workitem-dispatch-id.md)
provides that global coordinate directly when the decomposed workgroup and lane
values are not otherwise useful. All coordinate results are logical `index`
values; byte addressing begins only when the body projects or offsets storage.

The hierarchy has explicit query operations:

| Level | Position | Extent |
| --- | --- | --- |
| Whole dispatch | `kernel.workitem.dispatch.id` | Workgroup count × workgroup size. |
| Workgroup | `kernel.workgroup.id` | `kernel.workgroup.count`. |
| Workitem | `kernel.workitem.id` | `kernel.workgroup.size`. |
| Subgroup | `kernel.subgroup.id`, `kernel.subgroup.lane.id` | `kernel.subgroup.count`, `kernel.subgroup.size`. |
| Workgroup cluster | `kernel.cluster.id`, `kernel.cluster.workgroup.id` | `kernel.cluster.count`, `kernel.cluster.size`. |

Choose the level that expresses the algorithm. A subgroup reduction naturally
uses subgroup values; a tiled matrix kernel often uses separate workgroup and
workitem axes; a flat elementwise kernel can use the dispatch id.

## Over-dispatch safely and preserve the proof

Ceiling division usually creates inactive invocations in the final workgroup.
Guard memory access with the logical workload. The structured branch carries
the comparison as a path fact to operations in its region:

```loom
%in_bounds = index.cmp ult, %element_index, %element_count : index
scf.if %in_bounds {
  %value = view.load %input_view[%element_index] : view<[%element_count]xf32> -> f32
  view.store %value, %output_view[%element_index] : f32, view<[%element_count]xf32>
}
```

The branch controls execution and proves `%element_index < %element_count` on
its true path. [`index.assume`](../reference/dialects/index/ops/assume.md) is
still useful for facts established outside visible control flow, such as the
valid range of an expert index loaded from a routing table. [Facts and
specialization](facts-and-specialization.md#refine-the-value-that-carries-a-proven-contract)
separates those caller promises from compiler-derived path facts.

For vector tails, a vector mask can instead define precisely which lanes access
memory. The choice between a structured guard and a masked transfer is an
algorithm and target decision, not an ABI difference.

## Query target facts instead of naming a backend

Reusable kernels remain targetless unless their algorithm genuinely depends on
a target capability. Launch configuration can consume a normalized target fact
directly:

```loom
kernel.def @one_subgroup() {
  %unit = index.constant 1 : index
  %subgroup_size = target.subgroup.size : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%subgroup_size, %unit, %unit) : index
} launch(%output: buffer) {
  // Device body.
  kernel.return
}
```

`target.subgroup.size` asks what the selected compile target guarantees while
forming the launch contract. `kernel.subgroup.size` reads the subgroup size in
the executing device body. They often resolve to the same number, but they
belong to different stages and answer different questions.

Templates can state the same dependency as an eligibility requirement:

```loom
template.def<@guide.elementwise_transform> requires [#target.subgroup.size<32>] priority(20) @wave32_elementwise_transform(%value: f32) -> (f32) {
  %two = scalar.constant 2.0 : f32
  %result = scalar.mulf %value, %two : f32
  template.return %result : f32
}
```

A composition root supplies the target profile for offline compilation, or a
JIT projects normalized facts from the live device. The kernel and its library
providers do not need separate AMDGPU, SPIR-V, or other source variants merely
to learn subgroup size.

## Declare and launch the complete contract

A module that launches a kernel but does not define it uses
[`kernel.decl`](../reference/dialects/kernel/ops/decl.md). The declaration
preserves both signatures:

```loom
kernel.decl @fill(%element_count: index) launch(%element_count: index, %output: buffer, %value: f32)
```

[`kernel.launch`](../reference/dialects/kernel/ops/launch.md) spells workload
arguments in brackets and device arguments in parentheses:

```loom
kernel.launch @fill[%element_count](%element_count, %output, %value) : [index](index, buffer, f32)
```

This is not redundant syntax. The bracketed values evaluate the launch
configuration; the parenthesized values bind the device ABI. The verifier
checks both groups against the resolved definition or declaration.

As with function calls, the declaration is mandatory. Linking searches only
the explicitly supplied modules and libraries for that exact symbol; it never
discovers a kernel merely because an undeclared launch happens to name it.
A public compatible definition from an explicitly supplied library may satisfy
the declaration. Missing and ambiguous definitions are link errors for a
closed product.

## Evaluate the compiled launch contract

When a kernel-product request marks a root host-launchable, compilation emits a
VM launch-config companion beside the target-native executable. The immutable
product root carries both artifact ordinals and both artifact-local function
ordinals. An embedding links the optional `loomc/target/vm` package and uses
that projection in four steps:

1. Load the root's launch-config artifact with
   `loomc_vm_launch_config_program_load`.
2. Bind `root.launch_config_function_ordinal` with
   `loomc_vm_launch_config_program_function_at`.
3. Pass the workload scalar bit patterns to
   `loomc_vm_launch_config_program_invoke`.
4. Issue `root.executable_function_ordinal` from the matching executable with
   the returned workgroup count, workgroup size, optional cluster size,
   subgroup size, and workgroup-storage requirement.

Loading and invocation do not parse source, select a target, or compile device
code. Those decisions happened when the executable and companion artifact were
produced. The product builder establishes the cross-artifact relationship once
from compiler-owned function identity. Loading performs no artifact scan or
name join, and repeated invocations use the bound token without another lookup.

The generated [`loomc` VM launch-configuration API](../reference/c-api/generated/target_2vm_2launch__config_8h.html)
defines the exact ownership, thread-safety, argument representation, and result
contracts. [Embed kernel JIT compilation](../integration/jit-kernel.md) follows
the complete source, specialization, launch-evaluation, emission, and runtime
handoff; [Compile artifacts](../workflows/compile-artifacts.md) covers the
equivalent installed-tool products.

## Keep each decision at its owning boundary

| Symptom | Boundary confusion | Correct ownership |
| --- | --- | --- |
| Host code duplicates ceiling division | Workload confused with launch geometry | Compute geometry in the kernel configuration region. |
| A dynamic bound is a workload but missing from the device signature | Host and device needs conflated | Pass it through both signatures when both stages consume it. |
| A buffer appears in the workload signature | Device resource treated as host launch arithmetic | Keep it in the launch signature. |
| A kernel hardcodes one architecture name | Capability confused with backend identity | Query a normalized target fact or constrain a provider. |
| Tail invocations access before checking bounds | Physical grid confused with logical extent | Guard or mask access and carry the refined index. |
| Application code guesses LDS/shared-memory bytes | Compiler allocation result reconstructed externally | Consume `workgroup_storage_bytes` from the compiled launch contract. |

The complete kernel in the [source-to-artifacts walkthrough](../getting-started/source-to-artifacts.md#a-kernel-owns-two-contracts)
combines a dynamic workload, target-derived launch geometry, buffer views, a
selectable motif, and tail handling in one targetless entry.

Continue with facts and specialization: the same value predicates used for
bounds and launch geometry also drive linking, provider selection, and
target-specific optimization without changing the source ownership model.
