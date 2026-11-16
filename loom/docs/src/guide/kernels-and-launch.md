# Kernels and launch configuration

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

A Loom kernel owns the computation performed on a device and the mapping from
a logical workload to the physical launch that performs it. Keeping both
contracts beside the kernel lets an embedding evaluate launch geometry without
copying scheduling arithmetic into host code.

In this chapter, you will learn:

- how the launch-configuration region and device body divide responsibility;
- why workload arguments and launch arguments are separate signatures;
- how storage-backed workloads participate in launch computation;
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

The values after `@fill` are the **workload signature**. They feed the launch
configuration that computes the workgroup grid, workgroup size, and optional
cluster size. The values after `launch` are the **device signature** carried
through the selected target ABI.

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

`index` and `offset` are logical source types, not fixed-width device ABI
types. A target may select a 32- or 64-bit representation for a launch argument
from the facts proven about that value. Loom's execution tools pack arguments
from the loaded executable's parameter reflection; a direct HAL host must use
the same reflection instead of assuming a width from the source spelling.

## Launch configuration is deterministic executable logic

The launch-configuration region is a deterministic function of its explicit
workload arguments and immutable compilation inputs such as configuration and
target facts. Workloads may be scalars or storage. A configuration may load
through a buffer or view rooted in its workload signature, then perform
arbitrary arithmetic and control flow over the loaded values.

That permission is deliberately narrow. The region cannot write memory, use an
atomic or fence, observe non-deterministic state, perform a convergent or
independently observable operation, call code with unknown effects, or read
storage that does not originate at an explicit workload argument. Kernel
verification proves each admitted load's storage origin at the `kernel.def`
that owns the contract.

Storage-derived values may determine workgroup counts after their range is
proven to fit the dispatch ABI's unsigned 32-bit domain. They cannot determine
workgroup or cluster sizes: those sizes participate in device specialization,
resource allocation, and executable compatibility before issue-time storage is
available.

The retained effect contract keeps execution placement separate from meaning.
A pure scalar calculation may run in a host companion or move earlier in a
command program. A calculation that reads workload storage stays ordered after
the storage producer. That preserved ordering and identity gives command
lowering the information needed to fold known loads or realize unresolved
counts as device-produced indirect dispatch data without changing the kernel's
source contract.

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

### Share coordinate-dependent helpers

Ordinary helpers can read these coordinates without being forced inline:

```loom
func.def noinline @position(%bias: index) -> (index) {
  %lane = kernel.workitem.id<x> : index
  %result = index.add %lane, %bias : index
  func.return %result : index
}
```

When a kernel or worker calls `position`, compilation supplies its workitem ID
as an ordinary helper argument. Intermediate helpers forward that argument;
the public kernel signature stays unchanged. Two workers calling the same
helper each supply their own coordinates. Constant launch dimensions can fold
away through the same argument specialization used for explicitly passed values.

Every path to the helper must supply an invocation. A host-callable entry, a
launch-configuration region, or a fixed native signature cannot provide these
arguments implicitly and produces a diagnostic at that boundary. Active masks,
votes, shuffles, and barriers describe participation at the operation itself;
they are not stable coordinate inputs that can be moved to the caller.

## Form masks over active subgroup lanes

Subgroup votes describe cooperation without naming a backend.
[`kernel.subgroup.vote.any`](../reference/dialects/kernel/ops/subgroup-vote-any.md)
and
[`kernel.subgroup.vote.all`](../reference/dialects/kernel/ops/subgroup-vote-all.md)
reduce a predicate to one Boolean result.
[`kernel.subgroup.vote.ballot`](../reference/dialects/kernel/ops/subgroup-vote-ballot.md)
instead returns the physical lane positions whose predicates are true:

```loom
%lane = kernel.subgroup.lane.id : index
%unit = index.constant 1 : index
%parity = index.andi %lane, %unit : index
%selected = index.cmp eq, %parity, %unit : index
%selected_lanes = kernel.subgroup.vote.ballot %selected : i1 -> i64
```

Bit *i* in `%selected_lanes` represents physical lane *i*. The operation
observes only invocations active at that call, so a ballot inside divergent
control flow excludes lanes on other paths even when those lanes computed the
predicate earlier. This makes ballots suitable for subgroup-local compaction,
router selection, and sparse page or token lookups.

The result may be `i32` or `i64`, and its width must cover the selected target's
subgroup. Target selection diagnoses an unavailable ballot capability or a
subgroup wider than the authored mask instead of truncating it. AMDGPU and
Vulkan devices that report SPIR-V subgroup-ballot support use the same source
contract.

A vote exchanges predicate state; it does not make memory visible or wait for
independent asynchronous work. Use a barrier or memory fence when the algorithm
also requires a memory-ordering contract.

## Choose collective arithmetic requirements

Subgroup and workgroup reductions and scans combine each participating
invocation's value independently for every vector component. A reduction
publishes the group result; a scan publishes the prefix or suffix selected by
its direction and inclusive/exclusive mode. Association within that set is
unspecified: a parallel tree need not match a serial loop bit for bit.

Floating sums and products use native target arithmetic by default:

```loom
%sum = kernel.workgroup.reduce<addf> %partial : f32
%prefix = kernel.subgroup.scan<addf> %value {direction = forward, mode = inclusive} : f32
```

This default keeps native collective instructions available. The target's
arithmetic rules determine intermediate precision, signed zeros, subnormal
handling, and nonfinite behavior. Native arithmetic does not promise IEEE
exceptional-value behavior on every target. Ordinary finite ML workloads can
qualify their error envelope against an independent numerical oracle without
requiring an identical reduction tree.

An explicit `preserve` requirement instead requires each combine in the tree
to obey the source format's arithmetic semantics:

```loom
%sum = kernel.workgroup.reduce<addf, preserve> %partial : f32
```

`preserve` applies to each combine's arithmetic, not the reduction order. This
includes rounding, signed zeros, subnormals, and nonfinite values. For example,
a nonempty F32 sum of only negative zeros remains negative zero. For four
inputs, both `(a + b) + (c + d)` and `((a + b) + c) + d` remain permitted and
can produce different result bits. Preservation does not select a fixed tree,
guarantee reproducibility, or request a correctly rounded real-number sum.
It also does not introduce dynamic rounding modes or observable floating-point
exception flags. A target without a qualified implementation rejects the
requirement instead of dropping it or imposing software arithmetic on default
kernels.

Both modes retain the same participating invocations and synchronization
requirements. Integer arithmetic keeps its fixed-width semantics. Floating
extrema always retain their selected kind's NaN and signed-zero rules;
`preserve` adds no further requirement for extrema. An empty exclusive prefix
returns the combining kind's seed, not an invented contribution to a nonempty
prefix: negative zero for preserved floating addition, positive one for
multiplication, and the appropriate extreme bound for minimum or maximum.
Native addition permits either zero sign.

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

For a kernel whose launch computation has scalar workloads and remains pure,
compilation can emit a standalone launch-config companion. An embedding uses
the public `loomc` API in four steps:

1. Load the launch-config artifact with
   `loomc_launch_config_program_load`.
2. Resolve the kernel's public export name with
   `loomc_launch_config_program_lookup_function`.
3. Pass the workload scalar bit patterns to
   `loomc_launch_config_program_invoke`.
4. Issue the matching executable entry with the returned workgroup count,
   workgroup size, optional cluster size, subgroup size, and workgroup-storage
   requirement.

Loading and invocation do not parse source, select a target, or compile device
code. Those decisions happened when the executable and companion artifact were
produced. Lookup resolves a public export name to a program-local token once;
repeated invocations use that token without another string lookup.

A storage-backed configuration is not a raw-bits host callback. It remains in
command-program IR, where the workload buffer or view and its producer ordering
stay explicit. A closed command product must resolve the result to exact counts
or explicit indirect-count storage; the application does not reproduce the
kernel's launch arithmetic.

The generated [`loomc` launch-configuration API](../reference/c-api/generated/launch__config_8h.html)
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
| A produced count is copied into caller launch arithmetic | Producer data confused with caller policy | Pass its storage as a workload and let the kernel configuration load and transform it. |
| A kernel hardcodes one architecture name | Capability confused with backend identity | Query a normalized target fact or constrain a provider. |
| Tail invocations access before checking bounds | Physical grid confused with logical extent | Guard or mask access and carry the refined index. |
| Application code guesses LDS/shared-memory bytes | Compiler allocation result reconstructed externally | Consume `workgroup_storage_bytes` from the compiled launch contract. |

The complete kernel in the [source-to-artifacts walkthrough](../getting-started/source-to-artifacts.md#a-kernel-owns-two-contracts)
combines a dynamic workload, target-derived launch geometry, buffer views, a
selectable motif, and tail handling in one targetless entry.

Continue with facts and specialization: the same value predicates used for
bounds and launch geometry also drive linking, provider selection, and
target-specific optimization without changing the source ownership model.
