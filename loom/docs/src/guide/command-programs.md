# Command programs

**Example files:** [`loom/docs/examples/guide/command-programs/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/guide/command-programs)

A kernel specializes one device entry point. A command program specializes a
reusable subgraph: its kernel launches, immutable parameters, launch counts,
and explicit dependency structure. The result keeps the same source, linking,
fact, and target model as an individual kernel instead of introducing a second
graph language around it.

In this chapter, you will learn:

- how a command-program signature separates specialization inputs from
  explicit buffer roots and their materialization roles;
- how named parameter content becomes a typed kernel argument;
- how program declarations and launches compose independently packaged
  subgraphs;
- how `command.serial` and `command.concurrent` state dependency edges; and
- what command preparation preserves, specializes, and emits.

## The signature separates host decisions from storage

[`command.program.def`](../reference/dialects/command/ops/program-def.md) has
two argument groups:

```loom
command.program.def @project_layer(%layer: index, %element_count: index) launch(
    %parameters: buffer, %input: buffer, %output: buffer) {
  // Reusable command schedule.
  command.return
}
```

The arguments before `launch` specialize the program. The arguments after
`launch` are buffer roots supplied through the program ABI:

| Argument group | Becomes known | What it can control |
| --- | --- | --- |
| Specializations | Facts participate during linking and preparation; artifact-affecting scalar uses must resolve before portable lowering | Source control flow, parameter names, provider selection, direct launch counts, and exact scalar kernel arguments. |
| Buffer roots | Roles are fixed during preparation; storage is bound during materialization or issue | Immutable parameter storage or replaceable invocation storage. |

This distinction is stronger than a conventional kernel argument list.
Specializations are host-side program inputs, not push constants read by every
kernel. Their known facts participate in preparation of the complete command
root. An exact `%layer` can disappear into parameter placement and selected
branches. A source root may retain `%element_count` while it remains linkable,
but preparing a direct dispatch requires its physical count to become exact.
An issue-time dynamic count instead has an explicit `view<3xi32>` storage
producer and lowers to indirect dispatch; it never becomes an implicit host
callback outside the command artifact.

## Effects are explicit commands

A command program is not a pure function: launching a kernel and ordering work
are its purpose. Those observable effects must nevertheless be represented by
explicit command operations such as `kernel.launch`, `command.program.launch`,
`command.serial`, and `command.concurrent`. Ordinary SSA computation around
them is effect-free. Calls used to derive launch workloads therefore carry a
`pure` contract, just as calls in a kernel launch-configuration region do.

The distinction keeps preparation movable. The compiler can inline and fold
pure shape arithmetic at its real caller-owned control-flow site. When a count
must remain dynamic at issue time, an explicit device-side producer can write
the indirect dispatch tuple without silently moving a memory observation or
mutation. A `buffer.alloca` remains valid: it declares a fresh command-program
storage identity for later scheduling rather than reading, writing, or
synchronizing external state.

The verifier enforces this at the source operation that violates the contract.
An opaque ordinary call is rejected inside nested control flow instead of
surviving until command outlining or schedule materialization.

Schedule topology and parameter identity must be closed after preparation.
Pure launch-count expressions either fold to exact direct counts or terminate
in explicit indirect-count storage. They cannot leave an unresolved source
branch, dynamic parameter key, or free scalar result in the materialized
command program.

The launch-binding group is intentionally buffers only. Preparation classifies
roots used by `command.parameter` as fixed materialization resources and
ordinary roots as rebindable issue-time resources. Scalar specializations do
not silently become host callbacks or device push constants. A scalar used by
varying device arithmetic must live in device-visible storage, while an exact
scalar kernel argument may specialize away. The type checker rejects
non-buffer launch bindings so a caller and a command program cannot silently
disagree about which ABI carries a value.

!!! note "Workloads and specializations are different"

    A `kernel.launch @entry[%count](...)` workload configures that kernel's
    physical launch without changing its device ABI. A
    `command.program.launch @program[%variant](...)` specialization contributes
    facts to the complete reachable subgraph. Exact facts may change which
    kernels exist or make direct launch counts exact. Counts that intentionally
    remain issue-time dynamic use explicit indirect storage.

## Launch configuration stays with its caller

Command preparation does not open a selected kernel implementation. Linking
projects only its logical contract and pure launch-configuration
facet, then rewrites each workload-level launch at the same caller-owned CFG
site:

```loom
%x, %y, %z = func.call pure @project_block$config(%element_count) : (index) -> (index, index, index)
kernel.dispatch @project_block[%x, %y, %z](%weights, %input, %output) : [index, index, index](view<8xf32>, buffer, buffer)
```

Ordinary canonicalization, CSE, inlining, and folding optimize those calls
before schedule construction. Equal calls with equal SSA inputs can collapse;
calls in different branches or loop iterations retain the control-flow scope
that gives their values meaning. There is no aggregate result function or
command-specific memo table.

Exact scalar results become direct dispatch commands. An exact aligned
`view<3xi32>` rooted in an immutable parameter or command input becomes a
static indirect dispatch. The same view in transient command storage becomes
a dynamic indirect dispatch when an earlier execution wave writes it. Any
other residual count is diagnosed because the portable artifact has no hidden
place to evaluate or store it.

Each configured entry contributes one atomic executable-entry requirement.
The embedding may satisfy it with a Loom-compiled executable, a cached object,
or an externally supplied HIP or SPIR-V entry; command planning neither loads
the kernel body nor chooses that compilation policy.

## Turn named parameter content into typed views

The layer example declares the kernel contract it uses, derives a
typed view from one parameter root, and launches the kernel:

**Source:** [`loom/docs/examples/guide/command-programs/layer.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/command-programs/layer.loom)

```loom title="layer.loom"
--8<-- "examples/guide/command-programs/layer.loom"
```

[`command.parameter`](../reference/dialects/command/ops/parameter.md) is a
declarative association. It performs no string lookup, allocation, transfer,
or synchronization at execution time. During command preparation, each `{}` in
the pattern receives one canonical decimal index substitution. The complete
key and exact view footprint become a parameter requirement attached to the
source buffer root.

The model composition supplies exact layer constants. A `%layer` value of `12`
therefore produces the key `blk.12.projection.weight`, while the forwarded
`%element_count` remains a source specialization until a selected root fixes
its launch. The `view<8xf32>` result states the exact 32-byte parameter block
available to the kernel. `%element_count` independently selects a prefix from
one through eight, so the view shape does not create one kernel version per
active element count. Command lowering can place that content in a fixed
resource range while `%input` and `%output` remain replaceable issue-time
bindings.

The substitutions must become exact nonnegative indices before the program is
prepared. The result type must also have an exact byte footprint. These are
materialization contracts rather than runtime error cases: a program whose
parameter identity or size remains unknown cannot describe an immutable
command artifact.

The kernel receives the typed view directly:

**Source:** [`loom/docs/examples/guide/command-programs/kernels.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/command-programs/kernels.loom)

```loom title="kernels.loom"
--8<-- "examples/guide/command-programs/kernels.loom"
```

The view is more informative than an opaque pointer plus an offset. Its root,
byte range, element type, and shape remain available while the kernel is
specialized and compiled independently. The kernel can therefore use the same
ordinary view and vector operations as a standalone kernel; it has no
command-program-specific device implementation.

## Expand model structure from configuration

Model-level configuration can determine command topology without turning
per-invocation shapes into global variants. This program resolves
`@model.layer_count`, fully unrolls a source loop during command preparation,
and uses each exact induction value to select one layer's parameter block:

**Source:** [`loom/docs/examples/guide/command-programs/stack.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/command-programs/stack.loom)

```loom title="stack.loom"
--8<-- "examples/guide/command-programs/stack.loom"
```

The `unroll` policy is part of the source contract. Command preparation
requires the layer count to resolve, expands the loop into a finite sequence,
and then sees parameter keys such as `blk.0.projection.weight` and
`blk.1.projection.weight`. The `%element_count` specialization remains a
separate value that controls each kernel workload; it is not promoted to
configuration merely because every layer consumes it. A host may keep the
program parameterized in a source module, but it fixes that value before
preparing a direct portable command artifact.

## Compose programs through exact contracts

Command programs are symbols. A bodyless
[`command.program.decl`](../reference/dialects/command/ops/program-decl.md)
records the exact specialization and binding contract required by one source
module. [`command.program.launch`](../reference/dialects/command/ops/program-launch.md)
provides both groups explicitly at each call site.

The model module depends on the layer library and composes it in two
different ways:

**Source:** [`loom/docs/examples/guide/command-programs/model.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/command-programs/model.loom)

```loom title="model.loom"
--8<-- "examples/guide/command-programs/model.loom"
```

The declaration does not keep a definition live and the linker does not guess
which undeclared symbol a call intended. Linking the `model` module against
the `layer` library resolves the exact declaration; omitting that dependency
leaves an unresolved program symbol.

Its Bazel target supplies the complete layer module. The exact declaration is
resolved from the unique exported program definition in that explicit library
universe.
The [linking workflow](../workflows/link-and-package.md#link-transitive-dependencies-incrementally)
shows the same model-to-layer-to-kernel shape across a standalone partial
artifact.

During command preparation, reachable program launches are clone-inlined in
callee-before-caller order. Each selected root becomes a closed command body,
while independently selected roots and shared definitions remain ordinary
symbols in the linked source. Recursive command composition is rejected
because it cannot produce a finite materialized schedule.

This is the same composition model used by motifs and kernels:

```text
kernel library      layer program       model program
@project_block  <-  @project_layer  <-  @two_layer
                         ^           <-  @two_branch
                         |
                   typed parameter view
```

Packaging boundaries stay source boundaries. They do not become routing
layers in the application and they do not prevent specialization from seeing
through the final linked closure.

## State dependencies with structured schedules

[`command.serial`](../reference/dialects/command/ops/serial.md) orders each
child after the preceding child completes. In `@two_layer`, the first layer
writes `%hidden` before the second layer consumes it.

[`command.concurrent`](../reference/dialects/command/ops/concurrent.md) adds no
dependency edge between sibling commands and joins all siblings on exit. In
`@two_branch`, both projections may execute independently, but the command
program cannot return until both have completed.

These regions nest, so larger schedules can express serial chains, fork/join
sections, and pipelines without flattening them into host callbacks. The two
public roots in `model.loom` are the individual serial and concurrent building
blocks; nesting those same regions preserves each region's entry and exit
contract.

The schedule structure owns ordering. Buffer identity or aliasing does not
silently create dependency edges between concurrent siblings. If two commands
have a read/write hazard, the source must place them in a serial relationship.
This makes the schedule inspectable and keeps dependency analysis out of
opaque device ABIs.

## Specialize the subgraph as one program

Preparing one or more public command roots performs a bounded sequence of
compiler operations:

1. Link the dependency closure of the requested roots.
2. Flatten reachable command-program composition.
3. Project each referenced kernel's logical contract and pure launch
   configuration without opening its implementation.
4. Rewrite launches in caller IR, then specialize, inline, canonicalize, CSE,
   and unroll the resulting ordinary dataflow.
5. Resolve named parameters, physical dispatch counts, storage, and explicit
   schedule waves.
6. Assign dense slots for distinct executable-entry requirements and lower
   each closed root to portable command Low.

The dependency kernels retain their ordinary target compilation paths. A
command root can therefore lower to the portable `cmd.core` target while one
kernel dependency lowers to AMDGPU ISA and another target configuration lowers
the same source to SPIR-V. Command composition changes the unit of
specialization; it does not create a command-only kernel language.

Equivalent launch sites can share pure configuration computation through
ordinary CSE when their SSA inputs match. Repeated entry uses also share one
plan-wide executable-entry requirement while retaining root-local dense slots.
Kernel specialization and cache policy remain an independent host workflow
rather than a nested product of command compilation.

The prepared command plan owns three complementary products:

| Product | Responsibility |
| --- | --- |
| Command root | Portable resource bindings, command schedule, and executable slots. |
| Entry requirements | Atomic executable and entry bindings required by one or more command roots. |
| Parameter and storage requirements | Fixed parameter placements, rebindable bindings, transients, and any explicit indirect-count storage. |

## Keep the deployment boundary honest

The examples in this chapter format, verify, link, and package Loom
libraries. The compiler also has the target-neutral command preparation and
lowering path described above. The installed tools do not yet expose one
complete public command that prepares, materializes, binds, and issues a
command root, so this guide does not disguise a private runtime integration as
an end-user recipe.

That missing command is an ergonomics gap at the deployment boundary, not a
gap in the source representation. Kernel-only applications can continue to
compile and launch individual entries. Embedders can use command preparation
through the compiler integration while the stable installed workflow is being
defined.

## Keep failures in their owning contract

| Symptom | Contract to inspect |
| --- | --- |
| Source control flow remains in the prepared schedule | Its condition did not specialize to an exact value; portable command topology must be closed. |
| A direct dispatch count remains dynamic | Specialize it to an exact unsigned 32-bit value or produce an aligned `view<3xi32>` for indirect dispatch. |
| A launch binding is rejected | The command ABI accepts buffer roots only. |
| A parameter key cannot be prepared | A `{}` substitution did not become an exact nonnegative index. |
| A parameter view has unknown storage size | Its type does not provide an exact byte footprint. |
| A child program remains unresolved | Its module omitted the exact declaration or library dependency. |
| Two supposedly parallel commands race | The source placed a hazard inside `command.concurrent`. |
| Composition preparation reports a cycle | Reachable command programs recursively launch one another. |

[Source modules and canonical text](source-modules.md) explains declaration
and library ownership. [Facts and specialization](facts-and-specialization.md)
shows how configuration, assumptions, and target facts constrain the values
used here. [Compile artifacts](../workflows/compile-artifacts.md) covers the
public per-kernel compilation workflow available today.
