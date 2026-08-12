# Command programs

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
| Specializations | Facts participate during preparation; residual values feed host launch-count evaluation | Exact source control flow and parameter names, residual launch workloads, provider selection, and derived kernel facts. |
| Buffer roots | Roles are fixed during preparation; storage is bound during materialization or issue | Immutable parameter storage or replaceable invocation storage. |

This distinction is stronger than a conventional kernel argument list.
Specializations are host-side program inputs, not push constants read by every
kernel. Their known facts participate in preparation of the complete command
root. An exact `%layer` can disappear into parameter placement and selected
branches. A residual `%element_count` can remain as an input to the generated
host launch-count function without entering the command or device ABI.

Schedule topology and parameter identity must be closed after preparation.
Residual specialization values may feed pure launch-count expressions, but
they cannot leave an unresolved source branch or a dynamic parameter key in the
materialized command program.

The launch-binding group is intentionally buffers only. Preparation classifies
roots used by `command.parameter` as fixed materialization resources and
ordinary roots as rebindable issue-time resources. Residual scalar
specializations are host inputs to launch-count evaluation; they do not
silently become device push constants. A scalar used by varying device
arithmetic must live in device-visible storage today, while an exact scalar
kernel argument may specialize away. The type checker rejects non-buffer
launch bindings so a caller and a command program cannot silently disagree
about which ABI carries a value.

!!! note "Workloads and specializations are different"

    A `kernel.launch @entry[%count](...)` workload configures that kernel's
    physical launch without changing its device ABI. A
    `command.program.launch @program[%variant](...)` specialization contributes
    facts to the complete reachable subgraph. Exact facts may change which
    kernels exist; residual facts may remain in host launch-count evaluation.

## Turn named parameter content into typed views

The checked layer example declares the kernel contract it uses, derives a
typed view from one parameter root, and launches the kernel:

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
`%element_count` may remain residual. The `view<8xf32>` result states the exact
32-byte logical footprint needed by the kernel. Command lowering can place
that content in a fixed resource range while `%input` and `%output` remain
replaceable issue-time bindings.

The substitutions must become exact nonnegative indices before the program is
prepared. The result type must also have an exact byte footprint. These are
materialization contracts rather than runtime error cases: a program whose
parameter identity or size remains unknown cannot describe an immutable
command artifact.

The kernel receives the typed view directly:

```loom title="kernels.loom"
--8<-- "examples/guide/command-programs/kernels.loom"
```

The view is more informative than an opaque pointer plus an offset. Its root,
byte range, element type, and shape remain available while the dependency
kernel is derived. The kernel can therefore use the same ordinary view and
vector operations as a standalone kernel; it has no command-program-specific
device implementation.

## Compose programs through exact contracts

Command programs are symbols. A bodyless
[`command.program.decl`](../reference/dialects/command/ops/program-decl.md)
records the exact specialization and binding contract required by one source
module. [`command.program.launch`](../reference/dialects/command/ops/program-launch.md)
provides both groups explicitly at each call site.

The checked model module depends on the layer archive and composes it in two
different ways:

```loom title="model.loom"
--8<-- "examples/guide/command-programs/model.loom"
```

The declaration does not keep a definition live and the linker does not guess
which undeclared symbol a call intended. Linking the `model` archive against
the `layer` archive resolves the exact declaration; omitting that dependency
leaves an unresolved program symbol.

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

1. Selectively link the dependency closure of the requested roots.
2. Flatten reachable command-program composition.
3. Specialize source control flow, propagate facts, and retain pure residual
   launch-count expressions.
4. Resolve named parameter requirements and explicit schedule waves.
5. Derive independently compilable kernel units for reachable launch sites.
6. Share equivalent dependency units across roots and assign dense executable
   slots.
7. Lower each root to portable command Low and derive its launch-count
   function.

The dependency kernels retain their ordinary target compilation paths. A
command root can therefore lower to the portable `cmd.core` target while one
kernel dependency lowers to AMDGPU ISA and another target configuration lowers
the same source to SPIR-V. Command composition changes the unit of
specialization; it does not create a command-only kernel language.

Equivalent launch sites may share one derived kernel unit when they call the
same kernel and the scalar facts crossing their workloads and device ABI match.
Different facts produce distinct units when specialization can profit from
them. This is how one concise program can cover many shapes and variants
without carrying a hand-maintained dispatch table.

The prepared command plan owns three complementary artifacts:

| Artifact | Responsibility |
| --- | --- |
| Command root | Portable resource bindings, command schedule, and executable slots. |
| Launch-count function | Host evaluation of the dynamic launch tuples required by that root. |
| Dependency units | Selectively linked and specialized kernels compiled through their normal targets. |

## Keep the deployment boundary honest

The checked examples in this chapter format, verify, link, and archive as Loom
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
| Source control flow remains in the prepared schedule | Its condition did not specialize to an exact value; only launch-count expressions may remain dynamic. |
| A launch binding is rejected | The command ABI accepts buffer roots only. |
| A parameter key cannot be prepared | A `{}` substitution did not become an exact nonnegative index. |
| A parameter view has unknown storage size | Its type does not provide an exact byte footprint. |
| A child program remains unresolved | Its module omitted the exact declaration or archive dependency. |
| Two supposedly parallel commands race | The source placed a hazard inside `command.concurrent`. |
| Composition preparation reports a cycle | Reachable command programs recursively launch one another. |

[Source modules and canonical text](source-modules.md) explains declaration
and archive ownership. [Facts and specialization](facts-and-specialization.md)
shows how configuration, assumptions, and target facts constrain the values
used here. [Compile artifacts](../workflows/compile-artifacts.md) covers the
public per-kernel compilation workflow available today.
