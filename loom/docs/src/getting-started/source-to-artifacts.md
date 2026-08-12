# From source to artifacts

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

Loom keeps a program understandable while it moves from reusable source to a
target-specific artifact. Linking, specialization, launch configuration,
correctness cases, benchmark workloads, and compiler evidence operate on the
same program instead of meeting only after code has become opaque.

The useful unit is therefore larger than an instruction stream and smaller
than a framework. A Loom module can contain reusable computation, alternative
implementations, dispatchable kernels, executable checks, and command programs.
An application chooses which roots and facts matter, then asks Loom to link and
specialize only that reachable program.

The canonical `elementwise-transform` example maps a selectable doubling motif
over f32 buffers, then follows that program through linking, target
specialization, native kernel emission, and command-program lowering.

## One program, several forms

`.loom` is the canonical, human-readable source form. `.loombc` preserves the
same linkable program in bytecode. Neither form implies one target or one
deployment strategy.

```text
motif.loom + kernel.loom + model.loom + configuration + target profile
                                  │
                          link and specialize
                                  │
                  lower the selected executable roots
                                  │
           native code + launch contracts + compiler evidence
```

The same source can be compiled ahead of time, linked and specialized when a
model is loaded, or JIT-compiled for a concrete invocation. Those choices
change when facts become available; they do not require different kernel
languages.

Modules are ordinary composition boundaries. Public symbols can become roots
for another module or an embedding, private symbols remain implementation
details, and unreachable helpers and providers can disappear after selection.
The linker resolves declared symbol relationships; merely placing a module in
a library does not make its entire contents live.

## Exact calls and selectable implementations

Loom separates naming one implementation from requesting an implementation
that satisfies a contract.

| Source construct | Meaning |
| --- | --- |
| [`func.def`](../reference/dialects/func/ops/def.md) and [`func.call`](../reference/dialects/func/ops/call.md) | Define and call an exact helper by symbol. |
| [`func.template`](../reference/dialects/func/ops/template.md) | Provide a visible implementation of a named contract. |
| [`func.apply`](../reference/dialects/func/ops/apply.md) | Request an implementation of that contract at compile time. |

An exact bit-manipulation helper is naturally a `func.call`. An operation whose
best implementation depends on element format, shape facts, subgroup size, or
target capabilities is naturally a `func.apply`. During specialization, Loom
matches the reachable providers, their predicates and requirements, and the
known facts. The selected provider becomes an ordinary callable boundary that
normal inlining and optimization can remove.

This is the foundation of a scalable library: callers name semantic demands;
libraries provide reusable implementations; target selection remains a fact of
the final compilation rather than a global property baked into every helper.
This example defines an exact helper, a wave32-specific provider, and a
portable provider for the same `guide.elementwise_transform` contract:

**Source:** [`loom/docs/examples/elementwise-transform/motif.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/motif.loom)

```loom title="motif.loom"
--8<-- "examples/elementwise-transform/motif.loom"
```

All three definitions omit `public` and are therefore private. Supplying the
motif as an explicit library makes those implementations available while Loom
forms the requested program, but it does not add them to the module's public
interface. Provider selection and normal reachability can then remove the
implementations the selected root does not need.

The wave32 provider is eligible only when the selected target establishes a
subgroup size of 32. The fallback stays targetless, so the motif remains useful
to targets that know nothing about AMDGPU.

The complete operation inventory lives in the generated
[`func` dialect reference](../reference/dialects/func/index.md).

## A kernel owns two contracts

A [`kernel.def`](../reference/dialects/kernel/ops/def.md) contains a launch
configuration region and a device body. They answer different questions.

The launch configuration region receives workload values and computes the
physical workgroup grid, required workgroup size, and optional cluster size.
The body receives the arguments carried through the device launch ABI and uses
workgroup, workitem, subgroup, and memory operations to perform the work.

This distinction keeps a workload such as “process 1009 elements” separate
from the physical decision “launch four workgroups of 256 invocations.” An
embedding evaluates the compiled launch configuration for the workload; it
does not reverse-engineer grid dimensions from a kernel name or duplicate the
kernel's scheduling arithmetic.

Workload arguments and device arguments are explicit even when the same value
appears in both contracts. That explicit boundary lets Loom specialize launch
geometry when facts are known while preserving a correct dynamic path when
they are not. The generated [`kernel` dialect
reference](../reference/dialects/kernel/index.md) defines the complete launch,
execution, collective, and synchronization vocabulary.

The kernel below accepts its element count as a workload and device value,
derives a one-subgroup workgroup size from the target profile, and applies the
motif contract without naming either provider:

**Source:** [`loom/docs/examples/elementwise-transform/kernel.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/kernel.loom)

```loom title="kernel.loom"
--8<-- "examples/elementwise-transform/kernel.loom"
```

## Ask when a value becomes known

Most confusion about specialization disappears once every value has a clear
binding stage.

| Value category | Binding stage | Role |
| --- | --- | --- |
| Configuration values | Link or compile time | Resolve [`config.decl`](../reference/dialects/config/ops/decl.md) through a matching definition and seed facts used by specialization. |
| Target facts | Compile invocation | Describe capabilities such as subgroup size, supported types, and resource limits for the selected executable root. |
| Kernel workload values | Launch-configuration evaluation | Compute the physical launch for one workload without pretending the values were compile-time constants. |
| Kernel launch arguments | Device issue | Carry scalars and buffers through the selected target ABI into the kernel body. |
| Command specialization arguments | Command-program specialization | Shape a reusable command artifact and its aggregate launch requirements. |
| Command buffer bindings | Command-program issue | Attach concrete parameter, transient, input, and output storage without recompiling the schedule. |

Configuration is not a hidden global flag. A module declares artifact-level
choices such as a model's layer count or weight encoding, another input defines
them, and ordinary symbol dependency analysis keeps their effect visible:

```loom
config.decl @model.layer_count : %value: index where [range(%value, 1, 256)]
config.decl @model.weight_encoding : encoding<schema>
```

Per-launch element counts stay workload values instead of becoming global
configuration. Likewise, target requirements are compile-time selection
constraints rather than runtime branches. Reusable code generally stays
targetless; a provider names a target requirement only when its algorithm
actually depends on one.

Loom can preserve only information the program supplies. The ranges and
divisibility predicates on a
[`config.decl`](../reference/dialects/config/ops/decl.md) constrain a value even
before a composition root binds it to one constant. Control flow also supplies
path facts: inside the guarded region in `kernel.loom`, Loom knows that
`%element_index < %element_count` without an extra assumption. An
[`index.assume`](../reference/dialects/index/ops/assume.md) is for information
the compiler cannot derive, such as a range guaranteed by the producer of a
loaded routing index. The `buffer.assume.*` family carries root alignment,
memory-space, aliasing, and identity facts; the example's
[`buffer.assume.noalias`](../reference/dialects/buffer/ops/assume-noalias.md)
makes the input and output independence explicit.

These are optimization inputs, not documentation comments. Shapes, ranges,
alignment, aliasing, and target requirements survive linking and
specialization so that each later stage can make a stronger decision without
rediscovering information that the author or embedding already knew.

The generated [`config`](../reference/dialects/config/index.md) and
[`target`](../reference/dialects/target/index.md) references define those two
fact sources.

## Correctness and measurement stay beside the program

A [`check.case`](../reference/dialects/check/ops/case.md) is an executable SSA
program that creates inputs, launches code, obtains expected values, and states
the comparison policy. A
[`check.benchmark`](../reference/dialects/check/ops/benchmark.md) selects named
assignments from a case for timing. The benchmark row does not copy the setup or
invent a second workload description.

Checks are test-only symbols, not deployment entry points. Keeping them in the
same source gives formatters, verifiers, test runners, benchmark runners, and
agents one authoritative statement of the workload while allowing published
libraries to exclude the harness. The generated
[`check` dialect reference](../reference/dialects/check/index.md) covers input
generation, fixtures, requirements, expectations, and benchmark records.

## Command programs move the same model upward

A [`command.program.def`](../reference/dialects/command/ops/program-def.md)
composes kernel launches, parameter views, resources, and serial or concurrent
schedule regions into a reusable subgraph. Its leading specialization
arguments participate in staged specialization and launch-count evaluation;
its buffer bindings remain replaceable when the materialized program is
issued.

That separation allows one program to specialize around model structure and
target facts while still accepting different weights, cache storage, inputs,
and outputs. It is the same idea as a specialized kernel launch, applied to a
larger ownership boundary rather than hidden behind a separate graph compiler.

The outer module declares the exact kernel dependency and turns one concrete
1009-element invocation into a reusable command program. The kernel itself
retains its dynamic workload contract and can be launched with other counts by
other roots:

**Source:** [`loom/docs/examples/elementwise-transform/model.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/model.loom)

```loom title="model.loom"
--8<-- "examples/elementwise-transform/model.loom"
```

!!! info "Current command-program surface"

    Command-program source, preparation, target-neutral lowering, and immutable
    artifact infrastructure are present. A general installed-tool
    materialization path is not yet a published user surface, so the guide does
    not present an executable command-program tutorial yet.

The generated [`command` dialect
reference](../reference/dialects/command/index.md) documents the source
constructs that exist today.

## Follow one composition to Low

The three source listings above are repository `.loom` files, not prose copies.
With the Loom tools on `PATH`, this command formats them, links and specializes
the `@elementwise_transform` root, compiles its kernel for the generic GFX11
profile, and
prints every Loom command it runs:

```shell
loom/docs/examples/elementwise-transform/run.sh \
  gfx11-generic build/elementwise-transform/gfx11-generic
```

The resulting directory contains the specialized `elementwise-transform.loom`,
a VMFB, an HSACO, and the captured target Low IR. The documentation build
invokes that same script and regenerates the views below; neither output is
checked-in source.

=== "GFX11 kernel Low"

    ```loom
    --8<-- "generated/examples/elementwise-transform/elementwise-transform-gfx11.loom"
    ```

=== "Command-program Low"

    ```loom
    --8<-- "generated/examples/elementwise-transform/elementwise-transform-program.loom"
    ```

The GFX11 tab comes directly from the installed-tool workflow above. The
command-program tab shows compiler-owned materialization of the same linked
root. Until command-program materialization has a published installed-tool
surface, treat that tab as verified compiler output rather than a CLI recipe.

## Embedding chooses the deployment policy

Loom produces modules, target artifacts, launch contracts, diagnostics, and
structured compiler reports. The embedding decides when compilation happens,
where artifacts are cached, how executable code is loaded, and how work is
issued to a device runtime.

The public [`loomc` C API](../reference/c-api/index.md) exposes the in-memory
source, link, specialization, compilation, emission, and launch-configuration
boundaries needed to build those policies. Executable embedding examples in
this guide are included only from checked programs that build against that API;
the generated header reference remains authoritative for ownership, lifetime,
threading, and failure contracts.

[Embed kernel JIT compilation](../integration/jit-kernel.md) follows one of
those checked programs from targetless source through launch evaluation and an
in-memory native executable.

## Working rules

- Keep reusable algorithms targetless until a real implementation requirement
  needs target facts.
- Use an exact call for an exact helper and a contract application when the
  implementation is intentionally substitutable.
- Use configuration for compile-time product choices, workload arguments for
  per-launch shape, and launch arguments for the device ABI.
- Put correctness policy in `check.case`; make benchmarks select those checked
  workloads instead of reconstructing them.
- Keep motifs as functions and templates. Add a kernel ABI where the program
  actually becomes dispatchable, and a command-program ABI where a reusable
  subgraph becomes materializable.

Continue with [Source modules and canonical
text](../guide/source-modules.md) to learn the source and composition contract,
or browse the [generated language reference](../reference/index.md) for exact
syntax and operation contracts. If the Loom tools are not yet on `PATH`,
[Acquiring Loom](acquiring-loom.md) describes the current installation status.
