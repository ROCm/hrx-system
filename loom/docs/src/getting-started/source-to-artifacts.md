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

## Compose explicit modules

Every `.loom` file verifies independently. An exact call into another module
therefore needs a local declaration that states the complete contract. The
link invocation or embedding supplies the libraries that may satisfy it:

**Source:** [`loom/docs/examples/module-composition/root.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/module-composition/root.loom)

```loom title="root.loom"
--8<-- "examples/module-composition/root.loom"
```

The layer has its own declared dependency on `@scale`. The checked workflow
links only the available layer, writes a standalone partial `.loombc`, reloads
it, and then supplies the kernel library:

```shell
loom/docs/examples/module-composition/run.sh build/module-composition
```

| Source contract | Selection behavior |
| --- | --- |
| Exact declaration | Resolve the compatible direct definition or unique exported library definition from the explicitly supplied source universe. |
| `template.decl` plus `template.apply` | Select an eligible family implementation from the supplied libraries using facts, requirements, and priority. |
| Runtime import attributes on `func.decl` | Preserve an external ABI contract for runtime resolution instead of satisfying it from Loom libraries. |

Library paths and insertion order do not participate in symbol identity. The
application or build graph chooses the source universe; Loom declarations state
what that program requires. A partial link may preserve reachable declarations
for a later boundary, while a closed link rejects any exact requirement that
remains unresolved.

The [linking workflow](../workflows/link-and-package.md) follows partial and
transitive composition in detail. The [source-module
chapter](../guide/source-modules.md#definitions-declarations-and-roots)
defines the language contract.

## Exact calls and selectable implementations

Loom separates naming one implementation from requesting an implementation
that satisfies a contract.

| Source construct | Meaning |
| --- | --- |
| [`func.def`](../reference/dialects/func/ops/def.md) and [`func.call`](../reference/dialects/func/ops/call.md) | Define and call an exact helper by symbol. |
| [`template.decl`](../reference/dialects/template/ops/decl.md) | Declare a stable compile-time family contract in the module that uses or implements it. |
| [`template.def`](../reference/dialects/template/ops/def.md) | Provide one selectable implementation of that family. |
| [`template.apply`](../reference/dialects/template/ops/apply.md) | Request an implementation of that family at compile time. |
| [`template.call`](../reference/dialects/template/ops/call.md) | Call one named template implementation without candidate ranking. |

An exact bit-manipulation helper is naturally a `func.call`. An operation whose
best implementation depends on element format, shape facts, subgroup size, or
target capabilities is naturally a `template.apply`. During specialization, Loom
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

The complete operation inventories live in the generated
[`func`](../reference/dialects/func/index.md) and
[`template`](../reference/dialects/template/index.md) dialect references.

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
uses a portable 64-thread workgroup, and applies the motif contract without
naming either provider. The target profile still selects the eligible motif
implementation independently of that launch geometry:

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
| Target facts | Link or compile product boundary | Describe capabilities such as subgroup size, supported types, and resource limits for the selected executable root. |
| Kernel workload values | Launch-configuration evaluation | Compute the physical launch for one workload without pretending the values were compile-time constants. |
| Kernel launch arguments | Device issue | Carry scalars and buffers through the selected target ABI into the kernel body. |
| Command specialization arguments | Command-program specialization | Shape source control flow and launch computation; direct counts and scalar command data must become exact before portable preparation. |
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
arguments participate in staged specialization and pure launch-count
computation; its buffer bindings remain replaceable when the materialized
program is issued. Exact counts become direct commands, while intentionally
dynamic counts use an explicit `view<3xi32>` produced before an indirect
dispatch.

That separation allows one program to specialize around model structure and
target facts while still accepting different weights, cache storage, inputs,
and outputs. It is the same idea as a specialized kernel launch, applied to a
larger ownership boundary rather than hidden behind a separate graph compiler.

The outer module declares the exact kernel dependency and turns one concrete
1009-element invocation into a reusable command program. Its portable
workgroup geometry makes the physical count exact without loading the kernel
implementation. The kernel itself retains its dynamic workload contract and
can be launched with other counts by other roots:

**Source:** [`loom/docs/examples/elementwise-transform/model.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/model.loom)

```loom title="model.loom"
--8<-- "examples/elementwise-transform/model.loom"
```

!!! info "Command deployment products"

    `loom-compile --format=loom-command` materializes each selected command
    root as a portable `.loomcmd` and writes the shared executable-entry
    manifest. The target-specific kernel executable remains a separate
    artifact so an embedding can cache, replace, or prebuild it independently.

The generated [`command` dialect
reference](../reference/dialects/command/index.md) documents the source
constructs that exist today.

## Follow one composition to Low

The three source listings above are repository `.loom` files, not prose copies.
With the Loom tools on `PATH`, this command formats them, links the selected
program against the explicit provider universe, specializes that closure for
the generic GFX11 profile, and emits both kernel and command products. It
prints every Loom command it runs:

```shell
loom/docs/examples/elementwise-transform/run.sh \
  gfx11-generic build/elementwise-transform/gfx11-generic
```

The resulting directory contains the closed target-specialized
`elementwise-transform.loom` module, the GFX11 HSACO, a command manifest, one
portable `.loomcmd`, and the captured target Low IR. The
`--target=amdgpu:gfx11-generic` link argument makes subgroup facts
available before template-provider pruning, so the wave32 implementation is
selected without pulling the portable alternative into the product. The
documentation build invokes the same script and regenerates the views below;
neither output is checked-in source.

=== "GFX11 kernel Low"

    ```loom
    --8<-- "generated/examples/elementwise-transform/elementwise-transform-gfx11.loom"
    ```

=== "Command-program Low"

    ```loom
    --8<-- "generated/examples/elementwise-transform/elementwise-transform-program.loom"
    ```

The GFX11 tab comes directly from the installed-tool workflow above. The
command-program tab is the readable Low view of the `.loomcmd` produced from
the same closed program: launch configuration is projected as pure caller IR,
folded to an exact direct count, and lowered while the device implementation
remains unopened. The deployment artifact is binary and target-neutral; the
HSACO independently supplies its logical kernel entry.

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
