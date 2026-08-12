# Source modules and canonical text

A `.loom` file is a complete, independently verifiable module. It may contain
functions, templates, kernels, checks, command programs, target descriptions,
configuration declarations, and other symbols, but it does not inherit hidden
symbols or settings from the directory around it.

In this chapter, you will learn:

- what a source module owns;
- how definitions, declarations, libraries, and roots compose;
- why `.loom` and `.loombc` are two encodings of the same program;
- what canonical formatting preserves; and
- how source names carry information that the compiler should not discard.

## A module verifies on its own

`loom-format --check` parses and verifies the complete module before checking
its canonical spelling:

```shell
loom-format --check motif.loom kernel.loom model.loom
```

An exact symbol reference must resolve to a definition or a compatible
declaration in that module. The formatter does not search neighboring files,
scan a library directory, or infer a dependency from a missing call. This makes
the source boundary honest before a linker or build system is involved.

The mental-model command module demonstrates the pattern. It declares the
kernel signature it calls, then defines a command program using that exact
symbol:

```loom title="model.loom"
--8<-- "examples/mental-model/model.loom"
```

[`kernel.decl`](../reference/dialects/kernel/ops/decl.md) states the dependency
and its workload and launch signatures. It does not say where the definition is
stored. The link invocation supplies modules that may satisfy declarations;
the source itself remains meaningful when stored as text, packaged as bytecode,
or provided from memory through the C API.

The same rule applies to exact [`func.call`](../reference/dialects/func/ops/call.md)
and [`func.decl`](../reference/dialects/func/ops/decl.md) pairs. Contract-based
[`func.apply`](../reference/dialects/func/ops/apply.md) is different: it asks the
visible provider set for an implementation of a named contract rather than
naming one exact function symbol.

## Definitions, declarations, and roots

A definition owns a body. A declaration owns the signature expected from
another module or runtime boundary. Linking matches compatible declarations and
definitions among the inputs it was explicitly given.

A root states which symbol matters for one operation. Roots may be chosen by a
tool invocation, exported by a module, or selected through an embedding API.
Starting from those roots, linking retains the reachable declarations,
definitions, configuration, target records, and provider candidates needed to
form the requested program. Unreachable private helpers and unselected
providers can disappear.

This has two important consequences:

- Passing a library to the linker does not make every symbol in that library
  live.
- Passing one module does not authorize the linker to discover or pull in the
  transitive world. A declaration that no supplied module satisfies remains a
  declaration in a partial link and must be satisfied before a consumer that
  requires a closed program can use it.

The path or input order of a provider module does not give its templates
priority. Template matching uses contracts, signatures, facts, requirements,
and explicit provider priority. Module placement is a packaging choice, not a
hidden overload-resolution rule.

## Modules are not target boundaries

Reusable modules normally contain no selected target. A compile invocation
specializes the executable roots it is compiling for a target profile; it does
not turn every helper or every other root in the module into that target.

The checked motif below contains a wave32 provider and a portable provider. The
module itself is still target-independent:

```loom title="motif.loom"
--8<-- "examples/mental-model/motif.loom"
```

The exact helper and both providers are private: none carries the `public`
modifier. Explicitly supplying this module as a library makes them visible to
the linker for the requested composition without exporting their names from
the resulting module interface.

The specialized provider is eligible only when facts for the selected function
establish a subgroup size of 32. The fallback carries no target requirement.
This keeps one library useful across target families while allowing exact
specialization at the leaf that needs it.

## Text and bytecode preserve the same program

`.loom` is canonical human-readable source. `.loombc` is the compact bytecode
encoding of the same linkable module. Bytecode is not native code and does not
imply that target selection or specialization has already happened.

Convert a module in either direction with `loom-format`:

```shell
loom-format motif.loom --from=text --to=bc --output=motif.loombc
loom-format motif.loombc --from=bc --to=text --output=motif.loom
```

The converter verifies the complete input before writing output. A bytecode
library can therefore be indexed and linked without reparsing text, while a
round trip back to `.loom` recovers canonical source for inspection, review, or
migration.

Comments, SSA names, source syntax choices, and intentional blank-line grouping
are part of Loom's source representation and survive supported text/bytecode
round trips. Optimization and target lowering may of course produce a different
program; format conversion alone does not.

## Canonical formatting is the source contract

Canonical text removes whitespace and layout churn while retaining the source
structure people use to understand a program. Check several files without
writing them:

```shell
loom-format --check motif.loom kernel.loom model.loom
```

Rewrite only files that are valid but noncanonical:

```shell
loom-format --in-place motif.loom kernel.loom model.loom
```

Each input parses and verifies independently. Invalid source is reported and is
not rewritten. In-place mode reports how many files changed, remained
canonical, or failed.

Canonical formatting matters beyond aesthetics. Stable text gives reviews,
source migrations, compiler-report paths, and automated iteration a common
representation. When a source-format migration changes one construct, the diff
can show that construct rather than unrelated indentation or line wrapping.

## Names should preserve program meaning

SSA names do not affect execution, but they are valuable source facts for
people, diagnostics, reports, and agents. A value named `%batch_size` says why
the value exists. A value named `%fivehundredtwelve` merely spells its current
literal in words.

When a constant has a program role, use that role:

```loom
%batch_size = index.constant 512 : index
```

When the literal itself is the only available meaning, use the compact numeric
fallback. A type suffix distinguishes otherwise identical literals when
needed:

```loom
%c512 = index.constant 512 : index
%c0_i32 = scalar.constant 0 : i32
%c0_f32 = scalar.constant 0.0 : f32
```

Literal equality does not imply semantic identity. Two values that both happen
to be 16 may still deserve names such as `%channels_per_group` and
`%lane_partition_width` when they represent different constraints. Folding
those names into `%c16` would discard information the author already had.

The same principle applies above SSA values. Contract names describe reusable
semantic demands, configuration names describe product choices, and symbol
names describe the role an embedding or another module depends on. Target
mnemonics and one model's branding belong in those names only when they are
actually part of the contract.

## Follow the checked composition

The [mental-model example](../getting-started/mental-model.md#follow-one-composition-to-low)
links `motif.loom`, `kernel.loom`, and `model.loom`, selects one command root,
binds its configuration, and compiles the reachable kernel for GFX11. Its
`run.sh` prints every public command before executing it, and the documentation
build regenerates the displayed Low products from that same path.

Use [Format and verify source](../workflows/format-and-verify.md) for the complete
`loom-format` task contract and [Link and package
modules](../workflows/link-and-package.md) for archive, selective-link, config,
and plan commands.

Continue with [Values, types, and shapes](values-types-shapes.md), then follow
the guide through functions, memory, kernels, facts, checks, and command
programs as those ownership boundaries become relevant. For exact source
forms, use the generated [`func`](../reference/dialects/func/index.md),
[`kernel`](../reference/dialects/kernel/index.md), and
[`command`](../reference/dialects/command/index.md) references.
