# Format and verify source

`loom-lint` checks authoring policy that is meaningful to people and agents but
does not change program behavior. `loom-format` independently parses, verifies,
and prints complete Loom modules. Keeping those contracts separate lets the
formatter canonicalize syntax without inventing semantic names.

## Check authoring policy

Lint one or more explicit source files:

```shell
loom-lint motif.loom kernel.loom model.loom
```

The initial rule requires constant SSA names to carry a program role, such as
`%batch_size`, or use the compact `%c<literal>` form, such as `%c512`. A name
such as `%fivehundredtwelve` only repeats the literal and fails with its source
location and the `constant-name` rule identifier.

`loom-lint` also accepts `.loom-test` containers and checks only authored input
sections. Runner-owned expected output is excluded. Inputs are always explicit;
the tool does not discover a repository, build graph, or Git change set.

`loom-format` parses, verifies, and prints complete Loom modules. It is the
source equivalent of a language formatter and the text/bytecode conversion
boundary for linkable modules.

## Check canonical source

Check one file:

```shell
loom-format --check kernel.loom
```

Check several independent modules in one process:

```shell
loom-format --check motif.loom kernel.loom model.loom
```

Each file is parsed and verified as its own module. A successful batch prints a
summary and exits with status zero. A batch reports every invalid or
noncanonical input before returning failure, so one early file does not hide
the rest of the change set.

`--check` writes no source output. It accepts text input only; a single check
may read standard input, while a multi-input check names files.

## Rewrite canonical source

Format one or more verified text files in place:

```shell
loom-format --in-place motif.loom kernel.loom model.loom
```

The formatter rewrites only noncanonical files and reports changed, unchanged,
and failed counts. A file that does not parse or verify is never overwritten.
`--check` and `--in-place` are mutually exclusive.

Canonical formatting owns syntax and layout, not style policy. It preserves
comments, SSA names, supported source spellings, and intentional blank-line
grouping while making the surrounding representation deterministic. Semantic
naming conventions such as `%batch_size` versus `%fivehundredtwelve` are a
separate source-quality contract described in [Source modules and canonical
text](../guide/source-modules.md#names-should-preserve-program-meaning).

## Conversion verifies the module

Without `--check` or `--in-place`, `loom-format` converts one input and writes
the result to `--output` or standard output. Text-to-text conversion is useful
for generated source:

```shell
loom-format generated.loom --from=text --to=text --output=canonical.loom
```

Input defaults to standard input when no path is given:

```shell
loom-format --from=text --to=text --output=canonical.loom <generated.loom
```

The complete module must parse and verify before output is written. Conversion
therefore cannot be used to make an invalid module look canonical; missing
exact declarations, incompatible signatures, malformed regions, and violated
operation contracts remain errors.

## Convert text and bytecode

Package canonical source as Loom bytecode:

```shell
loom-format library.loom --from=text --to=bc --output=library.loombc
```

Recover canonical text from bytecode:

```shell
loom-format library.loombc --from=bc --to=text --output=library.loom
```

`--from=auto`, the default, recognizes Loom bytecode by its file magic and
treats other input as text. Explicit formats are useful in scripts because they
state the expected boundary. `bc` and `bytecode` are equivalent flag values.

Text and bytecode carry the same linkable program. Conversion does not select a
target, bind configuration, run optimization passes, or emit executable code.
Use [`loom-link`](link-and-package.md) to combine modules and `loom-compile` to
produce a runtime or target-native artifact.

## Diagnose a failure at its owner

Formatter failures establish a source-module problem before composition:

| Failure | Meaning |
| --- | --- |
| Noncanonical text | The module is valid; `loom-format --in-place` produces its canonical spelling. |
| Referenced symbol is not defined | The module lacks an exact definition or declaration for that reference. Add the declaration owned by this source boundary. |
| Signature or operation verification error | The authored contract is internally inconsistent. Linking another file cannot repair it. |
| Bytecode decode error | The input is not valid Loom bytecode for the active toolchain. Recover or regenerate the module rather than treating it as text. |
| Unknown target-Low syntax | The installed formatter does not have the target descriptor required to parse that Low assembly form. Use a tool distribution containing that target. |

A module that formats successfully can still fail later because its declared
dependency is absent from the supplied libraries, no template provider matches
the selected facts, or a target cannot lower a valid source operation. Those
failures belong to linking, specialization, or compilation and remain visible
at those boundaries.
