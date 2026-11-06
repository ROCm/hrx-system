# Source contracts

These cases assert how C++ authoring contracts become verified Loom IR:
named configuration, incoming buffer promises, retained predicate facts, and
their source diagnostics.
The tests retain symbolic values and source ownership without selecting an
execution target. Related cases use `// ====` separators and exact output.

| Contract | Fixtures |
| --- | --- |
| Named configuration, aliases, definitions, and typed reads | `config*` |
| Predicate identity, source arithmetic, scope, and bounded indexing | `assumption*` |
| Entry alignment, redeclarations, and pointer replacement | `parameter_alignment*` |
| Independent roots, scoped promises, and possible aliases | `parameter_noalias*` |

For example, reassigning an annotated input to an unmarked pointer must not
transfer the entry promise. The exact cases preserve that replacement root
and its reload after a potentially aliasing write. A row-gather case combines
independent roots, alignment, workgroup coordinates, and vector conversion.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/contracts:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/contracts:config --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
