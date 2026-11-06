# Source language and constant evaluation

These cases exercise C/C++ source admission and translation into verified Loom
IR. Each `// ====` case has its own source and exact expected output or expected
diagnostics, maintained by the shared `--update` workflow.

| Contract | Fixtures |
| --- | --- |
| Standard selection, parsing, and deduced types | `standards`, `frontend`, `auto_deduction` |
| Source widths and pointer arithmetic | `data_models` |
| Macro expansion and source locations | `macro_operators`, `locations` |
| Lexical names and initialization dependencies | `local_names`, `initializer_dependencies` |
| Enum representation, promotions, and memory values | `enum_semantics`, `enums`, `enum_storage` |
| Constant aggregates and references | `aggregate_constants`, `constexpr_aggregate_loads`, `constexpr_references` |
| Constant storage identity and lifetime | `constexpr_storage`, `constexpr_destruction` |
| Source selection and assertions | `constexpr_semantics`, `static_assert` |

Static assertions establish source-only properties such as enum underlying
types across LP64, LLP64, and ILP32. Exact IR establishes observable translation:
for example, a byte enum increment must truncate before widening its result,
and an inner type alias must stop shadowing the outer alias when its block ends.
The enum storage cases also check element-sized pointer arithmetic and typed
accesses independently of a device launch.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/source/test:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/source/test:enums --test_arg=--update
```

The [shared runner guide](../../../../tools/loom-check/README.md) describes
source options, declared includes, exact expectations, and diagnostic matching.
