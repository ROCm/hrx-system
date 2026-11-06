# Callable and immutable symbols

These fixtures check C++ declarations, calls, and named immutable data at the
verified Loom IR boundary. Related source cases share a `.cxx-test` file with
`// ====` separators. Exact expectations check signatures and operand wiring;
diagnostic cases check source admission.

| Contract | Fixtures |
| --- | --- |
| Visibility, reachability, and explicit names | `symbols`, `functions` |
| External declarations and C ABI object layout | `function_declarations`, `c_abi` |
| Inline linkage and private template identities | `linkage` |
| Call effects and independent aggregate results | `callable_effects`, `functions` |
| Dependent view shapes, layouts, and origins across calls | `view_calls` |
| Template deduction and overload substitution | `template_calls`, `enable_if` |
| Immutable object identity, layout, and linkage | `immutable_objects` |
| Literal code units, storage, aliases, and volatile initialization | `immutable_strings`, `inline_strings` |
| Mutable reference export and reimport | `function_ties_roundtrip` |

The view-call cases separately import providers and callers. For example, a
crop's result includes an encoding and a view typed by that encoding; its caller
must consume both together. These cases establish the source-produced contracts.
Cross-module resolution and private-symbol renaming are owned by linker tests.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/symbol/test:all

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/symbol/test:view_calls --test_arg=--update
```

The [shared runner guide](../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
