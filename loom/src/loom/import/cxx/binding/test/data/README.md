# Typed data bindings

These source tests preserve logical indices, physical offsets, view shapes,
encodings, and the dependent values carried through calls and control flow.
They assert verified IR without selecting a device or executing the program.

`typed_view_composition` joins pointer origins, static/dynamic layout factories,
predicated copies, loop-carried subviews, and encoded storage with their real
source callers. The remaining fixtures isolate individual facade operations,
invalid declarations, and dependent signatures. Export/reimport coverage keeps
the same dependent shape and layout identities through the source round trip.

Related cases use `// ====` and exact output maintained by the ordinary updater:

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/data:compiler_test \
  //loom/src/loom/import/cxx/binding/test/data:dependent_view_loops_roundtrip_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/data:typed_view_composition --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
