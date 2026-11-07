# Object lifetime

These source cases assert destruction at lexical exits, reverse destruction of
members and array elements, loop-condition cleanup, parameter ownership, and
full-expression temporary lifetimes. Exact IR expectations preserve call order
and conditional cleanup; no target code is executed.

Cases share `automatic_destruction.cxx-test` through `// ====` sections. The
[object suite](../../value/test/object/README.md) owns initialization, reference
identity, closure captures, and structured bindings, including their source
lifetime consequences.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/lifetime/test:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/lifetime/test:automatic_destruction --test_arg=--update
```

The [shared runner guide](../../../../tools/loom-check/README.md) describes exact
expectations and diagnostic matching.
