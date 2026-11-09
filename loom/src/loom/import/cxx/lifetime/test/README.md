# Object lifetime

These source cases assert destruction at lexical exits, reverse destruction of
members and array elements, loop-condition cleanup, parameter ownership, and
full-expression temporary lifetimes. Exact IR expectations preserve call order
and conditional cleanup; no target code is executed.

Cases share `automatic_destruction.cxx-test` through `// ====` sections. The
[object suite](../../value/test/object/README.md) owns initialization, reference
identity, closure captures, and structured bindings, including their source
lifetime consequences.

The neighboring `origins_test` checks construction identity and the reference
forms that extend lifetimes. The VM's
[temporary lifetime cases](../../../../target/arch/vm/test/temporary_lifetimes.cxx-test)
execute returned copies, closure cleanup, distinct equal-valued objects, and
references into nonzero-offset bases. The source roundtrip target executes
those same observations again after public C++ export and reimport.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/lifetime/test:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/lifetime/test:automatic_destruction --test_arg=--update
```

The [shared runner guide](../../../../tools/loom-check/README.md) describes exact
expectations and diagnostic matching.
