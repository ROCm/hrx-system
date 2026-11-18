# Source check construction

These fixtures test C++ check declarations as compiler input. Expected IR
records the actual subject/oracle calls, scalar and record projections,
tensor identities, slice origins, dependent shapes, comparison policies, and
benchmark references. No check program executes in this package.

`check` and `header_cases` cover declaration discovery and ordinary calls.
The remaining fixtures exercise generator bindings, tensor and resource types,
slices, and comparison policies. The [check translation suite](../../../translation/test/check/)
covers calls to subjects defined in separately linked modules, result
flattening, trial bodies, and comparison regions.

Each `// ====` case is independently reviewable. Update and test through the
ordinary runner:

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/check:check_resources \
  --test_arg=--update
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/check:compiler_test
```

Numerical comparison and resource execution are contracts of the check runtime.
These source assertions establish what program the runtime receives.
