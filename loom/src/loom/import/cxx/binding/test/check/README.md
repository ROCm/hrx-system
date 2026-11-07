# Source check construction

These fixtures test C++ check declarations as compiler input. Expected IR
records the actual subject/oracle calls, scalar and record projections,
tensor identities, slice origins, dependent shapes, comparison policies, and
benchmark references. No check program executes in this package.

`check` and `header_cases` cover declaration discovery and ordinary calls;
`check_records` covers flattened result components; `check_resources` covers
aliasing tensor handles; `check_scenarios` covers trial and comparison regions.
The typed expectation and vector tensor fixtures retain type and shape rules,
with invalid inputs diagnosed at the source boundary.

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
