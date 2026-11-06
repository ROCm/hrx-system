# Kernel bindings and memory effects

The importer translates source kernel contracts and explicitly ordered memory
effects into verified IR. These cases assert that translation without selecting
a device or executing a kernel.

Atomic cases distinguish returned-old and returned-new values, pointer origins,
source widths, signed selectors, memory order, and scope. Compare-exchange cases
also retain argument effects and the conditional update of the expected value.
Related source cases use `// ====` and complete output from the ordinary updater.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/kernel:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/binding/test/kernel:atomic_compare_values --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
