# Storage and pointer semantics

These source cases check object storage, allocation identity, byte origins,
typed accesses, and effects through aliases. Expectations stop at verified IR.
Related cases share a `.cxx-test` file with `// ====` separators; exact output
checks operands and order as well as the operations emitted.

| Contract | Fixtures |
| --- | --- |
| Local addresses, aliases, output parameters, and loop declarations | `local_addresses` |
| Array initialization, element origins, zero-fill, and volatile access | `local_arrays` |
| Addressable local objects across retained calls | `call_storage` |

For example, the guarded-array case returns the modified middle element and
both neighbors separately. The exact result distinguishes an update of the
right element from an update that merely produces the same sum. Other cases
preserve a mutable loop bound or compare-exchange's expected-value writeback.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/storage:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/storage:local_arrays --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
