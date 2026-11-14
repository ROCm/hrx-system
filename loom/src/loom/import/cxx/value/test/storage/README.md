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
| Ordinary pointer ABI, signed offsets, and cursor loops | `pointers`, `pointer_origins`, `pointer_walk` |
| Scoped restrict access paths, peers, returns, references, and receivers | `restrict_access*` |
| Selected roots and simultaneous pointer permutations | `buffer_selection` |
| Opaque transport, type recovery, and access qualification | `opaque_pointers*` |
| Typed allocation and borrowed memory-space refinement | `buffer_alloca*`, `buffer_memory_space*` |
| Literal payloads and independent initialized allocations | `inline_buffer_payloads` |
| Boolean value/byte representation, arrays, fields, and aliases | `boolean_storage*` |
| Packed class/member layout and unaligned typed accesses | `packed_memory` |
| Compound assignment, sequencing, and converted expression results | `storage_updates`, `storage_composition` |
| Observed accesses, qualified parameter objects, and strided subviews | `volatile*` |

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
