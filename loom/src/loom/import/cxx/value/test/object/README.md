# Object values and initialization

These `.cxx-test` fixtures check source object identity, value snapshots, array
partitions, and initialization order in verified Loom IR. Related cases share
files through `// ====` sections and retain exact expectations maintained with
`--update`.

| Contract | Fixtures |
| --- | --- |
| Record copies, returned values, and member aliases | `array_records` |
| Nested arrays, narrow fields, booleans, and vector components | `array_records`, `record_arrays` |
| Source layout, pointer-to-array strides, and qualified access | `record_arrays` |
| Completed-subobject visibility and destination copy elision | `record_initialization` |
| Defaults, zero initialization, and conditional initializer effects | `record_initialization` |

For example, a by-value helper may change a dynamically selected array member
without changing the caller's saved record. The golden checks the flattened
call operands, independent storage, and returned components. An initialization
case checks that a later clause reads the destination element already initialized
by an earlier clause. Ordinary source cleanup and private-storage promotion
produce compact results where the source admits them.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/object:compiler_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/object:array_records --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) describes
source options, exact expectations, and diagnostic matching.
