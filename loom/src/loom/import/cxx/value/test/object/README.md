# Object values and initialization

These `.cxx-test` fixtures check source object identity, value snapshots, array
partitions, and initialization order in verified Loom IR. Related cases share
files through `// ====` sections and retain exact expectations maintained with
`--update`.

| Contract | Fixtures |
| --- | --- |
| Record copies, returned values, and member aliases | `array_records` |
| Mixed record loop state, nested selections, and output component origins | `record_values`, `records`, `records_invalid` |
| Nested arrays, narrow fields, booleans, and vector components | `array_records`, `record_arrays` |
| References, selected aliases, and mutation across calls | `reference_accessors`, `reference_parameters*` |
| Value/storage receivers, member effects, and receiver aliases | `member_functions*`, `operators` |
| Constructor delegation, default targets, moves, and destination identity | `record_constructors` |
| Source layout, pointer-to-array strides, and qualified access | `record_arrays`, `record_memory` |
| Padding, packing, alignment attributes, and nested offset queries | `layout_attributes*`, `offsetof*` |
| Typed member loads/stores and unsupported object representations | `record_memory*` |
| Completed-subobject visibility and destination copy elision | `record_initialization` |
| Defaults, zero initialization, and conditional initializer effects | `record_initialization`, `value_initialization`, `const_default_initialization` |
| Object qualification and invalid list initialization | `const`, `list_initialization_invalid` |
| Class conversions and overload precedence | `class_conversions*` |
| Closure captures and recursive forwarding | `lambdas` |
| Record, array, and tuple decomposition | `structured_bindings` |

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
