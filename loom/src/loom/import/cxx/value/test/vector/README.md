# Vector source semantics

These source goldens cover native C++ vectors and Loom's shaped vector facades.
Cases retain exact lane types, operand wiring, evaluation order, and object
layout through verified IR. They use the ordinary `// ====` container and
`--update` workflow.

| Contract | Fixtures |
| --- | --- |
| Construction, omitted lanes, and scalar broadcasts | `constructors`, `vector_constants`, `vector_list_assignment` |
| HIP vector ABI, named lanes, and mixed-width kernel composition | `hip_vector_types` |
| Native promotions, comparisons, and narrow lane arithmetic | `native_arithmetic`, `packed_arithmetic`, `vectors` |
| Shaped arithmetic and value dimensions | `vector_arithmetic`, `vector_slices`, `vector_slices_family` |
| Numeric conversion and packed representations | `vector_conversion`, `vector_decode`, `group_dot`, `packed_slices` |
| Lane updates and memory projections | `vector_lane_updates`, `vector_memory`, `padded_vectors` |
| Vector blocks, scalar tails, and independent strides | `vector_spans` |
| Lookup, reductions, and seeded products | `vector_lookup`, `vector_math`, `shaped_intrinsics`, `shaped_composition` |

For example, the effectful list-assignment case checks four element calls in
source order, followed by the destination call and a store through the returned
pointer. The padded-vector cases check eight-byte object stride together with
three-lane accesses. Hardware preservation of padding bytes is a separate
execution contract.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/vector:compiler_test \
  //loom/src/loom/import/cxx/value/test/vector:vector_lane_updates_roundtrip_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/value/test/vector:vector_list_assignment \
  --test_arg=--update
```

The [shared runner guide](../../../../../tools/loom-check/README.md) explains
source modes, exact expectations, and updates. Numerical execution, target
register packing, and device scheduling belong to their respective owners.
