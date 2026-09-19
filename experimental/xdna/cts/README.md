# Native XDNA execution coverage

`execution_test` and `execution_test_instance` exercise image loading, binding,
queue submission, readback and resource retirement through libamdf. They select
the appropriate image for the available device family.

`vector_copy_npu2_test` exercises C++ import, bytecode linking, compilation and
native execution on NPU2 hardware using the Strix Halo profile. It is an explicit
hardware test (`manual` in Bazel). Its compiler and importer dependencies are
optional; the common importer package has no dependency on this native consumer.

The same C++ source supplies scalar and 512-bit vector copy controls. Two workers
process three records each, covering zero, one, four, nineteen and twenty blocks,
nonzero byte origins, and different source/destination strides. An independent
scalar oracle checks every output word, untouched gaps, a trailing binding guard
and unchanged input bytes. These checks qualify correctness, not performance.

```sh
iree-bazel-build --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=xdna --//loom/config/emit:enable=xdna \
  --//libamdf/config:enabled=true \
  //experimental/xdna/cts:vector_copy_npu2_test

benchmark-lock --label=xdna-copy -- \
  iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=xdna --//loom/config/emit:enable=xdna \
  --//libamdf/config:enabled=true \
  //experimental/xdna/cts:vector_copy_npu2_test
```
