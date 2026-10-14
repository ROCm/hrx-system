# Pixel-observing finite image component

This parameter-free component consumes the [finite image request
packet](../../media/image_request.h) through the ordinary finite model ABI.
Its VM preparation declares one direct device dispatch; there is no checkpoint,
tokenizer, command program, or native canned response.

For the first decoded RGB8 pixel `(R, G, B)`, projected row `i` contains four
little-endian `i32` values: `[R + i, G, B, i]`. Rows have a 16-byte stride. The
source result plan exposes exactly the packet's requested row count, with
matching device and feedback extents. The retained-text fixture consumes the
first word as its synthetic projected feature. This is a dataflow and lifetime
witness, not neural projection math or a performance model.

The component accepts one image per cohort, 1–64 output rows, and packets up to
65,536 bytes. Metadata is opaque: the kernel reads pixels at the packet's
published pixel offset, independently of the metadata length. Multiple images
are separate finite invocations, matching the serialized media consumer.

The prepare and control roots are `prepare.loom` and `control.loom`; the provider
catalog includes `project.loom`. Asset directory `.` and options `{}` suffice.
The package is a component, not a standalone HTTP model registration.

The adjacent test executes real PNG encoding/decoding, request construction,
VM preparation, native device dispatch, feedback, and detached result ownership:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //loom/src/loom/tooling/target/cpu/serve:projector_fixture_test \
  //loom/src/loom/tooling/target/amdgpu/serve:projector_fixture_test
```
