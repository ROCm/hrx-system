# Native Vulkan test support

These test-only C++ owners keep native Vulkan setup independent of the HAL,
libamdf, and Loom. They depend on the pinned Vulkan headers, GoogleTest, and
the platform's Vulkan loader. No Vulkan SDK library or shader compiler is
needed to build and run a consumer.

`Instance` owns the loader and native instance. `Device` creates exactly the
device described by its caller, including the caller's features, extensions,
and queues. `Buffer` owns an ordinary allocation and optional mapping;
`Compute` owns a descriptor-free pipeline with caller-specified SPIR-V and
push-constant ranges. `Submission` owns reusable command storage and completes
each submission with an infinite fence wait. Each owner is initialized once.
Its parent outlives it, and all GPU accesses complete before destruction.

The caller selects a physical device, establishes API capabilities, records
barriers and commands, owns shader argument layout, and checks numerical
results. Native function resolution is available through `Instance::Get` and
`Device::Get`; required core functions follow the requested Vulkan version,
while optional extension functions require explicit caller admission. External
memory imports and their FD/HANDLE ownership stay with the interop corpus.

Each fallible helper reports the native operation and result for an
`ASSERT_TRUE` call. `Instance::Create` preserves the native `VkResult` so tests
can distinguish an unavailable implementation from a broken initialization.
An accepted submission whose wait cannot establish completion fails the test
process before it can free reachable command storage or payloads. No implicit
device-wide idle, resource registry, retained parent, or process cache extends
these owners' lifetimes.

The libraries require `vulkan.api` for compilation. Device execution and
physical-device serialization are properties of the consuming test. For
example, the HAL borrowed-device corpus requires the Vulkan HAL and a Vulkan
device, and uses the shared `gpu-device` resource group:

```sh
iree-bazel-test --config=asan --//runtime/config/hal:drivers=vulkan \
  //runtime/src/iree/hal/drivers/vulkan/cts/interop:device_test
```

That test runs BDA compute before and after the complete lifetime of a wrapper
created by `iree_hal_vulkan_wrap_device`. It exercises real HAL queue work and
rejected wrapping, and verifies that no caller-supplied HAL allocation remains
when native execution resumes. The Vulkan loader is unloaded during fixture
teardown, so driver-owned leaks remain visible to sanitizers.

## Transform shader

`testdata/transform.comp` is the source for the small checked-in
`transform.spv` fixture. Its push constants contain a buffer device address at
byte 0, a changing uint salt at byte 8, and a uint word count at byte 12. The
address identifies a four-byte-aligned view; every invocation outside the word
count exits. Tests own the CPU oracle and all guard regions.

Regenerate and validate the fixture with glslang and SPIRV-Tools:

```sh
glslangValidator --target-env vulkan1.2 -V \
  build_tools/vulkan/testing/testdata/transform.comp \
  -o build_tools/vulkan/testing/testdata/transform.spv
spirv-val --target-env vulkan1.2 \
  build_tools/vulkan/testing/testdata/transform.spv
```

Normal builds embed the reviewed binary as aligned data; they do not invoke
either shader tool. There are no descriptor sets or reflection helpers.
