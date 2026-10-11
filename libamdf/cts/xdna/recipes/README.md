# CPU and XDNA memory recipes

The grouped `execution` corpus runs a finite arithmetic program through the
runtime-loaded public API. Its [ELF reader](../util/executable.h) and
[Loom program](../programs/README.md) are CTS-only. The ordinary build compiles
the checked-in `.loom` source and embeds the native images. The test executable
uses no runtime image-loader dependency.

`CpuXdnaRecipeTest.AllocatedRoundTrip` uses system allocations accessible to
the NPU. `RegisteredRoundTrip` retains separately allocated CPU storage through
registration, execution, unmapping and native attachment release. Each case
borrows the cached device and owns its context, queue and memory resources.
Process and instance native lifetimes are separate invocations of the same
dynamic executable.

The round-trip cases perform eight generations of sixteen unsigned 32-bit
products.
Inputs change between generations, output words start with the complement of
the expected result, and every byte outside the logical payload is guarded.
The oracle checks both inputs, the output, allocation-granularity padding and
the complete command owner. Repeated submissions retain the same bound command
bytes and independently queried DMA addresses.

`AllocatedFirstDeviceWriter` and `RegisteredFirstDeviceWriter` leave output
payload untouched on the CPU until the first completed NPU overwrite. Only
inputs and commands are published before submission; output is acquired once
after native completion and checked against sixteen nonzero products.
`RegisteredDirtyFirstDeviceWriter` writes old contents before registration,
then follows that same first-use path without a caller output flush and checks
that neighboring caller contents survived registration and execution. These
cases exercise initial cache readiness independently of the guarded round-trip
tests, whose output initialization already requires a CPU release.

The case queries both directional memory edges before execution. CPU
publication uses the mapping's exact flush recipe; CPU acquisition uses its
exact invalidate recipe. The device-side actions are NONE. Command publication
is separate and explicit. The finite output DMA join ends this program's
external payload accesses; its worker and compute DMA retain only tile-local
state. Every generation submits the complete establishing invocation, which
resets that local state before reconfiguration. Checked native completion
precedes output-first snapshots and all numerical comparisons.

The queue is destroyed before its instruction storage and context. Payload
attachments are released before registered CPU backing. A failed native
release stops destruction of dependent owners.

Enable `LOOM_BUILD` and `LOOM_TARGET_XDNA` alongside `AMDF_BUILD` to build
the compiler-generated fixtures.

```sh
iree-bazel-test --config=asan //libamdf/cts/xdna/recipes:execution
iree-cmake-test -R '^libamdf/cts/xdna/recipes/execution_dynamic'
```

The package carries the XDNA build/run requirement and the shared AMD hardware
resource group. Exact qualification invocations require both test names with
`--amdf_require_test=Suite.Case`; a missing target fixture or skipped required
case cannot count as successful execution. The fixtures cover the NPU4
Strix/Krackan and NPU5 Halo execution profiles.
