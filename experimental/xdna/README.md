# Experimental XDNA execution through libamdf

This directory connects Loom's native `.xdna` images to libamdf. The image
reader owns the immutable source and compact indexed metadata. Loading copies
explicit ranges straight into caller-owned backing; binding patches declared
address fields without interpreting native commands. libamdf owns device
admission, scoped memory and range submission. See
[XDNA native execution](../../libamdf/docs/xdna.md) for that boundary, the
[compiler target map](../../loom/src/loom/target/arch/amd/xdna/README.md)
for production of these programs, and the
[image format specification](../../runtime/src/iree/hal/drivers/amd/xdna/image/README.md)
for the storage and invocation ABI.

The load and bind operations allocate no memory and retain no resources. The
caller owns the image, mappings, native backing, contexts and logical HAL
buffers through actual terminal completion. Native command bytes remain opaque
to the loader; they are executable code, not sandboxed input.

`iree-xdna-run` executes one entry from an intact Loom `.xdna` file through the
runtime storage materializer and libamdf. It selects the image target from the
enumerated endpoint: Strix NPU4 `17f0:10` or Strix Halo NPU5 `17f0:11`.
The ELF must match that device's exact compiler profile identity; the shared
NPU2 array architecture does not make the images interchangeable. The platform
transport is native Linux DRM or Windows MCDM; the Linux path has no XRT, HSA,
or ROCr dependency.

From a checkout configured with libamdf and the XDNA family enabled:

```sh
iree-bazel-run --config=asan //experimental/xdna:iree-xdna-run -- \
  --image=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.xdna \
  --columns=1 --entry=mul_i32 --invocation_count=3 \
  --binding_memory=system \
  --binding=lhs.bin --binding=rhs.bin --binding=output-initial.bin \
  --output=2=mul-output.bin
```

For this fixture, the caller supplies three 64-byte files containing sixteen
little-endian i32 values each. The first two are inputs; the third initializes
the output. Compare `mul-output.bin` with the elementwise products, retaining
the low 32 bits of each result. On Strix NPU4, select the adjacent
`mul_i32_npu4.xdna` fixture instead; it has the same binding arithmetic and the
matching device identity.

`--columns` selects the logical context size; image qualification requires an
exact match. `--entry` names an export and defaults to ordinal zero when
omitted. `--device` selects an XDNA endpoint ordinal and defaults to zero.
`--binding_memory` selects provider-owned `system` memory or caller-owned
`registered_host` memory for every binding and defaults to `system`.

Each `--binding` supplies initial raw bytes in the entry's binding order. Its
file size determines the logical buffer length. Output buffers also receive
initial bytes, allowing a sentinel to expose missing writes. The image's
canonical binding records determine alignment and validate access and range
requirements. Each binding has one scope-created memory handle with explicit
access for the live device, a persistent host view, and a logical HAL buffer.
Registered bindings use caller storage aligned to the image contract and
deliberately offset within a native page whenever that alignment permits.
Successful teardown checks that every caller byte remains accessible after the
memory handle is destroyed.

Each `--output=ordinal=path` writes the selected buffer's complete logical
contents after successful command retirement and explicit cache invalidation.
Paths are overwritten. Multiple distinct bindings may be written. Output
comparison belongs to the caller or the artifact's accompanying checker. The
runner checks native completion status, not numerical correctness.

Each run creates one device and context. Image admission checks the exact
execution profile, backing requirements, load ranges, relocation fields and
invocation ranges. The runner allocates only the selected entry's backing:
command memory comes from the context's private scope and DMA catalogs use
ordinary device-addressable storage. Shared file ranges initialize their exact
destinations directly, and only declared zero-fill tails are cleared. Gaps are
undefined. Binding validates logical HAL ranges and patches DMA addresses;
publishing mapped writes remains an explicit cache operation.

Each independent invocation uses the entry's complete establishing command.
The adapter resolves that command once, and the runner reuses its immutable
backing for every `--invocation_count` repetition. Each submission resets and
configures the entry's device state, loads its tile programs, and performs the
input/output DMA. Host image loading, allocation and binding happen once.

The image also describes a continuation that can reuse resident workers, but
the time-sliced native contexts used here do not guarantee tile state survives
between submissions. Keeping a context and its backing alive does not establish
that guarantee. The finite adapter therefore selects invocation zero and does
not expose continuation selection. Role changes within a held invocation remain
compiled device behavior. libamdf receives memory handles and byte ranges, with
no per-submission binding list or argument patching.

The runner waits for each finite command before submitting the next. The
image's output DMA wait, together with native command retirement, establishes
completion for this fixture; retirement alone does not prove arbitrary
autonomous tile work is finished. Callers own the lifetime of every indirectly
referenced buffer. The runner waits without a hidden deadline, then destroys
the queue, unmaps and frees executable backing, releases
the image and data bindings, and finally closes the context, device,
endpoint, and instance. A native error is reported without retry. Failed
cleanup stops at its ownership boundary and returns failure.

## Independent execution benchmarks

`benchmarks/execution_benchmark` measures native publication of the same
canonical multiplication program. It retains one device, context, queue and
set of allocations across every row and repetition. Image loading, binding,
instruction publication and an initial correctness warmup finish before
measurement. Each measured submission reuses the complete setup-and-execution
command. Device setup repeats inside that command, including tile program
loading; its completion is included in the end-to-end row. libamdf receives
only the resolved immutable command range.

| Row | Timed region |
| --- | --- |
| `XdnaExecution/Independent/Submit` | One complete command submission, excluding its completion wait. |
| `XdnaExecution/Independent/SubmitAndWait` | The same submission plus device setup, execution and completion wait. |

Each iteration publishes changed inputs and poisoned output before timing.
After completion, outside timing, the caller checks native retirement, all
input/output values, guard regions, and instruction immutability. Completion
waits use the infinite timeout contract;
unexpected native errors or incorrect output terminate the benchmark instead
of producing later samples. Successful execution checks complete teardown.

Performance runs use an optimized, non-sanitized build of the exact benchmark
target. After building, run its executable with fixed iteration counts, for
example `--benchmark_min_time=200x --benchmark_repetitions=5`, and retain JSON
with `--benchmark_out=execution.json --benchmark_out_format=json`. Fixed counts
bound the untimed completion and verification work in submit-only rows.
Measurements require an otherwise idle device and host, separate from builds
and other hardware jobs. The two rows describe independent kernel-mediated
dispatch with prepared host resources. They exclude image preparation and do
not measure pipelined throughput or resident execution.
