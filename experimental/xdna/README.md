# Experimental XDNA execution through libamdf

This directory contains temporary executable and prepared-command adapters and
a standalone runner. They bridge the HAL image loader to `libamdf/` during
native bring-up. The adapters own ELF interpretation and cold relocation;
libamdf owns native device admission, scoped memory, and range submission.
See [XDNA native execution](../../libamdf/docs/xdna.md) for that boundary.

`iree-xdna-run` executes one entry from an intact Loom `.xdna` file through the
experimental adapters and libamdf. It selects the image target from the
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

Each run creates one device and context. The executable adapter parses,
qualifies, and lowers the image without creating native resources. The runner
then obtains the context's private memory scope, allocates one instruction
backing at the scope's required granularity, and maps the used prefix. Cold
preparation validates the bindings and writes fixed DMA addresses into that
storage. Publishing the bytes is an explicit cache operation.

The first submission selects the initialization-plus-execution range; later
`--invocation_count` iterations select only the execution range. Both occupy
the same allocation and keep the same bindings. ARRAY and CONTROL describe
those image-layer contents, not driver objects. libamdf receives memory handles
and byte ranges, with no per-submission binding list or argument patching.

The runner waits for each finite command before submitting the next. The
image's output DMA wait, together with native command retirement, establishes
completion for this fixture; retirement alone does not prove arbitrary
autonomous tile work is finished. Callers own the lifetime of every indirectly
referenced buffer. The runner waits without a hidden deadline, then destroys
the queue and prepared command, unmaps and frees instruction storage, releases
the executable and data bindings, and finally closes the context, device,
endpoint, and instance. A native error is reported without retry. Failed
cleanup stops at its ownership boundary and returns failure.

The adapter tests use real image parsing and caller-owned storage without a
fake native provider. CLI tests exercise host argument and file ownership
without creating a device.

The native consumer tests in `cts/` select the matching canonical compiler
image, allocate data and instruction backing through the public memory scopes,
and execute three different inputs through one cold-prepared command. They
check all 48 integer products, native retirement, byte-for-byte instruction
immutability, and caller-ordered teardown. Each output is poisoned before its
submission so missing writes cannot pass. Both process- and instance-scoped
native lifetimes use the shared CTS device owner.

```sh
iree-bazel-test --config=asan //experimental/xdna/cts/...
iree-cmake-test -R '^iree/experimental/xdna/cts/'
```

These tests carry the XDNA hardware requirement and share the AMDGPU resource
group with native and interop CTS. The same sources run on Linux and Windows;
hosts without an XDNA endpoint or a matching compiler fixture report a skip.

## Warm execution benchmarks

`benchmarks/execution_benchmark` measures native publication of the same
canonical multiplication program. It retains one device, context, queue and
set of allocations across every row and repetition. Image loading, cold
relocation, instruction publication and the first initialization command all
finish before measurement. Later submissions reuse immutable instructions and
resident data addresses; libamdf receives only the prepared command range.

| Row | Timed region |
| --- | --- |
| `XdnaExecution/Submit` | One native submission, excluding its completion wait. |
| `XdnaExecution/SubmitAndWait` | The same submission plus native completion wait. |

Each iteration publishes changed inputs and poisoned output before timing.
After completion, outside timing, the caller checks native retirement, all
input/output values, and guard regions. It also checks instruction immutability
around each repetition. Completion waits use the infinite timeout contract;
unexpected native errors or incorrect output terminate the benchmark instead
of producing later samples. Successful execution checks complete teardown.

The generated smoke test uses the XDNA hardware requirement and shared AMDGPU
resource group, like the native CTS:

```sh
iree-bazel-test --config=asan \
  //experimental/xdna/benchmarks:execution_benchmark_test
iree-cmake-test -R '^iree/experimental/xdna/benchmarks/'
```

Performance runs use an optimized, non-sanitized build of the exact benchmark
target. After building, run its executable with fixed iteration counts, for
example `--benchmark_min_time=200x --benchmark_repetitions=5`, and retain JSON
with `--benchmark_out=execution.json --benchmark_out_format=json`. Fixed counts
bound the untimed completion and verification work in submit-only rows.
Measurements require an otherwise idle device and host, separate from builds
and other hardware jobs. The two rows describe warm kernel-mediated dispatch,
not cold setup, pipelined throughput, or autonomous user-mode scheduling.
