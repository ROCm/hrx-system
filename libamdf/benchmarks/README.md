# Native memory and execution benchmarks

Memory lifecycle and warm command publication measure different boundaries.
The library benchmarks measure backing and access; the XDNA runtime consumer
measures execution of an already prepared image through libamdf.

## Memory

The GPU and XDNA memory benchmarks exercise the same public shared-library API
scenarios on Linux and Windows. Each executable reuses one device across all
cases and repetitions. Discovery, device activation, profile selection, and
full-range host correctness checks are outside timing.

Each family has four scenarios, named `GpuMemory/...` or `XdnaMemory/...`:

| Scenario | Timed operations |
| --- | --- |
| `Allocation` | Allocation, host mapping, device address lookup, and complete release. No explicit host write. |
| `Initialization` | The allocation lifecycle plus the first full-range host write. No explicit cache publication. |
| `Lifecycle` | The allocation lifecycle, first host write, and explicit cache publication. |
| `Publication` | Full-range host write and explicit cache publication of an already-resident mapping. |

All cover 4 KiB and 1 MiB ranges. Native allocation can itself initialize or
commit backing; `Allocation` excludes application first-touch, not native work.
The lifecycle rows are complete operations, not isolated stage measurements.
Their independent timings cannot establish precise stage costs by subtraction.
The address query requests the native GPU or XDNA DMA address for the selected
device. Explicit publication uses the mapping's native cache policy.

No memory scenario includes command submission or device completion; native
computation is covered by CTS and the execution benchmark below. The device is
activated but has not run a benchmark workload.
GPU VM residency and TLB invalidation costs can differ after execution, so an
exercised-context comparison must exercise both owners separately.

Build the optimized executable before collecting fixed-count measurements:

```sh
build_tools/bin/iree-bazel-build \
  //libamdf/benchmarks/xdna:memory_benchmark \
  -c opt --features=thin_lto --copt=-O3 --cxxopt=-O3 \
  --host_copt=-O3 --host_cxxopt=-O3 \
  --copt=-march=native --cxxopt=-march=native \
  --host_copt=-march=native --host_cxxopt=-march=native
bazel-bin/libamdf/benchmarks/xdna/memory_benchmark \
  --benchmark_min_time=64x --benchmark_repetitions=5 \
  --benchmark_out=memory.json --benchmark_out_format=json
```

Use `//libamdf/benchmarks/gpu:memory_benchmark` and its corresponding executable
path for GPU memory. Both use process-scoped native ownership.

Google Benchmark reports per-operation elapsed time and byte throughput. The
generated `memory_benchmark_test` performs a single-iteration smoke run through
the same shared GPU/XDNA hardware resource group as CTS. Family build and hardware
requirements are selected by package policy. Missing native capabilities are
reported as skips; native operation or byte-check failures terminate the run.

Measurements use nonsanitized optimized binaries, a quiet machine, and the same
compiler options and allocation/publication contract for every comparator.
CPU scaling, concurrent activity, and repetition spread remain part of the
result. Sanitizer and smoke-test output establish correctness, not performance.
The benchmark executable is not part of the installed libamdf distribution.

## Warm XDNA execution

[`//experimental/xdna/benchmarks:execution_benchmark`](../../experimental/xdna/benchmarks/execution_benchmark.cc)
uses the runtime ELF loader and the public libamdf API. It retains one device,
context, queue, prepared instruction range and data allocation set across cases
and repetitions. Image loading, relocation, cold initialization and allocation
are outside timing. Each iteration publishes fresh input and checks numerical
output, guards and instruction immutability outside timing.

| Scenario | Timed operations |
| --- | --- |
| `XdnaExecution/Submit` | Publication of one prepared native instruction range. Completion occurs outside timing. |
| `XdnaExecution/SubmitAndWait` | The same publication plus its completion wait. |

Both rows wait for each submission before reusing the queue, which admits one
unretired command. The submit-only row measures host publication cost, not
device throughput or completed-operation latency. Host input publication and
output acquisition are outside both rows, so the completed row is not a
measurement of the entire host-to-host application pipeline.

Build the execution target with the same optimized options shown above, then
run a fixed number of iterations:

```sh
bazel-bin/experimental/xdna/benchmarks/execution_benchmark \
  --benchmark_min_time=200x --benchmark_repetitions=7 \
  --benchmark_out=execution.json --benchmark_out_format=json
```

Fixed counts bound the real submission count when completion is excluded from
timing. Comparisons with XRT or another runtime match instruction bytes, work
per submission, residency, host cache operations and completion boundaries.
Allocation comparisons also match whether each owner's GPU VM has executed
work. Ratios between unlike lifecycle or submission shapes do not isolate
library overhead.
