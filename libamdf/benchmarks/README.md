# Native memory benchmarks

The library benchmarks measure native backing and access through the public
libamdf API. Application-facing execution measurements belong to the HAL
consumer, which owns image preparation, dispatch and completion.

Sustained XDNA results require the selected device's runtime power to be held
active for the measurement, with the previous policy restored afterward.
Deployment-representative runs instead preserve the deployment's normal power
policy and include its idle/wake behavior. These are separate result classes;
held-power numbers do not approximate default-policy user latency. The
[XDNA power and measurement contract](../docs/xdna.md#power-policy-and-measurement)
describes Linux control, verification, ownership and the required record.

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
computation belongs to the execution benchmark below. The device is
activated but has not run a benchmark workload.
GPU VM residency and TLB invalidation costs can differ after execution, so an
exercised-context comparison must exercise both owners separately.

Build the optimized executable before collecting fixed-count measurements:

```sh
build_tools/bin/iree-bazel-build --config=opt \
  //libamdf/benchmarks/xdna:memory_benchmark
bazel-bin/libamdf/benchmarks/xdna/memory_benchmark \
  --benchmark_min_time=64x --benchmark_repetitions=5 \
  --benchmark_out=memory.json --benchmark_out_format=json
```

Use `//libamdf/benchmarks/gpu:memory_benchmark` and its corresponding executable
path for GPU memory. Both use process-scoped native ownership.

Google Benchmark reports per-operation elapsed time and byte throughput.
Missing native capabilities are reported as skips; native operation or
byte-check failures terminate the run.

Measurements use nonsanitized optimized binaries, a quiet machine, and the same
compiler options and allocation/publication contract for every comparator.
CPU scaling, concurrent activity, and repetition spread remain part of the
result. Instrumented or smoke-run durations are not performance measurements.
The benchmark executable is not part of the installed libamdf distribution.

## XDNA execution through the HAL

The [XDNA HAL queue benchmark](../../runtime/src/iree/hal/driver/amd/xdna/README.md#queue-benchmarks)
measures ready dispatch, unresolved dependencies, streaming and executable
switching through the production HAL. It separates submit-only intervals from
checked completion. Its timings include HAL operation capture, dependency
handling and retirement as applicable; they are not isolated libamdf call costs.
The HAL guide owns the executable, invocation and exact timed boundaries.

Comparisons with XRT or another runtime match work per submission, residency,
host cache operations, dependency shape and completion boundaries. Allocation
comparisons also match whether each owner's GPU VM has executed work. Ratios
between unlike lifecycle or submission shapes do not isolate library overhead.
