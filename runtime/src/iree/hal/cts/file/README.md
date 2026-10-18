# Queued file benchmarks

`file_benchmark` measures reads through the public HAL API using the selected
device, a caller-owned proactor pool, and retained destination buffers. Each
in-flight read has its own semaphore: completion order is independent of
submission order. The file, pool, buffers, and semaphores survive repetitions.

The two rows measure different workloads:

- `BM_ReadBurst` submits `read_depth` independent reads and waits for all of
  them. Elapsed time covers the complete burst; dividing by `read_depth` gives
  amortized time per read, not individual read latency.
- `BM_ReadStream` reads the selected file prefix through a ring of `read_depth`
  buffers. A lane is refilled after its previous read completes. Elapsed time
  includes the final completion of every lane.

Device and pool construction, allocation, one warm-up pass, byte verification,
and teardown are outside timing. Verification downloads each lane's final read
through the selected queue and compares it with the source. It does not require
host-visible buffer placement. The source must remain immutable for the run.

## Running

Build with the canonical optimized configuration and run on an otherwise idle
device and storage path under the measurement environment's exclusion policy.
The default target smoke test uses Task and a declared source file; its timings
are not performance measurements.

```sh
build_tools/bin/iree-bazel-run --config=opt \
  //runtime/src/iree/hal/cts/file:file_benchmark -- \
  --device=rdna --file=/path/to/immutable.bin \
  --file_length=1073741824 --read_bytes=65536 --read_depth=8 \
  --registered_io=true --benchmark_filter=BM_ReadBurst \
  --benchmark_min_time=200x --benchmark_repetitions=5 \
  --benchmark_out=burst.json --benchmark_out_format=json
```

`read_bytes` is the maximum extent of one read. `buffer_bytes` can retain a
larger allocation per lane, allowing small reads to use the same backing
geometry as a bulk workload. Its default is `read_bytes`. `file_length=0` uses
the whole file; a stream's final read can be shorter than `read_bytes`.

For storage bandwidth, select `BM_ReadStream`, an aligned range, and
`--direct=true`. For example, 64 MiB reads at depth eight retain at least
512 MiB of destination storage. The native file's offset, length, and buffer
alignment requirements are enforced without rounding the requested range.
An unsupported direct mode fails instead of substituting buffered I/O.

`--registered_io=true` requires the pool to provide retained file-I/O bindings.
It can select different physical backing from the ordinary pool. Requested
registration failures are errors, not benchmark skips or an ordinary-I/O
fallback. The selected device and system must support that contract, including
any native device permissions and locked-memory allowance.

Buffered reads include the filesystem-cache path. The warm-up pass usually
warms that cache, but a file larger than available cache need not remain
resident. Storage-bandwidth claims require direct I/O or observed cold-source
residency, plus matching physical disk-read counters. Process-wide disk counters
also include untimed warm-up and byte verification; they are not automatically
the byte count of the timed region.

## Results

Google Benchmark reports real elapsed time and CPU time summed over the
process's threads. CPU time excludes external kernel workers. Byte and item
rates include all completed reads in the timed workload. Fixed iteration counts
keep small-operation runs bounded and make the actual request count explicit.

The additional counters describe the measured shape:

| Counter | Meaning |
| --- | --- |
| `read_bytes` | Maximum requested bytes per read |
| `read_depth` | Maximum number of independent reads in flight |
| `retained_bytes` | Sum of the HAL-reported destination allocation sizes |
| `host_allocations_per_read` | Successful allocation/reallocation calls through the supplied IREE host allocator, divided by completed reads |
| `direct` | Whether native uncached I/O was requested |
| `registered_io` | Whether retained file-I/O bindings were required |

The counted allocator adds atomic accounting to allocation and free calls. It
does not count C++ harness allocations, native driver allocations, or kernel
memory. A zero steady-state count proves only that the measured HAL path did
not call its supplied host allocator. Teardown joins the tracked resources'
asynchronous destruction before the allocator and verification storage expire.
