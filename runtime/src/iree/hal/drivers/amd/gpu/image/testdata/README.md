# HSACO image test inputs

`image.hip` produces the target-selected HIP code-object corpus used by image
tests, fuzzers, and benchmarks. It contains multiple kernels, initialized and
zero-filled globals, and a relocated global pointer. The build embeds the
ordinary linked artifacts without stripping away loader inputs.

## Fuzzing

Both targets use the repository's ASAN-enabled libFuzzer configuration:

```sh
iree-bazel-fuzz //runtime/src/iree/hal/drivers/amd/gpu/image:hsaco_fuzz -- \
  -max_len=65536 -runs=1000000
iree-bazel-fuzz //runtime/src/iree/hal/drivers/amd/gpu/image:hsaco_metadata_fuzz -- \
  -max_len=65536 -runs=1000000 \
  -dict=runtime/src/iree/hal/drivers/amd/gpu/image/hsaco_metadata_fuzz.dict
```

The wrapper retains corpora and failure inputs in the normal fuzz cache. Each
target periodically seeds ordinary byte mutation with a compiler artifact or
its MessagePack note. `max_len` must accommodate the linked corpus images.
Without a configured HIP compiler, raw-input fuzzing still works;
passing a populated corpus directory as another libFuzzer argument supplies
valid starting points.

`hsaco_fuzz` compares contiguous and fragmented admission, symbol lookup, and
host materialization. Writable segment storage is capped at 2 MiB per image;
larger virtual extents still exercise admission. No GPU instructions execute.
`hsaco_metadata_fuzz` exercises single notes, repeated notes, note pairs, and
rollback after a valid preceding note. The dictionary includes encoded metadata
keys, argument kinds, scalar encodings, and container boundaries.

## Benchmarks

```sh
iree-bazel-build --config=opt \
  //runtime/src/iree/hal/drivers/amd/gpu/image:hsaco_benchmark \
  //runtime/src/iree/hal/drivers/amd/gpu/image:hsaco_metadata_benchmark
```

`hsaco_benchmark` has separate admission and load rows for contiguous, 64-byte,
and 4096-byte source segments. Admission includes record allocation and
teardown. Loading uses an already-admitted image and preallocated writable
mappings, including copies, BSS initialization, and relocations.

`hsaco_metadata_benchmark` measures MessagePack admission and teardown after
extracting compiler notes outside timing. Batch rows repeat notes through the
multi-payload API to measure scaling independently of ELF processing. Labels
identify the compiler target; counters report source bytes and record counts.
An unavailable HIP corpus produces explicit skipped benchmark rows.

Run the built executables under the measurement host's exclusion mechanism,
with fixed iteration counts (for example `--benchmark_min_time=10000x`) and
multiple repetitions. The companion `_benchmark_test` targets are sanitizer
smoke tests; their timings are not performance measurements.
