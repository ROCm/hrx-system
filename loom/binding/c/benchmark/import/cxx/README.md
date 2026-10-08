# C++ import and compilation benchmarks

`source_to_module_benchmark` measures source text through a verified module and
release. Its C++ cases include preprocessing, source type checking, and import
through `loomc_module_import_cxx`. The no-use cases share one BF16 function
body while adding one facade or facade combination: `<stdfloat>`,
`<loomcxx/numeric.h>`, `<loomcxx/vector.h>`,
`<loomcxx/encoding_type.h>`, `<loomcxx/encoding.h>`,
`<loomcxx/view.h>`, `<loomcxx/predicate.h>`, `<loomcxx/kernel.h>`, or the kernel
and predicate facades together. Comparing them with `NoIncludes` isolates
header cost.
`RankTwoView` imports a dynamic-row, static-column view through a dense layout.
`RankThreeStorageView` composes a rank-three layout with an FP8 schema and loads
through the resulting physical-storage view. Together they retain the ordinary
view path as a control while tracking the higher-rank type and overflow-dimension
storage added for model-weight authoring.

`Q8S32Providers` imports the checked Q8S32 provider translation unit used by
the target specialization tests. Its user header is served through the public
source-provider callback, while the normal embedded facade supplies Loom
headers. This case measures a production-shaped library import with BF16
vectors, signed byte conversion, dot products, clustered reductions, semantic
predicates, and five template providers. The source handles, context, and
workspace are reused while every iteration preprocesses and parses the source
again; no parsed C++ state is cached. `Q8S32AuthoredLoomProviders` parses the
equivalent authored-Loom provider library through
`loomc_module_deserialize_from_source`. It is the control for separating the
one-time frontend cost from later specialization. These cases require no
target backend.

On an AMD Ryzen AI Max+ 395, 2026-10-07 optimized runs measured these median
wall times over seven repetitions of 50 source-to-module operations each:

| Source | Source text to verified module |
| --- | ---: |
| Tiny BF16 function, no includes | 0.317 ms |
| Tiny BF16 function and `predicate.h` | 0.562 ms |
| Tiny BF16 function and `kernel.h` | 0.977 ms |
| Tiny BF16 function and both facades | 1.235 ms |
| Q8S32 provider library, C++ | 2.363 ms |
| Q8S32 provider library, authored Loom | 0.0659 ms |

The benchmark lease was held, CPU scaling and ASLR were disabled, and the
canonical optimized configuration supplied optimization and ThinLTO. The Q8S32
C++ and authored-Loom repetition coefficients of variation were 2.1% and 0.4%,
respectively. The roughly 2.30 ms difference is the complete frontend premium
for this provider library; it is paid once before reusable loombc linking.

## Closed Q8S32 specialization

`q8s32_selection_benchmark` starts after either frontend has produced a
verified module. Setup serializes the common C++ root and the selected provider
library to bytecode, builds an immutable 12-symbol link index, and prepares a
`gfx1151` target profile. The timed region links and specializes
`@q8s32_specialize` with `model.q8s32.input_capacity=1280`. That capacity makes
the selector reject the four higher-priority target/value-specific providers
before accepting the target-independent eight-lane provider.

This measures the common production linker and template selector. Source
import, text parsing, bytecode serialization, index construction, and native
compilation stay outside the timed region. On the same machine, isolated
optimized runs measured these medians over seven repetitions of 5,000 closed
links each:

| Provider library | Closed selection | Repetition CV |
| --- | ---: | ---: |
| Imported C++ | 0.154 ms | 2.6% |
| Authored Loom | 0.153 ms | 2.3% |

The 0.7% median difference is inside run variation. Both paths allocate twelve
128 KiB workspace blocks on the first link, or 1.50 MiB, and allocate no new
workspace blocks during the warmed timed links. The imported C++ provider is
3,839 bytes of loombc and the authored control is 3,557 bytes. Their selected
modules are 4,427 and 4,679 bytes, respectively; the difference includes
durable source names and provenance rather than a different selected
implementation.

The two selected modules compile to byte-for-byte identical 9,184-byte HSACO
artifacts and identical assembly. Both contain 435 instructions and 2,588
bytes of machine code, use 20 SGPRs and 62 VGPRs, have no spills or local/private
memory, and report 100% occupancy with 16 resident subgroups per SIMD. A final
64-dispatch device-timestamp replay measured the same 1.844 us p50 and 2.364 us
p90 for both. The native artifact identity is the stronger result: authoring the
providers in C++ changes the one-time frontend cost, not target code or kernel
execution.

Build and run the selection comparison without enabling native emission:

```sh
iree-bazel-build --config=opt --config=loom-importer-cxx \
  --//loom/config/target:enable=amdgpu \
  //loom/binding/c/benchmark/import/cxx:q8s32_selection_benchmark

for name in SelectQ8S32CxxProviders SelectQ8S32AuthoredLoomProviders; do
  bazel-bin/loom/binding/c/benchmark/import/cxx/q8s32_selection_benchmark \
    --benchmark_filter="^${name}$" --benchmark_min_time=5000x \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true
done
```

## Source to HSACO

This Google Benchmark measures the public C++ importer through final HSACO
emission for the maintained attention, llama.cpp RMSNorm, and aiter FP16 SwiGLU
sources. It targets `gfx1151` without opening a GPU device. Numerical execution
coverage and source provenance live with the kernels under
`loom/src/loom/import/cxx/test/`.
Compilation permits approximate mathematical functions and supplies three
workgroups, matching that corpus's numerical configuration.

Each timed iteration preprocesses and type-checks source text, imports and
verifies High IR, specializes launch configs, compiles, emits an HSACO,
validates the artifact, and releases the module and results. The context,
compiler, prepared pipeline, target profile, config module, source handle, and
workspace are reused. Process startup and setup are outside timing. One warmup
compilation precedes measurement; parsed C++ ASTs are not cached.

Build an optimized executable before collecting numbers:

```sh
iree-bazel-build --config=opt --config=loom-importer-cxx \
  --//loom/config/target:enable=amdgpu --//loom/config/emit:enable=amdgpu \
  //loom/binding/c/benchmark/import/cxx:source_to_hsaco_benchmark

bazel-bin/loom/binding/c/benchmark/import/cxx/source_to_hsaco_benchmark \
  --benchmark_min_time=100x --benchmark_repetitions=7 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=source-to-hsaco.json --benchmark_out_format=json
```

The shared `opt` configuration owns optimization and ThinLTO settings. An
ISA-specific comparison names its intended measurement CPU explicitly:
`-march=native` in a remote build describes the build worker. Run matched
executables under the measurement runner's exclusion policy, after builds and
artifact transfer have finished.

The benchmark uses the normal embedded facade headers, so its build requires
`embed_includes`. Artifact byte counts and workspace allocation counters are
reported alongside latency. Workspace counters cover Loom arena growth, not
the C++ parser's allocations. ASAN smoke runs validate correctness; those
timings are not performance measurements.

On an AMD Ryzen AI Max+ 395, a 2026-09-18 optimized run with jemalloc measured
these median wall times over seven repetitions of 100 compilations each:

| Source | Source text to HSACO | HSACO size |
| --- | ---: | ---: |
| Flash attention | 5.29 ms | 9,184 bytes |
| llama.cpp RMSNorm | 3.88 ms | 9,176 bytes |
| aiter FP16 SwiGLU | 3.21 ms | 9,184 bytes |

The machine benchmark lease was held, CPU scaling and ASLR were disabled, and
the build processes had finished. Repetition coefficients of variation were
below 0.4%. The optimized `jit_amdgpu` C example was 15,042,192 bytes
(14.35 MiB) after stripping, including the importer, compiler, embedded
facades, and runtime execution path. The stripped example passed both kernels'
output and guard checks. It links the normal C/C++ system libraries and loads
ROCr for execution; it has no LLVM runtime dependency.
