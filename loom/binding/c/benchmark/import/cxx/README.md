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
emission for maintained attention, normalization, activation, storage-encoding,
and packed-quantized projection sources. Cases select `gfx1151` or `gfx1250`
without opening a GPU device. Numerical execution coverage and source
provenance live with each kernel's optional CXX test package.
Compilation permits approximate mathematical functions and supplies three
workgroups, matching that corpus's numerical configuration.

Each timed iteration preprocesses and type-checks source text, imports and
verifies High IR, specializes launch configs, compiles, emits an HSACO,
validates the artifact, and releases the module and results. The context,
compiler, prepared pipeline, target profile, config module, source handle, and
workspace are reused. Process startup and setup are outside timing. One warmup
compilation precedes measurement; parsed C++ ASTs are not cached.

The `CxxJitPhase` rows attribute this endpoint with maintained sources. llama.cpp
RMSNorm provides a small ordinary kernel; the MXFP8 group dot exercises storage
encodings, narrow-float conversion, vectors, reductions, and target-specific
lowering; the routed IQ4_XS projection combines original-format nonlinear
decode, dynamic schedules, workgroup staging, paired floating-point dots, and a
fused SwiGLU epilogue; and the routed Q4_K/Q8_1 SwiGLU combines packed records,
configuration, template application, integer dots, subgroup reductions, and
nontrivial address arithmetic. Each row invokes a complete public compiler
boundary:

| Phase | Timed operation |
| --- | --- |
| `Import` | Preprocess, parse, type-check, and import source to verified High IR. |
| `SourceToPreparedLow` | Import source and compile that module through prepared Low IR in one workspace. |
| `CloneHigh` | Clone a setup-imported High module into the invocation workspace. |
| `SourceLow` | Clone High IR and run the source-low target pipeline. |
| `PreparedLow` | Clone High IR and run the complete prepared-low target pipeline. |
| `HighToHsaco` | Clone High IR, compile it through prepared Low IR, emit HSACO, and validate the ELF artifact. |
| `ClonePreparedLow` | Clone setup-prepared Low IR with its retained specialization facts. |
| `EmitPreparedLow` | Clone prepared Low IR, emit HSACO, and validate the ELF artifact. |

The lowering and emission rows include their required fresh-module clone. The
clone-only rows expose that floor. `SourceToPreparedLow` and `HighToHsaco`
preserve each cumulative boundary in one workspace. `SourceLow`, `PreparedLow`,
and `EmitPreparedLow` isolate their inputs with retained templates, so those
phase times are not additive reconstructions of `SourceToHsaco`.
Validation, result construction, teardown, and arena reuse remain owned by the
public operation that performs them.

The routed Q4_K/Q8_1 source supplies input size 4096 through an ordinary
`config.def` and compiles its complete 768-channel kernel for `gfx1250`.
The IQ4_XS source fixes its production 2,560-element by 640-channel top-10
geometry in C++ and supplies its transport, staging depth, and scheduling
values through ordinary `config.def` operations. The benchmark retains the
single-buffered synchronous `gfx1151` case and compares single- and
double-buffered asynchronous `gfx942` specializations.
`Import`, `SourceToPreparedLow`, and `SourceToHsaco` form cumulative boundaries
whose differences provide an additive production-path breakdown. Clone-based
phase probes remain diagnostic controls and are not additive.

`ConfiguredWorkgroupStorage` measures a specialization-first kernel whose C++
source declares a constrained stage count, uses that value to size aligned
workgroup storage and a dynamic view, and carries the value through a loop. The
benchmark supplies four stages through a `config.def`; it does not rewrite or
reparse the source. A short optimized run on an AMD Ryzen AI Max+ 395 on
2026-10-08 measured these medians over seven 100-iteration repetitions:

| Public boundary | Configured workgroup storage |
| --- | ---: |
| Import to verified High IR | 3.55 ms |
| Clone High IR | 12.9 us |
| Through source-low | 0.362 ms |
| Through prepared-low | 0.397 ms |
| Clone prepared Low IR | 9.24 us |
| Emit prepared Low IR to HSACO | 84.1 us |
| Complete source to HSACO | 4.17 ms |

The benchmark lease was held and the complete row had a 5.2% repetition
coefficient of variation. Seven one-iteration cold-workspace repetitions had a
4.26 ms median and allocated eleven 128 KiB workspace blocks, or 1.375 MiB.
Warmed iterations reused those blocks. The emitted HSACO is 9,208 bytes.

The same imported module was also compiled with stage counts one and four and
compared with static High IR controls in which only the stage count was an
ordinary constant. Each specialization produced byte-for-byte identical HSACO
to its static control, including these resource and code properties:

| Stage count | Exact LDS | Instructions | Code bytes | SGPRs | VGPRs | Occupancy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 bytes | 19 | 100 | 4 | 3 | 100% |
| 4 | 512 bytes | 30 | 148 | 4 | 5 | 100% |

Configuration therefore changes the storage plan and specialized loop body
without leaving configuration machinery or a runtime-sized allocation in the
native artifact.

An optimized 2026-10-08 run on an AMD Ryzen AI Max+ 395 measured these warmed
medians. Phase rows used seven 100-iteration repetitions and complete rows used
fifteen 50-iteration repetitions under the benchmark lease with no process
allocator override.

| Public boundary | llama.cpp RMSNorm | MXFP8 group dot |
| --- | ---: | ---: |
| Import to verified High IR | 3.221 ms | 4.101 ms |
| Clone High IR | 22.7 us | 8.42 us |
| Through source-low | 1.439 ms | 0.233 ms |
| Through prepared-low | 1.614 ms | 0.291 ms |
| Clone prepared Low IR | 33.6 us | 33.4 us |
| Emit prepared Low IR to HSACO | 0.807 ms | 0.689 ms |
| Complete source to HSACO | 6.443 ms | 5.509 ms |

`SourceToHsacoColdWorkspace` skips scenario warmup and forces one compilation
per repetition. Context, compiler, target, pass-program, source, and config
setup remain outside timing, while the invocation workspace starts unused. Its
allocation counters therefore report the first compilation's arena growth
instead of mixing first-growth and reuse in one sample.
All fifteen one-iteration repetitions reported twelve 128 KiB blocks for
RMSNorm and eight blocks for MXFP8. Cold-workspace latency varied by 8.0% and
6.3%, respectively, so it is compared as a repeated distribution rather than a
single-sample threshold.

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
