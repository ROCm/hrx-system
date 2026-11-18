# C++ import and compilation benchmarks

`source_to_module_benchmark` measures source text through a verified module and
release. Its C++ cases include preprocessing, source type checking, and import
through `loomc_module_import_cxx`. The no-use cases share one BF16 function
body while adding one facade or facade combination: `<stdfloat>`,
`<loomcxx/numeric.h>`, `<loomcxx/vector.h>`,
`<loomcxx/encoding_type.h>`, `<loomcxx/encoding.h>`,
`<loomcxx/view.h>`, `<loomcxx/atomic.h>`, `<loomcxx/predicate.h>`,
`<loomcxx/pipeline.h>`, `<loomcxx/kernel.h>`, or the kernel with the pipeline
or predicate facade. Comparing them with `NoIncludes` isolates header cost;
comparing `KernelPipeline` with `Kernel` isolates the pipeline declarations
after their shared view and buffer dependencies are already parsed.
`RankTwoView` imports a dynamic-row, static-column view through a dense layout.
`RankThreeStorageView` composes a rank-three layout with an FP8 schema and loads
through the resulting physical-storage view. Together they retain the ordinary
view path as a control while tracking the higher-rank type and overflow-dimension
storage added for model-weight authoring.

`UnusedFacadeDeclarations` varies the number of attributed operation templates
visible to the same view consumer. Each template declares the buffer length
operation under a distinct name. The BF16 function is the only emitted body at
every size, so the difference measures frontend declaration cost.

`Q8S32Providers` imports the benchmark-owned Q8S32 provider source. Its user
header is served through the public source-provider callback, while the normal
embedded facade supplies Loom headers. This case measures a production-shaped
library import with BF16 vectors, signed byte conversion, dot products,
clustered reductions, semantic predicates, and five template providers. The
source handles, context, and workspace are reused while every iteration
preprocesses and parses the source again; no parsed C++ state is cached.
`Q8S32AuthoredLoomProviders` parses the equivalent authored-Loom provider with
`loomc_module_deserialize_from_source`. It is the control for separating the
one-time frontend cost from later specialization. These cases require no
target backend.

`PipelineDefinition` imports a complete configured pipeline with two placed
strands, pooled workgroup storage, FIFO channel ownership, asynchronous
transport, and BF16 vector math. Its cold, preamble-creation, and
preamble-restored rows keep the complete pipeline frontend cost visible without
coupling the timed bytes to the independently maintained target-specific
documentation example.

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
compilation stay outside the timed region.

Build and run the selection comparison with AMD GPU compiler support. The
benchmark does not open a GPU or invoke native emission in its timed path:

```sh
iree-bazel-build --config=opt --config=loom-importer-cxx \
  --config=loom-target-amd-gpu \
  //loom/binding/c/benchmark/import/cxx:q8s32_selection_benchmark

for name in SelectQ8S32CxxProviders SelectQ8S32AuthoredLoomProviders; do
  bazel-bin/loom/binding/c/benchmark/import/cxx/q8s32_selection_benchmark \
    --benchmark_filter="^${name}$" --benchmark_min_time=5000x \
    --benchmark_repetitions=7 --benchmark_report_aggregates_only=true
done
```

## Source to HSACO

This Google Benchmark measures the public C++ importer through final HSACO
emission for maintained attention, collective, storage-encoding, and
packed-quantized projection sources. Cases select `gfx1151` or `gfx1250`
without opening a GPU device. The timed translation units are private,
kernel-only benchmark inputs. Numerical execution coverage and source
provenance live in the C++ authoring corpus. Most source cases select the fast
translation-unit floating-point model. Cases that carry their numerical policy
in source can retain the precise model, so only their explicit approximate and
contractable operations reach High IR. The benchmark supplies three
workgroups, matching the corpus's numerical configuration.

Each timed iteration preprocesses and type-checks source text, imports and
verifies High IR, specializes launch configs, compiles, emits an HSACO,
validates the artifact, and releases the module and results. The context,
compiler, prepared pipeline, target profile, config module, source handle, and
workspace are reused. Process startup and setup are outside timing. One warmup
compilation precedes measurement; parsed C++ ASTs are not cached.

The `CxxJitPhase` rows attribute this endpoint with three representative source
shapes. Flash attention provides tiled shared-memory traffic, subgroup
reductions, and partial bounds. Paged online softmax exercises nested carried
loops, guarded addressing, vector dot and accumulation, and source-local
floating-point permissions. The MXFP8 group dot exercises storage encodings,
narrow-float conversion, vectors, reductions, and target-specific lowering.
Additional maintained kernels retain end-to-end
`SourceToHsaco` rows without multiplying frontend and prepared-module setup
across every cumulative phase. Each phase row invokes a complete public
compiler boundary:

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

The end-to-end rows retain larger and target-specialized compilation shapes.
The routed Q4_K/Q8_1 source supplies input size 4096 through an ordinary
`config.def` and compiles its complete 768-channel kernel for `gfx1250`. The
IQ4_XS source fixes its production 2,560-element by 640-channel top-10 geometry
in C++ and supplies its transport, staging depth, and scheduling values through
ordinary `config.def` operations. Its rows retain the single-buffered
synchronous `gfx1151` case and compare single- and double-buffered asynchronous
`gfx942` specializations. Configured workgroup storage and the multi-file NVFP4
provider source retain specialization and include-provider coverage at the same
complete endpoint.

`ConfiguredWorkgroupStorage` measures a specialization-first kernel whose C++
source declares a constrained stage count, uses that value to size aligned
workgroup storage and a dynamic view, and carries the value through a loop. The
benchmark supplies four stages through a `config.def`; it does not rewrite or
reparse the source.

`SourceToHsacoColdWorkspace` skips scenario warmup and forces one compilation
per repetition. Context, compiler, target, pass-program, source, and config
setup remain outside timing, while the invocation workspace starts unused. Its
allocation counters therefore report the first compilation's arena growth
instead of mixing first-growth and reuse in one sample.

Build an optimized executable before collecting numbers:

```sh
iree-bazel-build --config=opt --config=loom-importer-cxx \
  --config=loom-target-amd-gpu \
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
