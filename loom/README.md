# Loom

> [!NOTE]
> This README is a first-merge quickstart. It is intentionally narrow: get a
> motivated human or agent from checkout to the current examples, then point at
> the code and docs that explain the real surfaces. Deeper architecture,
> performance, target, and model-program docs are still being split out.

Loom is a source-first native compiler substrate for asynchronous
device-oriented programs. The current proof point is kernel authoring,
specialization, compilation, emission, testing, benchmarking, and tuning. The
larger unit is a model program: the same architecture is meant to scale upward
from individual kernels into model execution, serving, eval, adaptation,
LoRA/fine-tuning, and training loops where the scale makes sense.

The core idea is simple: keep source facts, target facts, linkable providers,
specialization choices, executable artifacts, correctness policy, and
performance evidence in one system. Loom is not another router over HIP, BLAS,
Triton, CK, AITER, and framework dispatch. It is the compiler layer where those
facts are still visible enough to compile, inspect, specialize, and tune.

## Current Status

| Area | Status |
| --- | --- |
| Loom text, bytecode, IR, passes, linking, reports | Active core infrastructure |
| `loomc` public C API | Active embedding surface for AOT, JIT, packaging, caller-owned artifact caches, and tuning/search |
| AMDGPU HSACO | Most established native target path |
| SPIR-V/Vulkan | Working examples and tests; still hardening as a product target |
| x86, Wasm | Real early target paths with providers, lowering/check coverage, and initial emission infrastructure; not mature product targets |
| Portable command programs | Source dialect, multi-root preparation, target-neutral Low ISA, and immutable artifact core; public embedding and materialization are staged separately |
| Whole host programs | Direction of travel beyond reusable command-buffer-shaped subgraphs |

## Build The First Slice

Run commands from the repository root.

Set up Bazel tools once:

```bash
python dev.py bazel setup
```

Build the core Loom tools and the smallest C API examples:

```bash
python dev.py bazel build \
  //loom/src/loom/tools/loom-check:loom-check \
  //loom/src/loom/tools/loom-compile:loom-compile \
  //loom/src/loom/tools/loom-link:loom-link \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom \
  //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom \
  //loom/py/loom/tools:loom-compile-report \
  //loom/binding/c:loomc \
  //loom/binding/c/example:source_info \
  //loom/binding/c/example:compile_text \
  //loom/binding/c/example:link_modules
```

That gives you the command-line tools, the benchmark/test runner used by the
authoring corpus, and the C ABI examples that show in-memory source, compile,
link, result, diagnostic, and artifact flow.

## Try The Authoring Corpus

The best starting point for Loom source is:

- [authoring README](src/loom/test/corpus/authoring/README.md)
- [q6/q8 gate-up SwiGLU example](src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom)
- [MLP down-projection residual example](src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom)

Run the host-only dry-run tests. These parse the source and prove
`check.case`/`check.benchmark` planning without requiring a local GPU:

```bash
python dev.py bazel test \
  //loom/src/loom/test/corpus/authoring:ffn_gate_up_swiglu_q6q8_plan_test \
  //loom/src/loom/test/corpus/authoring:mlp_down_projection_residual_bf16_plan_test
```

You can also run the benchmark planner directly:

```bash
python dev.py bazel run //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom -- \
  loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom \
  --dry-run \
  --config=mlp_down_projection_residual_bf16.row_capacity=3584
```

The Bazel HAL driver registry defaults to host-only drivers. On an
AMDGPU-capable build and machine, include the AMDGPU driver in the Bazel
configuration and run the smallest correctness and timing smoke through the
same source:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=amdgpu,task \
  //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom -- \
  loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom \
  --config=mlp_down_projection_residual_bf16.row_capacity=3584 \
  --device=amdgpu \
  --measure=dispatch_complete \
  --iterations=1 \
  --warmup-iterations=0 \
  --batch-size=1 \
  --min-time-ms=0 \
  --max-batches=1 \
  --input-ring-count=1
```

The authoring pattern to notice is that correctness policy and benchmark rows
live beside the source. `template.apply` requests an implementation from a
declared family, `template.def` operations provide the candidates, and explicit
config bindings select compile-time choices while case parameters remain
runtime values.

The authoring README includes the direct quantized AMDGPU flow for
`loom-compile` HSACO emission, artifact manifests, compile reports, IR dumps,
target listings, and correctness-gated dispatch benchmark bundles.

## Inspect Compile Reports

`loom-compile-report` provides a bounded first view over the structured reports
emitted by `loom-compile` and `iree-benchmark-loom`:

```bash
python dev.py bazel run //loom/py/loom/tools:loom-compile-report -- \
  show /tmp/kernel.compile-report.json
python dev.py bazel run //loom/py/loom/tools:loom-compile-report -- \
  diff /tmp/baseline.json /tmp/candidate.json --format=json
python dev.py bazel run //loom/py/loom/tools:loom-compile-report -- \
  suggest /tmp/candidate.json --format=json
python dev.py bazel run //loom/py/loom/tools:loom-compile-report -- \
  suggest /tmp/candidate.json --include-experimental --format=json
```

`show` separates emitted artifact facts from compiler analysis and omits
unavailable metrics from its compact JSON. `diff` rejects reports unless their
schema, target, specialization, workload, and entry identities match exactly.
`suggest` delegates interpretation to the selected target family and cites the
evidence behind each proposed experiment. Default findings require a documented
or silicon-calibrated target model. `--include-experimental` also exposes
structurally exact findings from hardware-unvalidated models and labels them
`experimental`, allowing pre-silicon search without presenting model
predictions as measured hardware behavior.

Final-native facts expose the scheduled body separately from the target-owned
entry envelope, native coissued instructions separately from their semantic
components, and each matrix family separately from the broad matrix total.
Dispatch-scaled operation counts appear only when the report carries an exact
workload scale. Wait analysis distinguishes waits already present in the low
stream from waits inserted by target planning, while target-insertion coverage
states whether dynamic packet counts are exact or unknown. These are structural
search signals rather than cycle estimates; absent values stay unavailable
instead of being inferred by the report tool.

Compile reports are version-zero, same-compiler-horizon diagnostics. Regenerate
them with the current checkout instead of treating them as durable records or
adding compatibility paths to the consumer.

## Try The C API

The public C ABI lives under [binding/c](binding/c). The main docs are:

- [C API main page](binding/c/doc/mainpage.md)
- [C API style guide](binding/c/doc/STYLE_GUIDE.md)
- [public headers](binding/c/include/loomc)
- [examples](binding/c/example)

Run the host-only examples:

```bash
python dev.py bazel run //loom/binding/c/example:source_info
python dev.py bazel run //loom/binding/c/example:compile_text
python dev.py bazel run //loom/binding/c/example:link_modules
```

These examples are deliberately small, but they show the intended embedding
shape:

```text
source -> module -> link/index -> compile -> result diagnostics/artifacts
```

The [in-memory composition guide](docs/src/integration/module-composition.md)
uses `link_modules` to show a caller-owned root input resolving declarations
against explicitly supplied libraries.

The API is staged instead of one file-oriented entry point because embedders
need different compositions:

- AOT packaging can compile and emit ahead of deployment.
- Runtime JITs can link and specialize around live target facts.
- Caller-owned artifact caches can define storage and invalidation policy.
- Tuning servers can reuse compilers, linkers, pass programs, target profiles,
  and frozen indexes across many worker-local workspaces.

## Try AMDGPU/HSA Emission

The AMDGPU examples show the target-profile and HSACO emission side of the C
API.

Offline synthetic AMDGPU processor profile compiling packaged Loom bytecode:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_offline -- \
  gfx11-generic \
  /tmp/targetless_store_i32.hsaco
```

The generic target emits a portable GFX11 code object. Pass `gfx1151` instead
when the artifact should be specialized for that exact processor. The example
build packages `targetless_store_i32.loom` as bytecode and invokes LoomC's
bytecode-specific deserializer, matching a kernel library that ships
precompiled Loom modules instead of source text.

Raw HSA probing, HSACO emission, code-object loading, and one kernel dispatch
without the IREE HAL:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa
```

Successful launch ends with `launched targetless_store_i32 via raw HSA:
output=42`.

The same raw-HSA example can start from a checked `.loom` or prelinked
`.loombc` file when the source defines the one-buffer `targetless_store_i32`
sample ABI:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa -- \
  /tmp/targetless_store_i32.loombc \
  @targetless_store_i32 \
  targetless_store_i32 \
  targetless_store_i32
```

Set `LOOMC_HSA_RUNTIME_PATH` when the HSA runtime is not discoverable through
the default dynamic loader search path. The value may be the exact
`libhsa-runtime64.so` path or a directory containing it:

```bash
LOOMC_HSA_RUNTIME_PATH=/opt/rocm/lib \
  python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa
```

The raw HSA path is useful when evaluating Loom as an embeddable compiler near
an application's own HSA/HRX/HIP loading boundary.

## Try SPIR-V/Vulkan Emission

The SPIR-V examples show the target-profile and emission side of the C API.

Offline synthetic SPIR-V profile:

```bash
python dev.py bazel run //loom/binding/c/example:emit_spirv_offline
```

Raw Vulkan probing and emission without the IREE HAL:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=vulkan,task \
  //loom/binding/c/example:emit_spirv_vulkan
```

IREE HAL-derived SPIR-V target facts and emission:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=vulkan,task \
  //loom/binding/c/example:emit_spirv_iree_hal
```

The raw Vulkan path is useful when evaluating Loom as an embeddable compiler
near an application's own shader/module loading boundary. The IREE HAL path is
useful when evaluating Loom as a compiler for target-explicit IREE HAL loading.

## Useful Entry Points

| Path | Why it matters |
| --- | --- |
| [binding/c/include/loomc](binding/c/include/loomc) | Public C ABI contracts, ownership, threading, diagnostics, targets, linking, compile, and emit |
| [binding/c/example](binding/c/example) | Minimal embedders for source info, compile, link, AMDGPU offline, raw HSA, SPIR-V offline, raw Vulkan, and IREE HAL |
| [loom/py/loom/dialect](/loom/py/loom/dialect) | Python op/dialect authoring DSL, assembly formats, and source-format migration breadcrumbs |
| [loom/py/loom/migration](/loom/py/loom/migration) | Source migration driver, rule generation, baselines, compatibility windows, and `loom-migrate` |
| [src/loom/test/corpus/authoring](src/loom/test/corpus/authoring) | Canonical hand-authored Loom examples for model/kernel-shaped source |
| [src/loom/tools/iree-benchmark-loom](src/loom/tools/iree-benchmark-loom) | Runner for `check.case`, `check.benchmark`, correctness samples, timing, and reports |
| [src/loom/tools/loom-check](src/loom/tools/loom-check) | Text/test corpus checking and `--update` workflow |
| [src/loom/target](src/loom/target) | Target providers, target-low infrastructure, artifact plans, and compile reports |
| [src/loom/target/arch/amdgpu](src/loom/target/arch/amdgpu) | AMDGPU source-to-low policy and target contracts |
| [src/loom/target/emit/native/amdgpu](src/loom/target/emit/native/amdgpu) | Native AMDGPU/HSACO emission path |
| [src/loom/target/arch/spirv](src/loom/target/arch/spirv) | SPIR-V target facts, profiles, and cooperative-matrix capability handling |
| [src/loom/target/arch/cmd](src/loom/target/arch/cmd/README.md) | Reusable command-program source, preparation, portable Low ISA, and artifact contracts |

## Mental Model

For AOT use:

```text
Loom source/bytecode + libraries + target/config
  -> link
  -> specialize
  -> compile
  -> emit artifact
  -> package
```

For JIT use:

```text
runtime loads source/bytecode
  -> reads live or saved target facts
  -> links the needed root
  -> specializes config
  -> emits a loadable artifact
  -> hands it to the runtime or native loader
```

For tuning/search:

```text
prepared compiler/linker/pass/target handles
  + frozen library indexes
  + worker-local workspaces
  + per-job config and target selections
  -> compile/report/benchmark loop
```

That is the current center of gravity: a tiny compiler that can be linked into
the place where executable artifacts are selected, loaded, cached, measured,
and tuned.
