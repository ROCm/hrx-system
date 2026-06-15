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
| `loomc` public C API | Active embedding surface for AOT, JIT, packaging, executable caches, and tuning/search |
| AMDGPU HSACO | Most established native target path |
| SPIR-V/Vulkan | Working examples and tests; still hardening as a product target |
| x86, Wasm, IREEVM | Real early target paths with providers, lowering/check coverage, and initial emission infrastructure; not mature product targets |
| Whole model programs | Direction of travel; current checked examples are kernel-focused |

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
  --dry-run
```

The Bazel HAL driver registry defaults to host-only drivers. On an
AMDGPU-capable build and machine, include the AMDGPU driver in the Bazel
configuration and run the smallest correctness and timing smoke through the
same source:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null \
  //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom -- \
  loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom \
  --device=amdgpu \
  --measure=dispatch_complete \
  --sample-compilation=per_sample \
  --iterations=1 \
  --warmup-iterations=0 \
  --batch-size=1 \
  --min-time-ms=0 \
  --max-batches=1 \
  --input-ring-count=1
```

The authoring pattern to notice is that correctness policy and benchmark rows
live beside the source. `func.apply` requests an implementation contract,
`func.template` providers satisfy those contracts, and per-case parameters can
become compile-time facts before lowering.

The authoring README includes the direct quantized AMDGPU flow for
`loom-compile` HSACO emission, artifact manifests, compile reports, IR dumps,
target listings, and correctness-gated dispatch benchmark bundles.

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

The API is staged instead of one file-oriented entry point because embedders
need different compositions:

- AOT packaging can compile and emit ahead of deployment.
- Runtime JITs can link and specialize around live target facts.
- Executable caches can own artifact storage and invalidation policy.
- Tuning servers can reuse compilers, linkers, pass programs, target profiles,
  and frozen indexes across many worker-local workspaces.

## Try AMDGPU/HSA Emission

The AMDGPU examples show the target-profile and HSACO emission side of the C
API.

Offline synthetic AMDGPU processor profile:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_offline -- \
  gfx1100 \
  /tmp/targetless_store_i32.hsaco
```

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
  --//runtime/config/hal:drivers=vulkan,local-sync,local-task,null \
  //loom/binding/c/example:emit_spirv_vulkan
```

IREE HAL-derived SPIR-V target facts and executable-cache validation:

```bash
python dev.py bazel run \
  --//runtime/config/hal:drivers=vulkan,local-sync,local-task,null \
  //loom/binding/c/example:emit_spirv_iree_hal
```

The raw Vulkan path is useful when evaluating Loom as an embeddable compiler
near an application's own shader/module loading boundary. The IREE HAL path is
useful when evaluating Loom as a companion to IREE runtime executable caches.

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
