# Loom

Loom is a language and embeddable compiler for functions, kernels, command
programs, and pipelines. Its microsecond-scale JIT, execution libraries, and
multi-model inference server fit in a megabyte-scale SDK, with no LLVM or ROCm
dependency.

Programs can be written in a subset of C++ or directly in Loom IR, linked with
reusable libraries, and specialized for the target and workload. The SDK includes
tools for testing correctness and benchmarking the resulting programs.

> [!TIP]
> **To run models or troubleshoot a model, start at [loom-dev](https://loom-fun.github.io/loom-dev/).**
> [Loom skills](https://github.com/loom-fun/loom-skills) help your agent set up,
> run, and adapt models; the [Loom docs](https://rocm.github.io/hrx-system/loom/)
> cover installation, serving, and authoring.
>
> `hrx-system` is the engineering repository for Loom's compiler, runtime, and tools.

The [repository overview](../README.md) maps the compiler, services, and execution
libraries across the monorepo.

## Choose an entry point

| You want to… | Start here |
| --- | --- |
| Learn the language | [One kernel in IR and C++](https://rocm.github.io/hrx-system/loom/getting-started/first-kernel/), then the [programming guide](https://rocm.github.io/hrx-system/loom/guide/). |
| Use the installed SDK | [Installation](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/) and [tool workflows](https://rocm.github.io/hrx-system/loom/workflows/). The `hrx-loom` package includes native tools, C++ import, and Python APIs. |
| Embed the compiler | The [`loomc` C API](binding/c/doc/mainpage.md), [public headers](binding/c/include/loomc/), and [small examples](binding/c/example/). |
| Emit code for your own runtime | [Compile artifacts](https://rocm.github.io/hrx-system/loom/workflows/compile-artifacts/), with [raw HSA](binding/c/example/emit_amdgpu_hsa.c) and [raw Vulkan](binding/c/example/emit_spirv_vulkan.c) integration examples. |
| Host or author models | [Fetch and serve](https://rocm.github.io/hrx-system/loom/serving/workflows/fetch-and-serve/), [model programs](https://rocm.github.io/hrx-system/loom/serving/model-programs/), and the [`loom-serve` source guide](src/loom/tools/loom-serve/README.md). |

Loom is a tech preview. The [target guide](https://rocm.github.io/hrx-system/loom/targets/)
tracks compilation, execution, correctness coverage, and performance maturity
separately for CPU, AMD GPU, XDNA, SPIR-V, and other targets.

## One program through the stack

Loom text (`.loom`) and bytecode (`.loombc`) represent the same IR. The C++
frontend imports into that IR, and the compiler C API can construct and compose
modules in memory. Functions and templates express reusable computation;
kernels add launch contracts; command programs and pipelines compose execution
and resource relationships.

```text
C++ / Loom text / bytecode / in-memory IR
  → link modules and select providers
  → specialize using configuration, shapes, and target facts
  → lower, schedule, allocate, and emit
  → executable artifacts and structured reports
```

A template family can have several implementations with explicit applicability
constraints. Linking and specialization select the implementation using the
facts available at that point. Shared code can remain portable while providers
use target-specific operations where they matter. The
[facts and specialization guide](https://rocm.github.io/hrx-system/loom/guide/facts-and-specialization/)
explains the authoring contract, and the
[source-to-artifacts walkthrough](https://rocm.github.io/hrx-system/loom/getting-started/source-to-artifacts/)
shows it in a complete program.

The compiler can emit ELF, HSACO, SPIR-V, WebAssembly, and command-program
artifacts. AOT compilation, model-load JIT, and invocation specialization use the
same modules and compiler. The application chooses when facts become available,
which roots to compile, and how to retain and cache the result. See
[Integrate Loom](https://rocm.github.io/hrx-system/loom/integration/) for the
available boundaries and their ownership model.

## Author, check, and tune

Correctness cases and benchmark workloads can live beside the program they
exercise. The tools carry those workloads through specialization and execution,
and produce structured diagnostics and reports for people and agents.

1. [Format and verify](https://rocm.github.io/hrx-system/loom/workflows/format-and-verify/)
   the source with `loom-lint` and `loom-format`.
2. [Check correctness](https://rocm.github.io/hrx-system/loom/workflows/test-correctness/)
   with `loom-test`, using the program's declared cases and numerical acceptance.
3. [Benchmark](https://rocm.github.io/hrx-system/loom/workflows/benchmark/)
   the proven workload with `loom-benchmark` and a defined measurement boundary.
4. [Inspect compile reports](https://rocm.github.io/hrx-system/loom/workflows/compile-reports/)
   and emitted code, then compare a candidate against the same correctness and
   performance evidence.

The [documentation examples](docs/examples/) are executable sources for the
published guides. The [authoring corpus](src/loom/test/corpus/authoring/README.md)
contains larger examples, including quantized model operations and reusable
providers. The [agent authoring workflow](https://rocm.github.io/hrx-system/loom/workflows/agent-driven-kernel-development/)
connects reference implementations, data reuse, numerical checks, and measured
performance. Reusable models and kernels live in the
[Loom library](https://github.com/ROCm/hrx-loom-library).

## Model execution and serving

A model package supplies its programs, weights, and descriptor as runtime input.
The execution layer prepares and specializes the program, manages devices and
storage, and retains work through completion. Modality services add text, image,
embedding, decision, recognition, and speech behavior. `loom-serve` composes
those services with HTTP, registration, and operational controls.

The [serving component map](src/loom/serve/README.md) traces the native owners and
points to their headers and tests. The shared [Loom runtime](src/loom/runtime/)
and [service components](src/loom/service/) are being factored for reuse outside
the HTTP server; their native interfaces are still evolving. The separately
documented `loomc` API is the public compiler embedding surface.

The [live model development workflow](https://rocm.github.io/hrx-system/loom/serving/workflows/live-model-development/)
shows how to upload implementations, compare variants using retained weights,
and change the selected execution without restarting the server.

## Source map

| Path | Responsibility |
| --- | --- |
| [src/loom](src/loom/) | Shipping C compiler: IR, verification, linking, analysis, transformations, and target lowering. |
| [py/loom/dialect](py/loom/dialect/README.md) | Source-of-truth dialect and operation declarations, syntax, and generated contracts. |
| [src/loom/import/cxx](src/loom/import/cxx/README.md) | C/C++ frontend, supported source constructs, and IR import/export. |
| [src/loom/target](src/loom/target/) | Target capabilities, lowering, scheduling, artifact emission, and compile reports. |
| [py/loom/target](py/loom/target/) | Target data, descriptors, generators, and validation. |
| [binding/c](binding/c/doc/mainpage.md) | Public compiler C API, target adapters, and embedding examples. |
| [src/loom/runtime](src/loom/runtime/) | Shared host/device ownership, program preparation, storage, parameters, and telemetry over the [system runtime](../runtime/README.md). |
| [src/loom/service](src/loom/service/) and [src/loom/serve](src/loom/serve/README.md) | Model service components, modality implementations, transports, and server composition. |
| [src/loom/tools](src/loom/tools/) and [py/loom/tools](py/loom/tools/) | Native and Python command-line tools. |
| [src/loom/test](src/loom/test/) | Semantic corpus and integration coverage; component unit tests live beside their implementations. |
| [py/loom/model](py/loom/model/README.md) | Model documents, acquisition, packaging, and publication. |
| [packaging/python](packaging/python/README.md) | `hrx-loom` SDK assembly, platform wheels, and release workflow. |
| [docs](docs/) | Documentation source and executable examples. |

## Develop from source

[Building](../BUILDING.md) owns prerequisites, compiler and driver selection,
Bazel/CMake configuration, and installation. [Contributing](../CONTRIBUTING.md)
owns the change and review workflow. Run source-development commands from the
repository root. For example, this builds and runs a small compiler embedder:

```bash
python dev.py bazel setup
python dev.py bazel run //loom/binding/c/example:compile_text
```

The [C API examples](binding/c/example/) cover in-memory compilation, linking,
offline emission, and runtime handoff. The
[`loom-check` guide](src/loom/tools/loom-check/README.md) covers compiler fixtures,
exact expected output, and updates. The
[C++ authoring examples](src/loom/test/corpus/authoring/cxx/README.md) cover the
frontend's repository workflow. Hardware execution tests additionally need the
selected driver and a compatible device; the target and build guides describe
those requirements.
