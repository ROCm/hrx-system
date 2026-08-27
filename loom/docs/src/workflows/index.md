# Workflows

Workflow pages answer task-oriented questions about public Loom tools. They
start with binaries on `PATH` and ordinary `.loom` or `.loombc` inputs. Source
checkout commands, Bazel labels, compiler internals, and target-specific runtime
setup have separate owners.

Use [Agent-driven kernel development](agent-driven-kernel-development.md) when
the task spans several tools and candidate iterations. It keeps numerical,
compiler, and physical evidence joined to one production witness while the
focused pages below own each command's exact contract.

## The ordinary authoring loop

Most kernel and program work repeats the same evidence sequence:

1. **Format and verify** each source module before composition.
2. **Link and specialize** the roots and libraries needed for one artifact.
3. **Test correctness** through the `check.case` programs owned beside the
   source.
4. **Benchmark** named `check.benchmark` rows that select those checked
   workloads.
5. **Compile** the selected roots for a target profile.
6. **Inspect evidence** only when a correctness, performance, or target question
   requires it.

The sequence is not a mandatory monolithic command. A one-file kernel can go
directly from formatting to testing or compilation. A reusable library may stop
at `.loombc`. A JIT embedding can perform the same operations in memory through
`loomc`.

## Source and composition

| Task | Workflow |
| --- | --- |
| Check or rewrite canonical source | [Format and verify source](format-and-verify.md) |
| Convert text and bytecode | [Format and verify source](format-and-verify.md#convert-text-and-bytecode) |
| Inspect symbols and dependency closure | [Link and package modules](link-and-package.md#inspect-before-linking) |
| Build a root-selected artifact input | [Link and package modules](link-and-package.md#link-one-program) |
| Merge a reusable bytecode catalog | [Link and package modules](link-and-package.md#merge-a-reusable-catalog) |

## Development loops

| Task | Workflow |
| --- | --- |
| Drive a kernel from production witness through controlled optimization | [Agent-driven kernel development](agent-driven-kernel-development.md) |
| Turn a score change into a compiler question | [Read compile reports](compile-reports.md#relate-reports-to-benchmark-results) |
| Preserve a complete result and its compiler evidence | [Benchmark checked work](benchmark.md#preserve-the-experiment) |

## External oracles

An external compiler or runtime can answer a narrowly defined question without
becoming Loom's source language or schedule. The oracle workflow preserves the
exact selected program, workload, tool identity, and evidence boundary.

| Oracle | Question it can answer |
| --- | --- |
| [RADV and Vulkan](oracles/radv.md) | Which shader and physical schedule an optimized Vulkan stack selected for one dispatch. |
| [LLVM MC](oracles/llvm-mc.md) | Whether selected native instructions and packets encode to the expected target bytes. |
| [GGML and llama.cpp](oracles/ggml-llama-cpp.md) | Which model, storage, graph, and optimized-runtime contracts a port must preserve. |

## Correctness and performance

`iree-test-loom` executes `check.case` records and emits structured correctness
results. `iree-benchmark-loom` plans `check.benchmark` records, gates timing on
the selected case's correctness, and records measurement evidence. Keeping
those tasks separate prevents a fast incorrect candidate from becoming a
performance result.

| Task | Workflow |
| --- | --- |
| Execute every checked case | [Test correctness](test-correctness.md#run-every-case) |
| Select one case or sample | [Test correctness](test-correctness.md#select-a-case-or-sample) |
| Test a wrapper against bytecode libraries | [Test correctness](test-correctness.md#link-test-wrappers-to-libraries) |
| Add target-pipeline sanitizers | [Test correctness](test-correctness.md#instrument-a-test-run) |
| Inspect a benchmark plan without execution | [Benchmark checked work](benchmark.md#inspect-the-plan-first) |
| Measure a complete checked case | [Benchmark checked work](benchmark.md#measure-the-complete-case) |
| Measure device dispatch completion | [Benchmark checked work](benchmark.md#measure-dispatch-completion) |
| Compare candidates in one run | [Benchmark checked work](benchmark.md#compare-candidates-in-one-run) |

The concise path belongs in the testing and benchmarking workflows. Sanitizers,
profiles, interleaved comparisons, artifact bundles, and report archaeology are
advanced sections reached from that ordinary path rather than flags every
tutorial must carry.

## Compilation and evidence

`loom-compile` owns offline specialization and artifact emission. It can select
roots from a catalog, bind configuration, target generic or exact profiles, and
emit runtime and target-native products. Artifact manifests describe what was
emitted; compile reports describe how the compiler produced it.

`loom-compile-report show`, `diff`, and `suggest` are the focused first view over
that evidence. Pass reports, IR traces, native listings, and raw JSON queries
remain available when the focused view identifies a question that needs deeper
inspection.

| Task | Workflow |
| --- | --- |
| Compile a loader-ready artifact | [Compile artifacts](compile-artifacts.md#compile-for-a-hal-loader) |
| Select roots and configuration | [Compile artifacts](compile-artifacts.md#select-roots-from-a-catalog) |
| Describe an emitted artifact | [Compile artifacts](compile-artifacts.md#emit-an-artifact-manifest) |
| Read one bounded report view | [Read compile reports](compile-reports.md#show-one-report) |
| Compare compiler evidence | [Read compile reports](compile-reports.md#diff-one-controlled-change) |
| Get target-owned experiments | [Read compile reports](compile-reports.md#ask-the-target-for-experiments) |
