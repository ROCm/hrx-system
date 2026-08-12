# Workflows

Workflow pages answer task-oriented questions about public Loom tools. They
start with binaries on `PATH` and ordinary `.loom` or `.loombc` inputs. Source
checkout commands, Bazel labels, compiler internals, and target-specific runtime
setup have separate owners.

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
| Build a selective artifact input | [Link and package modules](link-and-package.md#select-one-program) |
| Package a reusable bytecode archive | [Link and package modules](link-and-package.md#package-an-archive) |

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
