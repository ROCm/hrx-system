# Loom Integration Tests

This tree is for tests that cross product boundaries: source parsing, linking,
target selection, lowering, artifact emission, execution, reporting, and
benchmarking. The unit of coverage is an integration contract, not an individual
operation, pass, tool flag, model layer, or one-off user kernel.

Integration tests are intentionally scarce. They are valuable because they mix
concepts that smaller tests keep separate, and they are expensive for the same
reason. A checked-in integration case represents a reusable class of compiler
and runtime behavior with correctness and performance consequences. It does not
memorialize where a kernel was first seen.

## Ownership

| Area | Owner |
| --- | --- |
| `tools/` | CLI and toolchain wiring smoke tests. These may synthesize tiny source files to prove tool interoperability. |
| `checked_benchmarks/` | Full Loom programs with `check.case` correctness and `check.benchmark` rows, intended to feed both regression tests and continuous benchmark evidence. |

The existing source corpus has different ownership:

| Area | Owner |
| --- | --- |
| `corpus/source_low/` | Target-reusable source-to-low lowering coverage. These files pin accepted IR shapes and target expected output. |
| `corpus/text/` | Parser, printer, bytecode, and textual format coverage. |
| `corpus/authoring/` | Comment-rich examples for humans and agents porting kernels into Loom. These optimize for copyability and explanation. |
| Tool `testdata/` directories | Fixtures that test the tool itself. They are not product workload homes. |

## Checked Benchmark Integration

A checked benchmark integration case is a full workload-shaped Loom program
that has both:

- `check.case` correctness coverage with deterministic inputs and meaningful
  expected values.
- `check.benchmark<@case>` rows that select the same checked workload for
  timing and artifact/report capture.

Correctness-only execution is not enough for this suite. Performance-only rows
are not enough either. The point is to keep semantic and performance evidence
attached to the same source contract, so a future compiler change cannot make a
kernel faster by silently changing the math, and a future correctness fix cannot
hide a severe performance regression.

All-zero or all-one inputs can be useful as companion smoke samples, but they do
not carry enough information to be the only oracle for an arithmetic integration
case. Expected values should distinguish storage interpretation, signedness,
scale placement, address calculation, accumulation, tail handling, and any
other behavior class the case claims to represent.

## Admission

An integration case earns a permanent checked benchmark slot when it satisfies
all of these pressures:

- It represents a named compiler behavior class that cannot be reduced to an
  existing op, pass, source-low, tool, or authoring test without losing the
  failure mode.
- It carries at least one correctness sample that would fail for a plausible
  bug in the represented class.
- It carries at least one benchmark row whose name and parameters can remain
  stable enough for continuous benchmarking.
- Its source uses generated or compact deterministic data unless an external
  fixture is the behavior under test.
- Its shape is the smallest representative member of the class, plus named
  benchmark rows for larger classes only when scale changes compiler behavior.
- Its target requirements are explicit, so unavailable hardware reports a skip
  or unavailable result instead of silently reducing coverage.
- Its comments explain the represented behavior class and the non-obvious
  contract boundaries without becoming a porting cookbook.

The suite rejects cases whose main identity is a product, model, layer,
benchmark winner, local repro filename, or single hardware result. Those details
belong in issue notes, benchmark databases, external catalog metadata, or local
investigation notes. If such a kernel exposes a real compiler class, reduce it
to that class before adding it here.

## Naming

Names are semantic and class-oriented. A good name identifies the storage or
layout class, decode or movement contract, compute family, and shape class only
where shape is part of the represented behavior.

Examples of useful name axes:

- `block_quant_s8_fields`
- `f16_scale_f32_dot`
- `ragged_page_table_gather`
- `workgroup_reduction_tail`
- `subgroup_shuffle_reduce`
- `wide_vector_global_memory`

Names based on `q8`, model families, product names, layer numbers, local target
nicknames, or session provenance are not stable enough for this suite. A label
such as `q8` collapses storage width, signedness, layout, scale type, block
schema, decode semantics, and target strategy. The checked source should spell
the actual contract instead.

## Target Scaling

Integration cases should be target-neutral until the represented class is
target-specific. Target-specific behavior belongs in the case metadata,
requirements, benchmark row, or suite grouping, not in a copied kernel per
architecture.

When several targets should support the same class, one source should express
the shared workload and target requirements should decide which runners execute
it. A target-specific split is justified when the IR contract itself is
different, such as a target-owned matrix fragment shape or an exact native
helper boundary.

Unavailable hardware is an expected suite state. Skips and unavailable results
must stay structured enough for CI dashboards and benchmark ingestion to
distinguish missing hardware from compiler failure.

## Benchmark Scaling

Benchmark rows are workload selections, not harness policy. Iteration counts,
warmups, profiling mode, artifact bundle policy, input-ring sizes, and
statistical rigor belong to the runner or continuous benchmark configuration.

Named rows correspond to stable shape classes such as `decode`,
`small_prefill`, `full_tile`, `tail`, or `stress` when those names describe a
compiler or runtime behavior. A row named after a local experiment or a model
instance is not a reusable benchmark key.

Continuous benchmark ingestion should be able to identify:

- Source case name.
- Benchmark row name.
- Effective target.
- Target requirement result.
- Compile status and artifact identity.
- Correctness status.
- Measurement method.
- Timing/statistical payload.

The checked source should give the benchmark database stable semantic keys. The
benchmark runner decides how much timing data to collect for smoke, presubmit,
nightly, and tuning jobs.

## Lifecycle

Integration cases are allowed to evolve. If a smaller or more general case
covers the same behavior class, the old case should be retired or moved to an
external benchmark catalog instead of accumulating forever. If a user kernel
finds a new bug, the durable action is to identify the missing behavior class,
then decide whether the class belongs here, in source-low coverage, in an op or
pass test, in tool testdata, or in authoring docs.

The first checked benchmark in a class sets precedent for every future case in
that class. Establish the name, scope, data policy, target requirements, and
benchmark rows before adding a seed kernel.
