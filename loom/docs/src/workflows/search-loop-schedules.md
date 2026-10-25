# Search per-instance loop schedules

A kernel can instantiate the same motif with several independent schedules.
Template arguments make each caller's pipeline depth and unroll factor a search
dimension. Compile reports expose the resulting queues, native code size,
register allocation, waits, and modeled residency before device execution.

This walkthrough composes two instances of the
[vector-row reduction](tune-loop-schedules.md#give-each-motif-its-own-schedule).
Their inputs, row counts, outputs, and policies are independent. It uses global
configuration **only in the experiment entry** to vary those arguments. The
reusable motif takes SSA values; production callers supply their own constants,
template arguments, or calculations from specialized target properties.

## Keep the instances independent

Save these three files together:

- [vector-read-ahead.loom](../generated/examples/guide/functions-and-control/vector-read-ahead.loom): the reusable motif.
- [paired-read-ahead.loom](../generated/examples/guide/functions-and-control/paired-read-ahead.loom): the two-instance experiment entry.
- [paired-read-ahead-tests.loom](../generated/examples/guide/functions-and-control/paired-read-ahead-tests.loom): independent expected results and a named benchmark.

The caller converts lookahead to pipeline depth with ordinary arithmetic:

```loom
--8<-- "examples/guide/functions-and-control/paired-read-ahead.loom:caller-policies"
```

Lookahead zero gives depth one, the serial control. The defaults request depth
four/unroll four on the left and depth one/unroll one on the right. Pipelining
runs before unrolling; each instance can change either policy independently.

With [installed Loom tools](../getting-started/acquiring-loom.md), link the
caller, motif, and checks once:

```shell
loom-link vector-read-ahead.loom paired-read-ahead.loom \
  paired-read-ahead-tests.loom --mode=merge --to=bc \
  --output=paired-read-ahead.loombc
```

## Compile a bounded search

The following Bash session compiles sixteen combinations: each instance chooses
depth one or four and factor one or four. The deeper probes expose the cost of
retaining more live records. Compilation and report inspection need no GPU;
`gfx1151` makes the target-specific resource estimates concrete.

```bash
set -euo pipefail
mkdir -p paired
sha256sum "$(command -v loom-compile)" paired-read-ahead.loombc \
  >paired/inputs.sha256

compile_pair() {
  local left_depth="$1" left_factor="$2" right_depth="$3" right_factor="$4"
  local candidate="paired/l${1}u${2}-r${3}u${4}"
  loom-compile paired-read-ahead.loombc --root=@sum_paired_rows \
    --target=amdgpu:gfx1151 --format=amdgpu-hsaco \
    --config=paired_rows.left_lookahead="$((left_depth - 1))" \
    --config=paired_rows.left_unroll="$left_factor" \
    --config=paired_rows.right_lookahead="$((right_depth - 1))" \
    --config=paired_rows.right_unroll="$right_factor" \
    --output="$candidate.hsaco" --compile-report=details \
    --compile-report-output="$candidate.report.json" || return
  loom-compile-report show "$candidate.report.json" --format=json \
    >"$candidate.view.json"
  loom-compile-report suggest "$candidate.report.json" \
    >"$candidate.suggest.txt"
}

for left_depth in 1 4; do
  for left_factor in 1 4; do
    for right_depth in 1 4; do
      for right_factor in 1 4; do
        compile_pair "$left_depth" "$left_factor" "$right_depth" "$right_factor"
      done
    done
  done
done
for depth in 8 16 32; do
  compile_pair "$depth" 4 "$depth" 4
done
sha256sum paired/*.hsaco paired/*.report.json >paired/outputs.sha256
```

Each report's applied bindings identify the candidate. `loop_pipelines` in the
bounded `show` view records both applied depths, even after the template has
been inlined and the loops transformed. `suggest` names experiments justified
by the available evidence:

```text
--8<-- "generated/examples/guide/functions-and-control/paired-pipeline-suggest.txt"
```

## Inspect costs before measuring

The grid includes serial, unroll-only, pipeline-only, and combined controls.
Compare one instance at a time while holding the other instance and remaining
policy fixed; changing both depth and unroll factor cannot isolate either effect.

These rows are generated from the example's reports during the documentation
build. Each policy column is **depth / unroll factor**:

--8<-- "generated/examples/guide/functions-and-control/paired-resources.md"

Deeper queues can cross a register allocation threshold and lower modeled
residency without spilling. A search can reject such a candidate when it exceeds
an application's explicit resource budget, or retain it as a latency-hiding
experiment. Higher residency alone does not establish a faster kernel.

The report also separates static wait counts, allocation copies, and emitted
instructions. An unroll factor can duplicate waits while doing more useful work
per backedge. Static counts do not measure cycles spent waiting. Runtime bounds
can leave dynamic instruction and byte counts unavailable; unavailable evidence
is not zero cost.

For a controlled depth comparison, hold both unroll factors and the other
instance's policy fixed:

```shell
loom-compile-report diff paired/l1u4-r1u4.report.json \
  paired/l4u4-r1u4.report.json --force
```

The default comparison rejects the changed configuration. `--force` exposes the
intentional mismatch and labels the deltas observational. A reproducible search
retains the linked source, compiler identity, target, numeric contract, all
applied settings, and artifact/report identities. Report identity covers recorded
compilation fields; it is not a source or compiler content hash.

## Check and time the same candidate

The two inputs use distinct exact dyadic sequences. Closed-form expectations
cover eleven count pairs, including one empty instance beside a full instance,
short startup paths, unequal lengths, and remainders. Distinct output sentinels
also expose missing or crossed stores.

On a compatible AMDGPU device, check a candidate with access instrumentation:

```shell
loom-test paired-read-ahead.loombc --device=amdgpu \
  --target=amdgpu:gfx1151 --sanitizer=access \
  --config=paired_rows.left_lookahead=3 --config=paired_rows.left_unroll=4 \
  --config=paired_rows.right_lookahead=0 --config=paired_rows.right_unroll=1
```

Measure with optimized, uninstrumented tools using the same settings:

```shell
loom-benchmark paired-read-ahead.loombc --benchmark=@paired_rows_time \
  --device=amdgpu --target=amdgpu:gfx1151 \
  --config=paired_rows.left_lookahead=3 --config=paired_rows.left_unroll=4 \
  --config=paired_rows.right_lookahead=0 --config=paired_rows.right_unroll=1 \
  --measure=dispatch_complete --batch-size=128 --input-ring-count=1 \
  --artifact-bundle-dir=paired/mixed-run --artifact-bundle-policy=debug
```

The named workload has 64 rows on the left and 63 on the right. This command
measures host submission through completion of a batch, normalized per logical
operation, with hot input reuse. One workgroup makes it a compact workflow
example; it cannot establish large-grid throughput or demonstrate the benefit of
higher occupancy. An application supplies its production shape and reuse policy.
Alternate candidate order, retain correctness and compiler evidence with timing,
and apply the [benchmark workflow](benchmark.md) before selecting a winner.

## Run the search inside an embedding

The [C API](../reference/c-api/index.md) supports the same loop: reuse an immutable
context, compiler, and pass program; clone the authored module for each candidate;
compile with a typed configuration module; then emit the artifact and report.
The compiled module retains applied bindings even when compilation requested no
report. Configuration and result handles can be released before separate
emission, and module cloning preserves those binding strings. A later compilation
replaces the retained invocation facts. Serializing IR alone does not preserve
invocation bindings, so archive the report alongside its artifact.

Compile cost is a search dimension too. Record report-disabled, summary, and
detailed emission separately, distinguish reused from trimmed workspace storage,
and check repeated-candidate memory. After selecting a policy, move the values or
target-derived calculation into the production caller and keep the checked loop
body and boundary cases.
