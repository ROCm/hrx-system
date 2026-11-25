# Find duplicate code

`loom-analyze` is a static analyzer that finds operation sequences written more than
once across Loom sources. Each finding is a candidate motif: a
[`func.def inline`](../guide/functions-and-control.md) reached through
`func.call`, or a [template family](../guide/facts-and-specialization.md)
provider reached through `template.apply`. Replacing the copies with one motif
reduces the code you maintain and keeps fixes in one place.

## Analyze a source tree

Pass files or directories. Directories are searched recursively for `.loom`
files:

```shell
loom-analyze kernels/
```

Each finding is a warning anchored at the first copy, followed by one note per
copy:

```text
kernels/motifs/copy/concat_dim0.loom:121:7: warning: 10-op sequence is duplicated 3 times; factor it into a func.def inline or template.def motif [loom-analyze/duplicate-sequence]
kernels/motifs/copy/concat_dim0.loom:121:7: note: copy spans lines 121-130 in @ggml_copy_concat_dim0_strided_source_f32_body
kernels/motifs/copy/concat_dim0.loom:134:7: note: copy spans lines 134-143 in @ggml_copy_concat_dim0_strided_source_f32_body
kernels/motifs/copy/strided_source.loom:41:5: note: copy spans lines 41-50 in @ggml_copy_strided_source_f32_body
```

The run ends with a summary:

```text
loom-analyze: 1230 finding(s) in 286 module(s); 27998 of 46773 ops (59.9%) are in duplicated sequences
loom-analyze: refactor plan of 502 non-overlapping finding(s) removes 13170 ops (28.2%)
```

The first line counts the operations inside definitions that lie in any
reported copy. Findings can overlap; an unrolled loop, for example, matches at
every phase of its body. The second line counts only the
[refactor plan](#apply-the-refactor-plan): findings whose copies are disjoint,
so applying all of them is possible and removes exactly that many operations.
Module-level declarations, such as the `template.decl` and `func.decl` lines
each file repeats to link against a library, are never reported.

## Understand what matches

Two sequences match when they are the same program with different value names.
Operation names, attributes, symbols, constants, and types compare by content.
Operands compare by dataflow: a value defined inside the sequence must be used
in the same positions, and values flowing in from outside become the motif's
parameters. Operands of commutative operations compare as a multiset. A nested
region, such as an `scf.if` body, matches as part of its operation, and
sequences inside nested regions are candidates in their own right.

Size counts every operation, including nested ones. `--min-ops` sets the
smallest reported sequence (default 6) and `--min-occurrences` the fewest
copies (default 2). Findings are maximal: a shorter sequence is not reported
separately when every copy of it lies inside a larger reported finding.

## Find copies of an existing motif

When a sequence repeats the complete body of a `func.def` or `template.def`
found among the inputs, `loom-analyze` reports each copy against that definition:

```text
kernels/ops/mul_mat_q5_k_q8_plane_wmma.loom:119:3: warning: 17-op sequence repeats the body of motif @ggml_iq4nl_table_i8; call the motif instead of copying it [loom-analyze/reimplemented-motif]
kernels/ops/mul_mat_q5_k_q8_plane_wmma.loom:119:3: note: copy spans lines 119-135 in @ggml_q5_plane_dense_iq4xs_wmmai8_signed_values16
kernels/motifs/dequant.loom:2561:3: note: motif @ggml_iq4nl_table_i8 is defined here
```

Analyze the motif library together with its consumers so its definitions are
visible.

## Find sequences that differ only by constants

Sequences that differ only in literal values are motif candidates with a
parameter for each differing constant:

```shell
loom-analyze --parameterize=constants kernels/
```

With `constants`, `*.constant` operations compare by result type without their
literal value, and each message says the match holds up to constant values.

`--parameterize=types` also matches sequences that differ only in element types,
such as the same computation on `f32` and `i16`. Loom motifs cannot be generic
over element types, so these findings are reported with `"applicable": false`
and never join the refactor plan. The summary states how many more operations
type-generic motifs would remove. Relaxations combine:
`--parameterize=constants,types`. A finding records the relaxations its copies
actually needed in `parameterized_by`; copies that match exactly stay exact even
when relaxations are enabled.

## Copies in launch configuration

A copy inside a `kernel.def` launch-config region carries an extra note:
launch configuration cannot call functions today, so that copy cannot be
replaced by a call. The finding still identifies the duplicated computation.

## Apply the refactor plan

`loom-analyze` never edits source. Its JSON report is a complete plan for a
separate rewriter, whether a script or an agent:

```shell
loom-analyze --format=json --output=plan.json kernels/
```

Each finding carries `removable_ops`, the operations its suggested refactor
removes, and `selected`, which marks membership in the plan. Selected findings
are chosen largest saving first so that no two of them touch the same
operation; they can be applied together in any order. For every copy, the
report records:

| Field | Use |
| --- | --- |
| `start_line`, `start_column`, `end_line`, `end_column` | Exact source text to replace with a call |
| `inputs` | Values flowing in, as `{"name": "%x", "type": "index"}`, in motif parameter order |
| `outputs` | Values the call must bind, in motif result order |
| `callable` | False inside a launch-config region, where the copy must stay inline |

Entry *j* of `inputs` binds the same parameter in every copy of a finding, so a
rewriter builds one `func.def inline` from the first copy's text, using its
input names as parameters and its outputs as results, and replaces each
callable copy with
`%out0, ... = func.call @motif(%in0, ...) : (...) -> (...)` spelled with that
copy's names. A module calling a motif defined in another file also needs its
`func.decl`, and the program that links the kernel must supply the motif's
file. An inline motif is inlined during compilation, so each kernel's emitted
code can be compared before and after the rewrite. The motif returns every value
any copy uses afterwards, so all copies share one signature.

## Reduce the source

`loom-reduce` applies the plan. It copies the source tree to an empty output
directory and rewrites only the copy:

```shell
loom-analyze --format=json --output=plan.json kernels/
loom-reduce --plan=plan.json --output-dir=reduced/ --report=reduce.json
```

Run it from the directory where `loom-analyze` ran, or pass `--root`. Each
selected finding becomes one `func.def inline` motif, defined at the end of the
file holding its copies, and each callable copy becomes a call. Findings whose
copies differ in constants or types, and copies of an existing motif in another
file, are skipped with a reason. Every rewritten file is parsed and verified
with `loom-analyze`; a file that fails keeps only the motifs that verify on
their own. The summary and `reduce.json` list applied, skipped and reverted
motifs:

```text
loom-reduce: 293 motif(s) replace 998 copies in 79 file(s); removes 6542 of 46773 ops (14.0%)
loom-reduce: 49 skipped, 10 reverted after verification
```

Analyzing the reduced tree again can find further copies; a second pass on the
same corpus removed another 841 operations. Inlining makes most kernels compile
to identical code, but the compiler may select different instructions around a
call boundary, so compare the compiled kernels and measure any that change
before adopting the reduced sources.

## Use loom-analyze in CI

Write SARIF 2.1.0 for code-scanning services and baseline tools:

```shell
loom-analyze --format=sarif --output=loom-analyze.sarif kernels/
```

Each result carries a `partialFingerprints` entry derived from the matched
program, independent of file names and value names, so a finding keeps its
identity across runs while the code moves. `--format=json` writes a compact
report with the same findings for scripts.

`--exclude` skips paths matching a `*` and `?` wildcard pattern against the
whole `/`-separated path, and may be repeated:

```shell
loom-analyze --exclude='*/testdata/*' src/
```

The exit status separates findings from failures:

| Status | Meaning |
| --- | --- |
| 0 | No findings |
| 1 | Findings reported |
| 2 | Invalid flags, or an input could not be read, parsed, or verified |

Inputs that fail to load are reported on standard error; the remaining inputs
are still analyzed and the run exits with status 2.

## Limits

Sequences must be contiguous operations of one block, and a block's terminator
is never part of one. Copies interleaved with unrelated operations are not
matched. Copies that differ in element types are reported only with
`--parameterize=types` and cannot be reduced until motifs can be type-generic.
