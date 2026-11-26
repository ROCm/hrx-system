# loom-reduce

`loom-reduce` rewrites Loom sources to remove the duplication `loom-analyze`
reports. The user workflow is documented in
[Find duplicate code](../../../../docs/src/workflows/find-duplicate-code.md#reduce-the-source);
`loom-reduce --agents_md` prints the agent summary.

## Layout

- `reduce.h` owns the rewrite contract: scopes, jobs, skip and revert rules,
  and the report and link formats.
- `loom-reduce.c` loads and verifies the source root through the shared
  `loom-analyze` session, runs the clone analysis in process, applies the plan,
  and writes the rewritten copy of the root.

## Design

The analysis, source ranges and value bindings come from
`loom-analyze`'s `clone_analysis` library; `loom-reduce` does not re-match
anything. Facts about symbols also come from the verified native modules:

- the call graph used to reject recursive motifs is built from symbol references
  in operation attributes;
- declarations copied into a shared motif file are printed from their defining
  operations;
- an existing motif's parameters and results are its entry block arguments and
  `func.return` operands.

Only the rewrite itself is textual: copies are replaced at their exact source
ranges, so every untouched line, comment and layout choice is preserved. Every
rewritten file is parsed and verified in memory before it is written.

## Tests

`loom-reduce.test.json` covers the CLI contract: help, flag and output-directory
errors, a file-scope rewrite compared byte for byte with
`testdata/expected/kernel.loom`, and publication of the corpus-scope motif file
and link inputs.
