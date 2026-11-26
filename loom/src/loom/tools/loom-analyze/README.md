# loom-analyze

`loom-analyze` is a static analyzer for Loom sources. It reports operation sequences
written more than once so they can be factored into motifs. The user workflow
is documented in
[Find duplicate code](../../../../docs/src/workflows/find-duplicate-code.md);
`loom-analyze --agents_md` prints the agent summary.

## Layout

- `clone_analysis.h` owns the analysis contract: what counts as a sequence,
  when two sequences match, the rules, and the result structures.
- `report.h` owns the text, JSON and SARIF 2.1.0 output formats.
- `inputs.h` owns command-line path expansion and `--exclude` matching.
- `loom-analyze.c` parses and verifies inputs through LoomC, then borrows native
  modules through the interop view for analysis.

## Matching model

Every operation is printed once with regions elided, then normalized by
removing SSA names, block labels, comments and layout. The hash of that text
covers operation names, attributes, symbols, constants and types by content.
Interned string and symbol IDs are local to each module, so hashing their text
is what makes copies in separately parsed files comparable.

SSA references inside a candidate window (operands and values embedded in
result types, block argument types and attributes) are renamed to
window-relative tags: the Nth value defined in the window, or the Nth distinct
value flowing in. A window's canonical token stream is the operation hashes
interleaved with those tags and with region and block structure. Equal streams
are clones. Fingerprints use FNV-1a over that stream, so they are stable
across processes and platforms and suitable for SARIF baselines.

Detection follows the usual token-based clone detector shape: one candidate per
start position (the shortest window reaching `--min-ops` operations), buckets
by hash, exact comparison inside buckets, forward extension of each group, and
removal of classes subsumed by a larger reported class.

## Tests

`loom-analyze.test.json` covers the CLI contract: agent help, flag rejection, exit
status 0/1/2, report publication and `--exclude`. The fixtures in `testdata/`
are the minimal programs that drive each exit status.
