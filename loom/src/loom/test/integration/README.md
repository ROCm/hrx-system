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

Complete authored programs, including checked benchmark cases, live under
`test/corpus/`. A target provider composes those data-only packages and owns
device selection, architecture qualification, and physical execution.
