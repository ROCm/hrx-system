# Source-Low Corpus

This directory contains shared source programs for tests that assert lowered
output or execution results. Shared input without its own expectations is
exported `.loom` data, not a standalone roundtrip test. Its value comes from the
consumers' assertions: selected instructions, addressing, access widths, returned
values, or other observable behavior. Parser/printer roundtrip coverage belongs
beside the corresponding format implementation.

A target participates through a `TEMPLATE` fixture with its own compiler options,
RUN mode, and output assertions, or through an execution test with an independent
oracle. TEMPLATE copies the common source without adding declarations, changing
modifiers, or binding targets in the IR. Each target/profile instantiation needs
an independently useful assertion; corpus membership does not require a backend
matrix. Successful compilation alone adds no correctness assertion. Missing
lowering support belongs in work tracking, not a list of expected errors copied
across positive programs. Precise diagnostic expectations describe intentional
rejection behavior and live beside the subsystem that owns that contract.

Corpora without useful assertions for a target need no fixture or build
registration. A mixed fixture excludes exact cases outside its assertion contract
in its file preamble:

```text
// TEMPLATE: loom/src/loom/test/corpus/source_low/view_transport.loom-test
// TEMPLATE-EXCLUDE: @correlated_cfg_rotation SPIR-V requires structured control flow.
```

`TEMPLATE-EXCLUDE` requires a reason and uses the same function-symbol identity
as synchronization. Unknown or duplicate names fail, including when a corpus
case is renamed or removed. All other cases remain synchronized, including new
cases added to the corpus. Excluding every case is an error; an entirely
excluded corpus has no fixture. A target's dedicated rejection tests cover its
architectural boundary.

A case containing several functions has one public entry and private helpers.
The entry identifies the case and receives the requested compiler profile;
call-graph specialization supplies helper targets in compiler-owned function
versions. Helpers remain part of the shared input, including when a target
preserves their call boundaries. Compiler options preserve source visibility.

Shared analysis and report semantics have focused tests beside the owning
analysis or pass. The shared source-to-low tests use the backend-independent
test Low target for behavior such as loop execution counts and nested emission
accounting. Corpus programs here exercise backend capabilities such as memory
lowering or instruction selection; duplicating a shared analysis assertion
across target bindings adds no coverage of those semantics.

Do not put target-specific diagnostic or expected-failure cases here. Rejection
tests belong beside the target or pass that owns the diagnostic vocabulary, where
they can assert the exact structured error or remark without forcing every other
target to inherit unrelated failure semantics.
