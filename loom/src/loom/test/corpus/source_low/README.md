# Source-Low Corpus

This directory contains positive, target-reusable source programs for
source-to-low lowering coverage. A corpus case should be a program we want every
compatible target to accept, lower, and eventually execute or compare against an
oracle. Each applicable backend participates through a `TEMPLATE` fixture with
its own target compiler options, RUN mode, and output assertions. TEMPLATE copies
the common source without adding declarations, changing modifiers, or binding
targets in the IR. A backend that does not yet implement the required capability
records the precise structured diagnostic
in its fixture. This keeps the positive source shared and makes newly supported
behavior visible when that diagnostic stops appearing. Profile-specific
instruction selection and pass-mode checks can use separate fixtures consuming
the same corpus program.

Architecturally inapplicable corpora need no target fixture or build
registration. The target suite documents the source/target contract that makes
them inapplicable, with rejection coverage beside the owning subsystem. A mixed
fixture excludes exact cases in its file preamble:

```text
// TEMPLATE: loom/src/loom/test/corpus/source_low/view_transport.loom-test
// TEMPLATE-EXCLUDE: @correlated_cfg_rotation SPIR-V requires structured control flow.
```

`TEMPLATE-EXCLUDE` requires a reason and uses the same function-symbol identity
as synchronization. Unknown or duplicate names fail, including when a corpus
case is renamed or removed. All other cases remain synchronized, including new
cases added to the corpus. Excluding every case is an error; an entirely
inapplicable corpus has no fixture. Missing lowering implementations continue
to use precise diagnostics in applicable target fixtures.

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
