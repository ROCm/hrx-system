# Loom Tool Agent Guide

This guide applies to the generic CLI packages under `loom/src/loom/tools/` and
inherits the compiler-wide contract in `loom/AGENTS.md`.

## Target-Owned Integration Coverage

Generic tool packages own target-neutral binaries, unit tests, and smoke
manifests. Physical compilation and execution coverage for an optional target
lives with that target's tooling provider and invokes the generic tool through
its public Bazel label. The dependency points from the provider suite to the
tool; a generic tool package never acquires target-specific manifests,
fixtures, sanitizer suppressions, resource tags, or compatibility constraints
solely to run integration coverage.

Target-neutral execution programs shared by several providers live with the
generic execution subsystem as exported test data. Target-specific source,
requirements, and expected behavior remain beside the provider. This ownership
keeps broad `//loom/src/loom/tools/...` build paths independent of the complete
target and device matrix while preserving physical coverage where it belongs.
