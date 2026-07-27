# Source-Low Corpus

This directory contains positive, target-reusable source programs for
source-to-low lowering coverage. A corpus case should be a program we want every
compatible target to accept, lower, and eventually execute or compare against an
oracle. Individual targets may mark a positive case unsupported when the target
does not yet implement the required capability.

Do not put target-specific diagnostic or expected-failure cases here. Rejection
tests belong beside the target or pass that owns the diagnostic vocabulary, where
they can assert the exact structured error or remark without forcing every other
target to inherit unrelated failure semantics.
