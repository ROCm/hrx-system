# Compiler benchmark smoke kernels

This package contains the small, fixed inputs used to prove that compiler
throughput benchmarks exercise the production compilation path. It is a smoke
suite, not a benchmark corpus or model zoo. Representative model collections,
target tuning studies, and broad performance datasets have different ownership
and scale requirements and do not accumulate here.

The layout groups fixtures by algorithm:

- `attention/` contains fused attention workloads.
- `ffn/` contains fused feed-forward workloads.
- `synthetic/` contains minimal inputs that isolate benchmark-harness behavior.

Each logical benchmark cluster has one exported `iree_c_embed_data` target.
Target-specific implementations of the same algorithm live in that cluster and
use algorithmic names. The target suffix distinguishes implementation details;
model provenance is intentionally absent because it is not part of the measured
contract.

A fixture belongs here only when a registered smoke benchmark consumes it to
exercise a distinct compiler phase, scaling dimension, or target path. New
coverage normally extends or replaces an existing cluster. A collection whose
value comes from its breadth is a corpus and needs a separately designed
package rather than another entry in this one.

The embedded-data targets contain source text only and carry no target-library
dependencies. Target benchmark binaries select the clusters they consume and
remain guarded by their production target capabilities. Disabling AMDGPU or
SPIR-V therefore removes that backend benchmark without coupling the common
runner or the other backend to unavailable target code.
