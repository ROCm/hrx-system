# AMDGPU Execution Corpus

This package composes complete Loom cases into capability-coherent modules and
executes them through the AMDGPU HAL with `iree-test-loom`. Generic corpus
packages own target-independent semantics; this provider package owns device
selection, architecture qualification, and physical execution.

Architecture directories contain only cases whose authored contract names that
architecture. The generic module combines reusable corpus archives so all of
their cases share one device and execution context.
