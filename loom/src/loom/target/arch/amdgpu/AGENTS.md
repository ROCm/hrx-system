# Loom AMDGPU Target Guide

This guide applies to the AMDGPU target implementation under this directory and
to the matching Python target sources under `loom/py/loom/target/arch/amdgpu/`.

## Target Truth

Python is the source of truth for instruction descriptors, asm spellings,
feature predicates, generated lookup tables, and source-to-low contract rows.
Any invariant known when the compiler is generated belongs in Python or in the
generator validation path. Generated C should need, at most, an assertion for
those invariants.

C owns target policy, lowering mechanics, value-fact queries, and decisions that
depend on user IR. A growing chain of target-specific `if` statements in C is
usually evidence that a descriptor table or generated contract is missing.

`iree_status_t` is infrastructure failure transport. User IR rejection,
selection failure, unsupported operand forms, missing value facts, and
portability concerns are structured diagnostics, not statuses.

## Assembly Contract

AMDGPU asm form is a product surface. When asm is requested, every function in a
test should print asm form. A fallback to `low.op<>`, a missing function body, or
an opaque low packet is a backend coverage bug unless the test is explicitly
about generic low IR.

Asm tests should cover instruction families, not single lucky examples. If an
instruction family has profitable variants such as tied forms, literal forms,
packed forms, d16 payloads, VOPD planning, or architecture-specific aliases, the
coverage should make those forms visible.

Blocked or compatibility-only aliases should fail with deliberate structured
diagnostics. Silent fallback makes oracle comparison and agent authoring worse.

## Facts And Lowering

Value facts are the boundary between author intent and target legality. Lowering
should ask facts for ranges, alignment, memory space, vector footprints, native
mask demand, and descriptor bounds. Producer walking is a last resort because it
is local, slower, and misses facts published across CFG, SCF, block arguments,
and imported values.

Offset-domain arithmetic is the normal spelling for byte addressing. Index
values are shaped element coordinates. Kernel buffer arguments should publish
the facts the ABI already implies, such as global memory space, and user
annotations should only be required for facts genuinely dependent on the
program.

Packed quantized kernels are a core workload. Narrow integer constants,
sub-dword loads/stores, bit extraction/insertion, sign extension, mixed
f16/f32 arithmetic, tied accumulation, and scalar/vector bitcasts need to work
as ordinary source-low constructs rather than as private low-level tricks.

## Diagnostics

Diagnostics should use the structured diagnostic toolkit with templated fields.
Prose is for rendering the diagnostic, not for smuggling the machine-readable
reason. Useful AMDGPU diagnostics usually name the descriptor family, operand
role, expected register or scalar class, required value fact, target feature,
and selected or rejected packet form.

Runtime checks belong only where the answer depends on user input at compile
time. Descriptor-table consistency, packet-field compatibility, spelling
coverage, and feature-row completeness are build-time validation problems.

## Tests

Authoring fixtures should be generic and durable: describe the data movement,
numeric shape, and hardware behavior being modeled instead of product names,
model names, or private repro context.

Source-low tests should model realistic author IR: offsets for byte addresses,
dynamic descriptors for user buffers, compact but plausible control flow, and
comments only where they explain the contract being exercised.

Low and source-low asm tests should be updated with `--update` when the intended
asm form changes. Redundant `RUN` lines belong at the top-level test harness
only when the same invocation applies to all cases.

Large arbitrary kernels are integration tests, not corpus unit tests. They need
a scalable home, correctness evidence, and performance evidence before becoming
durable test suite material.
