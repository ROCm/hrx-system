# C++ importer tests

The C++ suite checks source semantics at the verified IR boundary. A fixture
contains C++ source followed by exact Loom output or source diagnostics. Tests
live beside the importer component that owns the contract; none select a device,
launch a program, or publish an execution module.

| Owner | Contracts |
| --- | --- |
| [Source](../source/test/README.md) | Language admission, preprocessing, locations, data models, constants, names, and enum representations. |
| [Symbols](../symbol/test/README.md) | Root selection, linkage, retained calls, template selection, immutable objects, and source exports. |
| [Control](../control/test/README.md) | Evaluation order, short-circuit effects, condition declarations, loops, exits, and switches. |
| [Numeric values](../value/test/numeric/README.md) | Promotions, signedness, conversions, bit representations, updates, and math facade selection. |
| [Vectors](../value/test/vector/README.md) | Lane initialization and effects, native and shaped arithmetic, packed encodings, memory spans, and padded layouts. |
| [Objects](../value/test/object/README.md) | Copies, initialization, references, overloaded operators, member calls, arrays, and mixed record state. |
| [Storage](../value/test/storage/README.md) | Pointer roots and origins, local allocation, volatile observations, updates, and constant buffers. |
| [Lifetime](../lifetime/test/README.md) | Cleanup order, temporary lifetime, and destruction across calls and control flow. |
| [Binding contracts](../binding/test/contracts/README.md) | Named configuration, entry promises, scoped assumptions, and explicit schedules. |
| [Data bindings](../binding/test/data/README.md) | Views, shapes, encodings, dependent calls, input construction, and reference computations. |
| [Kernel bindings](../binding/test/kernel/README.md) | Kernel arguments, topology, atomics, collectives, barriers, and shared storage. |
| [Pipelines](../binding/test/pipeline/README.md) | Command programs, channels, strands, launch metadata, and source projections. |
| [Checks](../binding/test/check/README.md) | Check declarations, generators, typed resources, slices, comparisons, and captured observations. |
| [Assembly](../binding/test/assembly/README.md) | Embedded source literals, operand/result flattening, effect ordering, and diagnostics. |
| [Loom templates](../binding/test/template/README.md) | Family declarations, applications, and provider metadata. |
| `tooling/test` | Source providers, import options, opaque imports, and retained source capture. |

## Authoring and updates

`.cxx-test` uses the same `loom-check` envelope as `.loom-test`. Each `// ====`
section is an independent source case. `// ----` introduces its exact expected
output; `// ERROR` annotations assert diagnostics. The first `RUN` and `INPUT`
lines establish defaults, with per-case overrides for different source options
or cleanup passes. A file groups one semantic family and its related cases.

```sh
# Check one family.
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/control/test:logical

# Regenerate its exact expectations after an intentional change.
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/control/test:logical --test_arg=--update

# Check the full importer and its API tests.
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/...
```

The generated IR makes the source contract inspectable. For example,
`logical` checks that an effectful RHS stays inside the selected branch;
`vector_list_assignment` checks initializer calls before the destination call;
`record_initialization` checks reads of already initialized destination members;
and `pointer_origins` checks the buffer and nonzero origin passed to a helper.
An operation name alone would not establish those ordering and identity facts.

`RUN: roundtrip` checks imported IR and its text round trip. `RUN: pass ...`
exercises the named ordinary passes before comparing exact output. Acceptance
cases use `RUN: verify` when a `static_assert` or source admission is itself the
contract. Shared source/header inputs are declared as Bazel data beside their
consuming fixtures.

`RUN: emit cxx` compares canonical exported C++ and declares
`REQUIRES: loom-export-cxx`; `RUN: emit cxx roundtrip` additionally reimports and
verifies it. These fixtures use the same case separators and updater. The
constant-buffer projection also has a process-level roundtrip check for
second-export equality, which is a stronger idempotence assertion than reimport
verification alone.

The [runner guide](../../../tools/loom-check/README.md) describes all modes,
source options, exact expectations, and diagnostic matching. Native `.cc` tests
exercise API ownership, source lifetimes, callbacks, and retained analysis facts.

## Backend and execution boundaries

C++ to Low expectations live with the backend whose output they assert, in its
`test/source_low` package or target qualification package. They preserve the
actual target configuration without adding target dependencies to importer tests.

Numerical execution, storage guards, access instrumentation, device topology,
and scheduling are separate contracts in the [IR corpus](../../../test/corpus)
and target qualification suites. Curated complete source programs live in the
[C++ authoring corpus](../../../test/corpus/authoring/cxx). CLI admission and
execution belong to the target tooling owner; the CPU tooling smoke checks the
public `loom-run` path from a C++ file through observable buffer results.

The [VM lifecycle fixtures](../../../target/arch/vm/test/special_members.cxx-test)
retain exact observable copy/move and destruction traces, including after C++
export and reimport. Their execution belongs to the VM owner; source rejection
cases remain in the object importer suite.
