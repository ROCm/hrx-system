# Raise a native schedule into Loom

A native kernel can be both a performance reference and a map of behavior that
Loom source has not recovered yet. The goal of reconstruction is a maintained
source or motif implementation that passes the default compiler pipeline,
produces the intended result through a Loom runner, and matches the relevant
physical performance boundary. Preserving the native program as an executable
oracle makes that migration incremental without mistaking the oracle for the
finished implementation.

This workflow applies whether the schedule came from hand-written assembly, a
vendor library, another compiler, or a prepared-Low reduction of one of those
programs.

## Keep the representations distinct

Each representation answers a different question:

| Form | What it establishes | What remains open |
| --- | --- | --- |
| Selected native artifact | The reference system executed these target instructions for the captured workload. | Which semantic and scheduling choices produced the result. |
| Prepared Low | Loom can represent, assemble, and execute the reconstructed target program. | Whether the default pipeline can recover or preserve it. |
| Source with `low.invoke` | Most of the algorithm survives normal lowering while one explicit physical fragment remains. | Whether that fragment can be raised or selected from source semantics. |
| Source or motif | The maintained representation survives the default pipeline. | Numerical and physical acceptance for the complete workload. |

Keep the exact workload, inputs, outputs, launch geometry, target, and timing
boundary joined to every form. A nearby shape or an instruction sequence copied
from a different dispatch is not the same oracle.

## Establish an executable Low oracle

Translate the selected native program into the smallest ABI-complete Low module
that retains its dataflow, resource requirements, and launch contract. Execute
it through Loom rather than loading its artifact through an unrelated runtime:

```shell
iree-run-loom oracle.loom \
  --backend=amdgpu-hal \
  --function=kernel \
  --pipeline=none \
  --kernel-input-buffer=64xf32=0 \
  --expected-kernel-buffer=64xf32=0
```

`--pipeline=none` disables all compiler transformations. The input passes
directly to the selected emitter and must already satisfy its complete input
contract. This is a supported assemble-and-run path for an already lowered
program. Its passing result establishes the prepared Low as an executable
oracle; it does not establish that ordinary Loom source survives compilation.

Use [`loom-compile`](../compile-artifacts.md) when the immediate question is
artifact construction rather than execution:

```shell
loom-compile oracle.loom \
  --format=amdgpu-hsaco \
  --pipeline=none \
  --output=oracle.hsaco
```

Preserve the unmodified Low input, emitted artifact, numerical result, and
measurement record. Instruction identity is useful diagnostic evidence, but
matching behavior and performance are the oracle contracts.

## Recover the schedule's behavior

Read the native program as a set of responsibilities rather than a line-by-line
translation exercise:

| Responsibility | Reconstruction question |
| --- | --- |
| Output ownership | Which workgroup, wave, lane, and accumulator owns each published value? |
| Memory traffic | Which addresses and packet widths does each lane issue, and in what reuse order? |
| Fragment representation | How are matrix inputs, accumulators, and packed values distributed and transformed? |
| Dependency latency | Which independent operations cover each load, matrix, or export dependency? |
| Synchronization | Which waits and barriers follow from real producer-consumer or reuse hazards? |
| Resource lifetime | Which values must remain live together, and which source choice lengthens that interval? |

This separates load-bearing choices from incidental register numbers and exact
instruction order. A native wait, fence, or move is evidence of a dependency to
explain, not automatically a construct the maintained source should copy.

## Raise the largest stable boundary first

Start from the highest source form that can express the recovered ownership,
layout, memory, and numerical semantics. Keep the prepared Low unchanged as the
oracle, then move one independently testable responsibility at a time into the
maintained implementation.

When a small physical fragment is the remaining boundary, an ordinary
schedule-free [`low.invoke`](../../guide/functions-and-control.md#invoke-an-authored-low-fragment)
can carry it while surrounding loads, stores, indexing, and launch structure
remain source-level. The helper is inlined into the caller and participates in
the caller's scheduling and allocation. This makes the first compiler
divergence visible without freezing the entire kernel.

`schedule(locked)` has a narrower role: it preserves authored order when exact
order is itself the question under investigation. It is useful for isolating a
scheduler or hazard discrepancy, but it is independent of writing a Low helper
and independent of fixed physical allocation. A locked helper that matches the
oracle shows that order matters somewhere in that fragment; it does not show
that the maintained implementation or default scheduler has recovered the
contract.

After each raised boundary, compare both behavior and compiler evidence. A
changed result identifies a semantic divergence. A passing result with a
different native program may be a valid alternative schedule; instruction
identity is not an acceptance requirement. A performance regression with
preserved results turns into a focused scheduling, allocation, wait, or memory
traffic question.

## Minimize the first compiler divergence

If the next reasonable source form cannot survive the default pipeline, retain
the desired source and reduce the first failing boundary. A useful compiler
packet contains:

- the smallest source or source-plus-`low.invoke` module that reproduces the
  divergence;
- the default-pipeline `iree-test-loom` command;
- expected and observed numerical behavior;
- the prepared-Low oracle and the last passing raised form; and
- the smallest report, IR snapshot, or native listing that identifies the
  divergent behavior.

The minimized packet belongs to the compiler boundary. Register pinning, extra
fences, and larger locked regions can localize the mechanism, but folding them
into the maintained program would hide the missing compiler contract.

Use [compile reports](../compile-reports.md) for the first bounded view of
allocation, scheduling, waits, and emitted resources. Follow a report into raw
IR or native disassembly only after it names the question that needs that
evidence.

## Accept the maintained form

Acceptance returns to the ordinary production path:

```shell
iree-test-loom kernel.loom \
  --case=@oracle_case \
  --device=amdgpu

iree-benchmark-loom kernel.loom \
  --benchmark=@production_shape \
  --device=amdgpu \
  --output=results.json
```

These commands use the default compiler pipeline. The accepted implementation
owns its numerical cases beside the source, covers the production shapes and
tails that matter, and measures the same physical boundary as the reference.
The prepared Low remains valuable as a regression oracle and archaeology aid;
it no longer carries a claim that only the default-pipeline source has proved.
