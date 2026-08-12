# Functions and structured control flow

**Example files:** [`loom/docs/examples/mental-model/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/mental-model)

Functions are typed callables that can execute on the host or device. Structured
control flow keeps conditions, iteration domains, and carried values visible to
verification, specialization, unrolling, distribution, and lowering.

In this chapter, you will learn:

- how definitions, declarations, exact calls, and returns compose;
- how function modifiers state visibility, purity, placement, and inline policy;
- how `scf.if` and `scf.for` produce SSA values;
- why loop-carried state is explicit; and
- when to request an implementation contract with `func.apply` instead of
  naming one function.

## Define a typed callable

A [`func.def`](../reference/dialects/func/ops/def.md) owns a symbol, signature,
and body. Every exit uses [`func.return`](../reference/dialects/func/ops/return.md)
with the declared result types:

```loom
func.def pure @affine(%value: f32, %scale: f32, %bias: f32) -> (f32) {
  %scaled = scalar.mulf %value, %scale : f32
  %result = scalar.addf %scaled, %bias : f32
  func.return %result : f32
}
```

Function bodies do not capture module-local SSA values. Arguments, globals,
configuration, and target queries make every input source explicit. That
isolation lets a linker move a function with its dependency closure and lets a
specializer reason about one callable without hidden lexical state.

Functions may return several values. Results remain ordinary SSA values at the
call site:

```loom
%minimum, %maximum = func.call @bounds(%values) : (vector<8xf32>) -> (f32, f32)
```

## Name exact dependencies

[`func.call`](../reference/dialects/func/ops/call.md) asks for one exact symbol.
The current module contains either its definition or a compatible
[`func.decl`](../reference/dialects/func/ops/decl.md):

```loom
func.decl pure @affine(%value: f32, %scale: f32, %bias: f32) -> (f32)

func.def @apply_affine(%value: f32, %scale: f32, %bias: f32) -> (f32) {
  %result = func.call @affine(%value, %scale, %bias) : (f32, f32, f32) -> (f32)
  func.return %result : f32
}
```

The declaration records what this module requires. A linker can satisfy it from
an explicitly supplied source or bytecode library. A missing declaration is a
source verification error; a library search is never allowed to invent the
dependency after seeing an unresolved call.

Exact calls are appropriate when identity is part of the algorithm: one bit
decoder, one reference function, or one helper whose precise implementation the
caller selected.

## Function modifiers state contracts

Modifiers sit on the definition or declaration whose behavior they constrain:

| Modifier | Contract |
| --- | --- |
| `public` | The symbol is visible outside the module. Absence means private. |
| `pure` | The function is deterministic and has no memory effects. |
| `host` / `device` | The calling convention and intended execution domain. |
| `inline` | The current IR stage must remove the callable boundary. |
| `noinline` | The current IR stage preserves the callable boundary. |
| `hot` / `cold` | Execution-temperature evidence for downstream policy. |

These are semantic inputs, not compiler folklore in comments. `inline` is not
needed merely because a helper is small; it is used when the source contract
requires the boundary to disappear at the current stage. Without an explicit
policy, the consuming pass decides from target, call graph, and cost evidence.

## Conditionals can return values

[`scf.if`](../reference/dialects/scf/ops/if.md) consumes an `i1` condition. A
result-producing conditional has an `else` region, and each region yields the
declared results:

```loom
%below_floor = scalar.cmpf olt, %value, %floor : f32
%clamped = scf.if %below_floor -> (f32) {
  scf.yield %floor : f32
} else {
  scf.yield %value : f32
}
```

There is no mutable temporary shared by both branches. The yielded value is the
definition of `%clamped`, so use-def analysis and specialization see exactly
which alternatives can reach each consumer.

A resultless `scf.if` can omit the textual `scf.yield`; the parser still
materializes its terminator in memory. This keeps the in-memory region contract
uniform without cluttering the common source form.

## Counted loops carry state explicitly

[`scf.for`](../reference/dialects/scf/ops/for.md) uses a half-open logical index
range: the lower bound is inclusive, the upper bound is exclusive, and the step
is positive. Parenthesized bindings initialize loop-carried values, and
`scf.yield` forwards the next iteration state:

```loom
%result = scf.for %iteration = [%begin to %end step %step](%current = %initial : f32) -> (f32) {
  %next = func.call @affine(%current, %scale, %bias) : (f32, f32, f32) -> (f32)
  scf.yield %next : f32
}
```

The induction variable is an `index`. Every carried value has the same type as
its corresponding result. Multiple accumulators are ordinary parallel
bindings:

```loom
%sum, %sum_of_squares = scf.for %column = [%begin to %column_count step %step](
    %sum_acc = %zero : f32,
    %square_acc = %zero : f32
  ) -> (f32, f32) {
  %value = view.load %input[%column] : view<[%column_count]xf32> -> f32
  %square = scalar.mulf %value, %value : f32
  %next_sum = scalar.addf %sum_acc, %value : f32
  %next_square = scalar.addf %square_acc, %square : f32
  scf.yield %next_sum, %next_square : f32, f32
}
```

This spelling preserves recurrence edges. A scheduler can distinguish the two
independent recurrence chains, an unroller can interleave their producers, and
reports can attribute pressure to the values that actually stay live.

## Unrolling is a loop policy

Keep the source loop when the algorithm is a loop. Request full local unrolling
on the loop instead of pasting body copies:

```loom
%sum = scf.for %part = [%begin to %part_count step %step](%acc = %zero : f32) -> (f32) unroll {
  %next = func.call @accumulate_part(%acc, %part) : (f32, index) -> (f32)
  scf.yield %next : f32
}
```

The unroll transform proves and materializes the trip structure. If source facts
are insufficient, the request fails with the unresolved bounds rather than
silently leaving a loop that the author required to be unrolled. More specific
unroll schedules can express linear, interleaved, or recurrence-aware body
ordering without changing the logical loop.

## Select among whole values

Not every choice needs a region. [`scf.select`](../reference/dialects/scf/ops/select.md)
chooses between two whole values under one scalar condition. `scf.lookup`
selects a value tuple from a keyed table, and `scf.switch` owns multi-region
control flow when cases perform different work.

Use the smallest construct that preserves the program distinction:

- `scf.select` for two already-computed values;
- `scf.lookup` for a static keyed value table;
- `scf.if` when alternatives execute different operations;
- `scf.switch` for several operation regions;
- `scf.for` for bounded iteration;
- `scf.while` when continuation is computed by the loop itself.

The structured form is part of the optimization input. Lowering to branches is
a target decision, not source authoring work.

## Request a contract when identity is not fixed

An exact call says *which symbol*. A
[`func.apply`](../reference/dialects/func/ops/apply.md) says *which
implementation contract* and lets specialization select an eligible provider.

The composition example's motif contains a concrete helper and two providers
for one contract:

**Source:** [`loom/docs/examples/mental-model/motif.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/mental-model/motif.loom)

```loom title="motif.loom"
--8<-- "examples/mental-model/motif.loom"
```

The kernel requests that contract without naming either provider:

```loom
%result = func.apply<guide.transform>(%value) : (f32) -> (f32)
```

Provider selection uses the contract key, exact signature, available facts,
requirements, and explicit priority. File order and module path do not break
ties. The wave32 provider is eligible only when the application-site target
facts prove a subgroup size of 32; the portable provider remains the fallback.

After selection, the apply becomes an exact callable edge and ordinary inlining
can erase the boundary. Templates therefore express library variability at
specialization time without turning every device invocation into a runtime
branch.

The next guide boundary is buffers, views, and structured compute: functions
will continue to own reusable algorithms while storage identity and memory
access become explicit.
