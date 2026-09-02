# Functions and structured control flow

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

Functions are typed callables that can execute on the host or device. Structured
control flow keeps conditions, iteration domains, and carried values visible to
verification, specialization, unrolling, distribution, and lowering.

In this chapter, you will learn:

- how definitions, declarations, exact calls, and returns compose;
- how function modifiers state visibility, purity, placement, and inline policy;
- how `scf.if` and `scf.for` produce SSA values;
- why loop-carried state is explicit; and
- when to request an implementation from a template family instead of naming
  one function.

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

The linker or embedding constructs the explicit library universe. A library
definition must be exported to satisfy another module's declaration, and two
eligible definitions with the same symbol are an error rather than an
input-order choice.

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

## Invoke an authored Low fragment

`low.invoke` is the explicit boundary for a source function that delegates a
value transformation to an authored `low.func.def`. It is a narrow migration
boundary for lifting a reverse-engineered instruction fragment while loads,
stores, launch geometry, and the surrounding algorithm move into source IR.
The source-to-Low pipeline always inlines the helper; it does not introduce a
runtime call or leave the boundary for a backend to interpret. The resulting
packets normally participate in scheduling and allocation with the caller.

The call site retains source types while the helper signature uses target-Low
register types. Physical register classes are carrier-only: source semantic
types are represented by the helper's predicates and facts rather than a
second type inside the physical register carrier.

```loom
amdgpu.target<gfx11-generic> @schedule_target

low.func.def target<amdgpu.gfx11.generic.core>(@schedule_target)
    @pack_pair(%even: reg<amdgpu.vgpr>, %odd: reg<amdgpu.vgpr>)
    -> (reg<amdgpu.vgpr>) asm {
  %selector = s_mov_b32 0x05040100
  %packed = v_perm_b32 %odd, %even, %selector
  return %packed
}

// %even and %odd are produced by ordinary source-level vector loads.
%packed = low.invoke @pack_pair(%even, %odd)
    : (vector<2xf16>, vector<2xf16>) -> (vector<2xf16>)
```

Lowering proves that each source operand and result maps exactly to the
helper's register signature. The helper has virtual register allocation, one
outer body block ending in `low.return`, and no function-entry resources or
live-ins. A nested call has its own target and representation boundary and is
therefore rejected instead of escaping into the caller accidentally.

Target compatibility is directional. A concrete `gfx1151` caller can invoke a
helper authored for `gfx11-generic` because the concrete target satisfies the
generic requirement. Lowering then projects the helper's register classes and
instruction descriptors by stable identity into the caller's
`amdgpu.rdna3_5.core` representation. A generic caller cannot invoke an
exact-only `gfx1151` helper. A helper may omit its target witness when only the
representation contract matters; the caller then supplies the target facts and
must still use a target contract supported by that representation.

Value-fact analysis remains local across this boundary. Source function
argument predicates are remapped onto the generated Low function, while facts
expressed by emitted Low producers are recomputed with the cloned operations.
The helper's authored argument predicates are preconditions: lowering resolves
each predicate against the call-site operands and proves it from caller-visible
facts. An unknown or contradicted precondition rejects the invocation. Only
after every precondition is proven does lowering remap the predicates to the
call-site values as `low.assume` identities and clone the body. The assumption
therefore reifies an established fact for downstream local analysis; it never
creates a fact needed to justify the call. This is the explicit source-to-Low
fact bridge for scalar dimensions, indices, and similar register arguments. A
helper states every fact its implementation requires in its `where` clause;
`low.invoke` does not serialize the caller's analysis table or turn
opportunistically inferred facts into hidden callee assumptions.

The schedule-free form is the ordinary `low.invoke` contract. It keeps SSA and
target instruction constraints while allowing the scheduler to place the
inlined packets among surrounding operations. This is also the useful boundary
for incrementally replacing an oracle fragment with higher-level source.

`schedule(locked)` remains an experimental escape hatch when exact source order
is itself part of an oracle. Inlining conservatively surrounds every authored
operation with source-order scheduling boundaries, preventing surrounding
source operations from interleaving with the fragment. Locked helpers are
straight-line; nested regions are rejected because preserving only their outer
position would not preserve their internal schedule. This heavy constraint is
not evidence that a recovered fragment is ready to become maintained source.

Acceptance uses the same default compiler pipeline as maintained Loom source
and executes through `iree-test-loom`. Direct execution of prepared Low can be
useful while reconstructing a schedule, but it establishes an oracle rather
than proving that the maintained source survives compilation.

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

## Request a template family when identity is not fixed

An exact call says *which symbol*. A
[`template.apply`](../reference/dialects/template/ops/apply.md) says *which
template family* and lets specialization select an eligible provider.

The composition example's motif contains a concrete helper and two providers
for one family:

**Source:** [`loom/docs/examples/elementwise-transform/motif.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/motif.loom)

```loom title="motif.loom"
--8<-- "examples/elementwise-transform/motif.loom"
```

The kernel requests that family without naming either provider:

```loom
%result = template.apply<@guide.elementwise_transform>(%value) : (f32) -> (f32)
```

Provider selection uses the family symbol, exact signature, available facts,
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
