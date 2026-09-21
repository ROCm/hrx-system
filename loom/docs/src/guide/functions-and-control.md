# Functions and structured control flow

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

Functions are typed callables that can execute on the host or device. Structured
control flow keeps conditions, iteration domains, and carried values visible to
verification, specialization, unrolling, distribution, and lowering.

In this chapter, you will learn:

- how definitions, declarations, exact calls, and returns compose;
- how function modifiers state visibility, purity, placement, and inline policy;
- how authored Low helpers control target instructions and their order;
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

Function arguments keep their initial values throughout an invocation. A
dependent result such as `vector<[%width]xf32>` names the width supplied by the
caller, and argument predicates describe those same inputs. Structured loops
bind separate carried values. When authoring CFG directly, a separate loop
header owns the changing block arguments; branches cannot target the function
entry block. This contract also applies to template, kernel, and Low bodies.

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

[`low.invoke`](../reference/dialects/low/ops/invoke.md) lets a source function
delegate part of its device implementation to an authored `low.func.def`.
A helper can express a target instruction sequence while indexing, launch
geometry, and the surrounding algorithm remain in source IR. On AMDGPU and
x86, the compiler inlines these helpers before emission, and their instructions
normally participate in scheduling and register allocation with the caller.
The call's inline policy follows the same rules as other callable boundaries;
`noinline` is rejected on targets that require Low calls to inline.

The call site retains source types while the helper signature uses target-Low
register types. Physical register classes are carrier-only: source semantic
types are represented by the helper's predicates and facts rather than a
second type inside the physical register carrier.

```loom
amdgpu.target<gfx11-generic> @schedule_target

low.func.def target<amdgpu.gfx11.generic.core>(@schedule_target) @pack_pair(%even: reg<amdgpu.vgpr>, %odd: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) asm {
  %selector = s_mov_b32 0x05040100
  %packed = v_perm_b32 %odd, %even, %selector
  return %packed
}

// %even and %odd are produced by ordinary source-level vector loads.
%packed = low.invoke @pack_pair(%even, %odd) : (vector<2xf16>, vector<2xf16>) -> (vector<2xf16>)
```

Lowering proves that each source operand and result maps exactly to the
helper's register signature. Inlined helpers use virtual register allocation
and receive inputs through arguments rather than `low.resource` or
`low.live_in`. They may contain multiple control-flow blocks and nested Low
calls. Each nested call must satisfy its own signature, target, and inline
contracts; every returning path supplies the helper's declared results.

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
call-site values as `low.assume` identities. The assumption therefore reifies
an established fact for downstream local analysis; it never
creates a fact needed to justify the call. This is the explicit source-to-Low
fact bridge for scalar dimensions, indices, and similar register arguments. A
helper states every fact its implementation requires in its `where` clause;
`low.invoke` does not serialize the caller's analysis table or turn
opportunistically inferred facts into hidden callee assumptions.

### Control instruction order

Authored Low uses a free schedule by default: SSA dependencies, memory effects,
and target hazards constrain instruction placement, while independent work can
move. This lets the compiler interleave a helper's instructions with its caller.
Explicit scheduling controls let an author further constrain that placement.

[`low.schedule.fence`](../reference/dialects/low/ops/schedule-fence.md) separates
reorderable ranges. Instructions before the fence stay before instructions
after it; instructions within either range remain free to move. For example,
this helper places two loads ahead of the arithmetic that consumes one:

```loom
amdgpu.target<gfx11-generic> @schedule_target

low.func.def target<amdgpu.gfx11.generic.core>(@schedule_target) @load_ahead_of_scale(%byte_offset: reg<amdgpu.vgpr>, %base: reg<amdgpu.sgpr x2>, %weight: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) asm {
  %current = global_load_b32_saddr %byte_offset, %base
  %future = global_load_b32_saddr %byte_offset, %base {offset = 4}
  low.schedule.fence
  %scaled = v_mul_f32 %current, %weight
  return %scaled, %future
}
```

The fence emits no instruction and performs no memory wait. The multiply still
needs `%current`, so the compiler inserts any required operand-readiness wait;
the fence itself does not require `%future` to complete.
[`scf.schedule.fence`](../reference/dialects/scf/ops/schedule-fence.md) supplies
the corresponding boundary in source IR and composes with
[scheduled unrolling](#preserve-source-order-through-unrolling).

`schedule(locked)` on a Low function or kernel preserves authored order within
each block under every scheduling strategy. Here the second load stays after
the multiply even though its address is already available:

```loom
amdgpu.target<gfx11-generic> @schedule_target

low.func.def schedule(locked) target<amdgpu.gfx11.generic.core>(@schedule_target) @scale_then_load(%byte_offset: reg<amdgpu.vgpr>, %base: reg<amdgpu.sgpr x2>, %weight: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) asm {
  %current = global_load_b32_saddr %byte_offset, %base
  %scaled = v_mul_f32 %current, %weight
  %future = global_load_b32_saddr %byte_offset, %base {offset = 4}
  return %scaled, %future
}
```

This contract also survives inlining through `low.invoke`. Inlining preserves
each block's order and keeps surrounding instructions outside the locked
sequence. Locked helpers can have control-flow blocks; operations with nested
regions inside a locked helper are rejected. Register allocation remains
virtual unless separately constrained, and required copies and hazard waits
still apply. Separately locked AMDGPU instructions keep separate issue slots
instead of being combined into a dual-issue instruction.

These controls govern compiler instruction order. They do not synchronize
lanes, publish shared-memory writes, or establish completion before storage
reuse. Native AMDGPU and x86 emission preserve the requested order; hardware
may still execute independent work concurrently or out of order. Intermediate
artifacts such as SPIR-V pass through another compiler, so their textual order
does not establish final native instruction order.

The [checked WMMA example](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/tooling/target/amdgpu/test/corpus/gfx11/low_schedule.loom)
compares free, fenced, locked, and explicitly overlapped two-tile schedules.
After downloading `low_schedule.loom`, run its four correctness cases on an
AMDGPU device supporting the example's GFX11 WMMA instructions:

```shell
iree-test-loom low_schedule.loom --device=amdgpu
```

Numerical checks establish that every schedule consumes the intended tiles.
[Native-code and compile-report inspection](../workflows/tune-loop-schedules.md#check-that-read-ahead-survives-native-code-generation)
then establishes which loads remain pending, where waits occur, and how the
chosen order changes register pressure.

### Use flat pointers in authored AMDGPU helpers

An AMDGPU flat pointer can address global or workgroup memory at runtime,
including different spaces in different lanes of the same wave. The compiler
tracks both completion domains for flat loads, stores, and atomics. Required
waits protect consumers, register reuse, and workgroup publication in authored
Low helpers, including `schedule(locked)` helpers inlined with `low.invoke`.

To form a flat pointer from an LDS byte offset, read the shared aperture with
`s_mov_b64_shared_base` and use its upper word. The 32-bit hardware read returns
zero and cannot supply this address. For example, inside a GFX11 Low helper:

```loom
%aperture = s_mov_b64_shared_base
%shared_base = slice %aperture[1] : reg<amdgpu.sgpr x2> -> reg<amdgpu.sgpr>
%upper = v_mov_b32_copy %shared_base
%address = concat(%local_byte_offset, %upper) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>
%value = flat_load_dword %address
```

Here `%local_byte_offset` includes the workgroup allocation's base and the
lane's offset. The [checked flat-pointer example](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/tooling/target/amdgpu/test/corpus/gfx11/flat_aperture.loom)
passes a `buffer.alloca<workgroup>` allocation into a Low helper, exercises
returning and non-returning atomics, and selects global or LDS addresses per
lane. Download it and run `iree-test-loom flat_aperture.loom --device=amdgpu`
on a GFX11 device.

Flat completion differs across families. GFX9 can report early completion in
the unused address domain, so consuming a pending flat result requires full
drains. GFX11 and GFX12 can retain younger requests with partial waits in both
domains. [Inspect the native waits](../workflows/tune-loop-schedules.md#check-that-read-ahead-survives-native-code-generation)
when evaluating how much of an authored schedule remains asynchronous.

### Compose independently scheduled helpers

A motif author can give a Low helper `schedule(phased)` and use
[`low.schedule.phase`](../reference/dialects/low/ops/schedule-phase.md) to begin
each subsequent phase. The first phase starts at function entry. Every
invocation owns its phase order, while instructions within a phase and work
from independent invocations remain free to interleave. This helper keeps its
multiply ahead of its second load:

```loom
amdgpu.target<gfx11-generic> @schedule_target

low.func.def schedule(phased) target<amdgpu.gfx11.generic.core>(@schedule_target) @scale_pair(%byte_offset: reg<amdgpu.vgpr>, %base: reg<amdgpu.sgpr x2>, %weight: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) asm {
  %first = global_load_b32_saddr %byte_offset, %base
  low.schedule.phase
  %scaled = v_mul_f32 %first, %weight
  low.schedule.phase
  %second = global_load_b32_saddr %byte_offset, %base {offset = 4}
  %result = v_add_f32 %scaled, %second
  return %result
}
```

Values keep their ordinary SSA scope: `%first` and `%scaled` remain available
after their phase boundaries without block arguments or region results. The
function owns the schedule boundary and closes it on every return.

The caller uses ordinary `low.invoke @scale_pair(...)` calls. Two calls create
two independent scopes: each preserves `first load → multiply → second load`,
while the compiler can issue another instance's load during the first
instance's computation. Inlining and full or partial unrolling preserve this
independence. The motif author chooses the separators; `scf.for pipeline(...)`
does not assign native phases automatically.

Expanded Low represents each inlined invocation with
[`low.schedule.begin`](../reference/dialects/low/ops/schedule-begin.md) and
[`low.schedule.end`](../reference/dialects/low/ops/schedule-end.md). These controls
also support explicitly authored subscopes. A nested scope belongs to its
parent's current phase, and empty phases are valid. Every reachable join and
loop backedge must agree on explicit nesting, and every exit must close it.
An explicit end cannot close the implicit function scope.

`schedule(phased)` also applies to directly authored Low kernels. Phase
separators appear in Low function or kernel body blocks; the enclosing scope
may span CFG edges introduced by inlining. Source loops and conditionals can
invoke phased Low helpers. Controls written directly inside nested Low
structured regions are rejected.

Scopes order surviving emitted instructions and emit no runtime instruction.
They do not wait for loads, synchronize lanes, or publish memory. Full fences
still order the whole instruction stream, and `schedule(locked)` still fixes
block order. Native AMDGPU and x86 support scopes; SPIR-V, Wasm, and VM
representations reject the native-order contract explicitly.

The [checked paired-matrix example](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/tooling/target/amdgpu/test/corpus/gfx11/schedule_scopes.loom)
invokes a two-tile WMMA helper twice with distinct inputs and expected results.
Its compile report exposes the materialized scope count alongside register use
and partial waits. [Scope report queries](../workflows/compile-report-queries.md#inspect-authored-scheduling-scopes)
help compare this composition with a fully fenced or manually interleaved
schedule. More overlap can keep more registers live; native evidence and
workload measurements determine whether that trade is useful.

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

The induction variable has the bounds' address type: `index` for logical
coordinates or `offset` for byte positions. If the lower bound is at least the
upper bound, the results are the initial values. Otherwise, they are the last
iteration's yielded values. A nonunit step visits only values below the upper
bound: `[3 to 10 step 4]` visits `3` and `7`.

Every carried value has the same type as its corresponding result. Multiple
accumulators are ordinary parallel bindings:

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

## Rotate views over reusable storage

Loop-carried state can include views over one buffer. Construct the slots from
the same buffer value, carrying any alignment or alias assumptions established
by the caller. Each view retains its shape and layout while its byte origin
rotates through the loop. The backing allocation remains outside the loop.

The yield tuple supplies the next state simultaneously. For example,
`scf.yield %next, %current` exchanges the two views: each operand denotes the
view from the current iteration. It does not copy their stored elements.

This example uses two private slots to compute `1 + 7 * trip_count`. Each
iteration consumes the current slot, writes the next slot, and rotates their
roles. The caller supplies a trip count in `[0, 257]`:

**Source:** [`rotating-views.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/functions-and-control/rotating-views.loom)

```loom title="rotating-views.loom"
--8<-- "examples/guide/functions-and-control/rotating-views.loom"
```

With zero iterations, `%last` is the initial `%first` view and the result is
`1`. After three iterations, `%last` identifies the last written slot and the
result is `22`. The returned views remain usable after the loop; they retain
the lifetime of their backing storage. A view is non-owning, so returning a
view never extends an allocation's lifetime.

Views can also pass through `scf.while`, region yields, and whole-value
selection. The [memory guide](buffers-views-memory.md) describes their storage,
alignment, and synchronization contracts. Rotating a view does not establish
completion of an asynchronous producer; consumption and storage reuse follow
the program's completion dependencies.

Scalar values can rotate through the same carried tuple. A loop may issue a
future input load before computing with an older carried input. On AMDGPU,
completion waits account for register copies used to carry those values: a
copied input is ready in the next iteration, while a value forwarded in place
retains its pending dependency. The future load can remain outstanding during
independent arithmetic, then completes before its result is read or copied.

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

`unroll(%factor)` requests partial unrolling with a specialized positive factor.
The two-slot example uses factor two with a runtime trip count. Each copied
iteration advances the whole carried tuple once. Empty ranges execute no body,
and a remainder executes only the iterations in the original half-open range:
three iterations still produce `22`, including the final view identity.
Unrolling preserves the program's data and completion dependencies.

The [loop-tuning walkthrough](../workflows/tune-loop-schedules.md) combines
these controls in checked row-sum and packed-dot examples, including serial
controls, configuration sweeps, and compiler evidence.

### Preserve source order through unrolling

Source fences compose with full and partial unrolling, including `interleaved`
and `recurrence` schedules. This boundary keeps a carried add ahead of an
independent ID-table lookup and its dependent payload. Here `%begin` is zero,
`%step` is one, `%end` is in `[0, 8]`, and the ID table contains indices in
`[0, 7]`:

```loom
%result = scf.for %row = [%begin to %end step %step](%sum = %initial : f32) -> (f32) unroll(%factor) schedule(recurrence) {
  %consumed = scalar.addf %sum, %bias : f32
  scf.schedule.fence
  %id = view.load %ids[%row] : view<8xi32> -> i32
  %position = index.cast %id : i32 to index
  %bounded = index.assume %position [range(%position, 0, 7)] : index
  %payload = view.load %payloads[%bounded] : view<8xf32> -> f32
  %next = scalar.addf %consumed, %payload : f32
  scf.yield %next : f32
}
```

`%factor` can be supplied by the surrounding template or computed from target
properties, just like an unfenced loop's unroll factor. Each materialized
iteration retains the boundary, and its unfenced ranges remain available for
scheduling. A fence inside an inlined helper participates in the caller's
source order. A fence inside a nested `scf.if` or `scf.for` retains that region's
scope when the enclosing operation is cloned as a unit.

The [checked dependent-lookup example](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/tooling/target/amdgpu/test/amdgpu_unroll_fences.loom)
composes two helpers and exercises empty, short, full, and remainder tiles.
A loop requesting `pipeline(%depth)` with depth greater than one rejects
authored fences: the read-ahead transform cannot carry an explicit source-range
constraint through its producer/consumer partition.

## Pipeline reads ahead of ordered computation

`pipeline(%depth)` requests read-ahead on one `scf.for`. The loop still describes
one logical iteration. Ordinary loads and their address prerequisites run ahead
of the ordered consumer; the compiler constructs startup, steady-state, and
drain code and carries the values between them.

This motif sums up to 64 rows of four-element vectors from four-byte-aligned
buffers. Its depth and unroll factor are template arguments: each caller owns
its schedule, and multiple instantiations can use different policies in one
kernel. The caller here chooses depth four and factor six.

**Source:** [`vector-read-ahead.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/functions-and-control/vector-read-ahead.loom)

```loom title="vector-read-ahead.loom"
--8<-- "examples/guide/functions-and-control/vector-read-ahead.loom"
```

Depth four keeps three iterations of loaded values queued. Each steady iteration
fetches the next input and consumes the oldest queued value. The drain
consumes the final three values in their original order. Empty loops perform no
loads; loops shorter than the depth take the serial path. A partial unroll
remainder also stays within the original half-open range. The accumulation order
is preserved, including for floating-point recurrences.

The clauses follow transformation order: `pipeline(%depth) unroll(%factor)`.
Pipelining constructs the queue first; unrolling then groups iterations of the
reconstructed loops. Depth three with unroll factor two still queues two original
iterations, and each unrolled steady body advances the queue twice. An optional
`schedule(...)` follows `unroll(...)` and controls how those copies are ordered.
The target schedules independent instructions using its normal dependency
constraints; the carried queue preserves the original iteration relationship.

The example uses factor six with `schedule(recurrence)` so old queue values
can be consumed before their registers receive new loads. Hardware overlap
also depends on allocation: a materialized queue copy consumes its source and
can require an early wait. The [native-overlap walkthrough](../workflows/tune-loop-schedules.md#check-that-read-ahead-survives-native-code-generation)
shows a steady backedge with pending loads and how to inspect copy waits.

Both controls are independent and explicit: loops without `pipeline(...)`
receive no read-ahead transformation, and pipelining does not request unrolling.
Depth one retains serial iteration and any separate unroll policy, providing a
useful control. The depth must specialize to a positive exact value; the unroll
factor must also specialize before its policy runs.

Nested `scf.if` and `scf.for` remain intact within their assigned stage.
A guarded load retains its guard, and a read-only inner reduction can produce
one queued result for each outer iteration. A pure inner loop can consume
queued values and the outer accumulator. Inner loops may also carry their own
explicit pipeline and unroll policies; pipelining processes the inner policy
before the outer one, then unrolling processes the reconstructed program.
The [guarded-row example](../workflows/tune-loop-schedules.md#keep-guards-and-inner-loops-in-the-source)
checks these combinations through native execution.

The read-ahead contract supports ordinary loads and memory-pure consumers with a
positive exact step. Read prerequisites may depend on the induction variable
and values outside the loop. For a nested unit containing reads, this includes
all captured values, guards, bounds, and initial inner state: the whole unit
must be independent of outer loop-carried state. A violation is diagnosed.
Cross-stage value types must be invariant across iterations. Writes, ordered
or volatile effects, explicit asynchronous groups, other nested control such
as `scf.while`, and consuming result-storage ties on the requested loop require
a different scheduling/ownership contract and are diagnosed when
requested at depth greater than one. The depth is bounded by 65,535 and by the
representable carried-state tuple; unsupported requests fail at the source
policy instead of silently running serially.

Subgroup and workgroup reductions can remain in the ordered consumer when the
requested loop's lower and upper bounds are compile-time exact. This keeps
every original participant at the same collective site during startup, steady
iteration, and drain. A fixed tile with runtime tail guards fits this contract:
place guarded loads in one `scf.if` and the guarded reduction in a separate
consumer `scf.if`. Each nested region is one scheduling unit, so a region mixing
loads and collectives cannot advance. A collective result also cannot determine
a read-ahead address or guard. Runtime bounds on the requested collective loop
receive a diagnostic; depth one preserves the original loop.

The [checked collective recurrence](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/test/corpus/conformance/collective_loop_state.loom)
applies one template with serial and pipelined policies. It combines 16-lane
cluster reductions, workgroup reductions, ragged reads, and a nested runtime
consumer loop, checking each schedule against independent integer results.
The [cooperative paged-attention workflow](../workflows/tune-loop-schedules.md#pipeline-cooperative-paged-attention)
applies this shape to shared K/V page lookups, subgroup score reductions, online
softmax state and ragged sequence tails, with matched checked benchmarks.
The [sparse token variant](../workflows/tune-loop-schedules.md#pipeline-sparse-token-attention)
adds dependent per-token IDs and an independent selected-prefix boundary,
including ignored suffixes that point at poisoned payloads.
The [grouped-query variant](../workflows/tune-loop-schedules.md#share-kv-loads-across-query-heads)
shares each K/V fragment across two independent softmax states. It compares
load reuse and pipelining separately, with per-head guards for unequal lengths.

The caller can also calculate these values with `index` arithmetic from
specialized template arguments or target properties. A global configuration
key for a motif's depth would couple all its instantiations. Config overrides
are useful in an experiment harness for sweeping schedules; the
[loop-tuning walkthrough](../workflows/tune-loop-schedules.md#keep-one-checked-source-for-experiments)
shows that separate workflow.

Compile this instantiation and inspect its schedule:

```shell
loom-compile vector-read-ahead.loom --root=@sum_vector_rows \
  --target=amdgpu:gfx1151 --format=amdgpu-hsaco --output=vector-rows.hsaco \
  --compile-report=details --compile-report-output=vector-rows.report.json
loom-compile-report show vector-rows.report.json
loom-compile-report suggest vector-rows.report.json
```

The report retains the applied policy and operation schedule alongside final
registers, spills, occupancy, and code size. A larger queue trades additional
live state and generated code for read overlap; the useful depth depends on the
kernel and target. [Compile reports](../workflows/compile-reports.md) explain
how to compare those costs with runtime measurements while holding unrolling
and workload fixed.

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
