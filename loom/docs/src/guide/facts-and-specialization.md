# Facts and specialization

Loom specializes from facts already present in the program. Constants, dynamic
shape bindings, configuration contracts, control-flow conditions, storage
assumptions, and target profiles all contribute knowledge to the same program
instead of disappearing into separate graph, kernel, and backend layers.

In this chapter, you will learn:

- where facts originate and how they follow SSA values;
- when to use configuration, workload arguments, assumptions, or executable
  assertions;
- how value predicates and target requirements select template providers;
- why unknown facts are different from false requirements; and
- how specialization removes alternatives without creating runtime dispatch.

## Facts are inputs to compilation

A fact is a compiler-visible statement about a value, storage root, execution
scope, or target version. Common fact categories include:

| Category | Examples |
| --- | --- |
| Scalar value | Exact constant, signed range, nonzero, power of two, known divisor, finite floating point. |
| Shape and representation | Dynamic dimension binding, element type, vector footprint, encoding, address layout. |
| Storage | Root identity, byte extent, alignment, memory space, alias scope. |
| Execution | Lane-varying or uniform value, workitem and workgroup topology domains. |
| Target | Subgroup size, resource limits, supported operations, ABI and artifact capabilities. |

Operations publish facts for their results and transfer facts from operands.
An exact constant stays exact through arithmetic that can be folded. Ranges and
divisibility propagate through index calculations. A view preserves its buffer
root and layout facts. Kernel coordinate queries carry their topology and
uniformity domains.

This analysis is always on. Authors add facts at boundaries where the compiler
cannot derive them; they do not manually annotate every intermediate result.

## Declare compile-time configuration

[`config.decl`](../reference/dialects/config/ops/decl.md) names a value that a
composition root must eventually supply. Its `where` clause constrains the
value even before an exact definition is available:

```loom
config.decl @model.hidden_size : %value: index where [range(%value, 256, 32768), mul(%value, 128)]

config.decl @kernel.workgroup_size : %value: index where [range(%value, 32, 256), pow2(%value)]
```

`range(%value, lo, hi)` is inclusive. `mul(%value, 128)` means that the value
is a multiple of 128; it does not perform multiplication. `pow2` records a
power-of-two contract. Relational predicates such as `lt`, `le`, `eq`, and
`ne` may relate a value to a literal or another SSA value.

[`config.get`](../reference/dialects/config/ops/get.md) makes the dependency an
ordinary symbol edge and exposes its typed facts in executable IR:

```loom
%hidden_size = config.get @model.hidden_size : index
%workgroup_size = config.get @kernel.workgroup_size : index
```

A source [`config.def`](../reference/dialects/config/ops/def.md), a `--config`
binding, a JSON/JSONC configuration object, or an embedding's `loomc` options
can provide the exact value. Materialization replaces the declaration with a
definition, validates its predicates, and lets ordinary canonicalization fold
dependent code.

Partial links may preserve reachable declarations. A final executable
compilation requires every reachable `config.get` to resolve to exactly one
definition. Bindings for keys that a selected root does not use are ignored,
so one model configuration bag can serve several smaller compiled programs.

The [linking workflow](../workflows/link-and-package.md) shows schema
inspection, individual bindings, and resolved-config enforcement.

## Keep per-launch values out of global configuration

Configuration describes a compiled artifact version. A kernel workload
describes one launch. The distinction matters when the same kernel appears
several times in one command program or accepts many shapes without
recompilation.

| Value changes when... | Source boundary |
| --- | --- |
| Building or selecting an artifact version | `config.decl` / `config.get`. |
| Evaluating one kernel launch | Kernel workload argument. |
| Issuing one device dispatch | Kernel launch argument. |
| Specializing one reusable command program | Command specialization argument. |
| Binding weights, cache, input, or output storage | Command buffer binding. |

A maximum supported row count can be configuration while the actual row count
remains a workload and device argument. This preserves one reusable artifact
and still gives the compiler a finite capacity for launch planning and bounds:

```loom
%row_capacity = config.get @model.row_capacity : index
%bounded_rows = index.assume %row_count [range(%row_count, 1, 65535), le(%row_count, %row_capacity)] : index
```

Moving every dynamic value into configuration creates global variants and
prevents one program from launching the same kernel with different workloads.
Keep a value dynamic unless changing it genuinely defines a new artifact or
specialization boundary.

## Refine the value that carries a proven contract

[`index.assume`](../reference/dialects/index/ops/assume.md) returns identity
aliases with stronger predicates:

```loom
%bounded_index = index.assume %element_index [range(%element_index, 0, 1048575), lt(%element_index, %element_count)] : index
```

The result equals the operand, but only `%bounded_index` carries the new fact
edge. Memory accesses, casts, loop transforms, and provider selection that need
the proof consume the refined value.

[`scalar.assume`](../reference/dialects/scalar/ops/assume.md) provides the same
contract for fixed-width integer and floating-point payloads:

```loom
%finite_scale = scalar.assume %scale [finite(%scale)] : f32
%aligned_length = scalar.assume %length [mul(%length, 16)] : i64
```

An assumption is not an executable check. It records a fact established by the
surrounding program, caller contract, or dominating condition. If runtime input
is untrusted, [`sanitizer.assert.value`](../reference/dialects/sanitizer/ops/assert-value.md)
checks the predicates, aborts the execution path on failure, and returns
refined aliases on success:

```loom
%checked_length = sanitizer.assert.value %length [range(%length, 0, 4096), mul(%length, 16)] : index
```

The distinction is semantic: `assume` makes a promise to the compiler;
`sanitizer.assert.value` enforces a promise at runtime.

## Control flow contributes path facts

A dominating structured condition can establish facts without repeating an
assumption. Template selection and later analysis evaluate an application site
in its control-flow context:

```loom
%tile_size_sixteen = index.constant 16 : index
%is_tile_size_sixteen = index.cmp eq, %tile_size, %tile_size_sixteen : index
%result = scf.if %is_tile_size_sixteen -> (i32) {
  %specialized = func.apply<guide.scale>(%tile_size, %value) : (index, i32) -> (i32)
  scf.yield %specialized : i32
} else {
  %fallback = func.apply<guide.scale>(%tile_size, %value) : (index, i32) -> (i32)
  scf.yield %fallback : i32
}
```

Inside the first region, `%tile_size == 16`; inside the second, it does not.
The condition controls runtime execution and simultaneously gives each
compile-time application site a stronger context. This lets a structured
program preserve the reason a specialization is valid instead of cloning a
function under an opaque generated name.

## A template provides an implementation contract

[`func.apply`](../reference/dialects/func/ops/apply.md) requests an
implementation by contract key. [`func.template`](../reference/dialects/func/ops/template.md)
provides one visible implementation whose signature and predicates may satisfy
that demand:

```loom
func.template<guide.scale> priority(20) @tile16_scale(%tile_size: index, %value: i32) -> (i32) where [eq(%tile_size, 16)] {
  %result = scalar.addi %value, %value : i32
  func.return %result : i32
}

func.template<guide.scale> priority(1) @portable_scale(%tile_size: index, %value: i32) -> (i32) {
  func.return %value : i32
}
```

An explicitly supplied library makes providers visible. For every reachable
apply, specialization evaluates candidates in this order:

1. Contract key and exact operand/result types must match.
2. Calling context and any explicit target identity must be compatible.
3. Value predicates in `where` must be proven at the application site.
4. Typed target requirements in `requires` must be proven by the active
   function-version target facts.
5. The highest-priority proven provider wins.

Selection rewrites the semantic demand to an inline exact call. Normal
inlining and dead-code elimination then remove the selected boundary and every
unreachable alternative. No per-invocation provider switch remains in the
device program.

Distinct providers tied at the best priority are ambiguous. File order,
library order, and symbol spelling never become accidental tie breakers.
Structurally equivalent duplicate providers may be coalesced because choosing
either cannot change the program.

## Requirements filter; they do not manufacture facts

A target requirement states when a provider is valid:

```loom
func.template<guide.transform> requires [#target.subgroup.size<32>] priority(20) @wave32_transform(%value: f32) -> (f32) {
  %result = func.call @double(%value) : (f32) -> (f32)
  func.return %result : f32
}
```

`requires [#target.subgroup.size<32>]` does not force wave32, select AMDGPU, or
teach the caller that subgroup size is 32. It filters this provider against the
facts of the function version containing the `func.apply`.

Use an exact `target(@symbol)` provider only when target identity itself is
part of the implementation contract. Most reusable providers stay targetless
and name the narrow normalized capability they need. The same requirement can
then match AMDGPU, SPIR-V, or another target family that establishes it.

The checked mental-model motif contains both the wave32 provider and its
portable fallback:

```loom title="motif.loom"
--8<-- "examples/mental-model/motif.loom"
```

## Unknown is not false

Target profiles may be exact, generic, partial, saved, synthetic, or projected
from a live device. A requirement can therefore be proven true, proven false,
or remain unknown.

| Candidate state | Early specialization | Final selection |
| --- | --- | --- |
| Requirement proven true | Candidate can be selected. | Candidate competes by priority. |
| Requirement proven false | Candidate is rejected. | Candidate is absent from the choice set. |
| Requirement unknown | A potentially better candidate remains live. | The best proven fallback may be selected; otherwise compilation diagnoses missing facts. |

This prevents an early generic link from prematurely erasing a provider that a
later device profile could prove valid. It also prevents final executable
lowering from carrying an unresolved semantic demand into target code.

Target facts attach to function versions rather than becoming one mutable
module-global mode. A compile invocation can specialize different entries for
different profiles while sharing the same linked libraries and targetless
helpers. The public [`loomc` target API](../reference/c-api/generated/target_8h.html)
defines target environments, immutable profiles, and per-function
specialization rows.

## Diagnose the missing fact, not the selected assembly

Provider selection emits structured report rows for selected, fallback,
rejected, ambiguous, and missing-fact outcomes. A blocked selection names the
contract, candidate, and unresolved value or target requirement. That is a
better repair boundary than inspecting final assembly and guessing why a
specialized implementation disappeared.

Typical mistakes expose the missing ownership decision:

| Symptom | Actual problem | Repair |
| --- | --- | --- |
| A specialized provider never matches | Its `where` predicate is not proven at the apply site. | Carry the relevant value or refined alias to the apply. |
| A target provider remains unknown | The active function version lacks the required target fact. | Supply a more exact profile or a portable fallback. |
| Two providers are ambiguous | They have equal priority and distinct implementations. | State the intended priority or make their predicates disjoint. |
| A fallback disappears during partial linking | The provider library was not an explicit input. | Declare and supply the library dependency. |
| The program recompiles for every shape | Per-launch workload was modeled as artifact configuration. | Move the value back to workload and launch signatures. |
| An assumption fixes a runtime failure | The source asserted an unverified promise. | Establish the fact with control flow or an executable assertion. |

[Read compile reports](../workflows/compile-reports.md) explains how to capture
details, inspect row-level JSON paths, and compare selection and emitted-code
evidence across configurations or targets.

Continue with checks and benchmarks: executable cases provide runtime values,
configuration bindings, expected results, and named performance rows without
creating a second representation of the program.
