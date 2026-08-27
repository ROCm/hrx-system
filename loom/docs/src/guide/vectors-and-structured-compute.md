# Vectors and structured compute

**Example files:** [`loom/docs/examples/guide/structured-compute/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/guide/structured-compute)

Loom vectors are typed SSA aggregates with a logical lane shape. They express
parallel arithmetic, coordinate construction, register rearrangement,
reductions, packed-format interpretation, and matrix contractions without
selecting an instruction set in source.

In this chapter, you will learn:

- why a vector shape is a computation contract rather than a hardware register
  declaration;
- how coordinates, masks, and lane-wise operations compose;
- when to use a reduction, a dot product, or a matrix fragment;
- how encoded carriers remain separate from their logical numeric values; and
- where structured computation ends and kernel distribution begins.

## Vector shape states logical work

The [`vector` type](../reference/types/vector.md) combines one or more lane
dimensions with a scalar element type:

```loom
vector<8xf32>
vector<4x8xbf16>
vector<[%lane_count]xi32>
```

Axes are ordered and lanes use logical row-major order. The shape participates
in verification and optimization: lane-wise operands agree in shape, a
transpose permutes axes, a reduction removes axes, and a grouped dot relates
source and accumulator extents.

The type does not promise one native instruction or one physical register. A
target may keep the value in one register group, split it into several native
operations, scalarize it, or eliminate it entirely. This distinction lets one
motif retain the algorithmic shape while several target providers choose its
physical implementation.

A dynamic vector dimension is still a compile-time lane count. It carries a
symbolic shape while templates and configuration remain unresolved, then
specialization must establish a concrete representation before target lowering.
Per-dispatch lengths belong in masks and loop bounds rather than in unresolved
physical vector extents.

## Build coordinates, masks, and data as ordinary values

Coordinate vectors and masks use the same SSA composition as numeric data. The
following function constructs eight logical coordinates, compares them
with a per-call limit, and selects scaled values for the active lanes:

**Source:** [`loom/docs/examples/guide/structured-compute/vector-values.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/structured-compute/vector-values.loom)

```loom
--8<-- "examples/guide/structured-compute/vector-values.loom"
```

[`vector.iota`](../reference/dialects/vector/ops/iota.md) uses a scalar base and
step to construct coordinates in logical lane order.
[`vector.splat`](../reference/dialects/vector/ops/splat.md) broadcasts one
scalar, [`vector.cmpi`](../reference/dialects/vector/ops/cmpi.md) compares lanes,
and [`vector.select`](../reference/dialects/vector/ops/select.md) combines the
two data paths. Each operation preserves the full vector shape unless its
contract explicitly changes it.

[`vector.mask.range`](../reference/dialects/vector/ops/mask-range.md) is a
shorter form when the mask is simply an inclusive-lower, exclusive-upper
coordinate interval:

```loom
%mask = vector.mask.range [%base to %limit step %step] : index -> vector<8xi1>
```

A selected value and a masked memory operation answer different questions.
`vector.select` chooses already-computed lane values; it does not undo a load
that happened earlier. A false lane on
[`vector.load.mask`](../reference/dialects/vector/ops/load-mask.md) suppresses
the memory access itself. Tail memory safety therefore belongs at the memory
operation or in control flow that proves the access unreachable.

## Keep lane-wise intent visible

Arithmetic, comparisons, casts, and math operations apply lane by lane. Their
names parallel the scalar dialect so the source makes numeric intent explicit:

```loom
%biases = vector.splat %bias : vector<8xf32>
%shifted = vector.addf %values, %biases : vector<8xf32>
%activated = vector.geluf<tanh> %shifted : vector<8xf32>
%narrowed = vector.fptrunc %activated : vector<8xf32> to vector<8xf16>
```

Integer signedness lives on the operation whose semantics need it. `vector.cmpi
slt` and `vector.cmpi ult` consume the same integer vector types but state
different orderings. Sign and zero extension similarly use `vector.extsi` and
`vector.extui`. This keeps physical carrier types reusable without hiding
numeric interpretation in a typedef or target convention.

Floating-point flags are permissions, not optimization levels. For example,
`reassoc` permits reassociation and `contract` permits contraction where the
participating operations carry compatible contracts. Omitting a flag preserves
the stricter operation semantics; adding a flag is an author-visible change to
the numeric contract.

## Rearrange registers without rewriting memory

Vector aggregate operations describe logical register layout changes:

```loom
%tile = vector.slice %matrix[0, 4] : vector<8x16xf32> -> vector<4x8xf32>
%transposed = vector.transpose<[1, 0]> %tile : vector<4x8xf32> -> vector<8x4xf32>
%even, %odd = vector.deinterleave<0> %lanes : vector<16xi8> -> vector<8xi8>, vector<8xi8>
```

[`vector.slice`](../reference/dialects/vector/ops/slice.md),
[`vector.concat`](../reference/dialects/vector/ops/concat.md),
[`vector.transpose`](../reference/dialects/vector/ops/transpose.md), shuffle,
interleave, and deinterleave operate on SSA values. They do not change a view's
address layout and do not move data between memory spaces. When the intended
operation is a tiled memory projection, construct a subview or use a structured
vector transfer instead.

The distinction gives lowering freedom. A transpose may become a register
permutation, disappear into the consumers' operand forms, or influence the
selected load layout without changing its source semantics.

## Choose the contraction that states the numeric contract

A reduction combines lanes; a dot product additionally states how products and
accumulation relate. These two example functions deliberately make that
difference visible:

**Source:** [`loom/docs/examples/guide/structured-compute/reductions.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/structured-compute/reductions.loom)

```loom
--8<-- "examples/guide/structured-compute/reductions.loom"
```

[`vector.reduce`](../reference/dialects/vector/ops/reduce.md) reduces every lane
into a scalar seed. [`vector.reduce.axes`](../reference/dialects/vector/ops/reduce-axes.md)
reduces selected axes and preserves the others. An explicit `vector.mulf`
followed by `vector.reduce<addf>` retains separately stated product and sum
operations.

[`vector.dotf`](../reference/dialects/vector/ops/dotf.md) instead defines one
floating-point dot accumulation equivalent to a sequence of fused multiply-add
terms. Grouped register contractions such as `vector.dot2f`, `vector.dot4i`,
`vector.dot8i4`, and `vector.dot4f8` make source grouping, packed field
interpretation, and accumulator shape explicit. A target may select a native
dot instruction when its provider satisfies that exact contract.

The initial accumulator is never implicit. It defines the result type, gives
zero-lane reductions a value, and makes accumulation into an existing partial
result an ordinary data dependency.

## Separate physical carriers from numeric interpretation

Packed model formats combine physical payload words with schemas, scales,
codebooks, zero points, or sparse metadata. Loom preserves those pieces as
separate SSA values instead of erasing them behind ad hoc unpacking code. This
GGML Q4_0 function carries four `i32` payload registers while producing 32
logical f32 values:

**Source:** [`loom/docs/examples/guide/structured-compute/encoded-values.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/structured-compute/encoded-values.loom)

```loom
--8<-- "examples/guide/structured-compute/encoded-values.loom"
```

[`encoding.define`](../reference/dialects/encoding/ops/define.md) materializes a
compact schema witness. Here `#ggml.q4_0` names the complete schema without
repeating its physical fields at every use. [`vector.decode`](../reference/dialects/vector/ops/decode.md)
combines the physical payload, that schema, and keyed auxiliary values at the
numeric interpretation boundary. [`vector.encode`](../reference/dialects/vector/ops/encode.md)
states the inverse direction.

The schema is reusable compile-time structure; scales and other bulk data
remain normal vector operands. A format library can therefore provide GGML,
MX, block-scaled, or application-specific carrier motifs without imposing a
kernel ABI. Kernels using the same stored format may decode into different
logical element types or fuse the interpretation directly into a contraction.

## Matrix fragments attach roles and logical shape

A matrix fragment is a physical vector value refined with a logical role and
matrix dimensions. Attaching fragment facts does not copy or numerically
convert the carrier:

**Source:** [`loom/docs/examples/guide/structured-compute/matrix-fragments.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/guide/structured-compute/matrix-fragments.loom)

```loom
--8<-- "examples/guide/structured-compute/matrix-fragments.loom"
```

[`vector.fragment`](../reference/dialects/vector/ops/fragment.md) marks carriers
as left-hand, right-hand, initial-accumulator, or result fragments. Logical M,
N, and K dimensions remain ordinary SSA values and may be specialized from
configuration and target facts. Encoded fragments may also carry a schema,
scale, table, or sparse metadata in the keyed `using` dictionary.

[`vector.mma`](../reference/dialects/vector/ops/mma.md) consumes the fragment
facts rather than spelling an AMDGPU, SPIR-V, or other target instruction in
source. Provider selection can choose a native matrix operation, a carrier
repack, or a valid decomposition while preserving the same contraction
contract. If no implementation satisfies the fragment shape and numeric
contract, compilation fails at the selection boundary instead of silently
changing the computation.

Fragment roles are not general-purpose layout casts. Use
[`vector.fragment.repack`](../reference/dialects/vector/ops/fragment-repack.md)
when an algorithm intentionally reinterprets or converts one native fragment
role into another and the target supplies that transformation.

## Keep compute reusable until distribution is required

Structured compute belongs in the narrowest boundary that owns it:

| Concern | Owning source construct |
| --- | --- |
| Decode a block format, reduce lanes, or compute one logical tile | `func.def` or `template.def` motif. |
| Map work across workgroups and workitems, access buffers, and synchronize | `kernel.def`. |
| Sequence launches and bind reusable resources | `command.program.def`. |
| Choose a target-specific implementation of a shared contract | Template provider plus target requirements. |

This separation is what lets a format decoder feed a scalar kernel, a vector
dot kernel, or a matrix-fragment kernel without cloning its ABI and launch
policy. The kernel chooses distribution and transfer shapes; the motif retains
the computation and representation contract it can actually own.

## Diagnose structure at the boundary that lost it

| Symptom | Contract to inspect |
| --- | --- |
| A dynamic vector cannot lower | The composition root did not specialize a physical lane extent. |
| A tail still reads out of bounds | Selection masked a value after an unmasked memory access. |
| A transpose changes the wrong data | Register rearrangement was confused with a view or memory layout. |
| A dot result differs after optimization | Fast-math permissions do not match the intended rounding and contraction contract. |
| Packed values have the right bits but wrong numbers | Signedness, schema, scale, or auxiliary encoding facts are missing or incorrect. |
| Matrix lowering finds no provider | Fragment roles, logical M/N/K shape, carrier types, and target facts do not describe a supported contraction. |

Continue with [Kernels and launch configuration](kernels-and-launch.md), where
reusable structured computation gains explicit workload, distribution, and
device ABI contracts.
