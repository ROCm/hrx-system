# Values, types, and shapes

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

Loom IR is statically typed SSA. Every operation names the values it consumes,
the values it produces, and the types at that boundary. The type system keeps
logical coordinates, physical byte addresses, register values, logical
aggregates, and aliasable storage distinct long enough for specialization and
lowering to use the distinction.

In this chapter, you will learn:

- which scalar domain owns coordinates, byte offsets, and fixed-width data;
- how static and dynamic dimensions appear in shaped types;
- why tensors, tiles, vectors, buffers, and views are separate types;
- how encodings describe physical representation without changing logical
  shape; and
- how refined SSA values carry facts forward.

## SSA values make dataflow explicit

An SSA value is defined once. Its name communicates source intent; its type is
the compiler contract. Operations repeat the relevant type at their boundary so
a reader can understand a local expression without recovering a hidden ambient
type.

```loom
%row = index.constant 7 : index
%row_stride = index.constant 4096 : offset
%row_byte_offset = index.scale %row, %row_stride : index, offset -> offset
```

`%row` is a logical coordinate. `%row_stride` and `%row_byte_offset` are
physical byte quantities. The result name explains its role, while the
`offset` type prevents later code from accidentally treating it as another
logical dimension.

There are no implicit numeric casts. A conversion is an operation in the
dataflow and therefore a place where the compiler can verify representation and
preserve range facts.

## Scalar types have distinct domains

The core scalar vocabulary has three categories:

| Category | Types | Meaning |
| --- | --- | --- |
| Address domains | `index`, `offset` | Logical coordinates and physical byte quantities whose carriers are selected by target lowering. |
| Fixed-width integers | `i1`, `i8`, `i16`, `i32`, `i64` | Booleans and integer payloads whose width is part of the program. |
| Floating point | `f8E4M3`, `f8E5M2`, `f16`, `bf16`, `f32`, `f64` | Arithmetic and storage formats with explicit precision. |

`index` is the signed address domain for dimensions, loop bounds, element
coordinates, and general indexing arithmetic. `offset` is the unsigned address
domain for byte counts and byte positions. Its physical carrier may differ from
the index carrier on the selected target.

Fixed-width integers carry payload representation. Integer operations state the
interpretation that matters at the use: signed and unsigned comparisons,
extensions, packed-field extraction, and quantized dot products do not infer
signedness from a pointer type or host language default.

Use [`index.constant`](../reference/dialects/index/ops/constant.md) for address
domains and [`scalar.constant`](../reference/dialects/scalar/ops/constant.md)
for fixed-width and floating-point values:

```loom
%element_count = index.constant 4096 : index
%base = index.constant 0 : offset
%zero_i32 = scalar.constant 0 : i32
%zero_f32 = scalar.constant 0.0 : f32
```

The generated [type reference](../reference/types/index.md) is the complete
catalog of scalar and aggregate forms.

## Cross address boundaries explicitly

[`index.scale`](../reference/dialects/index/ops/scale.md) is the ordinary
logical-to-physical boundary. It multiplies an element coordinate by a byte
stride and returns an offset:

```loom
%byte_offset = index.scale %element_index, %element_stride : index, offset -> offset
```

[`index.cast`](../reference/dialects/index/ops/cast.md) converts between address
domains and fixed-width integer payloads:

```loom
%logical_index = index.cast %payload : i64 to index
%byte_offset = index.cast %nonnegative_payload : i64 to offset
%physical_index = index.cast %logical_index : index to i32
```

Entering `index` interprets a fixed-width payload as signed. Entering `offset`
requires a nonnegative value and uses unsigned semantics. A target whose
physical carrier is narrower accepts the conversion only when facts prove that
the value is representable. The compiler never silently wraps an address to
make lowering succeed.

At an external ABI, the ABI contract decides the physical representation. Use
a fixed-width integer when width is itself part of the caller-visible payload;
use `index` or `offset` when the source value is an address-domain quantity and
let the selected ABI lowering establish its carrier. Casts at that boundary
remain explicit in IR.

## Shapes bind dimensions to SSA values

A shaped type combines dimensions with an element type:

```loom
tensor<4x128xf16>
vector<8xi32>
view<32x64xbf16>
```

A static dimension is an integer literal. A dynamic dimension is an
index-typed SSA binding written in brackets:

```loom
tensor<[%token_count]x4096xf16>
view<[%row_count]x[%column_count]xf32>
vector<[%vector_width]xi8>
```

The name in a dynamic dimension is not decorative text. It resolves to an SSA
value in the lexical scope and lets operations, facts, and specialization agree
on the exact extent. Two values can have the same structural dynamic type while
binding its dimensions to different SSA values at different call sites.

## Value space and storage are separate

Loom does not hide storage identity inside one polymorphic tensor type:

| Type | Ownership boundary |
| --- | --- |
| [`tensor`](../reference/types/tensor.md) | A logical n-dimensional aggregate before device storage and access have been chosen. |
| [`tile`](../reference/types/tile.md) | A logical aggregate whose local tiled workset is explicit. |
| [`vector`](../reference/types/vector.md) | A register-level lane grid used for structured SIMD computation. |
| [`buffer`](../reference/types/buffer.md) | Opaque, untyped, unshaped storage identity and the root of alias facts. |
| [`view`](../reference/types/view.md) | A typed, non-owning logical coordinate space projected over buffer storage. |

Tensors represent shaped values before a program chooses storage identity and
access. Kernels usually accept opaque `buffer` roots through their launch ABI.
Inside a kernel, `buffer.view` supplies the element type, extents, base byte
offset, and optional address layout needed for access.

The composition example shows that boundary in one place:

**Source:** [`loom/docs/examples/elementwise-transform/kernel.loom`](https://github.com/ROCm/hrx-system/blob/main/loom/docs/examples/elementwise-transform/kernel.loom)

```loom title="kernel.loom"
--8<-- "examples/elementwise-transform/kernel.loom"
```

The kernel signature accepts opaque buffer identities. The body creates f32
views using the per-launch element count, and `view.load` and `view.store`
operate in logical element coordinates. Alias independence is a fact on the
buffers rather than an accidental consequence of different parameter names.

## Vectors are values, not memory

A vector is a register lane grid. It can be constructed, transformed, reduced,
or transferred to and from a view, but it carries no storage identity or layout
attachment:

```loom
%scale_vector = vector.splat %scale : vector<8xf32>
%scaled = vector.mulf %values, %scale_vector : vector<8xf32>
```

The shape is part of the vector value. A `vector<8xf32>` may eventually map to
one native instruction, several instructions, or scalarized code depending on
the target and surrounding facts. Source vector width describes structured
computation, not a promise that one specific hardware opcode exists.

Memory operations make the transfer explicit:

```loom
%values = vector.load %input_view[%column] : view<4096xf32> -> vector<8xf32>
vector.store %scaled, %output_view[%column] : vector<8xf32>, view<4096xf32>
```

## Encodings describe representation

Tensors and tiles can carry an encoding that describes packing, quantization,
or another physical data representation. Views use the same attachment slot
for address layout. The attachment changes how logical elements map to storage;
it does not change the logical shape or element type.

```loom
tensor<[%row_count]x256xi8, #ggml.q4_0>
```

Named encodings such as `#ggml.q4_0` are concise aliases for complete physical
schema records. Parameterized forms remain available when a program needs to
state a new combination explicitly. An `encoding` SSA value can carry a dynamic
layout, schema, storage, or transform role when composition must compute the
attachment. Vectors deliberately have no encoding attachment: physical layout
is resolved before or while data enters the register value domain.

The generated [encoding reference](../reference/encodings/index.md) describes
the available representation records and their parameters.

## Facts refine values instead of replacing them

An assumption operation returns a new SSA value with a stronger fact set. This
is useful when the fact comes from outside the visible dataflow, such as a
routing table whose producer guarantees valid expert IDs:

```loom
%expert_id_i32 = view.load %routing[%token] : view<[%token_count]xi32> -> i32
%expert_id = index.cast %expert_id_i32 : i32 to index
%bounded_expert_id = index.assume %expert_id [range(%expert_id, 0, 127)] : index
```

The original value still exists. Uses that need the proven bound consume
`%bounded_index`, making the proof edge visible. The same pattern carries
alignment, memory space, aliasing, storage identity, ranges, multiplicity, and
target facts through later transformations.

A dominating comparison already contributes its true or false relation as a
path fact. [`Facts and specialization`](facts-and-specialization.md#control-flow-contributes-path-facts)
shows that case; repeating it with an assumption loses the distinction between
compiler-derived information and a caller promise.

This is why semantic type distinctions matter. Facts can say that an `index`
is in range, that an `offset` is aligned, that two buffers do not alias, or that
a vector lane grid has a known shape without reconstructing those truths from
lowered pointer arithmetic.

Continue with [Functions and structured control
flow](functions-and-control.md), which composes typed values without adding a
kernel launch boundary.
