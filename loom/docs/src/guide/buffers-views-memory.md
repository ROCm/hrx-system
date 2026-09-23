# Buffers, views, and memory

**Example files:** [`loom/docs/examples/elementwise-transform/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/elementwise-transform)

Loom separates storage identity from the logical shape used to access that
storage. This keeps aliasing, memory-space, alignment, layout, and bounds facts
available to the compiler instead of hiding them behind pointer arithmetic.

In this chapter, you will learn:

- why kernels receive opaque `buffer` values and construct typed `view` values;
- where byte offsets end and logical element coordinates begin;
- how assumptions attach storage facts to new SSA values;
- how scalar, vector, atomic, and masked accesses share one view model; and
- how workgroup storage and synchronization remain explicit.

## Storage identity and logical access are different contracts

Three value kinds meet at a memory operation:

| Value | What it owns | What it does not own |
| --- | --- | --- |
| `buffer` | Opaque storage identity and root facts such as memory space, alignment, and alias scope. | Element type, rank, shape, and address layout. |
| `view` | A typed, non-owning logical coordinate space projected over a buffer. | Allocation lifetime or independent storage identity. |
| `vector` or scalar | The value transferred to or from storage. | Address identity or a persistent attachment to memory. |

This separation allows one allocation to have several useful projections. A
packed weight buffer, for example, can expose byte payloads, integer words, and
scale values through different views without pretending those views are
different allocations.

## Project a buffer into logical coordinates

[`buffer.view`](../reference/dialects/buffer/ops/view.md) combines an opaque
storage root, a physical base byte offset, and a result view type:

```loom
%base = index.constant 0 : offset
%input_view = buffer.view %input[%base] : buffer -> view<[%row_count]x[%column_count]xf32>
```

The operation neither allocates nor copies. `%input_view` refers to the same
storage root as `%input`. Its two dimensions and `f32` element type establish
the logical coordinates accepted by later accesses.

The value between brackets on `buffer.view` is an `offset` measured in bytes.
Indices on [`view.load`](../reference/dialects/view/ops/load.md),
[`view.store`](../reference/dialects/view/ops/store.md), and vector memory
operations are `index` values measured in logical elements:

```loom
%value = view.load %input_view[%row, %column] : view<[%row_count]x[%column_count]xf32> -> f32
view.store %value, %output_view[%row, %column] : f32, view<[%row_count]x[%column_count]xf32>
```

The full-rank index list names a position in view coordinates. An attached
address layout, when present, maps that position to storage. Source operations
do not reproduce the layout as flattened pointer arithmetic.

## Cross the byte boundary once

Use [`index.scale`](../reference/dialects/index/ops/scale.md) when a logical
coordinate selects a byte-addressed subregion:

```loom
%row_stride = index.constant 4096 : offset
%row_byte_offset = index.scale %first_row, %row_stride : index, offset -> offset
%rows = buffer.view %storage[%row_byte_offset] : buffer -> view<[%row_count]x1024xf32>
```

Once `%rows` exists, its operations use row and column indices. Multiplying
every access by four or carrying a flattened byte pointer through the function
would discard the distinction that lets Loom reason about bounds and layouts.

[`view.subview`](../reference/dialects/view/ops/subview.md) moves an origin in
logical coordinates while retaining the same storage root and address layout:

```loom
%tile = view.subview %matrix[%row, %column] : view<[%m]x[%n]xf32> -> view<16x32xf32>
```

[`view.refine`](../reference/dialects/view/ops/refine.md) explicitly qualifies
the shape, layout, or element-access alignment of the same view and byte base.
Neither operation is an allocation or a data movement.

Views over a common buffer can be carried through structured loops or selected
as whole values. Their storage identity stays fixed while the chosen byte
origin changes. [Rotating views over reusable storage](functions-and-control.md#rotate-views-over-reusable-storage)
shows how to reuse two slots, preserve a live exit view, and handle empty or
partially unrolled loops.

When the choice changes the address mapping, construct the candidate
`encoding.layout.strided` values and select or carry the layout as ordinary SSA
state. The resulting view type names that layout, preserving runtime element
strides without flattening logical indices into source-level pointer
arithmetic. [Selecting and carrying address
layouts](functions-and-control.md#select-and-carry-address-layouts) shows the
complete relationship and a checked target-execution example.

## Natural alignment is the ordinary access contract

An ordinary typed access requires the natural alignment of its physical scalar
element. An `i32` or `f32` access requires four-byte alignment without an authored
assumption; `i64` and `f64` require eight. For byte-addressable physical scalars,
the requirement is their storage byte width. The width of a transferred vector
does not increase it: loading `vector<16xi8>` from an `i8` view requires only
byte alignment.

Packed storage carries an explicitly reduced requirement in the view type:

```loom
%packed = buffer.view %storage[%byte_offset] : buffer -> view<4xi32, align(1)>
%word = view.load %packed[0] : view<4xi32, align(1)> -> i32
```

`align(...)` takes a positive power of two no greater than natural element
alignment. Thus `view<4xi64, align(4)>` admits four-byte-aligned words, while
`view<4xi32, align(4)>` canonicalizes to `view<4xi32>`. When a layout is present,
the qualifier follows it: `view<4xi32, %layout, align(1)>`.

This is a requirement on an executed access, not an unconditional fact about
the view's origin. Forming, passing, or slicing a view performs no access.
Inactive masked lanes have no alignment requirement, and an all-false mask can
use an empty view at an otherwise unaligned origin. Prefetch hints likewise
make no semantic access. An active access that violates its declared alignment
is outside the language contract; it does not request runtime alignment checks
or automatic fallback.

The qualifier travels through function signatures and subviews. Exact call
matching distinguishes packed and natural views. An explicit `view.refine`
can relax or strengthen the requirement for subsequent accesses; automatic
shape refinement preserves the existing alignment contract. Stronger address
facts remain separate: `buffer.assume.alignment` describes the allocation root,
and an assumption of one-byte root alignment cannot relax an ordinary `i32`
access.

Reduced alignment does not guarantee that every target implements the access.
Targets diagnose unsupported memory contracts, including atomic operations
whose required indivisible access cannot be implemented at that alignment.
Alignment also grants no extra readable or writable bytes: footprint, masking,
and bounds remain independent contracts.

## State storage facts at the boundary that knows them

External buffer parameters do not become independent merely because their SSA
names differ. If the caller contract guarantees disjoint storage, make that
guarantee visible:

```loom
%input_noalias, %output_noalias = buffer.assume.noalias %input, %output : buffer, buffer
```

Each assumption produces a refined SSA value over the same storage. Later uses
must consume that result to benefit from the fact.

| Operation | Fact supplied by the author or embedding |
| --- | --- |
| [`buffer.assume.noalias`](../reference/dialects/buffer/ops/assume-noalias.md) | The listed roots participate in comparable disjoint alias scopes. |
| [`buffer.assume.alignment`](../reference/dialects/buffer/ops/assume-alignment.md) | Every listed root has at least the stated byte alignment. |
| [`buffer.assume.memory_space`](../reference/dialects/buffer/ops/assume-memory_space.md) | A root belongs to a concrete target-independent memory space. |
| [`buffer.assume.same_root`](../reference/dialects/buffer/ops/assume-same_root.md) | Two handles are known to refer to the same underlying allocation. |

Kernel ABI buffer arguments already carry their global-memory fact. Reusable
functions that accept buffers outside that boundary may remain generic or use
an explicit assumption when their caller contract knows more. An assumption is
a correctness promise: supplying a false alias, alignment, extent, or
memory-space fact makes the program invalid.

### Aligned bases enable wide transfers

Examples and entry wrappers should expose the alignment guaranteed by their
caller. For buffers allocated with at least 64-byte base alignment, an explicit
64-byte contract gives the compiler useful freedom to select wide memory
operations. This is particularly valuable for XDNA AIE2P's 512-bit transfers.

The following example copies sixteen words from aligned buffer bases:

```loom title="aligned-vector-copy.loom"
--8<-- "examples/guide/structured-compute/aligned-vector-copy.loom"
```

On AIE2P, this copy can use one 512-bit load and one 512-bit store. With only
the natural four-byte element requirement, the same vector copy lowers to
sixteen scalar loads and sixteen scalar stores. The assumption supplies address
information; it performs no allocation, realignment, or runtime check.

The contract applies to each passed buffer base. A byte offset that is a
multiple of 64 preserves the alignment; moving one `i32` element from that
base guarantees only four-byte alignment at the new origin. Subviews retain
these relationships without strengthening the root. When adapting an example,
change or remove the assumption to match the embedding's actual guarantee;
the scalar element-access requirement remains in force.

## Transfer structured values through views

Scalar access names one logical element. [`vector.load`](../reference/dialects/vector/ops/load.md)
and [`vector.store`](../reference/dialects/vector/ops/store.md) name a logical
origin and transfer a vector footprint:

```loom
%values = vector.load %matrix[%row, %column] : view<[%m]x[%n]xf32> -> vector<4x8xf32>
vector.store %values, %result[%row, %column] : vector<4x8xf32>, view<[%m]x[%n]xf32>
```

Vector axes map to the trailing view axes. Leading view axes select the slice;
the vector shape describes the transferred footprint. The source states the
structured operation while target lowering chooses native instructions,
splitting, or scalarization.

Masked operations make tail behavior part of the access itself. A false load
lane does not access memory and takes its passthrough value; a false store lane
does not modify memory:

```loom
%loaded = vector.load.mask %input[%row, %column], %mask, %zero : view<[%m]x[%n]xf32>, vector<8xi1>, vector<8xf32>
vector.store.mask %loaded, %output[%row, %column], %mask : vector<8xf32>, view<[%m]x[%n]xf32>, vector<8xi1>
```

Gather and scatter operations add per-lane logical offsets to the final view
axis. Fragment loads and stores add a structured matrix-operand role. These are
variations of the same view contract rather than separate pointer models.

## Allocate scratch by physical extent

[`buffer.alloca`](../reference/dialects/buffer/ops/alloca.md) creates a fresh
scratch root in an allocatable memory space. Its requested extent and alignment
are physical byte quantities:

```loom
%scratch_bytes = index.constant 4096 : offset
%base = index.constant 0 : offset
%scratch = buffer.alloca<workgroup> align(64) %scratch_bytes : buffer
%scratch_view = buffer.view %scratch[%base] : buffer -> view<32x32xf32>
```

Every execution produces a distinct storage identity. A target requiring a
static frame reserves the proven finite maximum, and the compiled launch
contract reports the total workgroup-local storage required by the kernel.

When several simultaneously live byte ranges share one slab,
[`buffer.pack`](../reference/dialects/buffer/ops/pack.md) computes aligned,
non-overlapping offsets and a repeatable total stride:

```loom
%total_bytes, %header_offset, %payload_offset = buffer.pack [align(16) %header_bytes, align(256) %payload_bytes] : offset
```

This is physical layout. Reusing one allocation across non-overlapping
lifetimes is a compiler allocation decision and is not encoded by pretending
the ranges are simultaneously live.

## Synchronization names both rendezvous and memory

Storage visibility is not implied by source order across invocations. When
workitems exchange values through workgroup storage, the program names the
execution scope, fenced memory space, and ordering:

```loom
vector.store %values, %scratch_view[%row, %column] : vector<4xf32>, view<32x32xf32>
kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
%transposed = vector.load %scratch_view[%column, %row] : view<32x32xf32> -> vector<4xf32>
```

[`kernel.barrier`](../reference/dialects/kernel/ops/barrier.md) is a rendezvous
for participating invocations plus a fence for the named memory space.
Asynchronous transfers use `kernel.async.group` and `kernel.async.wait` for
completion; a barrier is added only when invocations also need to rendezvous.

Atomic view operations similarly require an explicit scope and ordering. A
plain store does not become atomic because several invocations may reach it,
and a barrier does not resolve conflicting writes.

## Preserve the strongest useful representation

Common representation mistakes all erase information too early:

| Symptom | Lost contract | Stronger source shape |
| --- | --- | --- |
| Every access performs byte arithmetic | Logical shape and element stride | Construct a view once, then index it logically. |
| Two parameters are assumed independent by convention | Alias proof | Refine them together with `buffer.assume.noalias`. |
| A flat view is used for a logical matrix | Rank and axis structure | Use a ranked view with the actual logical dimensions. |
| Tail lanes load first and mask later | Memory safety | Use a guarded region or masked memory operation. |
| Shared-memory communication relies on source order | Cross-invocation visibility | State the required barrier or async completion edge. |
| Target address spaces appear throughout a motif | Reusability | Use target-independent memory spaces and specialize at the leaf. |

The [source-to-artifacts kernel](../getting-started/source-to-artifacts.md#a-kernel-owns-two-contracts)
shows the complete buffer-to-view path with dynamic extent, explicit no-alias
facts, a control-flow-derived in-bounds proof, and scalar access.

Continue with [Kernels and launch configuration](kernels-and-launch.md), where
the storage operations become part of a dispatchable entry with separate
workload and device contracts.
