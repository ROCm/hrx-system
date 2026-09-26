# Native XDNA executable images

A `.xdna` file describes native executable storage and how an exported entry
uses it. ELF program headers carry explicit load ranges; compact metadata
carries allocation requirements, external bindings, address relocations, and
invocation ranges. Native commands and AIE instructions remain opaque payloads.
The compiler owns the device program; the loader establishes its declared
storage and binding contract.

The image object in this directory is immutable and device-independent. The
[experimental executable adapter](../../../../../../../../experimental/xdna/executable.h) connects
it to caller-owned mappings and libamdf memory handles. The
[runner](../../../../../../../../experimental/xdna/README.md) demonstrates the complete lifecycle.
The [Loom producer](../../../../../../../../loom/src/loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h)
and reader share the fixed-width codecs in
[xdna_executable.h](../../../../../schemas/xdna_executable.h).

## Storage, identity, and ownership

An allocation row is a requirement, not a runtime object. An entry's allocation
use table selects the required global rows in a stable order. The caller
provides backing with those extents, alignments, and native address domains:

| Domain | Meaning |
| --- | --- |
| `COMMAND` | Context-private native instruction backing used for submission. |
| `DMA` | Device-addressable backing referenced by native commands or workers. |

Mutable invocation backing is private to its execution instance. Immutable
backing can be shared after all loading and static relocation are complete,
subject to the native memory scope and address lifetime. Exact file-range
aliases let one stored payload initialize multiple destinations without
replicating it in the file. This does not itself imply shared device backing.

Image admission compares the exact device-profile identity and revision,
firmware ABI, instruction generation, and context geometry. Strix NPU4 and
Strix Halo NPU5 share an array family but have distinct execution profiles.
The loader does not guess compatibility from a family name or patch an image
to fit another endpoint.

`iree_hal_amd_xdna_image_create` retains the source byte sequence, decodes the
load directory, and retains compact metadata. It does not flatten or copy the
native payload into another host image. Table queries borrow image-owned
storage. The image has no device allocations, mappings, context, or submission
ownership, and destroying it performs no device operation.

## Loading, binding, and execution

A real caller follows this sequence:

1. Create an image against the admitted context facts and select an export.
2. Allocate the entry's storage in allocation-use order, using the required
   native domain, extent, and alignment, and acquire writable mappings.
3. Load each declared file range directly to its destination. Clear only its
   explicit zero-fill tail. Apply static relocations using allocation addresses.
4. Supply external logical buffer ranges and bind their device addresses into
   the declared dynamic relocation fields.
5. Publish mapped writes through the native cache API, resolve invocation zero,
   and submit its command handle and range through libamdf.
6. Observe terminal completion, perform required output cache invalidation,
   and advance to the metadata's continuation ordinal while residency is valid.
7. Rebind or release resources only after their previous device users have
   drained. Keep mappings, backing, logical buffers, and context alive through
   their actual completion frontier.

`iree_hal_amd_xdna_executable_load` and `_bind` allocate no memory and retain no
resources. They check their supplied contracts before writes. A source I/O
failure during loading may leave backing partially initialized; that backing
is not ready for submission. Failure does not promise rollback of caller
storage. Native submission receives memory handles and byte ranges, with no
per-launch XRT patch list or ELF decoding.

Invocation zero establishes entry state. The current compiler emits a finite
protocol `0 -> 1 -> 1`: establishment first, then repeated work using resident
workers. A self-contained range can express `0 -> 0`. A continuation becomes
valid only after terminal completion while context, backing, and resident state
remain intact. Reset, replacement by another entry, or loss of backing requires
establishment again. The caller owns this ordinal and its completion frontier.

Native command retirement alone does not prove arbitrary autonomous tile work
has ended. The compiled program must establish the required output and worker
completion before the caller treats that frontier as final. Within a held
invocation, streaming, instruction shards, and role rotation are device-program
semantics. The file format imposes no fixed ARRAY/CONTROL dispatch categories
and contains no precompiled bootstrap PDI; family bootstrap belongs to libamdf.

## Implementation map

| Component | Contract |
| --- | --- |
| [directory.h](directory.h) | Retained source, canonical ELF envelope, load headers, and bounded source-range access. |
| [tables.h](tables.h) | Compact little-endian rows with indexed, allocation-free queries. |
| [validation.h](validation.h) | Admission of row relationships, ranges, and exact target requirements. |
| [image.h](image.h) | Immutable owner and export lookup. |
| [aie2p/target.h](aie2p/target.h) | Admitted context identity, geometry, and instruction alignment. |
| [Shared schema](../../../../../schemas/xdna_executable.h) | Wire constants and fixed-width codecs used by producer and consumer. |
| [Executable adapter](../../../../../../../../experimental/xdna/executable.h) | Loading, relocation, external binding, and native command resolution. |

Admission validates untrusted file structure once. Indexed consumers use the
established relationships. Native payload is executable code: these checks are
not a sandbox and do not certify instruction semantics.

## Wire format, metadata version 2

All multibyte fields are little endian. Offsets below are byte offsets from the
start of the relevant row; `u16`, `u32`, and `u64` are unsigned fixed-width
integers, and `i64` is signed. Rows are encoded explicitly, never by casting a
host struct over file storage. Unknown versions, identities, kinds, or flag
bits outside the sets below are rejected.

### ELF envelope and load directory

The envelope is ELF32, little endian, current ELF version, OSABI zero, ABI
version zero, with zero identity padding. `e_type = ET_EXEC (2)`,
`e_machine = EM_AIE (264)`, `e_entry = 0`, and AIE2P `e_flags = 3`.
The ELF header is 52 bytes. Its program-header directory starts at byte 52,
has 32-byte rows, and contains between 1 and 4,096 rows.

Section headers are optional and have no execution semantics. If any section
directory field is present, the directory has a nonzero four-byte-aligned
file offset, 40-byte rows, 1..4,096 entries, and an in-range string-table
ordinal. All directories are in bounds and nonoverlapping.

| Program header | Required values and interpretation |
| --- | --- |
| Ordinal 0 | `p_type = 0x6C584408` (`METADATA`), `p_flags = PF_R (4)`, `p_vaddr = p_paddr = 0`, and `p_filesz = p_memsz` between 64 bytes and 16 MiB. |
| Ordinals 1 onward | `p_type = PT_LOAD (1)`, `p_flags = PF_R (4)`. `p_paddr` is a global allocation ordinal; `p_vaddr` is a destination byte offset within that allocation, not a device virtual address. |

Every program header has a nonzero power-of-two `p_align`, with
`p_offset % p_align == p_vaddr % p_align`, and `p_filesz <= p_memsz`.
A load has nonzero `p_memsz`. Its file range is in bounds and does not intersect
the ELF header or either directory. Two nonempty payload ranges may be exactly
identical; partial overlaps are invalid.

An allocation owns a consecutive slice of load headers. These slices cover all
headers after metadata in allocation order. Within an allocation, destination
ranges are ordered, nonoverlapping, and contained in its declared extent.
Loading copies `p_filesz` bytes and zeroes only the following
`p_memsz - p_filesz` bytes. Gaps between loads remain undefined.

### Metadata layout and header

Metadata is the following concatenation, with no padding between tables:

```text
header[64]
allocation[32] * A
allocation_use[4] * U
entry[48] * E
binding[40] * B
relocation[48] * R
invocation[16] * I
name_bytes[S]
```

The exact length is `64 + 32*A + 4*U + 48*E + 40*B + 48*R + 16*I + S`.
Each record count is at most 65,535; `A` and `E` are nonzero. The whole metadata
extent is at most 16 MiB. Name bytes have no implicit terminators.

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | u32 | magic | `0x414E4458`, bytes `XDNA`. |
| 4 | u16 | version | `2`. |
| 6 | u16 | native_encoding | `1`, native transaction protocol 0.1. |
| 8 | u32 | target_generation | `3`, AIE2P. |
| 12 | u32 | device_profile_revision | Exact incompatible profile revision. |
| 16 | u64 | device_profile_id | Exact execution-profile identity. |
| 24 | u64 | firmware_abi_id | Exact native firmware protocol identity. |
| 32 | u16 | column_count | Required context-relative column count. |
| 34 | u16 | row_count | Required array row count. |
| 36 | u32 | allocation_count | `A`. |
| 40 | u32 | allocation_use_count | `U`. |
| 44 | u32 | entry_count | `E`. |
| 48 | u32 | binding_count | `B`. |
| 52 | u32 | relocation_count | `R`, static plus dynamic rows. |
| 56 | u32 | invocation_count | `I`. |
| 60 | u32 | string_byte_length | `S`. |

### Allocation and allocation-use rows

| Offset | Type | Allocation field | Meaning |
| --- | --- | --- | --- |
| 0 | u32 | domain | `COMMAND = 1`, `DMA = 2`. |
| 4 | u32 | flags | `0` mutable; `IMMUTABLE = 1`; `DEVICE_WRITE = 2`. The latter two are mutually exclusive. |
| 8 | u64 | byte_length | Nonzero required capacity, representable by the host. |
| 16 | u64 | alignment | Nonzero power-of-two device-address alignment. |
| 24 | u32 | first_load | First global ELF program-header ordinal. |
| 28 | u32 | load_count | Number of destination-ordered load headers. |

An allocation-use row is one `u32` global allocation ordinal. Each entry's use
slice contains strictly increasing, unique, valid ordinals. An entry-relative
use index selects a row from that slice; it is not itself a global allocation
ordinal.

### Entry rows

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u32 | name_offset |
| 4 | u32 | name_length |
| 8 | u32 | first_allocation_use |
| 12 | u32 | allocation_use_count |
| 16 | u32 | first_binding |
| 20 | u32 | binding_count |
| 24 | u32 | first_static_relocation |
| 28 | u32 | static_relocation_count |
| 32 | u32 | first_dynamic_relocation |
| 36 | u32 | dynamic_relocation_count |
| 40 | u32 | first_invocation |
| 44 | u32 | invocation_count |

The name range addresses the trailing byte table and has length 1..4,096.
Use and invocation counts are nonzero. Entry slices exhaust their corresponding
tables in entry order, with no gaps or unowned rows. In the relocation table,
each entry's static slice immediately precedes its dynamic slice. Bindings and
invocations are addressed by dense ordinals relative to the selected entry.

### Binding rows

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | u16 | kind | `NONE = 0` or `BUFFER = 1`. |
| 2 | u16 | address_space | `GLOBAL = 1` or `HOST = 2`. |
| 4 | u16 | access | Nonempty combination of `READ = 1`, `WRITE = 2`. |
| 6 | u16 | usage | Combination of `DEVICE_VISIBLE = 1`, `HOST_VISIBLE = 2`, `COHERENT = 4`, `CACHED = 8`. |
| 8 | u64 | minimum_byte_length | Minimum logical binding extent. |
| 16 | u64 | minimum_alignment | Nonzero power-of-two address alignment. |
| 24 | u64 | minimum_byte_offset | Inclusive lower bound on the supplied logical offset. |
| 32 | u64 | maximum_byte_offset | Inclusive upper bound; `UINT64_MAX` permits any offset satisfying the other bounds. |

`NONE` is an exact all-zero row preserving an unused position in the entry's
dense binding ABI. It has no supplied resource requirement and cannot source a
dynamic relocation. A `BUFFER` row has nonempty access and the supplied logical
range must fit its buffer and satisfy the visibility, extent, offset, and
alignment requirements. Minimum offset cannot exceed maximum offset. Binding
ordinals are independent of allocation-use ordinals: external tensors are not
executable backing allocations.

### Relocation rows

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | u32 | destination_use | Entry-relative allocation use receiving the patch. |
| 4 | u32 | source_ordinal | Allocation use for static fixups; external binding for dynamic fixups. |
| 8 | u32 | byte_offset | Four-byte-aligned offset of the eight-byte destination field. |
| 12 | u32 | kind | `SHIM_ADDRESS = 1`. |
| 16 | i64 | addend | Signed addition to the resolved source address. |
| 24 | u64 | minimum_value | Inclusive lower bound on the resulting address. |
| 32 | u64 | maximum_value | Inclusive upper bound, no greater than `0xFFFFFFFFFFFF`. |
| 40 | u64 | alignment | Required result alignment, a power of two at least four. |

The eight-byte destination lies entirely in initialized storage, including
explicit zero-fill tails. Each relocation slice is ordered by
`(destination_use, byte_offset)` and has nonoverlapping fields. Static and
dynamic fields also cannot overlap. Static sources are DMA-domain allocations;
dynamic destinations are mutable allocations.

After checked signed addition, bounds and alignment checks, the shim-address
patch writes two little-endian words:

```text
low  = (old_low  & 0x00000003) | uint32(address)
high = (old_high & 0xFFFF0000) | uint32(address >> 32)
```

This preserves the adjacent control bits. The loader patches only the declared
field; it does not search or decode native command streams.

### Invocation rows

| Offset | Type | Field | Meaning |
| --- | --- | --- | --- |
| 0 | u32 | allocation_use | Entry-relative COMMAND allocation use. |
| 4 | u32 | byte_offset | Command range offset, aligned to the target's instruction requirement. |
| 8 | u32 | byte_length | Nonzero byte length, a multiple of four. |
| 12 | u32 | next_invocation | Valid entry-relative continuation ordinal. |

The range fits its allocation and is fully initialized. Allocation alignment
is at least the target instruction alignment. Ordinal zero and continuation
validity have the lifecycle semantics above; the table describes ranges and
state transitions, not a requirement to rebuild or repatch an image on every
submission.
