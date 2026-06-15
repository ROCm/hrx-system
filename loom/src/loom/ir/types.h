// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom IR type system.
//
// Types describe the shapes and element types of values in loom IR.
// The type system has these core categories:
//
//   Scalar types:    f16, f32, f64, bf16, i1, i8, i16, i32, i64, index
//   Tensor types:    tensor<[%M]xf32>     (logical tensor value)
//   Tile types:      tile<[%M]x4xf32>     (tile-level aggregate value)
//   Vector types:    vector<[%M]xf32>     (register lane grid value)
//   Register types:  reg<amdgpu.vgpr x4> (target-owned low payload)
//   Buffer types:    buffer               (opaque storage identity)
//   View types:      view<[%M]xf32, %layout> (typed buffer projection)
//   Group types:     group<scope>         (barrier scoping)
//   Storage types:   low.storage<workgroup> (function-local byte storage)
//   Function types:  (f32, i32) -> (f64)  (callable signatures)
//
// Shape dimensions are either static integers or dynamic SSA value
// references. Static dims are known at compile time. Dynamic dims
// reference SSA values (index-typed) that provide the runtime size.
// Within function signatures, Scope() enables forward references so
// result types can reference co-result dims by name:
//
//   tile<4x4xf32>           All static.
//   tile<[%M]x4xf32>        First dim dynamic, named %M.
//   tensor<[%M]x[%K]xf32>   Both dims dynamic.
//   vector<16xf32>          Static 1-D register vector.
//   vector<[%N]xi32>        Dynamic register vector extent.
//   view<[%N]xf32, %layout> Dynamic view layout carried by an SSA value.
//   buffer                  Opaque buffer storage identity.
//
// Tensor, tile, and vector types describe SSA values, not aliasable storage.
// A tensor is a logical n-D aggregate used before storage has been chosen, a
// tile is the same value-space idea after tiling decisions have made local
// worksets explicit, and a vector is the register-level lane grid used by
// vector lowering. Vector types are always rank >= 1 and cannot carry
// encodings/layouts; use an explicit splat/broadcast/conversion op when a
// scalar value must feed vector code.
//
// Buffers and views carry the storage story. A buffer is an untyped, unshaped
// storage identity: it says "this SSA value may alias other values derived from
// the same storage," but it says nothing about element type, logical extents,
// or byte layout. A view is a non-owning projection over a buffer-like storage
// identity. The view type records the logical element space and optional
// address layout, while the op that creates the view records which buffer,
// offset, and dynamic bindings the view comes from. This keeps alias reasoning
// attached to SSA use-def structure instead of hiding it in a polymorphic
// kernel signature.
//
// Dynamic dims are printed with SSA value names. At use sites, those names
// resolve to ordinary index-typed SSA values in the current lexical scope. In
// declaration wrappers such as function/global signatures, names that appear
// first inside a type may be parsed as declaration-local placeholders, but
// function body visibility still comes from explicit binders or op results.
//
// Tile and tensor types may carry an encoding that describes the physical data
// layout (quantization, packing, etc.). View types use the same representation
// slot for address layout. Vector types are pure register lane grids and do not
// carry layout attachments.
//
//   tile<256x256xf32, #q6_k>
//   tensor<[%N]x[%K]xi8, #q8_0<block=32>>
//   view<[%N]xf32, #strided<stride=64>>
//
// The encoding/layout attachment is metadata: it doesn't change the logical
// shape or element type, but determines how bytes map to logical elements.
// The first-class `encoding` type is an SSA value type used in attachment
// position (`tile<4xf32, %enc>`, `view<[%N]xf32, %layout>`). Encoding values
// may use role-qualified types (`encoding<layout>`, `encoding<schema>`,
// `encoding<storage>`, `encoding<transform>`) so op signatures can declare the
// exact role they consume. It is parsed as a built-in keyword rather than as a
// parameterized TypeDef registry entry.
// Static attachments use only attribute parameters:
//
//   tile<256xi8, #q8_0<block=32>>
//
// Dynamic parameters are named operands on encoding.define, and the resulting
// `%enc` value is attached to shaped types:
//
//   %enc = encoding.define #q8_0<block=32> {group_size = %g : index}
//       : encoding<schema>
//   tile<256xi8, %enc>
//
// ==========================================================================
// Memory layout
// ==========================================================================
//
// Types are 24 bytes, fitting 2-3 per cache line. For the common case
// (rank <= 2 tiles/vectors), everything is inline with no pointer
// chases. Values store types by value, while each module also interns
// canonical type entries for deduplication.
//
//   bytes 0-3:  packed header (kind, element_type/role/space, rank, flags)
//   bytes 4-5:  encoding_id
//   bytes 6-7:  encoding_flags
//   bytes 8-23: dims (inline for rank <= 2, overflow pointer otherwise)
//
// Inline dim packing (rank <= 2):
//   Each dim is a uint64_t. Top bit encodes the kind:
//   bit 63 = 0: static (bits 0-62 = size).
//   bit 63 = 1: dynamic SSA value (bits 0-61 = value_id, bit 62 reserved).
//
// Overflow (rank > 2):
//   dims[0] = pointer to arena-allocated loom_dim_t array.
//   dims[1] = unused.
//
// Function types use dims[0] as a pointer to an arena-allocated
// loom_func_type_data_t containing the arg and result type arrays.
// Function types may carry SSA-specific dim bindings within their
// arg/result types, so the embedded types are stored by value (24
// bytes each), not interned.
//
// Register types reserve dims[0..1] as target-owned payload storage. Core IR
// only stores and compares the payload structurally; target-low helpers
// interpret the fields as a register-class identity plus contiguous unit count.
//
// Type equality is structural: callers should use loom_type_equal() on the
// by-value representation. The module type table deduplicates equal entries,
// but pointer identity only applies to table entries themselves, not arbitrary
// loom_type_t values copied into SSA values or temporary parser storage.

#ifndef LOOM_IR_TYPES_H_
#define LOOM_IR_TYPES_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// IDs
//===----------------------------------------------------------------------===//

// Index into the module's value table (module->values.entries[]).
// Values are 64-byte cache-line-aligned entries. Prefer passing
// value IDs over value pointers for stability across IR mutations.
typedef uint32_t loom_value_id_t;
#define LOOM_VALUE_ID_INVALID ((loom_value_id_t)UINT32_MAX)

// Dense local ordinal for a value in an explicit value domain.
//
// Unlike loom_value_id_t, ordinals are not module-stable identities. They are
// valid only against the domain or frame that produced them and are used for
// direct indexing into compact function-local tables.
typedef uint32_t loom_value_ordinal_t;
#define LOOM_VALUE_ORDINAL_INVALID ((loom_value_ordinal_t)UINT32_MAX)

// Index into the module's interned string table.
// Strings are deduplicated: identical strings share the same ID.
typedef uint32_t loom_string_id_t;
#define LOOM_STRING_ID_INVALID ((loom_string_id_t)UINT32_MAX)

// Index into the module's symbol table (module->symbols.entries[]).
typedef uint16_t loom_symbol_id_t;

// Index into the module's interned type table (module->types.entries[]).
// Types are deduplicated: structurally identical types share the same ID.
typedef uint32_t loom_type_id_t;
#define LOOM_TYPE_ID_INVALID ((loom_type_id_t)UINT32_MAX)

//===----------------------------------------------------------------------===//
// Dims and enums
//===----------------------------------------------------------------------===//

// Dimension packing uses the top bit to distinguish two kinds:
//
//   bit 63 = 0: static dimension (bits 0-62 = size).
//   bit 63 = 1: dynamic SSA value (bits 0-61 = value_id).
//
// Bit 62 is reserved for dynamic dims and must remain zero.

// Dynamic flag: top bit of a packed dimension value.
#define LOOM_DIM_DYNAMIC_FLAG ((uint64_t)1 << 63)

// Mask covering the payload bits (0-61).
#define LOOM_DIM_PAYLOAD_MASK (((uint64_t)1 << 62) - 1)

// Maximum static dimension size (2^63 - 1).
#define LOOM_DIM_MAX_STATIC_SIZE INT64_MAX

// Maximum SSA value ID that fits in a packed dim.
#define LOOM_DIM_MAX_VALUE_ID (((uint64_t)1 << 62) - 1)

// Packs a static dimension into the inline format.
static inline uint64_t loom_dim_pack_static(int64_t size) {
  IREE_ASSERT(size >= 0 && size <= LOOM_DIM_MAX_STATIC_SIZE,
              "static dim size out of range");
  return (uint64_t)size;  // Top bit clear = static.
}

// Packs a dynamic dimension (SSA value ID) into the inline format.
static inline uint64_t loom_dim_pack_dynamic(uint64_t value_id) {
  IREE_ASSERT(value_id <= LOOM_DIM_MAX_VALUE_ID,
              "dynamic dim value ID out of range");
  return LOOM_DIM_DYNAMIC_FLAG | value_id;
}

// Returns true if a packed dimension is dynamic (SSA value reference).
static inline bool loom_dim_is_dynamic(uint64_t packed) {
  return (packed & LOOM_DIM_DYNAMIC_FLAG) != 0;
}

// Returns the static size from a packed dimension.
// Caller must check !loom_dim_is_dynamic() first.
static inline int64_t loom_dim_static_size(uint64_t packed) {
  return (int64_t)packed;
}

// Returns the SSA value ID from a packed dynamic dimension.
// Caller must check loom_dim_is_dynamic() first.
static inline loom_value_id_t loom_dim_value_id(uint64_t packed) {
  uint64_t payload = packed & LOOM_DIM_PAYLOAD_MASK;
  IREE_ASSERT(payload <= UINT32_MAX,
              "dim value ID exceeds loom_value_id_t range");
  return (loom_value_id_t)payload;
}

// Overflow dimension for rank > 2. Same packing as inline dims.
// Arena-allocated array, pointed to by dims[0] in the type struct.
typedef uint64_t loom_overflow_dim_t;

// Type kind tag constants.
//
// These are internal compiler values. The bytecode format has its own wire enum
// (loom_bytecode_type_kind_t in format.h) with independent versioning. Keep
// these constants append-only and map values explicitly in the bytecode
// reader/writer when the wire format diverges.
enum loom_type_kind_e {
  LOOM_TYPE_NONE = 0,       // Absence of a type (no-result ops).
  LOOM_TYPE_SCALAR = 1,     // f32, i8, index, etc.
  LOOM_TYPE_TILE = 2,       // tile<[%M]x4xf32>
  LOOM_TYPE_TENSOR = 3,     // tensor<[%M]xf32>
  LOOM_TYPE_GROUP = 4,      // group<scope>
  LOOM_TYPE_FUNCTION = 5,   // (types) -> (types)
  LOOM_TYPE_DIALECT = 6,    // hal.buffer, vm.ref<T>, etc.
  LOOM_TYPE_ENCODING = 7,   // encoding<role> (first-class SSA encoding value)
  LOOM_TYPE_POOL = 8,       // pool<[%block_size]> (block-managed memory)
  LOOM_TYPE_VECTOR = 9,     // vector<[%M]xf32> (register lane grid)
  LOOM_TYPE_VIEW = 10,      // view<[%M]xf32, %layout> (buffer projection)
  LOOM_TYPE_BUFFER = 11,    // buffer (opaque storage identity)
  LOOM_TYPE_REGISTER = 12,  // reg<amdgpu.vgpr x4> (target-owned low payload)
  LOOM_TYPE_STORAGE = 13,   // low.storage<workgroup> (function-local storage)
  LOOM_TYPE_COUNT_,
};

typedef uint8_t loom_type_kind_t;

// Returns true if |kind| names a real type kind. The LOOM_TYPE_COUNT_
// sentinel is not a type and must not be serialized or interpreted.
static inline bool loom_type_kind_is_valid(loom_type_kind_t kind) {
  return (uint32_t)kind < LOOM_TYPE_COUNT_;
}

// Group scope kind. Stored in the element_type byte of the type header
// for LOOM_TYPE_GROUP (that byte is unused for non-shaped types).
enum loom_group_scope_e {
  LOOM_GROUP_SCOPE_WORKGROUP = 0,
  LOOM_GROUP_SCOPE_SUBGROUP = 1,
  LOOM_GROUP_SCOPE_COUNT_,
};
// Raw group-scope storage.
typedef uint8_t loom_group_scope_t;

// Returns the name string for a group scope kind, or NULL if invalid.
const char* loom_group_scope_name(loom_group_scope_t scope);

// Semantic encoding role. Stored in the element_type byte of the type header
// for LOOM_TYPE_ENCODING.
enum loom_encoding_role_e {
  // Role is intentionally unspecified.
  LOOM_ENCODING_ROLE_UNKNOWN = 0,
  // Address-layout mapping for view address arithmetic.
  LOOM_ENCODING_ROLE_ADDRESS_LAYOUT = 1,
  // Physical storage schema such as a packed quantized block format.
  LOOM_ENCODING_ROLE_STORAGE_SCHEMA = 2,
  // Composition of an address layout and storage schema.
  LOOM_ENCODING_ROLE_PHYSICAL_STORAGE = 3,
  // Numeric transform descriptor applied by explicit transform ops.
  LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM = 4,
  LOOM_ENCODING_ROLE_COUNT_,
};
// Raw encoding-role storage. Bytecode validation may inspect future role
// ordinals before rejecting them, so storage cannot use a C enum type.
typedef uint8_t loom_encoding_role_t;

// Returns true if |role| names a real encoding role.
static inline bool loom_encoding_role_is_valid(loom_encoding_role_t role) {
  return (uint32_t)role < LOOM_ENCODING_ROLE_COUNT_;
}

// Returns the short text spelling for an encoding role, or NULL if invalid.
const char* loom_encoding_role_name(loom_encoding_role_t role);

// Parses the short text spelling for an encoding role.
bool loom_encoding_role_parse(iree_string_view_t text,
                              loom_encoding_role_t* out_role);

// Function-local storage space. Stored in the element_type byte of the type
// header for LOOM_TYPE_STORAGE.
enum loom_storage_space_e {
  // Host stack-frame storage.
  LOOM_STORAGE_SPACE_STACK = 0,
  // Per-lane spill/scratch storage.
  LOOM_STORAGE_SPACE_SCRATCH = 1,
  // Target-private per-invocation storage.
  LOOM_STORAGE_SPACE_PRIVATE = 2,
  // Workgroup-local shared storage.
  LOOM_STORAGE_SPACE_WORKGROUP = 3,
  LOOM_STORAGE_SPACE_COUNT_,
};
// Raw function-local storage-space storage.
typedef uint8_t loom_storage_space_t;

// Returns true if |space| names a real storage space.
static inline bool loom_storage_space_is_valid(loom_storage_space_t space) {
  return (uint32_t)space < LOOM_STORAGE_SPACE_COUNT_;
}

// Returns the short text spelling for a storage space, or NULL if invalid.
const char* loom_storage_space_name(loom_storage_space_t space);

// Parses the short text spelling for a storage space.
bool loom_storage_space_parse(iree_string_view_t text,
                              loom_storage_space_t* out_space);

// Type flags (packed into header bits 20-23).
enum loom_type_flags_e {
  // Shape dims are stored inline in dims[0..1].
  // Clear = dims[0] is a pointer to overflow array.
  LOOM_TYPE_FLAG_INLINE_DIMS = (1 << 0),
  // Type has all-static dimensions (no dynamic dims).
  // Enables fast-path checks without inspecting individual dims.
  LOOM_TYPE_FLAG_ALL_STATIC = (1 << 1),
};

// Maximum rank representable in the type header. Rank is packed into four bits
// so well-formed shaped types are rank 0-15.
#define LOOM_TYPE_MAX_RANK 15

// Encoding flag bits for the encoding_flags field of loom_type_t.
//
// When LOOM_ENCODING_FLAG_SSA is set, encoding_id holds a value_id
// (uint16_t) referencing an encoding-typed SSA value instead of a
// module encoding table index. This enables dynamic encodings that
// propagate through function signatures as SSA values.
//
enum loom_encoding_flag_bits_e {
  LOOM_ENCODING_FLAG_SSA = (1u << 0),
};
typedef uint16_t loom_encoding_flags_t;

//===----------------------------------------------------------------------===//
// Type struct and accessors
//===----------------------------------------------------------------------===//

// 24-byte by-value type representation. Fits 2-3 per cache line.
//
// Modules also intern canonical copies for deduplication, but a loom_type_t
// value itself should be compared with loom_type_equal(), not pointer identity.
typedef struct loom_type_t {
  // Packed header:
  //   [0:7]   loom_type_kind_t
  //   [8:15]  loom_scalar_type_t (shaped types), loom_group_scope_t (group),
  //           loom_encoding_role_t (encoding), or loom_storage_space_t
  //           (storage)
  //   [16:19] rank (0-LOOM_TYPE_MAX_RANK for shaped types, 0 otherwise)
  //   [20:23] loom_type_flags_e (inline_dims, all_static)
  //   [24:31] reserved
  uint32_t header;

  // Index into the module's encoding table (1-based, 0 = no encoding).
  // When encoding_flags has LOOM_ENCODING_FLAG_SSA set, this is instead
  // a value_id referencing an SSA value of type LOOM_TYPE_ENCODING.
  uint16_t encoding_id;

  // Encoding interpretation flags. Zero for most types.
  loom_encoding_flags_t encoding_flags;

  // Inline dim storage for rank <= 2. Each element is a packed dim
  // (see loom_dim_pack_*). For rank > 2, dims[0] is a pointer to an
  // arena-allocated loom_overflow_dim_t array and dims[1] is unused. For
  // function
  // types, dims[0] is a pointer to loom_func_type_data_t.
  uint64_t dims[2];
} loom_type_t;

// Static assert: type must be exactly 24 bytes, no padding.
_Static_assert(sizeof(loom_type_t) == 24, "loom_type_t must be 24 bytes");

// Arena-allocated overflow data for function types. Stored via pointer
// in dims[0] of a LOOM_TYPE_FUNCTION loom_type_t. The types array
// holds arg types followed by result types contiguously:
//   types[0 .. arg_count-1]                      = argument types
//   types[arg_count .. arg_count+result_count-1]  = result types
//
// The embedded types are stored by value (not interned) because they
// may carry SSA-specific dim bindings that differ per use site.
typedef struct loom_func_type_data_t {
  uint16_t arg_count;
  uint16_t result_count;
  uint32_t reserved;  // Padding for loom_type_t alignment (8 bytes).
  loom_type_t types[];
} loom_func_type_data_t;

_Static_assert(sizeof(loom_func_type_data_t) == 8,
               "loom_func_type_data_t header must be 8 bytes");

// --- Header accessors ---

static inline loom_type_kind_t loom_type_kind(loom_type_t type) {
  return (loom_type_kind_t)(type.header & 0xFF);
}

static inline loom_scalar_type_t loom_type_element_type(loom_type_t type) {
  return (loom_scalar_type_t)((type.header >> 8) & 0xFF);
}

static inline uint8_t loom_type_rank(loom_type_t type) {
  return (uint8_t)((type.header >> 16) & 0xF);
}

static inline uint8_t loom_type_flags(loom_type_t type) {
  return (uint8_t)((type.header >> 20) & 0xF);
}

static inline bool loom_type_has_inline_dims(loom_type_t type) {
  return (loom_type_flags(type) & LOOM_TYPE_FLAG_INLINE_DIMS) != 0;
}

static inline bool loom_type_is_all_static(loom_type_t type) {
  return (loom_type_flags(type) & LOOM_TYPE_FLAG_ALL_STATIC) != 0;
}

// Returns the group scope for a type. Only valid when kind == LOOM_TYPE_GROUP.
// The scope is stored in the element_type byte of the header.
static inline loom_group_scope_t loom_type_group_scope(loom_type_t type) {
  return (loom_group_scope_t)((type.header >> 8) & 0xFF);
}

// Returns the role for an encoding type. Only valid when
// kind == LOOM_TYPE_ENCODING.
static inline loom_encoding_role_t loom_type_encoding_role(loom_type_t type) {
  return (loom_encoding_role_t)((type.header >> 8) & 0xFF);
}

// Returns the storage space for a storage type. Only valid when
// kind == LOOM_TYPE_STORAGE.
static inline loom_storage_space_t loom_type_storage_space(loom_type_t type) {
  return (loom_storage_space_t)((type.header >> 8) & 0xFF);
}

static inline uint32_t loom_type_make_raw_header(loom_type_kind_t kind,
                                                 uint8_t payload, uint8_t rank,
                                                 uint8_t flags) {
  return ((uint32_t)kind & 0xFF) | (((uint32_t)payload & 0xFF) << 8) |
         (((uint32_t)rank & 0xF) << 16) | (((uint32_t)flags & 0xF) << 20);
}

static inline uint32_t loom_type_make_header(loom_type_kind_t kind,
                                             loom_scalar_type_t element_type,
                                             uint8_t rank, uint8_t flags) {
  return loom_type_make_raw_header(kind, element_type, rank, flags);
}

// --- Dim accessors ---

// Returns the packed dim at the given index. |index| must be less than the
// type rank.
// For inline types (rank <= 2): reads from dims[index].
// For overflow types: reads from the overflow array.
static inline uint64_t loom_type_dim(loom_type_t type, iree_host_size_t index) {
  IREE_ASSERT(index < loom_type_rank(type));
  if (IREE_LIKELY(loom_type_has_inline_dims(type))) {
    return type.dims[index];
  }
  const loom_overflow_dim_t* overflow =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  return overflow[index];
}

static inline bool loom_type_dim_is_dynamic_at(loom_type_t type,
                                               iree_host_size_t index) {
  return loom_dim_is_dynamic(loom_type_dim(type, index));
}

static inline int64_t loom_type_dim_static_size_at(loom_type_t type,
                                                   iree_host_size_t index) {
  return loom_dim_static_size(loom_type_dim(type, index));
}

static inline loom_value_id_t loom_type_dim_value_id_at(
    loom_type_t type, iree_host_size_t index) {
  return loom_dim_value_id(loom_type_dim(type, index));
}

// --- Function type accessors ---

// Returns the function type data for a LOOM_TYPE_FUNCTION type.
// The pointer is stored in dims[0]. Only valid when kind == LOOM_TYPE_FUNCTION.
static inline const loom_func_type_data_t* loom_type_func_data(
    loom_type_t type) {
  return (const loom_func_type_data_t*)(uintptr_t)type.dims[0];
}

// Returns the number of argument types in a function type.
static inline uint16_t loom_type_func_arg_count(loom_type_t type) {
  return loom_type_func_data(type)->arg_count;
}

// Returns the number of result types in a function type.
static inline uint16_t loom_type_func_result_count(loom_type_t type) {
  return loom_type_func_data(type)->result_count;
}

// Returns a pointer to the argument type array.
static inline const loom_type_t* loom_type_func_arg_types(loom_type_t type) {
  return loom_type_func_data(type)->types;
}

// Returns a pointer to the result type array.
static inline const loom_type_t* loom_type_func_result_types(loom_type_t type) {
  const loom_func_type_data_t* data = loom_type_func_data(type);
  return data->types + data->arg_count;
}

// --- Scalar type classification ---

// Returns true if the scalar type is a pure integer (I1, I8, I16, I32, I64).
// INDEX and OFFSET are address types, not pure integers — they have
// target-dependent width and sign semantics.
static inline bool loom_scalar_type_is_integer(loom_scalar_type_t type) {
  return type >= LOOM_SCALAR_TYPE_I1 && type <= LOOM_SCALAR_TYPE_I64;
}

// Returns true if the scalar type is a floating-point type
// (F8E4M3, F8E5M2, F16, BF16, F32, F64).
static inline bool loom_scalar_type_is_float(loom_scalar_type_t type) {
  return type >= LOOM_SCALAR_TYPE_F8E4M3 && type <= LOOM_SCALAR_TYPE_F64;
}

// --- Kind checks ---

static inline bool loom_type_is_scalar(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_SCALAR;
}

static inline bool loom_type_is_tile(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_TILE;
}

static inline bool loom_type_is_tensor(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_TENSOR;
}

static inline bool loom_type_is_vector(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_VECTOR;
}

static inline bool loom_type_is_view(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_VIEW;
}

static inline bool loom_type_is_buffer(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_BUFFER;
}

static inline bool loom_type_is_register(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_REGISTER;
}

static inline bool loom_type_is_storage(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_STORAGE;
}

static inline bool loom_type_is_function(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_FUNCTION;
}

static inline bool loom_type_is_encoding(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_ENCODING;
}

static inline bool loom_type_is_dialect(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_DIALECT;
}

static inline bool loom_type_is_pool(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_POOL;
}

// Returns true if the type is shaped (has rank, dims, element type):
// tile, tensor, vector, or view. Scalar types are NOT shaped.
static inline bool loom_type_is_shaped(loom_type_t type) {
  loom_type_kind_t kind = loom_type_kind(type);
  return kind == LOOM_TYPE_TILE || kind == LOOM_TYPE_TENSOR ||
         kind == LOOM_TYPE_VECTOR || kind == LOOM_TYPE_VIEW;
}

// Returns true if the type kind can carry the shared encoding/layout
// attachment slot. Vector types are shaped but intentionally layout-free.
static inline bool loom_type_can_have_encoding(loom_type_t type) {
  loom_type_kind_t kind = loom_type_kind(type);
  return kind == LOOM_TYPE_TILE || kind == LOOM_TYPE_TENSOR ||
         kind == LOOM_TYPE_VIEW;
}

// Returns true if |name| has the namespace-qualified register-class shape used
// by target-low register types, such as "amdgpu.vgpr" or "x86.zmm".
static inline bool loom_register_class_name_is_qualified(
    iree_string_view_t name) {
  iree_host_size_t dot = iree_string_view_find_char(name, '.', 0);
  return dot != IREE_STRING_VIEW_NPOS && dot > 0 && dot + 1 < name.size;
}

// --- Type comparison ---

// One-way SSA value map used when comparing types across forwarding
// boundaries.
typedef struct loom_type_value_remap_t {
  // Values appearing in the source-side type.
  const loom_value_id_t* source_values;
  // Values that the corresponding source values forward to.
  const loom_value_id_t* target_values;
  // Number of source/target value pairs.
  iree_host_size_t count;
} loom_type_value_remap_t;

// Returns true if two types have the same element type.
// Meaningful for scalar and shaped types. For non-element-bearing types
// (group, function, encoding, pool), compares the raw header byte.
static inline bool loom_type_element_type_equals(loom_type_t a, loom_type_t b) {
  return loom_type_element_type(a) == loom_type_element_type(b);
}

// Returns true if two types have the same rank.
static inline bool loom_type_rank_equals(loom_type_t a, loom_type_t b) {
  return loom_type_rank(a) == loom_type_rank(b);
}

// Returns true if two types have identical shapes (same rank and all
// dims match — static sizes and dynamic value IDs).
bool loom_type_shape_equals(loom_type_t a, loom_type_t b);

// Returns true if |type| is shaped and one dimension is statically zero.
// A static zero dimension makes the total element count zero even when other
// dimensions are dynamic.
bool loom_type_has_static_zero_extent(loom_type_t type);

// Computes the total element count for an all-static shaped type. Returns false
// for non-shaped types, dynamic shapes, and unrepresentable products.
bool loom_type_static_element_count(loom_type_t type,
                                    uint64_t* out_element_count);

// Returns true if two types are structurally equal.
//
// Shaped/pool types compare kind, element type, rank, dimensions, and
// encoding. Function types compare argument/result type sequences
// recursively. Dialect types compare the dialect type name and parameter
// list recursively.
bool loom_type_equal(loom_type_t a, loom_type_t b);

// Returns true if |source_type| equals |target_type| after applying |remap| to
// SSA value references embedded in |source_type|.
//
// This is used for region forwarding checks where a parent result type may
// reference a sibling result (`view<4xf32, %layout>`) while each region yields
// an equivalent branch-local value (`view<4xf32, %branch_layout>`). The helper
// is allocation-free and does not intern remapped types.
bool loom_type_equal_after_value_remap(loom_type_t source_type,
                                       loom_type_t target_type,
                                       const loom_type_value_remap_t* remap);

// Returns a content-based hash for |type|.
//
// Guarantee: if loom_type_equal(a, b), then
// loom_type_hash(a) == loom_type_hash(b).
uint32_t loom_type_hash(loom_type_t type);

// Callback invoked for each SSA value reference embedded in a type.
typedef iree_status_t (*loom_type_value_ref_callback_t)(
    loom_value_id_t value_id, void* user_data);

// Walks SSA value references embedded in |type|.
//
// This includes dynamic shape dimensions, dynamic pool sizes, SSA
// encoding/layout attachments, and references in nested function or dialect
// type parameters. References are reported in structural order and are not
// deduplicated: a type that mentions the same value twice emits two callbacks.
iree_status_t loom_type_walk_value_refs(loom_type_t type,
                                        loom_type_value_ref_callback_t callback,
                                        void* user_data);

// Returns true if |type| embeds a reference to |value_id|.
bool loom_type_references_value(loom_type_t type, loom_value_id_t value_id);

// Returns true if two types have the same encoding (same encoding_id
// and encoding_flags). Both types having no encoding counts as equal.
static inline bool loom_type_encoding_equals(loom_type_t a, loom_type_t b) {
  return a.encoding_id == b.encoding_id && a.encoding_flags == b.encoding_flags;
}

// Returns true if a shaped type carries any encoding (static or SSA).
//
// Static encodings have encoding_id > 0 with no flags. SSA encodings may have
// encoding_id == 0 (valid value_id), so we also check encoding_flags. Returns
// false for non-shaped types.
static inline bool loom_type_has_encoding(loom_type_t type) {
  return loom_type_can_have_encoding(type) &&
         (type.encoding_id != 0 || type.encoding_flags != 0);
}

// Returns true if the encoding is an SSA value reference rather than
// a static encoding table index. Returns false for non-shaped types.
static inline bool loom_type_has_ssa_encoding(loom_type_t type) {
  return loom_type_can_have_encoding(type) &&
         (type.encoding_flags & LOOM_ENCODING_FLAG_SSA) != 0;
}

// Returns true if the shaped type has a static (non-SSA) encoding.
static inline bool loom_type_has_static_encoding(loom_type_t type) {
  return loom_type_can_have_encoding(type) && type.encoding_id != 0 &&
         !loom_type_has_ssa_encoding(type);
}

// Returns the SSA value_id for a dynamic encoding. Only valid when
// loom_type_has_ssa_encoding() returns true.
static inline uint16_t loom_type_encoding_value_id(loom_type_t type) {
  return type.encoding_id;
}

//===----------------------------------------------------------------------===//
// Type construction and encodings
//===----------------------------------------------------------------------===//

// Creates an encoding type (the type of encoding SSA values) with |role|.
// No shape, no element type, no dims — just the kind tag and role byte.
static inline loom_type_t loom_type_encoding_with_role(
    loom_encoding_role_t role) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_ENCODING, role, 0,
                                          LOOM_TYPE_FLAG_INLINE_DIMS);
  return type;
}

// Creates an encoding type with an unspecified role.
static inline loom_type_t loom_type_encoding(void) {
  return loom_type_encoding_with_role(LOOM_ENCODING_ROLE_UNKNOWN);
}

// Creates a buffer type used as an opaque storage identity for views.
static inline loom_type_t loom_type_buffer(void) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(
      LOOM_TYPE_BUFFER, 0, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  return type;
}

// Creates a target-owned low register payload type. Target code should prefer
// loom/target/registers.h so core IR remains only the by-value storage owner.
static inline loom_type_t loom_type_register_payload(uint64_t payload0,
                                                     uint64_t payload1) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(
      LOOM_TYPE_REGISTER, 0, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  type.dims[0] = payload0;
  type.dims[1] = payload1;
  return type;
}

// Creates a function-local storage handle type.
static inline loom_type_t loom_type_storage(loom_storage_space_t space) {
  IREE_ASSERT(loom_storage_space_is_valid(space), "invalid storage space");
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(
      LOOM_TYPE_STORAGE, space, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  return type;
}

// Returns the first target-owned register payload field.
static inline uint64_t loom_type_register_payload0(loom_type_t type) {
  return type.dims[0];
}

// Returns the second target-owned register payload field.
static inline uint64_t loom_type_register_payload1(loom_type_t type) {
  return type.dims[1];
}

// Creates a pool type with a single block_size dimension.
// Uses rank=1 and stores the packed dim in dims[0], following the same
// packing as shaped types (loom_dim_pack_static / loom_dim_pack_dynamic).
static inline loom_type_t loom_type_pool(uint64_t block_size_dim) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(block_size_dim)) flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_POOL, 0, 1, flags);
  type.dims[0] = block_size_dim;
  return type;
}

// Creates a none type used as a parser/construction placeholder before the
// value's real type is assigned.
static inline loom_type_t loom_type_none(void) { return (loom_type_t){0}; }

// Creates a scalar type (rank 0, no shape, no encoding).
static inline loom_type_t loom_type_scalar(loom_scalar_type_t scalar_type) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(
      LOOM_TYPE_SCALAR, scalar_type, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  return type;
}

// Creates a rank-0 shaped type for kinds that permit scalar-shaped forms.
// Vector types are always rank >= 1 and are rejected by verifiers if a caller
// hand-constructs them with this helper.
static inline loom_type_t loom_type_shaped_0d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint16_t encoding_id) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(
      kind, element_type, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  type.encoding_id = encoding_id;
  return type;
}

// Creates a rank-1 shaped type with one inline dim.
static inline loom_type_t loom_type_shaped_1d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint64_t dim0,
                                              uint16_t encoding_id) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(dim0)) flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header = loom_type_make_header(kind, element_type, 1, flags);
  type.encoding_id = encoding_id;
  type.dims[0] = dim0;
  return type;
}

// Creates a rank-2 shaped type with two inline dims.
static inline loom_type_t loom_type_shaped_2d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint64_t dim0, uint64_t dim1,
                                              uint16_t encoding_id) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(dim0) && !loom_dim_is_dynamic(dim1))
    flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header = loom_type_make_header(kind, element_type, 2, flags);
  type.encoding_id = encoding_id;
  type.dims[0] = dim0;
  type.dims[1] = dim1;
  return type;
}

// Creates a function type from a pre-allocated loom_func_type_data_t.
// The data must be arena-allocated and outlive the type. The caller
// is responsible for populating the types[] array before use.
static inline loom_type_t loom_type_function(loom_func_type_data_t* func_data) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_FUNCTION, 0, 0, 0);
  type.dims[0] = (uint64_t)(uintptr_t)func_data;
  return type;
}

// Allocates and constructs a function type. The arg and result type
// arrays are copied into the allocated data. The allocation is owned
// by the provided allocator (typically an arena).
iree_status_t loom_type_function_build(const loom_type_t* arg_types,
                                       uint16_t arg_count,
                                       const loom_type_t* result_types,
                                       uint16_t result_count,
                                       iree_allocator_t allocator,
                                       loom_type_t* out_type);

//===----------------------------------------------------------------------===//
// Dialect type construction and accessors
//===----------------------------------------------------------------------===//
//
// Dialect types represent types from external dialects (hal.buffer,
// vm.ref<T>, etc.). They reuse the 24-byte loom_type_t layout:
//   dims[1]        → string_id (type name in the module string table)
//   encoding_id    → unused
//   encoding_flags → param_count (number of type parameters)
//   dims[0]        → pointer to loom_type_t* array (params), or 0

// Creates an opaque dialect type (no parameters). For types like
// hal.buffer that have no interior syntax.
static inline loom_type_t loom_type_dialect_opaque(loom_string_id_t name_id) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_DIALECT, 0, 0,
                                          LOOM_TYPE_FLAG_INLINE_DIMS);
  type.encoding_flags = 0;
  type.dims[1] = name_id;
  return type;
}

// Creates a parameterized dialect type. The params array must be
// arena-allocated and outlive the type.
static inline loom_type_t loom_type_dialect(loom_string_id_t name_id,
                                            uint16_t param_count,
                                            const loom_type_t* params) {
  loom_type_t type = {0};
  type.header = loom_type_make_raw_header(LOOM_TYPE_DIALECT, 0, 0,
                                          LOOM_TYPE_FLAG_INLINE_DIMS);
  type.encoding_flags = param_count;
  type.dims[0] = (uint64_t)(uintptr_t)params;
  type.dims[1] = name_id;
  return type;
}

// Returns the string_id for the dialect type name.
static inline loom_string_id_t loom_type_dialect_name_id(loom_type_t type) {
  return (loom_string_id_t)type.dims[1];
}

// Returns the number of type parameters.
static inline uint16_t loom_type_dialect_param_count(loom_type_t type) {
  return type.encoding_flags;
}

// Returns the type parameter array (may be NULL if param_count == 0).
static inline const loom_type_t* loom_type_dialect_params(loom_type_t type) {
  return (const loom_type_t*)(uintptr_t)type.dims[0];
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_TYPES_H_
