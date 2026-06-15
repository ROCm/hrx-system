// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom IR graph: Module -> Region -> Block -> Operation -> Value.
//
// The IR is a lightweight, arena-allocated graph that represents loom
// programs. All IR for a module lives in a single arena: creation is
// bump-pointer allocation, destruction is freeing the arena.
//
// ==========================================================================
// Core concepts
// ==========================================================================
//
// Module:     Top-level container. Owns the arena, string table, type
//             table, value table, symbol table. One per file/compilation.
//
// Symbol:     A named module-level entity: function-like op, global,
//             executable. Flat symbol table per module (no nesting).
//             Symbol dependency/use queries are built by analysis passes; the
//             core table stores identity and definition state.
//
// Block:      A basic block within a region. Contains a linear
//             sequence of operations. May have block arguments (for
//             function entry, loop carried values, branch targets).
//
// Operation:  A single IR node: an op kind, operands (value IDs),
//             results (value IDs), attributes, regions, and optional
//             tied result bindings.
//
// Value:      An SSA value (op result or block argument). 64-byte
//             cache-line-aligned entry in the module's value table.
//             Carries the type inline (24 bytes) and up to 3 uses
//             inline (no pointer chase for the common case).
//
// Region:     A list of blocks. Used for function-like bodies, loop bodies,
//             conditional branches. Regions nest (a region's block
//             contains ops that may have their own regions).
//
// ==========================================================================
// ID-based references
// ==========================================================================
//
// Table-owned entities use integer IDs; ownership/backreference edges use
// stable arena pointers:
//   value_id   -> index into module->values.entries[]
//   symbol_id  -> index into module->symbols.entries[]
//   string_id  -> index into module->strings.entries[]
//   type_id    -> index into module->types.entries[] (for interned types)
//   use/def    -> loom_op_t* / loom_block_t* stable arena pointers
//
// Benefits: stable table references for serialization, compact scalar IDs
// where the IR already has dense tables, and direct pointers for hot local
// walks where arena allocation keeps objects stable.
//
// ==========================================================================
// Symbol references
// ==========================================================================
//
// A symbol reference is (module_id, symbol_id): 4 bytes total.
//
// In the in-memory IR, symbol references are module-local: valid refs use
// module_id = 0 and symbol_id indexes into the current module's symbol table.
// The module_id field is retained so the same payload shape can represent
// cross-module links in serialized formats and external linker metadata.
//
// ==========================================================================
// Tied results (ownership transfer)
// ==========================================================================
//
// A tied result reuses an operand's storage (linear ownership):
//   -> %operand        Same type, operand consumed.
//   -> %operand as T   Type change, operand consumed.
//   -> tensor<...>     Fresh allocation (not tied).
//
// Tied results are stored per-operation as a mapping from result
// index to the tied operand index. The operand is consumed: no uses
// after the consuming op (verified statically).
//
// Well-formed tied metadata has three invariants:
//   - Each result index appears in at most one tied-result entry.
//   - Each operand index appears in at most one tied-result entry.
//   - A tied operand's value appears in exactly one operand slot on
//     the operation, so name-based surface syntax cannot be ambiguous.

#ifndef LOOM_IR_IR_H_
#define LOOM_IR_IR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/attribute.h"
#include "loom/ir/encoding.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_context_t loom_context_t;
typedef struct loom_module_t loom_module_t;
typedef struct loom_symbol_t loom_symbol_t;
typedef struct loom_block_t loom_block_t;
typedef struct loom_op_t loom_op_t;
typedef struct loom_region_t loom_region_t;
typedef struct loom_value_t loom_value_t;
typedef struct loom_op_vtable_t loom_op_vtable_t;
typedef uint8_t loom_symbol_kind_t;
typedef struct loom_symbol_fact_domain_t loom_symbol_fact_domain_t;
typedef struct loom_symbol_definition_descriptor_t
    loom_symbol_definition_descriptor_t;
typedef struct loom_operand_descriptor_t loom_operand_descriptor_t;
typedef struct loom_result_descriptor_t loom_result_descriptor_t;
typedef struct loom_attr_descriptor_t loom_attr_descriptor_t;
typedef struct loom_region_descriptor_t loom_region_descriptor_t;
typedef struct loom_op_placement_descriptor_t loom_op_placement_descriptor_t;
typedef struct loom_format_element_t loom_format_element_t;
typedef struct loom_constraint_t loom_constraint_t;
typedef struct loom_rewriter_t loom_rewriter_t;
typedef struct loom_call_like_vtable_t loom_call_like_vtable_t;
typedef struct loom_func_like_vtable_t loom_func_like_vtable_t;
typedef struct loom_target_like_descriptor_t loom_target_like_descriptor_t;
typedef struct loom_target_like_vtable_t loom_target_like_vtable_t;
typedef struct loom_loop_like_vtable_t loom_loop_like_vtable_t;
typedef struct loom_region_branch_vtable_t loom_region_branch_vtable_t;
typedef struct loom_memory_access_vtable_t loom_memory_access_vtable_t;
typedef struct loom_type_transfer_context_t loom_type_transfer_context_t;

//===----------------------------------------------------------------------===//
// References
//===----------------------------------------------------------------------===//

// loom_value_id_t and loom_string_id_t are defined in types.h.

// Packed reference to an op field in semantic-constraint tables: 2-bit
// category (operand, result, attr, region) in the top bits, 6-bit index in the
// bottom bits. Printer callbacks use loom_print_field_ref_t instead so wide
// variadic fields can preserve full indices in diagnostic highlights.
typedef uint8_t loom_field_ref_t;

//===----------------------------------------------------------------------===//
// Use-def tracking
//===----------------------------------------------------------------------===//
//
// A use entry records "op X uses this value as operand Y." 8 bytes on
// all platforms, stored inline in the value (up to 3) or in an arena-
// allocated overflow array.
//
// On 64-bit: the op pointer and operand index are packed into a single
// uint64_t. Userspace heap pointers on x86-64 and AArch64 use at most
// 48 bits; the upper 16 bits carry the operand index. This gives direct
// pointer access with no table indirection.
//
// On 32-bit builds: the natural struct layout is 8 bytes because
// pointers are 4 bytes. No packing needed.
//
// Both representations fit 3 uses inline in the value's 24-byte union,
// preserving the 64-byte cache-line-aligned value layout.

#if defined(IREE_PTR_SIZE_64)

typedef uint64_t loom_use_t;

// Mask for extracting the 48-bit pointer from a tagged use entry.
// Validated at creation time on debug builds. Holds on all current
// x86-64, AArch64, and RISC-V 64-bit hardware. Intel LAM and ARM TBI
// do not affect untagged userspace pointers.
#define LOOM_USE_POINTER_MASK UINT64_C(0x0000FFFFFFFFFFFF)

static inline loom_use_t loom_use_make(loom_op_t* user_op,
                                       uint16_t operand_index) {
  IREE_ASSERT(((uintptr_t)user_op & ~LOOM_USE_POINTER_MASK) == 0,
              "op pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)user_op | ((uint64_t)operand_index << 48);
}

static inline loom_op_t* loom_use_user_op(loom_use_t use) {
  return (loom_op_t*)(uintptr_t)(use & LOOM_USE_POINTER_MASK);
}

static inline uint16_t loom_use_operand_index(loom_use_t use) {
  return (uint16_t)(use >> 48);
}

#else  // 32-bit

typedef struct loom_use_t {
  loom_op_t* user_op;
  uint16_t operand_index;
  uint16_t reserved;
} loom_use_t;

static inline loom_use_t loom_use_make(loom_op_t* user_op,
                                       uint16_t operand_index) {
  loom_use_t use = {user_op, operand_index, 0};
  return use;
}

static inline loom_op_t* loom_use_user_op(loom_use_t use) {
  return use.user_op;
}

static inline uint16_t loom_use_operand_index(loom_use_t use) {
  return use.operand_index;
}

#endif  // IREE_PTR_SIZE

static_assert(sizeof(loom_use_t) == 8, "loom_use_t must be 8 bytes");

// Index of a use entry within a value's use list.
typedef uint32_t loom_use_index_t;
#define LOOM_USE_INDEX_INVALID ((loom_use_index_t)UINT32_MAX)

//===----------------------------------------------------------------------===//
// Value definition pointer
//===----------------------------------------------------------------------===//
//
// A tagged pointer that records how a value was defined: either as the
// result of an operation (loom_op_t*) or as a block argument
// (loom_block_t*). The LOOM_VALUE_FLAG_BLOCK_ARG flag on the value
// disambiguates which pointer type is stored.
//
// On 64-bit: the pointer and index are packed into a single uint64_t.
// Low 48 bits hold the pointer (same constraint as loom_use_t), high
// 16 bits hold the result_index (op results) or arg_index (block args).
//
// On 32-bit builds: natural struct layout, 8 bytes.
//
// This enables O(1) pattern matching: given a value, follow the def
// pointer to the defining op and inspect its kind, operands, or
// attributes without walking the region tree.

#if defined(IREE_PTR_SIZE_64)

typedef uint64_t loom_value_def_t;

// Mask for extracting the 48-bit pointer from a def entry.
// Same constraint as loom_use_t: validated at creation on debug builds.
#define LOOM_VALUE_DEF_POINTER_MASK UINT64_C(0x0000FFFFFFFFFFFF)

// Creates a def pointer for an op result.
static inline loom_value_def_t loom_value_def_make_op(loom_op_t* op,
                                                      uint16_t result_index) {
  IREE_ASSERT(((uintptr_t)op & ~LOOM_VALUE_DEF_POINTER_MASK) == 0,
              "op pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)op | ((uint64_t)result_index << 48);
}

// Creates a def pointer for a block argument.
static inline loom_value_def_t loom_value_def_make_block(loom_block_t* block,
                                                         uint16_t arg_index) {
  IREE_ASSERT(((uintptr_t)block & ~LOOM_VALUE_DEF_POINTER_MASK) == 0,
              "block pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)block | ((uint64_t)arg_index << 48);
}

// Creates an empty def pointer before the value's defining site is wired.
static inline loom_value_def_t loom_value_def_make_none(void) { return 0; }

// Extracts the defining op pointer. Caller must check that the value
// is not a block argument. Returns NULL if the def is unset.
static inline loom_op_t* loom_def_op(loom_value_def_t def) {
  return (loom_op_t*)(uintptr_t)(def & LOOM_VALUE_DEF_POINTER_MASK);
}

// Extracts the owning block pointer. Caller must check that the value
// is a block argument. Returns NULL if the def is unset.
static inline loom_block_t* loom_def_block(loom_value_def_t def) {
  return (loom_block_t*)(uintptr_t)(def & LOOM_VALUE_DEF_POINTER_MASK);
}

// Extracts the result index (op results) or arg index (block args).
static inline uint16_t loom_def_index(loom_value_def_t def) {
  return (uint16_t)(def >> 48);
}

#else  // 32-bit

typedef struct loom_value_def_t {
  void* pointer;   // loom_op_t* or loom_block_t*.
  uint16_t index;  // result_index or arg_index.
  uint16_t reserved;
} loom_value_def_t;

static inline loom_value_def_t loom_value_def_make_op(loom_op_t* op,
                                                      uint16_t result_index) {
  loom_value_def_t def = {op, result_index, 0};
  return def;
}

static inline loom_value_def_t loom_value_def_make_block(loom_block_t* block,
                                                         uint16_t arg_index) {
  loom_value_def_t def = {block, arg_index, 0};
  return def;
}

static inline loom_value_def_t loom_value_def_make_none(void) {
  loom_value_def_t def = {NULL, 0, 0};
  return def;
}

static inline loom_op_t* loom_def_op(loom_value_def_t def) {
  return (loom_op_t*)def.pointer;
}

static inline loom_block_t* loom_def_block(loom_value_def_t def) {
  return (loom_block_t*)def.pointer;
}

static inline uint16_t loom_def_index(loom_value_def_t def) {
  return def.index;
}

#endif  // IREE_PTR_SIZE

static_assert(sizeof(loom_value_def_t) == 8,
              "loom_value_def_t must be 8 bytes");

//===----------------------------------------------------------------------===//
// Value flags
//===----------------------------------------------------------------------===//

// Maximum number of uses stored inline in a loom_value_t before
// falling back to an arena-allocated overflow array.
// 3 covers the vast majority: most op results are used 1-2 times.
#define LOOM_VALUE_INLINE_USE_COUNT 3

// Individual flag bits for loom_value_flags_t.
enum loom_value_flag_bits_e {
  // This value is a block argument (entry to a function, loop carried,
  // branch target), not an operation result. When set, the def field
  // stores a loom_block_t* (use loom_value_def_block to extract).
  LOOM_VALUE_FLAG_BLOCK_ARG = 1u << 0,

  // This value has been consumed by a tied operand (linear ownership
  // transfer). Any use of this value after the consuming op is a
  // verification error. Set by the verifier or during IR construction.
  LOOM_VALUE_FLAG_CONSUMED = 1u << 1,

  // The use list has overflowed inline storage. When set, access uses
  // through overflow_uses pointer instead of inline_uses array.
  // Check: if use_count > LOOM_VALUE_INLINE_USE_COUNT, this must be set.
  LOOM_VALUE_FLAG_OVERFLOW_USES = 1u << 2,

  // This value may be referenced by operation attributes such as predicate
  // lists or type attributes. Attribute references are not operand uses and are
  // therefore tracked separately so common SSA-only replacements can avoid a
  // full module attribute walk.
  LOOM_VALUE_FLAG_ATTRIBUTE_USES = 1u << 3,
};
typedef uint16_t loom_value_flags_t;

// Maximum representable use count. Count and flags share one 32-bit storage
// unit in loom_value_t so values can track high-fanout uses while preserving
// the 64-byte value layout and three inline uses.
#define LOOM_VALUE_USE_COUNT_BITS 28
#define LOOM_VALUE_MAX_USE_COUNT ((1u << LOOM_VALUE_USE_COUNT_BITS) - 1u)

//===----------------------------------------------------------------------===//
// Value — 64-byte cache-line-aligned
//===----------------------------------------------------------------------===//

// An SSA value: either an operation result or a block argument.
//
// Values live in the module's value table (module->values), accessed
// by loom_value_id_t. The entire value (name, type, definition site,
// and first 3 uses) fits in a single 64-byte cache line.
//
// Thread safety: values are owned by their module. Read access is safe
// from any thread when the module is immutable (during parallel
// compilation phases). Mutation requires exclusive module access.
//
// Lifetime: values are arena-allocated and live until the module is
// destroyed. Individual values are never freed. Erased operations
// leave their result values in the table (with use_count == 0).
//
// The type field is stored inline (24 bytes) to avoid a pointer chase.
// For the hot path of type checking during pattern matching, this
// means one cache-line fetch gives you everything about a value.
typedef iree_alignas(64) struct loom_value_t {
  // Interned bare SSA name (e.g., "x", "tile", "0") without sigil.
  // The printer adds '%' when emitting. Used for round-trip printing.
  // The name is a property of the value, not the type: two values can
  // have different names but the same type.
  loom_string_id_t name_id;

  // Number of operand uses of this value. When 0, the value is dead
  // (candidate for dead code elimination). Stored as a 29-bit bitfield so
  // high-fanout values scale past 64K uses without growing loom_value_t.
  uint32_t use_count : LOOM_VALUE_USE_COUNT_BITS;

  // Bitfield of loom_value_flag_bits_e. Stored in the upper 4 bits of the same
  // 32-bit unit as use_count.
  uint32_t flags : 4;

  // --- 8 bytes ---

  // Full type, stored inline. 24 bytes. No pointer chase.
  // Includes shape dims (static sizes or SSA value IDs for dynamic
  // dims), element type, and encoding reference.
  loom_type_t type;

  // --- 32 bytes ---

  // Tagged pointer to the defining site. For op results, stores the
  // loom_op_t* and result index. For block arguments, stores the
  // loom_block_t* and arg index. Use loom_value_def_op/block/index
  // to extract (checking LOOM_VALUE_FLAG_BLOCK_ARG first).
  // Set by loom_builder_finalize_op (op results) or
  // loom_block_add_arg (block arguments). Zero until set.
  loom_value_def_t def;

  // --- 40 bytes ---

  // Inline use storage (common path) or overflow pointer.
  //
  // When use_count <= LOOM_VALUE_INLINE_USE_COUNT:
  //   Uses are stored directly in inline_uses[0..use_count-1].
  //   No pointer chase, no arena allocation.
  //
  // When use_count > LOOM_VALUE_INLINE_USE_COUNT:
  //   LOOM_VALUE_FLAG_OVERFLOW_USES is set.
  //   overflow_uses points to an arena-allocated array of
  //   overflow_capacity entries. When use_count reaches
  //   overflow_capacity, a new 2x array is arena-allocated and
  //   the old one is abandoned (arena frees all at module destruction).
  union {
    loom_use_t inline_uses[LOOM_VALUE_INLINE_USE_COUNT];
    struct {
      loom_use_t* overflow_uses;
      uint32_t overflow_capacity;
      uint32_t _reserved_0;
      uint64_t _reserved_1;
    };
  };

  // --- 64 bytes ---
} loom_value_t;

static_assert(sizeof(loom_value_t) == 64, "loom_value_t must be 64 bytes");

//===----------------------------------------------------------------------===//
// Value accessors
//===----------------------------------------------------------------------===//

static inline bool loom_value_is_block_arg(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_BLOCK_ARG);
}

static inline bool loom_value_is_consumed(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_CONSUMED);
}

static inline bool loom_value_has_overflow_uses(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_OVERFLOW_USES);
}

static inline bool loom_value_has_attribute_uses(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_ATTRIBUTE_USES);
}

// Returns a const pointer to the use array (inline or overflow).
// Valid for iteration over use_count entries.
static inline const loom_use_t* loom_value_uses(const loom_value_t* value) {
  if (loom_value_has_overflow_uses(value)) {
    return value->overflow_uses;
  }
  return value->inline_uses;
}

// Returns a mutable pointer to the use array (inline or overflow).
// Used by use-list maintenance functions (add_use, remove_use).
static inline loom_use_t* loom_value_uses_mutable(loom_value_t* value) {
  if (loom_value_has_overflow_uses(value)) {
    return value->overflow_uses;
  }
  return value->inline_uses;
}

// Returns true if the value has no uses (dead, candidate for DCE).
static inline bool loom_value_has_no_uses(const loom_value_t* value) {
  return value->use_count == 0;
}

// Returns true if the value has exactly one use.
static inline bool loom_value_has_single_use(const loom_value_t* value) {
  return value->use_count == 1;
}

// Returns the operation that defines this value. The value must be an
// op result (not a block argument). Returns NULL if the def pointer
// has not been set yet (value was just created, finalize_op not called).
//
// Usage (pattern matching — "is this value defined by a constant?"):
//   loom_op_t* def_op = loom_value_def_op(value);
//   if (def_op && loom_test_constant_isa(def_op)) { ... }
static inline loom_op_t* loom_value_def_op(const loom_value_t* value) {
  IREE_ASSERT(!loom_value_is_block_arg(value));
  return loom_def_op(value->def);
}

// Returns the block that owns this block argument. The value must be
// a block argument (LOOM_VALUE_FLAG_BLOCK_ARG set).
static inline loom_block_t* loom_value_def_block(const loom_value_t* value) {
  IREE_ASSERT(loom_value_is_block_arg(value));
  return loom_def_block(value->def);
}

// Returns the result index (for op results) or argument index (for
// block arguments) of this value within its defining op or block.
static inline uint16_t loom_value_def_index(const loom_value_t* value) {
  return loom_def_index(value->def);
}

//===----------------------------------------------------------------------===//
// Tied result
//===----------------------------------------------------------------------===//

// Binding between a result and the operand whose storage it reuses.
// Stored in the operation's tied result list.
//
// The operand is consumed (linear ownership transfer). The result
// occupies the operand's buffer. If has_type_change is set, the
// result has a different type than the operand (e.g., a reshape or
// encoding change over the same storage).
typedef struct loom_tied_result_t {
  uint16_t result_index;
  uint16_t operand_index;
  bool has_type_change;
} loom_tied_result_t;

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

// Op kind: (dialect_id << 8 | op_index). The high byte identifies the
// dialect, the low byte identifies the op within the dialect.
typedef uint16_t loom_op_kind_t;

// Each dialect has a unique 8-bit ID. The high byte of loom_op_kind_t
// carries the dialect ID, the low byte the op index within that
// dialect. This gives 256 dialects x 256 ops each = 65536 total.
//
//   0x00       = unknown/invalid
//   0x01..0x7F = internal dialects (assigned by loom)
//   0x80..0xFE = external/user dialects (assigned at registration)
//   0xFF       = reserved
typedef enum loom_dialect_id_e {
  LOOM_DIALECT_UNKNOWN = 0x00,
  LOOM_DIALECT_TEST = 0x01,
  LOOM_DIALECT_SCALAR = 0x02,
  LOOM_DIALECT_TILE = 0x03,
  LOOM_DIALECT_TENSOR = 0x04,
  LOOM_DIALECT_SCF = 0x05,
  LOOM_DIALECT_FUNC = 0x06,
  LOOM_DIALECT_HAL = 0x07,
  LOOM_DIALECT_VM = 0x08,
  LOOM_DIALECT_ENCODING = 0x09,
  LOOM_DIALECT_POOL = 0x0A,
  LOOM_DIALECT_GLOBAL = 0x0B,
  LOOM_DIALECT_BUFFER = 0x0C,
  LOOM_DIALECT_VIEW = 0x0D,
  LOOM_DIALECT_VECTOR = 0x0E,
  LOOM_DIALECT_INDEX = 0x0F,
  LOOM_DIALECT_KERNEL = 0x10,
  LOOM_DIALECT_LLVMIR = 0x11,
  LOOM_DIALECT_CFG = 0x12,
  LOOM_DIALECT_TARGET = 0x13,
  LOOM_DIALECT_LOW = 0x14,
  LOOM_DIALECT_PASS = 0x15,
  LOOM_DIALECT_CHECK = 0x16,
  LOOM_DIALECT_AMDGPU = 0x17,
  LOOM_DIALECT_X86 = 0x18,
  LOOM_DIALECT_WASM = 0x19,
  LOOM_DIALECT_IREEVM = 0x1A,
  LOOM_DIALECT_SPIRV = 0x1B,
  LOOM_DIALECT_CONFIG = 0x1C,
  LOOM_DIALECT_RESERVED = 0xFF,
} loom_dialect_id_t;
#define LOOM_OP_KIND_UNKNOWN ((loom_op_kind_t)0)

// Constructs a loom_op_kind_t from a dialect ID and an op index.
// Both must fit in a uint8_t. The result is a uint16_t.
#define LOOM_OP_KIND(dialect, index) \
  ((loom_op_kind_t)(((dialect) << 8) | (index)))

// Maximum number of built-in dialects. Dialect IDs must be less than
// this value. Matches the size of the dialect vtable registry array.
#define LOOM_DIALECT_BUILTIN_COUNT_ 29

// Extracts the dialect ID (high byte) from an op kind.
static inline uint8_t loom_op_dialect_id(loom_op_kind_t kind) {
  return (uint8_t)(kind >> 8);
}

// Extracts the op index (low byte) from an op kind.
static inline uint8_t loom_op_dialect_index(loom_op_kind_t kind) {
  return (uint8_t)(kind & 0xFF);
}

enum loom_op_flag_bits_e {
  // The operation has been logically erased. Enumeration macros skip
  // dead ops. The memory is not freed (arena-owned) but the op is
  // invisible to passes and will not be serialized.
  LOOM_OP_FLAG_DEAD = 1u << 0,

  // The operation is on the rewriter's worklist. Set when added,
  // cleared when popped. Prevents duplicate entries in the worklist
  // (O(1) dedup instead of hash set lookup). Harmless if stale —
  // overwritten on the next driver invocation.
  LOOM_OP_FLAG_ON_WORKLIST = 1u << 1,

  // The operation's direct effects have been added to the transitive counters
  // on its containing region and ancestor regions. Nested operations carry
  // their own counted flags. Set when an op is finalized or use-def data is
  // recomputed, and cleared when the op is erased.
  LOOM_OP_FLAG_EFFECTS_COUNTED = 1u << 2,
};
typedef uint8_t loom_op_flags_t;

// Generic semantic trait bitfield. Op vtables carry construction defaults and
// op instances carry the effective trait word consumed by pass hot paths.
enum loom_trait_bits_e {
  LOOM_TRAIT_PURE = 1u << 0,
  LOOM_TRAIT_COMMUTATIVE = 1u << 1,
  LOOM_TRAIT_IDEMPOTENT = 1u << 2,
  LOOM_TRAIT_INVOLUTION = 1u << 3,
  LOOM_TRAIT_TERMINATOR = 1u << 4,
  LOOM_TRAIT_CONSTANT_LIKE = 1u << 5,
  LOOM_TRAIT_ELEMENTWISE = 1u << 6,
  LOOM_TRAIT_DECOMPOSABLE = 1u << 7,
  // Op defines a named symbol (function, global) rather than producing
  // SSA values. The printer omits the LHS result list (%name =) for
  // symbol-defining ops even when the op carries result values (which
  // hold return type information for the printer's ResultTypeList).
  LOOM_TRAIT_SYMBOL_DEFINE = 1u << 8,
  // Op reads from at least one resource operand (has LOOM_OPERAND_READS).
  // Derived by the generator from per-operand effect flags.
  LOOM_TRAIT_READS_MEMORY = 1u << 9,
  // Op writes to at least one resource operand (has LOOM_OPERAND_WRITES).
  // Derived by the generator from per-operand effect flags.
  LOOM_TRAIT_WRITES_MEMORY = 1u << 10,
  // Op may produce different results for identical inputs (RNG,
  // timestamps, hardware state queries). Prevents CSE and LICM but
  // does not prevent DCE — a non-deterministic op with unused results
  // and no write effects is dead.
  LOOM_TRAIT_NON_DETERMINISTIC = 1u << 11,
  // Op has dynamic effects that depend on runtime state (e.g., func.call
  // effects depend on the callee). Passes treat this conservatively as
  // both READS_MEMORY and WRITES_MEMORY.
  LOOM_TRAIT_UNKNOWN_EFFECTS = 1u << 12,
  // Op's regions cannot reference values defined outside the op.
  // Values enter the region only through block arguments. CSE and
  // other passes must not substitute inner values with outer ones.
  LOOM_TRAIT_ISOLATED_FROM_ABOVE = 1u << 13,
  // Each execution produces a result with a distinct identity, even
  // when operands and attributes are identical. Two identical allocation
  // ops must not be merged. Prevents CSE but allows DCE (an unused
  // allocation with no write effects is dead) and LICM (a
  // loop-invariant allocation can be hoisted out of the loop). Derived
  // automatically by the generator when any result has LOOM_RESULT_ALLOCATES.
  LOOM_TRAIT_UNIQUE_IDENTITY = 1u << 14,
  // Op is a compiler hint that may affect generated code quality but does not
  // affect program values, memory contents, or control flow. Ordinary DCE and
  // canonicalization preserve hints; dedicated hint-stripping passes may erase
  // them explicitly.
  LOOM_TRAIT_HINT = 1u << 15,
  // Op execution may be moved to a program point where it executes more often
  // than in the source IR. This is stronger than PURE: the op must not trap,
  // allocate a distinct identity, read runtime state, write memory, or rely on
  // being control-dependent. Common-tail motion does not need this trait when
  // every original control path already executed an equivalent op exactly once.
  LOOM_TRAIT_SAFE_TO_SPECULATE = 1u << 16,
  // Op owns the SSA references embedded in its result types. Local
  // canonicalization may rewrite those references when the rewritten result
  // value has no users that require the original type/shape/encoding identity.
  LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS = 1u << 17,
  // Op observes poison operands at a semantic boundary where poison can no
  // longer propagate as an ordinary SSA value. Examples include return-like
  // terminators and externally-visible boundary ops.
  LOOM_TRAIT_POISON_BOUNDARY = 1u << 18,
  // Op refines facts or static type information while preserving a one-to-one
  // SSA value identity between each operand and result. Source-to-target-low
  // lowering binds each result to the lowered operand instead of asking a
  // target policy to rediscover the identity op.
  LOOM_TRAIT_FACT_IDENTITY = 1u << 19,
  // Op attaches metadata or facts to operand 0 and produces one result that
  // aliases the same physical value. Extra operands are interpretation data
  // and must not force target-low storage. Source-to-target-low lowering binds
  // result 0 to lowered operand 0.
  LOOM_TRAIT_VALUE_ALIAS = 1u << 20,
  // Op must preserve the dynamic participant set of its original execution
  // site. Convergent ops may be memory-pure value transforms, but they cannot
  // be duplicated, removed, CSE'd, or moved to a different control predicate.
  LOOM_TRAIT_CONVERGENT = 1u << 21,
  // Op fact inference defines target-independent result distribution facts.
  LOOM_TRAIT_DISTRIBUTION_TRANSFER = 1u << 22,
};
typedef uint32_t loom_trait_flags_t;

// Returns true if the trait flags indicate the op may write to memory
// or has unknown effects. Works on raw bitfields to avoid redundant
// vtable lookups when the vtable is already resolved.
static inline bool loom_traits_may_write(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS)) !=
         0;
}

// Returns true if the trait flags indicate the op may observe memory or
// external state in a way that prevents pure/deterministic reasoning.
// NON_DETERMINISTIC is counted with read-like effects because CSE and purity
// checks must treat it as an observation even when it does not read memory.
static inline bool loom_traits_may_read(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS |
                    LOOM_TRAIT_NON_DETERMINISTIC)) != 0;
}

// Returns true if the trait flags indicate the op may produce different
// results for identical inputs, writes memory, or has unknown effects.
static inline bool loom_traits_has_side_effects(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS |
                    LOOM_TRAIT_NON_DETERMINISTIC)) != 0;
}

// Returns true if the trait flags indicate the op's regions cannot
// reference values defined outside the op.
static inline bool loom_traits_is_isolated(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_ISOLATED_FROM_ABOVE) != 0;
}

// Returns true if each execution of the op produces a result with a
// distinct identity. Prevents CSE but not DCE or LICM.
static inline bool loom_traits_has_unique_identity(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_UNIQUE_IDENTITY) != 0;
}

// Returns true if the trait flags declare an op safe to execute on additional
// control paths. This deliberately does not infer safety from PURE: partial
// operations such as integer division may be pure but still cannot be
// speculated without extra value facts.
static inline bool loom_traits_are_safe_to_speculate(
    loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_SAFE_TO_SPECULATE) != 0;
}

// Returns true when the op depends on the dynamic participant set at its
// execution site. This is independent of ordinary memory effects.
static inline bool loom_traits_are_convergent(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_CONVERGENT) != 0;
}

// Returns true when result types carry op-owned SSA references that local
// rewrites may retarget after separately checking downstream type-sensitive
// uses.
static inline bool loom_traits_have_refinable_result_type_refs(
    loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS) != 0;
}

// Returns true when each result preserves the identity of the operand at the
// same ordinal while carrying stronger facts or static type information.
static inline bool loom_traits_are_fact_identity(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_FACT_IDENTITY) != 0;
}

// Returns true when op fact inference defines uniform/lane-varying
// distribution facts that targets may use for value placement.
static inline bool loom_traits_have_distribution_transfer(
    loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_DISTRIBUTION_TRANSFER) != 0;
}

// Returns true when result 0 aliases operand 0 and remaining operands carry
// metadata or fact inputs that need not survive target-low lowering.
static inline bool loom_traits_are_value_alias(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_VALUE_ALIAS) != 0;
}

// Structural flags on the op vtable (shared by all instances of an op kind).
enum loom_op_vtable_flag_bits_e {
  LOOM_OP_VTABLE_VARIADIC_OPERANDS = 1u << 0,
  LOOM_OP_VTABLE_VARIADIC_RESULTS = 1u << 1,
  LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS = 1u << 2,
  LOOM_OP_VTABLE_VARIADIC_REGIONS = 1u << 3,
  LOOM_OP_VTABLE_SEGMENTED_OPERANDS = 1u << 4,
  // The op kind has generated constraints or type-transfer hooks that can
  // narrow dynamic type properties during table-driven type propagation.
  LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE = 1u << 5,
};
typedef uint8_t loom_op_vtable_flags_t;

// Compact control-flow metadata on the op vtable. These flags describe
// structural successor semantics that are cheaper to store inline than as a
// pointer-backed interface.
enum loom_op_control_flow_flag_bits_e {
  // One operand selects among this op's successor alternatives.
  LOOM_OP_CONTROL_FLOW_HAS_SUCCESSOR_SELECTOR = 1u << 0,
};
typedef uint8_t loom_op_control_flow_flags_t;

// Op-specific verification callback. Called after the standard
// table-driven checks have established the op's structural invariants.
typedef iree_status_t (*loom_op_verify_fn_t)(const loom_module_t* module,
                                             const loom_op_t* op,
                                             iree_diagnostic_emitter_t emitter);

// Canonicalization callback. Attempts to simplify |op| using the
// rewriter.
typedef iree_status_t (*loom_canonicalize_fn_t)(loom_op_t* op,
                                                loom_rewriter_t* rewriter);

// Per-instance effective traits callback. Recomputes the trait flags for a
// specific op instance when construction or mutation changes attrs/flags that
// affect generic semantics. NULL means "use the op vtable construction default
// as-is" and is the common case. Pass hot paths read loom_op_t::traits instead
// of invoking this callback.
//
// Instance flags are dialect-specific per op kind: bit 0 on func.call
// means "callee pure" but bit 0 on scalar.addf means a fast-math flag.
// The callback gives each op kind control over interpretation.
typedef loom_trait_flags_t (*loom_effective_traits_fn_t)(const loom_op_t* op);

typedef struct loom_fact_context_t loom_fact_context_t;
typedef struct loom_value_facts_t loom_value_facts_t;

// Fact inference callback. Computes output facts from operand facts for value
// analysis. Constant folding is a consumer of these facts, not part of this
// callback's contract: the rewriter may materialize constants when all result
// facts are exact.
//
// Writes one entry per result into |result_facts|. Single-result ops
// write result_facts[0]. Multi-result ops fill the entire array.
// |module| provides type access (int vs float detection, bitwidth
// queries via loom_module_value_type). Must not mutate IR.
typedef iree_status_t (*loom_op_infer_facts_fn_t)(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Semantic type-transfer callback. Feeds op-specific candidate refinements into
// the transactional type propagator. The callback must not mutate IR directly;
// it may only seed candidates through loom_type_transfer_context_t helpers.
typedef iree_status_t (*loom_type_transfer_fn_t)(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op);

//===----------------------------------------------------------------------===//
// FuncLike interface vtable
//===----------------------------------------------------------------------===//

// Sentinel values for interface vtable index fields. All interface
// vtables use 0xFF to indicate "this field is not applicable to this
// op kind" — e.g., scf.while has no induction variable, so its
// loop_like vtable sets iv_block_arg_index = LOOM_BLOCK_ARG_INDEX_NONE.
#define LOOM_ATTR_INDEX_NONE ((uint8_t)0xFF)
#define LOOM_REGION_INDEX_NONE ((uint8_t)0xFF)
#define LOOM_OPERAND_INDEX_NONE ((uint8_t)0xFF)
#define LOOM_RESULT_INDEX_NONE ((uint8_t)0xFF)
#define LOOM_BLOCK_ARG_INDEX_NONE ((uint8_t)0xFF)

//===----------------------------------------------------------------------===//
// CallLike interface vtable
//===----------------------------------------------------------------------===//

typedef uint8_t loom_call_like_kind_t;

enum loom_call_like_kind_e {
  // Invalid or non-call-like op.
  LOOM_CALL_LIKE_KIND_NONE = 0,
  // Ordinary runtime/semantic dispatch.
  LOOM_CALL_LIKE_KIND_SEMANTIC = 1,
  // Direct target-low function body to target-low function body call.
  LOOM_CALL_LIKE_KIND_LOW_INTERNAL = 2,
  // Explicit semantic-to-target-low invocation of a selected low function.
  LOOM_CALL_LIKE_KIND_LOW_INVOKE = 3,
};

// Interface descriptor for direct symbol call-like ops. The operand and result
// offsets identify trailing call argument/result slices, so generic analyses
// can read call edges without knowing dialect-specific op names.
typedef struct loom_call_like_vtable_t {
  // Index of the symbol ref attr that names the direct callee.
  uint8_t callee_attr_index;

  // Index of the optional purity enum attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t purity_attr_index;

  // Index of the optional temperature enum attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t temperature_attr_index;

  // Index of the optional inline policy enum attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t inline_policy_attr_index;

  // Operand offset where call arguments begin.
  uint8_t operand_offset;

  // Result offset where call results begin.
  uint8_t result_offset;

  // Semantic class used by analyses to opt into the call shapes they own.
  loom_call_like_kind_t kind;
} loom_call_like_vtable_t;

// Fat reference to a direct call-like op. 16 bytes, passed by value.
typedef struct loom_call_like_t {
  // Operation implementing the CallLike interface.
  loom_op_t* op;

  // Interface vtable for |op|.
  const loom_call_like_vtable_t* vtable;
} loom_call_like_t;

// Interface descriptor for function-like ops. Pure .rodata — a struct
// of attr/region indices that let generic code read the right fields
// from any implementing op without knowing its specific kind.
//
// Generated by c_tables.py from FuncLikeInterface declarations in the
// Python DSL. One instance per implementing op kind.
typedef struct loom_func_like_vtable_t {
  // Index of the symbol ref attr that names this function.
  uint8_t callee_attr_index;

  // Index of the optional import module string attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t import_module_attr_index;

  // Index of the optional import symbol string attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t import_symbol_attr_index;

  // Index of the optional target record attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t target_attr_index;

  // Index of the optional target ABI enum attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t abi_attr_index;

  // Index of the optional target ABI payload dict. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t abi_attrs_attr_index;

  // Index of the optional export symbol attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t export_symbol_attr_index;

  // Index of the optional export payload dict. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t export_attrs_attr_index;

  // Index of the optional export linkage attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t export_linkage_attr_index;

  // Index of the optional visibility enum attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t visibility_attr_index;

  // Index of the optional calling convention enum attr. LOOM_ATTR_INDEX_NONE
  // if absent.
  uint8_t cc_attr_index;

  // Index of the optional purity enum attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t purity_attr_index;

  // Index of the optional temperature enum attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t temperature_attr_index;

  // Index of the optional inline policy enum attr. LOOM_ATTR_INDEX_NONE if
  // absent.
  uint8_t inline_policy_attr_index;

  // Index of the predicate list attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t predicates_attr_index;

  // Body region index. LOOM_REGION_INDEX_NONE for bodyless ops
  // (func.decl, func.ukernel) that only declare a signature.
  uint8_t body_region_index;

  // Index of the implements string attr (for template/ukernel dispatch).
  // LOOM_ATTR_INDEX_NONE for def/decl.
  uint8_t implements_attr_index;

  // Index of the priority i64 attr (for template/ukernel dispatch).
  // LOOM_ATTR_INDEX_NONE for def/decl.
  uint8_t priority_attr_index;

  // When true, function arguments are stored as the op's operands
  // rather than as block arguments of the body region. Used for ops
  // that declare a signature without providing a body — the parser
  // stores FUNC_ARGS as operands when no REGION follows.
  bool args_as_operands;
} loom_func_like_vtable_t;

// Fat reference to a function-like op: pairs the op with its interface
// vtable. 16 bytes, passed by value. Constructed by the pass manager
// when iterating symbols, or by any code that has an op and knows it
// implements FuncLike.
typedef struct loom_func_like_t {
  loom_op_t* op;
  const loom_func_like_vtable_t* vtable;
} loom_func_like_t;

//===----------------------------------------------------------------------===//
// TargetLike interface vtable
//===----------------------------------------------------------------------===//

// Interface descriptor for ops that define target environment records. Target
// facts resolve a target symbol, cast the defining op through this interface,
// and use the target-family descriptor plus generated attr indices to project
// common and target-specific facts without switching on concrete op names.
typedef struct loom_target_like_vtable_t {
  // Index of the symbol attr that names this target record.
  uint8_t symbol_attr_index;

  // Index of the typed attr selecting the target row used as the projection
  // base. The attr kind is target-family defined.
  uint8_t selector_attr_index;

  // Index of the optional target-specific extension dictionary attr.
  // LOOM_ATTR_INDEX_NONE if absent.
  uint8_t extension_attrs_attr_index;

  // Opaque target-family projection descriptor.
  const loom_target_like_descriptor_t* descriptor;
} loom_target_like_vtable_t;

// Fat reference to a target-like op. 16 bytes, passed by value.
typedef struct loom_target_like_t {
  // Operation implementing the TargetLike interface.
  const loom_op_t* op;

  // Interface vtable for |op|.
  const loom_target_like_vtable_t* vtable;
} loom_target_like_t;

//===----------------------------------------------------------------------===//
// LoopLike interface vtable
//===----------------------------------------------------------------------===//

// Interface descriptor for loop-like ops. Lets generic passes (LICM,
// loop-invariant sinking, trip count analysis, loop transformation
// passes) operate on any iterating op without knowing its specific
// kind. All ops implementing this interface represent iteration over
// a body region with optional loop-carried state.
//
// Generated by c_tables.py from LoopLikeInterface declarations in
// the Python DSL. One instance per implementing op kind.
typedef struct loom_loop_like_vtable_t {
  // Index of the primary loop body region. For scf.for this is the
  // one body region; for scf.while this is the "after" region that
  // contains the iteration body (the "before" region holds the
  // condition).
  uint8_t body_region_index;

  // Index of the condition region, or LOOM_REGION_INDEX_NONE for
  // loops without a separate condition region. For scf.for: NONE.
  // For scf.while: 0 (the "before" region).
  uint8_t condition_region_index;

  // Index of the induction variable in the body region's entry
  // block arguments, or LOOM_BLOCK_ARG_INDEX_NONE for loops without
  // an IV. For scf.for: 0 (the implicit first block arg).
  // For scf.while: NONE (no induction variable).
  uint8_t iv_block_arg_index;

  // Operand field index carrying loop state (iter_args). Accessors resolve the
  // field to the current flat operand span so segmented operands stay hidden
  // from analyses using the LoopLike interface.
  uint8_t iter_args_operand_field_index;

  // Number of author-facing operand fields on the implementing op.
  uint8_t operand_field_count;

  // True when operand fields are backed by per-instance segment counts.
  bool segmented_operands;

  // Index of the inclusive lower-bound operand for counted loops.
  // LOOM_OPERAND_INDEX_NONE for non-counted loop forms.
  uint8_t lower_bound_operand_index;

  // Index of the exclusive upper-bound operand for counted loops.
  // LOOM_OPERAND_INDEX_NONE for non-counted loop forms.
  uint8_t upper_bound_operand_index;

  // Index of the step operand for counted loops.
  // LOOM_OPERAND_INDEX_NONE for non-counted loop forms.
  uint8_t step_operand_index;
} loom_loop_like_vtable_t;

// Fat reference to a loop-like op. 16 bytes, passed by value.
typedef struct loom_loop_like_t {
  loom_op_t* op;
  const loom_loop_like_vtable_t* vtable;
} loom_loop_like_t;

//===----------------------------------------------------------------------===//
// RegionBranch interface vtable
//===----------------------------------------------------------------------===//

// Interface descriptor for ops whose regions are mutually-exclusive
// conditional branches of a single decision. Enables generic sinking
// into branches, hoisting identical ops out of all branches, and
// constant-folding of the decision.
//
// Implementing this interface is itself the mutually-exclusive
// declaration: every region of an op that implements RegionBranch
// is an alternative of the same decision. Ops with iterating body
// regions do NOT implement this interface — their regions are
// iterated, not branched.
typedef struct loom_region_branch_vtable_t {
  // Index of the operand that drives the branch decision. For
  // scf.if this is the i1 condition; for scf.switch this is the
  // index selector. LOOM_OPERAND_INDEX_NONE is not valid — every
  // branch-like op has a selector operand.
  uint8_t selector_operand_index;
} loom_region_branch_vtable_t;

// Fat reference to a region-branch op. 16 bytes, passed by value.
typedef struct loom_region_branch_t {
  loom_op_t* op;
  const loom_region_branch_vtable_t* vtable;
} loom_region_branch_t;

//===----------------------------------------------------------------------===//
// MemoryAccess interface vtable
//===----------------------------------------------------------------------===//

// Interface descriptor for ops that access memory through a view-like operand.
// Every field is an operand or attr index resolved from MemoryAccessInterface
// declarations in the Python DSL. LOOM_*_INDEX_NONE marks roles that are not
// part of a particular op shape.
typedef struct loom_memory_access_vtable_t {
  // Index of the view or memory-object operand being accessed.
  uint8_t view_operand_index;

  // Index of the written value or atomic update contribution operand.
  uint8_t value_operand_index;

  // Index of the compare-exchange expected-value operand.
  uint8_t expected_operand_index;

  // Index of the compare-exchange replacement-value operand.
  uint8_t replacement_operand_index;

  // Index of the lane/activity mask operand.
  uint8_t mask_operand_index;

  // Index of the passthrough operand for inactive result lanes.
  uint8_t passthrough_operand_index;

  // Index of the per-lane offsets operand.
  uint8_t offsets_operand_index;

  // Operand offset of the variadic dynamic logical-origin index slice.
  uint8_t indices_operand_offset;

  // Index of the static logical-origin indices attr.
  uint8_t static_indices_attr_index;

  // Index of the optional cache/coherency-scope attr.
  uint8_t cache_scope_attr_index;

  // Index of the optional temporal cache-policy attr.
  uint8_t cache_temporal_attr_index;

  // Index of the atomic update-kind attr.
  uint8_t atomic_kind_attr_index;

  // Index of the single atomic memory-ordering attr.
  uint8_t atomic_ordering_attr_index;

  // Index of the compare-exchange success memory-ordering attr.
  uint8_t atomic_success_ordering_attr_index;

  // Index of the compare-exchange failure memory-ordering attr.
  uint8_t atomic_failure_ordering_attr_index;

  // Index of the atomic synchronization-scope attr.
  uint8_t atomic_scope_attr_index;
} loom_memory_access_vtable_t;

// Fat reference to a memory-access op. 16 bytes, passed by value.
typedef struct loom_memory_access_t {
  // Operation implementing the MemoryAccess interface.
  const loom_op_t* op;

  // Interface vtable for |op|.
  const loom_memory_access_vtable_t* vtable;
} loom_memory_access_t;

//===----------------------------------------------------------------------===//
// Op vtable
//===----------------------------------------------------------------------===//

// Per-op metadata in .rodata. One vtable per op kind.
//
// Contains everything the printer, parser, verifier, and diagnostics
// need to handle an op without op-specific code. Vtables are registered
// per-dialect at context creation time.
//
// The name field is a B-string: [total_len][ns_len]"namespace.opname\0".
// Full name = name+2 for name[0] bytes. Namespace = name+2 for
// name[1] bytes. Short name = name+2+name[1]+1 for the remainder.
// Layout is cache-optimized for construction, verification, and pass dispatch:
//
//   Cache line 1 (bytes 0-63): scalars and pointers touched by common pass
//   dispatch — construction-default traits, counts, canonicalize/fact/type-
//   transfer fn pointers, attr/operand descriptors. Effective per-instance
//   traits live on loom_op_t so descriptor-backed ops do not need vtable or
//   descriptor queries in generic pass hot paths.
//
//   Cache line 2 (bytes 64-127): verification, parse/print, and
//   diagnostics — descriptor arrays only needed by the verifier,
//   format tables only needed by the parser/printer, and the name
//   string only needed by diagnostics.
//
//   Cache line 3 (bytes 128-191): interface and placement pointers — only
//   touched by passes that query a specific interface (e.g., LICM reads
//   loop_like, call graph construction reads call_like) and by verification.
//   Each pointer is NULL for ops that don't implement that interface/contract,
//   so passes that don't use any interfaces never fetch this line. With typical
//   op counts (~200-500 kinds), the NULL pointers for the majority of ops
//   occupy .rodata address space but do not cause L1 cache misses because no
//   code path reads them.
//
// With ~500 op kinds, keeping the hot path in one cache line avoids
// ~500 × 64B = 32KB of cold .rodata fetches per pass.
struct loom_op_vtable_t {
  // --- Cache line 1: compiler pass hot path (0-63) ---

  // Construction-default traits stamped onto new op instances.
  loom_trait_flags_t traits;
  uint8_t fixed_operand_count;
  uint8_t fixed_result_count;
  uint8_t attribute_count;
  uint8_t region_count;
  loom_op_vtable_flags_t vtable_flags;
  // Legacy bytecode symbol payload kind for symbol-defining ops. Symbol
  // legality and interfaces come from |symbol_def|, not this wire tag.
  loom_symbol_kind_t symbol_kind;
  uint8_t constraint_count;
  // Number of operand descriptors when it differs from the implied count.
  // Zero uses fixed_operand_count plus the variadic operand flag.
  uint8_t operand_descriptor_count;
  // Structural control-flow semantics declared by the op kind.
  loom_op_control_flow_flags_t control_flow_flags;
  // Reserved for future compact control-flow metadata. Always zero.
  uint8_t control_flow_reserved;
  // Selector operand index for multi-successor terminators. Valid only when
  // control_flow_flags has LOOM_OP_CONTROL_FLOW_HAS_SUCCESSOR_SELECTOR.
  uint16_t successor_selector_operand_index;

  loom_canonicalize_fn_t canonicalize;
  loom_op_infer_facts_fn_t infer_facts;
  loom_effective_traits_fn_t effective_traits;
  const loom_attr_descriptor_t* attr_descriptors;
  const loom_operand_descriptor_t* operand_descriptors;
  loom_type_transfer_fn_t type_transfer;

  // --- Cache line 2: verify + parse/print + diagnostics (64-127) ---

  const loom_result_descriptor_t* result_descriptors;
  const loom_region_descriptor_t* region_descriptors;
  const loom_constraint_t* constraints;
  loom_op_verify_fn_t verify;
  const uint8_t* name;
  const loom_format_element_t* format_elements;
  const loom_bstring_t* instance_flags_case_names;
  uint16_t format_element_count;
  uint8_t instance_flags_case_count;
  // 5 bytes padding to 128.

  // --- Cache line 3: interface and placement pointers (128-191) ---
  //
  // Each pointer is NULL for ops that don't implement that interface.
  // Passes query interfaces through cast functions like
  // loom_loop_like_cast() that return {NULL, NULL} when the op does
  // not implement the interface; callers check the result with the
  // corresponding isa() helper.

  const loom_call_like_vtable_t* call_like;
  const loom_func_like_vtable_t* func_like;
  const loom_target_like_vtable_t* target_like;
  const loom_loop_like_vtable_t* loop_like;
  const loom_region_branch_vtable_t* region_branch;
  // Generated semantic contract for memory-access op roles.
  const loom_memory_access_vtable_t* memory_access;
  // Generated symbol-definition contract for SYMBOL_DEFINE ops.
  const loom_symbol_definition_descriptor_t* symbol_def;
  // Generated structural placement contract for ancestor constraints.
  const loom_op_placement_descriptor_t* placement;
};

static_assert(sizeof(loom_op_vtable_t) == 192,
              "loom_op_vtable_t must be exactly three cache lines");

static inline uint8_t loom_op_vtable_operand_descriptor_count(
    const loom_op_vtable_t* vtable) {
  if (vtable->operand_descriptor_count != 0) {
    return vtable->operand_descriptor_count;
  }
  const uint8_t variadic_count =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) ? 1 : 0;
  return (uint8_t)(vtable->fixed_operand_count + variadic_count);
}

// Returns the full dotted name as a string view (e.g., "test.addi").
static inline iree_string_view_t loom_op_vtable_name(
    const loom_op_vtable_t* vtable) {
  return iree_make_string_view((const char*)(vtable->name + 2),
                               vtable->name[0]);
}

// Returns the namespace portion of the name (e.g., "test").
static inline iree_string_view_t loom_op_vtable_namespace(
    const loom_op_vtable_t* vtable) {
  return iree_make_string_view((const char*)(vtable->name + 2),
                               vtable->name[1]);
}

// Returns the short name after the dot (e.g., "addi").
static inline iree_string_view_t loom_op_vtable_short_name(
    const loom_op_vtable_t* vtable) {
  uint8_t namespace_length = vtable->name[1];
  return iree_make_string_view(
      (const char*)(vtable->name + 2 + namespace_length + 1),
      vtable->name[0] - namespace_length - 1);
}

// An IR operation.
//
// Operations are arena-allocated within their owning block. Operands
// and results are stored as value IDs (indices into the module's value
// table) in variable-length trailing data accessed through helpers.
//
// Thread safety: operations are owned by their module. Same rules as
// loom_value_t: read-safe when module is immutable, mutation requires
// exclusive access.
//
// Lifetime: arena-allocated, lives until module destruction. Erased
// ops are flagged LOOM_OP_FLAG_DEAD but not freed.
typedef struct loom_op_t {
  // Encoded dialect/op identifier.
  loom_op_kind_t kind;
  // Number of operand value IDs in trailing storage.
  uint16_t operand_count;
  // Number of result value IDs in trailing storage.
  uint16_t result_count;
  // Number of in-place result/operand ties in trailing storage.
  uint16_t tied_result_count;
  // Effective generic semantic traits for this op instance.
  loom_trait_flags_t traits;
  // Source location (index into module's location table).
  // Carries the full source range for diagnostics and debug info.
  // Value locations are derived: value -> def -> op -> location.
  loom_location_id_t location;
  // Number of child region pointers in trailing storage.
  uint8_t region_count;
  // Number of successor block pointers in trailing storage.
  uint8_t successor_count;
  // Number of attributes in trailing storage.
  uint8_t attribute_count;
  // Per-op lifecycle/worklist flags.
  loom_op_flags_t flags;
  // Per-op-instance flags: fast-math flags for float ops, overflow
  // flags for integer ops. Declared via the Flags format element in
  // the Python DSL. Zero when the op has no flags or uses strict
  // semantics. Bit layout is dialect-specific (each flag enum defines
  // its own mask constants in the per-dialect ops.h).
  uint8_t instance_flags;
  // Monotonic position key within parent_block. Live ops in a block have
  // strictly increasing ordinals, enabling O(1) same-block order comparisons
  // without carrying mutable array indices on every insertion.
  uint64_t block_ordinal;
  // Op whose region contains this op's block. NULL for module-level
  // ops (direct children of the module body). Set during construction
  // by the builder and maintained by the rewriter on op movement.
  // Enables O(depth) ancestry queries without full IR walks.
  struct loom_op_t* parent_op;
  // Block that contains this op. NULL until inserted into a block.
  // Set by loom_builder_allocate_op and maintained by the rewriter.
  // Enables O(1) same-block checks for canonicalization patterns.
  loom_block_t* parent_block;
  // Previous live op in parent_block's ordered list. NULL for the first op.
  struct loom_op_t* prev_op;
  // Next live op in parent_block's ordered list. NULL for the last op.
  struct loom_op_t* next_op;
  // Variable-length trailing data (arena-allocated, accessed via helpers).
  // Pointer-sized fields first for natural alignment:
  //   loom_block_t*      successors[successor_count]
  //   loom_region_t*     regions[region_count]
  //   loom_value_id_t    operands[operand_count]
  //   loom_value_id_t    results[result_count]
  //   loom_use_index_t   operand_use_indices[operand_count]
  //   loom_tied_result_t tied_results[tied_result_count]
  //   <padding to alignof(loom_attribute_t)>
  //   loom_attribute_t   attributes[attribute_count]
  //   uint16_t           operand_segment_counts[operand_descriptor_count]
} loom_op_t;

static_assert(sizeof(loom_op_t) == 64, "loom_op_t must be 64 bytes");

//===----------------------------------------------------------------------===//
// Op trailing data accessors
//===----------------------------------------------------------------------===//
//
// Operations store variable-length data contiguously after the loom_op_t
// header (arena-allocated in one bump). Pointer-sized fields come first
// so they are naturally aligned after the header:
//
//   loom_block_t*      successors[successor_count] (8 bytes each, aligned)
//   loom_region_t*     regions[region_count]       (8 bytes each, aligned)
//   loom_value_id_t    operands[operand_count]     (4 bytes each)
//   loom_value_id_t    results[result_count]        (4 bytes each)
//   loom_use_index_t   operand_use_indices[operand_count]
//   loom_tied_result_t tied_results[tied_result_count]
//   <padding to alignof(loom_attribute_t)>
//   loom_attribute_t   attributes[attribute_count]  (16 bytes each)
//   uint16_t           operand_segment_counts[]     (segmented ops only)

// Returns a pointer to the successor block pointer array.
static inline loom_block_t** loom_op_successors(const loom_op_t* op) {
  return (loom_block_t**)((uint8_t*)op + sizeof(loom_op_t));
}
static inline loom_block_t* const* loom_op_const_successors(
    const loom_op_t* op) {
  return (loom_block_t* const*)loom_op_successors(op);
}

// Returns a pointer to the region pointer array (after successors).
static inline loom_region_t** loom_op_regions(const loom_op_t* op) {
  return (loom_region_t**)(loom_op_successors(op) + op->successor_count);
}

// Returns a pointer to the operand value ID array (after regions).
// The non-const variant accepts const loom_op_t* for backward
// compatibility (C trailing data doesn't propagate const). Prefer
// loom_op_const_operands when mutation is not needed.
static inline loom_value_id_t* loom_op_operands(const loom_op_t* op) {
  return (loom_value_id_t*)(loom_op_regions(op) + op->region_count);
}
static inline const loom_value_id_t* loom_op_const_operands(
    const loom_op_t* op) {
  return (const loom_value_id_t*)(loom_op_regions(op) + op->region_count);
}

// Returns a pointer to the result value ID array (after operands).
static inline loom_value_id_t* loom_op_results(const loom_op_t* op) {
  return loom_op_operands(op) + op->operand_count;
}
static inline const loom_value_id_t* loom_op_const_results(
    const loom_op_t* op) {
  return loom_op_const_operands(op) + op->operand_count;
}

// Returns a pointer to the per-operand use-list index array (after results).
// Each entry gives the operand's current position in the referenced value's use
// list, enabling O(1) use removal.
static inline loom_use_index_t* loom_op_operand_use_indices(
    const loom_op_t* op) {
  return (loom_use_index_t*)(loom_op_results(op) + op->result_count);
}

// Returns a pointer to the tied result binding array (after operand use
// indices).
static inline loom_tied_result_t* loom_op_tied_results(const loom_op_t* op) {
  return (loom_tied_result_t*)(loom_op_operand_use_indices(op) +
                               op->operand_count);
}

// Returns a pointer to the attribute array. Aligned to
// alignof(loom_attribute_t) because the union contains int64_t/double.
static inline loom_attribute_t* loom_op_attrs(const loom_op_t* op) {
  uintptr_t after_tied_results =
      (uintptr_t)(loom_op_tied_results(op) + op->tied_result_count);
  return (loom_attribute_t*)iree_host_align(after_tied_results,
                                            iree_alignof(loom_attribute_t));
}
// Returns a const pointer to the attribute array.
static inline const loom_attribute_t* loom_op_const_attrs(const loom_op_t* op) {
  return (const loom_attribute_t*)loom_op_attrs(op);
}

// Returns a pointer to the operand segment count array for segmented ops. The
// array is present only when the op vtable has
// LOOM_OP_VTABLE_SEGMENTED_OPERANDS, with one uint16_t count per operand
// descriptor. Non-segmented callers must not dereference the returned pointer.
static inline uint16_t* loom_op_operand_segment_counts(const loom_op_t* op) {
  return (uint16_t*)(loom_op_attrs(op) + op->attribute_count);
}
static inline const uint16_t* loom_op_const_operand_segment_counts(
    const loom_op_t* op) {
  return (const uint16_t*)(loom_op_attrs(op) + op->attribute_count);
}

//===----------------------------------------------------------------------===//
// Block
//===----------------------------------------------------------------------===//

// Sentinel region_index value for blocks not attached to a region.
#define LOOM_BLOCK_REGION_INDEX_INVALID UINT16_MAX

// A basic block: a linear sequence of operations with optional
// block arguments.
//
// Block arguments serve as the entry interface (function args, loop
// carried values, branch target parameters). They are stored as
// value IDs into the module's value table.
typedef struct loom_block_t {
  // Interned block label, or LOOM_STRING_ID_INVALID for anonymous blocks.
  loom_string_id_t label_id;
  // Number of block argument value IDs in arg_ids.
  uint16_t arg_count;
  // Allocated capacity of arg_ids.
  uint16_t arg_capacity;
  // Number of live operations linked into this block.
  uint32_t op_count;
  // Per-block instance flags.
  uint16_t flags;
  // Position in parent_region->blocks, or LOOM_BLOCK_REGION_INDEX_INVALID.
  uint16_t region_index;
  // Block argument value IDs.
  loom_value_id_t* arg_ids;
  // First live operation in block order.
  loom_op_t* first_op;
  // Last live operation in block order.
  loom_op_t* last_op;
  // Region that owns this block. NULL only for manually allocated blocks that
  // have not been attached to a region.
  loom_region_t* parent_region;
} loom_block_t;

static_assert(sizeof(loom_block_t) == 48, "loom_block_t must be 48 bytes");

// Returns the |arg_index|-th block argument value ID.
static inline loom_value_id_t loom_block_arg_id(const loom_block_t* block,
                                                uint16_t arg_index) {
  IREE_ASSERT(arg_index < block->arg_count);
  return block->arg_ids[arg_index];
}

// Returns the |op_index|-th operation in |block| by walking the ordered list.
// This is a cold helper for tests/diagnostics; hot paths should carry op
// pointers or use loom_block_for_each_op.
static inline const loom_op_t* loom_block_const_op(const loom_block_t* block,
                                                   iree_host_size_t op_index) {
  IREE_ASSERT(op_index < block->op_count);
  const loom_op_t* op = block->first_op;
  for (iree_host_size_t i = 0; i < op_index; ++i) {
    op = op->next_op;
  }
  return op;
}
static inline loom_op_t* loom_block_op(loom_block_t* block,
                                       iree_host_size_t op_index) {
  return (loom_op_t*)loom_block_const_op(block, op_index);
}

// Returns the terminator/last op in |block|. |block| must be non-empty.
static inline const loom_op_t* loom_block_const_last_op(
    const loom_block_t* block) {
  IREE_ASSERT(block->op_count > 0);
  return block->last_op;
}

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

// Per-instance region flags. These are stored on each loom_region_t
// and describe the runtime state of a specific region, as opposed to
// the vtable's loom_region_flag_bits_e which describe the op-kind's
// static requirements. Stored per-region (not per-op-kind) because
// the same op kind may own both structured and CFG regions at
// different points in the pipeline, and mixed IR (some functions
// structured, others CFG) is valid on input.
enum loom_region_instance_flag_bits_e {
  // Region uses unstructured control flow with explicit branch
  // terminators and successor blocks. When clear, dominance is
  // structural: entry block dominates all siblings, nesting gives
  // the dominator tree. When set, dominance must be computed from
  // predecessor edges (dominator tree construction).
  LOOM_REGION_INSTANCE_FLAG_CFG = 1u << 0,
};
typedef uint16_t loom_region_instance_flags_t;

// A region: an ordered list of blocks. Used for function bodies,
// loop bodies (scf.for), conditional branches (scf.if then/else).
// Regions nest: a region's block contains ops that may have their
// own regions.
//
// Block 0 is embedded directly in the region for the common single-block
// case and is always the entry block when block_count > 0. Additional blocks
// are arena-allocated individually and referenced through the growable
// |blocks| table, so appending blocks does not relocate existing block
// objects or invalidate loom_value_def_t / loom_op_t block pointers.
typedef struct loom_region_t {
  // Number of blocks in the region.
  uint16_t block_count;
  // Allocated capacity of the blocks pointer table.
  uint16_t block_capacity;
  // Per-region structural flags.
  loom_region_instance_flags_t flags;
  // Reserved for future flags while keeping effect counters aligned.
  uint16_t reserved;
  // Transitive count of read-like effects in all live ops nested in this
  // region. READS_MEMORY, NON_DETERMINISTIC, and UNKNOWN_EFFECTS contribute.
  uint32_t read_effect_count;
  // Transitive count of write-like effects in all live ops nested in this
  // region. WRITES_MEMORY and UNKNOWN_EFFECTS contribute.
  uint32_t write_effect_count;
  // Transitive count of convergent effects in all live ops nested in this
  // region. Convergent ops cannot be removed or moved across control structure
  // even when they are otherwise memory-pure.
  uint32_t convergent_effect_count;
  // Inline storage for the entry block.
  loom_block_t entry_block;
  // Ordered block pointer table. Points at inline_blocks for one-block regions.
  loom_block_t** blocks;
  // Inline block pointer table for the common single-block case.
  loom_block_t* inline_blocks[1];
} loom_region_t;

static_assert(sizeof(loom_region_t) == 88, "loom_region_t must be 88 bytes");

// Returns true and writes |out_block_index| when |block| is owned by |region|.
static inline bool loom_region_try_block_index(const loom_region_t* region,
                                               const loom_block_t* block,
                                               uint16_t* out_block_index) {
  if (!region || !region->blocks || !block || block->parent_region != region ||
      block->region_index >= region->block_count ||
      region->blocks[block->region_index] != block) {
    return false;
  }
  if (out_block_index) {
    *out_block_index = block->region_index;
  }
  return true;
}

// Returns |block|'s current ordinal in its parent region's block table.
static inline uint16_t loom_block_region_index(const loom_block_t* block) {
  IREE_ASSERT(block != NULL);
  IREE_ASSERT(block != NULL &&
              loom_region_try_block_index(block->parent_region, block, NULL));
  return block->region_index;
}

// Returns the block at |block_index| in |region|.
static inline loom_block_t* loom_region_block(loom_region_t* region,
                                              uint16_t block_index) {
  IREE_ASSERT(block_index < region->block_count);
  return region->blocks[block_index];
}
static inline const loom_block_t* loom_region_const_block(
    const loom_region_t* region, uint16_t block_index) {
  IREE_ASSERT(block_index < region->block_count);
  return region->blocks[block_index];
}

// Returns the entry block for |region|.
static inline loom_block_t* loom_region_entry_block(loom_region_t* region) {
  return loom_region_block(region, 0);
}
static inline const loom_block_t* loom_region_const_entry_block(
    const loom_region_t* region) {
  return loom_region_const_block(region, 0);
}

// Returns true when any live op nested in |region| has a read-like effect.
static inline bool loom_region_has_read_effects(const loom_region_t* region) {
  return region && region->read_effect_count != 0;
}

// Returns true when any live op nested in |region| has a write-like effect.
static inline bool loom_region_has_write_effects(const loom_region_t* region) {
  return region && region->write_effect_count != 0;
}

// Returns true when any live op nested in |region| has a convergent effect.
static inline bool loom_region_has_convergent_effects(
    const loom_region_t* region) {
  return region && region->convergent_effect_count != 0;
}

// Returns true when any child region of |op| has a read-like effect.
static inline bool loom_op_regions_have_read_effects(const loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_region_has_read_effects(regions[i])) return true;
  }
  return false;
}

// Returns true when any child region of |op| has a write-like effect.
static inline bool loom_op_regions_have_write_effects(const loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_region_has_write_effects(regions[i])) return true;
  }
  return false;
}

// Returns true when any child region of |op| has a convergent effect.
static inline bool loom_op_regions_have_convergent_effects(
    const loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_region_has_convergent_effects(regions[i])) return true;
  }
  return false;
}

// Returns the number of arguments on the entry block of |region|. |region|
// must be non-empty.
static inline uint16_t loom_region_entry_arg_count(
    const loom_region_t* region) {
  return loom_region_const_entry_block(region)->arg_count;
}

// Returns the |arg_index|-th argument ID from the entry block of |region|.
// |region| must be non-empty and |arg_index| must be in range.
static inline loom_value_id_t loom_region_entry_arg_id(
    const loom_region_t* region, uint16_t arg_index) {
  return loom_block_arg_id(loom_region_const_entry_block(region), arg_index);
}

//===----------------------------------------------------------------------===//
// Symbol flags and kinds
//===----------------------------------------------------------------------===//

// Symbol kind stored as a fixed-size integer for explicit control
// over struct layout. The enum defines the values.
//
// LOOM_SYMBOL_NONE (0) is the zero-initialized default for unlinked
// symbols. This is intentional: symbols are created when the parser
// first encounters a SYMBOL_REF (which may be a forward reference)
// and linked later when the SYMBOL_DEFINE op is finalized. Code that
// iterates symbols must handle NONE gracefully.
typedef enum loom_symbol_kind_e {
  // Unlinked or unknown. Zero-initialized symbols start here.
  LOOM_SYMBOL_NONE = 0,
  LOOM_SYMBOL_FUNC_DEF = 1,
  LOOM_SYMBOL_FUNC_DECL = 2,
  LOOM_SYMBOL_FUNC_TEMPLATE = 3,
  LOOM_SYMBOL_FUNC_UKERNEL = 4,
  // Sentinel: first non-function-like symbol kind.
  LOOM_SYMBOL_FUNC_COUNT_ = 5,
  LOOM_SYMBOL_GLOBAL = 5,
  LOOM_SYMBOL_EXECUTABLE = 6,
  LOOM_SYMBOL_RECORD = 7,
  LOOM_SYMBOL_COUNT_,
} loom_symbol_kind_e;

// Returns true if the symbol kind is a function-like (def, decl,
// template, or ukernel). Function-like symbols carry a defining_op
// pointer to the op that implements them.
static inline bool loom_symbol_kind_is_function_like(loom_symbol_kind_t kind) {
  return kind >= LOOM_SYMBOL_FUNC_DEF && kind < LOOM_SYMBOL_FUNC_COUNT_;
}

enum loom_symbol_flag_bits_e {
  // Symbol is visible outside the module (exported for linking).
  LOOM_SYMBOL_FLAG_PUBLIC = 1u << 0,
};
typedef uint16_t loom_symbol_flags_t;

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

// A module-level named symbol.
//
// The symbol table stores identity and link state only. The shape of the symbol
// is described by the defining op's generated symbol-definition descriptor,
// which lets dialects add new symbol families without extending a central IR
// enum. The |kind| field is retained as the existing bytecode wire payload tag
// for serialized function/global symbols; it is not the verifier's source of
// symbol-reference legality.
typedef struct loom_symbol_t {
  // Interned symbol name in module->strings.
  loom_string_id_t name_id;
  // Legacy bytecode symbol payload kind derived from |definition|.
  loom_symbol_kind_t kind;
  // Visibility and link-state flags.
  loom_symbol_flags_t flags;
  // Generated symbol contract implemented by |defining_op|.
  const loom_symbol_definition_descriptor_t* definition;
  // The op that defines this symbol, or NULL until a forward reference is
  // linked to its definition.
  loom_op_t* defining_op;
} loom_symbol_t;

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

// Interned string table. All strings in a module are deduplicated here.
// String IDs are stable across the module's lifetime.
//
// Lookup by content (for interning during construction) uses a hash
// map. Lookup by ID (for printing) is a direct array index.
typedef struct loom_string_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_string_view_t* entries;
} loom_string_table_t;

// Value table. Cache-line-aligned entries for fast iteration.
// All values (op results and block arguments) in a module live here.
typedef struct loom_value_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  // 64-byte aligned value entries.
  loom_value_t* entries;
} loom_value_table_t;

// Active state for module value-id indexed u32 scratch storage.
typedef enum loom_value_u32_scratch_state_e {
  // Scratch contents are not initialized for a specific typed user.
  LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_UNKNOWN = 0,
  // Scratch contents are initialized as value ordinals and not acquired.
  LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS = 1,
  // Scratch contents were last used as zero-initialized payloads and are not
  // acquired.
  LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ZEROED = 2,
  // Scratch is acquired as value ordinals by one compiler frame.
  LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ORDINALS = 3,
  // Scratch is acquired as zero-initialized u32 payloads by one frame.
  LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED = 4,
} loom_value_u32_scratch_state_t;

// Compiler scratch mapping module value IDs to one active u32 payload.
//
// The entries mirror the value table capacity and are reused by phase frames
// that need value-id keyed scratch. The table does not describe semantic IR
// state: value IDs remain the durable identity, while payloads are transient
// facts assigned by the active frame. Only one typed frame may acquire the
// table at a time.
typedef struct loom_value_u32_scratch_t {
  // Dense value_id -> typed u32 payload entries.
  uint32_t* values_by_value_id;
  // Number of entries allocated in values_by_value_id.
  iree_host_size_t capacity;
  // Active ownership and payload interpretation state.
  loom_value_u32_scratch_state_t state;
} loom_value_u32_scratch_t;

// Reusable non-semantic compiler scratch owned by a module.
typedef struct loom_module_scratch_t {
  // Exclusive value-id indexed u32 scratch storage.
  loom_value_u32_scratch_t values;
} loom_module_scratch_t;

// Symbol table. Flat (no nesting), one per module.
typedef struct loom_symbol_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_symbol_t* entries;
} loom_symbol_table_t;

// Type table. Interned types for pointer-equality comparison.
typedef struct loom_type_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_type_t* entries;
} loom_type_table_t;

// Index into the module's type-use record table.
typedef uint32_t loom_type_use_id_t;
#define LOOM_TYPE_USE_ID_INVALID ((loom_type_use_id_t)UINT32_MAX)

// Per-value heads for the type-use adjacency lists.
typedef struct loom_value_type_use_heads_t {
  // First record whose referenced_value_id is this value.
  loom_type_use_id_t first_incoming_use_id;
  // First record whose user_value_id is this value.
  loom_type_use_id_t first_outgoing_use_id;
} loom_value_type_use_heads_t;

// A reference from one SSA value's type to another SSA value.
//
// Type uses are not operands: they describe symbolic type structure such as
// dynamic dimensions and SSA encodings. They still participate in liveness and
// RAUW because printed/serialized types contain the referenced SSA names.
typedef struct loom_type_use_t {
  // SSA value referenced by a type payload.
  loom_value_id_t referenced_value_id;
  // SSA value whose type carries the reference.
  loom_value_id_t user_value_id;
  // Next record in referenced_value_id's incoming list.
  loom_type_use_id_t next_incoming_use_id;
  // Previous record in referenced_value_id's incoming list.
  loom_type_use_id_t previous_incoming_use_id;
  // Next record in user_value_id's outgoing list.
  loom_type_use_id_t next_outgoing_use_id;
  // Previous record in user_value_id's outgoing list.
  loom_type_use_id_t previous_outgoing_use_id;
} loom_type_use_t;

// Dense side metadata for SSA references embedded in value types.
typedef struct loom_type_use_table_t {
  // Number of per-value head entries allocated in value_heads.
  iree_host_size_t value_capacity;
  // Dense per-value incoming/outgoing list heads, indexed by value ID.
  loom_value_type_use_heads_t* value_heads;
  // Number of record slots ever allocated from records.
  iree_host_size_t record_count;
  // Number of record slots allocated in records.
  iree_host_size_t record_capacity;
  // Number of currently active type-use records.
  iree_host_size_t active_count;
  // Number of inactive record slots linked through first_free_use_id.
  iree_host_size_t free_count;
  // First inactive record slot available for reuse.
  loom_type_use_id_t first_free_use_id;
  // Sparse type-use records, indexed by loom_type_use_id_t.
  loom_type_use_t* records;
} loom_type_use_table_t;

// Kind of IR object that owns source text comments.
typedef enum loom_comment_owner_kind_e {
  LOOM_COMMENT_OWNER_OP = 0,
  LOOM_COMMENT_OWNER_BLOCK = 1,
} loom_comment_owner_kind_t;

// Leading comments attached to one operation or block label.
typedef struct loom_comment_attachment_t {
  // Operation or block pointer that owns the leading comments.
  const void* owner;
  // Kind tag describing the pointer stored in owner.
  loom_comment_owner_kind_t owner_kind;
  // Reserved padding for pointer alignment.
  uint16_t reserved;
  // Number of line comments stored in comments.
  uint16_t comment_count;
  // Module-arena-owned post-// comment payloads in source order.
  iree_string_view_t* comments;
} loom_comment_attachment_t;

// Cold side table for source comments preserved by text parsing/printing.
typedef struct loom_comment_table_t {
  // Number of comment attachments stored in entries.
  iree_host_size_t count;
  // Allocated capacity of entries.
  iree_host_size_t capacity;
  // Sparse source-comment attachments keyed by IR object pointer.
  loom_comment_attachment_t* entries;
} loom_comment_table_t;

// Open-addressing hash table for deduplicating strings and types during
// construction. Arena-allocated, freed when the module is destroyed.
// Lazy-initialized: capacity 0 means uninitialized, first use allocates.
typedef struct loom_intern_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  uint32_t* hashes;
  uint32_t* indices;
} loom_intern_table_t;

//===----------------------------------------------------------------------===//
// Module flags
//===----------------------------------------------------------------------===//

enum loom_module_flag_bits_e {
  // Declaration-only module: contains function signatures and type
  // definitions but no function bodies. Used for fast symbol
  // resolution during linking without parsing the bulk IR.
  LOOM_MODULE_FLAG_DECLARATION = 1u << 0,
};
typedef uint16_t loom_module_flags_t;

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

// A loom module: the top-level IR container.
//
// Owns all IR through an arena allocator. Creating a module allocates
// the arena. Destroying the module frees the arena and all IR within
// it in O(1) time. No individual deallocation of IR nodes.
//
// Thread safety: a module is single-owner. During parallel compilation
// phases, the module is immutable (const access from worker threads).
// Mutation (adding ops, modifying values) requires exclusive access.
// The module is NOT thread-safe for concurrent mutation.
//
// Lifetime: the module owns everything referenced through its tables.
// Values, operations, blocks, regions, strings, types: all arena-
// allocated, all freed when the module is destroyed. Pointers into
// the module's IR are invalid after module destruction.
typedef struct loom_module_t {
  // Flags (accessed frequently, placed first for cache locality).
  loom_module_flags_t flags;
  uint16_t reserved;

  // Module name. Every module is named (required for linking).
  loom_string_id_t name_id;

  // Owning context (provides vtables for op resolution).
  loom_context_t* context;

  // Allocator used to allocate and free the module struct itself.
  iree_allocator_t allocator;

  // Arena backing all IR storage. Bump-pointer allocation during
  // construction. O(1) destruction. Grows in large blocks (~64KB).
  iree_arena_allocator_t arena;

  // Interned strings (SSA names, function names, attribute keys).
  loom_string_table_t strings;

  // Interned types. Pointer equality works for interned types.
  loom_type_table_t types;

  // Encoding instances (e.g., q8_0 with block=32). Indexed 1-based
  // by the encoding_id field in loom_type_t (0 = no encoding).
  loom_encoding_table_t encodings;

  // All SSA values (op results and block arguments).
  // 64-byte aligned for cache-line access.
  loom_value_table_t values;

  // Reusable compiler scratch indexed by value ID.
  loom_module_scratch_t scratch;

  // SSA references carried by value types.
  loom_type_use_table_t type_uses;

  // Module-level named symbols (functions, globals, executables).
  loom_symbol_table_t symbols;

  // Source identifiers referenced by this module's locations.
  loom_source_table_t sources;

  // Source locations for diagnostics, debug info, and source mapping.
  // Entry 0 is always LOOM_LOCATION_NONE (unknown/absent). Ops
  // reference entries by loom_location_id_t index. When locations are
  // stripped (e.g., release builds), the table is empty and all ops
  // reference LOOM_LOCATION_UNKNOWN (0).
  loom_location_table_t locations;

  // Source comments attached to operations and explicit block labels.
  loom_comment_table_t comments;

  // Module body: a region with a single entry block. All top-level symbol ops
  // and other module-scope ops live in this block. Created automatically by
  // loom_module_allocate().
  loom_region_t* body;

  // Intern hash tables for deduplicating strings and types during
  // construction. Arena-allocated, lazy-initialized on first use.
  loom_intern_table_t string_intern;
  loom_intern_table_t type_intern;
} loom_module_t;

//===----------------------------------------------------------------------===//
// Enumeration macros
//===----------------------------------------------------------------------===//
//
// Linux kernel-style iteration macros for walking IR structures.
// Each macro declares a typed loop variable and handles skipping
// dead/invalid entries. Stack-allocated, no heap allocation.
//
// Usage:
//
//   loom_op_t* op = NULL;
//   loom_block_for_each_op(block, op) {
//     if (loom_op_kind(op) == MY_OP_KIND) {
//       // Process op.
//     }
//   }
//
//   const loom_use_t* use = NULL;
//   loom_value_for_each_use(value, use) {
//     loom_op_t* user = loom_use_user_op(*use);
//     // Process user.
//   }

// Walk all live operations in a block. Erased ops are unlinked from the list.
#define loom_block_for_each_op(block, op_var) \
  for ((op_var) = (block)->first_op; (op_var); (op_var) = (op_var)->next_op)

// Walk all blocks in a region.
#define loom_region_for_each_block(region, block_var)                         \
  for (iree_host_size_t _blk_i = 0; _blk_i < (region)->block_count; ++_blk_i) \
    if (((block_var) = (region)->blocks[_blk_i]))

// Walk all uses of a value.
#define loom_value_for_each_use(value, use_var)                            \
  for (iree_host_size_t _use_i = 0; _use_i < (value)->use_count; ++_use_i) \
    if (((use_var) = &loom_value_uses(value)[_use_i]))

// Walk all symbols in a module.
#define loom_module_for_each_symbol(module, sym_var)                  \
  for (iree_host_size_t _sym_i = 0; _sym_i < (module)->symbols.count; \
       ++_sym_i)                                                      \
    if (((sym_var) = &(module)->symbols.entries[_sym_i]))

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_IR_H_
