// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic operation construction.
//
// Generated dialect constructors and handwritten IR materialization share this
// builder API. The module owns semantic metadata; the builder supplies the
// allocation arena, insertion point, and optional publication callbacks.
// Operation interfaces and descriptor definitions live in op_defs.h.

#ifndef LOOM_OPS_BUILDER_H_
#define LOOM_OPS_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

// Annotation for parameters that may be NULL or zero. No-op macro that
// improves readability at call sites and in generated declarations.
#define loom_optional

// Annotation for operand parameters that may be consumed by a tied result.
// The operand's storage may be reused by a result of this op (linear
// ownership transfer). After calling the builder, the annotated operand
// must not be used if it was tied to a result.
#define loom_may_consume

#ifdef __cplusplus
extern "C" {
#endif

// A keyed SSA value used by OperandDict builders. The name is static metadata
// identifying the dictionary entry; the value remains an ordinary operand and
// participates in the normal use-def lists.
typedef struct loom_named_value_t {
  // Interned dictionary key in the destination module string table.
  loom_string_id_t name_id;
  // Reserved for stable ABI alignment and must be zero.
  uint32_t reserved;
  // SSA value ID stored in the variadic operand segment.
  loom_value_id_t value_id;
} loom_named_value_t;

// A borrowed range of named SSA values.
typedef struct loom_named_value_slice_t {
  // Borrowed entry pointer. NULL is valid only when count is zero.
  const loom_named_value_t* entries;
  // Number of entries in the borrowed range.
  iree_host_size_t count;
} loom_named_value_slice_t;

// Bundles |entries| and |count| as a named value slice.
static inline loom_named_value_slice_t loom_make_named_value_slice(
    const loom_named_value_t* entries, iree_host_size_t count) {
  loom_named_value_slice_t slice = {
      /*.entries=*/count > 0 ? entries : NULL,
      /*.count=*/count,
  };
  return slice;
}

// Saved insertion point for nested region construction. Stack-allocated.
// Use loom_builder_save/loom_builder_restore to switch between blocks.
// The parent_op field is saved/restored so that ops created inside
// nested regions get the correct ancestry on their parent_op pointer.
typedef struct loom_builder_ip_t {
  // Block to insert into.
  loom_block_t* block;
  // Op whose region contains |block|. NULL at the module level.
  // Stamped onto every op created at this insertion point as
  // op->parent_op, giving O(depth) ancestry queries.
  loom_op_t* parent_op;
  // Live op to insert before. NULL means append to the end of the block.
  loom_op_t* before_op;
} loom_builder_ip_t;

// Callback invoked after an op's direct fields are finalized. Region-owning op
// builders create their regions before invoking this callback, but callers may
// populate those regions afterward. Used by the rewriter to add newly created
// ops to its worklist.
typedef iree_status_t (*loom_builder_op_fn_t)(void* user_data, loom_op_t* op);
typedef struct loom_builder_callback_t {
  // Receives the op after its direct fields and metadata are finalized.
  loom_builder_op_fn_t fn;
  // Borrowed state retained for the builder lifetime.
  void* user_data;
} loom_builder_callback_t;

// Fired after an argument's definition, type ownership, and block ordinal are
// published. The containing region may still be under construction.
typedef struct loom_builder_argument_callback_t {
  // Receives the newly defined argument in its owning block.
  iree_status_t (*fn)(void* user_data, loom_value_id_t argument);
  // Borrowed state retained for the builder lifetime.
  void* user_data;
} loom_builder_argument_callback_t;

// The builder is the single API surface for IR construction.
//
// It bundles the target module (for value/string/type tables), the
// allocation arena, and the current insertion point into one object.
// Every generated op builder takes a loom_builder_t* as its first
// parameter. Generic helpers (inlining, cloning, pattern rewriting)
// take a builder without knowing the source or target module.
//
// The arena is separate from the module's arena so that callers can
// control storage lifetime: linking into a fresh arena, per-thread
// arenas during parallel compilation, etc.
//
// The optional on_op_finalized callback fires after finalization bookkeeping
// (uses registered, def pointers set). Nested regions may still be empty at
// this point. The callback function is NULL when unused.
typedef struct loom_builder_t {
  // Borrowed module owning the value, string, type, and semantic tables.
  loom_module_t* module;
  // Borrowed allocator whose storage must outlive the constructed IR.
  iree_arena_allocator_t* arena;
  // Current block, parent operation, and insertion position.
  loom_builder_ip_t ip;
  // Optional publication of newly finalized operations.
  loom_builder_callback_t on_op_finalized;
  // Optional publication of newly defined block arguments.
  loom_builder_argument_callback_t on_argument_defined;
  // Borrowed array of pre-allocated value IDs for the next op build. When
  // reserved_value_count > 0, loom_builder_define_value consumes from this
  // array instead of allocating new value IDs. Cleared by
  // loom_builder_finalize_op after verifying all were consumed.
  const loom_value_id_t* reserved_value_ids;
  // Number of reserved IDs in |reserved_value_ids|.
  iree_host_size_t reserved_value_count;
  // Index of the next reserved ID to consume.
  iree_host_size_t reserved_value_next;
} loom_builder_t;

// Initializes a builder that appends to |block|.
void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder);

// Moves the insertion point to before |op| and inherits |op|'s parent op.
// Used during rewrites: insert replacement ops, then erase the original.
void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to after |op| and inherits |op|'s parent op.
void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to the end of |block| (append mode).
// Does not change parent_op — use loom_builder_enter_region when
// entering a nested region to ensure correct parent ancestry.
void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block);

// Enters a nested region for building ops inside it. Saves the
// current insertion point (use loom_builder_restore to return),
// sets the insertion block to the region's entry block, and sets
// parent_op so that ops created inside inherit correct ancestry.
//
//   loom_builder_ip_t saved = loom_builder_enter_region(
//       &builder, parent_op, region);
//   // ... build ops inside region ...
//   loom_builder_restore(&builder, saved);
loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region);

// Saves the current insertion point. Returns a value that can be
// passed to loom_builder_restore to return to this position.
loom_builder_ip_t loom_builder_save(const loom_builder_t* builder);

// Restores a previously saved insertion point.
void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip);

// Pre-allocates |count| value IDs in the module's value table. The values are
// real entries with uninitialized types. The next |count| calls to
// loom_builder_define_value (typically from a generated builder) assign types
// to these values instead of allocating fresh IDs. loom_builder_finalize_op
// verifies all reserved values were consumed.
//
// Generated builders consume identities in declaration order. Function
// arguments precede function results. Independently typed region bindings
// precede operation results. Other region entry arguments are fresh
// definitions and do not consume reservations.
//
// This enables constructing result types that reference other results
// by value_id before the build call:
//
//   loom_value_id_t result_ids[2];
//   loom_builder_reserve_values(&builder, 2, result_ids);
//   loom_type_t output_type = loom_type_shaped_1d(
//       LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
//       loom_dim_pack_dynamic(result_ids[1]), 0);
//   loom_type_t result_types[] = {output_type, index_type};
//   loom_test_deflate_build(&builder, input, result_types, 2, ...);
//
iree_status_t loom_builder_reserve_values(loom_builder_t* builder,
                                          iree_host_size_t count,
                                          loom_value_id_t* out_value_ids);

// Creates a fresh value in the module's value table with the given type.
// Returns the value ID. The value has no defining op yet (set by the builder
// when the op is finalized). If values were reserved via
// loom_builder_reserve_values, consumes the next reserved ID and assigns the
// type to it.
iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id);

// Interns a string in the module's string table. Returns the string ID.
// Identical strings share the same ID.
iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id);

// Canonicalizes a keyed variadic operand dictionary.
//
// Writes the sorted operand values into |operand_storage| and writes a
// canonical DICT attribute to |out_names_attr| mapping each key to its operand
// ordinal relative to |operand_storage|. Empty dictionaries produce an absent
// names attribute so optional OperandDict fields print nothing. Input entries
// are borrowed and may appear in any order; duplicate keys are rejected. The
// module owns the final name entries, with no retained sorting scratch. Sorting
// takes linear comparisons for ordered input and O(n log n) in the worst case.
iree_status_t loom_builder_set_operand_dict(
    loom_builder_t* builder, loom_named_value_slice_t named_values,
    loom_value_id_t* operand_storage, loom_attribute_t* out_names_attr);

// Checks that a generated builder count fits the target storage width.
iree_status_t loom_builder_check_count_range(iree_host_size_t count,
                                             iree_host_size_t max_count,
                                             iree_string_view_t label);

// Copies an i64-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_i64_array_attr_storage(loom_builder_t* builder,
                                                       const int64_t* values,
                                                       iree_host_size_t count,
                                                       iree_string_view_t label,
                                                       int64_t** out_storage);

// Copies an f64-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_f64_array_attr_storage(
    loom_builder_t* builder, const double* values, iree_host_size_t count,
    const double** out_storage);

// Copies a string-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_string_array_attr_storage(
    loom_builder_t* builder, loom_string_id_array_t values,
    iree_string_view_t label, const loom_string_id_t** out_storage);

// Copies an enum-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_enum_array_attr_storage(
    loom_builder_t* builder, loom_enum_array_t values, iree_string_view_t label,
    const uint8_t** out_storage);

// Validates, canonicalizes, and copies a signed enum-set attribute payload into
// the builder arena.
iree_status_t loom_builder_copy_signed_enum_set_attr_storage(
    loom_builder_t* builder, loom_signed_enum_set_t set,
    iree_string_view_t label, const uint64_t** out_storage,
    uint16_t* out_word_count);

// Copies a symbol-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_symbol_array_attr_storage(
    loom_builder_t* builder, loom_symbol_ref_array_t values,
    iree_string_view_t label, const loom_symbol_ref_t** out_storage);

// Copies a predicate-list attribute payload into the builder arena.
iree_status_t loom_builder_copy_predicate_list_attr_storage(
    loom_builder_t* builder, const loom_predicate_t* predicates,
    iree_host_size_t count, iree_string_view_t label,
    loom_predicate_t** out_storage);

// Copies a byte attribute payload into the builder arena.
iree_status_t loom_builder_copy_bytes_attr_storage(loom_builder_t* builder,
                                                   iree_const_byte_span_t bytes,
                                                   iree_string_view_t label,
                                                   const uint8_t** out_storage);

// Creates a one-block region and installs it into |op| at |region_index|.
iree_status_t loom_builder_create_region(loom_builder_t* builder, loom_op_t* op,
                                         uint8_t region_index,
                                         loom_block_t** out_entry_block);

// Creates a fresh value with the given type and adds it as a block argument.
// Region arguments never consume identities reserved for the enclosing op's
// results, even when generated builders create regions before those results.
iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id);

// Defines one op result value and writes the result id to |out_result|.
iree_status_t loom_builder_define_result(loom_builder_t* builder,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_result);

// Defines |result_count| op result values into |result_storage|.
iree_status_t loom_builder_define_results(loom_builder_t* builder,
                                          const loom_type_t* result_types,
                                          iree_host_size_t result_count,
                                          loom_value_id_t* result_storage);

// Copies tied-result metadata into |op|.
iree_status_t loom_builder_copy_tied_results(
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_op_t* op);

// Allocates an op with the given field counts and inserts it at the
// builder's current insertion point. This is the low-level primitive
// that generated builders call. The caller fills in trailing data
// (operands, results, regions, tied results, attributes) through the
// loom_op_* accessors in ir.h. Successor storage is omitted.
iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op);

// Allocates an op with explicit successor storage and inserts it at the
// builder's current insertion point. Successor slots are semantic block
// targets used by CFG terminators; labels remain display names and parser
// syntax. The caller fills successors through loom_op_successors(op), then
// fills the ordinary trailing fields through their accessors.
iree_status_t loom_builder_allocate_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op);

// Allocates an op with segmented operand metadata. |operand_segment_counts|
// must have one entry per operand descriptor on op kinds whose vtable has
// LOOM_OP_VTABLE_SEGMENTED_OPERANDS. The counts must sum to |operand_count|.
iree_status_t loom_builder_allocate_segmented_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op);

// Allocates a segmented-operand op with explicit successor storage.
iree_status_t loom_builder_allocate_segmented_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op);

// Finalizes a newly built op: verifies reserved value consumption, registers
// operand and attribute uses, assigns result definitions, and maintains
// effective traits, symbol definitions, and semantic summaries before notifying
// the listener.
// Nested regions may still be populated after the parent builder returns.
// Called as the tail return from every generated builder:
//   return loom_builder_finalize_op(builder, *out_op);
iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_BUILDER_H_
