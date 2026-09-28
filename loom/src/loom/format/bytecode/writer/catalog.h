// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable bytecode catalogs and IR-to-wire numbering.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_
#define LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/type_index.h"
#include "loom/format/low_repr.h"
#include "loom/ir/context.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/ir.h"
#include "loom/ir/string_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a trusted native type kind to its independently versioned wire tag.
uint8_t loom_bytecode_type_kind_byte(loom_type_kind_t kind);

// Number of external strings retained without allocating a content index.
#define LOOM_BYTECODE_INLINE_EXTERNAL_STRING_CAPACITY 16u

// Operation kind registered in the bytecode operation table.
typedef struct loom_bytecode_op_entry_t {
  // Internal operation kind represented by this entry.
  loom_op_kind_t kind;
  // Dense bytecode operation-table ID.
  uint32_t writer_op_id;
  // Bytecode string-table ID naming |kind|.
  uint32_t string_writer_id;
} loom_bytecode_op_entry_t;

// One module value's state in the currently active writer-local namespace.
typedef struct loom_bytecode_value_scope_row_t {
  // Scope generation that owns this row, or zero before first use.
  uint32_t generation;
  // Wire-local number in numbering scopes; unused in membership-only scopes.
  uint32_t number;
} loom_bytecode_value_scope_row_t;

// Direct-index rows corresponding to one stable module value segment.
typedef iree_alignas(64) struct loom_bytecode_value_scope_segment_t {
  // Scope-local mappings indexed by the row within the module value segment.
  loom_bytecode_value_scope_row_t rows[LOOM_VALUE_SEGMENT_CAPACITY];
} loom_bytecode_value_scope_segment_t;

static_assert(sizeof(loom_bytecode_value_scope_segment_t) == 2048,
              "writer value scope segment must fit in arena blocks");

typedef struct loom_bytecode_global_value_list_t
    loom_bytecode_global_value_list_t;

// Number of retained global-value closures in each symbol-index segment.
#define LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY 256u

// Retained declaration-local closures for one range of module symbol IDs.
typedef iree_alignas(64) struct loom_bytecode_global_value_segment_t {
  // Closure pointers indexed by the row within the symbol-ID segment.
  loom_bytecode_global_value_list_t*
      values[LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY];
} loom_bytecode_global_value_segment_t;

static_assert(sizeof(loom_bytecode_global_value_segment_t) <= 2048,
              "writer global-value segment must fit in arena blocks");

// Number of entries in each direction of a symbol-order segment.
#define LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY 512u
#define LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT 9u
#define LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK \
  (LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY - 1u)

// Bidirectional projections for one range of symbol IDs and wire ordinals.
typedef struct loom_bytecode_symbol_order_segment_t {
  // Module symbol IDs indexed by the low bits of a wire ordinal.
  loom_symbol_id_t module_ids[LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY];
  // Wire ordinals indexed by the low bits of a module symbol ID.
  loom_symbol_id_t wire_ordinals[LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY];
} loom_bytecode_symbol_order_segment_t;

static_assert(sizeof(loom_bytecode_symbol_order_segment_t) == 2048,
              "writer symbol-order segment must fit in arena blocks");
static_assert((1u << LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT) ==
                  LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY,
              "writer symbol-order segment capacity must match its shift");

// Sequential catalog-completion facts retained until ENCODINGS emission.
// Fixed-size chunks fit the arena pool and are consumed without random lookup.
typedef struct loom_bytecode_encoding_prefix_chunk_t {
  // Next chunk in encoding order, or NULL at the end.
  struct loom_bytecode_encoding_prefix_chunk_t* next;
  // Completed static type counts before the corresponding encoding entries.
  uint32_t type_counts[128];
} loom_bytecode_encoding_prefix_chunk_t;

// First-use-ordered bytecode catalogs derived while streaming one module.
typedef struct loom_bytecode_numbering_t {
  // Module being serialized.
  const loom_module_t* module;
  // Temporary arena owning all catalog storage.
  iree_arena_allocator_t* arena;
  // File-level location mode controlling operation location references.
  loom_bytecode_location_mode_t location_mode;
  // Low representation state shared by catalog and body serialization.
  struct {
    // Stable-key codec supplied by the embedding compiler.
    loom_low_repr_environment_t environment;
    // Representation contract active while numbering one Low function.
    const loom_low_repr_descriptor_set_t* active_descriptor_set;
  } low_repr;

  // Bidirectional stable module ID and presentation-order mapping. Modules
  // fitting one segment use flat; larger modules use segments.
  union {
    // Direct projections retained by ordinary modules.
    struct {
      // Module symbol IDs indexed by presentation-ordered wire ordinal.
      loom_symbol_id_t* module_ids;
      // Presentation-ordered wire ordinals indexed by module symbol ID.
      loom_symbol_id_t* wire_ordinals;
    } flat;
    // Paired projections indexed by the high bits of either ID.
    loom_bytecode_symbol_order_segment_t** segments;
  } symbol_order;

  // Invocation-owned direct index for body-local numbering and membership.
  struct {
    // Segmented rows sharing the module value table's ID geometry.
    loom_segmented_storage_t segments;
  } value_scopes;

  // Declaration-local value closures prepared before section emission.
  struct {
    // Segmented closure pointers indexed directly by module symbol ID.
    loom_segmented_storage_t segments;
  } global_values;

  // First-use-ordered string catalog and its module/external projections.
  struct {
    // Segmented borrowed views indexed by bytecode string ID.
    loom_string_table_t table;
    // Module string ID projection.
    struct {
      // Segmented bytecode string IDs indexed by module string ID.
      loom_segmented_storage_t segments;
    } module_ids;
    // Strings absent from the source module.
    struct {
      // Borrowed views retained before the content index is needed.
      iree_string_view_t
          inline_views[LOOM_BYTECODE_INLINE_EXTERNAL_STRING_CAPACITY];
      // Writer IDs paired with |inline_views|.
      uint32_t inline_writer_ids[LOOM_BYTECODE_INLINE_EXTERNAL_STRING_CAPACITY];
      // Number of external strings in the catalog.
      iree_host_size_t count;
      // Lazily built content index after the inline tier fills.
      loom_intern_table_t index;
    } external;
  } strings;

  // Global static type catalog and scope-local dependency discovery.
  struct {
    // Bytecode type IDs indexed by module type-table index.
    uint32_t* writer_ids_by_module_index;
    // Serialization facts and dependency slices indexed by module type ID.
    loom_bytecode_type_index_t index;
    // Module type-table indices indexed by bytecode type ID.
    iree_host_size_t* module_indices_by_writer_id;
    // Number of assigned bytecode type IDs.
    iree_host_size_t count;
    // Allocated capacity of |module_indices_by_writer_id|.
    iree_host_size_t capacity;
  } types;

  // Static type completion order established by encoding parameter numbering.
  struct {
    // First chunk, consumed in encoding order by the section writer.
    loom_bytecode_encoding_prefix_chunk_t* first;
    // Current chunk receiving newly numbered encoding completion facts.
    loom_bytecode_encoding_prefix_chunk_t* last;
  } encoding_prefixes;

  // Reusable continuations for interleaved type and attribute discovery.
  struct {
    // Arena-owned suspended work; borrowed source payloads remain immutable.
    struct loom_bytecode_catalog_frame_t* frames;
    // Allocated frame capacity, reused by successive catalog roots.
    iree_host_size_t capacity;
  } traversal;

  // Reusable length-prefixed type payload storage, owned by |arena| and reset
  // between emissions. Parameter TYPE references never recursively emit it.
  loom_bytecode_buffer_t type_record_buffer;

  // First-use-ordered operation catalog.
  struct {
    // Operation entries indexed by bytecode operation ID.
    loom_bytecode_op_entry_t* values;
    // Number of assigned bytecode operation IDs.
    iree_host_size_t count;
    // Allocated capacity of |values|.
    iree_host_size_t capacity;
  } ops;
} loom_bytecode_numbering_t;

// Returns the number of assigned bytecode string IDs.
static inline iree_host_size_t loom_bytecode_numbering_string_count(
    const loom_bytecode_numbering_t* numbering) {
  return numbering->strings.table.count;
}

// Returns the string assigned to |writer_id|.
static inline iree_string_view_t loom_bytecode_numbering_string(
    const loom_bytecode_numbering_t* numbering, uint32_t writer_id) {
  return loom_string_table_get(&numbering->strings.table, writer_id);
}

// Initializes empty catalogs and the stable symbol-order projection.
iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena);

// Returns the module symbol ID assigned to |wire_ordinal|.
static inline loom_symbol_id_t loom_bytecode_module_symbol_id(
    const loom_bytecode_numbering_t* numbering, loom_symbol_id_t wire_ordinal) {
  IREE_ASSERT(wire_ordinal < numbering->module->symbols.count);
  if (IREE_LIKELY(numbering->module->symbols.count <=
                  LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY)) {
    return numbering->symbol_order.flat.module_ids[wire_ordinal];
  }
  const loom_bytecode_symbol_order_segment_t* segment =
      numbering->symbol_order
          .segments[wire_ordinal >> LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT];
  return segment
      ->module_ids[wire_ordinal & LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK];
}

// Returns the wire ordinal assigned to |module_symbol_id|.
static inline loom_symbol_id_t loom_bytecode_wire_symbol_ordinal(
    const loom_bytecode_numbering_t* numbering,
    loom_symbol_id_t module_symbol_id) {
  IREE_ASSERT(module_symbol_id < numbering->module->symbols.count);
  if (IREE_LIKELY(numbering->module->symbols.count <=
                  LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY)) {
    return numbering->symbol_order.flat.wire_ordinals[module_symbol_id];
  }
  const loom_bytecode_symbol_order_segment_t* segment =
      numbering->symbol_order
          .segments[module_symbol_id >>
                    LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT];
  return segment->wire_ordinals[module_symbol_id &
                                LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK];
}

// Interns a module-owned string into the bytecode string catalog.
iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id);

// Interns an arbitrary borrowed string into the bytecode string catalog.
iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id);

// Interns a known module type and its dependencies into the bytecode catalog.
iree_status_t loom_bytecode_numbering_intern_module_type(
    loom_bytecode_numbering_t* numbering, loom_type_id_t type_id,
    uint32_t* out_writer_id);

// Resolves a by-value source type and interns its dependencies. When non-NULL,
// |out_type_id| receives the canonical module ID for scope-local records.
iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id, loom_type_id_t* out_type_id);

// Interns the registered kind of |op| into the operation catalog.
iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id);

// Validates and returns a closed or open enum ordinal.
iree_status_t loom_bytecode_get_enum_ordinal(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_ordinal);

// Validates and returns a descriptor-backed enum array.
iree_status_t loom_bytecode_get_enum_array(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_enum_array_t* out_array);

// Validates and returns a descriptor-backed signed enum set.
iree_status_t loom_bytecode_get_signed_enum_set(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_signed_enum_set_t* out_set);

// Validates and returns a module-local symbol collection.
iree_status_t loom_bytecode_get_symbol_collection(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_ref_array_t* out_array);

// Resolves and validates one registered parameterized attribute family.
iree_status_t loom_bytecode_get_parameterized_attr(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind,
    const loom_parameterized_attr_descriptor_t** out_family_descriptor);

// Resolves whether one parameterized attribute slot is present.
iree_status_t loom_bytecode_parameter_is_present(
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    loom_attribute_t value, uint8_t parameter_index, bool* out_present);

// Resolves the representation contract selected by |func_like|.
iree_status_t loom_bytecode_resolve_function_low_descriptor_set(
    const loom_bytecode_numbering_t* numbering, loom_func_like_t func_like,
    const loom_low_repr_descriptor_set_t** out_descriptor_set);

// Numbers the strings and types owned by one attribute payload, including its
// nested aggregates. Referenced encoding payloads belong to the ordered
// encoding pass and are not traversed again at each use.
iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor);

// Numbers the catalogs owned by one static encoding payload. The writer calls
// this once per entry in module-table order before signature/body discovery.
// Construction restricts encoding parameters to prior entries, whose payloads
// have therefore already been numbered.
iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_
