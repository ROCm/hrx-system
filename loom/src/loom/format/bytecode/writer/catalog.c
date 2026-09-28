// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/catalog.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"

uint8_t loom_bytecode_type_kind_byte(loom_type_kind_t kind) {
  switch (kind) {
    case LOOM_TYPE_NONE:
      return LOOM_BYTECODE_TYPE_NONE;
    case LOOM_TYPE_SCALAR:
      return LOOM_BYTECODE_TYPE_SCALAR;
    case LOOM_TYPE_TILE:
      return LOOM_BYTECODE_TYPE_TILE;
    case LOOM_TYPE_TENSOR:
      return LOOM_BYTECODE_TYPE_TENSOR;
    case LOOM_TYPE_VECTOR:
      return LOOM_BYTECODE_TYPE_VECTOR;
    case LOOM_TYPE_VIEW:
      return LOOM_BYTECODE_TYPE_VIEW;
    case LOOM_TYPE_BUFFER:
      return LOOM_BYTECODE_TYPE_BUFFER;
    case LOOM_TYPE_FUNCTION:
      return LOOM_BYTECODE_TYPE_FUNCTION;
    case LOOM_TYPE_DIALECT:
      return LOOM_BYTECODE_TYPE_DIALECT;
    case LOOM_TYPE_REGISTER:
      return LOOM_BYTECODE_TYPE_REGISTER;
    case LOOM_TYPE_STORAGE:
      return LOOM_BYTECODE_TYPE_STORAGE;
    case LOOM_TYPE_PARAMETERIZED:
      return LOOM_BYTECODE_TYPE_PARAMETERIZED;
    case LOOM_TYPE_ENCODING:
      return LOOM_BYTECODE_TYPE_ENCODING;
    case LOOM_TYPE_POOL:
      return LOOM_BYTECODE_TYPE_POOL;
    default:
      IREE_ASSERT_UNREACHABLE("verified native type kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

//===----------------------------------------------------------------------===//
// Numbering context
//===----------------------------------------------------------------------===//

// Sentinel for "not yet assigned" in mapping arrays.
#define LOOM_WRITER_ID_NONE UINT32_MAX

// Number of module-to-writer string IDs in each projection segment.
#define LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY 512u
#define LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_SHIFT 9u
#define LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_MASK \
  (LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY - 1u)

// Maximum module string count cheaper to scan than hash for borrowed views.
#define LOOM_BYTECODE_LINEAR_MODULE_STRING_LIMIT 16u

static_assert((1u << LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_SHIFT) ==
                  LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY,
              "writer string ID segment capacity must match its index shift");
static_assert((uint64_t)LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY *
                      LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT >=
                  UINT32_MAX,
              "writer string ID storage must cover the module ID domain");

// Ensures that one additional borrowed string view can be published.
static iree_status_t loom_bytecode_numbering_reserve_string(
    loom_bytecode_numbering_t* numbering) {
  loom_string_table_t* table = &numbering->strings.table;
  if (table->count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode string count exceeds format maximum "
                            "(16M)");
  }
  if (table->count == loom_string_table_capacity(table)) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &table->segments, numbering->arena, &segment));
  }
  return iree_ok_status();
}

// Publishes one view after all fallible capacity preparation is complete.
static uint32_t loom_bytecode_numbering_publish_string(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view) {
  loom_string_table_t* table = &numbering->strings.table;
  IREE_ASSERT(table->count < (1u << 24));
  const uint32_t writer_id = (uint32_t)table->count;
  loom_string_segment_t* segment =
      (loom_string_segment_t*)loom_segmented_storage_segment(
          &table->segments, writer_id >> LOOM_STRING_SEGMENT_SHIFT);
  segment->entries[writer_id & LOOM_STRING_SEGMENT_MASK] = view;
  ++table->count;
  return writer_id;
}

// Appends one borrowed view to the ordered string catalog.
static iree_status_t loom_bytecode_numbering_append_string(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_reserve_string(numbering));
  *out_writer_id = loom_bytecode_numbering_publish_string(numbering, view);
  return iree_ok_status();
}

// Returns the module-to-writer projection entry for |string_id|.
static uint32_t* loom_bytecode_numbering_module_string_writer_id(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id) {
  IREE_ASSERT(string_id < numbering->module->strings.count);
  const uint32_t segment_index =
      string_id >> LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_SHIFT;
  uint32_t* segment = (uint32_t*)loom_segmented_storage_segment(
      &numbering->strings.module_ids.segments, segment_index);
  return &segment[string_id & LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_MASK];
}

typedef struct loom_bytecode_external_string_equal_context_t {
  // Numbering context owning the external string index and catalog.
  const loom_bytecode_numbering_t* numbering;
  // Candidate borrowed string contents.
  iree_string_view_t view;
} loom_bytecode_external_string_equal_context_t;

// Compares borrowed spellings without rereading identical static storage.
static inline bool loom_bytecode_borrowed_string_equal(iree_string_view_t lhs,
                                                       iree_string_view_t rhs) {
  if (lhs.data == rhs.data) {
    return lhs.size == rhs.size;
  }
  return iree_string_view_equal(lhs, rhs);
}

// Compares one candidate with an external catalog entry by writer ID.
static bool loom_bytecode_external_string_equal(const void* context,
                                                uint32_t writer_id) {
  const loom_bytecode_external_string_equal_context_t* equal_context =
      (const loom_bytecode_external_string_equal_context_t*)context;
  return loom_bytecode_borrowed_string_equal(
      loom_bytecode_numbering_string(equal_context->numbering, writer_id),
      equal_context->view);
}

// Returns a process-local content hash for an external borrowed string.
static uint32_t loom_bytecode_external_string_hash(iree_string_view_t view) {
  return loom_structural_hash_finalize(loom_structural_hash_mix_bytes(
      loom_structural_hash_initialize(), view.data, view.size));
}

// Finds a module-owned spelling with bounded small-table linear lookup.
static loom_string_id_t loom_bytecode_numbering_lookup_module_string(
    const loom_bytecode_numbering_t* numbering, iree_string_view_t view) {
  const loom_string_table_t* table = &numbering->module->strings;
  if (table->count <= LOOM_BYTECODE_LINEAR_MODULE_STRING_LIMIT) {
    for (loom_string_id_t string_id = 0; string_id < table->count;
         ++string_id) {
      if (iree_string_view_equal(loom_string_table_get(table, string_id),
                                 view)) {
        return string_id;
      }
    }
    return LOOM_STRING_ID_INVALID;
  }
  return loom_module_lookup_string(numbering->module, view);
}

// Publishes one entry in both directions of the symbol-order projection.
static void loom_bytecode_numbering_publish_symbol_order(
    loom_bytecode_numbering_t* numbering, loom_symbol_id_t module_symbol_id,
    loom_symbol_id_t wire_ordinal) {
  if (IREE_LIKELY(numbering->module->symbols.count <=
                  LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY)) {
    numbering->symbol_order.flat.module_ids[wire_ordinal] = module_symbol_id;
    numbering->symbol_order.flat.wire_ordinals[module_symbol_id] = wire_ordinal;
    return;
  }
  loom_bytecode_symbol_order_segment_t* wire_segment =
      numbering->symbol_order
          .segments[wire_ordinal >> LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT];
  wire_segment
      ->module_ids[wire_ordinal & LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK] =
      module_symbol_id;
  loom_bytecode_symbol_order_segment_t* module_segment =
      numbering->symbol_order
          .segments[module_symbol_id >>
                    LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT];
  module_segment->wire_ordinals[module_symbol_id &
                                LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK] =
      wire_ordinal;
}

// Builds the stable module-ID to presentation-ordered wire-ordinal mapping.
static iree_status_t loom_bytecode_numbering_initialize_symbol_order(
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  if (module->symbols.count <= LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY) {
    // Preserve one direct projection for ordinary modules. The complete pair
    // fits in one segment-sized arena allocation.
    loom_symbol_id_t* storage = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(numbering->arena, module->symbols.count,
                                  2 * sizeof(*storage), (void**)&storage));
    numbering->symbol_order.flat.module_ids = storage;
    numbering->symbol_order.flat.wire_ordinals =
        storage + module->symbols.count;
    memset(numbering->symbol_order.flat.wire_ordinals, 0xFF,
           module->symbols.count * sizeof(*storage));
  } else {
    // Both directions share pool-fitting segments. Symbol IDs are bounded
    // below LOOM_SYMBOL_ID_INVALID by module construction.
    const iree_host_size_t segment_count =
        (module->symbols.count + LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_MASK) >>
        LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_SHIFT;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(numbering->arena, segment_count,
                                  sizeof(*numbering->symbol_order.segments),
                                  (void**)&numbering->symbol_order.segments));
    for (iree_host_size_t i = 0; i < segment_count; ++i) {
      loom_bytecode_symbol_order_segment_t* segment = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          numbering->arena, sizeof(*segment), (void**)&segment));
      numbering->symbol_order.segments[i] = segment;
      memset(segment->wire_ordinals, 0xFF, sizeof(segment->wire_ordinals));
    }
  }

  loom_symbol_id_t wire_ordinal = 0;
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    const loom_symbol_id_t symbol_id =
        loom_op_defining_symbol_id(module, op, loom_op_vtable(module, op));
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    if (module->symbols.entries[symbol_id].defining_op != op) {
      continue;
    }
    IREE_ASSERT_EQ(loom_bytecode_wire_symbol_ordinal(numbering, symbol_id),
                   LOOM_SYMBOL_ID_INVALID);
    loom_bytecode_numbering_publish_symbol_order(numbering, symbol_id,
                                                 wire_ordinal);
    ++wire_ordinal;
  }

  // Symbols without a live top-level defining op have no physical presentation
  // anchor. Preserve their stable module-table order after all definitions.
  for (loom_symbol_id_t module_symbol_id = 0;
       module_symbol_id < module->symbols.count; ++module_symbol_id) {
    if (loom_bytecode_wire_symbol_ordinal(numbering, module_symbol_id) !=
        LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_bytecode_numbering_publish_symbol_order(numbering, module_symbol_id,
                                                 wire_ordinal);
    ++wire_ordinal;
  }
  IREE_ASSERT_EQ(wire_ordinal, module->symbols.count);
  return iree_ok_status();
}

// Initializes the numbering context. All allocations come from |arena|,
// which the caller owns. No individual frees needed — the arena handles
// bulk deallocation.
iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena) {
  memset(numbering, 0, sizeof(*numbering));
  numbering->module = module;
  numbering->arena = arena;
  loom_segmented_storage_initialize(
      sizeof(loom_bytecode_value_scope_segment_t),
      iree_alignof(loom_bytecode_value_scope_segment_t),
      &numbering->value_scopes.segments);
  loom_segmented_storage_initialize(
      sizeof(loom_bytecode_global_value_segment_t),
      iree_alignof(loom_bytecode_global_value_segment_t),
      &numbering->global_values.segments);
  loom_segmented_storage_initialize(sizeof(loom_string_segment_t),
                                    iree_alignof(loom_string_segment_t),
                                    &numbering->strings.table.segments);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY * sizeof(uint32_t),
      iree_alignof(uint32_t), &numbering->strings.module_ids.segments);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_initialize_symbol_order(numbering));

  // Module string map: segmented direct index for module ID to writer ID.
  const iree_host_size_t module_string_segment_count =
      (module->strings.count + LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_MASK) >>
      LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_SHIFT;
  for (iree_host_size_t i = 0; i < module_string_segment_count; ++i) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &numbering->strings.module_ids.segments, arena, &segment));
    memset(segment, 0xFF,
           LOOM_BYTECODE_MODULE_STRING_ID_SEGMENT_CAPACITY * sizeof(uint32_t));
  }
  // Writer string id 0 is reserved as "no SSA name" in value definitions.
  // Keep the empty string in slot 0 so named values never alias the sentinel.
  uint32_t empty_string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_append_string(
      numbering, iree_string_view_empty(), &empty_string_writer_id));
  const loom_string_id_t module_empty_string_id =
      loom_module_lookup_string(module, iree_string_view_empty());
  if (module_empty_string_id != LOOM_STRING_ID_INVALID) {
    *loom_bytecode_numbering_module_string_writer_id(
        numbering, module_empty_string_id) = empty_string_writer_id;
  }

  // Type map: parallel array for O(1) module_type_index → writer_type_id.
  if (module->types.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->types.count, sizeof(uint32_t),
        (void**)&numbering->types.writer_ids_by_module_index));
    memset(numbering->types.writer_ids_by_module_index, 0xFF,
           module->types.count * sizeof(uint32_t));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_type_index_initialize(
      module, arena, &numbering->types.index));

  return iree_ok_status();
}

// Interns a module string by its module string_id. Returns writer string ID.
iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id) {
  if (string_id == LOOM_STRING_ID_INVALID) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  if (string_id >= numbering->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string_id %u out of range (module has %" PRIhsz
                            " strings)",
                            string_id, numbering->module->strings.count);
  }
  uint32_t* writer_id_entry =
      loom_bytecode_numbering_module_string_writer_id(numbering, string_id);
  if (*writer_id_entry != LOOM_WRITER_ID_NONE) {
    *out_writer_id = *writer_id_entry;
    return iree_ok_status();
  }
  iree_string_view_t view =
      loom_string_table_get(&numbering->module->strings, string_id);
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  *writer_id_entry = writer_id;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Resolves an inline miss or any lookup after the external index is active.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_numbering_intern_string_view_slow(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  uint32_t hash = 0;
  loom_intern_probe_t probe = {/*.index=*/UINT32_MAX, /*.slot=*/0};
  if (numbering->strings.external.index.capacity != 0) {
    hash = loom_bytecode_external_string_hash(view);
    const loom_bytecode_external_string_equal_context_t equal_context = {
        numbering,
        view,
    };
    probe = loom_intern_table_probe(&numbering->strings.external.index, hash,
                                    loom_bytecode_external_string_equal,
                                    &equal_context);
    if (probe.index != UINT32_MAX) {
      *out_writer_id = probe.index;
      return iree_ok_status();
    }
  }

  // Module strings use the source module's canonical content index and retain
  // first-use writer IDs independently of module intern order.
  const loom_string_id_t module_string_id =
      loom_bytecode_numbering_lookup_module_string(numbering, view);
  if (module_string_id != LOOM_STRING_ID_INVALID) {
    return loom_bytecode_numbering_intern_module_string(
        numbering, module_string_id, out_writer_id);
  }

  // Prepare both owners before publishing an externally indexed catalog row.
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_reserve_string(numbering));
  iree_host_size_t slot = probe.slot;
  if (numbering->strings.external.index.capacity == 0) {
    if (numbering->strings.external.count <
        LOOM_BYTECODE_INLINE_EXTERNAL_STRING_CAPACITY) {
      const uint32_t writer_id =
          loom_bytecode_numbering_publish_string(numbering, view);
      const iree_host_size_t external_index =
          numbering->strings.external.count++;
      numbering->strings.external.inline_views[external_index] = view;
      numbering->strings.external.inline_writer_ids[external_index] = writer_id;
      *out_writer_id = writer_id;
      return iree_ok_status();
    }

    // Initialize the index failure-atomically. This is the last fallible step;
    // all retained inline rows and the first overflow row can then publish.
    loom_intern_table_t* index = &numbering->strings.external.index;
    IREE_RETURN_IF_ERROR(
        loom_intern_table_initialize(numbering->arena,
                                     loom_intern_table_capacity_for_entries(
                                         numbering->strings.external.count + 1),
                                     index));
    for (iree_host_size_t i = 0; i < numbering->strings.external.count; ++i) {
      const uint32_t existing_hash = loom_bytecode_external_string_hash(
          numbering->strings.external.inline_views[i]);
      const iree_host_size_t existing_slot =
          loom_intern_table_find_empty_slot(index, existing_hash);
      loom_intern_table_insert(
          index, existing_slot, existing_hash,
          numbering->strings.external.inline_writer_ids[i]);
    }
    hash = loom_bytecode_external_string_hash(view);
    slot = loom_intern_table_find_empty_slot(index, hash);
  } else {
    IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
        numbering->arena, &numbering->strings.external.index, hash,
        /*insertion_count=*/1, &slot));
  }

  const uint32_t writer_id =
      loom_bytecode_numbering_publish_string(numbering, view);
  loom_intern_table_insert(&numbering->strings.external.index, slot, hash,
                           writer_id);
  ++numbering->strings.external.count;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an arbitrary string_view (for vtable names not in the module).
iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (iree_string_view_is_empty(view)) {
    *out_writer_id = 0;
    return iree_ok_status();
  }

  // Tiny external catalogs stay as a bounded linear scan. Descriptor-heavy
  // modules spill once to the content index before the scan could scale.
  if (numbering->strings.external.index.capacity == 0) {
    for (iree_host_size_t i = 0; i < numbering->strings.external.count; ++i) {
      if (loom_bytecode_borrowed_string_equal(
              numbering->strings.external.inline_views[i], view)) {
        *out_writer_id = numbering->strings.external.inline_writer_ids[i];
        return iree_ok_status();
      }
    }
  }

  return loom_bytecode_numbering_intern_string_view_slow(numbering, view,
                                                         out_writer_id);
}

// Publishes a completed type only after its ordered catalog storage is ready.
static iree_status_t loom_bytecode_numbering_append_type(
    loom_bytecode_numbering_t* numbering, loom_type_id_t module_index) {
  const loom_bytecode_type_node_t* node =
      &numbering->types.index.nodes[module_index];
  if (node->has_bindings) {
    numbering->types.writer_ids_by_module_index[module_index] = UINT32_MAX - 1;
    return iree_ok_status();
  }
  if (numbering->types.count >= (1u << 16)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode type count exceeds format maximum (64K)");
  }
  if (numbering->types.count >= numbering->types.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->types.count, /*minimum_capacity=*/16,
        sizeof(iree_host_size_t), &numbering->types.capacity,
        (void**)&numbering->types.module_indices_by_writer_id));
  }
  const uint32_t writer_id = (uint32_t)numbering->types.count++;
  numbering->types.module_indices_by_writer_id[writer_id] = module_index;
  numbering->types.writer_ids_by_module_index[module_index] = writer_id;
  return iree_ok_status();
}

// Interns an op kind. Returns writer op name ID.
iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id) {
  // Check existing entries.
  for (uint32_t i = 0; i < numbering->ops.count; ++i) {
    if (numbering->ops.values[i].kind == op->kind) {
      *out_writer_op_id = numbering->ops.values[i].writer_op_id;
      return iree_ok_status();
    }
  }
  // New op kind.
  iree_string_view_t name = loom_op_name(numbering->module, op);
  uint32_t string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, name, &string_writer_id));

  if (numbering->ops.count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode op count exceeds format maximum (16M)");
  }
  uint32_t writer_op_id = (uint32_t)numbering->ops.count;
  if (numbering->ops.count >= numbering->ops.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->ops.count, /*minimum_capacity=*/16,
        sizeof(loom_bytecode_op_entry_t), &numbering->ops.capacity,
        (void**)&numbering->ops.values));
  }
  numbering->ops.values[numbering->ops.count++] = (loom_bytecode_op_entry_t){
      .kind = op->kind,
      .writer_op_id = writer_op_id,
      .string_writer_id = string_writer_id,
  };
  *out_writer_op_id = writer_op_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_enum_ordinal(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_ordinal) {
  if (attr.raw > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value exceeds uint8_t range");
  }
  *out_ordinal = (uint8_t)attr.raw;
  if (!descriptor ||
      iree_all_bits_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_ok_status();
  }
  if (!loom_attr_descriptor_has_enum_case(descriptor, *out_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value has no declared case");
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_enum_array(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_enum_array_t* out_array) {
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_ENUM_ARRAY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum arrays require a descriptor-backed field");
  }
  if (attr.count > 0 && !attr.enum_array) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum array has nonzero count but NULL values");
  }
  *out_array = loom_attr_as_enum_array(attr);
  if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < out_array->count; ++i) {
    if (!loom_attr_descriptor_has_enum_case(descriptor, out_array->values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "enum array value has no declared case");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_signed_enum_set(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_signed_enum_set_t* out_set) {
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_SIGNED_ENUM_SET ||
      iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signed enum sets require a closed descriptor-backed field");
  }
  *out_set = loom_attr_as_signed_enum_set(attr);
  iree_host_size_t canonical_word_count = 0;
  IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
      *out_set, &canonical_word_count));
  if (canonical_word_count != out_set->word_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "signed enum set is not canonically trimmed");
  }
  for (iree_host_size_t value = 0; value < 256; ++value) {
    if (!loom_signed_enum_set_contains_positive(*out_set, (uint8_t)value) &&
        !loom_signed_enum_set_contains_negative(*out_set, (uint8_t)value)) {
      continue;
    }
    if (!loom_attr_descriptor_has_enum_case(descriptor, (uint8_t)value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signed enum-set value %u has no declared case",
                              (unsigned)value);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_symbol_collection(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_ref_array_t* out_array) {
  if (!descriptor || descriptor->attr_kind != attr.kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol collections require a descriptor-backed field");
  }
  if (attr.count > 0 && !attr.symbol_refs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol collection has nonzero count but NULL references");
  }
  *out_array = loom_make_symbol_ref_array(attr.symbol_refs, attr.count);
  for (iree_host_size_t i = 0; i < out_array->count; ++i) {
    const loom_symbol_ref_t ref = out_array->values[i];
    if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
        ref.symbol_id >= numbering->module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "symbol collection reference %" PRIhsz
                              " is not a local module symbol",
                              i);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_get_parameterized_attr(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind,
    const loom_parameterized_attr_descriptor_t** out_family_descriptor) {
  if (attr.kind != LOOM_ATTR_PARAMETERIZED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized attribute payload has kind %u",
                            (unsigned)attr.kind);
  }
  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(attr);
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_resolve_parameterized_attr(numbering->module->context,
                                              family_kind);
  if (!family_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u is not registered",
        (unsigned)family_kind);
  }
  if (descriptor && descriptor->attr_kind != expected_descriptor_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute does not match field kind %u",
        (unsigned)descriptor->attr_kind);
  }
  if (descriptor &&
      descriptor->reference.parameterized_attr_kind !=
          LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      descriptor->reference.parameterized_attr_kind != family_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u does not match field contract "
        "%u",
        (unsigned)family_kind,
        (unsigned)descriptor->reference.parameterized_attr_kind);
  }
  if (attr.count != family_descriptor->parameter_count ||
      (attr.count > 0 && !attr.parameterized_slots)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute '%.*s' has malformed slot storage",
        (int)loom_bstring_view(family_descriptor->name).size,
        loom_bstring_view(family_descriptor->name).data);
  }
  *out_family_descriptor = family_descriptor;
  return iree_ok_status();
}

iree_status_t loom_bytecode_parameter_is_present(
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    loom_attribute_t value, uint8_t parameter_index, bool* out_present) {
  const loom_attr_descriptor_t* parameter_descriptor =
      &family_descriptor->parameter_descriptors[parameter_index];
  if (!loom_attr_is_absent(value)) {
    *out_present = true;
    return iree_ok_status();
  }
  if (iree_any_bit_set(parameter_descriptor->flags, LOOM_ATTR_OPTIONAL)) {
    *out_present = false;
    return iree_ok_status();
  }
  const iree_string_view_t family_name =
      loom_bstring_view(family_descriptor->name);
  const iree_string_view_t parameter_name =
      loom_attr_descriptor_name(parameter_descriptor);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "parameterized attribute '%.*s' has absent required parameter '%.*s'",
      (int)family_name.size, family_name.data, (int)parameter_name.size,
      parameter_name.data);
}

iree_status_t loom_bytecode_resolve_function_low_descriptor_set(
    const loom_bytecode_numbering_t* numbering, loom_func_like_t func_like,
    const loom_low_repr_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (func_like.vtable->repr_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  const loom_string_id_t descriptor_set_key_id =
      loom_func_like_repr_contract(func_like);
  if (descriptor_set_key_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function representation contract is required");
  }
  if (descriptor_set_key_id >= numbering->module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function representation contract string ID is out of range");
  }
  if (!numbering->low_repr.environment.vtable) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "serializing Low functions requires a representation codec");
  }
  const iree_string_view_t descriptor_set_key =
      loom_string_table_get(&numbering->module->strings, descriptor_set_key_id);
  *out_descriptor_set = loom_low_repr_lookup_descriptor_set(
      &numbering->low_repr.environment, descriptor_set_key);
  if (!*out_descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function representation contract '%.*s' is not available",
        (int)descriptor_set_key.size, descriptor_set_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_scoped_enum(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr) {
  if (!numbering->low_repr.active_descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum attribute is outside a representation contract");
  }
  const iree_string_view_t key =
      loom_low_repr_descriptor_key(&numbering->low_repr.environment,
                                   numbering->low_repr.active_descriptor_set,
                                   loom_attr_as_scoped_enum(attr));
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum ordinal is outside the active representation contract");
  }
  uint32_t unused_id = 0;
  return loom_bytecode_numbering_intern_string_view(numbering, key, &unused_id);
}

// Numbers leaf payloads without entering another type or aggregate.
static iree_status_t loom_bytecode_number_leaf_attribute(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor) {
  uint32_t unused_id = 0;
  switch (attr.kind) {
    case LOOM_ATTR_STRING:
      return loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &unused_id);
    case LOOM_ATTR_ENUM_ARRAY: {
      loom_enum_array_t unused_array = loom_enum_array_empty();
      return loom_bytecode_get_enum_array(attr, descriptor, &unused_array);
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      loom_signed_enum_set_t unused_set = loom_signed_enum_set_empty();
      return loom_bytecode_get_signed_enum_set(attr, descriptor, &unused_set);
    }
    case LOOM_ATTR_SCOPED_ENUM:
      return loom_bytecode_number_scoped_enum(numbering, attr);
    case LOOM_ATTR_SYMBOL: {
      const loom_symbol_ref_t ref = attr.symbol;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        return loom_bytecode_numbering_intern_module_string(
            numbering,
            numbering->module->symbols.entries[ref.symbol_id].name_id,
            &unused_id);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_symbol_ref_array_t array = loom_symbol_ref_array_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_get_symbol_collection(
          numbering, attr, descriptor, &array));
      iree_status_t status = iree_ok_status();
      for (iree_host_size_t i = 0; i < array.count && iree_status_is_ok(status);
           ++i) {
        status = loom_bytecode_numbering_intern_module_string(
            numbering,
            numbering->module->symbols.entries[array.values[i].symbol_id]
                .name_id,
            &unused_id);
      }
      return status;
    }
    case LOOM_ATTR_ENCODING: {
      const uint16_t encoding_id = loom_attr_as_encoding_id(attr);
      if (encoding_id == 0 ||
          encoding_id > numbering->module->encodings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "encoding_id %u out of range (module has %" PRIhsz " encodings)",
            (unsigned)encoding_id, numbering->module->encodings.count);
      }
      // The ordered encoding pass owns the referenced payload's catalogs.
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

enum loom_bytecode_catalog_frame_kind_e {
  LOOM_BYTECODE_CATALOG_TYPE_CHILDREN,
  LOOM_BYTECODE_CATALOG_TYPE_COMPLETE,
  LOOM_BYTECODE_CATALOG_TYPE_PARAMETERS,
  LOOM_BYTECODE_CATALOG_ATTRIBUTE_PARAMETERS,
  LOOM_BYTECODE_CATALOG_ATTRIBUTE,
  LOOM_BYTECODE_CATALOG_DICTIONARY,
  LOOM_BYTECODE_CATALOG_PARAMETERIZED_ARRAY,
  LOOM_BYTECODE_CATALOG_PARAMETERIZED_ENTRY,
};

// One suspended first-use traversal. Frame indices survive arena-array growth;
// no continuation borrows another frame's address.
typedef struct loom_bytecode_catalog_frame_t {
  // Immutable source payload selected by kind.
  union {
    // Canonical source type owning the immediate dependency slice.
    loom_type_id_t type_id;
    // Parameterized type payload and its family contract.
    struct {
      // Borrowed slots in declaration order.
      const loom_attribute_t* values;
      // Family owning slot names and presence requirements.
      const loom_parameterized_type_descriptor_t* descriptor;
    } type_parameters;
    // Attribute payloads and their field contracts.
    struct {
      // Borrowed single value or contiguous array elements.
      const loom_attribute_t* values;
      // Field contract for the single value or parameterized array.
      const loom_attr_descriptor_t* descriptor;
    } attributes;
    // Parameterized attribute payloads after family resolution.
    struct {
      // Borrowed family slots in declaration order.
      const loom_attribute_t* values;
      // Resolved family owning parameter names and field contracts.
      const loom_parameterized_attr_descriptor_t* descriptor;
    } parameters;
    // Borrowed dictionary entries in canonical stored order.
    const loom_named_attr_t* dictionary;
  } value;
  // Next child edge, dictionary entry, array element or parameter.
  iree_host_size_t position;
  // Number of entries in a suspended aggregate.
  uint16_t count;
  // Continuation operation from loom_bytecode_catalog_frame_kind_e.
  uint8_t kind;
  // Attribute-only nesting depth; each type starts a new attribute scope.
  uint8_t aggregate_depth;
} loom_bytecode_catalog_frame_t;

static iree_status_t loom_bytecode_catalog_push(
    loom_bytecode_numbering_t* numbering, iree_host_size_t* count,
    loom_bytecode_catalog_frame_t frame) {
  if (*count >= numbering->traversal.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, *count, /*minimum_capacity=*/16, sizeof(frame),
        &numbering->traversal.capacity, (void**)&numbering->traversal.frames));
  }
  numbering->traversal.frames[(*count)++] = frame;
  return iree_ok_status();
}

// Begins a first-use type. Leaf and completed types need no continuation.
static iree_status_t loom_bytecode_catalog_enter_type(
    loom_bytecode_numbering_t* numbering, iree_host_size_t* count,
    loom_type_id_t type_id) {
  if (numbering->types.writer_ids_by_module_index[type_id] !=
      LOOM_WRITER_ID_NONE) {
    return iree_ok_status();
  }

  const loom_bytecode_type_node_t* node =
      &numbering->types.index.nodes[type_id];
  const loom_type_t type =
      loom_type_table_get(&numbering->module->types, type_id);
  uint32_t unused_id = 0;
  if (loom_type_kind(type) == LOOM_TYPE_DIALECT) {
    const loom_string_id_t name_id = loom_type_dialect_name_id(type);
    if (name_id < numbering->module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, name_id, &unused_id));
    }
  }
  if (loom_type_kind(type) == LOOM_TYPE_PARAMETERIZED) {
    const loom_parameterized_type_descriptor_t* descriptor =
        loom_type_parameterized_descriptor(type);
    if (!descriptor) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameterized type has no descriptor");
    }
    const uint8_t parameter_count =
        loom_type_parameterized_parameter_count(type);
    const loom_attribute_t* parameters =
        loom_type_parameterized_parameters(type);
    if (parameter_count != descriptor->parameter_count ||
        (parameter_count > 0 && !parameters)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameterized type '%.*s' has malformed slot storage",
          (int)loom_bstring_view(descriptor->name).size,
          loom_bstring_view(descriptor->name).data);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_bstring_view(descriptor->name), &unused_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_catalog_push(
        numbering, count,
        (loom_bytecode_catalog_frame_t){
            .value.type_id = type_id,
            .kind = LOOM_BYTECODE_CATALOG_TYPE_COMPLETE}));
    return loom_bytecode_catalog_push(
        numbering, count,
        (loom_bytecode_catalog_frame_t){
            .value.type_parameters = {parameters, descriptor},
            .count = parameter_count,
            .kind = LOOM_BYTECODE_CATALOG_TYPE_PARAMETERS});
  }
  // A completed child prefix needs no suspended work. Flat and topologically
  // requested types commonly have their entire dependency slice numbered.
  iree_host_size_t position = 0;
  while (position < node->dependencies.count) {
    const loom_type_id_t child =
        numbering->types.index
            .dependencies[node->dependencies.begin + position];
    if (numbering->types.writer_ids_by_module_index[child] ==
        LOOM_WRITER_ID_NONE) {
      break;
    }
    ++position;
  }
  if (position == node->dependencies.count) {
    return loom_bytecode_numbering_append_type(numbering, type_id);
  }
  return loom_bytecode_catalog_push(
      numbering, count,
      (loom_bytecode_catalog_frame_t){
          .value.type_id = type_id,
          .position = position,
          .kind = LOOM_BYTECODE_CATALOG_TYPE_CHILDREN});
}

// Completes all suspended roots without C recursion between types and
// attributes.
static iree_status_t loom_bytecode_catalog_complete(
    loom_bytecode_numbering_t* numbering, iree_host_size_t count) {
  iree_status_t status = iree_ok_status();
  while (count != 0 && iree_status_is_ok(status)) {
    const iree_host_size_t top = count - 1;
    const loom_bytecode_catalog_frame_t frame =
        numbering->traversal.frames[top];
    uint32_t unused_id = 0;
    switch (frame.kind) {
      case LOOM_BYTECODE_CATALOG_TYPE_CHILDREN:
      case LOOM_BYTECODE_CATALOG_TYPE_COMPLETE: {
        const loom_bytecode_type_node_t* node =
            &numbering->types.index.nodes[frame.value.type_id];
        if (frame.kind == LOOM_BYTECODE_CATALOG_TYPE_CHILDREN &&
            frame.position < node->dependencies.count) {
          ++numbering->traversal.frames[top].position;
          const loom_type_id_t child =
              numbering->types.index
                  .dependencies[node->dependencies.begin + frame.position];
          status = loom_bytecode_catalog_enter_type(numbering, &count, child);
          break;
        }
        --count;
        status =
            loom_bytecode_numbering_append_type(numbering, frame.value.type_id);
        break;
      }
      case LOOM_BYTECODE_CATALOG_TYPE_PARAMETERS:
      case LOOM_BYTECODE_CATALOG_ATTRIBUTE_PARAMETERS: {
        if (frame.position == frame.count) {
          --count;
          break;
        }
        ++numbering->traversal.frames[top].position;
        const loom_attribute_t* value = NULL;
        const loom_attr_descriptor_t* descriptor = NULL;
        uint8_t aggregate_depth = 0;
        if (frame.kind == LOOM_BYTECODE_CATALOG_TYPE_PARAMETERS) {
          const loom_parameterized_type_descriptor_t* family =
              frame.value.type_parameters.descriptor;
          value = &frame.value.type_parameters.values[frame.position];
          descriptor = &family->parameter_descriptors[frame.position];
          if (loom_attr_is_absent(*value)) {
            if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPTIONAL)) {
              break;
            }
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "parameterized type '%.*s' has absent required parameter "
                "'%.*s'",
                (int)loom_bstring_view(family->name).size,
                loom_bstring_view(family->name).data,
                (int)loom_attr_descriptor_name(descriptor).size,
                loom_attr_descriptor_name(descriptor).data);
            break;
          }
        } else {
          const loom_parameterized_attr_descriptor_t* family =
              frame.value.parameters.descriptor;
          value = &frame.value.parameters.values[frame.position];
          descriptor = &family->parameter_descriptors[frame.position];
          bool present = false;
          status = loom_bytecode_parameter_is_present(
              family, *value, (uint8_t)frame.position, &present);
          if (!iree_status_is_ok(status) || !present) {
            break;
          }
          aggregate_depth = frame.aggregate_depth + 1;
        }
        status = loom_bytecode_numbering_intern_string_view(
            numbering, loom_attr_descriptor_name(descriptor), &unused_id);
        if (iree_status_is_ok(status)) {
          status = loom_bytecode_catalog_push(
              numbering, &count,
              (loom_bytecode_catalog_frame_t){
                  .value.attributes = {value, descriptor},
                  .kind = LOOM_BYTECODE_CATALOG_ATTRIBUTE,
                  .aggregate_depth = aggregate_depth});
        }
        break;
      }
      case LOOM_BYTECODE_CATALOG_DICTIONARY:
      case LOOM_BYTECODE_CATALOG_PARAMETERIZED_ARRAY: {
        if (frame.position == frame.count) {
          --count;
          break;
        }
        ++numbering->traversal.frames[top].position;
        const loom_attribute_t* value = NULL;
        const loom_attr_descriptor_t* descriptor = NULL;
        uint8_t kind = LOOM_BYTECODE_CATALOG_ATTRIBUTE;
        if (frame.kind == LOOM_BYTECODE_CATALOG_DICTIONARY) {
          const loom_named_attr_t* entry =
              &frame.value.dictionary[frame.position];
          status = loom_bytecode_numbering_intern_module_string(
              numbering, entry->name_id, &unused_id);
          value = &entry->value;
        } else {
          value = &frame.value.attributes.values[frame.position];
          descriptor = frame.value.attributes.descriptor;
          kind = LOOM_BYTECODE_CATALOG_PARAMETERIZED_ENTRY;
        }
        if (iree_status_is_ok(status)) {
          status = loom_bytecode_catalog_push(
              numbering, &count,
              (loom_bytecode_catalog_frame_t){
                  .value.attributes = {value, descriptor},
                  .kind = kind,
                  .aggregate_depth = frame.aggregate_depth + 1});
        }
        break;
      }
      case LOOM_BYTECODE_CATALOG_ATTRIBUTE:
      case LOOM_BYTECODE_CATALOG_PARAMETERIZED_ENTRY: {
        --count;
        const loom_attribute_t attr = *frame.value.attributes.values;
        const loom_attr_descriptor_t* descriptor =
            frame.value.attributes.descriptor;
        // Array entries must resolve the array field contract even when the
        // provided payload has the wrong kind.
        if (attr.kind == LOOM_ATTR_PARAMETERIZED ||
            frame.kind == LOOM_BYTECODE_CATALOG_PARAMETERIZED_ENTRY) {
          if (frame.aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "aggregate attribute nesting exceeds max depth %u",
                (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
            break;
          }
          const loom_parameterized_attr_descriptor_t* family = NULL;
          const loom_attr_kind_t expected_kind =
              frame.kind == LOOM_BYTECODE_CATALOG_PARAMETERIZED_ENTRY
                  ? LOOM_ATTR_PARAMETERIZED_ARRAY
                  : LOOM_ATTR_PARAMETERIZED;
          status = loom_bytecode_get_parameterized_attr(
              numbering, attr, descriptor, expected_kind, &family);
          if (iree_status_is_ok(status)) {
            status = loom_bytecode_numbering_intern_string_view(
                numbering, loom_bstring_view(family->name), &unused_id);
          }
          if (iree_status_is_ok(status)) {
            status = loom_bytecode_catalog_push(
                numbering, &count,
                (loom_bytecode_catalog_frame_t){
                    .value.parameters = {attr.parameterized_slots, family},
                    .count = family->parameter_count,
                    .kind = LOOM_BYTECODE_CATALOG_ATTRIBUTE_PARAMETERS,
                    .aggregate_depth = frame.aggregate_depth});
          }
          break;
        }
        switch (attr.kind) {
          case LOOM_ATTR_TYPE: {
            if (attr.type_id >= numbering->module->types.count) {
              status = iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "type attribute id %u out of range (module has %" PRIhsz
                  " types)",
                  (unsigned)attr.type_id, numbering->module->types.count);
              break;
            }
            status = loom_bytecode_catalog_enter_type(numbering, &count,
                                                      attr.type_id);
            break;
          }
          case LOOM_ATTR_DICT:
            if (frame.aggregate_depth >=
                LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
              status = iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "aggregate attribute nesting exceeds max depth %u",
                  (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
              break;
            }
            status = loom_bytecode_catalog_push(
                numbering, &count,
                (loom_bytecode_catalog_frame_t){
                    .value.dictionary = attr.dict_entries,
                    .count = attr.count,
                    .kind = LOOM_BYTECODE_CATALOG_DICTIONARY,
                    .aggregate_depth = frame.aggregate_depth});
            break;
          case LOOM_ATTR_PARAMETERIZED_ARRAY:
            if (!descriptor ||
                descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY) {
              status = iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "parameterized attribute arrays require a descriptor-backed "
                  "field");
              break;
            }
            if (frame.aggregate_depth >=
                    LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
                (attr.count > 0 && !attr.parameterized_array)) {
              status = iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "parameterized attribute array has malformed storage or "
                  "nesting");
              break;
            }
            status = loom_bytecode_catalog_push(
                numbering, &count,
                (loom_bytecode_catalog_frame_t){
                    .value.attributes = {attr.parameterized_array, descriptor},
                    .count = attr.count,
                    .kind = LOOM_BYTECODE_CATALOG_PARAMETERIZED_ARRAY,
                    .aggregate_depth = frame.aggregate_depth});
            break;
          default:
            status = loom_bytecode_number_leaf_attribute(numbering, attr,
                                                         descriptor);
            break;
        }
        break;
      }
    }
  }
  return status;
}

iree_status_t loom_bytecode_numbering_intern_module_type(
    loom_bytecode_numbering_t* numbering, loom_type_id_t type_id,
    uint32_t* out_writer_id) {
  if (numbering->types.writer_ids_by_module_index[type_id] ==
      LOOM_WRITER_ID_NONE) {
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_catalog_enter_type(numbering, &count, type_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_catalog_complete(numbering, count));
  }
  *out_writer_id = numbering->types.writer_ids_by_module_index[type_id];
  return iree_ok_status();
}

iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id, loom_type_id_t* out_type_id) {
  const loom_type_id_t type_id =
      loom_bytecode_type_index_lookup(&numbering->types.index, type);
  if (type_id == LOOM_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type not found in module type table (kind=%u, "
                            "rank=%u, module types=%" PRIhsz ")",
                            (unsigned)loom_type_kind(type),
                            (unsigned)loom_type_rank(type),
                            numbering->module->types.count);
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_type(
      numbering, type_id, out_writer_id));
  if (out_type_id) {
    *out_type_id = type_id;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor) {
  switch (attr.kind) {
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      break;
    default:
      return loom_bytecode_number_leaf_attribute(numbering, attr, descriptor);
  }
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_catalog_push(numbering, &count,
                                 (loom_bytecode_catalog_frame_t){
                                     .value.attributes = {&attr, descriptor},
                                     .kind = LOOM_BYTECODE_CATALOG_ATTRIBUTE}));
  return loom_bytecode_catalog_complete(numbering, count);
}

iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id) {
  uint32_t unused_id = 0;
  const loom_encoding_t* encoding =
      &numbering->module->encodings.entries[encoding_id - 1];
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
      numbering, encoding->name_id, &unused_id));
  if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, encoding->alias_id, &unused_id));
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* attr = &encoding->attributes[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, attr->name_id, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attr->value, NULL));
  }
  const iree_host_size_t chunk_index =
      (encoding_id - 1) %
      IREE_ARRAYSIZE(numbering->encoding_prefixes.last->type_counts);
  if (chunk_index == 0) {
    loom_bytecode_encoding_prefix_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(numbering->arena, sizeof(*chunk), (void**)&chunk));
    chunk->next = NULL;
    if (numbering->encoding_prefixes.last) {
      numbering->encoding_prefixes.last->next = chunk;
    } else {
      numbering->encoding_prefixes.first = chunk;
    }
    numbering->encoding_prefixes.last = chunk;
  }
  numbering->encoding_prefixes.last->type_counts[chunk_index] =
      (uint32_t)numbering->types.count;
  return iree_ok_status();
}
