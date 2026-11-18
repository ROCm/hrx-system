// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/numbering.h"

#include <string.h>

#include "loom/format/block_order.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refs.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Value numbering (per-function)
//===----------------------------------------------------------------------===//

// Issues a nonzero generation selecting a fresh value-index namespace.
static uint32_t loom_bytecode_value_scope_begin(
    loom_bytecode_numbering_t* numbering) {
  // A module has fewer than 2^16 symbols. Each function issues at most 2^8
  // scopes total: one signature and fewer than 2^8 root regions. Each global
  // uses only its retained closure and signature. One writer invocation
  // therefore issues fewer than 2^24 generations across all passes.
  return ++numbering->types.index.binding_generation;
}

// Returns the mutable direct-index row for |value_id|, growing the segmented
// index through that ID when first encountered.
static iree_status_t loom_bytecode_value_scope_row(
    loom_bytecode_numbering_t* numbering, loom_value_id_t value_id,
    loom_bytecode_value_scope_row_t** out_row) {
  const uint32_t segment_index = value_id >> LOOM_VALUE_SEGMENT_SHIFT;
  while (numbering->value_scopes.segments.segment_count <= segment_index) {
    loom_bytecode_value_scope_segment_t* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &numbering->value_scopes.segments, numbering->arena, (void**)&segment));
    memset(segment, 0, sizeof(*segment));
  }
  loom_bytecode_value_scope_segment_t* segment =
      (loom_bytecode_value_scope_segment_t*)loom_segmented_storage_segment(
          &numbering->value_scopes.segments, segment_index);
  *out_row = &segment->rows[value_id & LOOM_VALUE_SEGMENT_MASK];
  return iree_ok_status();
}

void loom_bytecode_value_numbering_initialize(
    loom_bytecode_value_numbering_t* value_numbering,
    loom_bytecode_numbering_t* numbering) {
  *value_numbering = (loom_bytecode_value_numbering_t){
      .numbering = numbering,
  };
}

iree_status_t loom_bytecode_value_numbering_assign_value(
    loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id) {
  loom_bytecode_numbering_t* numbering = value_numbering->numbering;
  if (value_id >= numbering->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u exceeds module value count %" PRIhsz,
                            value_id, numbering->module->values.count);
  }
  if (value_numbering->scope_generation == 0) {
    value_numbering->scope_generation =
        loom_bytecode_value_scope_begin(numbering);
  }
  loom_bytecode_value_scope_row_t* row = NULL;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_value_scope_row(numbering, value_id, &row));
  if (row->generation == value_numbering->scope_generation) {
    return iree_ok_status();
  }
  if (value_numbering->next_number == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode local value number exceeds uint32");
  }
  row->generation = value_numbering->scope_generation;
  row->number = value_numbering->next_number++;
  return iree_ok_status();
}

// Resolves a module value_id to its active local value number.
// Returns INVALID_ARGUMENT if the value_id is out of bounds or was
// never assigned a number (indicates a malformed IR graph).
iree_status_t loom_bytecode_resolve_value_number(
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id, uint32_t* out_number) {
  const loom_bytecode_numbering_t* numbering = value_numbering->numbering;
  if (value_id >= numbering->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u exceeds module value count %" PRIhsz,
                            value_id, numbering->module->values.count);
  }
  const uint32_t segment_index = value_id >> LOOM_VALUE_SEGMENT_SHIFT;
  const loom_bytecode_value_scope_row_t* row = NULL;
  if (segment_index < numbering->value_scopes.segments.segment_count) {
    const loom_bytecode_value_scope_segment_t* segment =
        (const loom_bytecode_value_scope_segment_t*)
            loom_segmented_storage_const_segment(
                &numbering->value_scopes.segments, segment_index);
    row = &segment->rows[value_id & LOOM_VALUE_SEGMENT_MASK];
  }
  if (value_numbering->scope_generation == 0 || row == NULL ||
      row->generation != value_numbering->scope_generation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u has no assigned value number "
                            "(undefined or not in scope)",
                            value_id);
  }
  *out_number = row->number;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_retain_region_order(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, loom_bytecode_region_order_t** out_order) {
  *out_order = NULL;
  if (region->block_count <= 1) {
    return iree_ok_status();
  }
  loom_bytecode_numbering_t* numbering = value_numbering->numbering;
  loom_bytecode_region_order_t* retained = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(numbering->arena, sizeof(*retained),
                                           (void**)&retained));
  *retained = (loom_bytecode_region_order_t){0};
  loom_format_block_order_t order = {0};
  IREE_RETURN_IF_ERROR(
      loom_format_block_order_initialize(numbering->module, region, &order));
  iree_status_t status = iree_ok_status();
  if (order.indices) {
    uint16_t* indices = NULL;
    status =
        iree_arena_allocate_array(numbering->arena, 2 * region->block_count,
                                  sizeof(*indices), (void**)&indices);
    if (iree_status_is_ok(status)) {
      uint16_t* positions = indices + region->block_count;
      for (uint16_t i = 0; i < region->block_count; ++i) {
        indices[i] = order.indices[i];
        positions[indices[i]] = i;
      }
      retained->indices = indices;
      retained->positions = positions;
    }
  }
  loom_format_block_order_deinitialize(&order);
  if (iree_status_is_ok(status)) {
    if (value_numbering->regions.last) {
      value_numbering->regions.last->next = retained;
    } else {
      value_numbering->regions.next = retained;
    }
    value_numbering->regions.last = retained;
    *out_order = retained;
  }
  return status;
}

iree_status_t loom_bytecode_value_numbering_assign_region(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region) {
  loom_bytecode_region_order_t* order = NULL;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_retain_region_order(value_numbering, region, &order));
  for (uint16_t position = 0; position < region->block_count; ++position) {
    const uint16_t block_index =
        order && order->indices ? order->indices[position] : position;
    const loom_block_t* block = loom_region_const_block(region, block_index);
    // Block arguments define values.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
          value_numbering, loom_block_arg_id(block, arg_index)));
    }
    // Op results define values.
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
            value_numbering, results[result_index]));
      }
      // Recurse into nested regions.
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        if (regions[region_index]) {
          IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_region(
              value_numbering, regions[region_index]));
        }
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Numbering walk
//===----------------------------------------------------------------------===//
//
// Walks the module in the same order as the Python writer's
// _number_module to produce identical string/type/op ordering.

// Forward declarations for recursive numbering.
static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth);
static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth);

iree_status_t loom_bytecode_op_attr_is_present(
    const loom_op_t* op, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t attr, bool* out_present) {
  if (attr.kind != LOOM_ATTR_ABSENT) {
    *out_present = true;
    return iree_ok_status();
  }
  if (descriptor && iree_all_bits_set(descriptor->flags, LOOM_ATTR_OPTIONAL)) {
    *out_present = false;
    return iree_ok_status();
  }
  iree_string_view_t attr_name =
      descriptor ? loom_attr_descriptor_name(descriptor) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "op kind 0x%04x has absent required attribute '%.*s'",
                          (unsigned)op->kind, (int)attr_name.size,
                          attr_name.data);
}

static bool loom_bytecode_attr_descriptor_is_symbol(
    const loom_attr_descriptor_t* descriptor) {
  return descriptor && descriptor->attr_kind == LOOM_ATTR_SYMBOL;
}

bool loom_bytecode_attr_is_symbol_identity(const loom_op_vtable_t* vtable,
                                           uint8_t attr_index) {
  return vtable && vtable->symbol_def &&
         attr_index == vtable->symbol_def->name_attr_index;
}

static uint8_t loom_bytecode_find_symbol_attr_index(
    const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->attr_descriptors) {
    return LOOM_ATTR_INDEX_NONE;
  }
  if (vtable->symbol_def) {
    uint8_t attr_index = vtable->symbol_def->name_attr_index;
    if (attr_index < vtable->attribute_count &&
        loom_bytecode_attr_descriptor_is_symbol(
            &vtable->attr_descriptors[attr_index])) {
      return attr_index;
    }
    return LOOM_ATTR_INDEX_NONE;
  }
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (loom_bytecode_attr_descriptor_is_symbol(&vtable->attr_descriptors[i])) {
      return i;
    }
  }
  return LOOM_ATTR_INDEX_NONE;
}

static iree_status_t loom_bytecode_global_value_list_push_unique(
    loom_bytecode_global_value_list_t* list, loom_value_id_t value_id) {
  loom_bytecode_numbering_t* numbering = list->numbering;
  if (value_id >= numbering->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global declaration-local value id %u out of range (module has %" PRIhsz
        " values)",
        value_id, numbering->module->values.count);
  }
  loom_bytecode_value_scope_row_t* row = NULL;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_value_scope_row(numbering, value_id, &row));
  if (row->generation == list->generation) {
    return iree_ok_status();
  }

  const iree_host_size_t chunk_offset =
      list->count % LOOM_BYTECODE_GLOBAL_VALUE_CHUNK_CAPACITY;
  if (chunk_offset == 0) {
    loom_bytecode_global_value_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(list->numbering->arena,
                                             sizeof(*chunk), (void**)&chunk));
    chunk->next = NULL;
    if (list->last) {
      list->last->next = chunk;
    } else {
      list->first = chunk;
    }
    list->last = chunk;
  }
  list->last->values[chunk_offset] = value_id;
  ++list->count;
  row->generation = list->generation;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_type_value_ref(
    loom_value_id_t value_id, void* user_data) {
  return loom_bytecode_global_value_list_push_unique(
      (loom_bytecode_global_value_list_t*)user_data, value_id);
}

static iree_status_t loom_bytecode_collect_global_value_type_refs(
    loom_bytecode_global_value_list_t* list,
    loom_bytecode_global_value_iterator_t* iterator) {
  const loom_module_t* module = list->numbering->module;
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  while (loom_bytecode_global_value_iterator_next(iterator, &value_id)) {
    loom_type_t type = loom_module_value_type(module, value_id);
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        module, type, loom_bytecode_collect_global_type_value_ref, list));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_attr_value_refs_at_depth(
    loom_bytecode_global_value_list_t* list, loom_attribute_t attr,
    uint8_t aggregate_depth) {
  switch (attr.kind) {
    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        if (predicate->arg_count > IREE_ARRAYSIZE(predicate->arg_tags)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "predicate arg count %u exceeds capacity",
                                  (unsigned)predicate->arg_count);
        }
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          if (predicate->arg_tags[arg_index] != LOOM_PRED_ARG_VALUE) {
            continue;
          }
          IREE_RETURN_IF_ERROR(loom_bytecode_global_value_list_push_unique(
              list, (loom_value_id_t)predicate->args[arg_index]));
        }
      }
      break;
    case LOOM_ATTR_DICT:
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_collect_global_attr_value_refs_at_depth(
                list, attr.dict_entries[i].value, aggregate_depth + 1));
      }
      break;
    case LOOM_ATTR_PARAMETERIZED:
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (loom_attr_is_absent(attr.parameterized_slots[i])) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_collect_global_attr_value_refs_at_depth(
                list, attr.parameterized_slots[i], aggregate_depth + 1));
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_attr_value_refs(
    loom_bytecode_global_value_list_t* list, loom_attribute_t attr) {
  return loom_bytecode_collect_global_attr_value_refs_at_depth(
      list, attr, /*aggregate_depth=*/0);
}

static iree_status_t loom_bytecode_global_value_slot(
    loom_bytecode_numbering_t* numbering, loom_symbol_id_t symbol_id,
    loom_bytecode_global_value_list_t*** out_slot) {
  IREE_ASSERT(symbol_id < numbering->module->symbols.count);
  const uint32_t segment_index =
      symbol_id / LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY;
  while (numbering->global_values.segments.segment_count <= segment_index) {
    loom_bytecode_global_value_segment_t* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&numbering->global_values.segments,
                                      numbering->arena, (void**)&segment));
    memset(segment, 0, sizeof(*segment));
  }
  loom_bytecode_global_value_segment_t* segment =
      (loom_bytecode_global_value_segment_t*)loom_segmented_storage_segment(
          &numbering->global_values.segments, segment_index);
  *out_slot =
      &segment->values[symbol_id % LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY];
  return iree_ok_status();
}

iree_status_t loom_bytecode_prepare_global_values(
    loom_bytecode_numbering_t* numbering, loom_symbol_id_t symbol_id,
    const loom_op_t* op, const loom_bytecode_global_value_list_t** out_values) {
  *out_values = NULL;
  loom_bytecode_global_value_list_t** slot = NULL;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_global_value_slot(numbering, symbol_id, &slot));
  if (*slot) {
    *out_values = *slot;
    return iree_ok_status();
  }

  loom_bytecode_global_value_list_t* values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(numbering->arena, sizeof(*values), (void**)&values));
  *values = (loom_bytecode_global_value_list_t){
      .numbering = numbering,
      .generation = loom_bytecode_value_scope_begin(numbering),
  };

  const loom_value_id_t* result_ids = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_global_value_list_push_unique(values, result_ids[i]));
  }

  loom_bytecode_global_value_iterator_t iterator =
      loom_bytecode_global_value_iterator_begin(values);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_collect_global_value_type_refs(values, &iterator));

  const loom_op_vtable_t* vtable = loom_op_vtable(numbering->module, op);
  if (!vtable || (op->attribute_count > 0 && !vtable->attr_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global symbol op kind 0x%04x has missing attr descriptors",
        (unsigned)op->kind);
  }
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_collect_global_attr_value_refs(values, attrs[i]));
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_collect_global_value_type_refs(values, &iterator));
  *slot = values;
  *out_values = values;
  return iree_ok_status();
}

const loom_bytecode_global_value_list_t* loom_bytecode_global_values_for_symbol(
    const loom_bytecode_numbering_t* numbering, loom_symbol_id_t symbol_id) {
  IREE_ASSERT(symbol_id < numbering->module->symbols.count);
  const uint32_t segment_index =
      symbol_id / LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY;
  IREE_ASSERT(segment_index < numbering->global_values.segments.segment_count);
  const loom_bytecode_global_value_segment_t* segment =
      (const loom_bytecode_global_value_segment_t*)
          loom_segmented_storage_const_segment(
              &numbering->global_values.segments, segment_index);
  const loom_bytecode_global_value_list_t* values =
      segment->values[symbol_id % LOOM_BYTECODE_GLOBAL_VALUE_SEGMENT_CAPACITY];
  IREE_ASSERT(values != NULL);
  return values;
}

iree_status_t loom_bytecode_number_global(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    const loom_bytecode_global_value_list_t* local_values) {
  uint32_t unused_id = 0;

  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  const loom_module_t* module = numbering->module;
  loom_bytecode_global_value_iterator_t iterator =
      loom_bytecode_global_value_iterator_begin(local_values);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  while (loom_bytecode_global_value_iterator_next(&iterator, &value_id)) {
    const loom_value_t* value = loom_module_value(module, value_id);
    if (!numbering->strip_debug && value->name_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, value->name_id, &unused_id));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, value->type, &unused_id, NULL));
  }

  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (!vtable || (op->attribute_count > 0 && !vtable->attr_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global symbol op kind 0x%04x has missing attr descriptors",
        (unsigned)op->kind);
  }

  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  return iree_ok_status();
}

iree_status_t loom_bytecode_validate_record_symbol_op(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      !loom_symbol_definition_implements(vtable->symbol_def,
                                         LOOM_SYMBOL_INTERFACE_RECORD)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must define a RECORD symbol",
        (unsigned)op->kind);
  }
  if (op->operand_count != 0 || op->result_count != 0 ||
      op->tied_result_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must not have operands, results, or "
        "tied results",
        (unsigned)op->kind);
  }
  if (vtable->region_count > 1 ||
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must declare at most one fixed region",
        (unsigned)op->kind);
  }
  if (op->region_count != vtable->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x region count does not match its vtable",
        (unsigned)op->kind);
  }
  if (op->region_count == 1 && !loom_op_regions(op)[0]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must have a materialized body region",
        (unsigned)op->kind);
  }
  if (loom_bytecode_find_symbol_attr_index(vtable) == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must declare a symbol attribute",
        (unsigned)op->kind);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_number_record(loom_bytecode_numbering_t* numbering,
                                          const loom_op_t* op) {
  uint32_t unused_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_validate_record_symbol_op(numbering->module, op));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(numbering->module->context, op->kind);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i) ||
        loom_symbol_definition_attr_is_policy(vtable->symbol_def, i)) {
      continue;
    }

    iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  if (op->region_count == 1) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_region(numbering, loom_op_regions(op)[0], 0));
  }

  return iree_ok_status();
}

iree_status_t loom_bytecode_number_function(
    loom_bytecode_numbering_t* numbering, loom_func_like_t func_like) {
  uint32_t unused_id = 0;

  // Defining func-like op name.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, func_like.op, &unused_id));

  // Kernel workload and ordinary FuncLike signature names and types.
  loom_value_slice_t workload_args =
      loom_kernel_workload_arg_ids(numbering->module, func_like.op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  for (uint16_t i = 0; i < workload_args.count; ++i) {
    loom_value_id_t value_id = workload_args.values[i];
    if (value_id < numbering->module->values.count) {
      loom_string_id_t name_id =
          loom_module_value(numbering->module, value_id)->name_id;
      if (!numbering->strip_debug && name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, name_id, &unused_id));
      }
    }
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = arg_ids[i];
    if (value_id < numbering->module->values.count) {
      loom_string_id_t name_id =
          loom_module_value(numbering->module, value_id)->name_id;
      if (!numbering->strip_debug && name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, name_id, &unused_id));
      }
    }
  }
  for (uint16_t i = 0; i < workload_args.count; ++i) {
    loom_type_t arg_type =
        loom_module_value_type(numbering->module, workload_args.values[i]);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, arg_type, &unused_id, NULL));
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = arg_ids[i];
    loom_type_t arg_type = loom_module_value_type(numbering->module, value_id);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, arg_type, &unused_id, NULL));
  }

  // Result types (recursive).
  uint16_t result_count = func_like.op->result_count;
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    loom_type_t result_type =
        loom_module_value_type(numbering->module, result_ids[i]);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, result_type, &unused_id, NULL));
  }

  // Function metadata predicates reference signature SSA values. Their value
  // numbers are resolved while the metadata section is emitted.

  // Root regions in declared slot order. Each serialized payload owns an
  // independent value namespace.
  loom_region_t** regions = loom_op_regions(func_like.op);
  for (uint8_t i = 0; i < func_like.op->region_count; ++i) {
    if (!regions[i]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_number_region(numbering, regions[i], 0));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth) {
  if (depth >= LOOM_BYTECODE_WRITER_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_WRITER_MAX_REGION_DEPTH);
  }
  uint32_t unused_id = 0;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    // Block label.
    if (!numbering->strip_debug && block->label_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, block->label_id, &unused_id));
    }

    // Block arg names and types.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t value_id = loom_block_arg_id(block, arg_index);
      const loom_value_t* value =
          loom_module_value(numbering->module, value_id);
      if (!numbering->strip_debug && value->name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, value->name_id, &unused_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, value->type, &unused_id, NULL));
    }

    // Operations.
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_operation(numbering, op, depth));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth) {
  uint32_t unused_id = 0;

  // Op name (into both op table and string table).
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  // Result names and types.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value =
        loom_module_value(numbering->module, results[i]);
    if (!numbering->strip_debug && value->name_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, value->name_id, &unused_id));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, value->type, &unused_id, NULL));
  }

  // Attribute keys and values.
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(numbering->module->context, op->kind);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor =
        (vtable && vtable->attr_descriptors) ? &vtable->attr_descriptors[i]
                                             : NULL;
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present) {
      continue;
    }

    // Attribute key name (from vtable descriptor).
    if (descriptor) {
      iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
          numbering, key_name, &unused_id));
    }
    // Attribute value strings.
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  // Nested regions.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_region(numbering, regions[i], depth + 1));
    }
  }

  return iree_ok_status();
}
