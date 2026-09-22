// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/func/locations.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/func/location.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

static const loom_pass_info_t loom_materialize_locations_info = {
    .name = IREE_SVL("materialize-locations"),
    .description = IREE_SVL("Intern captured locations as executable rodata."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_materialize_locations_pass_info(void) {
  return &loom_materialize_locations_info;
}

typedef struct loom_location_collection_t {
  // Scratch storage for the single traversal's retained operation list.
  iree_arena_allocator_t* arena;
  // Captures in source traversal order.
  loom_op_t** values;
  // Number of retained captures.
  iree_host_size_t count;
  // Allocated operation slots.
  iree_host_size_t capacity;
} loom_location_collection_t;

static iree_status_t loom_materialize_locations_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_func_location_isa(op)) {
    return iree_ok_status();
  }
  loom_location_collection_t* captures = user_data;
  if (captures->count == captures->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        captures->arena, captures->count, 8, sizeof(*captures->values),
        &captures->capacity, (void**)&captures->values));
  }
  captures->values[captures->count++] = op;
  return iree_ok_status();
}

typedef struct loom_location_intern_entry_t {
  // Complete semantic capture key; absent denotes an unoccupied slot.
  loom_attribute_t nodes;
  // Defined rodata symbol for this capture.
  loom_symbol_ref_t symbol;
} loom_location_intern_entry_t;

static iree_status_t loom_materialize_location_define(
    loom_module_t* module, loom_symbol_map_t* names, uint32_t* next_name,
    loom_attribute_t nodes, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* out_symbol) {
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  do {
    char text[48];
    iree_snprintf(text, sizeof(text), "__location_%u", (*next_name)++);
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, iree_make_cstring_view(text), &name));
  } while (loom_symbol_map_find(names, name) != LOOM_SYMBOL_ID_INVALID);
  loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol));
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(names, arena, name, symbol));
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_func_location_encode(
      module, loom_attr_as_parameterized_array(nodes), arena, &bytes));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* definition = NULL;
  IREE_RETURN_IF_ERROR(loom_global_rodata_def_build(
      &builder, 0, (loom_symbol_ref_t){0, symbol}, 0, bytes,
      LOOM_LOCATION_UNKNOWN, &definition));
  *out_symbol = (loom_symbol_ref_t){0, symbol};
  return iree_ok_status();
}

iree_status_t loom_materialize_locations_run(loom_pass_t* pass,
                                             loom_module_t* module) {
  loom_location_collection_t captures = {.arena = pass->arena};
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, module->body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_materialize_locations_collect, &captures},
      pass->arena, &walk_result));
  if (!captures.count) {
    return iree_ok_status();
  }

  loom_symbol_map_t names = {0};
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_string_id_t name = module->symbols.entries[i].name_id;
    if (name == LOOM_STRING_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_symbol_map_insert(&names, pass->arena, name, (loom_symbol_id_t)i));
  }
  iree_host_size_t capacity = 8;
  while (capacity / 2 < captures.count) {
    capacity *= 2;
  }
  loom_location_intern_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      pass->arena, capacity, sizeof(*entries), (void**)&entries));
  memset(entries, 0, capacity * sizeof(*entries));
  uint32_t next_name = 0;
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < captures.count && iree_status_is_ok(status);
       ++i) {
    loom_op_t* op = captures.values[i];
    const loom_attribute_t nodes =
        loom_op_attrs(op)[loom_func_location_nodes_ATTR_INDEX];
    iree_host_size_t slot = loom_attribute_hash(&nodes) & (capacity - 1);
    while (!loom_attr_is_absent(entries[slot].nodes) &&
           !loom_attribute_equal(&entries[slot].nodes, &nodes)) {
      slot = (slot + 1) & (capacity - 1);
    }
    loom_location_intern_entry_t* entry = &entries[slot];
    if (loom_attr_is_absent(entry->nodes)) {
      status = loom_materialize_location_define(
          module, &names, &next_name, nodes, pass->arena, &entry->symbol);
      entry->nodes = nodes;
    }
    if (iree_status_is_ok(status)) {
      loom_builder_set_before(&rewriter.builder, op);
      const loom_value_id_t checkpoint =
          loom_rewriter_value_checkpoint(&rewriter);
      const loom_type_t type =
          loom_module_value_type(module, loom_func_location_result(op));
      loom_op_t* load = NULL;
      status = loom_global_load_build(&rewriter.builder, entry->symbol, &type,
                                      1, op->location, &load);
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_preserve_result_names_on_new_values(
            &rewriter, op, loom_global_load_result(load).values, 1, checkpoint);
      }
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_replace_all_uses_and_erase(
            &rewriter, op, loom_global_load_result(load).values, 1);
      }
    }
  }
  if (rewriter.flags & LOOM_REWRITER_FLAG_CHANGED) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
