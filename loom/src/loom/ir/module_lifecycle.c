// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/module.h"

static iree_status_t loom_module_initialize_tables(
    loom_module_t* module, const loom_module_size_hints_t* hints) {
  iree_arena_allocator_t* arena = &module->arena;

  iree_host_size_t string_capacity = 512;
  iree_host_size_t type_capacity = 64;
  iree_host_size_t encoding_capacity = 0;
  iree_host_size_t source_capacity = 0;
  iree_host_size_t symbol_capacity = 32;

  if (hints) {
    string_capacity =
        (iree_host_size_t)(hints->string_count * LOOM_MODULE_GROWTH_FACTOR);
    type_capacity =
        (iree_host_size_t)(hints->type_count * LOOM_MODULE_GROWTH_FACTOR);
    encoding_capacity =
        (iree_host_size_t)(hints->encoding_count * LOOM_MODULE_GROWTH_FACTOR);
    source_capacity =
        (iree_host_size_t)(hints->source_count * LOOM_MODULE_GROWTH_FACTOR);
    symbol_capacity =
        (iree_host_size_t)(hints->symbol_count * LOOM_MODULE_GROWTH_FACTOR);
    if (string_capacity < 8) {
      string_capacity = 8;
    }
    if (type_capacity < 8) {
      type_capacity = 8;
    }
    if (hints->encoding_count > 0 && encoding_capacity < 8) {
      encoding_capacity = 8;
    }
    if (hints->source_count > 0 && source_capacity < 4) {
      source_capacity = 4;
    }
    if (symbol_capacity < 4) {
      symbol_capacity = 4;
    }
  }

  // Value segments are allocated lazily as values are defined. The matching
  // scratch and type-use head tables share each segment.
  loom_segmented_storage_initialize(sizeof(loom_value_segment_t),
                                    iree_alignof(loom_value_segment_t),
                                    &module->values.segments);
  module->scratch.values.value_table = &module->values;
  module->scratch.values.state =
      LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS;
  module->type_uses.value_table = &module->values;
  iree_arena_initialize(module->arena.block_pool, &module->type_uses.arena);

  // Locations. Rows are allocated lazily in stable segments.
  loom_segmented_storage_initialize(sizeof(loom_location_segment_t),
                                    iree_alignof(loom_location_segment_t),
                                    &module->locations.segments);

  // String views are lazy; hints size only the interner's buckets.
  loom_segmented_storage_initialize(sizeof(loom_string_segment_t),
                                    iree_alignof(loom_string_segment_t),
                                    &module->strings.segments);

  // Canonical rows are lazy; type hints size only the interner's buckets.
  loom_segmented_storage_initialize(sizeof(loom_type_segment_t),
                                    iree_alignof(loom_type_segment_t),
                                    &module->types.segments);

  // Encodings. Modules without an encoding count hint retain lazy allocation.
  if (encoding_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, encoding_capacity, sizeof(loom_encoding_t),
        (void**)&module->encodings.entries));
    module->encodings.capacity = encoding_capacity;
  }

  // Sources. Modules without a source count hint retain lazy allocation.
  if (source_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source_capacity, sizeof(iree_string_view_t),
        (void**)&module->sources.entries));
    module->sources.capacity = source_capacity;
  }

  // Symbols.
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, symbol_capacity, sizeof(loom_symbol_t),
                                (void**)&module->symbols.entries));
  module->symbols.capacity = symbol_capacity;
  memset(module->symbols.entries, 0, symbol_capacity * sizeof(loom_symbol_t));

  // Intern tables sized to match entry capacity.
  iree_host_size_t string_intern_capacity =
      loom_intern_table_capacity_for_entries(string_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(
      arena, string_intern_capacity, &module->string_intern));
  iree_host_size_t type_intern_capacity =
      loom_intern_table_capacity_for_entries(type_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(arena, type_intern_capacity,
                                                    &module->type_intern));
  iree_host_size_t encoding_intern_capacity = 0;
  if (encoding_capacity > 0) {
    encoding_intern_capacity =
        loom_intern_table_capacity_for_entries(encoding_capacity);
  }
  return loom_intern_table_initialize(arena, encoding_intern_capacity,
                                      &module->encoding_intern);
}

iree_status_t loom_module_allocate(loom_context_t* context,
                                   iree_string_view_t name,
                                   iree_arena_block_pool_t* block_pool,
                                   const loom_module_size_hints_t* hints,
                                   iree_allocator_t allocator,
                                   loom_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_module = NULL;

  loom_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(loom_module_t), (void**)&module));
  memset(module, 0, sizeof(loom_module_t));

  module->context = context;
  module->allocator = allocator;
  iree_arena_initialize(block_pool, &module->arena);

  iree_status_t status = loom_module_initialize_tables(module, hints);
  if (iree_status_is_ok(status)) {
    status = loom_module_intern_string(module, name, &module->name_id);
  }
  if (iree_status_is_ok(status)) {
    status = loom_module_allocate_region(module, 1, &module->body);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_module_free(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void loom_module_free(loom_module_t* module) {
  if (!module) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = module->allocator;
  iree_arena_deinitialize(&module->type_uses.arena);
  iree_arena_deinitialize(&module->arena);
  iree_allocator_free(allocator, module);
  IREE_TRACE_ZONE_END(z0);
}
