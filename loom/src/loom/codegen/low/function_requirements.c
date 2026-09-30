// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_requirements.h"

#include <string.h>

#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_function_requirements_append_read_only_data(
    const loom_module_t* module, loom_symbol_ref_t symbol,
    iree_arena_allocator_t* arena,
    loom_low_read_only_data_requirement_t** inout_requirements,
    iree_host_size_t* inout_capacity, uint32_t* ordinal_by_symbol,
    loom_low_function_requirements_t* out_requirements) {
  if (!loom_symbol_ref_is_valid(symbol) || symbol.module_id != 0 ||
      symbol.symbol_id >= out_requirements->symbol_count) {
    return iree_ok_status();
  }
  if (ordinal_by_symbol[symbol.symbol_id] != UINT32_MAX) {
    return iree_ok_status();
  }
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  if (!loom_global_rodata_def_isa(definition)) {
    return iree_ok_status();
  }

  const iree_host_size_t minimum_capacity =
      out_requirements->read_only_data_count + 1;
  if (minimum_capacity > *inout_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, out_requirements->read_only_data_count,
        iree_max(minimum_capacity, 4u), sizeof(**inout_requirements),
        inout_capacity, (void**)inout_requirements));
  }
  const uint32_t ordinal = (uint32_t)out_requirements->read_only_data_count;
  (*inout_requirements)[ordinal] = (loom_low_read_only_data_requirement_t){
      .symbol = symbol,
      .definition = definition,
      .contents = loom_global_rodata_def_contents(definition),
      .minimum_alignment =
          loom_global_rodata_def_has_alignment(definition)
              ? (uint64_t)loom_global_rodata_def_alignment(definition)
              : 1,
  };
  ordinal_by_symbol[symbol.symbol_id] = ordinal;
  ++out_requirements->read_only_data_count;
  return iree_ok_status();
}

uint32_t loom_low_function_requirements_read_only_data_ordinal(
    const loom_low_function_requirements_t* requirements,
    loom_symbol_ref_t symbol) {
  if (!loom_symbol_ref_is_valid(symbol) || symbol.module_id != 0 ||
      symbol.symbol_id >= requirements->symbol_count ||
      requirements->read_only_data_ordinal_by_symbol == NULL) {
    return UINT32_MAX;
  }
  return requirements->read_only_data_ordinal_by_symbol[symbol.symbol_id];
}

iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements) {
  *out_requirements = (loom_low_function_requirements_t){0};
  const loom_op_t** resources = NULL;
  iree_host_size_t resource_capacity = 0;
  loom_low_read_only_data_requirement_t* read_only_data = NULL;
  iree_host_size_t read_only_data_capacity = 0;
  uint32_t* read_only_data_ordinal_by_symbol = NULL;
  out_requirements->symbol_count = module->symbols.count;
  if (out_requirements->symbol_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_requirements->symbol_count,
                                  sizeof(*read_only_data_ordinal_by_symbol),
                                  (void**)&read_only_data_ordinal_by_symbol));
    memset(read_only_data_ordinal_by_symbol, 0xFF,
           out_requirements->symbol_count *
               sizeof(*read_only_data_ordinal_by_symbol));
  }
  loom_low_storage_layout_builder_t storage_builder;
  loom_low_storage_layout_builder_initialize(&storage_builder);
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    out_requirements->node_count += block->op_count;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_resource_isa(op)) {
        const iree_host_size_t minimum_capacity =
            out_requirements->resource_count + 1;
        if (minimum_capacity > resource_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              arena, out_requirements->resource_count,
              iree_max(minimum_capacity, 4u), sizeof(*resources),
              &resource_capacity, (void**)&resources));
        }
        resources[out_requirements->resource_count++] = op;
      } else if (loom_low_storage_reserve_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_low_storage_layout_builder_append(
            module, op, arena, &storage_builder));
      } else if (loom_low_return_isa(op)) {
        ++out_requirements->return_count;
      }
      if (loom_low_op_isa(op)) {
        const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
        for (iree_host_size_t i = 0; i < attrs.count; ++i) {
          if (attrs.entries[i].value.kind != LOOM_ATTR_SYMBOL) {
            continue;
          }
          IREE_RETURN_IF_ERROR(
              loom_low_function_requirements_append_read_only_data(
                  module, loom_attr_as_symbol(attrs.entries[i].value), arena,
                  &read_only_data, &read_only_data_capacity,
                  read_only_data_ordinal_by_symbol, out_requirements));
        }
      }
    }
  }
  out_requirements->resources = resources;
  out_requirements->read_only_data = read_only_data;
  out_requirements->read_only_data_ordinal_by_symbol =
      read_only_data_ordinal_by_symbol;
  loom_low_storage_layout_builder_finish(&storage_builder,
                                         &out_requirements->storage_layout);
  return iree_ok_status();
}
