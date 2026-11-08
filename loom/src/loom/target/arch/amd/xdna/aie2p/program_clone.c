// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/program_clone.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/materialize.h"

static iree_status_t loom_aie2p_program_clone_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  const loom_aie2p_program_clone_t* clone = user_data;
  *out_target_ref = clone->symbols_by_source[source_ref.symbol_id];
  // Source admission inventories every symbol before private IR is built.
  IREE_ASSERT(loom_symbol_ref_is_valid(*out_target_ref));
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_initialize(
    const loom_module_t* source_module, loom_module_t* module,
    iree_arena_allocator_t* arena, loom_aie2p_program_clone_t* out_clone) {
  *out_clone = (loom_aie2p_program_clone_t){
      .source_module = source_module,
      .module = module,
      .arena = arena,
  };
  const iree_host_size_t symbol_count = source_module->symbols.count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(*out_clone->symbols_by_source),
      (void**)&out_clone->symbols_by_source));
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    out_clone->symbols_by_source[i] = loom_symbol_ref_null();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(*out_clone->functions_by_source),
      (void**)&out_clone->functions_by_source));
  memset(out_clone->functions_by_source, 0,
         symbol_count * sizeof(*out_clone->functions_by_source));
  const loom_ir_remap_options_t options = {
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_aie2p_program_clone_remap_symbol, out_clone),
  };
  return loom_ir_remap_initialize(source_module, module, arena, &options,
                                  &out_clone->definition_remap);
}

static iree_status_t loom_aie2p_program_clone_reserve_symbol(
    loom_aie2p_program_clone_t* clone, loom_symbol_id_t source_id) {
  if (loom_symbol_ref_is_valid(clone->symbols_by_source[source_id])) {
    return iree_ok_status();
  }
  const loom_symbol_t* source_symbol =
      &clone->source_module->symbols.entries[source_id];
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      clone->module,
      loom_string_table_get(&clone->source_module->strings,
                            source_symbol->name_id),
      &name_id));
  loom_symbol_ref_t target_ref = {
      .module_id = 0,
      .symbol_id = LOOM_SYMBOL_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(clone->module, name_id, &target_ref.symbol_id));
  clone->symbols_by_source[source_id] = target_ref;
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_reserve_function(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source) {
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_reserve_symbol(clone, source->symbol.symbol_id));
  const loom_symbol_ref_t target =
      loom_low_func_def_target(source->function_op);
  if (loom_symbol_ref_is_valid(target)) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_clone_reserve_symbol(clone, target.symbol_id));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_reserve_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* data) {
  for (iree_host_size_t i = 0; i < data->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_reserve_symbol(
        clone, data->values[i].symbol.symbol_id));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* data) {
  loom_builder_t builder;
  loom_builder_initialize(clone->module, &clone->module->arena,
                          loom_module_block(clone->module), &builder);
  for (iree_host_size_t i = 0; i < data->count; ++i) {
    const loom_low_read_only_data_requirement_t* requirement = &data->values[i];
    const loom_symbol_ref_t target =
        clone->symbols_by_source[requirement->symbol.symbol_id];
    if (clone->module->symbols.entries[target.symbol_id].defining_op != NULL) {
      continue;
    }
    loom_symbol_ref_t* conflicts = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        clone->arena, requirement->bank_conflicts.count, sizeof(*conflicts),
        (void**)&conflicts));
    iree_host_size_t conflict_count = 0;
    for (iree_host_size_t j = 0; j < requirement->bank_conflicts.count; ++j) {
      const loom_symbol_ref_t conflict =
          clone->symbols_by_source[requirement->bank_conflicts.values[j]
                                       .symbol_id];
      if (loom_symbol_ref_is_valid(conflict)) {
        conflicts[conflict_count++] = conflict;
      }
    }
    loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
        &clone->definition_remap, requirement->definition->location,
        &location));
    const bool has_alignment =
        loom_global_rodata_def_has_alignment(requirement->definition);
    loom_global_rodata_def_build_flags_t flags =
        has_alignment ? LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT : 0;
    if (conflict_count != 0) {
      flags |= LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_BANK_CONFLICTS;
    }
    loom_op_t* definition = NULL;
    IREE_RETURN_IF_ERROR(loom_global_rodata_def_build(
        &builder, flags, /*visibility=*/0, /*linkage=*/0, target,
        has_alignment
            ? loom_global_rodata_def_alignment(requirement->definition)
            : 0,
        loom_make_symbol_ref_array(conflicts, conflict_count),
        requirement->contents, location, &definition));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_project_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* source_data,
    loom_low_read_only_data_set_t* out_data) {
  loom_low_read_only_data_builder_t builder = {0};
  for (iree_host_size_t i = 0; i < source_data->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_read_only_data_builder_insert(
        &builder, clone->module,
        clone->symbols_by_source[source_data->values[i].symbol.symbol_id],
        clone->arena));
  }
  *out_data = loom_low_read_only_data_builder_finish(&builder);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_clone_observe_products(
    void* user_data, const loom_op_t* source_op, loom_op_t* target_op) {
  loom_aie2p_program_clone_products_t* products = user_data;
  if (products->memory != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_memory_access_clone_op(products->memory,
                                                         source_op, target_op));
  }
  if (products->loop_execution != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_clone_op(
        products->loop_execution, source_op, target_op));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_clone_body_remap(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source,
    loom_aie2p_program_clone_products_t* products,
    loom_low_memory_access_map_t** out_memory_accesses,
    loom_low_loop_execution_map_t** out_loop_execution,
    loom_ir_remap_t* out_remap) {
  *out_memory_accesses = NULL;
  *out_loop_execution = NULL;
  *products = (loom_aie2p_program_clone_products_t){0};
  if (source->memory_accesses != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_memory_access_map_allocate(
        &clone->module->arena, out_memory_accesses));
    IREE_RETURN_IF_ERROR(loom_low_memory_access_clone_allocate(
        source->memory_accesses, *out_memory_accesses, clone->arena,
        &products->memory));
  }
  if (source->loop_execution != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_map_allocate(
        &clone->module->arena, out_loop_execution));
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_clone_allocate(
        source->loop_execution, *out_loop_execution, clone->arena,
        &products->loop_execution));
  }
  loom_ir_remap_options_t options = {
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_aie2p_program_clone_remap_symbol, clone),
  };
  if (products->memory != NULL || products->loop_execution != NULL) {
    options.clone_observer = (loom_ir_clone_observer_t){
        .fn = loom_aie2p_program_clone_observe_products,
        .user_data = products,
    };
  }
  return loom_ir_remap_initialize(clone->source_module, clone->module,
                                  clone->arena, &options, out_remap);
}

iree_status_t loom_aie2p_program_clone_function(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source) {
  const loom_symbol_id_t source_id = source->symbol.symbol_id;
  if (clone->functions_by_source[source_id] != NULL) {
    return iree_ok_status();
  }
  loom_builder_t builder;
  loom_builder_initialize(clone->module, &clone->module->arena,
                          loom_module_block(clone->module), &builder);
  const loom_symbol_ref_t source_target =
      loom_low_func_def_target(source->function_op);
  if (loom_symbol_ref_is_valid(source_target)) {
    const loom_symbol_ref_t target =
        clone->symbols_by_source[source_target.symbol_id];
    if (clone->module->symbols.entries[target.symbol_id].defining_op == NULL) {
      loom_op_t* definition = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_clone_op(
          &builder,
          clone->source_module->symbols.entries[source_target.symbol_id]
              .defining_op,
          &clone->definition_remap, &definition));
    }
  }
  loom_aie2p_program_clone_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(clone->arena, sizeof(*function), (void**)&function));
  *function = (loom_aie2p_program_clone_function_t){
      .entry = clone->symbols_by_source[source_id],
      .function_target_facts = source->target.target_facts,
  };
  loom_aie2p_program_clone_products_t products;
  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_body_remap(
      clone, source, &products, &function->memory_accesses,
      &function->loop_execution, &remap));
  loom_op_t* definition = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ir_clone_op(&builder, source->function_op, &remap, &definition));
  clone->functions_by_source[source_id] = function;
  return iree_ok_status();
}
