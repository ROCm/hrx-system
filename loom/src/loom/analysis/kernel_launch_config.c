// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_launch_config.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/facts.h"
#include "loom/ir/types.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/function_contract.h"

static iree_string_view_t loom_kernel_launch_config_normalize_symbol(
    iree_string_view_t symbol) {
  if (!iree_string_view_is_empty(symbol) && symbol.data[0] == '@') {
    return iree_make_string_view(symbol.data + 1, symbol.size - 1);
  }
  return symbol;
}

bool loom_kernel_launch_config_fields_are_valid(
    loom_kernel_launch_config_field_flags_t fields) {
  const loom_kernel_launch_config_field_flags_t known_fields =
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE |
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  return (fields & ~known_fields) == 0;
}

static const loom_symbol_t* loom_kernel_launch_config_find_symbol(
    const loom_module_t* module, iree_string_view_t symbol_name,
    loom_symbol_id_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  loom_string_id_t name_id = loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return NULL;
  }
  *out_symbol_id = symbol_id;
  return &module->symbols.entries[symbol_id];
}

static iree_status_t loom_kernel_launch_config_symbol_has_target(
    const loom_module_t* module, loom_symbol_id_t symbol_id,
    iree_arena_block_pool_t* block_pool, bool* out_has_target) {
  *out_has_target = false;

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  iree_status_t status = loom_symbol_fact_table_lookup(&symbol_facts, module,
                                                       symbol_id, &base_facts);
  if (iree_status_is_ok(status)) {
    const loom_func_symbol_facts_t* func_facts =
        loom_func_symbol_facts_cast(base_facts);
    *out_has_target =
        func_facts && loom_symbol_ref_is_valid(func_facts->target_symbol);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

static iree_status_t loom_kernel_launch_config_resolve_target_bundle(
    const loom_module_t* module, loom_symbol_id_t symbol_id,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_valid, loom_target_bundle_storage_t* out_storage,
    const loom_target_bundle_t** out_bundle) {
  *out_valid = true;
  *out_storage = (loom_target_bundle_storage_t){0};
  *out_bundle = NULL;

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(&symbol_facts, module,
                                                     symbol_id, &base_facts));
  const loom_func_symbol_facts_t* func_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (!func_facts || !loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    return iree_ok_status();
  }

  bool contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve(
      module, &symbol_facts, func_facts, emitter, &contract_valid,
      out_storage));
  *out_valid = contract_valid;
  if (contract_valid) {
    *out_bundle = &out_storage->bundle;
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_define_workload_arguments(
    const loom_kernel_launch_config_options_t* options,
    const loom_module_t* module, loom_op_t* kernel_op,
    loom_value_fact_table_t* fact_table,
    loom_kernel_launch_config_t* out_config) {
  if (options->workload_argument_count == 0) {
    return iree_ok_status();
  }
  loom_region_t* config_region = loom_kernel_def_config(kernel_op);
  if (!config_region || config_region->block_count == 0) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_COUNT;
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(config_region);
  if (!entry_block) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_COUNT;
    return iree_ok_status();
  }
  if (options->workload_argument_count != entry_block->arg_count) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_COUNT;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < options->workload_argument_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(entry_block, (uint16_t)i);
    loom_type_t type = loom_module_value_type(module, value_id);
    if (!loom_type_is_scalar(type)) {
      out_config->failure =
          LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_TYPE;
      return iree_ok_status();
    }
    const loom_scalar_type_t scalar_type = loom_type_element_type(type);
    if (scalar_type != LOOM_SCALAR_TYPE_INDEX &&
        scalar_type != LOOM_SCALAR_TYPE_OFFSET &&
        !loom_scalar_type_is_integer(scalar_type)) {
      out_config->failure =
          LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_TYPE;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        fact_table, value_id,
        loom_value_facts_exact_i64(options->workload_arguments[i])));
  }
  return iree_ok_status();
}

static void loom_kernel_launch_config_fill_known_fields(
    const loom_module_t* module, loom_op_t* kernel_op,
    const loom_target_bundle_t* target_bundle,
    const loom_value_fact_table_t* fact_table,
    loom_kernel_launch_config_t* out_config) {
  loom_target_dispatch_workgroup_count_t count = {0};
  if (loom_kernel_def_static_workgroup_count_from_facts(module, kernel_op,
                                                        fact_table, &count)) {
    out_config->workgroup_count = count;
    out_config->fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT;
  }

  loom_target_workgroup_size_t size = {0};
  if (loom_kernel_def_static_workgroup_size_from_facts(module, kernel_op,
                                                       fact_table, &size)) {
    out_config->workgroup_size = size;
    out_config->fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE;
  }

  if (target_bundle && target_bundle->snapshot &&
      target_bundle->snapshot->subgroup_size != 0) {
    out_config->subgroup_size = target_bundle->snapshot->subgroup_size;
    out_config->fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE;
  }
}

static void loom_kernel_launch_config_report_required_fields(
    loom_kernel_launch_config_field_flags_t required_fields,
    const loom_kernel_launch_config_t* config,
    loom_kernel_launch_config_t* out_config) {
  const loom_kernel_launch_config_field_flags_t missing =
      required_fields & ~config->fields;
  if (missing & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_COUNT;
  } else if (missing & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_SIZE;
  } else if (missing & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_SUBGROUP_SIZE;
  } else if (missing &
             LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES) {
    out_config->failure =
        LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES;
  }
}

iree_status_t loom_kernel_launch_config_try_evaluate_direct(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    const loom_kernel_launch_config_options_t* options,
    loom_kernel_launch_config_t* out_config, bool* out_evaluated) {
  if (!module || !block_pool || !options || !out_config || !out_evaluated) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, block_pool, options, out_config, and out_evaluated are "
        "required");
  }
  if (!loom_kernel_launch_config_fields_are_valid(options->required_fields)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required_fields contains unknown bits");
  }

  *out_config = (loom_kernel_launch_config_t){0};
  *out_evaluated = false;
  if (options->workload_argument_count != 0) {
    return iree_ok_status();
  }

  const iree_string_view_t symbol_name =
      loom_kernel_launch_config_normalize_symbol(options->function_symbol);
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  const loom_symbol_t* symbol =
      loom_kernel_launch_config_find_symbol(module, symbol_name, &symbol_id);
  if (symbol == NULL || !loom_kernel_def_isa(symbol->defining_op)) {
    return iree_ok_status();
  }

  bool has_target = false;
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_symbol_has_target(
      module, symbol_id, block_pool, &has_target));
  if (has_target) {
    return iree_ok_status();
  }

  loom_kernel_launch_config_fill_known_fields(module, symbol->defining_op,
                                              /*target_bundle=*/NULL,
                                              /*fact_table=*/NULL, out_config);
  loom_kernel_launch_config_report_required_fields(options->required_fields,
                                                   out_config, out_config);
  if (!loom_kernel_launch_config_has_failure(out_config->failure)) {
    *out_evaluated = true;
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_evaluate(
    loom_module_t* module, iree_arena_block_pool_t* block_pool,
    const loom_kernel_launch_config_options_t* options,
    loom_kernel_launch_config_t* out_config) {
  if (!module || !block_pool || !options || !out_config) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, block_pool, options, and out_config are required");
  }
  if (!loom_kernel_launch_config_fields_are_valid(options->required_fields)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required_fields contains unknown bits");
  }

  *out_config = (loom_kernel_launch_config_t){0};
  const iree_string_view_t symbol_name =
      loom_kernel_launch_config_normalize_symbol(options->function_symbol);
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  const loom_symbol_t* symbol =
      loom_kernel_launch_config_find_symbol(module, symbol_name, &symbol_id);
  if (symbol == NULL) {
    out_config->failure = LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_FUNCTION_NOT_FOUND;
    return iree_ok_status();
  }
  if (!loom_kernel_def_isa(symbol->defining_op)) {
    out_config->failure = LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NOT_KERNEL;
    return iree_ok_status();
  }

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(block_pool, &target_arena);
  loom_target_bundle_storage_t target_storage = {0};
  const loom_target_bundle_t* target_bundle = NULL;
  bool target_valid = true;
  iree_status_t status = loom_kernel_launch_config_resolve_target_bundle(
      module, symbol_id, &target_arena, options->diagnostic_emitter,
      &target_valid, &target_storage, &target_bundle);
  if (iree_status_is_ok(status) && !target_valid) {
    out_config->failure = LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_TARGET_CONTRACT;
  }

  loom_pass_value_fact_owner_t fact_owner = {0};
  loom_pass_value_fact_owner_initialize(block_pool, &fact_owner);
  loom_value_fact_table_t* fact_table = NULL;
  if (iree_status_is_ok(status) &&
      !loom_kernel_launch_config_has_failure(out_config->failure)) {
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    status = loom_pass_value_fact_owner_prepare(
        &fact_owner, module,
        loom_pass_value_fact_scope_region_for_target(
            function, loom_kernel_def_config(symbol->defining_op),
            symbol->defining_op, target_bundle),
        &fact_table);
  }
  if (iree_status_is_ok(status) &&
      !loom_kernel_launch_config_has_failure(out_config->failure)) {
    status = loom_kernel_launch_config_define_workload_arguments(
        options, module, symbol->defining_op, fact_table, out_config);
  }
  if (iree_status_is_ok(status) &&
      !loom_kernel_launch_config_has_failure(out_config->failure)) {
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    status = loom_value_fact_table_compute_region(
        fact_table, module, function,
        loom_kernel_def_config(symbol->defining_op), symbol->defining_op);
  }
  if (iree_status_is_ok(status) &&
      !loom_kernel_launch_config_has_failure(out_config->failure)) {
    loom_kernel_launch_config_fill_known_fields(
        module, symbol->defining_op, target_bundle, fact_table, out_config);
    loom_kernel_launch_config_report_required_fields(options->required_fields,
                                                     out_config, out_config);
  }

  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&target_arena);
  return status;
}
