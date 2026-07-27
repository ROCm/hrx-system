// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/one_shot.h"

#include "loom/ops/kernel/launch_config.h"

void loom_run_one_shot_options_initialize(
    loom_run_one_shot_options_t* out_options) {
  *out_options = (loom_run_one_shot_options_t){0};
  out_options->hal_workgroup_count[0] = 1;
  out_options->hal_workgroup_count[1] = 1;
  out_options->hal_workgroup_count[2] = 1;
}

static const loom_op_t* loom_run_one_shot_lookup_entry_kernel(
    const loom_module_t* module, iree_string_view_t function_name) {
  // Symbols are stored without their textual sigil.
  (void)iree_string_view_consume_prefix_char(&function_name, '@');
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, function_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  return loom_kernel_def_isa(symbol->defining_op) ? symbol->defining_op : NULL;
}

static const loom_op_t* loom_run_one_shot_find_single_kernel(
    const loom_module_t* module) {
  const loom_op_t* selected_kernel = NULL;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_kernel_def_isa(symbol->defining_op)) {
      continue;
    }
    if (selected_kernel != NULL) {
      return NULL;
    }
    selected_kernel = symbol->defining_op;
  }
  return selected_kernel;
}

bool loom_run_one_shot_options_apply_static_hal_workgroup_count(
    const loom_module_t* module, iree_string_view_t function_name,
    loom_run_one_shot_options_t* options) {
  if (module == NULL || options == NULL) {
    return false;
  }
  const loom_op_t* kernel =
      iree_string_view_is_empty(function_name)
          ? loom_run_one_shot_find_single_kernel(module)
          : loom_run_one_shot_lookup_entry_kernel(module, function_name);
  if (kernel == NULL) {
    return false;
  }
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  if (!loom_kernel_def_static_workgroup_count(module, kernel,
                                              &workgroup_count)) {
    return false;
  }
  options->hal_workgroup_count[0] = workgroup_count.x;
  options->hal_workgroup_count[1] = workgroup_count.y;
  options->hal_workgroup_count[2] = workgroup_count.z;
  return true;
}

void loom_run_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_one_shot_result_t* out_result) {
  *out_result = (loom_run_one_shot_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_one_shot_result_deinitialize(loom_run_one_shot_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_one_shot_result_t){0};
}
