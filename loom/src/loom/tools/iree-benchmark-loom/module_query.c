// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/module_query.h"

#include "loom/ops/special_values.h"

iree_string_view_t iree_benchmark_loom_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t iree_benchmark_loom_value_name(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = module->values.entries[value_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[name_id];
}

iree_string_view_t iree_benchmark_loom_normalize_selection_name(
    iree_string_view_t selection_name) {
  selection_name = iree_string_view_trim(selection_name);
  if (iree_string_view_starts_with(selection_name, IREE_SV("@"))) {
    return iree_string_view_substr(selection_name, 1, IREE_HOST_SIZE_MAX);
  }
  return selection_name;
}

bool iree_benchmark_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(case_plan->name, selected_case_name);
}

bool iree_benchmark_loom_benchmark_matches_selection(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_string_view_t selected_benchmark_name) {
  return iree_string_view_is_empty(selected_benchmark_name) ||
         iree_string_view_equal(benchmark_plan->name, selected_benchmark_name);
}

iree_status_t iree_benchmark_loom_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u is outside the module symbol table",
                            (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u has an invalid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}
