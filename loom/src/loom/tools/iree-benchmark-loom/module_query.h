// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Read-only Loom module and attribute queries used by iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODULE_QUERY_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODULE_QUERY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a borrowed module string or an empty view when |string_id| is
// invalid.
iree_string_view_t iree_benchmark_loom_module_string(
    const loom_module_t* module, loom_string_id_t string_id);

// Returns a borrowed value name or an empty view when unnamed/invalid.
iree_string_view_t iree_benchmark_loom_value_name(const loom_module_t* module,
                                                  loom_value_id_t value_id);

// Trims a CLI selection name and removes one leading '@' when present.
iree_string_view_t iree_benchmark_loom_normalize_selection_name(
    iree_string_view_t selection_name);

// Returns true when |case_plan| matches an empty or named selection.
bool iree_benchmark_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name);

// Returns true when |benchmark_plan| matches an empty or named selection.
bool iree_benchmark_loom_benchmark_matches_selection(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_string_view_t selected_benchmark_name);

// Resolves a symbol reference attribute to the target symbol name.
iree_status_t iree_benchmark_loom_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODULE_QUERY_H_
