// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_dce.h"

#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/module.h"
#include "loom/target/selection.h"
#include "loom/transforms/symbol/symbol_pruning.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_SYMBOL_DCE_STATISTICS(V, statistics_type)             \
  V(statistics_type, symbols_eliminated, "symbols-eliminated",     \
    "Number of unreachable symbol definitions removed.")           \
  V(statistics_type, functions_eliminated, "functions-eliminated", \
    "Number of unreachable private function-like symbols removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_symbol_dce_statistics,
                            loom_symbol_dce_statistics_t,
                            LOOM_SYMBOL_DCE_STATISTICS)

static const loom_pass_info_t loom_symbol_dce_pass_info_storage = {
    .name = IREE_SVL("symbol-dce"),
    .description = IREE_SVL("Remove unreachable private symbol definitions."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_symbol_dce_statistics_layout,
};

const loom_pass_info_t* loom_symbol_dce_pass_info(void) {
  return &loom_symbol_dce_pass_info_storage;
}

typedef struct loom_symbol_dce_state_t {
  // Active pass instance for scratch allocation and statistics.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_symbol_dce_statistics_t* statistics;
  // Module being rewritten.
  loom_module_t* module;
  // Rebuilt module symbol dependency table.
  loom_symbol_dependency_table_t dependencies;
  // Computed live symbol set.
  loom_symbol_liveness_t liveness;
} loom_symbol_dce_state_t;

//===----------------------------------------------------------------------===//
// Reachability
//===----------------------------------------------------------------------===//

static iree_status_t loom_symbol_dce_compute_live_symbols(
    loom_symbol_dce_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(
      state->module, state->pass->arena, &state->dependencies));
  loom_symbol_liveness_options_t options = {
      // Encodings are module-table records that serialize with the module.
      // Until there is encoding-table DCE, their symbol refs are roots.
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_pruning_symbol_is_root,
  };
  return loom_symbol_liveness_compute(state->module, &state->dependencies,
                                      &options, state->pass->arena,
                                      &state->liveness);
}

static iree_status_t loom_symbol_dce_erase_unreachable_symbols(
    loom_symbol_dce_state_t* state) {
  loom_symbol_pruning_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_symbol_pruning_erase_unreachable(
      state->module, &state->liveness, /*options=*/NULL, state->pass->arena,
      &result));
  if (result.symbol_count == 0) {
    return iree_ok_status();
  }
  loom_pass_mark_changed(state->pass);
  state->statistics->symbols_eliminated += result.symbol_count;
  state->statistics->functions_eliminated += result.function_like_count;
  return iree_ok_status();
}

iree_status_t loom_symbol_dce_run(loom_pass_t* pass, loom_module_t* module) {
  loom_symbol_dce_state_t state = {
      .pass = pass,
      .statistics = loom_symbol_dce_statistics(pass),
      .module = module,
  };
  IREE_RETURN_IF_ERROR(loom_symbol_dce_compute_live_symbols(&state));
  IREE_RETURN_IF_ERROR(loom_symbol_dce_erase_unreachable_symbols(&state));
  return loom_target_pass_compact_symbols_preserving_target_ref(
      pass, module, pass->arena, NULL);
}
