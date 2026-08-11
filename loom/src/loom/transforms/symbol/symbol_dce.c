// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_dce.h"

#include "loom/analysis/symbol_liveness.h"
#include "loom/analysis/symbol_references.h"
#include "loom/ir/module.h"
#include "loom/ops/module/ops.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/symbol/symbol_pruning.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_SYMBOL_DCE_STATISTICS(V, statistics_type)                       \
  V(statistics_type, symbols_eliminated, "symbols-eliminated",               \
    "Number of unreachable symbol definitions removed.")                     \
  V(statistics_type, functions_eliminated, "functions-eliminated",           \
    "Number of unreachable private function-like symbols removed.")          \
  V(statistics_type, import_anchors_eliminated, "import-anchors-eliminated", \
    "Number of unreachable provider import anchors removed.")                \
  V(statistics_type, imports_eliminated, "imports-eliminated",               \
    "Number of provider imports made empty and removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_symbol_dce_statistics,
                            loom_symbol_dce_statistics_t,
                            LOOM_SYMBOL_DCE_STATISTICS)

static const loom_pass_info_t loom_symbol_dce_pass_info_storage = {
    .name = IREE_SVL("symbol-dce"),
    .description =
        IREE_SVL("Remove unreachable private symbols and provider imports."),
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
  // Rebuilt module symbol reference table.
  loom_symbol_reference_table_t references;
  // Computed live symbol set.
  loom_symbol_liveness_t liveness;
} loom_symbol_dce_state_t;

//===----------------------------------------------------------------------===//
// Reachability
//===----------------------------------------------------------------------===//

static iree_status_t loom_symbol_dce_compute_live_symbols(
    loom_symbol_dce_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(
      state->module, state->pass->arena, &state->references));
  loom_symbol_liveness_options_t options = {
      // Encodings are module-table records that serialize with the module.
      // Until there is encoding-table DCE, their symbol refs are roots.
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_pruning_symbol_is_root,
  };
  return loom_symbol_liveness_compute(state->module, &state->references,
                                      &options, state->pass->arena,
                                      &state->liveness);
}

static iree_status_t loom_symbol_dce_prune_imports(
    loom_symbol_dce_state_t* state) {
  loom_block_t* module_block = loom_module_block(state->module);
  for (loom_op_t* op = module_block->last_op; op;) {
    loom_op_t* previous_op = op->prev_op;
    if (!loom_module_import_isa(op)) {
      op = previous_op;
      continue;
    }

    loom_symbol_ref_array_t anchors = loom_module_import_symbols(op);
    uint16_t live_anchor_count = 0;
    for (uint16_t i = 0; i < anchors.count; ++i) {
      live_anchor_count +=
          state->liveness.live_symbols[anchors.values[i].symbol_id] != 0;
    }
    if (live_anchor_count == anchors.count) {
      op = previous_op;
      continue;
    }

    const uint16_t eliminated_anchor_count =
        (uint16_t)(anchors.count - live_anchor_count);
    if (live_anchor_count == 0) {
      IREE_RETURN_IF_ERROR(loom_op_erase(state->module, op));
      ++state->statistics->imports_eliminated;
    } else {
      loom_symbol_ref_t* live_anchors = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &state->module->arena, live_anchor_count, sizeof(*live_anchors),
          (void**)&live_anchors));
      uint16_t target_index = 0;
      for (uint16_t i = 0; i < anchors.count; ++i) {
        loom_symbol_ref_t anchor = anchors.values[i];
        if (state->liveness.live_symbols[anchor.symbol_id] == 0) continue;
        live_anchors[target_index++] = anchor;
      }
      // Filtering a strictly ordered set preserves its canonical order.
      loom_op_attrs(op)[loom_module_import_symbols_ATTR_INDEX] =
          loom_attr_symbol_set(live_anchors, live_anchor_count);
    }
    state->statistics->import_anchors_eliminated += eliminated_anchor_count;
    loom_pass_mark_changed(state->pass);
    op = previous_op;
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_symbol_dce_prune_imports(&state));
  IREE_RETURN_IF_ERROR(loom_symbol_dce_erase_unreachable_symbols(&state));
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
