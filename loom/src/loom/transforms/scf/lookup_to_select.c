// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/lookup_to_select.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

#define LOOM_SCF_LOOKUP_TO_SELECT_STATISTICS(V, statistics_type) \
  V(statistics_type, lookups_projected, "lookups-projected",     \
    "Number of scf.lookup ops projected to scf.select chains.")

LOOM_PASS_STATISTICS_DEFINE(loom_scf_lookup_to_select_statistics,
                            loom_scf_lookup_to_select_statistics_t,
                            LOOM_SCF_LOOKUP_TO_SELECT_STATISTICS)

static const loom_pass_info_t loom_scf_lookup_to_select_pass_info_storage = {
    .name = IREE_SVL("scf-lookup-to-select"),
    .description =
        IREE_SVL("Project keyed SSA tables to conditional selections."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_scf_lookup_to_select_statistics_layout,
};

const loom_pass_info_t* loom_scf_lookup_to_select_pass_info(void) {
  return &loom_scf_lookup_to_select_pass_info_storage;
}

typedef struct loom_scf_lookup_to_select_state_t {
  // Pass instance owning scratch storage and statistics.
  loom_pass_t* pass;
  // Typed statistics for this pass invocation.
  loom_scf_lookup_to_select_statistics_t* statistics;
  // Module containing the function being transformed.
  loom_module_t* module;
  // Rewriter maintaining value uses during projection.
  loom_rewriter_t* rewriter;
  // Whether this function invocation projected any lookup.
  bool changed;
} loom_scf_lookup_to_select_state_t;

static iree_status_t loom_scf_lookup_to_select_project(
    loom_scf_lookup_to_select_state_t* state, loom_op_t* op) {
  const loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  const loom_value_slice_t table = loom_scf_lookup_values(op);
  const uint16_t column_count = op->result_count;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, column_count,
                                sizeof(*replacements), (void**)&replacements));
  memcpy(replacements,
         table.values + (iree_host_size_t)case_keys.count * column_count,
         (iree_host_size_t)column_count * sizeof(*replacements));

  loom_builder_t* builder = &state->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t result_checkpoint =
      loom_rewriter_value_checkpoint(state->rewriter);
  const loom_type_t selector_type =
      loom_module_value_type(state->module, loom_scf_lookup_selector(op));
  for (uint16_t row = case_keys.count; row > 0; --row) {
    const uint16_t case_index = row - 1;
    loom_op_t* key_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(case_keys.i64_array[case_index]), selector_type,
        op->location, &key_op));
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, loom_scf_lookup_selector(op),
        loom_index_constant_result(key_op), op->location, &comparison_op));
    for (uint16_t column = 0; column < column_count; ++column) {
      const loom_value_id_t case_value =
          table.values[(iree_host_size_t)case_index * column_count + column];
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          builder, loom_index_cmp_result(comparison_op), case_value,
          replacements[column],
          loom_module_value_type(state->module,
                                 loom_op_const_results(op)[column]),
          op->location, &select_op));
      replacements[column] = loom_scf_select_result(select_op);
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, op, replacements, column_count, result_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, op, replacements, column_count));
  ++state->statistics->lookups_projected;
  state->changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_lookup_to_select_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  return loom_scf_lookup_isa(op)
             ? loom_scf_lookup_to_select_project(
                   (loom_scf_lookup_to_select_state_t*)user_data, op)
             : iree_ok_status();
}

iree_status_t loom_scf_lookup_to_select_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  loom_scf_lookup_to_select_state_t state = {
      .pass = pass,
      .statistics = loom_scf_lookup_to_select_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_status_t status = loom_walk_region(
      module, body, LOOM_WALK_POST_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_lookup_to_select_visit,
                             .user_data = &state},
      &walk_result);
  if (iree_status_is_ok(status) && state.changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
