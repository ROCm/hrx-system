// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/projection_driver.h"

#include "loom/transforms/boundary/projection_apply.h"
#include "loom/transforms/boundary/projection_plan.h"

static void loom_boundary_projection_capture_statistics(
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_statistics_t* out_statistics) {
  *out_statistics = (loom_boundary_projection_statistics_t){
      .functions_rewritten = plan->functions_rewritten,
      .calls_rewritten = plan->calls_rewritten,
      .returns_rewritten = plan->returns_rewritten,
      .cfg_edges_rewritten = plan->cfg_edges_rewritten,
      .rules = plan->rule_statistics,
      .rule_count = plan->rules.count,
  };
}

static bool loom_boundary_projection_function_has_selected_slot(
    const loom_boundary_projection_function_t* function) {
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    if (function->candidates[i].selected) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_boundary_projection_run(
    loom_pass_t* pass, loom_module_t* module,
    const loom_function_version_list_t* version_list,
    loom_boundary_projection_rule_list_t rules,
    loom_boundary_projection_statistics_t* out_statistics) {
  *out_statistics = (loom_boundary_projection_statistics_t){0};
  loom_boundary_projection_plan_t plan = {
      .pass = pass,
      .module = module,
      .arena = pass->arena,
  };
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_plan_prepare(&plan, version_list, rules));

  bool has_changes = false;
  for (iree_host_size_t i = 0; i < plan.function_count; ++i) {
    const loom_boundary_projection_function_t* function = &plan.functions[i];
    has_changes |=
        function->selected &&
        (function->signature_changes || function->call_count != 0 ||
         loom_boundary_projection_function_has_selected_slot(function));
  }
  if (!has_changes) {
    loom_boundary_projection_capture_statistics(&plan, out_statistics);
    return iree_ok_status();
  }

  loom_rewriter_initialize(&plan.rewriter, module, pass->arena);
  iree_status_t status = loom_boundary_projection_apply(&plan);
  const bool changed =
      iree_any_bit_set(plan.rewriter.flags, LOOM_REWRITER_FLAG_CHANGED);
  loom_rewriter_deinitialize(&plan.rewriter);
  if (changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }

  loom_boundary_projection_capture_statistics(&plan, out_statistics);
  return status;
}
