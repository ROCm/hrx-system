// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/dce.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/cleanup/dce.h"

static const loom_pass_statistic_field_t kLowDceStatisticFields[] = {
    LOOM_PASS_STATISTIC_FIELD(loom_dce_statistics_t, ops_eliminated,
                              "ops-eliminated",
                              "Number of low operations removed."),
};

static const loom_pass_statistic_layout_t kLowDceStatisticLayout =
    LOOM_PASS_STATISTIC_LAYOUT(loom_dce_statistics_t, kLowDceStatisticFields);

static const loom_pass_info_t loom_low_dce_pass_info_storage = {
    .name = IREE_SVL("low-dce"),
    .description = IREE_SVL("Remove dead descriptor-backed low packets."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &kLowDceStatisticLayout,
};

const loom_pass_info_t* loom_low_dce_pass_info(void) {
  return &loom_low_dce_pass_info_storage;
}

typedef struct loom_low_dce_deadness_context_t {
  // Resolved target selected by the low function target bundle.
  const loom_low_resolved_target_t* target;
} loom_low_dce_deadness_context_t;

static iree_status_t loom_low_dce_deadness_query(void* user_data,
                                                 const loom_module_t* module,
                                                 const loom_op_t* op,
                                                 bool* out_is_dead) {
  loom_low_dce_deadness_context_t* context =
      (loom_low_dce_deadness_context_t*)user_data;
  *out_is_dead = false;

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, context->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    *out_is_dead = loom_op_is_trivially_dead(module, op);
    return iree_ok_status();
  }

  if (op->result_count == 0 || !loom_op_results_unused(module, op)) {
    return iree_ok_status();
  }

  if (packet.descriptor == NULL) {
    return iree_ok_status();
  }
  *out_is_dead = iree_any_bit_set(packet.descriptor->flags,
                                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE);
  return iree_ok_status();
}

static iree_status_t loom_low_dce_function(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_func_op = function.op;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_function_target(module, low_func_op, descriptor_registry,
                                       target_selection, emitter, &target));
  if (!target.descriptor_set) {
    return iree_ok_status();
  }

  loom_low_dce_deadness_context_t deadness_context = {
      .target = &target,
  };
  const loom_dce_deadness_query_callback_t deadness_query = {
      .fn = loom_low_dce_deadness_query,
      .user_data = &deadness_context,
  };
  return loom_dce_run_with_deadness_query(pass, module, function,
                                          deadness_query);
}

iree_status_t loom_low_dce_run(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_target_selection_t target_selection =
      loom_target_pass_capability_target_selection(target_capability);

  return loom_low_dce_function(pass, module, function, descriptor_registry,
                               target_selection, pass->diagnostic_emitter);
}
