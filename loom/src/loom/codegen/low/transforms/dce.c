// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/dce.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/function_version.h"
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

static loom_dead_state_op_kind_t loom_low_dce_deadness_query(
    void* user_data, const loom_module_t* module, const loom_op_t* op) {
  loom_low_dce_deadness_context_t* context =
      (loom_low_dce_deadness_context_t*)user_data;

  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(context->target->descriptor_set, op,
                                        &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return loom_dead_state_classify_op(NULL, module, op);
  }

  if (op->result_count == 0 ||
      iree_any_bit_set(loom_op_effective_traits(module, op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return LOOM_DEAD_STATE_OP_ROOT;
  }

  return iree_any_bit_set(packet.descriptor->flags,
                          LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE)
             ? LOOM_DEAD_STATE_OP_VALUE
             : LOOM_DEAD_STATE_OP_ROOT;
}

static iree_status_t loom_low_dce_function(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_func_op = function.op;
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, low_func_op,
      loom_target_function_version_target_facts(pass->function_version),
      descriptor_registry, emitter, &target));
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
  return loom_low_dce_function(pass, module, function, descriptor_registry,
                               pass->diagnostic_emitter);
}
