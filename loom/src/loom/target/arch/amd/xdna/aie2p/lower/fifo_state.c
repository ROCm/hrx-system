// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/fifo_state.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/ops.h"

enum {
  LOOM_AIE2P_FIFO_STATE_PLAN_IMPORT = 0x404,
};

typedef struct loom_aie2p_fifo_state_import_t {
  // First source query selected for this function. Preamble emission binds its
  // two results; later queries alias those canonical values.
  const loom_op_t* source_op;
  // Native carrier shared by both architectural load-FIFO states.
  loom_type_t low_type;
} loom_aie2p_fifo_state_import_t;

static const char loom_aie2p_fifo_state_import_key;

iree_status_t loom_aie2p_query_fifo_state_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  (void)environment;
  *out_result = loom_target_contract_query_result_empty();
  if (loom_aie2p_load_fifo_state_isa(source_op)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_aie2p_fifo_state_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_FIFO_STATE_PLAN_IMPORT;
}

iree_status_t loom_aie2p_select_fifo_state_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_aie2p_load_fifo_state_isa(source_op)) {
    return iree_ok_status();
  }

  loom_aie2p_fifo_state_import_t* import = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_aie2p_fifo_state_import_key, sizeof(*import),
      (void**)&import));
  if (import->source_op == NULL) {
    import->source_op = source_op;
    const loom_type_t source_type =
        loom_module_value_type(loom_low_lower_context_module(context),
                               loom_aie2p_load_fifo_state_fifo0(source_op));
    IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
        context, source_op, source_type, &import->low_type));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_aie2p_load_fifo_state_fifo0(source_op), import->low_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_aie2p_load_fifo_state_fifo1(source_op), import->low_type));
  *out_plan =
      loom_low_lower_plan_make(LOOM_AIE2P_FIFO_STATE_PLAN_IMPORT, import);
  return iree_ok_status();
}

iree_status_t loom_aie2p_emit_fifo_state_preamble(
    void* user_data, loom_low_lower_context_t* context) {
  (void)user_data;
  const loom_aie2p_fifo_state_import_t* import =
      loom_low_lower_lookup_target_state(
          context, &loom_aie2p_fifo_state_import_key, sizeof(*import));
  if (import == NULL) {
    return iree_ok_status();
  }

  static const iree_string_view_t source_names[] = {
      IREE_SVL("amd.xdna.aie2p.load_fifo.0"),
      IREE_SVL("amd.xdna.aie2p.load_fifo.1"),
  };
  const loom_value_id_t source_results[] = {
      loom_aie2p_load_fifo_state_fifo0(import->source_op),
      loom_aie2p_load_fifo_state_fifo1(import->source_op),
  };
  loom_module_t* module = loom_low_lower_context_module(context);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(source_names) && iree_status_is_ok(status); ++i) {
    loom_string_id_t source_name = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(module, source_names[i], &source_name);
    loom_op_t* live_in = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_low_live_in_build(
          loom_low_lower_context_builder(context), 0, source_name,
          loom_named_attr_slice_empty(), import->low_type,
          import->source_op->location, &live_in);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_bind_value(context, source_results[i],
                                         loom_low_live_in_result(live_in));
    }
  }
  return status;
}

void loom_aie2p_describe_fifo_state_plan(
    loom_low_lower_plan_report_t* out_report) {
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = IREE_SV("load-fifo.entry-state"),
  };
}

iree_status_t loom_aie2p_emit_fifo_state_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  const loom_aie2p_fifo_state_import_t* import = plan.target_data;
  if (source_op == import->source_op) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
      context, loom_aie2p_load_fifo_state_fifo0(import->source_op),
      loom_aie2p_load_fifo_state_fifo0(source_op)));
  return loom_low_lower_bind_value_alias(
      context, loom_aie2p_load_fifo_state_fifo1(import->source_op),
      loom_aie2p_load_fifo_state_fifo1(source_op));
}
