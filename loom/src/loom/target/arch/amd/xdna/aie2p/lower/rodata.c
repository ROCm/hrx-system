// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/rodata.h"

#include "loom/ops/global/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

typedef enum loom_aie2p_rodata_plan_kind_e {
  LOOM_AIE2P_RODATA_PLAN_ADDRESS = 0x200,
} loom_aie2p_rodata_plan_kind_t;

bool loom_aie2p_rodata_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_RODATA_PLAN_ADDRESS;
}

iree_status_t loom_aie2p_select_rodata_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_global_load_isa(source_op)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_symbol_ref_t symbol = loom_global_load_global(source_op);
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  if (loom_global_rodata_def_isa(definition)) {
    *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_RODATA_PLAN_ADDRESS, NULL);
  }
  return iree_ok_status();
}

void loom_aie2p_mark_rodata_plan_demands(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)context;
  (void)source_op;
  (void)plan;
}

void loom_aie2p_describe_rodata_plan(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  (void)plan;
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = IREE_SV("read-only-data.address"),
  };
}

iree_status_t loom_aie2p_emit_rodata_plan(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  (void)plan;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_string_id_t immediate_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("i"), &immediate_name));
  const loom_named_attr_t attr = {
      .name_id = immediate_name,
      .value = loom_attr_symbol(loom_global_load_global(source_op)),
  };
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &result_type));
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(context)->descriptors
               [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attr, 1), &result_type, /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, loom_value_slice_get(loom_global_load_result(source_op), 0),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}
