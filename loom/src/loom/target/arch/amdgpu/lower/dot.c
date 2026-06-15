// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/dot.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_DOTF_RELAXED_FOREST_MAX_ACCUMULATORS 4u

static bool loom_amdgpu_dotf_fastmath_allows_relaxed_forest(uint8_t fastmath) {
  const uint8_t required_flags =
      LOOM_VECTOR_FASTMATHFLAGS_REASSOC | LOOM_VECTOR_FASTMATHFLAGS_NNAN |
      LOOM_VECTOR_FASTMATHFLAGS_NINF | LOOM_VECTOR_FASTMATHFLAGS_NSZ |
      LOOM_VECTOR_FASTMATHFLAGS_CONTRACT;
  return iree_all_bits_set(fastmath, required_flags);
}

static iree_status_t loom_amdgpu_dotf_descriptor_present(
    loom_low_lower_context_t* context, loom_amdgpu_descriptor_ref_t ref,
    bool* out_present) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, ref, &descriptor, out_present);
}

static iree_status_t loom_amdgpu_dotf_result_is_one_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t result, bool* out_match) {
  *out_match = false;
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &low_result_type));
  if (loom_low_register_type_unit_count(low_result_type) != 1) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_type_register_class_is(
      context, low_result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, out_match);
}

static iree_status_t loom_amdgpu_dotf_low_value_is_one_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t value, bool* out_match) {
  *out_match = false;
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != 1) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_type_register_class_is(
      context, type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, out_match);
}

iree_status_t loom_amdgpu_select_vector_dotf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_dotf_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_dotf_plan_t){0};
  *out_selected = false;
  if (!loom_vector_dotf_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t lhs = loom_vector_dotf_lhs(source_op);
  const loom_value_id_t rhs = loom_vector_dotf_rhs(source_op);
  const loom_value_id_t init = loom_vector_dotf_init(source_op);
  const loom_value_id_t result = loom_vector_dotf_result(source_op);
  const loom_type_t lhs_type = loom_module_value_type(module, lhs);
  const loom_type_t rhs_type = loom_module_value_type(module, rhs);
  const loom_type_t init_type = loom_module_value_type(module, init);
  const loom_type_t result_type = loom_module_value_type(module, result);

  const uint32_t lane_count = loom_amdgpu_vector_f32_lane_count(lhs_type);
  if (lane_count == 0 ||
      loom_amdgpu_vector_f32_lane_count(rhs_type) != lane_count ||
      !loom_amdgpu_type_is_f32(init_type) ||
      !loom_amdgpu_type_is_f32(result_type)) {
    return iree_ok_status();
  }

  bool result_is_one_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_result_is_one_vgpr(
      context, source_op, result, &result_is_one_vgpr));
  if (!result_is_one_vgpr) {
    return iree_ok_status();
  }

  bool fma_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_descriptor_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_F32, &fma_present));
  if (!fma_present) {
    return iree_ok_status();
  }
  loom_amdgpu_descriptor_ref_t tied_accumulate_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool fmac_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_descriptor_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAC_F32, &fmac_present));
  if (fmac_present) {
    tied_accumulate_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAC_F32;
  }

  loom_amdgpu_dotf_accumulation_kind_t accumulation_kind =
      LOOM_AMDGPU_DOTF_ACCUMULATION_STRICT_CHAIN;
  if (loom_amdgpu_dotf_fastmath_allows_relaxed_forest(
          loom_vector_dotf_fastmath(source_op))) {
    bool multiply_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_descriptor_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, &multiply_present));
    bool add_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_descriptor_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32, &add_present));
    if (multiply_present && add_present) {
      accumulation_kind = LOOM_AMDGPU_DOTF_ACCUMULATION_RELAXED_FOREST;
    }
  }

  uint32_t init_bit_pattern = 0;
  const loom_amdgpu_dotf_init_kind_t init_kind =
      loom_amdgpu_value_as_f32_constant(context, init, &init_bit_pattern) &&
              init_bit_pattern == 0
          ? LOOM_AMDGPU_DOTF_INIT_ZERO
          : LOOM_AMDGPU_DOTF_INIT_GENERIC;

  out_plan->lhs = lhs;
  out_plan->rhs = rhs;
  out_plan->init = init;
  out_plan->result = result;
  out_plan->lane_count = lane_count;
  out_plan->accumulation_kind = accumulation_kind;
  out_plan->init_kind = init_kind;
  out_plan->tied_accumulate_descriptor_ref = tied_accumulate_descriptor_ref;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dotf_emit_lane_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dotf_emit_mul(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_amdgpu_dotf_emit_lane_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, operands,
      IREE_ARRAYSIZE(operands), lane_type, out_result);
}

static iree_status_t loom_amdgpu_dotf_emit_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_amdgpu_dotf_emit_lane_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32, operands,
      IREE_ARRAYSIZE(operands), lane_type, out_result);
}

static iree_status_t loom_amdgpu_dotf_emit_fma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t accumulator,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs, accumulator};
  return loom_amdgpu_dotf_emit_lane_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_F32, operands,
      IREE_ARRAYSIZE(operands), lane_type, out_result);
}

static iree_status_t loom_amdgpu_dotf_emit_fmac(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t accumulator, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {accumulator, lhs, rhs};
  const loom_tied_result_t tied_results[] = {
      {
          .result_index = 0,
          .operand_index = 0,
          .has_type_change = false,
      },
  };
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), source_op->location, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_dotf_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return IREE_SV("<missing>");
  }
  const iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_is_empty(descriptor_set_key) ? IREE_SV("<empty>")
                                                       : descriptor_set_key;
}

static iree_status_t loom_amdgpu_dotf_emit_tied_accumulator_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t lane,
    bool accumulator_is_dot_local, bool selected_tied_form,
    iree_string_view_t reason) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t descriptor_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor.descriptor->key_string_offset);
  const iree_string_view_t descriptor_set_name =
      loom_amdgpu_dotf_descriptor_set_key(descriptor_set);
  const iree_string_view_t accumulator_kind =
      accumulator_is_dot_local ? IREE_SV("dot_local") : IREE_SV("incoming");
  const iree_string_view_t decision =
      selected_tied_form ? IREE_SV("selected") : IREE_SV("rejected");
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_u32(lane),
      loom_param_string(descriptor_set_name),
      loom_param_string(accumulator_kind),
      loom_param_string(decision),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_028_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_dotf_emit_accumulate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t accumulator, uint32_t lane,
    bool accumulator_is_dot_local, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  if (!accumulator_is_dot_local) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_tied_accumulator_diagnostic(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_F32, lane,
        accumulator_is_dot_local, /*selected_tied_form=*/false,
        IREE_SV("incoming_accumulator")));
    return loom_amdgpu_dotf_emit_fma(context, source_op, lhs, rhs, accumulator,
                                     lane_type, out_result);
  }

  if (plan->tied_accumulate_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_tied_accumulator_diagnostic(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_F32, lane,
        accumulator_is_dot_local, /*selected_tied_form=*/false,
        IREE_SV("tied_descriptor_unavailable")));
    return loom_amdgpu_dotf_emit_fma(context, source_op, lhs, rhs, accumulator,
                                     lane_type, out_result);
  }

  bool rhs_is_one_vgpr = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_dotf_low_value_is_one_vgpr(context, rhs, &rhs_is_one_vgpr));
  if (!rhs_is_one_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_tied_accumulator_diagnostic(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_F32, lane,
        accumulator_is_dot_local, /*selected_tied_form=*/false,
        IREE_SV("rhs_not_one_vgpr")));
    return loom_amdgpu_dotf_emit_fma(context, source_op, lhs, rhs, accumulator,
                                     lane_type, out_result);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_tied_accumulator_diagnostic(
      context, source_op, plan->tied_accumulate_descriptor_ref, lane,
      accumulator_is_dot_local, /*selected_tied_form=*/true,
      IREE_SV("dot_local_accumulator")));
  return loom_amdgpu_dotf_emit_fmac(context, source_op,
                                    plan->tied_accumulate_descriptor_ref, lhs,
                                    rhs, accumulator, lane_type, out_result);
}

static iree_status_t loom_amdgpu_dotf_extract_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint32_t lane_count, uint32_t lane,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = source;
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_module_value_type(loom_low_lower_context_module(context), source);
  if (!loom_low_type_is_register(source_type) ||
      loom_low_register_type_unit_count(source_type) <= lane) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid AMDGPU dotf lane source type");
  }
  const loom_type_t lane_type =
      loom_low_register_type_with_unit_count(source_type, 1);
  return loom_amdgpu_emit_low_slice(context, source_op, source, lane, lane_type,
                                    out_lane);
}

static iree_status_t loom_amdgpu_dotf_extract_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, uint32_t lane, loom_value_id_t* out_lhs_lane,
    loom_value_id_t* out_rhs_lane) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_extract_lane(
      context, source_op, low_lhs, plan->lane_count, lane, out_lhs_lane));
  return loom_amdgpu_dotf_extract_lane(context, source_op, low_rhs,
                                       plan->lane_count, lane, out_rhs_lane);
}

static iree_status_t loom_amdgpu_dotf_emit_strict_chain(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  loom_value_id_t accumulator = low_init;
  bool accumulator_is_dot_local = plan->init_kind == LOOM_AMDGPU_DOTF_INIT_ZERO;
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_dotf_extract_operands(context, source_op, plan, low_lhs,
                                          low_rhs, lane, &lhs_lane, &rhs_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_accumulate(
        context, source_op, plan, lhs_lane, rhs_lane, accumulator, lane,
        accumulator_is_dot_local, lane_type, &accumulator));
    accumulator_is_dot_local = true;
  }
  *out_result = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dotf_emit_accumulator_sum(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* accumulators, uint32_t accumulator_count,
    loom_type_t lane_type, loom_value_id_t* out_result) {
  while (accumulator_count > 1) {
    uint32_t next_count = 0;
    uint32_t i = 0;
    for (; i + 1 < accumulator_count; i += 2) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_add(
          context, source_op, accumulators[i], accumulators[i + 1], lane_type,
          &accumulators[next_count++]));
    }
    if (i < accumulator_count) {
      accumulators[next_count++] = accumulators[i];
    }
    accumulator_count = next_count;
  }
  *out_result = accumulators[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dotf_emit_relaxed_forest(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  const uint32_t accumulator_count =
      plan->lane_count < LOOM_AMDGPU_DOTF_RELAXED_FOREST_MAX_ACCUMULATORS
          ? plan->lane_count
          : LOOM_AMDGPU_DOTF_RELAXED_FOREST_MAX_ACCUMULATORS;
  loom_value_id_t
      accumulators[LOOM_AMDGPU_DOTF_RELAXED_FOREST_MAX_ACCUMULATORS];

  uint32_t lane = 0;
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_extract_operands(
      context, source_op, plan, low_lhs, low_rhs, lane, &lhs_lane, &rhs_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_accumulate(
      context, source_op, plan, lhs_lane, rhs_lane, low_init, lane,
      /*accumulator_is_dot_local=*/plan->init_kind ==
          LOOM_AMDGPU_DOTF_INIT_ZERO,
      lane_type, &accumulators[0]));
  ++lane;

  for (; lane < accumulator_count; ++lane) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_dotf_extract_operands(context, source_op, plan, low_lhs,
                                          low_rhs, lane, &lhs_lane, &rhs_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_mul(context, source_op, lhs_lane,
                                                   rhs_lane, lane_type,
                                                   &accumulators[lane]));
  }

  for (; lane < plan->lane_count; ++lane) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_dotf_extract_operands(context, source_op, plan, low_lhs,
                                          low_rhs, lane, &lhs_lane, &rhs_lane));
    const uint32_t accumulator_index = lane % accumulator_count;
    IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_accumulate(
        context, source_op, plan, lhs_lane, rhs_lane,
        accumulators[accumulator_index], lane,
        /*accumulator_is_dot_local=*/true, lane_type,
        &accumulators[accumulator_index]));
  }

  return loom_amdgpu_dotf_emit_accumulator_sum(context, source_op, accumulators,
                                               accumulator_count, lane_type,
                                               out_result);
}

iree_status_t loom_amdgpu_lower_vector_dotf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan) {
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->rhs, &low_rhs));
  loom_value_id_t low_init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->init, &low_init));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  switch (plan->accumulation_kind) {
    case LOOM_AMDGPU_DOTF_ACCUMULATION_STRICT_CHAIN: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_strict_chain(
          context, source_op, plan, low_lhs, low_rhs, low_init, lane_type,
          &result));
      break;
    }
    case LOOM_AMDGPU_DOTF_ACCUMULATION_RELAXED_FOREST: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_dotf_emit_relaxed_forest(
          context, source_op, plan, low_lhs, low_rhs, low_init, lane_type,
          &result));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported AMDGPU dotf accumulation kind");
  }
  return loom_low_lower_bind_value(context, plan->result, result);
}
