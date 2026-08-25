// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"

static bool loom_amdgpu_address_i64_alu_kind_uses_vgpr(
    loom_amdgpu_address_i64_alu_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO:
      return true;
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD:
      return false;
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU address i64 ALU plan kind");
  return false;
}

iree_status_t loom_amdgpu_lower_index_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_index_cast_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32: {
      IREE_ASSERT_EQ(plan->index_bitwidth, 32u);
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32: {
      IREE_ASSERT_EQ(plan->index_bitwidth, 32u);
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      const loom_module_t* module = loom_low_lower_context_module(context);
      const loom_type_t source_type =
          loom_module_value_type(module, low_source);
      const loom_type_t lane_type =
          loom_low_register_carrier_type_with_unit_count(source_type, 1);
      loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                      plan->zero_descriptor_ref,
                                                      0, lane_type, &low_zero));
      const loom_value_id_t lanes[] = {
          low_source,
          low_zero,
      };
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
          context, source_op, lanes, IREE_ARRAYSIZE(lanes), result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED:
      return iree_ok_status();
    case LOOM_AMDGPU_INDEX_CAST_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU index cast plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_materialize_address_i64_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_amdgpu_address_i64_alu_kind_t kind,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address i64 ALU plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);

  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(kind)) {
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    if (is_vgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (unit_count == 2) {
      return loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_value, out_low_value);
    }
    if (unit_count == 1) {
      return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  } else {
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (is_sgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (is_sgpr && unit_count == 1) {
      return loom_amdgpu_emit_sgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU address i64 ALU plan selected incompatible register type");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_i64_compare_operand_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_source);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 1 && lane_index == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_low_lane);
  }
  if (unit_count != 1 && unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected wrong operand register count");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_type_t source_lane_type =
      loom_low_register_carrier_type_with_unit_count(low_type, 1);
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, unit_count, lane_index, source_lane_type,
      out_low_lane));
  return loom_amdgpu_materialize_low_vgpr_b32(context, source_op, *out_low_lane,
                                              out_low_lane);
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* combine_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &combine_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(combine_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_i64_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_i64_compare_plan_t* plan) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 0, vgpr_type, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 1, vgpr_type, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 0, vgpr_type, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 1, vgpr_type, &rhs_hi));

  loom_value_id_t high_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->high_descriptor_ref, lhs_hi, rhs_hi, mask_type,
      &high_mask));
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->low_descriptor_ref, lhs_lo, rhs_lo, mask_type,
      &low_mask));

  loom_value_id_t combined_low_mask = low_mask;
  if (plan->needs_high_equal) {
    loom_value_id_t high_equal_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, lhs_hi,
        rhs_hi, mask_type, &high_equal_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        high_equal_mask, low_mask, mask_type, &combined_low_mask));
  }

  loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
      context, source_op, plan->combine_descriptor_ref, high_mask,
      combined_low_mask, mask_type, &result_mask));
  return loom_low_lower_bind_value(context, plan->result, result_mask);
}

iree_status_t loom_amdgpu_extract_low_32_bits_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source_pair));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source_pair);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar source lowered to a non-register type");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source_pair, /*lane_offset=*/0, source_lane_type,
      &low_source_lane));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source_lane, out_low_source);
}

iree_status_t loom_amdgpu_lower_address_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_address_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t product = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &product));
      loom_value_id_t low_addend = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->addend, plan->kind, &low_addend));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, product, low_addend, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU address i64 ALU plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr64_lshr_literal(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint8_t shift_amount,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(shift_amount, 64u);
  if (shift_amount == 0) {
    *out_low_result = low_value;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, low_value);
  IREE_ASSERT(loom_amdgpu_low_type_is_register_class_count(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));
  const loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(result_type, 1);

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_value, /*offset=*/0, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_value, /*offset=*/1, lane_type, &high_lane));

  loom_value_id_t result_low_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result_high_lane = LOOM_VALUE_ID_INVALID;
  if (shift_amount < 32) {
    loom_value_id_t shifted_low_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        shift_amount, low_lane, lane_type, &shifted_low_lane));
    loom_value_id_t carried_high_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        32u - shift_amount, high_lane, lane_type, &carried_high_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
        shifted_low_lane, carried_high_lane, lane_type, &result_low_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        shift_amount, high_lane, lane_type, &result_high_lane));
  } else {
    if (shift_amount == 32) {
      result_low_lane = high_lane;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          shift_amount - 32u, high_lane, lane_type, &result_low_lane));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &result_high_lane));
  }

  const loom_value_id_t result_lanes[] = {
      result_low_lane,
      result_high_lane,
  };
  return loom_amdgpu_build_low_register_range(context, source_op, result_lanes,
                                              IREE_ARRAYSIZE(result_lanes),
                                              result_type, out_low_result);
}

iree_status_t loom_amdgpu_lower_scalar_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_value));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_lshr_literal(
          context, source_op, low_value, plan->shift_amount, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU scalar i64 ALU plan kind");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_scalar_i64_ctpop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_ctpop_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const uint32_t result_unit_count =
      loom_low_register_type_unit_count(result_type);
  loom_value_id_t low_count = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32:
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
      loom_op_t* count_op = NULL;
      const loom_value_id_t operands[] = {low_source};
      const loom_amdgpu_descriptor_ref_t descriptor_ref =
          plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32
              ? LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B32
              : LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B64;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, descriptor_ref, operands,
          IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sgpr_type,
          1, &count_op));
      low_count = loom_value_slice_get(loom_low_op_results(count_op), 0);
      if (result_unit_count == 2) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
            sgpr_type, &low_zero));
      }
      break;
    }
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32:
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_source, &low_source));
      loom_type_t vgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
      loom_value_id_t low_source_half = low_source;
      if (plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, low_source, /*lane_offset=*/0, vgpr_type,
            &low_source_half));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32_SRC1_ZERO, low_source_half,
          vgpr_type, &low_count));
      if (plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64) {
        loom_value_id_t high_source_half = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, low_source, /*lane_offset=*/1, vgpr_type,
            &high_source_half));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32,
            high_source_half, low_count, vgpr_type, &low_count));
      }
      if (result_unit_count == 2) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
            vgpr_type, &low_zero));
      }
      break;
    }
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_NONE:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU scalar i64 ctpop plan kind");
      return iree_ok_status();
  }

  if (result_unit_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_count);
  }
  const loom_value_id_t low_halves[] = {
      low_count,
      low_zero,
  };
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_halves, IREE_ARRAYSIZE(low_halves), result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
