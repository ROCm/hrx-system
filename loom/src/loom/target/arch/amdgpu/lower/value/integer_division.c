// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/integer_division.h"

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/util/fact_table.h"

typedef enum loom_amdgpu_wide_division_domain_e {
  LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_UNSIGNED = 0,
  LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED = 1,
} loom_amdgpu_wide_division_domain_t;

static loom_amdgpu_wide_division_domain_t loom_amdgpu_wide_division_domain(
    const loom_op_t* op) {
  return loom_scalar_divsi_isa(op) || loom_scalar_remsi_isa(op) ||
                 loom_scalar_divremsi_isa(op)
             ? LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED
             : LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_UNSIGNED;
}

static void loom_amdgpu_wide_division_results(const loom_op_t* op,
                                              loom_value_id_t* out_quotient,
                                              loom_value_id_t* out_remainder) {
  *out_quotient = LOOM_VALUE_ID_INVALID;
  *out_remainder = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t* results = loom_op_const_results(op);
  switch (op->kind) {
    case LOOM_OP_SCALAR_DIVSI:
    case LOOM_OP_SCALAR_DIVUI:
    case LOOM_OP_INDEX_DIV:
      *out_quotient = results[0];
      return;
    case LOOM_OP_SCALAR_REMSI:
    case LOOM_OP_SCALAR_REMUI:
    case LOOM_OP_INDEX_REM:
      *out_remainder = results[0];
      return;
    case LOOM_OP_SCALAR_DIVREMSI:
    case LOOM_OP_SCALAR_DIVREMUI:
    case LOOM_OP_INDEX_DIVREM:
      *out_quotient = results[0];
      *out_remainder = results[1];
      return;
    default:
      IREE_ASSERT_UNREACHABLE("unexpected wide integer division operation");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_wide_divisor(const loom_value_fact_table_t* facts,
                                     loom_value_id_t rhs,
                                     uint64_t* out_divisor) {
  int64_t value = 0;
  if (!facts || !loom_value_facts_as_exact_i64(
                    loom_value_fact_table_lookup(facts, rhs), &value)) {
    return false;
  }
  if (value == 0) {
    return false;
  }
  *out_divisor = (uint64_t)value;
  return true;
}

static bool loom_amdgpu_wide_division_descriptors_supported(
    const loom_low_descriptor_set_t* descriptors, uint16_t register_class_id,
    loom_amdgpu_wide_division_domain_t domain) {
  static const loom_amdgpu_descriptor_ref_t sgpr[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_HI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_SUBB_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B64,
  };
  static const loom_amdgpu_descriptor_ref_t vgpr[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B64,
  };
  static const loom_amdgpu_descriptor_ref_t signed_sgpr[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I64,
  };
  static const loom_amdgpu_descriptor_ref_t signed_vgpr[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I64,
  };
  const bool uniform = register_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  if (!loom_amdgpu_descriptor_set_has_all_refs(
          descriptors, uniform ? sgpr : vgpr,
          uniform ? IREE_ARRAYSIZE(sgpr) : IREE_ARRAYSIZE(vgpr))) {
    return false;
  }
  return domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_UNSIGNED ||
         loom_amdgpu_descriptor_set_has_all_refs(
             descriptors, uniform ? signed_sgpr : signed_vgpr,
             uniform ? IREE_ARRAYSIZE(signed_sgpr)
                     : IREE_ARRAYSIZE(signed_vgpr));
}

iree_status_t loom_amdgpu_select_wide_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_wide_division_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_wide_division_plan_t){
      .source = LOOM_VALUE_ID_INVALID,
      .quotient = LOOM_VALUE_ID_INVALID,
      .remainder = LOOM_VALUE_ID_INVALID,
  };
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  const loom_scalar_type_t source_element_type =
      loom_type_element_type(loom_module_value_type(module, operands[0]));
  if (source_element_type != LOOM_SCALAR_TYPE_I64 &&
      source_element_type != LOOM_SCALAR_TYPE_INDEX) {
    return iree_ok_status();
  }
  const loom_amdgpu_wide_division_domain_t domain =
      loom_amdgpu_wide_division_domain(source_op);
  uint64_t divisor = 0;
  if (!loom_amdgpu_wide_divisor(loom_low_lower_context_fact_table(context),
                                operands[1], &divisor)) {
    return iree_ok_status();
  }

  loom_value_id_t quotient = LOOM_VALUE_ID_INVALID;
  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_wide_division_results(source_op, &quotient, &remainder);
  const loom_value_id_t results[] = {quotient, remainder};
  uint8_t result_unit_counts[] = {0, 0};
  uint16_t register_class_id = 0;
  bool register_class_selected = false;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                     results[i], &result_type));
    uint16_t result_class_id = 0;
    uint8_t result_unit_count = 0;
    bool result_type_supported = false;
    for (uint8_t unit_count = 1; unit_count <= 2; ++unit_count) {
      if (loom_amdgpu_low_type_is_register_class_count(
              context, result_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
              unit_count)) {
        result_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
        result_unit_count = unit_count;
        result_type_supported = true;
        break;
      }
      if (loom_amdgpu_low_type_is_register_class_count(
              context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
              unit_count)) {
        result_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
        result_unit_count = unit_count;
        result_type_supported = true;
        break;
      }
    }
    if (!result_type_supported ||
        (register_class_selected && register_class_id != result_class_id)) {
      return iree_ok_status();
    }
    register_class_id = result_class_id;
    register_class_selected = true;
    result_unit_counts[i] = result_unit_count;
  }

  loom_type_t source_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, operands[0], &source_type));
  if (!loom_amdgpu_low_type_is_register_class_count(context, source_type,
                                                    register_class_id, 2) &&
      !(register_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
        loom_amdgpu_low_type_is_register_class_count(
            context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2))) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_wide_division_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), register_class_id,
          domain)) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_wide_division_plan_t){
      .source = operands[0],
      .quotient = quotient,
      .remainder = remainder,
      .divisor = divisor,
      .register_class_id = register_class_id,
      .quotient_result_unit_count = result_unit_counts[0],
      .remainder_result_unit_count = result_unit_counts[1],
  };
  if (domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED) {
    const int64_t signed_divisor = (int64_t)divisor;
    if (signed_divisor != 1 && signed_divisor != -1) {
      out_plan->magic.signed_value =
          loom_low_lower_signed_divisor_magic_info(signed_divisor, 64);
    }
  } else {
    const uint64_t numerator_maximum =
        loom_low_lower_unsigned_numerator_maximum(
            loom_low_lower_context_fact_table(context), operands[0], 64);
    out_plan->quotient_unit_count =
        numerator_maximum / divisor <= UINT32_MAX ? 1 : 2;
    if (divisor > 1) {
      out_plan->magic.unsigned_value =
          loom_low_lower_unsigned_divisor_magic_info(divisor, 64,
                                                     numerator_maximum);
    }
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_wide_division_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_wide_division_plan_t* plan) {
  const loom_value_id_t results[] = {plan->quotient, plan->remainder};
  const uint8_t result_unit_counts[] = {plan->quotient_result_unit_count,
                                        plan->remainder_result_unit_count};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_type_t result_type = loom_low_register_type(
        loom_low_lower_context_descriptor_set(context)->stable_id,
        plan->register_class_id, result_unit_counts[i]);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_plan_value_type(context, results[i], result_type));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_wide_division(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_target_low_legality_fact_table(context);
  const loom_value_id_t source = loom_op_const_operands(op)[0];
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_scalar_type_t source_element_type =
      loom_type_element_type(source_type);
  const bool is_index = source_element_type == LOOM_SCALAR_TYPE_INDEX;
  if (source_element_type != LOOM_SCALAR_TYPE_I64 &&
      (!is_index || !loom_amdgpu_source_address_value_needs_64bit(
                        module, fact_table, source, source_type))) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_amdgpu_wide_division_domain_t domain =
      loom_amdgpu_wide_division_domain(op);
  const bool is_unsigned_remainder = loom_scalar_remui_isa(op);
  uint64_t divisor = 0;
  if (!loom_amdgpu_wide_divisor(fact_table, loom_op_const_operands(op)[1],
                                &divisor)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        is_index ? IREE_SV("unsigned_index64_division.constant_nonzero_divisor")
        : domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED
            ? IREE_SV("signed_i64_division.constant_nonzero_divisor")
        : is_unsigned_remainder
            ? IREE_SV("unsigned_i64_remainder."
                      "constant_nonzero_or_positive_u32_divisor")
            : IREE_SV("unsigned_i64_division.constant_nonzero_divisor"));
  }

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, loom_op_const_results(op)[0], &result_prefers_vgpr));
  if (!loom_amdgpu_wide_division_descriptors_supported(
          loom_target_low_legality_descriptor_set(context),
          result_prefers_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                              : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          domain)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        is_index ? IREE_SV("unsigned_index64_division.word_arithmetic")
        : domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED
            ? IREE_SV("signed_i64_division.word_arithmetic")
            : IREE_SV("unsigned_i64_division.word_arithmetic"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_division_emit_constant(
    loom_low_lower_context_t* context, const loom_op_t* op, uint64_t value,
    loom_type_t result_type, loom_value_id_t* out_value) {
  const loom_type_t word_type =
      loom_low_register_carrier_type_with_unit_count(result_type, 1);
  const loom_amdgpu_descriptor_ref_t move =
      loom_low_register_type_class_id(result_type) ==
              LOOM_AMDGPU_REG_CLASS_ID_SGPR
          ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  loom_value_id_t words[2];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, op, move, (uint32_t)value, word_type, &words[0]));
  if (loom_low_register_type_unit_count(result_type) == 1) {
    *out_value = words[0];
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, op, move, (uint32_t)(value >> 32), word_type, &words[1]));
  return loom_amdgpu_build_low_register_range(
      context, op, words, IREE_ARRAYSIZE(words), result_type, out_value);
}

static iree_status_t loom_amdgpu_division_emit_add(
    loom_low_lower_context_t* context, const loom_op_t* op, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  if (loom_low_register_type_class_id(type) == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_emit_sgpr64_binary_carry(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, lhs, rhs, out_value);
  }
  return loom_amdgpu_emit_vgpr64_add(context, op, lhs, rhs, out_value);
}

static iree_status_t loom_amdgpu_division_emit_subtract(
    loom_low_lower_context_t* context, const loom_op_t* op, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  if (loom_low_register_type_class_id(type) == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_emit_sgpr64_binary_carry(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_CO_U32,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_SUBB_U32, lhs, rhs, out_value);
  }
  return loom_amdgpu_emit_vgpr64_sub(context, op, lhs, rhs, out_value);
}

typedef enum loom_amdgpu_division_shift_kind_e {
  LOOM_AMDGPU_DIVISION_SHIFT_LOGICAL_RIGHT = 0,
  LOOM_AMDGPU_DIVISION_SHIFT_ARITHMETIC_RIGHT = 1,
} loom_amdgpu_division_shift_kind_t;

static iree_status_t loom_amdgpu_division_emit_shift(
    loom_low_lower_context_t* context, const loom_op_t* op,
    loom_value_id_t value, uint8_t shift,
    loom_amdgpu_division_shift_kind_t kind, loom_value_id_t* out_value) {
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value);
  const loom_type_t word_type =
      loom_low_register_carrier_type_with_unit_count(type, 1);
  const bool uniform =
      loom_low_register_type_class_id(type) == LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  loom_value_id_t amount;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_const_u32(context, op,
                                 uniform ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
                                         : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                 shift, word_type, &amount));
  return loom_amdgpu_emit_binary(
      context, op,
      uniform ? (kind == LOOM_AMDGPU_DIVISION_SHIFT_ARITHMETIC_RIGHT
                     ? LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I64
                     : LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B64)
              : (kind == LOOM_AMDGPU_DIVISION_SHIFT_ARITHMETIC_RIGHT
                     ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I64
                     : LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B64),
      uniform ? value : amount, uniform ? amount : value, type, out_value);
}

static iree_status_t loom_amdgpu_division_emit_and(
    loom_low_lower_context_t* context, const loom_op_t* op, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  const loom_type_t pair_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  if (loom_low_register_type_class_id(pair_type) ==
      LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_emit_binary(context, op,
                                   LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, lhs,
                                   rhs, pair_type, out_value);
  }
  const loom_type_t word_type =
      loom_low_register_carrier_type_with_unit_count(pair_type, 1);
  loom_value_id_t words[2];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(words); ++i) {
    loom_value_id_t lhs_word;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, op, lhs, i, word_type, &lhs_word));
    loom_value_id_t rhs_word;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, op, rhs, i, word_type, &rhs_word));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32, lhs_word, rhs_word,
        word_type, &words[i]));
  }
  return loom_amdgpu_build_low_register_range(
      context, op, words, IREE_ARRAYSIZE(words), pair_type, out_value);
}

// At radix B=2^32, form t1=a1*b0+hi(a0*b0), t2=a0*b1+lo(t1),
// then high64(a*b)=a1*b1+hi(t1)+hi(t2). Each temporary is an exact u64:
// a 32x32 product is at most (B-1)^2 and its added word is at most B-1.
// Native carry results stay explicit SSA edges through scheduling/allocation.
static iree_status_t loom_amdgpu_division_emit_multiply_high(
    loom_low_lower_context_t* context, const loom_op_t* op, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  const loom_type_t pair_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  const loom_type_t word_type =
      loom_low_register_carrier_type_with_unit_count(pair_type, 1);
  const bool uniform = loom_low_register_type_class_id(pair_type) ==
                       LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  const loom_amdgpu_descriptor_ref_t multiply_low =
      uniform ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32
              : LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32;
  const loom_amdgpu_descriptor_ref_t multiply_high =
      uniform ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_HI_U32
              : LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32;
  loom_value_id_t lhs_words[2], rhs_words[2];
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, op, lhs, i,
                                                    word_type, &lhs_words[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, op, rhs, i,
                                                    word_type, &rhs_words[i]));
  }
  loom_value_id_t zero;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_const_u32(context, op,
                                 uniform ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
                                         : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                 0, word_type, &zero));
  loom_value_id_t low_product_high;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(context, op, multiply_high,
                                               lhs_words[0], rhs_words[0],
                                               word_type, &low_product_high));

  // Accumulate each cross product before publishing its high word. Splitting
  // a pair here never discards the carry from the preceding low-word sum.
  loom_value_id_t cross_high[2];
  loom_value_id_t addend = low_product_high;
  for (uint32_t i = 0; i < 2; ++i) {
    loom_value_id_t product_words[2];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(context, op, multiply_low,
                                                 lhs_words[1 - i], rhs_words[i],
                                                 word_type, &product_words[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(context, op, multiply_high,
                                                 lhs_words[1 - i], rhs_words[i],
                                                 word_type, &product_words[1]));
    loom_value_id_t product;
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
        context, op, product_words, 2, pair_type, &product));
    const loom_value_id_t addend_words[] = {addend, zero};
    loom_value_id_t wide_addend;
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
        context, op, addend_words, 2, pair_type, &wide_addend));
    loom_value_id_t sum;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_division_emit_add(context, op, product, wide_addend, &sum));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, op, sum, 0, word_type, &addend));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, op, sum, 1,
                                                    word_type, &cross_high[i]));
  }
  loom_value_id_t high_product_words[2];
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_binary(context, op, multiply_low, lhs_words[1],
                              rhs_words[1], word_type, &high_product_words[0]));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_binary(context, op, multiply_high, lhs_words[1],
                              rhs_words[1], word_type, &high_product_words[1]));
  loom_value_id_t result;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, op, high_product_words, 2, pair_type, &result));
  for (uint32_t i = 0; i < 2; ++i) {
    const loom_value_id_t words[] = {cross_high[i], zero};
    loom_value_id_t wide_cross_high;
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
        context, op, words, 2, pair_type, &wide_cross_high));
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_add(
        context, op, result, wide_cross_high, &result));
  }
  *out_value = result;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_division_emit_unsigned_quotient(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_wide_division_plan_t* plan, loom_value_id_t numerator,
    loom_type_t pair_type, loom_value_id_t* out_quotient) {
  if (plan->divisor == 1) {
    *out_quotient = numerator;
    return iree_ok_status();
  }
  if (iree_is_power_of_two_uint64(plan->divisor)) {
    return loom_amdgpu_division_emit_shift(
        context, source_op, numerator,
        (uint8_t)iree_math_floor_log2_u64(plan->divisor),
        LOOM_AMDGPU_DIVISION_SHIFT_LOGICAL_RIGHT, out_quotient);
  }
  const loom_low_lower_unsigned_divisor_magic_info_t magic =
      plan->magic.unsigned_value;
  if (magic.multiplier == 0) {
    return loom_amdgpu_division_emit_constant(context, source_op, 0, pair_type,
                                              out_quotient);
  }
  loom_value_id_t multiplier;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_constant(
      context, source_op, magic.multiplier, pair_type, &multiplier));
  loom_value_id_t quotient;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_multiply_high(
      context, source_op, numerator, multiplier, &quotient));
  if (magic.is_add) {
    loom_value_id_t difference;
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_subtract(
        context, source_op, numerator, quotient, &difference));
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_shift(
        context, source_op, difference, 1,
        LOOM_AMDGPU_DIVISION_SHIFT_LOGICAL_RIGHT, &difference));
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_add(
        context, source_op, difference, quotient, &quotient));
  }
  return loom_amdgpu_division_emit_shift(
      context, source_op, quotient, magic.post_shift,
      LOOM_AMDGPU_DIVISION_SHIFT_LOGICAL_RIGHT, out_quotient);
}

static iree_status_t loom_amdgpu_division_emit_signed_quotient(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_wide_division_plan_t* plan, loom_value_id_t numerator,
    loom_type_t pair_type, loom_value_id_t* out_quotient) {
  const int64_t divisor = (int64_t)plan->divisor;
  if (divisor == 1) {
    *out_quotient = numerator;
    return iree_ok_status();
  }
  if (divisor == -1) {
    loom_value_id_t zero;
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_constant(
        context, source_op, 0, pair_type, &zero));
    return loom_amdgpu_division_emit_subtract(context, source_op, zero,
                                              numerator, out_quotient);
  }

  const loom_low_lower_signed_divisor_magic_info_t magic =
      plan->magic.signed_value;
  loom_value_id_t multiplier;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_constant(
      context, source_op, magic.multiplier, pair_type, &multiplier));
  loom_value_id_t quotient;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_multiply_high(
      context, source_op, numerator, multiplier, &quotient));

  // Convert the unsigned high product to the adjusted signed high product in
  // one combined recipe. The numerator sign contributes -multiplier when
  // negative. The signed magic numerator coefficient cancels the multiplier
  // sign correction for positive divisors and combines to -numerator for every
  // negative divisor.
  loom_value_id_t numerator_sign;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_shift(
      context, source_op, numerator, 63,
      LOOM_AMDGPU_DIVISION_SHIFT_ARITHMETIC_RIGHT, &numerator_sign));
  loom_value_id_t signed_high_correction;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_and(
      context, source_op, numerator_sign, multiplier, &signed_high_correction));
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_subtract(
      context, source_op, quotient, signed_high_correction, &quotient));
  if (divisor < 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_subtract(
        context, source_op, quotient, numerator, &quotient));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_shift(
      context, source_op, quotient, magic.post_shift,
      LOOM_AMDGPU_DIVISION_SHIFT_ARITHMETIC_RIGHT, &quotient));
  loom_value_id_t sign_correction;
  IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_shift(
      context, source_op, quotient, 63,
      LOOM_AMDGPU_DIVISION_SHIFT_LOGICAL_RIGHT, &sign_correction));
  return loom_amdgpu_division_emit_add(context, source_op, quotient,
                                       sign_correction, out_quotient);
}

iree_status_t loom_amdgpu_lower_wide_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_wide_division_plan_t* plan) {
  loom_value_id_t numerator =
      loom_low_lower_lookup_value(context, plan->source);
  if (plan->register_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, numerator, &numerator));
  }
  const loom_type_t pair_type =
      loom_module_value_type(loom_low_lower_context_module(context), numerator);
  const loom_amdgpu_wide_division_domain_t domain =
      loom_amdgpu_wide_division_domain(source_op);

  loom_value_id_t quotient;
  if (domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_signed_quotient(
        context, source_op, plan, numerator, pair_type, &quotient));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_unsigned_quotient(
        context, source_op, plan, numerator, pair_type, &quotient));
  }
  if (plan->quotient != LOOM_VALUE_ID_INVALID) {
    loom_value_id_t quotient_result = quotient;
    if (plan->quotient_result_unit_count == 1) {
      const loom_type_t word_type =
          loom_low_register_carrier_type_with_unit_count(pair_type, 1);
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, quotient, 0, word_type, &quotient_result));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, plan->quotient, quotient_result));
  }
  if (plan->remainder != LOOM_VALUE_ID_INVALID) {
    const loom_type_t word_type =
        loom_low_register_carrier_type_with_unit_count(pair_type, 1);
    loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
    if (plan->divisor == 1 ||
        (domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_SIGNED &&
         (int64_t)plan->divisor == -1)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_constant(
          context, source_op, 0, pair_type, &remainder));
    } else if (domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_UNSIGNED &&
               plan->magic.unsigned_value.multiplier == 0) {
      remainder = numerator;
    } else {
      loom_type_t divisor_type = pair_type;
      loom_value_id_t product_quotient = quotient;
      if (domain == LOOM_AMDGPU_WIDE_DIVISION_DOMAIN_UNSIGNED &&
          plan->quotient_unit_count == 1) {
        divisor_type = word_type;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, quotient, 0, divisor_type, &product_quotient));
      }
      loom_value_id_t divisor;
      IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_constant(
          context, source_op, plan->divisor,
          plan->divisor <= UINT32_MAX ? word_type : pair_type, &divisor));
      loom_value_id_t product;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_mul_lo(
          context, source_op, product_quotient, divisor, &product));
      IREE_RETURN_IF_ERROR(loom_amdgpu_division_emit_subtract(
          context, source_op, numerator, product, &remainder));
    }
    if (plan->remainder_result_unit_count == 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, remainder, 0, word_type, &remainder));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, plan->remainder, remainder));
  }
  return iree_ok_status();
}
