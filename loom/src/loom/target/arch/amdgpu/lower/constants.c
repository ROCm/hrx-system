// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/constants.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_i32_immediate(attr)) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_f32_immediate(attr)) {
    return false;
  }
  *out_bit_pattern = loom_amdgpu_attr_f32_bit_pattern(attr);
  return true;
}

bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo < 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

bool loom_amdgpu_value_facts_as_exact_i32(loom_value_facts_t facts,
                                          int64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

bool loom_amdgpu_value_as_exact_i32(const loom_module_t* module,
                                    const loom_value_fact_table_t* fact_table,
                                    loom_value_id_t value_id,
                                    int64_t* out_value) {
  *out_value = 0;
  if (fact_table != NULL &&
      loom_amdgpu_value_facts_as_exact_i32(
          loom_value_fact_table_lookup(fact_table, value_id), out_value)) {
    return true;
  }
  return loom_amdgpu_module_value_as_i32_constant(module, value_id, out_value);
}

bool loom_amdgpu_value_facts_as_f32_bit_pattern(loom_value_facts_t facts,
                                                uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F32, facts, &value)) {
    return false;
  }
  const float f32_value = (float)value;
  memcpy(out_bit_pattern, &f32_value, sizeof(*out_bit_pattern));
  return true;
}

bool loom_amdgpu_value_as_f32_bit_pattern(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  if (fact_table != NULL &&
      loom_amdgpu_value_facts_as_f32_bit_pattern(
          loom_value_fact_table_lookup(fact_table, value_id),
          out_bit_pattern)) {
    return true;
  }
  return loom_amdgpu_module_value_as_f32_constant(module, value_id,
                                                  out_bit_pattern);
}

static bool loom_amdgpu_i64_value_as_u32_bits(int64_t value,
                                              uint32_t* out_bits) {
  if (value < INT32_MIN || value > UINT32_MAX) {
    return false;
  }
  *out_bits = (uint32_t)value;
  return true;
}

bool loom_amdgpu_value_facts_as_u32_bits(loom_value_facts_t facts,
                                         uint32_t* out_bits) {
  *out_bits = 0;
  if (loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts)) {
    return loom_amdgpu_value_facts_as_f32_bit_pattern(facts, out_bits);
  }
  int64_t value = 0;
  return loom_value_facts_as_exact_i64(facts, &value) &&
         loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
}

bool loom_amdgpu_source_lane_as_u32_bits(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t source, uint32_t lane, uint32_t* out_bits) {
  *out_bits = 0;
  if (fact_table == NULL || module == NULL) {
    return false;
  }

  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, source);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const bool source_is_integer_scalar =
      loom_type_is_scalar(source_type) &&
      loom_scalar_type_is_integer(loom_type_element_type(source_type));
  const int32_t source_bit_count =
      source_is_integer_scalar
          ? loom_scalar_type_bitwidth(loom_type_element_type(source_type))
          : 0;
  const uint32_t lane_offset = lane * 32u;
  int64_t exact_value = 0;
  if (source_is_integer_scalar && lane_offset < (uint32_t)source_bit_count &&
      loom_value_facts_as_exact_i64(facts, &exact_value)) {
    *out_bits = (uint32_t)((uint64_t)exact_value >> lane_offset);
    return true;
  }
  // A non-negative scalar range contained in 32 bits has zero high lanes.
  if (source_is_integer_scalar && lane_offset < (uint32_t)source_bit_count &&
      facts.range_lo >= 0 && facts.range_hi <= UINT32_MAX && lane > 0) {
    *out_bits = 0;
    return true;
  }

  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    return loom_amdgpu_value_facts_as_u32_bits(uniform.element, out_bits);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context, facts,
                                                &lanes)) {
    return lane < lanes.count &&
           loom_amdgpu_value_facts_as_u32_bits(lanes.lanes[lane], out_bits);
  }

  loom_value_fact_vector_iota_t iota = {0};
  if (loom_value_facts_query_vector_iota(&fact_table->context, facts, &iota)) {
    int64_t base = 0;
    int64_t step = 0;
    int64_t delta = 0;
    int64_t value = 0;
    return loom_value_facts_as_exact_i64(iota.base, &base) &&
           loom_value_facts_as_exact_i64(iota.step, &step) &&
           iree_checked_mul_i64((int64_t)lane, step, &delta) &&
           iree_checked_add_i64(base, delta, &value) &&
           loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
  }

  return loom_amdgpu_value_facts_as_u32_bits(facts, out_bits);
}

bool loom_amdgpu_u32_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

bool loom_amdgpu_attr_is_u32_address_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= 0 &&
         value.i64 <= UINT32_MAX;
}

bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

bool loom_amdgpu_attr_is_16bit_float_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_16bit_float_bit_pattern(loom_scalar_type_t type,
                                                  loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  switch (type) {
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16(f32_value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16(f32_value);
    default:
      IREE_ASSERT_UNREACHABLE("expected f16 or bf16");
      return 0;
  }
}

bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value) {
  *out_value = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL ? loom_amdgpu_value_as_exact_i32(
                                  module, fact_table, value_id, out_value)
                            : loom_amdgpu_module_value_as_i32_constant(
                                  module, value_id, out_value);
}

bool loom_amdgpu_value_as_i1_constant(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id,
                                      bool* out_value) {
  *out_value = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_value_facts_as_exact_bool(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return loom_amdgpu_value_as_f32_bit_pattern(module, fact_table, value_id,
                                              out_bit_pattern);
}

bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

iree_status_t loom_amdgpu_resolve_imm32_descriptor(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present) {
  *out_imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"), out_imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_select_u32_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint32_t bit_pattern,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, descriptor_ref, &descriptor, &imm32_attr_name_id,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_U32_BITS,
      .result = result,
      .descriptor = descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .register_count = 1,
  };
  out_plan->bit_patterns[0] = bit_pattern;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  if (!loom_amdgpu_attr_is_i32_immediate(value)) {
    *out_selected = false;
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)(int32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static uint32_t loom_amdgpu_integer_sign_extended_bits(int64_t value,
                                                       uint32_t bit_count) {
  IREE_ASSERT_GT(bit_count, 0);
  IREE_ASSERT_LT(bit_count, 32);
  const uint32_t low_bits =
      iree_math_mask_low_bits_u32((uint32_t)value, (int32_t)bit_count);
  const uint32_t sign_bit = UINT32_C(1) << (bit_count - 1u);
  if ((low_bits & sign_bit) == 0) {
    return low_bits;
  }
  return low_bits |
         ~iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)bit_count);
}

static bool loom_amdgpu_sub32_integer_payload_bit_count(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_count) {
  const uint32_t bit_count =
      loom_amdgpu_integer_scalar_type_bit_count(scalar_type);
  if (bit_count == 0 || bit_count >= 32) {
    *out_bit_count = 0;
    return false;
  }
  *out_bit_count = bit_count;
  return true;
}

static bool loom_amdgpu_type_sub32_integer_payload_bit_count(
    loom_type_t type, uint32_t* out_bit_count) {
  *out_bit_count = 0;
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  return loom_amdgpu_sub32_integer_payload_bit_count(
      loom_type_element_type(type), out_bit_count);
}

static iree_status_t loom_amdgpu_select_narrow_integer_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_type_t result_type,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t bit_count = 0;
  if (!loom_amdgpu_type_sub32_integer_payload_bit_count(result_type,
                                                        &bit_count) ||
      value.kind != LOOM_ATTR_I64) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_integer_sign_extended_bits(value.i64, bit_count),
      result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_i1_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t result, loom_amdgpu_constant_plan_t* out_plan,
    bool* out_selected) {
  *out_selected = false;

  bool value = false;
  if (!loom_amdgpu_value_as_i1_constant(context, result, &value)) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool is_scc = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  if (is_scc) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context,
        value ? LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32
              : LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32,
        &descriptor, &descriptor_present));
    if (!descriptor_present) {
      return iree_ok_status();
    }
    loom_low_lower_resolved_descriptor_t zero_descriptor = {0};
    loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
    bool zero_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, &zero_descriptor,
        &imm32_attr_name_id, &zero_descriptor_present));
    if (!zero_descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_constant_plan_t){
        .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_SCC,
        .result = result,
        .descriptor = descriptor,
        .zero_descriptor = zero_descriptor,
        .imm32_attr_name_id = imm32_attr_name_id,
        .i1_value = value,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  const bool is_native_mask = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!is_native_mask) {
    return iree_ok_status();
  }

  if (value) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ, &descriptor,
        &descriptor_present));
    if (!descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_constant_plan_t){
        .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK,
        .result = result,
        .descriptor = descriptor,
        .i1_value = value,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t zero_descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool zero_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, &zero_descriptor,
      &imm32_attr_name_id, &zero_descriptor_present));
  if (!zero_descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK,
      .result = result,
      .zero_descriptor = zero_descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .i1_value = value,
  };
  *out_selected = true;
  return iree_ok_status();
}

static void loom_amdgpu_repeat_first_constant_bit_pattern(
    loom_amdgpu_constant_plan_t* plan, uint32_t register_count) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  const uint32_t bit_pattern = plan->bit_patterns[0];
  plan->register_count = register_count;
  for (uint32_t i = 1; i < register_count; ++i) {
    plan->bit_patterns[i] = bit_pattern;
  }
}

static uint32_t loom_amdgpu_repeated_integer_lane_pattern(uint32_t lane_bits,
                                                          uint32_t bit_count) {
  IREE_ASSERT(bit_count == 8 || bit_count == 16);
  const uint32_t masked_lane_bits =
      lane_bits & iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)bit_count);
  uint32_t bit_pattern = 0;
  for (uint32_t bit_offset = 0; bit_offset < 32; bit_offset += bit_count) {
    bit_pattern |= masked_lane_bits << bit_offset;
  }
  return bit_pattern;
}

static iree_status_t loom_amdgpu_select_f32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_attr_is_f32_immediate(value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_attr_f32_bit_pattern(value), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_u64_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint64_t bit_pattern,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_EQ(register_count % 2u, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)bit_pattern, result, descriptor_ref, out_plan,
      out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->register_count = register_count;
  for (uint32_t i = 0; i < register_count; i += 2) {
    out_plan->bit_patterns[i] = (uint32_t)bit_pattern;
    out_plan->bit_patterns[i + 1] = (uint32_t)(bit_pattern >> 32);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i64_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (value.kind != LOOM_ATTR_I64) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_u64_bit_pattern_constant_plan(
      context, (uint64_t)value.i64, result, register_count, descriptor_ref,
      out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_f64_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (value.kind != LOOM_ATTR_F64) {
    return iree_ok_status();
  }
  uint64_t bit_pattern = 0;
  memcpy(&bit_pattern, &value.f64, sizeof(bit_pattern));
  return loom_amdgpu_select_u64_bit_pattern_constant_plan(
      context, bit_pattern, result, register_count, descriptor_ref, out_plan,
      out_selected);
}

static iree_status_t loom_amdgpu_select_packed_integer_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (value.kind != LOOM_ATTR_I64) {
    return iree_ok_status();
  }
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER ||
      (storage.element_bit_count != 8 && storage.element_bit_count != 16)) {
    return iree_ok_status();
  }
  const uint32_t lane_bits = iree_math_mask_low_bits_u32(
      (uint32_t)value.i64, (int32_t)storage.element_bit_count);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context,
      loom_amdgpu_repeated_integer_lane_pattern(lane_bits,
                                                storage.element_bit_count),
      result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan,
                                                storage.register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_packed_16bit_float_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t unused_payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          result_type, &unused_payload_bit_count, &register_count) ||
      !loom_amdgpu_attr_is_16bit_float_immediate(value)) {
    return iree_ok_status();
  }
  const uint32_t lane_bit_pattern = loom_amdgpu_attr_16bit_float_bit_pattern(
      loom_type_element_type(result_type), value);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, lane_bit_pattern | (lane_bit_pattern << 16), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_index_constant_result(source_op);
  const loom_attribute_t value = loom_index_constant_value(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, result) ||
      !loom_amdgpu_attr_is_u32_address_immediate(value)) {
    return iree_ok_status();
  }
  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), result);
  if (loom_amdgpu_type_is_i1(result_type)) {
    return loom_amdgpu_select_i1_constant_plan(context, source_op, result,
                                               out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, /*register_count=*/1, out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_f16_or_bf16(context, result)) {
    if (!loom_amdgpu_attr_is_16bit_float_immediate(value)) {
      return iree_ok_status();
    }
    const loom_type_t result_type =
        loom_module_value_type(loom_low_lower_context_module(context), result);
    return loom_amdgpu_select_u32_bit_pattern_constant_plan(
        context,
        loom_amdgpu_attr_16bit_float_bit_pattern(
            loom_type_element_type(result_type), value),
        result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_narrow_integer_constant_plan(
      context, value, result, result_type, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  if (loom_amdgpu_type_is_i64(result_type)) {
    bool result_prefers_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
        context, result, &result_prefers_vgpr));
    const loom_amdgpu_descriptor_ref_t descriptor_ref =
        result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                            : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
    return loom_amdgpu_select_i64_constant_plan(
        context, value, result, /*register_count=*/2, descriptor_ref, out_plan,
        out_selected);
  }
  if (loom_amdgpu_type_is_f64(result_type)) {
    bool result_prefers_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
        context, result, &result_prefers_vgpr));
    const loom_amdgpu_descriptor_ref_t descriptor_ref =
        result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                            : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
    return loom_amdgpu_select_f64_constant_plan(
        context, value, result, /*register_count=*/2, descriptor_ref, out_plan,
        out_selected);
  }
  if (!loom_amdgpu_type_is_i32(result_type)) {
    return iree_ok_status();
  }
  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_i32_constant_plan(
      context, value, result, descriptor_ref, out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  const uint32_t i32_register_count =
      loom_amdgpu_vector_i32_register_count(result_type);
  if (i32_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_i32_constant_plan(
        context, value, result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan,
        out_selected));
    if (!*out_selected) {
      return iree_ok_status();
    }
    loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, i32_register_count);
    return iree_ok_status();
  }
  const uint32_t f32_register_count =
      loom_amdgpu_vector_f32_register_count(result_type);
  if (f32_register_count != 0) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, f32_register_count, out_plan, out_selected);
  }
  loom_amdgpu_vector_storage_t storage = {0};
  if (loom_amdgpu_type_vector_storage(result_type, &storage) &&
      storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT) {
    if (storage.element_type == LOOM_SCALAR_TYPE_I64) {
      return loom_amdgpu_select_i64_constant_plan(
          context, value, result, storage.register_count,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
    }
    if (storage.element_type == LOOM_SCALAR_TYPE_F64) {
      return loom_amdgpu_select_f64_constant_plan(
          context, value, result, storage.register_count,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_16bit_float_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_integer_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_bind_register_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), lane_count);
  const loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(result_type, 1);

  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id,
        lane_bit_patterns[i], lane_type, &low_lane_values[i]));
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_lane_values, lane_count, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, descriptor, imm32_attr_name_id, bit_pattern,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_i1_scc_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &plan->zero_descriptor, plan->imm32_attr_name_id, 0,
      sgpr_type, &zero));
  const loom_value_id_t operands[] = {zero, zero};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &compare_op));
  return loom_low_lower_bind_value(
      context, plan->result,
      loom_value_slice_get(loom_low_op_results(compare_op), 0));
}

static iree_status_t loom_amdgpu_lower_i1_mask_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  if (plan->i1_value) {
    loom_op_t* exec_read_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor,
        /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
        &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &exec_read_op));
    return loom_low_lower_bind_value(
        context, plan->result,
        loom_value_slice_get(loom_low_op_results(exec_read_op), 0));
  }

  const uint32_t bit_patterns[] = {0, 0};
  return loom_amdgpu_bind_register_u32_lane_constants(
      context, source_op, plan->result, &plan->zero_descriptor,
      plan->imm32_attr_name_id, bit_patterns, IREE_ARRAYSIZE(bit_patterns));
}

iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_U32_BITS:
      break;
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_SCC:
      return loom_amdgpu_lower_i1_scc_constant(context, source_op, plan);
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK:
      return loom_amdgpu_lower_i1_mask_constant(context, source_op, plan);
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_NONE:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU constant plan kind");
      return iree_ok_status();
  }
  if (plan->register_count == 1) {
    return loom_amdgpu_lower_u32_constant(context, source_op, &plan->descriptor,
                                          plan->imm32_attr_name_id,
                                          plan->bit_patterns[0], plan->result);
  }
  return loom_amdgpu_bind_register_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, plan->bit_patterns, plan->register_count);
}
