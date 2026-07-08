// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/constants.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/float_facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/util/math.h"

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
           loom_checked_mul_i64((int64_t)lane, step, &delta) &&
           loom_checked_add_i64(base, delta, &value) &&
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
