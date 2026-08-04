// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"

#include "iree/base/bitfield.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/abi/asan.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_shadow_config_values_t {
  // Address of the runtime-published ASAN config global.
  loom_value_id_t address;
  // Device-visible shadow base loaded from loom_amdgpu_asan_config_layout_e.
  loom_value_id_t shadow_base;
} loom_amdgpu_sanitizer_shadow_config_values_t;

typedef struct loom_amdgpu_sanitizer_access_packet_check_t {
  // Native lane mask identifying lanes that failed this packet.
  loom_value_id_t failure_mask;
  // Shadow address selected for reporting this packet.
  loom_value_id_t shadow_address;
  // Loaded shadow byte selected for reporting this packet.
  loom_value_id_t shadow_value;
} loom_amdgpu_sanitizer_access_packet_check_t;

// Bitfield controlling which values the access-check builder materializes.
typedef uint32_t loom_amdgpu_sanitizer_access_build_flags_t;

enum loom_amdgpu_sanitizer_access_build_flag_bits_e {
  // Only builds the lane failure mask required by the hot assertion predicate.
  LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_NONE = 0u,
  // Also preserves the selected shadow address/value used in feedback reports.
  LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_CAPTURE_REPORT_VALUES = 1u << 0,
};

static void loom_amdgpu_sanitizer_access_require_register(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count) {
  IREE_ASSERT_LT(value, builder->module->values.count);
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_low_type_is_register(type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(type),
                 descriptor_set->stable_id);
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(type), unit_count);
  const uint16_t register_class = loom_low_register_type_class_id(type);
  IREE_ASSERT(register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
              register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR);
}

static void loom_amdgpu_sanitizer_access_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, uint16_t register_class) {
  loom_amdgpu_sanitizer_access_require_register(builder, descriptor_set, value,
                                                unit_count);
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT_EQ(loom_low_register_type_class_id(type), register_class);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){0};
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, name, &out_attr->name_id));
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

static const loom_low_asm_form_t*
loom_amdgpu_sanitizer_access_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_ASSERT_UNREACHABLE(
        "validated AMDGPU sanitizer access descriptor asm form");
    IREE_BUILTIN_UNREACHABLE();
  }
  return &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
}

static iree_status_t loom_amdgpu_sanitizer_access_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count, attrs,
      result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_sanitizer_access_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_sanitizer_access_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_sanitizer_access_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_sgpr_u32_binary(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, lhs, 1, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, rhs, 1, LOOM_AMDGPU_REG_CLASS_ID_SGPR);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, descriptor_ref, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_register_lane(
    loom_builder_t* builder, loom_value_id_t source, uint32_t lane_index,
    loom_type_t lane_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source, lane_index,
                                            lane_type, location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr_u32_binary(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, lhs, /*expected_unit_count=*/1, location,
      &vgpr_lhs));
  loom_value_id_t vgpr_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, rhs, /*expected_unit_count=*/1, location,
      &vgpr_rhs));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  const loom_value_id_t operands[] = {vgpr_lhs, vgpr_rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, descriptor_ref, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      /*result_count=*/1, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_location_id_t location,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, value, /*expected_unit_count=*/1, location,
      &vgpr_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_u32_attr(
      builder, IREE_SV("imm32"), immediate, &imm32_attr));
  const loom_value_id_t operands[] = {vgpr_value};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, descriptor_ref, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&imm32_attr, 1),
      &vgpr_type, /*result_count=*/1, location, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr64_add(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_sum) {
  *out_sum = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register(builder, descriptor_set, lhs,
                                                2);
  loom_amdgpu_sanitizer_access_require_register(builder, descriptor_set, rhs,
                                                2);
  loom_value_id_t vgpr_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, lhs, /*expected_unit_count=*/2, location,
      &vgpr_lhs));
  loom_value_id_t vgpr_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, rhs, /*expected_unit_count=*/2, location,
      &vgpr_rhs));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_lhs, /*lane_index=*/0, vgpr_type, location, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_lhs, /*lane_index=*/1, vgpr_type, location, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_rhs, /*lane_index=*/0, vgpr_type, location, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_rhs, /*lane_index=*/1, vgpr_type, location, &rhs_hi));

  const loom_value_id_t add_lo_operands[] = {lhs_lo, rhs_lo};
  const loom_type_t add_result_types[] = {vgpr_type, sgpr_x2_type};
  loom_op_t* add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      add_lo_operands, IREE_ARRAYSIZE(add_lo_operands),
      loom_make_named_attr_slice(NULL, 0), add_result_types,
      IREE_ARRAYSIZE(add_result_types), location, &add_lo_op));
  const loom_value_id_t sum_lo =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 0);
  const loom_value_id_t carry =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 1);

  const loom_value_id_t add_hi_operands[] = {lhs_hi, rhs_hi, carry};
  loom_op_t* add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      add_hi_operands, IREE_ARRAYSIZE(add_hi_operands),
      loom_make_named_attr_slice(NULL, 0), add_result_types,
      IREE_ARRAYSIZE(add_result_types), location, &add_hi_op));
  const loom_value_id_t sum_hi =
      loom_value_slice_get(loom_low_op_results(add_hi_op), 0);

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {sum_lo, sum_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_sum = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr_u32_select(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, condition_mask, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  loom_value_id_t vgpr_false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, false_value, /*expected_unit_count=*/1, location,
      &vgpr_false_value));
  loom_value_id_t vgpr_true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, true_value, /*expected_unit_count=*/1, location,
      &vgpr_true_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  const loom_value_id_t operands[] = {vgpr_false_value, vgpr_true_value,
                                      condition_mask};
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      &vgpr_type, /*result_count=*/1, location, &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_vgpr64_select(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, false_value, /*expected_unit_count=*/2, location,
      &vgpr_false_value));
  loom_value_id_t vgpr_true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, true_value, /*expected_unit_count=*/2, location,
      &vgpr_true_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_value_id_t false_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_false_value, /*lane_index=*/0, vgpr_type, location,
      &false_lo));
  loom_value_id_t false_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_false_value, /*lane_index=*/1, vgpr_type, location,
      &false_hi));
  loom_value_id_t true_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_true_value, /*lane_index=*/0, vgpr_type, location,
      &true_lo));
  loom_value_id_t true_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_true_value, /*lane_index=*/1, vgpr_type, location,
      &true_hi));

  loom_value_id_t selected_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_select(
      builder, descriptor_set, false_lo, true_lo, condition_mask, location,
      &selected_lo));
  loom_value_id_t selected_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_select(
      builder, descriptor_set, false_hi, true_hi, condition_mask, location,
      &selected_hi));
  const loom_value_id_t parts[] = {selected_lo, selected_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_vgpr64_select(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_sanitizer_access_build_vgpr64_select(
      builder, descriptor_set, false_value, true_value, condition_mask,
      location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_shadow_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_sanitizer_shadow_config_values_t* out_values) {
  *out_values = (loom_amdgpu_sanitizer_shadow_config_values_t){0};

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_sanitizer_shadow_config_values_t values = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY);
  loom_named_attr_t offset_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, LOOM_AMDGPU_ASAN_CONFIG_SHADOW_BASE_OFFSET, &offset_attr));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, &values.address,
      /*operand_count=*/1, loom_make_named_attr_slice(&offset_attr, 1),
      &sgpr_x2_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  values.shadow_base = loom_value_slice_get(loom_low_op_results(load_op), 0);

  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(out_config);
  *out_config = (loom_amdgpu_sanitizer_access_config_t){0};

  loom_amdgpu_sanitizer_shadow_config_values_t config_values = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_config_values(
      builder, descriptor_set, asan_config_symbol, location, &config_values));

  loom_amdgpu_sanitizer_access_config_t config = {
      .shadow_base = config_values.shadow_base,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &config.zero));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
      builder, descriptor_set, 128, location, &config.poison_min));
  *out_config = config;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_shadow_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t application_address, loom_value_id_t shadow_base,
    loom_location_id_t location, loom_value_id_t* out_shadow_address) {
  *out_shadow_address = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, application_address,
      /*expected_unit_count=*/2, location, &vgpr_address));
  loom_value_id_t vgpr_shadow_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, shadow_base, /*expected_unit_count=*/2, location,
      &vgpr_shadow_base));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));

  loom_value_id_t address_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_address, /*lane_index=*/0, vgpr_type, location,
      &address_lo));
  loom_value_id_t address_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, vgpr_address, /*lane_index=*/1, vgpr_type, location,
      &address_hi));

  loom_value_id_t shifted_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      address_lo, LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT, location, &shifted_lo));
  loom_value_id_t high_carry_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      address_hi, 32u - LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT, location,
      &high_carry_bits));
  loom_value_id_t scaled_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, shifted_lo,
      high_carry_bits, location, &scaled_lo));
  loom_value_id_t scaled_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      address_hi, LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT, location, &scaled_hi));

  const loom_value_id_t scaled_parts[] = {scaled_lo, scaled_hi};
  loom_op_t* scaled_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, scaled_parts, IREE_ARRAYSIZE(scaled_parts),
                            vgpr_x2_type, location, &scaled_op));
  return loom_amdgpu_sanitizer_access_build_vgpr64_add(
      builder, descriptor_set, vgpr_shadow_base,
      loom_low_concat_result(scaled_op), location, out_shadow_address);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_shadow_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t shadow_address, uint32_t byte_length,
    loom_location_id_t location, loom_value_id_t* out_shadow_value) {
  *out_shadow_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, shadow_address, 2,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR);

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  uint32_t result_unit_count = 1;
  switch (byte_length) {
    case 1:
      descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8;
      break;
    case 2:
      descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U16;
      break;
    case 4:
      descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_B32;
      break;
    case 8:
      descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_B64;
      result_unit_count = 2;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU ASAN shadow load width");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, result_unit_count,
      &result_type));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  const loom_low_asm_form_t* asm_form =
      loom_amdgpu_sanitizer_access_canonical_asm_form(descriptor_set,
                                                      descriptor);
  IREE_ASSERT(asm_form->operand_index_count == 1 ||
              asm_form->operand_index_count == 2);
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);
  loom_value_id_t operands[2] = {shadow_address, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  if (asm_form->operand_index_count == 2) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &load_op));
  *out_shadow_value = loom_value_slice_get(loom_low_op_results(load_op), 0);

  return loom_amdgpu_system_memory_build_load_wait(builder, descriptor_set,
                                                   location);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_shadow_failure_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t shadow_value, loom_value_id_t last_byte,
    loom_value_id_t zero, loom_value_id_t poison_min,
    loom_location_id_t location, loom_value_id_t* out_failure_mask) {
  *out_failure_mask = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));

  const loom_value_id_t nonzero_operands[] = {shadow_value, zero};
  loom_op_t* nonzero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      nonzero_operands, IREE_ARRAYSIZE(nonzero_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &nonzero_op));
  const loom_value_id_t shadow_nonzero_mask =
      loom_value_slice_get(loom_low_op_results(nonzero_op), 0);

  const loom_value_id_t full_poison_operands[] = {shadow_value, poison_min};
  loom_op_t* full_poison_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
      full_poison_operands, IREE_ARRAYSIZE(full_poison_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &full_poison_op));
  const loom_value_id_t full_poison_mask =
      loom_value_slice_get(loom_low_op_results(full_poison_op), 0);

  const loom_value_id_t partial_oob_operands[] = {last_byte, shadow_value};
  loom_op_t* partial_oob_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
      partial_oob_operands, IREE_ARRAYSIZE(partial_oob_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &partial_oob_op));
  const loom_value_id_t partial_oob_mask =
      loom_value_slice_get(loom_low_op_results(partial_oob_op), 0);

  const loom_value_id_t partial_operands[] = {shadow_nonzero_mask,
                                              partial_oob_mask};
  loom_op_t* partial_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      partial_operands, IREE_ARRAYSIZE(partial_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &partial_op));
  const loom_value_id_t partial_failure_mask =
      loom_value_slice_get(loom_low_op_results(partial_op), 0);

  const loom_value_id_t failure_operands[] = {full_poison_mask,
                                              partial_failure_mask};
  loom_op_t* failure_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      failure_operands, IREE_ARRAYSIZE(failure_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &failure_op));
  *out_failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_access_build_shadow_chunk_failure_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t shadow_value, uint32_t shadow_byte_length,
    loom_value_id_t zero, loom_location_id_t location,
    loom_value_id_t* out_failure_mask) {
  *out_failure_mask = LOOM_VALUE_ID_INVALID;

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  if (shadow_byte_length <= 4) {
    const loom_value_id_t operands[] = {shadow_value, zero};
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &mask_type, /*result_count=*/1, location, &op));
    *out_failure_mask = loom_value_slice_get(loom_low_op_results(op), 0);
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(shadow_byte_length, 8u);
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_value_id_t shadow_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, shadow_value, /*lane_index=*/0, vgpr_type, location,
      &shadow_lo));
  loom_value_id_t shadow_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, shadow_value, /*lane_index=*/1, vgpr_type, location,
      &shadow_hi));

  const loom_value_id_t lo_operands[] = {shadow_lo, zero};
  loom_op_t* lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      lo_operands, IREE_ARRAYSIZE(lo_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &lo_op));
  const loom_value_id_t lo_mask =
      loom_value_slice_get(loom_low_op_results(lo_op), 0);

  const loom_value_id_t hi_operands[] = {shadow_hi, zero};
  loom_op_t* hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      hi_operands, IREE_ARRAYSIZE(hi_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &hi_op));
  const loom_value_id_t hi_mask =
      loom_value_slice_get(loom_low_op_results(hi_op), 0);

  const loom_value_id_t failure_operands[] = {lo_mask, hi_mask};
  loom_op_t* failure_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      failure_operands, IREE_ARRAYSIZE(failure_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
      location, &failure_op));
  *out_failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_shadow_value_report(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t shadow_value, loom_location_id_t location,
    loom_value_id_t* out_shadow_value_report) {
  *out_shadow_value_report = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {shadow_value, zero};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_shadow_value_report = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_access_build_packet_check(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address_vgpr, uint32_t packet_size,
    uint32_t minimum_alignment,
    loom_amdgpu_sanitizer_access_build_flags_t flags,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_packet_check_t* out_check) {
  *out_check = (loom_amdgpu_sanitizer_access_packet_check_t){0};
  IREE_ASSERT_GT(packet_size, 0u);
  IREE_ASSERT_LE(packet_size, 8u);
  const bool capture_report_values = iree_any_bit_set(
      flags, LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_CAPTURE_REPORT_VALUES);

  loom_value_id_t first_shadow_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_address(
      builder, descriptor_set, fault_address_vgpr, config->shadow_base,
      location, &first_shadow_address));
  loom_value_id_t first_shadow_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_load(
      builder, descriptor_set, first_shadow_address, /*byte_length=*/1,
      location, &first_shadow_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_value_id_t fault_address_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
      builder, fault_address_vgpr, /*lane_index=*/0, vgpr_type, location,
      &fault_address_lo));
  loom_value_id_t first_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      fault_address_lo, 7, location, &first_byte_offset));
  loom_value_id_t first_last_byte = first_byte_offset;
  if (packet_size > 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
        first_byte_offset, packet_size - 1, location, &first_last_byte));
  }
  loom_value_id_t first_failure_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_failure_mask(
      builder, descriptor_set, first_shadow_value, first_last_byte,
      config->zero, config->poison_min, location, &first_failure_mask));

  loom_value_id_t failure_mask = first_failure_mask;
  loom_value_id_t report_shadow_address = LOOM_VALUE_ID_INVALID;
  loom_value_id_t report_shadow_value = LOOM_VALUE_ID_INVALID;
  if (capture_report_values) {
    report_shadow_address = first_shadow_address;
    report_shadow_value = first_shadow_value;
  }
  const bool packet_spans_one_shadow_byte = minimum_alignment >= 8;
  if (packet_size > 1 && !packet_spans_one_shadow_byte) {
    loom_value_id_t last_address_delta = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u64_const(
        builder, descriptor_set, packet_size - 1, location,
        &last_address_delta));
    loom_value_id_t last_address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_add(
        builder, descriptor_set, fault_address_vgpr, last_address_delta,
        location, &last_address));
    loom_value_id_t last_shadow_address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_address(
        builder, descriptor_set, last_address, config->shadow_base, location,
        &last_shadow_address));
    loom_value_id_t last_shadow_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_load(
        builder, descriptor_set, last_shadow_address, /*byte_length=*/1,
        location, &last_shadow_value));

    loom_value_id_t last_address_lo = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_register_lane(
        builder, last_address, /*lane_index=*/0, vgpr_type, location,
        &last_address_lo));
    loom_value_id_t last_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_immediate(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        last_address_lo, 7, location, &last_byte_offset));
    loom_value_id_t last_failure_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_failure_mask(
        builder, descriptor_set, last_shadow_value, last_byte_offset,
        config->zero, config->poison_min, location, &last_failure_mask));

    if (capture_report_values) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_select(
          builder, descriptor_set, first_shadow_address, last_shadow_address,
          last_failure_mask, location, &report_shadow_address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_select(
          builder, descriptor_set, first_shadow_value, last_shadow_value,
          last_failure_mask, location, &report_shadow_value));
    }

    loom_type_t mask_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_build_register_type(
        descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
    const loom_value_id_t failure_operands[] = {first_failure_mask,
                                                last_failure_mask};
    loom_op_t* failure_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        failure_operands, IREE_ARRAYSIZE(failure_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
        location, &failure_op));
    failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  }

  *out_check = (loom_amdgpu_sanitizer_access_packet_check_t){
      .failure_mask = failure_mask,
      .shadow_address = report_shadow_address,
      .shadow_value = report_shadow_value,
  };
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_access_build_full_granule_failure_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment, uint32_t wavefront_size,
    loom_location_id_t location, loom_value_id_t* out_failure_mask) {
  *out_failure_mask = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register(builder, descriptor_set,
                                                fault_address, 2);

  loom_value_id_t fault_address_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, fault_address, /*expected_unit_count=*/2,
      location, &fault_address_vgpr));

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));

  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  uint32_t remaining_shadow_bytes =
      access_size >> LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT;
  for (uint32_t chunk_offset = 0; remaining_shadow_bytes != 0;) {
    uint32_t shadow_byte_length = 1;
    if (remaining_shadow_bytes >= 8) {
      shadow_byte_length = 8;
    } else if (remaining_shadow_bytes >= 4) {
      shadow_byte_length = 4;
    } else if (remaining_shadow_bytes >= 2) {
      shadow_byte_length = 2;
    }
    loom_value_id_t chunk_address = fault_address_vgpr;
    if (chunk_offset != 0) {
      loom_value_id_t chunk_delta = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u64_const(
          builder, descriptor_set, chunk_offset, location, &chunk_delta));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_add(
          builder, descriptor_set, fault_address_vgpr, chunk_delta, location,
          &chunk_address));
    }

    loom_value_id_t shadow_address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_address(
        builder, descriptor_set, chunk_address, config->shadow_base, location,
        &shadow_address));
    loom_value_id_t shadow_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_load(
        builder, descriptor_set, shadow_address, shadow_byte_length, location,
        &shadow_value));
    loom_value_id_t chunk_failure_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_access_build_shadow_chunk_failure_mask(
            builder, descriptor_set, shadow_value, shadow_byte_length,
            config->zero, location, &chunk_failure_mask));
    chunk_offset += shadow_byte_length << LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT;
    remaining_shadow_bytes -= shadow_byte_length;

    if (failure_mask == LOOM_VALUE_ID_INVALID) {
      failure_mask = chunk_failure_mask;
      continue;
    }
    const loom_value_id_t failure_operands[] = {failure_mask,
                                                chunk_failure_mask};
    loom_op_t* failure_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        failure_operands, IREE_ARRAYSIZE(failure_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
        location, &failure_op));
    failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  }

  // Unknown address alignment can make the access reach one final partial
  // shadow granule beyond the full shadow chunks loaded above. Check that final
  // byte precisely so row-strided fragments do not report on untouched padding.
  if (minimum_alignment < 8) {
    loom_value_id_t last_address_delta = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u64_const(
        builder, descriptor_set, access_size - 1, location,
        &last_address_delta));
    loom_value_id_t last_address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_add(
        builder, descriptor_set, fault_address_vgpr, last_address_delta,
        location, &last_address));
    loom_amdgpu_sanitizer_access_packet_check_t last_check = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_packet_check(
        builder, descriptor_set, config, last_address, /*packet_size=*/1,
        minimum_alignment, LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_NONE,
        location, &last_check));

    const loom_value_id_t failure_operands[] = {failure_mask,
                                                last_check.failure_mask};
    loom_op_t* failure_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        failure_operands, IREE_ARRAYSIZE(failure_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
        location, &failure_op));
    failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  }

  return loom_amdgpu_build_feedback_canonical_exec_mask(
      builder, descriptor_set, failure_mask, wavefront_size, location,
      out_failure_mask);
}

static iree_status_t loom_amdgpu_sanitizer_access_build_check(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment,
    loom_amdgpu_sanitizer_access_build_flags_t flags, uint32_t wavefront_size,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check) {
  IREE_ASSERT_ARGUMENT(out_check);
  *out_check = (loom_amdgpu_sanitizer_access_check_t){0};
  const loom_amdgpu_sanitizer_access_build_flags_t valid_flags =
      LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_CAPTURE_REPORT_VALUES;
  IREE_ASSERT_EQ(flags & ~valid_flags, 0u);
  loom_amdgpu_sanitizer_access_require_register(builder, descriptor_set,
                                                fault_address, 2);
  IREE_ASSERT_GT(access_size, 0u);

  loom_value_id_t fault_address_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, fault_address, /*expected_unit_count=*/2,
      location, &fault_address_vgpr));

  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t report_shadow_address = LOOM_VALUE_ID_INVALID;
  loom_value_id_t report_shadow_value = LOOM_VALUE_ID_INVALID;
  const bool capture_report_values = iree_any_bit_set(
      flags, LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_CAPTURE_REPORT_VALUES);
  for (uint64_t packet_offset = 0; packet_offset < access_size;
       packet_offset += 8) {
    const uint64_t remaining_size = access_size - packet_offset;
    const uint32_t packet_size =
        remaining_size > 8 ? 8 : (uint32_t)remaining_size;
    loom_value_id_t packet_address = fault_address_vgpr;
    if (packet_offset != 0) {
      loom_value_id_t packet_delta = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u64_const(
          builder, descriptor_set, packet_offset, location, &packet_delta));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_add(
          builder, descriptor_set, fault_address_vgpr, packet_delta, location,
          &packet_address));
    }

    loom_amdgpu_sanitizer_access_packet_check_t packet_check = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_packet_check(
        builder, descriptor_set, config, packet_address, packet_size,
        minimum_alignment, flags, location, &packet_check));
    if (packet_offset == 0) {
      failure_mask = packet_check.failure_mask;
      if (capture_report_values) {
        report_shadow_address = packet_check.shadow_address;
        report_shadow_value = packet_check.shadow_value;
      }
      continue;
    }

    if (capture_report_values) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr64_select(
          builder, descriptor_set, report_shadow_address,
          packet_check.shadow_address, packet_check.failure_mask, location,
          &report_shadow_address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_vgpr_u32_select(
          builder, descriptor_set, report_shadow_value,
          packet_check.shadow_value, packet_check.failure_mask, location,
          &report_shadow_value));
    }

    loom_type_t mask_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_build_register_type(
        descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
    const loom_value_id_t failure_operands[] = {failure_mask,
                                                packet_check.failure_mask};
    loom_op_t* failure_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
        builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        failure_operands, IREE_ARRAYSIZE(failure_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_type, /*result_count=*/1,
        location, &failure_op));
    failure_mask = loom_value_slice_get(loom_low_op_results(failure_op), 0);
  }

  loom_amdgpu_sanitizer_access_check_t check = {
      .shadow_address = report_shadow_address,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_canonical_exec_mask(
      builder, descriptor_set, failure_mask, wavefront_size, location,
      &check.failure_mask));
  if (capture_report_values) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_shadow_value_report(
        builder, descriptor_set, report_shadow_value, location,
        &check.shadow_value));
  }
  *out_check = check;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_value_id_t fault_address,
    uint32_t access_size, uint32_t wavefront_size, loom_location_id_t location,
    loom_value_id_t* out_failure_mask) {
  IREE_ASSERT_ARGUMENT(out_failure_mask);
  *out_failure_mask = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_config_t config = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_config(
      builder, descriptor_set, asan_config_symbol, location, &config));
  return loom_amdgpu_build_sanitizer_access_failure_mask_with_config(
      builder, descriptor_set, &config, fault_address, access_size,
      /*minimum_alignment=*/1, wavefront_size, location, out_failure_mask);
}

iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask_with_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment, uint32_t wavefront_size,
    loom_location_id_t location, loom_value_id_t* out_failure_mask) {
  IREE_ASSERT_ARGUMENT(out_failure_mask);
  *out_failure_mask = LOOM_VALUE_ID_INVALID;
  if (access_size != 0 && access_size % 8 == 0) {
    return loom_amdgpu_sanitizer_access_build_full_granule_failure_mask(
        builder, descriptor_set, config, fault_address, access_size,
        minimum_alignment, wavefront_size, location, out_failure_mask);
  }
  loom_amdgpu_sanitizer_access_check_t check = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_check(
      builder, descriptor_set, config, fault_address, access_size,
      minimum_alignment, LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_NONE,
      wavefront_size, location, &check));
  *out_failure_mask = check.failure_mask;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_failure_mask_union(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs_failure_mask, loom_value_id_t rhs_failure_mask,
    loom_location_id_t location, loom_value_id_t* out_failure_mask) {
  IREE_ASSERT_ARGUMENT(out_failure_mask);
  *out_failure_mask = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, lhs_failure_mask, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  loom_amdgpu_sanitizer_access_require_register_class(
      builder, descriptor_set, rhs_failure_mask, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR);

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  const loom_value_id_t operands[] = {lhs_failure_mask, rhs_failure_mask};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_access_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &mask_type,
      /*result_count=*/1, location, &op));
  *out_failure_mask = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_check(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_value_id_t fault_address,
    uint32_t access_size, uint32_t wavefront_size, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check) {
  loom_amdgpu_sanitizer_access_config_t config = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_config(
      builder, descriptor_set, asan_config_symbol, location, &config));
  return loom_amdgpu_build_sanitizer_access_check_with_config(
      builder, descriptor_set, &config, fault_address, access_size,
      /*minimum_alignment=*/1, wavefront_size, location, out_check);
}

iree_status_t loom_amdgpu_build_sanitizer_access_check_with_config(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_config_t* config,
    loom_value_id_t fault_address, uint32_t access_size,
    uint32_t minimum_alignment, uint32_t wavefront_size,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check) {
  return loom_amdgpu_sanitizer_access_build_check(
      builder, descriptor_set, config, fault_address, access_size,
      minimum_alignment,
      LOOM_AMDGPU_SANITIZER_ACCESS_BUILD_FLAG_CAPTURE_REPORT_VALUES,
      wavefront_size, location, out_check);
}
