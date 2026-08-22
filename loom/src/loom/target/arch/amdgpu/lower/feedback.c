// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/signal.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static loom_amdgpu_feedback_config_values_t
loom_amdgpu_feedback_config_values_empty(void) {
  return (loom_amdgpu_feedback_config_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .channel_base = LOOM_VALUE_ID_INVALID,
      .notify_signal = LOOM_VALUE_ID_INVALID,
      .source_context = LOOM_VALUE_ID_INVALID,
      .ring_base = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_channel_header_values_t
loom_amdgpu_feedback_channel_header_values_empty(void) {
  return (loom_amdgpu_feedback_channel_header_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .record_length = LOOM_VALUE_ID_INVALID,
      .abi_version = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .ring_capacity = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_packet_address_t
loom_amdgpu_feedback_packet_address_empty(void) {
  return (loom_amdgpu_feedback_packet_address_t){
      .base = LOOM_VALUE_ID_INVALID,
      .byte_offset = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_reservation_attempt_t
loom_amdgpu_feedback_reservation_attempt_empty(void) {
  return (loom_amdgpu_feedback_reservation_attempt_t){
      .expected_head = LOOM_VALUE_ID_INVALID,
      .next_head = LOOM_VALUE_ID_INVALID,
      .observed_head = LOOM_VALUE_ID_INVALID,
      .cas_succeeded = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_reservation_t
loom_amdgpu_feedback_reservation_empty(void) {
  return (loom_amdgpu_feedback_reservation_t){
      .packet_address = loom_amdgpu_feedback_packet_address_empty(),
      .sequence = LOOM_VALUE_ID_INVALID,
      .reserved_mask = LOOM_VALUE_ID_INVALID,
  };
}

typedef uint32_t loom_amdgpu_feedback_global_load_flags_t;

enum loom_amdgpu_feedback_global_load_flag_bits_e {
  LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_NONE = 0u,
  LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE = 1u << 0,
};

static bool loom_amdgpu_feedback_packet_record_length_is_valid(
    uint32_t record_length) {
  return record_length >= LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH &&
         (record_length & (LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) == 0 &&
         record_length <= loom_amdgpu_feedback_packet_length(
                              LOOM_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH);
}

static iree_status_t loom_amdgpu_feedback_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_feedback_build_u32_attr(builder, IREE_SV("offset"),
                                             byte_offset, out_attr);
}

static iree_host_size_t loom_amdgpu_feedback_packet_operand_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  iree_host_size_t count = 0;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++count;
    }
  }
  return count;
}

static bool loom_amdgpu_feedback_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static void loom_amdgpu_feedback_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  IREE_ASSERT(value < builder->module->values.count,
              "AMDGPU feedback builder received an invalid low value");
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_amdgpu_feedback_type_is_register_class(descriptor_set, type,
                                                          reg_class_id) &&
                  loom_low_register_type_unit_count(type) == unit_count,
              "AMDGPU feedback builder received a low value with an "
              "unsupported register shape");
}

static iree_status_t loom_amdgpu_feedback_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(NULL, 0), result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_feedback_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_feedback_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_feedback_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
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

static iree_status_t loom_amdgpu_feedback_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_feedback_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_sgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_register_lane(
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

iree_status_t loom_amdgpu_build_feedback_canonical_exec_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t mask, uint32_t wavefront_size, loom_location_id_t location,
    loom_value_id_t* out_mask) {
  IREE_ASSERT_ARGUMENT(out_mask);
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, mask,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (wavefront_size == 64) {
    *out_mask = mask;
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(wavefront_size, 32);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, mask, /*lane_index=*/0, sgpr_type, location, &low_mask));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, 0, location, &zero));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {low_mask, zero};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_mask = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr64_add(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_sum) {
  *out_sum = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, lhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, rhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/0, vgpr_type, location, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/1, vgpr_type, location, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/0, vgpr_type, location, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/1, vgpr_type, location, &rhs_hi));

  const loom_value_id_t add_lo_operands[] = {lhs_lo, rhs_lo};
  const loom_type_t add_result_types[] = {vgpr_type, sgpr_x2_type};
  loom_op_t* add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      add_lo_operands, IREE_ARRAYSIZE(add_lo_operands), add_result_types,
      IREE_ARRAYSIZE(add_result_types), location, &add_lo_op));
  const loom_value_id_t sum_lo =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 0);
  const loom_value_id_t carry =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 1);

  const loom_value_id_t add_hi_operands[] = {lhs_hi, rhs_hi, carry};
  loom_op_t* add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      add_hi_operands, IREE_ARRAYSIZE(add_hi_operands), add_result_types,
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

static iree_status_t loom_amdgpu_feedback_build_vgpr64_equal_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, lhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, rhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/0, vgpr_type, location, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/1, vgpr_type, location, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/0, vgpr_type, location, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/1, vgpr_type, location, &rhs_hi));

  const loom_value_id_t compare_hi_operands[] = {lhs_hi, rhs_hi};
  loom_op_t* compare_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_hi_operands, IREE_ARRAYSIZE(compare_hi_operands), &mask_type,
      /*result_count=*/1, location, &compare_hi_op));
  const loom_value_id_t high_mask =
      loom_value_slice_get(loom_low_op_results(compare_hi_op), 0);

  const loom_value_id_t compare_lo_operands[] = {lhs_lo, rhs_lo};
  loom_op_t* compare_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_lo_operands, IREE_ARRAYSIZE(compare_lo_operands), &mask_type,
      /*result_count=*/1, location, &compare_lo_op));
  const loom_value_id_t low_mask =
      loom_value_slice_get(loom_low_op_results(compare_lo_op), 0);

  const loom_value_id_t and_operands[] = {high_mask, low_mask};
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      and_operands, IREE_ARRAYSIZE(and_operands), &mask_type,
      /*result_count=*/1, location, &and_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(and_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, loom_value_id_t zero64, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, value,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, zero64,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_value_id_t operands[] = {value, zero64};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      operands, IREE_ARRAYSIZE(operands), &scc_type, /*result_count=*/1,
      location, &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_sgpr32_nonzero_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, loom_value_id_t zero32, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, value,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, zero32,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_value_id_t operands[] = {value, zero32};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32,
      operands, IREE_ARRAYSIZE(operands), &scc_type, /*result_count=*/1,
      location, &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_exec_narrow(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  if (out_saved_exec != NULL) {
    *out_saved_exec = LOOM_VALUE_ID_INVALID;
  }
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, lane_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {sgpr_x2_type, scc_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &lane_mask, /*operand_count=*/1, result_types,
      IREE_ARRAYSIZE(result_types), location, &saveexec_op));
  if (out_saved_exec != NULL) {
    *out_saved_exec = loom_value_slice_get(loom_low_op_results(saveexec_op), 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_exec_restore(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t saved_exec, loom_location_id_t location) {
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, saved_exec, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_op_t* restore_op = NULL;
  return loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, /*operand_count=*/1, /*result_types=*/NULL,
      /*result_count=*/0, location, &restore_op);
}

static iree_status_t loom_amdgpu_feedback_build_cmpswap_pair(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t expected_value, loom_value_id_t desired_value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4, &vgpr_x4_type));
  // AMDGPU compare-and-swap data operands match LLVM's lowering: the desired
  // value occupies the low lanes and the expected value occupies the high
  // lanes.
  const loom_value_id_t parts[] = {desired_value, expected_value};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x4_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_b32_copy(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY);
  loom_value_id_t operands[] = {source};
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &copy_op));
  *out_value = loom_value_slice_get(loom_low_op_results(copy_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_materialize_vgpr_registers(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, uint32_t expected_unit_count,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = source;
  IREE_ASSERT(source < builder->module->values.count,
              "AMDGPU feedback builder received an invalid low value");
  const loom_type_t source_type =
      loom_module_value_type(builder->module, source);
  IREE_ASSERT(
      loom_low_type_is_register(source_type) &&
          loom_low_register_type_unit_count(source_type) == expected_unit_count,
      "AMDGPU feedback builder cannot materialize value with "
      "unsupported register shape");
  if (loom_amdgpu_feedback_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_amdgpu_feedback_type_is_register_class(
                  descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR),
              "AMDGPU feedback builder cannot materialize non-SGPR value into "
              "VGPR");
  IREE_ASSERT_LE(expected_unit_count, 2);
  if (expected_unit_count == 1) {
    return loom_amdgpu_feedback_build_vgpr_b32_copy(
        builder, descriptor_set, source, location, out_value);
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t lanes[2];
  for (uint32_t i = 0; i < expected_unit_count; ++i) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source, i, sgpr_type,
                                              location, &slice_op));
    const loom_value_id_t sgpr_lane = loom_low_slice_result(slice_op);
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_b32_copy(
        builder, descriptor_set, sgpr_lane, location, &lanes[i]));
  }

  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, expected_unit_count,
      &vgpr_range_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, lanes, expected_unit_count,
                            vgpr_range_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_vgpr_registers(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, uint32_t expected_unit_count,
    loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  return loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, source, expected_unit_count, location,
      out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_binary(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, lhs, /*expected_unit_count=*/1, location,
      &vgpr_lhs));
  loom_value_id_t vgpr_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, rhs, /*expected_unit_count=*/1, location,
      &vgpr_rhs));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  const loom_value_id_t operands[] = {vgpr_lhs, vgpr_rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, descriptor_ref, operands,
      IREE_ARRAYSIZE(operands), &vgpr_type, /*result_count=*/1, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_sub(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_feedback_build_vgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, lhs, rhs,
      location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_and(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_feedback_build_vgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32, lhs, rhs,
      location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_sgpr_u32_sub(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, lhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set, rhs,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32, operands,
      IREE_ARRAYSIZE(operands), &sgpr_type, /*result_count=*/1, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_ule_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vgpr_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, lhs, /*expected_unit_count=*/1, location,
      &vgpr_lhs));
  loom_value_id_t vgpr_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, rhs, /*expected_unit_count=*/1, location,
      &vgpr_rhs));

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));
  const loom_value_id_t operands[] = {vgpr_lhs, vgpr_rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULE_U32,
      operands, IREE_ARRAYSIZE(operands), &mask_type, /*result_count=*/1,
      location, &op));
  *out_mask = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_capacity_available_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t ring_capacity, loom_value_id_t reservation_head,
    loom_value_id_t read_tail, uint32_t packet_length,
    loom_location_id_t location, loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, ring_capacity, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              reservation_head,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, read_tail, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));

  loom_value_id_t capacity_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, ring_capacity, /*lane_index=*/0, sgpr_type, location,
      &capacity_low));
  loom_value_id_t packet_length_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, packet_length, location, &packet_length_low));
  loom_value_id_t capacity_limit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_sub(
      builder, descriptor_set, capacity_low, packet_length_low, location,
      &capacity_limit));

  loom_value_id_t head_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, reservation_head, /*lane_index=*/0, vgpr_type, location,
      &head_low));
  loom_value_id_t tail_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, read_tail, /*lane_index=*/0, vgpr_type, location, &tail_low));
  loom_value_id_t used_capacity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_sub(
      builder, descriptor_set, head_low, tail_low, location, &used_capacity));

  loom_value_id_t available_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_ule_mask(
      builder, descriptor_set, used_capacity, capacity_limit, location,
      &available_mask));
  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));
  return loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
      builder, descriptor_set, available_mask, zero64, location, out_scc);
}

static iree_status_t loom_amdgpu_feedback_build_ring_offset(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t ring_capacity, loom_value_id_t reservation_head,
    loom_location_id_t location, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, ring_capacity, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              reservation_head,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));

  loom_value_id_t capacity_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, ring_capacity, /*lane_index=*/0, sgpr_type, location,
      &capacity_low));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, 1, location, &one));
  loom_value_id_t capacity_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_sub(
      builder, descriptor_set, capacity_low, one, location, &capacity_mask));

  loom_value_id_t head_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, reservation_head, /*lane_index=*/0, vgpr_type, location,
      &head_low));
  return loom_amdgpu_feedback_build_vgpr_u32_and(
      builder, descriptor_set, head_low, capacity_mask, location, out_offset);
}

static iree_status_t loom_amdgpu_feedback_build_global_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, uint32_t value_unit_count,
    const loom_named_attr_t* extra_attrs, iree_host_size_t extra_attr_count,
    loom_location_id_t location) {
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->base,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->byte_offset,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  loom_value_id_t vgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, value, value_unit_count, location, &vgpr_value));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  IREE_ASSERT_LT(descriptor->canonical_asm_form_ordinal,
                 descriptor_set->asm_form_count);
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_feedback_packet_operand_count(descriptor_set, descriptor);
  IREE_ASSERT(packet_operand_count == 3 || packet_operand_count == 4,
              "AMDGPU feedback global store descriptor has an unsupported "
              "packet operand count");
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  IREE_ASSERT_LE(extra_attr_count, IREE_ARRAYSIZE(attrs) - attr_count);
  for (iree_host_size_t i = 0; i < extra_attr_count; ++i) {
    attrs[attr_count++] = extra_attrs[i];
  }
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);
  loom_value_id_t operands[4] = {packet_address->byte_offset, vgpr_value,
                                 packet_address->base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (packet_operand_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &store_op);
}

static iree_status_t loom_amdgpu_feedback_build_global_store_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, packet_address,
      byte_offset, value, /*value_unit_count=*/1,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

static iree_status_t loom_amdgpu_feedback_build_global_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR, packet_address,
      byte_offset, value, /*value_unit_count=*/2,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

iree_status_t loom_amdgpu_build_feedback_packet_store_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address, byte_offset, value, location);
}

iree_status_t loom_amdgpu_build_feedback_packet_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address, byte_offset, value, location);
}

iree_status_t loom_amdgpu_build_feedback_packet_store_u32_constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, uint32_t value, loom_location_id_t location) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, value, location, &value_id));
  return loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address, byte_offset, value_id, location);
}

iree_status_t loom_amdgpu_build_feedback_packet_store_u64_constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, uint64_t value, loom_location_id_t location) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, value, location, &value_id));
  return loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address, byte_offset, value_id, location);
}

static iree_status_t loom_amdgpu_feedback_build_publish_state_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_value_id_t ready_value, loom_location_id_t location) {
  loom_named_attr_t extra_attrs[2] = {0};
  iree_host_size_t extra_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_release_store_attrs(
      builder, descriptor_set, extra_attrs, IREE_ARRAYSIZE(extra_attrs),
      &extra_attr_count));
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, ready_value,
      /*value_unit_count=*/1, extra_attrs, extra_attr_count, location);
}

static const loom_low_descriptor_t*
loom_amdgpu_feedback_resolve_global_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  IREE_ASSERT_LT(descriptor->canonical_asm_form_ordinal,
                 descriptor_set->asm_form_count);
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_feedback_packet_operand_count(descriptor_set, descriptor);
  IREE_ASSERT(packet_operand_count == 2 || packet_operand_count == 3 ||
                  packet_operand_count == 4,
              "AMDGPU feedback descriptor has an unsupported packet operand "
              "count");
  return descriptor;
}

static iree_status_t loom_amdgpu_feedback_build_global_load_b64_system(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, uint32_t byte_offset,
    loom_amdgpu_feedback_global_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_feedback_resolve_global_memory_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_feedback_packet_operand_count(descriptor_set, descriptor);
  IREE_ASSERT(packet_operand_count == 2 || packet_operand_count == 3,
              "AMDGPU feedback load descriptor has an unsupported packet "
              "operand count");

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);

  loom_value_id_t operands[3] = {zero_vaddr, channel_base,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  if (packet_operand_count == 3) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
        builder, descriptor_set, location));
  }
  *out_value = value;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_feedback_config_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_config_values_empty();

  loom_amdgpu_feedback_config_values_t values =
      loom_amdgpu_feedback_config_values_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.channel_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.notify_signal));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_SOURCE_CONTEXT_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.source_context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_RING_BASE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.ring_base));

  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_config_enabled_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t config_flags, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  IREE_ASSERT_ARGUMENT(out_scc);
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, config_flags, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));

  loom_value_id_t enabled_flag = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED,
      location, &enabled_flag));
  const loom_value_id_t and_operands[] = {config_flags, enabled_flag};
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      and_operands, IREE_ARRAYSIZE(and_operands), &sgpr_type,
      /*result_count=*/1, location, &and_op));
  const loom_value_id_t masked_flags =
      loom_value_slice_get(loom_low_op_results(and_op), 0);

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u32_const(
      builder, descriptor_set, 0, location, &zero));
  return loom_amdgpu_feedback_build_sgpr32_nonzero_scc(
      builder, descriptor_set, masked_flags, zero, location, out_scc);
}

iree_status_t loom_amdgpu_build_feedback_uniform_packet_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base, loom_location_id_t location,
    loom_amdgpu_feedback_packet_address_t* out_address) {
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = loom_amdgpu_feedback_packet_address_empty();

  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_amdgpu_feedback_packet_address_t address =
      loom_amdgpu_feedback_packet_address_empty();
  address.base = packet_base;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &address.byte_offset));
  *out_address = address;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_dropped_packet_count_increment(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location) {
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t one64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, 1, location, &one64));

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  IREE_ASSERT_LT(descriptor->canonical_asm_form_ordinal,
                 descriptor_set->asm_form_count);
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_feedback_packet_operand_count(descriptor_set, descriptor);
  IREE_ASSERT(packet_operand_count == 3 || packet_operand_count == 4,
              "AMDGPU feedback atomic descriptor has an unsupported packet "
              "operand count");

  loom_value_id_t counter_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_saddr_byte_offset(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET, location,
      &counter_address));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_no_return_atomic_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);

  loom_value_id_t operands[4] = {zero_vaddr, one64, counter_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (packet_operand_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op);
}

iree_status_t loom_amdgpu_build_feedback_reservation_head_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  return loom_amdgpu_feedback_build_global_load_b64_system(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
      LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_NONE, location, out_value);
}

iree_status_t loom_amdgpu_build_feedback_read_tail_acquire_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  return loom_amdgpu_feedback_build_global_load_b64_system(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET,
      LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE, location, out_value);
}

iree_status_t
loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t expected_head,
    loom_value_id_t desired_head, loom_location_id_t location,
    loom_value_id_t* out_old_head) {
  IREE_ASSERT_ARGUMENT(out_old_head);
  *out_old_head = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t expected_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, expected_head, /*expected_unit_count=*/2,
      location, &expected_vgpr));
  loom_value_id_t desired_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, desired_head, /*expected_unit_count=*/2,
      location, &desired_vgpr));
  loom_value_id_t compare_exchange_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_cmpswap_pair(
      builder, descriptor_set, expected_vgpr, desired_vgpr, location,
      &compare_exchange_pair));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_feedback_resolve_global_memory_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_feedback_packet_operand_count(descriptor_set, descriptor);
  IREE_ASSERT(packet_operand_count == 3 || packet_operand_count == 4,
              "AMDGPU feedback compare-exchange descriptor has an unsupported "
              "packet operand count");

  loom_value_id_t reservation_head_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_saddr_byte_offset(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, location,
      &reservation_head_address));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_return_atomic_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));

  loom_value_id_t operands[4] = {zero_vaddr, compare_exchange_pair,
                                 reservation_head_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (packet_operand_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t old_head =
      loom_value_slice_get(loom_low_op_results(op), 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
      builder, descriptor_set, location));
  *out_old_head = old_head;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_reservation_attempt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t reservation_head,
    uint32_t packet_length, loom_location_id_t location,
    loom_amdgpu_feedback_reservation_attempt_t* out_attempt) {
  IREE_ASSERT_ARGUMENT(out_attempt);
  *out_attempt = loom_amdgpu_feedback_reservation_attempt_empty();
  IREE_ASSERT(loom_amdgpu_feedback_packet_record_length_is_valid(packet_length),
              "AMDGPU feedback reservation packet length violates the "
              "feedback ABI");
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              reservation_head,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  loom_value_id_t packet_length_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, packet_length, location, &packet_length_value));
  loom_value_id_t next_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr64_add(
      builder, descriptor_set, reservation_head, packet_length_value, location,
      &next_head));

  loom_value_id_t observed_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          builder, descriptor_set, channel_base, reservation_head, next_head,
          location, &observed_head));

  loom_value_id_t cas_succeeded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr64_equal_mask(
      builder, descriptor_set, observed_head, reservation_head, location,
      &cas_succeeded));

  *out_attempt = (loom_amdgpu_feedback_reservation_attempt_t){
      .expected_head = reservation_head,
      .next_head = next_head,
      .observed_head = observed_head,
      .cas_succeeded = cas_succeeded,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_failure_scc_split(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t failure_scc, loom_location_id_t location,
    loom_amdgpu_feedback_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_feedback_failure_branch_t){0};
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, failure_scc, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  IREE_ASSERT(builder->ip.before_op == NULL,
              "AMDGPU feedback failure branch must be built at the end of a "
              "low block");

  loom_block_t* hot_block = builder->ip.block;
  loom_amdgpu_feedback_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, hot_block->parent_region,
      (uint16_t)(hot_block->region_index + 1), &branch.continuation_block));
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, branch.continuation_block->parent_region,
      (uint16_t)(branch.continuation_block->region_index + 1),
      &branch.failure_block));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      builder, failure_scc, branch.failure_block, branch.continuation_block,
      location, &cond_branch_op));

  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_failure_mask_split(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t failure_mask, loom_location_id_t location,
    loom_amdgpu_feedback_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_feedback_failure_branch_t){0};
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, failure_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero_mask));
  loom_value_id_t failure_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
      builder, descriptor_set, failure_mask, zero_mask, location,
      &failure_scc));

  loom_amdgpu_feedback_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_failure_scc_split(
      builder, descriptor_set, failure_scc, location, &branch));

  loom_builder_set_block(builder, branch.failure_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_exec_narrow(
      builder, descriptor_set, failure_mask, location,
      /*out_saved_exec=*/NULL));

  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_reservation(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t ring_base,
    loom_value_id_t ring_capacity, uint32_t packet_length,
    loom_location_id_t location,
    loom_amdgpu_feedback_reservation_t* out_reservation) {
  IREE_ASSERT_ARGUMENT(out_reservation);
  *out_reservation = loom_amdgpu_feedback_reservation_empty();
  IREE_ASSERT(loom_amdgpu_feedback_packet_record_length_is_valid(packet_length),
              "AMDGPU feedback reservation packet length violates the "
              "feedback ABI");
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, ring_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, ring_capacity, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  IREE_ASSERT(builder->ip.before_op == NULL,
              "AMDGPU feedback reservation must be built at the end of a low "
              "block");

  loom_block_t* check_block = builder->ip.block;
  loom_block_t* attempt_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, check_block->parent_region,
      (uint16_t)(check_block->region_index + 1), &attempt_block));
  loom_block_t* reserved_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, attempt_block->parent_region,
      (uint16_t)(attempt_block->region_index + 1), &reserved_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, reserved_block->parent_region,
      (uint16_t)(reserved_block->region_index + 1), &continuation_block));
  loom_block_t* dropped_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, continuation_block->parent_region,
      (uint16_t)(continuation_block->region_index + 1), &dropped_block));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));

  loom_value_id_t sequence_arg = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, continuation_block, vgpr_x2_type, &sequence_arg));
  loom_value_id_t byte_offset_arg = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, continuation_block, vgpr_type, &byte_offset_arg));
  loom_value_id_t reserved_mask_arg = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, continuation_block, mask_type, &reserved_mask_arg));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_head_load(
      builder, descriptor_set, channel_base, location, &reservation_head));
  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_read_tail_acquire_load(
      builder, descriptor_set, channel_base, location, &read_tail));
  loom_value_id_t has_capacity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_capacity_available_scc(
      builder, descriptor_set, ring_capacity, reservation_head, read_tail,
      packet_length, location, &has_capacity));
  loom_op_t* check_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_capacity,
                                              attempt_block, dropped_block,
                                              location, &check_branch_op));

  loom_builder_set_block(builder, attempt_block);
  loom_amdgpu_feedback_reservation_attempt_t attempt =
      loom_amdgpu_feedback_reservation_attempt_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_attempt(
      builder, descriptor_set, channel_base, reservation_head, packet_length,
      location, &attempt));
  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero_mask));
  loom_value_id_t cas_succeeded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
      builder, descriptor_set, attempt.cas_succeeded, zero_mask, location,
      &cas_succeeded));
  loom_op_t* attempt_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, cas_succeeded,
                                              reserved_block, check_block,
                                              location, &attempt_branch_op));

  loom_builder_set_block(builder, reserved_block);
  loom_value_id_t ring_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_ring_offset(
      builder, descriptor_set, ring_capacity, attempt.expected_head, location,
      &ring_offset));
  const loom_value_id_t reserved_args[] = {
      attempt.expected_head,
      ring_offset,
      attempt.cas_succeeded,
  };
  loom_op_t* reserved_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(
      builder, continuation_block, reserved_args, IREE_ARRAYSIZE(reserved_args),
      location, &reserved_branch_op));

  loom_builder_set_block(builder, dropped_block);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_feedback_dropped_packet_count_increment(
          builder, descriptor_set, channel_base, location));
  loom_value_id_t zero_sequence = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, 0, location, &zero_sequence));
  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_offset));
  loom_value_id_t zero_reserved_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero_reserved_mask));
  const loom_value_id_t dropped_args[] = {
      zero_sequence,
      zero_offset,
      zero_reserved_mask,
  };
  loom_op_t* dropped_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(
      builder, continuation_block, dropped_args, IREE_ARRAYSIZE(dropped_args),
      location, &dropped_branch_op));

  loom_builder_set_block(builder, continuation_block);
  *out_reservation = (loom_amdgpu_feedback_reservation_t){
      .packet_address =
          {
              .base = ring_base,
              .byte_offset = byte_offset_arg,
          },
      .sequence = sequence_arg,
      .reserved_mask = reserved_mask_arg,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_reservation_succeeded_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t reservation_mask, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  IREE_ASSERT_ARGUMENT(out_scc);
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero_mask));
  return loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
      builder, descriptor_set, reservation_mask, zero_mask, location, out_scc);
}

iree_status_t loom_amdgpu_build_feedback_packet_header(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_feedback_packet_header_t* header,
    loom_location_id_t location) {
  IREE_ASSERT(
      loom_amdgpu_feedback_packet_record_length_is_valid(header->record_length),
      "AMDGPU feedback packet header record length violates the "
      "feedback ABI");
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->base,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->byte_offset,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  loom_value_id_t zero32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero32));
  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));

  loom_value_id_t record_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, header->record_length, location,
      &record_length));
  const uint32_t packed_header_kind =
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH | ((uint32_t)header->kind << 16);
  loom_value_id_t header_kind = LOOM_VALUE_ID_INVALID;
  // The 16-bit header length and 16-bit kind occupy one little-endian dword.
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, packed_header_kind, location, &header_kind));
  loom_value_id_t packet_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, header->flags, location, &packet_flags));

  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, record_length,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, header_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, packet_flags, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, zero32, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, header->sequence, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET,
      header->source_dispatch_ptr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET,
      header->source_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET,
      header->source_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_CONTEXT_OFFSET, header->source_context,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, zero64, location));
  return loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, zero64, location);
}

iree_status_t loom_amdgpu_build_feedback_publish_packet_state(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_location_id_t location) {
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->base,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              packet_address->byte_offset,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));

  loom_value_id_t ready_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY,
      location, &ready_value));
  return loom_amdgpu_feedback_build_publish_state_store(
      builder, descriptor_set, packet_address, ready_value, location);
}

iree_status_t loom_amdgpu_build_feedback_notify_host(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t notify_signal, loom_location_id_t location) {
  IREE_ASSERT(builder->ip.before_op == NULL,
              "AMDGPU feedback notification must be built at the end of a "
              "low block");
  loom_amdgpu_signal_values_t signal_values = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_values(
      builder, descriptor_set, notify_signal, location, &signal_values));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_add_one_release(
      builder, descriptor_set, signal_values.address, location));

  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));
  loom_value_id_t has_mailbox_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_sgpr64_nonzero_scc(
      builder, descriptor_set, signal_values.event_mailbox_ptr, zero64,
      location, &has_mailbox_scc));

  loom_block_t* check_block = builder->ip.block;
  loom_block_t* mailbox_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, check_block->parent_region,
      (uint16_t)(check_block->region_index + 1), &mailbox_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, mailbox_block->parent_region,
      (uint16_t)(mailbox_block->region_index + 1), &continuation_block));
  loom_op_t* check_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, has_mailbox_scc,
                                              mailbox_block, continuation_block,
                                              location, &check_branch_op));

  loom_builder_set_block(builder, mailbox_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_poke_mailbox(
      builder, descriptor_set, signal_values.event_mailbox_ptr,
      signal_values.event_id, location));
  loom_op_t* mailbox_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, continuation_block,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &mailbox_branch_op));

  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_publish_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_value_id_t notify_signal, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet_state(
      builder, descriptor_set, packet_address, location));
  return loom_amdgpu_build_feedback_notify_host(builder, descriptor_set,
                                                notify_signal, location);
}

iree_status_t loom_amdgpu_build_feedback_channel_header_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_amdgpu_feedback_channel_header_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_channel_header_values_empty();

  loom_amdgpu_feedback_channel_header_values_t values =
      loom_amdgpu_feedback_channel_header_values_empty();
  values.address = channel_base;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.record_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.abi_version));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.ring_capacity));

  *out_values = values;
  return iree_ok_status();
}

static void loom_amdgpu_feedback_validate_packet_source(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_source_t* source) {
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              source->dispatch_ptr,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_feedback_require_register_class(builder, descriptor_set,
                                              source->workgroup_id_x,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  IREE_ASSERT(source->workitem_id_x < builder->module->values.count,
              "AMDGPU feedback packet source has an invalid workitem id value");
  const loom_type_t workitem_type =
      loom_module_value_type(builder->module, source->workitem_id_x);
  IREE_ASSERT(loom_low_type_is_register(workitem_type) &&
                  loom_low_register_type_descriptor_set_stable_id(
                      workitem_type) == descriptor_set->stable_id &&
                  loom_low_register_type_unit_count(workitem_type) == 1,
              "AMDGPU feedback packet source has an unsupported workitem id "
              "shape");
  const uint16_t workitem_register_class =
      loom_low_register_type_class_id(workitem_type);
  IREE_ASSERT(workitem_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
                  workitem_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR,
              "AMDGPU feedback packet source workitem id must be an SGPR or "
              "VGPR");
}

iree_status_t loom_amdgpu_build_feedback_packet_producer_terminate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_producer_t* producer,
    loom_location_id_t location, loom_block_t** out_terminal_block) {
  IREE_ASSERT_ARGUMENT(producer);
  IREE_ASSERT_ARGUMENT(producer->source);
  if (out_terminal_block != NULL) {
    *out_terminal_block = NULL;
  }
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      producer->payload_byte_length);
  IREE_ASSERT(loom_amdgpu_feedback_packet_record_length_is_valid(packet_length),
              "AMDGPU feedback packet producer length violates the feedback "
              "ABI");
  IREE_ASSERT_NE(producer->packet_kind, LOOM_AMDGPU_FEEDBACK_PACKET_KIND_NONE);
  IREE_ASSERT(producer->build_payload != NULL,
              "AMDGPU feedback packet producer requires a payload builder");
  loom_amdgpu_feedback_validate_packet_source(builder, descriptor_set,
                                              producer->source);
  IREE_ASSERT(builder->ip.before_op == NULL,
              "AMDGPU feedback packet producer must be built at the end of a "
              "low block");

  loom_block_t* config_block = builder->ip.block;
  loom_block_t* feedback_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, config_block->parent_region,
      (uint16_t)(config_block->region_index + 1), &feedback_block));
  loom_block_t* terminal_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, feedback_block->parent_region,
      (uint16_t)(feedback_block->region_index + 1), &terminal_block));
  if (out_terminal_block != NULL) {
    *out_terminal_block = terminal_block;
  }

  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
      builder, descriptor_set, feedback_config_symbol, location,
      &config_values));
  loom_value_id_t feedback_enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_enabled_scc(
      builder, descriptor_set, config_values.flags, location,
      &feedback_enabled_scc));
  loom_op_t* enabled_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, feedback_enabled_scc,
                                              feedback_block, terminal_block,
                                              location, &enabled_branch_op));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, config_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* continuation_block = builder->ip.block;
  loom_block_t* report_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, continuation_block->parent_region,
      (uint16_t)(continuation_block->region_index + 1), &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             terminal_block, location, &reserved_branch_op));

  loom_builder_set_block(builder, report_block);
  loom_value_id_t saved_report_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_exec_narrow(
      builder, descriptor_set, reservation.reserved_mask, location,
      &saved_report_exec));
  const loom_amdgpu_feedback_packet_header_t header = {
      .record_length = packet_length,
      .kind = producer->packet_kind,
      .flags = producer->packet_flags,
      .sequence = reservation.sequence,
      .source_dispatch_ptr = producer->source->dispatch_ptr,
      .source_workgroup_id_x = producer->source->workgroup_id_x,
      .source_workitem_id_x = producer->source->workitem_id_x,
      .source_context = config_values.source_context,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(producer->build_payload(
      builder, descriptor_set, &reservation.packet_address,
      producer->payload_context, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_exec_restore(
      builder, descriptor_set, saved_report_exec, location));
  loom_op_t* report_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, terminal_block, /*args=*/NULL,
                                         /*args_count=*/0, location,
                                         &report_branch_op));

  loom_builder_set_block(builder, terminal_block);
  loom_op_t* return_op = NULL;
  return loom_low_return_build(builder, /*values=*/NULL, /*value_count=*/0,
                               location, &return_op);
}
