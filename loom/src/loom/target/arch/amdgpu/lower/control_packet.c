// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_packet.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_AMDHSA_TRAP_ID_FATAL 2u
#define LOOM_AMDGPU_AMDHSA_TRAP_ID_DEBUG 3u

#define LOOM_AMDGPU_SENDMSG_ID_INTERRUPT 1u
#define LOOM_AMDGPU_SENDMSG_ID_RTN_GET_DOORBELL 128u

#define LOOM_AMDGPU_DOORBELL_ID_MASK 0x3FFu
#define LOOM_AMDGPU_DOORBELL_QUEUE_WAVE_ABORT 0x400u

static iree_status_t loom_amdgpu_control_packet_build_u32_attr(
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

static iree_status_t loom_amdgpu_control_packet_descriptor_operand_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  const uint32_t operand_index =
      (uint32_t)descriptor->operand_start + descriptor_operand_index;
  IREE_ASSERT_LT(operand_index, descriptor_set->operand_count);
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    return loom_low_build_register_type(descriptor_set, alt->reg_class_id,
                                        operand->unit_count, out_type);
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU control packet descriptor operand has no register alternative");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_control_packet_descriptor_result_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index,
    loom_type_t* out_type) {
  IREE_ASSERT_LT(result_index, descriptor->result_count);
  return loom_amdgpu_control_packet_descriptor_operand_type(
      descriptor_set, descriptor, result_index, out_type);
}

static iree_status_t loom_amdgpu_control_packet_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_control_packet_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t consumer_descriptor_ref, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* consumer_descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set,
                                        consumer_descriptor_ref);

  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_control_packet_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_control_packet_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_control_packet_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_control_packet_build_m0_from_sgpr(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_descriptor_result_type(
      descriptor_set, descriptor, /*result_index=*/0, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, &source,
      /*operand_count=*/1, loom_make_named_attr_slice(NULL, 0), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_control_packet_build_control_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, iree_string_view_t immediate,
    uint32_t immediate_value, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t immediate_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_u32_attr(
      builder, immediate, immediate_value, &immediate_attr));
  const bool has_result = !loom_type_equal(result_type, loom_type_none());
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(&immediate_attr, 1),
      has_result ? &result_type : NULL, has_result ? 1 : 0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_control_packet_build_binary_sgpr_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static bool loom_amdgpu_control_packet_has_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                               descriptor_ref) != NULL;
}

iree_status_t loom_amdgpu_build_control_packet_send_message(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location) {
  loom_value_id_t m0_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_m0_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG, 0,
      location, &m0_payload));
  return loom_amdgpu_build_control_packet_send_message_with_m0(
      builder, descriptor_set, message, m0_payload, location);
}

iree_status_t loom_amdgpu_build_control_packet_send_message_with_m0(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_value_id_t m0_payload, loom_location_id_t location) {
  loom_op_t* op = NULL;
  const loom_value_id_t operands[] = {m0_payload};
  return loom_amdgpu_control_packet_build_control_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG,
      IREE_SV("message"), message, operands, IREE_ARRAYSIZE(operands),
      loom_type_none(), location, &op);
}

iree_status_t loom_amdgpu_build_control_packet_send_message_rtn_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t message, loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_control_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG_RTN_B32,
      IREE_SV("message"), message, /*operands=*/NULL, /*operand_count=*/0,
      sgpr_type, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_control_packet_halt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t reason, loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_amdgpu_control_packet_build_control_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SETHALT,
      IREE_SV("reason"), reason, /*operands=*/NULL, /*operand_count=*/0,
      loom_type_none(), location, &op);
}

iree_status_t loom_amdgpu_build_control_packet_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t trap_id, loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_amdgpu_control_packet_build_control_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP,
      IREE_SV("trapid"), trap_id, /*operands=*/NULL, /*operand_count=*/0,
      loom_type_none(), location, &op);
}

iree_status_t loom_amdgpu_build_control_packet_debug_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_build_control_packet_trap(
      builder, descriptor_set, LOOM_AMDGPU_AMDHSA_TRAP_ID_DEBUG, location);
}

iree_status_t loom_amdgpu_build_control_packet_fatal_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_control_packet_trap(
      builder, descriptor_set, LOOM_AMDGPU_AMDHSA_TRAP_ID_FATAL, location));

  if (!loom_amdgpu_control_packet_has_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG_RTN_B32)) {
    return iree_ok_status();
  }

  loom_value_id_t doorbell = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_control_packet_send_message_rtn_b32(
      builder, descriptor_set, LOOM_AMDGPU_SENDMSG_ID_RTN_GET_DOORBELL,
      location, &doorbell));
  loom_value_id_t doorbell_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_sgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_DOORBELL_ID_MASK, location,
      &doorbell_mask));
  loom_value_id_t doorbell_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_binary_sgpr_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, doorbell,
      doorbell_mask, location, &doorbell_id));
  loom_value_id_t wave_abort_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_sgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_DOORBELL_QUEUE_WAVE_ABORT, location,
      &wave_abort_bit));
  loom_value_id_t interrupt_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_binary_sgpr_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, doorbell_id,
      wave_abort_bit, location, &interrupt_payload));
  loom_value_id_t m0_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_control_packet_build_m0_from_sgpr(
      builder, descriptor_set, interrupt_payload, location, &m0_payload));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_control_packet_send_message_with_m0(
      builder, descriptor_set, LOOM_AMDGPU_SENDMSG_ID_INTERRUPT, m0_payload,
      location));
  return iree_ok_status();
}
