// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/signal.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/control_packet.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static loom_amdgpu_signal_values_t loom_amdgpu_signal_values_empty(void) {
  return (loom_amdgpu_signal_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .event_mailbox_ptr = LOOM_VALUE_ID_INVALID,
      .event_id = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_amdgpu_signal_build_u32_attr(
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

static iree_status_t loom_amdgpu_signal_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_signal_build_u32_attr(builder, IREE_SV("offset"),
                                           byte_offset, out_attr);
}

static bool loom_amdgpu_signal_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static void loom_amdgpu_signal_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  IREE_ASSERT(value < builder->module->values.count,
              "AMDGPU signal builder received an invalid low value");
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(
      loom_amdgpu_signal_type_is_register_class(descriptor_set, type,
                                                reg_class_id) &&
          loom_low_register_type_unit_count(type) == unit_count,
      "AMDGPU signal builder received a low value with an unsupported register "
      "shape");
}

static iree_status_t loom_amdgpu_signal_descriptor_operand_type(
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
      "AMDGPU signal descriptor operand has no register alternative");
  IREE_BUILTIN_UNREACHABLE();
}

static const loom_low_asm_form_t* loom_amdgpu_signal_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const uint32_t canonical_asm_form_ordinal =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor)
          ->canonical_asm_form_ordinal;
  IREE_ASSERT_LT(canonical_asm_form_ordinal, descriptor_set->asm_form_count);
  return &descriptor_set->asm_forms[canonical_asm_form_ordinal];
}

static iree_status_t loom_amdgpu_signal_descriptor_result_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index,
    loom_type_t* out_type) {
  IREE_ASSERT_LT(result_index, descriptor->result_count);
  return loom_amdgpu_signal_descriptor_operand_type(descriptor_set, descriptor,
                                                    result_index, out_type);
}

static iree_status_t loom_amdgpu_signal_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_b32_copy(
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
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &copy_op));
  *out_value = loom_value_slice_get(loom_low_op_results(copy_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_materialize_vgpr_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = source;
  IREE_ASSERT(source < builder->module->values.count,
              "AMDGPU signal builder received an invalid low value");
  const loom_type_t source_type =
      loom_module_value_type(builder->module, source);
  IREE_ASSERT(loom_low_type_is_register(source_type) &&
                  loom_low_register_type_unit_count(source_type) == 1,
              "AMDGPU signal builder cannot materialize value with unsupported "
              "register shape");
  if (loom_amdgpu_signal_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_amdgpu_signal_type_is_register_class(
                  descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR),
              "AMDGPU signal builder cannot materialize non-SGPR value into "
              "VGPR");
  return loom_amdgpu_signal_build_vgpr_b32_copy(builder, descriptor_set, source,
                                                location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u64_zero_extend(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_materialize_vgpr_b32(
      builder, descriptor_set, source, location, &low_source));
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &low_zero));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {low_source, low_zero};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, sources, IREE_ARRAYSIZE(sources),
                            vgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, sources, IREE_ARRAYSIZE(sources),
                            vgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static uint16_t loom_amdgpu_signal_packet_operand_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_asm_form_t* asm_form =
      loom_amdgpu_signal_asm_form(descriptor_set, descriptor);
  return asm_form->operand_index_count;
}

static iree_status_t loom_amdgpu_signal_append_optional_m0_operand(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_location_id_t location,
    loom_value_id_t* operands, iree_host_size_t operand_capacity,
    iree_host_size_t* inout_operand_count) {
  const uint16_t packet_operand_count =
      loom_amdgpu_signal_packet_operand_count(descriptor_set, descriptor);
  if (packet_operand_count == *inout_operand_count) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(packet_operand_count, *inout_operand_count + 1,
                 "AMDGPU signal descriptor has an unsupported packet operand "
                 "count");
  IREE_ASSERT_LT(*inout_operand_count, operand_capacity);
  loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_m0_const_u32(
      builder, descriptor_set, descriptor, 0, location, &m0_value));
  operands[*inout_operand_count] = m0_value;
  *inout_operand_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_global_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t zero_vaddr, loom_value_id_t saddr, loom_value_id_t value,
    loom_location_id_t location) {
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, zero_vaddr,
                                            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, value,
                                            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, saddr,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);

  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_signal_build_offset_attr(builder, 0, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_release_store_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);

  loom_value_id_t operands[4] = {zero_vaddr, value, saddr,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_append_optional_m0_operand(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_signal_build_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_system_memory_append_no_return_atomic_attrs(
      builder, descriptor_set, attrs, attr_capacity, out_attr_count);
}

static iree_status_t loom_amdgpu_signal_build_m0_from_sgpr(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, source,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_descriptor_result_type(
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

iree_status_t loom_amdgpu_build_signal_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location,
    loom_amdgpu_signal_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_signal_values_empty();
  loom_amdgpu_signal_require_register_class(builder, descriptor_set,
                                            signal_address,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_amdgpu_signal_values_t values = loom_amdgpu_signal_values_empty();
  values.address = signal_address;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.event_mailbox_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.event_id));
  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_signal_add_one_release(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location) {
  loom_amdgpu_signal_require_register_class(builder, descriptor_set,
                                            signal_address,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t one64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u64_const(
      builder, descriptor_set, 1, location, &one64));

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  loom_value_id_t value_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_saddr_byte_offset(
      builder, descriptor_set, signal_address, LOOM_AMDGPU_SIGNAL_VALUE_OFFSET,
      location, &value_address));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_atomic_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);
  loom_value_id_t operands[4] = {zero_vaddr, one64, value_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_append_optional_m0_operand(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op);
}

iree_status_t loom_amdgpu_build_signal_mailbox_message_id(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_id, loom_location_id_t location,
    loom_value_id_t* out_message_id) {
  IREE_ASSERT_ARGUMENT(out_message_id);
  *out_message_id = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, event_id,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  if (descriptor_set->descriptor_set_ordinal >=
      LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU signal descriptor set");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_sgpr_u32_const(
      builder, descriptor_set,
      LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_GFX9_11_12_MASK, location, &mask));
  loom_value_id_t operands[] = {event_id, mask};
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_message_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_signal_poke_mailbox(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_mailbox_ptr, loom_value_id_t event_id,
    loom_location_id_t location) {
  loom_amdgpu_signal_require_register_class(builder, descriptor_set,
                                            event_mailbox_ptr,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  loom_amdgpu_signal_require_register_class(builder, descriptor_set, event_id,
                                            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t event_id64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u64_zero_extend(
      builder, descriptor_set, event_id, location, &event_id64));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, event_mailbox_ptr, event_id64,
      location));

  loom_value_id_t message_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_mailbox_message_id(
      builder, descriptor_set, event_id, location, &message_id));
  loom_value_id_t m0_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_m0_from_sgpr(
      builder, descriptor_set, message_id, location, &m0_payload));
  return loom_amdgpu_build_control_packet_send_message_with_m0(
      builder, descriptor_set, LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG, m0_payload,
      location);
}
