// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/register_class.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/emit/native/assembly.h"

typedef struct loom_amdgpu_assembly_emit_state_t {
  // Wait-packet plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Wait-state plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_state_plan_t* wait_states;
  // VOPD plan used to replace paired descriptor packets.
  const loom_amdgpu_vopd_plan_t* vopd_plan;
  // Low byte of MODE's current VGPR-MSB selector state.
  uint8_t current_vgpr_msb_mode;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_wait_packet_index;
  // Next wait-state row to compare with the current scheduled packet.
  iree_host_size_t next_wait_state_index;
} loom_amdgpu_assembly_emit_state_t;

static iree_status_t loom_amdgpu_descriptor_key(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_key) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset, out_key);
}

static iree_status_t loom_amdgpu_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, &mnemonic));
  return iree_string_builder_append_string(context->builder, mnemonic);
}

static const loom_low_allocation_assignment_t* loom_amdgpu_map_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(context->allocation,
                                                         value_id, NULL);
}

static bool loom_amdgpu_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_amdgpu_append_register_range(
    const loom_native_assembly_packet_context_t* context, const char* prefix,
    const loom_low_allocation_assignment_t* assignment) {
  const uint32_t base_register = assignment->location_base;
  const uint32_t register_count = assignment->location_count;
  if (register_count == 1) {
    return iree_string_builder_append_format(context->builder, "%s%" PRIu32,
                                             prefix, base_register);
  }
  const uint32_t last_register = base_register + register_count - 1;
  return iree_string_builder_append_format(
      context->builder, "%s[%" PRIu32 ":%" PRIu32 "]", prefix, base_register,
      last_register);
}

static iree_status_t loom_amdgpu_append_register_range_units(
    const loom_native_assembly_packet_context_t* context, const char* prefix,
    uint32_t base_register, uint32_t register_count) {
  if (register_count == 1) {
    return iree_string_builder_append_format(context->builder, "%s%" PRIu32,
                                             prefix, base_register);
  }
  const uint32_t last_register = base_register + register_count - 1;
  return iree_string_builder_append_format(
      context->builder, "%s[%" PRIu32 ":%" PRIu32 "]", prefix, base_register,
      last_register);
}

static iree_status_t loom_amdgpu_append_assignment(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_append_register_range(context, "s", assignment);
  }
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return loom_amdgpu_append_register_range(context, "v", assignment);
  }
  if (loom_amdgpu_register_class_is_agpr(
          context->allocation->target.descriptor_set,
          assignment->descriptor_reg_class_id)) {
    return loom_amdgpu_append_register_range(context, "acc", assignment);
  }
  if (loom_amdgpu_register_class_is_m0(
          context->allocation->target.descriptor_set,
          assignment->descriptor_reg_class_id)) {
    if (assignment->location_base != 0 || assignment->location_count != 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly M0 assignment must name physical register 0");
    }
    return iree_string_builder_append_cstring(context->builder, "m0");
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      context->allocation, assignment, &register_class));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly register class '%.*s' is unsupported",
      (int)register_class.size, register_class.data);
}

static iree_status_t loom_amdgpu_descriptor_operand_assignment(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_operand_index,
    const loom_low_allocation_assignment_t** out_assignment) {
  *out_assignment = NULL;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_op_t* op = context->packet->node->op;
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly descriptor operand %" PRIu16
                            " is out of range",
                            descriptor_operand_index);
  }
  if (descriptor_operand_index < descriptor->result_count) {
    if (descriptor_operand_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU assembly descriptor result %" PRIu16
                              " has no matching SSA result",
                              descriptor_operand_index);
    }
    *out_assignment = loom_amdgpu_map_assignment(
        context, loom_op_const_results(op)[descriptor_operand_index]);
    return iree_ok_status();
  }

  uint16_t packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    if (i == descriptor_operand_index) {
      if (packet_operand_index >= op->operand_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU assembly descriptor operand %" PRIu16
                                " maps outside the packet operands",
                                descriptor_operand_index);
      }
      *out_assignment = loom_amdgpu_map_assignment(
          context, loom_op_const_operands(op)[packet_operand_index]);
      return iree_ok_status();
    }
    ++packet_operand_index;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU assembly descriptor operand %" PRIu16
                          " does not name an explicit packet value",
                          descriptor_operand_index);
}

static iree_status_t loom_amdgpu_append_descriptor_assignment(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_operand_index) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  const loom_low_allocation_assignment_t* assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_operand_assignment(
      context, descriptor_operand_index, &assignment));
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      operand->address_map_kind != LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
    return loom_amdgpu_append_assignment(context, assignment);
  }
  const uint32_t addressable_unit_count = operand->addressable_unit_count;
  if (addressable_unit_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly target-state VGPR operand has no "
                            "addressable window size");
  }
  if (assignment->location_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly target-state VGPR operand has an "
                            "empty assignment");
  }
  const uint64_t assigned_last =
      (uint64_t)assignment->location_base + assignment->location_count - 1u;
  if (assignment->location_base / addressable_unit_count !=
      assigned_last / addressable_unit_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly target-state VGPR range v[%" PRIu32 ":%" PRIu64
        "] crosses a %" PRIu32 "-register window",
        assignment->location_base, assigned_last, addressable_unit_count);
  }
  return loom_amdgpu_append_register_range_units(
      context, "v", assignment->location_base % addressable_unit_count,
      assignment->location_count);
}

static iree_status_t loom_amdgpu_append_descriptor_operand(
    const loom_native_assembly_packet_context_t* context,
    uint16_t packet_operand_index) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  uint16_t current_packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT) ||
        !loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    if (current_packet_operand_index == packet_operand_index) {
      return loom_amdgpu_append_descriptor_assignment(context, i);
    }
    ++current_packet_operand_index;
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU assembly packet operand %" PRIu16
                          " has no matching descriptor operand",
                          packet_operand_index);
}

static iree_status_t loom_amdgpu_append_move_location(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_move_location_t* location) {
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_string_builder_append_format(context->builder, "s%" PRIu32,
                                             location->location);
  }
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_string_builder_append_format(context->builder, "v%" PRIu32,
                                             location->location);
  }
  if (loom_amdgpu_register_class_is_agpr(
          context->allocation->target.descriptor_set,
          location->descriptor_reg_class_id)) {
    return iree_string_builder_append_format(context->builder, "acc%" PRIu32,
                                             location->location);
  }
  if (loom_amdgpu_register_class_is_m0(
          context->allocation->target.descriptor_set,
          location->descriptor_reg_class_id)) {
    if (location->location != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly M0 move location must name physical register 0");
    }
    return iree_string_builder_append_cstring(context->builder, "m0");
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly descriptor register class ID %" PRIu16 " is unsupported",
      location->descriptor_reg_class_id);
}

static iree_status_t loom_amdgpu_append_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  assignment = loom_amdgpu_map_assignment(context, value_id);
  return loom_amdgpu_append_assignment(context, assignment);
}

static iree_status_t loom_amdgpu_append_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly result index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_amdgpu_append_operand(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly operand index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_operands(op)[operand_index]);
}

static loom_named_attr_slice_t loom_amdgpu_packet_attrs(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static iree_status_t loom_amdgpu_read_packet_i64_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, int64_t* out_value) {
  return loom_native_assembly_read_i64_attr(context->schedule->module,
                                            loom_amdgpu_packet_attrs(context),
                                            name, out_value);
}

static iree_status_t loom_amdgpu_read_packet_immediate_i64(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_immediate_t* immediate, int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset, &name));
  const loom_named_attr_t* attr = loom_native_assembly_find_attr(
      context->schedule->module, loom_amdgpu_packet_attrs(context), name);
  if (attr == NULL) {
    if (iree_all_bits_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly requires attribute '%.*s'",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly attribute '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_packet_immediate_by_index_i64(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index, int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor_immediate_index >= descriptor->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate index is out of range");
  }
  const uint32_t immediate_row =
      descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >= descriptor_set->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate row is outside the descriptor "
        "set");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  return loom_amdgpu_read_packet_immediate_i64(context, immediate, out_value);
}

static iree_status_t loom_amdgpu_append_packet_immediate_unsigned_hex(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index, uint8_t bit_width) {
  IREE_ASSERT(bit_width > 0 && bit_width <= 32);
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  const uint64_t max_value = (UINT64_C(1) << bit_width) - 1;
  if (value < 0 || (uint64_t)value > max_value) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate value %" PRId64 " is not a u%u",
        value, (unsigned)bit_width);
  }
  const int hex_digit_count = (int)((bit_width + 3) / 4);
  return iree_string_builder_append_format(context->builder, "0x%0*" PRIx64,
                                           hex_digit_count, (uint64_t)value);
}

static iree_status_t loom_amdgpu_append_packet_immediate_i64(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  return iree_string_builder_append_format(context->builder, "%" PRId64, value);
}

static iree_status_t loom_amdgpu_find_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t field_name, const loom_low_immediate_t** out_immediate) {
  *out_immediate = NULL;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor->immediate_count >
          descriptor_set->immediate_count - descriptor->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate range is out of range");
  }
  if (descriptor->immediate_count != 0 && descriptor_set->immediates == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly descriptor immediates table is "
                            "missing");
  }
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &name));
    if (iree_string_view_equal(name, field_name)) {
      *out_immediate = immediate;
      return iree_ok_status();
    }
  }
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &descriptor_key));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU assembly descriptor '%.*s' has no immediate "
                          "attribute '%.*s'",
                          (int)descriptor_key.size, descriptor_key.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_amdgpu_descriptor_has_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_has_memory_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    loom_low_memory_space_t memory_space, bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind && effect->memory_space == memory_space) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_is_global_to_lds(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, bool* out_is_global_to_lds) {
  bool has_global_read = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ,
      LOOM_LOW_MEMORY_SPACE_GLOBAL, &has_global_read));
  bool has_workgroup_write = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      LOOM_LOW_MEMORY_SPACE_WORKGROUP, &has_workgroup_write));
  *out_is_global_to_lds = has_global_read && has_workgroup_write;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind,
    bool* out_uses_resource_kind) {
  *out_uses_resource_kind = false;
  if (descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor schedule class is out of range");
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  if (schedule_class->issue_use_count == 0) {
    return iree_ok_status();
  }
  if (schedule_class->issue_use_start > descriptor_set->issue_use_count ||
      schedule_class->issue_use_count >
          descriptor_set->issue_use_count - schedule_class->issue_use_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU descriptor schedule-class issue-use range is out of range");
  }
  if (descriptor_set->issue_uses == NULL || descriptor_set->resources == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor schedule resources are missing");
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU descriptor issue-use resource is out of "
                              "range");
    }
    const loom_low_resource_t* resource =
        &descriptor_set->resources[issue_use->resource_id];
    if (resource->kind == kind) {
      *out_uses_resource_kind = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_comma(
    const loom_native_assembly_packet_context_t* context) {
  return iree_string_builder_append_cstring(context->builder, ", ");
}

static iree_status_t loom_amdgpu_append_result_operand_list(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  bool needs_comma = false;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, i));
    needs_comma = true;
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, i));
    needs_comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_canonical_asm_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t** out_form) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly descriptor '%.*s' has no canonical asm form",
        (int)key.size, key.data);
  }
  *out_form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor->canonical_asm_form_ordinal);
  if (*out_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly canonical asm form is out of "
                            "range");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_asm_form_native_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form, iree_string_view_t* out_mnemonic) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  loom_bstring_table_offset_t string_offset = form->mnemonic_string_offset;
  if (form->native_assembly_mnemonic_string_offset !=
      LOOM_LOW_STRING_OFFSET_NONE) {
    string_offset = form->native_assembly_mnemonic_string_offset;
  }
  return loom_native_assembly_descriptor_string(descriptor_set, string_offset,
                                                out_mnemonic);
}

static iree_status_t loom_amdgpu_append_descriptor_value_list(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->operand_start > descriptor_set->operand_count ||
      descriptor->operand_count >
          descriptor_set->operand_count - descriptor->operand_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor operand range is out of range");
  }
  if (descriptor->operand_count == 0) {
    return iree_ok_status();
  }
  if (descriptor_set->operands == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor operands table is missing");
  }

  bool needs_comma = false;
  for (uint16_t i = 0; i < descriptor->result_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " "));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, i));
    needs_comma = true;
  }

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
      continue;
    }
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU descriptor operand row %" PRIu16
          " has role %u that cannot map to a packet operand",
          i, (unsigned)operand->role);
    }
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " "));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, i));
    needs_comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_asm_form_separator(
    const loom_native_assembly_packet_context_t* context, bool* in_list) {
  if (*in_list) {
    return loom_amdgpu_append_comma(context);
  }
  *in_list = true;
  return iree_string_builder_append_cstring(context->builder, " ");
}

typedef enum loom_amdgpu_assembly_value_kind_e {
  LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_RESULT,
  LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_OPERAND,
  LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_PACKET_IMMEDIATE_I64,
} loom_amdgpu_assembly_value_kind_t;

typedef struct loom_amdgpu_assembly_value_spec_t {
  // Kind of value appended at this position.
  loom_amdgpu_assembly_value_kind_t kind;
  // Result, operand, or immediate index for indexed value kinds.
  uint16_t index;
} loom_amdgpu_assembly_value_spec_t;

#define LOOM_AMDGPU_ASSEMBLY_PACKET_FORM_VALUE_CAPACITY 8

typedef struct loom_amdgpu_assembly_packet_form_t {
  // Number of rows in |values|.
  iree_host_size_t value_count;
  // Ordered values appended after the mnemonic.
  loom_amdgpu_assembly_value_spec_t
      values[LOOM_AMDGPU_ASSEMBLY_PACKET_FORM_VALUE_CAPACITY];
} loom_amdgpu_assembly_packet_form_t;

#define LOOM_AMDGPU_ASM_VALUE_RESULT(value_index)     \
  {                                                   \
      .kind = LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_RESULT, \
      .index = value_index,                           \
  }
#define LOOM_AMDGPU_ASM_VALUE_OPERAND(value_index)     \
  {                                                    \
      .kind = LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_OPERAND, \
      .index = value_index,                            \
  }
#define LOOM_AMDGPU_ASM_VALUE_IMMEDIATE_I64(value_index)            \
  {                                                                 \
      .kind = LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_PACKET_IMMEDIATE_I64, \
      .index = value_index,                                         \
  }

static iree_status_t loom_amdgpu_append_assembly_value(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_assembly_value_spec_t* value) {
  switch (value->kind) {
    case LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_RESULT:
      return loom_amdgpu_append_result(context, value->index);
    case LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_OPERAND:
      return loom_amdgpu_append_operand(context, value->index);
    case LOOM_AMDGPU_ASSEMBLY_VALUE_KIND_PACKET_IMMEDIATE_I64:
      return loom_amdgpu_append_packet_immediate_i64(context, value->index);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "AMDGPU assembly value kind %u is unsupported",
                              (unsigned)value->kind);
  }
}

static iree_status_t loom_amdgpu_append_assembly_packet_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_assembly_packet_form_t* form) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  bool in_list = false;
  if (form->value_count > IREE_ARRAYSIZE(form->values)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly packet form value count is out of range");
  }
  for (iree_host_size_t i = 0; i < form->value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, &in_list));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_assembly_value(context, &form->values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_asm_form_value(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    bool is_result) {
  const loom_op_t* op = context->packet->node->op;
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU asm-form operand index is outside the descriptor");
  }
  if (is_result) {
    if (descriptor_operand_index >= descriptor->result_count ||
        descriptor_operand_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU asm-form result field does not name an "
                              "emitted result");
    }
    return loom_amdgpu_append_descriptor_assignment(context,
                                                    descriptor_operand_index);
  }
  if (descriptor_operand_index < descriptor->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU asm-form operand field unexpectedly names "
                            "a descriptor result");
  }
  const uint16_t operand_index =
      descriptor_operand_index - descriptor->result_count;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU asm-form operand field does not name an "
                            "emitted operand");
  }
  (void)operand_index;
  return loom_amdgpu_append_descriptor_assignment(context,
                                                  descriptor_operand_index);
}

static iree_status_t loom_amdgpu_append_asm_form_values(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint32_t start, uint16_t count,
    bool is_result, bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < count; ++i) {
    const uint32_t asm_operand_index = start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form operand row is outside the descriptor set");
    }
    const uint16_t descriptor_operand_index =
        descriptor_set->asm_operand_indices[asm_operand_index];
    if (descriptor_operand_index >= descriptor->operand_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form operand index is outside the descriptor");
    }
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set
             ->operands[descriptor->operand_start + descriptor_operand_index];
    if (iree_any_bit_set(descriptor_operand->flags,
                         LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_value(
        context, descriptor, descriptor_operand_index, is_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_native_asm_form_value(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor,
    const loom_low_native_asm_value_t* value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  switch (value->kind) {
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL: {
      iree_string_view_t literal = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
          descriptor_set, value->literal_string_offset, &literal));
      return iree_string_builder_append_string(context->builder, literal);
    }
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT:
      return loom_amdgpu_append_asm_form_value(
          context, descriptor, value->index, /*is_result=*/true);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_OPERAND:
      return loom_amdgpu_append_asm_form_value(
          context, descriptor, value->index, /*is_result=*/false);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64:
      return loom_amdgpu_append_packet_immediate_i64(context, value->index);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX:
      return loom_amdgpu_append_packet_immediate_unsigned_hex(
          context, value->index, value->bit_width);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "AMDGPU native asm value kind %u is unsupported",
                              (unsigned)value->kind);
  }
}

static iree_status_t loom_amdgpu_append_native_asm_form_values(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form, bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  for (uint16_t i = 0; i < form->native_assembly_value_count; ++i) {
    const uint32_t native_value_index = form->native_assembly_value_start + i;
    if (native_value_index >= descriptor_set->native_asm_value_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU native asm-form value row is outside the descriptor set");
    }
    const loom_low_native_asm_value_t* value =
        &descriptor_set->native_asm_values[native_value_index];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_native_asm_form_value(context, descriptor, value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_asm_form_immediates(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, const loom_low_asm_form_t* form,
    bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < form->immediate_count; ++i) {
    const uint32_t asm_immediate_index = form->immediate_start + i;
    if (asm_immediate_index >= descriptor_set->asm_immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form immediate row is outside the descriptor set");
    }
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_immediate_index];
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form immediate references an invalid descriptor field");
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    asm_immediate->immediate_index];
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      iree_string_view_t spelling = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
          descriptor_set, asm_immediate->name_string_offset, &spelling));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%.*s(%" PRId64 ")", (int)spelling.size,
          spelling.data, value));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%" PRId64, value));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_canonical_asm_form_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_asm_form_native_mnemonic(context, form, &mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  bool in_list = false;
  if (form->native_assembly_value_count > 0) {
    return loom_amdgpu_append_native_asm_form_values(context, form, &in_list);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
      context, descriptor, form->result_operand_index_start,
      form->result_operand_index_count, /*is_result=*/true, &in_list));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
      context, descriptor, form->operand_index_start, form->operand_index_count,
      /*is_result=*/false, &in_list));
  return loom_amdgpu_append_asm_form_immediates(context, descriptor, form,
                                                &in_list);
}

static iree_status_t loom_amdgpu_append_basic_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  if (result_count + operand_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  return loom_amdgpu_append_result_operand_list(context, result_count,
                                                operand_count);
}

static iree_status_t loom_amdgpu_append_memory_immediate_suffixes(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->immediate_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor->immediate_count >
          descriptor_set->immediate_count - descriptor->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate range is out of range");
  }
  if (descriptor_set->immediates == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly descriptor immediates table is "
                            "missing");
  }
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &name));
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s:%" PRId64, (int)name.size, name.data, value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_append_native_memory_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
  if (form->native_assembly_value_count == 0) {
    return iree_ok_status();
  }
  *out_matched = true;
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_asm_form_native_mnemonic(context, form, &mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  bool in_list = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_native_asm_form_values(context, form, &in_list));
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static iree_status_t loom_amdgpu_append_memory_packet(
    const loom_native_assembly_packet_context_t* context) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_native_memory_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_value_list(context));
  } else {
    const loom_low_asm_form_t* form = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
    iree_string_view_t mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_asm_form_native_mnemonic(context, form, &mnemonic));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(context->builder, mnemonic));
    bool in_list = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
        context, descriptor, form->result_operand_index_start,
        form->result_operand_index_count, /*is_result=*/true, &in_list));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
        context, descriptor, form->operand_index_start,
        form->operand_index_count, /*is_result=*/false, &in_list));
  }
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static iree_status_t loom_amdgpu_append_offset_suffix(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_immediate_t* immediate = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_packet_immediate(
      context, IREE_SV("offset"), &immediate));
  int64_t offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_immediate_i64(context, immediate, &offset));
  return iree_string_builder_append_format(context->builder, " offset:%" PRId64,
                                           offset);
}

static bool loom_amdgpu_mubuf_load_uses_off_zero_form(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  return op->result_count == 1 && op->operand_count == 1;
}

static bool loom_amdgpu_mubuf_store_uses_off_zero_form(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  return op->result_count == 0 && op->operand_count == 2;
}

static iree_status_t loom_amdgpu_append_mubuf_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_native_memory_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_mubuf_load_uses_off_zero_form(context)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "0"));
    return loom_amdgpu_append_offset_suffix(context);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 2));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_mubuf_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_native_memory_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_mubuf_store_uses_off_zero_form(context)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "0"));
    return loom_amdgpu_append_offset_suffix(context);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 3));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_buffer_atomic_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  if (descriptor->result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, 0));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 3));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static bool loom_amdgpu_descriptor_uses_global_pointer_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_uses_scratch_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_SCRATCH:
    case LOOM_AMDGPU_ENCODING_FORMAT_VSCRATCH:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_uses_data_share_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_uses_buffer_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
      return true;
    default:
      return false;
  }
}

static iree_host_size_t loom_amdgpu_explicit_packet_operand_count(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  iree_host_size_t explicit_operand_count = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT) ||
        !loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    ++explicit_operand_count;
  }
  return explicit_operand_count;
}

static iree_status_t loom_amdgpu_append_mubuf_load_lds_packet(
    const loom_native_assembly_packet_context_t* context) {
  const iree_host_size_t explicit_operand_count =
      loom_amdgpu_explicit_packet_operand_count(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  if (explicit_operand_count == 1) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "0"));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 2));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, " offen"));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_offset_suffix(context));
  return iree_string_builder_append_cstring(context->builder, " lds");
}

static bool loom_amdgpu_descriptor_operand_uses_m0(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand) {
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return false;
    }
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (loom_amdgpu_register_class_is_m0(descriptor_set, alt->reg_class_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_descriptor_clobbers_hidden_m0(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + (uint32_t)i];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_IMPLICIT ||
        !loom_amdgpu_descriptor_operand_uses_m0(descriptor_set, operand)) {
      continue;
    }
    return true;
  }
  return false;
}

static iree_status_t loom_amdgpu_append_hidden_m0_zero_if_required(
    const loom_native_assembly_packet_context_t* context) {
  if (!loom_amdgpu_descriptor_clobbers_hidden_m0(context)) {
    return iree_ok_status();
  }
  return iree_string_builder_append_cstring(context->builder,
                                            "s_mov_b32_m0_imm m0, 0\n  ");
}

static bool loom_amdgpu_descriptor_uses_global_scalar_base_format(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const uint16_t address_operand_index = descriptor->result_count;
  if (address_operand_index >= descriptor->operand_count) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  return operands[address_operand_index].unit_count == 1;
}

static iree_status_t loom_amdgpu_append_global_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_global_load_lds_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_global_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 2));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_scratch_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const iree_host_size_t explicit_operand_count =
      loom_amdgpu_explicit_packet_operand_count(context);
  if (op->result_count != 1 || explicit_operand_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU scratch load packet has unexpected "
                            "operand shape");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_hidden_m0_zero_if_required(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (explicit_operand_count == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "off"));
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static iree_status_t loom_amdgpu_append_scratch_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const iree_host_size_t explicit_operand_count =
      loom_amdgpu_explicit_packet_operand_count(context);
  if (op->result_count != 0 || explicit_operand_count == 0 ||
      explicit_operand_count > 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU scratch store packet has unexpected "
                            "operand shape");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_hidden_m0_zero_if_required(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  if (explicit_operand_count == 2) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 1));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_operand(context, 0));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "off"));
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static iree_status_t loom_amdgpu_append_waitcnt_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t vmcnt = 0;
  int64_t lgkmcnt = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("vmcnt"), &vmcnt));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("lgkmcnt"), &lgkmcnt));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  return iree_string_builder_append_format(
      context->builder, " vmcnt(%" PRId64 ") lgkmcnt(%" PRId64 ")", vmcnt,
      lgkmcnt);
}

typedef enum loom_amdgpu_descriptor_packet_route_flag_bits_e {
  // Descriptor has a read effect.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT = 1u << 0,
  // Descriptor has a write effect.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT = 1u << 1,
  // Descriptor has a counter effect.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_COUNTER_EFFECT = 1u << 2,
  // Descriptor reads global memory and writes LDS.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS = 1u << 3,
  // Descriptor uses MUBUF or VBUFFER encoding.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT = 1u << 4,
  // Descriptor uses a global flat/vector-global encoding.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT = 1u << 5,
  // Descriptor uses a flat scratch/vector-scratch encoding.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT = 1u << 6,
  // Descriptor uses DS or VDS encoding.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT = 1u << 7,
  // Descriptor has exactly the two immediates used by s_waitcnt.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_TWO_IMMEDIATES = 1u << 8,
} loom_amdgpu_descriptor_packet_route_flag_bits_t;
typedef uint32_t loom_amdgpu_descriptor_packet_route_flags_t;

typedef iree_status_t (*loom_amdgpu_append_descriptor_packet_fn_t)(
    const loom_native_assembly_packet_context_t* context);

typedef struct loom_amdgpu_descriptor_packet_route_t {
  // Fact bits required for this route to match.
  loom_amdgpu_descriptor_packet_route_flags_t required_flags;
  // Fact bits that prevent this route from matching.
  loom_amdgpu_descriptor_packet_route_flags_t forbidden_flags;
  // Assembly appender used when the route matches.
  loom_amdgpu_append_descriptor_packet_fn_t append;
} loom_amdgpu_descriptor_packet_route_t;

static bool loom_amdgpu_descriptor_packet_route_matches(
    const loom_amdgpu_descriptor_packet_route_t* route,
    loom_amdgpu_descriptor_packet_route_flags_t flags) {
  return iree_all_bits_set(flags, route->required_flags) &&
         !iree_any_bit_set(flags, route->forbidden_flags);
}

static iree_status_t loom_amdgpu_read_descriptor_packet_route_flags(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_descriptor_packet_route_flags_t* out_flags) {
  *out_flags = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;

  bool has_read_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ, &has_read_effect));
  if (has_read_effect) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT;
  }
  bool has_write_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      &has_write_effect));
  if (has_write_effect) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT;
  }
  bool has_counter_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_COUNTER,
      &has_counter_effect));
  if (has_counter_effect) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_COUNTER_EFFECT;
  }
  if (has_read_effect && has_write_effect) {
    bool is_global_to_lds = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_is_global_to_lds(
        descriptor_set, descriptor, &is_global_to_lds));
    if (is_global_to_lds) {
      *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS;
    }
  }
  if (loom_amdgpu_descriptor_uses_buffer_format(descriptor)) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT;
  }
  if (loom_amdgpu_descriptor_uses_global_pointer_format(descriptor)) {
    *out_flags |=
        LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT;
  }
  if (loom_amdgpu_descriptor_uses_scratch_format(descriptor)) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT;
  }
  if (loom_amdgpu_descriptor_uses_data_share_format(descriptor)) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT;
  }
  if (descriptor->immediate_count == 2) {
    *out_flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_TWO_IMMEDIATES;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_unsupported_read_write_packet(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly descriptor '%.*s' has both read and write effects",
      (int)key.size, key.data);
}

static iree_status_t loom_amdgpu_try_append_descriptor_packet_route(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_descriptor_packet_route_flags_t flags, bool* out_matched) {
  static const loom_amdgpu_descriptor_packet_route_t kRoutes[] = {
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .append = loom_amdgpu_append_global_load_lds_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .append = loom_amdgpu_append_mubuf_load_lds_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .append = loom_amdgpu_append_buffer_atomic_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .append = loom_amdgpu_append_memory_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT,
          .append = loom_amdgpu_append_memory_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .append = loom_amdgpu_append_unsupported_read_write_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .append = loom_amdgpu_append_mubuf_load_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .append = loom_amdgpu_append_mubuf_store_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .append = loom_amdgpu_append_scratch_load_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .append = loom_amdgpu_append_scratch_store_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .append = loom_amdgpu_append_global_load_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .append = loom_amdgpu_append_global_store_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT,
          .append = loom_amdgpu_append_memory_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .append = loom_amdgpu_append_memory_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .append = loom_amdgpu_append_memory_packet,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_COUNTER_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_TWO_IMMEDIATES,
          .append = loom_amdgpu_append_waitcnt_packet,
      },
  };
  *out_matched = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRoutes); ++i) {
    const loom_amdgpu_descriptor_packet_route_t* route = &kRoutes[i];
    if (!loom_amdgpu_descriptor_packet_route_matches(route, flags)) {
      continue;
    }
    *out_matched = true;
    return route->append(context);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_materialized_wait_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set,
                                                 wait_packet->descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet descriptor row does not belong to the selected "
        "descriptor set");
  }
  if (wait_packet->immediate_start > wait_packets->immediate_count ||
      wait_packet->immediate_count >
          wait_packets->immediate_count - wait_packet->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet immediate range is out of range");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, descriptor->mnemonic_string_offset, &mnemonic));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "  "));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s(%" PRIu16 ")", (int)immediate->name.size,
        immediate->name.data, immediate->value));
  }
  return iree_string_builder_append_cstring(context->builder, "\n");
}

static bool loom_amdgpu_wait_packet_is_before_node(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_schedule_node_t* node) {
  return wait_packet->block_index < node->block_index ||
         (wait_packet->block_index == node->block_index &&
          wait_packet->scheduled_ordinal < node->scheduled_ordinal);
}

static bool loom_amdgpu_wait_packet_matches_packet(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_packet->block_index == node->block_index &&
         wait_packet->scheduled_ordinal == node->scheduled_ordinal &&
         wait_packet->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_append_wait_packets_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state->wait_packets == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = context->packet->node;
  while (state->next_wait_packet_index < state->wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->wait_packets->packets[state->next_wait_packet_index];
    if (loom_amdgpu_wait_packet_is_before_node(wait_packet, node)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait packet plan contains an insertion point before the "
          "current scheduled packet");
    }
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_materialized_wait_packet(
        context, wait_packet, state->wait_packets));
    ++state->next_wait_packet_index;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_state_is_before_node(
    const loom_amdgpu_wait_state_t* wait_state,
    const loom_low_schedule_node_t* node) {
  return wait_state->block_index < node->block_index ||
         (wait_state->block_index == node->block_index &&
          wait_state->scheduled_ordinal < node->scheduled_ordinal);
}

static bool loom_amdgpu_wait_state_matches_packet(
    const loom_amdgpu_wait_state_t* wait_state,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_state->block_index == node->block_index &&
         wait_state->scheduled_ordinal == node->scheduled_ordinal &&
         wait_state->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_append_s_nop_cycles(
    const loom_native_assembly_packet_context_t* context,
    uint16_t cycle_count) {
  while (cycle_count != 0) {
    const uint16_t chunk = cycle_count > LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES
                               ? LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES
                               : cycle_count;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, "  s_nop %" PRIu16 "\n", (uint16_t)(chunk - 1)));
    cycle_count -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_wait_state_action(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_wait_state_t* wait_state) {
  switch (wait_state->action) {
    case LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP:
      return loom_amdgpu_append_s_nop_cycles(context, wait_state->cycle_count);
    case LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN:
    default: {
      iree_string_view_t action_name =
          loom_amdgpu_wait_state_action_name(wait_state->action);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unsupported AMDGPU wait-state action '%.*s'",
                              (int)action_name.size, action_name.data);
    }
  }
}

static iree_status_t loom_amdgpu_append_wait_states_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state->wait_states == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = context->packet->node;
  while (state->next_wait_state_index < state->wait_states->state_count) {
    const loom_amdgpu_wait_state_t* wait_state =
        &state->wait_states->states[state->next_wait_state_index];
    if (loom_amdgpu_wait_state_is_before_node(wait_state, node)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait-state plan contains an insertion point before the "
          "current scheduled packet");
    }
    if (!loom_amdgpu_wait_state_matches_packet(wait_state, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_wait_state_action(context, wait_state));
    ++state->next_wait_state_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_waits_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_wait_packets_before_packet(user_data, context));
  return loom_amdgpu_append_wait_states_before_packet(user_data, context);
}

static iree_status_t loom_amdgpu_copy_mnemonic(
    uint16_t descriptor_reg_class_id, iree_string_view_t* out_mnemonic) {
  if (descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    *out_mnemonic = IREE_SV("s_mov_b32");
    return iree_ok_status();
  }
  if (descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    *out_mnemonic = IREE_SV("v_mov_b32");
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly copy descriptor register class ID %" PRIu16
      " is unsupported",
      descriptor_reg_class_id);
}

typedef struct loom_amdgpu_assembly_move_state_t {
  // Packet context receiving assembly text.
  const loom_native_assembly_packet_context_t* context;
  // Surrounding assembly state tracking MODE across scheduled packets.
  loom_amdgpu_assembly_emit_state_t* emit_state;
  // Target move mnemonic used for each emitted unit move.
  iree_string_view_t mnemonic;
  // Number of non-empty lines emitted so far.
  uint32_t emitted_count;
} loom_amdgpu_assembly_move_state_t;

static iree_status_t loom_amdgpu_vgpr_msb_insert_requirement(
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank, uint8_t* mask,
    uint8_t* value) {
  if (slot == LOOM_AMDGPU_VGPR_MSB_SLOT_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU VGPR-MSB assembly state has no encoding "
                            "slot");
  }
  if (bank > 3) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VGPR-MSB assembly state bank %" PRIu32
                            " exceeds the two-bit selector range",
                            bank);
  }
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  if ((*mask & slot_mask) != 0 && (*value & slot_mask) != slot_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU VGPR-MSB assembly state requires conflicting banks for one "
        "encoding slot");
  }
  *mask |= slot_mask;
  *value = (uint8_t)((*value & ~slot_mask) | slot_value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_move_line_prefix(
    loom_amdgpu_assembly_move_state_t* state) {
  if (state->emitted_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_cstring(state->context->builder, "\n  ");
}

static iree_status_t loom_amdgpu_append_vgpr_msb_mode(
    loom_amdgpu_assembly_move_state_t* state, uint8_t new_mode) {
  IREE_ASSERT(state->emit_state != NULL);
  if (state->emit_state->current_vgpr_msb_mode == new_mode) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          state->context->schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
  IREE_ASSERT(descriptor != NULL);
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      state->context->schedule->target.descriptor_set,
      descriptor->mnemonic_string_offset, &mnemonic));
  const uint16_t immediate =
      (uint16_t)(((uint16_t)state->emit_state->current_vgpr_msb_mode << 8) |
                 new_mode);
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_line_prefix(state));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      state->context->builder, "%.*s %" PRIu16, (int)mnemonic.size,
      mnemonic.data, immediate));
  ++state->emitted_count;
  state->emit_state->current_vgpr_msb_mode = new_mode;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_vgpr_msb_requirement(
    loom_amdgpu_assembly_move_state_t* state, uint8_t mask, uint8_t value) {
  const uint8_t current_mode =
      state->emit_state == NULL ? 0 : state->emit_state->current_vgpr_msb_mode;
  const uint8_t new_mode = (uint8_t)((current_mode & ~mask) | (value & mask));
  return loom_amdgpu_append_vgpr_msb_mode(state, new_mode);
}

static iree_status_t loom_amdgpu_append_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_assembly_move_state_t* state =
      (loom_amdgpu_assembly_move_state_t*)user_data;
  const loom_native_assembly_packet_context_t* context = state->context;
  const uint32_t window = LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE;
  if (destination->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      source->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    if (destination->descriptor_reg_class_id ==
            source->descriptor_reg_class_id &&
        destination->location == source->location) {
      return iree_ok_status();
    }
    const uint32_t destination_bank = destination->location / window;
    const uint32_t source_bank = source->location / window;
    uint8_t mask = 0;
    uint8_t value = 0;
    if (destination->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vgpr_msb_insert_requirement(
          LOOM_AMDGPU_VGPR_MSB_SLOT_DST, destination_bank, &mask, &value));
    }
    if (source->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vgpr_msb_insert_requirement(
          LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0, source_bank, &mask, &value));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_vgpr_msb_requirement(state, mask, value));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_line_prefix(state));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, state->mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  if (destination->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, "v%" PRIu32, destination->location % window));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_move_location(context, destination));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (source->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, "v%" PRIu32, source->location % window));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_location(context, source));
  }
  ++state->emitted_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_move_sequence(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_assembly_emit_state_t* emit_state, iree_string_view_t mnemonic,
    iree_host_size_t move_count,
    const loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count) {
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .emit_state = emit_state,
      .mnemonic = mnemonic,
  };
  const uint8_t saved_mode =
      emit_state == NULL ? 0 : emit_state->current_vgpr_msb_mode;
  loom_low_move_sequence_options_t options = {
      .descriptor_set = context->allocation->target.descriptor_set,
      .temporary_locations = temporary_locations,
      .temporary_location_count = temporary_location_count,
      .emit_move =
          {
              .fn = loom_amdgpu_append_move,
              .user_data = &move_state,
          },
  };
  IREE_RETURN_IF_ERROR(
      loom_low_move_sequence_emit(context->move_scratch, move_count, &options));
  return loom_amdgpu_append_vgpr_msb_mode(&move_state, saved_mode);
}

static iree_status_t loom_amdgpu_packet_move_temporaries(
    const loom_native_assembly_packet_context_t* context,
    loom_low_move_location_t** out_temporary_locations,
    iree_host_size_t* out_temporary_location_count) {
  *out_temporary_locations = NULL;
  *out_temporary_location_count = 0;
  const loom_low_allocation_packet_move_temporary_group_t* group =
      loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
          context->allocation, context->packet->node->source_ordinal);
  if (!group) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_temporaries(
      context->move_scratch, group->temporary_count, out_temporary_locations));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_packet_move_temporaries(
      context->allocation, group, *out_temporary_locations,
      group->temporary_count));
  *out_temporary_location_count = group->temporary_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_edge_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_assembly_move_state_t* state =
      (loom_amdgpu_assembly_move_state_t*)user_data;
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      destination->descriptor_reg_class_id, &mnemonic));
  state->mnemonic = mnemonic;
  return loom_amdgpu_append_move(user_data, destination, source);
}

static iree_status_t loom_amdgpu_emit_edge_copy_group(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_assembly_emit_state_t* emit_state,
    const loom_low_allocation_edge_copy_group_t* group, bool* out_emitted) {
  *out_emitted = false;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      context->allocation, group, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }
  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  loom_low_move_location_t* temporaries = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_temporaries(
      context->move_scratch, group->temporary_count, &temporaries));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_units(
      context->allocation, group, moves, move_count));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_temporaries(
      context->allocation, group, temporaries, group->temporary_count));
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .emit_state = emit_state,
  };
  const uint8_t saved_mode =
      emit_state == NULL ? 0 : emit_state->current_vgpr_msb_mode;
  loom_low_move_sequence_options_t options = {
      .descriptor_set = context->allocation->target.descriptor_set,
      .temporary_locations = temporaries,
      .temporary_location_count = group->temporary_count,
      .emit_move =
          {
              .fn = loom_amdgpu_append_edge_move,
              .user_data = &move_state,
          },
  };
  IREE_RETURN_IF_ERROR(
      loom_low_move_sequence_emit(context->move_scratch, move_count, &options));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_vgpr_msb_mode(&move_state, saved_mode));
  if (move_state.emitted_count != 0) {
    *out_emitted = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_map_assignment(context, loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_copy_result(op));
  if (loom_amdgpu_assignments_match(source_assignment, result_assignment)) {
    return iree_ok_status();
  }
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly copy requires matching register ranges");
  }
  const uint32_t register_count = source_assignment->location_count;
  const uint32_t last_register_offset = register_count - 1;
  if (source_assignment->location_base > UINT32_MAX - last_register_offset ||
      result_assignment->location_base > UINT32_MAX - last_register_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly copy register range exceeds uint32_t");
  }

  if (source_assignment->descriptor_reg_class_id !=
      result_assignment->descriptor_reg_class_id) {
    iree_string_view_t source_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, source_assignment, &source_register_class));
    iree_string_view_t result_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, result_assignment, &result_register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, register_count, &moves));
  for (uint32_t i = 0; i < register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, i, &moves[i].source));
  }
  loom_low_move_location_t* temporaries = NULL;
  iree_host_size_t temporary_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_move_temporaries(
      context, &temporaries, &temporary_count));
  return loom_amdgpu_emit_move_sequence(context, emit_state, mnemonic,
                                        register_count, temporaries,
                                        temporary_count);
}

static iree_status_t loom_amdgpu_append_slice_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
      context->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_slice_result(op));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_slice_units(
      context->allocation, op, moves, move_count));
  loom_low_move_location_t* temporaries = NULL;
  iree_host_size_t temporary_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_move_temporaries(
      context, &temporaries, &temporary_count));
  return loom_amdgpu_emit_move_sequence(
      context, emit_state, mnemonic, move_count, temporaries, temporary_count);
}

static iree_status_t loom_amdgpu_append_concat_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
      context->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_concat_result(op));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_concat_units(
      context->allocation, op, moves, move_count));
  loom_low_move_location_t* temporaries = NULL;
  iree_host_size_t temporary_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_move_temporaries(
      context, &temporaries, &temporary_count));
  return loom_amdgpu_emit_move_sequence(
      context, emit_state, mnemonic, move_count, temporaries, temporary_count);
}

static iree_status_t loom_amdgpu_append_storage_address_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  loom_amdgpu_storage_layout_reference_t reference;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_resolve_reference(
      context->schedule->module, context->schedule->function_op,
      loom_low_storage_address_storage(op), &reference));
  const int64_t signed_offset = loom_low_storage_address_offset(op);
  if (signed_offset < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly low.storage.address offset must be non-negative");
  }
  const uint64_t offset = (uint64_t)signed_offset;
  if (offset >= reference.byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly low.storage.address offset "
                            "exceeds storage reference size");
  }
  uint64_t byte_offset = reference.reservation.byte_offset;
  if (byte_offset > UINT32_MAX ||
      reference.byte_offset > UINT32_MAX - byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly low.storage.address byte offset exceeds u32");
  }
  byte_offset += reference.byte_offset;
  if (offset > UINT32_MAX - byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly low.storage.address byte offset exceeds u32");
  }

  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(context, loom_low_storage_address_result(op));
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly low.storage.address result must "
                            "be one VGPR");
  }
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .emit_state = emit_state,
  };
  const uint8_t saved_mode =
      emit_state == NULL ? 0 : emit_state->current_vgpr_msb_mode;
  const uint32_t window = LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE;
  uint8_t mask = 0;
  uint8_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vgpr_msb_insert_requirement(
      LOOM_AMDGPU_VGPR_MSB_SLOT_DST, assignment->location_base / window, &mask,
      &value));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_vgpr_msb_requirement(&move_state, mask, value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_line_prefix(&move_state));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "v_mov_b32 "));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder, "v%" PRIu32, assignment->location_base % window));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder, "%" PRIu32, (uint32_t)(byte_offset + offset)));
  ++move_state.emitted_count;
  return loom_amdgpu_append_vgpr_msb_mode(&move_state, saved_mode);
}

static iree_status_t loom_amdgpu_append_matrix_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  uint16_t accumulator_operand_index = UINT16_MAX;
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == 0) {
      accumulator_operand_index = tied_results[i].operand_index;
      break;
    }
  }
  if (accumulator_operand_index == UINT16_MAX ||
      accumulator_operand_index >= op->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU matrix result must be tied to an accumulator operand");
  }
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_op_const_results(op)[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_amdgpu_map_assignment(
          context, loom_op_const_operands(op)[accumulator_operand_index]);
  if (!loom_amdgpu_assignments_match(result_assignment,
                                     accumulator_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU matrix result must share the accumulator physical register");
  }
  return loom_amdgpu_append_canonical_asm_form_packet(context);
}

typedef struct loom_amdgpu_assembly_packet_shape_t {
  // Diagnostic name used when the packet does not match this shape.
  const char* diagnostic_name;
  // Number of expected low results.
  uint16_t result_count;
  // Number of expected explicit packet operands.
  iree_host_size_t explicit_operand_count;
  // Number of expected descriptor immediates.
  uint16_t immediate_count;
} loom_amdgpu_assembly_packet_shape_t;

static iree_status_t loom_amdgpu_validate_assembly_packet_shape(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_assembly_packet_shape_t* shape) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->result_count == shape->result_count &&
      loom_amdgpu_explicit_packet_operand_count(context) ==
          shape->explicit_operand_count &&
      descriptor->immediate_count == shape->immediate_count) {
    return iree_ok_status();
  }
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly %s descriptor '%.*s' has unsupported shape",
      shape->diagnostic_name, (int)key.size, key.data);
}

typedef iree_status_t (*loom_amdgpu_asm_form_match_fn_t)(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form, iree_string_view_t canonical_mnemonic,
    bool* out_matched);

static iree_status_t loom_amdgpu_source0_immediate_asm_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form, iree_string_view_t canonical_mnemonic,
    bool* out_matched) {
  (void)context;
  (void)form;
  const bool plain_literal =
      iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src0_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src1_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src2_lit"));
  *out_matched =
      iree_string_view_ends_with(canonical_mnemonic,
                                 IREE_SV("_src0_16_low16")) ||
      iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_vop3_imm")) ||
      plain_literal;
  return iree_ok_status();
}

typedef struct loom_amdgpu_source_immediate_suffix_t {
  // Canonical low mnemonic suffix identifying the omitted source operand.
  const char* suffix;
  // Zero-based native source operand position occupied by the packet immediate.
  uint8_t immediate_source_index;
} loom_amdgpu_source_immediate_suffix_t;

static bool loom_amdgpu_match_source_immediate_asm_form(
    iree_string_view_t canonical_mnemonic,
    iree_string_view_t* out_base_mnemonic,
    uint8_t* out_immediate_source_index) {
  static const loom_amdgpu_source_immediate_suffix_t
      kSourceImmediateSuffixes[] = {
          {.suffix = "_src0_lit", .immediate_source_index = 0},
          {.suffix = "_src1_lit", .immediate_source_index = 1},
          {.suffix = "_src2_lit", .immediate_source_index = 2},
          {.suffix = "_src0_inline", .immediate_source_index = 0},
          {.suffix = "_src1_inline", .immediate_source_index = 1},
          {.suffix = "_src2_inline", .immediate_source_index = 2},
      };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kSourceImmediateSuffixes);
       ++i) {
    iree_string_view_t base_mnemonic = canonical_mnemonic;
    if (iree_string_view_consume_suffix(
            &base_mnemonic,
            iree_make_cstring_view(kSourceImmediateSuffixes[i].suffix))) {
      *out_base_mnemonic = base_mnemonic;
      *out_immediate_source_index =
          kSourceImmediateSuffixes[i].immediate_source_index;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_amdgpu_append_source_immediate_asm_form_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t base_mnemonic, uint8_t immediate_source_index) {
  const iree_host_size_t explicit_operand_count =
      loom_amdgpu_explicit_packet_operand_count(context);
  if (context->packet->descriptor->result_count != 1 ||
      explicit_operand_count == 0 ||
      immediate_source_index > explicit_operand_count ||
      context->packet->descriptor->immediate_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly source-immediate descriptor '%.*s' has unsupported "
        "shape",
        (int)key.size, key.data);
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, base_mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  iree_host_size_t operand_index = 0;
  const iree_host_size_t source_count = explicit_operand_count + 1;
  for (iree_host_size_t source_index = 0; source_index < source_count;
       ++source_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    if (source_index == immediate_source_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_packet_immediate_i64(context, 0));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, operand_index));
      ++operand_index;
    }
  }
  return iree_ok_status();
}

static const loom_amdgpu_assembly_packet_form_t kSource0ImmediatePacketForm = {
    .value_count = 3,
    .values =
        {
            LOOM_AMDGPU_ASM_VALUE_RESULT(0),
            LOOM_AMDGPU_ASM_VALUE_IMMEDIATE_I64(0),
            LOOM_AMDGPU_ASM_VALUE_OPERAND(0),
        },
};

static const loom_amdgpu_assembly_packet_shape_t kSource0ImmediatePacketShape =
    {
        .diagnostic_name = "source0-immediate",
        .result_count = 1,
        .explicit_operand_count = 1,
        .immediate_count = 1,
};

typedef struct loom_amdgpu_asm_form_packet_rule_t {
  // Predicate over the canonical asm form.
  loom_amdgpu_asm_form_match_fn_t match;
  // Required packet shape for this native form.
  const loom_amdgpu_assembly_packet_shape_t* shape;
  // Native assembly form appended when |match| succeeds.
  const loom_amdgpu_assembly_packet_form_t* form;
} loom_amdgpu_asm_form_packet_rule_t;

static iree_status_t loom_amdgpu_try_append_asm_form_rule_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form, iree_string_view_t canonical_mnemonic,
    bool* out_matched) {
  iree_string_view_t base_mnemonic = iree_string_view_empty();
  uint8_t immediate_source_index = 0;
  if (loom_amdgpu_match_source_immediate_asm_form(
          canonical_mnemonic, &base_mnemonic, &immediate_source_index)) {
    *out_matched = true;
    return loom_amdgpu_append_source_immediate_asm_form_packet(
        context, base_mnemonic, immediate_source_index);
  }

  static const loom_amdgpu_asm_form_packet_rule_t kRules[] = {
      {
          .match = loom_amdgpu_source0_immediate_asm_form,
          .shape = &kSource0ImmediatePacketShape,
          .form = &kSource0ImmediatePacketForm,
      },
  };
  *out_matched = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRules); ++i) {
    const loom_amdgpu_asm_form_packet_rule_t* rule = &kRules[i];
    bool rule_matched = false;
    IREE_RETURN_IF_ERROR(
        rule->match(context, form, canonical_mnemonic, &rule_matched));
    if (!rule_matched) {
      continue;
    }
    *out_matched = true;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_validate_assembly_packet_shape(context, rule->shape));
    return loom_amdgpu_append_assembly_packet_form(context, rule->form);
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fma_mix_source_part_selectors(
    iree_string_view_t source_part, uint8_t bit, uint8_t* op_sel,
    uint8_t* op_sel_hi) {
  if (iree_string_view_equal(source_part, IREE_SV("f32"))) {
    return true;
  }
  if (iree_string_view_equal(source_part, IREE_SV("f16lo"))) {
    *op_sel_hi |= bit;
    return true;
  }
  if (iree_string_view_equal(source_part, IREE_SV("f16hi"))) {
    *op_sel |= bit;
    *op_sel_hi |= bit;
    return true;
  }
  return false;
}

static bool loom_amdgpu_fma_mix_src2_literal_selectors(
    iree_string_view_t mnemonic, uint8_t* out_op_sel, uint8_t* out_op_sel_hi) {
  *out_op_sel = 0;
  *out_op_sel_hi = 0;
  if (!iree_string_view_consume_prefix(&mnemonic, IREE_SV("v_fma_mix_f32_")) ||
      !iree_string_view_consume_suffix(&mnemonic, IREE_SV("_src2_lit"))) {
    return false;
  }

  iree_string_view_t source_parts[3] = {0};
  iree_string_view_t remaining = mnemonic;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_parts); ++i) {
    iree_string_view_t part = iree_string_view_empty();
    if (i + 1 == IREE_ARRAYSIZE(source_parts)) {
      part = remaining;
      remaining = iree_string_view_empty();
    } else if (iree_string_view_split(remaining, '_', &part, &remaining) < 0) {
      return false;
    }
    source_parts[i] = part;
  }
  if (!iree_string_view_is_empty(remaining) ||
      !iree_string_view_equal(source_parts[2], IREE_SV("f32"))) {
    return false;
  }

  uint8_t op_sel = 0;
  uint8_t op_sel_hi = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_parts); ++i) {
    if (!loom_amdgpu_fma_mix_source_part_selectors(
            source_parts[i], (uint8_t)(1u << i), &op_sel, &op_sel_hi)) {
      return false;
    }
  }
  *out_op_sel = op_sel;
  *out_op_sel_hi = op_sel_hi;
  return true;
}

static iree_status_t loom_amdgpu_append_fma_mix_selector_modifier(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, uint8_t selectors) {
  if (selectors == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      context->builder, " %.*s:[%u,%u,%u]", (int)name.size, name.data,
      selectors & 1u ? 1u : 0u, selectors & 2u ? 1u : 0u,
      selectors & 4u ? 1u : 0u);
}

static iree_status_t loom_amdgpu_append_fma_mix_src2_literal_packet(
    const loom_native_assembly_packet_context_t* context, uint8_t op_sel,
    uint8_t op_sel_hi) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "v_fma_mix_f32 "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));

  const loom_low_immediate_t* imm32 = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_find_packet_immediate(context, IREE_SV("imm32"), &imm32));
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_immediate_i64(context, imm32, &value));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(context->builder, "%" PRId64, value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_fma_mix_selector_modifier(
      context, IREE_SV("op_sel"), op_sel));
  return loom_amdgpu_append_fma_mix_selector_modifier(
      context, IREE_SV("op_sel_hi"), op_sel_hi);
}

typedef iree_status_t (*loom_amdgpu_mnemonic_packet_rule_fn_t)(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic, bool* out_matched);

static iree_status_t loom_amdgpu_try_append_fma_mix_src2_literal_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic, bool* out_matched) {
  *out_matched = false;
  uint8_t op_sel = 0;
  uint8_t op_sel_hi = 0;
  if (!loom_amdgpu_fma_mix_src2_literal_selectors(mnemonic, &op_sel,
                                                  &op_sel_hi)) {
    return iree_ok_status();
  }
  *out_matched = true;
  return loom_amdgpu_append_fma_mix_src2_literal_packet(context, op_sel,
                                                        op_sel_hi);
}

static iree_status_t loom_amdgpu_try_append_mnemonic_rule_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic, bool* out_matched) {
  static const loom_amdgpu_mnemonic_packet_rule_fn_t kRules[] = {
      loom_amdgpu_try_append_fma_mix_src2_literal_packet,
  };
  *out_matched = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRules); ++i) {
    bool rule_matched = false;
    IREE_RETURN_IF_ERROR(kRules[i](context, mnemonic, &rule_matched));
    if (rule_matched) {
      *out_matched = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_vopd_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  return iree_string_builder_append_cstring(context->builder, " ");
}

static iree_string_view_t loom_amdgpu_vopd_component_assembly_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_info_t* info) {
  const uint32_t descriptor_set_ordinal =
      context->schedule->target.descriptor_set->descriptor_set_ordinal;
  if ((descriptor_set_ordinal == LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4 ||
       descriptor_set_ordinal ==
           LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X) &&
      info->rdna4_assembly_mnemonic.size != 0) {
    return info->rdna4_assembly_mnemonic;
  }
  return info->assembly_mnemonic;
}

static iree_status_t loom_amdgpu_append_vopd_tied_accumulate_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_vopd_component_info_t* info,
    iree_string_view_t mnemonic) {
  loom_native_assembly_packet_context_t component_context = *context;
  component_context.packet = packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(&component_context,
                                                  info->operands.src0_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_operand(&component_context,
                                    info->operands.vsrc1_index);
}

static iree_status_t loom_amdgpu_append_vopd_fmaak_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t mnemonic,
    uint32_t literal_u32) {
  loom_native_assembly_packet_context_t component_context = *context;
  component_context.packet = packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(&component_context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return iree_string_builder_append_format(context->builder, "0x%08" PRIx32,
                                           literal_u32);
}

static iree_status_t loom_amdgpu_append_vopd_fmamk_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t mnemonic,
    uint32_t literal_u32) {
  loom_native_assembly_packet_context_t component_context = *context;
  component_context.packet = packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder, "0x%08" PRIx32, literal_u32));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_operand(&component_context, 1);
}

static iree_status_t loom_amdgpu_append_vopd_binary_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t mnemonic) {
  loom_native_assembly_packet_context_t component_context = *context;
  component_context.packet = packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_operand(&component_context, 1);
}

static iree_status_t loom_amdgpu_append_vopd_mov_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t mnemonic) {
  loom_native_assembly_packet_context_t component_context = *context;
  component_context.packet = packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(&component_context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_packet_immediate_i64(&component_context, 0);
}

static iree_status_t loom_amdgpu_append_vopd_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_packet_view_t* packet, uint16_t vopd_op,
    uint32_t literal_u32) {
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_op(vopd_op);
  if (info == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU assembly VOPD component op %" PRIu16
                            " is not supported",
                            vopd_op);
  }
  const iree_string_view_t mnemonic =
      loom_amdgpu_vopd_component_assembly_mnemonic(context, info);
  switch (info->form) {
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE:
      return loom_amdgpu_append_vopd_tied_accumulate_component(context, packet,
                                                               info, mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL:
      return loom_amdgpu_append_vopd_fmaak_component(context, packet, mnemonic,
                                                     literal_u32);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL:
      return loom_amdgpu_append_vopd_fmamk_component(context, packet, mnemonic,
                                                     literal_u32);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR:
      return loom_amdgpu_append_vopd_binary_component(context, packet,
                                                      mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV:
      return loom_amdgpu_append_vopd_mov_component(context, packet, mnemonic);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "AMDGPU assembly VOPD component has unknown "
                              "form");
  }
}

static iree_status_t loom_amdgpu_append_vopd_pair_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_pair_t* pair) {
  loom_low_packet_view_t second = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_packet_view_at(context->schedule, context->allocation,
                              pair->second_packet_index, &second));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_component(
      context, context->packet, pair->op_x, pair->literal_u32));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " :: "));
  return loom_amdgpu_append_vopd_component(context, &second, pair->op_y,
                                           pair->literal_u32);
}

typedef iree_status_t (*loom_amdgpu_try_append_descriptor_dispatch_fn_t)(
    const loom_native_assembly_packet_context_t* context, bool* out_matched);

typedef struct loom_amdgpu_descriptor_packet_dispatch_rule_t {
  // Attempts one descriptor-packet assembly spelling and reports whether it
  // consumed the packet.
  loom_amdgpu_try_append_descriptor_dispatch_fn_t try_append;
} loom_amdgpu_descriptor_packet_dispatch_rule_t;

static iree_status_t loom_amdgpu_try_append_mnemonic_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, &mnemonic));
  return loom_amdgpu_try_append_mnemonic_rule_packet(context, mnemonic,
                                                     out_matched);
}

static iree_status_t loom_amdgpu_try_append_canonical_asm_form_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  const loom_low_asm_form_t* canonical_form = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_canonical_asm_form(context, &canonical_form));
  iree_string_view_t canonical_mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      canonical_form->mnemonic_string_offset, &canonical_mnemonic));
  return loom_amdgpu_try_append_asm_form_rule_packet(
      context, canonical_form, canonical_mnemonic, out_matched);
}

static iree_status_t loom_amdgpu_try_append_effect_route_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  loom_amdgpu_descriptor_packet_route_flags_t route_flags = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_descriptor_packet_route_flags(context, &route_flags));
  return loom_amdgpu_try_append_descriptor_packet_route(context, route_flags,
                                                        out_matched);
}

static iree_status_t loom_amdgpu_try_append_matrix_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  bool uses_matrix_resource = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_uses_resource_kind(
      context->schedule->target.descriptor_set, context->packet->descriptor,
      LOOM_LOW_RESOURCE_KIND_MATRIX, &uses_matrix_resource));
  if (!uses_matrix_resource) {
    return iree_ok_status();
  }
  *out_matched = true;
  return loom_amdgpu_append_matrix_packet(context);
}

static iree_status_t loom_amdgpu_append_canonical_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = true;
  return loom_amdgpu_append_canonical_asm_form_packet(context);
}

static iree_status_t loom_amdgpu_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  static const loom_amdgpu_descriptor_packet_dispatch_rule_t kRules[] = {
      {.try_append = loom_amdgpu_try_append_mnemonic_dispatch_packet},
      {.try_append = loom_amdgpu_try_append_canonical_asm_form_dispatch_packet},
      {.try_append = loom_amdgpu_try_append_effect_route_dispatch_packet},
      {.try_append = loom_amdgpu_try_append_matrix_dispatch_packet},
      {.try_append = loom_amdgpu_append_canonical_dispatch_packet},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRules); ++i) {
    bool matched = false;
    IREE_RETURN_IF_ERROR(kRules[i].try_append(context, &matched));
    if (matched) {
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "AMDGPU assembly descriptor packet dispatch reached no fallback");
}

static iree_status_t loom_amdgpu_update_vgpr_msb_mode_after_descriptor(
    loom_amdgpu_assembly_emit_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  if (state == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* set_vgpr_msb_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          context->schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
  if (set_vgpr_msb_descriptor == NULL ||
      context->packet->descriptor != set_vgpr_msb_descriptor) {
    return iree_ok_status();
  }
  int64_t mode = 0;
  IREE_RETURN_IF_ERROR(loom_native_assembly_read_i64_attr(
      context->schedule->module, loom_amdgpu_packet_attrs(context),
      IREE_SV("mode"), &mode));
  if (mode < 0 || mode > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly s_set_vgpr_msb mode immediate "
                            "%" PRId64 " is not a u16",
                            mode);
  }
  state->current_vgpr_msb_mode = (uint8_t)((uint16_t)mode & 0xFFu);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_stateful_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_packet(NULL, context));
  return loom_amdgpu_update_vgpr_msb_mode_after_descriptor(state, context);
}

static iree_status_t loom_amdgpu_append_vopd_or_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state == NULL || state->vopd_plan == NULL) {
    return loom_amdgpu_append_stateful_descriptor_packet(user_data, context);
  }
  const loom_amdgpu_vopd_packet_t* vopd_packet =
      loom_amdgpu_vopd_plan_packet_at(state->vopd_plan,
                                      context->packet->packet_index);
  if (vopd_packet == NULL) {
    return loom_amdgpu_append_stateful_descriptor_packet(user_data, context);
  }
  if (vopd_packet->pair_index >= state->vopd_plan->pair_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly VOPD packet references pair "
                            "%" PRIu32 " but plan has %" PRIhsz " pair(s)",
                            vopd_packet->pair_index,
                            state->vopd_plan->pair_count);
  }
  if (vopd_packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
    return iree_ok_status();
  }
  if (vopd_packet->role != LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST) {
    iree_string_view_t role =
        loom_amdgpu_vopd_packet_role_name(vopd_packet->role);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly VOPD packet has unsupported role "
                            "'%.*s'",
                            (int)role.size, role.data);
  }
  const loom_amdgpu_vopd_pair_t* pair =
      &state->vopd_plan->pairs[vopd_packet->pair_index];
  return loom_amdgpu_append_vopd_pair_packet(context, pair);
}

static iree_status_t loom_amdgpu_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  return iree_string_builder_append_cstring(context->builder, "s_endpgm");
}

static iree_status_t loom_amdgpu_append_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_block_t* dest = loom_low_br_dest(op);
  loom_value_slice_t args = loom_low_br_args(op);
  bool emitted_edge_copies = false;
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            context->allocation, context->packet->node->source_ordinal);
    if (!group) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly branch edge copies are missing from allocation");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_edge_copy_group(
        context, emit_state, group, &emitted_edge_copies));
  }
  const uint32_t current_block_index = context->packet->node->block_index;
  const uint32_t dest_block_index =
      loom_low_packet_block_index(context->schedule, dest);
  if (dest_block_index == current_block_index + 1) {
    return iree_ok_status();
  }
  if (emitted_edge_copies) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_branch "));
  return loom_native_assembly_append_block_label(context->schedule, dest,
                                                 context->builder);
}

static iree_status_t loom_amdgpu_verify_scc_condition_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t condition_value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(context, condition_value_id);
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly conditional branch condition must "
                            "be allocated to SCC");
  }
  if (assignment->location_base != 0 || assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly SCC condition must use the single "
                            "architectural SCC register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_cond_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_scc_condition_assignment(
      context, loom_low_cond_br_condition(op)));
  const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
  const uint32_t current_block_index = context->packet->node->block_index;
  const uint32_t true_block_index =
      loom_low_packet_block_index(context->schedule, true_dest);
  const uint32_t false_block_index =
      loom_low_packet_block_index(context->schedule, false_dest);
  if (true_dest == false_dest) {
    if (true_block_index == current_block_index + 1) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "s_branch "));
    return loom_native_assembly_append_block_label(context->schedule, true_dest,
                                                   context->builder);
  }
  if (true_block_index == current_block_index + 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(context->builder,
                                                            "s_cbranch_scc0 "));
    return loom_native_assembly_append_block_label(
        context->schedule, false_dest, context->builder);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_cbranch_scc1 "));
  IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
      context->schedule, true_dest, context->builder));
  if (false_block_index == current_block_index + 1) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  s_branch "));
  return loom_native_assembly_append_block_label(context->schedule, false_dest,
                                                 context->builder);
}

static iree_status_t loom_amdgpu_verify_assembly_target(
    const loom_low_schedule_table_t* schedule) {
  if (schedule->target.descriptor_set->target_stable_id !=
      LOOM_AMDGPU_TARGET_STABLE_ID) {
    iree_string_view_t target_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        schedule->target.descriptor_set,
        schedule->target.descriptor_set->target_key_string_offset,
        &target_key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

typedef struct loom_amdgpu_assembly_structural_packet_row_t {
  // Structural op handled by this row.
  loom_op_kind_t op_kind;
  // Formatter for packets with |op_kind|.
  loom_native_assembly_append_packet_fn_t fn;
} loom_amdgpu_assembly_structural_packet_row_t;

static const loom_amdgpu_assembly_structural_packet_row_t
    kLoomAmdgpuAssemblyStructuralPacketRows[] = {
        {LOOM_OP_LOW_COPY, loom_amdgpu_append_copy_packet},
        {LOOM_OP_LOW_SLICE, loom_amdgpu_append_slice_packet},
        {LOOM_OP_LOW_CONCAT, loom_amdgpu_append_concat_packet},
        {LOOM_OP_LOW_STORAGE_ADDRESS,
         loom_amdgpu_append_storage_address_packet},
        {LOOM_OP_LOW_RETURN, loom_amdgpu_append_return_packet},
        {LOOM_OP_LOW_BR, loom_amdgpu_append_branch_packet},
        {LOOM_OP_LOW_COND_BR, loom_amdgpu_append_cond_branch_packet},
};

static void loom_amdgpu_initialize_assembly_structural_packet_callbacks(
    loom_amdgpu_assembly_emit_state_t* emit_state,
    loom_native_assembly_structural_packet_callback_t* callbacks) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuAssemblyStructuralPacketRows); ++i) {
    callbacks[i] = (loom_native_assembly_structural_packet_callback_t){
        .op_kind = kLoomAmdgpuAssemblyStructuralPacketRows[i].op_kind,
        .append_packet =
            {
                .fn = kLoomAmdgpuAssemblyStructuralPacketRows[i].fn,
                .user_data = emit_state,
            },
    };
  }
}

static loom_native_assembly_format_options_t loom_amdgpu_assembly_options(
    const loom_native_assembly_structural_packet_callback_t*
        structural_packet_callbacks,
    loom_native_assembly_append_packet_callback_t append_before_packet,
    loom_native_assembly_append_packet_callback_t append_descriptor_packet) {
  return (loom_native_assembly_format_options_t){
      .append_before_packet = append_before_packet,
      .append_descriptor_packet = append_descriptor_packet,
      .structural_packet_callbacks = structural_packet_callbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(kLoomAmdgpuAssemblyStructuralPacketRows),
  };
}

iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  loom_amdgpu_assembly_emit_state_t emit_state = {0};
  loom_native_assembly_structural_packet_callback_t
      structural_packet_callbacks[IREE_ARRAYSIZE(
          kLoomAmdgpuAssemblyStructuralPacketRows)];
  loom_amdgpu_initialize_assembly_structural_packet_callbacks(
      &emit_state, structural_packet_callbacks);
  const loom_native_assembly_format_options_t options =
      loom_amdgpu_assembly_options(
          structural_packet_callbacks,
          (loom_native_assembly_append_packet_callback_t){0},
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_stateful_descriptor_packet,
              .user_data = &emit_state,
          });
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder, scratch_arena);
}

iree_status_t loom_amdgpu_emit_assembly_fragment_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_assembly_fragment_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  const loom_amdgpu_packet_plan_t* packet_plan =
      options ? options->packet_plan : NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_packet_plan_verify(schedule, allocation, packet_plan));
  const loom_amdgpu_wait_packet_plan_t* wait_packets =
      packet_plan ? &packet_plan->wait_packets : NULL;
  const loom_amdgpu_wait_state_plan_t* wait_states =
      packet_plan ? &packet_plan->wait_states : NULL;
  const loom_amdgpu_vopd_plan_t* vopd_plan =
      packet_plan ? &packet_plan->vopd_plan : NULL;

  loom_amdgpu_assembly_emit_state_t emit_state = {
      .wait_packets = wait_packets,
      .wait_states = wait_states,
      .vopd_plan = vopd_plan,
  };
  loom_native_assembly_structural_packet_callback_t
      structural_packet_callbacks[IREE_ARRAYSIZE(
          kLoomAmdgpuAssemblyStructuralPacketRows)];
  loom_amdgpu_initialize_assembly_structural_packet_callbacks(
      &emit_state, structural_packet_callbacks);
  const loom_native_assembly_format_options_t format_options =
      loom_amdgpu_assembly_options(
          structural_packet_callbacks,
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_waits_before_packet,
              .user_data = &emit_state,
          },
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_vopd_or_descriptor_packet,
              .user_data = &emit_state,
          });
  IREE_RETURN_IF_ERROR(loom_native_assembly_format_fragment(
      schedule, allocation, &format_options, builder, scratch_arena));
  if (wait_packets != NULL &&
      emit_state.next_wait_packet_index != wait_packets->packet_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly wait packet plan contains an unmatched insertion "
        "point");
  }
  if (wait_states != NULL &&
      emit_state.next_wait_state_index != wait_states->state_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly wait-state plan contains an unmatched insertion "
        "point");
  }
  return iree_ok_status();
}
