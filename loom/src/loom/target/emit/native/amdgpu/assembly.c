// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/register_class.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/emit/native/assembly.h"

typedef enum loom_amdgpu_native_asm_immediate_format_e {
  // Target-format ID for S_DELAY_ALU's packed SIMM16 dependency immediate.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DELAY_ALU = 1,
  // Target-format ID for RDNA4 scale-select immediate suffixes.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_SCALE_SEL = 2,
  // Target-format ID for DPP lane-control immediate syntax.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP_CTRL = 3,
  // Target-format ID for a four-bit DPP destination bank mask.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP_BANK_MASK = 4,
  // Target-format ID for an omitted-at-default named bit-list modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST = 5,
  // Target-format ID for an omitted-at-default named integer modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_I64 = 6,
  // Target-format ID for an omitted-at-default named presence modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG = 7,
  // Target-format ID for a GFX12 SCOPE symbolic modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE = 8,
  // Target-format ID for a GFX12 load TH symbolic modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL = 9,
  // Target-format ID for a required named integer modifier.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64 = 10,
  // Packed eight-lane DPP selector tuple.
  LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP8 = 11,
} loom_amdgpu_native_asm_immediate_format_t;

typedef enum loom_amdgpu_register_part_mask_e {
  LOOM_AMDGPU_REGISTER_PART_MASK_LOW16 = 0x1u,
  LOOM_AMDGPU_REGISTER_PART_MASK_HIGH16 = 0x2u,
} loom_amdgpu_register_part_mask_t;

typedef struct loom_amdgpu_assembly_packet_plan_state_t {
  // Address-state plan consumed in scheduled insertion order.
  const loom_amdgpu_address_state_plan_t* address_state;
  // Wait-packet plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Wait-state plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_state_plan_t* wait_states;
  // VOPD plan used to replace paired descriptor packets.
  const loom_amdgpu_vopd_plan_t* vopd_plan;
  // Next address-state transition to compare with the scheduled packet.
  iree_host_size_t next_address_state_index;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_wait_packet_index;
  // Next wait-state row to compare with the current scheduled packet.
  iree_host_size_t next_wait_state_index;
} loom_amdgpu_assembly_packet_plan_state_t;

typedef struct loom_amdgpu_assembly_traversal_state_t {
  // Current scheduled block, or LOOM_LOW_PACKET_INDEX_NONE before emission.
  uint32_t current_block_index;
  // Low byte of MODE's current VGPR-MSB selector state.
  uint8_t current_vgpr_msb_mode;
} loom_amdgpu_assembly_traversal_state_t;

typedef struct loom_amdgpu_assembly_branch_state_t {
  // Exact converged branch-island layout, or NULL when no islands were needed.
  const loom_amdgpu_branch_layout_t* layout;
  // Next original branch edge consumed from |layout|.
  iree_host_size_t next_edge_index;
  // Next island group consumed from |layout|.
  iree_host_size_t next_group_index;
} loom_amdgpu_assembly_branch_state_t;

typedef struct loom_amdgpu_assembly_emit_state_t {
  // Function-local storage layout shared by all storage-address packets.
  const loom_amdgpu_storage_layout_t* storage_layout;
  // Target packet plan and its emission-order consumption cursors.
  loom_amdgpu_assembly_packet_plan_state_t packet_plan;
  // Function-local branch placement and emission cursors.
  loom_amdgpu_assembly_branch_state_t branches;
  // Packet traversal and simulated architectural state.
  loom_amdgpu_assembly_traversal_state_t traversal;
} loom_amdgpu_assembly_emit_state_t;

static iree_string_view_t loom_amdgpu_descriptor_key(
    const loom_native_assembly_packet_context_t* context) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset);
}

static const loom_low_descriptor_view_t* loom_amdgpu_descriptor_view(
    const loom_native_assembly_packet_context_t* context) {
  return loom_low_descriptor_set_descriptor_view_at(
      context->schedule->target.descriptor_set,
      context->packet->descriptor_ordinal);
}

static iree_status_t loom_amdgpu_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  const iree_string_view_t mnemonic = loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset);
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

static bool loom_amdgpu_op_ties_result_to_operand(const loom_op_t* op,
                                                  uint16_t result_index,
                                                  uint16_t operand_index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == result_index &&
        tied_results[i].operand_index == operand_index) {
      return true;
    }
  }
  return false;
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

static iree_status_t loom_amdgpu_append_sgpr_range_units(
    const loom_native_assembly_packet_context_t* context,
    uint32_t base_register, uint32_t register_count) {
  if (loom_amdgpu_sgpr_location_range_is_ttmp(base_register, register_count)) {
    return loom_amdgpu_append_register_range_units(
        context, "ttmp", base_register - LOOM_AMDGPU_TTMP_SGPR_LOCATION_BASE,
        register_count);
  }
  return loom_amdgpu_append_register_range_units(context, "s", base_register,
                                                 register_count);
}

static iree_status_t loom_amdgpu_append_assignment(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_append_sgpr_range_units(
        context, assignment->location_base, assignment->location_count);
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
  if (loom_amdgpu_register_class_is_vcc(
          context->allocation->target.descriptor_set,
          assignment->descriptor_reg_class_id)) {
    if (assignment->location_base != 0 || assignment->location_count != 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly VCC assignment must name physical register 0");
    }
    return iree_string_builder_append_cstring(context->builder, "vcc");
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      context->allocation, assignment, &register_class));
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU assembly register class '%.*s' is unsupported",
      (int)register_class.size, register_class.data);
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
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(
          context->allocation, context->packet, descriptor_operand_index);
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      operand->address_map_kind != LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
    return loom_amdgpu_append_assignment(context, assignment);
  }
  return loom_amdgpu_append_register_range_units(
      context, "v", assignment->location_base % operand->addressable_unit_count,
      assignment->location_count);
}

static iree_status_t loom_amdgpu_append_descriptor_register_part_assignment(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_operand_index) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  const loom_low_operand_t* operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  IREE_ASSERT_NE(operand->register_part_id, LOOM_LOW_REGISTER_PART_NONE);
  IREE_ASSERT_LT(operand->register_part_id,
                 descriptor_set->register_part_count);
  const loom_low_register_part_t* register_part =
      &descriptor_set->register_parts[operand->register_part_id];
  IREE_ASSERT(register_part->reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
              register_part->reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR);

  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_assignment(
      context, descriptor_operand_index));
  switch (register_part->mask) {
    case LOOM_AMDGPU_REGISTER_PART_MASK_LOW16:
      return iree_string_builder_append_cstring(context->builder, ".l");
    case LOOM_AMDGPU_REGISTER_PART_MASK_HIGH16:
      return iree_string_builder_append_cstring(context->builder, ".h");
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU assembly register part");
      return iree_ok_status();
  }
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
    return loom_amdgpu_append_sgpr_range_units(context, location->location,
                                               /*register_count=*/1);
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
  if (loom_amdgpu_register_class_is_vcc(
          context->allocation->target.descriptor_set,
          location->descriptor_reg_class_id)) {
    if (location->location != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly VCC move location must name physical register 0");
    }
    return iree_string_builder_append_cstring(context->builder, "vcc");
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
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
  const iree_string_view_t name = loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset);
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

static iree_status_t loom_amdgpu_append_packet_symbol_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t field_name, const loom_named_attr_t* attr) {
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly requires attribute '%.*s'",
                            (int)field_name.size, field_name.data);
  }
  if (attr->value.kind != LOOM_ATTR_SYMBOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly attribute '%.*s' must be a "
                            "symbol reference",
                            (int)field_name.size, field_name.data);
  }
  const loom_symbol_ref_t symbol_ref = attr->value.symbol;
  const loom_module_t* module = context->schedule->module;
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly attribute '%.*s' references an invalid symbol",
        (int)field_name.size, field_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly attribute '%.*s' references an unnamed symbol",
        (int)field_name.size, field_name.data);
  }
  const iree_string_view_t name = module->strings.entries[symbol->name_id];
  return iree_string_builder_append_format(context->builder, "@%.*s",
                                           (int)name.size, name.data);
}

static iree_status_t loom_amdgpu_append_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_immediate_t* immediate) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const iree_string_view_t name = loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset);
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ORDINAL &&
      iree_all_bits_set(immediate->flags, LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC)) {
    const loom_named_attr_t* attr = loom_native_assembly_find_attr(
        context->schedule->module, loom_amdgpu_packet_attrs(context), name);
    if (attr != NULL && attr->value.kind == LOOM_ATTR_SYMBOL) {
      return loom_amdgpu_append_packet_symbol_attr(context, name, attr);
    }
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
  return iree_string_builder_append_format(context->builder, "%" PRId64, value);
}

static iree_status_t loom_amdgpu_read_packet_immediate_by_index_i64(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index, int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_ASSERT_LT(descriptor_immediate_index, descriptor->immediate_count);
  const uint32_t immediate_row =
      descriptor->immediate_start + descriptor_immediate_index;
  IREE_ASSERT_LT(immediate_row, descriptor_set->immediate_count);
  IREE_ASSERT(descriptor_set->immediates != NULL);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  return loom_amdgpu_read_packet_immediate_i64(context, immediate, out_value);
}

static iree_status_t loom_amdgpu_read_packet_immediate_by_name_i64(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t field_name, int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor->immediate_start + i;
    IREE_ASSERT_LT(immediate_index, descriptor_set->immediate_count);
    IREE_ASSERT(descriptor_set->immediates != NULL);
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    const iree_string_view_t name = loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset);
    if (iree_string_view_equal(name, field_name)) {
      return loom_amdgpu_read_packet_immediate_i64(context, immediate,
                                                   out_value);
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU assembly descriptor has no '%.*s' immediate",
                          (int)field_name.size, field_name.data);
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

static iree_status_t loom_amdgpu_append_delay_alu_dependency(
    const loom_native_assembly_packet_context_t* context, uint8_t value) {
  if (value == 0) {
    return iree_string_builder_append_cstring(context->builder, "NO_DEP");
  } else if (value <= 4) {
    return iree_string_builder_append_format(context->builder,
                                             "VALU_DEP_%" PRIu8, value);
  } else if (value >= 5 && value <= 7) {
    return iree_string_builder_append_format(
        context->builder, "TRANS32_DEP_%" PRIu8, (uint8_t)(value - 4));
  } else if (value >= 9 && value <= 11) {
    return iree_string_builder_append_format(
        context->builder, "SALU_CYCLE_%" PRIu8, (uint8_t)(value - 8));
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "AMDGPU s_delay_alu dependency code %" PRIu8 " is reserved", value);
}

static iree_status_t loom_amdgpu_append_delay_alu_skip(
    const loom_native_assembly_packet_context_t* context, uint8_t value) {
  static const char* const kDelayAluSkipNames[] = {
      "SAME", "NEXT", "SKIP_1", "SKIP_2", "SKIP_3", "SKIP_4",
  };
  if (value >= IREE_ARRAYSIZE(kDelayAluSkipNames)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU s_delay_alu skip code %" PRIu8 " is reserved", value);
  }
  return iree_string_builder_append_cstring(context->builder,
                                            kDelayAluSkipNames[value]);
}

static iree_status_t loom_amdgpu_append_delay_alu_immediate(
    const loom_native_assembly_packet_context_t* context, uint16_t value) {
  if ((value >> 11) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU s_delay_alu immediate value 0x%04" PRIx16
                            " sets reserved high bits",
                            value);
  }

  const uint8_t instid0 = (uint8_t)(value & 0x000F);
  const uint8_t instskip = (uint8_t)((value >> 4) & 0x0007);
  const uint8_t instid1 = (uint8_t)((value >> 7) & 0x000F);

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "instid0("));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_delay_alu_dependency(context, instid0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ")"));
  if (instskip != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, " | instskip("));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_delay_alu_skip(context, instskip));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, ")"));
  }
  if (instid1 != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, " | instid1("));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_delay_alu_dependency(context, instid1));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, ")"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_packet_immediate_delay_alu(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  if (value < 0 || value > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU s_delay_alu immediate value %" PRId64 " is not a u16", value);
  }
  return loom_amdgpu_append_delay_alu_immediate(context, (uint16_t)value);
}

static iree_status_t loom_amdgpu_append_packet_immediate_scale_sel(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  return iree_string_builder_append_format(context->builder,
                                           "scale_sel:%" PRId64, value);
}

static bool loom_amdgpu_native_asm_format_is_named_modifier(
    uint8_t target_format_id) {
  switch (target_format_id) {
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_I64:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_append_packet_immediate_named_modifier(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_native_asm_value_t* native_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_ASSERT_LT(native_value->index, descriptor->immediate_count);
  const uint32_t immediate_row =
      descriptor->immediate_start + native_value->index;
  IREE_ASSERT_LT(immediate_row, descriptor_set->immediate_count);
  IREE_ASSERT(descriptor_set->immediates != NULL);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];

  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
  if (native_value->target_format_id !=
          LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64 &&
      value == immediate->default_value) {
    return iree_ok_status();
  }

  const iree_string_view_t name = loom_native_assembly_descriptor_string(
      descriptor_set, native_value->literal_string_offset);
  switch (native_value->target_format_id) {
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST: {
      IREE_ASSERT(native_value->bit_width > 0 && native_value->bit_width < 64);
      IREE_ASSERT(value >= 0 &&
                  (uint64_t)value < (UINT64_C(1) << native_value->bit_width));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, " %.*s:[", (int)name.size, name.data));
      for (uint8_t bit = 0; bit < native_value->bit_width; ++bit) {
        if (bit > 0) {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(context->builder, ","));
        }
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            context->builder, "%u", (unsigned)(((uint64_t)value >> bit) & 1)));
      }
      return iree_string_builder_append_cstring(context->builder, "]");
    }
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_I64:
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64:
      return iree_string_builder_append_format(
          context->builder, " %.*s:%" PRId64, (int)name.size, name.data, value);
    case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG:
      IREE_ASSERT_EQ(immediate->default_value, 0);
      IREE_ASSERT_EQ(value, 1);
      return iree_string_builder_append_format(context->builder, " %.*s",
                                               (int)name.size, name.data);
    default:
      IREE_ASSERT_UNREACHABLE("not a named AMDGPU assembly modifier");
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_append_packet_immediate_gfx12_scope(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  static const char* const kScopeNames[] = {"SCOPE_CU", "SCOPE_SE", "SCOPE_DEV",
                                            "SCOPE_SYS"};
  int64_t scope = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &scope));
  if (scope == 0) {
    return iree_ok_status();
  }
  if (scope < 0 || (uint64_t)scope >= IREE_ARRAYSIZE(kScopeNames)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU GFX12 SCOPE value %" PRId64 " is not in [0, 3]", scope);
  }
  return iree_string_builder_append_format(context->builder, " scope:%s",
                                           kScopeNames[scope]);
}

static iree_status_t loom_amdgpu_append_packet_immediate_gfx12_load_temporal(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t temporal = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &temporal));
  if (temporal == 0) {
    return iree_ok_status();
  }

  int64_t scope = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_name_i64(
      context, IREE_SV("scope"), &scope));
  if (scope < 0 || scope > 3) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU GFX12 SCOPE value %" PRId64 " is not in [0, 3]", scope);
  }

  const char* name = NULL;
  switch (temporal) {
    case 1:
      name = "TH_LOAD_NT";
      break;
    case 2:
      name = "TH_LOAD_HT";
      break;
    case 3:
      name = scope == 3 ? "TH_LOAD_BYPASS" : "TH_LOAD_LU";
      break;
    case 4:
      name = "TH_LOAD_NT_RT";
      break;
    case 5:
      name = "TH_LOAD_RT_NT";
      break;
    case 6:
      name = "TH_LOAD_NT_HT";
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU GFX12 load TH value %" PRId64 " is reserved", temporal);
  }
  return iree_string_builder_append_format(context->builder, " th:%s", name);
}

static iree_status_t loom_amdgpu_append_dpp_control(
    const loom_native_assembly_packet_context_t* context, uint16_t value) {
  loom_amdgpu_dpp_control_decoding_t decoding = {0};
  if (!loom_amdgpu_dpp_control_decode(value, &decoding)) {
    IREE_ASSERT_UNREACHABLE("generated AMDGPU DPP control immediate");
    IREE_BUILTIN_UNREACHABLE();
  }
  switch (decoding.syntax) {
    case LOOM_AMDGPU_DPP_CONTROL_SYNTAX_QUAD_PERM:
      return iree_string_builder_append_format(
          context->builder, "quad_perm:[%u,%u,%u,%u]",
          (unsigned)(decoding.selector & 0x3u),
          (unsigned)((decoding.selector >> 2) & 0x3u),
          (unsigned)((decoding.selector >> 4) & 0x3u),
          (unsigned)((decoding.selector >> 6) & 0x3u));
    case LOOM_AMDGPU_DPP_CONTROL_SYNTAX_INDEXED:
      return iree_string_builder_append_format(
          context->builder, "%.*s%u", (int)decoding.text.size,
          decoding.text.data, (unsigned)decoding.selector);
    case LOOM_AMDGPU_DPP_CONTROL_SYNTAX_FIXED:
      return iree_string_builder_append_string(context->builder, decoding.text);
    default:
      IREE_ASSERT_UNREACHABLE("invalid decoded AMDGPU DPP control syntax");
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_append_packet_immediate_dpp_control(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  IREE_ASSERT(value >= 0 && value <= UINT16_MAX);
  return loom_amdgpu_append_dpp_control(context, (uint16_t)value);
}

static iree_status_t loom_amdgpu_append_packet_immediate_dpp8(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  IREE_ASSERT(value >= 0 && value <= 0xFFFFFF);
  return iree_string_builder_append_format(
      context->builder, "dpp8:[%u,%u,%u,%u,%u,%u,%u,%u]",
      (unsigned)((value >> 0) & 0x7), (unsigned)((value >> 3) & 0x7),
      (unsigned)((value >> 6) & 0x7), (unsigned)((value >> 9) & 0x7),
      (unsigned)((value >> 12) & 0x7), (unsigned)((value >> 15) & 0x7),
      (unsigned)((value >> 18) & 0x7), (unsigned)((value >> 21) & 0x7));
}

static iree_status_t loom_amdgpu_append_packet_immediate_dpp_bank_mask(
    const loom_native_assembly_packet_context_t* context,
    uint16_t descriptor_immediate_index) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_index_i64(
      context, descriptor_immediate_index, &value));
  IREE_ASSERT(value >= 0 && value <= 0xF);
  return iree_string_builder_append_format(context->builder, "bank_mask:0x%x",
                                           (unsigned)value);
}

static iree_status_t loom_amdgpu_find_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t field_name, const loom_low_immediate_t** out_immediate) {
  *out_immediate = NULL;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_ASSERT_LE(descriptor->immediate_start, descriptor_set->immediate_count);
  IREE_ASSERT_LE(descriptor->immediate_count,
                 descriptor_set->immediate_count - descriptor->immediate_start);
  IREE_ASSERT(descriptor->immediate_count == 0 ||
              descriptor_set->immediates != NULL);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    const iree_string_view_t name = loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset);
    if (iree_string_view_equal(name, field_name)) {
      *out_immediate = immediate;
      return iree_ok_status();
    }
  }
  const iree_string_view_t descriptor_key = loom_amdgpu_descriptor_key(context);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU assembly descriptor '%.*s' has no immediate "
                          "attribute '%.*s'",
                          (int)descriptor_key.size, descriptor_key.data,
                          (int)field_name.size, field_name.data);
}

static bool loom_amdgpu_descriptor_has_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind) {
  if (descriptor->effect_count == 0) {
    return false;
  }
  IREE_ASSERT(descriptor_set->effects != NULL);
  IREE_ASSERT_LE(descriptor->effect_start, descriptor_set->effect_count);
  IREE_ASSERT_LE(descriptor->effect_count,
                 descriptor_set->effect_count - descriptor->effect_start);
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_descriptor_has_memory_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    loom_low_memory_space_t memory_space) {
  if (descriptor->effect_count == 0) {
    return false;
  }
  IREE_ASSERT(descriptor_set->effects != NULL);
  IREE_ASSERT_LE(descriptor->effect_start, descriptor_set->effect_count);
  IREE_ASSERT_LE(descriptor->effect_count,
                 descriptor_set->effect_count - descriptor->effect_start);
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind && effect->memory_space == memory_space) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_descriptor_is_global_to_lds(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const bool has_global_read = loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ,
      LOOM_LOW_MEMORY_SPACE_GLOBAL);
  const bool has_workgroup_write = loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  return has_global_read && has_workgroup_write;
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
  const loom_low_descriptor_view_t* descriptor_view =
      loom_amdgpu_descriptor_view(context);
  if (descriptor_view->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    const iree_string_view_t key = loom_amdgpu_descriptor_key(context);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly descriptor '%.*s' has no canonical asm form",
        (int)key.size, key.data);
  }
  *out_form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor_view->canonical_asm_form_ordinal);
  if (*out_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly canonical asm form is out of "
                            "range");
  }
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_asm_form_native_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t* form) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  loom_bstring_table_offset_t string_offset = form->mnemonic_string_offset;
  if (form->native_assembly_mnemonic_string_offset !=
      LOOM_LOW_STRING_OFFSET_NONE) {
    string_offset = form->native_assembly_mnemonic_string_offset;
  }
  return loom_native_assembly_descriptor_string(descriptor_set, string_offset);
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
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU assembly value kind %u is unsupported",
                              (unsigned)value->kind);
  }
}

static iree_status_t loom_amdgpu_append_assembly_packet_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_assembly_packet_form_t* form) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  bool in_list = false;
  IREE_ASSERT_LE(form->value_count, IREE_ARRAYSIZE(form->values));
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
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  const uint16_t operand_index = descriptor_operand->source_value_index;
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
    IREE_ASSERT_LT(asm_operand_index, descriptor_set->asm_operand_index_count);
    IREE_ASSERT(descriptor_set->asm_operand_indices != NULL);
    const uint16_t descriptor_operand_index =
        descriptor_set->asm_operand_indices[asm_operand_index];
    IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
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
      const iree_string_view_t literal = loom_native_assembly_descriptor_string(
          descriptor_set, value->literal_string_offset);
      return iree_string_builder_append_string(context->builder, literal);
    }
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_MODIFIER_LITERAL: {
      const iree_string_view_t literal = loom_native_assembly_descriptor_string(
          descriptor_set, value->literal_string_offset);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " "));
      return iree_string_builder_append_string(context->builder, literal);
    }
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT:
      return loom_amdgpu_append_asm_form_value(
          context, descriptor, value->index, /*is_result=*/true);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_OPERAND:
      return loom_amdgpu_append_asm_form_value(
          context, descriptor, value->index, /*is_result=*/false);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_REGISTER_PART:
      return loom_amdgpu_append_descriptor_register_part_assignment(
          context, value->index);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64:
      return loom_amdgpu_append_packet_immediate_i64(context, value->index);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX:
      return loom_amdgpu_append_packet_immediate_unsigned_hex(
          context, value->index, value->bit_width);
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_TARGET_FORMAT:
      switch (value->target_format_id) {
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DELAY_ALU:
          return loom_amdgpu_append_packet_immediate_delay_alu(context,
                                                               value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_SCALE_SEL:
          return loom_amdgpu_append_packet_immediate_scale_sel(context,
                                                               value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP_CTRL:
          return loom_amdgpu_append_packet_immediate_dpp_control(context,
                                                                 value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP8:
          return loom_amdgpu_append_packet_immediate_dpp8(context,
                                                          value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DPP_BANK_MASK:
          return loom_amdgpu_append_packet_immediate_dpp_bank_mask(
              context, value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST:
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_I64:
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64:
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG:
          return loom_amdgpu_append_packet_immediate_named_modifier(context,
                                                                    value);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE:
          return loom_amdgpu_append_packet_immediate_gfx12_scope(context,
                                                                 value->index);
        case LOOM_AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL:
          return loom_amdgpu_append_packet_immediate_gfx12_load_temporal(
              context, value->index);
        default:
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AMDGPU native asm target immediate format id %u is unsupported",
              (unsigned)value->target_format_id);
      }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
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
    IREE_ASSERT_LT(native_value_index, descriptor_set->native_asm_value_count);
    IREE_ASSERT(descriptor_set->native_asm_values != NULL);
    const loom_low_native_asm_value_t* value =
        &descriptor_set->native_asm_values[native_value_index];
    const bool is_named_modifier =
        value->kind == LOOM_LOW_NATIVE_ASM_VALUE_KIND_MODIFIER_LITERAL ||
        (value->kind ==
             LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_TARGET_FORMAT &&
         loom_amdgpu_native_asm_format_is_named_modifier(
             value->target_format_id));
    if (!is_named_modifier) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_append_asm_form_separator(context, in_list));
    }
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
    IREE_ASSERT_LT(asm_immediate_index, descriptor_set->asm_immediate_count);
    IREE_ASSERT(descriptor_set->asm_immediates != NULL);
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_immediate_index];
    IREE_ASSERT_LT(asm_immediate->immediate_index, descriptor->immediate_count);
    IREE_ASSERT(descriptor_set->immediates != NULL);
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    asm_immediate->immediate_index];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      const iree_string_view_t spelling =
          loom_native_assembly_descriptor_string(
              descriptor_set, asm_immediate->name_string_offset);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%.*s(", (int)spelling.size, spelling.data));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_append_packet_immediate(context, immediate));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, ")"));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_append_packet_immediate(context, immediate));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_canonical_asm_form_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
  const iree_string_view_t mnemonic =
      loom_amdgpu_asm_form_native_mnemonic(context, form);
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
  IREE_ASSERT_LE(descriptor->immediate_start, descriptor_set->immediate_count);
  IREE_ASSERT_LE(descriptor->immediate_count,
                 descriptor_set->immediate_count - descriptor->immediate_start);
  IREE_ASSERT(descriptor_set->immediates != NULL);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    const iree_string_view_t name = loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset);
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s:%" PRId64, (int)name.size, name.data, value));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_native_asm_form_owns_immediate_syntax(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* form) {
  for (uint16_t i = 0; i < form->native_assembly_value_count; ++i) {
    const uint32_t value_index = form->native_assembly_value_start + i;
    IREE_ASSERT_LT(value_index, descriptor_set->native_asm_value_count);
    IREE_ASSERT(descriptor_set->native_asm_values != NULL);
    switch (descriptor_set->native_asm_values[value_index].kind) {
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64:
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX:
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_TARGET_FORMAT:
        return true;
      default:
        break;
    }
  }
  return false;
}

static iree_status_t loom_amdgpu_try_append_native_memory_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_view_t* descriptor_view =
      loom_amdgpu_descriptor_view(context);
  if (descriptor_view->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
  if (form->native_assembly_value_count == 0) {
    return iree_ok_status();
  }
  *out_matched = true;
  const iree_string_view_t mnemonic =
      loom_amdgpu_asm_form_native_mnemonic(context, form);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  bool in_list = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_native_asm_form_values(context, form, &in_list));
  // A native value list containing immediates owns their complete spelling.
  // Forms that only reorder operands retain the generic memory suffixes.
  if (loom_amdgpu_native_asm_form_owns_immediate_syntax(descriptor_set, form)) {
    return iree_ok_status();
  }
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
  const loom_low_descriptor_view_t* descriptor_view =
      loom_amdgpu_descriptor_view(context);
  if (descriptor_view->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_value_list(context));
  } else {
    const loom_low_asm_form_t* form = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
    const iree_string_view_t mnemonic =
        loom_amdgpu_asm_form_native_mnemonic(context, form);
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
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_native_memory_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_name_i64(
      context, IREE_SV("vmcnt"), &vmcnt));
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_packet_immediate_by_name_i64(
      context, IREE_SV("lgkmcnt"), &lgkmcnt));
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
  // Descriptor's canonical form completely defines its native spelling.
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_NATIVE_FORM = 1u << 9,
} loom_amdgpu_descriptor_packet_route_flag_bits_t;
typedef uint16_t loom_amdgpu_descriptor_packet_route_flags_t;

typedef enum loom_amdgpu_descriptor_packet_route_kind_e {
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD_LDS = 0,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD_LDS,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_BUFFER_ATOMIC,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_UNSUPPORTED_READ_WRITE,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_STORE,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_LOAD,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_STORE,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_STORE,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_WAITCNT,
  LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_CANONICAL,
} loom_amdgpu_descriptor_packet_route_kind_t;

typedef struct loom_amdgpu_descriptor_packet_route_t {
  // Fact bits required for this route to match.
  loom_amdgpu_descriptor_packet_route_flags_t required_flags;
  // Fact bits that prevent this route from matching.
  loom_amdgpu_descriptor_packet_route_flags_t forbidden_flags;
  // Assembly route to append when the row matches.
  loom_amdgpu_descriptor_packet_route_kind_t route_kind;
} loom_amdgpu_descriptor_packet_route_t;

static bool loom_amdgpu_descriptor_packet_route_matches(
    const loom_amdgpu_descriptor_packet_route_t* route,
    loom_amdgpu_descriptor_packet_route_flags_t flags) {
  return iree_all_bits_set(flags, route->required_flags) &&
         !iree_any_bit_set(flags, route->forbidden_flags);
}

static loom_amdgpu_descriptor_packet_route_flags_t
loom_amdgpu_descriptor_packet_route_flags(
    const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_descriptor_packet_route_flags_t flags = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_view_t* descriptor_view =
      loom_amdgpu_descriptor_view(context);

  const bool has_read_effect = loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ);
  if (has_read_effect) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT;
  }
  const bool has_write_effect = loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE);
  if (has_write_effect) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT;
  }
  const bool has_counter_effect = loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_COUNTER);
  if (has_counter_effect) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_COUNTER_EFFECT;
  }
  if (has_read_effect && has_write_effect &&
      loom_amdgpu_descriptor_is_global_to_lds(descriptor_set, descriptor)) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS;
  }
  const loom_amdgpu_encoding_format_flags_t encoding_flags =
      loom_amdgpu_encoding_format_flags(descriptor->encoding_format_id);
  if (iree_any_bit_set(encoding_flags,
                       LOOM_AMDGPU_ENCODING_FORMAT_FLAG_BUFFER_ADDRESS)) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT;
  }
  if (iree_any_bit_set(
          encoding_flags,
          LOOM_AMDGPU_ENCODING_FORMAT_FLAG_GLOBAL_POINTER_ADDRESS)) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT;
  }
  if (iree_any_bit_set(encoding_flags,
                       LOOM_AMDGPU_ENCODING_FORMAT_FLAG_SCRATCH_ADDRESS)) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT;
  }
  if (iree_any_bit_set(encoding_flags,
                       LOOM_AMDGPU_ENCODING_FORMAT_FLAG_DATA_SHARE_ADDRESS)) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT;
  }
  if (descriptor->immediate_count == 2) {
    flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_TWO_IMMEDIATES;
  }
  if (descriptor_view->canonical_asm_form_ordinal !=
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_ASSERT_LT(descriptor_view->canonical_asm_form_ordinal,
                   descriptor_set->asm_form_count);
    IREE_ASSERT(descriptor_set->asm_forms != NULL);
    const loom_low_asm_form_t* form =
        &descriptor_set->asm_forms[descriptor_view->canonical_asm_form_ordinal];
    if (form->native_assembly_value_count > 0) {
      flags |= LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_NATIVE_FORM;
    }
  }
  return flags;
}

static iree_status_t loom_amdgpu_append_unsupported_read_write_packet(
    const loom_native_assembly_packet_context_t* context) {
  const iree_string_view_t key = loom_amdgpu_descriptor_key(context);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU assembly descriptor '%.*s' has both read and write effects",
      (int)key.size, key.data);
}

static iree_status_t loom_amdgpu_append_descriptor_packet_route(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_descriptor_packet_route_kind_t route_kind) {
  switch (route_kind) {
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD_LDS:
      return loom_amdgpu_append_global_load_lds_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD_LDS:
      return loom_amdgpu_append_mubuf_load_lds_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_BUFFER_ATOMIC:
      return loom_amdgpu_append_buffer_atomic_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY:
      return loom_amdgpu_append_memory_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_UNSUPPORTED_READ_WRITE:
      return loom_amdgpu_append_unsupported_read_write_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD:
      return loom_amdgpu_append_mubuf_load_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_STORE:
      return loom_amdgpu_append_mubuf_store_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_LOAD:
      return loom_amdgpu_append_scratch_load_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_STORE:
      return loom_amdgpu_append_scratch_store_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD:
      return loom_amdgpu_append_global_load_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_STORE:
      return loom_amdgpu_append_global_store_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_WAITCNT:
      return loom_amdgpu_append_waitcnt_packet(context);
    case LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_CANONICAL:
      return loom_amdgpu_append_canonical_asm_form_packet(context);
    default:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU assembly descriptor route table must use a known route kind");
      IREE_BUILTIN_UNREACHABLE();
  }
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
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD_LDS,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_TO_LDS |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD_LDS,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_BUFFER_ATOMIC,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_NATIVE_FORM,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_CANONICAL,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .route_kind =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_UNSUPPORTED_READ_WRITE,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_LOAD,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_BUFFER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MUBUF_STORE,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_LOAD,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_SCRATCH_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_SCRATCH_STORE,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_LOAD,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_GLOBAL_POINTER_FORMAT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_GLOBAL_STORE,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_DATA_SHARE_FORMAT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_WRITE_EFFECT,
          .forbidden_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_READ_EFFECT,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_MEMORY,
      },
      {
          .required_flags =
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_COUNTER_EFFECT |
              LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_FLAG_TWO_IMMEDIATES,
          .route_kind = LOOM_AMDGPU_DESCRIPTOR_PACKET_ROUTE_WAITCNT,
      },
  };
  *out_matched = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRoutes); ++i) {
    const loom_amdgpu_descriptor_packet_route_t* route = &kRoutes[i];
    if (!loom_amdgpu_descriptor_packet_route_matches(route, flags)) {
      continue;
    }
    *out_matched = true;
    return loom_amdgpu_append_descriptor_packet_route(context,
                                                      route->route_kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_materialized_wait_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const iree_string_view_t mnemonic = loom_native_assembly_descriptor_string(
      descriptor_set, wait_packet->descriptor->mnemonic_string_offset);

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

static iree_status_t loom_amdgpu_append_branch_target_label(
    const loom_amdgpu_assembly_emit_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_branch_target_t target) {
  switch (target.kind) {
    case LOOM_AMDGPU_BRANCH_TARGET_BLOCK: {
      IREE_ASSERT_LT(target.index, context->schedule->block_count);
      return loom_native_assembly_append_block_label(
          context->schedule, context->schedule->blocks[target.index].block,
          context->builder);
    }
    case LOOM_AMDGPU_BRANCH_TARGET_ISLAND:
      IREE_ASSERT(state->branches.layout != NULL);
      IREE_ASSERT_LT(target.index, state->branches.layout->island_count);
      return iree_string_builder_append_format(
          context->builder, ".Lbranch_island%" PRIu32, target.index);
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU branch target kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_append_branch_groups_before_packet(
    loom_amdgpu_assembly_emit_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  if (state->branches.layout == NULL) return iree_ok_status();
  while (state->branches.next_group_index <
         state->branches.layout->group_count) {
    const iree_host_size_t group_index = state->branches.next_group_index;
    const loom_amdgpu_branch_layout_group_t* group =
        &state->branches.layout->groups[group_index];
    if (group->packet_index != context->packet->packet_index) break;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, "  s_branch .Lbranch_island_group_end%" PRIhsz "\n",
        group_index));
    for (uint32_t i = 0; i < group->island_count; ++i) {
      const uint32_t island_index = group->island_start + i;
      IREE_ASSERT_LT(island_index, state->branches.layout->island_count);
      const loom_amdgpu_branch_layout_island_t* island =
          &state->branches.layout->islands[island_index];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, ".Lbranch_island%" PRIu32 ":\n  s_branch ",
          island_index));
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_branch_target_label(
          state, context, island->target));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, "\n"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, ".Lbranch_island_group_end%" PRIhsz ":\n",
        group_index));
    ++state->branches.next_group_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const uint32_t block_index = context->packet->node->block_index;
  if (state->traversal.current_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    state->traversal.current_block_index = block_index;
    return iree_ok_status();
  }
  if (state->traversal.current_block_index == block_index) {
    return iree_ok_status();
  }
  if (state->packet_plan.address_state == NULL) {
    if (state->traversal.current_vgpr_msb_mode != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly left VGPR-MSB address state active at the end of "
          "block %" PRIu32,
          state->traversal.current_block_index);
    }
  } else {
    IREE_ASSERT_EQ(state->traversal.current_vgpr_msb_mode, 0);
  }
  state->traversal.current_block_index = block_index;
  return iree_ok_status();
}

static bool loom_amdgpu_address_state_matches_packet(
    const loom_amdgpu_address_state_transition_t* transition,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return transition->block_index == node->block_index &&
         transition->scheduled_ordinal == node->scheduled_ordinal &&
         transition->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_append_address_state_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state->packet_plan.address_state == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_address_state_index <
         state->packet_plan.address_state->transition_count) {
    const loom_amdgpu_address_state_transition_t* transition =
        &state->packet_plan.address_state
             ->transitions[state->packet_plan.next_address_state_index];
    if (!loom_amdgpu_address_state_matches_packet(transition,
                                                  context->packet)) {
      return iree_ok_status();
    }
    const uint8_t new_mode = (uint8_t)(transition->mode_immediate & 0xFFu);
    const loom_low_descriptor_t* descriptor =
        loom_amdgpu_descriptor_ref_descriptor(
            context->schedule->target.descriptor_set,
            LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
    IREE_ASSERT(descriptor != NULL);
    const iree_string_view_t mnemonic = loom_native_assembly_descriptor_string(
        context->schedule->target.descriptor_set,
        descriptor->mnemonic_string_offset);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, "  %.*s %" PRIu16 "\n", (int)mnemonic.size,
        mnemonic.data, transition->mode_immediate));
    state->traversal.current_vgpr_msb_mode = new_mode;
    ++state->packet_plan.next_address_state_index;
  }
  return iree_ok_status();
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
  if (state->packet_plan.wait_packets == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_wait_packet_index <
         state->packet_plan.wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->packet_plan.wait_packets
             ->packets[state->packet_plan.next_wait_packet_index];
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_materialized_wait_packet(
        context, wait_packet, state->packet_plan.wait_packets));
    ++state->packet_plan.next_wait_packet_index;
  }
  return iree_ok_status();
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
    case LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          context->builder, "  s_delay_alu "));
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_delay_alu_immediate(
          context, wait_state->delay_alu_immediate));
      return iree_string_builder_append_cstring(context->builder, "\n");
    }
    case LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP:
      for (uint16_t i = 0; i < wait_state->cycle_count; ++i) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(context->builder, "  v_nop\n"));
      }
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN:
    default: {
      IREE_ASSERT_UNREACHABLE(
          "verified AMDGPU wait-state plans must use known actions");
      IREE_BUILTIN_UNREACHABLE();
    }
  }
}

static iree_status_t loom_amdgpu_append_wait_states_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state->packet_plan.wait_states == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_wait_state_index <
         state->packet_plan.wait_states->state_count) {
    const loom_amdgpu_wait_state_t* wait_state =
        &state->packet_plan.wait_states
             ->states[state->packet_plan.next_wait_state_index];
    if (!loom_amdgpu_wait_state_matches_packet(wait_state, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_wait_state_action(context, wait_state));
    ++state->packet_plan.next_wait_state_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_insertions_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_packet(user_data, context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_branch_groups_before_packet(
      (loom_amdgpu_assembly_emit_state_t*)user_data, context));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_address_state_before_packet(user_data, context));
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
      IREE_STATUS_FAILED_PRECONDITION,
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

static void loom_amdgpu_vgpr_msb_insert_requirement(
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank, uint8_t* mask,
    uint8_t* value) {
  IREE_ASSERT_LT(bank, LOOM_AMDGPU_VGPR_MSB_BANK_COUNT);
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  *mask |= slot_mask;
  *value = (uint8_t)((*value & ~slot_mask) | slot_value);
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
  if (state->emit_state->traversal.current_vgpr_msb_mode == new_mode) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          state->context->schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
  IREE_ASSERT(descriptor != NULL);
  const iree_string_view_t mnemonic = loom_native_assembly_descriptor_string(
      state->context->schedule->target.descriptor_set,
      descriptor->mnemonic_string_offset);
  const uint16_t immediate =
      (uint16_t)(((uint16_t)state->emit_state->traversal.current_vgpr_msb_mode
                  << 8) |
                 new_mode);
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_line_prefix(state));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      state->context->builder, "%.*s %" PRIu16, (int)mnemonic.size,
      mnemonic.data, immediate));
  ++state->emitted_count;
  state->emit_state->traversal.current_vgpr_msb_mode = new_mode;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_vgpr_msb_requirement(
    loom_amdgpu_assembly_move_state_t* state, uint8_t mask, uint8_t value) {
  const uint8_t current_mode =
      state->emit_state == NULL
          ? 0
          : state->emit_state->traversal.current_vgpr_msb_mode;
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
      loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_DST,
                                              destination_bank, &mask, &value);
    }
    if (source->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0,
                                              source_bank, &mask, &value);
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

static iree_status_t loom_amdgpu_emit_move_range(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_assembly_emit_state_t* emit_state,
    loom_low_move_range_t move_range, bool* out_emitted) {
  if (out_emitted != NULL) {
    *out_emitted = false;
  }
  if (move_range.count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .emit_state = emit_state,
  };
  const uint8_t saved_mode =
      emit_state == NULL ? 0 : emit_state->traversal.current_vgpr_msb_mode;
  for (iree_host_size_t i = 0; i < move_range.count; ++i) {
    const loom_low_move_t* move =
        &context->allocation->moves[move_range.start + i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
        move->destination.descriptor_reg_class_id, &move_state.mnemonic));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_move(
        &move_state, &move->destination, &move->source));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_vgpr_msb_mode(&move_state, saved_mode));
  if (out_emitted != NULL) {
    *out_emitted = move_state.emitted_count != 0;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_edge_copy_group(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_assembly_emit_state_t* emit_state,
    const loom_low_allocation_edge_copy_group_t* group, bool* out_emitted) {
  return loom_amdgpu_emit_move_range(context, emit_state,
                                     group->move_group.moves, out_emitted);
}

static iree_status_t loom_amdgpu_append_packet_moves(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          context->allocation, context->packet->node->source_ordinal);
  return loom_amdgpu_emit_move_range(
      context, emit_state,
      group == NULL ? (loom_low_move_range_t){0} : group->move_group.moves,
      /*out_emitted=*/NULL);
}

static iree_status_t loom_amdgpu_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  return loom_amdgpu_append_packet_moves(user_data, context);
}

static iree_status_t loom_amdgpu_append_slice_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  return loom_amdgpu_append_packet_moves(user_data, context);
}

static iree_status_t loom_amdgpu_append_concat_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  return loom_amdgpu_append_packet_moves(user_data, context);
}

static iree_status_t loom_amdgpu_append_storage_address_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  loom_amdgpu_storage_layout_reference_t reference;
  loom_amdgpu_storage_layout_lookup_reference(
      emit_state->storage_layout, context->schedule->module,
      loom_low_storage_address_storage(op), &reference);
  const uint64_t offset = (uint64_t)loom_low_storage_address_offset(op);
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
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .emit_state = emit_state,
  };
  const uint8_t saved_mode =
      emit_state == NULL ? 0 : emit_state->traversal.current_vgpr_msb_mode;
  const uint32_t window = LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE;
  uint8_t mask = 0;
  uint8_t value = 0;
  loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_DST,
                                          assignment->location_base / window,
                                          &mask, &value);
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
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_op_t* op = context->packet->node->op;
  uint16_t accumulator_operand_index = UINT16_MAX;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED &&
        constraint->lhs_operand_index == 0 &&
        constraint->rhs_operand_index != LOOM_LOW_ID_NONE) {
      const loom_low_operand_t* accumulator_operand =
          &descriptor_set->operands[descriptor->operand_start +
                                    constraint->rhs_operand_index];
      accumulator_operand_index = accumulator_operand->source_value_index;
      IREE_ASSERT_LT(accumulator_operand_index, op->operand_count);
      break;
    }
  }
  if (accumulator_operand_index == UINT16_MAX) {
    return loom_amdgpu_append_canonical_asm_form_packet(context);
  }
  if (!loom_amdgpu_op_ties_result_to_operand(op, /*result_index=*/0,
                                             accumulator_operand_index)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU matrix descriptor tie is missing from the "
                            "scheduled low packet");
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
  const iree_string_view_t key = loom_amdgpu_descriptor_key(context);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU assembly %s descriptor '%.*s' has unsupported shape",
      shape->diagnostic_name, (int)key.size, key.data);
}

static bool loom_amdgpu_source0_immediate_asm_form(
    iree_string_view_t canonical_mnemonic) {
  const bool plain_literal =
      iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src0_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src1_lit")) &&
      !iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_src2_lit"));
  return iree_string_view_ends_with(canonical_mnemonic,
                                    IREE_SV("_src0_16_low16")) ||
         iree_string_view_ends_with(canonical_mnemonic, IREE_SV("_vop3_imm")) ||
         plain_literal;
}

enum {
  LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY = 3,
};

typedef struct loom_amdgpu_source_immediate_suffix_t {
  // Canonical low mnemonic suffix component identifying a source operand.
  const char* suffix;
  // Native source operand position named by the suffix component.
  uint8_t source_index;
  // True when the suffix component is backed by a descriptor immediate.
  bool supplies_immediate;
} loom_amdgpu_source_immediate_suffix_t;

static bool loom_amdgpu_match_source_immediate_asm_form(
    iree_string_view_t canonical_mnemonic,
    iree_string_view_t* out_base_mnemonic, uint8_t* out_immediate_count,
    uint8_t out_immediate_source_indices
        [LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY]) {
  static const loom_amdgpu_source_immediate_suffix_t
      kSourceImmediateSuffixes[] = {
          {.suffix = "_src0_lit",
           .source_index = 0,
           .supplies_immediate = true},
          {.suffix = "_src1_lit",
           .source_index = 1,
           .supplies_immediate = true},
          {.suffix = "_src2_lit",
           .source_index = 2,
           .supplies_immediate = true},
          {.suffix = "_src0_inline",
           .source_index = 0,
           .supplies_immediate = true},
          {.suffix = "_src1_inline",
           .source_index = 1,
           .supplies_immediate = true},
          {.suffix = "_src2_inline",
           .source_index = 2,
           .supplies_immediate = true},
          {.suffix = "_src0_zero", .source_index = 0},
          {.suffix = "_src1_zero", .source_index = 1},
          {.suffix = "_src2_zero", .source_index = 2},
      };
  iree_string_view_t base_mnemonic = canonical_mnemonic;
  uint8_t reversed_immediate_source_indices
      [LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY] = {0};
  uint8_t immediate_count = 0;
  bool matched_any_suffix = false;
  while (true) {
    bool matched_suffix = false;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kSourceImmediateSuffixes);
         ++i) {
      iree_string_view_t next_base_mnemonic = base_mnemonic;
      if (!iree_string_view_consume_suffix(
              &next_base_mnemonic,
              iree_make_cstring_view(kSourceImmediateSuffixes[i].suffix))) {
        continue;
      }
      if (kSourceImmediateSuffixes[i].supplies_immediate) {
        if (immediate_count ==
            IREE_ARRAYSIZE(reversed_immediate_source_indices)) {
          return false;
        }
        reversed_immediate_source_indices[immediate_count++] =
            kSourceImmediateSuffixes[i].source_index;
      }
      base_mnemonic = next_base_mnemonic;
      matched_any_suffix = true;
      matched_suffix = true;
      break;
    }
    if (!matched_suffix) {
      break;
    }
  }
  if (!matched_any_suffix) {
    return false;
  }
  *out_base_mnemonic = base_mnemonic;
  *out_immediate_count = immediate_count;
  for (uint8_t i = 0; i < immediate_count; ++i) {
    out_immediate_source_indices[i] =
        reversed_immediate_source_indices[immediate_count - 1 - i];
  }
  return true;
}

static iree_status_t loom_amdgpu_source_immediate_unsupported_shape(
    const loom_native_assembly_packet_context_t* context) {
  const iree_string_view_t key = loom_amdgpu_descriptor_key(context);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU assembly source-immediate descriptor '%.*s' has unsupported "
      "shape",
      (int)key.size, key.data);
}

static bool loom_amdgpu_source_encoding_field_index(uint16_t encoding_field_id,
                                                    uint8_t* out_source_index) {
  switch (encoding_field_id) {
    case LOOM_AMDGPU_ENCODING_FIELD_SRC0:
    case LOOM_AMDGPU_ENCODING_FIELD_SSRC0:
    case LOOM_AMDGPU_ENCODING_FIELD_VSRC0:
      *out_source_index = 0;
      return true;
    case LOOM_AMDGPU_ENCODING_FIELD_SRC1:
    case LOOM_AMDGPU_ENCODING_FIELD_SSRC1:
    case LOOM_AMDGPU_ENCODING_FIELD_VSRC1:
      *out_source_index = 1;
      return true;
    case LOOM_AMDGPU_ENCODING_FIELD_SRC2:
    case LOOM_AMDGPU_ENCODING_FIELD_VSRC2:
      *out_source_index = 2;
      return true;
    default:
      return false;
  }
}

typedef enum loom_amdgpu_source_immediate_slot_kind_e {
  LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_EMPTY = 0,
  LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_OPERAND = 1,
  LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_IMMEDIATE = 2,
  LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_FIXED = 3,
} loom_amdgpu_source_immediate_slot_kind_t;

typedef struct loom_amdgpu_source_immediate_slot_t {
  // Source value kind occupying this native source position.
  loom_amdgpu_source_immediate_slot_kind_t kind;
  // Packet operand index, descriptor immediate index, or fixed source selector.
  uint16_t value;
} loom_amdgpu_source_immediate_slot_t;

static iree_status_t loom_amdgpu_occupy_source_immediate_slot(
    const loom_native_assembly_packet_context_t* context,
    loom_amdgpu_source_immediate_slot_t* slots, uint8_t source_index,
    loom_amdgpu_source_immediate_slot_kind_t kind, uint16_t value) {
  if (source_index >= LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY ||
      slots[source_index].kind != LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_EMPTY) {
    return loom_amdgpu_source_immediate_unsupported_shape(context);
  }
  slots[source_index].kind = kind;
  slots[source_index].value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_fixed_source_selector(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_encoding_table_t* table, uint16_t source_selector) {
  if (table != NULL && source_selector >= table->scalar_inline_u32_zero &&
      source_selector <
          table->scalar_inline_u32_zero + table->scalar_inline_u32_count) {
    return iree_string_builder_append_format(
        context->builder, "%" PRIu32,
        (uint32_t)(source_selector - table->scalar_inline_u32_zero));
  }
  if (table != NULL) {
    for (uint16_t i = 0; i < table->inline_f32_source_count; ++i) {
      const loom_amdgpu_encoding_inline_f32_source_t* source =
          &table->inline_f32_sources[i];
      if (source->source == source_selector) {
        return iree_string_builder_append_format(context->builder, "%" PRIu32,
                                                 source->bit_pattern);
      }
    }
  }
  return loom_amdgpu_source_immediate_unsupported_shape(context);
}

static iree_status_t loom_amdgpu_append_source_immediate_asm_form_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t base_mnemonic, uint8_t immediate_count,
    const uint8_t
        immediate_source_indices[LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY]) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  loom_amdgpu_source_immediate_slot_t
      slots[LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY] = {0};
  if (descriptor->result_count != 1 ||
      descriptor->immediate_count != immediate_count) {
    return loom_amdgpu_source_immediate_unsupported_shape(context);
  }
  for (uint8_t immediate_index = 0; immediate_index < immediate_count;
       ++immediate_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupy_source_immediate_slot(
        context, slots, immediate_source_indices[immediate_index],
        LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_IMMEDIATE, immediate_index));
  }

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t descriptor_operand_index = descriptor->result_count;
       descriptor_operand_index < descriptor->operand_count;
       ++descriptor_operand_index) {
    const loom_low_operand_t* operand = &operands[descriptor_operand_index];
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT) ||
        !loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    uint8_t source_index = 0;
    if (!loom_amdgpu_source_encoding_field_index(operand->encoding_field_id,
                                                 &source_index)) {
      return loom_amdgpu_source_immediate_unsupported_shape(context);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupy_source_immediate_slot(
        context, slots, source_index, LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_OPERAND,
        operand->source_value_index));
  }

  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
          descriptor_set->descriptor_set_ordinal);
  if (descriptor->encoding_field_value_start >
          descriptor_set->encoding_field_value_count ||
      descriptor->encoding_field_value_count >
          descriptor_set->encoding_field_value_count -
              descriptor->encoding_field_value_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor encoding field range is out of range");
  }
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set
             ->encoding_field_values[descriptor->encoding_field_value_start +
                                     i];
    uint8_t source_index = 0;
    if (!loom_amdgpu_source_encoding_field_index(field_value->encoding_field_id,
                                                 &source_index)) {
      continue;
    }
    if (field_value->value > UINT16_MAX || encoding_table == NULL) {
      return loom_amdgpu_source_immediate_unsupported_shape(context);
    }
    const uint16_t source_selector = (uint16_t)field_value->value;
    if (source_selector == encoding_table->source_literal) {
      if (slots[source_index].kind ==
          LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_IMMEDIATE) {
        continue;
      }
      return loom_amdgpu_source_immediate_unsupported_shape(context);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupy_source_immediate_slot(
        context, slots, source_index, LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_FIXED,
        source_selector));
  }

  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
    if (slots[i].kind != LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_EMPTY) {
      source_count = i + 1;
    }
  }
  if (source_count == 0) {
    return loom_amdgpu_source_immediate_unsupported_shape(context);
  }
  for (iree_host_size_t source_index = 0; source_index < source_count;
       ++source_index) {
    if (slots[source_index].kind == LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_EMPTY) {
      return loom_amdgpu_source_immediate_unsupported_shape(context);
    }
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, base_mnemonic));
  bool in_list = false;
  if (!iree_any_bit_set(operands[0].flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, &in_list));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  }
  for (iree_host_size_t source_index = 0; source_index < source_count;
       ++source_index) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, &in_list));
    switch (slots[source_index].kind) {
      case LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_OPERAND: {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_append_operand(context, slots[source_index].value));
        break;
      }
      case LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_IMMEDIATE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_append_packet_immediate_i64(
            context, slots[source_index].value));
        break;
      }
      case LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_FIXED: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_append_fixed_source_selector(
            context, encoding_table, slots[source_index].value));
        break;
      }
      default:
        return loom_amdgpu_source_immediate_unsupported_shape(context);
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

static iree_status_t loom_amdgpu_try_append_asm_form_rule_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t canonical_mnemonic, bool* out_matched) {
  iree_string_view_t base_mnemonic = iree_string_view_empty();
  uint8_t immediate_count = 0;
  uint8_t immediate_source_indices[LOOM_AMDGPU_SOURCE_IMMEDIATE_SLOT_CAPACITY] =
      {0};
  if (loom_amdgpu_match_source_immediate_asm_form(
          canonical_mnemonic, &base_mnemonic, &immediate_count,
          immediate_source_indices)) {
    *out_matched = true;
    return loom_amdgpu_append_source_immediate_asm_form_packet(
        context, base_mnemonic, immediate_count, immediate_source_indices);
  }

  *out_matched = false;
  if (loom_amdgpu_source0_immediate_asm_form(canonical_mnemonic)) {
    *out_matched = true;
    IREE_RETURN_IF_ERROR(loom_amdgpu_validate_assembly_packet_shape(
        context, &kSource0ImmediatePacketShape));
    return loom_amdgpu_append_assembly_packet_form(
        context, &kSource0ImmediatePacketForm);
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

static iree_status_t loom_amdgpu_try_append_mnemonic_rule_packet(
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

static iree_status_t loom_amdgpu_append_vopd_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  return iree_string_builder_append_cstring(context->builder, " ");
}

static iree_status_t loom_amdgpu_append_vopd_vgpr(
    const loom_native_assembly_packet_context_t* context, uint16_t vgpr) {
  return iree_string_builder_append_format(context->builder, "v%" PRIu16, vgpr);
}

static iree_string_view_t loom_amdgpu_vopd_component_assembly_mnemonic(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_info_t* info) {
  const uint16_t descriptor_set_ordinal =
      context->schedule->target.descriptor_set->descriptor_set_ordinal;
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(descriptor_set_ordinal);
  IREE_ASSERT(descriptor_set_info != NULL);
  if (loom_amdgpu_descriptor_set_info_has_flags(
          descriptor_set_info,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS) &&
      info->numeric_minmax_mnemonic.size != 0) {
    return info->numeric_minmax_mnemonic;
  }
  return info->assembly_mnemonic;
}

static iree_status_t loom_amdgpu_append_vopd_tied_accumulate_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->src0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_vopd_vgpr(context, component->vsrc1);
}

static iree_status_t loom_amdgpu_append_vopd_fmaak_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->src0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vsrc1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return iree_string_builder_append_format(context->builder, "0x%08" PRIx32,
                                           component->immediate_u32);
}

static iree_status_t loom_amdgpu_append_vopd_fmamk_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->src0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder, "0x%08" PRIx32, component->immediate_u32));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_vopd_vgpr(context, component->vsrc1);
}

static iree_status_t loom_amdgpu_append_vopd_binary_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->src0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_vopd_vgpr(context, component->vsrc1);
}

static iree_status_t loom_amdgpu_append_vopd_inline_mov_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return iree_string_builder_append_format(context->builder, "%" PRIu32,
                                           component->immediate_u32);
}

static iree_status_t loom_amdgpu_append_vopd_register_mov_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component,
    iree_string_view_t mnemonic) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_mnemonic(context, mnemonic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_vgpr(context, component->vdst));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_vopd_vgpr(context, component->src0);
}

static iree_status_t loom_amdgpu_append_vopd_component(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_component_t* component) {
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_op(component->op);
  if (info == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "verified AMDGPU VOPD plans must reference supported component ops");
    IREE_BUILTIN_UNREACHABLE();
  }
  const iree_string_view_t mnemonic =
      loom_amdgpu_vopd_component_assembly_mnemonic(context, info);
  switch (component->form) {
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE:
      return loom_amdgpu_append_vopd_tied_accumulate_component(
          context, component, mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL:
      return loom_amdgpu_append_vopd_fmaak_component(context, component,
                                                     mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL:
      return loom_amdgpu_append_vopd_fmamk_component(context, component,
                                                     mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR:
      return loom_amdgpu_append_vopd_binary_component(context, component,
                                                      mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV:
      return loom_amdgpu_append_vopd_inline_mov_component(context, component,
                                                          mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_REGISTER_MOV:
      return loom_amdgpu_append_vopd_register_mov_component(context, component,
                                                            mnemonic);
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_CNDMASK_VCC:
      return loom_amdgpu_append_vopd_binary_component(context, component,
                                                      mnemonic);
    default:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU VOPD component metadata must use a known form");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_append_vopd_pair_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_vopd_pair_t* pair) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_vopd_component(context, &pair->x));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " :: "));
  return loom_amdgpu_append_vopd_component(context, &pair->y);
}

static iree_status_t loom_amdgpu_try_append_mnemonic_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  const iree_string_view_t mnemonic = loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset);
  return loom_amdgpu_try_append_mnemonic_rule_packet(context, mnemonic,
                                                     out_matched);
}

static iree_status_t loom_amdgpu_try_append_canonical_asm_form_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  const loom_low_descriptor_view_t* descriptor_view =
      loom_amdgpu_descriptor_view(context);
  if (descriptor_view->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  const loom_low_asm_form_t* canonical_form = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_canonical_asm_form(context, &canonical_form));
  if (canonical_form->native_assembly_value_count > 0) {
    // Explicit native values are the terminal canonical spelling. Leave them
    // to the final canonical fallback after route-specific dispatchers run.
    return iree_ok_status();
  }
  const iree_string_view_t canonical_mnemonic =
      loom_native_assembly_descriptor_string(
          context->schedule->target.descriptor_set,
          canonical_form->mnemonic_string_offset);
  return loom_amdgpu_try_append_asm_form_rule_packet(
      context, canonical_mnemonic, out_matched);
}

static iree_status_t loom_amdgpu_try_append_effect_route_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  return loom_amdgpu_try_append_descriptor_packet_route(
      context, loom_amdgpu_descriptor_packet_route_flags(context), out_matched);
}

static iree_status_t loom_amdgpu_try_append_matrix_dispatch_packet(
    const loom_native_assembly_packet_context_t* context, bool* out_matched) {
  *out_matched = false;
  if (!iree_any_bit_set(loom_amdgpu_descriptor_traits(
                            context->schedule->target.descriptor_set,
                            context->packet->descriptor),
                        LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX)) {
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
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_mnemonic_dispatch_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_canonical_asm_form_dispatch_packet(context,
                                                                &matched));
  if (matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_effect_route_dispatch_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_append_matrix_dispatch_packet(context, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_amdgpu_append_canonical_dispatch_packet(context, &matched);
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
  state->traversal.current_vgpr_msb_mode = (uint8_t)((uint16_t)mode & 0xFFu);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_stateful_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state == NULL || state->packet_plan.address_state == NULL) {
    const loom_amdgpu_address_state_requirement_t requirement =
        loom_amdgpu_address_state_requirement_for_packet(context->allocation,
                                                         context->packet);
    const uint8_t current_mode =
        state == NULL ? 0 : state->traversal.current_vgpr_msb_mode;
    if ((current_mode & requirement.mask) !=
        (requirement.value & requirement.mask)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor packet reached assembly emission without its "
          "required address-state transition");
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_packet(NULL, context));
  return loom_amdgpu_update_vgpr_msb_mode_after_descriptor(state, context);
}

static iree_status_t loom_amdgpu_append_vopd_or_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_assembly_emit_state_t* state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
  if (state == NULL || state->packet_plan.vopd_plan == NULL) {
    return loom_amdgpu_append_stateful_descriptor_packet(user_data, context);
  }
  const loom_amdgpu_vopd_packet_t* vopd_packet =
      loom_amdgpu_vopd_plan_packet_at(state->packet_plan.vopd_plan,
                                      context->packet->packet_index);
  if (vopd_packet == NULL) {
    return loom_amdgpu_append_stateful_descriptor_packet(user_data, context);
  }
  IREE_ASSERT(vopd_packet->pair_index <
              state->packet_plan.vopd_plan->pair_count);
  if (vopd_packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
    return iree_ok_status();
  }
  IREE_ASSERT(vopd_packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST);
  const loom_amdgpu_vopd_pair_t* pair =
      &state->packet_plan.vopd_plan->pairs[vopd_packet->pair_index];
  return loom_amdgpu_append_vopd_pair_packet(context, pair);
}

static iree_status_t loom_amdgpu_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  return iree_string_builder_append_cstring(context->builder, "s_endpgm");
}

static iree_status_t loom_amdgpu_append_original_branch_target(
    loom_amdgpu_assembly_emit_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    const loom_block_t* direct_target) {
  if (state->branches.layout == NULL) {
    return loom_native_assembly_append_block_label(
        context->schedule, direct_target, context->builder);
  }
  IREE_ASSERT_LT(state->branches.next_edge_index,
                 state->branches.layout->edge_count);
  const loom_amdgpu_branch_layout_edge_t* edge =
      &state->branches.layout->edges[state->branches.next_edge_index++];
  return loom_amdgpu_append_branch_target_label(state, context, edge->target);
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
  return loom_amdgpu_append_original_branch_target(emit_state, context, dest);
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
  loom_amdgpu_assembly_emit_state_t* emit_state =
      (loom_amdgpu_assembly_emit_state_t*)user_data;
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
    return loom_amdgpu_append_original_branch_target(emit_state, context,
                                                     true_dest);
  }
  if (true_block_index == current_block_index + 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(context->builder,
                                                            "s_cbranch_scc0 "));
    return loom_amdgpu_append_original_branch_target(emit_state, context,
                                                     false_dest);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_cbranch_scc1 "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_original_branch_target(
      emit_state, context, true_dest));
  if (false_block_index == current_block_index + 1) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  s_branch "));
  return loom_amdgpu_append_original_branch_target(emit_state, context,
                                                   false_dest);
}

static iree_status_t loom_amdgpu_verify_assembly_target(
    const loom_low_schedule_table_t* schedule) {
  if (schedule->target.descriptor_set->target_stable_id !=
      LOOM_AMDGPU_TARGET_STABLE_ID) {
    const iree_string_view_t target_key =
        loom_native_assembly_descriptor_string(
            schedule->target.descriptor_set,
            schedule->target.descriptor_set->target_key_string_offset);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_structural_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  switch (op->kind) {
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_MOVE:
      return loom_amdgpu_append_copy_packet(user_data, context);
    case LOOM_OP_LOW_SLICE:
      return loom_amdgpu_append_slice_packet(user_data, context);
    case LOOM_OP_LOW_CONCAT:
      return loom_amdgpu_append_concat_packet(user_data, context);
    case LOOM_OP_LOW_STORAGE_ADDRESS:
      return loom_amdgpu_append_storage_address_packet(user_data, context);
    case LOOM_OP_LOW_RETURN:
      return loom_amdgpu_append_return_packet(user_data, context);
    case LOOM_OP_LOW_BR:
      return loom_amdgpu_append_branch_packet(user_data, context);
    case LOOM_OP_LOW_COND_BR:
      return loom_amdgpu_append_cond_branch_packet(user_data, context);
    default:
      break;
  }
  return loom_native_assembly_make_unsupported_structural_packet_status(
      IREE_SV("AMDGPU"), context);
}

static loom_native_assembly_format_options_t loom_amdgpu_assembly_options(
    loom_amdgpu_assembly_emit_state_t* emit_state,
    loom_native_assembly_append_packet_callback_t append_before_packet,
    loom_native_assembly_append_packet_callback_t append_descriptor_packet) {
  return (loom_native_assembly_format_options_t){
      .append_before_packet = append_before_packet,
      .append_descriptor_packet = append_descriptor_packet,
      .append_structural_packet =
          {
              .fn = loom_amdgpu_append_structural_packet,
              .user_data = emit_state,
          },
  };
}

iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  return loom_amdgpu_emit_assembly_fragment_with_options(
      schedule, allocation, /*options=*/NULL, builder, scratch_arena);
}

iree_status_t loom_amdgpu_emit_assembly_fragment_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_assembly_fragment_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  loom_amdgpu_storage_layout_t derived_storage_layout;
  const loom_amdgpu_storage_layout_t* storage_layout =
      options ? options->storage_layout : NULL;
  if (storage_layout == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_build(
        &schedule->storage_layout, scratch_arena, &derived_storage_layout));
    storage_layout = &derived_storage_layout;
  }
  const loom_amdgpu_packet_plan_t* packet_plan =
      options ? options->packet_plan : NULL;
  const loom_amdgpu_address_state_plan_t* address_state =
      packet_plan ? &packet_plan->address_state : NULL;
  const loom_amdgpu_wait_packet_plan_t* wait_packets =
      packet_plan ? &packet_plan->wait_packets : NULL;
  const loom_amdgpu_wait_state_plan_t* wait_states =
      packet_plan ? &packet_plan->wait_states : NULL;
  const loom_amdgpu_vopd_plan_t* vopd_plan =
      packet_plan ? &packet_plan->vopd_plan : NULL;
  const loom_amdgpu_branch_layout_t* branch_layout =
      options && options->branch_layout && options->branch_layout->island_count
          ? options->branch_layout
          : NULL;

  loom_amdgpu_assembly_emit_state_t emit_state = {
      .storage_layout = storage_layout,
      .packet_plan =
          {
              .address_state = address_state,
              .wait_packets = wait_packets,
              .wait_states = wait_states,
              .vopd_plan = vopd_plan,
          },
      .branches =
          {
              .layout = branch_layout,
          },
      .traversal =
          {
              .current_block_index = LOOM_LOW_PACKET_INDEX_NONE,
          },
  };
  const loom_native_assembly_format_options_t format_options =
      loom_amdgpu_assembly_options(
          &emit_state,
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_insertions_before_packet,
              .user_data = &emit_state,
          },
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_vopd_or_descriptor_packet,
              .user_data = &emit_state,
          });
  IREE_RETURN_IF_ERROR(loom_native_assembly_format_fragment(
      schedule, allocation, &format_options, builder, scratch_arena));
  if (address_state != NULL) {
    IREE_ASSERT_EQ(emit_state.packet_plan.next_address_state_index,
                   address_state->transition_count);
  }
  IREE_ASSERT_EQ(emit_state.traversal.current_vgpr_msb_mode, 0);
  if (wait_packets != NULL) {
    IREE_ASSERT_EQ(emit_state.packet_plan.next_wait_packet_index,
                   wait_packets->packet_count);
  }
  if (wait_states != NULL) {
    IREE_ASSERT_EQ(emit_state.packet_plan.next_wait_state_index,
                   wait_states->state_count);
  }
  if (branch_layout != NULL) {
    IREE_ASSERT_EQ(emit_state.branches.next_edge_index,
                   branch_layout->edge_count);
    IREE_ASSERT_EQ(emit_state.branches.next_group_index,
                   branch_layout->group_count);
  }
  return iree_ok_status();
}
