// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/address_state.h"

#include <inttypes.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_vgpr_msb_mode_requirement_t {
  // Two-bit S_SET_VGPR_MSB slot mask in low-immediate layout.
  uint8_t mask;
  // Two-bit S_SET_VGPR_MSB slot values in low-immediate layout.
  uint8_t value;
} loom_amdgpu_vgpr_msb_mode_requirement_t;

typedef struct loom_amdgpu_address_state_context_t {
  // Module being rewritten.
  loom_module_t* module;
  // Descriptor set selected by the scheduled low function.
  const loom_low_descriptor_set_t* descriptor_set;
  // s_set_vgpr_msb descriptor row, or NULL on targets without that packet.
  const loom_low_descriptor_t* set_vgpr_msb_descriptor;
  // Module string ID for the s_set_vgpr_msb descriptor key.
  loom_string_id_t set_vgpr_msb_opcode_id;
  // Module string ID for the s_set_vgpr_msb mode immediate attribute.
  loom_string_id_t mode_attr_id;
  // Builder that owns target-state packet insertion.
  loom_builder_t builder;
} loom_amdgpu_address_state_context_t;

static bool loom_amdgpu_descriptor_is_s_set_vgpr_msb(
    const loom_amdgpu_address_state_context_t* context,
    const loom_low_descriptor_t* descriptor) {
  return context->set_vgpr_msb_descriptor != NULL &&
         descriptor == context->set_vgpr_msb_descriptor;
}

static iree_status_t loom_amdgpu_intern_descriptor_opcode(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t* out_opcode_id) {
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU address-state descriptor has no key");
  }
  return loom_module_intern_string(module, key, out_opcode_id);
}

static iree_status_t loom_amdgpu_address_state_initialize_context(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_address_state_context_t* out_context) {
  *out_context = (loom_amdgpu_address_state_context_t){
      .module = module,
      .descriptor_set = descriptor_set,
      .set_vgpr_msb_descriptor = loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB),
      .set_vgpr_msb_opcode_id = LOOM_STRING_ID_INVALID,
      .mode_attr_id = LOOM_STRING_ID_INVALID,
  };
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &out_context->builder);
  if (out_context->set_vgpr_msb_descriptor == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern_descriptor_opcode(
      module, descriptor_set, out_context->set_vgpr_msb_descriptor,
      &out_context->set_vgpr_msb_opcode_id));
  return loom_module_intern_string(module, IREE_SV("mode"),
                                   &out_context->mode_attr_id);
}

static iree_status_t loom_amdgpu_vgpr_msb_insert_slot_bank(
    loom_amdgpu_vgpr_msb_mode_requirement_t* requirement,
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank) {
  if (slot == LOOM_AMDGPU_VGPR_MSB_SLOT_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target-state VGPR operand has no S_SET_VGPR_MSB slot");
  }
  if (bank > 3) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU target-state VGPR operand requires VGPR MSB bank %" PRIu32,
        bank);
  }
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  if ((requirement->mask & slot_mask) != 0 &&
      (requirement->value & slot_mask) != slot_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target-state packet requires conflicting VGPR MSB banks for "
        "one encoding slot");
  }
  requirement->mask |= slot_mask;
  requirement->value =
      (uint8_t)((requirement->value & ~slot_mask) | slot_value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_target_state_operand_assignment(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node, uint16_t descriptor_operand_index,
    const loom_low_allocation_assignment_t** out_assignment) {
  *out_assignment = NULL;
  const uint16_t result_count = node->descriptor->result_count;
  if (descriptor_operand_index < result_count) {
    if (descriptor_operand_index >= node->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU target-state descriptor result %" PRIu16
                              " has no matching schedule result",
                              descriptor_operand_index);
    }
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    *out_assignment = loom_low_allocation_assignment_for_value_ordinal(
        &frame->allocation, result_ordinals[descriptor_operand_index], NULL);
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  uint16_t packet_operand_ordinal = 0;
  for (uint16_t i = result_count; i < node->descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set
             ->operands[node->descriptor->operand_start + (uint32_t)i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    if (i == descriptor_operand_index) {
      if (packet_operand_ordinal >= node->operand_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU target-state descriptor operand %" PRIu16
            " has no matching schedule operand",
            descriptor_operand_index);
      }
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      *out_assignment = loom_low_allocation_assignment_for_value_ordinal(
          &frame->allocation, operand_ordinals[packet_operand_ordinal], NULL);
      return iree_ok_status();
    }
    ++packet_operand_ordinal;
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU target-state descriptor operand %" PRIu16
                          " has no matching schedule operand",
                          descriptor_operand_index);
}

static iree_status_t loom_amdgpu_collect_vgpr_msb_mode_requirement(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node,
    loom_amdgpu_vgpr_msb_mode_requirement_t* out_requirement) {
  *out_requirement = (loom_amdgpu_vgpr_msb_mode_requirement_t){0};
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + (uint32_t)i];
    if (operand->address_map_kind !=
        LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_target_state_operand_assignment(
        frame, node, i, &assignment));
    if (assignment == NULL ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      continue;
    }
    if (assignment->location_count != operand->unit_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU target-state value %" PRIu32 " has %" PRIu32
          " assigned registers but descriptor field needs %" PRIu16,
          assignment->value_id, assignment->location_count,
          operand->unit_count);
    }
    if (operand->addressable_unit_count == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU target-state VGPR operand has no "
                              "addressable window size");
    }
    const uint32_t bank =
        assignment->location_base / operand->addressable_unit_count;
    const loom_amdgpu_vgpr_msb_slot_t slot = loom_amdgpu_encoding_vgpr_msb_slot(
        descriptor->encoding_format_id, operand->encoding_field_id);
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vgpr_msb_insert_slot_bank(out_requirement, slot, bank));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_s_set_vgpr_msb_mode(
    const loom_amdgpu_address_state_context_t* context,
    const loom_low_schedule_node_t* node, uint8_t* out_mode) {
  *out_mode = 0;
  const loom_op_t* op = node->op;
  if (!loom_low_op_isa(op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU s_set_vgpr_msb packet is not a low.op");
  }
  const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  const loom_named_attr_t* attr = NULL;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == context->mode_attr_id) {
      attr = &attrs.entries[i];
      break;
    }
  }
  if (attr == NULL || attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU s_set_vgpr_msb packet is missing an i64 "
                            "mode immediate");
  }
  if (attr->value.i64 < 0 || attr->value.i64 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU s_set_vgpr_msb mode immediate %" PRId64
                            " is not a u16",
                            attr->value.i64);
  }
  *out_mode = (uint8_t)((uint16_t)attr->value.i64 & 0xFFu);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_build_s_set_vgpr_msb(
    loom_amdgpu_address_state_context_t* context, const loom_op_t* before_op,
    loom_block_t* block, uint16_t mode, loom_location_id_t location) {
  if (before_op != NULL) {
    loom_builder_set_before(&context->builder, before_op);
  } else {
    loom_builder_set_block(&context->builder, block);
  }
  loom_named_attr_t attr = {
      .name_id = context->mode_attr_id,
      .reserved = 0,
      .value = loom_attr_i64(mode),
  };
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &context->builder, context->descriptor_set,
      context->set_vgpr_msb_descriptor, context->set_vgpr_msb_opcode_id,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attr, 1), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op);
}

static iree_status_t loom_amdgpu_materialize_vgpr_msb_for_block(
    loom_amdgpu_address_state_context_t* context,
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_block_t* block,
    loom_low_emission_frame_materialize_address_state_result_t* result) {
  uint8_t current_mode = 0;
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t node_index =
        frame->schedule
            .scheduled_node_indices[block->scheduled_node_start + (uint32_t)i];
    if (node_index >= frame->schedule.node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU address-state schedule references node "
                              "%" PRIu32 " but schedule has %" PRIhsz " nodes",
                              node_index, frame->schedule.node_count);
    }
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[node_index];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR) {
      continue;
    }
    if (loom_amdgpu_descriptor_is_s_set_vgpr_msb(context, node->descriptor)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_read_s_set_vgpr_msb_mode(context, node, &current_mode));
      continue;
    }

    loom_amdgpu_vgpr_msb_mode_requirement_t requirement = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_collect_vgpr_msb_mode_requirement(
        frame, node, &requirement));
    if (requirement.mask == 0 || (current_mode & requirement.mask) ==
                                     (requirement.value & requirement.mask)) {
      continue;
    }

    const uint8_t new_mode = (uint8_t)((current_mode & ~requirement.mask) |
                                       (requirement.value & requirement.mask));
    const uint16_t mode_immediate =
        (uint16_t)(((uint16_t)current_mode << 8) | new_mode);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_s_set_vgpr_msb(
        context, node->op, NULL, mode_immediate, node->op->location));
    current_mode = new_mode;
    result->changed = true;
  }

  if (current_mode == 0) {
    return iree_ok_status();
  }
  const loom_block_t* const_block = block->block;
  loom_block_t* mutable_block = (loom_block_t*)const_block;
  const loom_op_t* before_op = NULL;
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  if (const_block != NULL && const_block->last_op != NULL &&
      iree_all_bits_set(const_block->last_op->traits, LOOM_TRAIT_TERMINATOR)) {
    before_op = const_block->last_op;
    location = const_block->last_op->location;
  }
  const uint16_t mode_immediate = (uint16_t)((uint16_t)current_mode << 8);
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_s_set_vgpr_msb(
      context, before_op, mutable_block, mode_immediate, location));
  result->changed = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_materialize_address_state(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_low_emission_frame_materialize_address_state_result_t){0};
  (void)function_op;
  (void)arena;

  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_amdgpu_address_state_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_initialize_context(
      module, descriptor_set, &context));
  if (context.set_vgpr_msb_descriptor == NULL) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < frame->schedule.block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_vgpr_msb_for_block(
        &context, frame, &frame->schedule.blocks[i], out_result));
  }
  return iree_ok_status();
}
