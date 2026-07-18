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

typedef struct loom_amdgpu_vgpr_msb_desired_transition_t {
  // Operation before which the transition is inserted, or NULL to append.
  const loom_op_t* before_op;
  // Block receiving an appended transition when |before_op| is NULL.
  loom_block_t* block;
  // Packed previous/new S_SET_VGPR_MSB mode immediate.
  uint16_t mode_immediate;
  // Source location assigned to the materialized transition.
  loom_location_id_t location;
} loom_amdgpu_vgpr_msb_desired_transition_t;

typedef struct loom_amdgpu_vgpr_msb_existing_transition_t {
  // Existing S_SET_VGPR_MSB operation.
  const loom_op_t* op;
  // Packed previous/new S_SET_VGPR_MSB mode immediate.
  uint16_t mode_immediate;
} loom_amdgpu_vgpr_msb_existing_transition_t;

typedef struct loom_amdgpu_vgpr_msb_block_plan_t {
  // Canonical transitions required by the accepted allocation.
  loom_amdgpu_vgpr_msb_desired_transition_t* desired_transitions;
  // Number of entries in |desired_transitions|.
  iree_host_size_t desired_transition_count;
  // Scheduled transitions already present in the block.
  loom_amdgpu_vgpr_msb_existing_transition_t* existing_transitions;
  // Number of entries in |existing_transitions|.
  iree_host_size_t existing_transition_count;
  // True when the existing stream satisfies every scheduled state read.
  bool existing_stream_valid;
} loom_amdgpu_vgpr_msb_block_plan_t;

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
  IREE_ASSERT(!iree_string_view_is_empty(key),
              "generated address-state descriptor must have a key");
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

static void loom_amdgpu_vgpr_msb_insert_slot_bank(
    loom_amdgpu_vgpr_msb_mode_requirement_t* requirement,
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank) {
  IREE_ASSERT_GE(slot, LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0);
  IREE_ASSERT_LE(slot, LOOM_AMDGPU_VGPR_MSB_SLOT_DST);
  IREE_ASSERT_LE(bank, 3u);
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  if ((requirement->mask & slot_mask) != 0) {
    IREE_ASSERT_EQ(requirement->value & slot_mask, slot_value);
    return;
  }
  requirement->mask |= slot_mask;
  requirement->value =
      (uint8_t)((requirement->value & ~slot_mask) | slot_value);
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
    IREE_ASSERT_EQ(assignment->location_count, operand->unit_count);
    IREE_ASSERT(
        operand->addressable_unit_count == LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE,
        "generated target-state VGPR operand must use the "
        "S_SET_VGPR_MSB addressable window");
    const uint32_t bank =
        assignment->location_base / LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE;
    loom_amdgpu_vgpr_msb_insert_slot_bank(
        out_requirement,
        (loom_amdgpu_vgpr_msb_slot_t)operand->address_state_slot, bank);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_s_set_vgpr_msb_mode_immediate(
    const loom_amdgpu_address_state_context_t* context,
    const loom_low_schedule_node_t* node, uint16_t* out_mode_immediate) {
  *out_mode_immediate = 0;
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
  *out_mode_immediate = (uint16_t)attr->value.i64;
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

static void loom_amdgpu_append_desired_vgpr_msb_transition(
    loom_amdgpu_vgpr_msb_block_plan_t* plan, const loom_op_t* before_op,
    loom_block_t* block, uint8_t previous_mode, uint8_t new_mode,
    loom_location_id_t location) {
  plan->desired_transitions[plan->desired_transition_count++] =
      (loom_amdgpu_vgpr_msb_desired_transition_t){
          .before_op = before_op,
          .block = block,
          .mode_immediate =
              (uint16_t)(((uint16_t)previous_mode << 8) | new_mode),
          .location = location,
      };
}

static iree_status_t loom_amdgpu_plan_vgpr_msb_for_block(
    const loom_amdgpu_address_state_context_t* context,
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_block_t* block, iree_arena_allocator_t* arena,
    loom_amdgpu_vgpr_msb_block_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vgpr_msb_block_plan_t){
      .existing_stream_valid = true,
  };
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, (iree_host_size_t)block->node_count + 1,
                                sizeof(*out_plan->desired_transitions),
                                (void**)&out_plan->desired_transitions));
  if (block->scheduled_node_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, block->scheduled_node_count,
                                  sizeof(*out_plan->existing_transitions),
                                  (void**)&out_plan->existing_transitions));
  }

  // Build the desired stream in the source order that the rewrite can
  // represent. State dependencies then constrain the scheduler to keep each
  // affected packet inside the source interval established by these writes.
  uint8_t desired_mode = 0;
  for (uint32_t i = 0; i < block->node_count; ++i) {
    const uint32_t node_index = block->node_start + i;
    if (node_index >= frame->schedule.node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU address-state block references node "
                              "%" PRIu32 " but schedule has %" PRIhsz " nodes",
                              node_index, frame->schedule.node_count);
    }
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[node_index];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR) {
      continue;
    }
    if (loom_amdgpu_descriptor_is_s_set_vgpr_msb(context, node->descriptor)) {
      continue;
    }

    loom_amdgpu_vgpr_msb_mode_requirement_t requirement = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_collect_vgpr_msb_mode_requirement(
        frame, node, &requirement));
    if (requirement.mask == 0 || (desired_mode & requirement.mask) ==
                                     (requirement.value & requirement.mask)) {
      continue;
    }

    const uint8_t new_mode = (uint8_t)((desired_mode & ~requirement.mask) |
                                       (requirement.value & requirement.mask));
    loom_amdgpu_append_desired_vgpr_msb_transition(
        out_plan, node->op, /*block=*/NULL, desired_mode, new_mode,
        node->op->location);
    desired_mode = new_mode;
  }

  // Validate the existing stream in the order packets will be emitted. An
  // identical immediate sequence is insufficient: a scheduler defect or a
  // malformed authored stream may place the right transition after its read.
  uint8_t existing_mode = 0;
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
      uint16_t mode_immediate = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_read_s_set_vgpr_msb_mode_immediate(
          context, node, &mode_immediate));
      out_plan->existing_transitions[out_plan->existing_transition_count++] =
          (loom_amdgpu_vgpr_msb_existing_transition_t){
              .op = node->op,
              .mode_immediate = mode_immediate,
          };
      if ((uint8_t)(mode_immediate >> 8) != existing_mode) {
        out_plan->existing_stream_valid = false;
      }
      existing_mode = (uint8_t)(mode_immediate & 0xFFu);
      continue;
    }

    loom_amdgpu_vgpr_msb_mode_requirement_t requirement = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_collect_vgpr_msb_mode_requirement(
        frame, node, &requirement));
    if (requirement.mask != 0 && (existing_mode & requirement.mask) !=
                                     (requirement.value & requirement.mask)) {
      out_plan->existing_stream_valid = false;
    }
  }

  if (existing_mode != 0) {
    out_plan->existing_stream_valid = false;
  }
  if (desired_mode != 0) {
    const loom_block_t* const_block = block->block;
    loom_block_t* mutable_block = (loom_block_t*)const_block;
    const loom_op_t* before_op = NULL;
    loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
    if (const_block != NULL && const_block->last_op != NULL &&
        iree_all_bits_set(const_block->last_op->traits,
                          LOOM_TRAIT_TERMINATOR)) {
      before_op = const_block->last_op;
      location = const_block->last_op->location;
    }
    loom_amdgpu_append_desired_vgpr_msb_transition(out_plan, before_op,
                                                   mutable_block, desired_mode,
                                                   /*new_mode=*/0, location);
  }
  return iree_ok_status();
}

static bool loom_amdgpu_vgpr_msb_block_plan_matches(
    const loom_amdgpu_vgpr_msb_block_plan_t* plan) {
  if (!plan->existing_stream_valid ||
      plan->existing_transition_count != plan->desired_transition_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < plan->desired_transition_count; ++i) {
    if (plan->existing_transitions[i].mode_immediate !=
        plan->desired_transitions[i].mode_immediate) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_apply_vgpr_msb_block_plan(
    loom_amdgpu_address_state_context_t* context,
    const loom_amdgpu_vgpr_msb_block_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->existing_transition_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_op_erase(
        context->module, (loom_op_t*)plan->existing_transitions[i].op));
  }
  for (iree_host_size_t i = 0; i < plan->desired_transition_count; ++i) {
    const loom_amdgpu_vgpr_msb_desired_transition_t* transition =
        &plan->desired_transitions[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_s_set_vgpr_msb(
        context, transition->before_op, transition->block,
        transition->mode_immediate, transition->location));
  }
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

  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_amdgpu_address_state_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_initialize_context(
      module, descriptor_set, &context));
  if (context.set_vgpr_msb_descriptor == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_vgpr_msb_block_plan_t* plans = NULL;
  if (frame->schedule.block_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, frame->schedule.block_count, sizeof(*plans), (void**)&plans));
  }
  for (iree_host_size_t i = 0; i < frame->schedule.block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_plan_vgpr_msb_for_block(
        &context, frame, &frame->schedule.blocks[i], arena, &plans[i]));
  }
  for (iree_host_size_t i = 0; i < frame->schedule.block_count; ++i) {
    if (loom_amdgpu_vgpr_msb_block_plan_matches(&plans[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_apply_vgpr_msb_block_plan(&context, &plans[i]));
    out_result->changed = true;
  }
  return iree_ok_status();
}
