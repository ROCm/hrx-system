// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"

#include <stddef.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

typedef enum loom_aie2p_block_terminator_e {
  LOOM_AIE2P_BLOCK_TERMINATOR_NONE = 0,
  LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH = 1,
  LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH = 2,
  LOOM_AIE2P_BLOCK_TERMINATOR_RETURN = 3,
} loom_aie2p_block_terminator_t;

typedef struct loom_aie2p_block_analysis_t {
  // Structural terminator packet ending this block.
  uint32_t terminator_packet_index;
  // Logical issue cycle at which the structural terminator completes.
  uint32_t terminator_issue_cycle;
  // Greatest logical issue cycle in this scheduled block.
  uint32_t maximum_issue_cycle;
  // Structural control kind selected for this block.
  loom_aie2p_block_terminator_t terminator;
} loom_aie2p_block_analysis_t;

typedef struct loom_aie2p_bundle_plan_analysis_t {
  // Per-block control and cycle summaries in source block order.
  loom_aie2p_block_analysis_t* blocks;
  // Total logical issue cycles across all source blocks.
  iree_host_size_t logical_issue_cycle_count;
  // Final sequential allocation moves required by low.br payloads.
  iree_host_size_t edge_move_count;
  // Conservative branch, return, and delay-bundle capacity.
  iree_host_size_t control_bundle_capacity;
  // Structural local-storage addresses requiring placement fixups.
  iree_host_size_t storage_fixup_count;
} loom_aie2p_bundle_plan_analysis_t;

typedef struct loom_aie2p_bundle_plan_builder_t {
  // Emission frame being converted into physical bundles.
  const loom_low_emission_frame_t* frame;
  // Exact AIE2P descriptor set selected by the frame.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena-backed bundle storage owned by the final plan.
  loom_aie2p_planned_bundle_t* bundles;
  // Maximum number of records available in |bundles|.
  iree_host_size_t bundle_capacity;
  // Number of populated records in |bundles|.
  iree_host_size_t bundle_count;
  // Arena-backed slot storage owned by the final plan.
  loom_aie2p_planned_slot_t* slots;
  // Maximum number of records available in |slots|.
  iree_host_size_t slot_capacity;
  // Number of populated records in |slots|.
  iree_host_size_t slot_count;
  // Arena-backed branch target records owned by the final plan.
  loom_aie2p_planned_branch_fixup_t* branch_fixups;
  // Maximum number of records available in |branch_fixups|.
  iree_host_size_t branch_fixup_capacity;
  // Number of populated records in |branch_fixups|.
  iree_host_size_t branch_fixup_count;
  // Arena-backed local-storage fixups owned by the final plan.
  loom_aie2p_planned_storage_fixup_t* storage_fixups;
  // Fixup index for each scheduled packet, or UINT32_MAX when absent.
  uint32_t* storage_fixup_indices_by_packet;
  // Maximum number of records available in |storage_fixups|.
  iree_host_size_t storage_fixup_capacity;
  // Number of populated records in |storage_fixups|.
  iree_host_size_t storage_fixup_count;
  // Arena-backed contribution offsets in source block order.
  uint32_t* block_byte_offsets;
  // Number of source blocks represented by |block_byte_offsets|.
  iree_host_size_t block_count;
  // Exact byte length of all bundles appended so far.
  iree_host_size_t encoded_byte_length;
  // Source-order block receiving newly appended bundles.
  uint32_t current_block_index;
} loom_aie2p_bundle_plan_builder_t;

static uint32_t loom_aie2p_bundle_plan_descriptor_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  IREE_ASSERT(descriptor >= descriptor_set->descriptors);
  IREE_ASSERT(descriptor <
              descriptor_set->descriptors + descriptor_set->descriptor_count);
  return (uint32_t)(descriptor - descriptor_set->descriptors);
}

static loom_aie2p_instruction_info_t loom_aie2p_bundle_plan_instruction_info(
    loom_aie2p_instruction_id_t id) {
  loom_aie2p_instruction_info_t info;
  const bool found = loom_aie2p_encoding_query_instruction_info(id, &info);
  IREE_ASSERT(found && "generated descriptor encoding ID must be valid");
  return info;
}

static iree_host_size_t loom_aie2p_bundle_plan_control_bundle_count(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  return (iree_host_size_t)loom_aie2p_bundle_plan_instruction_info(
             descriptor->encoding_id)
             .delay_slot_count +
         1u;
}

static uint8_t loom_aie2p_bundle_plan_single_slot_byte_length(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  const loom_aie2p_instruction_info_t instruction_info =
      loom_aie2p_bundle_plan_instruction_info(descriptor->encoding_id);
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&instruction_info.slot,
                                                       1);
  IREE_ASSERT(format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID &&
              "structural AIE2P instruction must have a standalone bundle");
  loom_aie2p_bundle_format_info_t format_info;
  const bool found =
      loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
  IREE_ASSERT(found && format_info.slot_count == 1 &&
              format_info.bit_count % 8 == 0);
  return format_info.bit_count / 8;
}

static bool loom_aie2p_bundle_plan_structural_move_isa(const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_move_isa(op) ||
         loom_low_slice_isa(op) || loom_low_concat_isa(op);
}

static bool loom_aie2p_bundle_plan_non_emitting_structural_isa(
    const loom_op_t* op) {
  return loom_low_live_in_isa(op) || loom_low_resource_isa(op) ||
         loom_low_storage_reserve_isa(op) || loom_low_storage_view_isa(op);
}

static const loom_low_allocation_packet_move_group_t*
loom_aie2p_bundle_plan_packet_move_group(const loom_low_emission_frame_t* frame,
                                         const loom_low_packet_view_t* packet) {
  if (!loom_aie2p_bundle_plan_structural_move_isa(packet->node->op)) {
    return NULL;
  }
  return loom_low_allocation_find_packet_move_group_by_source_ordinal(
      &frame->allocation, packet->node->source_ordinal);
}

static bool loom_aie2p_bundle_plan_physical_control_descriptor(
    uint32_t descriptor_ordinal) {
  switch (descriptor_ordinal) {
    case AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT:
    case AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO:
    case AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO:
    case AIE2P_CORE_DESCRIPTOR_REF_RETURN_:
      return true;
    default:
      return false;
  }
}

static loom_aie2p_block_terminator_t loom_aie2p_bundle_plan_block_terminator(
    const loom_op_t* op) {
  if (loom_low_br_isa(op)) return LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH;
  if (loom_low_cond_br_isa(op)) {
    return LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH;
  }
  if (loom_low_return_isa(op)) return LOOM_AIE2P_BLOCK_TERMINATOR_RETURN;
  return LOOM_AIE2P_BLOCK_TERMINATOR_NONE;
}

static iree_status_t loom_aie2p_bundle_plan_analyze(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_bundle_plan_analysis_t* out_analysis) {
  *out_analysis = (loom_aie2p_bundle_plan_analysis_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  if (frame->target.descriptor_set != descriptor_set ||
      frame->schedule.target.descriptor_set != descriptor_set ||
      frame->allocation.target.descriptor_set != descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P bundle planning requires the amd.xdna.aie2p.core descriptor "
        "set");
  }
  if (frame->schedule.error_count != 0 || frame->allocation.error_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P bundle planning requires a successful "
                            "schedule and allocation");
  }
  if (frame->allocation.spill_count != 0 ||
      frame->allocation.spill_plan_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P bundle planning requires a spill-free "
                            "allocation");
  }
  if (frame->schedule.block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P Low leaf has no blocks");
  }

  loom_aie2p_block_analysis_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, frame->schedule.block_count, sizeof(*blocks), (void**)&blocks));
  memset(blocks, 0,
         frame->schedule.block_count * sizeof(loom_aie2p_block_analysis_t));
  for (iree_host_size_t block_index = 0;
       block_index < frame->schedule.block_count; ++block_index) {
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
    loom_aie2p_block_analysis_t* block_analysis = &blocks[block_index];
    block_analysis->terminator_packet_index =
        LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE;
    if (block->issue_group_count != 0) {
      const loom_low_schedule_issue_group_t* last_group =
          &frame->schedule.issue_groups[block->issue_group_start +
                                        block->issue_group_count - 1];
      block_analysis->maximum_issue_cycle = last_group->issue_cycle;
    }

    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
          &frame->schedule, (uint32_t)block_index, scheduled_ordinal);
      if (loom_low_packet_is_compile_time_only(&packet)) continue;
      if (packet.descriptor != NULL) {
        const uint32_t descriptor_ordinal =
            loom_aie2p_bundle_plan_descriptor_ordinal(descriptor_set,
                                                      packet.descriptor);
        if (!loom_aie2p_bundle_plan_physical_control_descriptor(
                descriptor_ordinal)) {
          continue;
        }
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P physical control is materialized from structural Low CFG; "
            "descriptor-backed control packets are not accepted");
      }
      if (loom_low_storage_address_isa(packet.node->op)) {
        if (out_analysis->storage_fixup_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AIE2P local-storage fixup count exceeds host size");
        }
        ++out_analysis->storage_fixup_count;
        continue;
      }
      if (loom_aie2p_bundle_plan_structural_move_isa(packet.node->op) ||
          loom_aie2p_bundle_plan_non_emitting_structural_isa(packet.node->op)) {
        continue;
      }

      const loom_aie2p_block_terminator_t terminator =
          loom_aie2p_bundle_plan_block_terminator(packet.node->op);
      if (terminator == LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P core emission does not lower structural operation %u",
            (unsigned)packet.node->op->kind);
      }
      if (block_analysis->terminator != LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P Low block %zu has more than one structural terminator",
            block_index);
      }
      block_analysis->terminator = terminator;
      block_analysis->terminator_packet_index = (uint32_t)packet.packet_index;
      block_analysis->terminator_issue_cycle = packet.node->issue_cycle;
    }

    if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P Low block %zu has no terminator",
                              block_index);
    }
    if (block_analysis->terminator_issue_cycle <
        block_analysis->maximum_issue_cycle) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P structural terminator does not complete Low block %zu",
          block_index);
    }
    if (!iree_host_size_checked_add(
            out_analysis->logical_issue_cycle_count,
            (iree_host_size_t)block_analysis->maximum_issue_cycle + 1u,
            &out_analysis->logical_issue_cycle_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P logical cycle count exceeds host size");
    }
    iree_host_size_t control_bundle_count = 0;
    switch (block_analysis->terminator) {
      case LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH:
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT);
        break;
      case LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH:
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT);
        control_bundle_count += iree_max(
            loom_aie2p_bundle_plan_control_bundle_count(
                descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO),
            loom_aie2p_bundle_plan_control_bundle_count(
                descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO));
        break;
      case LOOM_AIE2P_BLOCK_TERMINATOR_RETURN:
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_);
        break;
      case LOOM_AIE2P_BLOCK_TERMINATOR_NONE:
        IREE_ASSERT(false && "analyzed AIE2P block must have a terminator");
        break;
    }
    if (!iree_host_size_checked_add(out_analysis->control_bundle_capacity,
                                    control_bundle_count,
                                    &out_analysis->control_bundle_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P control capacity exceeds host size");
    }
    if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH) {
      const loom_low_packet_view_t terminator_packet = loom_low_packet_at(
          &frame->schedule, block_analysis->terminator_packet_index);
      const loom_low_allocation_edge_copy_group_t* edge_copy_group =
          loom_low_allocation_find_edge_copy_group_by_source_ordinal(
              &frame->allocation, terminator_packet.node->source_ordinal);
      if (edge_copy_group != NULL &&
          !iree_host_size_checked_add(out_analysis->edge_move_count,
                                      edge_copy_group->move_group.moves.count,
                                      &out_analysis->edge_move_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P edge move count exceeds host size");
      }
    }
  }
  out_analysis->blocks = blocks;
  return iree_ok_status();
}

static const loom_named_attr_t* loom_aie2p_bundle_plan_find_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) return &attrs.entries[i];
  }
  return NULL;
}

static int64_t loom_aie2p_bundle_plan_enum_value(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate, loom_attribute_t attr) {
  if (attr.kind == LOOM_ATTR_I64) return attr.i64;
  IREE_ASSERT(attr.kind == LOOM_ATTR_STRING);
  const iree_string_view_t token = module->strings.entries[attr.string_id];
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[immediate->enum_domain_id];
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const loom_low_enum_value_t* value =
        &descriptor_set->enum_values[domain->value_start + i];
    if (iree_string_view_equal(
            token, loom_low_descriptor_set_string(
                       descriptor_set, value->token_string_offset))) {
      return value->value;
    }
  }
  IREE_ASSERT(false && "verified enum token must be in its descriptor domain");
  return 0;
}

static iree_status_t loom_aie2p_bundle_plan_encode_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_allocation_assignment_t** operand_assignments =
      descriptor->operand_count
          ? (const loom_low_allocation_assignment_t**)iree_alloca(
                descriptor->operand_count * sizeof(*operand_assignments))
          : NULL;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (operand->encoding_field_id == 0) {
      operand_assignments[i] = NULL;
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(&frame->allocation,
                                                      packet, i);
    IREE_ASSERT(assignment->location_kind ==
                    LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
                "spill-free physical descriptor operand must be allocated");
    operand_assignments[i] = assignment;
  }

  int64_t* immediate_values =
      descriptor->immediate_count
          ? (int64_t*)iree_alloca(descriptor->immediate_count *
                                  sizeof(*immediate_values))
          : NULL;
  const loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    if (iree_any_bit_set(immediate->flags, LOOM_LOW_IMMEDIATE_FLAG_RELATIVE)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P relative immediates require packet-offset fixup planning");
    }
    const iree_string_view_t field_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    const loom_string_id_t field_name_id =
        loom_module_lookup_string(frame->module, field_name);
    const loom_named_attr_t* attr =
        loom_aie2p_bundle_plan_find_attr(attrs, field_name_id);
    if (attr == NULL) {
      IREE_ASSERT(iree_any_bit_set(immediate->flags,
                                   LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE));
      immediate_values[i] = immediate->default_value;
      continue;
    }
    if (attr->value.kind == LOOM_ATTR_SYMBOL) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P symbolic immediates require native object fixup planning");
    }
    if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ENUM) {
      immediate_values[i] = loom_aie2p_bundle_plan_enum_value(
          frame->module, descriptor_set, immediate, attr->value);
    } else {
      IREE_ASSERT(attr->value.kind == LOOM_ATTR_I64);
      immediate_values[i] = attr->value.i64;
    }
  }

  *out_encoded_slot = loom_aie2p_descriptor_encode(
      descriptor_set,
      loom_aie2p_bundle_plan_descriptor_ordinal(descriptor_set, descriptor),
      operand_assignments, immediate_values);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_encode_storage_address(
    const loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot,
    loom_storage_space_t* out_storage_space, uint64_t* out_byte_offset) {
  const loom_low_schedule_node_t* node = packet->node;
  IREE_ASSERT(loom_low_storage_address_isa(node->op));
  IREE_ASSERT_EQ(node->result_count, 1u);
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[0];
  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_allocation_assignment_for_value_ordinal(
          &builder->frame->allocation, result_ordinal, NULL);
  if (result_assignment == NULL ||
      result_assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      result_assignment->descriptor_reg_class_id !=
          AIE2P_CORE_REG_CLASS_ID_AIE2P_EP ||
      result_assignment->unit_count != 1 ||
      result_assignment->location_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P local-storage address requires one physical aie2p.ep register");
  }

  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors
           [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32];
  IREE_ASSERT_EQ(descriptor->operand_count, 1u);
  IREE_ASSERT_EQ(descriptor->result_count, 1u);
  IREE_ASSERT_EQ(descriptor->immediate_count, 1u);
  const loom_low_allocation_assignment_t* operand_assignments[] = {
      result_assignment,
  };
  const int64_t immediate_values[] = {0};
  *out_encoded_slot = loom_aie2p_descriptor_encode(
      builder->descriptor_set,
      AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
      operand_assignments, immediate_values);

  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &builder->frame->schedule.storage_layout, builder->frame->module,
      loom_low_storage_address_storage(node->op), &reference);
  const int64_t operation_offset = loom_low_storage_address_offset(node->op);
  IREE_ASSERT_GE(operation_offset, 0);
  uint64_t byte_offset = 0;
  if (!iree_checked_add_u64(reference.reservation.byte_offset,
                            reference.byte_offset, &byte_offset) ||
      !iree_checked_add_u64(byte_offset, (uint64_t)operation_offset,
                            &byte_offset) ||
      byte_offset > INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P local-storage address exceeds native fixup range");
  }
  *out_storage_space = reference.reservation.space;
  *out_byte_offset = byte_offset;
  return iree_ok_status();
}

static loom_aie2p_encoded_slot_t
loom_aie2p_bundle_plan_encode_structural_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    IREE_ASSERT(operand->encoding_field_id == 0 &&
                "structural descriptor operands must not encode bits");
  }
  IREE_ASSERT(descriptor->immediate_count == 0 &&
              "structural descriptors must not carry immediates");
  return loom_aie2p_descriptor_encode(descriptor_set, descriptor_ordinal, NULL,
                                      NULL);
}

static iree_status_t loom_aie2p_bundle_plan_encode_move(
    const loom_low_emission_frame_t* frame, const loom_low_move_t* move,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  IREE_ASSERT_EQ(move->destination.location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  IREE_ASSERT_EQ(move->source.location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  IREE_ASSERT_LT(move->destination.location,
                 descriptor_set->physical_register_count);
  IREE_ASSERT_LT(move->source.location,
                 descriptor_set->physical_register_count);
  if (!loom_low_descriptor_set_find_physical_register_candidate(
          descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
          move->destination.location, NULL) ||
      !loom_low_descriptor_set_find_physical_register_candidate(
          descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
          move->source.location, NULL)) {
    const iree_string_view_t destination_name = loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->physical_registers[move->destination.location]
            .name_string_offset);
    const iree_string_view_t source_name = loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->physical_registers[move->source.location]
            .name_string_offset);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P has no selected physical move route from %.*s to %.*s",
        (int)source_name.size, source_name.data, (int)destination_name.size,
        destination_name.data);
  }

  const loom_low_allocation_assignment_t destination_assignment = {
      .descriptor_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = move->destination.location,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t source_assignment = {
      .descriptor_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = move->source.location,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t* operand_assignments[] = {
      &destination_assignment,
      &source_assignment,
  };
  *out_encoded_slot = loom_aie2p_descriptor_encode(
      descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR,
      operand_assignments, NULL);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_slot(
    loom_aie2p_bundle_plan_builder_t* builder, loom_aie2p_planned_slot_t slot,
    iree_host_size_t* out_slot_index) {
  if (builder->slot_count >= builder->slot_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P bundle-plan slot capacity exhausted");
  }
  const iree_host_size_t slot_index = builder->slot_count++;
  builder->slots[slot_index] = slot;
  if (out_slot_index != NULL) *out_slot_index = slot_index;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_storage_fixup(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t scheduled_packet_index,
    loom_storage_space_t storage_space, uint64_t byte_offset) {
  if (builder->storage_fixup_count >= builder->storage_fixup_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P storage-fixup capacity exhausted");
  }
  const iree_host_size_t fixup_index = builder->storage_fixup_count++;
  builder->storage_fixups[fixup_index] = (loom_aie2p_planned_storage_fixup_t){
      .bundle_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
      .storage_space = storage_space,
      .byte_offset = byte_offset,
  };
  IREE_ASSERT_LT(scheduled_packet_index,
                 builder->frame->schedule.scheduled_node_count);
  IREE_ASSERT_EQ(
      builder->storage_fixup_indices_by_packet[scheduled_packet_index],
      UINT32_MAX);
  builder->storage_fixup_indices_by_packet[scheduled_packet_index] =
      (uint32_t)fixup_index;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_resolve_storage_fixups(
    loom_aie2p_bundle_plan_builder_t* builder) {
  for (iree_host_size_t bundle_index = 0; bundle_index < builder->bundle_count;
       ++bundle_index) {
    const loom_aie2p_planned_bundle_t* bundle = &builder->bundles[bundle_index];
    for (uint8_t slot_ordinal = 0; slot_ordinal < bundle->slot_count;
         ++slot_ordinal) {
      const loom_aie2p_planned_slot_t* slot =
          &builder->slots[bundle->slot_start + slot_ordinal];
      if (!iree_any_bit_set(
              slot->flags,
              LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS)) {
        continue;
      }
      if (slot->scheduled_packet_index >=
          builder->frame->schedule.scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AIE2P storage-address slot has no scheduled packet");
      }
      const uint32_t matched_fixup_index =
          builder
              ->storage_fixup_indices_by_packet[slot->scheduled_packet_index];
      if (matched_fixup_index == UINT32_MAX ||
          matched_fixup_index >= builder->storage_fixup_count ||
          builder->storage_fixups[matched_fixup_index].bundle_index !=
              LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AIE2P storage-address slot has no unique planned fixup");
      }
      builder->storage_fixups[matched_fixup_index].bundle_index =
          (uint32_t)bundle_index;
    }
  }
  for (iree_host_size_t i = 0; i < builder->storage_fixup_count; ++i) {
    if (builder->storage_fixups[i].bundle_index ==
        LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AIE2P planned storage fixup has no emitted instruction");
    }
  }
  return iree_ok_status();
}

static uint8_t loom_aie2p_bundle_plan_format_byte_length(
    loom_aie2p_bundle_format_id_t format, iree_host_size_t slot_count) {
  loom_aie2p_bundle_format_info_t format_info;
  const bool found =
      loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
  IREE_ASSERT(found && format_info.slot_count == slot_count &&
              format_info.bit_count % 8 == 0);
  return format_info.bit_count / 8;
}

static iree_status_t loom_aie2p_bundle_plan_append_bundle(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start, iree_host_size_t slot_count) {
  if (slot_count > LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u requires %zu physical slots in one "
        "ordered segment",
        (unsigned)logical_issue_cycle, slot_count);
  }
  loom_aie2p_slot_t physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    physical_slots[i] = builder->slots[slot_start + i].encoded_slot.slot;
  }
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(physical_slots,
                                                       slot_count);
  if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u has no legal physical bundle format",
        (unsigned)logical_issue_cycle);
  }
  if (builder->bundle_count >= builder->bundle_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P bundle-plan bundle capacity exhausted");
  }
  const uint8_t byte_length =
      loom_aie2p_bundle_plan_format_byte_length(format, slot_count);
  if (builder->encoded_byte_length >
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE - byte_length) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf exceeds 16 KiB of core program memory");
  }
  const iree_host_size_t bundle_index = builder->bundle_count++;
  builder->bundles[bundle_index] = (loom_aie2p_planned_bundle_t){
      .issue_cycle = (uint32_t)bundle_index,
      .block_index = builder->current_block_index,
      .logical_issue_cycle = logical_issue_cycle,
      .byte_offset = (uint32_t)builder->encoded_byte_length,
      .slot_start = (uint32_t)slot_start,
      .format = format,
      .slot_count = (uint8_t)slot_count,
  };
  builder->encoded_byte_length += byte_length;
  return iree_ok_status();
}

// Partitions one scheduled descriptor run into the minimum number of
// contiguous physical bundles. AIE2P's exact bundle-format domain is not
// downward closed: a wide format can exist while one of its slot subsets does
// not. Keeping the scheduled order while minimizing contiguous partitions
// preserves every dependency and timing separation without rejecting those
// representable runs.
static iree_status_t loom_aie2p_bundle_plan_append_descriptor_run(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start, iree_host_size_t slot_count) {
  IREE_ASSERT_GT(slot_count, 0u);
  if (slot_count > LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u contains %zu descriptor slots in one "
        "ordered run",
        (unsigned)logical_issue_cycle, slot_count);
  }

  uint8_t minimum_bundle_counts[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT + 1];
  uint8_t predecessor_indices[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT + 1];
  memset(minimum_bundle_counts, UINT8_MAX, sizeof(minimum_bundle_counts));
  memset(predecessor_indices, UINT8_MAX, sizeof(predecessor_indices));
  minimum_bundle_counts[0] = 0;
  for (iree_host_size_t end = 1; end <= slot_count; ++end) {
    for (iree_host_size_t begin = 0; begin < end; ++begin) {
      if (minimum_bundle_counts[begin] == UINT8_MAX) continue;
      loom_aie2p_slot_t
          physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
      const iree_host_size_t candidate_slot_count = end - begin;
      for (iree_host_size_t i = 0; i < candidate_slot_count; ++i) {
        physical_slots[i] =
            builder->slots[slot_start + begin + i].encoded_slot.slot;
      }
      if (loom_aie2p_encoding_find_bundle_format_for_slots(
              physical_slots, candidate_slot_count) ==
          LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
        continue;
      }
      const uint8_t candidate_bundle_count =
          (uint8_t)(minimum_bundle_counts[begin] + 1u);
      if (candidate_bundle_count < minimum_bundle_counts[end]) {
        minimum_bundle_counts[end] = candidate_bundle_count;
        predecessor_indices[end] = (uint8_t)begin;
      }
    }
  }
  if (minimum_bundle_counts[slot_count] == UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u has an unrepresentable descriptor run",
        (unsigned)logical_issue_cycle);
  }

  uint8_t partition_starts[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  uint8_t partition_ends[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  uint8_t partition_count = 0;
  for (uint8_t end = (uint8_t)slot_count; end != 0;) {
    const uint8_t begin = predecessor_indices[end];
    IREE_ASSERT_NE(begin, UINT8_MAX);
    partition_starts[partition_count] = begin;
    partition_ends[partition_count] = end;
    ++partition_count;
    end = begin;
  }
  while (partition_count != 0) {
    --partition_count;
    const iree_host_size_t begin = partition_starts[partition_count];
    const iree_host_size_t end = partition_ends[partition_count];
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
        builder, logical_issue_cycle, slot_start + begin, end - begin));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_nop_bundle(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle) {
  iree_host_size_t slot_start = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
              builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP),
          .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
      },
      &slot_start));
  return loom_aie2p_bundle_plan_append_bundle(builder, logical_issue_cycle,
                                              slot_start, 1);
}

static iree_status_t loom_aie2p_bundle_plan_try_place_return(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t return_packet_index,
    iree_host_size_t candidate_bundle_index, bool* out_placed) {
  *out_placed = false;
  loom_aie2p_planned_bundle_t* candidate_bundle =
      &builder->bundles[candidate_bundle_index];
  loom_aie2p_planned_slot_t return_slot = {
      .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
          builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
      .scheduled_packet_index = return_packet_index,
      .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
  };
  loom_aie2p_slot_t physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  const iree_host_size_t candidate_slot_start = candidate_bundle->slot_start;
  const bool replaces_nop =
      candidate_bundle->slot_count == 1 &&
      iree_any_bit_set(builder->slots[candidate_slot_start].flags,
                       LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
  const iree_host_size_t retained_slot_count =
      replaces_nop ? 0 : candidate_bundle->slot_count;
  if (retained_slot_count == LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < retained_slot_count; ++i) {
    physical_slots[i] =
        builder->slots[candidate_slot_start + i].encoded_slot.slot;
  }
  physical_slots[retained_slot_count] = return_slot.encoded_slot.slot;
  const iree_host_size_t candidate_slot_count = retained_slot_count + 1;
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(physical_slots,
                                                       candidate_slot_count);
  if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
    return iree_ok_status();
  }

  const uint8_t old_byte_length = loom_aie2p_bundle_plan_format_byte_length(
      candidate_bundle->format, candidate_bundle->slot_count);
  const uint8_t new_byte_length =
      loom_aie2p_bundle_plan_format_byte_length(format, candidate_slot_count);
  if (new_byte_length > old_byte_length &&
      builder->encoded_byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE -
                                         (new_byte_length - old_byte_length)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf exceeds 16 KiB of core program memory");
  }
  if (replaces_nop) {
    builder->slots[candidate_slot_start] = return_slot;
  } else {
    if (builder->slot_count >= builder->slot_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P bundle-plan slot capacity exhausted");
    }
    const iree_host_size_t insert_index =
        candidate_slot_start + candidate_bundle->slot_count;
    memmove(&builder->slots[insert_index + 1], &builder->slots[insert_index],
            (builder->slot_count - insert_index) * sizeof(*builder->slots));
    builder->slots[insert_index] = return_slot;
    ++builder->slot_count;
    for (iree_host_size_t i = candidate_bundle_index + 1;
         i < builder->bundle_count; ++i) {
      ++builder->bundles[i].slot_start;
    }
  }
  if (new_byte_length != old_byte_length) {
    const int32_t byte_delta =
        (int32_t)new_byte_length - (int32_t)old_byte_length;
    for (iree_host_size_t i = candidate_bundle_index + 1;
         i < builder->bundle_count; ++i) {
      builder->bundles[i].byte_offset =
          (uint32_t)((int32_t)builder->bundles[i].byte_offset + byte_delta);
    }
    builder->encoded_byte_length =
        builder->encoded_byte_length - old_byte_length + new_byte_length;
  }
  candidate_bundle->format = format;
  candidate_bundle->slot_count = (uint8_t)candidate_slot_count;
  *out_placed = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_encode_branch(
    const loom_aie2p_bundle_plan_builder_t* builder,
    uint32_t scheduled_packet_index, uint32_t descriptor_ordinal,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors[descriptor_ordinal];
  IREE_ASSERT(descriptor->immediate_count == 1 &&
              "AIE2P structural branch must carry one target immediate");
  const loom_low_allocation_assignment_t* operand_assignments[1] = {NULL};
  if (descriptor->operand_count != 0) {
    IREE_ASSERT(descriptor->operand_count == 1 &&
                "AIE2P conditional branch must carry one condition");
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&builder->frame->schedule, scheduled_packet_index);
    IREE_ASSERT(loom_low_cond_br_isa(packet.node->op));
    const loom_value_ordinal_t condition_ordinal =
        loom_low_schedule_node_const_operand_ordinals(packet.node)[0];
    const loom_low_allocation_assignment_t* condition_assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            &builder->frame->allocation, condition_ordinal, NULL);
    if (condition_assignment == NULL ||
        condition_assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        condition_assignment->descriptor_reg_class_id !=
            AIE2P_CORE_REG_CLASS_ID_AIE2P_ER ||
        condition_assignment->unit_count != 1 ||
        condition_assignment->location_count != 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P conditional branch requires one physical aie2p.er register");
    }
    operand_assignments[0] = condition_assignment;
  }
  const int64_t target_immediate[] = {0};
  *out_encoded_slot =
      loom_aie2p_descriptor_encode(builder->descriptor_set, descriptor_ordinal,
                                   operand_assignments, target_immediate);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_edge_moves(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* terminator_packet) {
  const loom_low_allocation_edge_copy_group_t* edge_copy_group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          &builder->frame->allocation, terminator_packet->node->source_ordinal);
  if (edge_copy_group == NULL) return iree_ok_status();
  for (iree_host_size_t move_ordinal = 0;
       move_ordinal < edge_copy_group->move_group.moves.count; ++move_ordinal) {
    const iree_host_size_t move_index =
        edge_copy_group->move_group.moves.start + move_ordinal;
    loom_aie2p_encoded_slot_t encoded_move;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_move(
        builder->frame, &builder->frame->allocation.moves[move_index],
        &encoded_move));
    iree_host_size_t slot_start = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
        builder,
        (loom_aie2p_planned_slot_t){
            .encoded_slot = encoded_move,
            .scheduled_packet_index = (uint32_t)terminator_packet->packet_index,
            .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
        },
        &slot_start));
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
        builder, terminator_packet->node->issue_cycle, slot_start, 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_branch(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t scheduled_packet_index,
    uint32_t descriptor_ordinal, uint32_t target_block_index,
    uint32_t logical_issue_cycle) {
  IREE_ASSERT_LT(target_block_index, builder->block_count);
  loom_aie2p_encoded_slot_t encoded_branch;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_branch(
      builder, scheduled_packet_index, descriptor_ordinal, &encoded_branch));
  iree_host_size_t slot_start = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = encoded_branch,
          .scheduled_packet_index = scheduled_packet_index,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
      },
      &slot_start));
  const iree_host_size_t bundle_index = builder->bundle_count;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
      builder, logical_issue_cycle, slot_start, 1));
  if (builder->branch_fixup_count >= builder->branch_fixup_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P branch-fixup capacity exhausted");
  }
  builder->branch_fixups[builder->branch_fixup_count++] =
      (loom_aie2p_planned_branch_fixup_t){
          .bundle_index = (uint32_t)bundle_index,
          .target_block_index = target_block_index,
      };

  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors[descriptor_ordinal];
  const uint8_t delay_slot_count =
      loom_aie2p_bundle_plan_instruction_info(descriptor->encoding_id)
          .delay_slot_count;
  for (uint8_t i = 0; i < delay_slot_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_nop_bundle(builder, logical_issue_cycle));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_return(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_block_analysis_t* block_analysis,
    iree_host_size_t block_bundle_start) {
  const loom_low_descriptor_t* return_descriptor =
      &builder->descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_RETURN_];
  const uint8_t delay_slot_count =
      loom_aie2p_bundle_plan_instruction_info(return_descriptor->encoding_id)
          .delay_slot_count;
  bool return_placed = false;
  iree_host_size_t candidate_bundle_index = block_bundle_start;
  if (builder->bundle_count != block_bundle_start) {
    const iree_host_size_t block_bundle_count =
        builder->bundle_count - block_bundle_start;
    candidate_bundle_index = block_bundle_count > delay_slot_count
                                 ? builder->bundle_count - 1u - delay_slot_count
                                 : block_bundle_start;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_place_return(
        builder, block_analysis->terminator_packet_index,
        candidate_bundle_index, &return_placed));
  }
  if (return_placed) {
    const iree_host_size_t required_bundle_count =
        candidate_bundle_index + delay_slot_count + 1u;
    while (builder->bundle_count < required_bundle_count) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          builder, block_analysis->terminator_issue_cycle));
    }
    return iree_ok_status();
  }

  iree_host_size_t return_slot_start = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
              builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
          .scheduled_packet_index = block_analysis->terminator_packet_index,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
      },
      &return_slot_start));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
      builder, block_analysis->terminator_issue_cycle, return_slot_start, 1));
  for (uint8_t i = 0; i < delay_slot_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
        builder, block_analysis->terminator_issue_cycle));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_align_next_block(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle) {
  if ((builder->encoded_byte_length & 1u) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P bundle stream cannot reach 16-byte block alignment");
  }
  while ((builder->encoded_byte_length & 15u) != 0) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_nop_bundle(builder, logical_issue_cycle));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_terminator(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_block_analysis_t* block_analysis,
    iree_host_size_t block_bundle_start) {
  const loom_low_packet_view_t packet = loom_low_packet_at(
      &builder->frame->schedule, block_analysis->terminator_packet_index);
  const uint32_t block_index = builder->current_block_index;
  const uint32_t next_block_index = block_index + 1u < builder->block_count
                                        ? block_index + 1u
                                        : LOOM_LOW_PACKET_INDEX_NONE;
  switch (block_analysis->terminator) {
    case LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH: {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_bundle_plan_append_edge_moves(builder, &packet));
      const uint32_t target_block_index = loom_low_packet_block_index(
          &builder->frame->schedule, loom_low_br_dest(packet.node->op));
      if (target_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "AIE2P branch target is outside its function");
      }
      if (target_block_index == next_block_index) return iree_ok_status();
      return loom_aie2p_bundle_plan_append_branch(
          builder, (uint32_t)packet.packet_index,
          AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT, target_block_index,
          block_analysis->terminator_issue_cycle);
    }
    case LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH: {
      const uint32_t true_block_index = loom_low_packet_block_index(
          &builder->frame->schedule,
          loom_low_cond_br_true_dest(packet.node->op));
      const uint32_t false_block_index = loom_low_packet_block_index(
          &builder->frame->schedule,
          loom_low_cond_br_false_dest(packet.node->op));
      if (true_block_index == LOOM_LOW_PACKET_INDEX_NONE ||
          false_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P conditional branch target is outside its function");
      }
      if (true_block_index == false_block_index) {
        if (true_block_index == next_block_index) return iree_ok_status();
        return loom_aie2p_bundle_plan_append_branch(
            builder, (uint32_t)packet.packet_index,
            AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT, true_block_index,
            block_analysis->terminator_issue_cycle);
      }
      if (false_block_index == next_block_index) {
        return loom_aie2p_bundle_plan_append_branch(
            builder, (uint32_t)packet.packet_index,
            AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO, true_block_index,
            block_analysis->terminator_issue_cycle);
      }
      if (true_block_index == next_block_index) {
        return loom_aie2p_bundle_plan_append_branch(
            builder, (uint32_t)packet.packet_index,
            AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO, false_block_index,
            block_analysis->terminator_issue_cycle);
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_branch(
          builder, (uint32_t)packet.packet_index,
          AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO, true_block_index,
          block_analysis->terminator_issue_cycle));
      return loom_aie2p_bundle_plan_append_branch(
          builder, (uint32_t)packet.packet_index,
          AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT, false_block_index,
          block_analysis->terminator_issue_cycle);
    }
    case LOOM_AIE2P_BLOCK_TERMINATOR_RETURN:
      return loom_aie2p_bundle_plan_append_return(builder, block_analysis,
                                                  block_bundle_start);
    case LOOM_AIE2P_BLOCK_TERMINATOR_NONE:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "AIE2P block terminator analysis is invalid");
}

iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_bundle_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_aie2p_bundle_plan_t){0};

  loom_aie2p_bundle_plan_analysis_t analysis;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze(frame, arena, &analysis));
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  if (frame->schedule.block_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P block count exceeds target index range");
  }
  iree_host_size_t alignment_bundle_capacity = 0;
  if (frame->schedule.block_count > 1) {
    const iree_host_size_t boundary_count = frame->schedule.block_count - 1u;
    const uint8_t nop_byte_length =
        loom_aie2p_bundle_plan_single_slot_byte_length(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP);
    IREE_ASSERT(nop_byte_length != 0 && nop_byte_length < 16u &&
                16u % nop_byte_length == 0 &&
                "standalone AIE2P NOP must reach block alignment");
    const iree_host_size_t maximum_boundary_nop_count =
        16u / nop_byte_length - 1u;
    if (boundary_count > IREE_HOST_SIZE_MAX / maximum_boundary_nop_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P block alignment capacity exceeds host "
                              "size");
    }
    alignment_bundle_capacity = boundary_count * maximum_boundary_nop_count;
  }
  iree_host_size_t bundle_capacity = analysis.logical_issue_cycle_count;
  if (!iree_host_size_checked_add(bundle_capacity,
                                  frame->schedule.scheduled_node_count,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(bundle_capacity,
                                  frame->allocation.packet_move_count,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(bundle_capacity, analysis.edge_move_count,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(bundle_capacity,
                                  analysis.control_bundle_capacity,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(bundle_capacity, alignment_bundle_capacity,
                                  &bundle_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P bundle-plan storage capacity exceeds host size");
  }
  // The same upper bound covers slots: each scheduled node and allocation
  // move can contribute at most one slot, while every empty logical cycle and
  // return-delay cycle contributes one synthetic slot.
  const iree_host_size_t slot_capacity = bundle_capacity;
  if (frame->schedule.block_count > IREE_HOST_SIZE_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P branch-fixup capacity exceeds host size");
  }
  const iree_host_size_t branch_fixup_capacity =
      frame->schedule.block_count * 2u;

  loom_aie2p_bundle_plan_builder_t builder = {
      .frame = frame,
      .descriptor_set = descriptor_set,
      .bundle_capacity = bundle_capacity,
      .slot_capacity = slot_capacity,
      .branch_fixup_capacity = branch_fixup_capacity,
      .storage_fixup_capacity = analysis.storage_fixup_count,
      .block_count = frame->schedule.block_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, builder.bundle_capacity,
                                                 sizeof(*builder.bundles),
                                                 (void**)&builder.bundles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, builder.slot_capacity,
                                                 sizeof(*builder.slots),
                                                 (void**)&builder.slots));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, builder.branch_fixup_capacity, sizeof(*builder.branch_fixups),
      (void**)&builder.branch_fixups));
  if (builder.storage_fixup_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, builder.storage_fixup_capacity, sizeof(*builder.storage_fixups),
        (void**)&builder.storage_fixups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, frame->schedule.scheduled_node_count,
        sizeof(*builder.storage_fixup_indices_by_packet),
        (void**)&builder.storage_fixup_indices_by_packet));
    for (iree_host_size_t i = 0; i < frame->schedule.scheduled_node_count;
         ++i) {
      builder.storage_fixup_indices_by_packet[i] = UINT32_MAX;
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, builder.block_count, sizeof(*builder.block_byte_offsets),
      (void**)&builder.block_byte_offsets));
  for (uint32_t block_index = 0;
       block_index < (uint32_t)frame->schedule.block_count; ++block_index) {
    if (block_index != 0) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_align_next_block(
          &builder, analysis.blocks[block_index - 1u].terminator_issue_cycle));
    }
    builder.current_block_index = block_index;
    builder.block_byte_offsets[block_index] =
        (uint32_t)builder.encoded_byte_length;
    const iree_host_size_t block_bundle_start = builder.bundle_count;
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
    const loom_aie2p_block_analysis_t* block_analysis =
        &analysis.blocks[block_index];
    uint32_t issue_group_index = 0;
    for (uint32_t issue_cycle = 0;
         issue_cycle <= block_analysis->maximum_issue_cycle; ++issue_cycle) {
      const loom_low_schedule_issue_group_t* group = NULL;
      if (issue_group_index < block->issue_group_count) {
        const loom_low_schedule_issue_group_t* next_group =
            &frame->schedule
                 .issue_groups[block->issue_group_start + issue_group_index];
        if (next_group->issue_cycle == issue_cycle) {
          group = next_group;
          ++issue_group_index;
        }
      }
      const iree_host_size_t cycle_bundle_start = builder.bundle_count;
      iree_host_size_t segment_slot_start = builder.slot_count;
      if (group != NULL) {
        for (uint32_t i = 0; i < group->scheduled_node_count; ++i) {
          const uint32_t packet_index = group->scheduled_node_start + i;
          const loom_low_packet_view_t packet =
              loom_low_packet_at(&frame->schedule, packet_index);
          if (loom_low_packet_is_compile_time_only(&packet) ||
              loom_aie2p_bundle_plan_block_terminator(packet.node->op) !=
                  LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
            continue;
          }
          if (packet.descriptor != NULL) {
            loom_aie2p_encoded_slot_t encoded_slot;
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_packet(
                frame, &packet, &encoded_slot));
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
                &builder,
                (loom_aie2p_planned_slot_t){
                    .encoded_slot = encoded_slot,
                    .scheduled_packet_index = packet_index,
                },
                NULL));
            continue;
          }
          if (loom_low_storage_address_isa(packet.node->op)) {
            loom_aie2p_encoded_slot_t encoded_slot;
            loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_STACK;
            uint64_t storage_byte_offset = 0;
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_storage_address(
                &builder, &packet, &encoded_slot, &storage_space,
                &storage_byte_offset));
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
                &builder,
                (loom_aie2p_planned_slot_t){
                    .encoded_slot = encoded_slot,
                    .scheduled_packet_index = packet_index,
                    .flags =
                        LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS,
                },
                NULL));
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_storage_fixup(
                &builder, packet_index, storage_space, storage_byte_offset));
            continue;
          }
          if (loom_aie2p_bundle_plan_non_emitting_structural_isa(
                  packet.node->op)) {
            continue;
          }

          const loom_low_allocation_packet_move_group_t* move_group =
              loom_aie2p_bundle_plan_packet_move_group(frame, &packet);
          if (move_group == NULL) continue;
          if (builder.slot_count != segment_slot_start) {
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_descriptor_run(
                &builder, issue_cycle, segment_slot_start,
                builder.slot_count - segment_slot_start));
            segment_slot_start = builder.slot_count;
          }
          for (iree_host_size_t move_ordinal = 0;
               move_ordinal < move_group->move_group.moves.count;
               ++move_ordinal) {
            const iree_host_size_t move_index =
                move_group->move_group.moves.start + move_ordinal;
            loom_aie2p_encoded_slot_t encoded_move;
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_move(
                frame, &frame->allocation.moves[move_index], &encoded_move));
            iree_host_size_t move_slot_start = 0;
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_slot(
                &builder,
                (loom_aie2p_planned_slot_t){
                    .encoded_slot = encoded_move,
                    .scheduled_packet_index = packet_index,
                    .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
                },
                &move_slot_start));
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
                &builder, issue_cycle, move_slot_start, 1));
            segment_slot_start = builder.slot_count;
          }
        }
      }
      if (builder.slot_count != segment_slot_start) {
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_descriptor_run(
            &builder, issue_cycle, segment_slot_start,
            builder.slot_count - segment_slot_start));
      }
      if (builder.bundle_count == cycle_bundle_start &&
          issue_cycle != block_analysis->terminator_issue_cycle) {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_bundle_plan_append_nop_bundle(&builder, issue_cycle));
      }
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_terminator(
        &builder, block_analysis, block_bundle_start));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_resolve_storage_fixups(&builder));

  *out_plan = (loom_aie2p_bundle_plan_t){
      .frame = frame,
      .block_byte_offsets = builder.block_byte_offsets,
      .block_count = builder.block_count,
      .bundles = builder.bundles,
      .bundle_count = builder.bundle_count,
      .slots = builder.slots,
      .slot_count = builder.slot_count,
      .branch_fixups = builder.branch_fixups,
      .branch_fixup_count = builder.branch_fixup_count,
      .storage_fixups = builder.storage_fixups,
      .storage_fixup_count = builder.storage_fixup_count,
      .encoded_byte_length = builder.encoded_byte_length,
  };
  return iree_ok_status();
}
