// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/physical_issue.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

typedef enum loom_aie2p_block_terminator_e {
  LOOM_AIE2P_BLOCK_TERMINATOR_NONE = 0,
  LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH = 1,
  LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH = 2,
  LOOM_AIE2P_BLOCK_TERMINATOR_RETURN = 3,
} loom_aie2p_block_terminator_t;

typedef struct loom_aie2p_block_branch_t {
  // Native branch descriptor selected for the block terminator.
  uint32_t descriptor_ordinal;
  // Source-order destination block used by the final branch fixup.
  uint32_t target_block_index;
} loom_aie2p_block_branch_t;

typedef struct loom_aie2p_block_analysis_t {
  // Structural terminator packet ending this block.
  uint32_t terminator_packet_index;
  // Logical issue cycle at which the structural terminator completes.
  uint32_t terminator_issue_cycle;
  // Greatest logical issue cycle in this scheduled block.
  uint32_t maximum_issue_cycle;
  // Structural control kind selected for this block.
  loom_aie2p_block_terminator_t terminator;
  // Selected physical branches, shared by destination alignment and emission.
  struct {
    // Native branches in emission order; fallthrough contributes no entry.
    loom_aie2p_block_branch_t values[2];
    // Number of selected native branches, from zero through two.
    uint8_t count;
  } branches;
  // Entry or an actual native branch destination requiring 16-byte alignment.
  bool requires_alignment;
} loom_aie2p_block_analysis_t;

typedef struct loom_aie2p_bundle_plan_analysis_t {
  // Per-block control and cycle summaries in source block order.
  loom_aie2p_block_analysis_t* blocks;
  // Total logical issue cycles across all source blocks.
  iree_host_size_t logical_issue_cycle_count;
  // Native instruction slots required by packet and edge allocation moves.
  iree_host_size_t move_slot_count;
  // Conservative branch, return, and delay-bundle capacity.
  iree_host_size_t control_bundle_capacity;
  // Structural local-storage addresses requiring placement fixups.
  iree_host_size_t storage_fixup_count;
} loom_aie2p_bundle_plan_analysis_t;

// Native bindings retained until physical issue admission. The slot's
// STRUCTURAL_MOVE flag selects move operands instead of a descriptor ordinal.
typedef union loom_aie2p_slot_realization_t {
  // Selected descriptor for an ordinary instruction.
  uint32_t descriptor_ordinal;
  // Actual native registers after allocation-move decomposition.
  loom_aie2p_register_move_t move;
} loom_aie2p_slot_realization_t;

static_assert(sizeof(loom_aie2p_slot_realization_t) == sizeof(uint32_t),
              "native slot bindings must remain four bytes");

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
  // Next physical issue cycle, including implicit NOP gaps.
  uint32_t next_issue_cycle;
  // Physical origin of this block's logical cycles, fixed at block entry.
  uint32_t block_issue_cycle;
  // First scheduled packet not yet represented by a native or virtual issue.
  uint32_t next_source_packet;
  // Shared physical event and resource admission state.
  loom_low_physical_issue_t issue;
  // Scratch native bindings per slot. Not retained by the final plan.
  loom_aie2p_slot_realization_t* slot_realizations;
  // Borrowed instruction views for one hardware-bounded issue group.
  loom_low_physical_instruction_t
      instructions[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  // Fixed per-group physical binding scratch, sized from generated arity.
  uint16_t* instruction_registers;
  // Accumulated native writes, copied into the final plan.
  loom_aie2p_register_unit_set_t register_writes;
} loom_aie2p_bundle_plan_builder_t;

static iree_status_t loom_aie2p_bundle_plan_copy_function_name(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_string_view_t* out_function_name) {
  *out_function_name = iree_string_view_empty();
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  if (iree_string_view_is_empty(function_name)) {
    return iree_ok_status();
  }
  char* function_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, function_name.size,
                                           (void**)&function_name_data));
  memcpy(function_name_data, function_name.data, function_name.size);
  *out_function_name =
      iree_make_string_view(function_name_data, function_name.size);
  return iree_ok_status();
}

static void loom_aie2p_bundle_plan_retain_storage_requirements(
    const loom_low_emission_frame_t* frame,
    loom_aie2p_leaf_program_plan_t* plan) {
  for (loom_storage_space_t space = 0; space < LOOM_STORAGE_SPACE_COUNT_;
       ++space) {
    const loom_low_storage_layout_requirement_t requirement =
        loom_low_storage_layout_requirement(
            &frame->schedule.requirements.storage_layout, space);
    plan->storage_requirements[space] = (loom_aie2p_leaf_storage_requirement_t){
        .byte_length = requirement.byte_length,
        .minimum_alignment = requirement.minimum_alignment,
    };
  }
  plan->spill = (loom_aie2p_leaf_storage_requirement_t){
      .byte_length = frame->materialized_spill_storage_bytes,
      .minimum_alignment = frame->materialized_spill_storage_minimum_alignment,
  };
}

static iree_status_t loom_aie2p_bundle_plan_retain_resource_imports(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* plan) {
  const loom_low_function_requirements_t* requirements =
      &frame->schedule.requirements;
  const iree_host_size_t resource_count = requirements->resource_count;
  if (resource_count == 0) {
    return iree_ok_status();
  }

  loom_aie2p_leaf_resource_import_t* resources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resources), (void**)&resources));
  for (iree_host_size_t resource_index = 0; resource_index < resource_count;
       ++resource_index) {
    const loom_op_t* op = requirements->resources[resource_index];

    const loom_low_schedule_node_t* node =
        loom_low_schedule_node_for_op(&frame->schedule, op);
    IREE_ASSERT(node != NULL && node->result_count == 1);
    const loom_low_packet_view_t packet = loom_low_packet_at_node(
        &frame->schedule, (uint32_t)(node - frame->schedule.nodes));
    const loom_low_allocation_assignment_t* result_assignment =
        loom_low_packet_result_assignment(&frame->allocation, &packet, 0);
    IREE_ASSERT(result_assignment != NULL);
    IREE_ASSERT_EQ(result_assignment->location_kind,
                   LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);

    loom_aie2p_leaf_resource_flags_t flags = 0;
    uint64_t extent = 0;
    if (loom_low_resource_has_extent(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT;
      extent = (uint64_t)loom_low_resource_extent(op);
    }
    uint32_t cache_swizzle_stride = 0;
    if (loom_low_resource_has_cache_swizzle_stride(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE;
      cache_swizzle_stride =
          (uint32_t)loom_low_resource_cache_swizzle_stride(op);
    }

    uint32_t extent_physical_register = UINT32_MAX;
    uint16_t extent_descriptor_register_class_id = 0;
    uint32_t extent_physical_register_count = 0;
    if (loom_low_resource_extent_value_is_present(op)) {
      flags |= LOOM_AIE2P_LEAF_RESOURCE_FLAG_DYNAMIC_EXTENT;
      const loom_low_allocation_assignment_t* extent_assignment =
          loom_low_packet_operand_assignment(&frame->allocation, &packet, 0);
      IREE_ASSERT(extent_assignment != NULL);
      IREE_ASSERT_EQ(extent_assignment->location_kind,
                     LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
      extent_physical_register = extent_assignment->location_base;
      extent_descriptor_register_class_id =
          extent_assignment->descriptor_reg_class_id;
      extent_physical_register_count = extent_assignment->location_count;
    }

    const loom_type_id_t source_type_id = loom_low_resource_source_type(op);
    IREE_ASSERT_LT(source_type_id, frame->module->types.count);
    resources[resource_index] = (loom_aie2p_leaf_resource_import_t){
        .index = (uint64_t)loom_low_resource_index(op),
        .extent = extent,
        .cache_swizzle_stride = cache_swizzle_stride,
        .physical_register = result_assignment->location_base,
        .physical_register_count = result_assignment->location_count,
        .extent_physical_register = extent_physical_register,
        .descriptor_register_class_id =
            result_assignment->descriptor_reg_class_id,
        .extent_descriptor_register_class_id =
            extent_descriptor_register_class_id,
        .extent_physical_register_count = extent_physical_register_count,
        .flags = flags,
        .import_kind = loom_low_resource_import_kind(op),
        .source_type_kind = loom_type_kind(
            loom_type_table_get(&frame->module->types, source_type_id)),
    };
  }
  plan->resource_imports = resources;
  plan->resource_import_count = resource_count;
  return iree_ok_status();
}

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

static const loom_low_allocation_packet_move_group_t*
loom_aie2p_bundle_plan_packet_move_group(const loom_low_emission_frame_t* frame,
                                         const loom_low_packet_view_t* packet) {
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

static loom_aie2p_block_terminator_t
loom_aie2p_bundle_plan_terminator_from_structure_kind(
    loom_aie2p_core_structure_kind_t structure_kind) {
  switch (structure_kind) {
    case LOOM_AIE2P_CORE_STRUCTURE_BRANCH:
      return LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH;
    case LOOM_AIE2P_CORE_STRUCTURE_CONDITIONAL_BRANCH:
      return LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH;
    case LOOM_AIE2P_CORE_STRUCTURE_RETURN:
      return LOOM_AIE2P_BLOCK_TERMINATOR_RETURN;
    default:
      return LOOM_AIE2P_BLOCK_TERMINATOR_NONE;
  }
}

static iree_host_size_t loom_aie2p_bundle_plan_move_slot_count(
    const loom_low_emission_frame_t* frame, loom_low_move_range_t range) {
  iree_host_size_t slot_count = 0;
  for (iree_host_size_t i = 0; i < range.count; ++i) {
    const loom_low_move_t* move = &frame->allocation.moves[range.start + i];
    loom_aie2p_register_move_t parts[2];
    slot_count += loom_aie2p_descriptor_move_parts(
        (loom_aie2p_register_move_t){
            .source = (loom_aie2p_physical_register_id_t)move->source.location,
            .destination =
                (loom_aie2p_physical_register_id_t)move->destination.location,
        },
        parts);
  }
  return slot_count;
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
  if (frame->schedule.block_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P block count exceeds target index range");
  }

  loom_aie2p_block_analysis_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, frame->schedule.block_count, sizeof(*blocks), (void**)&blocks));
  memset(blocks, 0,
         frame->schedule.block_count * sizeof(loom_aie2p_block_analysis_t));
  blocks[0].requires_alignment = true;
  // Destination marks may be set before a block's own analysis. Each iteration
  // preserves those marks in the once-initialized table.
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
      if (loom_low_packet_is_compile_time_only(&packet)) {
        continue;
      }
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
      const loom_aie2p_core_structure_kind_t structure_kind =
          loom_aie2p_core_structure_classify(packet.node->op);
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS) {
        if (out_analysis->storage_fixup_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AIE2P local-storage fixup count exceeds host size");
        }
        ++out_analysis->storage_fixup_count;
        continue;
      }
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE) {
        const loom_low_allocation_packet_move_group_t* group =
            loom_aie2p_bundle_plan_packet_move_group(frame, &packet);
        if (group != NULL) {
          out_analysis->move_slot_count +=
              loom_aie2p_bundle_plan_move_slot_count(frame,
                                                     group->move_group.moves);
        }
        continue;
      }
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_DECLARATION) {
        continue;
      }

      const loom_aie2p_block_terminator_t terminator =
          loom_aie2p_bundle_plan_terminator_from_structure_kind(structure_kind);
      if (terminator == LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
        IREE_ASSERT_UNREACHABLE(
            "AIE2P core verification must reject unsupported structural "
            "operations");
        IREE_BUILTIN_UNREACHABLE();
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
    const loom_low_packet_view_t terminator_packet = loom_low_packet_at(
        &frame->schedule, block_analysis->terminator_packet_index);
    const uint32_t next_block_index =
        block_index + 1u < frame->schedule.block_count
            ? (uint32_t)(block_index + 1u)
            : LOOM_LOW_PACKET_INDEX_NONE;
    iree_host_size_t control_bundle_count = 0;
    switch (block_analysis->terminator) {
      case LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH: {
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT);
        const uint32_t target_block_index = loom_low_packet_block_index(
            &frame->schedule, loom_low_br_dest(terminator_packet.node->op));
        if (target_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P branch target is outside its function");
        }
        if (target_block_index != next_block_index) {
          block_analysis->branches.values[block_analysis->branches.count++] =
              (loom_aie2p_block_branch_t){
                  .descriptor_ordinal = AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT,
                  .target_block_index = target_block_index,
              };
        }
        break;
      }
      case LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH: {
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT);
        control_bundle_count += iree_max(
            loom_aie2p_bundle_plan_control_bundle_count(
                descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO),
            loom_aie2p_bundle_plan_control_bundle_count(
                descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO));
        const uint32_t true_block_index = loom_low_packet_block_index(
            &frame->schedule,
            loom_low_cond_br_true_dest(terminator_packet.node->op));
        const uint32_t false_block_index = loom_low_packet_block_index(
            &frame->schedule,
            loom_low_cond_br_false_dest(terminator_packet.node->op));
        if (true_block_index == LOOM_LOW_PACKET_INDEX_NONE ||
            false_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AIE2P conditional branch target is outside its function");
        }
        if (true_block_index == false_block_index) {
          if (true_block_index != next_block_index) {
            block_analysis->branches.values[block_analysis->branches.count++] =
                (loom_aie2p_block_branch_t){
                    .descriptor_ordinal =
                        AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT,
                    .target_block_index = true_block_index,
                };
          }
        } else if (false_block_index == next_block_index) {
          block_analysis->branches.values[block_analysis->branches.count++] =
              (loom_aie2p_block_branch_t){
                  .descriptor_ordinal =
                      AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO,
                  .target_block_index = true_block_index,
              };
        } else if (true_block_index == next_block_index) {
          block_analysis->branches.values[block_analysis->branches.count++] =
              (loom_aie2p_block_branch_t){
                  .descriptor_ordinal = AIE2P_CORE_DESCRIPTOR_REF_BRANCH_ZERO,
                  .target_block_index = false_block_index,
              };
        } else {
          block_analysis->branches.values[block_analysis->branches.count++] =
              (loom_aie2p_block_branch_t){
                  .descriptor_ordinal =
                      AIE2P_CORE_DESCRIPTOR_REF_BRANCH_NONZERO,
                  .target_block_index = true_block_index,
              };
          block_analysis->branches.values[block_analysis->branches.count++] =
              (loom_aie2p_block_branch_t){
                  .descriptor_ordinal = AIE2P_CORE_DESCRIPTOR_REF_BRANCH_DIRECT,
                  .target_block_index = false_block_index,
              };
        }
        break;
      }
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
    for (uint8_t i = 0; i < block_analysis->branches.count; ++i) {
      blocks[block_analysis->branches.values[i].target_block_index]
          .requires_alignment = true;
    }
    if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH) {
      const loom_low_allocation_edge_copy_group_t* edge_copy_group =
          loom_low_allocation_find_edge_copy_group_by_source_ordinal(
              &frame->allocation, terminator_packet.node->source_ordinal);
      if (edge_copy_group != NULL) {
        out_analysis->move_slot_count += loom_aie2p_bundle_plan_move_slot_count(
            frame, edge_copy_group->move_group.moves);
      }
    }
  }
  out_analysis->blocks = blocks;
  return iree_ok_status();
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
  // Required core immediates and canonical IR dictionaries share field-name
  // order. Construction and Low verification establish their positional
  // binding.
  const loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_attribute_t value = attrs.entries[i].value;
    if (value.kind == LOOM_ATTR_SYMBOL) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P symbolic immediates require native object fixup planning");
    }
    immediate_values[i] = value.i64;
  }

  *out_encoded_slot =
      loom_aie2p_descriptor_encode(descriptor_set, packet->descriptor_ordinal,
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
  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_packet_result_assignment(&builder->frame->allocation, packet, 0);
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
      &builder->frame->schedule.requirements.storage_layout,
      builder->frame->module, loom_low_storage_address_storage(node->op),
      &reference);
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
    const loom_low_emission_frame_t* frame, loom_aie2p_register_move_t move,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  const uint32_t descriptor_ordinal =
      loom_aie2p_descriptor_select_move(move.source, move.destination);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const iree_string_view_t destination_name = loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->physical_registers[move.destination].name_string_ref);
    const iree_string_view_t source_name = loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->physical_registers[move.source].name_string_ref);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P has no selected physical move route from %.*s to %.*s",
        (int)source_name.size, source_name.data, (int)destination_name.size,
        destination_name.data);
  }

  const loom_low_allocation_assignment_t destination_assignment = {
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = move.destination,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t source_assignment = {
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = move.source,
      .location_count = 1,
  };
  const loom_low_allocation_assignment_t* operand_assignments[] = {
      &destination_assignment,
      &source_assignment,
  };
  *out_encoded_slot = loom_aie2p_descriptor_encode(
      descriptor_set, descriptor_ordinal, operand_assignments, NULL);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_slot(
    loom_aie2p_bundle_plan_builder_t* builder, loom_aie2p_planned_slot_t slot,
    loom_aie2p_slot_realization_t realization,
    iree_host_size_t* out_slot_index) {
  if (builder->slot_count >= builder->slot_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P bundle-plan slot capacity exhausted");
  }
  const iree_host_size_t slot_index = builder->slot_count++;
  builder->slots[slot_index] = slot;
  builder->slot_realizations[slot_index] = realization;
  if (out_slot_index != NULL) {
    *out_slot_index = slot_index;
  }
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

// Resolves only this selected instruction's bindings. The shared issue owner
// consumes these views; neither instruction decoding nor an IR walk is needed
// to recover allocation-generated accesses.
static void loom_aie2p_bundle_plan_instruction(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_planned_slot_t* slot,
    loom_aie2p_slot_realization_t realization, uint16_t group_index) {
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  const loom_aie2p_register_move_t* move = NULL;
  uint32_t descriptor_ordinal = realization.descriptor_ordinal;
  if (iree_any_bit_set(slot->flags,
                       LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE)) {
    move = &realization.move;
    descriptor_ordinal =
        loom_aie2p_descriptor_select_move(move->source, move->destination);
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  uint16_t* registers =
      builder->instruction_registers +
      group_index * descriptor_set->maximum_descriptor_operand_count;
  builder->instructions[group_index] = (loom_low_physical_instruction_t){
      .descriptor_ordinal = descriptor_ordinal,
      .physical_registers = registers,
  };
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (operand->read_event_id == LOOM_LOW_TIMING_EVENT_NONE &&
        operand->write_event_id == LOOM_LOW_TIMING_EVENT_NONE) {
      registers[i] = 0;
    } else if (operand->source_value_index == LOOM_LOW_ID_NONE) {
      const uint16_t reg_class_id =
          descriptor_set->reg_class_alts[operand->reg_class_alt_start]
              .reg_class_id;
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[reg_class_id];
      registers[i] = descriptor_set->physical_register_candidate_ids
                         [reg_class->physical_register_candidate_start];
    } else if (move != NULL) {
      registers[i] = i == 0 ? move->destination : move->source;
    } else {
      const loom_low_packet_view_t packet = loom_low_packet_at(
          &builder->frame->schedule, slot->scheduled_packet_index);
      const loom_low_allocation_assignment_t* assignment;
      if (iree_any_bit_set(
              slot->flags,
              LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS)) {
        assignment = loom_low_packet_result_assignment(
            &builder->frame->allocation, &packet, 0);
      } else if (iree_any_bit_set(
                     slot->flags,
                     LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL)) {
        assignment = loom_low_packet_operand_assignment(
            &builder->frame->allocation, &packet, 0);
      } else {
        assignment = loom_low_packet_descriptor_operand_assignment(
            &builder->frame->allocation, &packet, i);
      }
      registers[i] = (uint16_t)assignment->location_base;
    }
  }
}

static void loom_aie2p_bundle_plan_instruction_group(
    loom_aie2p_bundle_plan_builder_t* builder, iree_host_size_t slot_start,
    iree_host_size_t slot_count) {
  for (uint16_t i = 0; i < slot_count; ++i) {
    loom_aie2p_bundle_plan_instruction(
        builder, &builder->slots[slot_start + i],
        builder->slot_realizations[slot_start + i], i);
  }
}

static iree_status_t loom_aie2p_bundle_plan_advance(
    loom_aie2p_bundle_plan_builder_t* builder, uint64_t next_cycle) {
  if (next_cycle <= builder->next_issue_cycle) {
    return iree_ok_status();
  }
  const uint64_t gap_bytes =
      (next_cycle - builder->next_issue_cycle) *
      loom_aie2p_bundle_plan_single_slot_byte_length(
          builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP);
  if (gap_bytes >
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE - builder->encoded_byte_length) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf exceeds 16 KiB of core program memory");
  }
  builder->encoded_byte_length += (iree_host_size_t)gap_bytes;
  builder->next_issue_cycle = (uint32_t)next_cycle;
  return iree_ok_status();
}

static void loom_aie2p_bundle_plan_record_writes(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_physical_instruction_t* instructions,
    iree_host_size_t slot_count) {
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    const loom_low_physical_instruction_t* instruction = &instructions[i];
    const loom_low_descriptor_t* descriptor =
        &descriptor_set->descriptors[instruction->descriptor_ordinal];
    for (uint16_t j = 0; j < descriptor->operand_count; ++j) {
      const loom_low_operand_t* operand =
          &descriptor_set->operands[descriptor->operand_start + j];
      if (operand->write_event_id == LOOM_LOW_TIMING_EVENT_NONE) {
        continue;
      }
      const loom_low_physical_register_t* physical_register =
          &descriptor_set
               ->physical_registers[instruction->physical_registers[j]];
      uint32_t part_units =
          loom_aie2p_descriptor_register_part_units(operand->register_part_id) &
          (UINT32_MAX >> (32u - physical_register->atomic_unit_count));
      do {
        const uint32_t k = iree_math_count_trailing_zeros_u32(part_units);
        const uint16_t unit = descriptor_set->physical_register_atomic_units
                                  [physical_register->atomic_unit_start + k];
        builder->register_writes.words[unit / 64] |= UINT64_C(1) << (unit % 64);
        part_units &= part_units - 1u;
      } while (part_units != 0);
    }
  }
}

static uint64_t loom_aie2p_bundle_plan_quiescent_cycle(
    const loom_aie2p_bundle_plan_builder_t* builder,
    uint32_t terminator_packet_index) {
  // Structural control completes after its architectural delay window. Its
  // source deadline constrains retirement, while native condition and LR
  // reads retain their concrete register-event issue admission.
  return iree_max(loom_low_physical_issue_quiescent_cycle(&builder->issue),
                  (uint64_t)loom_low_physical_issue_source_ready_cycle(
                      &builder->issue, terminator_packet_index));
}

// Commits the partition's selected format and already resolved bindings.
static iree_status_t loom_aie2p_bundle_plan_commit_bundle(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start, iree_host_size_t slot_count,
    loom_aie2p_bundle_format_id_t format,
    const loom_low_physical_instruction_t* instructions) {
  if (builder->bundle_count >= builder->bundle_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P bundle-plan bundle capacity exhausted");
  }
  const uint8_t byte_length =
      loom_aie2p_bundle_plan_format_byte_length(format, slot_count);
  uint32_t proposed_cycle = builder->next_issue_cycle;
  uint32_t source_packet_end = builder->next_source_packet;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    const uint32_t packet_index =
        builder->slots[slot_start + i].scheduled_packet_index;
    if (packet_index != LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
      // Controls follow publication of all nonterminal source packets. Their
      // source deadline is enforced at retirement rather than at native issue.
      if (!iree_any_bit_set(builder->slots[slot_start + i].flags,
                            LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL)) {
        proposed_cycle = iree_max(
            proposed_cycle, builder->block_issue_cycle + logical_issue_cycle);
        proposed_cycle =
            iree_max(proposed_cycle, loom_low_physical_issue_source_ready_cycle(
                                         &builder->issue, packet_index));
      }
      source_packet_end = iree_max(source_packet_end, packet_index + 1);
    }
  }
  // Coalesced packets retain source availability without splitting the native
  // descriptor run. They are published in schedule order with this bundle.
  for (uint32_t packet = builder->next_source_packet;
       packet < source_packet_end; ++packet) {
    proposed_cycle = iree_max(
        proposed_cycle,
        loom_low_physical_issue_source_ready_cycle(&builder->issue, packet));
  }
  uint32_t issue_cycle = 0;
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_place(
      &builder->issue, instructions, (uint16_t)slot_count, proposed_cycle,
      &issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_advance(builder, issue_cycle));
  const uint32_t source_packet_begin = builder->next_source_packet;
  while (builder->next_source_packet < source_packet_end) {
    loom_low_physical_issue_commit_source(
        &builder->issue, builder->next_source_packet++, issue_cycle);
  }
  // A structural copy can realize as several native moves. Each later piece
  // extends its consumers' deadlines from the same retained source identity.
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    const uint32_t packet_index =
        builder->slots[slot_start + i].scheduled_packet_index;
    if (packet_index < source_packet_begin) {
      loom_low_physical_issue_commit_source(&builder->issue, packet_index,
                                            issue_cycle);
    }
  }
  if (builder->encoded_byte_length >
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE - byte_length) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf exceeds 16 KiB of core program memory");
  }
  const iree_host_size_t bundle_index = builder->bundle_count++;
  builder->bundles[bundle_index] = (loom_aie2p_planned_bundle_t){
      .issue_cycle = builder->next_issue_cycle++,
      .block_index = builder->current_block_index,
      .logical_issue_cycle = logical_issue_cycle,
      .byte_offset = (uint32_t)builder->encoded_byte_length,
      .slot_start = (uint32_t)slot_start,
      .format = format,
      .slot_count = (uint8_t)slot_count,
  };
  builder->encoded_byte_length += byte_length;
  loom_aie2p_bundle_plan_record_writes(builder, instructions, slot_count);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_single_slot(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start) {
  const loom_aie2p_slot_t physical_slot =
      builder->slots[slot_start].encoded_slot.slot;
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&physical_slot, 1);
  if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u has no legal single-slot bundle format",
        (unsigned)logical_issue_cycle);
  }
  loom_aie2p_bundle_plan_instruction_group(builder, slot_start, 1);
  return loom_aie2p_bundle_plan_commit_bundle(builder, logical_issue_cycle,
                                              slot_start, 1, format,
                                              builder->instructions);
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
  if (slot_count == 1) {
    return loom_aie2p_bundle_plan_append_single_slot(
        builder, logical_issue_cycle, slot_start);
  }
  if (slot_count > LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u contains %zu descriptor slots in one "
        "ordered run",
        (unsigned)logical_issue_cycle, slot_count);
  }

  uint8_t minimum_bundle_counts[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT + 1];
  uint8_t predecessor_indices[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT + 1];
  loom_aie2p_bundle_format_id_t
      partition_formats[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT + 1];
  memset(minimum_bundle_counts, UINT8_MAX, sizeof(minimum_bundle_counts));
  memset(predecessor_indices, UINT8_MAX, sizeof(predecessor_indices));
  minimum_bundle_counts[0] = 0;
  loom_aie2p_bundle_plan_instruction_group(builder, slot_start, slot_count);
  for (iree_host_size_t end = 1; end <= slot_count; ++end) {
    for (iree_host_size_t begin = 0; begin < end; ++begin) {
      if (minimum_bundle_counts[begin] == UINT8_MAX) {
        continue;
      }
      loom_aie2p_slot_t
          physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
      const iree_host_size_t candidate_slot_count = end - begin;
      for (iree_host_size_t i = 0; i < candidate_slot_count; ++i) {
        physical_slots[i] =
            builder->slots[slot_start + begin + i].encoded_slot.slot;
      }
      const loom_aie2p_bundle_format_id_t format =
          loom_aie2p_encoding_find_bundle_format_for_slots(
              physical_slots, candidate_slot_count);
      if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
        continue;
      }
      if (!loom_low_physical_issue_group_fits(builder->descriptor_set,
                                              &builder->instructions[begin],
                                              (uint16_t)candidate_slot_count)) {
        continue;
      }
      const uint8_t candidate_bundle_count =
          (uint8_t)(minimum_bundle_counts[begin] + 1u);
      if (candidate_bundle_count < minimum_bundle_counts[end]) {
        minimum_bundle_counts[end] = candidate_bundle_count;
        predecessor_indices[end] = (uint8_t)begin;
        partition_formats[end] = format;
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
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_commit_bundle(
        builder, logical_issue_cycle, slot_start + begin, end - begin,
        partition_formats[end], &builder->instructions[begin]));
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
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_NOP},
      &slot_start));
  return loom_aie2p_bundle_plan_append_single_slot(builder, logical_issue_cycle,
                                                   slot_start);
}

static iree_status_t loom_aie2p_bundle_plan_try_place_return(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_block_analysis_t* block_analysis, uint32_t candidate_cycle,
    iree_host_size_t candidate_bundle_index, bool* out_placed) {
  *out_placed = false;
  loom_aie2p_planned_bundle_t* candidate_bundle =
      candidate_bundle_index < builder->bundle_count
          ? &builder->bundles[candidate_bundle_index]
          : NULL;
  const bool inserts_bundle = candidate_bundle == NULL ||
                              candidate_bundle->issue_cycle != candidate_cycle;
  loom_aie2p_planned_slot_t return_slot = {
      .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
          builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
      .scheduled_packet_index = block_analysis->terminator_packet_index,
      .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
  };
  loom_aie2p_slot_t physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  const iree_host_size_t candidate_slot_start =
      candidate_bundle != NULL ? candidate_bundle->slot_start
                               : builder->slot_count;
  const bool replaces_nop =
      !inserts_bundle && candidate_bundle->slot_count == 1 &&
      iree_any_bit_set(builder->slots[candidate_slot_start].flags,
                       LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
  const iree_host_size_t retained_slot_count =
      inserts_bundle || replaces_nop ? 0 : candidate_bundle->slot_count;
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

  loom_aie2p_bundle_plan_instruction_group(builder, candidate_slot_start,
                                           retained_slot_count);
  loom_aie2p_bundle_plan_instruction(
      builder, &return_slot,
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_RETURN_},
      (uint16_t)retained_slot_count);
  if (!loom_low_physical_issue_group_fits(builder->descriptor_set,
                                          builder->instructions,
                                          (uint16_t)candidate_slot_count) ||
      loom_low_physical_issue_register_ready_cycle(
          &builder->issue, &builder->instructions[retained_slot_count], 1) >
          candidate_cycle) {
    return iree_ok_status();
  }
  // RET's resources are issue-stage-only and shared only with issue-stage
  // uses, as validated with the target tables. Its only register event reads
  // LR. Thus the group check and retained LR frontier also prove retrospective
  // placement among already-timed tail bundles without replaying the block.

  const uint8_t old_byte_length =
      inserts_bundle
          ? loom_aie2p_bundle_plan_single_slot_byte_length(
                builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP)
          : loom_aie2p_bundle_plan_format_byte_length(
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
  if (!replaces_nop && builder->slot_count >= builder->slot_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P bundle-plan slot capacity exhausted");
  }
  if (inserts_bundle) {
    // The next row, or the stream endpoint, owns the end of this implicit NOP
    // run. Only the selected cycle becomes a row; the control reservation
    // already covers this RET and its remaining delay bundles.
    const uint32_t next_cycle = candidate_bundle != NULL
                                    ? candidate_bundle->issue_cycle
                                    : builder->next_issue_cycle;
    const uint32_t next_byte_offset =
        candidate_bundle != NULL ? candidate_bundle->byte_offset
                                 : (uint32_t)builder->encoded_byte_length;
    const uint32_t byte_offset =
        next_byte_offset - (next_cycle - candidate_cycle) * old_byte_length;
    memmove(&builder->bundles[candidate_bundle_index + 1],
            &builder->bundles[candidate_bundle_index],
            (builder->bundle_count - candidate_bundle_index) *
                sizeof(*builder->bundles));
    ++builder->bundle_count;
    candidate_bundle = &builder->bundles[candidate_bundle_index];
    *candidate_bundle = (loom_aie2p_planned_bundle_t){
        .issue_cycle = candidate_cycle,
        .block_index = builder->current_block_index,
        .logical_issue_cycle = block_analysis->terminator_issue_cycle,
        .byte_offset = byte_offset,
        .slot_start = (uint32_t)candidate_slot_start,
    };
  }
  if (replaces_nop) {
    builder->slots[candidate_slot_start] = return_slot;
    builder->slot_realizations[candidate_slot_start].descriptor_ordinal =
        AIE2P_CORE_DESCRIPTOR_REF_RETURN_;
  } else {
    const iree_host_size_t insert_index =
        candidate_slot_start + retained_slot_count;
    memmove(&builder->slots[insert_index + 1], &builder->slots[insert_index],
            (builder->slot_count - insert_index) * sizeof(*builder->slots));
    builder->slots[insert_index] = return_slot;
    memmove(&builder->slot_realizations[insert_index + 1],
            &builder->slot_realizations[insert_index],
            (builder->slot_count - insert_index) *
                sizeof(*builder->slot_realizations));
    builder->slot_realizations[insert_index].descriptor_ordinal =
        AIE2P_CORE_DESCRIPTOR_REF_RETURN_;
    ++builder->slot_count;
  }
  const int32_t byte_delta =
      (int32_t)new_byte_length - (int32_t)old_byte_length;
  for (iree_host_size_t i = candidate_bundle_index + 1;
       i < builder->bundle_count; ++i) {
    builder->bundles[i].slot_start += replaces_nop ? 0 : 1;
    builder->bundles[i].byte_offset =
        (uint32_t)((int32_t)builder->bundles[i].byte_offset + byte_delta);
  }
  builder->encoded_byte_length =
      builder->encoded_byte_length - old_byte_length + new_byte_length;
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
    const loom_low_allocation_assignment_t* condition_assignment =
        loom_low_packet_operand_assignment(&builder->frame->allocation, &packet,
                                           0);
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

static iree_status_t loom_aie2p_bundle_plan_append_move(
    loom_aie2p_bundle_plan_builder_t* builder, const loom_low_move_t* move,
    uint32_t packet_index, uint32_t logical_issue_cycle) {
  loom_aie2p_register_move_t parts[2];
  const uint8_t part_count = loom_aie2p_descriptor_move_parts(
      (loom_aie2p_register_move_t){
          .source = (loom_aie2p_physical_register_id_t)move->source.location,
          .destination =
              (loom_aie2p_physical_register_id_t)move->destination.location,
      },
      parts);
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < part_count && iree_status_is_ok(status); ++i) {
    loom_aie2p_encoded_slot_t encoded_move;
    status = loom_aie2p_bundle_plan_encode_move(builder->frame, parts[i],
                                                &encoded_move);
    iree_host_size_t slot_start = 0;
    if (iree_status_is_ok(status)) {
      status = loom_aie2p_bundle_plan_append_slot(
          builder,
          (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_move,
              .scheduled_packet_index = packet_index,
              .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
          },
          (loom_aie2p_slot_realization_t){.move = parts[i]}, &slot_start);
    }
    if (iree_status_is_ok(status)) {
      status = loom_aie2p_bundle_plan_append_single_slot(
          builder, logical_issue_cycle, slot_start);
    }
  }
  return status;
}

static iree_status_t loom_aie2p_bundle_plan_append_edge_moves(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* terminator_packet) {
  const loom_low_allocation_edge_copy_group_t* edge_copy_group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          &builder->frame->allocation, terminator_packet->node->source_ordinal);
  if (edge_copy_group == NULL) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t move_ordinal = 0;
       move_ordinal < edge_copy_group->move_group.moves.count &&
       iree_status_is_ok(status);
       ++move_ordinal) {
    const iree_host_size_t move_index =
        edge_copy_group->move_group.moves.start + move_ordinal;
    status = loom_aie2p_bundle_plan_append_move(
        builder, &builder->frame->allocation.moves[move_index],
        (uint32_t)terminator_packet->packet_index,
        terminator_packet->node->issue_cycle);
  }
  return status;
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
      (loom_aie2p_slot_realization_t){.descriptor_ordinal = descriptor_ordinal},
      &slot_start));
  const iree_host_size_t bundle_index = builder->bundle_count;
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors[descriptor_ordinal];
  const uint8_t delay_slot_count =
      loom_aie2p_bundle_plan_instruction_info(descriptor->encoding_id)
          .delay_slot_count;
  const uint64_t quiescent_cycle =
      loom_aie2p_bundle_plan_quiescent_cycle(builder, scheduled_packet_index);
  const uint32_t control_cycles = (uint32_t)delay_slot_count + 1;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_advance(
      builder,
      quiescent_cycle > control_cycles ? quiescent_cycle - control_cycles : 0));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_single_slot(
      builder, logical_issue_cycle, slot_start));
  if (builder->branch_fixup_count >= builder->branch_fixup_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P branch-fixup capacity exhausted");
  }
  builder->branch_fixups[builder->branch_fixup_count++] =
      (loom_aie2p_planned_branch_fixup_t){
          .bundle_index = (uint32_t)bundle_index,
          .target_block_index = target_block_index,
      };

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
  const uint32_t control_cycles = (uint32_t)delay_slot_count + 1;
  const uint64_t retire_cycle =
      iree_max((uint64_t)builder->next_issue_cycle,
               loom_aie2p_bundle_plan_quiescent_cycle(
                   builder, block_analysis->terminator_packet_index));
  const uint64_t earliest_return_cycle =
      retire_cycle > control_cycles ? retire_cycle - control_cycles : 0;
  // At most one hardware delay window is inspected, including implicit NOP
  // positions without stored rows. The block origin excludes predecessor gaps.
  uint64_t candidate_cycle =
      iree_max(earliest_return_cycle, (uint64_t)builder->block_issue_cycle);
  iree_host_size_t candidate = builder->bundle_count;
  while (candidate > block_bundle_start &&
         builder->bundles[candidate - 1].issue_cycle >= candidate_cycle) {
    --candidate;
  }
  bool return_placed = false;
  for (; candidate_cycle < builder->next_issue_cycle; ++candidate_cycle) {
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_place_return(
        builder, block_analysis, (uint32_t)candidate_cycle, candidate,
        &return_placed));
    if (return_placed) {
      break;
    }
    if (candidate < builder->bundle_count &&
        builder->bundles[candidate].issue_cycle == candidate_cycle) {
      ++candidate;
    }
  }
  if (return_placed) {
    loom_low_physical_issue_commit_source(
        &builder->issue, block_analysis->terminator_packet_index,
        (uint32_t)candidate_cycle);
    const uint64_t required_cycle = candidate_cycle + control_cycles;
    while (builder->next_issue_cycle < required_cycle) {
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
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_RETURN_},
      &return_slot_start));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_advance(builder, earliest_return_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_single_slot(
      builder, block_analysis->terminator_issue_cycle, return_slot_start));
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
  if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_RETURN) {
    return loom_aie2p_bundle_plan_append_return(builder, block_analysis,
                                                block_bundle_start);
  }
  if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH) {
    const loom_low_packet_view_t packet = loom_low_packet_at(
        &builder->frame->schedule, block_analysis->terminator_packet_index);
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_edge_moves(builder, &packet));
  }
  if (block_analysis->branches.count == 0) {
    return loom_aie2p_bundle_plan_advance(
        builder, loom_aie2p_bundle_plan_quiescent_cycle(
                     builder, block_analysis->terminator_packet_index));
  }
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0;
       i < block_analysis->branches.count && iree_status_is_ok(status); ++i) {
    const loom_aie2p_block_branch_t* branch =
        &block_analysis->branches.values[i];
    status = loom_aie2p_bundle_plan_append_branch(
        builder, block_analysis->terminator_packet_index,
        branch->descriptor_ordinal, branch->target_block_index,
        block_analysis->terminator_issue_cycle);
  }
  return status;
}

static iree_status_t loom_aie2p_bundle_plan_build_impl(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_aie2p_leaf_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_aie2p_leaf_program_plan_t){0};

  loom_aie2p_bundle_plan_analysis_t analysis;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_analyze(frame, scratch_arena, &analysis));
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
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
      !iree_host_size_checked_add(bundle_capacity, analysis.move_slot_count,
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
  // The same upper bound covers slots: scheduled nodes and native allocation
  // move parts contribute one slot each, while every empty logical cycle and
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
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_initialize(
      &frame->schedule, scratch_arena, &builder.issue));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, slot_capacity, sizeof(*builder.slot_realizations),
      (void**)&builder.slot_realizations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena,
      LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT *
          descriptor_set->maximum_descriptor_operand_count,
      sizeof(*builder.instruction_registers),
      (void**)&builder.instruction_registers));
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
        scratch_arena, frame->schedule.scheduled_node_count,
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
    const loom_aie2p_block_analysis_t* block_analysis =
        &analysis.blocks[block_index];
    if (block_index != 0 && block_analysis->requires_alignment) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_align_next_block(
          &builder, analysis.blocks[block_index - 1u].terminator_issue_cycle));
    }
    builder.current_block_index = block_index;
    builder.block_issue_cycle = builder.next_issue_cycle;
    builder.block_byte_offsets[block_index] =
        (uint32_t)builder.encoded_byte_length;
    const iree_host_size_t block_bundle_start = builder.bundle_count;
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
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
          if (loom_low_packet_is_compile_time_only(&packet)) {
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
                (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                                    packet.descriptor_ordinal},
                NULL));
            continue;
          }
          const loom_aie2p_core_structure_kind_t structure_kind =
              loom_aie2p_core_structure_classify(packet.node->op);
          if (loom_aie2p_bundle_plan_terminator_from_structure_kind(
                  structure_kind) != LOOM_AIE2P_BLOCK_TERMINATOR_NONE) {
            continue;
          }
          if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS) {
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
                (loom_aie2p_slot_realization_t){
                    .descriptor_ordinal =
                        AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32},
                NULL));
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_storage_fixup(
                &builder, packet_index, storage_space, storage_byte_offset));
            continue;
          }
          if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_DECLARATION) {
            continue;
          }

          if (structure_kind != LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE) {
            IREE_ASSERT_UNREACHABLE(
                "AIE2P bundle analysis must admit every emitted structural "
                "operation");
            IREE_BUILTIN_UNREACHABLE();
          }

          const loom_low_allocation_packet_move_group_t* move_group =
              loom_aie2p_bundle_plan_packet_move_group(frame, &packet);
          if (move_group == NULL) {
            continue;
          }
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
            IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_move(
                &builder, &frame->allocation.moves[move_index], packet_index,
                issue_cycle));
            segment_slot_start = builder.slot_count;
          }
        }
      }
      if (builder.slot_count != segment_slot_start) {
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_descriptor_run(
            &builder, issue_cycle, segment_slot_start,
            builder.slot_count - segment_slot_start));
      }
      if (group != NULL) {
        const uint32_t packet_end =
            group->scheduled_node_start + group->scheduled_node_count;
        const uint32_t virtual_issue_cycle = iree_max(
            builder.next_issue_cycle ? builder.next_issue_cycle - 1 : 0,
            builder.block_issue_cycle + issue_cycle);
        while (builder.next_source_packet < packet_end) {
          const uint32_t packet_index = builder.next_source_packet++;
          if (packet_index != block_analysis->terminator_packet_index) {
            loom_low_physical_issue_commit_source(&builder.issue, packet_index,
                                                  virtual_issue_cycle);
          }
        }
      }
      if (builder.bundle_count == cycle_bundle_start &&
          issue_cycle != block_analysis->terminator_issue_cycle &&
          builder.next_issue_cycle <=
              (uint64_t)builder.block_issue_cycle + issue_cycle) {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_bundle_plan_append_nop_bundle(&builder, issue_cycle));
      }
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_terminator(
        &builder, block_analysis, block_bundle_start));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_resolve_storage_fixups(&builder));

  loom_aie2p_leaf_program_plan_t plan = {
      .register_writes = builder.register_writes,
      .block_byte_offsets = builder.block_byte_offsets,
      .block_count = builder.block_count,
      .bundles = builder.bundles,
      .bundle_count = builder.bundle_count,
      .issue_cycle_count = builder.next_issue_cycle,
      .slots = builder.slots,
      .slot_count = builder.slot_count,
      .branch_fixups = builder.branch_fixups,
      .branch_fixup_count = builder.branch_fixup_count,
      .storage_fixups = builder.storage_fixups,
      .storage_fixup_count = builder.storage_fixup_count,
      .encoded_byte_length = builder.encoded_byte_length,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_copy_function_name(
      frame, arena, &plan.function_name));
  loom_aie2p_bundle_plan_retain_storage_requirements(frame, &plan);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_retain_resource_imports(frame, arena, &plan));
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* out_plan) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status =
      loom_aie2p_bundle_plan_build_impl(frame, arena, &scratch_arena, out_plan);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
