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
  // Native instruction slots required by packet and edge allocation moves.
  iree_host_size_t move_slot_count;
  // Conservative branch, return, and delay-bundle capacity.
  iree_host_size_t control_bundle_capacity;
  // Structural local-storage addresses requiring placement fixups.
  iree_host_size_t storage_fixup_count;
  // Symbolic read-only data addresses requiring placement fixups.
  iree_host_size_t read_only_data_fixup_count;
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

// Encoded instructions already admitted in semantic order, awaiting closure
// of the bounded native issue window. No compiler analysis is repeated when
// this packet is appended to the detached program.
typedef struct loom_aie2p_pending_bundle_t {
  // Native slots in semantic publication order within this issue cycle.
  loom_aie2p_planned_slot_t slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  // Concrete bindings retained for the final program's control-tail packing.
  loom_aie2p_slot_realization_t
      realizations[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  // Greatest logical source cycle represented by this physical packet.
  uint32_t logical_issue_cycle;
  // Exact format for the currently admitted slot union.
  loom_aie2p_bundle_format_id_t format;
  // Number of populated slots; zero denotes an empty issue position.
  uint8_t slot_count;
} loom_aie2p_pending_bundle_t;

typedef struct loom_aie2p_bundle_plan_builder_t {
  // Emission frame being converted into physical bundles.
  const loom_low_emission_frame_t* frame;
  // Exact AIE2P descriptor set selected by the frame.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena-backed bundle storage owned by the final plan.
  loom_aie2p_planned_bundle_t* bundles;
  // Number of populated records in |bundles|.
  iree_host_size_t bundle_count;
  // Arena-backed slot storage owned by the final plan.
  loom_aie2p_planned_slot_t* slots;
  // Number of populated records in |slots|.
  iree_host_size_t slot_count;
  // Arena-backed branch target records owned by the final plan.
  loom_aie2p_planned_branch_fixup_t* branch_fixups;
  // Number of populated records in |branch_fixups|.
  iree_host_size_t branch_fixup_count;
  // Arena-backed local-storage fixups owned by the final plan.
  loom_aie2p_planned_storage_fixup_t* storage_fixups;
  // Fixup index for each scheduled packet, or UINT32_MAX when absent.
  uint32_t* storage_fixup_indices_by_packet;
  // Number of populated records in |storage_fixups|.
  iree_host_size_t storage_fixup_count;
  // Arena-backed read-only data fixups owned by the final plan.
  loom_aie2p_planned_read_only_data_fixup_t* read_only_data_fixups;
  // Fixup index for each scheduled packet, or UINT32_MAX when absent.
  uint32_t* read_only_data_fixup_indices_by_packet;
  // Number of populated records in |read_only_data_fixups|.
  iree_host_size_t read_only_data_fixup_count;
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
  // Bounded encoded-packet history matching shared physical admission.
  struct {
    // Scratch ring indexed by absolute issue cycle modulo |capacity|.
    loom_aie2p_pending_bundle_t* bundles;
    // One entry for each retained issue position, including the high-water.
    uint32_t capacity;
    // First issue cycle after all currently pending instructions.
    uint32_t end_cycle;
    // Native issue floor established by block entry and source-order fences.
    uint32_t minimum_issue_cycle;
  } pending;
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

static iree_status_t loom_aie2p_bundle_plan_retain_read_only_data(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* plan) {
  const loom_low_function_requirements_t* requirements =
      &frame->schedule.requirements;
  if (requirements->read_only_data_count == 0) {
    return iree_ok_status();
  }
  loom_aie2p_leaf_read_only_data_t* read_only_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, requirements->read_only_data_count, sizeof(*read_only_data),
      (void**)&read_only_data));
  for (iree_host_size_t i = 0; i < requirements->read_only_data_count; ++i) {
    const loom_low_read_only_data_requirement_t* requirement =
        &requirements->read_only_data[i];
    IREE_ASSERT_EQ(requirement->symbol.module_id, 0u);
    IREE_ASSERT_LT(requirement->symbol.symbol_id, frame->module->symbols.count);
    const loom_symbol_t* symbol =
        &frame->module->symbols.entries[requirement->symbol.symbol_id];
    const iree_string_view_t source_name =
        loom_string_table_get(&frame->module->strings, symbol->name_id);
    char* name_data = NULL;
    if (!iree_string_view_is_empty(source_name)) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, source_name.size, (void**)&name_data));
      memcpy(name_data, source_name.data, source_name.size);
    }
    uint8_t* contents_data = NULL;
    if (requirement->contents.data_length != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          arena, requirement->contents.data_length, (void**)&contents_data));
      memcpy(contents_data, requirement->contents.data,
             requirement->contents.data_length);
    }
    read_only_data[i] = (loom_aie2p_leaf_read_only_data_t){
        .name = iree_make_string_view(name_data, source_name.size),
        .contents = iree_make_const_byte_span(
            contents_data, requirement->contents.data_length),
        .minimum_alignment = requirement->minimum_alignment,
    };
  }
  plan->read_only_data = read_only_data;
  plan->read_only_data_count = requirements->read_only_data_count;
  return iree_ok_status();
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
        if (descriptor_ordinal ==
            AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32) {
          const loom_named_attr_slice_t attrs = loom_low_packet_attrs(&packet);
          IREE_ASSERT_EQ(attrs.count, 1u);
          if (attrs.entries[0].value.kind == LOOM_ATTR_SYMBOL) {
            if (out_analysis->read_only_data_fixup_count ==
                IREE_HOST_SIZE_MAX) {
              return iree_make_status(
                  IREE_STATUS_OUT_OF_RANGE,
                  "AIE2P read-only data fixup count exceeds host size");
            }
            ++out_analysis->read_only_data_fixup_count;
          }
        }
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
    loom_aie2p_encoded_slot_t* out_encoded_slot,
    uint32_t* out_read_only_data_ordinal) {
  *out_read_only_data_ordinal = UINT32_MAX;
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
      if (packet->descriptor_ordinal !=
          AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P descriptor %u has no symbolic immediate relocation",
            packet->descriptor_ordinal);
      }
      const uint32_t ordinal =
          loom_low_function_requirements_read_only_data_ordinal(
              &frame->schedule.requirements, loom_attr_as_symbol(value));
      if (ordinal == UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P symbolic local address does not name retained read-only "
            "data");
      }
      IREE_ASSERT_EQ(*out_read_only_data_ordinal, UINT32_MAX);
      *out_read_only_data_ordinal = ordinal;
      immediate_values[i] = 0;
      continue;
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

static iree_host_size_t loom_aie2p_bundle_plan_append_slot(
    loom_aie2p_bundle_plan_builder_t* builder, loom_aie2p_planned_slot_t slot,
    loom_aie2p_slot_realization_t realization) {
  const iree_host_size_t slot_index = builder->slot_count++;
  builder->slots[slot_index] = slot;
  builder->slot_realizations[slot_index] = realization;
  return slot_index;
}

static void loom_aie2p_bundle_plan_append_storage_fixup(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t scheduled_packet_index,
    loom_storage_space_t storage_space, uint64_t byte_offset) {
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
}

static void loom_aie2p_bundle_plan_append_read_only_data_fixup(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t scheduled_packet_index,
    uint32_t read_only_data_ordinal) {
  const iree_host_size_t fixup_index = builder->read_only_data_fixup_count++;
  builder->read_only_data_fixups[fixup_index] =
      (loom_aie2p_planned_read_only_data_fixup_t){
          .bundle_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
          .read_only_data_ordinal = read_only_data_ordinal,
      };
  IREE_ASSERT_LT(scheduled_packet_index,
                 builder->frame->schedule.scheduled_node_count);
  IREE_ASSERT_EQ(
      builder->read_only_data_fixup_indices_by_packet[scheduled_packet_index],
      UINT32_MAX);
  builder->read_only_data_fixup_indices_by_packet[scheduled_packet_index] =
      (uint32_t)fixup_index;
}

static iree_status_t loom_aie2p_bundle_plan_resolve_address_fixups(
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
        if (!iree_any_bit_set(
                slot->flags,
                LOOM_AIE2P_PLANNED_SLOT_FLAG_READ_ONLY_DATA_ADDRESS)) {
          continue;
        }
        if (slot->scheduled_packet_index >=
            builder->frame->schedule.scheduled_node_count) {
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "AIE2P read-only data address has no scheduled packet");
        }
        const uint32_t matched_fixup_index =
            builder->read_only_data_fixup_indices_by_packet
                [slot->scheduled_packet_index];
        if (matched_fixup_index == UINT32_MAX ||
            matched_fixup_index >= builder->read_only_data_fixup_count ||
            builder->read_only_data_fixups[matched_fixup_index].bundle_index !=
                LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "AIE2P read-only data address has no unique planned fixup");
        }
        builder->read_only_data_fixups[matched_fixup_index].bundle_index =
            (uint32_t)bundle_index;
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
  for (iree_host_size_t i = 0; i < builder->read_only_data_fixup_count; ++i) {
    if (builder->read_only_data_fixups[i].bundle_index ==
        LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AIE2P planned read-only data fixup has no emitted instruction");
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

// Appends an already admitted packet in chronological native issue order.
static iree_status_t loom_aie2p_bundle_plan_append_bundle(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start, iree_host_size_t slot_count,
    loom_aie2p_bundle_format_id_t format, uint32_t issue_cycle) {
  const uint8_t byte_length =
      loom_aie2p_bundle_plan_format_byte_length(format, slot_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_advance(builder, issue_cycle));
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
  return iree_ok_status();
}

// Control and padding packets follow a closed native window. Their source
// dependency constrains retirement, while register reads constrain issue.
static iree_status_t loom_aie2p_bundle_plan_append_single_slot(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    iree_host_size_t slot_start) {
  const loom_aie2p_slot_t physical_slot =
      builder->slots[slot_start].encoded_slot.slot;
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&physical_slot, 1);
  loom_aie2p_bundle_plan_instruction_group(builder, slot_start, 1);
  const uint32_t issue_cycle =
      loom_low_physical_issue_find_earliest_issue_cycle(
          &builder->issue, builder->instructions, 1, builder->next_issue_cycle);
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_commit(
      &builder->issue, builder->instructions, 1, issue_cycle));
  const uint32_t packet_index =
      builder->slots[slot_start].scheduled_packet_index;
  if (packet_index != LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
    loom_low_physical_issue_commit_source(&builder->issue, packet_index,
                                          issue_cycle);
  }
  loom_aie2p_bundle_plan_record_writes(builder, builder->instructions, 1);
  return loom_aie2p_bundle_plan_append_bundle(
      builder, logical_issue_cycle, slot_start, 1, format, issue_cycle);
}

// Closes only issue positions that no later instruction can reach. Empty
// positions remain implicit gaps rather than growing per-cycle plan records.
static iree_status_t loom_aie2p_bundle_plan_flush_pending(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t end_cycle) {
  const uint32_t populated_end =
      iree_min(end_cycle, builder->pending.end_cycle);
  while (builder->next_issue_cycle < populated_end) {
    const uint32_t cycle = builder->next_issue_cycle;
    loom_aie2p_pending_bundle_t* pending =
        &builder->pending.bundles[cycle % builder->pending.capacity];
    if (pending->slot_count == 0) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_advance(builder, cycle + 1));
      continue;
    }
    const iree_host_size_t slot_start = builder->slot_count;
    for (uint8_t i = 0; i < pending->slot_count; ++i) {
      loom_aie2p_bundle_plan_append_slot(builder, pending->slots[i],
                                         pending->realizations[i]);
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
        builder, pending->logical_issue_cycle, slot_start, pending->slot_count,
        pending->format, cycle));
    pending->slot_count = 0;
  }
  return loom_aie2p_bundle_plan_advance(builder, end_cycle);
}

// Publishes one actual instruction in accepted source/expansion order. Its
// timestamp may precede a prior publication, but never the retained-history or
// source-fence floor. The shared event frontier therefore sees the intended
// old-value read before a storage-reusing delayed write, not a false RAW edge.
static iree_status_t loom_aie2p_bundle_plan_admit_instruction(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    loom_aie2p_planned_slot_t slot, loom_aie2p_slot_realization_t realization) {
  loom_aie2p_bundle_plan_instruction(builder, &slot, realization, 0);
  uint32_t proposed_cycle =
      iree_max(builder->pending.minimum_issue_cycle, builder->next_issue_cycle);
  proposed_cycle = iree_max(proposed_cycle,
                            loom_low_physical_issue_source_ready_cycle(
                                &builder->issue, slot.scheduled_packet_index));
  uint32_t cycle = loom_low_physical_issue_find_earliest_issue_cycle(
      &builder->issue, builder->instructions, 1, proposed_cycle);
  loom_aie2p_pending_bundle_t* pending = NULL;
  loom_aie2p_bundle_format_id_t format;
  do {
    const uint32_t lookback =
        builder->descriptor_set->resource_calendar_lookback_cycles;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
        builder, cycle > lookback ? cycle - lookback : 0));
    pending = &builder->pending.bundles[cycle % builder->pending.capacity];
    loom_aie2p_slot_t slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    format = LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
    if (pending->slot_count < LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
      for (uint8_t i = 0; i < pending->slot_count; ++i) {
        slots[i] = pending->slots[i].encoded_slot.slot;
      }
      slots[pending->slot_count] = slot.encoded_slot.slot;
      format = loom_aie2p_encoding_find_bundle_format_for_slots(
          slots, pending->slot_count + 1);
    }
    if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
      cycle = loom_low_physical_issue_find_earliest_issue_cycle(
          &builder->issue, builder->instructions, 1, cycle + 1);
    }
  } while (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_commit(
      &builder->issue, builder->instructions, 1, cycle));
  loom_low_physical_issue_commit_source(&builder->issue,
                                        slot.scheduled_packet_index, cycle);
  loom_aie2p_bundle_plan_record_writes(builder, builder->instructions, 1);
  pending->logical_issue_cycle =
      pending->slot_count == 0
          ? logical_issue_cycle
          : iree_max(pending->logical_issue_cycle, logical_issue_cycle);
  pending->slots[pending->slot_count] = slot;
  pending->realizations[pending->slot_count++] = realization;
  pending->format = format;
  builder->pending.end_cycle = iree_max(builder->pending.end_cycle, cycle + 1);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_nop_bundle(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle) {
  const iree_host_size_t slot_start = loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
              builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP),
          .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
      },
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_NOP});
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
    if (iree_status_is_ok(status)) {
      status = loom_aie2p_bundle_plan_admit_instruction(
          builder, logical_issue_cycle,
          (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_move,
              .scheduled_packet_index = packet_index,
              .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
          },
          (loom_aie2p_slot_realization_t){.move = parts[i]});
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
  const iree_host_size_t slot_start = loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = encoded_branch,
          .scheduled_packet_index = scheduled_packet_index,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
      },
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          descriptor_ordinal});
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

  const iree_host_size_t return_slot_start = loom_aie2p_bundle_plan_append_slot(
      builder,
      (loom_aie2p_planned_slot_t){
          .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
              builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
          .scheduled_packet_index = block_analysis->terminator_packet_index,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
      },
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_RETURN_});
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
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
        builder, builder->pending.end_cycle));
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

// Realizes exactly one accepted source packet. Coalesced packets publish only
// their incoming payload availability, not the native issue high-water.
static iree_status_t loom_aie2p_bundle_plan_append_packet(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  const uint32_t packet_index = (uint32_t)packet->packet_index;
  const uint32_t logical_issue_cycle = packet->node->issue_cycle;
  if (!loom_low_packet_is_compile_time_only(packet)) {
    if (packet->descriptor != NULL) {
      loom_aie2p_encoded_slot_t encoded_slot;
      uint32_t read_only_data_ordinal = UINT32_MAX;
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_packet(
          builder->frame, packet, &encoded_slot, &read_only_data_ordinal));
      if (read_only_data_ordinal != UINT32_MAX) {
        loom_aie2p_bundle_plan_append_read_only_data_fixup(
            builder, packet_index, read_only_data_ordinal);
      }
      return loom_aie2p_bundle_plan_admit_instruction(
          builder, logical_issue_cycle,
          (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_slot,
              .scheduled_packet_index = packet_index,
              .flags = read_only_data_ordinal != UINT32_MAX
                           ? LOOM_AIE2P_PLANNED_SLOT_FLAG_READ_ONLY_DATA_ADDRESS
                           : 0,
          },
          (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                              packet->descriptor_ordinal});
    }
    const loom_aie2p_core_structure_kind_t structure_kind =
        loom_aie2p_core_structure_classify(packet->node->op);
    if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS) {
      loom_aie2p_encoded_slot_t encoded_slot;
      loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_STACK;
      uint64_t storage_byte_offset = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_storage_address(
          builder, packet, &encoded_slot, &storage_space,
          &storage_byte_offset));
      loom_aie2p_bundle_plan_append_storage_fixup(
          builder, packet_index, storage_space, storage_byte_offset);
      return loom_aie2p_bundle_plan_admit_instruction(
          builder, logical_issue_cycle,
          (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_slot,
              .scheduled_packet_index = packet_index,
              .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_STORAGE_ADDRESS,
          },
          (loom_aie2p_slot_realization_t){
              .descriptor_ordinal =
                  AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32});
    }
    if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE) {
      const loom_low_allocation_packet_move_group_t* group =
          loom_aie2p_bundle_plan_packet_move_group(builder->frame, packet);
      if (group != NULL && group->move_group.moves.count != 0) {
        const loom_low_move_range_t moves = group->move_group.moves;
        iree_status_t status = iree_ok_status();
        for (iree_host_size_t i = 0;
             i < moves.count && iree_status_is_ok(status); ++i) {
          status = loom_aie2p_bundle_plan_append_move(
              builder, &builder->frame->allocation.moves[moves.start + i],
              packet_index, logical_issue_cycle);
        }
        return status;
      }
    }
  }
  loom_low_physical_issue_commit_source(&builder->issue, packet_index, 0);
  return iree_ok_status();
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
  iree_host_size_t bundle_capacity = frame->schedule.scheduled_node_count;
  if (!iree_host_size_checked_add(bundle_capacity, analysis.move_slot_count,
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
  // move parts contribute one slot each. Control delays and alignment reserve
  // their synthetic slots; other timing gaps have no stored records.
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
      .block_count = frame->schedule.block_count,
      .pending.capacity = descriptor_set->resource_calendar_lookback_cycles + 1,
  };
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_initialize(
      &frame->schedule, scratch_arena, &builder.issue));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, builder.pending.capacity, sizeof(*builder.pending.bundles),
      (void**)&builder.pending.bundles));
  memset(builder.pending.bundles, 0,
         builder.pending.capacity * sizeof(*builder.pending.bundles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, slot_capacity, sizeof(*builder.slot_realizations),
      (void**)&builder.slot_realizations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena,
      LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT *
          descriptor_set->maximum_descriptor_operand_count,
      sizeof(*builder.instruction_registers),
      (void**)&builder.instruction_registers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, bundle_capacity,
                                                 sizeof(*builder.bundles),
                                                 (void**)&builder.bundles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, slot_capacity, sizeof(*builder.slots), (void**)&builder.slots));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, branch_fixup_capacity, sizeof(*builder.branch_fixups),
      (void**)&builder.branch_fixups));
  if (analysis.storage_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, analysis.storage_fixup_count, sizeof(*builder.storage_fixups),
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
  if (analysis.read_only_data_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, analysis.read_only_data_fixup_count,
                                  sizeof(*builder.read_only_data_fixups),
                                  (void**)&builder.read_only_data_fixups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, frame->schedule.scheduled_node_count,
        sizeof(*builder.read_only_data_fixup_indices_by_packet),
        (void**)&builder.read_only_data_fixup_indices_by_packet));
    for (iree_host_size_t i = 0; i < frame->schedule.scheduled_node_count;
         ++i) {
      builder.read_only_data_fixup_indices_by_packet[i] = UINT32_MAX;
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
    builder.pending.minimum_issue_cycle = builder.next_issue_cycle;
    builder.pending.end_cycle = builder.next_issue_cycle;
    for (uint32_t ordinal = 0; ordinal < block->scheduled_node_count;
         ++ordinal) {
      const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
          &frame->schedule, block_index, ordinal);
      if (packet.packet_index == block_analysis->terminator_packet_index) {
        break;
      }
      const bool is_boundary =
          iree_any_bit_set(packet.node->flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY);
      if (is_boundary) {
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
            &builder, builder.pending.end_cycle));
        builder.pending.minimum_issue_cycle = builder.next_issue_cycle;
      }
      IREE_RETURN_IF_ERROR(
          loom_aie2p_bundle_plan_append_packet(&builder, &packet));
      if (is_boundary) {
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
            &builder, builder.pending.end_cycle));
        builder.pending.minimum_issue_cycle = builder.next_issue_cycle;
      }
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
        &builder, builder.pending.end_cycle));
    builder.pending.minimum_issue_cycle = builder.next_issue_cycle;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_terminator(
        &builder, block_analysis, block_bundle_start));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_resolve_address_fixups(&builder));

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
      .read_only_data_fixups = builder.read_only_data_fixups,
      .read_only_data_fixup_count = builder.read_only_data_fixup_count,
      .encoded_byte_length = builder.encoded_byte_length,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_copy_function_name(
      frame, arena, &plan.function_name));
  loom_aie2p_bundle_plan_retain_storage_requirements(frame, &plan);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_retain_read_only_data(frame, arena, &plan));
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
