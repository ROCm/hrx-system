// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_analysis.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/loop_execution.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_builders.h"
#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

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

static const loom_low_allocation_packet_move_group_t*
loom_aie2p_bundle_plan_packet_move_group(const loom_low_emission_frame_t* frame,
                                         const loom_low_packet_view_t* packet) {
  return loom_low_allocation_find_packet_move_group_by_source_ordinal(
      &frame->allocation, packet->node_index);
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

static iree_status_t loom_aie2p_bundle_plan_analyze_move_range(
    const loom_low_emission_frame_t* frame, loom_low_move_range_t range,
    const loom_op_t* op, iree_diagnostic_emitter_t diagnostic_emitter,
    loom_aie2p_bundle_plan_analysis_t* analysis, bool* out_accepted) {
  *out_accepted = false;
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  for (iree_host_size_t i = 0; i < range.count; ++i) {
    const iree_host_size_t move_index = range.start + i;
    const loom_low_move_t* move = &frame->allocation.moves[move_index];
    loom_aie2p_move_route_id_t route_id =
        analysis->move_route_ids_by_move[move_index];
    if (route_id == LOOM_AIE2P_MOVE_ROUTE_ID_INVALID) {
      route_id = loom_aie2p_descriptor_resolve_move_route(
          (loom_aie2p_physical_register_id_t)move->source.location,
          (loom_aie2p_physical_register_id_t)move->destination.location);
      if (route_id == LOOM_AIE2P_MOVE_ROUTE_ID_INVALID) {
        const iree_string_view_t source_register =
            loom_low_descriptor_set_string(
                descriptor_set,
                descriptor_set->physical_registers[move->source.location]
                    .name_string_ref);
        const iree_string_view_t destination_register =
            loom_low_descriptor_set_string(
                descriptor_set,
                descriptor_set->physical_registers[move->destination.location]
                    .name_string_ref);
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_low_diagnostic_function_name(
                frame->module, frame->function_op)),
            loom_param_string(source_register),
            loom_param_string(destination_register),
        };
        const loom_diagnostic_emission_t emission = {
            .op = op,
            .error = LOOM_ERR_XDNA_061,
            .params = params,
            .param_count = IREE_ARRAYSIZE(params),
        };
        return iree_diagnostic_emit(diagnostic_emitter, &emission);
      }
      analysis->move_route_ids_by_move[move_index] = route_id;
    }
    analysis->move_slot_count +=
        loom_aie2p_descriptor_move_route_step_count(route_id);
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_analyze_storage_address(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    const loom_low_storage_layout_t* layout,
    iree_diagnostic_emitter_t diagnostic_emitter,
    loom_aie2p_planned_storage_fixup_t* out_fixup, bool* out_accepted) {
  *out_accepted = false;
  const loom_op_t* op = packet->node->op;
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(&layout->index, layout->records,
                                           loom_low_storage_address_storage(op),
                                           &reference);
  const int64_t operation_offset = loom_low_storage_address_offset(op);
  IREE_ASSERT_GE(operation_offset, 0);
  IREE_ASSERT_LE(reference.reservation.byte_offset,
                 UINT64_MAX - reference.byte_offset);
  const uint64_t reference_offset =
      reference.reservation.byte_offset + reference.byte_offset;
  IREE_ASSERT_LE((uint64_t)operation_offset, UINT64_MAX - reference_offset);
  const uint64_t byte_offset = reference_offset + (uint64_t)operation_offset;
  if (byte_offset <= INT64_MAX) {
    *out_fixup = (loom_aie2p_planned_storage_fixup_t){
        .storage_space = reference.reservation.space,
        .byte_offset = byte_offset,
    };
    *out_accepted = true;
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(frame->module, frame->function_op)),
      loom_param_string(
          loom_low_storage_type_space_name(reference.reservation.space)),
      loom_param_u64(byte_offset),
      loom_param_u64(INT64_MAX),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_060,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static bool loom_aie2p_bundle_plan_header_is_hardware_loop_only(
    const loom_low_emission_frame_t* frame, uint16_t header_block_index,
    uint32_t terminator_packet_index) {
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  const loom_low_schedule_block_t* block =
      &schedule->blocks[header_block_index];
  const loom_block_t* header = block->block;
  for (uint32_t ordinal = 0; ordinal < block->scheduled_node_count; ++ordinal) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at_block_ordinal(schedule, header_block_index, ordinal);
    if (packet.packet_index == terminator_packet_index ||
        loom_low_packet_is_compile_time_only(&packet)) {
      continue;
    }
    if (packet.descriptor == NULL ||
        loom_traits_has_side_effects(packet.node->traits)) {
      return false;
    }
    const loom_op_t* op = packet.node->op;
    for (uint16_t result_index = 0; result_index < op->result_count;
         ++result_index) {
      const loom_value_id_t result_id = loom_op_results(op)[result_index];
      const loom_value_t* result = loom_module_value(frame->module, result_id);
      if (loom_value_has_attribute_uses(result) ||
          loom_module_value_has_type_uses(frame->module, result_id)) {
        return false;
      }
      const loom_use_t* use = NULL;
      loom_value_for_each_use(result, use) {
        if (loom_use_user_op(*use)->parent_block != header) {
          return false;
        }
      }
    }
  }
  return true;
}

static bool loom_aie2p_bundle_plan_body_is_hardware_loop_compatible(
    const loom_low_emission_frame_t* frame, uint16_t body_block_index,
    uint32_t terminator_packet_index) {
  const loom_low_schedule_block_t* block =
      &frame->schedule.blocks[body_block_index];
  for (uint32_t ordinal = 0; ordinal < block->scheduled_node_count; ++ordinal) {
    const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
        &frame->schedule, body_block_index, ordinal);
    if (packet.packet_index == terminator_packet_index ||
        packet.descriptor != NULL ||
        loom_low_packet_is_compile_time_only(&packet)) {
      continue;
    }
    switch (loom_aie2p_core_structure_classify(packet.node->op)) {
      case LOOM_AIE2P_CORE_STRUCTURE_DECLARATION:
      case LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE:
      case LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS:
        break;
      default:
        return false;
    }
  }
  return true;
}

static void loom_aie2p_bundle_plan_analyze_hardware_loops(
    const loom_low_emission_frame_t* frame,
    loom_aie2p_bundle_plan_analysis_t* analysis) {
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  const loom_cfg_loop_nest_t* nest = &schedule->loop_nest;
  if (!nest->reducible || nest->loop_count == 0 ||
      schedule->loop_execution.block_count != schedule->block_count) {
    return;
  }
  for (uint16_t loop_index = 0; loop_index < nest->loop_count; ++loop_index) {
    const loom_cfg_natural_loop_t* loop = &nest->loops[loop_index];
    if (loop->entries.count != 1 || loop->backedges.count != 1 ||
        loop->exits.count != 1 ||
        loop->continuation_index == LOOM_CFG_LOOP_CONTINUATION_NONE) {
      continue;
    }
    const loom_cfg_edge_info_t* entry =
        loom_cfg_graph_edge(graph, loop->entries.unique_index);
    const loom_cfg_edge_info_t* backedge =
        loom_cfg_graph_edge(graph, loop->backedges.unique_index);
    const loom_cfg_edge_info_t* exit =
        loom_cfg_graph_edge(graph, loop->exits.unique_index);
    if (entry == NULL || backedge == NULL || exit == NULL) {
      continue;
    }
    const uint16_t preheader_index = entry->source_block_index;
    const uint16_t header_index = loop->header_index;
    const uint16_t body_index = backedge->source_block_index;
    const uint16_t continuation_index = loop->continuation_index;
    if (entry->target_block_index != header_index ||
        backedge->target_block_index != header_index ||
        exit->source_block_index != header_index ||
        exit->target_block_index != continuation_index ||
        body_index == header_index || preheader_index + 1u != header_index ||
        header_index + 1u != body_index ||
        body_index + 1u != continuation_index) {
      continue;
    }
    if (loom_cfg_loop_nest_innermost(nest, header_index) != loop_index ||
        loom_cfg_loop_nest_innermost(nest, body_index) != loop_index) {
      continue;
    }
    bool has_other_member = false;
    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      if (block_index != header_index && block_index != body_index &&
          loom_cfg_loop_nest_contains(nest, loop_index, block_index)) {
        has_other_member = true;
        break;
      }
    }
    if (has_other_member) {
      continue;
    }
    const loom_cfg_block_index_span_t preheader_successors =
        loom_cfg_graph_successors(graph, preheader_index);
    const loom_cfg_block_index_span_t header_successors =
        loom_cfg_graph_successors(graph, header_index);
    const loom_cfg_block_index_span_t body_successors =
        loom_cfg_graph_successors(graph, body_index);
    if (preheader_successors.count != 1 ||
        preheader_successors.values[0] != header_index ||
        header_successors.count != 2 || body_successors.count != 1 ||
        body_successors.values[0] != header_index) {
      continue;
    }
    const bool header_reaches_body_and_continuation =
        (header_successors.values[0] == body_index &&
         header_successors.values[1] == continuation_index) ||
        (header_successors.values[1] == body_index &&
         header_successors.values[0] == continuation_index);
    if (!header_reaches_body_and_continuation) {
      continue;
    }
    loom_aie2p_block_analysis_t* preheader = &analysis->blocks[preheader_index];
    loom_aie2p_block_analysis_t* header = &analysis->blocks[header_index];
    loom_aie2p_block_analysis_t* body = &analysis->blocks[body_index];
    if (preheader->terminator != LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH ||
        header->terminator != LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH ||
        body->terminator != LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH ||
        !loom_aie2p_bundle_plan_header_is_hardware_loop_only(
            frame, header_index, header->terminator_packet_index) ||
        !loom_aie2p_bundle_plan_body_is_hardware_loop_compatible(
            frame, body_index, body->terminator_packet_index)) {
      continue;
    }
    const loom_low_loop_execution_facts_t* facts =
        loom_low_loop_execution_index_lookup(
            &schedule->loop_execution, schedule->blocks[header_index].block);
    if (facts == NULL ||
        facts->count_kind != LOOM_LOW_LOOP_EXECUTION_COUNT_EXACT ||
        facts->trip_count == 0 || facts->trip_count > 1023u) {
      continue;
    }
    header->hardware_loop = (loom_aie2p_hardware_loop_analysis_t){
        .preheader_block_index = preheader_index,
        .header_block_index = header_index,
        .body_block_index = body_index,
        .continuation_block_index = continuation_index,
        .trip_count = (uint16_t)facts->trip_count,
        .is_candidate = true,
    };
    ++analysis->hardware_loop_candidate_count;
  }
}

iree_status_t loom_aie2p_bundle_plan_analyze(
    const loom_low_emission_frame_t* frame,
    const loom_low_storage_layout_t* storage_layout,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* plan_arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_bundle_plan_analysis_t* out_analysis, bool* out_accepted) {
  *out_analysis = (loom_aie2p_bundle_plan_analysis_t){0};
  *out_accepted = false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  loom_aie2p_block_analysis_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, frame->schedule.block_count,
                                sizeof(*blocks), (void**)&blocks));
  memset(blocks, 0,
         frame->schedule.block_count * sizeof(loom_aie2p_block_analysis_t));
  if (frame->allocation.move_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, frame->allocation.move_count,
        sizeof(*out_analysis->move_route_ids_by_move),
        (void**)&out_analysis->move_route_ids_by_move));
    memset(out_analysis->move_route_ids_by_move,
           LOOM_AIE2P_MOVE_ROUTE_ID_INVALID,
           frame->allocation.move_count *
               sizeof(*out_analysis->move_route_ids_by_move));
  }
  const iree_host_size_t expected_storage_fixup_count =
      frame->schedule.requirements.storage_address_count;
  if (expected_storage_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan_arena, expected_storage_fixup_count,
                                  sizeof(*out_analysis->storage_fixups),
                                  (void**)&out_analysis->storage_fixups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan_arena, frame->schedule.scheduled_node_count,
        sizeof(*out_analysis->address_fixup_indices_by_packet),
        (void**)&out_analysis->address_fixup_indices_by_packet));
    memset(out_analysis->address_fixup_indices_by_packet, 0xFF,
           frame->schedule.scheduled_node_count *
               sizeof(*out_analysis->address_fixup_indices_by_packet));
  }
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
      frame, frame->allocation.entry_moves.moves, frame->function_op,
      diagnostic_emitter, out_analysis, &accepted));
  if (!accepted) {
    return iree_ok_status();
  }
  // Destination marks may be set before a block's own analysis. Each iteration
  // preserves those marks in the once-initialized table.
  for (iree_host_size_t block_index = 0;
       block_index < frame->schedule.block_count; ++block_index) {
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
    loom_aie2p_block_analysis_t* block_analysis = &blocks[block_index];
    block_analysis->terminator_packet_index = UINT32_MAX;

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
        continue;
      }
      const loom_aie2p_core_structure_kind_t structure_kind =
          loom_aie2p_core_structure_classify(packet.node->op);
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_STORAGE_ADDRESS) {
        IREE_ASSERT_LT(out_analysis->storage_fixup_count,
                       expected_storage_fixup_count);
        accepted = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_storage_address(
            frame, &packet, storage_layout, diagnostic_emitter,
            &out_analysis->storage_fixups[out_analysis->storage_fixup_count],
            &accepted));
        if (!accepted) {
          return iree_ok_status();
        }
        out_analysis->address_fixup_indices_by_packet[packet.packet_index] =
            (uint32_t)out_analysis->storage_fixup_count++;
        continue;
      }
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_REGISTER_MOVE) {
        const loom_low_allocation_packet_move_group_t* group =
            loom_aie2p_bundle_plan_packet_move_group(frame, &packet);
        if (group != NULL) {
          accepted = false;
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
              frame, group->move_group.moves, packet.node->op,
              diagnostic_emitter, out_analysis, &accepted));
          if (!accepted) {
            return iree_ok_status();
          }
        }
        continue;
      }
      if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_CALL) {
        const loom_low_allocation_call_moves_t* call_moves =
            loom_low_allocation_find_call_moves_by_source_ordinal(
                &frame->allocation, packet.node_index);
        IREE_ASSERT(call_moves != NULL &&
                    "admitted AIE2P call must retain ABI transport");
        accepted = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
            frame, call_moves->arguments.moves, packet.node->op,
            diagnostic_emitter, out_analysis, &accepted));
        if (!accepted) {
          return iree_ok_status();
        }
        accepted = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
            frame, call_moves->results.moves, packet.node->op,
            diagnostic_emitter, out_analysis, &accepted));
        if (!accepted) {
          return iree_ok_status();
        }
        const iree_host_size_t control_bundle_count =
            loom_aie2p_bundle_plan_control_bundle_count(
                descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_CALL_DIRECT);
        if (!iree_host_size_checked_add(
                out_analysis->control_bundle_capacity, control_bundle_count,
                &out_analysis->control_bundle_capacity) ||
            out_analysis->call_fixup_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "AIE2P call capacity exceeds host size");
        }
        ++out_analysis->call_fixup_count;
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
      block_analysis->terminator = terminator;
      block_analysis->terminator_packet_index = (uint32_t)packet.packet_index;
      block_analysis->terminator_issue_cycle = packet.node->issue_cycle;
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
            &frame->schedule,
            loom_direct_branch_dest(terminator_packet.node->op));
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
      case LOOM_AIE2P_BLOCK_TERMINATOR_RETURN: {
        const loom_low_allocation_exit_moves_t* exit_moves =
            loom_low_allocation_find_exit_moves_by_source_ordinal(
                &frame->allocation, terminator_packet.node_index);
        if (exit_moves != NULL) {
          block_analysis->return_moves = exit_moves->results.moves;
          accepted = false;
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
              frame, block_analysis->return_moves, terminator_packet.node->op,
              diagnostic_emitter, out_analysis, &accepted));
          if (!accepted) {
            return iree_ok_status();
          }
        }
        control_bundle_count = loom_aie2p_bundle_plan_control_bundle_count(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_);
        break;
      }
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
      const uint32_t target_block_index =
          block_analysis->branches.values[i].target_block_index;
      // Repeated backward transfers must not pay an alignment NOP prefix on
      // every iteration. This direct-entry requirement wins over forward edges.
      const loom_aie2p_block_alignment_t alignment =
          target_block_index <= block_index
              ? LOOM_AIE2P_BLOCK_ALIGNMENT_DIRECT
              : LOOM_AIE2P_BLOCK_ALIGNMENT_NOP_PREFIX;
      blocks[target_block_index].alignment =
          iree_max(blocks[target_block_index].alignment, alignment);
    }
    if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH) {
      const loom_low_allocation_edge_copy_group_t* edge_copy_group =
          loom_low_allocation_find_edge_copy_group_by_source_ordinal(
              &frame->allocation, terminator_packet.node_index);
      if (edge_copy_group != NULL) {
        accepted = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze_move_range(
            frame, edge_copy_group->move_group.moves,
            terminator_packet.node->op, diagnostic_emitter, out_analysis,
            &accepted));
        if (!accepted) {
          return iree_ok_status();
        }
      }
    }
  }
  IREE_ASSERT_EQ(out_analysis->storage_fixup_count,
                 expected_storage_fixup_count);
  out_analysis->blocks = blocks;
  loom_aie2p_bundle_plan_analyze_hardware_loops(frame, out_analysis);
  *out_accepted = true;
  return iree_ok_status();
}
