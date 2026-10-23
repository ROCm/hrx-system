// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/physical_issue.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_alignment.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_analysis.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_fixup.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_plan_metadata.h"

// Native control transfers after a returned JL require two intervening issue
// positions. Ordinary caller work may occupy them; otherwise they encode as
// NOPs. This is distinct from JL's five architectural delay positions.
enum { LOOM_AIE2P_RETURN_CONTROL_GAP_CYCLES = 2 };

// AIE2P requires at least 112 bytes from LE setup to the loop-end packet.
enum { LOOM_AIE2P_HARDWARE_LOOP_MINIMUM_SETUP_TO_END_BYTES = 112 };

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

// One realized instruction retained in semantic publication order while a
// hardware-loop body is being planned.
typedef struct loom_aie2p_recurrence_instruction_t {
  // Planned instruction and source-packet identity.
  loom_aie2p_planned_slot_t slot;
  // Concrete descriptor or move binding selected during native realization.
  loom_aie2p_slot_realization_t realization;
  // Physical cycle selected in the first loop-body iteration.
  uint32_t issue_cycle;
} loom_aie2p_recurrence_instruction_t;

typedef struct loom_aie2p_bundle_plan_builder_t {
  // Emission frame being converted into physical bundles.
  const loom_low_emission_frame_t* frame;
  // Target-owned entry and exit mechanics for this function.
  const loom_aie2p_bundle_plan_options_t* options;
  // Arena owning the detached program plan.
  iree_arena_allocator_t* arena;
  // Build-lifetime storage used by transactional alignment selection.
  iree_arena_allocator_t* scratch_arena;
  // Exact AIE2P descriptor set selected by the frame.
  const loom_low_descriptor_set_t* descriptor_set;
  // Machine schedule model selected by the frame.
  const loom_low_schedule_model_t* schedule_model;
  // Arena-backed bundle storage owned by the final plan.
  loom_aie2p_planned_bundle_t* bundles;
  // Number of records available in |bundles|.
  iree_host_size_t bundle_capacity;
  // Number of populated records in |bundles|.
  iree_host_size_t bundle_count;
  // Arena-backed slot storage owned by the final plan.
  loom_aie2p_planned_slot_t* slots;
  // Number of records available in |slots| and |slot_realizations|.
  iree_host_size_t slot_capacity;
  // Number of populated records in |slots|.
  iree_host_size_t slot_count;
  // Arena-backed branch target records owned by the final plan.
  loom_aie2p_planned_branch_fixup_t* branch_fixups;
  // Number of populated records in |branch_fixups|.
  iree_host_size_t branch_fixup_count;
  // Arena-backed direct-call target records owned by the final plan.
  loom_aie2p_planned_call_fixup_t* call_fixups;
  // Number of populated records in |call_fixups|.
  iree_host_size_t call_fixup_count;
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
  // Arena-backed executable-address fixups owned by the final plan.
  loom_aie2p_planned_code_address_fixup_t* code_address_fixups;
  // Number of records available in |code_address_fixups|.
  iree_host_size_t code_address_fixup_capacity;
  // Number of populated records in |code_address_fixups|.
  iree_host_size_t code_address_fixup_count;
  // Arena-backed contribution offsets in source block order.
  uint32_t* block_byte_offsets;
  // Number of source blocks represented by |block_byte_offsets|.
  iree_host_size_t block_count;
  // Exact byte length of all bundles appended so far.
  iree_host_size_t encoded_byte_length;
  // Contiguous NOP bytes after the last instruction with observable behavior.
  iree_host_size_t trailing_nop_byte_length;
  // Source-order block receiving newly appended bundles.
  uint32_t current_block_index;
  // Next physical issue cycle, including implicit NOP gaps.
  uint32_t next_issue_cycle;
  // Physical origin of this block's logical cycles, fixed at block entry.
  uint32_t block_issue_cycle;
  // First bundle in the current synchronous execution segment. Retrospective
  // control placement cannot cross a call boundary that reset physical state.
  iree_host_size_t execution_bundle_start;
  // First issue cycle in the current synchronous execution segment. A segment
  // may begin in an implicit timing gap with no bundle at this cycle.
  uint32_t execution_issue_cycle;
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
  // Semantic-order native instructions captured for cyclic admission.
  loom_aie2p_recurrence_instruction_t* recurrence_instructions;
  // Number of records available in |recurrence_instructions|.
  iree_host_size_t recurrence_instruction_capacity;
  // Number of records captured for the current hardware-loop candidate.
  iree_host_size_t recurrence_instruction_count;
  // Whether native commits belong to the current hardware-loop body.
  bool capture_recurrence_instructions;
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

static void loom_aie2p_bundle_plan_record_recurrence_instruction(
    loom_aie2p_bundle_plan_builder_t* builder, loom_aie2p_planned_slot_t slot,
    loom_aie2p_slot_realization_t realization, uint32_t issue_cycle) {
  if (!builder->capture_recurrence_instructions) {
    return;
  }
  IREE_ASSERT_LT(builder->recurrence_instruction_count,
                 builder->recurrence_instruction_capacity);
  builder->recurrence_instructions[builder->recurrence_instruction_count++] =
      (loom_aie2p_recurrence_instruction_t){
          .slot = slot,
          .realization = realization,
          .issue_cycle = issue_cycle,
      };
}

static iree_host_size_t loom_aie2p_bundle_plan_append_slot(
    loom_aie2p_bundle_plan_builder_t* builder, loom_aie2p_planned_slot_t slot,
    loom_aie2p_slot_realization_t realization) {
  IREE_ASSERT_LT(builder->slot_count, builder->slot_capacity);
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

static loom_aie2p_bundle_alignment_storage_t
loom_aie2p_bundle_plan_alignment_storage(
    loom_aie2p_bundle_plan_builder_t* builder) {
  return (loom_aie2p_bundle_alignment_storage_t){
      .scratch_arena = builder->scratch_arena,
      .bundles = builder->bundles,
      .bundle_count = builder->bundle_count,
      .slots = builder->slots,
      .slot_capacity = builder->slot_capacity,
      .slot_count = builder->slot_count,
      .slot_realizations = builder->slot_realizations,
      .encoded_byte_length = builder->encoded_byte_length,
      .trailing_nop_byte_length = builder->trailing_nop_byte_length,
  };
}

static void loom_aie2p_bundle_plan_commit_alignment_storage(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_bundle_alignment_storage_t* storage) {
  builder->bundle_count = storage->bundle_count;
  builder->slot_count = storage->slot_count;
  builder->encoded_byte_length = storage->encoded_byte_length;
  builder->trailing_nop_byte_length = storage->trailing_nop_byte_length;
}

static iree_status_t loom_aie2p_bundle_plan_try_align_loop_boundary(
    loom_aie2p_bundle_plan_builder_t* builder,
    iree_host_size_t region_bundle_start,
    iree_host_size_t boundary_bundle_index, bool* out_aligned) {
  loom_aie2p_bundle_alignment_storage_t storage =
      loom_aie2p_bundle_plan_alignment_storage(builder);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_alignment_try_loop_boundary(
      &storage, region_bundle_start, boundary_bundle_index, out_aligned));
  loom_aie2p_bundle_plan_commit_alignment_storage(builder, &storage);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_align_call_continuation(
    loom_aie2p_bundle_plan_builder_t* builder,
    iree_host_size_t call_bundle_index, uint32_t continuation_cycle) {
  loom_aie2p_bundle_alignment_storage_t storage =
      loom_aie2p_bundle_plan_alignment_storage(builder);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_alignment_align_call_continuation(
      &storage, call_bundle_index, continuation_cycle));
  loom_aie2p_bundle_plan_commit_alignment_storage(builder, &storage);
  return iree_ok_status();
}

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
  builder->instructions[group_index] = loom_low_physical_instruction_make(
      descriptor_set, builder->schedule_model, descriptor_ordinal, registers);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
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
      loom_aie2p_bundle_encoding_single_slot_byte_length(
          builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP);
  builder->encoded_byte_length += (iree_host_size_t)gap_bytes;
  builder->trailing_nop_byte_length += (iree_host_size_t)gap_bytes;
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
          loom_low_descriptor_operand(descriptor_set, descriptor, j);
      if (operand->write_event_id == LOOM_LOW_TIMING_EVENT_NONE) {
        continue;
      }
      const loom_low_physical_register_t* physical_register =
          &descriptor_set
               ->physical_registers[instruction->physical_registers[j]];
      IREE_ASSERT_EQ(operand->reg_class_alt_count, 1,
                     "AIE2P write operand register-class alternative");
      const loom_low_reg_class_alt_t* alternative =
          &descriptor_set->reg_class_alts[operand->reg_class_alt_start];
      uint32_t part_units =
          loom_aie2p_descriptor_register_part_units(
              alternative->register_part_id) &
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
      loom_aie2p_bundle_encoding_format_byte_length(format, slot_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_advance(builder, issue_cycle));
  IREE_ASSERT_LT(builder->bundle_count, builder->bundle_capacity);
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
  const bool is_nop =
      slot_count == 1 &&
      builder->slots[slot_start].encoded_slot.slot == LOOM_AIE2P_SLOT_NOP;
  builder->trailing_nop_byte_length =
      is_nop ? builder->trailing_nop_byte_length + byte_length : 0;
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
  loom_aie2p_bundle_plan_record_recurrence_instruction(
      builder, builder->slots[slot_start],
      builder->slot_realizations[slot_start], issue_cycle);
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
  if (slot.scheduled_packet_index != LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
    proposed_cycle = iree_max(
        proposed_cycle, loom_low_physical_issue_source_ready_cycle(
                            &builder->issue, slot.scheduled_packet_index));
  }
  uint32_t cycle = loom_low_physical_issue_find_earliest_issue_cycle(
      &builder->issue, builder->instructions, 1, proposed_cycle);
  loom_aie2p_pending_bundle_t* pending = NULL;
  loom_aie2p_bundle_format_id_t format;
  do {
    const uint32_t lookback =
        builder->schedule_model->resource_calendar_lookback_cycles;
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
  loom_aie2p_bundle_plan_record_recurrence_instruction(builder, slot,
                                                       realization, cycle);
  if (slot.scheduled_packet_index != LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
    loom_low_physical_issue_commit_source(&builder->issue,
                                          slot.scheduled_packet_index, cycle);
  }
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
          .encoded_slot =
              loom_aie2p_bundle_encoding_encode_structural_descriptor(
                  builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP),
          .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
      },
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          AIE2P_CORE_DESCRIPTOR_REF_NOP});
  return loom_aie2p_bundle_plan_append_single_slot(builder, logical_issue_cycle,
                                                   slot_start);
}

static iree_status_t loom_aie2p_bundle_plan_try_place_control(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t source_packet_index,
    uint32_t slot_packet_index, uint32_t logical_issue_cycle,
    uint32_t descriptor_ordinal, loom_aie2p_encoded_slot_t encoded_control,
    uint32_t candidate_cycle, iree_host_size_t candidate_bundle_index,
    bool* out_placed) {
  *out_placed = false;
  loom_aie2p_planned_bundle_t* candidate_bundle =
      candidate_bundle_index < builder->bundle_count
          ? &builder->bundles[candidate_bundle_index]
          : NULL;
  const bool inserts_bundle = candidate_bundle == NULL ||
                              candidate_bundle->issue_cycle != candidate_cycle;
  loom_aie2p_planned_slot_t control_slot = {
      .encoded_slot = encoded_control,
      .scheduled_packet_index = slot_packet_index,
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
  physical_slots[retained_slot_count] = control_slot.encoded_slot.slot;
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
      builder, &control_slot,
      (loom_aie2p_slot_realization_t){.descriptor_ordinal = descriptor_ordinal},
      (uint16_t)retained_slot_count);
  if (!loom_low_physical_issue_group_fits(
          builder->descriptor_set, builder->schedule_model,
          builder->instructions, (uint16_t)candidate_slot_count) ||
      loom_low_physical_issue_register_ready_cycle(
          &builder->issue, &builder->instructions[retained_slot_count], 1) >
          candidate_cycle) {
    return iree_ok_status();
  }
  // Structural control only reads registers, and its resources are shared
  // only with issue-stage uses, as validated with the target tables. The
  // group check and retained register frontier therefore prove retrospective
  // placement among already-timed tail bundles without replaying the block.

  const uint8_t old_byte_length =
      inserts_bundle
          ? loom_aie2p_bundle_encoding_single_slot_byte_length(
                builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP)
          : loom_aie2p_bundle_encoding_format_byte_length(
                candidate_bundle->format, candidate_bundle->slot_count);
  const uint8_t new_byte_length = loom_aie2p_bundle_encoding_format_byte_length(
      format, candidate_slot_count);
  if (inserts_bundle) {
    // The next row, or the stream endpoint, owns the end of this implicit NOP
    // run. Only the selected cycle becomes a row; the control reservation
    // already covers this instruction and its remaining delay bundles.
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
        .logical_issue_cycle = logical_issue_cycle,
        .byte_offset = byte_offset,
        .slot_start = (uint32_t)candidate_slot_start,
    };
  }
  if (replaces_nop) {
    builder->slots[candidate_slot_start] = control_slot;
    builder->slot_realizations[candidate_slot_start].descriptor_ordinal =
        descriptor_ordinal;
  } else {
    const iree_host_size_t insert_index =
        candidate_slot_start + retained_slot_count;
    memmove(&builder->slots[insert_index + 1], &builder->slots[insert_index],
            (builder->slot_count - insert_index) * sizeof(*builder->slots));
    builder->slots[insert_index] = control_slot;
    memmove(&builder->slot_realizations[insert_index + 1],
            &builder->slot_realizations[insert_index],
            (builder->slot_count - insert_index) *
                sizeof(*builder->slot_realizations));
    builder->slot_realizations[insert_index].descriptor_ordinal =
        descriptor_ordinal;
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
  // A control instruction placed inside the NOP tail shortens the shareable
  // tail. Bytes after this bundle are unchanged even when the bundle itself
  // grows.
  builder->trailing_nop_byte_length =
      iree_min(builder->trailing_nop_byte_length,
               builder->encoded_byte_length - candidate_bundle->byte_offset -
                   old_byte_length);
  builder->encoded_byte_length =
      builder->encoded_byte_length - old_byte_length + new_byte_length;
  candidate_bundle->format = format;
  candidate_bundle->slot_count = (uint8_t)candidate_slot_count;
  IREE_ASSERT(!builder->capture_recurrence_instructions &&
              "hardware-loop bodies cannot contain structural control");
  IREE_RETURN_IF_ERROR(loom_low_physical_issue_commit(
      &builder->issue, &builder->instructions[retained_slot_count], 1,
      candidate_cycle));
  if (slot_packet_index != LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
    loom_low_physical_issue_commit_source(&builder->issue, source_packet_index,
                                          candidate_cycle);
  }
  loom_aie2p_bundle_plan_record_writes(
      builder, &builder->instructions[retained_slot_count], 1);
  *out_placed = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_encode_branch(
    const loom_aie2p_bundle_plan_builder_t* builder,
    uint32_t scheduled_packet_index, uint32_t descriptor_ordinal,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors[descriptor_ordinal];
  IREE_ASSERT(loom_low_descriptor_layout(builder->descriptor_set, descriptor)
                      ->immediate_count == 1 &&
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

static loom_aie2p_encoded_slot_t loom_aie2p_bundle_plan_encode_call(
    const loom_aie2p_bundle_plan_builder_t* builder) {
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set
           ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_CALL_DIRECT];
  IREE_ASSERT_EQ(descriptor->operand_count, 1u);
  const loom_low_operand_t* link =
      loom_low_descriptor_operand(builder->descriptor_set, descriptor, 0);
  IREE_ASSERT_EQ(link->encoding_field_id, 0u);
  IREE_ASSERT_EQ(loom_low_descriptor_layout(builder->descriptor_set, descriptor)
                     ->immediate_count,
                 1u);
  const int64_t target_immediate[] = {0};
  return loom_aie2p_descriptor_encode(
      builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_CALL_DIRECT,
      /*operand_assignments=*/NULL, target_immediate);
}

static iree_status_t loom_aie2p_bundle_plan_append_control(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t source_packet_index,
    uint32_t slot_packet_index, uint32_t logical_issue_cycle,
    iree_host_size_t block_bundle_start, uint32_t minimum_issue_cycle,
    uint32_t descriptor_ordinal, loom_aie2p_encoded_slot_t encoded_control,
    bool close_delay_window, uint32_t* out_bundle_index);

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
    status = loom_aie2p_bundle_encoding_encode_move(builder->frame, parts[i],
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

static iree_status_t loom_aie2p_bundle_plan_append_link_move(
    loom_aie2p_bundle_plan_builder_t* builder, bool save_link,
    uint32_t logical_issue_cycle) {
  const loom_aie2p_physical_register_id_t link_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("lr"));
  IREE_ASSERT(link_register != LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  const loom_low_move_t move = {
      .destination =
          {
              .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              .descriptor_reg_class_id =
                  save_link ? AIE2P_CORE_REG_CLASS_ID_AIE2P_ER
                            : AIE2P_CORE_REG_CLASS_ID_AIE2P_MLRM,
              .location = save_link ? builder->options->link_shadow_register
                                    : link_register,
          },
      .source =
          {
              .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              .descriptor_reg_class_id =
                  save_link ? AIE2P_CORE_REG_CLASS_ID_AIE2P_MLRM
                            : AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
              .location = save_link ? link_register
                                    : builder->options->link_shadow_register,
          },
  };
  return loom_aie2p_bundle_plan_append_move(
      builder, &move, LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE, logical_issue_cycle);
}

static iree_status_t loom_aie2p_bundle_plan_append_edge_moves(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* terminator_packet) {
  const loom_low_allocation_edge_copy_group_t* edge_copy_group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          &builder->frame->allocation, terminator_packet->node_index);
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

static iree_status_t loom_aie2p_bundle_plan_append_move_range(
    loom_aie2p_bundle_plan_builder_t* builder, loom_low_move_range_t moves,
    uint32_t logical_issue_cycle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < moves.count && iree_status_is_ok(status);
       ++i) {
    status = loom_aie2p_bundle_plan_append_move(
        builder, &builder->frame->allocation.moves[moves.start + i],
        LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE, logical_issue_cycle);
  }
  return status;
}

static iree_status_t loom_aie2p_bundle_plan_copy_call_target(
    loom_aie2p_bundle_plan_builder_t* builder, const loom_op_t* call,
    iree_string_view_t* out_name) {
  const loom_symbol_ref_t callee = loom_low_func_call_callee(call);
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, builder->frame->module->symbols.count);
  const loom_symbol_t* symbol =
      &builder->frame->module->symbols.entries[callee.symbol_id];
  const iree_string_view_t name =
      loom_string_table_get(&builder->frame->module->strings, symbol->name_id);
  char* data = NULL;
  if (!iree_string_view_is_empty(name)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(builder->arena, name.size, (void**)&data));
    memcpy(data, name.data, name.size);
  }
  *out_name = iree_make_string_view(data, name.size);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_call(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  const uint32_t packet_index = (uint32_t)packet->packet_index;
  const uint32_t logical_issue_cycle = packet->node->issue_cycle;
  const loom_low_allocation_call_moves_t* moves =
      loom_low_allocation_find_call_moves_by_source_ordinal(
          &builder->frame->allocation, packet->node_index);
  IREE_ASSERT(moves != NULL &&
              "AIE2P call analysis requires retained ABI transport");

  const uint8_t delay_slot_count =
      loom_aie2p_bundle_encoding_instruction_info(
          builder->descriptor_set
              ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_CALL_DIRECT]
              .encoding_id)
          .delay_slot_count;
  const uint32_t control_cycles = (uint32_t)delay_slot_count + 1;
  const uint32_t source_ready_cycle =
      loom_low_physical_issue_source_ready_cycle(&builder->issue, packet_index);

  // Keep the largest complete suffix of resolved parallel moves that fits in
  // JL's architectural delay window. Earlier transport remains before JL;
  // move order is unchanged, including any cycle-breaking scratch move.
  iree_host_size_t delay_move_start = moves->arguments.count;
  iree_host_size_t delay_move_slot_count = 0;
  while (delay_move_start != 0) {
    const loom_low_move_t* move =
        &builder->frame->allocation
             .moves[moves->arguments.start + delay_move_start - 1];
    loom_aie2p_register_move_t parts[2];
    const uint8_t part_count = loom_aie2p_descriptor_move_parts(
        (loom_aie2p_register_move_t){
            .source = (loom_aie2p_physical_register_id_t)move->source.location,
            .destination =
                (loom_aie2p_physical_register_id_t)move->destination.location,
        },
        parts);
    if (delay_move_slot_count + part_count > delay_slot_count) {
      break;
    }
    delay_move_slot_count += part_count;
    --delay_move_start;
  }
  const loom_low_move_range_t before_call_moves = {
      .start = moves->arguments.start,
      .count = delay_move_start,
  };
  const loom_low_move_range_t delay_moves = {
      .start = moves->arguments.start + delay_move_start,
      .count = moves->arguments.count - delay_move_start,
  };

  builder->pending.minimum_issue_cycle =
      iree_max(builder->pending.minimum_issue_cycle, source_ready_cycle);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_move_range(
      builder, before_call_moves, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));

  uint32_t required_delay_end =
      iree_max(builder->next_issue_cycle, source_ready_cycle);
  if (required_delay_end > UINT32_MAX - delay_move_slot_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P call delay window exceeds cycle domain");
  }
  required_delay_end += (uint32_t)delay_move_slot_count;
  const uint32_t minimum_call_issue_cycle = iree_max(
      builder->execution_issue_cycle, required_delay_end > control_cycles
                                          ? required_delay_end - control_cycles
                                          : 0);
  uint32_t call_bundle_index = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_control(
      builder, packet_index, LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
      logical_issue_cycle, builder->execution_bundle_start,
      minimum_call_issue_cycle, AIE2P_CORE_DESCRIPTOR_REF_CALL_DIRECT,
      loom_aie2p_bundle_plan_encode_call(builder),
      /*close_delay_window=*/false, &call_bundle_index));
  const uint32_t call_issue_cycle =
      builder->bundles[call_bundle_index].issue_cycle;
  if (call_issue_cycle > UINT32_MAX - control_cycles) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P call continuation exceeds cycle domain");
  }
  const uint32_t continuation_cycle = call_issue_cycle + control_cycles;

  builder->pending.minimum_issue_cycle =
      iree_max(builder->next_issue_cycle, source_ready_cycle);
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_move_range(
      builder, delay_moves, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  if (builder->next_issue_cycle > continuation_cycle ||
      loom_low_physical_issue_quiescent_cycle(&builder->issue) >
          continuation_cycle) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P argument transport does not fit the direct-call delay window");
  }
  while (builder->next_issue_cycle < continuation_cycle) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_nop_bundle(builder, logical_issue_cycle));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_align_call_continuation(
      builder, call_bundle_index, continuation_cycle));

  loom_aie2p_planned_call_fixup_t* fixup =
      &builder->call_fixups[builder->call_fixup_count++];
  fixup->bundle_index = call_bundle_index;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_copy_call_target(
      builder, packet->node->op, &fixup->target_name));

  // The callee begins after the closed JL delay window and returns only after
  // its own physical stream is quiescent. Its exact clobbers were consumed by
  // allocation; caller-side timing history therefore restarts at continuation.
  loom_low_physical_issue_reset_execution_state(&builder->issue);
  builder->execution_bundle_start = builder->bundle_count;
  if (builder->next_issue_cycle >
      UINT32_MAX - LOOM_AIE2P_RETURN_CONTROL_GAP_CYCLES) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P call continuation exceeds cycle domain");
  }
  // Keep ordinary continuation work available to fill the required gap, but
  // prevent a following call, branch, or return from entering it.
  builder->execution_issue_cycle =
      builder->next_issue_cycle + LOOM_AIE2P_RETURN_CONTROL_GAP_CYCLES;
  builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
  builder->pending.end_cycle = builder->next_issue_cycle;
  const uint32_t result_move_start_cycle = builder->next_issue_cycle;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_move_range(
      builder, moves->results, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  const uint32_t result_available_cycle =
      builder->next_issue_cycle == result_move_start_cycle
          ? result_move_start_cycle
          : builder->next_issue_cycle - 1u;
  loom_low_physical_issue_commit_completed_source(&builder->issue, packet_index,
                                                  result_available_cycle);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_control(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t source_packet_index,
    uint32_t slot_packet_index, uint32_t logical_issue_cycle,
    iree_host_size_t block_bundle_start, uint32_t minimum_issue_cycle,
    uint32_t descriptor_ordinal, loom_aie2p_encoded_slot_t encoded_control,
    bool close_delay_window, uint32_t* out_bundle_index) {
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set->descriptors[descriptor_ordinal];
  const uint8_t delay_slot_count =
      loom_aie2p_bundle_encoding_instruction_info(descriptor->encoding_id)
          .delay_slot_count;
  const uint32_t control_cycles = (uint32_t)delay_slot_count + 1;
  const uint64_t retire_cycle = iree_max(
      (uint64_t)builder->next_issue_cycle,
      loom_aie2p_bundle_plan_quiescent_cycle(builder, source_packet_index));
  const uint64_t earliest_control_cycle =
      retire_cycle > control_cycles ? retire_cycle - control_cycles : 0;
  // At most one hardware delay window is inspected, including implicit NOP
  // positions without stored rows. The block origin excludes predecessor gaps.
  uint64_t candidate_cycle =
      iree_max(earliest_control_cycle, (uint64_t)minimum_issue_cycle);
  iree_host_size_t candidate = builder->bundle_count;
  while (candidate > block_bundle_start &&
         builder->bundles[candidate - 1].issue_cycle >= candidate_cycle) {
    --candidate;
  }
  bool control_placed = false;
  for (; candidate_cycle < builder->next_issue_cycle; ++candidate_cycle) {
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_place_control(
        builder, source_packet_index, slot_packet_index, logical_issue_cycle,
        descriptor_ordinal, encoded_control, (uint32_t)candidate_cycle,
        candidate, &control_placed));
    if (control_placed) {
      break;
    }
    if (candidate < builder->bundle_count &&
        builder->bundles[candidate].issue_cycle == candidate_cycle) {
      ++candidate;
    }
  }
  if (control_placed) {
    *out_bundle_index = (uint32_t)candidate;
    if (!close_delay_window) {
      return iree_ok_status();
    }
    const uint64_t required_cycle = candidate_cycle + control_cycles;
    while (builder->next_issue_cycle < required_cycle) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          builder, logical_issue_cycle));
    }
    return iree_ok_status();
  }

  const iree_host_size_t control_slot_start =
      loom_aie2p_bundle_plan_append_slot(
          builder,
          (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_control,
              .scheduled_packet_index = slot_packet_index,
              .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
          },
          (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                              descriptor_ordinal});
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_advance(builder, candidate_cycle));
  *out_bundle_index = (uint32_t)builder->bundle_count;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_single_slot(
      builder, logical_issue_cycle, control_slot_start));
  if (close_delay_window) {
    for (uint8_t i = 0; i < delay_slot_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          builder, logical_issue_cycle));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_align_next_block(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t logical_issue_cycle,
    loom_aie2p_block_alignment_t alignment, uint32_t* out_byte_offset) {
  // Reuse an aligned address in the existing NOP tail without removing the
  // cycles needed by the predecessor. A taken edge executes only those NOPs
  // before reaching this block; fallthrough keeps its original retirement gap.
  const iree_host_size_t aligned_offset = builder->encoded_byte_length & ~15u;
  if (alignment == LOOM_AIE2P_BLOCK_ALIGNMENT_NOP_PREFIX &&
      builder->encoded_byte_length - aligned_offset <=
          builder->trailing_nop_byte_length) {
    *out_byte_offset = (uint32_t)aligned_offset;
    return iree_ok_status();
  }
  if ((builder->encoded_byte_length & 1u) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P bundle stream cannot reach 16-byte block alignment");
  }
  while ((builder->encoded_byte_length & 15u) != 0) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_nop_bundle(builder, logical_issue_cycle));
  }
  *out_byte_offset = (uint32_t)builder->encoded_byte_length;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_terminator(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_block_analysis_t* block_analysis,
    iree_host_size_t block_bundle_start) {
  if (block_analysis->terminator == LOOM_AIE2P_BLOCK_TERMINATOR_RETURN) {
    if (builder->options->preserve_link_register) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_link_move(
          builder, /*save_link=*/false,
          block_analysis->terminator_issue_cycle));
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
          builder, builder->pending.end_cycle));
    }
    uint32_t bundle_index = 0;
    return loom_aie2p_bundle_plan_append_control(
        builder, block_analysis->terminator_packet_index,
        block_analysis->terminator_packet_index,
        block_analysis->terminator_issue_cycle, block_bundle_start,
        builder->execution_issue_cycle, AIE2P_CORE_DESCRIPTOR_REF_RETURN_,
        loom_aie2p_bundle_encoding_encode_structural_descriptor(
            builder->descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
        /*close_delay_window=*/true, &bundle_index);
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
    loom_aie2p_encoded_slot_t encoded_branch;
    status = loom_aie2p_bundle_plan_encode_branch(
        builder, block_analysis->terminator_packet_index,
        branch->descriptor_ordinal, &encoded_branch);
    uint32_t bundle_index = 0;
    if (iree_status_is_ok(status)) {
      // A second branch follows the first branch's complete delay window.
      const uint32_t minimum_issue_cycle =
          i == 0 ? builder->execution_issue_cycle : builder->next_issue_cycle;
      status = loom_aie2p_bundle_plan_append_control(
          builder, block_analysis->terminator_packet_index,
          block_analysis->terminator_packet_index,
          block_analysis->terminator_issue_cycle, block_bundle_start,
          minimum_issue_cycle, branch->descriptor_ordinal, encoded_branch,
          /*close_delay_window=*/true, &bundle_index);
    }
    if (iree_status_is_ok(status)) {
      builder->branch_fixups[builder->branch_fixup_count++] =
          (loom_aie2p_planned_branch_fixup_t){
              .bundle_index = bundle_index,
              .target_block_index = branch->target_block_index,
          };
    }
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
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_encoding_encode_packet(
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
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_encoding_encode_storage_address(
          builder->frame, builder->descriptor_set, packet, &encoded_slot,
          &storage_space, &storage_byte_offset));
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
          loom_aie2p_bundle_encoding_packet_move_group(builder->frame, packet);
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
    if (structure_kind == LOOM_AIE2P_CORE_STRUCTURE_CALL) {
      return loom_aie2p_bundle_plan_append_call(builder, packet);
    }
  }
  loom_low_physical_issue_commit_source(&builder->issue, packet_index, 0);
  return iree_ok_status();
}

typedef struct loom_aie2p_hardware_loop_snapshot_t {
  // Number of committed bundles before speculative loop construction.
  iree_host_size_t bundle_count;
  // Number of committed slot records before speculative loop construction.
  iree_host_size_t slot_count;
  // Number of committed branch fixups before speculative loop construction.
  iree_host_size_t branch_fixup_count;
  // Number of committed call fixups before speculative loop construction.
  iree_host_size_t call_fixup_count;
  // Number of committed storage fixups before speculative loop construction.
  iree_host_size_t storage_fixup_count;
  // Number of committed constant-data fixups before speculative construction.
  iree_host_size_t read_only_data_fixup_count;
  // Number of committed code-address fixups before speculative construction.
  iree_host_size_t code_address_fixup_count;
  // Encoded program length before speculative loop construction.
  iree_host_size_t encoded_byte_length;
  // Shareable trailing NOP bytes before speculative loop construction.
  iree_host_size_t trailing_nop_byte_length;
  // First bundle in the active synchronous execution segment.
  iree_host_size_t execution_bundle_start;
  // Source block receiving bundles before speculative loop construction.
  uint32_t current_block_index;
  // Next physical issue cycle before speculative loop construction.
  uint32_t next_issue_cycle;
  // Physical origin of the active source block.
  uint32_t block_issue_cycle;
  // First issue cycle in the active synchronous execution segment.
  uint32_t execution_issue_cycle;
  // First cycle after all pending instructions.
  uint32_t pending_end_cycle;
  // Native issue floor for the pending window.
  uint32_t pending_minimum_issue_cycle;
  // Native register units written before speculative loop construction.
  loom_aie2p_register_unit_set_t register_writes;
  // Copied logical availability deadlines indexed by schedule node.
  uint32_t* node_ready_cycles;
  // Copied physical register-event deadlines.
  uint32_t* event_deadlines;
  // First cycle with no retained physical register event.
  uint32_t event_quiescent_cycle;
  // Copied physical resource occupancy calendar.
  loom_low_schedule_resource_calendar_slot_t* resource_slots;
  // First cycle after every retained resource stage.
  uint64_t resource_quiescent_cycle;
  // Latest committed physical issue cycle.
  uint32_t resource_issue_cycle_high_water;
  // Forward-issue floor imposed by saturated resources.
  uint64_t resource_minimum_issue_cycle;
  // First cycle after every retained effect event.
  uint32_t effect_quiescent_cycle;
} loom_aie2p_hardware_loop_snapshot_t;

static iree_status_t loom_aie2p_bundle_plan_snapshot_hardware_loop(
    const loom_aie2p_bundle_plan_builder_t* builder,
    loom_aie2p_hardware_loop_snapshot_t* out_snapshot) {
  *out_snapshot = (loom_aie2p_hardware_loop_snapshot_t){
      .bundle_count = builder->bundle_count,
      .slot_count = builder->slot_count,
      .branch_fixup_count = builder->branch_fixup_count,
      .call_fixup_count = builder->call_fixup_count,
      .storage_fixup_count = builder->storage_fixup_count,
      .read_only_data_fixup_count = builder->read_only_data_fixup_count,
      .code_address_fixup_count = builder->code_address_fixup_count,
      .encoded_byte_length = builder->encoded_byte_length,
      .trailing_nop_byte_length = builder->trailing_nop_byte_length,
      .execution_bundle_start = builder->execution_bundle_start,
      .current_block_index = builder->current_block_index,
      .next_issue_cycle = builder->next_issue_cycle,
      .block_issue_cycle = builder->block_issue_cycle,
      .execution_issue_cycle = builder->execution_issue_cycle,
      .pending_end_cycle = builder->pending.end_cycle,
      .pending_minimum_issue_cycle = builder->pending.minimum_issue_cycle,
      .register_writes = builder->register_writes,
      .event_quiescent_cycle = builder->issue.events.quiescent_cycle,
      .resource_quiescent_cycle = builder->issue.resources.quiescent_cycle,
      .resource_issue_cycle_high_water =
          builder->issue.resources.issue_cycle_high_water,
      .resource_minimum_issue_cycle =
          builder->issue.resources.minimum_issue_cycle,
      .effect_quiescent_cycle = builder->issue.effect_quiescent_cycle,
  };
  const iree_host_size_t node_count = builder->frame->schedule.node_count;
  if (node_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->scratch_arena, node_count,
                                  sizeof(*out_snapshot->node_ready_cycles),
                                  (void**)&out_snapshot->node_ready_cycles));
    memcpy(out_snapshot->node_ready_cycles, builder->issue.node_ready_cycles,
           node_count * sizeof(*out_snapshot->node_ready_cycles));
  }
  const iree_host_size_t event_deadline_count =
      (iree_host_size_t)builder->descriptor_set->physical_register_unit_count *
      builder->schedule_model->timing_event_count;
  if (event_deadline_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->scratch_arena, event_deadline_count,
                                  sizeof(*out_snapshot->event_deadlines),
                                  (void**)&out_snapshot->event_deadlines));
    memcpy(out_snapshot->event_deadlines, builder->issue.events.deadlines,
           event_deadline_count * sizeof(*out_snapshot->event_deadlines));
  }
  const iree_host_size_t resource_slot_count =
      builder->schedule_model->resource_calendar_slot_count;
  if (resource_slot_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->scratch_arena, resource_slot_count,
                                  sizeof(*out_snapshot->resource_slots),
                                  (void**)&out_snapshot->resource_slots));
    memcpy(out_snapshot->resource_slots, builder->issue.resources.slots,
           resource_slot_count * sizeof(*out_snapshot->resource_slots));
  }
  return iree_ok_status();
}

static void loom_aie2p_bundle_plan_restore_hardware_loop(
    loom_aie2p_bundle_plan_builder_t* builder,
    const loom_aie2p_hardware_loop_snapshot_t* snapshot,
    const loom_aie2p_hardware_loop_analysis_t* loop) {
  for (iree_host_size_t i = 0;
       i < builder->frame->schedule.scheduled_node_count; ++i) {
    if (builder->storage_fixup_indices_by_packet != NULL &&
        builder->storage_fixup_indices_by_packet[i] >=
            snapshot->storage_fixup_count) {
      builder->storage_fixup_indices_by_packet[i] = UINT32_MAX;
    }
    if (builder->read_only_data_fixup_indices_by_packet != NULL &&
        builder->read_only_data_fixup_indices_by_packet[i] >=
            snapshot->read_only_data_fixup_count) {
      builder->read_only_data_fixup_indices_by_packet[i] = UINT32_MAX;
    }
  }
  builder->bundle_count = snapshot->bundle_count;
  builder->slot_count = snapshot->slot_count;
  builder->branch_fixup_count = snapshot->branch_fixup_count;
  builder->call_fixup_count = snapshot->call_fixup_count;
  builder->storage_fixup_count = snapshot->storage_fixup_count;
  builder->read_only_data_fixup_count = snapshot->read_only_data_fixup_count;
  builder->code_address_fixup_count = snapshot->code_address_fixup_count;
  builder->encoded_byte_length = snapshot->encoded_byte_length;
  builder->trailing_nop_byte_length = snapshot->trailing_nop_byte_length;
  builder->execution_bundle_start = snapshot->execution_bundle_start;
  builder->current_block_index = snapshot->current_block_index;
  builder->next_issue_cycle = snapshot->next_issue_cycle;
  builder->block_issue_cycle = snapshot->block_issue_cycle;
  builder->execution_issue_cycle = snapshot->execution_issue_cycle;
  builder->pending.end_cycle = snapshot->pending_end_cycle;
  builder->pending.minimum_issue_cycle = snapshot->pending_minimum_issue_cycle;
  builder->register_writes = snapshot->register_writes;
  memset(builder->pending.bundles, 0,
         builder->pending.capacity * sizeof(*builder->pending.bundles));
  const iree_host_size_t node_count = builder->frame->schedule.node_count;
  if (node_count != 0) {
    memcpy(builder->issue.node_ready_cycles, snapshot->node_ready_cycles,
           node_count * sizeof(*builder->issue.node_ready_cycles));
  }
  const iree_host_size_t event_deadline_count =
      (iree_host_size_t)builder->descriptor_set->physical_register_unit_count *
      builder->schedule_model->timing_event_count;
  if (event_deadline_count != 0) {
    memcpy(builder->issue.events.deadlines, snapshot->event_deadlines,
           event_deadline_count * sizeof(*builder->issue.events.deadlines));
  }
  builder->issue.events.quiescent_cycle = snapshot->event_quiescent_cycle;
  const iree_host_size_t resource_slot_count =
      builder->schedule_model->resource_calendar_slot_count;
  if (resource_slot_count != 0) {
    memcpy(builder->issue.resources.slots, snapshot->resource_slots,
           resource_slot_count * sizeof(*builder->issue.resources.slots));
  }
  builder->issue.resources.quiescent_cycle = snapshot->resource_quiescent_cycle;
  builder->issue.resources.issue_cycle_high_water =
      snapshot->resource_issue_cycle_high_water;
  builder->issue.resources.minimum_issue_cycle =
      snapshot->resource_minimum_issue_cycle;
  builder->issue.effect_quiescent_cycle = snapshot->effect_quiescent_cycle;
  builder->block_byte_offsets[loop->header_block_index] = 0;
  builder->block_byte_offsets[loop->body_block_index] = 0;
}

static void loom_aie2p_bundle_plan_begin_block(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t block_index,
    uint32_t block_byte_offset) {
  builder->current_block_index = block_index;
  builder->block_issue_cycle = builder->next_issue_cycle;
  builder->execution_bundle_start = builder->bundle_count;
  builder->execution_issue_cycle =
      iree_max(builder->execution_issue_cycle, builder->next_issue_cycle);
  builder->block_byte_offsets[block_index] = block_byte_offset;
  builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
  builder->pending.end_cycle = builder->next_issue_cycle;
}

static iree_status_t loom_aie2p_bundle_plan_append_block_packets(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t block_index,
    const loom_aie2p_block_analysis_t* block_analysis) {
  const loom_low_schedule_block_t* block =
      &builder->frame->schedule.blocks[block_index];
  for (uint32_t ordinal = 0; ordinal < block->scheduled_node_count; ++ordinal) {
    const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
        &builder->frame->schedule, block_index, ordinal);
    if (packet.packet_index == block_analysis->terminator_packet_index) {
      break;
    }
    const bool is_boundary = iree_any_bit_set(
        packet.node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY);
    if (is_boundary) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
          builder, builder->pending.end_cycle));
      builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
    }
    IREE_RETURN_IF_ERROR(
        loom_aie2p_bundle_plan_append_packet(builder, &packet));
    if (is_boundary) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
          builder, builder->pending.end_cycle));
      builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
    }
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_loop_setup_instruction(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t descriptor_ordinal,
    int64_t immediate, uint32_t logical_issue_cycle) {
  return loom_aie2p_bundle_plan_admit_instruction(
      builder, logical_issue_cycle,
      (loom_aie2p_planned_slot_t){
          .encoded_slot =
              loom_aie2p_bundle_encoding_encode_loop_setup_descriptor(
                  builder->descriptor_set, descriptor_ordinal, immediate),
          .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
          .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_HARDWARE_LOOP_SETUP,
      },
      (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                          descriptor_ordinal});
}

// Proves that the fixed native body can repeat at its encoded issue period.
// Source scheduling has already placed every required cross-iteration effect
// deadline on the body terminator. This check owns only the concrete register
// events and resource reservations introduced by native realization.
static iree_status_t loom_aie2p_bundle_plan_check_hardware_loop_recurrence(
    loom_aie2p_bundle_plan_builder_t* builder, uint32_t body_issue_cycle,
    uint16_t trip_count, bool* out_admitted) {
  *out_admitted = false;
  IREE_ASSERT_NE(builder->recurrence_instruction_count, 0u);
  IREE_ASSERT_LT(body_issue_cycle, builder->next_issue_cycle);

  const uint32_t period = builder->next_issue_cycle - body_issue_cycle;
  const uint64_t native_quiescent_cycle =
      iree_max(builder->issue.resources.quiescent_cycle,
               (uint64_t)builder->issue.events.quiescent_cycle);
  if (native_quiescent_cycle <= builder->next_issue_cycle || trip_count <= 1) {
    *out_admitted = true;
    return iree_ok_status();
  }

  // Copy only the physical histories mutated by instruction admission. Node
  // readiness is immutable here because replay has no semantic source commit.
  loom_low_physical_issue_t replay = builder->issue;
  const iree_host_size_t event_deadline_count =
      (iree_host_size_t)builder->descriptor_set->physical_register_unit_count *
      builder->schedule_model->timing_event_count;
  if (event_deadline_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, event_deadline_count,
        sizeof(*replay.events.deadlines), (void**)&replay.events.deadlines));
    memcpy(replay.events.deadlines, builder->issue.events.deadlines,
           event_deadline_count * sizeof(*replay.events.deadlines));
  }
  const iree_host_size_t resource_slot_count =
      builder->schedule_model->resource_calendar_slot_count;
  if (resource_slot_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, resource_slot_count,
        sizeof(*replay.resources.slots), (void**)&replay.resources.slots));
    memcpy(replay.resources.slots, builder->issue.resources.slots,
           resource_slot_count * sizeof(*replay.resources.slots));
  }

  const uint64_t retained_tail =
      native_quiescent_cycle - builder->next_issue_cycle;
  uint64_t replay_count = retained_tail / period;
  replay_count += retained_tail % period != 0;
  replay_count = iree_min(replay_count, (uint64_t)trip_count - 1u);
  for (uint64_t repetition = 1; repetition <= replay_count; ++repetition) {
    const uint64_t cycle_shift = repetition * period;
    const uint64_t prior_native_quiescent_cycle =
        iree_max(replay.resources.quiescent_cycle,
                 (uint64_t)replay.events.quiescent_cycle);
    for (iree_host_size_t i = 0; i < builder->recurrence_instruction_count;
         ++i) {
      const loom_aie2p_recurrence_instruction_t* instruction =
          &builder->recurrence_instructions[i];
      const uint64_t proposed_cycle = instruction->issue_cycle + cycle_shift;
      if (proposed_cycle > UINT32_MAX) {
        return iree_ok_status();
      }
      // Intermediate repetitions must be complete to construct every live
      // overlap in the final repetition. There, only instructions issued
      // before prior native state quiesces can invalidate the fixed pattern.
      // Publication order is not chronological, so filter each record.
      if (repetition == replay_count &&
          proposed_cycle >= prior_native_quiescent_cycle) {
        continue;
      }
      loom_aie2p_bundle_plan_instruction(builder, &instruction->slot,
                                         instruction->realization, 0);
      const uint32_t admitted_cycle =
          loom_low_physical_issue_find_earliest_issue_cycle(
              &replay, builder->instructions, 1, (uint32_t)proposed_cycle);
      if (admitted_cycle != proposed_cycle) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_low_physical_issue_commit(
          &replay, builder->instructions, 1, admitted_cycle));
    }
  }
  *out_admitted = true;
  return iree_ok_status();
}

static bool loom_aie2p_bundle_plan_bundle_has_loop_setup_descriptor(
    const loom_aie2p_bundle_plan_builder_t* builder,
    iree_host_size_t bundle_index, uint32_t descriptor_ordinal) {
  const loom_aie2p_planned_bundle_t* bundle = &builder->bundles[bundle_index];
  for (uint8_t i = 0; i < bundle->slot_count; ++i) {
    const iree_host_size_t slot_index = bundle->slot_start + i;
    if (iree_any_bit_set(builder->slots[slot_index].flags,
                         LOOM_AIE2P_PLANNED_SLOT_FLAG_HARDWARE_LOOP_SETUP) &&
        builder->slot_realizations[slot_index].descriptor_ordinal ==
            descriptor_ordinal) {
      return true;
    }
  }
  return false;
}

static loom_aie2p_block_alignment_t
loom_aie2p_bundle_plan_effective_block_alignment(
    const loom_aie2p_bundle_plan_analysis_t* analysis,
    iree_host_size_t block_count, uint32_t target_block_index) {
  loom_aie2p_block_alignment_t alignment = LOOM_AIE2P_BLOCK_ALIGNMENT_NONE;
  for (uint32_t source_block_index = 0; source_block_index < block_count;
       ++source_block_index) {
    const loom_aie2p_block_analysis_t* source =
        &analysis->blocks[source_block_index];
    if (source->elide_control) {
      continue;
    }
    for (uint8_t i = 0; i < source->branches.count; ++i) {
      if (source->branches.values[i].target_block_index != target_block_index) {
        continue;
      }
      alignment =
          iree_max(alignment, target_block_index <= source_block_index
                                  ? LOOM_AIE2P_BLOCK_ALIGNMENT_DIRECT
                                  : LOOM_AIE2P_BLOCK_ALIGNMENT_NOP_PREFIX);
    }
  }
  return alignment;
}

static iree_status_t loom_aie2p_bundle_plan_try_append_hardware_loop_impl(
    loom_aie2p_bundle_plan_builder_t* builder,
    loom_aie2p_bundle_plan_analysis_t* analysis,
    const loom_aie2p_hardware_loop_analysis_t* loop,
    const loom_aie2p_hardware_loop_snapshot_t* snapshot, bool* out_selected) {
  IREE_ASSERT_EQ(builder->current_block_index, loop->preheader_block_index);
  builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
  builder->pending.end_cycle = builder->next_issue_cycle;

  const uint32_t logical_issue_cycle =
      analysis->blocks[loop->preheader_block_index].terminator_issue_cycle;
  const iree_host_size_t setup_bundle_start = builder->bundle_count;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_loop_setup_instruction(
      builder, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_COUNT_SHORT,
      loop->trip_count, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_loop_setup_instruction(
      builder, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_START_ADDRESS,
      /*immediate=*/0, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  if (builder->bundle_count != setup_bundle_start + 1u ||
      !loom_aie2p_bundle_plan_bundle_has_loop_setup_descriptor(
          builder, setup_bundle_start,
          AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_COUNT_SHORT) ||
      !loom_aie2p_bundle_plan_bundle_has_loop_setup_descriptor(
          builder, setup_bundle_start,
          AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_START_ADDRESS)) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_loop_setup_instruction(
      builder, AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_END_ADDRESS,
      /*immediate=*/0, logical_issue_cycle));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  if (builder->bundle_count != setup_bundle_start + 2u ||
      builder->bundles[setup_bundle_start + 1u].issue_cycle !=
          builder->bundles[setup_bundle_start].issue_cycle + 1u ||
      !loom_aie2p_bundle_plan_bundle_has_loop_setup_descriptor(
          builder, setup_bundle_start + 1u,
          AIE2P_CORE_DESCRIPTOR_REF_STATE_LOOP_END_ADDRESS)) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }
  bool setup_aligned = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_align_loop_boundary(
      builder, setup_bundle_start, builder->bundle_count, &setup_aligned));
  if (!setup_aligned) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }

  const uint32_t body_byte_offset = (uint32_t)builder->encoded_byte_length;
  IREE_ASSERT_EQ(body_byte_offset % LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES,
                 0u);
  builder->block_byte_offsets[loop->header_block_index] = body_byte_offset;
  loom_aie2p_bundle_plan_begin_block(builder, loop->body_block_index,
                                     body_byte_offset);
  const uint32_t body_issue_cycle = builder->next_issue_cycle;
  const iree_host_size_t body_bundle_start = builder->bundle_count;
  builder->recurrence_instruction_count = 0;
  builder->capture_recurrence_instructions = true;
  loom_aie2p_block_analysis_t* body_analysis =
      &analysis->blocks[loop->body_block_index];
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_block_packets(
      builder, loop->body_block_index, body_analysis));
  const loom_low_packet_view_t terminator_packet = loom_low_packet_at(
      &builder->frame->schedule, body_analysis->terminator_packet_index);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_append_edge_moves(builder, &terminator_packet));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      builder, builder->pending.end_cycle));
  const uint32_t semantic_ready_cycle =
      loom_low_physical_issue_source_ready_cycle(
          &builder->issue, body_analysis->terminator_packet_index);
  while (builder->next_issue_cycle < semantic_ready_cycle) {
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
        builder, body_analysis->terminator_issue_cycle));
  }
  builder->capture_recurrence_instructions = false;
  if (builder->bundle_count == body_bundle_start) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }
  bool recurrence_admitted = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_check_hardware_loop_recurrence(
      builder, body_issue_cycle, loop->trip_count, &recurrence_admitted));
  if (!recurrence_admitted) {
    const uint64_t quiescent_cycle =
        loom_low_physical_issue_quiescent_cycle(&builder->issue);
    while (builder->next_issue_cycle < quiescent_cycle) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          builder, body_analysis->terminator_issue_cycle));
    }
  }
  const iree_host_size_t loop_end_bundle_index = builder->bundle_count - 1u;
  bool loop_end_aligned = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_align_loop_boundary(
      builder, body_bundle_start, loop_end_bundle_index, &loop_end_aligned));
  if (!loop_end_aligned) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }

  // LE identifies an entire 16-byte execution group. Keep the continuation
  // out of that group by widening the final loop bundle through its endpoint.
  // Otherwise a short LE bundle would place fallthrough instructions in bytes
  // that the hardware still repeats as part of the loop.
  bool loop_continuation_aligned = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_align_loop_boundary(
      builder, loop_end_bundle_index, builder->bundle_count,
      &loop_continuation_aligned));
  const uint32_t loop_end_byte_offset =
      builder->bundles[loop_end_bundle_index].byte_offset;
  const uint32_t final_setup_byte_offset =
      builder->bundles[setup_bundle_start + 1u].byte_offset;
  if (!loop_continuation_aligned ||
      builder->encoded_byte_length - loop_end_byte_offset !=
          LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES ||
      loop_end_byte_offset < final_setup_byte_offset ||
      loop_end_byte_offset % LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES != 0 ||
      loop_end_byte_offset - final_setup_byte_offset <
          LOOM_AIE2P_HARDWARE_LOOP_MINIMUM_SETUP_TO_END_BYTES) {
    loom_aie2p_bundle_plan_restore_hardware_loop(builder, snapshot, loop);
    return iree_ok_status();
  }

  IREE_ASSERT_LE(builder->code_address_fixup_count + 2u,
                 builder->code_address_fixup_capacity);
  builder->code_address_fixups[builder->code_address_fixup_count++] =
      (loom_aie2p_planned_code_address_fixup_t){
          .bundle_index = (uint32_t)setup_bundle_start,
          .target_byte_offset = body_byte_offset,
      };
  builder->code_address_fixups[builder->code_address_fixup_count++] =
      (loom_aie2p_planned_code_address_fixup_t){
          .bundle_index = (uint32_t)(setup_bundle_start + 1u),
          .target_byte_offset = loop_end_byte_offset,
      };
  analysis->blocks[loop->header_block_index].elide_control = true;
  body_analysis->elide_control = true;

  loom_low_physical_issue_reset_execution_state(&builder->issue);
  builder->execution_bundle_start = builder->bundle_count;
  builder->execution_issue_cycle = builder->next_issue_cycle;
  builder->pending.minimum_issue_cycle = builder->next_issue_cycle;
  builder->pending.end_cycle = builder->next_issue_cycle;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_try_append_hardware_loop(
    loom_aie2p_bundle_plan_builder_t* builder,
    loom_aie2p_bundle_plan_analysis_t* analysis,
    const loom_aie2p_hardware_loop_analysis_t* loop, bool* out_selected) {
  *out_selected = false;
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(builder->scratch_arena);
  loom_aie2p_hardware_loop_snapshot_t snapshot;
  iree_status_t status =
      loom_aie2p_bundle_plan_snapshot_hardware_loop(builder, &snapshot);
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_bundle_plan_try_append_hardware_loop_impl(
        builder, analysis, loop, &snapshot, out_selected);
  }
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}

static iree_status_t loom_aie2p_bundle_plan_build_impl(
    const loom_low_emission_frame_t* frame,
    const loom_aie2p_bundle_plan_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_leaf_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_aie2p_leaf_program_plan_t){0};

  loom_aie2p_bundle_plan_analysis_t analysis;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_bundle_plan_analyze(frame, scratch_arena, &analysis));
  if (options->preserve_link_register) {
    iree_host_size_t link_move_count = 1;
    for (iree_host_size_t i = 0; i < frame->schedule.block_count; ++i) {
      link_move_count +=
          analysis.blocks[i].terminator == LOOM_AIE2P_BLOCK_TERMINATOR_RETURN;
    }
    if (!iree_host_size_checked_add(analysis.move_slot_count, link_move_count,
                                    &analysis.move_slot_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P link-preservation capacity exceeds host "
                              "size");
    }
  }
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  iree_host_size_t alignment_bundle_capacity = 0;
  if (frame->schedule.block_count > 1) {
    const iree_host_size_t boundary_count = frame->schedule.block_count - 1u;
    const uint8_t nop_byte_length =
        loom_aie2p_bundle_encoding_single_slot_byte_length(
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
  // Scheduled nodes and native allocation move parts contribute one slot each.
  // Control capacity reserves one slot per issue position; call alignment may
  // fill every remaining physical slot in all six retained JL packets.
  iree_host_size_t call_alignment_slot_capacity = 0;
  if (!iree_host_size_checked_mul(
          analysis.call_fixup_count,
          LOOM_AIE2P_BUNDLE_ALIGNMENT_CALL_PACKET_COUNT *
              (LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT - 1u),
          &call_alignment_slot_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P call-alignment slot capacity exceeds host size");
  }
  iree_host_size_t slot_capacity = bundle_capacity;
  if (!iree_host_size_checked_add(slot_capacity, call_alignment_slot_capacity,
                                  &slot_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P slot capacity exceeds host size");
  }
  iree_host_size_t loop_alignment_slot_capacity = 0;
  if (!iree_host_size_checked_mul(
          analysis.hardware_loop_candidate_count,
          2u * LOOM_AIE2P_BUNDLE_ALIGNMENT_LOOP_MAX_ADDED_SLOTS,
          &loop_alignment_slot_capacity) ||
      !iree_host_size_checked_add(slot_capacity, loop_alignment_slot_capacity,
                                  &slot_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P hardware-loop alignment capacity exceeds host size");
  }
  if (frame->schedule.block_count > IREE_HOST_SIZE_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P branch-fixup capacity exceeds host size");
  }
  const iree_host_size_t branch_fixup_capacity =
      frame->schedule.block_count * 2u;

  loom_aie2p_bundle_plan_builder_t builder = {
      .frame = frame,
      .options = options,
      .arena = arena,
      .scratch_arena = scratch_arena,
      .descriptor_set = descriptor_set,
      .schedule_model = frame->schedule.target.schedule_model,
      .bundle_capacity = bundle_capacity,
      .slot_capacity = slot_capacity,
      .block_count = frame->schedule.block_count,
      .recurrence_instruction_capacity =
          analysis.hardware_loop_candidate_count != 0 ? slot_capacity : 0,
      .code_address_fixup_capacity =
          analysis.hardware_loop_candidate_count * 2u,
      .pending.capacity = frame->schedule.target.schedule_model
                              ->resource_calendar_lookback_cycles +
                          1,
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
  if (builder.recurrence_instruction_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, builder.recurrence_instruction_capacity,
        sizeof(*builder.recurrence_instructions),
        (void**)&builder.recurrence_instructions));
  }
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
  if (analysis.call_fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, analysis.call_fixup_count, sizeof(*builder.call_fixups),
        (void**)&builder.call_fixups));
  }
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
  if (builder.code_address_fixup_capacity != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, builder.code_address_fixup_capacity,
                                  sizeof(*builder.code_address_fixups),
                                  (void**)&builder.code_address_fixups));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, builder.block_count, sizeof(*builder.block_byte_offsets),
      (void**)&builder.block_byte_offsets));

  // Invocation transport and the caller link save execute once, before the
  // body CFG. A branch to the first body block must not replay either one.
  builder.current_block_index = 0;
  builder.pending.minimum_issue_cycle = 0;
  builder.pending.end_cycle = 0;
  if (options->preserve_link_register) {
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_link_move(
        &builder, /*save_link=*/true, /*logical_issue_cycle=*/0));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_move_range(
      &builder, frame->allocation.entry_moves.moves,
      /*logical_issue_cycle=*/0));
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_flush_pending(
      &builder, builder.pending.end_cycle));

  for (uint32_t block_index = 0;
       block_index < (uint32_t)frame->schedule.block_count;) {
    const loom_aie2p_block_analysis_t* block_analysis =
        &analysis.blocks[block_index];
    if (block_analysis->hardware_loop.is_candidate) {
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_append_hardware_loop(
          &builder, &analysis, &block_analysis->hardware_loop, &selected));
      if (selected) {
        block_index = block_analysis->hardware_loop.body_block_index + 1u;
        continue;
      }
    }
    uint32_t block_byte_offset = (uint32_t)builder.encoded_byte_length;
    const loom_aie2p_block_alignment_t alignment =
        loom_aie2p_bundle_plan_effective_block_alignment(
            &analysis, frame->schedule.block_count, block_index);
    if (alignment != LOOM_AIE2P_BLOCK_ALIGNMENT_NONE) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_align_next_block(
          &builder,
          block_index == 0
              ? 0
              : analysis.blocks[block_index - 1u].terminator_issue_cycle,
          alignment, &block_byte_offset));
    }
    loom_aie2p_bundle_plan_begin_block(&builder, block_index,
                                       block_byte_offset);
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_block_packets(
        &builder, block_index, block_analysis));
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_terminator(
        &builder, block_analysis, builder.execution_bundle_start));
    ++block_index;
  }
  loom_aie2p_bundle_address_fixups_t address_fixups = {
      .scheduled_packet_count = frame->schedule.scheduled_node_count,
      .storage_fixups = builder.storage_fixups,
      .storage_fixup_count = builder.storage_fixup_count,
      .storage_fixup_indices_by_packet =
          builder.storage_fixup_indices_by_packet,
      .read_only_data_fixups = builder.read_only_data_fixups,
      .read_only_data_fixup_count = builder.read_only_data_fixup_count,
      .read_only_data_fixup_indices_by_packet =
          builder.read_only_data_fixup_indices_by_packet,
  };
  loom_aie2p_bundle_fixup_resolve_addresses(
      builder.bundles, builder.bundle_count, builder.slots, &address_fixups);

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
      .call_fixups = builder.call_fixups,
      .call_fixup_count = builder.call_fixup_count,
      .storage_fixups = builder.storage_fixups,
      .storage_fixup_count = builder.storage_fixup_count,
      .read_only_data_fixups = builder.read_only_data_fixups,
      .read_only_data_fixup_count = builder.read_only_data_fixup_count,
      .code_address_fixups = builder.code_address_fixups,
      .code_address_fixup_count = builder.code_address_fixup_count,
      .encoded_byte_length = builder.encoded_byte_length,
  };
  IREE_RETURN_IF_ERROR(
      loom_aie2p_leaf_plan_retain_metadata(frame, arena, &plan));
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame,
    const loom_aie2p_bundle_plan_options_t* options,
    iree_arena_allocator_t* arena, loom_aie2p_leaf_program_plan_t* out_plan) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status = loom_aie2p_bundle_plan_build_impl(
      frame, options, arena, &scratch_arena, out_plan);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
