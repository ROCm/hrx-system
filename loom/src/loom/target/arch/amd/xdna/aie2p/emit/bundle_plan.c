// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"

#include <stddef.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

typedef struct loom_aie2p_bundle_plan_analysis_t {
  // Structural Low return packet materialized as physical RET.
  uint32_t return_packet_index;
  // Logical issue cycle at which the structural return completes.
  uint32_t return_issue_cycle;
  // Greatest logical issue cycle in the single scheduled block.
  uint32_t maximum_issue_cycle;
} loom_aie2p_bundle_plan_analysis_t;

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

static bool loom_aie2p_bundle_plan_structural_move_isa(const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_move_isa(op) ||
         loom_low_slice_isa(op) || loom_low_concat_isa(op);
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

static iree_status_t loom_aie2p_bundle_plan_analyze(
    const loom_low_emission_frame_t* frame,
    loom_aie2p_bundle_plan_analysis_t* out_analysis) {
  *out_analysis = (loom_aie2p_bundle_plan_analysis_t){
      .return_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
  };
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
  if (frame->schedule.block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P core emission currently requires exactly one Low block");
  }

  const loom_low_schedule_block_t* block = &frame->schedule.blocks[0];
  if (block->issue_group_count != 0) {
    const loom_low_schedule_issue_group_t* last_group =
        &frame->schedule.issue_groups[block->issue_group_start +
                                      block->issue_group_count - 1];
    out_analysis->maximum_issue_cycle = last_group->issue_cycle;
  }
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(&frame->schedule); ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&frame->schedule, packet_index);
    if (loom_low_packet_is_compile_time_only(&packet)) continue;
    if (packet.descriptor != NULL) {
      const uint32_t descriptor_ordinal =
          loom_aie2p_bundle_plan_descriptor_ordinal(descriptor_set,
                                                    packet.descriptor);
      if (descriptor_ordinal == AIE2P_CORE_DESCRIPTOR_REF_RETURN_) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P physical RET is materialized from structural low.return; "
            "descriptor-backed ret packets are not accepted");
      }
      continue;
    }
    if (loom_aie2p_bundle_plan_structural_move_isa(packet.node->op)) {
      continue;
    }
    if (!loom_low_return_isa(packet.node->op)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P core emission does not yet lower structural operation %u",
          (unsigned)packet.node->op->kind);
    }
    if (out_analysis->return_packet_index !=
        LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P Low leaf has more than one return");
    }
    out_analysis->return_packet_index = (uint32_t)packet_index;
    out_analysis->return_issue_cycle = packet.node->issue_cycle;
  }
  if (out_analysis->return_packet_index == LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P Low leaf has no structural return");
  }
  if (out_analysis->return_issue_cycle < out_analysis->maximum_issue_cycle) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P structural return does not complete the "
                            "scheduled block");
  }
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

static iree_status_t loom_aie2p_bundle_plan_append_bundle(
    uint32_t logical_issue_cycle, iree_host_size_t slot_start,
    iree_host_size_t slot_count, const loom_aie2p_planned_slot_t* slots,
    loom_aie2p_planned_bundle_t* bundles,
    iree_host_size_t* inout_bundle_count) {
  if (slot_count > LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P logical issue cycle %u requires %zu physical slots in one "
        "ordered segment",
        (unsigned)logical_issue_cycle, slot_count);
  }
  loom_aie2p_slot_t physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    physical_slots[i] = slots[slot_start + i].encoded_slot.slot;
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
  loom_aie2p_bundle_format_info_t format_info;
  const bool found =
      loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
  IREE_ASSERT(found && format_info.slot_count == slot_count &&
              format_info.bit_count % 8 == 0);
  const uint32_t physical_issue_cycle = (uint32_t)*inout_bundle_count;
  bundles[(*inout_bundle_count)++] = (loom_aie2p_planned_bundle_t){
      .issue_cycle = physical_issue_cycle,
      .logical_issue_cycle = logical_issue_cycle,
      .slot_start = (uint32_t)slot_start,
      .format = format,
      .slot_count = (uint8_t)slot_count,
  };
  return iree_ok_status();
}

// Partitions one scheduled descriptor run into the minimum number of
// contiguous physical bundles. AIE2P's exact bundle-format domain is not
// downward closed: a wide format can exist while one of its slot subsets does
// not. Keeping the scheduled order while minimizing contiguous partitions
// preserves every dependency and timing separation without rejecting those
// representable runs.
static iree_status_t loom_aie2p_bundle_plan_append_descriptor_run(
    uint32_t logical_issue_cycle, iree_host_size_t slot_start,
    iree_host_size_t slot_count, const loom_aie2p_planned_slot_t* slots,
    loom_aie2p_planned_bundle_t* bundles,
    iree_host_size_t* inout_bundle_count) {
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
        physical_slots[i] = slots[slot_start + begin + i].encoded_slot.slot;
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
        logical_issue_cycle, slot_start + begin, end - begin, slots, bundles,
        inout_bundle_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_bundle_plan_append_nop_bundle(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t logical_issue_cycle, loom_aie2p_planned_bundle_t* bundles,
    iree_host_size_t* inout_bundle_count, loom_aie2p_planned_slot_t* slots,
    iree_host_size_t* inout_slot_count) {
  const iree_host_size_t slot_start = *inout_slot_count;
  slots[(*inout_slot_count)++] = (loom_aie2p_planned_slot_t){
      .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
          descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP),
      .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
      .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
  };
  return loom_aie2p_bundle_plan_append_bundle(
      logical_issue_cycle, slot_start, 1, slots, bundles, inout_bundle_count);
}

static iree_status_t loom_aie2p_bundle_plan_try_place_return(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t return_packet_index, iree_host_size_t candidate_bundle_index,
    loom_aie2p_planned_bundle_t* bundles, iree_host_size_t bundle_count,
    loom_aie2p_planned_slot_t* slots, iree_host_size_t* inout_slot_count,
    bool* out_placed) {
  *out_placed = false;
  loom_aie2p_planned_bundle_t* candidate_bundle =
      &bundles[candidate_bundle_index];
  loom_aie2p_planned_slot_t return_slot = {
      .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
          descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
      .scheduled_packet_index = return_packet_index,
      .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
  };
  loom_aie2p_slot_t physical_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  const iree_host_size_t candidate_slot_start = candidate_bundle->slot_start;
  const bool replaces_nop =
      candidate_bundle->slot_count == 1 &&
      iree_any_bit_set(slots[candidate_slot_start].flags,
                       LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
  const iree_host_size_t retained_slot_count =
      replaces_nop ? 0 : candidate_bundle->slot_count;
  if (retained_slot_count == LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < retained_slot_count; ++i) {
    physical_slots[i] = slots[candidate_slot_start + i].encoded_slot.slot;
  }
  physical_slots[retained_slot_count] = return_slot.encoded_slot.slot;
  const iree_host_size_t candidate_slot_count = retained_slot_count + 1;
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(physical_slots,
                                                       candidate_slot_count);
  if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
    return iree_ok_status();
  }

  if (replaces_nop) {
    slots[candidate_slot_start] = return_slot;
  } else {
    const iree_host_size_t insert_index =
        candidate_slot_start + candidate_bundle->slot_count;
    memmove(&slots[insert_index + 1], &slots[insert_index],
            (*inout_slot_count - insert_index) * sizeof(*slots));
    slots[insert_index] = return_slot;
    ++*inout_slot_count;
    for (iree_host_size_t i = candidate_bundle_index + 1; i < bundle_count;
         ++i) {
      ++bundles[i].slot_start;
    }
  }
  candidate_bundle->format = format;
  candidate_bundle->slot_count = (uint8_t)candidate_slot_count;
  *out_placed = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_bundle_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_aie2p_bundle_plan_t){0};

  loom_aie2p_bundle_plan_analysis_t analysis;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_analyze(frame, &analysis));
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  const loom_low_descriptor_t* return_descriptor =
      &descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_RETURN_];
  const uint8_t return_delay_slot_count =
      loom_aie2p_bundle_plan_instruction_info(return_descriptor->encoding_id)
          .delay_slot_count;
  const loom_low_schedule_block_t* block = &frame->schedule.blocks[0];
  const uint64_t maximum_bundle_count =
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE / 2u;
  const uint64_t logical_bundle_count_64 =
      (uint64_t)analysis.maximum_issue_cycle + 1u;
  if (logical_bundle_count_64 > maximum_bundle_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P bundle plan exceeds the 16 KiB core program memory");
  }
  iree_host_size_t bundle_capacity = (iree_host_size_t)logical_bundle_count_64;
  if (!iree_host_size_checked_add(bundle_capacity,
                                  frame->schedule.scheduled_node_count,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(bundle_capacity,
                                  frame->allocation.packet_move_count,
                                  &bundle_capacity) ||
      !iree_host_size_checked_add(
          bundle_capacity, (iree_host_size_t)return_delay_slot_count + 1u,
          &bundle_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P bundle-plan storage capacity exceeds host size");
  }
  // The same upper bound covers slots: each scheduled node and allocation
  // move can contribute at most one slot, while every empty logical cycle and
  // return-delay cycle contributes one synthetic slot.
  const iree_host_size_t slot_capacity = bundle_capacity;

  loom_aie2p_planned_bundle_t* bundles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bundle_capacity, sizeof(*bundles), (void**)&bundles));
  loom_aie2p_planned_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, slot_capacity, sizeof(*slots), (void**)&slots));
  iree_host_size_t bundle_count = 0;
  iree_host_size_t slot_count = 0;
  uint32_t issue_group_index = 0;
  for (uint32_t issue_cycle = 0; issue_cycle <= analysis.maximum_issue_cycle;
       ++issue_cycle) {
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
    const iree_host_size_t cycle_bundle_start = bundle_count;
    iree_host_size_t segment_slot_start = slot_count;
    if (group != NULL) {
      for (uint32_t i = 0; i < group->scheduled_node_count; ++i) {
        const uint32_t packet_index = group->scheduled_node_start + i;
        const loom_low_packet_view_t packet =
            loom_low_packet_at(&frame->schedule, packet_index);
        if (loom_low_packet_is_compile_time_only(&packet) ||
            loom_low_return_isa(packet.node->op)) {
          continue;
        }
        if (packet.descriptor != NULL) {
          loom_aie2p_encoded_slot_t encoded_slot;
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_packet(
              frame, &packet, &encoded_slot));
          slots[slot_count++] = (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_slot,
              .scheduled_packet_index = packet_index,
          };
          continue;
        }

        const loom_low_allocation_packet_move_group_t* move_group =
            loom_aie2p_bundle_plan_packet_move_group(frame, &packet);
        if (move_group == NULL) continue;
        if (slot_count != segment_slot_start) {
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_descriptor_run(
              issue_cycle, segment_slot_start, slot_count - segment_slot_start,
              slots, bundles, &bundle_count));
          segment_slot_start = slot_count;
        }
        for (iree_host_size_t move_ordinal = 0;
             move_ordinal < move_group->move_group.moves.count;
             ++move_ordinal) {
          const iree_host_size_t move_index =
              move_group->move_group.moves.start + move_ordinal;
          loom_aie2p_encoded_slot_t encoded_move;
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_move(
              frame, &frame->allocation.moves[move_index], &encoded_move));
          const iree_host_size_t move_slot_start = slot_count;
          slots[slot_count++] = (loom_aie2p_planned_slot_t){
              .encoded_slot = encoded_move,
              .scheduled_packet_index = packet_index,
              .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
          };
          IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
              issue_cycle, move_slot_start, 1, slots, bundles, &bundle_count));
          segment_slot_start = slot_count;
        }
      }
    }
    if (slot_count != segment_slot_start) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_descriptor_run(
          issue_cycle, segment_slot_start, slot_count - segment_slot_start,
          slots, bundles, &bundle_count));
    }
    if (bundle_count == cycle_bundle_start) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          descriptor_set, issue_cycle, bundles, &bundle_count, slots,
          &slot_count));
    }
  }

  IREE_ASSERT_GT(bundle_count, 0u);
  const iree_host_size_t candidate_return_cycle =
      bundle_count - 1 >= return_delay_slot_count
          ? bundle_count - 1 - return_delay_slot_count
          : 0;
  bool return_placed = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_try_place_return(
      descriptor_set, analysis.return_packet_index, candidate_return_cycle,
      bundles, bundle_count, slots, &slot_count, &return_placed));
  if (return_placed) {
    const iree_host_size_t required_bundle_count =
        candidate_return_cycle + return_delay_slot_count + 1;
    while (bundle_count < required_bundle_count) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          descriptor_set, analysis.return_issue_cycle, bundles, &bundle_count,
          slots, &slot_count));
    }
  } else {
    const iree_host_size_t return_slot_start = slot_count;
    slots[slot_count++] = (loom_aie2p_planned_slot_t){
        .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
        .scheduled_packet_index = analysis.return_packet_index,
        .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_bundle(
        analysis.return_issue_cycle, return_slot_start, 1, slots, bundles,
        &bundle_count));
    for (uint8_t i = 0; i < return_delay_slot_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_append_nop_bundle(
          descriptor_set, analysis.return_issue_cycle, bundles, &bundle_count,
          slots, &slot_count));
    }
  }

  if (bundle_count > maximum_bundle_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P bundle plan exceeds the 16 KiB core program memory");
  }
  iree_host_size_t encoded_byte_length = 0;
  for (iree_host_size_t i = 0; i < bundle_count; ++i) {
    loom_aie2p_bundle_format_info_t format_info;
    const bool found = loom_aie2p_encoding_query_bundle_format_info(
        bundles[i].format, &format_info);
    IREE_ASSERT(found && format_info.slot_count == bundles[i].slot_count &&
                format_info.bit_count % 8 == 0);
    encoded_byte_length += format_info.bit_count / 8;
  }
  if (encoded_byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf requires %zu bytes of 16 KiB program memory",
        encoded_byte_length);
  }

  *out_plan = (loom_aie2p_bundle_plan_t){
      .frame = frame,
      .bundles = bundles,
      .bundle_count = bundle_count,
      .slots = slots,
      .slot_count = slot_count,
      .encoded_byte_length = encoded_byte_length,
  };
  return iree_ok_status();
}
