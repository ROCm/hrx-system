// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"

#include <stddef.h>

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

static const loom_low_schedule_issue_group_t*
loom_aie2p_bundle_plan_find_issue_group(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_block_t* block, uint32_t issue_cycle) {
  for (uint32_t i = 0; i < block->issue_group_count; ++i) {
    const loom_low_schedule_issue_group_t* group =
        &schedule->issue_groups[block->issue_group_start + i];
    if (group->issue_cycle == issue_cycle) return group;
    if (group->issue_cycle > issue_cycle) break;
  }
  return NULL;
}

static bool loom_aie2p_bundle_plan_append_slot(loom_aie2p_slot_t slot,
                                               loom_aie2p_slot_t* slots,
                                               iree_host_size_t* slot_count) {
  for (iree_host_size_t i = 0; i < *slot_count; ++i) {
    if (slots[i] == slot) return false;
  }
  if (*slot_count == LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT) return false;
  slots[(*slot_count)++] = slot;
  return true;
}

static bool loom_aie2p_bundle_plan_gather_cycle_slots(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_issue_group_t* group, bool include_return,
    loom_aie2p_slot_t* slots, iree_host_size_t* out_slot_count) {
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  iree_host_size_t slot_count = 0;
  if (group != NULL) {
    for (uint32_t i = 0; i < group->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index = group->scheduled_node_start + i;
      const loom_low_packet_view_t packet =
          loom_low_packet_at(schedule, packet_index);
      if (loom_low_packet_is_compile_time_only(&packet) ||
          packet.descriptor == NULL) {
        continue;
      }
      const loom_aie2p_instruction_info_t info =
          loom_aie2p_bundle_plan_instruction_info(
              packet.descriptor->encoding_id);
      if (!loom_aie2p_bundle_plan_append_slot(info.slot, slots, &slot_count)) {
        return false;
      }
    }
  }
  if (include_return) {
    const loom_low_descriptor_set_t* descriptor_set =
        frame->target.descriptor_set;
    const loom_low_descriptor_t* return_descriptor =
        &descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_RETURN_];
    const loom_aie2p_instruction_info_t return_info =
        loom_aie2p_bundle_plan_instruction_info(return_descriptor->encoding_id);
    if (!loom_aie2p_bundle_plan_append_slot(return_info.slot, slots,
                                            &slot_count)) {
      return false;
    }
  }
  *out_slot_count = slot_count;
  return true;
}

static loom_aie2p_bundle_format_id_t loom_aie2p_bundle_plan_select_cycle_format(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_issue_group_t* group, bool include_return,
    iree_host_size_t* out_slot_count) {
  loom_aie2p_slot_t slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  iree_host_size_t slot_count = 0;
  if (!loom_aie2p_bundle_plan_gather_cycle_slots(frame, group, include_return,
                                                 slots, &slot_count)) {
    return LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
  }
  if (slot_count == 0) {
    const loom_low_descriptor_set_t* descriptor_set =
        frame->target.descriptor_set;
    const loom_low_descriptor_t* nop_descriptor =
        &descriptor_set->descriptors[AIE2P_CORE_DESCRIPTOR_REF_NOP];
    slots[slot_count++] =
        loom_aie2p_bundle_plan_instruction_info(nop_descriptor->encoding_id)
            .slot;
  }
  *out_slot_count = slot_count;
  return loom_aie2p_encoding_find_bundle_format_for_slots(slots, slot_count);
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

static iree_status_t loom_aie2p_bundle_plan_encode_cycle(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_issue_group_t* group, uint32_t return_packet_index,
    bool include_return, loom_aie2p_planned_slot_t* out_slots,
    iree_host_size_t* out_slot_count) {
  iree_host_size_t slot_count = 0;
  if (group != NULL) {
    for (uint32_t i = 0; i < group->scheduled_node_count; ++i) {
      const uint32_t packet_index = group->scheduled_node_start + i;
      const loom_low_packet_view_t packet =
          loom_low_packet_at(&frame->schedule, packet_index);
      if (loom_low_packet_is_compile_time_only(&packet) ||
          packet.descriptor == NULL) {
        continue;
      }
      loom_aie2p_encoded_slot_t encoded_slot;
      IREE_RETURN_IF_ERROR(
          loom_aie2p_bundle_plan_encode_packet(frame, &packet, &encoded_slot));
      out_slots[slot_count++] = (loom_aie2p_planned_slot_t){
          .encoded_slot = encoded_slot,
          .scheduled_packet_index = packet_index,
      };
    }
  }
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  if (include_return) {
    out_slots[slot_count++] = (loom_aie2p_planned_slot_t){
        .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_RETURN_),
        .scheduled_packet_index = return_packet_index,
        .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL,
    };
  }
  if (slot_count == 0) {
    out_slots[slot_count++] = (loom_aie2p_planned_slot_t){
        .encoded_slot = loom_aie2p_bundle_plan_encode_structural_descriptor(
            descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_NOP),
        .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
        .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
    };
  }
  *out_slot_count = slot_count;
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
  const uint32_t candidate_return_cycle =
      analysis.return_issue_cycle >= return_delay_slot_count
          ? analysis.return_issue_cycle - return_delay_slot_count
          : 0;
  const loom_low_schedule_block_t* block = &frame->schedule.blocks[0];
  const loom_low_schedule_issue_group_t* candidate_group =
      loom_aie2p_bundle_plan_find_issue_group(&frame->schedule, block,
                                              candidate_return_cycle);
  iree_host_size_t candidate_slot_count = 0;
  const loom_aie2p_bundle_format_id_t candidate_format =
      loom_aie2p_bundle_plan_select_cycle_format(frame, candidate_group,
                                                 /*include_return=*/true,
                                                 &candidate_slot_count);
  const uint64_t return_cycle_64 =
      candidate_format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID
          ? candidate_return_cycle
          : (uint64_t)analysis.maximum_issue_cycle + 1;
  const uint64_t return_delay_end_cycle_64 =
      return_cycle_64 + return_delay_slot_count;
  const uint64_t final_cycle_64 = iree_max(
      return_delay_end_cycle_64, (uint64_t)analysis.maximum_issue_cycle);
  const uint64_t bundle_count_64 = final_cycle_64 + 1;
  const uint64_t maximum_bundle_count =
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE / 2u;
  if (bundle_count_64 > maximum_bundle_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P bundle plan exceeds the 16 KiB core program memory");
  }
  const uint32_t return_cycle = (uint32_t)return_cycle_64;
  const uint32_t final_cycle = (uint32_t)final_cycle_64;
  const iree_host_size_t bundle_count = (iree_host_size_t)bundle_count_64;

  loom_aie2p_planned_bundle_t* bundles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bundle_count, sizeof(*bundles), (void**)&bundles));
  iree_host_size_t slot_count = 0;
  iree_host_size_t encoded_byte_length = 0;
  uint32_t issue_group_index = 0;
  for (uint32_t issue_cycle = 0; issue_cycle <= final_cycle; ++issue_cycle) {
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
    iree_host_size_t cycle_slot_count = 0;
    const loom_aie2p_bundle_format_id_t format =
        loom_aie2p_bundle_plan_select_cycle_format(
            frame, group, issue_cycle == return_cycle, &cycle_slot_count);
    if (format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P issue cycle %u has no legal physical bundle format",
          (unsigned)issue_cycle);
    }
    loom_aie2p_bundle_format_info_t format_info;
    const bool found =
        loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
    IREE_ASSERT(found && format_info.slot_count == cycle_slot_count &&
                format_info.bit_count % 8 == 0);
    bundles[issue_cycle] = (loom_aie2p_planned_bundle_t){
        .issue_cycle = issue_cycle,
        .slot_start = (uint32_t)slot_count,
        .format = format,
        .slot_count = (uint8_t)cycle_slot_count,
    };
    slot_count += cycle_slot_count;
    encoded_byte_length += format_info.bit_count / 8;
  }
  if (encoded_byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P encoded leaf requires %zu bytes of 16 KiB program memory",
        encoded_byte_length);
  }

  loom_aie2p_planned_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, slot_count, sizeof(*slots), (void**)&slots));
  issue_group_index = 0;
  for (uint32_t issue_cycle = 0; issue_cycle <= final_cycle; ++issue_cycle) {
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
    const loom_aie2p_planned_bundle_t* bundle = &bundles[issue_cycle];
    iree_host_size_t encoded_slot_count = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_bundle_plan_encode_cycle(
        frame, group, analysis.return_packet_index, issue_cycle == return_cycle,
        &slots[bundle->slot_start], &encoded_slot_count));
    IREE_ASSERT(encoded_slot_count == bundle->slot_count);
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
