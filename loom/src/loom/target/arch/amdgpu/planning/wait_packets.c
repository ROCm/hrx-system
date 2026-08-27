// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packets.h"

#include <inttypes.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function.h"
#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE UINT16_MAX

typedef struct loom_amdgpu_wait_packet_group_t {
  // Region block containing the insertion point.
  uint32_t block_index;
  // Schedule node before which wait packets are inserted.
  uint32_t node_index;
  // Scheduled ordinal before which wait packets are inserted.
  uint32_t scheduled_ordinal;
  // First input action in this coalescing group.
  iree_host_size_t source_action_start;
  // Number of input actions in this coalescing group.
  iree_host_size_t source_action_count;
  // Logical counters required by this group.
  uint32_t counter_mask;
  // Target wait count per logical wait-counter slot.
  uint16_t target_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
} loom_amdgpu_wait_packet_group_t;

typedef struct loom_amdgpu_wait_packet_builder_t {
  // Logical wait plan being materialized.
  const loom_amdgpu_wait_plan_t* wait_plan;
  // Descriptor set selected by the scheduled low function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena owning all output rows.
  iree_arena_allocator_t* arena;
  // Concrete wait packet descriptors available on this descriptor set.
  loom_amdgpu_wait_packet_target_t target;
  // Output concrete packet rows.
  loom_amdgpu_wait_packet_t* packets;
  // Number of populated concrete packet rows.
  iree_host_size_t packet_count;
  // Allocated concrete packet row capacity.
  iree_host_size_t packet_capacity;
  // Output immediate rows.
  loom_amdgpu_wait_packet_immediate_t* immediates;
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
  // Allocated immediate row capacity.
  iree_host_size_t immediate_capacity;
} loom_amdgpu_wait_packet_builder_t;

static uint16_t loom_amdgpu_wait_packet_target_count(
    const loom_amdgpu_wait_packet_group_t* group, uint32_t counter_mask) {
  uint16_t target_count = LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE;
  for (uint32_t slot = 0; slot < IREE_ARRAYSIZE(group->target_counts); ++slot) {
    const uint32_t slot_mask = 1u << slot;
    if ((counter_mask & slot_mask) == 0 ||
        group->target_counts[slot] ==
            LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE) {
      continue;
    }
    if (group->target_counts[slot] < target_count) {
      target_count = group->target_counts[slot];
    }
  }
  IREE_ASSERT_NE(target_count, LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE);
  return target_count;
}

static iree_status_t loom_amdgpu_wait_packet_allocate(
    loom_amdgpu_wait_packet_builder_t* builder) {
  iree_host_size_t planned_action_count = 0;
  for (iree_host_size_t i = 0; i < builder->wait_plan->action_count; ++i) {
    if (builder->wait_plan->actions[i].kind ==
        LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
      ++planned_action_count;
    }
  }
  builder->packet_capacity = planned_action_count;
  if (builder->packet_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->packet_capacity, sizeof(*builder->packets),
        (void**)&builder->packets));
  }

  const iree_host_size_t max_immediate_count =
      builder->target.max_descriptor_immediate_count == 0
          ? 1
          : builder->target.max_descriptor_immediate_count;
  IREE_ASSERT_LE(planned_action_count,
                 IREE_HOST_SIZE_MAX / max_immediate_count);
  builder->immediate_capacity = planned_action_count * max_immediate_count;
  if (builder->immediate_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->immediate_capacity,
        sizeof(*builder->immediates), (void**)&builder->immediates));
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_packet_append_immediate(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t*
        immediate_descriptor,
    uint16_t value) {
  IREE_ASSERT_LT(builder->immediate_count, builder->immediate_capacity);
  const uint16_t descriptor_immediate_index =
      immediate_descriptor->descriptor_immediate_index;
  value = iree_min(value, immediate_descriptor->no_wait_value);

  builder->immediates[builder->immediate_count++] =
      (loom_amdgpu_wait_packet_immediate_t){
          .descriptor_immediate_index = descriptor_immediate_index,
          .name = immediate_descriptor->name,
          .value = value,
      };
}

static void loom_amdgpu_wait_packet_append_packet(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor,
    uint32_t counter_mask) {
  IREE_ASSERT_LT(builder->packet_count, builder->packet_capacity);

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_wait_packet_resolve_descriptor(builder->descriptor_set,
                                                 packet_descriptor);

  const iree_host_size_t immediate_start = builder->immediate_count;
  for (uint16_t i = 0; i < packet_descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate =
        loom_amdgpu_wait_packet_descriptor_immediate(packet_descriptor, i);
    uint16_t value = immediate->no_wait_value;
    const uint32_t immediate_counter_mask =
        immediate->counter_mask & counter_mask;
    if (immediate_counter_mask != 0) {
      value =
          loom_amdgpu_wait_packet_target_count(group, immediate_counter_mask);
    }
    loom_amdgpu_wait_packet_append_immediate(builder, immediate, value);
  }
  builder->packets[builder->packet_count++] = (loom_amdgpu_wait_packet_t){
      .descriptor = descriptor,
      .block_index = group->block_index,
      .node_index = group->node_index,
      .scheduled_ordinal = group->scheduled_ordinal,
      .counter_mask = counter_mask,
      .source_action_start = group->source_action_start,
      .source_action_count = group->source_action_count,
      .immediate_start = immediate_start,
      .immediate_count = builder->immediate_count - immediate_start,
  };
}

static const loom_amdgpu_wait_packet_descriptor_template_t*
loom_amdgpu_wait_packet_select_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder, uint32_t remaining_counter_mask,
    uint32_t* out_covered_counter_mask) {
  *out_covered_counter_mask = 0;
  if (remaining_counter_mask >= builder->target.selection_count) {
    return NULL;
  }
  const loom_amdgpu_wait_packet_selection_template_t* selection =
      &builder->target.selections[remaining_counter_mask];
  const uint32_t covered_counter_mask = selection->covered_counter_mask;
  if (covered_counter_mask == 0) {
    return NULL;
  }
  IREE_ASSERT_LT(selection->descriptor_index, builder->target.descriptor_count);
  *out_covered_counter_mask = covered_counter_mask;
  return &builder->target.descriptors[selection->descriptor_index];
}

bool loom_amdgpu_wait_packet_try_select_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    uint16_t target_count, loom_amdgpu_wait_packet_selection_t* out_selection) {
  *out_selection = (loom_amdgpu_wait_packet_selection_t){0};
  if (counter_mask == 0) {
    IREE_ASSERT_UNREACHABLE("AMDGPU wait packet selection requires a counter");
    return false;
  }

  loom_amdgpu_wait_packet_builder_t builder = {
      .descriptor_set = descriptor_set,
  };
  loom_amdgpu_wait_packet_analyze_target(descriptor_set, &builder.target);

  uint32_t covered_counter_mask = 0;
  const loom_amdgpu_wait_packet_descriptor_template_t* packet_descriptor =
      loom_amdgpu_wait_packet_select_descriptor(&builder, counter_mask,
                                                &covered_counter_mask);
  if (packet_descriptor == NULL || covered_counter_mask != counter_mask) {
    return false;
  }
  IREE_ASSERT_LE(packet_descriptor->immediate_count,
                 LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_wait_packet_resolve_descriptor(descriptor_set,
                                                 packet_descriptor);

  *out_selection = (loom_amdgpu_wait_packet_selection_t){
      .descriptor = descriptor,
      .descriptor_ref = packet_descriptor->descriptor_ref,
      .counter_mask = covered_counter_mask,
      .immediate_count = packet_descriptor->immediate_count,
  };
  for (uint16_t i = 0; i < packet_descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_template_t* immediate =
        loom_amdgpu_wait_packet_descriptor_immediate(packet_descriptor, i);
    uint16_t value = immediate->no_wait_value;
    if (iree_any_bit_set(immediate->counter_mask, covered_counter_mask)) {
      value = target_count;
    }
    value = iree_min(value, immediate->no_wait_value);
    out_selection->immediates[i] = (loom_amdgpu_wait_packet_immediate_t){
        .descriptor_immediate_index = immediate->descriptor_immediate_index,
        .name = immediate->name,
        .value = value,
    };
  }
  return true;
}

static void loom_amdgpu_wait_packet_materialize_group(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group) {
  IREE_ASSERT_EQ(group->counter_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL, 0u);
  uint32_t remaining_counter_mask = group->counter_mask;
  while (remaining_counter_mask != 0) {
    uint32_t covered_counter_mask = 0;
    const loom_amdgpu_wait_packet_descriptor_template_t* descriptor =
        loom_amdgpu_wait_packet_select_descriptor(
            builder, remaining_counter_mask, &covered_counter_mask);
    IREE_ASSERT_NE(descriptor, NULL);
    IREE_ASSERT_NE(covered_counter_mask, 0u);
    loom_amdgpu_wait_packet_append_packet(builder, group, descriptor,
                                          covered_counter_mask);
    remaining_counter_mask &= ~covered_counter_mask;
  }
}

static bool loom_amdgpu_wait_packet_same_insertion_point(
    const loom_amdgpu_wait_plan_action_t* lhs,
    const loom_amdgpu_wait_plan_action_t* rhs) {
  return lhs->block_index == rhs->block_index &&
         lhs->node_index == rhs->node_index &&
         lhs->scheduled_ordinal == rhs->scheduled_ordinal;
}

static void loom_amdgpu_wait_packet_group_initialize(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_host_size_t action_index,
    loom_amdgpu_wait_packet_group_t* out_group) {
  const loom_amdgpu_wait_plan_action_t* action =
      &wait_plan->actions[action_index];
  *out_group = (loom_amdgpu_wait_packet_group_t){
      .block_index = action->block_index,
      .node_index = action->node_index,
      .scheduled_ordinal = action->scheduled_ordinal,
      .source_action_start = action_index,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(out_group->target_counts); ++i) {
    out_group->target_counts[i] = LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE;
  }
}

static void loom_amdgpu_wait_packet_group_accumulate(
    loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_plan_action_t* action) {
  const uint32_t counter_mask =
      loom_amdgpu_wait_counter_mask(action->counter_id);
  const uint32_t slot =
      loom_amdgpu_wait_counter_slot_from_id(action->counter_id);
  IREE_ASSERT_LT(slot, IREE_ARRAYSIZE(group->target_counts));
  group->counter_mask |= counter_mask;
  if (action->target_count < group->target_counts[slot]) {
    group->target_counts[slot] = action->target_count;
  }
  ++group->source_action_count;
}

static void loom_amdgpu_wait_packet_build_packets(
    loom_amdgpu_wait_packet_builder_t* builder) {
  const loom_amdgpu_wait_plan_t* wait_plan = builder->wait_plan;
  for (iree_host_size_t i = 0; i < wait_plan->action_count;) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    if (action->kind != LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
      ++i;
      continue;
    }
    loom_amdgpu_wait_packet_group_t group = {0};
    loom_amdgpu_wait_packet_group_initialize(wait_plan, i, &group);
    iree_host_size_t next_action_index = i;
    while (next_action_index < wait_plan->action_count &&
           loom_amdgpu_wait_packet_same_insertion_point(
               action, &wait_plan->actions[next_action_index])) {
      const loom_amdgpu_wait_plan_action_t* group_action =
          &wait_plan->actions[next_action_index];
      if (group_action->kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
        loom_amdgpu_wait_packet_group_accumulate(&group, group_action);
      }
      ++next_action_index;
    }
    loom_amdgpu_wait_packet_materialize_group(builder, &group);
    i = next_action_index;
  }
}

iree_status_t loom_amdgpu_wait_packet_plan_build(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_packet_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(wait_plan);
  IREE_ASSERT_ARGUMENT(wait_plan->schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_wait_packet_plan_t){0};
  loom_amdgpu_wait_packet_builder_t builder = {
      .wait_plan = wait_plan,
      .descriptor_set = wait_plan->schedule->target.descriptor_set,
      .arena = arena,
  };
  loom_amdgpu_wait_packet_analyze_target(builder.descriptor_set,
                                         &builder.target);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_allocate(&builder));
  loom_amdgpu_wait_packet_build_packets(&builder);
  *out_plan = (loom_amdgpu_wait_packet_plan_t){
      .wait_plan = wait_plan,
      .packets = builder.packets,
      .packet_count = builder.packet_count,
      .immediates = builder.immediates,
      .immediate_count = builder.immediate_count,
  };
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_wait_packet_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (module == NULL || !loom_symbol_ref_is_valid(symbol_ref) ||
      symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_amdgpu_wait_packet_json_function_name(
    const loom_amdgpu_wait_packet_plan_t* plan) {
  const loom_low_schedule_table_t* schedule = plan->wait_plan->schedule;
  if (loom_low_function_def_isa(schedule->function_op)) {
    return loom_amdgpu_wait_packet_json_symbol_name(
        schedule->module, loom_low_function_callee(schedule->function_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_status_t loom_amdgpu_wait_packet_json_write_counters(
    loom_output_stream_t* stream, uint32_t counter_mask) {
  loom_json_array_writer_t counters;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &counters));
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t slot_mask = 1u << slot;
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &counters, loom_amdgpu_wait_counter_name(counter_id)));
  }
  return loom_json_array_end(&counters);
}

iree_status_t loom_amdgpu_wait_packet_plan_format_json(
    const loom_amdgpu_wait_packet_plan_t* plan,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_low_schedule_table_t* schedule = plan->wait_plan->schedule;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.amdgpu.wait_packet_plan.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_amdgpu_wait_packet_json_function_name(plan)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), schedule->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"), schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("packet_count"), plan->packet_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("packets")));
  loom_json_array_writer_t packets;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &packets));
  for (iree_host_size_t i = 0; i < plan->packet_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&packets));
    const loom_amdgpu_wait_packet_t* packet = &plan->packets[i];
    loom_json_object_writer_t packet_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &packet_object));
    IREE_ASSERT_NE(packet->descriptor, NULL);
    iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
        schedule->target.descriptor_set, packet->descriptor->key_string_offset);
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &packet_object, IREE_SV("descriptor"), descriptor_key));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &packet_object, IREE_SV("block"), packet->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &packet_object, IREE_SV("node"), packet->node_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &packet_object, IREE_SV("scheduled_ordinal"),
        packet->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &packet_object, IREE_SV("counter_mask"), packet->counter_mask));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&packet_object, IREE_SV("counters")));
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_json_write_counters(
        &stream, packet->counter_mask));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &packet_object, IREE_SV("source_action_start"),
        packet->source_action_start));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &packet_object, IREE_SV("source_action_count"),
        packet->source_action_count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&packet_object, IREE_SV("immediates")));
    loom_json_array_writer_t immediates;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &immediates));
    for (iree_host_size_t j = 0; j < packet->immediate_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&immediates));
      const iree_host_size_t immediate_index = packet->immediate_start + j;
      IREE_ASSERT_LT(immediate_index, plan->immediate_count);
      const loom_amdgpu_wait_packet_immediate_t* immediate =
          &plan->immediates[immediate_index];
      loom_json_object_writer_t immediate_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &immediate_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &immediate_object, IREE_SV("index"),
          immediate->descriptor_immediate_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &immediate_object, IREE_SV("name"), immediate->name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &immediate_object, IREE_SV("value"), immediate->value));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&immediate_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&immediates));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&packet_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&packets));
  return loom_json_object_end(&object);
}
