// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_packets.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE UINT16_MAX
#define LOOM_AMDGPU_WAIT_PACKET_COUNTER_COUNT 5
#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_CAPACITY 16
#define LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_IMMEDIATE_CAPACITY 4

typedef struct loom_amdgpu_wait_packet_descriptor_immediate_t {
  // Descriptor-local immediate index populated by this row.
  uint16_t descriptor_immediate_index;
  // Borrowed immediate field name from the selected descriptor set.
  iree_string_view_t name;
  // Logical counters controlled by this immediate field.
  uint32_t counter_mask;
  // Immediate value that leaves this field unconstrained.
  uint16_t no_wait_value;
} loom_amdgpu_wait_packet_descriptor_immediate_t;

typedef struct loom_amdgpu_wait_packet_descriptor_t {
  // Borrowed descriptor row.
  const loom_low_descriptor_t* descriptor;
  // Borrowed descriptor key string.
  iree_string_view_t key;
  // Logical counters this descriptor can drain.
  uint32_t counter_mask;
  // Per-immediate logical counter mappings.
  loom_amdgpu_wait_packet_descriptor_immediate_t
      immediates[LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_IMMEDIATE_CAPACITY];
  // Number of populated immediate mappings.
  uint16_t immediate_count;
} loom_amdgpu_wait_packet_descriptor_t;

typedef struct loom_amdgpu_wait_packet_target_t {
  // Concrete wait descriptors available on this target.
  loom_amdgpu_wait_packet_descriptor_t
      descriptors[LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_CAPACITY];
  // Number of populated descriptor rows.
  iree_host_size_t descriptor_count;
  // Maximum immediate row count for any available wait descriptor.
  iree_host_size_t max_descriptor_immediate_count;
} loom_amdgpu_wait_packet_target_t;

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
  uint16_t target_counts[LOOM_AMDGPU_WAIT_PACKET_COUNTER_COUNT];
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

static uint16_t loom_amdgpu_wait_packet_counter_id_from_slot(uint32_t slot) {
  return (uint16_t)(slot + 1);
}

static iree_status_t loom_amdgpu_wait_packet_counter_slot(uint16_t counter_id,
                                                          uint32_t* out_slot) {
  if (counter_id < LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD ||
      counter_id > LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU wait counter id %" PRIu16,
                            counter_id);
  }
  *out_slot = (uint32_t)(counter_id - 1);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_target_count(
    const loom_amdgpu_wait_packet_group_t* group, uint32_t counter_mask,
    uint16_t* out_target_count) {
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
  if (target_count == LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet group has no target count for counter mask 0x%x",
        counter_mask);
  }
  *out_target_count = target_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_verify_target(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait packet materialization requires a descriptor set");
  }
  if (descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    iree_string_view_t target_key = loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->target_key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU wait packet materialization received target '%.*s'",
        (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_descriptor_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t* out_counter_mask) {
  *out_counter_mask = 0;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor effect range is out of range");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind != LOOM_LOW_EFFECT_KIND_COUNTER) {
      continue;
    }
    if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
      continue;
    }
    uint32_t counter_mask = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_counter_mask(effect->counter_id, &counter_mask));
    *out_counter_mask |= counter_mask;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_immediate_counter_mask(
    const loom_low_immediate_t* immediate, uint32_t* out_counter_mask) {
  switch (immediate->encoding_id) {
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_VMEM:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_LGKM:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                          LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_VMEM_LOAD:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_VMEM_STORE:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_LDS:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_SMEM:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM;
      return iree_ok_status();
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_WAIT_COUNTER_ALU:
      *out_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
      return iree_ok_status();
    default:
      *out_counter_mask = 0;
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unsupported AMDGPU wait immediate encoding id %" PRIu16,
          immediate->encoding_id);
  }
}

static iree_status_t loom_amdgpu_wait_packet_append_target_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_low_descriptor_t* descriptor, uint32_t counter_mask) {
  if (builder->target.descriptor_count >=
      LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU wait descriptor table capacity exceeded");
  }
  if (descriptor->immediate_count >
      LOOM_AMDGPU_WAIT_PACKET_DESCRIPTOR_IMMEDIATE_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU wait descriptor has too many immediates");
  }
  iree_string_view_t key = loom_low_descriptor_set_string(
      builder->descriptor_set, descriptor->key_string_offset);

  loom_amdgpu_wait_packet_descriptor_t* target_descriptor =
      &builder->target.descriptors[builder->target.descriptor_count++];
  *target_descriptor = (loom_amdgpu_wait_packet_descriptor_t){
      .descriptor = descriptor,
      .key = key,
      .counter_mask = counter_mask,
      .immediate_count = descriptor->immediate_count,
  };
  uint32_t mapped_counter_mask = 0;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_row = descriptor->immediate_start + i;
    if (immediate_row >= builder->descriptor_set->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait descriptor '%.*s' immediate row %" PRIu32
          " is out of range",
          (int)key.size, key.data, immediate_row);
    }
    const loom_low_immediate_t* immediate =
        &builder->descriptor_set->immediates[immediate_row];
    if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU wait descriptor '%.*s' immediate %" PRIu16
                              " must be unsigned",
                              (int)key.size, key.data, i);
    }
    iree_string_view_t name = loom_low_descriptor_set_string(
        builder->descriptor_set, immediate->field_name_string_offset);
    uint32_t immediate_counter_mask = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_immediate_counter_mask(
        immediate, &immediate_counter_mask));
    immediate_counter_mask &= counter_mask;
    if (immediate_counter_mask == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU wait descriptor '%.*s' immediate '%.*s' does not map to any "
          "descriptor counter effect",
          (int)key.size, key.data, (int)name.size, name.data);
    }
    if (immediate->unsigned_max > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait descriptor '%.*s' immediate '%.*s' "
                              "no-wait value %" PRIu64
                              " exceeds packet planner storage",
                              (int)key.size, key.data, (int)name.size,
                              name.data, immediate->unsigned_max);
    }
    target_descriptor->immediates[i] =
        (loom_amdgpu_wait_packet_descriptor_immediate_t){
            .descriptor_immediate_index = i,
            .name = name,
            .counter_mask = immediate_counter_mask,
            .no_wait_value = (uint16_t)immediate->unsigned_max,
        };
    mapped_counter_mask |= immediate_counter_mask;
  }
  if (mapped_counter_mask != counter_mask) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait descriptor '%.*s' maps counter mask "
                            "0x%x but advertises 0x%x",
                            (int)key.size, key.data, mapped_counter_mask,
                            counter_mask);
  }
  if (descriptor->immediate_count >
      builder->target.max_descriptor_immediate_count) {
    builder->target.max_descriptor_immediate_count =
        descriptor->immediate_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_classify_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  uint32_t counter_mask = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_descriptor_counter_mask(
      builder->descriptor_set, descriptor, &counter_mask));
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_packet_append_target_descriptor(builder, descriptor,
                                                          counter_mask);
}

static iree_status_t loom_amdgpu_wait_packet_analyze_target(
    loom_amdgpu_wait_packet_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_verify_target(builder->descriptor_set));
  builder->target = (loom_amdgpu_wait_packet_target_t){0};
  for (uint32_t i = 0; i < builder->descriptor_set->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_classify_descriptor(
        builder, &builder->descriptor_set->descriptors[i]));
  }
  return iree_ok_status();
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
  if (!iree_host_size_checked_mul(planned_action_count, max_immediate_count,
                                  &builder->immediate_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait packet immediate capacity overflows");
  }
  if (builder->immediate_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->immediate_capacity,
        sizeof(*builder->immediates), (void**)&builder->immediates));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_append_immediate(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    const loom_amdgpu_wait_packet_descriptor_immediate_t* immediate_descriptor,
    uint16_t value) {
  if (builder->immediate_count >= builder->immediate_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet plan exceeded precomputed immediate capacity");
  }
  const uint16_t descriptor_immediate_index =
      immediate_descriptor->descriptor_immediate_index;
  if (value > immediate_descriptor->no_wait_value) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor '%.*s' immediate %" PRIu16 " value %" PRIu16
        " exceeds maximum %" PRIu16,
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        descriptor_immediate_index, value, immediate_descriptor->no_wait_value);
  }

  builder->immediates[builder->immediate_count++] =
      (loom_amdgpu_wait_packet_immediate_t){
          .descriptor_immediate_index = descriptor_immediate_index,
          .name = immediate_descriptor->name,
          .value = value,
      };
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_packet_popcount(uint32_t value) {
  uint32_t count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

static iree_status_t loom_amdgpu_wait_packet_append_packet(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    uint32_t counter_mask) {
  if (builder->packet_count >= builder->packet_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet plan exceeded precomputed packet capacity");
  }

  const iree_host_size_t immediate_start = builder->immediate_count;
  for (uint16_t i = 0; i < packet_descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_t* immediate =
        &packet_descriptor->immediates[i];
    uint16_t value = immediate->no_wait_value;
    const uint32_t immediate_counter_mask =
        immediate->counter_mask & counter_mask;
    if (immediate_counter_mask != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_target_count(
          group, immediate_counter_mask, &value));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_immediate(
        builder, packet_descriptor, immediate, value));
  }
  builder->packets[builder->packet_count++] = (loom_amdgpu_wait_packet_t){
      .descriptor = packet_descriptor->descriptor,
      .block_index = group->block_index,
      .node_index = group->node_index,
      .scheduled_ordinal = group->scheduled_ordinal,
      .counter_mask = counter_mask,
      .source_action_start = group->source_action_start,
      .source_action_count = group->source_action_count,
      .immediate_start = immediate_start,
      .immediate_count = builder->immediate_count - immediate_start,
  };
  return iree_ok_status();
}

static const loom_amdgpu_wait_packet_descriptor_t*
loom_amdgpu_wait_packet_select_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder, uint32_t remaining_counter_mask,
    uint32_t* out_covered_counter_mask) {
  const loom_amdgpu_wait_packet_descriptor_t* best_descriptor = NULL;
  uint32_t best_covered_counter_mask = 0;
  uint32_t best_covered_count = 0;
  uint32_t best_extra_count = UINT32_MAX;
  uint16_t best_immediate_count = UINT16_MAX;
  for (iree_host_size_t i = 0; i < builder->target.descriptor_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_t* descriptor =
        &builder->target.descriptors[i];
    const uint32_t covered_counter_mask =
        descriptor->counter_mask & remaining_counter_mask;
    if (covered_counter_mask == 0) {
      continue;
    }
    const uint32_t covered_count =
        loom_amdgpu_wait_packet_popcount(covered_counter_mask);
    const uint32_t extra_count = loom_amdgpu_wait_packet_popcount(
        descriptor->counter_mask & ~remaining_counter_mask);
    if (best_descriptor == NULL || covered_count > best_covered_count ||
        (covered_count == best_covered_count &&
         extra_count < best_extra_count) ||
        (covered_count == best_covered_count &&
         extra_count == best_extra_count &&
         descriptor->immediate_count < best_immediate_count)) {
      best_descriptor = descriptor;
      best_covered_counter_mask = covered_counter_mask;
      best_covered_count = covered_count;
      best_extra_count = extra_count;
      best_immediate_count = descriptor->immediate_count;
    }
  }
  *out_covered_counter_mask = best_covered_counter_mask;
  return best_descriptor;
}

iree_status_t loom_amdgpu_wait_packet_try_select_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    uint16_t target_count, loom_amdgpu_wait_packet_selection_t* out_selection,
    bool* out_selected) {
  *out_selection = (loom_amdgpu_wait_packet_selection_t){0};
  *out_selected = false;
  if (counter_mask == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait packet selection requires a counter");
  }

  loom_amdgpu_wait_packet_builder_t builder = {
      .descriptor_set = descriptor_set,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_analyze_target(&builder));

  uint32_t covered_counter_mask = 0;
  const loom_amdgpu_wait_packet_descriptor_t* descriptor =
      loom_amdgpu_wait_packet_select_descriptor(&builder, counter_mask,
                                                &covered_counter_mask);
  if (descriptor == NULL || covered_counter_mask != counter_mask) {
    return iree_ok_status();
  }
  if (descriptor->immediate_count >
      LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU wait packet selection immediate capacity exceeded");
  }

  *out_selection = (loom_amdgpu_wait_packet_selection_t){
      .descriptor = descriptor->descriptor,
      .counter_mask = covered_counter_mask,
      .immediate_count = descriptor->immediate_count,
  };
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_amdgpu_wait_packet_descriptor_immediate_t* immediate =
        &descriptor->immediates[i];
    uint16_t value = immediate->no_wait_value;
    if (iree_any_bit_set(immediate->counter_mask, covered_counter_mask)) {
      value = target_count;
    }
    if (value > immediate->no_wait_value) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait descriptor '%.*s' immediate %" PRIu16
                              " value %" PRIu16 " exceeds maximum %" PRIu16,
                              (int)descriptor->key.size, descriptor->key.data,
                              immediate->descriptor_immediate_index, value,
                              immediate->no_wait_value);
    }
    out_selection->immediates[i] = (loom_amdgpu_wait_packet_immediate_t){
        .descriptor_immediate_index = immediate->descriptor_immediate_index,
        .name = immediate->name,
        .value = value,
    };
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_select_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    uint16_t target_count, loom_amdgpu_wait_packet_selection_t* out_selection) {
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_try_select_counter_mask(
      descriptor_set, counter_mask, target_count, out_selection, &selected));
  if (!selected) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU target cannot materialize wait packet for counter mask 0x%x",
        counter_mask);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_materialize_group(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group) {
  const uint32_t unknown_mask =
      group->counter_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  if (unknown_mask != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait packet group has unknown counter "
                            "mask 0x%x",
                            unknown_mask);
  }
  uint32_t remaining_counter_mask = group->counter_mask;
  while (remaining_counter_mask != 0) {
    uint32_t covered_counter_mask = 0;
    const loom_amdgpu_wait_packet_descriptor_t* descriptor =
        loom_amdgpu_wait_packet_select_descriptor(
            builder, remaining_counter_mask, &covered_counter_mask);
    if (descriptor == NULL) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU target cannot materialize wait packet for counter mask 0x%x",
          remaining_counter_mask);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_packet(
        builder, group, descriptor, covered_counter_mask));
    remaining_counter_mask &= ~covered_counter_mask;
  }
  return iree_ok_status();
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

static iree_status_t loom_amdgpu_wait_packet_group_accumulate(
    loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_plan_action_t* action) {
  uint32_t counter_mask = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_counter_mask(action->counter_id, &counter_mask));
  uint32_t slot = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_counter_slot(action->counter_id, &slot));
  group->counter_mask |= counter_mask;
  if (action->target_count < group->target_counts[slot]) {
    group->target_counts[slot] = action->target_count;
  }
  ++group->source_action_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_build_packets(
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
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_packet_group_accumulate(&group, group_action));
      }
      ++next_action_index;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_packet_materialize_group(builder, &group));
    i = next_action_index;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_plan_build(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_packet_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_packet_plan_t){0};
  if (wait_plan == NULL || wait_plan->schedule == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "wait plan, schedule, and arena are required for AMDGPU wait packet "
        "materialization");
  }
  loom_amdgpu_wait_packet_builder_t builder = {
      .wait_plan = wait_plan,
      .descriptor_set = wait_plan->schedule->target.descriptor_set,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_analyze_target(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_build_packets(&builder));
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool needs_comma = false;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_PACKET_COUNTER_COUNT;
       ++slot) {
    const uint32_t slot_mask = 1u << slot;
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    const uint16_t counter_id =
        loom_amdgpu_wait_packet_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_amdgpu_wait_counter_name(counter_id)));
    needs_comma = true;
  }
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t loom_amdgpu_wait_packet_plan_format_json(
    const loom_amdgpu_wait_packet_plan_t* plan,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_low_schedule_table_t* schedule = plan->wait_plan->schedule;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.wait_packet_plan.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_amdgpu_wait_packet_json_function_name(plan)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"packet_count\":%zu,\"packets\":[", plan->packet_count));
  for (iree_host_size_t i = 0; i < plan->packet_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_amdgpu_wait_packet_t* packet = &plan->packets[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, "\"descriptor\":"));
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_descriptor_ordinal(
            schedule->target.descriptor_set, packet->descriptor);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait packet descriptor row does not belong to the selected "
          "descriptor set");
    }
    iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
        schedule->target.descriptor_set, packet->descriptor->key_string_offset);
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, descriptor_key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"block\":%" PRIu32 ",\"node\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"counter_mask\":%" PRIu32
        ",\"counters\":",
        packet->block_index, packet->node_index, packet->scheduled_ordinal,
        packet->counter_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_json_write_counters(
        &stream, packet->counter_mask));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"source_action_start\":%zu,\"source_action_count\":%zu"
        ",\"immediates\":[",
        packet->source_action_start, packet->source_action_count));
    for (iree_host_size_t j = 0; j < packet->immediate_count; ++j) {
      if (j > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const iree_host_size_t immediate_index = packet->immediate_start + j;
      if (immediate_index >= plan->immediate_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait packet immediate index %zu is out of range",
            immediate_index);
      }
      const loom_amdgpu_wait_packet_immediate_t* immediate =
          &plan->immediates[immediate_index];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "{\"index\":%" PRIu16 ",\"name\":",
          immediate->descriptor_immediate_index));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(&stream, immediate->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"value\":%" PRIu16 "}", immediate->value));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
