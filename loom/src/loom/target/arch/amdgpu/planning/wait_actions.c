// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_actions.h"

#include <string.h>

#include "iree/base/bitfield.h"
#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"

// Target payload size for lazily appended action segments. Segments stay small
// enough to share normal compiler workspace blocks while avoiding repeated
// copies as the final action count becomes known.
#define LOOM_AMDGPU_WAIT_ACTION_SEGMENT_BYTE_LENGTH (4u * 1024u)

// Number of action rows stored in each append segment.
#define LOOM_AMDGPU_WAIT_ACTIONS_PER_SEGMENT     \
  (LOOM_AMDGPU_WAIT_ACTION_SEGMENT_BYTE_LENGTH / \
   sizeof(loom_amdgpu_wait_plan_action_t))

static_assert(LOOM_AMDGPU_WAIT_ACTIONS_PER_SEGMENT > 0,
              "wait action must fit in one append segment");

// One stable segment populated before exact finalization.
typedef struct loom_amdgpu_wait_action_segment_t {
  // Wait actions in append order.
  loom_amdgpu_wait_plan_action_t actions[LOOM_AMDGPU_WAIT_ACTIONS_PER_SEGMENT];
} loom_amdgpu_wait_action_segment_t;

static_assert(sizeof(loom_amdgpu_wait_action_segment_t) <=
                  LOOM_AMDGPU_WAIT_ACTION_SEGMENT_BYTE_LENGTH,
              "wait action segment exceeds its byte budget");

// Common-table replay state scoped only to immutable publication.
typedef struct loom_amdgpu_wait_action_publication_t {
  // Schedule table whose packet order indexes the published records.
  const loom_low_schedule_table_t* schedule;
  // Immutable target classification indexed by schedule node.
  const loom_amdgpu_wait_classification_t* classification;
  // Node-indexed bitset of authored waits removed by simulation.
  const uint64_t* elided_wait_nodes;
  // Contiguous finalized action sequence and its cardinalities.
  const loom_amdgpu_wait_actions_t* actions;
  // Cursor into packet-ordered actions while projecting residual hazards.
  iree_host_size_t hazard_action_cursor;
  // Packet index of the next residual hazard, or IREE_HOST_SIZE_MAX.
  iree_host_size_t next_hazard_packet_index;
} loom_amdgpu_wait_action_publication_t;

static bool loom_amdgpu_wait_action_is_residual_hazard(
    const loom_amdgpu_wait_plan_action_t* action) {
  return action->kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
         !iree_any_bit_set(action->flags,
                           LOOM_AMDGPU_WAIT_PLAN_ACTION_FLAG_STORAGE_RELEASE);
}

void loom_amdgpu_wait_actions_initialize(loom_amdgpu_wait_actions_t* actions) {
  *actions = (loom_amdgpu_wait_actions_t){0};
  loom_segmented_storage_initialize(
      sizeof(loom_amdgpu_wait_action_segment_t),
      iree_alignof(loom_amdgpu_wait_action_segment_t), &actions->segments);
}

iree_status_t loom_amdgpu_wait_actions_append(
    loom_amdgpu_wait_actions_t* actions,
    const loom_amdgpu_wait_plan_action_t* action,
    iree_arena_allocator_t* transient_arena) {
  if (actions->tail_count == LOOM_AMDGPU_WAIT_ACTIONS_PER_SEGMENT) {
    actions->tail = NULL;
    actions->tail_count = 0;
  }
  if (actions->tail == NULL) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &actions->segments, transient_arena, &actions->tail));
  }
  loom_amdgpu_wait_action_segment_t* tail =
      (loom_amdgpu_wait_action_segment_t*)actions->tail;
  tail->actions[actions->tail_count++] = *action;
  ++actions->action_count;
  if (loom_amdgpu_wait_action_is_residual_hazard(action)) {
    ++actions->hazard_event_count;
  }
  return iree_ok_status();
}

// Finalization has one hot caller. Preserve its caller specialization while
// retaining an external definition for ordinary non-LTO builds.
IREE_ATTRIBUTE_ALWAYS_INLINE extern inline iree_status_t
loom_amdgpu_wait_actions_finalize(loom_amdgpu_wait_actions_t* actions,
                                  iree_arena_allocator_t* arena) {
  if (actions->action_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_wait_plan_action_t* retained_actions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, actions->action_count,
                                                 sizeof(*retained_actions),
                                                 (void**)&retained_actions));
  iree_host_size_t output_count = 0;
  for (uint32_t segment_index = 0;
       segment_index < actions->segments.segment_count; ++segment_index) {
    const loom_amdgpu_wait_action_segment_t* segment =
        (const loom_amdgpu_wait_action_segment_t*)
            loom_segmented_storage_const_segment(&actions->segments,
                                                 segment_index);
    const iree_host_size_t segment_action_count =
        iree_min(actions->action_count - output_count,
                 (iree_host_size_t)LOOM_AMDGPU_WAIT_ACTIONS_PER_SEGMENT);
    memcpy(&retained_actions[output_count], segment->actions,
           segment_action_count * sizeof(*retained_actions));
    output_count += segment_action_count;
  }
  IREE_ASSERT_EQ(output_count, actions->action_count);
  actions->actions = retained_actions;
  return iree_ok_status();
}

static void loom_amdgpu_wait_actions_emit_counter_progress(
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
    uint16_t counter_id, loom_low_packet_progress_action_t action,
    uint32_t units) {
  const loom_low_packet_progress_event_t event = {
      .progress_class_id = counter_id,
      .action = action,
      .units = units,
  };
  emit(emit_user_data, &event);
}

static void loom_amdgpu_wait_actions_emit_counter_progress_mask(
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
    uint32_t counter_mask, loom_low_packet_progress_action_t action,
    uint32_t units) {
  while (counter_mask != 0) {
    const uint32_t slot =
        (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    loom_amdgpu_wait_actions_emit_counter_progress(emit, emit_user_data,
                                                   counter_id, action, units);
    counter_mask &= counter_mask - 1;
  }
}

static bool loom_amdgpu_wait_actions_node_is_elided(
    const loom_amdgpu_wait_action_publication_t* publication,
    uint32_t node_index) {
  return publication->elided_wait_nodes != NULL &&
         (publication->elided_wait_nodes[node_index / 64] &
          (UINT64_C(1) << (node_index % 64))) != 0;
}

static void loom_amdgpu_wait_actions_progress_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  const loom_amdgpu_wait_action_publication_t* publication =
      (const loom_amdgpu_wait_action_publication_t*)user_data;
  const loom_amdgpu_wait_node_state_t* node_state =
      &publication->classification->node_states[packet->node_index];
  if (loom_amdgpu_wait_actions_node_is_elided(publication,
                                              packet->node_index)) {
    return;
  }
  loom_amdgpu_wait_actions_emit_counter_progress_mask(
      emit, emit_user_data,
      node_state->explicit_wait_counter_mask |
          node_state->implicit_wait_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_RESET, 0);
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT)) {
    const loom_amdgpu_wait_packet_bounds_t* wait_bounds =
        &publication->classification
             ->wait_bounds[node_state->state.wait_bounds_index];
    for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
         ++slot) {
      const uint16_t bound = wait_bounds->target_counts[slot];
      if (bound == 0 || bound == UINT16_MAX) {
        continue;
      }
      loom_amdgpu_wait_actions_emit_counter_progress(
          emit, emit_user_data, loom_amdgpu_wait_counter_id_from_slot(slot),
          LOOM_LOW_PACKET_PROGRESS_ACTION_BOUND, bound);
    }
  }
  const uint32_t producer_counter_mask =
      publication->classification->completion_nodes[packet->node_index]
          .producer_counter_mask;
  loom_amdgpu_wait_actions_emit_counter_progress_mask(
      emit, emit_user_data, producer_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 1);
}

static void loom_amdgpu_wait_actions_emit_hazard(
    const loom_amdgpu_wait_plan_action_t* action,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  const uint32_t observed_progress = action->target_count;
  uint32_t required_progress = action->outstanding_before;
  if (required_progress <= observed_progress) {
    // Some wait-counter predicates are epoch or control-flow hazards whose
    // counted outstanding packets are block-local. Record the action as one
    // unsatisfied target progress unit instead of losing the residual hazard.
    required_progress = observed_progress + 1;
  }
  const loom_low_packet_hazard_plan_event_t event = {
      .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
      .action_id = LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET,
      .action_name = loom_amdgpu_wait_plan_residual_action_name(
          LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET),
      .reason_id = (uint16_t)action->reason,
      .reason_name = loom_amdgpu_wait_plan_reason_name(action->reason),
      .producer_node_index = action->producer_node,
      .progress_class_id = action->counter_id,
      .progress_class_name =
          loom_amdgpu_wait_counter_progress_class_name(action->counter_id),
      .required_progress = required_progress,
      .observed_progress = observed_progress,
      .residual_progress = required_progress - observed_progress,
  };
  emit(emit_user_data, &event);
}

static void loom_amdgpu_wait_actions_advance_hazard(
    loom_amdgpu_wait_action_publication_t* publication) {
  const loom_amdgpu_wait_actions_t* actions = publication->actions;
  while (publication->hazard_action_cursor < actions->action_count) {
    const loom_amdgpu_wait_plan_action_t* action =
        &actions->actions[publication->hazard_action_cursor];
    if (loom_amdgpu_wait_action_is_residual_hazard(action)) {
      const loom_low_schedule_block_t* block =
          &publication->schedule->blocks[action->block_index];
      publication->next_hazard_packet_index =
          (iree_host_size_t)block->scheduled_node_start +
          action->scheduled_ordinal;
      return;
    }
    ++publication->hazard_action_cursor;
  }
  publication->next_hazard_packet_index = IREE_HOST_SIZE_MAX;
}

static void loom_amdgpu_wait_actions_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  loom_amdgpu_wait_action_publication_t* publication =
      (loom_amdgpu_wait_action_publication_t*)user_data;
  if (publication->next_hazard_packet_index != packet->packet_index) {
    return;
  }
  do {
    const loom_amdgpu_wait_plan_action_t* action =
        &publication->actions->actions[publication->hazard_action_cursor];
    loom_amdgpu_wait_actions_emit_hazard(action, emit, emit_user_data);
    ++publication->hazard_action_cursor;
    loom_amdgpu_wait_actions_advance_hazard(publication);
  } while (publication->next_hazard_packet_index == packet->packet_index);
}

// Publication likewise has one hot caller and exposes its short-lived replay
// context to optimized builds when inlined.
IREE_ATTRIBUTE_ALWAYS_INLINE extern inline iree_status_t
loom_amdgpu_wait_actions_build_common_tables(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_classification_t* classification,
    const uint64_t* elided_wait_nodes, iree_host_size_t progress_event_count,
    const loom_amdgpu_wait_actions_t* actions, iree_arena_allocator_t* arena,
    loom_low_packet_progress_table_t* out_progress,
    loom_low_packet_hazard_plan_t* out_hazard_plan) {
  *out_progress = (loom_low_packet_progress_table_t){0};
  *out_hazard_plan = (loom_low_packet_hazard_plan_t){0};
  loom_amdgpu_wait_action_publication_t publication = {
      .schedule = schedule,
      .classification = classification,
      .elided_wait_nodes = elided_wait_nodes,
      .actions = actions,
  };

  const loom_low_packet_progress_table_t* progress = NULL;
  if (allocation != NULL) {
    const loom_low_packet_progress_provider_t progress_provider = {
        .user_data = &publication,
        .event_count = progress_event_count,
        .query = loom_amdgpu_wait_actions_progress_query,
        .class_name = loom_amdgpu_wait_counter_progress_class_name,
    };
    IREE_RETURN_IF_ERROR(loom_low_packet_progress_build(
        schedule, allocation, &progress_provider, arena, out_progress));
    progress = out_progress;
  }

  loom_amdgpu_wait_actions_advance_hazard(&publication);
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      .user_data = &publication,
      .event_count = actions->hazard_event_count,
      .query = loom_amdgpu_wait_actions_hazard_query,
  };
  return loom_low_packet_hazard_plan_build(
      schedule, allocation, progress, &hazard_provider, arena, out_hazard_plan);
}
