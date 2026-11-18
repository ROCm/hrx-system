// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan.h"

#include <string.h>

typedef struct loom_low_packet_hazard_plan_build_state_t {
  // Schedule table being walked.
  const loom_low_schedule_table_t* schedule;
  // Optional allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Optional progress table paired with |schedule|.
  const loom_low_packet_progress_table_t* progress;
  // Target residual hazard provider.
  const loom_low_packet_hazard_plan_provider_t* provider;
  // Packet currently being queried.
  const loom_low_packet_view_t* current_packet;
  // Mutable output record storage during the populate pass.
  loom_low_packet_hazard_plan_record_t* records;
  // Maximum entries available in |records| from the provider.
  iree_host_size_t record_capacity;
  // Number of records populated so far from the provider.
  iree_host_size_t record_count;
} loom_low_packet_hazard_plan_build_state_t;

static void loom_low_packet_hazard_plan_producer_packet_index(
    const loom_low_schedule_table_t* schedule, uint32_t producer_node_index,
    iree_host_size_t* out_packet_index, uint32_t* out_scheduled_ordinal) {
  *out_packet_index = LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE;
  *out_scheduled_ordinal = LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE;
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return;
  }
  IREE_ASSERT_LT(producer_node_index, schedule->node_count);
  const loom_low_schedule_node_t* producer =
      &schedule->nodes[producer_node_index];
  const loom_low_schedule_block_t* block =
      &schedule->blocks[producer->block_index];
  *out_packet_index = (iree_host_size_t)block->scheduled_node_start +
                      producer->scheduled_ordinal;
  *out_scheduled_ordinal = producer->scheduled_ordinal;
}

static void loom_low_packet_hazard_plan_append_event(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_build_state_t* state = user_data;
  IREE_ASSERT_LT(state->record_count, state->record_capacity);
  const loom_low_packet_view_t* packet = state->current_packet;
  iree_host_size_t producer_packet_index =
      LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE;
  uint32_t producer_scheduled_ordinal =
      LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE;
  loom_low_packet_hazard_plan_producer_packet_index(
      state->schedule, event->producer_node_index, &producer_packet_index,
      &producer_scheduled_ordinal);
  state->records[state->record_count++] =
      (loom_low_packet_hazard_plan_record_t){
          .kind = event->kind,
          .action_id = event->action_id,
          .reason_id = event->reason_id,
          .action_name = event->action_name,
          .reason_name = event->reason_name,
          .producer_node_index = event->producer_node_index,
          .producer_scheduled_ordinal = producer_scheduled_ordinal,
          .producer_packet_index = producer_packet_index,
          .consumer_node_index = event->consumer_node_index,
          .block_index = packet->node->block_index,
          .insertion_packet_index = packet->packet_index,
          .scheduled_ordinal = packet->node->scheduled_ordinal,
          .progress_class_id = event->progress_class_id,
          .progress_class_name = event->progress_class_name,
          .required_progress = event->required_progress,
          .observed_progress = event->observed_progress,
          .residual_progress = event->residual_progress,
      };
}

static void loom_low_packet_hazard_plan_query_packets(
    loom_low_packet_hazard_plan_build_state_t* state) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(state->schedule); ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(state->schedule, packet_index);
    state->current_packet = &packet;
    state->provider->query(state->provider->user_data, state->schedule,
                           state->allocation, state->progress, &packet,
                           loom_low_packet_hazard_plan_append_event, state);
    state->current_packet = NULL;
  }
}

iree_status_t loom_low_packet_hazard_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_hazard_plan_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_hazard_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));

  loom_low_packet_hazard_plan_build_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress,
      .provider = provider,
      .record_capacity = provider->event_count,
  };
  if (state.record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, state.record_capacity,
                                                   sizeof(*state.records),
                                                   (void**)&state.records));
  }
  loom_low_packet_hazard_plan_query_packets(&state);
  IREE_ASSERT_EQ(state.record_count, state.record_capacity);
  *out_plan = (loom_low_packet_hazard_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress ? *progress : (loom_low_packet_progress_table_t){0},
      .records = state.records,
      .record_count = state.record_count,
  };
  return iree_ok_status();
}
