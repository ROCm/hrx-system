// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_progress.h"

#include <string.h>

typedef struct loom_low_packet_progress_build_state_t {
  // Schedule table being walked.
  const loom_low_schedule_table_t* schedule;
  // Allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Target progress provider.
  const loom_low_packet_progress_provider_t* provider;
  // Packet currently being queried.
  const loom_low_packet_view_t* current_packet;
  // Mutable output record storage during the populate pass.
  loom_low_packet_progress_record_t* records;
  // Maximum entries available in |records|.
  iree_host_size_t record_capacity;
  // Number of records populated so far.
  iree_host_size_t record_count;
} loom_low_packet_progress_build_state_t;

static void loom_low_packet_progress_append_event(
    void* user_data, const loom_low_packet_progress_event_t* event) {
  loom_low_packet_progress_build_state_t* state =
      (loom_low_packet_progress_build_state_t*)user_data;
  IREE_ASSERT_LT(state->record_count, state->record_capacity);
  const loom_low_packet_view_t* packet = state->current_packet;
  state->records[state->record_count++] = (loom_low_packet_progress_record_t){
      .packet_index = packet->packet_index,
      .progress_class_id = event->progress_class_id,
      .action = event->action,
      .units = event->units,
  };
}

static void loom_low_packet_progress_query_packets(
    loom_low_packet_progress_build_state_t* state) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(state->schedule); ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(state->schedule, packet_index);
    state->current_packet = &packet;
    state->provider->query(state->provider->user_data, state->schedule,
                           state->allocation, &packet,
                           loom_low_packet_progress_append_event, state);
    state->current_packet = NULL;
  }
}

iree_status_t loom_low_packet_progress_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_provider_t* provider,
    iree_arena_allocator_t* arena,
    loom_low_packet_progress_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));

  loom_low_packet_progress_build_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .provider = provider,
      .record_capacity = provider->event_count,
  };

  loom_low_packet_progress_record_t* records = NULL;
  if (state.record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state.record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  loom_low_packet_progress_query_packets(&state);
  IREE_ASSERT_EQ(state.record_count, state.record_capacity);

  *out_table = (loom_low_packet_progress_table_t){
      .schedule = schedule,
      .allocation = allocation,
      .records = records,
      .record_count = state.record_capacity,
      .class_name = provider->class_name,
  };
  return iree_ok_status();
}
