// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_progress.h"

#include <inttypes.h>
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
  // Number of records counted or populated so far.
  iree_host_size_t record_count;
} loom_low_packet_progress_build_state_t;

static bool loom_low_packet_progress_action_is_valid(
    loom_low_packet_progress_action_t action) {
  return action == LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE ||
         action == LOOM_LOW_PACKET_PROGRESS_ACTION_RESET;
}

static iree_status_t loom_low_packet_progress_validate_event(
    const loom_low_packet_progress_event_t* event) {
  if (event->progress_class_id == LOOM_LOW_PACKET_PROGRESS_CLASS_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet progress event has no progress class");
  }
  if (iree_string_view_is_empty(event->progress_class_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packet progress event has no stable progress class name");
  }
  if (!loom_low_packet_progress_action_is_valid(event->action)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet progress event has invalid action %u",
                            (unsigned)event->action);
  }
  if (event->action == LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE &&
      event->units == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packet progress advance event must have non-zero units");
  }
  if (event->action == LOOM_LOW_PACKET_PROGRESS_ACTION_RESET &&
      event->units != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet progress reset event cannot advance units");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_progress_count_event(
    void* user_data, const loom_low_packet_progress_event_t* event) {
  loom_low_packet_progress_build_state_t* state =
      (loom_low_packet_progress_build_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_validate_event(event));
  iree_host_size_t next_record_count = 0;
  if (!iree_host_size_checked_add(state->record_count, 1, &next_record_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "packet progress record count exceeds host size");
  }
  state->record_count = next_record_count;
  return iree_ok_status();
}

static iree_status_t loom_low_packet_progress_append_event(
    void* user_data, const loom_low_packet_progress_event_t* event) {
  loom_low_packet_progress_build_state_t* state =
      (loom_low_packet_progress_build_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_validate_event(event));
  if (state->record_count >= state->record_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "packet progress provider emitted inconsistent record count");
  }
  const loom_low_packet_view_t* packet = state->current_packet;
  state->records[state->record_count++] = (loom_low_packet_progress_record_t){
      .packet_index = packet->packet_index,
      .node_index = packet->node_index,
      .block_index = packet->node->block_index,
      .scheduled_ordinal = packet->node->scheduled_ordinal,
      .progress_class_id = event->progress_class_id,
      .progress_class_name = event->progress_class_name,
      .action = event->action,
      .units = event->units,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_packet_progress_run_pass(
    loom_low_packet_progress_build_state_t* state,
    loom_low_packet_progress_emit_fn_t emit) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(state->schedule); ++packet_index) {
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        state->schedule, state->allocation, packet_index, &packet));
    state->current_packet = &packet;
    IREE_RETURN_IF_ERROR(
        state->provider->query(state->provider->user_data, state->schedule,
                               state->allocation, &packet, emit, state));
    state->current_packet = NULL;
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_progress_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_provider_t* provider,
    iree_arena_allocator_t* arena,
    loom_low_packet_progress_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));

  loom_low_packet_progress_build_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .provider = provider,
  };
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_run_pass(
      &state, loom_low_packet_progress_count_event));
  const iree_host_size_t record_capacity = state.record_count;

  loom_low_packet_progress_record_t* records = NULL;
  if (record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  state.record_capacity = record_capacity;
  state.record_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_run_pass(
      &state, loom_low_packet_progress_append_event));
  if (state.record_count != record_capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "packet progress provider emitted %" PRIhsz
                            " record(s) after counting %" PRIhsz,
                            state.record_count, record_capacity);
  }

  *out_table = (loom_low_packet_progress_table_t){
      .schedule = schedule,
      .allocation = allocation,
      .records = records,
      .record_count = record_capacity,
  };
  return iree_ok_status();
}
