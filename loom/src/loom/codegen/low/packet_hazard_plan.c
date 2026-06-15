// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan.h"

#include <inttypes.h>
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
  // Maximum entries available in |records|.
  iree_host_size_t record_capacity;
  // Number of records counted or populated so far.
  iree_host_size_t record_count;
} loom_low_packet_hazard_plan_build_state_t;

static bool loom_low_packet_hazard_plan_record_kind_is_valid(
    loom_low_packet_hazard_plan_record_kind_t kind) {
  return kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION ||
         kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA ||
         kind ==
             LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION ||
         kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION;
}

static bool loom_low_packet_hazard_plan_record_kind_has_residual_progress(
    loom_low_packet_hazard_plan_record_kind_t kind) {
  return kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION ||
         kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION;
}

static bool loom_low_packet_hazard_plan_record_kind_is_diagnostic(
    loom_low_packet_hazard_plan_record_kind_t kind) {
  return kind != LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION;
}

static iree_status_t loom_low_packet_hazard_plan_schedule_view_at(
    const loom_low_schedule_table_t* schedule, iree_host_size_t packet_index,
    loom_low_packet_view_t* out_packet) {
  memset(out_packet, 0, sizeof(*out_packet));
  const uint32_t node_index = schedule->scheduled_node_indices[packet_index];

  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  *out_packet = (loom_low_packet_view_t){
      .packet_index = packet_index,
      .node_index = node_index,
      .node = node,
      .descriptor = node->descriptor,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_view_at(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_host_size_t packet_index, loom_low_packet_view_t* out_packet) {
  if (allocation != NULL) {
    return loom_low_packet_view_at(schedule, allocation, packet_index,
                                   out_packet);
  }
  return loom_low_packet_hazard_plan_schedule_view_at(schedule, packet_index,
                                                      out_packet);
}

static iree_status_t loom_low_packet_hazard_plan_producer_packet_index(
    const loom_low_schedule_table_t* schedule, uint32_t producer_node_index,
    iree_host_size_t* out_packet_index, uint32_t* out_scheduled_ordinal) {
  *out_packet_index = LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE;
  *out_scheduled_ordinal = LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE;
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  if (producer_node_index >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hazard plan producer node %" PRIu32
                            " is out of range for %" PRIhsz " node(s)",
                            producer_node_index, schedule->node_count);
  }
  const loom_low_schedule_node_t* producer =
      &schedule->nodes[producer_node_index];
  IREE_RETURN_IF_ERROR(loom_low_packet_index_at_block_ordinal(
      schedule, producer->block_index, producer->scheduled_ordinal,
      out_packet_index));
  *out_scheduled_ordinal = producer->scheduled_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_validate_event(
    const loom_low_packet_hazard_plan_event_t* event) {
  if (!loom_low_packet_hazard_plan_record_kind_is_valid(event->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "hazard plan event has invalid kind %u",
                            (unsigned)event->kind);
  }
  if (event->reason_id == LOOM_LOW_PACKET_HAZARD_PLAN_REASON_NONE ||
      iree_string_view_is_empty(event->reason_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan event must have a target reason id and name");
  }
  const bool has_action =
      event->action_id != LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE ||
      !iree_string_view_is_empty(event->action_name);
  if (event->kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION) {
    if (event->action_id == LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE ||
        iree_string_view_is_empty(event->action_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "hazard plan action event must have a target action id and name");
    }
  } else if (has_action) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan diagnostic event cannot have a target action");
  }
  if (loom_low_packet_hazard_plan_record_kind_is_diagnostic(event->kind) &&
      iree_string_view_is_empty(event->target_detail)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan diagnostic event must have target detail");
  }
  if (event->kind == LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION &&
      !iree_string_view_is_empty(event->target_detail)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan action event cannot have target detail");
  }
  if (!loom_low_packet_hazard_plan_record_kind_has_residual_progress(
          event->kind)) {
    if (event->producer_node_index != LOOM_LOW_SCHEDULE_NODE_NONE ||
        event->progress_class_id != LOOM_LOW_PACKET_PROGRESS_CLASS_NONE ||
        !iree_string_view_is_empty(event->progress_class_name) ||
        event->required_progress != 0 || event->observed_progress != 0 ||
        event->residual_progress != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "hazard plan diagnostic event cannot have residual progress fields");
    }
    return iree_ok_status();
  }
  if (event->progress_class_id == LOOM_LOW_PACKET_PROGRESS_CLASS_NONE ||
      iree_string_view_is_empty(event->progress_class_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan residual event must have a progress class id and name");
  }
  if (event->observed_progress >= event->required_progress) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan residual event already satisfies required progress");
  }
  const uint32_t residual_progress =
      event->required_progress - event->observed_progress;
  if (event->residual_progress != residual_progress) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan residual event has inconsistent residual progress");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_count_event(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_build_state_t* state =
      (loom_low_packet_hazard_plan_build_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_event(event));
  iree_host_size_t next_record_count = 0;
  if (!iree_host_size_checked_add(state->record_count, 1, &next_record_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "hazard plan record count exceeds host size");
  }
  state->record_count = next_record_count;
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_append_event(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_build_state_t* state =
      (loom_low_packet_hazard_plan_build_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_event(event));
  if (state->record_count >= state->record_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "hazard plan provider emitted inconsistent record count");
  }
  const loom_low_packet_view_t* packet = state->current_packet;
  iree_host_size_t producer_packet_index =
      LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE;
  uint32_t producer_scheduled_ordinal =
      LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_producer_packet_index(
      state->schedule, event->producer_node_index, &producer_packet_index,
      &producer_scheduled_ordinal));
  if (loom_low_packet_hazard_plan_record_kind_has_residual_progress(
          event->kind) &&
      producer_packet_index != LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE &&
      producer_packet_index >= packet->packet_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hazard plan residual producer must precede insertion packet");
  }
  state->records[state->record_count++] =
      (loom_low_packet_hazard_plan_record_t){
          .kind = event->kind,
          .action_id = event->action_id,
          .action_name = event->action_name,
          .reason_id = event->reason_id,
          .reason_name = event->reason_name,
          .producer_node_index = event->producer_node_index,
          .producer_packet_index = producer_packet_index,
          .producer_scheduled_ordinal = producer_scheduled_ordinal,
          .consumer_node_index = packet->node_index,
          .insertion_packet_index = packet->packet_index,
          .block_index = packet->node->block_index,
          .scheduled_ordinal = packet->node->scheduled_ordinal,
          .progress_class_id = event->progress_class_id,
          .progress_class_name = event->progress_class_name,
          .required_progress = event->required_progress,
          .observed_progress = event->observed_progress,
          .residual_progress = event->residual_progress,
          .target_detail = event->target_detail,
      };
  return iree_ok_status();
}

static uint32_t loom_low_packet_hazard_plan_observed_progress(
    const loom_low_packet_progress_table_t* progress,
    iree_host_size_t start_packet_index, iree_host_size_t end_packet_index,
    uint16_t progress_class_id) {
  if (progress == NULL) {
    return 0;
  }
  uint32_t observed_progress = 0;
  for (iree_host_size_t i = 0; i < progress->record_count; ++i) {
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    if (record->packet_index <= start_packet_index ||
        record->packet_index >= end_packet_index ||
        record->progress_class_id != progress_class_id) {
      continue;
    }
    if (record->action == LOOM_LOW_PACKET_PROGRESS_ACTION_RESET) {
      observed_progress = 0;
    } else if (observed_progress <= UINT32_MAX - record->units) {
      observed_progress += record->units;
    } else {
      observed_progress = UINT32_MAX;
    }
  }
  return observed_progress;
}

static iree_status_t loom_low_packet_hazard_plan_emit_storage_release_actions(
    loom_low_packet_hazard_plan_build_state_t* state,
    loom_low_packet_hazard_plan_emit_fn_t emit) {
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  const loom_low_packet_view_t* packet = state->current_packet;
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[i];
    if (action->insertion_packet_index != packet->packet_index) {
      continue;
    }
    const loom_low_storage_lease_record_t* lease_record =
        &allocation->storage_leases.records[action->lease_record_index];
    const uint32_t observed_progress =
        loom_low_packet_hazard_plan_observed_progress(
            state->progress, lease_record->packet_index,
            action->insertion_packet_index, action->release_class_id);
    if (observed_progress >= action->required_progress) {
      continue;
    }
    const uint32_t residual_progress =
        action->required_progress - observed_progress;
    const loom_low_packet_hazard_plan_event_t event = {
        .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
        .action_id = action->release_action_id,
        .action_name = action->release_action_name,
        .reason_id = action->release_reason_id,
        .reason_name = action->release_reason_name,
        .producer_node_index = lease_record->node_index,
        .progress_class_id = action->release_class_id,
        .progress_class_name = action->release_class_name,
        .required_progress = action->required_progress,
        .observed_progress = observed_progress,
        .residual_progress = residual_progress,
    };
    IREE_RETURN_IF_ERROR(emit(state, &event));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_run_pass(
    loom_low_packet_hazard_plan_build_state_t* state,
    loom_low_packet_hazard_plan_emit_fn_t emit) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(state->schedule); ++packet_index) {
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_view_at(
        state->schedule, state->allocation, packet_index, &packet));
    state->current_packet = &packet;
    IREE_RETURN_IF_ERROR(state->provider->query(
        state->provider->user_data, state->schedule, state->allocation,
        state->progress, &packet, emit, state));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_hazard_plan_emit_storage_release_actions(state, emit));
    state->current_packet = NULL;
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_hazard_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_hazard_plan_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_hazard_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  if (allocation != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  }
  if (progress != NULL && progress->schedule != schedule) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "hazard plan progress table must use schedule");
  }
  if (progress != NULL && progress->allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "hazard plan progress table must use allocation");
  }

  loom_low_packet_hazard_plan_build_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress,
      .provider = provider,
  };
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_run_pass(
      &state, loom_low_packet_hazard_plan_count_event));
  const iree_host_size_t record_capacity = state.record_count;

  loom_low_packet_hazard_plan_record_t* records = NULL;
  if (record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  state.record_capacity = record_capacity;
  state.record_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_run_pass(
      &state, loom_low_packet_hazard_plan_append_event));
  if (state.record_count != record_capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "hazard plan provider emitted %" PRIhsz
                            " record(s) after counting %" PRIhsz,
                            state.record_count, record_capacity);
  }

  *out_plan = (loom_low_packet_hazard_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress,
      .records = records,
      .record_count = record_capacity,
  };
  return iree_ok_status();
}
