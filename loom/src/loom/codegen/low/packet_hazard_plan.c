// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan.h"

#include <inttypes.h>
#include <string.h>

typedef struct loom_low_packet_hazard_plan_build_state_t {
  // Validated packet sequence being queried.
  const loom_low_packet_sequence_t* packets;
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
  // Maximum entries available in |records| across all event sources.
  iree_host_size_t record_capacity;
  // Number of records populated so far across all event sources.
  iree_host_size_t record_count;
  // Number of target-provider records populated so far.
  iree_host_size_t target_record_count;
  // Storage-release actions grouped by insertion packet.
  loom_low_storage_release_action_index_t storage_release_action_index;
  // Packet progress records grouped by progress class.
  loom_low_packet_progress_class_index_t progress_class_index;
  // Exact number of generic allocation storage-release records.
  iree_host_size_t storage_release_record_capacity;
  // Number of generic allocation storage-release records populated so far.
  iree_host_size_t storage_release_record_count;
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

static iree_status_t loom_low_packet_hazard_plan_append_validated_event(
    loom_low_packet_hazard_plan_build_state_t* state,
    const loom_low_packet_hazard_plan_event_t* event) {
  const loom_low_packet_view_t* packet = state->current_packet;
  iree_host_size_t producer_packet_index =
      LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE;
  uint32_t producer_scheduled_ordinal =
      LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_producer_packet_index(
      state->schedule, event->producer_node_index, &producer_packet_index,
      &producer_scheduled_ordinal));
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
      };
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_append_target_event(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_build_state_t* state =
      (loom_low_packet_hazard_plan_build_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_event(event));
  if (state->target_record_count >= state->provider->event_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "hazard plan provider declared %" PRIhsz
                            " event(s) but attempted to emit more",
                            state->provider->event_count);
  }
  IREE_RETURN_IF_ERROR(
      loom_low_packet_hazard_plan_append_validated_event(state, event));
  ++state->target_record_count;
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_append_storage_release_event(
    loom_low_packet_hazard_plan_build_state_t* state,
    const loom_low_packet_hazard_plan_event_t* event) {
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_event(event));
  if (state->storage_release_record_count >=
      state->storage_release_record_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "hazard plan storage-release count changed during construction");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_packet_hazard_plan_append_validated_event(state, event));
  ++state->storage_release_record_count;
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_prepare_storage_releases(
    loom_low_packet_hazard_plan_build_state_t* state,
    iree_arena_allocator_t* arena) {
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t packet_count = loom_low_packet_count(state->schedule);
  IREE_RETURN_IF_ERROR(loom_low_storage_release_action_index_build(
      allocation->storage_release_actions,
      allocation->storage_release_action_count,
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_PACKET, packet_count,
      arena, &state->storage_release_action_index));
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_class_index_build(
      state->progress, arena, &state->progress_class_index));
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[i];
    if (action->lease_record_index >= allocation->storage_leases.record_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "storage release action references lease record %" PRIu32
          " but allocation has %" PRIhsz " lease record(s)",
          action->lease_record_index, allocation->storage_leases.record_count);
    }
    const loom_low_storage_lease_record_t* lease_record =
        &allocation->storage_leases.records[action->lease_record_index];
    const uint32_t observed_progress =
        loom_low_packet_progress_class_index_observed_progress(
            &state->progress_class_index, lease_record->packet_index,
            action->insertion_packet_index, action->release_class_id);
    if (observed_progress < action->required_progress) {
      iree_host_size_t next_record_capacity = 0;
      if (!iree_host_size_checked_add(state->storage_release_record_capacity, 1,
                                      &next_record_capacity)) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "storage-release hazard record count exceeds host size");
      }
      state->storage_release_record_capacity = next_record_capacity;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_emit_storage_release_actions(
    loom_low_packet_hazard_plan_build_state_t* state) {
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation == NULL ||
      state->storage_release_action_index.first_action_indices == NULL) {
    return iree_ok_status();
  }
  const loom_low_storage_release_action_index_t* index =
      &state->storage_release_action_index;
  const loom_low_packet_view_t* packet = state->current_packet;
  for (uint32_t action_index =
           index->first_action_indices[packet->packet_index];
       action_index != LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
       action_index = index->next_action_indices[action_index]) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[action_index];
    const loom_low_storage_lease_record_t* lease_record =
        &allocation->storage_leases.records[action->lease_record_index];
    const uint32_t observed_progress =
        loom_low_packet_progress_class_index_observed_progress(
            &state->progress_class_index, lease_record->packet_index,
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
    IREE_RETURN_IF_ERROR(
        loom_low_packet_hazard_plan_append_storage_release_event(state,
                                                                 &event));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_hazard_plan_query_packets(
    loom_low_packet_hazard_plan_build_state_t* state) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_sequence_count(state->packets);
       ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_sequence_at(state->packets, packet_index);
    state->current_packet = &packet;
    IREE_RETURN_IF_ERROR(state->provider->query(
        state->provider->user_data, state->schedule, state->allocation,
        state->progress, &packet,
        loom_low_packet_hazard_plan_append_target_event, state));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_hazard_plan_emit_storage_release_actions(state));
    state->current_packet = NULL;
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_hazard_plan_build(
    const loom_low_packet_sequence_t* packets,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_hazard_plan_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_hazard_plan_t* out_plan) {
  if (packets == NULL || packets->schedule == NULL || provider == NULL ||
      provider->query == NULL || arena == NULL || out_plan == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packets, provider, arena, and output plan are required for packet "
        "hazard planning");
  }
  memset(out_plan, 0, sizeof(*out_plan));
  const loom_low_schedule_table_t* schedule = packets->schedule;
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
      .packets = packets,
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress,
      .provider = provider,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_packet_hazard_plan_prepare_storage_releases(&state, arena));
  if (!iree_host_size_checked_add(provider->event_count,
                                  state.storage_release_record_capacity,
                                  &state.record_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "hazard plan record count exceeds host size");
  }

  loom_low_packet_hazard_plan_record_t* records = NULL;
  if (state.record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state.record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_query_packets(&state));
  if (state.target_record_count != provider->event_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "hazard plan provider declared %" PRIhsz
                            " event(s) but emitted %" PRIhsz,
                            provider->event_count, state.target_record_count);
  }
  if (state.storage_release_record_count !=
      state.storage_release_record_capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "hazard plan expected %" PRIhsz
                            " storage-release event(s) but emitted %" PRIhsz,
                            state.storage_release_record_capacity,
                            state.storage_release_record_count);
  }

  *out_plan = (loom_low_packet_hazard_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .progress = progress,
      .records = records,
      .record_count = state.record_capacity,
  };
  return iree_ok_status();
}
