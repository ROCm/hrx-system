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
  // Maximum entries available in |records| across all event sources.
  iree_host_size_t record_capacity;
  // Number of records populated so far across all event sources.
  iree_host_size_t record_count;
  // Storage-release actions grouped by insertion packet.
  loom_low_storage_release_action_index_t storage_release_action_index;
  // Packet progress records chained by progress class.
  loom_low_packet_progress_class_chain_index_t progress_class_chain_index;
  // Optional prefix range index for repeated long progress queries.
  loom_low_packet_progress_class_range_index_t progress_class_range_index;
  // Exact number of generic allocation storage-release records.
  iree_host_size_t storage_release_record_capacity;
  // Observed progress indexed by allocation storage-release action.
  uint32_t* storage_release_observed_progress;
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
    loom_low_packet_hazard_plan_build_state_t* state,
    const loom_low_packet_hazard_plan_event_t* event) {
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
}

static void loom_low_packet_hazard_plan_append_target_event(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_build_state_t* state =
      (loom_low_packet_hazard_plan_build_state_t*)user_data;
  loom_low_packet_hazard_plan_append_event(state, event);
}

static void loom_low_packet_hazard_plan_append_storage_release_event(
    loom_low_packet_hazard_plan_build_state_t* state,
    const loom_low_packet_hazard_plan_event_t* event) {
  loom_low_packet_hazard_plan_append_event(state, event);
}

static uint32_t loom_low_packet_hazard_plan_storage_release_observed_progress(
    const loom_low_packet_hazard_plan_build_state_t* state,
    const loom_low_storage_release_action_t* action,
    const loom_low_storage_lease_record_t* lease_record) {
  if (state->progress_class_range_index.progress != NULL) {
    return loom_low_packet_progress_class_range_index_observed_progress(
        &state->progress_class_range_index, lease_record->packet_index,
        action->insertion_packet_index, action->release_class_id);
  }
  return loom_low_packet_progress_class_chain_index_observed_progress(
      &state->progress_class_chain_index, lease_record->packet_index,
      action->insertion_packet_index, action->release_class_id);
}

static bool loom_low_packet_hazard_plan_should_build_progress_class_range_index(
    const loom_low_packet_hazard_plan_build_state_t* state) {
  const loom_low_packet_progress_class_chain_index_t* chain_index =
      &state->progress_class_chain_index;
  if (chain_index->progress == NULL ||
      chain_index->progress->record_count == 0) {
    return false;
  }

  // The prefix index adds two complete progress-record passes plus two binary
  // bounds per query. Require the chains' reachable-record upper bound to
  // exceed four complete table passes. This leaves amortization headroom even
  // when packet bounds or RESET records terminate some chain walks early.
  const uint64_t amortization_record_pass_count = 4;
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation->storage_release_action_count <=
      amortization_record_pass_count) {
    return false;
  }
  const uint64_t reachable_record_threshold =
      (uint64_t)chain_index->progress->record_count *
      amortization_record_pass_count;
  uint64_t reachable_record_count = 0;
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_packet_progress_class_chain_entry_t* class_entry =
        loom_low_packet_progress_class_chain_index_lookup(
            chain_index,
            allocation->storage_release_actions[i].release_class_id);
    if (class_entry == NULL) continue;
    if (reachable_record_count >
        UINT64_MAX - (uint64_t)class_entry->record_count) {
      return true;
    }
    reachable_record_count += class_entry->record_count;
    if (reachable_record_count > reachable_record_threshold) return true;
  }
  return false;
}

static iree_status_t loom_low_packet_hazard_plan_prepare_storage_releases(
    loom_low_packet_hazard_plan_build_state_t* state,
    iree_arena_allocator_t* transient_arena) {
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      transient_arena, allocation->storage_release_action_count,
      sizeof(*state->storage_release_observed_progress),
      (void**)&state->storage_release_observed_progress));
  const iree_host_size_t packet_count = loom_low_packet_count(state->schedule);
  IREE_RETURN_IF_ERROR(loom_low_storage_release_action_index_build(
      allocation->storage_release_actions,
      allocation->storage_release_action_count,
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_PACKET, packet_count,
      transient_arena, &state->storage_release_action_index));
  IREE_RETURN_IF_ERROR(loom_low_packet_progress_class_chain_index_build(
      state->progress, transient_arena, &state->progress_class_chain_index));
  if (loom_low_packet_hazard_plan_should_build_progress_class_range_index(
          state)) {
    IREE_RETURN_IF_ERROR(loom_low_packet_progress_class_range_index_build(
        &state->progress_class_chain_index, transient_arena,
        &state->progress_class_range_index));
  }
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[i];
    IREE_ASSERT_LT(action->lease_record_index,
                   allocation->storage_leases.record_count);
    const loom_low_storage_lease_record_t* lease_record =
        &allocation->storage_leases.records[action->lease_record_index];
    const uint32_t observed_progress =
        loom_low_packet_hazard_plan_storage_release_observed_progress(
            state, action, lease_record);
    state->storage_release_observed_progress[i] = observed_progress;
    if (observed_progress < action->required_progress) {
      ++state->storage_release_record_capacity;
    }
  }
  return iree_ok_status();
}

static void loom_low_packet_hazard_plan_emit_storage_release_actions(
    loom_low_packet_hazard_plan_build_state_t* state) {
  const loom_low_allocation_table_t* allocation = state->allocation;
  if (allocation == NULL ||
      state->storage_release_action_index.first_action_indices == NULL) {
    return;
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
        state->storage_release_observed_progress[action_index];
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
    loom_low_packet_hazard_plan_append_storage_release_event(state, &event);
  }
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
                           loom_low_packet_hazard_plan_append_target_event,
                           state);
    loom_low_packet_hazard_plan_emit_storage_release_actions(state);
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
  };
  iree_arena_allocator_t transient_arena;
  iree_arena_initialize(arena->block_pool, &transient_arena);
  iree_status_t status = loom_low_packet_hazard_plan_prepare_storage_releases(
      &state, &transient_arena);
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(provider->event_count,
                                  state.storage_release_record_capacity,
                                  &state.record_capacity)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "hazard plan record count exceeds host size");
  }

  loom_low_packet_hazard_plan_record_t* records = NULL;
  if (iree_status_is_ok(status) && state.record_capacity != 0) {
    status = iree_arena_allocate_array(arena, state.record_capacity,
                                       sizeof(*records), (void**)&records);
  }

  if (iree_status_is_ok(status)) {
    state.records = records;
    loom_low_packet_hazard_plan_query_packets(&state);
    IREE_ASSERT_EQ(state.record_count, state.record_capacity);
  }

  if (iree_status_is_ok(status)) {
    *out_plan = (loom_low_packet_hazard_plan_t){
        .schedule = schedule,
        .allocation = allocation,
        .progress = progress,
        .records = records,
        .record_count = state.record_capacity,
    };
  }
  iree_arena_deinitialize(&transient_arena);
  return status;
}
