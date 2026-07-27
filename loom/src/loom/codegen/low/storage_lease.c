// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_lease.h"

#include <string.h>

typedef struct loom_low_storage_lease_build_state_t {
  // Schedule table being walked.
  const loom_low_schedule_table_t* schedule;
  // Target storage-lease provider.
  const loom_low_storage_lease_provider_t* provider;
  // Packet ordinal currently being queried.
  iree_host_size_t current_packet_index;
  // Schedule-node index currently being queried.
  uint32_t current_node_index;
  // Schedule node currently being queried.
  const loom_low_schedule_node_t* current_node;
  // Mutable output record storage during the populate pass.
  loom_low_storage_lease_record_t* records;
  // Maximum entries available in |records|.
  iree_host_size_t record_capacity;
  // Number of records counted or populated so far.
  iree_host_size_t record_count;
} loom_low_storage_lease_build_state_t;

static bool loom_low_storage_lease_kind_is_valid(
    loom_low_storage_lease_kind_t kind) {
  return kind == LOOM_LOW_STORAGE_LEASE_SOURCE_READ ||
         kind == LOOM_LOW_STORAGE_LEASE_RESULT_WRITE;
}

static bool loom_low_storage_lease_attachment_is_valid(
    loom_low_storage_lease_attachment_t attachment) {
  return attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND ||
         attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT;
}

static bool loom_low_storage_lease_release_scope_is_valid(
    loom_low_storage_lease_release_scope_t scope) {
  return scope == LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS;
}

static bool loom_low_storage_lease_flags_are_valid(
    loom_low_storage_lease_flags_t flags) {
  const loom_low_storage_lease_flags_t known_flags =
      LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE |
      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY |
      LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY |
      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE;
  if ((flags & (loom_low_storage_lease_flags_t)~known_flags) != 0) {
    return false;
  }
  const loom_low_storage_lease_flags_t contradictory_boundary_flags =
      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY |
      LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY;
  if ((flags & contradictory_boundary_flags) == contradictory_boundary_flags) {
    return false;
  }
  return true;
}

static void loom_low_storage_lease_validate_event(
    const loom_low_storage_lease_build_state_t* state,
    const loom_low_storage_lease_event_t* event) {
  IREE_ASSERT(loom_low_storage_lease_kind_is_valid(event->kind));
  IREE_ASSERT(loom_low_storage_lease_attachment_is_valid(event->attachment));
  IREE_ASSERT(
      loom_low_storage_lease_release_scope_is_valid(event->release_scope));
  IREE_ASSERT_NE(event->release_class_id,
                 LOOM_LOW_STORAGE_LEASE_RELEASE_CLASS_NONE);
  IREE_ASSERT(!iree_string_view_is_empty(event->release_class_name));
  IREE_ASSERT_NE(event->release_action_id,
                 LOOM_LOW_STORAGE_RELEASE_ACTION_NONE);
  IREE_ASSERT(!iree_string_view_is_empty(event->release_action_name));
  IREE_ASSERT_NE(event->release_reason_id,
                 LOOM_LOW_STORAGE_RELEASE_REASON_NONE);
  IREE_ASSERT(!iree_string_view_is_empty(event->release_reason_name));
  IREE_ASSERT_NE(event->unit_count, 0u);
  IREE_ASSERT(loom_low_storage_lease_flags_are_valid(event->flags));

  const loom_low_schedule_node_t* node = state->current_node;
  const uint16_t attachment_count =
      event->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND
          ? node->operand_count
          : node->result_count;
  IREE_ASSERT_LT(event->attachment_index, attachment_count);
}

static iree_status_t loom_low_storage_lease_count_event(
    void* user_data, const loom_low_storage_lease_event_t* event) {
  loom_low_storage_lease_build_state_t* state =
      (loom_low_storage_lease_build_state_t*)user_data;
  loom_low_storage_lease_validate_event(state, event);
  IREE_ASSERT_NE(state->record_count, IREE_HOST_SIZE_MAX);
  ++state->record_count;
  return iree_ok_status();
}

static iree_status_t loom_low_storage_lease_append_event(
    void* user_data, const loom_low_storage_lease_event_t* event) {
  loom_low_storage_lease_build_state_t* state =
      (loom_low_storage_lease_build_state_t*)user_data;
  loom_low_storage_lease_validate_event(state, event);
  IREE_ASSERT_LT(state->record_count, state->record_capacity);
  const loom_low_schedule_node_t* node = state->current_node;
  state->records[state->record_count++] = (loom_low_storage_lease_record_t){
      .packet_index = state->current_packet_index,
      .node_index = state->current_node_index,
      .block_index = node->block_index,
      .scheduled_ordinal = node->scheduled_ordinal,
      .kind = event->kind,
      .attachment = event->attachment,
      .attachment_index = event->attachment_index,
      .unit_offset = event->unit_offset,
      .unit_count = event->unit_count,
      .release_scope = event->release_scope,
      .release_class_id = event->release_class_id,
      .release_class_name = event->release_class_name,
      .release_action_id = event->release_action_id,
      .release_action_name = event->release_action_name,
      .release_reason_id = event->release_reason_id,
      .release_reason_name = event->release_reason_name,
      .flags = event->flags,
  };
  return iree_ok_status();
}

iree_status_t loom_low_storage_lease_query_descriptor_rows(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(emit);
  if (node->descriptor == NULL || schedule->target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (descriptor->storage_lease_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(descriptor->storage_lease_start,
                 descriptor_set->storage_lease_count);
  IREE_ASSERT_LE(
      descriptor->storage_lease_count,
      descriptor_set->storage_lease_count - descriptor->storage_lease_start);
  for (uint16_t i = 0; i < descriptor->storage_lease_count; ++i) {
    const loom_low_descriptor_storage_lease_t* row =
        &descriptor_set->storage_leases[descriptor->storage_lease_start + i];
    const loom_low_storage_lease_event_t event = {
        .kind = row->kind,
        .attachment = row->attachment,
        .attachment_index = row->attachment_index,
        .unit_offset = row->unit_offset,
        .unit_count = row->unit_count,
        .release_scope = row->release_scope,
        .release_class_id = row->release_class_id,
        .release_class_name = loom_low_descriptor_set_string(
            descriptor_set, row->release_class_name_string_offset),
        .release_action_id = row->release_action_id,
        .release_action_name = loom_low_descriptor_set_string(
            descriptor_set, row->release_action_name_string_offset),
        .release_reason_id = row->release_reason_id,
        .release_reason_name = loom_low_descriptor_set_string(
            descriptor_set, row->release_reason_name_string_offset),
        .flags = row->flags,
    };
    IREE_RETURN_IF_ERROR(emit(emit_user_data, &event));
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_release_action_index_build(
    const loom_low_storage_release_action_t* actions,
    iree_host_size_t action_count,
    loom_low_storage_release_action_index_key_t key, iree_host_size_t key_count,
    iree_arena_allocator_t* arena,
    loom_low_storage_release_action_index_t* out_index) {
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = (loom_low_storage_release_action_index_t){0};
  if (action_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(actions);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_LE(action_count, UINT32_MAX);
  IREE_ASSERT_NE(key_count, 0u);
  IREE_ASSERT(key ==
                  LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_PACKET ||
              key == LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_NODE);
  const bool use_packet_index =
      key == LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_PACKET;

  uint32_t* first_action_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, key_count, sizeof(*first_action_indices),
                                (void**)&first_action_indices));
  for (iree_host_size_t i = 0; i < key_count; ++i) {
    first_action_indices[i] = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
  }

  uint32_t* next_action_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, action_count,
                                                 sizeof(*next_action_indices),
                                                 (void**)&next_action_indices));
  for (iree_host_size_t i = 0; i < action_count; ++i) {
    next_action_indices[i] = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
  }

  for (iree_host_size_t i = action_count; i > 0; --i) {
    const uint32_t action_index = (uint32_t)(i - 1);
    const loom_low_storage_release_action_t* action = &actions[action_index];
    const iree_host_size_t key_index = use_packet_index
                                           ? action->insertion_packet_index
                                           : action->insertion_node_index;
    IREE_ASSERT_LT(key_index, key_count);
    next_action_indices[action_index] = first_action_indices[key_index];
    first_action_indices[key_index] = action_index;
  }

  *out_index = (loom_low_storage_release_action_index_t){
      .first_action_indices = first_action_indices,
      .next_action_indices = next_action_indices,
      .key_count = key_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_storage_lease_run_pass(
    loom_low_storage_lease_build_state_t* state,
    loom_low_storage_lease_emit_fn_t emit) {
  for (iree_host_size_t packet_index = 0;
       packet_index < state->schedule->scheduled_node_count; ++packet_index) {
    const uint32_t node_index =
        state->schedule->scheduled_node_indices[packet_index];
    IREE_ASSERT_LT(node_index, state->schedule->node_count);
    const loom_low_schedule_node_t* node = &state->schedule->nodes[node_index];
    state->current_packet_index = packet_index;
    state->current_node_index = node_index;
    state->current_node = node;
    IREE_RETURN_IF_ERROR(state->provider->query(
        state->provider->user_data, state->schedule, node, emit, state));
    state->current_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
    state->current_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
    state->current_node = NULL;
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_lease_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_storage_lease_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_storage_lease_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(provider->query);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_table);
  memset(out_table, 0, sizeof(*out_table));

  loom_low_storage_lease_build_state_t state = {
      .schedule = schedule,
      .provider = provider,
      .current_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE,
      .current_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_low_storage_lease_run_pass(
      &state, loom_low_storage_lease_count_event));
  const iree_host_size_t record_capacity = state.record_count;

  loom_low_storage_lease_record_t* records = NULL;
  if (record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  state.record_capacity = record_capacity;
  state.record_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_storage_lease_run_pass(
      &state, loom_low_storage_lease_append_event));
  IREE_ASSERT_EQ(state.record_count, record_capacity);

  *out_table = (loom_low_storage_lease_table_t){
      .schedule = schedule,
      .records = records,
      .record_count = record_capacity,
  };
  return iree_ok_status();
}
