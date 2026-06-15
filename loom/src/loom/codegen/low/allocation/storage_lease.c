// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation/storage.h"

struct loom_low_allocation_storage_lease_unit_entry_t {
  // Storage-lease instance occupying this leased unit.
  uint32_t storage_lease_index;
  // Next entry in the hashed unit bucket.
  uint32_t next_entry;
  // Target-storage identity key for the register class owning this unit.
  uint32_t storage_key;
  // Target-visible storage kind for this unit.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register or target ID for this unit.
  uint32_t location;
};

static uint32_t loom_low_allocation_storage_lease_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

static uint32_t loom_low_allocation_storage_lease_unit_hash(
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  uint32_t hash = location ^ ((uint32_t)location_kind * 0x9E3779B9u);
  hash ^= storage_key * 0x7F4A7C15u;
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  hash ^= hash >> 16;
  return hash;
}

static bool loom_low_allocation_storage_lease_unit_index_is_enabled(
    const loom_low_allocation_storage_lease_unit_index_t* index) {
  return index->bucket_heads != NULL && index->bucket_count != 0 &&
         index->entries != NULL;
}

static uint32_t loom_low_allocation_storage_lease_unit_bucket_index(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  return loom_low_allocation_storage_lease_unit_hash(location_kind, storage_key,
                                                     location) &
         (index->bucket_count - 1u);
}

static iree_status_t loom_low_allocation_storage_lease_unit_index_initialize(
    loom_low_allocation_storage_lease_unit_index_t* index,
    iree_host_size_t lease_unit_capacity, iree_arena_allocator_t* arena) {
  if (lease_unit_capacity == 0) {
    return iree_ok_status();
  }
  if (lease_unit_capacity > UINT32_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease unit count exceeds "
                            "index range");
  }

  const uint32_t bucket_count =
      loom_low_allocation_storage_lease_round_up_to_power_of_two_u32(
          (uint32_t)lease_unit_capacity * 2u);
  if (bucket_count == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease bucket count exceeds "
                            "u32 range");
  }

  index->bucket_count = bucket_count;
  index->entry_capacity = lease_unit_capacity;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, bucket_count,
                                                 sizeof(*index->bucket_heads),
                                                 (void**)&index->bucket_heads));
  for (uint32_t i = 0; i < bucket_count; ++i) {
    index->bucket_heads[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, lease_unit_capacity,
                                                 sizeof(*index->entries),
                                                 (void**)&index->entries));
  memset(index->entries, 0, lease_unit_capacity * sizeof(*index->entries));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_unit_index_insert(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_storage_lease_t* instances,
    uint32_t storage_lease_index) {
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(index)) {
    return iree_ok_status();
  }
  const loom_low_allocation_storage_lease_t* lease =
      &instances[storage_lease_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          lease->location_kind)) {
    return iree_ok_status();
  }
  if (lease->location_count > index->entry_capacity - index->entry_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage lease unit index capacity was exhausted");
  }

  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, lease->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < lease->location_count;
       ++unit_offset) {
    if (lease->location_base > UINT32_MAX - unit_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation storage lease unit location exceeds "
                              "u32 range");
    }
    const uint32_t location = lease->location_base + unit_offset;
    const uint32_t bucket_index =
        loom_low_allocation_storage_lease_unit_bucket_index(
            index, lease->location_kind, storage_key, location);
    const uint32_t entry_index = (uint32_t)index->entry_count++;
    loom_low_allocation_storage_lease_unit_entry_t* entry =
        &index->entries[entry_index];
    *entry = (loom_low_allocation_storage_lease_unit_entry_t){
        .storage_lease_index = storage_lease_index,
        .next_entry = index->bucket_heads[bucket_index],
        .storage_key = storage_key,
        .location_kind = lease->location_kind,
        .location = location,
    };
    index->bucket_heads[bucket_index] = entry_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_storage_lease_table(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op) {
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (lease_table->schedule == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have records but no schedule");
  }
  if (lease_table->records == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have a record count but no records");
  }
  if (lease_table->schedule->module != module ||
      lease_table->schedule->function_op != function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases must describe the allocated low function");
  }
  if (lease_table->record_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease count exceeds u32 "
                            "range");
  }
  return iree_ok_status();
}

static bool loom_low_allocation_value_ordinal_for_liveness_value(
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    if (liveness->value_ids[i] == value_id) {
      *out_value_ordinal = (loom_value_ordinal_t)i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_storage_lease_value_id(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_low_storage_lease_record_t* record,
    loom_value_id_t* out_value_id) {
  const loom_low_schedule_table_t* schedule = lease_table->schedule;
  if (record->node_index >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease references schedule node %" PRIu32
                            " outside the schedule",
                            record->node_index);
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[record->node_index];
  if (node->block_index != record->block_index ||
      node->scheduled_ordinal != record->scheduled_ordinal) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease record no longer matches its schedule node");
  }

  const loom_value_ordinal_t* value_ordinals = NULL;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND) {
    if (record->attachment_index >= node->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease operand index exceeds schedule "
                              "node operand count");
    }
    value_ordinals = loom_low_schedule_node_const_operand_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT) {
    if (record->attachment_index >= node->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease result index exceeds schedule "
                              "node result count");
    }
    value_ordinals = loom_low_schedule_node_const_result_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "storage lease record has invalid attachment kind %u",
        (unsigned)record->attachment);
  }
  if (value_ordinal >= schedule->liveness.value_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease references a value outside schedule liveness");
  }
  *out_value_id = schedule->liveness.value_ids[value_ordinal];
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_start_point(
    const loom_liveness_analysis_t* liveness,
    const loom_low_storage_lease_record_t* record, uint32_t* out_start_point) {
  if (record->block_index >= liveness->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease block index exceeds allocation "
                            "liveness block count");
  }
  const loom_liveness_block_info_t* block_info =
      &liveness->blocks[record->block_index];
  if (record->scheduled_ordinal > UINT32_MAX - block_info->start_point) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease start point exceeds u32 range");
  }
  const uint32_t start_point =
      block_info->start_point + record->scheduled_ordinal;
  if (start_point >= block_info->end_point) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease scheduled ordinal exceeds "
                            "allocation liveness block extent");
  }
  *out_start_point = start_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_end_point(
    const loom_liveness_analysis_t* liveness,
    const loom_low_storage_lease_record_t* record, uint32_t* out_end_point) {
  if (record->block_index >= liveness->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease block index exceeds allocation "
                            "liveness block count");
  }
  if (iree_any_bit_set(record->flags,
                       LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY)) {
    *out_end_point = liveness->blocks[record->block_index].end_point;
    return iree_ok_status();
  }
  uint32_t function_end_point = 0;
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    if (function_end_point < liveness->blocks[i].end_point) {
      function_end_point = liveness->blocks[i].end_point;
    }
  }
  *out_end_point = function_end_point;
  return iree_ok_status();
}

static bool loom_low_allocation_storage_lease_instance_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  if (lease->release_action_index !=
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
    return false;
  }
  if (lease->end_point <= candidate->start_point ||
      lease->start_point >= candidate->end_point) {
    return false;
  }
  if (lease->location_kind != candidate->location_kind) {
    return false;
  }
  if (!loom_low_allocation_storage_reg_classes_share(
          descriptor_set, lease->descriptor_reg_class_id,
          candidate->descriptor_reg_class_id)) {
    return false;
  }
  const uint64_t lease_begin = lease->location_base;
  const uint64_t candidate_begin = candidate->location_base;
  const uint64_t lease_end = lease_begin + lease->location_count;
  const uint64_t candidate_end = candidate_begin + candidate->location_count;
  return lease_begin < candidate_end && candidate_begin < lease_end;
}

static bool loom_low_allocation_storage_lease_value_is_ignored(
    const loom_low_allocation_storage_lease_t* lease,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  if (ignored_value_ids == NULL) {
    return false;
  }
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (lease->value_id == ignored_value_ids[i]) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_try_packet_at_program_point(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  *out_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  *out_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  *out_block_index = UINT32_MAX;
  *out_scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;

  const loom_low_schedule_table_t* schedule = state->lease_table->schedule;
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &liveness->blocks[i];
    if (program_point < block_info->start_point ||
        program_point >= block_info->end_point) {
      continue;
    }
    const uint32_t scheduled_ordinal = program_point - block_info->start_point;
    if (scheduled_ordinal >= schedule->blocks[i].scheduled_node_count) {
      return false;
    }
    const uint64_t packet_index =
        (uint64_t)schedule->blocks[i].scheduled_node_start + scheduled_ordinal;
    if (packet_index >= schedule->scheduled_node_count ||
        packet_index > IREE_HOST_SIZE_MAX) {
      return false;
    }
    const uint32_t node_index =
        schedule->scheduled_node_indices[(iree_host_size_t)packet_index];
    if (node_index >= schedule->node_count) {
      return false;
    }
    *out_packet_index = (iree_host_size_t)packet_index;
    *out_node_index = node_index;
    *out_block_index = (uint32_t)i;
    *out_scheduled_ordinal = scheduled_ordinal;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_at_program_point(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  if (loom_low_allocation_try_packet_at_program_point(
          state, liveness, program_point, out_packet_index, out_node_index,
          out_block_index, out_scheduled_ordinal)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "storage release insertion point does not map to a "
                          "scheduled packet");
}

static bool loom_low_allocation_storage_lease_can_release_before(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_storage_lease_record_t* record =
      &state->lease_table->records[lease->lease_record_index];
  if (record->release_scope !=
      LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS) {
    return false;
  }
  if (candidate->start_point <= lease->start_point) {
    return false;
  }
  iree_host_size_t packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  uint32_t node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  uint32_t block_index = UINT32_MAX;
  uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
  if (!loom_low_allocation_try_packet_at_program_point(
          state, liveness, candidate->start_point, &packet_index, &node_index,
          &block_index, &scheduled_ordinal)) {
    return false;
  }
  return packet_index != LOOM_LOW_STORAGE_LEASE_PACKET_NONE &&
         packet_index > record->packet_index;
}

static bool loom_low_allocation_storage_lease_scan_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate) {
  const iree_host_size_t record_count = state->lease_table->record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->instance_written[i] == 0) {
      continue;
    }
    if (loom_low_allocation_storage_lease_instance_conflicts(
            descriptor_set, &state->instances[i], candidate)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_storage_lease_state_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(lease_table);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = (loom_low_allocation_storage_lease_state_t){0};
  out_state->lease_table = lease_table;
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_storage_lease_table(
      lease_table, module, function_op));
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->instances),
      (void**)&out_state->instances));
  memset(out_state->instances, 0,
         lease_table->record_count * sizeof(*out_state->instances));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    out_state->instances[i].release_action_index =
        LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->release_actions),
      (void**)&out_state->release_actions));
  memset(out_state->release_actions, 0,
         lease_table->record_count * sizeof(*out_state->release_actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->instance_written),
      (void**)&out_state->instance_written));
  memset(out_state->instance_written, 0,
         lease_table->record_count * sizeof(*out_state->instance_written));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->next_record_indices),
      (void**)&out_state->next_record_indices));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    out_state->next_record_indices[i] = UINT32_MAX;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->value_count,
      sizeof(*out_state->record_heads_by_value_ordinal),
      (void**)&out_state->record_heads_by_value_ordinal));
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    out_state->record_heads_by_value_ordinal[i] = UINT32_MAX;
  }

  iree_host_size_t lease_unit_capacity = 0;
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    const loom_low_storage_lease_record_t* record = &lease_table->records[i];
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_value_id(
        lease_table, record, &value_id));
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (!loom_low_allocation_value_ordinal_for_liveness_value(
            liveness, value_id, &value_ordinal)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "storage lease references value outside allocation liveness");
    }
    out_state->next_record_indices[i] =
        out_state->record_heads_by_value_ordinal[value_ordinal];
    out_state->record_heads_by_value_ordinal[value_ordinal] = (uint32_t)i;
    if (!iree_host_size_checked_add(lease_unit_capacity, record->unit_count,
                                    &lease_unit_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease unit count exceeds host size");
    }
  }
  return loom_low_allocation_storage_lease_unit_index_initialize(
      &out_state->units, lease_unit_capacity, arena);
}

bool loom_low_allocation_storage_lease_state_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(candidate);
  if (state->lease_table == NULL || state->lease_table->record_count == 0) {
    return false;
  }
  if (policy == LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED) {
    const iree_host_size_t record_count = state->lease_table->record_count;
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      if (state->instance_written[i] == 0) {
        continue;
      }
      const loom_low_allocation_storage_lease_t* lease = &state->instances[i];
      if (loom_low_allocation_storage_lease_value_is_ignored(
              lease, ignored_value_ids, ignored_value_count)) {
        continue;
      }
      if (!loom_low_allocation_storage_lease_instance_conflicts(
              descriptor_set, lease, candidate)) {
        continue;
      }
      if (!loom_low_allocation_storage_lease_can_release_before(
              state, liveness, lease, candidate)) {
        return true;
      }
    }
    return false;
  }
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(&state->units)) {
    return loom_low_allocation_storage_lease_scan_conflicts(
        state, descriptor_set, candidate);
  }
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, candidate->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < candidate->location_count;
       ++unit_offset) {
    if (candidate->location_base > UINT32_MAX - unit_offset) {
      break;
    }
    const uint32_t location = candidate->location_base + unit_offset;
    const uint32_t bucket_index =
        loom_low_allocation_storage_lease_unit_bucket_index(
            &state->units, candidate->location_kind, storage_key, location);
    uint32_t entry_index = state->units.bucket_heads[bucket_index];
    while (entry_index != UINT32_MAX) {
      const loom_low_allocation_storage_lease_unit_entry_t* entry =
          &state->units.entries[entry_index];
      const loom_low_allocation_storage_lease_t* lease =
          &state->instances[entry->storage_lease_index];
      if (loom_low_allocation_storage_lease_value_is_ignored(
              lease, ignored_value_ids, ignored_value_count)) {
        entry_index = entry->next_entry;
        continue;
      }
      if (entry->location_kind == candidate->location_kind &&
          entry->storage_key == storage_key && entry->location == location &&
          lease->release_action_index ==
              LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE &&
          lease->end_point > candidate->start_point &&
          lease->start_point < candidate->end_point) {
        return true;
      }
      entry_index = entry->next_entry;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_storage_lease_state_record_release_actions(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(candidate);
  if (state->instances == NULL) {
    return iree_ok_status();
  }
  const iree_host_size_t record_count = state->lease_table->record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->instance_written[i] == 0) {
      continue;
    }
    loom_low_allocation_storage_lease_t* lease = &state->instances[i];
    if (loom_low_allocation_storage_lease_value_is_ignored(
            lease, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (!loom_low_allocation_storage_lease_instance_conflicts(
            descriptor_set, lease, candidate)) {
      continue;
    }
    const loom_low_storage_lease_record_t* record =
        &state->lease_table->records[i];
    if (state->release_action_count >= state->lease_table->record_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "allocation storage release action count exceeded lease count");
    }
    if (!loom_low_allocation_storage_lease_can_release_before(
            state, liveness, lease, candidate)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "storage lease conflict has no legal release before assignment");
    }

    iree_host_size_t insertion_packet_index =
        LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
    uint32_t insertion_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
    uint32_t block_index = UINT32_MAX;
    uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_at_program_point(
        state, liveness, candidate->start_point, &insertion_packet_index,
        &insertion_node_index, &block_index, &scheduled_ordinal));

    const uint32_t release_action_index =
        (uint32_t)state->release_action_count++;
    state->release_actions[release_action_index] =
        (loom_low_storage_release_action_t){
            .insertion_packet_index = insertion_packet_index,
            .insertion_node_index = insertion_node_index,
            .block_index = block_index,
            .scheduled_ordinal = scheduled_ordinal,
            .release_class_id = record->release_class_id,
            .release_class_name = record->release_class_name,
            .release_action_id = record->release_action_id,
            .release_action_name = record->release_action_name,
            .release_reason_id = record->release_reason_id,
            .release_reason_name = record->release_reason_name,
            .required_progress = 1,
            .lease_record_index = (uint32_t)i,
        };
    lease->release_action_index = release_action_index;
    lease->end_point = candidate->start_point;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_storage_lease_state_record_assignment(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(assignment);
  if (state->record_heads_by_value_ordinal == NULL) {
    return iree_ok_status();
  }
  uint32_t lease_record_index =
      state->record_heads_by_value_ordinal[value_ordinal];
  while (lease_record_index != UINT32_MAX) {
    const loom_low_storage_lease_record_t* record =
        &state->lease_table->records[lease_record_index];
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_value_id(
        state->lease_table, record, &value_id));
    if (assignment->value_id != value_id) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease record does not match assigned "
                              "value");
    }
    if (!loom_low_allocation_location_kind_is_register_like(
            assignment->location_kind)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease references value %u assigned to "
                              "non-register storage",
                              (unsigned)value_id);
    }
    if (record->unit_offset > UINT32_MAX - record->unit_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease unit range exceeds u32 range");
    }
    const uint32_t lease_assignment_end =
        record->unit_offset + record->unit_count;
    if (lease_assignment_end > assignment->location_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease unit range exceeds assigned "
                              "physical storage");
    }
    if (assignment->location_base > UINT32_MAX - record->unit_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease physical location exceeds u32 "
                              "range");
    }
    const uint32_t location_base =
        assignment->location_base + record->unit_offset;
    if (!iree_any_bit_set(record->flags,
                          LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "storage lease record has no supported start "
                              "condition");
    }
    uint32_t start_point = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_start_point(
        liveness, record, &start_point));
    uint32_t end_point = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_end_point(
        liveness, record, &end_point));
    if (end_point <= start_point) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease end point must be after its "
                              "start point");
    }

    if (state->instance_written[lease_record_index] != 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease record was materialized twice");
    }
    state->instances[lease_record_index] =
        (loom_low_allocation_storage_lease_t){
            .lease_record_index = lease_record_index,
            .assignment_index = assignment_index,
            .value_id = value_id,
            .start_point = start_point,
            .end_point = end_point,
            .release_action_index = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE,
            .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
            .location_kind = assignment->location_kind,
            .location_base = location_base,
            .location_count = record->unit_count,
        };
    state->instance_written[lease_record_index] = 1;
    ++state->instance_count;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_unit_index_insert(
        &state->units, descriptor_set, state->instances, lease_record_index));
    lease_record_index = state->next_record_indices[lease_record_index];
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_storage_lease_state_finalize(
    const loom_low_allocation_storage_lease_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);
  if (state->lease_table == NULL || state->lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (state->instance_count != state->lease_table->record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation materialized %" PRIhsz
        " storage lease instance(s) for %" PRIhsz " storage lease record(s)",
        state->instance_count, state->lease_table->record_count);
  }
  for (iree_host_size_t i = 0; i < state->lease_table->record_count; ++i) {
    if (state->instance_written[i] == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease record %" PRIhsz
                              " has no assigned physical storage",
                              i);
    }
  }
  return iree_ok_status();
}
