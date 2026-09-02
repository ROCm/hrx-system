// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/module_state.h"

#include <string.h>

#include "iree/base/internal/math.h"

typedef struct loom_low_lower_module_target_state_record_t {
  // Target-owned static key identifying this module-scope state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the module-state arena.
  void* data;
} loom_low_lower_module_target_state_record_t;

struct loom_low_lower_module_state_t {
  // Arena used for module-scope target state records and payloads.
  iree_arena_allocator_t* arena;
  // Module-scope target state records keyed by target-owned static storage.
  loom_low_lower_module_target_state_record_t* target_state_records;
  // Number of populated target_state_records entries.
  iree_host_size_t target_state_record_count;
  // Number of allocated target_state_records entries.
  iree_host_size_t target_state_record_capacity;
};

iree_status_t loom_low_lower_module_state_create(
    iree_arena_allocator_t* arena,
    loom_low_lower_module_state_t** out_module_state) {
  *out_module_state = NULL;
  loom_low_lower_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*module_state), (void**)&module_state));
  memset(module_state, 0, sizeof(*module_state));
  module_state->arena = arena;
  *out_module_state = module_state;
  return iree_ok_status();
}

iree_status_t loom_low_lower_module_state_get_or_allocate(
    loom_low_lower_module_state_t* module_state, const void* key,
    iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT(key != NULL);
  IREE_ASSERT_GT(data_length, 0);
  *out_data = NULL;
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  for (iree_host_size_t i = 0; i < module_state->target_state_record_count;
       ++i) {
    loom_low_lower_module_target_state_record_t* record =
        &module_state->target_state_records[i];
    if (record->key != key) continue;
    IREE_ASSERT_EQ(record->data_length, data_length);
    *out_data = record->data;
    return iree_ok_status();
  }

  if (module_state->target_state_record_count ==
      module_state->target_state_record_capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(module_state->target_state_record_count, 1,
                                    &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        module_state->arena, module_state->target_state_record_count,
        minimum_capacity, sizeof(*module_state->target_state_records),
        &module_state->target_state_record_capacity,
        (void**)&module_state->target_state_records));
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(module_state->arena, data_length, &data));
  memset(data, 0, data_length);
  const iree_host_size_t record_index =
      module_state->target_state_record_count++;
  loom_low_lower_module_target_state_record_t* record =
      &module_state->target_state_records[record_index];
  *record = (loom_low_lower_module_target_state_record_t){
      .key = key,
      .data_length = data_length,
      .data = data,
  };
  *out_data = data;
  return iree_ok_status();
}

iree_status_t loom_low_lower_module_state_allocate(
    loom_low_lower_module_state_t* module_state, iree_host_size_t byte_length,
    void** out_ptr) {
  *out_ptr = NULL;
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  if (byte_length == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate(module_state->arena, byte_length, out_ptr);
}

iree_status_t loom_low_lower_module_state_allocate_array(
    loom_low_lower_module_state_t* module_state, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (module_state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module-scope target lowering state is required");
  }
  return iree_arena_allocate_array(module_state->arena, count, element_size,
                                   out_ptr);
}
