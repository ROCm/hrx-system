// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_transfer_scheduler.h"

#include <string.h>

struct iree_hal_remote_bulk_transfer_scheduler_t {
  // Host allocator used for scheduler allocation.
  iree_allocator_t host_allocator;

  // Fixed-capacity active transfer table.
  iree_net_bulk_transfer_table_t* table;

  // Owner callbacks for transfer lifecycle.
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks;
};

static iree_status_t iree_hal_remote_bulk_transfer_scheduler_resolve_options(
    const iree_hal_remote_bulk_transfer_scheduler_options_t* options,
    iree_hal_remote_bulk_transfer_scheduler_options_t* out_options) {
  iree_hal_remote_bulk_transfer_scheduler_options_t resolved_options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  if (options) resolved_options = *options;
  if (resolved_options.capacity == 0) {
    resolved_options.capacity = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY;
  }
  if (resolved_options.user_storage_alignment == 0) {
    resolved_options.user_storage_alignment = iree_max_align_t;
  }
  *out_options = resolved_options;
  return iree_ok_status();
}

iree_status_t iree_hal_remote_bulk_transfer_scheduler_allocate(
    const iree_hal_remote_bulk_transfer_scheduler_options_t* options,
    iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler) {
  IREE_ASSERT_ARGUMENT(out_scheduler);
  *out_scheduler = NULL;

  iree_hal_remote_bulk_transfer_scheduler_options_t resolved_options;
  iree_status_t status =
      iree_hal_remote_bulk_transfer_scheduler_resolve_options(
          options, &resolved_options);

  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*scheduler),
                                   (void**)&scheduler);
  }
  if (iree_status_is_ok(status)) {
    memset(scheduler, 0, sizeof(*scheduler));
    iree_net_bulk_transfer_table_options_t table_options =
        iree_net_bulk_transfer_table_options_default();
    table_options.capacity = resolved_options.capacity;
    table_options.user_storage_size = resolved_options.user_storage_size;
    table_options.user_storage_alignment =
        resolved_options.user_storage_alignment;
    table_options.initial_transfer_id = resolved_options.initial_transfer_id;
    table_options.transfer_id_stride = resolved_options.transfer_id_stride;
    status = iree_net_bulk_transfer_table_allocate(
        &table_options, host_allocator, &scheduler->table);
  }

  if (iree_status_is_ok(status)) {
    scheduler->host_allocator = host_allocator;
    scheduler->callbacks = callbacks;
    *out_scheduler = scheduler;
  } else if (scheduler) {
    iree_net_bulk_transfer_table_free(scheduler->table);
    iree_allocator_free(host_allocator, scheduler);
  }
  return status;
}

void iree_hal_remote_bulk_transfer_scheduler_free(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler) {
  if (!scheduler) return;
  iree_allocator_t host_allocator = scheduler->host_allocator;
  iree_hal_remote_bulk_transfer_scheduler_clear(scheduler);
  iree_net_bulk_transfer_table_free(scheduler->table);
  iree_allocator_free(host_allocator, scheduler);
}

iree_host_size_t iree_hal_remote_bulk_transfer_scheduler_capacity(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_net_bulk_transfer_table_capacity(scheduler->table);
}

iree_host_size_t iree_hal_remote_bulk_transfer_scheduler_count(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_net_bulk_transfer_table_count(scheduler->table);
}

bool iree_hal_remote_bulk_transfer_scheduler_has_capacity(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_hal_remote_bulk_transfer_scheduler_count(scheduler) <
         iree_hal_remote_bulk_transfer_scheduler_capacity(scheduler);
}

iree_status_t iree_hal_remote_bulk_transfer_scheduler_insert_peer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler, uint64_t transfer_id,
    uint64_t total_size, uint64_t user_value,
    iree_net_bulk_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_net_bulk_transfer_table_insert(
      scheduler->table, transfer_id, total_size, user_value, out_transfer);
}

iree_status_t iree_hal_remote_bulk_transfer_scheduler_allocate_local(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler, uint64_t total_size,
    uint64_t user_value, iree_net_bulk_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_net_bulk_transfer_table_allocate_transfer(
      scheduler->table, total_size, user_value, out_transfer);
}

iree_net_bulk_transfer_t* iree_hal_remote_bulk_transfer_scheduler_lookup(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    uint64_t transfer_id) {
  IREE_ASSERT_ARGUMENT(scheduler);
  return iree_net_bulk_transfer_table_lookup(scheduler->table, transfer_id);
}

bool iree_hal_remote_bulk_transfer_scheduler_release(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer) {
  if (!scheduler || !transfer) return false;
  const uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(scheduler->table, transfer_id);
  if (table_transfer != transfer) return false;
  if (scheduler->callbacks.deinitialize) {
    scheduler->callbacks.deinitialize(scheduler->callbacks.user_data, transfer);
  }
  return iree_net_bulk_transfer_table_remove(scheduler->table, transfer_id);
}

typedef struct iree_hal_remote_bulk_transfer_scheduler_first_id_t {
  // First visited transfer ID.
  uint64_t transfer_id;

  // True when |transfer_id| contains a valid transfer ID.
  bool has_transfer_id;
} iree_hal_remote_bulk_transfer_scheduler_first_id_t;

static void iree_hal_remote_bulk_transfer_scheduler_select_first(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_scheduler_first_id_t* first_id =
      (iree_hal_remote_bulk_transfer_scheduler_first_id_t*)user_data;
  if (!first_id->has_transfer_id) {
    first_id->transfer_id = iree_net_bulk_transfer_id(transfer);
    first_id->has_transfer_id = true;
  }
}

void iree_hal_remote_bulk_transfer_scheduler_clear(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler) {
  if (!scheduler) return;
  while (iree_hal_remote_bulk_transfer_scheduler_count(scheduler) != 0) {
    iree_hal_remote_bulk_transfer_scheduler_first_id_t first_id = {
        .transfer_id = 0,
        .has_transfer_id = false,
    };
    iree_net_bulk_transfer_table_visit(
        scheduler->table, iree_hal_remote_bulk_transfer_scheduler_select_first,
        &first_id);
    if (!first_id.has_transfer_id) break;
    iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
        scheduler->table, first_id.transfer_id);
    if (!iree_hal_remote_bulk_transfer_scheduler_release(scheduler, transfer)) {
      break;
    }
  }
}

void iree_hal_remote_bulk_transfer_scheduler_visit(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_hal_remote_bulk_transfer_visit_fn_t visitor, void* user_data) {
  IREE_ASSERT_ARGUMENT(scheduler);
  iree_net_bulk_transfer_table_visit(scheduler->table, visitor, user_data);
}

typedef struct iree_hal_remote_bulk_transfer_scheduler_collect_state_t {
  // Selection predicate.
  iree_hal_remote_bulk_transfer_select_fn_t select;

  // User data passed to |select|.
  void* select_user_data;

  // Transfer IDs selected so far.
  uint64_t* transfer_ids;

  // Maximum entries in |transfer_ids|.
  iree_host_size_t capacity;

  // Number of transfer IDs selected.
  iree_host_size_t count;

  // True if at least one selected transfer could not be recorded.
  bool overflowed;
} iree_hal_remote_bulk_transfer_scheduler_collect_state_t;

static void iree_hal_remote_bulk_transfer_scheduler_collect_selected(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_scheduler_collect_state_t* state =
      (iree_hal_remote_bulk_transfer_scheduler_collect_state_t*)user_data;
  if (state->select && !state->select(state->select_user_data, transfer)) {
    return;
  }
  if (state->count >= state->capacity) {
    state->overflowed = true;
    return;
  }
  state->transfer_ids[state->count++] = iree_net_bulk_transfer_id(transfer);
}

bool iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_hal_remote_bulk_transfer_select_fn_t select, void* select_user_data,
    uint64_t* out_transfer_ids, iree_host_size_t capacity,
    iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(scheduler);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  if (capacity > 0) {
    IREE_ASSERT_ARGUMENT(out_transfer_ids);
  }

  iree_hal_remote_bulk_transfer_scheduler_collect_state_t state = {
      .select = select,
      .select_user_data = select_user_data,
      .transfer_ids = out_transfer_ids,
      .capacity = capacity,
      .count = 0,
      .overflowed = false,
  };
  iree_net_bulk_transfer_table_visit(
      scheduler->table,
      iree_hal_remote_bulk_transfer_scheduler_collect_selected, &state);
  *out_count = state.count;
  return !state.overflowed;
}
