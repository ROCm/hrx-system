// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer table: maps device/host pointers to hrx_buffer_t handles.
// Unified implementation used by both libhrx and the HIP/CUDA bindings.

#include "buffer_table.h"

#include <stdlib.h>
#include <string.h>

#define HRX_BUFFER_TABLE_INITIAL_CAPACITY 256

void hrx_buffer_table_initialize(hrx_buffer_table_t* table) {
  memset(table, 0, sizeof(*table));
  iree_slim_mutex_initialize(&table->mutex);
}

void hrx_buffer_table_deinitialize(hrx_buffer_table_t* table) {
  IREE_ASSERT(table->reserved_insert_count == 0);
  iree_slim_mutex_deinitialize(&table->mutex);
  free(table->entries);
  memset(table, 0, sizeof(*table));
}

static size_t hrx_buffer_table_find_index(hrx_buffer_table_t* table,
                                          uint64_t any_ptr) {
  for (size_t i = 0; i < table->count; ++i) {
    hrx_buffer_table_entry_t* e = &table->entries[i];
    if (any_ptr >= e->device_ptr &&
        any_ptr - e->device_ptr < (uint64_t)e->size) {
      return i;
    }
    if (e->host_ptr) {
      uint64_t host_addr = (uint64_t)(uintptr_t)e->host_ptr;
      if (any_ptr >= host_addr && any_ptr - host_addr < (uint64_t)e->size) {
        return i;
      }
    }
  }
  return table->count;
}

static hrx_status_t hrx_buffer_table_grow(hrx_buffer_table_t* table) {
  size_t new_cap = HRX_BUFFER_TABLE_INITIAL_CAPACITY;
  if (table->capacity) {
    if (table->capacity > SIZE_MAX / 2) {
      return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                             "buffer table capacity overflow");
    }
    new_cap = table->capacity * 2;
  }
  if (new_cap > SIZE_MAX / sizeof(hrx_buffer_table_entry_t)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table allocation size overflow");
  }
  hrx_buffer_table_entry_t* new_entries =
      realloc(table->entries, new_cap * sizeof(hrx_buffer_table_entry_t));
  if (!new_entries) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table grow failed");
  }
  table->entries = new_entries;
  table->capacity = new_cap;
  return hrx_ok_status();
}

static hrx_status_t hrx_buffer_table_ensure_insert_capacity_locked(
    hrx_buffer_table_t* table, size_t additional_reservations) {
  if (table->count > SIZE_MAX - table->reserved_insert_count ||
      table->count + table->reserved_insert_count >
          SIZE_MAX - additional_reservations) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table entry count overflow");
  }
  const size_t required_capacity =
      table->count + table->reserved_insert_count + additional_reservations;
  while (table->capacity < required_capacity) {
    hrx_status_t status = hrx_buffer_table_grow(table);
    if (!hrx_status_is_ok(status)) return status;
  }
  return hrx_ok_status();
}

static hrx_status_t hrx_buffer_table_validate_insert_locked(
    hrx_buffer_table_t* table, uint64_t device_ptr, void* host_ptr) {
  if (hrx_buffer_table_find_index(table, device_ptr) < table->count) {
    return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                           "device pointer already registered");
  }

  if (host_ptr && (uint64_t)(uintptr_t)host_ptr != device_ptr) {
    uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
    if (hrx_buffer_table_find_index(table, host_addr) < table->count) {
      return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                             "host pointer already registered");
    }
  }
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_table_insert(hrx_buffer_table_t* table,
                                     uint64_t device_ptr, void* host_ptr,
                                     size_t size, hrx_buffer_t buffer,
                                     void* user_data) {
  iree_slim_mutex_lock(&table->mutex);

  hrx_status_t status =
      hrx_buffer_table_validate_insert_locked(table, device_ptr, host_ptr);
  if (hrx_status_is_ok(status)) {
    status = hrx_buffer_table_ensure_insert_capacity_locked(
        table, /*additional_reservations=*/1);
  }
  if (!hrx_status_is_ok(status)) {
    iree_slim_mutex_unlock(&table->mutex);
    return status;
  }

  table->entries[table->count++] = (hrx_buffer_table_entry_t){
      .device_ptr = device_ptr,
      .host_ptr = host_ptr,
      .size = size,
      .buffer = buffer,
      .user_data = user_data,
  };

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_table_insert_if_new(hrx_buffer_table_t* table,
                                            uint64_t device_ptr, void* host_ptr,
                                            size_t size, hrx_buffer_t buffer,
                                            void* user_data) {
  hrx_status_t status = hrx_buffer_table_insert(table, device_ptr, host_ptr,
                                                size, buffer, user_data);
  if (!hrx_status_is_ok(status) &&
      hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
    // Already registered — that is the intended idempotent outcome here.
    hrx_status_ignore(status);
    return hrx_ok_status();
  }
  return status;
}

hrx_status_t hrx_buffer_table_reserve_insert(hrx_buffer_table_t* table) {
  iree_slim_mutex_lock(&table->mutex);
  hrx_status_t status = hrx_buffer_table_ensure_insert_capacity_locked(
      table, /*additional_reservations=*/1);
  if (hrx_status_is_ok(status)) {
    ++table->reserved_insert_count;
  }
  iree_slim_mutex_unlock(&table->mutex);
  return status;
}

hrx_status_t hrx_buffer_table_insert_reserved(hrx_buffer_table_t* table,
                                              uint64_t device_ptr,
                                              void* host_ptr, size_t size,
                                              hrx_buffer_t buffer,
                                              void* user_data) {
  iree_slim_mutex_lock(&table->mutex);
  IREE_ASSERT(table->reserved_insert_count > 0);
  --table->reserved_insert_count;

  hrx_status_t status =
      hrx_buffer_table_validate_insert_locked(table, device_ptr, host_ptr);
  if (hrx_status_is_ok(status)) {
    IREE_ASSERT(table->count < table->capacity);
    table->entries[table->count++] = (hrx_buffer_table_entry_t){
        .device_ptr = device_ptr,
        .host_ptr = host_ptr,
        .size = size,
        .buffer = buffer,
        .user_data = user_data,
    };
  }

  iree_slim_mutex_unlock(&table->mutex);
  return status;
}

void hrx_buffer_table_cancel_reserved_insert(hrx_buffer_table_t* table) {
  iree_slim_mutex_lock(&table->mutex);
  IREE_ASSERT(table->reserved_insert_count > 0);
  --table->reserved_insert_count;
  iree_slim_mutex_unlock(&table->mutex);
}

hrx_status_t hrx_buffer_table_remove(hrx_buffer_table_t* table,
                                     uint64_t any_ptr) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = hrx_buffer_table_find_index(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "pointer not found in buffer table");
  }

  if (idx < table->count - 1) {
    memmove(&table->entries[idx], &table->entries[idx + 1],
            (table->count - idx - 1) * sizeof(hrx_buffer_table_entry_t));
  }
  table->count--;

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_ok_status();
}

static void hrx_buffer_table_fill_result(hrx_buffer_table_entry_t* e,
                                         uint64_t any_ptr,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data) {
  if (out_buffer) *out_buffer = e->buffer;
  if (out_offset) {
    if (any_ptr >= e->device_ptr &&
        any_ptr - e->device_ptr < (uint64_t)e->size) {
      *out_offset = (size_t)(any_ptr - e->device_ptr);
    } else {
      uint64_t host_addr = (uint64_t)(uintptr_t)e->host_ptr;
      *out_offset = (size_t)(any_ptr - host_addr);
    }
  }
  if (out_user_data) *out_user_data = e->user_data;
}

hrx_status_t hrx_buffer_table_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                                   hrx_buffer_t* out_buffer, size_t* out_offset,
                                   void** out_user_data) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = hrx_buffer_table_find_index(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    if (out_buffer) *out_buffer = NULL;
    if (out_offset) *out_offset = 0;
    if (out_user_data) *out_user_data = NULL;
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "pointer not found in buffer table");
  }

  hrx_buffer_table_fill_result(&table->entries[idx], any_ptr, out_buffer,
                               out_offset, out_user_data);
  iree_slim_mutex_unlock(&table->mutex);
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_table_find_range(hrx_buffer_table_t* table,
                                         uint64_t any_ptr, size_t size,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data) {
  if (out_buffer) *out_buffer = NULL;
  if (out_offset) *out_offset = 0;
  if (out_user_data) *out_user_data = NULL;

  if (size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "range size must be > 0");
  }
  if ((uint64_t)size > UINT64_MAX - any_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "range would overflow");
  }
  iree_slim_mutex_lock(&table->mutex);

  for (size_t i = 0; i < table->count; ++i) {
    hrx_buffer_table_entry_t* e = &table->entries[i];
    if (any_ptr >= e->device_ptr) {
      const uint64_t offset = any_ptr - e->device_ptr;
      if (offset <= (uint64_t)e->size &&
          (uint64_t)size <= (uint64_t)e->size - offset) {
        hrx_buffer_table_fill_result(e, any_ptr, out_buffer, out_offset,
                                     out_user_data);
        iree_slim_mutex_unlock(&table->mutex);
        return hrx_ok_status();
      }
    }
    if (e->host_ptr) {
      const uint64_t host_start = (uint64_t)(uintptr_t)e->host_ptr;
      if (any_ptr < host_start) continue;
      const uint64_t offset = any_ptr - host_start;
      if (offset <= (uint64_t)e->size &&
          (uint64_t)size <= (uint64_t)e->size - offset) {
        hrx_buffer_table_fill_result(e, any_ptr, out_buffer, out_offset,
                                     out_user_data);
        iree_slim_mutex_unlock(&table->mutex);
        return hrx_ok_status();
      }
    }
  }

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_make_status(HRX_STATUS_NOT_FOUND,
                         "no buffer contains the requested range");
}
