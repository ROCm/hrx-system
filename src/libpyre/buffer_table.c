// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer table: maps device/host pointers to pyre_buffer_t handles.
// Unified implementation used by both libpyre and the HIP/CUDA bindings.

#include "buffer_table.h"

#include <stdlib.h>
#include <string.h>

#define PYRE_BUFFER_TABLE_INITIAL_CAPACITY 256

void pyre_buffer_table_initialize(pyre_buffer_table_t* table) {
  memset(table, 0, sizeof(*table));
  iree_slim_mutex_initialize(&table->mutex);
}

void pyre_buffer_table_deinitialize(pyre_buffer_table_t* table) {
  iree_slim_mutex_deinitialize(&table->mutex);
  free(table->entries);
  memset(table, 0, sizeof(*table));
}

static size_t pyre_buffer_table_find_index(pyre_buffer_table_t* table,
                                           uint64_t any_ptr) {
  for (size_t i = 0; i < table->count; ++i) {
    pyre_buffer_table_entry_t* e = &table->entries[i];
    if (any_ptr >= e->device_ptr && any_ptr < e->device_ptr + e->size) {
      return i;
    }
    if (e->host_ptr) {
      uint64_t host_addr = (uint64_t)(uintptr_t)e->host_ptr;
      if (any_ptr >= host_addr && any_ptr < host_addr + e->size) {
        return i;
      }
    }
  }
  return table->count;
}

static pyre_status_t pyre_buffer_table_grow(pyre_buffer_table_t* table) {
  size_t new_cap = table->capacity ? table->capacity * 2
                                   : PYRE_BUFFER_TABLE_INITIAL_CAPACITY;
  pyre_buffer_table_entry_t* new_entries =
      realloc(table->entries, new_cap * sizeof(pyre_buffer_table_entry_t));
  if (!new_entries) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "buffer table grow failed");
  }
  table->entries = new_entries;
  table->capacity = new_cap;
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_table_insert(pyre_buffer_table_t* table,
                                       uint64_t device_ptr, void* host_ptr,
                                       size_t size, pyre_buffer_t buffer,
                                       void* user_data) {
  iree_slim_mutex_lock(&table->mutex);

  if (pyre_buffer_table_find_index(table, device_ptr) < table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    return pyre_make_status(PYRE_STATUS_ALREADY_EXISTS,
                            "device pointer already registered");
  }

  if (host_ptr && (uint64_t)(uintptr_t)host_ptr != device_ptr) {
    uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
    if (pyre_buffer_table_find_index(table, host_addr) < table->count) {
      iree_slim_mutex_unlock(&table->mutex);
      return pyre_make_status(PYRE_STATUS_ALREADY_EXISTS,
                              "host pointer already registered");
    }
  }

  if (table->count >= table->capacity) {
    pyre_status_t status = pyre_buffer_table_grow(table);
    if (!pyre_status_is_ok(status)) {
      iree_slim_mutex_unlock(&table->mutex);
      return status;
    }
  }

  table->entries[table->count++] = (pyre_buffer_table_entry_t){
      .device_ptr = device_ptr,
      .host_ptr = host_ptr,
      .size = size,
      .buffer = buffer,
      .user_data = user_data,
  };

  iree_slim_mutex_unlock(&table->mutex);
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_table_remove(pyre_buffer_table_t* table,
                                       uint64_t any_ptr) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = pyre_buffer_table_find_index(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    return pyre_make_status(PYRE_STATUS_NOT_FOUND,
                            "pointer not found in buffer table");
  }

  if (idx < table->count - 1) {
    memmove(&table->entries[idx], &table->entries[idx + 1],
            (table->count - idx - 1) * sizeof(pyre_buffer_table_entry_t));
  }
  table->count--;

  iree_slim_mutex_unlock(&table->mutex);
  return pyre_ok_status();
}

static void pyre_buffer_table_fill_result(pyre_buffer_table_entry_t* e,
                                          uint64_t any_ptr,
                                          pyre_buffer_t* out_buffer,
                                          size_t* out_offset,
                                          void** out_user_data) {
  if (out_buffer) *out_buffer = e->buffer;
  if (out_offset) {
    if (any_ptr >= e->device_ptr && any_ptr < e->device_ptr + e->size) {
      *out_offset = (size_t)(any_ptr - e->device_ptr);
    } else {
      uint64_t host_addr = (uint64_t)(uintptr_t)e->host_ptr;
      *out_offset = (size_t)(any_ptr - host_addr);
    }
  }
  if (out_user_data) *out_user_data = e->user_data;
}

pyre_status_t pyre_buffer_table_find(pyre_buffer_table_t* table,
                                     uint64_t any_ptr,
                                     pyre_buffer_t* out_buffer,
                                     size_t* out_offset,
                                     void** out_user_data) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = pyre_buffer_table_find_index(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    if (out_buffer) *out_buffer = NULL;
    if (out_offset) *out_offset = 0;
    if (out_user_data) *out_user_data = NULL;
    return pyre_make_status(PYRE_STATUS_NOT_FOUND,
                            "pointer not found in buffer table");
  }

  pyre_buffer_table_fill_result(&table->entries[idx], any_ptr, out_buffer,
                                out_offset, out_user_data);
  iree_slim_mutex_unlock(&table->mutex);
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_table_find_range(pyre_buffer_table_t* table,
                                           uint64_t any_ptr, size_t size,
                                           pyre_buffer_t* out_buffer,
                                           size_t* out_offset,
                                           void** out_user_data) {
  if (out_buffer) *out_buffer = NULL;
  if (out_offset) *out_offset = 0;
  if (out_user_data) *out_user_data = NULL;

  if (size == 0) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "range size must be > 0");
  }
  uint64_t range_end = any_ptr + size;
  if (range_end < any_ptr) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "range would overflow");
  }

  iree_slim_mutex_lock(&table->mutex);

  for (size_t i = 0; i < table->count; ++i) {
    pyre_buffer_table_entry_t* e = &table->entries[i];
    uint64_t buf_end = e->device_ptr + e->size;
    if (any_ptr >= e->device_ptr && range_end <= buf_end) {
      pyre_buffer_table_fill_result(e, any_ptr, out_buffer, out_offset,
                                    out_user_data);
      iree_slim_mutex_unlock(&table->mutex);
      return pyre_ok_status();
    }
    if (e->host_ptr) {
      uint64_t host_start = (uint64_t)(uintptr_t)e->host_ptr;
      uint64_t host_end = host_start + e->size;
      if (any_ptr >= host_start && range_end <= host_end) {
        pyre_buffer_table_fill_result(e, any_ptr, out_buffer, out_offset,
                                      out_user_data);
        iree_slim_mutex_unlock(&table->mutex);
        return pyre_ok_status();
      }
    }
  }

  iree_slim_mutex_unlock(&table->mutex);
  return pyre_make_status(PYRE_STATUS_NOT_FOUND,
                          "no buffer contains the requested range");
}
