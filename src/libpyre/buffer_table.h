// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-device buffer table: maps device/host pointers to pyre_buffer_t.
// Thread-safe via internal mutex.  Also carries an opaque user_data pointer
// per entry so that higher layers (e.g. the HIP binding) can attach their
// own wrapper object alongside each pyre_buffer_t.

#ifndef PYRE_BUFFER_TABLE_H_
#define PYRE_BUFFER_TABLE_H_

#include "pyre_runtime.h"

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pyre_buffer_table_entry_t {
  uint64_t device_ptr;
  void* host_ptr;
  size_t size;
  pyre_buffer_t buffer;
  void* user_data;
} pyre_buffer_table_entry_t;

typedef struct pyre_buffer_table_t {
  iree_slim_mutex_t mutex;
  pyre_buffer_table_entry_t* entries;
  size_t count;
  size_t capacity;
} pyre_buffer_table_t;

PYRE_API void pyre_buffer_table_initialize(pyre_buffer_table_t* table);
PYRE_API void pyre_buffer_table_deinitialize(pyre_buffer_table_t* table);

PYRE_API pyre_status_t pyre_buffer_table_insert(pyre_buffer_table_t* table,
                                                uint64_t device_ptr,
                                                void* host_ptr, size_t size,
                                                pyre_buffer_t buffer,
                                                void* user_data);

PYRE_API pyre_status_t pyre_buffer_table_remove(pyre_buffer_table_t* table,
                                                uint64_t any_ptr);

// Looks up a buffer containing |any_ptr| (device or host).
// Returns the buffer, byte offset within it, and optional user_data.
// Any out-parameter may be NULL if not needed.
PYRE_API pyre_status_t pyre_buffer_table_find(pyre_buffer_table_t* table,
                                              uint64_t any_ptr,
                                              pyre_buffer_t* out_buffer,
                                              size_t* out_offset,
                                              void** out_user_data);

// Looks up a buffer containing the entire range [any_ptr, any_ptr + size).
// Returns NOT_FOUND if no single buffer covers the full range.
PYRE_API pyre_status_t pyre_buffer_table_find_range(
    pyre_buffer_table_t* table, uint64_t any_ptr, size_t size,
    pyre_buffer_t* out_buffer, size_t* out_offset, void** out_user_data);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_BUFFER_TABLE_H_
