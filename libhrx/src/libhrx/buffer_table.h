// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-device buffer table: maps device/host pointers to hrx_buffer_t.
// Thread-safe via internal mutex.  Also carries an opaque user_data pointer
// per entry so that higher layers (e.g. the HIP binding) can attach their
// own wrapper object alongside each hrx_buffer_t.

#ifndef HRX_BUFFER_TABLE_H_
#define HRX_BUFFER_TABLE_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_buffer_table_entry_t {
  uint64_t device_ptr;
  void* host_ptr;
  size_t size;
  hrx_buffer_t buffer;
  void* user_data;
} hrx_buffer_table_entry_t;

typedef struct hrx_buffer_table_t {
  iree_slim_mutex_t mutex;
  hrx_buffer_table_entry_t* entries;
  size_t count;
  size_t capacity;
  // Slots promised to callers that require allocation-free rollback.
  size_t reserved_insert_count;
} hrx_buffer_table_t;

void hrx_buffer_table_initialize(hrx_buffer_table_t* table);
void hrx_buffer_table_deinitialize(hrx_buffer_table_t* table);

hrx_status_t hrx_buffer_table_insert(hrx_buffer_table_t* table,
                                     uint64_t device_ptr, void* host_ptr,
                                     size_t size, hrx_buffer_t buffer,
                                     void* user_data);

// Like hrx_buffer_table_insert, but treats an already-registered pointer as
// success (returns OK). Genuine failures (e.g. out-of-memory growing the
// table) are propagated to the caller instead of being silently dropped.
hrx_status_t hrx_buffer_table_insert_if_new(hrx_buffer_table_t* table,
                                            uint64_t device_ptr, void* host_ptr,
                                            size_t size, hrx_buffer_t buffer,
                                            void* user_data);

// Reserves capacity for one future insertion. Regular insertions cannot consume
// the reserved slot. A successful call must be paired with either
// hrx_buffer_table_insert_reserved or hrx_buffer_table_cancel_reserved_insert.
hrx_status_t hrx_buffer_table_reserve_insert(hrx_buffer_table_t* table);

// Inserts an entry using capacity reserved by hrx_buffer_table_reserve_insert.
// This consumes one reservation even if pointer validation rejects the entry.
hrx_status_t hrx_buffer_table_insert_reserved(hrx_buffer_table_t* table,
                                              uint64_t device_ptr,
                                              void* host_ptr, size_t size,
                                              hrx_buffer_t buffer,
                                              void* user_data);

// Cancels one insertion reservation without inserting an entry.
void hrx_buffer_table_cancel_reserved_insert(hrx_buffer_table_t* table);

hrx_status_t hrx_buffer_table_remove(hrx_buffer_table_t* table,
                                     uint64_t any_ptr);

// Looks up a buffer containing |any_ptr| (device or host).
// Returns the buffer, byte offset within it, and optional user_data.
// Any out-parameter may be NULL if not needed.
hrx_status_t hrx_buffer_table_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                                   hrx_buffer_t* out_buffer, size_t* out_offset,
                                   void** out_user_data);

// Looks up a buffer containing the entire range [any_ptr, any_ptr + size).
// Returns NOT_FOUND if no single buffer covers the full range.
hrx_status_t hrx_buffer_table_find_range(hrx_buffer_table_t* table,
                                         uint64_t any_ptr, size_t size,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data);

#ifdef __cplusplus
}
#endif

#endif  // HRX_BUFFER_TABLE_H_
