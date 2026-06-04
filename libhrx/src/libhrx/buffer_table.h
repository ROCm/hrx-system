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
} hrx_buffer_table_t;

HRX_API void hrx_buffer_table_initialize(hrx_buffer_table_t* table);
HRX_API void hrx_buffer_table_deinitialize(hrx_buffer_table_t* table);

HRX_API hrx_status_t hrx_buffer_table_insert(hrx_buffer_table_t* table,
                                             uint64_t device_ptr,
                                             void* host_ptr, size_t size,
                                             hrx_buffer_t buffer,
                                             void* user_data);

HRX_API hrx_status_t hrx_buffer_table_remove(hrx_buffer_table_t* table,
                                             uint64_t any_ptr);

// Looks up a buffer containing |any_ptr| (device or host).
// Returns the buffer, byte offset within it, and optional user_data.
// Any out-parameter may be NULL if not needed.
HRX_API hrx_status_t hrx_buffer_table_find(hrx_buffer_table_t* table,
                                           uint64_t any_ptr,
                                           hrx_buffer_t* out_buffer,
                                           size_t* out_offset,
                                           void** out_user_data);

// Looks up a buffer containing the entire range [any_ptr, any_ptr + size).
// Returns NOT_FOUND if no single buffer covers the full range.
HRX_API hrx_status_t hrx_buffer_table_find_range(hrx_buffer_table_t* table,
                                                 uint64_t any_ptr, size_t size,
                                                 hrx_buffer_t* out_buffer,
                                                 size_t* out_offset,
                                                 void** out_user_data);

#ifdef __cplusplus
}
#endif

#endif  // HRX_BUFFER_TABLE_H_
