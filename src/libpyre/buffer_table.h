// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-device buffer table: maps device pointers to pyre_buffer_t.
// Thread-safe via internal mutex.

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
} pyre_buffer_table_entry_t;

typedef struct pyre_buffer_table_t {
  iree_slim_mutex_t mutex;
  pyre_buffer_table_entry_t* entries;
  size_t count;
  size_t capacity;
} pyre_buffer_table_t;

void pyre_buffer_table_initialize(pyre_buffer_table_t* table);
void pyre_buffer_table_deinitialize(pyre_buffer_table_t* table);

pyre_status_t pyre_buffer_table_insert(pyre_buffer_table_t* table,
                                       uint64_t device_ptr, void* host_ptr,
                                       size_t size, pyre_buffer_t buffer);

pyre_status_t pyre_buffer_table_remove(pyre_buffer_table_t* table,
                                       uint64_t any_ptr);

pyre_status_t pyre_buffer_table_find(pyre_buffer_table_t* table,
                                     uint64_t any_ptr,
                                     pyre_buffer_t* out_buffer,
                                     size_t* out_offset);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_BUFFER_TABLE_H_
