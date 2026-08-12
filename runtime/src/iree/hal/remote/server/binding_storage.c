// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/binding_storage.h"

#include <string.h>

iree_status_t iree_hal_remote_server_buffer_ref_list_storage_initialize(
    iree_host_size_t count,
    iree_hal_remote_server_buffer_ref_list_storage_t* out_storage,
    iree_allocator_t host_allocator) {
  memset(out_storage, 0, sizeof(*out_storage));
  out_storage->host_allocator = host_allocator;
  if (count == 0) return iree_ok_status();

  if (count <= IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY) {
    out_storage->values = out_storage->inline_values;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        host_allocator, count, sizeof(*out_storage->allocated_values),
        (void**)&out_storage->allocated_values));
    out_storage->values = out_storage->allocated_values;
  }
  out_storage->list.count = count;
  out_storage->list.values = out_storage->values;
  return iree_ok_status();
}

void iree_hal_remote_server_buffer_ref_list_storage_deinitialize(
    iree_hal_remote_server_buffer_ref_list_storage_t* storage) {
  iree_allocator_free(storage->host_allocator, storage->allocated_values);
  memset(storage, 0, sizeof(*storage));
}

iree_status_t iree_hal_remote_server_buffer_binding_table_storage_initialize(
    iree_host_size_t count,
    iree_hal_remote_server_buffer_binding_table_storage_t* out_storage,
    iree_allocator_t host_allocator) {
  memset(out_storage, 0, sizeof(*out_storage));
  out_storage->host_allocator = host_allocator;
  if (count == 0) return iree_ok_status();

  if (count <= IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY) {
    out_storage->bindings = out_storage->inline_bindings;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        host_allocator, count, sizeof(*out_storage->allocated_bindings),
        (void**)&out_storage->allocated_bindings));
    out_storage->bindings = out_storage->allocated_bindings;
  }
  out_storage->table.count = count;
  out_storage->table.bindings = out_storage->bindings;
  return iree_ok_status();
}

void iree_hal_remote_server_buffer_binding_table_storage_deinitialize(
    iree_hal_remote_server_buffer_binding_table_storage_t* storage) {
  iree_allocator_free(storage->host_allocator, storage->allocated_bindings);
  memset(storage, 0, sizeof(*storage));
}
