// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/buffer_lease.h"

#include <string.h>

typedef struct iree_net_buffer_lease_storage_t {
  // Allocator owning this storage allocation.
  iree_allocator_t host_allocator;
  // Host-visible message storage.
  uint8_t data[];
} iree_net_buffer_lease_storage_t;

static void iree_net_buffer_lease_release(void* user_data,
                                          iree_async_buffer_index_t index) {
  (void)index;
  iree_net_buffer_lease_storage_t* storage =
      (iree_net_buffer_lease_storage_t*)user_data;
  iree_allocator_free(storage->host_allocator, storage);
}

iree_status_t iree_net_buffer_lease_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_async_buffer_lease_t* out_lease) {
  IREE_ASSERT_ARGUMENT(out_lease);
  memset(out_lease, 0, sizeof(*out_lease));

  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_net_buffer_lease_storage_t), &total_size,
                         IREE_STRUCT_FIELD_FAM(capacity, uint8_t)));

  iree_net_buffer_lease_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  storage->host_allocator = host_allocator;

  out_lease->span = iree_async_span_from_ptr(storage->data, capacity);
  out_lease->release.fn = iree_net_buffer_lease_release;
  out_lease->release.user_data = storage;
  return iree_ok_status();
}
