// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_BINDING_STORAGE_H_
#define IREE_HAL_REMOTE_SERVER_BINDING_STORAGE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Number of binding entries retained inline in temporary server storage.
#define IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY 32

// Callback-lifetime storage for a HAL buffer reference list.
//
// Lists through IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY entries use
// inline storage. Larger lists use one host allocation. The storage owns only
// the backing array and does not retain buffers referenced by its entries.
typedef struct iree_hal_remote_server_buffer_ref_list_storage_t {
  // HAL list referencing the active |values| storage.
  iree_hal_buffer_ref_list_t list;
  // Mutable backing array selected by initialization.
  iree_hal_buffer_ref_t* values;
  // Heap backing array, or NULL when the list is inline or empty.
  iree_hal_buffer_ref_t* allocated_values;
  // Host allocator used to release |allocated_values|.
  iree_allocator_t host_allocator;
  // Inline backing array used for common binding lists.
  iree_hal_buffer_ref_t
      inline_values[IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY];
} iree_hal_remote_server_buffer_ref_list_storage_t;

// Initializes |out_storage| for |count| buffer references.
//
// The returned entries are uninitialized and must be completely populated
// before exposing |out_storage->list| to the HAL. On failure, |out_storage|
// remains safe to deinitialize.
iree_status_t iree_hal_remote_server_buffer_ref_list_storage_initialize(
    iree_host_size_t count,
    iree_hal_remote_server_buffer_ref_list_storage_t* out_storage,
    iree_allocator_t host_allocator);

// Releases any allocated backing array and resets |storage| to zero.
void iree_hal_remote_server_buffer_ref_list_storage_deinitialize(
    iree_hal_remote_server_buffer_ref_list_storage_t* storage);

// Callback-lifetime storage for a HAL buffer binding table.
//
// Tables through IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY entries use
// inline storage. Larger tables use one host allocation. The storage owns only
// the backing array and does not retain buffers referenced by its entries.
typedef struct iree_hal_remote_server_buffer_binding_table_storage_t {
  // HAL table referencing the active |bindings| storage.
  iree_hal_buffer_binding_table_t table;
  // Mutable backing array selected by initialization.
  iree_hal_buffer_binding_t* bindings;
  // Heap backing array, or NULL when the table is inline or empty.
  iree_hal_buffer_binding_t* allocated_bindings;
  // Host allocator used to release |allocated_bindings|.
  iree_allocator_t host_allocator;
  // Inline backing array used for common binding tables.
  iree_hal_buffer_binding_t
      inline_bindings[IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY];
} iree_hal_remote_server_buffer_binding_table_storage_t;

// Initializes |out_storage| for |count| buffer bindings.
//
// The returned entries are uninitialized and must be completely populated
// before exposing |out_storage->table| to the HAL. On failure, |out_storage|
// remains safe to deinitialize.
iree_status_t iree_hal_remote_server_buffer_binding_table_storage_initialize(
    iree_host_size_t count,
    iree_hal_remote_server_buffer_binding_table_storage_t* out_storage,
    iree_allocator_t host_allocator);

// Releases any allocated backing array and resets |storage| to zero.
void iree_hal_remote_server_buffer_binding_table_storage_deinitialize(
    iree_hal_remote_server_buffer_binding_table_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BINDING_STORAGE_H_
