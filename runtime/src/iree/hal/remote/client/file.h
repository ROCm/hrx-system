// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Remote client file wrappers.
//
// Remote queue file I/O needs to know where a file lives instead of probing
// generic HAL file vtable capabilities. Client-local files expose transfer
// capabilities here; server-local files expose remote resource IDs resolved by
// the FILE_OPEN control path.

#ifndef IREE_HAL_REMOTE_CLIENT_FILE_H_
#define IREE_HAL_REMOTE_CLIENT_FILE_H_

#include "iree/async/file.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/common.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_file_t
//===----------------------------------------------------------------------===//

typedef enum iree_hal_remote_client_file_kind_e {
  // File contents are directly addressable in client host memory.
  IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION = 0u,

  // File contents are accessible through a proactor-managed async file handle.
  IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE = 1u,

  // File contents are local to the remote server and referenced by resource ID.
  IREE_HAL_REMOTE_CLIENT_FILE_KIND_REMOTE_FILE = 2u,
} iree_hal_remote_client_file_kind_e;
typedef uint8_t iree_hal_remote_client_file_kind_t;

// Resolved transfer capabilities for a remote client file.
typedef struct iree_hal_remote_client_file_view_t {
  // Capability lane used for queue file I/O.
  iree_hal_remote_client_file_kind_t kind;

  // Allowed HAL access bits.
  iree_hal_memory_access_t access;

  // Accessible file length in bytes.
  uint64_t length;

  // Client-local host allocation for HOST_ALLOCATION files.
  iree_byte_span_t host_allocation;

  // Proactor-managed async file handle for ASYNC_FILE files.
  iree_async_file_t* async_file;

  // Server-side resource ID for REMOTE_FILE files.
  iree_hal_remote_resource_id_t remote_file_id;
} iree_hal_remote_client_file_view_t;

// Imports |handle| as a remote-native client file.
//
// Supported client-local handles expose either a direct host allocation byte
// span or a proactor-managed async file handle. When |device| is connected over
// a transport that can transfer file handle rights, descriptor-backed files are
// registered on the server as remote files instead.
iree_status_t iree_hal_remote_client_file_import(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

// Opens a server-side file from the server's configured logical namespace and
// returns its resolved metadata.
iree_status_t iree_hal_remote_client_file_open(
    iree_hal_remote_client_device_t* device, iree_string_view_t logical_name,
    iree_hal_memory_access_t access, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

// Returns true if |file| is a remote client file wrapper.
bool iree_hal_remote_client_file_isa(iree_hal_file_t* file);

// Resolves the transfer capabilities for |file|.
iree_status_t iree_hal_remote_client_file_resolve(
    iree_hal_file_t* file, iree_hal_remote_client_file_view_t* out_view);

// Marks |file| as referenced by at least one remote queue command.
void iree_hal_remote_client_file_mark_queue_referenced(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_FILE_H_
