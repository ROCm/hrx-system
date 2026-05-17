// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_FILE_INDEX_H_
#define IREE_HAL_REMOTE_SERVER_FILE_INDEX_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_file_index_t iree_hal_remote_file_index_t;

// Creates an initially empty server-side file index.
//
// The index is an explicit allow-list from client-visible logical names to
// server-local host paths. It is intended to be populated during server
// configuration and then retained by iree_hal_remote_server_t.
IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_create(
    iree_allocator_t host_allocator,
    iree_hal_remote_file_index_t** out_file_index);

// Retains |file_index| for the caller.
IREE_API_EXPORT void iree_hal_remote_file_index_retain(
    iree_hal_remote_file_index_t* file_index);

// Releases |file_index| from the caller.
IREE_API_EXPORT void iree_hal_remote_file_index_release(
    iree_hal_remote_file_index_t* file_index);

// Allows opening the exact file |host_path| via |logical_name|.
//
// |allowed_access| must only contain IREE_HAL_MEMORY_ACCESS_READ and/or
// IREE_HAL_MEMORY_ACCESS_WRITE.
IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_file(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access);

// Allows opening files beneath directory |host_path| via |logical_prefix|.
//
// The logical prefix is normalized to end in '/' so that a prefix such as
// "models" exposes "models/foo.bin". Client paths must stay within the logical
// namespace; absolute suffixes and "."/".." path segments are rejected.
IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_directory(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_prefix,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access);

// Allows either a file or directory by stat-ing |host_path|.
//
// Files become exact logical-name entries and directories become logical-prefix
// entries. Returns UNAVAILABLE when IREE_FILE_IO_ENABLE=0.
IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_allow_path(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_string_view_t host_path, iree_hal_memory_access_t allowed_access);

// Opens |logical_name| if permitted by the index.
//
// The returned handle is opened with IREE_IO_FILE_MODE_ASYNC and must be
// released by the caller. |out_granted_access| is the access used to open the
// handle and is always a subset of the matched allow-list entry.
IREE_API_EXPORT iree_status_t iree_hal_remote_file_index_open(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t logical_name,
    iree_hal_memory_access_t requested_access, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle,
    iree_hal_memory_access_t* out_granted_access);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_FILE_INDEX_H_
