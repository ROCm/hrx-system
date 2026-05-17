// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/file.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/file_registry.h"

typedef struct iree_hal_remote_client_file_t {
  iree_hal_resource_t resource;

  // Allocator used for this structure.
  iree_allocator_t host_allocator;

  // Capability lane used for queue file I/O.
  iree_hal_remote_client_file_kind_t kind;

  // Allowed HAL access bits.
  iree_hal_memory_access_t access;

  // Accessible file length in bytes.
  uint64_t length;

  // Retained external handle for HOST_ALLOCATION files.
  iree_io_file_handle_t* handle;

  // Retained common HAL file for ASYNC_FILE files.
  iree_hal_file_t* inner_file;

  // Client-local host allocation for HOST_ALLOCATION files.
  iree_byte_span_t host_allocation;

  // Server-side resource ID for REMOTE_FILE files.
  iree_hal_remote_resource_id_t remote_file_id;
} iree_hal_remote_client_file_t;

static const iree_hal_file_vtable_t iree_hal_remote_client_file_vtable;

static iree_hal_remote_client_file_t* iree_hal_remote_client_file_cast(
    iree_hal_file_t* IREE_RESTRICT base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_file_vtable);
  return (iree_hal_remote_client_file_t*)base_value;
}

static bool iree_hal_remote_client_file_handle_allows_access(
    iree_io_file_handle_t* handle, iree_hal_memory_access_t access) {
  const iree_io_file_access_t handle_access =
      iree_io_file_handle_access(handle);
  return (!iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_READ) ||
          iree_all_bits_set(handle_access, IREE_IO_FILE_ACCESS_READ)) &&
         (!iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE) ||
          iree_all_bits_set(handle_access, IREE_IO_FILE_ACCESS_WRITE));
}

static iree_status_t iree_hal_remote_client_file_create(
    iree_hal_remote_client_file_kind_t kind, iree_hal_memory_access_t access,
    uint64_t length, iree_io_file_handle_t* handle, iree_hal_file_t* inner_file,
    iree_byte_span_t host_allocation,
    iree_hal_remote_resource_id_t remote_file_id,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file));

  iree_hal_resource_initialize(&iree_hal_remote_client_file_vtable,
                               &file->resource);
  file->host_allocator = host_allocator;
  file->kind = kind;
  file->access = access;
  file->length = length;
  file->handle = handle;
  file->inner_file = inner_file;
  file->host_allocation = host_allocation;
  file->remote_file_id = remote_file_id;

  iree_io_file_handle_retain(file->handle);
  iree_hal_file_retain(file->inner_file);

  *out_file = (iree_hal_file_t*)file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_file_import_host_allocation(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  const iree_byte_span_t contents =
      iree_io_file_handle_value(handle).host_allocation;
  return iree_hal_remote_client_file_create(
      IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION, access,
      (uint64_t)contents.data_length, handle, /*inner_file=*/NULL, contents,
      /*remote_file_id=*/0, host_allocator, out_file);
}

static iree_status_t iree_hal_remote_client_file_import_file_descriptor(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  if (!proactor || !iree_io_file_handle_uses_async_io(handle)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote file descriptor import requires an async proactor file handle; "
        "open the file with IREE_IO_FILE_MODE_ASYNC or use a host allocation "
        "or server-side remote file");
  }

  iree_hal_file_t* inner_file = NULL;
  iree_status_t status = iree_hal_fd_file_from_handle(
      access, handle, proactor, host_allocator, &inner_file);
  if (iree_status_is_ok(status) && !iree_hal_file_async_handle(inner_file)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote file descriptor import requires a proactor-backed async file "
        "handle");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_file_create(
        IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE, access,
        iree_hal_file_length(inner_file), /*handle=*/NULL, inner_file,
        iree_byte_span_empty(), /*remote_file_id=*/0, host_allocator, out_file);
  }
  iree_hal_file_release(inner_file);
  (void)queue_affinity;
  return status;
}

iree_status_t iree_hal_remote_client_file_import(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_EXTERNAL_FILE_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote file import flags must be zero");
  } else if (!iree_hal_remote_client_file_handle_allows_access(handle,
                                                               access)) {
    status = iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "remote file import requested access not allowed by the file handle");
  }

  if (iree_status_is_ok(status)) {
    switch (iree_io_file_handle_type(handle)) {
      case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION:
        status = iree_hal_remote_client_file_import_host_allocation(
            access, handle, host_allocator, out_file);
        break;
      case IREE_IO_FILE_HANDLE_TYPE_FD:
        status = iree_hal_remote_client_file_import_file_descriptor(
            queue_affinity, access, handle, proactor, host_allocator, out_file);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "remote file import requires a host allocation, async file, or "
            "server-side remote file handle; unsupported file handle type %d",
            (int)iree_io_file_handle_type(handle));
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_remote_client_file_isa(iree_hal_file_t* file) {
  return iree_hal_resource_is((const iree_hal_resource_t*)file,
                              &iree_hal_remote_client_file_vtable);
}

iree_status_t iree_hal_remote_client_file_resolve(
    iree_hal_file_t* base_file, iree_hal_remote_client_file_view_t* out_view) {
  IREE_ASSERT_ARGUMENT(base_file);
  IREE_ASSERT_ARGUMENT(out_view);
  memset(out_view, 0, sizeof(*out_view));

  if (!iree_hal_remote_client_file_isa(base_file)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file was not imported by the remote client device");
  }

  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  out_view->kind = file->kind;
  out_view->access = file->access;
  out_view->length = file->length;
  out_view->host_allocation = file->host_allocation;
  out_view->remote_file_id = file->remote_file_id;
  if (file->inner_file) {
    out_view->async_file = iree_hal_file_async_handle(file->inner_file);
  }
  return iree_ok_status();
}

static void iree_hal_remote_client_file_destroy(
    iree_hal_file_t* IREE_RESTRICT base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_release(file->inner_file);
  iree_io_file_handle_release(file->handle);

  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_memory_access_t iree_hal_remote_client_file_allowed_access(
    iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->access;
}

static uint64_t iree_hal_remote_client_file_length(iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->length;
}

static iree_hal_buffer_t* iree_hal_remote_client_file_storage_buffer(
    iree_hal_file_t* base_file) {
  return NULL;
}

static iree_async_file_t* iree_hal_remote_client_file_async_handle(
    iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->inner_file ? iree_hal_file_async_handle(file->inner_file) : NULL;
}

static bool iree_hal_remote_client_file_supports_synchronous_io(
    iree_hal_file_t* base_file) {
  return false;
}

static iree_status_t iree_hal_remote_client_file_read(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote client files do not support synchronous read; use queue_read "
      "with the remote bulk transfer path");
}

static iree_status_t iree_hal_remote_client_file_write(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote client files do not support synchronous write; use queue_write "
      "with the remote bulk transfer path");
}

static const iree_hal_file_vtable_t iree_hal_remote_client_file_vtable = {
    .destroy = iree_hal_remote_client_file_destroy,
    .allowed_access = iree_hal_remote_client_file_allowed_access,
    .length = iree_hal_remote_client_file_length,
    .storage_buffer = iree_hal_remote_client_file_storage_buffer,
    .async_handle = iree_hal_remote_client_file_async_handle,
    .supports_synchronous_io =
        iree_hal_remote_client_file_supports_synchronous_io,
    .read = iree_hal_remote_client_file_read,
    .write = iree_hal_remote_client_file_write,
};
