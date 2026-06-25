// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/readback.h"

#include <string.h>

#include "iree/io/file_handle.h"

void id4_tooling_host_bytes_deinitialize(id4_tooling_host_bytes_t* bytes,
                                         iree_allocator_t host_allocator) {
  if (!bytes) return;
  iree_allocator_free(host_allocator, bytes->data);
  memset(bytes, 0, sizeof(*bytes));
}

static iree_status_t id4_tooling_readback_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_tooling_readback_validate_semaphore_list(
    iree_hal_semaphore_list_t list, iree_string_view_t role) {
  if (list.count == 0) return iree_ok_status();
  if (!list.semaphores || !list.payload_values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s semaphore list has %" PRIhsz
        " entries but is missing semaphores or payload values",
        (int)role.size, role.data, list.count);
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (!list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore list entry %" PRIhsz " is NULL",
                              (int)role.size, role.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_tooling_readback_validate_options(
    const id4_tooling_readback_buffer_binding_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback options are required");
  }
  IREE_RETURN_IF_ERROR(id4_tooling_readback_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("readback")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "readback extension structures are not supported");
  }
  if (!options->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback device is required");
  }
  if (!options->binding.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback source buffer is required");
  }
  if (options->binding.length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback byte length must be non-zero");
  }
  if (options->binding.length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "readback byte length %" PRIu64
                            " exceeds host capacity",
                            (uint64_t)options->binding.length);
  }
  if (iree_allocator_is_null(options->host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback host allocator is required");
  }
  return id4_tooling_readback_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("readback wait"));
}

static iree_hal_semaphore_list_t id4_tooling_readback_signal_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of signal edges in this stack-backed list.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = semaphore_storage,
      // Stack-backed payload value.
      .payload_values = payload_storage,
  };
  return list;
}

iree_status_t id4_tooling_readback_buffer_binding(
    const id4_tooling_readback_buffer_binding_options_t* options,
    id4_tooling_host_bytes_t* out_bytes) {
  IREE_ASSERT_ARGUMENT(out_bytes);
  memset(out_bytes, 0, sizeof(*out_bytes));
  IREE_RETURN_IF_ERROR(id4_tooling_readback_validate_options(options));

  id4_tooling_host_bytes_t bytes;
  memset(&bytes, 0, sizeof(bytes));
  bytes.length = (iree_host_size_t)options->binding.length;

  iree_io_file_handle_t* file_handle = NULL;
  iree_hal_file_t* target_file = NULL;
  iree_hal_semaphore_t* readback_semaphore = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(options->host_allocator, bytes.length,
                                  sizeof(bytes.data[0]), (void**)&bytes.data);
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(bytes.data, bytes.length),
        iree_io_file_handle_release_callback_null(), options->host_allocator,
        &file_handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        options->device, options->queue_affinity, IREE_HAL_MEMORY_ACCESS_WRITE,
        file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(options->device, options->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &readback_semaphore);
  }
  iree_hal_semaphore_t* readback_semaphore_storage = NULL;
  uint64_t readback_payload_storage = 1;
  iree_hal_semaphore_list_t readback_signal_list =
      iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    readback_signal_list = id4_tooling_readback_signal_list(
        &readback_semaphore_storage, &readback_payload_storage,
        readback_semaphore, readback_payload_storage);
    status = iree_hal_device_queue_write(
        options->device, options->queue_affinity, options->wait_semaphore_list,
        readback_signal_list, options->binding.buffer, options->binding.offset,
        target_file, /*target_offset=*/0, options->binding.length,
        IREE_HAL_WRITE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(readback_signal_list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_semaphore_release(readback_semaphore);
  iree_hal_file_release(target_file);
  iree_io_file_handle_release(file_handle);
  if (iree_status_is_ok(status)) {
    *out_bytes = bytes;
  } else {
    id4_tooling_host_bytes_deinitialize(&bytes, options->host_allocator);
  }
  return status;
}
