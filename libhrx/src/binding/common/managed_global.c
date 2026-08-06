// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/managed_global.h"

#include <inttypes.h>
#include <string.h>

#include "common/direct_transfer.h"
#include "common/internal.h"

iree_status_t iree_hal_streaming_managed_global_storage_name(
    iree_allocator_t host_allocator, iree_string_view_t pointer_name,
    char** out_storage_name) {
  IREE_ASSERT_ARGUMENT(out_storage_name);
  *out_storage_name = NULL;
  static const iree_string_view_t suffix = IREE_SVL(".managed");
  iree_host_size_t storage_name_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          pointer_name.size, suffix.size + 1, &storage_name_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed symbol name size overflow");
  }

  char* storage_name = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, storage_name_size,
                                             (void**)&storage_name));
  memcpy(storage_name, pointer_name.data, pointer_name.size);
  memcpy(storage_name + pointer_name.size, suffix.data, suffix.size);
  storage_name[storage_name_size - 1] = '\0';
  *out_storage_name = storage_name;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_managed_global_initialize(
    iree_hal_streaming_context_t* context, iree_string_view_t pointer_name,
    const void* initial_value, iree_host_size_t initial_value_size,
    iree_hal_streaming_symbol_t* pointer_symbol,
    iree_hal_streaming_symbol_t* storage_symbol) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(pointer_symbol);
  IREE_ASSERT_ARGUMENT(storage_symbol);
  if (!pointer_symbol->global_buffer ||
      !pointer_symbol->global_buffer->buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "managed pointer `%.*s` has no HAL buffer",
                            (int)pointer_name.size, pointer_name.data);
  }
  if (pointer_symbol->size_bytes < sizeof(storage_symbol->device_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "managed pointer `%.*s` is too small (%" PRIu64
                            " bytes)",
                            (int)pointer_name.size, pointer_name.data,
                            (uint64_t)pointer_symbol->size_bytes);
  }
  if (!storage_symbol->global_buffer ||
      !storage_symbol->global_buffer->buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "managed storage `%.*s` has no HAL buffer",
                            (int)pointer_name.size, pointer_name.data);
  }
  if (initial_value_size > storage_symbol->size_bytes) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "managed storage `%.*s` is too small (%" PRIhsz
                            " requested, %" PRIu64 " available)",
                            (int)pointer_name.size, pointer_name.data,
                            initial_value_size,
                            (uint64_t)storage_symbol->size_bytes);
  }

  iree_status_t status = iree_ok_status();
  if (initial_value && initial_value_size != 0) {
    status = iree_hal_streaming_direct_transfer_h2d(
        context, initial_value, storage_symbol->global_buffer->buffer,
        /*target_offset=*/0, initial_value_size);
  }
  if (!iree_status_is_ok(status)) return status;

  const iree_hal_streaming_deviceptr_t storage_device_address =
      storage_symbol->device_address;
  return iree_hal_streaming_direct_transfer_h2d(
      context, &storage_device_address, pointer_symbol->global_buffer->buffer,
      /*target_offset=*/0, sizeof(storage_device_address));
}
