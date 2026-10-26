// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/direct_transfer.h"

#include "common/context.h"
#include "common/device.h"
#include "common/memory.h"
#include "common/stream.h"
#include "transfer.h"

iree_status_t iree_hal_streaming_direct_transfer_h2d(
    iree_hal_streaming_context_t* context, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(source || length == 0);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "direct H2D transfer exceeds host address range");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = source,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
          },
  };
  return hrx_hal_queue_transfer_and_wait(context->device_entry->hrx_device,
                                         /*operation_count=*/1, &operation);
}

iree_status_t iree_hal_streaming_direct_transfer_d2h(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, void* target, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target || length == 0);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "direct D2H transfer exceeds host address range");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target = target,
              .length = length,
          },
  };
  return hrx_hal_queue_transfer_and_wait(context->device_entry->hrx_device,
                                         /*operation_count=*/1, &operation);
}

iree_status_t iree_hal_streaming_direct_transfer_d2d(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
              .flags = IREE_HAL_COPY_FLAG_NONE,
          },
  };
  return hrx_hal_queue_transfer_and_wait(context->device_entry->hrx_device,
                                         /*operation_count=*/1, &operation);
}

typedef struct iree_hal_streaming_host_download_t {
  // Borrowed device whose native transfer path performs host visibility work.
  hrx_device_t device;
  // Number of row downloads in this transaction.
  iree_host_size_t operation_count;
  // Borrowed capture, live on the caller until the transaction completes.
  const iree_hal_transfer_operation_t* operations;
} iree_hal_streaming_host_download_t;

static iree_status_t iree_hal_streaming_host_download_execute(void* user_data) {
  const iree_hal_streaming_host_download_t* download = user_data;
  return hrx_hal_queue_transfer_and_wait(
      download->device, download->operation_count, download->operations);
}

// Unregistered host memory is a blocking compatibility destination. Keep its
// completion on the caller, ordered only after this stream, instead of making
// it depend on unrelated application callbacks or allocating CPU-copy staging.
iree_status_t iree_hal_streaming_direct_transfer_d2h_rows(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream,
    void* destination, iree_device_size_t destination_pitch,
    iree_device_size_t width, iree_host_size_t row_count,
    const iree_hal_streaming_buffer_ref_t* sources) {
  iree_hal_transfer_operation_t single_operation;
  iree_hal_transfer_operation_t* operations = &single_operation;
  if (row_count > 1) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        context->host_allocator, row_count, sizeof(*operations),
        (void**)&operations));
  }
  for (iree_host_size_t row = 0; row < row_count; ++row) {
    operations[row] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = sources[row].buffer->buffer,
                .source_offset = sources[row].offset,
                .target = (uint8_t*)destination + row * destination_pitch,
                .length = width,
            },
    };
  }
  iree_hal_streaming_host_download_t download = {
      .device = context->device_entry->hrx_device,
      .operation_count = row_count,
      .operations = operations,
  };
  iree_status_t status = iree_hal_streaming_execute_host_operation(
      stream, iree_hal_streaming_host_download_execute, &download);
  if (operations != &single_operation) {
    iree_allocator_free(context->host_allocator, operations);
  }
  return status;
}
