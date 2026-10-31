// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/direct_transfer.h"

#include "common/context.h"
#include "common/device.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "transfer.h"

iree_status_t iree_hal_streaming_direct_transfer_enqueue(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_flush_locked(stream);
  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
  }
  iree_hal_streaming_submission_t submission = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_submission_prepare_locked(
        stream, stream->queue, signal_value,
        IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t waits = {
        .count = wait_value > 0 ? 1 : 0,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    status =
        iree_hal_queue_transfer(submission.queue, waits, submission.signals,
                                operation_count, operations, /*barriers=*/NULL);
  }
  const bool accepted = iree_status_is_ok(status);
  iree_hal_streaming_submission_finish_locked(stream, &submission, accepted);
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_flush(submission.queue);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  if (accepted && !iree_status_is_ok(status)) {
    status =
        iree_status_join(status, iree_hal_streaming_stream_synchronize(stream));
  }
  return status;
}

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
  return hrx_hal_queue_transfer_and_wait(
      context->device_entry->hrx_device,
      /*operation_count=*/1, &operation,
      iree_hal_streaming_context_wait_flags(context));
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
  return hrx_hal_queue_transfer_and_wait(
      context->device_entry->hrx_device,
      /*operation_count=*/1, &operation,
      iree_hal_streaming_context_wait_flags(context));
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
  return hrx_hal_queue_transfer_and_wait(
      context->device_entry->hrx_device,
      /*operation_count=*/1, &operation,
      iree_hal_streaming_context_wait_flags(context));
}

typedef struct iree_hal_streaming_host_download_t {
  // Borrowed device whose native transfer path performs host visibility work.
  hrx_device_t device;
  // Number of row downloads in this transaction.
  iree_host_size_t operation_count;
  // Borrowed capture, live on the caller until the transaction completes.
  const iree_hal_transfer_operation_t* operations;
  // Frontend context's scheduling policy for this blocking host call.
  iree_async_wait_flags_t wait_flags;
} iree_hal_streaming_host_download_t;

static iree_status_t iree_hal_streaming_host_download_execute(void* user_data) {
  const iree_hal_streaming_host_download_t* download = user_data;
  return hrx_hal_queue_transfer_and_wait(
      download->device, download->operation_count, download->operations,
      download->wait_flags);
}

// Unregistered host memory is a blocking compatibility destination. Keep its
// completion on the caller, ordered only after this stream, instead of making
// it depend on unrelated application callbacks or allocating CPU-copy staging.
iree_status_t iree_hal_streaming_direct_transfer_and_wait(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_hal_streaming_host_download_t download = {
      .device = stream->context->device_entry->hrx_device,
      .operation_count = operation_count,
      .operations = operations,
      .wait_flags = iree_hal_streaming_context_wait_flags(stream->context),
  };
  return iree_hal_streaming_execute_host_operation(
      stream, iree_hal_streaming_host_download_execute, &download);
}
