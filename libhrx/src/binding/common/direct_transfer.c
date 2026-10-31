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

// Internal producers have already validated the operations and byte ranges.
// Accumulate only up to the policy boundary, without overflowing large batches.
static bool iree_hal_streaming_direct_transfer_is_bulk(
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_device_size_t remaining = IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    iree_device_size_t length = 0;
    switch (operations[i].type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        length = operations[i].fill.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        length = operations[i].update.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        length = operations[i].copy.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        length = operations[i].upload.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        length = operations[i].download.length;
        break;
    }
    if (length > remaining) {
      return true;
    }
    remaining -= length;
  }
  return false;
}

static iree_status_t iree_hal_streaming_direct_transfer_queue(
    iree_hal_streaming_context_t* context, iree_hal_queue_t** out_queue) {
  iree_slim_mutex_lock(&context->transfer_queue_mutex);
  iree_status_t status = iree_ok_status();
  if (!context->transfer_queue) {
    status = iree_hal_streaming_device_acquire_transfer_queue(
        context->device_entry, context->queue, &context->transfer_queue);
  }
  *out_queue = context->transfer_queue;
  iree_slim_mutex_unlock(&context->transfer_queue_mutex);
  return status;
}

// Synchronous context copies share staging with ordinary stream copies. Their
// caller has already established any ordering required with preceding work.
static iree_status_t iree_hal_streaming_direct_transfer_context_and_wait(
    iree_hal_streaming_context_t* context, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_hal_queue_t* queue = context->queue;
  if (iree_hal_streaming_direct_transfer_is_bulk(operation_count, operations)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_direct_transfer_queue(context, &queue));
  }
  return hrx_hal_queue_transfer_and_wait(
      queue, operation_count, operations,
      iree_hal_streaming_context_wait_flags(context));
}

// Accepts the native batch before exposing its stream dependency to another
// caller. A blocking copy adds an independent completion proof so a later
// public timeline failure cannot release its borrowed host bytes early.
static iree_status_t iree_hal_streaming_direct_transfer_submit(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_semaphore_t* completion_semaphore, bool* out_accepted) {
  *out_accepted = false;
  iree_hal_queue_t* queue = stream->queue;
  if (queue == stream->context->queue &&
      iree_hal_streaming_direct_transfer_is_bulk(operation_count, operations)) {
    iree_slim_mutex_lock(&stream->mutex);
    const bool needs_transfer_queue = stream->value_wait_lane == NULL;
    iree_slim_mutex_unlock(&stream->mutex);
    if (needs_transfer_queue) {
      // Native queue construction can join completion callbacks. Keep it
      // outside the stream lock, then honor any lease acquired in the meantime.
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_direct_transfer_queue(stream->context, &queue));
    }
  }
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
        stream, stream->value_wait_lane ? stream->queue : queue, signal_value,
        IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t waits = {
        .count = wait_value > 0 ? 1 : 0,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    iree_hal_semaphore_t* signal_semaphores[3];
    uint64_t signal_values[3];
    iree_hal_semaphore_list_t signals = submission.signals;
    if (completion_semaphore) {
      for (iree_host_size_t i = 0; i < signals.count; ++i) {
        signal_semaphores[i] = signals.semaphores[i];
        signal_values[i] = signals.payload_values[i];
      }
      signal_semaphores[signals.count] = completion_semaphore;
      signal_values[signals.count] = 1;
      signals = (iree_hal_semaphore_list_t){
          .count = signals.count + 1,
          .semaphores = signal_semaphores,
          .payload_values = signal_values,
      };
    }
    status = iree_hal_queue_transfer(submission.queue, waits, signals,
                                     operation_count, operations,
                                     /*barriers=*/NULL);
  }
  const bool accepted = iree_status_is_ok(status);
  iree_hal_streaming_submission_finish_locked(stream, &submission, accepted);
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_flush(submission.queue);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  *out_accepted = accepted;
  return status;
}

iree_status_t iree_hal_streaming_direct_transfer_enqueue(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  bool accepted = false;
  iree_status_t status = iree_hal_streaming_direct_transfer_submit(
      stream, operation_count, operations, /*completion_semaphore=*/NULL,
      &accepted);
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
  return iree_hal_streaming_direct_transfer_context_and_wait(
      context, /*operation_count=*/1, &operation);
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
  return iree_hal_streaming_direct_transfer_context_and_wait(
      context, /*operation_count=*/1, &operation);
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
  return iree_hal_streaming_direct_transfer_context_and_wait(
      context, /*operation_count=*/1, &operation);
}

iree_status_t iree_hal_streaming_direct_transfer_and_wait(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  const iree_hal_queue_family_t* family = iree_hal_queue_family(stream->queue);
  iree_hal_semaphore_t* completion_semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      iree_hal_queue_family_device(family),
      iree_hal_make_queue_family_affinity(
          iree_hal_queue_family_ordinal(family)),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &completion_semaphore));
  bool accepted = false;
  iree_status_t status = iree_hal_streaming_direct_transfer_submit(
      stream, operation_count, operations, completion_semaphore, &accepted);
  if (accepted) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(
                    completion_semaphore, 1, iree_infinite_timeout(),
                    iree_hal_streaming_context_wait_flags(stream->context)));
  }
  iree_hal_semaphore_release(completion_semaphore);
  return status;
}
