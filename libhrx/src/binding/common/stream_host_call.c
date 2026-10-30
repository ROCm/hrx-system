// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/device.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/stream.h"
#include "common/stream_queue.h"

typedef struct iree_hal_streaming_capture_host_function_t {
  void (*fn)(void* user_data);
  void* user_data;
} iree_hal_streaming_capture_host_function_t;

static iree_status_t iree_hal_streaming_capture_record_host_function(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_host_function_t* capture =
      (iree_hal_streaming_capture_host_function_t*)user_data;
  return iree_hal_streaming_graph_add_host_call_node(
      graph, dependencies, dependency_count, capture->fn, capture->user_data,
      out_terminal_node);
}

iree_status_t iree_hal_streaming_queue_host_call(
    iree_hal_streaming_stream_t* stream, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(call.fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);
  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value);
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_value > 0 ? 1 : 0,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    iree_hal_streaming_submission_t submission;
    status = iree_hal_streaming_submission_prepare_locked(
        stream, stream->queue, signal_value,
        IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_host_call(submission.queue, wait_semaphores,
                                        submission.signals, call, args, flags);
    }
    iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                iree_status_is_ok(status));
  }
  iree_slim_mutex_unlock(&stream->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_capture_host_function_t capture = {
      .fn = fn,
      .user_data = user_data,
  };
  bool was_capturing = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_capture_try_record_node(
              stream, iree_hal_streaming_capture_record_host_function, &capture,
              &was_capturing));
  if (was_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_hal_host_call_t call;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_host_callback_prepare(
              stream->context->device_entry->host_callbacks,
              iree_hal_streaming_make_host_function(fn), user_data,
              /*resource=*/NULL, &call));

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_status_t status = iree_hal_streaming_queue_host_call(
      stream, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);
  iree_hal_resource_release(call.resource);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
