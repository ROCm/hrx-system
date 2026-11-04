// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/stream.h"

#include <limits.h>

#include "binding/hip/handle_registry.h"
#include "common/context.h"
#include "common/stream.h"
#include "iree/base/threading/call_once.h"

static iree_once_flag iree_hip_stream_registry_once = IREE_ONCE_FLAG_INIT;
static iree_hip_handle_registry_t iree_hip_stream_registry;

static void iree_hip_stream_registry_initialize(void) {
  iree_hip_handle_registry_initialize(&iree_hip_stream_registry);
}

void iree_hip_stream_trim(void) {
  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  iree_hip_handle_registry_trim(&iree_hip_stream_registry);
}

static void iree_hip_stream_handle_retain(uintptr_t handle) {
  hipStream_t stream = (hipStream_t)handle;
  iree_atomic_ref_count_inc(&stream->ref_count);
}

static iree_host_size_t iree_hip_queue_family_normal_priority_index(
    const iree_hal_queue_family_spec_t* family_spec) {
  iree_host_size_t normal_priority_index = 0;
  while (family_spec->priorities[normal_priority_index] !=
         IREE_HAL_QUEUE_PRIORITY_NORMAL) {
    ++normal_priority_index;
  }
  return normal_priority_index;
}

void iree_hip_queue_family_priority_range(
    const iree_hal_queue_family_t* queue_family, int* out_least_priority,
    int* out_greatest_priority) {
  IREE_ASSERT_ARGUMENT(queue_family);
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  const iree_host_size_t normal_priority_index =
      iree_hip_queue_family_normal_priority_index(family_spec);
  const int least_priority =
      (int)iree_min(normal_priority_index, (iree_host_size_t)INT_MAX);
  const iree_host_size_t higher_priority_count =
      family_spec->priority_count - normal_priority_index - 1;
  const int greatest_priority =
      -(int)iree_min(higher_priority_count, (iree_host_size_t)INT_MAX);
  if (out_least_priority) {
    *out_least_priority = least_priority;
  }
  if (out_greatest_priority) {
    *out_greatest_priority = greatest_priority;
  }
}

iree_hal_queue_priority_t iree_hip_queue_family_select_priority(
    const iree_hal_queue_family_t* queue_family, int requested_priority,
    int* out_hip_priority) {
  IREE_ASSERT_ARGUMENT(queue_family);
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  const iree_host_size_t normal_priority_index =
      iree_hip_queue_family_normal_priority_index(family_spec);
  int least_priority = 0;
  int greatest_priority = 0;
  iree_hip_queue_family_priority_range(queue_family, &least_priority,
                                       &greatest_priority);
  const int hip_priority =
      iree_min(iree_max(requested_priority, greatest_priority), least_priority);
  if (out_hip_priority) {
    *out_hip_priority = hip_priority;
  }
  const iree_host_size_t queue_priority_index =
      hip_priority < 0
          ? normal_priority_index + (iree_host_size_t) - (int64_t)hip_priority
          : normal_priority_index - (iree_host_size_t)hip_priority;
  return family_spec->priorities[queue_priority_index];
}

iree_status_t iree_hip_stream_create(
    iree_hal_streaming_context_t* context, iree_hal_queue_t* queue,
    unsigned int flags, int priority,
    iree_hal_semaphore_list_t initial_wait_semaphores,
    hipStream_t* out_handle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_handle);

  iree_hal_streaming_stream_flags_t stream_flags =
      IREE_HAL_STREAMING_STREAM_FLAG_NONE;
  if (flags & hipStreamNonBlocking) {
    stream_flags |= IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING;
  }

  iree_hal_streaming_stream_t* stream = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, queue, stream_flags, priority, context->host_allocator, &stream);
  if (iree_status_is_ok(status) && initial_wait_semaphores.count > 0) {
    status = iree_hal_streaming_stream_wait_semaphores(stream,
                                                       initial_wait_semaphores);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hip_stream_publish(stream, context->host_allocator, out_handle);
  }
  if (!iree_status_is_ok(status) && stream) {
    iree_hal_streaming_context_unregister_stream(context, stream);
    iree_hal_streaming_stream_release(stream);
  }
  return status;
}

iree_status_t iree_hip_stream_publish(iree_hal_streaming_stream_t* stream,
                                      iree_allocator_t host_allocator,
                                      hipStream_t* out_handle) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_handle);

  hipStream_t handle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*handle), (void**)&handle));
  iree_atomic_ref_count_init(&handle->ref_count);
  iree_slim_mutex_initialize(&handle->mutex);
  handle->stream = stream;
  handle->execution_context = NULL;
  handle->next_execution_context_stream = NULL;
  handle->host_allocator = host_allocator;

  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  iree_status_t status = iree_hip_handle_registry_insert(
      &iree_hip_stream_registry, (uintptr_t)handle);
  if (iree_status_is_ok(status)) {
    *out_handle = handle;
  } else {
    iree_slim_mutex_deinitialize(&handle->mutex);
    iree_allocator_free(host_allocator, handle);
  }
  return status;
}

bool iree_hip_stream_lookup_retain(hipStream_t handle,
                                   hipStream_t* out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  if (!handle || handle == hipStreamLegacy || handle == hipStreamPerThread) {
    return false;
  }

  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  if (!iree_hip_handle_registry_lookup_retain(&iree_hip_stream_registry,
                                              (uintptr_t)handle,
                                              iree_hip_stream_handle_retain)) {
    return false;
  }
  *out_handle = handle;
  return true;
}

bool iree_hip_stream_take(hipStream_t handle, hipStream_t* out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  if (!handle || handle == hipStreamLegacy || handle == hipStreamPerThread) {
    return false;
  }

  iree_call_once(&iree_hip_stream_registry_once,
                 iree_hip_stream_registry_initialize);
  if (!iree_hip_handle_registry_remove(&iree_hip_stream_registry,
                                       (uintptr_t)handle)) {
    return false;
  }
  *out_handle = handle;
  return true;
}

bool iree_hip_stream_retain_attached(
    hipStream_t handle, iree_hal_streaming_stream_t** out_stream,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_stream);
  IREE_ASSERT_ARGUMENT(out_context);

  iree_hal_streaming_stream_t* stream = NULL;
  iree_hal_streaming_context_t* context = NULL;
  iree_slim_mutex_lock(&handle->mutex);
  stream = handle->stream;
  if (stream) {
    iree_hal_streaming_stream_retain(stream);
    if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
      iree_hal_streaming_stream_release(stream);
      stream = NULL;
    }
  }
  iree_slim_mutex_unlock(&handle->mutex);

  if (!stream) {
    return false;
  }
  *out_stream = stream;
  *out_context = context;
  return true;
}

bool iree_hip_stream_detach(hipStream_t handle,
                            iree_hal_streaming_stream_t** out_stream,
                            iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_stream);
  IREE_ASSERT_ARGUMENT(out_context);

  iree_slim_mutex_lock(&handle->mutex);
  iree_hal_streaming_stream_t* stream = handle->stream;
  iree_hal_streaming_context_t* context = NULL;
  if (stream) {
    iree_hal_streaming_stream_retain_context(stream, &context);
    handle->stream = NULL;
  }
  iree_slim_mutex_unlock(&handle->mutex);

  if (!stream) {
    return false;
  }
  *out_stream = stream;
  *out_context = context;
  return true;
}

void iree_hip_stream_retain(hipStream_t handle) {
  if (!handle) {
    return;
  }
  iree_atomic_ref_count_inc(&handle->ref_count);
}

void iree_hip_stream_release(hipStream_t handle) {
  if (!handle) {
    return;
  }
  if (iree_atomic_ref_count_dec(&handle->ref_count) != 1) {
    return;
  }

  iree_slim_mutex_lock(&handle->mutex);
  IREE_ASSERT(handle->stream == NULL);
  IREE_ASSERT(handle->execution_context == NULL);
  IREE_ASSERT(handle->next_execution_context_stream == NULL);
  iree_slim_mutex_unlock(&handle->mutex);

  iree_slim_mutex_deinitialize(&handle->mutex);
  const iree_allocator_t host_allocator = handle->host_allocator;
  iree_allocator_free(host_allocator, handle);
}
