// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream.h"

#include <stdlib.h>
#include <string.h>

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/graph.h"
#include "common/kernel_arguments.h"
#include "common/memory.h"
#include "common/stream_queue.h"
#include "libhrx/src/libhrx/device.h"

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream);

static iree_status_t iree_hal_streaming_stream_reserve_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    bool* out_was_added) {
  *out_was_added = false;
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    if (stream->memory_reuse_dependencies[i].source_stream_id ==
        source_stream_id) {
      iree_slim_mutex_unlock(&stream->mutex);
      return iree_ok_status();
    }
  }

  iree_status_t status = iree_ok_status();
  if (stream->memory_reuse_dependency_count ==
      stream->memory_reuse_dependency_capacity) {
    // A stream may wait on many producer streams. Grow geometrically so
    // recording those dependencies remains amortized constant time.
    iree_host_size_t new_capacity =
        stream->memory_reuse_dependency_capacity
            ? stream->memory_reuse_dependency_capacity
            : 4;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(new_capacity, 2, &new_capacity) ||
            !iree_host_size_checked_mul(
                new_capacity, sizeof(*stream->memory_reuse_dependencies),
                &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream dependency count overflow");
    } else {
      status =
          iree_allocator_realloc(stream->host_allocator, allocation_size,
                                 (void**)&stream->memory_reuse_dependencies);
      if (iree_status_is_ok(status)) {
        stream->memory_reuse_dependency_capacity = new_capacity;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    stream->memory_reuse_dependencies[stream->memory_reuse_dependency_count++] =
        (iree_hal_streaming_memory_reuse_dependency_t){
            .source_stream_id = source_stream_id,
            .source_timeline_value = 0,
        };
    *out_was_added = true;
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return status;
}

static void iree_hal_streaming_stream_record_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value) {
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id) {
      dependency->source_timeline_value =
          iree_max(dependency->source_timeline_value, source_timeline_value);
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

static void
iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id) {
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id &&
        dependency->source_timeline_value == 0) {
      --stream->memory_reuse_dependency_count;
      stream->memory_reuse_dependencies[i] =
          stream->memory_reuse_dependencies
              [stream->memory_reuse_dependency_count];
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

bool iree_hal_streaming_stream_has_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value) {
  bool has_dependency = false;
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    const iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id &&
        dependency->source_timeline_value >= source_timeline_value) {
      has_dependency = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return has_dependency;
}

void iree_hal_streaming_stream_set_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t new_status) {
  const iree_hal_streaming_capture_status_t old_status = stream->capture_status;
  if (old_status == new_status) {
    return;
  }
  if (old_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      new_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_hal_streaming_context_enter_capture(stream->context);
  }
  stream->capture_status = new_status;
  if (old_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      new_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_hal_streaming_context_leave_capture(stream->context);
  }
}

iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context, iree_hal_queue_t* queue,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->ref_count);
  stream->context = context;
  stream->registration_state =
      IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED;
  stream->flags = flags;
  stream->priority = priority;
  stream->stream_id = 0;
  stream->command_buffer = NULL;
  stream->value_flush_timer = NULL;
  stream->pending_launch_count = 0;
  stream->timeline_semaphore = NULL;
  stream->pending_value = 0;
  stream->completed_value = 0;
  stream->queue = queue;
  iree_hal_queue_retain(stream->queue);
  stream->cooperative_queue = NULL;
  stream->value_wait_lane = NULL;
  stream->memory_reuse_dependencies = NULL;
  stream->memory_reuse_dependency_count = 0;
  stream->memory_reuse_dependency_capacity = 0;

  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  stream->capture_graph = NULL;
  stream->capture_graph_owned = false;
  stream->capture_origin = false;
  stream->capture_id = 0;
  stream->capture_owner_thread_id = 0;
  stream->capture_dependencies = NULL;
  stream->capture_dependency_count = 0;
  stream->capture_dependency_capacity = 0;

  stream->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&stream->mutex);

  // Create timeline semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(
      context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ULL,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &stream->timeline_semaphore);

  // Register stream with context.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_register_stream(context, stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_hal_streaming_stream_destroy(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_select_cooperative_queue_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_queue);

  if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  if (iree_all_bits_set(iree_hal_queue_features(stream->queue),
                        IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)) {
    *out_queue = stream->queue;
    return iree_ok_status();
  }
  if (stream->cooperative_queue) {
    *out_queue = stream->cooperative_queue;
    return iree_ok_status();
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(stream->queue);
  params.placement = iree_hal_queue_placement(stream->queue);
  params.features = iree_hal_queue_features(stream->queue) |
                    IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
  params.execution_resources =
      iree_hal_queue_execution_resources(stream->queue);
  iree_hal_queue_t* cooperative_queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_queue_acquire(
      iree_hal_queue_family(stream->queue), &params, &cooperative_queue));

  stream->cooperative_queue = cooperative_queue;
  *out_queue = cooperative_queue;
  return iree_ok_status();
}

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_context_t* context = stream->context;
  iree_slim_mutex_unlock(&stream->mutex);
  if (context) {
    iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) {
      iree_hal_queue_t* queue = stream->queue;
      iree_hal_queue_t* cooperative_queue = stream->cooperative_queue;
      stream->queue = NULL;
      stream->cooperative_queue = NULL;
      stream->context = NULL;
      iree_slim_mutex_unlock(&stream->mutex);
      iree_hal_queue_release(cooperative_queue);
      iree_hal_queue_release(queue);
    } else {
      iree_slim_mutex_unlock(&stream->mutex);
    }
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

  iree_allocator_free(stream->host_allocator,
                      stream->memory_reuse_dependencies);

  IREE_ASSERT(stream->value_flush_timer == NULL,
              "a flush timer reference must keep its stream alive");

  // Release command buffer.
  iree_hal_command_buffer_release(stream->command_buffer);

  if (stream->capture_graph_owned) {
    iree_hal_streaming_graph_release(stream->capture_graph);
  }
  iree_allocator_free(stream->host_allocator, stream->capture_dependencies);

  // Release timeline semaphore.
  iree_hal_semaphore_release(stream->timeline_semaphore);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&stream->mutex);

  // Free stream memory.
  const iree_allocator_t host_allocator = stream->host_allocator;
  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream) {
  if (stream) {
    iree_atomic_ref_count_inc(&stream->ref_count);
  }
}

void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream) {
  if (stream && iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    iree_hal_streaming_stream_destroy(stream);
  }
}

bool iree_hal_streaming_stream_retain_context(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_context_t* context = stream->context;
  if (iree_hal_streaming_context_try_retain(context)) {
    *out_context = context;
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return *out_context != NULL;
}

static const iree_hal_access_scope_t iree_hal_streaming_device_access_scope =
    IREE_HAL_ACCESS_SCOPE_DISPATCH_READ | IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
    IREE_HAL_ACCESS_SCOPE_TRANSFER_READ | IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;

// Empty source/target access scopes express entry acquire/terminal release.
// Internal dependencies need both; submission completion alone does not publish
// recorded writes or acquire earlier stream writes for a later recording.
static iree_status_t iree_hal_streaming_stream_record_dependency(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_access_scope_t source_scope,
    iree_hal_access_scope_t target_scope) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = source_scope,
      .target_scope = target_scope,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = source_scope
                               ? IREE_HAL_EXECUTION_STAGE_DISPATCH |
                                     IREE_HAL_EXECUTION_STAGE_TRANSFER
                               : IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = target_scope
                               ? IREE_HAL_EXECUTION_STAGE_DISPATCH |
                                     IREE_HAL_EXECUTION_STAGE_TRANSFER
                               : IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  return iree_hal_command_buffer_barrier(command_buffer, &barrier);
}

iree_status_t iree_hal_streaming_stream_begin_locked(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);

  // Create command buffer if not already created. HIP pointer lifetime is
  // enforced by free/unregister ordering, not by launch-time pointer discovery:
  // synchronous destruction waits for active streams, and asynchronous
  // destruction queues the release after prior stream work.
  iree_status_t status = iree_ok_status();
  if (!stream->command_buffer) {
    iree_hal_command_buffer_t* command_buffer = NULL;
    status = iree_hal_command_buffer_create(
        iree_hal_queue_family(stream->queue),
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED |
            hrx_device_profile_command_buffer_mode(
                stream->context->device_entry->hrx_device),
        IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/0, &command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_begin(command_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_stream_record_dependency(
          command_buffer, 0, iree_hal_streaming_device_access_scope);
    }
    if (iree_status_is_ok(status)) {
      stream->command_buffer = command_buffer;
    } else {
      iree_hal_command_buffer_release(command_buffer);
    }
  } else if (stream->pending_launch_count) {
    // Acquire a dispatch's writes only when there is a following consumer.
    // Transfer recorders already publish their own dependencies, including
    // stronger system releases when required for host-visible results.
    status = iree_hal_streaming_stream_record_dependency(
        stream->command_buffer, iree_hal_streaming_device_access_scope,
        iree_hal_streaming_device_access_scope);
  }

  return status;
}

iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_flush_locked(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  if (!stream->command_buffer) {
    return iree_ok_status();
  }

  iree_hal_command_buffer_t* command_buffer = stream->command_buffer;
  iree_status_t status = iree_ok_status();
  if (stream->pending_launch_count) {
    // Retain the producer release at the submission boundary without an
    // acquire for nonexistent later commands. The next recording acquires at
    // entry, after its stream timeline prerequisite has completed.
    status = iree_hal_streaming_stream_record_dependency(
        command_buffer, iree_hal_streaming_device_access_scope, 0);
  }
  stream->command_buffer = NULL;
  stream->pending_launch_count = 0;

  // Once flushing has been attempted, a one-shot command buffer cannot accept
  // more commands and must be released on every path below.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  if (iree_status_is_ok(status)) {
    // Timeline chaining preserves stream order across command-buffer
    // boundaries. A rejected submission does not consume |signal_value|.
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
  }
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
      status = iree_hal_queue_execute(submission.queue, wait_semaphores,
                                      submission.signals, command_buffer,
                                      iree_hal_buffer_binding_table_empty(),
                                      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    }
    iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                iree_status_is_ok(status));
    if (iree_status_is_ok(status)) {
      // An accepted submission owns the value it signals, even if flushing the
      // hardware queue subsequently reports an error.
      status = iree_hal_queue_flush(submission.queue);
    }
  }

  iree_hal_command_buffer_release(command_buffer);
  return status;
}

iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_flush_locked(stream);
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_completion_state_t* out_completion_state) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_completion_state);

  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_flush(stream));

  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value));

  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t pending_value = stream->pending_value;
  if (current_value >= pending_value) {
    *out_completion_state = IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
    stream->completed_value = iree_max(stream->completed_value, current_value);
  } else {
    *out_completion_state = IREE_HAL_STREAMING_COMPLETION_STATE_PENDING;
  }
  iree_slim_mutex_unlock(&stream->mutex);

  return iree_ok_status();
}

typedef struct iree_hal_streaming_wait_dependency_t {
  iree_hal_streaming_stream_t* source_stream;
  uint64_t source_timeline_value;
  bool added_memory_reuse_dependency;
} iree_hal_streaming_wait_dependency_t;

enum { IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT = 8 };

iree_status_t iree_hal_streaming_stream_wait_streams(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* const* source_streams,
    iree_host_size_t source_stream_count) {
  IREE_ASSERT_ARGUMENT(stream);
  if (source_stream_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(source_streams);

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);
  const bool destination_is_capturing =
      stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_hal_queue_t* destination_queue = stream->queue;
  iree_slim_mutex_unlock(&stream->mutex);
  if (destination_is_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "destination stream is capturing");
  }

  iree_status_t status = iree_ok_status();
  iree_hal_streaming_wait_dependency_t
      inline_dependencies[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT];
  iree_hal_semaphore_t*
      inline_wait_semaphores[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT +
                             1];
  uint64_t
      inline_wait_values[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT + 1];
  iree_hal_streaming_wait_dependency_t* dependencies = inline_dependencies;
  iree_hal_semaphore_t** wait_semaphores_storage = inline_wait_semaphores;
  uint64_t* wait_values_storage = inline_wait_values;
  bool uses_heap_storage =
      source_stream_count > IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT;
  if (uses_heap_storage) {
    iree_host_size_t wait_capacity = 0;
    iree_host_size_t dependencies_size = 0;
    iree_host_size_t semaphores_size = 0;
    iree_host_size_t values_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(source_stream_count, 1,
                                                  &wait_capacity) ||
                      !iree_host_size_checked_mul(
                          source_stream_count,
                          sizeof(iree_hal_streaming_wait_dependency_t),
                          &dependencies_size) ||
                      !iree_host_size_checked_mul(wait_capacity,
                                                  sizeof(iree_hal_semaphore_t*),
                                                  &semaphores_size) ||
                      !iree_host_size_checked_mul(
                          wait_capacity, sizeof(uint64_t), &values_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream dependency count overflow");
    }
    dependencies = NULL;
    wait_semaphores_storage = NULL;
    wait_values_storage = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, dependencies_size,
                                     (void**)&dependencies);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, semaphores_size,
                                     (void**)&wait_semaphores_storage);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, values_size,
                                     (void**)&wait_values_storage);
    }
  }

  iree_host_size_t dependency_count = 0;
  iree_hal_streaming_submission_flags_t submission_flags =
      IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE;
  for (iree_host_size_t i = 0;
       i < source_stream_count && iree_status_is_ok(status); ++i) {
    iree_hal_streaming_stream_t* source_stream = source_streams[i];
    if (!source_stream || source_stream == stream) {
      continue;
    }
    if (source_stream->context != stream->context) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stream dependency crosses contexts");
      break;
    }
    iree_slim_mutex_lock(&source_stream->mutex);
    const bool source_is_capturing =
        source_stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
    iree_slim_mutex_unlock(&source_stream->mutex);
    if (source_is_capturing) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "source stream is capturing");
      break;
    }

    // Flush first so the captured value names all source work preceding this
    // operation. A source that has never submitted contributes no wait.
    status = iree_hal_streaming_stream_flush(source_stream);
    if (!iree_status_is_ok(status)) {
      break;
    }
    iree_slim_mutex_lock(&source_stream->mutex);
    uint64_t source_timeline_value = 0;
    const bool source_requires_independent_queue =
        source_stream->value_wait_lane != NULL ||
        source_stream->queue != destination_queue;
    if (IREE_UNLIKELY(source_stream->capture_status !=
                      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "source stream is capturing");
    } else {
      source_timeline_value = source_stream->pending_value;
    }
    iree_slim_mutex_unlock(&source_stream->mutex);
    if (!iree_status_is_ok(status)) {
      break;
    }

    status = iree_status_join(
        status,
        iree_hal_streaming_memory_release_completed_async_frees(source_stream));
    if (!iree_status_is_ok(status) || source_timeline_value == 0) {
      continue;
    }

    // Every later destination operation already follows an accepted join of
    // this source point. Repeating it adds no ordering and can move a pending
    // external wait back onto a shared queue used by its producer.
    if (iree_hal_streaming_stream_has_memory_reuse_dependency(
            stream, source_stream->stream_id, source_timeline_value)) {
      continue;
    }

    iree_hal_streaming_wait_dependency_t* dependency =
        &dependencies[dependency_count];
    dependency->source_stream = source_stream;
    dependency->source_timeline_value = source_timeline_value;
    dependency->added_memory_reuse_dependency = false;
    status = iree_hal_streaming_stream_reserve_memory_reuse_dependency(
        stream, source_stream->stream_id,
        &dependency->added_memory_reuse_dependency);
    if (iree_status_is_ok(status)) {
      ++dependency_count;
      if (source_requires_independent_queue) {
        submission_flags |= IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT;
      }
    }
  }

  // Submit prior destination work before appending a queue dependency. The
  // barrier advances its timeline only after every captured source point.
  bool submitted = false;
  if (iree_status_is_ok(status) && dependency_count > 0) {
    status = iree_hal_streaming_stream_flush(stream);
  }
  if (iree_status_is_ok(status) && dependency_count > 0) {
    iree_slim_mutex_lock(&stream->mutex);
    uint64_t destination_timeline_value = 0;
    uint64_t signal_value = 0;
    if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream execution context has been destroyed");
    } else if (IREE_UNLIKELY(stream->capture_status !=
                             IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "destination stream is capturing");
    } else {
      status = iree_hal_streaming_stream_reserve_next_value_locked(
          stream, &destination_timeline_value, &signal_value);
    }
    if (iree_status_is_ok(status)) {
      iree_host_size_t wait_count = 0;
      if (destination_timeline_value > 0) {
        wait_semaphores_storage[wait_count] = stream->timeline_semaphore;
        wait_values_storage[wait_count] = destination_timeline_value;
        ++wait_count;
      }
      for (iree_host_size_t i = 0; i < dependency_count; ++i) {
        wait_semaphores_storage[wait_count] =
            dependencies[i].source_stream->timeline_semaphore;
        wait_values_storage[wait_count] = dependencies[i].source_timeline_value;
        ++wait_count;
      }
      const iree_hal_semaphore_list_t wait_semaphores = {
          .count = wait_count,
          .semaphores = wait_semaphores_storage,
          .payload_values = wait_values_storage,
      };
      iree_hal_streaming_submission_t submission;
      status = iree_hal_streaming_submission_prepare_locked(
          stream, stream->queue, signal_value, submission_flags, &submission);
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_barrier(submission.queue, wait_semaphores,
                                        submission.signals, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
      }
      iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                  iree_status_is_ok(status));
      if (iree_status_is_ok(status)) {
        // The accepted barrier owns the value it signals, so the timeline
        // advances here and stays advanced even when the flush below fails.
        submitted = true;
        status = iree_hal_queue_flush(submission.queue);
      }
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  for (iree_host_size_t i = 0; i < dependency_count; ++i) {
    iree_hal_streaming_wait_dependency_t* dependency = &dependencies[i];
    if (submitted) {
      iree_hal_streaming_stream_record_memory_reuse_dependency(
          stream, dependency->source_stream->stream_id,
          dependency->source_timeline_value);
    } else if (dependency->added_memory_reuse_dependency) {
      iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
          stream, dependency->source_stream->stream_id);
    }
  }

  if (uses_heap_storage) {
    iree_allocator_free(stream->host_allocator, wait_values_storage);
    iree_allocator_free(stream->host_allocator, wait_semaphores_storage);
    iree_allocator_free(stream->host_allocator, dependencies);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_wait_stream(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* source_stream) {
  IREE_ASSERT_ARGUMENT(source_stream);
  return iree_hal_streaming_stream_wait_streams(stream, &source_stream, 1);
}

iree_status_t iree_hal_streaming_stream_wait_semaphores(
    iree_hal_streaming_stream_t* stream,
    iree_hal_semaphore_list_t wait_semaphores) {
  IREE_ASSERT_ARGUMENT(stream);
  if (wait_semaphores.count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!wait_semaphores.semaphores ||
                    !wait_semaphores.payload_values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream wait semaphore list storage is null");
  }

  bool has_wait = false;
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    if (IREE_UNLIKELY(!wait_semaphores.semaphores[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stream wait semaphore is null");
    }
    has_wait |= wait_semaphores.payload_values[i] != 0;
  }
  if (!has_wait) {
    return iree_ok_status();
  }

  // Reject known-invalid states before flushing any pending stream work. The
  // checks repeat under the submission lock below because either state may
  // change while the flush is in progress.
  iree_slim_mutex_lock(&stream->mutex);
  const bool is_attached = stream->context && stream->queue;
  const bool is_capturing =
      stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_slim_mutex_unlock(&stream->mutex);
  if (IREE_UNLIKELY(!is_attached)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  if (IREE_UNLIKELY(is_capturing)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is capturing");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_t*
      inline_semaphores[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT + 1];
  uint64_t inline_values[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT + 1];
  iree_hal_semaphore_t** semaphore_storage = inline_semaphores;
  uint64_t* value_storage = inline_values;
  iree_status_t status = iree_ok_status();
  iree_host_size_t required_capacity = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(wait_semaphores.count, 1,
                                                &required_capacity))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "stream wait semaphore count overflow");
  }
  const bool uses_heap_storage =
      required_capacity > IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT + 1;
  if (uses_heap_storage) {
    iree_host_size_t semaphore_storage_size = 0;
    iree_host_size_t value_storage_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(required_capacity,
                                                  sizeof(*semaphore_storage),
                                                  &semaphore_storage_size) ||
                      !iree_host_size_checked_mul(required_capacity,
                                                  sizeof(*value_storage),
                                                  &value_storage_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream wait semaphore count overflow");
    }
    semaphore_storage = NULL;
    value_storage = NULL;
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(stream->host_allocator, semaphore_storage_size,
                                (void**)&semaphore_storage);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, value_storage_size,
                                     (void**)&value_storage);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_flush(stream);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);
    if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream execution context has been destroyed");
    } else if (IREE_UNLIKELY(stream->capture_status !=
                             IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream is capturing");
    }

    // A wait on this stream's own current or earlier timeline point adds no
    // ordering. Refuse a future point, which only this stream could signal and
    // would therefore deadlock it, and count the external dependencies that
    // require a barrier.
    const uint64_t current_stream_value = stream->pending_value;
    iree_host_size_t external_wait_count = 0;
    for (iree_host_size_t i = 0;
         i < wait_semaphores.count && iree_status_is_ok(status); ++i) {
      iree_hal_semaphore_t* semaphore = wait_semaphores.semaphores[i];
      const uint64_t value = wait_semaphores.payload_values[i];
      if (value == 0) {
        continue;
      } else if (semaphore == stream->timeline_semaphore) {
        // The stream is the sole signaler of its timeline. A point no later
        // than its current tail is already covered by natural stream ordering;
        // waiting on a future point would deadlock the stream against itself.
        if (IREE_UNLIKELY(value > current_stream_value)) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "stream cannot wait on a future point of its own timeline");
        }
      } else {
        ++external_wait_count;
      }
    }

    uint64_t stream_wait_value = 0;
    uint64_t stream_signal_value = 0;
    if (iree_status_is_ok(status) && external_wait_count > 0) {
      status = iree_hal_streaming_stream_reserve_next_value_locked(
          stream, &stream_wait_value, &stream_signal_value);
    }

    iree_host_size_t wait_count = 0;
    if (iree_status_is_ok(status) && external_wait_count > 0) {
      if (stream_wait_value > 0) {
        semaphore_storage[wait_count] = stream->timeline_semaphore;
        value_storage[wait_count] = stream_wait_value;
        ++wait_count;
      }
      for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
        iree_hal_semaphore_t* semaphore = wait_semaphores.semaphores[i];
        const uint64_t value = wait_semaphores.payload_values[i];
        if (value > 0 && semaphore != stream->timeline_semaphore) {
          semaphore_storage[wait_count] = semaphore;
          value_storage[wait_count] = value;
          ++wait_count;
        }
      }

      const iree_hal_semaphore_list_t combined_waits = {
          .count = wait_count,
          .semaphores = semaphore_storage,
          .payload_values = value_storage,
      };
      iree_hal_streaming_submission_t submission;
      status = iree_hal_streaming_submission_prepare_locked(
          stream, stream->queue, stream_signal_value,
          IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT, &submission);
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_barrier(submission.queue, combined_waits,
                                        submission.signals, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
      }
      iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                  iree_status_is_ok(status));
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_flush(submission.queue);
      }
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  if (uses_heap_storage) {
    iree_allocator_free(stream->host_allocator, value_storage);
    iree_allocator_free(stream->host_allocator, semaphore_storage);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_stream_synchronize_impl(
    iree_hal_streaming_stream_t* stream, bool flush_context) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flush_context) {
    // HIP launches are logically submitted work even when HRX batches command
    // buffer recording. A flush error may be reported after queue submission
    // has already advanced this stream's accepted tail, so retain that error
    // while still draining the published point below.
    status = iree_hal_streaming_context_flush(stream->context);
    // Context flushing snapshots its registered streams. If that snapshot
    // allocation fails the target stream may still own an unsubmitted command
    // buffer, so always flush it directly before selecting the wait target.
    // A second flush is a no-op when the context flush already handled it.
    status = iree_status_join(status, iree_hal_streaming_stream_flush(stream));
  }

  // Snapshot the synchronization target under the same mutex used to reserve
  // timeline values. Operations submitted after this point are not part of
  // this synchronization operation.
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t target_value = stream->pending_value;
  uint64_t completed_value = stream->completed_value;
  const iree_async_wait_flags_t wait_flags =
      stream->context ? iree_hal_streaming_context_wait_flags(stream->context)
                      : IREE_ASYNC_WAIT_FLAG_NONE;
  iree_slim_mutex_unlock(&stream->mutex);

  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value);
  if (iree_status_is_ok(query_status) && current_value > completed_value) {
    completed_value = current_value;
  }
  status = iree_status_join(status, query_status);
  // The stream timeline is private: each published value is a HAL queue signal
  // (whose contract publishes success only after terminal completion and
  // reports post-accept failures through that same signal), except host
  // operations that fail it only after their callback has returned. Therefore
  // either result of this infinite wait is a terminal resolution of the
  // accepted tail, even though only success advances |completed_value|.
  bool target_completed = target_value <= completed_value;
  if (!target_completed) {
    iree_status_t wait_status =
        iree_hal_semaphore_wait(stream->timeline_semaphore, target_value,
                                iree_infinite_timeout(), wait_flags);
    target_completed = iree_status_is_ok(wait_status);
    status = iree_status_join(status, wait_status);
    if (target_completed) {
      completed_value = target_value;
    }
  }

  if (target_completed) {
    iree_slim_mutex_lock(&stream->mutex);
    stream->completed_value =
        iree_max(stream->completed_value, completed_value);
    iree_slim_mutex_unlock(&stream->mutex);
    status = iree_status_join(
        status,
        iree_hal_streaming_memory_release_completed_async_frees(stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream) {
  return iree_hal_streaming_stream_synchronize_impl(stream,
                                                    /*flush_context=*/true);
}

iree_status_t iree_hal_streaming_stream_synchronize_flushed(
    iree_hal_streaming_stream_t* stream) {
  return iree_hal_streaming_stream_synchronize_impl(stream,
                                                    /*flush_context=*/false);
}

iree_status_t iree_hal_streaming_execute_host_operation(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_host_operation_fn_t fn, void* user_data) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value);
  if (iree_status_is_ok(status)) {
    // Claim the timeline point before waiting. Later submissions may proceed
    // concurrently, but they will wait for this operation's terminal signal.
    stream->pending_value = signal_value;
  }
  iree_slim_mutex_unlock(&stream->mutex);

  if (iree_status_is_ok(status) && wait_value != 0) {
    status = iree_hal_semaphore_wait(
        stream->timeline_semaphore, wait_value, iree_infinite_timeout(),
        stream->context ? iree_hal_streaming_context_wait_flags(stream->context)
                        : IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = fn(user_data);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_fail(stream->timeline_semaphore,
                              iree_status_clone(status));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_signal(stream->timeline_semaphore, signal_value,
                                       /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_fail(stream->timeline_semaphore,
                              iree_status_clone(status));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_wait_submitted(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for already-submitted work to complete WITHOUT flushing.
  // Snapshot the tail of accepted work under the stream lock. New submissions
  // after this point are outside this wait operation.
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t pending_value = stream->pending_value;
  iree_slim_mutex_unlock(&stream->mutex);
  if (pending_value > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_semaphore_wait(
            stream->timeline_semaphore, pending_value, iree_infinite_timeout(),
            stream->context
                ? iree_hal_streaming_context_wait_flags(stream->context)
                : IREE_ASYNC_WAIT_FLAG_NONE));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

typedef struct iree_hal_streaming_capture_event_wait_t {
  iree_hal_streaming_event_t* event;
} iree_hal_streaming_capture_event_wait_t;

static iree_status_t iree_hal_streaming_capture_record_event_wait(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_event_wait_t* capture =
      (iree_hal_streaming_capture_event_wait_t*)user_data;
  return iree_hal_streaming_graph_add_event_node(
      graph, dependencies, dependency_count,
      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT, capture->event,
      out_terminal_node);
}

typedef struct iree_hal_streaming_capture_kernel_t {
  iree_hal_streaming_symbol_t* symbol;
  const iree_hal_streaming_dispatch_params_t* params;
} iree_hal_streaming_capture_kernel_t;

static iree_status_t iree_hal_streaming_capture_record_kernel(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_kernel_t* capture =
      (iree_hal_streaming_capture_kernel_t*)user_data;
  return iree_hal_streaming_graph_add_kernel_node(
      graph, dependencies, dependency_count, capture->symbol, capture->params,
      out_terminal_node);
}
typedef struct iree_hal_streaming_captured_event_snapshot_t {
  iree_hal_streaming_graph_t* graph;
  unsigned long long capture_id;
  iree_hal_streaming_graph_node_t** dependencies;
  iree_host_size_t dependency_count;
} iree_hal_streaming_captured_event_snapshot_t;

static iree_status_t iree_hal_streaming_event_snapshot_capture(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_captured_event_snapshot_t* out_snapshot) {
  *out_snapshot = (iree_hal_streaming_captured_event_snapshot_t){0};

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&event->mutex);
  if (event->capture_graph) {
    iree_host_size_t allocation_size = 0;
    if (event->capture_id == 0) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "captured event has no session identifier");
    } else if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                   event->capture_dependency_count,
                   sizeof(*out_snapshot->dependencies), &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "captured event frontier size overflow");
    } else if (allocation_size > 0) {
      status = iree_allocator_malloc(event->host_allocator, allocation_size,
                                     (void**)&out_snapshot->dependencies);
    }
    if (iree_status_is_ok(status)) {
      if (allocation_size > 0) {
        memcpy(out_snapshot->dependencies, event->capture_dependencies,
               allocation_size);
      }
      out_snapshot->dependency_count = event->capture_dependency_count;
      out_snapshot->capture_id = event->capture_id;
      out_snapshot->graph = event->capture_graph;
      iree_hal_streaming_graph_retain(out_snapshot->graph);
    }
  }
  iree_slim_mutex_unlock(&event->mutex);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(event->host_allocator, out_snapshot->dependencies);
    *out_snapshot = (iree_hal_streaming_captured_event_snapshot_t){0};
  }
  return status;
}

static void iree_hal_streaming_captured_event_snapshot_deinitialize(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_captured_event_snapshot_t* snapshot) {
  iree_allocator_free(event->host_allocator, snapshot->dependencies);
  iree_hal_streaming_graph_release(snapshot->graph);
  *snapshot = (iree_hal_streaming_captured_event_snapshot_t){0};
}

iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    bool capture_external_wait) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  // An external wait remains an explicit node so each graph launch resolves
  // the event point supplied by the application at execution time.
  if (capture_external_wait) {
    iree_hal_streaming_capture_event_wait_t capture = {
        .event = event,
    };
    bool was_captured = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_try_record_node(
                stream, iree_hal_streaming_capture_record_event_wait, &capture,
                &was_captured));
    if (was_captured) {
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Capture graph, exact session, and frontier are copied as one event value.
  // The event lock is released before the graph transaction is acquired, which
  // preserves graph -> stream -> event lock ordering.
  iree_hal_streaming_captured_event_snapshot_t capture_snapshot;
  iree_status_t capture_status =
      iree_hal_streaming_event_snapshot_capture(event, &capture_snapshot);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, capture_status);
  if (capture_snapshot.graph) {
    iree_hal_streaming_context_t* context = NULL;
    if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
      capture_status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "stream execution context has been destroyed");
    } else {
      iree_hal_streaming_capture_admission_begin_transition(
          &context->capture_admission);
      capture_status = iree_hal_streaming_capture_join_graph(
          stream, capture_snapshot.graph, capture_snapshot.capture_id,
          capture_snapshot.dependencies, capture_snapshot.dependency_count);
      iree_hal_streaming_capture_admission_end_transition(
          &context->capture_admission);
      iree_hal_streaming_context_release(context);
    }
    iree_hal_streaming_captured_event_snapshot_deinitialize(event,
                                                            &capture_snapshot);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, capture_status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Read the point once. The barrier below waits on exactly the point whose
  // stream ordering is filed as a reuse dependency, so a record landing on this
  // event concurrently cannot make the filed dependency describe a point other
  // than the one waited on.
  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);

  // Waiting on the point orders every later submission on this stream behind
  // it, and therefore behind the stream timeline point it follows. That is the
  // ordering a deferred free on that stream needs before its allocation may be
  // reused here. A point that follows no stream timeline point files nothing:
  // there is no ordering to claim. A point on this stream's own timeline files
  // nothing either, because same-stream ordering already covers it.
  const unsigned long long source_stream_id =
      recorded_point.ordered_after_stream_id;
  const uint64_t source_timeline_value =
      recorded_point.ordered_after_stream_value;
  const bool files_memory_reuse_dependency =
      event->context == stream->context && source_stream_id != 0 &&
      source_stream_id != stream->stream_id && source_timeline_value != 0;
  bool added_memory_reuse_dependency = false;
  // True once the queue has accepted the barrier that establishes the ordering.
  bool submitted = false;
  iree_status_t status = iree_ok_status();
  if (files_memory_reuse_dependency) {
    status = iree_hal_streaming_stream_reserve_memory_reuse_dependency(
        stream, source_stream_id, &added_memory_reuse_dependency);
  }

  // Flush the stream to ensure all prior operations are submitted.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_flush(stream);
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);

    // Reserve the next stream timeline value and submit a barrier that
    // completes only after the event is signaled. The value is submitted, not
    // completed; query/synchronize advance completed_value after observing the
    // semaphore.
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
    if (iree_status_is_ok(status)) {
      // The barrier waits on the point the event was recorded at and on
      // everything already on this stream, so the value it signals stays behind
      // the value below it. Either wait is dropped when there is nothing behind
      // it: a stream that has never submitted, or an event with no submitted
      // record.
      iree_hal_semaphore_t* wait_semaphore_storage[2];
      uint64_t wait_value_storage[2];
      iree_host_size_t wait_count = 0;
      if (wait_value > 0) {
        wait_semaphore_storage[wait_count] = stream->timeline_semaphore;
        wait_value_storage[wait_count] = wait_value;
        ++wait_count;
      }
      if (recorded_point.semaphore) {
        wait_semaphore_storage[wait_count] = recorded_point.semaphore;
        wait_value_storage[wait_count] = recorded_point.value;
        ++wait_count;
      }
      iree_hal_semaphore_list_t wait_semaphores = {
          .count = wait_count,
          .semaphores = wait_semaphore_storage,
          .payload_values = wait_value_storage,
      };
      iree_hal_streaming_submission_t submission;
      status = iree_hal_streaming_submission_prepare_locked(
          stream, stream->queue, signal_value,
          recorded_point.requires_independent_queue
              ? IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT
              : IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE,
          &submission);

      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_barrier(submission.queue, wait_semaphores,
                                        submission.signals, /*barriers=*/NULL,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
      }
      iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                  iree_status_is_ok(status));
      if (iree_status_is_ok(status)) {
        // The accepted barrier owns the value it signals, so the timeline
        // advances here and stays advanced even when the flush below fails.
        submitted = true;
        status = iree_hal_queue_flush(submission.queue);
      }
    }

    iree_slim_mutex_unlock(&stream->mutex);
  }

  iree_hal_streaming_event_release_recorded_point(&recorded_point);

  // The reservation holds the dependency slot from before the submission so a
  // concurrent reservation for the same source stream cannot displace it. It
  // carries no value until the submission that establishes the ordering is
  // accepted, and is withdrawn when that submission is not.
  if (files_memory_reuse_dependency) {
    if (submitted) {
      iree_hal_streaming_stream_record_memory_reuse_dependency(
          stream, source_stream_id, source_timeline_value);
    } else if (added_memory_reuse_dependency) {
      iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
          stream, source_stream_id);
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

typedef struct iree_hal_streaming_launch_arguments_t {
  // Native constants passed to HAL dispatch.
  void* constants;
  // Number of bytes in |constants|.
  iree_host_size_t constants_size;
  // HAL buffer bindings passed to dispatch.
  iree_hal_buffer_ref_list_t bindings;
  // Whether |constants| already contains a target-native argument image.
  bool use_raw_arguments;
} iree_hal_streaming_launch_arguments_t;

static iree_status_t iree_hal_streaming_prepare_launch_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_context_t* context, bool is_native_kernel,
    bool is_empty_native_kernel, bool is_pre_packed, bool is_args_array,
    uint8_t* argument_storage, iree_host_size_t constants_storage_size,
    iree_host_size_t constants_capacity, iree_host_size_t binding_capacity,
    iree_hal_streaming_launch_arguments_t* out_arguments) {
  *out_arguments = (iree_hal_streaming_launch_arguments_t){
      .constants = constants_capacity ? argument_storage : NULL,
      .constants_size = symbol->parameters.constant_bytes,
      .bindings =
          {
              .count = binding_capacity,
              .values = binding_capacity
                            ? (iree_hal_buffer_ref_t*)(argument_storage +
                                                       constants_storage_size)
                            : NULL,
          },
      .use_raw_arguments = false,
  };

  if (is_pre_packed) {
    // Pre-packed buffers are already in native kernarg layout. They may
    // contain device pointers either as formal pointer arguments or inside
    // opaque data, so preserve the bytes exactly and rely on HIP lifetime
    // ordering instead of trying to translate visible pointer slots.
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_validate_prepacked_kernel_arguments(symbol, params));
    out_arguments->constants = params->buffer;
    out_arguments->constants_size = params->buffer_size;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_args_array) {
    // Pointer-array launches are converted to native kernarg bytes. This keeps
    // formal pointer arguments and pointers nested inside copied structs on the
    // same device-pointer contract.
    out_arguments->constants_size = symbol->parameters.direct_arg_bytes
                                        ? symbol->parameters.direct_arg_bytes
                                        : symbol->parameters.constant_bytes;
    if (out_arguments->constants_size == 0) {
      out_arguments->constants_size = symbol->parameters.buffer_size;
    }
    IREE_RETURN_IF_ERROR(iree_hal_streaming_pack_raw_argument_list(
        &symbol->parameters, (void**)params->buffer, out_arguments->constants,
        &out_arguments->constants_size));
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_native_kernel && params->buffer) {
    // Native kernel with pre-packed buffer: pass raw arguments directly. This
    // path is only valid when the caller did not declare the buffer as a HIP
    // args array, because a void** parameter list is not a native kernarg pack.
    out_arguments->constants = params->buffer;
    if (params->buffer_size > 0) {
      out_arguments->constants_size = params->buffer_size;
    }
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (params->buffer) {
    // Unflagged launches use the shared metadata path. HIP entry points mark
    // native kernarg bytes with PRE_PACKED or provide ARGS_ARRAY so device
    // pointer values are preserved without depending on reflected pointer
    // slots.
    iree_status_t status = iree_hal_streaming_unpack_parameters(
        context, &symbol->parameters, params->buffer, out_arguments->constants,
        &out_arguments->bindings);
    if (iree_status_is_ok(status)) {
      return status;
    }
    if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
      return status;
    }

    // External device pointers cannot be expressed as HAL bindings. Preserve
    // the caller's raw argument image instead of partially translating it.
    iree_status_ignore(status);
    out_arguments->constants = params->buffer;
    out_arguments->constants_size = params->buffer_size;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_empty_native_kernel) {
    out_arguments->constants = NULL;
    out_arguments->constants_size = 0;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "kernel launch missing parameter storage");
}

// Appends one dispatch to a stream command buffer while |stream->mutex| is
// held. The next recorded operation acquires this dispatch's writes; flushing
// records the terminal release when there is no subsequent consumer.
static iree_status_t iree_hal_streaming_record_dispatch_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_symbol_t* symbol,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_dispatch(
        stream->command_buffer, symbol->executable,
        iree_hal_executable_function_from_index(symbol->export_ordinal), config,
        constants, bindings, flags);
  }

  if (iree_status_is_ok(status)) {
    ++stream->pending_launch_count;
  }
  return status;
}

iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  const bool cooperative_dispatch = iree_any_bit_set(
      params->flags, IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE);

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Verify parameter storage early for metadata-described launches. The
  // concrete branch below decides whether the storage is a native byte image,
  // a pointer array to pack, or a shared unflagged metadata buffer.
  if (!params->buffer && symbol->parameters.buffer_size > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct kernel launch missing expected parameters");
  }

  // Avoid entering the capture transaction on the ordinary launch path. The
  // submission lock below revalidates the stream in case capture begins after
  // this context-wide snapshot.
  iree_hal_streaming_capture_kernel_t capture = {
      .symbol = symbol,
      .params = params,
  };
  bool was_captured = false;
  if (IREE_UNLIKELY(
          stream->context &&
          iree_hal_streaming_context_has_capture_streams(stream->context))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_try_record_node(
                stream, iree_hal_streaming_capture_record_kernel, &capture,
                &was_captured));
  }
  if (was_captured) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Check if this is a "native" kernel without IREE parameter metadata.
  // Native kernels have no bindings and no copy operations.
  const bool is_native_kernel = symbol->parameters.binding_count == 0 &&
                                symbol->parameters.copy_count == 0;
  const bool is_empty_native_kernel =
      is_native_kernel &&
      iree_hal_streaming_parameter_info_is_empty(&symbol->parameters);

  // Check if this is a pre-packed buffer (HIP_LAUNCH_PARAM_BUFFER format).
  // Pre-packed buffers are already in the kernel's native ABI format and must
  // be passed directly without unpacking or pointer rewriting.
  const bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;
  const bool is_args_array =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
  if (is_args_array && is_native_kernel && !is_empty_native_kernel) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-empty args-array kernel launch requires parameter metadata");
  }
  if (!is_pre_packed && !is_args_array && !params->buffer &&
      !is_empty_native_kernel) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel launch missing parameter storage");
  }

  // Use one aligned temporary allocation for constants and binding refs. Most
  // kernels fit in a bounded inline buffer; larger reflected layouts fall back
  // to the stream host allocator and leave through one cleanup path.
  iree_host_size_t constants_capacity = 0;
  iree_host_size_t binding_capacity = 0;
  if (is_args_array && !is_empty_native_kernel) {
    constants_capacity = iree_max(symbol->parameters.direct_arg_bytes,
                                  iree_max(symbol->parameters.constant_bytes,
                                           symbol->parameters.buffer_size));
  } else if (!is_pre_packed && !is_native_kernel) {
    constants_capacity = symbol->parameters.constant_bytes;
    binding_capacity = symbol->parameters.binding_count;
  }
  iree_host_size_t constants_storage_size = 0;
  iree_host_size_t bindings_storage_size = 0;
  iree_host_size_t temporary_storage_size = 0;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
                        constants_capacity, iree_alignof(iree_hal_buffer_ref_t),
                        &constants_storage_size) ||
                    !iree_host_size_checked_mul(binding_capacity,
                                                sizeof(iree_hal_buffer_ref_t),
                                                &bindings_storage_size) ||
                    !iree_host_size_checked_add(constants_storage_size,
                                                bindings_storage_size,
                                                &temporary_storage_size))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel argument storage size overflow");
  }

  enum { IREE_HAL_STREAMING_INLINE_ARGUMENT_STORAGE_SIZE = 256 };
  iree_alignas(iree_max_align_t) uint8_t
      inline_argument_storage[IREE_HAL_STREAMING_INLINE_ARGUMENT_STORAGE_SIZE];
  uint8_t* allocated_argument_storage = NULL;
  uint8_t* argument_storage = inline_argument_storage;
  if (iree_status_is_ok(status) &&
      temporary_storage_size > sizeof(inline_argument_storage)) {
    status = iree_allocator_malloc_uninitialized(
        stream->host_allocator, temporary_storage_size,
        (void**)&allocated_argument_storage);
    if (iree_status_is_ok(status)) {
      argument_storage = allocated_argument_storage;
    }
  }

  iree_hal_streaming_launch_arguments_t arguments = {0};
  if (iree_status_is_ok(status)) {
    if (temporary_storage_size > 0 && !is_args_array) {
      memset(argument_storage, 0, temporary_storage_size);
    }
    status = iree_hal_streaming_prepare_launch_arguments(
        symbol, params, stream->context, is_native_kernel,
        is_empty_native_kernel, is_pre_packed, is_args_array, argument_storage,
        constants_storage_size, constants_capacity, binding_capacity,
        &arguments);
  }

  if (iree_status_is_ok(status)) {
    // Create IREE dispatch config.
    const iree_hal_dispatch_config_t config = {
        .workgroup_size =
            {
                params->block_dim[0],
                params->block_dim[1],
                params->block_dim[2],
            },
        .workgroup_count =
            {
                params->grid_dim[0],
                params->grid_dim[1],
                params->grid_dim[2],
            },
        .workitem_count =
            {
                params->workitem_count[0],
                params->workitem_count[1],
                params->workitem_count[2],
            },
        .dynamic_workgroup_local_memory = params->shared_memory_bytes,
    };

    // HIP launches use native kernarg bytes. The AMDGPU queue code still
    // populates the dispatch implicit arguments for CUSTOM_DIRECT_ARGUMENTS;
    // this flag only says that the explicit argument payload is already native.
    iree_hal_dispatch_flags_t flags =
        (arguments.use_raw_arguments || is_pre_packed)
            ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
            : IREE_HAL_DISPATCH_FLAG_NONE;
    if (cooperative_dispatch) {
      flags |= IREE_HAL_DISPATCH_FLAG_COOPERATIVE;
    }
    if (params->workitem_count[0] || params->workitem_count[1] ||
        params->workitem_count[2]) {
      flags |= IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
    }

    for (;;) {
      iree_slim_mutex_lock(&stream->mutex);
      if (IREE_UNLIKELY(stream->capture_status !=
                        IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
        iree_slim_mutex_unlock(&stream->mutex);
        was_captured = false;
        status = iree_hal_streaming_capture_try_record_node(
            stream, iree_hal_streaming_capture_record_kernel, &capture,
            &was_captured);
        if (!iree_status_is_ok(status) || was_captured) {
          break;
        }
        continue;
      }

      // Ordinary launches must progress even if this is the final API call on
      // their stream. Capture owns deferred execution; an uncaptured launch
      // transfers its arguments and dependencies to the queue before returning.
      if (stream->command_buffer) {
        status = iree_hal_streaming_stream_flush_locked(stream);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_queue_t* dispatch_queue = stream->queue;
        if (cooperative_dispatch) {
          status = iree_hal_streaming_stream_select_cooperative_queue_locked(
              stream, &dispatch_queue);
        }
        uint64_t wait_value = 0;
        uint64_t signal_value = 0;
        if (iree_status_is_ok(status)) {
          status = iree_hal_streaming_stream_reserve_next_value_locked(
              stream, &wait_value, &signal_value);
        }
        const iree_hal_semaphore_list_t wait_semaphores = {
            .count = wait_value > 0 ? 1 : 0,
            .semaphores = &stream->timeline_semaphore,
            .payload_values = &wait_value,
        };
        iree_hal_streaming_submission_t submission = {0};
        if (iree_status_is_ok(status)) {
          status = iree_hal_streaming_submission_prepare_locked(
              stream, dispatch_queue, signal_value,
              IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_queue_dispatch(
              submission.queue, wait_semaphores, submission.signals,
              symbol->executable,
              iree_hal_executable_function_from_index(symbol->export_ordinal),
              config,
              iree_make_const_byte_span(arguments.constants,
                                        arguments.constants_size),
              arguments.bindings, /*barriers=*/NULL, flags);
        }
        iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                    iree_status_is_ok(status));
        if (iree_status_is_ok(status)) {
          status = iree_hal_queue_flush(submission.queue);
        }
      }
      iree_slim_mutex_unlock(&stream->mutex);
      break;
    }
  }
  iree_allocator_free(stream->host_allocator, allocated_argument_storage);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_streaming_prepared_launch_t {
  // Temporary native argument bytes consumed during command recording.
  uint8_t* constants;
  // Number of bytes in |constants|.
  iree_host_size_t constants_size;
} iree_hal_streaming_prepared_launch_t;

static int iree_hal_streaming_compare_stream_ids(const void* lhs,
                                                 const void* rhs) {
  const iree_hal_streaming_stream_t* lhs_stream =
      *(iree_hal_streaming_stream_t* const*)lhs;
  const iree_hal_streaming_stream_t* rhs_stream =
      *(iree_hal_streaming_stream_t* const*)rhs;
  if (lhs_stream->stream_id < rhs_stream->stream_id) {
    return -1;
  }
  if (lhs_stream->stream_id > rhs_stream->stream_id) {
    return 1;
  }
  const uintptr_t lhs_address = (uintptr_t)lhs_stream;
  const uintptr_t rhs_address = (uintptr_t)rhs_stream;
  return lhs_address < rhs_address ? -1 : lhs_address > rhs_address ? 1 : 0;
}

iree_status_t iree_hal_streaming_launch_kernel_batch(
    iree_host_size_t launch_count,
    const iree_hal_streaming_kernel_launch_t* launches) {
  if (launch_count == 0 || !launches) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel launch batch must not be empty");
  }

  iree_host_size_t prepared_size = 0;
  iree_host_size_t lock_order_size = 0;
  if (!iree_host_size_checked_mul(launch_count,
                                  sizeof(iree_hal_streaming_prepared_launch_t),
                                  &prepared_size) ||
      !iree_host_size_checked_mul(launch_count, sizeof(launches[0].stream),
                                  &lock_order_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel launch batch allocation size overflow");
  }

  iree_host_size_t constants_size = 0;
  for (iree_host_size_t i = 0; i < launch_count; ++i) {
    const iree_hal_streaming_kernel_launch_t* launch = &launches[i];
    if (!launch->symbol || !launch->stream) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel launch batch contains a null handle");
    }
    if (launch->symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel launch batch symbol is not a function");
    }
    if (launch->params.flags != IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel launch batch requires pointer-array arguments");
    }

    const iree_hal_streaming_parameter_info_t* parameters =
        &launch->symbol->parameters;
    const bool is_native_kernel =
        parameters->binding_count == 0 && parameters->copy_count == 0;
    if (!launch->params.buffer && parameters->buffer_size > 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel launch is missing parameter storage");
    }
    if (is_native_kernel && launch->params.buffer &&
        !iree_hal_streaming_parameter_info_is_empty(parameters)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "non-empty pointer-array launch requires parameter metadata");
    }

    iree_host_size_t launch_constants_size = parameters->direct_arg_bytes
                                                 ? parameters->direct_arg_bytes
                                                 : parameters->constant_bytes;
    if (launch_constants_size == 0) {
      launch_constants_size = parameters->buffer_size;
    }
    if (!iree_host_size_checked_add(constants_size, launch_constants_size,
                                    &constants_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel argument storage size overflow");
    }
  }

  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(prepared_size, lock_order_size,
                                  &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, constants_size,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel launch batch allocation size overflow");
  }

  const iree_allocator_t host_allocator = launches[0].stream->host_allocator;
  uint8_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, allocation_size, (void**)&allocation));
  iree_hal_streaming_prepared_launch_t* prepared =
      (iree_hal_streaming_prepared_launch_t*)allocation;
  iree_hal_streaming_stream_t** lock_order =
      (iree_hal_streaming_stream_t**)(allocation + prepared_size);
  uint8_t* constants = allocation + prepared_size + lock_order_size;
  memset(prepared, 0, prepared_size);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < launch_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_streaming_kernel_launch_t* launch = &launches[i];
    lock_order[i] = launch->stream;
    iree_host_size_t constants_capacity =
        launch->symbol->parameters.direct_arg_bytes
            ? launch->symbol->parameters.direct_arg_bytes
            : launch->symbol->parameters.constant_bytes;
    if (constants_capacity == 0) {
      constants_capacity = launch->symbol->parameters.buffer_size;
    }
    prepared[i].constants = constants_capacity ? constants : NULL;
    prepared[i].constants_size = constants_capacity;
    status = iree_hal_streaming_pack_raw_argument_list(
        &launch->symbol->parameters, (void**)launch->params.buffer,
        prepared[i].constants, &prepared[i].constants_size);
    constants += constants_capacity;
  }

  if (iree_status_is_ok(status)) {
    qsort(lock_order, launch_count, sizeof(lock_order[0]),
          iree_hal_streaming_compare_stream_ids);
    for (iree_host_size_t i = 1; i < launch_count; ++i) {
      if (lock_order[i - 1] == lock_order[i]) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "kernel launch batch repeats a stream");
        break;
      }
    }
  }

  iree_host_size_t locked_count = 0;
  if (iree_status_is_ok(status)) {
    for (; locked_count < launch_count; ++locked_count) {
      iree_slim_mutex_lock(&lock_order[locked_count]->mutex);
    }
    for (iree_host_size_t i = 0; i < launch_count; ++i) {
      if (lock_order[i]->capture_status !=
          IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "kernel launch batch does not support stream capture");
        break;
      }
    }
  }

  iree_host_size_t recorded_count = 0;
  for (iree_host_size_t i = 0; i < launch_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_streaming_kernel_launch_t* launch = &launches[i];
    const iree_hal_dispatch_config_t config = {
        .workgroup_size =
            {
                launch->params.block_dim[0],
                launch->params.block_dim[1],
                launch->params.block_dim[2],
            },
        .workgroup_count =
            {
                launch->params.grid_dim[0],
                launch->params.grid_dim[1],
                launch->params.grid_dim[2],
            },
        .workitem_count =
            {
                launch->params.workitem_count[0],
                launch->params.workitem_count[1],
                launch->params.workitem_count[2],
            },
        .dynamic_workgroup_local_memory = launch->params.shared_memory_bytes,
    };
    iree_hal_dispatch_flags_t flags =
        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
    if (launch->params.workitem_count[0] || launch->params.workitem_count[1] ||
        launch->params.workitem_count[2]) {
      flags |= IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
    }
    status = iree_hal_streaming_record_dispatch_locked(
        launch->stream, launch->symbol, config,
        iree_make_const_byte_span(prepared[i].constants,
                                  prepared[i].constants_size),
        iree_hal_buffer_ref_list_empty(), flags);
    if (iree_status_is_ok(status)) {
      ++recorded_count;
    }
  }

  while (locked_count > 0) {
    iree_slim_mutex_unlock(&lock_order[--locked_count]->mutex);
  }
  // Submit every recorded member before returning so independent device queues
  // can overlap. This remains asynchronous: flushing only transfers ownership
  // to the queues and does not wait for completion.
  for (iree_host_size_t i = 0; i < recorded_count; ++i) {
    status = iree_status_join(
        status, iree_hal_streaming_stream_flush(launches[i].stream));
  }

  iree_allocator_free(host_allocator, allocation);
  return status;
}
