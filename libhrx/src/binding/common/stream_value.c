// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_value.h"

#include "common/context.h"
#include "common/device.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/math.h"
#include "libhrx/src/libhrx/runtime.h"

// Maximum time a write-only batch may remain recorded without another stream
// operation submitting it. A short delay coalesces adjacent scalar writes while
// ensuring that the final successful call makes progress on its own.
enum {
  IREE_HAL_STREAMING_VALUE_FLUSH_DELAY_MS = 1,
};

struct iree_hal_streaming_value_flush_timer_t {
  // One-shot timer submitted to the process async runtime.
  iree_async_timer_operation_t operation;
  // Stream to flush. Retained until the timer callback completes.
  iree_hal_streaming_stream_t* stream;
  // Proactor executing |operation|. Retained until callback completion.
  iree_async_proactor_t* proactor;
  // Allocator that owns this timer state.
  iree_allocator_t host_allocator;
};

static void iree_hal_streaming_value_flush_timer_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_streaming_value_flush_timer_t* timer =
      (iree_hal_streaming_value_flush_timer_t*)user_data;
  iree_hal_streaming_stream_t* stream = timer->stream;

  iree_slim_mutex_lock(&stream->mutex);
  IREE_ASSERT(stream->value_flush_timer == timer,
              "stream must reference its outstanding flush timer");
  stream->value_flush_timer = NULL;
  if (stream->context && stream->queue) {
    status = iree_status_join(status,
                              iree_hal_streaming_stream_flush_locked(stream));
  }
  iree_slim_mutex_unlock(&stream->mutex);

  // There is no initiating host call left to receive an asynchronous submission
  // failure. Poisoning the stream timeline makes every later query, wait, or
  // synchronization observe the original status instead of silently losing it.
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(stream->timeline_semaphore, status);
  }

  iree_async_proactor_release(timer->proactor);
  iree_allocator_free(timer->host_allocator, timer);
  iree_hal_streaming_stream_release(stream);
}

// Schedules one bounded flush for the current write burst. Called with the
// stream mutex held after the write batch has been recorded.
static iree_status_t iree_hal_streaming_schedule_value_flush_locked(
    iree_hal_streaming_stream_t* stream) {
  if (stream->value_flush_timer) {
    return iree_ok_status();
  }

  iree_async_proactor_pool_t* proactor_pool = hrx_runtime_proactor_pool();
  if (IREE_UNLIKELY(!proactor_pool)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "async runtime is unavailable");
  }

  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_pool_get(proactor_pool, /*index=*/0, &proactor));

  iree_hal_streaming_value_flush_timer_t* timer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(stream->host_allocator,
                                             sizeof(*timer), (void**)&timer));
  memset(timer, 0, sizeof(*timer));
  timer->stream = stream;
  timer->proactor = proactor;
  timer->host_allocator = stream->host_allocator;
  iree_async_operation_initialize(
      &timer->operation.base, IREE_ASYNC_OPERATION_TYPE_TIMER,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_streaming_value_flush_timer_callback, timer);
  timer->operation.deadline_ns =
      iree_time_now() +
      iree_make_duration_ms(IREE_HAL_STREAMING_VALUE_FLUSH_DELAY_MS);

  iree_hal_streaming_stream_retain(stream);
  iree_async_proactor_retain(proactor);
  stream->value_flush_timer = timer;
  iree_status_t status =
      iree_async_proactor_submit_one(proactor, &timer->operation.base);
  if (!iree_status_is_ok(status)) {
    stream->value_flush_timer = NULL;
    iree_async_proactor_release(proactor);
    iree_allocator_free(stream->host_allocator, timer);
    iree_hal_streaming_stream_release(stream);
  }
  return status;
}

bool iree_hal_streaming_queue_family_supports_value_waits(
    const iree_hal_queue_family_spec_t* family_spec) {
  if (!family_spec ||
      iree_math_count_ones_u64(family_spec->physical_device_affinity) != 1 ||
      !iree_all_bits_set(family_spec->role_flags,
                         IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC) ||
      !iree_any_bit_set(family_spec->flags,
                        IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
    return false;
  }

  const iree_hal_atomic_capabilities_t* capabilities =
      &family_spec->zero_compute_atomic_capabilities;
  return iree_all_bits_set(capabilities->operations.device_scope_32,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT) &&
         iree_all_bits_set(capabilities->operations.device_scope_64,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT) &&
         iree_all_bits_set(capabilities->operations.system_scope_32,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT) &&
         iree_all_bits_set(capabilities->operations.system_scope_64,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT) &&
         iree_all_bits_set(capabilities->wait_conditions.device_scope_32,
                           IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL) &&
         iree_all_bits_set(capabilities->wait_conditions.device_scope_64,
                           IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL) &&
         iree_all_bits_set(capabilities->wait_conditions.system_scope_32,
                           IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL) &&
         iree_all_bits_set(capabilities->wait_conditions.system_scope_64,
                           IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL);
}

static bool iree_hal_streaming_value_operations_contain_wait(
    iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations) {
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    if (operations[i].kind == IREE_HAL_STREAMING_VALUE_OPERATION_WAIT) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_streaming_value_operation_target_ref(
    const iree_hal_streaming_value_operation_t* operation,
    iree_hal_buffer_ref_t* out_target_ref) {
  IREE_ASSERT_ARGUMENT(operation);
  IREE_ASSERT_ARGUMENT(out_target_ref);
  if (IREE_UNLIKELY(!operation->target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream value target buffer is null");
  }

  iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32;
  switch (operation->kind) {
    case IREE_HAL_STREAMING_VALUE_OPERATION_WAIT:
      width = operation->params.wait.width;
      break;
    case IREE_HAL_STREAMING_VALUE_OPERATION_STORE:
      width = operation->params.store.width;
      break;
    case IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE:
      width = operation->params.update.width;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid stream value operation kind %u",
                              operation->kind);
  }
  if (IREE_UNLIKELY(width != IREE_HAL_ATOMIC_WIDTH_32 &&
                    width != IREE_HAL_ATOMIC_WIDTH_64)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid stream value atomic width %u", width);
  }

  *out_target_ref = iree_hal_make_buffer_ref(
      operation->target_buffer, operation->target_offset, width / 8);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_value_operation_validate(
    const iree_hal_streaming_value_operation_t* operation) {
  iree_hal_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_value_operation_target_ref(operation, &target_ref));
  return iree_hal_buffer_validate_range(target_ref.buffer, target_ref.offset,
                                        target_ref.length);
}

static iree_status_t iree_hal_streaming_append_value_operation(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_streaming_value_operation_t* operation,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage) {
  iree_hal_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_value_operation_target_ref(operation, &target_ref));
  switch (operation->kind) {
    case IREE_HAL_STREAMING_VALUE_OPERATION_WAIT:
      return iree_hal_command_buffer_atomic_wait(command_buffer, source_stage,
                                                 target_stage, target_ref,
                                                 operation->params.wait);
    case IREE_HAL_STREAMING_VALUE_OPERATION_STORE:
      return iree_hal_command_buffer_atomic_store(command_buffer, source_stage,
                                                  target_stage, target_ref,
                                                  operation->params.store);
    case IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE:
      return iree_hal_command_buffer_atomic_rmw(command_buffer, source_stage,
                                                target_stage, target_ref,
                                                operation->params.update);
    default:
      IREE_ASSERT_UNREACHABLE("stream value operation must be valid");
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "invalid stream value operation");
  }
}

iree_status_t iree_hal_streaming_command_buffer_append_value_operations(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_execution_stage_t initial_source_stage,
    iree_hal_execution_stage_t target_stage) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (IREE_UNLIKELY(operation_count == 0 || !operations)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream value operation batch is empty");
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_execution_stage_t source_stage =
        i == 0 ? initial_source_stage : IREE_HAL_EXECUTION_STAGE_ATOMIC;
    status = iree_hal_streaming_append_value_operation(
        command_buffer, &operations[i], source_stage, target_stage);
  }
  return status;
}

static iree_status_t iree_hal_streaming_record_value_operations(
    const iree_hal_queue_family_t* queue_family,
    iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      queue_family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, /*binding_capacity=*/0,
      &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_append_value_operations(
        command_buffer, operation_count, operations,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

static iree_status_t iree_hal_streaming_validate_value_stream_locked(
    iree_hal_streaming_stream_t* stream) {
  if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  if (IREE_UNLIKELY(stream->capture_status !=
                    IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
      return iree_make_status(
          IREE_STATUS_ABORTED,
          "stream capture began before value operation submission");
    }
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "stream capture was already invalidated before value operation");
  }
  return iree_ok_status();
}

// Installs a write-only batch as the first commands in a retained stream
// command buffer. Ordinary stream command buffers remain unretained; this one
// retains its target buffers because a peer allocation can otherwise be freed
// by its owning context before this stream is flushed.
static iree_status_t iree_hal_streaming_record_write_batch_locked(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations) {
  IREE_RETURN_IF_ERROR(iree_hal_streaming_validate_value_stream_locked(stream));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_flush_locked(stream));

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      iree_hal_queue_family(stream->queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH |
          IREE_HAL_COMMAND_CATEGORY_ATOMIC,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_append_value_operations(
        command_buffer, operation_count, operations,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC | IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER);
  }
  if (iree_status_is_ok(status)) {
    stream->command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

static iree_status_t iree_hal_streaming_submit_value_operations_locked(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_command_buffer_t* command_buffer) {
  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value));
  iree_hal_streaming_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_submission_prepare_locked(
      stream, stream->queue, signal_value,
      IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT, &submission));
  const iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  iree_status_t status = iree_ok_status();
  if (operation_count == 1) {
    const iree_hal_streaming_value_operation_t* operation = &operations[0];
    switch (operation->kind) {
      case IREE_HAL_STREAMING_VALUE_OPERATION_WAIT:
        status = iree_hal_queue_atomic_wait(
            submission.queue, wait_semaphores, submission.signals,
            operation->target_buffer, operation->target_offset,
            operation->params.wait, /*barriers=*/NULL);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_STORE:
        status = iree_hal_queue_atomic_store(
            submission.queue, wait_semaphores, submission.signals,
            operation->target_buffer, operation->target_offset,
            operation->params.store, /*barriers=*/NULL);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE:
        status = iree_hal_queue_atomic_rmw(
            submission.queue, wait_semaphores, submission.signals,
            operation->target_buffer, operation->target_offset,
            operation->params.update, /*barriers=*/NULL);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("stream value operation must be valid");
        break;
    }
  } else {
    status = iree_hal_queue_execute(submission.queue, wait_semaphores,
                                    submission.signals, command_buffer,
                                    iree_hal_buffer_binding_table_empty(),
                                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  iree_hal_streaming_submission_finish_locked(stream, &submission,
                                              iree_status_is_ok(status));
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_flush(submission.queue);
  }
  return status;
}

iree_status_t iree_hal_streaming_queue_value_operations(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations) {
  IREE_ASSERT_ARGUMENT(stream);
  if (IREE_UNLIKELY(operation_count == 0 || !operations)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream value operation batch is empty");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation_count);

  const bool contains_wait = iree_hal_streaming_value_operations_contain_wait(
      operation_count, operations);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_value_operation_validate(&operations[i]);
  }

  iree_hal_streaming_context_t* context = NULL;
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_slim_mutex_lock(&stream->mutex);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_validate_value_stream_locked(stream);
  }
  if (iree_status_is_ok(status)) {
    if (iree_hal_streaming_context_try_retain(stream->context)) {
      context = stream->context;
    } else {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream execution context has been destroyed");
    }
  }
  if (iree_status_is_ok(status) && contains_wait &&
      !iree_hal_streaming_queue_family_supports_value_waits(
          iree_hal_queue_family_spec(iree_hal_queue_family(stream->queue)))) {
    status =
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "stream family cannot provide native memory waits");
  }
  // Prepare the complete retained batch before flushing earlier stream work.
  // The stream mutex keeps queue selection and acceptance in submission order.
  if (iree_status_is_ok(status) && contains_wait && operation_count > 1) {
    status = iree_hal_streaming_record_value_operations(
        iree_hal_queue_family(stream->queue), operation_count, operations,
        &command_buffer);
  }
  if (iree_status_is_ok(status) && contains_wait) {
    status = iree_hal_streaming_stream_flush_locked(stream);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_submit_value_operations_locked(
          stream, operation_count, operations, command_buffer);
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_record_write_batch_locked(
        stream, operation_count, operations);
    if (iree_status_is_ok(status)) {
      iree_status_t schedule_status =
          iree_hal_streaming_schedule_value_flush_locked(stream);
      if (!iree_status_is_ok(schedule_status)) {
        // Immediate publication preserves progress when the coalescing timer
        // cannot be scheduled; report both errors if publication also fails.
        status = iree_hal_streaming_stream_flush_locked(stream);
        if (iree_status_is_ok(status)) {
          iree_status_free(schedule_status);
        } else {
          status = iree_status_join(status, schedule_status);
        }
      }
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_streaming_context_release(context);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_queue_wait_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_wait_params_t params) {
  const iree_hal_streaming_value_operation_t operation = {
      .kind = IREE_HAL_STREAMING_VALUE_OPERATION_WAIT,
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .params.wait = params,
  };
  return iree_hal_streaming_queue_value_operations(stream, 1, &operation);
}

iree_status_t iree_hal_streaming_queue_store_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_store_params_t params) {
  const iree_hal_streaming_value_operation_t operation = {
      .kind = IREE_HAL_STREAMING_VALUE_OPERATION_STORE,
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .params.store = params,
  };
  return iree_hal_streaming_queue_value_operations(stream, 1, &operation);
}

iree_status_t iree_hal_streaming_queue_update_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_rmw_params_t params) {
  const iree_hal_streaming_value_operation_t operation = {
      .kind = IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE,
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .params.update = params,
  };
  return iree_hal_streaming_queue_value_operations(stream, 1, &operation);
}
