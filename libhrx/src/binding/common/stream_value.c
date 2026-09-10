// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_value.h"

#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/internal/math.h"

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

// Selects the stream-owned queue used by operations that may block on a memory
// predicate. The caller must hold |stream->mutex|. Queue acquisition is a cold
// first-use path and preserves the stream's scheduling domain.
static iree_status_t iree_hal_streaming_select_value_wait_queue_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  if (stream->value_wait_queue) {
    *out_queue = stream->value_wait_queue;
    return iree_ok_status();
  }

  const iree_hal_queue_family_t* family = iree_hal_queue_family(stream->queue);
  if (!iree_hal_streaming_queue_family_supports_value_waits(
          iree_hal_queue_family_spec(family))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "stream family cannot provide an independent value-wait queue");
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(stream->queue);
  params.execution_resources =
      iree_hal_queue_execution_resources(stream->queue);
  iree_hal_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_acquire_queue(stream->context->device,
                                                     family, &params, &queue));
  if (IREE_UNLIKELY(queue == stream->queue)) {
    iree_hal_queue_release(queue);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "dynamic queue acquisition returned the stream operation queue");
  }

  stream->value_wait_queue = queue;
  *out_queue = queue;
  return iree_ok_status();
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

static iree_status_t iree_hal_streaming_record_value_operations(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, queue_family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, /*binding_capacity=*/0,
      &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_streaming_value_operation_t* operation = &operations[i];
    iree_hal_buffer_ref_t target_ref = {0};
    status =
        iree_hal_streaming_value_operation_target_ref(operation, &target_ref);
    if (!iree_status_is_ok(status)) break;

    const iree_hal_execution_stage_t source_stage =
        i == 0 ? IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE
               : IREE_HAL_EXECUTION_STAGE_ATOMIC;
    switch (operation->kind) {
      case IREE_HAL_STREAMING_VALUE_OPERATION_WAIT:
        status = iree_hal_command_buffer_atomic_wait(
            command_buffer, source_stage, IREE_HAL_EXECUTION_STAGE_ATOMIC,
            target_ref, operation->params.wait);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_STORE:
        status = iree_hal_command_buffer_atomic_store(
            command_buffer, source_stage, IREE_HAL_EXECUTION_STAGE_ATOMIC,
            target_ref, operation->params.store);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE:
        status = iree_hal_command_buffer_atomic_rmw(
            command_buffer, source_stage, IREE_HAL_EXECUTION_STAGE_ATOMIC,
            target_ref, operation->params.update);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("stream value operation must be valid");
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "invalid stream value operation");
        break;
    }
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

static iree_status_t iree_hal_streaming_submit_value_operations_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t* operation_queue,
    iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_command_buffer_t* command_buffer) {
  if (IREE_UNLIKELY(!stream->context || !stream->queue)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  if (IREE_UNLIKELY(stream->capture_status !=
                    IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
    }
    return iree_make_status(
        IREE_STATUS_ABORTED,
        "stream capture began before value operation submission");
  }

  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value));
  const iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  iree_status_t status = iree_ok_status();
  if (operation_count == 1) {
    const iree_hal_streaming_value_operation_t* operation = &operations[0];
    switch (operation->kind) {
      case IREE_HAL_STREAMING_VALUE_OPERATION_WAIT:
        status = iree_hal_queue_atomic_wait(
            operation_queue, wait_semaphores, signal_semaphores,
            operation->target_buffer, operation->target_offset,
            operation->params.wait);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_STORE:
        status = iree_hal_queue_atomic_store(
            operation_queue, wait_semaphores, signal_semaphores,
            operation->target_buffer, operation->target_offset,
            operation->params.store);
        break;
      case IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE:
        status = iree_hal_queue_atomic_rmw(
            operation_queue, wait_semaphores, signal_semaphores,
            operation->target_buffer, operation->target_offset,
            operation->params.update);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("stream value operation must be valid");
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "invalid stream value operation");
        break;
    }
  } else {
    status = iree_hal_queue_execute(operation_queue, wait_semaphores,
                                    signal_semaphores, command_buffer,
                                    iree_hal_buffer_binding_table_empty(),
                                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_value = signal_value;
    status = iree_hal_queue_flush(operation_queue);
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
  iree_hal_streaming_context_t* context = NULL;
  iree_hal_queue_t* operation_queue = NULL;
  iree_status_t status = iree_ok_status();

  // Snapshot an attached stream and reject known capture state before doing
  // fallible preparation. Capture is rechecked at the submission point because
  // it may begin while a flush is in progress.
  iree_slim_mutex_lock(&stream->mutex);
  if (IREE_UNLIKELY(!stream->context || !stream->queue ||
                    !iree_hal_streaming_context_try_retain(stream->context))) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream execution context has been destroyed");
  } else {
    context = stream->context;
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(stream->capture_status !=
                    IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
    }
    status = iree_make_status(
        IREE_STATUS_ABORTED, "stream capture does not support value operation");
  }
  if (iree_status_is_ok(status)) {
    operation_queue = stream->queue;
    if (contains_wait) {
      status = iree_hal_streaming_select_value_wait_queue_locked(
          stream, &operation_queue);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_queue_retain(operation_queue);
  }
  iree_slim_mutex_unlock(&stream->mutex);

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status) && operation_count > 1) {
    status = iree_hal_streaming_record_value_operations(
        context->device, iree_hal_queue_family(operation_queue),
        operation_count, operations, &command_buffer);
  } else if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t target_ref = {0};
    status = iree_hal_streaming_value_operation_target_ref(&operations[0],
                                                           &target_ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_flush(stream);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_streaming_submit_value_operations_locked(
        stream, operation_queue, operation_count, operations, command_buffer);
    iree_slim_mutex_unlock(&stream->mutex);
  }

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_queue_release(operation_queue);
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
