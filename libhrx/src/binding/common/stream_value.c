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

static bool iree_hal_streaming_value_wait_lane_matches(
    const iree_hal_streaming_value_wait_lane_t* lane,
    const iree_hal_queue_family_t* family, iree_hal_queue_priority_t priority,
    iree_hal_queue_execution_resource_list_t execution_resources) {
  return lane->family == family && lane->priority == priority &&
         lane->execution_resources.count == execution_resources.count &&
         (execution_resources.count == 0 ||
          memcmp(lane->execution_resources.ordinals,
                 execution_resources.ordinals,
                 execution_resources.count *
                     sizeof(*execution_resources.ordinals)) == 0);
}

static iree_status_t iree_hal_streaming_recycle_value_wait_lanes_locked(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_value_wait_lane_t** next_lane =
      &context->pending_value_wait_lanes;
  while (*next_lane) {
    iree_hal_streaming_value_wait_lane_t* lane = *next_lane;
    uint64_t value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(lane->completion_semaphore, &value);
    if (!iree_status_is_ok(status)) return status;
    if (value < lane->completion_value) {
      next_lane = &lane->next;
      continue;
    }

    *next_lane = lane->next;
    iree_hal_semaphore_release(lane->completion_semaphore);
    lane->completion_semaphore = NULL;
    lane->completion_value = 0;
    lane->next = context->idle_value_wait_lanes;
    context->idle_value_wait_lanes = lane;
  }
  return iree_ok_status();
}

// Acquires a queue that remains exclusive until the submission using it has
// completed. Acquisition is a cold path and preserves the stream's scheduling
// domain; completed queues are recycled across logical stream lifetimes.
static iree_status_t iree_hal_streaming_acquire_value_wait_lane(
    iree_hal_streaming_context_t* context,
    const iree_hal_queue_family_t* family, iree_hal_queue_priority_t priority,
    iree_hal_queue_execution_resource_list_t execution_resources,
    iree_hal_queue_t* excluded_queue,
    iree_hal_streaming_value_wait_lane_t** out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = NULL;

  if (!iree_hal_streaming_queue_family_supports_value_waits(
          iree_hal_queue_family_spec(family))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "stream family cannot provide an independent value-wait queue");
  }

  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  iree_status_t status =
      iree_hal_streaming_recycle_value_wait_lanes_locked(context);
  iree_hal_streaming_value_wait_lane_t** next_lane =
      &context->idle_value_wait_lanes;
  while (iree_status_is_ok(status) && *next_lane && !*out_lane) {
    iree_hal_streaming_value_wait_lane_t* lane = *next_lane;
    if (lane->queue != excluded_queue &&
        iree_hal_streaming_value_wait_lane_matches(lane, family, priority,
                                                   execution_resources)) {
      *next_lane = lane->next;
      lane->next = NULL;
      *out_lane = lane;
    } else {
      next_lane = &lane->next;
    }
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  if (!iree_status_is_ok(status) || *out_lane) return status;

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = priority;
  params.execution_resources = execution_resources;
  iree_hal_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_acquire_queue(context->device, family, &params, &queue));
  if (IREE_UNLIKELY(queue == excluded_queue)) {
    iree_hal_queue_release(queue);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "dynamic queue acquisition returned the stream operation queue");
  }

  iree_hal_streaming_value_wait_lane_t* lane = NULL;
  status = iree_allocator_malloc(context->host_allocator, sizeof(*lane),
                                 (void**)&lane);
  if (!iree_status_is_ok(status)) {
    iree_hal_queue_release(queue);
    return status;
  }
  memset(lane, 0, sizeof(*lane));
  lane->queue = queue;
  lane->family = iree_hal_queue_family(queue);
  lane->priority = iree_hal_queue_priority(queue);
  lane->execution_resources = iree_hal_queue_execution_resources(queue);
  *out_lane = lane;
  return iree_ok_status();
}

static void iree_hal_streaming_release_idle_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane) {
  if (!lane) return;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  lane->next = context->idle_value_wait_lanes;
  context->idle_value_wait_lanes = lane;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
}

static void iree_hal_streaming_publish_pending_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_semaphore_t* completion_semaphore, uint64_t completion_value) {
  iree_hal_semaphore_retain(completion_semaphore);
  lane->completion_semaphore = completion_semaphore;
  lane->completion_value = completion_value;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  lane->next = context->pending_value_wait_lanes;
  context->pending_value_wait_lanes = lane;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
}

void iree_hal_streaming_value_wait_lanes_deinitialize(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_value_wait_lane_t* lists[] = {
      context->idle_value_wait_lanes,
      context->pending_value_wait_lanes,
  };
  context->idle_value_wait_lanes = NULL;
  context->pending_value_wait_lanes = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lists); ++i) {
    iree_hal_streaming_value_wait_lane_t* lane = lists[i];
    while (lane) {
      iree_hal_streaming_value_wait_lane_t* next = lane->next;
      iree_hal_semaphore_release(lane->completion_semaphore);
      iree_hal_queue_release(lane->queue);
      iree_allocator_free(context->host_allocator, lane);
      lane = next;
    }
  }
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
    const iree_hal_execution_stage_t source_stage =
        i == 0 ? IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE
               : IREE_HAL_EXECUTION_STAGE_ATOMIC;
    status = iree_hal_streaming_append_value_operation(
        command_buffer, &operations[i], source_stage,
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
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
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
      stream->context->device, iree_hal_queue_family(stream->queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH |
          IREE_HAL_COMMAND_CATEGORY_ATOMIC,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_execution_stage_t source_stage =
        i == 0 ? IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE
               : IREE_HAL_EXECUTION_STAGE_ATOMIC;
    status = iree_hal_streaming_append_value_operation(
        command_buffer, &operations[i], source_stage,
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
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t* operation_queue,
    iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations,
    iree_hal_command_buffer_t* command_buffer, bool* out_submission_accepted,
    uint64_t* out_signal_value) {
  *out_submission_accepted = false;
  *out_signal_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_validate_value_stream_locked(stream));

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
    *out_submission_accepted = true;
    *out_signal_value = signal_value;
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
  iree_hal_queue_t* excluded_wait_queue = NULL;
  iree_hal_streaming_value_wait_lane_t* wait_lane = NULL;
  const iree_hal_queue_family_t* wait_family = NULL;
  iree_hal_queue_priority_t wait_priority = IREE_HAL_QUEUE_PRIORITY_NORMAL;
  iree_hal_queue_execution_resource_list_t wait_execution_resources = {0};
  iree_status_t status = iree_ok_status();

  // Validate the complete batch before flushing or replacing any recorded
  // stream work. The retained command-buffer path below may still fail while
  // recording, but such a failure only discards that new batch.
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_ref_t target_ref = {0};
    status = iree_hal_streaming_value_operation_target_ref(&operations[i],
                                                           &target_ref);
  }

  // Snapshot an attached stream and reject known capture state before doing
  // fallible preparation. Capture is rechecked at the submission point because
  // it may begin while a flush is in progress.
  iree_slim_mutex_lock(&stream->mutex);
  if (iree_status_is_ok(status)) {
    if (IREE_UNLIKELY(
            !stream->context || !stream->queue ||
            !iree_hal_streaming_context_try_retain(stream->context))) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream execution context has been destroyed");
    } else {
      context = stream->context;
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(stream->capture_status !=
                    IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      status =
          iree_make_status(IREE_STATUS_ABORTED,
                           "stream capture does not support value operation");
    } else {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "stream capture was already invalidated by an earlier operation");
    }
  }
  if (iree_status_is_ok(status)) {
    if (contains_wait) {
      wait_family = iree_hal_queue_family(stream->queue);
      wait_priority = iree_hal_queue_priority(stream->queue);
      wait_execution_resources =
          iree_hal_queue_execution_resources(stream->queue);
      excluded_wait_queue = stream->queue;
      iree_hal_queue_retain(excluded_wait_queue);
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);

  if (iree_status_is_ok(status) && contains_wait) {
    status = iree_hal_streaming_acquire_value_wait_lane(
        context, wait_family, wait_priority, wait_execution_resources,
        excluded_wait_queue, &wait_lane);
    if (iree_status_is_ok(status)) operation_queue = wait_lane->queue;
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status) && contains_wait && operation_count > 1) {
    status = iree_hal_streaming_record_value_operations(
        context->device, iree_hal_queue_family(operation_queue),
        operation_count, operations, &command_buffer);
  }

  if (iree_status_is_ok(status) && contains_wait) {
    bool submission_accepted = false;
    uint64_t signal_value = 0;
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_streaming_stream_flush_locked(stream);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_submit_value_operations_locked(
          stream, operation_queue, operation_count, operations, command_buffer,
          &submission_accepted, &signal_value);
    }
    if (submission_accepted && wait_lane) {
      iree_hal_streaming_publish_pending_value_wait_lane(
          context, wait_lane, stream->timeline_semaphore, signal_value);
      wait_lane = NULL;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  } else if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_streaming_record_write_batch_locked(
        stream, operation_count, operations);
    iree_slim_mutex_unlock(&stream->mutex);
  }

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_streaming_release_idle_value_wait_lane(context, wait_lane);
  iree_hal_queue_release(excluded_wait_queue);
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
