// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/operation.h"

#include "iree/hal/driver/amd/gpu/queue/sdma/state.h"

iree_status_t iree_hal_amd_gpu_sdma_operation_create(
    iree_hal_amd_gpu_sdma_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_size,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_sdma_operation_t** out_operation, void** out_payload) {
  *out_operation = NULL;
  *out_payload = NULL;
  const iree_host_size_t header_size =
      sizeof(iree_hal_amd_gpu_sdma_operation_t) +
      (queue->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0);
  iree_host_size_t size = 0, waits_offset = 0, values_offset = 0;
  iree_host_size_t signals_offset = 0, payload_offset = 0;
  iree_host_size_t pending_offset = 0, tracker_offset = 0, tracker_size = 0;
  if (waits.count) {
    IREE_RETURN_IF_ERROR(
        iree_async_semaphore_wait_tracker_storage_size(1, &tracker_size));
  }
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      header_size, &size,
      IREE_STRUCT_FIELD(waits.count, iree_hal_amd_gpu_sdma_wait_t,
                        &waits_offset),
      IREE_STRUCT_FIELD(signals.count, uint64_t, &values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*, &signals_offset),
      IREE_STRUCT_FIELD(waits.count ? 1 : 0, iree_hal_amd_gpu_sdma_pending_t,
                        &pending_offset),
      IREE_STRUCT_FIELD_ALIGNED(tracker_size, uint8_t, iree_max_align_t,
                                &tracker_offset)));
  if (!iree_host_size_checked_align(size, 8, &payload_offset) ||
      !iree_host_size_checked_add(payload_offset, payload_size, &size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SDMA operation capture size overflows");
  }
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_capture_pool_acquire(&queue->captures, size, &storage));
  iree_hal_amd_gpu_sdma_operation_t* operation = (void*)storage.data;
  *operation = (iree_hal_amd_gpu_sdma_operation_t){
      .queue = queue,
      .allocation_size = storage.data_length,
      .waits = (void*)(storage.data + waits_offset),
      .wait_count = waits.count,
      .signals =
          {
              .count = signals.count,
              .semaphores = (void*)(storage.data + signals_offset),
              .payload_values = (void*)(storage.data + values_offset),
          },
      .before = iree_hal_amd_gpu_sdma_barrier_prepare(
          barriers ? barriers->before : NULL),
      .after = iree_hal_amd_gpu_sdma_barrier_prepare(barriers ? barriers->after
                                                              : NULL),
      .profiled = queue->profile != NULL,
  };
  iree_atomic_ref_count_init(&operation->ref_count);
  if (waits.count) {
    operation->pending = (void*)(storage.data + pending_offset);
    *operation->pending = (iree_hal_amd_gpu_sdma_pending_t){
        .operation = operation,
        .tracker = {.buffer = storage.data + tracker_offset,
                    .capacity = tracker_size}};
  }
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    operation->waits[i] = (iree_hal_amd_gpu_sdma_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  *out_operation = operation;
  *out_payload = storage.data + payload_offset;
  return iree_ok_status();
}

void iree_hal_amd_gpu_sdma_operation_release_payload(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER:
      iree_hal_amd_gpu_transfer_deinitialize(&operation->transfer);
      break;
    case IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE:
      if (operation->execute.instance) {
        if (!operation->program_published) {
          iree_hal_amd_gpu_program_cancel(operation->execute.storage,
                                          operation->execute.instance);
        } else {
          iree_hal_amd_gpu_program_release(operation->execute.storage,
                                           operation->execute.instance);
        }
      }
      iree_hal_amd_gpu_sdma_command_invocation_deinitialize(
          &operation->execute.invocation);
      break;
    case IREE_HAL_AMD_GPU_SDMA_OPERATION_TIMESTAMP:
      iree_hal_buffer_release(operation->timestamp.buffer);
      break;
    case IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION:
      iree_hal_amd_gpu_allocation_destroy(operation->allocation);
      break;
    default:
      break;
  }
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_BARRIER;
}

static void iree_hal_amd_gpu_sdma_operation_free(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  iree_hal_amd_gpu_capture_pool_t* pool = &operation->queue->captures;
  const iree_host_size_t allocation_size = operation->allocation_size;
  iree_hal_amd_gpu_sdma_operation_release_payload(operation);
  for (iree_host_size_t i = 0; i < operation->wait_count; ++i) {
    iree_hal_amd_gpu_signal_release(operation->waits[i].signal);
    iree_hal_semaphore_release(operation->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < operation->signals.count; ++i) {
    iree_hal_semaphore_release(operation->signals.semaphores[i]);
  }
  iree_hal_amd_gpu_signal_release(operation->signal);
  iree_hal_amd_gpu_capture_pool_release(
      pool, iree_make_byte_span(operation, allocation_size));
}

static void iree_hal_amd_gpu_sdma_operation_destroy(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  if (!operation->submitted) {
    iree_hal_amd_gpu_sdma_operation_free(operation);
    return;
  }
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  const iree_hal_amd_gpu_retirement_callback_t callback = operation->callback;
  const iree_async_single_frontier_t completed = operation->frontier;
  const iree_async_frontier_t* frontier =
      completed.entry_count
          ? iree_async_single_frontier_as_const_frontier(&completed)
          : NULL;
  iree_status_t status =
      iree_status_from_code((iree_status_code_t)iree_atomic_load(
          &operation->failure_code, iree_memory_order_relaxed));
  iree_hal_amd_gpu_sdma_operation_release_payload(operation);
  iree_hal_amd_gpu_profile_operation_t* profile =
      iree_hal_amd_gpu_sdma_operation_profile(operation);
  if (profile) {
    iree_hal_amd_gpu_profile_operation_record(queue->profile, profile,
                                              operation->accepted, 0);
    if (operation->accepted) {
      iree_hal_amd_gpu_profile_operation_complete(queue->profile, profile,
                                                  &profile->completion);
    }
  }
  for (iree_host_size_t i = 0;
       i < operation->signals.count && iree_status_is_ok(status); ++i) {
    iree_hal_semaphore_t* semaphore = operation->signals.semaphores[i];
    const uint64_t value = operation->signals.payload_values[i];
    status = iree_hal_amd_gpu_semaphore_isa(semaphore)
                 ? iree_hal_amd_gpu_semaphore_publish_completed(semaphore,
                                                                value, frontier)
                 : iree_hal_semaphore_signal(semaphore, value, frontier);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    iree_hal_semaphore_list_fail(operation->signals, iree_status_clone(status));
  }
  iree_hal_amd_gpu_signal_retire(operation->signal, iree_status_code(status));
  iree_hal_amd_gpu_sdma_operation_free(operation);
  if (callback.fn) {
    callback.fn(callback.user_data, status,
                completed.entry_count ? &completed : NULL, true);
  } else {
    iree_status_free(status);
  }
  // Completion may publish a public signal or invoke a callback that releases
  // the queue. Its membership join still sees this operation and source tail.
  iree_slim_mutex_lock(&queue->host_mutex);
  iree_atomic_fetch_sub(&queue->operation_count, 1, iree_memory_order_release);
  iree_hal_amd_gpu_sdma_wake_shutdown(queue);
  iree_slim_mutex_unlock(&queue->host_mutex);
}

void iree_hal_amd_gpu_sdma_operation_retain(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  if (operation) {
    iree_atomic_ref_count_inc(&operation->ref_count);
  }
}

void iree_hal_amd_gpu_sdma_operation_release(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  if (operation && iree_atomic_ref_count_dec(&operation->ref_count) == 1) {
    iree_hal_amd_gpu_sdma_operation_destroy(operation);
  }
}
