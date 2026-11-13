// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/host_call.h"

#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"

typedef struct iree_hal_amd_gpu_host_call_t {
  // Independent submitting-call and native-entry ownership.
  iree_atomic_ref_count_t ref_count;
  // Borrowed queue-owned services, live through invocation retirement.
  const iree_hal_amd_gpu_host_call_executor_t* executor;
  // Owned extent returned to the reusable capture pool.
  iree_host_size_t allocation_size;
  // Captured public completion list with retained semaphore references.
  iree_hal_semaphore_list_t signals;
  // Captured callback and retained optional state resource.
  iree_hal_host_call_t call;
  // Copied callback arguments.
  uint64_t arguments[4];
  // Admitted public ordering and cache hints.
  iree_hal_host_call_flags_t flags;
} iree_hal_amd_gpu_host_call_t;

static void iree_hal_amd_gpu_host_call_join(
    iree_hal_amd_gpu_host_call_t* operation) {
  if (iree_atomic_ref_count_dec(&operation->ref_count) != 1) {
    return;
  }
  const iree_hal_amd_gpu_host_call_executor_t* executor = operation->executor;
  iree_hal_resource_release(operation->call.resource);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_gpu_capture_pool_release(
      executor->captures,
      iree_make_byte_span(operation, operation->allocation_size));
  executor->end(executor->queue);
}

static void iree_hal_amd_gpu_host_call_finish(
    iree_hal_amd_gpu_host_call_t* operation, iree_status_t status,
    const iree_async_single_frontier_t* frontier) {
  iree_hal_resource_release(operation->call.resource);
  operation->call.resource = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        frontier ? iree_async_single_frontier_as_const_frontier(frontier)
                 : NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(
        operation->executor->context,
        operation->executor->queue->queue_family->device, status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_host_call_join(operation);
}

static void iree_hal_amd_gpu_host_call_enter(
    void* user_data, iree_status_t status,
    const iree_async_single_frontier_t* frontier, bool retired) {
  iree_hal_amd_gpu_host_call_t* operation = user_data;
  if (!retired) {
    // The native owner retains the first failure and delivers it again after
    // all accepted uses end. Early notification neither invokes nor completes
    // the public host call.
    iree_status_free(status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_host_call_finish(operation, status, frontier);
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  const bool non_blocking =
      iree_any_bit_set(operation->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (non_blocking) {
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        frontier ? iree_async_single_frontier_as_const_frontier(frontier)
                 : NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_host_call_finish(operation, status, frontier);
      IREE_TRACE_ZONE_END(z0);
      return;
    }
    iree_hal_semaphore_list_release(operation->signals);
    operation->signals = iree_hal_semaphore_list_empty();
  }
  iree_hal_host_call_context_t context = {
      .queue = operation->executor->queue,
      .signal_semaphore_list = operation->signals,
  };
  status = operation->call.fn(operation->call.user_data, operation->arguments,
                              &context);
  if (iree_status_is_deferred(status)) {
    // The callback's asynchronous work owns the original output points. Its
    // own queue operations carry their visibility boundaries and provenance.
    // Retain the invocation until fn returns even if those outputs have already
    // completed; there is no second completion or return-to-device operation.
    iree_status_free(status);
    iree_hal_amd_gpu_host_call_join(operation);
  } else {
    iree_hal_amd_gpu_host_call_finish(operation, status, frontier);
  }
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amd_gpu_host_call_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_host_call_t call, const uint64_t arguments[4],
    iree_hal_host_call_flags_t flags) {
  iree_host_size_t total_size = 0;
  iree_host_size_t values_offset = 0, semaphores_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_host_call_t), &total_size,
      IREE_STRUCT_FIELD(signals.count, uint64_t, &values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*,
                        &semaphores_offset)));
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      executor->captures, total_size, &storage));
  iree_hal_amd_gpu_host_call_t* operation = (void*)storage.data;
  *operation = (iree_hal_amd_gpu_host_call_t){
      .executor = executor,
      .allocation_size = storage.data_length,
      .signals =
          {
              .count = signals.count,
              .semaphores = (void*)(storage.data + semaphores_offset),
              .payload_values = (void*)(storage.data + values_offset),
          },
      .call = call,
      .flags = flags,
  };
  executor->begin(executor->queue);
  iree_atomic_ref_count_init(&operation->ref_count);
  memcpy(operation->arguments, arguments, sizeof(operation->arguments));
  iree_hal_resource_retain(call.resource);
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                           IREE_HAL_EXECUTION_STAGE_DISPATCH |
                           IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_barrier_list_t effects = {
      .count = iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_RELAXED) ? 0 : 1,
      .values = &barrier,
  };
  const iree_hal_queue_barriers_t barriers = {&effects, &empty};
  // Native retirement can invoke the callback before entry submission returns.
  // The submitting caller independently owns its capture and borrowed inputs.
  iree_atomic_ref_count_inc(&operation->ref_count);
  iree_status_t status =
      executor->enter(executor->queue, waits,
                      (iree_hal_amd_gpu_retirement_callback_t){
                          iree_hal_amd_gpu_host_call_enter, operation},
                      &barriers, signals.count);
  if (!iree_status_is_ok(status)) {
    // Rejection accepted no native callback ownership.
    iree_hal_amd_gpu_host_call_join(operation);
  }
  iree_hal_amd_gpu_host_call_join(operation);
  return status;
}
