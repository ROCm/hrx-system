// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/host_call.h"

#include "iree/async/semaphore.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"

enum {
  IREE_HAL_AMD_GPU_HOST_CALL_ENTRY = 1,
  IREE_HAL_AMD_GPU_HOST_CALL_COMPLETE = 2,
  IREE_HAL_AMD_GPU_HOST_CALL_RETURN = 3,
};

typedef struct iree_hal_amd_gpu_host_call_t {
  // Service continuation for native mappings or the return cache boundary.
  iree_hal_memory_maintenance_entry_t entry;
  // Captured queue services, dominated by device.
  iree_hal_amd_gpu_host_call_executor_t executor;
  // Retained parent through the final callback and native boundary.
  iree_hal_device_t* device;
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
  // Mapping batches execute on service and retain buffers through return.
  bool mapping;
  // Owned private timeline for entry, callback completion, and return.
  iree_hal_semaphore_t* semaphore;
  // One active point, owned by its callback after dispatch.
  iree_async_semaphore_timepoint_t timepoint;
} iree_hal_amd_gpu_host_call_t;

static void iree_hal_amd_gpu_host_call_free(
    iree_hal_amd_gpu_host_call_t* operation) {
  iree_hal_device_t* device = operation->device;
  iree_hal_resource_release(operation->call.resource);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_semaphore_release(operation->semaphore);
  iree_hal_amd_gpu_capture_pool_release(
      operation->executor.captures,
      iree_make_byte_span(operation, operation->allocation_size));
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_host_call_finish(
    iree_hal_amd_gpu_host_call_t* operation, iree_status_t status) {
  // Mapping buffers remain protected through the final cache boundary. Public
  // completion permits immediately destroying their source pools.
  iree_hal_resource_release(operation->call.resource);
  operation->call.resource = NULL;
  if (iree_status_is_ok(status)) {
    // Both native boundaries and all callback effects have actually completed;
    // no outstanding device frontier needs to travel with this host signal.
    status = iree_hal_semaphore_list_signal(operation->signals, NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(operation->executor.context,
                                           operation->device, status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_host_call_free(operation);
}

static void iree_hal_amd_gpu_host_call_retire(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amd_gpu_host_call_finish(user_data, status);
}

static iree_status_t iree_hal_amd_gpu_host_call_boundary(
    iree_hal_amd_gpu_host_call_t* operation, iree_hal_semaphore_list_t waits,
    uint64_t value, iree_async_semaphore_timepoint_fn_t callback) {
  operation->timepoint = (iree_async_semaphore_timepoint_t){
      .callback = callback,
      .user_data = operation,
  };
  IREE_RETURN_IF_ERROR(iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)operation->semaphore, value,
      &operation->timepoint));
  const bool entering = value == IREE_HAL_AMD_GPU_HOST_CALL_ENTRY;
  const iree_hal_execution_stage_t device_stages =
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER;
  const iree_hal_barrier_t barrier = {
      .source_stage_mask =
          entering ? device_stages : IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask =
          entering ? IREE_HAL_EXECUTION_STAGE_HOST : device_stages,
      .flags = entering ? IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE
                        : IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_barrier_list_t effects = {
      .count =
          iree_any_bit_set(operation->flags, IREE_HAL_HOST_CALL_FLAG_RELAXED)
              ? 0
              : 1,
      .values = &barrier,
  };
  const iree_hal_queue_barriers_t barriers = {
      .before = &effects,
      .after = &empty,
  };
  const iree_hal_semaphore_list_t completion = {
      .count = 1,
      .semaphores = &operation->semaphore,
      .payload_values = &value,
  };
  // Only this native submission can satisfy the enrolled point. A rejected
  // submission has no producer, so cancellation rejoins storage synchronously.
  iree_status_t status = iree_hal_queue_barrier(
      operation->executor.queue, waits, completion, &barriers, 0);
  if (!iree_status_is_ok(status)) {
    const bool cancelled = iree_async_semaphore_cancel_timepoint(
        (iree_async_semaphore_t*)operation->semaphore, &operation->timepoint);
    IREE_ASSERT(cancelled);
  }
  return status;
}

static void iree_hal_amd_gpu_host_call_return(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_host_call_t* operation = (void*)entry;
  iree_status_t status = iree_hal_amd_gpu_host_call_boundary(
      operation, iree_hal_semaphore_list_empty(),
      IREE_HAL_AMD_GPU_HOST_CALL_RETURN, iree_hal_amd_gpu_host_call_retire);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_host_call_finish(operation, status);
  }
}

static void iree_hal_amd_gpu_host_call_complete(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amd_gpu_host_call_t* operation = user_data;
  if (!iree_status_is_ok(status) ||
      iree_any_bit_set(operation->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
                                             IREE_HAL_HOST_CALL_FLAG_RELAXED)) {
    iree_hal_amd_gpu_host_call_finish(operation, status);
    return;
  }
  operation->entry.fn = iree_hal_amd_gpu_host_call_return;
  iree_hal_memory_maintenance_enqueue(operation->executor.service,
                                      &operation->entry);
}

static void iree_hal_amd_gpu_host_call_invoke(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_host_call_t* operation = (void*)entry;
  IREE_TRACE_ZONE_BEGIN(z0);
  const bool non_blocking =
      iree_any_bit_set(operation->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (non_blocking) {
    iree_status_t status =
        iree_hal_semaphore_list_signal(operation->signals, NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_host_call_finish(operation, status);
      IREE_TRACE_ZONE_END(z0);
      return;
    }
    iree_hal_semaphore_list_release(operation->signals);
    operation->signals = iree_hal_semaphore_list_empty();
  }
  uint64_t value = IREE_HAL_AMD_GPU_HOST_CALL_COMPLETE;
  iree_hal_host_call_context_t context = {
      .queue = operation->executor.queue,
      .signal_semaphore_list =
          {
              .count = non_blocking ? 0 : 1,
              .semaphores = &operation->semaphore,
              .payload_values = &value,
          },
  };
  iree_status_t status = operation->call.fn(operation->call.user_data,
                                            operation->arguments, &context);
  if (!operation->mapping) {
    iree_hal_resource_release(operation->call.resource);
    operation->call.resource = NULL;
  }
  if (iree_status_is_deferred(status)) {
    iree_status_free(status);
    if (non_blocking) {
      iree_hal_amd_gpu_host_call_free(operation);
    } else {
      // The callback may have already signaled, but no completion may release
      // this capture until the invocation and optional resource release end.
      operation->timepoint = (iree_async_semaphore_timepoint_t){
          .callback = iree_hal_amd_gpu_host_call_complete,
          .user_data = operation,
      };
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)operation->semaphore, value,
          &operation->timepoint);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_host_call_finish(operation, status);
      }
    }
  } else {
    iree_hal_amd_gpu_host_call_complete(operation, NULL, status);
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amd_gpu_host_call_enter(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amd_gpu_host_call_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_host_call_finish(operation, status);
  } else if (operation->mapping) {
    iree_hal_memory_maintenance_enqueue(operation->executor.service,
                                        &operation->entry);
  } else {
    // Success comes only from actual native barrier retirement, on the sole
    // completion worker. A submission hint cannot invoke user code.
    iree_hal_amd_gpu_host_call_invoke(&operation->entry);
  }
}

static iree_status_t iree_hal_amd_gpu_host_call_submit_on(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_host_call_t call, const uint64_t arguments[4], bool mapping,
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
      .entry = {.fn = iree_hal_amd_gpu_host_call_invoke},
      .executor = *executor,
      .device = executor->queue->queue_family->device,
      .allocation_size = storage.data_length,
      .signals =
          {
              .count = signals.count,
              .semaphores = (void*)(storage.data + semaphores_offset),
              .payload_values = (void*)(storage.data + values_offset),
          },
      .call = call,
      .flags = flags,
      .mapping = mapping,
  };
  memcpy(operation->arguments, arguments, sizeof(operation->arguments));
  iree_hal_device_retain(operation->device);
  iree_hal_resource_retain(call.resource);
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  iree_status_t status = iree_hal_semaphore_create(
      operation->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &operation->semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_host_call_boundary(
        operation, waits, IREE_HAL_AMD_GPU_HOST_CALL_ENTRY,
        iree_hal_amd_gpu_host_call_enter);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_host_call_free(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_host_call_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_host_call_t call, const uint64_t arguments[4],
    iree_hal_host_call_flags_t flags) {
  return iree_hal_amd_gpu_host_call_submit_on(executor, waits, signals, call,
                                              arguments, false, flags);
}

iree_status_t iree_hal_amd_gpu_memory_mapping_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t mapping_count, const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_host_call_t call;
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_prepare(
      executor->queue, mapping_count, mappings, flags,
      executor->captures->host_allocator, &call));
  const uint64_t arguments[4] = {0};
  iree_status_t status = iree_hal_amd_gpu_host_call_submit_on(
      executor, waits, signals, call, arguments, true,
      IREE_HAL_HOST_CALL_FLAG_NONE);
  iree_hal_resource_release(call.resource);
  return status;
}
