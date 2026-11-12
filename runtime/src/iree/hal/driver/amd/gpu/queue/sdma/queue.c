// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/queue.h"

#include "iree/base/cpu_data.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/execution.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/growth.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/state.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_sdma_queue_vtable;

void iree_hal_amd_gpu_sdma_wake_shutdown(iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (IREE_UNLIKELY(
          iree_atomic_load(&queue->closed, iree_memory_order_acquire)) &&
      !iree_hal_amd_gpu_context_wake_failure(queue->context)) {
    const amdf_status_t status =
        iree_hal_amd_gpu_context_signal(queue->context);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(queue->context, status,
                                      "SDMA shutdown notification");
    }
  }
}

void iree_hal_amd_gpu_sdma_host_begin(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  iree_atomic_fetch_add(&queue->host_owners, 1, iree_memory_order_relaxed);
}

void iree_hal_amd_gpu_sdma_host_end(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  iree_slim_mutex_lock(&queue->host_mutex);
  iree_atomic_fetch_sub(&queue->host_owners, 1, iree_memory_order_release);
  iree_hal_amd_gpu_sdma_wake_shutdown(queue);
  iree_slim_mutex_unlock(&queue->host_mutex);
}

iree_status_t iree_hal_amd_gpu_sdma_check_failure(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  return iree_status_from_code((iree_status_code_t)iree_atomic_load(
      &queue->failure_code, iree_memory_order_acquire));
}

iree_status_t iree_hal_amd_gpu_sdma_check_acceptance(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SDMA queue is closed");
  }
  return iree_hal_amd_gpu_sdma_check_failure(queue);
}

static void iree_hal_amd_gpu_sdma_profile(
    iree_hal_amd_gpu_sdma_operation_t* operation,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t count) {
  if (operation->profiled) {
    iree_hal_amd_gpu_profile_operation_initialize(
        operation->queue->profile, operation->queue->options.queue_ordinal,
        type, operation->wait_count, operation->signals.count, count,
        iree_hal_amd_gpu_sdma_operation_profile(operation));
  }
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals, 0, barriers, &operation, &payload));
  iree_hal_amd_gpu_sdma_profile(operation,
                                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, 0);
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_host_enter(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_retirement_callback_t callback,
    const iree_hal_queue_barriers_t* barriers,
    iree_host_size_t public_signal_count) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_failure(queue));
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, iree_hal_semaphore_list_empty(), 0, barriers, &operation,
      &payload));
  operation->callback = callback;
  iree_hal_amd_gpu_sdma_profile(operation,
                                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL, 1);
  if (operation->profiled) {
    iree_hal_amd_gpu_sdma_operation_profile(operation)->event.signal_count =
        public_signal_count;
  }
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_host_call(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t arguments[4], iree_hal_host_call_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  return iree_hal_amd_gpu_host_call_submit(&queue->host_executor, waits,
                                           signals, call, arguments, flags);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_execute(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals,
      iree_hal_amd_gpu_sdma_command_buffer_invocation_size(command_buffer),
      &barriers, &operation, &payload));
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE;
  iree_status_t status = iree_hal_amd_gpu_sdma_command_invocation_initialize(
      command_buffer, bindings, flags, payload, &operation->execute.invocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_operation_release(operation);
    return status;
  }
  operation->execute.storage =
      iree_hal_amd_gpu_sdma_command_buffer_storage(command_buffer);
  iree_hal_amd_gpu_sdma_profile(
      operation, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE,
      iree_hal_amd_gpu_sdma_command_buffer_operation_count(command_buffer));
  if (operation->profiled) {
    iree_hal_amd_gpu_sdma_operation_profile(operation)
        ->event.command_buffer_id =
        iree_hal_command_buffer_profile_id(command_buffer);
  }
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* transfers,
    const iree_hal_queue_barriers_t* barriers) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  iree_hal_amd_gpu_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_transfer_measure(count, transfers, &layout));
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals, layout.byte_length, barriers, &operation,
      &payload));
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER;
  iree_status_t status = iree_hal_amd_gpu_transfer_initialize(
      base->queue_family, IREE_HAL_MEMORY_SITE_QUEUE,
      (iree_hal_amd_gpu_transfer_staging_limits_t){
          queue->options.staging_capacity, queue->options.staging_capacity},
      count, transfers, layout, payload, &operation->transfer);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_operation_release(operation);
    return status;
  }
  iree_hal_amd_gpu_sdma_profile(operation,
                                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER,
                                operation->transfer.count);
  if (operation->profiled) {
    iree_hal_amd_gpu_transfer_profile(
        &operation->transfer,
        &iree_hal_amd_gpu_sdma_operation_profile(operation)->event);
  }
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_file_submit(
    iree_hal_queue_t* base,
    const iree_hal_amd_gpu_transfer_operation_t* transfer, const void* before,
    const void* after, iree_hal_amd_gpu_retirement_callback_t callback) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_failure(queue));
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      transfer ? sizeof(*transfer) : 0, &barriers, &operation, &payload));
  operation->callback = callback;
  operation->before =
      before ? *(const iree_hal_amd_gpu_sdma_barrier_t*)before : 0;
  operation->after = after ? *(const iree_hal_amd_gpu_sdma_barrier_t*)after : 0;
  if (transfer) {
    operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER;
    iree_hal_amd_gpu_transfer_initialize_operation(
        (iree_hal_amd_gpu_transfer_staging_limits_t){
            queue->options.staging_capacity, queue->options.staging_capacity},
        transfer, payload, &operation->transfer);
  }
  iree_hal_amd_gpu_sdma_profile(operation,
                                transfer
                                    ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER
                                    : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
                                transfer ? 1 : 0);
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static void iree_hal_amd_gpu_sdma_file_effects(
    iree_hal_queue_t* base, iree_hal_memory_effects_t effects, void* barrier) {
  *(iree_hal_amd_gpu_sdma_barrier_t*)barrier |=
      iree_hal_amd_gpu_sdma_barrier_effects(effects);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_read(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* source,
    uint64_t source_offset, iree_hal_buffer_t* target,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  const iree_hal_amd_gpu_sdma_barrier_t before =
      iree_hal_amd_gpu_sdma_barrier_prepare(barriers ? barriers->before : NULL);
  const iree_hal_amd_gpu_sdma_barrier_t after =
      iree_hal_amd_gpu_sdma_barrier_prepare(barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, source,
      IREE_HAL_MEMORY_ACCESS_READ, source_offset, target, target_offset, length,
      &before, &after);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_file_t* target,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  const iree_hal_amd_gpu_sdma_barrier_t before =
      iree_hal_amd_gpu_sdma_barrier_prepare(barriers ? barriers->before : NULL);
  const iree_hal_amd_gpu_sdma_barrier_t after =
      iree_hal_amd_gpu_sdma_barrier_prepare(barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, target,
      IREE_HAL_MEMORY_ACCESS_WRITE, target_offset, source, source_offset,
      length, &before, &after);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_timestamp(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  iree_hal_amd_gpu_buffer_reference_t reference;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_buffer_reference_initialize(
      base->queue_family, IREE_HAL_MEMORY_SITE_QUEUE, target, offset,
      &reference));
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals, 0, barriers, &operation, &payload));
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_TIMESTAMP;
  operation->timestamp = reference;
  iree_hal_buffer_retain(target);
  iree_hal_amd_gpu_sdma_profile(operation,
                                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TIMESTAMP, 1);
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals, 0, &barriers, &operation, &payload));
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_alloca(
      queue->options.allocations, pool, count, requests, out_buffers,
      &operation->allocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_operation_release(operation);
    return status;
  }
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION;
  iree_hal_amd_gpu_sdma_profile(
      operation, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA, count);
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_sdma_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_operation_create(
      queue, waits, signals, 0, &barriers, &operation, &payload));
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_dealloca(
      queue->options.allocations, count, buffers, &operation->allocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_operation_release(operation);
    return status;
  }
  operation->kind = IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION;
  iree_hal_amd_gpu_sdma_profile(
      operation, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA, count);
  iree_hal_amd_gpu_sdma_execution_submit(operation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_map(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_map(queue->options.mappings, base,
                                    queue->options.queue_ordinal, waits,
                                    signals, count, mappings, flags);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_flush(iree_hal_queue_t* base) {
  return iree_hal_amd_gpu_sdma_check_acceptance((void*)base);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA queues cannot dispatch");
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_dispatch(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA queues cannot dispatch");
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA queues do not expose public atomic operations");
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA queues do not expose public atomic operations");
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA queues do not expose public atomic operations");
}

static void iree_hal_amd_gpu_sdma_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  if (queue->release.fn && !queue->release.fn(queue->release.user_data, base)) {
    return;
  }
  iree_hal_amd_gpu_host_wait_set_deinitialize(&queue->waits);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_hal_amd_gpu_capture_pool_deinitialize(&queue->captures);
  iree_slim_mutex_deinitialize(&queue->host_mutex);
  iree_allocator_free(queue->host_allocator, queue->producer.allocation);
  iree_allocator_free_aligned(queue->host_allocator, queue);
}

static amdf_status_t iree_hal_amd_gpu_sdma_queue_shutdown(
    iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  const amdf_status_t status =
      iree_hal_amd_gpu_sdma_native_queue_free(queue->native);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->native = NULL;
  iree_hal_amd_gpu_byte_storage_free(queue->context, queue->staging);
  queue->staging = NULL;
  return AMDF_STATUS_OK;
}

static iree_status_t iree_hal_amd_gpu_sdma_prepare_signals(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (queue->options.barriers.submission ||
      iree_hal_amd_gpu_signal_pool_has_available(
          &queue->options.signals->pool)) {
    return iree_ok_status();
  }
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_signal_storage_prepare(
      IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY, queue->producer.count,
      queue->host_allocator, &storage));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      queue->context->api, queue->options.memory,
      (uint64_t)storage->capacity * queue->options.signal_layout.stride,
      queue->options.signal_layout.stride, queue->context->event_sink,
      &storage->memory, NULL);
  if (iree_status_is_ok(status)) {
    status = queue->options.signal_layout.initialize(
        queue->context, &queue->options.signals->pool, storage);
  }
  if (iree_status_is_ok(status)) {
    storage->next = queue->options.signals->storage;
    queue->options.signals->storage = storage;
  } else {
    iree_hal_amd_gpu_signal_storage_free(queue->context, storage);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_sdma_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_sdma_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  if (iree_cpu_architecture_host() != IREE_CPU_ARCHITECTURE_X86_64 ||
      !iree_atomic_int32_is_lock_free() || !iree_atomic_int64_is_lock_free()) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SDMA publication requires qualified x86-64 ordering");
  }
  if (options->staging_capacity < 64 ||
      (options->staging_capacity & (options->staging_capacity - 1))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SDMA staging capacity must be a power of two >= 64");
  }
  iree_host_size_t size = 0;
  IREE_RETURN_IF_ERROR(
      iree_host_size_checked_mul_add(sizeof(iree_hal_amd_gpu_sdma_queue_t),
                                     options->queue_count, 2 * sizeof(uint64_t),
                                     &size)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "SDMA queue allocation overflows"));
  iree_hal_amd_gpu_sdma_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, size, iree_alignof(iree_hal_amd_gpu_sdma_queue_t), 0,
      (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->options = *options;
  queue->features = context->families[options->family_ordinal].format_features;
  queue->encoding = iree_hal_amd_gpu_sdma_transfer_encoding(queue->features);
  queue->producer.count = options->queue_count;
  queue->producer.published = (uint64_t*)(queue + 1);
  queue->producer.pending = queue->producer.published + options->queue_count;
  queue->producer.published[options->queue_ordinal] = options->initial_epoch;
  iree_atomic_store(&queue->completed_epoch, options->initial_epoch,
                    iree_memory_order_relaxed);
  iree_hal_queue_initialize(family, params, &iree_hal_amd_gpu_sdma_queue_vtable,
                            &queue->base);
  iree_slim_mutex_initialize(&queue->host_mutex);
  iree_async_semaphore_wait_context_initialize(&queue->wait_context);
  iree_hal_amd_gpu_host_wait_set_initialize(&queue->wait_context,
                                            &queue->waits);
  iree_hal_amd_gpu_capture_pool_initialize(options->capture_block_size,
                                           host_allocator, &queue->captures);
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_sdma_host_begin,
      .end = iree_hal_amd_gpu_sdma_host_end,
      .enter = iree_hal_amd_gpu_sdma_host_enter};
  queue->file_executor = (iree_hal_amd_gpu_file_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_sdma_host_begin,
      .end = iree_hal_amd_gpu_sdma_host_end,
      .staging = options->file_staging,
      .waits = &queue->wait_context,
      .queue_ordinal = options->queue_ordinal,
      .native_family_ordinal = options->family_ordinal,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(iree_hal_amd_gpu_sdma_barrier_t),
      .add_effects = iree_hal_amd_gpu_sdma_file_effects,
      .submit = iree_hal_amd_gpu_sdma_file_submit};
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_sdma_prepare_signals(queue);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_byte_storage_allocate(
        context, options->memory,
        (uint64_t)options->staging_capacity *
            IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY,
        0, host_allocator, &queue->staging);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_sdma_native_queue_allocate(
        context, options->family_ordinal, options->publication_mode,
        options->memory, options->publication, options->source_ordinal,
        iree_hal_amd_gpu_user_queue_native_priority(params->priority),
        (amdf_queue_placement_t)params->placement, host_allocator,
        &queue->native);
  }
  if (iree_status_is_ok(status)) {
    queue->base.placement =
        iree_hal_amd_gpu_sdma_native_queue_placement(queue->native);
    *out_queue = &queue->base;
  } else {
    const amdf_status_t cleanup =
        iree_hal_amd_gpu_sdma_queue_shutdown(&queue->base);
    if (amdf_status_is_ok(cleanup)) {
      iree_hal_queue_release(&queue->base);
    } else {
      iree_atomic_store(&context->cleanup_blocked, 1,
                        iree_memory_order_relaxed);
      status = iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(cleanup, "SDMA construction rollback"));
    }
  }
  return status;
}

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_sdma_queue_vtable = {
    .destroy = iree_hal_amd_gpu_sdma_queue_destroy,
    .barrier = iree_hal_amd_gpu_sdma_queue_barrier,
    .execute = iree_hal_amd_gpu_sdma_queue_execute,
    .host_call = iree_hal_amd_gpu_sdma_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_gpu_sdma_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_gpu_sdma_queue_dispatch,
    .atomic_wait = iree_hal_amd_gpu_sdma_queue_atomic_wait,
    .atomic_store = iree_hal_amd_gpu_sdma_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_gpu_sdma_queue_atomic_rmw,
    .timestamp = iree_hal_amd_gpu_sdma_queue_timestamp,
    .flush = iree_hal_amd_gpu_sdma_queue_flush,
    .alloca = iree_hal_amd_gpu_sdma_queue_alloca,
    .dealloca = iree_hal_amd_gpu_sdma_queue_dealloca,
    .map = iree_hal_amd_gpu_sdma_queue_map,
    .transfer = iree_hal_amd_gpu_sdma_queue_transfer,
    .read = iree_hal_amd_gpu_sdma_queue_read,
    .write = iree_hal_amd_gpu_sdma_queue_write,
};

static void iree_hal_amd_gpu_sdma_queue_close(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  iree_atomic_store(&queue->closed, 1, iree_memory_order_release);
  iree_hal_amd_gpu_sdma_wake_shutdown(queue);
}

static void iree_hal_amd_gpu_sdma_queue_notify_failure(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  int32_t expected = 0;
  iree_atomic_compare_exchange_strong(
      &queue->failure_code, &expected, IREE_STATUS_ABORTED,
      iree_memory_order_release, iree_memory_order_relaxed);
  iree_hal_amd_gpu_host_wait_set_cancel(&queue->waits);
  iree_hal_amd_gpu_sdma_native_queue_fail(
      queue->native, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
}

static bool iree_hal_amd_gpu_sdma_queue_is_host_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  if (!iree_hal_amd_gpu_sdma_native_queue_is_host_idle(queue->native)) {
    return false;
  }
  iree_slim_mutex_lock(&queue->host_mutex);
  const bool idle =
      !iree_atomic_load(&queue->host_owners, iree_memory_order_acquire);
  iree_slim_mutex_unlock(&queue->host_mutex);
  return idle;
}

static bool iree_hal_amd_gpu_sdma_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  return !iree_atomic_load(&queue->operation_count,
                           iree_memory_order_acquire) &&
         iree_hal_amd_gpu_sdma_native_queue_is_idle(queue->native) &&
         iree_hal_amd_gpu_sdma_queue_is_host_idle(base);
}

static bool iree_hal_amd_gpu_sdma_queue_try_inactivate(
    iree_hal_queue_t* base, amdf_status_t* out_status) {
  return iree_hal_amd_gpu_sdma_native_queue_try_inactivate(
      ((iree_hal_amd_gpu_sdma_queue_t*)base)->native, out_status);
}

static bool iree_hal_amd_gpu_sdma_queue_retire_failed(iree_hal_queue_t* base,
                                                      iree_status_code_t code) {
  return iree_hal_amd_gpu_sdma_native_queue_retire_failed(
      ((iree_hal_amd_gpu_sdma_queue_t*)base)->native);
}

static void iree_hal_amd_gpu_sdma_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  queue->frontier.tracker = tracker;
  queue->frontier.axis = axis;
}

static uint64_t iree_hal_amd_gpu_sdma_queue_completed_epoch(
    iree_hal_queue_t* base) {
  return (uint64_t)iree_atomic_load(
      &((iree_hal_amd_gpu_sdma_queue_t*)base)->completed_epoch,
      iree_memory_order_acquire);
}

static void iree_hal_amd_gpu_sdma_queue_set_release(
    iree_hal_queue_t* base, bool (*release)(void*, iree_hal_queue_t*),
    void* user_data) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  queue->release.fn = release;
  queue->release.user_data = user_data;
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_set_profile(
    iree_hal_queue_t* base, iree_hal_amd_gpu_profile_t* profile) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_register_queue(
      profile, queue->options.queue_ordinal));
  queue->profile = profile;
  queue->file_executor.profile = profile;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_sdma_trim_process(void* user_data,
                                               amdf_status_t status) {
  iree_hal_amd_gpu_sdma_queue_t* queue = user_data;
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
  iree_hal_amd_gpu_signal_cache_trim(queue->context, 1, queue->options.signals);
  iree_atomic_store(&queue->trim.pending, 0, iree_memory_order_release);
}

static void iree_hal_amd_gpu_sdma_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  if (!iree_atomic_exchange(&queue->trim.pending, 1,
                            iree_memory_order_acq_rel)) {
    queue->trim.metadata = (iree_hal_amd_gpu_sdma_metadata_t){
        .process = iree_hal_amd_gpu_sdma_trim_process, .user_data = queue};
    iree_hal_amd_gpu_sdma_native_queue_defer_metadata(queue->native,
                                                      &queue->trim.metadata);
  }
}

typedef struct iree_hal_amd_gpu_sdma_frontiers_t {
  // Metadata-only publication entry, borrowed until the device notification.
  iree_hal_amd_gpu_sdma_metadata_t metadata;
  // Exact queue borrowed by the serialized membership caller.
  iree_hal_amd_gpu_sdma_queue_t* queue;
  // Shared caller-owned vector/block expansion state.
  iree_hal_amd_gpu_frontier_request_t request;
} iree_hal_amd_gpu_sdma_frontiers_t;

static void iree_hal_amd_gpu_sdma_frontiers_process(void* user_data,
                                                    amdf_status_t status) {
  iree_hal_amd_gpu_sdma_frontiers_t* entry = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = entry->queue;
  iree_hal_amd_gpu_frontier_request_t* request = &entry->request;
  if (request->phase == IREE_HAL_AMD_GPU_FRONTIER_BEGIN && request->vectors) {
    uint64_t* vectors = request->vectors;
    memcpy(vectors, queue->producer.published,
           queue->producer.count * sizeof(uint64_t));
    request->vectors = queue->producer.allocation;
    queue->producer.allocation = vectors;
    queue->producer.published = vectors;
    queue->producer.pending = vectors + request->queue_count;
    queue->producer.count = request->queue_count;
  }
  iree_hal_amd_gpu_frontier_request_process(
      queue->options.signals,
      iree_hal_amd_gpu_sdma_growth_pending_signals(queue), request);
}

static void iree_hal_amd_gpu_sdma_frontiers_enqueue(void* user_data) {
  iree_hal_amd_gpu_sdma_frontiers_t* entry = user_data;
  entry->metadata = (iree_hal_amd_gpu_sdma_metadata_t){
      .process = iree_hal_amd_gpu_sdma_frontiers_process, .user_data = entry};
  iree_hal_amd_gpu_sdma_native_queue_defer_metadata(entry->queue->native,
                                                    &entry->metadata);
}

static iree_status_t iree_hal_amd_gpu_sdma_queue_reserve_frontiers(
    iree_hal_queue_t* base, uint32_t count, iree_notification_t* notification) {
  iree_hal_amd_gpu_sdma_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_sdma_frontiers_t entry = {
      .queue = queue,
      .request = {.queue_count = iree_max(count, queue->producer.count),
                  .notification = notification}};
  if (count > queue->producer.count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, count, 2 * sizeof(uint64_t),
        (void**)&entry.request.vectors));
  }
  return iree_hal_amd_gpu_frontier_request_run(
      &entry.request, queue->host_allocator,
      iree_hal_amd_gpu_sdma_frontiers_enqueue, &entry);
}

static bool iree_hal_amd_gpu_sdma_queue_publish(void* user_data) {
  iree_hal_amd_gpu_sdma_queue_t* queue = user_data;
  if (iree_hal_amd_gpu_signal_cache_take_ready(queue->options.signals)) {
    iree_hal_amd_gpu_sdma_growth_signals_available(queue);
  }
  const iree_hal_amd_gpu_publication_source_t source =
      iree_hal_amd_gpu_sdma_native_queue_publication_source(queue->native);
  const bool ready = source.publish(source.user_data);
  iree_hal_amd_gpu_sdma_wake_shutdown(queue);
  return ready;
}

static iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_sdma_queue_publication_source(iree_hal_queue_t* base) {
  return (iree_hal_amd_gpu_publication_source_t){
      iree_hal_amd_gpu_sdma_queue_publish, base};
}

static iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_sdma_queue_completion_source(iree_hal_queue_t* base) {
  return iree_hal_amd_gpu_sdma_native_queue_completion_source(
      ((iree_hal_amd_gpu_sdma_queue_t*)base)->native);
}

const iree_hal_amd_gpu_queue_lifecycle_t iree_hal_amd_gpu_sdma_queue_lifecycle =
    {
        .close = iree_hal_amd_gpu_sdma_queue_close,
        .is_idle = iree_hal_amd_gpu_sdma_queue_is_idle,
        .is_host_idle = iree_hal_amd_gpu_sdma_queue_is_host_idle,
        .try_inactivate = iree_hal_amd_gpu_sdma_queue_try_inactivate,
        .retire_failed = iree_hal_amd_gpu_sdma_queue_retire_failed,
        .shutdown = iree_hal_amd_gpu_sdma_queue_shutdown,
        .trim = iree_hal_amd_gpu_sdma_queue_trim,
        .set_profile = iree_hal_amd_gpu_sdma_queue_set_profile,
        .notify_failure = iree_hal_amd_gpu_sdma_queue_notify_failure,
        .assign_frontier = iree_hal_amd_gpu_sdma_queue_assign_frontier,
        .reserve_frontiers = iree_hal_amd_gpu_sdma_queue_reserve_frontiers,
        .set_release = iree_hal_amd_gpu_sdma_queue_set_release,
        .completed_epoch = iree_hal_amd_gpu_sdma_queue_completed_epoch,
        .publication_source = iree_hal_amd_gpu_sdma_queue_publication_source,
        .completion_source = iree_hal_amd_gpu_sdma_queue_completion_source,
};
