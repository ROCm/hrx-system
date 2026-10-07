// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/execution.h"

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/memory/tlsf_pool.h"

typedef struct loom_serve_execution_timeline_t {
  // Retained timeline identifying accepted operations on this branch.
  iree_hal_semaphore_t* semaphore;
  // Last accepted submission, advanced only after successful queue submission.
  uint64_t submitted_value;
} loom_serve_execution_timeline_t;

struct loom_serve_execution_t {
  // Intrusive ownership shared by the host and native VM module.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for this object.
  iree_allocator_t host_allocator;
  // Retained owner of the queues and semaphore domain.
  iree_hal_device_t* device;
  // Retained exact queue for command execution.
  iree_hal_queue_t* dispatch_queue;
  // Retained exact queue for host transfers.
  iree_hal_queue_t* transfer_queue;
  // Ordered commands and input transfers owning shared model scratch.
  loom_serve_execution_timeline_t work;
  // Ordered downloads consuming produced results without gating model work.
  loom_serve_execution_timeline_t feedback;
  // Protects timeline reservation through queue acceptance, never device waits.
  iree_slim_mutex_t mutex;
  // Shared private command storage, independent of model residency.
  struct {
    // Owned HAL pool, created on first use and retained across invocations.
    iree_hal_pool_t* pool;
    // Slab bytes, rounded to a searchable TLSF size class.
    iree_device_size_t length;
    // Largest registered base alignment, in bytes.
    iree_device_size_t alignment;
    // Owned roots awaiting the final caller/queue release, in capture order.
    iree_hal_buffer_t** roots;
    // Capacity of the reusable root ring.
    iree_host_size_t capacity;
    // First occupied ring slot.
    iree_host_size_t head;
    // Number of occupied ring slots.
    iree_host_size_t count;
  } workspace;
};

// The registry owns one reference; private roots are not published to models.
// A unique root has no remaining caller or queue users, including failed work
// whose semaphore was failed before its captured resources were released.
// Reaping never waits on the submission path and reuses the host ring storage.
static void execution_reap_workspace(loom_serve_execution_t* execution) {
  while (execution->workspace.count) {
    iree_hal_buffer_t* root =
        execution->workspace.roots[execution->workspace.head];
    if (iree_atomic_ref_count_load(&((iree_hal_resource_t*)root)->ref_count) !=
        1) {
      break;
    }
    iree_hal_buffer_release(root);
    execution->workspace.head =
        (execution->workspace.head + 1) % execution->workspace.capacity;
    --execution->workspace.count;
  }
}

static iree_status_t execution_reserve_workspace_root(
    loom_serve_execution_t* execution) {
  execution_reap_workspace(execution);
  if (execution->workspace.count < execution->workspace.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t capacity =
      iree_max(16, execution->workspace.capacity * 2);
  iree_hal_buffer_t** roots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      execution->host_allocator, capacity, sizeof(*roots), (void**)&roots));
  for (iree_host_size_t i = 0; i < execution->workspace.count; ++i) {
    roots[i] = execution->workspace.roots[(execution->workspace.head + i) %
                                          execution->workspace.capacity];
  }
  iree_allocator_free(execution->host_allocator, execution->workspace.roots);
  execution->workspace.roots = roots;
  execution->workspace.capacity = capacity;
  execution->workspace.head = 0;
  return iree_ok_status();
}

void loom_serve_execution_retain(loom_serve_execution_t* execution) {
  iree_atomic_ref_count_inc(&execution->ref_count);
}

void loom_serve_execution_release(loom_serve_execution_t* execution) {
  if (execution && iree_atomic_ref_count_dec(&execution->ref_count) == 1) {
    IREE_ASSERT_EQ(execution->workspace.count, 0);
    iree_hal_pool_release(execution->workspace.pool);
    iree_allocator_free(execution->host_allocator, execution->workspace.roots);
    iree_slim_mutex_deinitialize(&execution->mutex);
    iree_hal_semaphore_release(execution->feedback.semaphore);
    iree_hal_semaphore_release(execution->work.semaphore);
    iree_hal_queue_release(execution->transfer_queue);
    iree_hal_queue_release(execution->dispatch_queue);
    iree_hal_device_release(execution->device);
    iree_allocator_free(execution->host_allocator, execution);
  }
}

iree_status_t loom_serve_execution_reserve_workspace(
    loom_serve_execution_t* execution, iree_device_size_t length,
    iree_device_size_t alignment) {
  if (!length || (length <= execution->workspace.length &&
                  alignment <= execution->workspace.alignment)) {
    return iree_ok_status();
  }
  length = iree_max(length, execution->workspace.length);
  alignment = iree_max(iree_max(alignment, execution->workspace.alignment),
                       IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT);
  // A slab inside a TLSF size class can be skipped by a maximum-size search,
  // which rounds up to the next class. Give backing a searchable geometry;
  // individual command requests retain their exact compiler-reflected size.
  const uint32_t first_level = 63 - iree_math_count_leading_zeros_u64(length);
  const uint32_t shift = first_level > IREE_HAL_MEMORY_TLSF_SL_LOG2
                             ? first_level - IREE_HAL_MEMORY_TLSF_SL_LOG2
                             : 0;
  const iree_device_size_t quantum =
      iree_max((iree_device_size_t)1 << shift, alignment);
  if (length > IREE_DEVICE_SIZE_MAX - (quantum - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workspace slab size overflows device range");
  }
  length = iree_device_align(length, quantum);
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(execution));
  iree_hal_pool_release(execution->workspace.pool);
  execution->workspace.pool = NULL;
  execution->workspace.length = length;
  execution->workspace.alignment = alignment;
  return iree_ok_status();
}

// TLSF creates its first physical slab eagerly. Defer that cold acquisition
// until execution actually needs workspace rather than model registration.
static iree_status_t execution_prepare_workspace_pool(
    loom_serve_execution_t* execution) {
  if (execution->workspace.pool) {
    return iree_ok_status();
  }
  iree_hal_queue_pool_backend_t backend;
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      execution->device, iree_hal_queue_family(execution->dispatch_queue),
      &backend));
  const iree_hal_tlsf_pool_options_t options = {
      .tlsf_options = {.range_length = execution->workspace.length,
                       .alignment = execution->workspace.alignment},
      .asan = backend.asan,
      .trace_name = IREE_SVL("loom_serve_workspace"),
  };
  return iree_hal_tlsf_pool_create(
      options, backend.slab_provider, backend.notification, backend.epoch_query,
      execution->host_allocator, &execution->workspace.pool);
}

iree_status_t loom_serve_execution_alloca(
    loom_serve_execution_t* execution,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_slim_mutex_lock(&execution->mutex);
  iree_status_t status = execution_reserve_workspace_root(execution);
  if (iree_status_is_ok(status)) {
    status = execution_prepare_workspace_pool(execution);
  }
  if (iree_status_is_ok(status) &&
      execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "execution timeline is exhausted");
  }
  uint64_t next_value = execution->work.submitted_value + 1;
  const iree_hal_semaphore_list_t waits = {1, &execution->work.semaphore,
                                           &execution->work.submitted_value};
  const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                             &next_value};
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_alloca(execution->dispatch_queue, waits, signals,
                                   execution->workspace.pool, 1, request,
                                   out_buffer);
  }
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    const iree_host_size_t tail =
        (execution->workspace.head + execution->workspace.count++) %
        execution->workspace.capacity;
    execution->workspace.roots[tail] = *out_buffer;
    iree_hal_buffer_retain(*out_buffer);
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_serve_execution_dealloca(loom_serve_execution_t* execution,
                                            iree_hal_buffer_t* buffer,
                                            uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    iree_slim_mutex_unlock(&execution->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "execution timeline is exhausted");
  }
  uint64_t next_value = execution->work.submitted_value + 1;
  const iree_hal_semaphore_list_t waits = {1, &execution->work.semaphore,
                                           &execution->work.submitted_value};
  const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                             &next_value};
  iree_status_t status = iree_hal_queue_dealloca(execution->dispatch_queue,
                                                 waits, signals, 1, &buffer);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_hal_pool_stats_t loom_serve_execution_workspace_statistics(
    const loom_serve_execution_t* execution) {
  iree_hal_pool_stats_t statistics = {0};
  if (execution->workspace.pool) {
    iree_hal_pool_query_stats(execution->workspace.pool, &statistics);
  }
  return statistics;
}

iree_status_t loom_serve_execution_trim_workspace(
    loom_serve_execution_t* execution) {
  return execution->workspace.pool
             ? iree_hal_pool_trim(execution->workspace.pool)
             : iree_ok_status();
}

iree_status_t loom_serve_execution_create(
    iree_hal_queue_t* dispatch_queue, iree_hal_queue_t* transfer_queue,
    iree_allocator_t host_allocator, loom_serve_execution_t** out_execution) {
  iree_hal_device_t* device =
      iree_hal_queue_family_device(iree_hal_queue_family(dispatch_queue));
  if (device !=
      iree_hal_queue_family_device(iree_hal_queue_family(transfer_queue))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "execution queues must belong to the same device");
  }
  loom_serve_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  iree_atomic_ref_count_init(&execution->ref_count);
  iree_slim_mutex_initialize(&execution->mutex);
  execution->host_allocator = host_allocator;
  execution->device = device;
  iree_hal_device_retain(device);
  execution->dispatch_queue = dispatch_queue;
  iree_hal_queue_retain(dispatch_queue);
  execution->transfer_queue = transfer_queue;
  iree_hal_queue_retain(transfer_queue);
  iree_status_t status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution->work.semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution->feedback.semaphore);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    loom_serve_execution_release(execution);
  }
  return status;
}

iree_status_t loom_serve_execution_execute(
    loom_serve_execution_t* execution,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    iree_slim_mutex_unlock(&execution->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "execution timeline is exhausted");
  }
  uint64_t next_value = execution->work.submitted_value + 1;
  const iree_hal_semaphore_list_t waits = {1, &execution->work.semaphore,
                                           &execution->work.submitted_value};
  const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                             &next_value};
  iree_status_t status = iree_hal_queue_execute(
      execution->dispatch_queue, waits, signals, command_buffer, bindings,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_serve_execution_transfer(
    loom_serve_execution_t* execution, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    iree_slim_mutex_unlock(&execution->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "execution timeline is exhausted");
  }
  uint64_t next_value = execution->work.submitted_value + 1;
  const iree_hal_semaphore_list_t waits = {1, &execution->work.semaphore,
                                           &execution->work.submitted_value};
  const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                             &next_value};
  iree_status_t status = iree_hal_queue_transfer(
      execution->transfer_queue, waits, signals, operation_count, operations);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_serve_execution_feedback(
    loom_serve_execution_t* execution, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->feedback.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    iree_slim_mutex_unlock(&execution->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "feedback timeline is exhausted");
  }
  uint64_t next_value = execution->feedback.submitted_value + 1;
  iree_hal_semaphore_t* wait_semaphores[] = {execution->work.semaphore,
                                             execution->feedback.semaphore};
  uint64_t wait_values[] = {execution->work.submitted_value,
                            execution->feedback.submitted_value};
  const iree_hal_semaphore_list_t waits = {IREE_ARRAYSIZE(wait_semaphores),
                                           wait_semaphores, wait_values};
  const iree_hal_semaphore_list_t signals = {1, &execution->feedback.semaphore,
                                             &next_value};
  iree_status_t status = iree_hal_queue_transfer(
      execution->transfer_queue, waits, signals, operation_count, operations);
  if (iree_status_is_ok(status)) {
    execution->feedback.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_serve_execution_wait(loom_serve_execution_t* execution,
                                        uint64_t value) {
  return iree_hal_semaphore_wait(execution->work.semaphore, value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

iree_status_t loom_serve_execution_feedback_wait(
    loom_serve_execution_t* execution, uint64_t value) {
  return iree_hal_semaphore_wait(execution->feedback.semaphore, value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

iree_status_t loom_serve_execution_drain(loom_serve_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  const uint64_t work_value = execution->work.submitted_value;
  const uint64_t feedback_value = execution->feedback.submitted_value;
  iree_slim_mutex_unlock(&execution->mutex);
  iree_status_t status = loom_serve_execution_wait(execution, work_value);
  status = iree_status_join(
      status, loom_serve_execution_feedback_wait(execution, feedback_value));
  // Host-owner drain only. No queue callback tears down shared state. A failed
  // timeline can precede the queue's final resource release, so join private
  // roots as well before the shared pool can be trimmed or destroyed.
  for (;;) {
    iree_slim_mutex_lock(&execution->mutex);
    execution_reap_workspace(execution);
    const bool retired = execution->workspace.count == 0;
    iree_slim_mutex_unlock(&execution->mutex);
    if (retired) {
      break;
    }
    iree_thread_yield();
  }
  return status;
}
