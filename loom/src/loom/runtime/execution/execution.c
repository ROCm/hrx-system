// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/execution.h"

#include "iree/base/threading/mutex.h"
#include "loom/runtime/execution/retirement.h"

typedef struct loom_runtime_execution_timeline_t {
  // Retained timeline identifying accepted operations on this branch.
  iree_hal_semaphore_t* semaphore;
  // Last accepted submission, advanced only after successful queue submission.
  uint64_t submitted_value;
} loom_runtime_execution_timeline_t;

struct loom_runtime_execution_t {
  // Intrusive ownership shared by the host and native VM module.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for this object.
  iree_allocator_t host_allocator;
  // Exact queues and inventory ownership held through final execution release.
  loom_runtime_execution_binding_t binding;
  // Shared physical command backing, independent of this model's timeline.
  loom_runtime_workspace_t* workspace;
  // Ordered commands and input transfers for this model.
  loom_runtime_execution_timeline_t work;
  // Ordered downloads consuming produced results without gating model work.
  loom_runtime_execution_timeline_t feedback;
  // Protects timeline reservation through queue acceptance, never device waits.
  iree_slim_mutex_t mutex;
  // Transient allocation/pool captures governing workspace activity.
  loom_runtime_workspace_scope_t* scope;
  // Persistent allocation/pool captures, independent of workspace activity.
  loom_runtime_workspace_scope_t* storage_scope;
  // Model-visible views captured by execution and host payload transfers.
  loom_runtime_retirement_t consumers;
};

void loom_runtime_execution_retain(loom_runtime_execution_t* execution) {
  if (execution) {
    iree_atomic_ref_count_inc(&execution->ref_count);
  }
}

void loom_runtime_execution_release(loom_runtime_execution_t* execution) {
  if (execution && iree_atomic_ref_count_dec(&execution->ref_count) == 1) {
    loom_runtime_retirement_deinitialize(&execution->consumers);
    loom_runtime_workspace_scope_retire(execution->storage_scope);
    loom_runtime_workspace_scope_retire(execution->scope);
    loom_runtime_workspace_release(execution->workspace);
    iree_slim_mutex_deinitialize(&execution->mutex);
    iree_hal_semaphore_release(execution->feedback.semaphore);
    iree_hal_semaphore_release(execution->work.semaphore);
    loom_runtime_execution_binding_release(execution->binding);
    iree_allocator_free(execution->host_allocator, execution);
  }
}

const iree_hal_queue_family_t* loom_runtime_execution_dispatch_family(
    const loom_runtime_execution_t* execution) {
  return iree_hal_queue_family(execution->binding.dispatch);
}

void loom_runtime_execution_set_retirement_observer(
    loom_runtime_execution_t* execution, void (*wake)(void* user_data),
    void* user_data) {
  loom_runtime_retirement_set_observer(&execution->consumers, wake, user_data);
}

bool loom_runtime_execution_is_retired(loom_runtime_execution_t* execution) {
  return loom_runtime_retirement_is_idle(&execution->consumers);
}

iree_status_t loom_runtime_execution_reserve_workspace(
    loom_runtime_execution_t* execution, iree_device_size_t length,
    iree_device_size_t alignment) {
  return loom_runtime_workspace_reserve(execution->workspace, length,
                                        alignment);
}

iree_status_t loom_runtime_execution_alloca(
    loom_runtime_execution_t* execution, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  memset(out_buffers, 0, count * sizeof(*out_buffers));
  iree_slim_mutex_lock(&execution->mutex);
  iree_hal_pool_t* pool = NULL;
  iree_status_t status =
      loom_runtime_workspace_scope_acquire(execution->scope, count, &pool);
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
    status = iree_hal_queue_alloca(execution->binding.dispatch, waits, signals,
                                   pool, count, requests, out_buffers);
  }
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    // Capture the complete accepted batch before wrapping caller views. Every
    // raw root borrows its source pool independently of its siblings.
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (i) {
        iree_hal_pool_retain(pool);
      }
      loom_runtime_workspace_scope_capture(execution->scope, pool,
                                           out_buffers[i]);
    }
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      status = loom_runtime_retirement_track(
          &execution->consumers, &out_buffers[i], execution->host_allocator);
    }
    if (!iree_status_is_ok(status)) {
      // Alloca was accepted even if a model-view allocation failed. The shared
      // scope keeps every original root and pool through queue retirement.
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_hal_buffer_release(out_buffers[i]);
        out_buffers[i] = NULL;
      }
    }
  } else {
    iree_hal_pool_release(pool);
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_alloca_from_pool(
    loom_runtime_execution_t* execution, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  memset(out_buffers, 0, count * sizeof(*out_buffers));
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    iree_slim_mutex_unlock(&execution->mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "execution timeline is exhausted");
  }
  uint64_t next_value = execution->work.submitted_value + 1;
  iree_status_t status = iree_ok_status();
  if (!execution->storage_scope) {
    status = loom_runtime_workspace_scope_allocate(execution->workspace,
                                                   /*tracks_workspace=*/false,
                                                   &execution->storage_scope);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_runtime_workspace_scope_reserve(execution->storage_scope, count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_alloca(
        execution->binding.dispatch,
        (iree_hal_semaphore_list_t){1, &execution->work.semaphore,
                                    &execution->work.submitted_value},
        (iree_hal_semaphore_list_t){1, &execution->work.semaphore, &next_value},
        pool, count, requests, out_buffers);
  }
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_pool_retain(pool);
      loom_runtime_workspace_scope_capture(execution->storage_scope, pool,
                                           out_buffers[i]);
    }
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_dealloca(
    loom_runtime_execution_t* execution, iree_hal_buffer_t* buffer,
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
  iree_hal_buffer_t* root = iree_hal_buffer_allocated_buffer(buffer);
  iree_status_t status = iree_hal_queue_dealloca(execution->binding.dispatch,
                                                 waits, signals, 1, &root);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_create(
    loom_runtime_execution_queues_t* queues,
    loom_runtime_workspace_t* workspace, iree_allocator_t host_allocator,
    loom_runtime_execution_t** out_execution) {
  *out_execution = NULL;
  loom_runtime_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  iree_atomic_ref_count_init(&execution->ref_count);
  iree_slim_mutex_initialize(&execution->mutex);
  loom_runtime_retirement_initialize(&execution->consumers);
  execution->host_allocator = host_allocator;
  execution->workspace = workspace;
  loom_runtime_workspace_retain(workspace);
  execution->binding = loom_runtime_execution_queues_acquire(queues);
  iree_hal_device_t* device = iree_hal_queue_family_device(
      iree_hal_queue_family(execution->binding.dispatch));
  iree_status_t status = loom_runtime_workspace_scope_allocate(
      workspace, /*tracks_workspace=*/true, &execution->scope);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution->work.semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution->feedback.semaphore);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    loom_runtime_execution_release(execution);
  }
  return status;
}

loom_runtime_execution_frontier_t loom_runtime_execution_frontier(
    loom_runtime_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  const loom_runtime_execution_frontier_t frontier = {
      .semaphores = {execution->work.semaphore, execution->feedback.semaphore},
      .values = {execution->work.submitted_value,
                 execution->feedback.submitted_value},
  };
  iree_slim_mutex_unlock(&execution->mutex);
  return frontier;
}

iree_status_t loom_runtime_execution_barrier(
    loom_runtime_execution_t* execution, iree_hal_semaphore_list_t waits) {
  iree_hal_semaphore_t* inline_semaphores[8];
  uint64_t inline_values[8];
  iree_hal_semaphore_t** semaphores = inline_semaphores;
  uint64_t* values = inline_values;
  const iree_host_size_t count = waits.count + 1;
  void* storage = NULL;
  if (count > IREE_ARRAYSIZE(inline_semaphores)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        execution->host_allocator, count, sizeof(*semaphores) + sizeof(*values),
        &storage));
    semaphores = storage;
    values = (uint64_t*)(semaphores + count);
  }
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    semaphores[i] = waits.semaphores[i];
    values[i] = waits.payload_values[i];
  }
  iree_slim_mutex_lock(&execution->mutex);
  semaphores[waits.count] = execution->work.semaphore;
  values[waits.count] = execution->work.submitted_value;
  uint64_t next_value = execution->work.submitted_value + 1;
  const iree_hal_semaphore_list_t dependencies = {count, semaphores, values};
  const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                             &next_value};
  iree_status_t status =
      execution->work.submitted_value == IREE_HAL_SEMAPHORE_MAX_VALUE
          ? iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "execution timeline is exhausted")
          : iree_hal_queue_barrier(execution->binding.dispatch, dependencies,
                                   signals, NULL,
                                   IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  iree_allocator_free(execution->host_allocator, storage);
  return status;
}

iree_status_t loom_runtime_execution_execute(
    loom_runtime_execution_t* execution,
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
      execution->binding.dispatch, waits, signals, command_buffer, bindings,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_timestamp(
    loom_runtime_execution_t* execution, iree_hal_buffer_t* target,
    iree_device_size_t offset, uint64_t* out_value) {
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
  iree_status_t status = iree_hal_queue_timestamp(
      execution->binding.dispatch, waits, signals, target, offset, NULL,
      IREE_HAL_TIMESTAMP_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_dispatch(
    loom_runtime_execution_t* execution, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    iree_hal_buffer_ref_list_t bindings, uint64_t* out_value) {
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
  iree_status_t status = iree_hal_queue_dispatch(
      execution->binding.dispatch, waits, signals, executable, function, config,
      iree_const_byte_span_empty(), bindings, NULL,
      IREE_HAL_DISPATCH_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_transfer(
    loom_runtime_execution_t* execution, iree_host_size_t operation_count,
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
  iree_status_t status =
      iree_hal_queue_transfer(execution->binding.transfer, waits, signals,
                              operation_count, operations, NULL);
  if (iree_status_is_ok(status)) {
    execution->work.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_feedback(
    loom_runtime_execution_t* execution, iree_host_size_t operation_count,
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
  iree_status_t status =
      iree_hal_queue_transfer(execution->binding.transfer, waits, signals,
                              operation_count, operations, NULL);
  if (iree_status_is_ok(status)) {
    execution->feedback.submitted_value = next_value;
    *out_value = next_value;
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_runtime_execution_wait(loom_runtime_execution_t* execution,
                                          uint64_t value) {
  return loom_runtime_workspace_wait(execution->workspace,
                                     execution->work.semaphore, value);
}

iree_status_t loom_runtime_execution_feedback_wait(
    loom_runtime_execution_t* execution, uint64_t value) {
  return loom_runtime_workspace_wait(execution->workspace,
                                     execution->feedback.semaphore, value);
}

iree_status_t loom_runtime_execution_drain(
    loom_runtime_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  const uint64_t work_value = execution->work.submitted_value;
  const uint64_t feedback_value = execution->feedback.submitted_value;
  iree_slim_mutex_unlock(&execution->mutex);
  iree_status_t status = loom_runtime_execution_wait(execution, work_value);
  status = iree_status_join(
      status, loom_runtime_execution_feedback_wait(execution, feedback_value));
  // Consumer captures protect model payloads after early failed readiness.
  // Raw allocation objects independently remain protected by the shared pool.
  return iree_status_join(
      status, loom_runtime_workspace_await_retirement(execution->workspace,
                                                      &execution->consumers));
}
