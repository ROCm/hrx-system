// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/execution.h"

#include "iree/base/threading/mutex.h"

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
};

void loom_serve_execution_retain(loom_serve_execution_t* execution) {
  iree_atomic_ref_count_inc(&execution->ref_count);
}

void loom_serve_execution_release(loom_serve_execution_t* execution) {
  if (execution && iree_atomic_ref_count_dec(&execution->ref_count) == 1) {
    iree_slim_mutex_deinitialize(&execution->mutex);
    iree_hal_semaphore_release(execution->feedback.semaphore);
    iree_hal_semaphore_release(execution->work.semaphore);
    iree_hal_queue_release(execution->transfer_queue);
    iree_hal_queue_release(execution->dispatch_queue);
    iree_hal_device_release(execution->device);
    iree_allocator_free(execution->host_allocator, execution);
  }
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
  return iree_status_join(
      status, loom_serve_execution_feedback_wait(execution, feedback_value));
}
