// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/execution.h"

#include <dlfcn.h>

#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/execution/retirement.h"

#define LOOM_SERVE_EXECUTION_GROUP_CAPACITY 32

typedef enum loom_serve_execution_deferred_kind_e {
  LOOM_SERVE_EXECUTION_DEFERRED_EXECUTE = 0,
  LOOM_SERVE_EXECUTION_DEFERRED_DISPATCH = 1,
  LOOM_SERVE_EXECUTION_DEFERRED_DEALLOCA = 2,
  LOOM_SERVE_EXECUTION_DEFERRED_FEEDBACK = 3,
} loom_serve_execution_deferred_kind_t;

typedef struct loom_serve_execution_deferred_t {
  loom_serve_execution_deferred_kind_t kind;
  iree_hal_command_buffer_t* command_buffer;
  iree_host_size_t binding_count;
  iree_hal_buffer_binding_t* bindings;
  iree_hal_executable_t* executable;
  iree_hal_executable_function_t function;
  iree_hal_dispatch_config_t config;
  iree_hal_buffer_ref_t* refs;
  iree_hal_buffer_t* buffer;
  iree_host_size_t operation_count;
  iree_hal_transfer_operation_t operations[2];
} loom_serve_execution_deferred_t;

// Signature of the optional group hook exported by an interposing adapter.
typedef iree_status_t (*loom_serve_execution_group_hook_t)(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_command_buffer_t* const* command_buffers,
    const iree_hal_buffer_binding_table_t* tables);

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
  // Exact queues and inventory ownership held through final execution release.
  loom_serve_execution_binding_t binding;
  // Shared physical command backing, independent of this model's timeline.
  loom_serve_workspace_t* workspace;
  // Ordered commands and input transfers for this model.
  loom_serve_execution_timeline_t work;
  // Ordered downloads consuming produced results without gating model work.
  loom_serve_execution_timeline_t feedback;
  // Protects timeline reservation through queue acceptance, never device waits.
  iree_slim_mutex_t mutex;
  // Transient allocation/pool captures governing workspace activity.
  loom_serve_workspace_scope_t* scope;
  // Persistent allocation/pool captures, independent of workspace activity.
  loom_serve_workspace_scope_t* storage_scope;
  // Model-visible views captured by execution and host payload transfers.
  loom_serve_retirement_t consumers;
  // Deferred work between begin_group and end_group.
  struct {
    bool active;
    iree_host_size_t count;
    loom_serve_execution_deferred_t
        entries[LOOM_SERVE_EXECUTION_GROUP_CAPACITY];
  } group;
};

static iree_status_t loom_serve_execution_defer(
    loom_serve_execution_t* execution,
    loom_serve_execution_deferred_t** out_entry) {
  if (execution->group.count == LOOM_SERVE_EXECUTION_GROUP_CAPACITY) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "execution group capacity exhausted");
  }
  loom_serve_execution_deferred_t* entry =
      &execution->group.entries[execution->group.count++];
  memset(entry, 0, sizeof(*entry));
  *out_entry = entry;
  return iree_ok_status();
}

static void loom_serve_execution_deferred_release(
    loom_serve_execution_t* execution, loom_serve_execution_deferred_t* entry) {
  iree_hal_command_buffer_release(entry->command_buffer);
  for (iree_host_size_t i = 0; i < entry->binding_count; ++i) {
    iree_hal_buffer_release(entry->bindings ? entry->bindings[i].buffer
                                            : entry->refs[i].buffer);
  }
  iree_allocator_free(execution->host_allocator, entry->bindings);
  iree_allocator_free(execution->host_allocator, entry->refs);
  iree_hal_executable_release(entry->executable);
  iree_hal_buffer_release(entry->buffer);
  for (iree_host_size_t i = 0; i < entry->operation_count; ++i) {
    iree_hal_buffer_release(entry->operations[i].download.source_buffer);
  }
  memset(entry, 0, sizeof(*entry));
}

void loom_serve_execution_retain(loom_serve_execution_t* execution) {
  if (execution) {
    iree_atomic_ref_count_inc(&execution->ref_count);
  }
}

void loom_serve_execution_release(loom_serve_execution_t* execution) {
  if (execution && iree_atomic_ref_count_dec(&execution->ref_count) == 1) {
    loom_serve_retirement_deinitialize(&execution->consumers);
    loom_serve_workspace_scope_retire(execution->storage_scope);
    loom_serve_workspace_scope_retire(execution->scope);
    loom_serve_workspace_release(execution->workspace);
    iree_slim_mutex_deinitialize(&execution->mutex);
    iree_hal_semaphore_release(execution->feedback.semaphore);
    iree_hal_semaphore_release(execution->work.semaphore);
    loom_serve_execution_binding_release(execution->binding);
    iree_allocator_free(execution->host_allocator, execution);
  }
}

void loom_serve_execution_set_retirement_observer(
    loom_serve_execution_t* execution, void (*wake)(void* user_data),
    void* user_data) {
  loom_serve_retirement_set_observer(&execution->consumers, wake, user_data);
}

bool loom_serve_execution_is_retired(loom_serve_execution_t* execution) {
  return loom_serve_retirement_is_idle(&execution->consumers);
}

iree_status_t loom_serve_execution_reserve_workspace(
    loom_serve_execution_t* execution, iree_device_size_t length,
    iree_device_size_t alignment) {
  return loom_serve_workspace_reserve(execution->workspace, length, alignment);
}

iree_status_t loom_serve_execution_alloca(
    loom_serve_execution_t* execution, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  memset(out_buffers, 0, count * sizeof(*out_buffers));
  iree_slim_mutex_lock(&execution->mutex);
  iree_hal_pool_t* pool = NULL;
  iree_status_t status =
      loom_serve_workspace_scope_acquire(execution->scope, count, &pool);
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
      loom_serve_workspace_scope_capture(execution->scope, pool,
                                         out_buffers[i]);
    }
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      status = loom_serve_retirement_track(
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

iree_status_t loom_serve_execution_alloca_from_pool(
    loom_serve_execution_t* execution, iree_hal_pool_t* pool,
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
    status = loom_serve_workspace_scope_allocate(execution->workspace,
                                                 /*tracks_workspace=*/false,
                                                 &execution->storage_scope);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_workspace_scope_reserve(execution->storage_scope, count);
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
      loom_serve_workspace_scope_capture(execution->storage_scope, pool,
                                         out_buffers[i]);
    }
  }
  iree_slim_mutex_unlock(&execution->mutex);
  return status;
}

iree_status_t loom_serve_execution_dealloca(loom_serve_execution_t* execution,
                                            iree_hal_buffer_t* buffer,
                                            uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->group.active) {
    loom_serve_execution_deferred_t* entry = NULL;
    iree_status_t status = loom_serve_execution_defer(execution, &entry);
    if (iree_status_is_ok(status)) {
      entry->kind = LOOM_SERVE_EXECUTION_DEFERRED_DEALLOCA;
      entry->buffer = buffer;
      iree_hal_buffer_retain(buffer);
      *out_value = execution->work.submitted_value;
    }
    iree_slim_mutex_unlock(&execution->mutex);
    return status;
  }
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

iree_status_t loom_serve_execution_create(
    loom_serve_execution_queues_t* queues, loom_serve_workspace_t* workspace,
    iree_allocator_t host_allocator, loom_serve_execution_t** out_execution) {
  *out_execution = NULL;
  loom_serve_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  iree_atomic_ref_count_init(&execution->ref_count);
  iree_slim_mutex_initialize(&execution->mutex);
  loom_serve_retirement_initialize(&execution->consumers);
  execution->host_allocator = host_allocator;
  execution->workspace = workspace;
  loom_serve_workspace_retain(workspace);
  execution->binding = loom_serve_execution_queues_acquire(queues);
  iree_hal_device_t* device = iree_hal_queue_family_device(
      iree_hal_queue_family(execution->binding.dispatch));
  iree_status_t status = loom_serve_workspace_scope_allocate(
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
    loom_serve_execution_release(execution);
  }
  return status;
}

loom_serve_execution_frontier_t loom_serve_execution_frontier(
    loom_serve_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  const loom_serve_execution_frontier_t frontier = {
      .semaphores = {execution->work.semaphore, execution->feedback.semaphore},
      .values = {execution->work.submitted_value,
                 execution->feedback.submitted_value},
  };
  iree_slim_mutex_unlock(&execution->mutex);
  return frontier;
}

iree_status_t loom_serve_execution_barrier(loom_serve_execution_t* execution,
                                           iree_hal_semaphore_list_t waits) {
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

iree_status_t loom_serve_execution_execute(
    loom_serve_execution_t* execution,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->group.active) {
    loom_serve_execution_deferred_t* entry = NULL;
    iree_status_t status = loom_serve_execution_defer(execution, &entry);
    if (iree_status_is_ok(status)) {
      entry->kind = LOOM_SERVE_EXECUTION_DEFERRED_EXECUTE;
      status = iree_allocator_malloc_array(
          execution->host_allocator, iree_max(bindings.count, 1),
          sizeof(*entry->bindings), (void**)&entry->bindings);
    }
    if (iree_status_is_ok(status)) {
      entry->command_buffer = command_buffer;
      iree_hal_command_buffer_retain(command_buffer);
      entry->binding_count = bindings.count;
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        entry->bindings[i] = bindings.bindings[i];
        iree_hal_buffer_retain(entry->bindings[i].buffer);
      }
      *out_value = execution->work.submitted_value;
    }
    iree_slim_mutex_unlock(&execution->mutex);
    return status;
  }
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

iree_status_t loom_serve_execution_dispatch(
    loom_serve_execution_t* execution, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    iree_hal_buffer_ref_list_t bindings, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->group.active) {
    loom_serve_execution_deferred_t* entry = NULL;
    iree_status_t status = loom_serve_execution_defer(execution, &entry);
    if (iree_status_is_ok(status)) {
      entry->kind = LOOM_SERVE_EXECUTION_DEFERRED_DISPATCH;
      status = iree_allocator_malloc_array(
          execution->host_allocator, iree_max(bindings.count, 1),
          sizeof(*entry->refs), (void**)&entry->refs);
    }
    if (iree_status_is_ok(status)) {
      entry->executable = executable;
      iree_hal_executable_retain(executable);
      entry->function = function;
      entry->config = config;
      entry->binding_count = bindings.count;
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        entry->refs[i] = bindings.values[i];
        iree_hal_buffer_retain(entry->refs[i].buffer);
      }
      *out_value = execution->work.submitted_value;
    }
    iree_slim_mutex_unlock(&execution->mutex);
    return status;
  }
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

iree_status_t loom_serve_execution_feedback(
    loom_serve_execution_t* execution, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, uint64_t* out_value) {
  iree_slim_mutex_lock(&execution->mutex);
  if (execution->group.active) {
    loom_serve_execution_deferred_t* entry = NULL;
    iree_status_t status =
        operation_count > IREE_ARRAYSIZE(entry->operations)
            ? iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                               "deferred feedback has too many operations")
            : loom_serve_execution_defer(execution, &entry);
    if (iree_status_is_ok(status)) {
      entry->kind = LOOM_SERVE_EXECUTION_DEFERRED_FEEDBACK;
      entry->operation_count = operation_count;
      for (iree_host_size_t i = 0; i < operation_count; ++i) {
        entry->operations[i] = operations[i];
        iree_hal_buffer_retain(operations[i].download.source_buffer);
      }
      *out_value = execution->feedback.submitted_value;
    }
    iree_slim_mutex_unlock(&execution->mutex);
    return status;
  }
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

iree_status_t loom_serve_execution_wait(loom_serve_execution_t* execution,
                                        uint64_t value) {
  return loom_serve_workspace_wait(execution->workspace,
                                   execution->work.semaphore, value);
}

iree_status_t loom_serve_execution_feedback_wait(
    loom_serve_execution_t* execution, uint64_t value) {
  return loom_serve_workspace_wait(execution->workspace,
                                   execution->feedback.semaphore, value);
}

iree_status_t loom_serve_execution_drain(loom_serve_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  const uint64_t work_value = execution->work.submitted_value;
  const uint64_t feedback_value = execution->feedback.submitted_value;
  iree_slim_mutex_unlock(&execution->mutex);
  iree_status_t status = loom_serve_execution_wait(execution, work_value);
  status = iree_status_join(
      status, loom_serve_execution_feedback_wait(execution, feedback_value));
  // Consumer captures protect model payloads after early failed readiness.
  // Raw allocation objects independently remain protected by the shared pool.
  return iree_status_join(
      status, loom_serve_workspace_await_retirement(execution->workspace,
                                                    &execution->consumers));
}

void loom_serve_execution_begin_group(loom_serve_execution_t* execution) {
  iree_slim_mutex_lock(&execution->mutex);
  execution->group.active = true;
  execution->group.count = 0;
  iree_slim_mutex_unlock(&execution->mutex);
}

iree_status_t loom_serve_execution_end_group(
    loom_serve_execution_t* execution) {
  static loom_serve_execution_group_hook_t hook = NULL;
  static bool resolved = false;
  if (!resolved) {
    hook = (loom_serve_execution_group_hook_t)dlsym(
        RTLD_DEFAULT, "loom_hybrid_execute_group");
    resolved = true;
  }
  iree_slim_mutex_lock(&execution->mutex);
  execution->group.active = false;
  const iree_host_size_t count = execution->group.count;
  iree_hal_command_buffer_t*
      command_buffers[LOOM_SERVE_EXECUTION_GROUP_CAPACITY];
  iree_hal_buffer_binding_table_t tables[LOOM_SERVE_EXECUTION_GROUP_CAPACITY];
  iree_host_size_t work_count = 0;
  bool executes_only = true;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_execution_deferred_t* entry = &execution->group.entries[i];
    if (entry->kind == LOOM_SERVE_EXECUTION_DEFERRED_EXECUTE) {
      command_buffers[work_count] = entry->command_buffer;
      tables[work_count++] = (iree_hal_buffer_binding_table_t){
          entry->binding_count, entry->bindings};
    } else if (entry->kind == LOOM_SERVE_EXECUTION_DEFERRED_DISPATCH) {
      executes_only = false;
    }
  }
  iree_status_t status = iree_ok_status();
  if (hook && executes_only && work_count) {
    uint64_t next_value = execution->work.submitted_value + 1;
    const iree_hal_semaphore_list_t waits = {1, &execution->work.semaphore,
                                             &execution->work.submitted_value};
    const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                               &next_value};
    status = hook(execution->binding.dispatch, waits, signals, work_count,
                  command_buffers, tables);
    if (iree_status_is_ok(status)) {
      execution->work.submitted_value = next_value;
    }
  } else {
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      loom_serve_execution_deferred_t* entry = &execution->group.entries[i];
      if (entry->kind != LOOM_SERVE_EXECUTION_DEFERRED_EXECUTE &&
          entry->kind != LOOM_SERVE_EXECUTION_DEFERRED_DISPATCH) {
        continue;
      }
      uint64_t next_value = execution->work.submitted_value + 1;
      const iree_hal_semaphore_list_t waits = {
          1, &execution->work.semaphore, &execution->work.submitted_value};
      const iree_hal_semaphore_list_t signals = {1, &execution->work.semaphore,
                                                 &next_value};
      status = entry->kind == LOOM_SERVE_EXECUTION_DEFERRED_EXECUTE
                   ? iree_hal_queue_execute(
                         execution->binding.dispatch, waits, signals,
                         entry->command_buffer,
                         (iree_hal_buffer_binding_table_t){entry->binding_count,
                                                           entry->bindings},
                         IREE_HAL_QUEUE_EXECUTE_FLAG_NONE)
                   : iree_hal_queue_dispatch(
                         execution->binding.dispatch, waits, signals,
                         entry->executable, entry->function, entry->config,
                         iree_const_byte_span_empty(),
                         (iree_hal_buffer_ref_list_t){entry->binding_count,
                                                      entry->refs},
                         NULL, IREE_HAL_DISPATCH_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        execution->work.submitted_value = next_value;
      }
    }
  }
  iree_slim_mutex_unlock(&execution->mutex);
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_execution_deferred_t* entry = &execution->group.entries[i];
    uint64_t value = 0;
    if (iree_status_is_ok(status) &&
        entry->kind == LOOM_SERVE_EXECUTION_DEFERRED_DEALLOCA) {
      status = loom_serve_execution_dealloca(execution, entry->buffer, &value);
    } else if (iree_status_is_ok(status) &&
               entry->kind == LOOM_SERVE_EXECUTION_DEFERRED_FEEDBACK) {
      status = loom_serve_execution_feedback(execution, entry->operation_count,
                                             entry->operations, &value);
    }
    loom_serve_execution_deferred_release(execution, entry);
  }
  execution->group.count = 0;
  return status;
}
