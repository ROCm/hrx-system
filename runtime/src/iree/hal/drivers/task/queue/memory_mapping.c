// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/hal/virtual_buffer.h"

typedef struct iree_hal_task_queue_mapping_t {
  // Finite worker process retained by the operation's arena until release_fn.
  iree_task_process_t process;
  // Accepted HAL operation, completed after the final process access.
  iree_hal_task_queue_op_t* operation;
  // Prepared native mapping batch retained by the operation's resource set.
  iree_hal_host_call_t call;
  // Owned native result transferred to HAL completion by release_fn.
  iree_status_t status;
} iree_hal_task_queue_mapping_t;

static iree_status_t iree_hal_task_queue_map_execute(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_task_queue_mapping_t* mapping = process->user_data;
  iree_hal_task_queue_profile_start_host_execution(mapping->operation);
  iree_hal_host_call_context_t context = {
      .queue = &mapping->operation->queue->base,
      .signal_semaphore_list = iree_hal_semaphore_list_empty(),
  };
  const uint64_t args[4] = {0};
  mapping->status = mapping->call.fn(mapping->call.user_data, args, &context);
  out_result->did_work = true;
  out_result->completed = true;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_task_queue_map_release(iree_task_process_t* process) {
  iree_hal_task_queue_mapping_t* mapping = process->user_data;
  if (iree_status_is_ok(mapping->status)) {
    iree_hal_task_queue_op_complete(mapping->operation);
  } else {
    iree_hal_task_queue_op_fail(mapping->operation, mapping->status);
  }
}

void iree_hal_task_queue_issue_map(iree_hal_task_queue_op_t* operation) {
  iree_task_executor_schedule_process(operation->queue->executor,
                                      &operation->mapping->process);
}

iree_status_t iree_hal_task_queue_submit_map(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)base_queue;
  iree_hal_host_call_t call;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(
      iree_hal_queue_family_device(iree_hal_queue_family(base_queue)));
  iree_status_t status = iree_hal_memory_mapping_prepare(
      base_queue, mapping_count, mappings, flags, host_allocator, &call);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_t* operation = NULL;
    status = iree_hal_task_queue_submit_op_begin(
        queue, IREE_HAL_TASK_QUEUE_OP_MAP, &signals, &operation);
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_aligned(
          &operation->arena, sizeof(*operation->mapping),
          iree_alignof(iree_hal_task_queue_mapping_t),
          (void**)&operation->mapping);
      if (iree_status_is_ok(status)) {
        iree_hal_task_queue_mapping_t* mapping = operation->mapping;
        mapping->operation = operation;
        mapping->call = call;
        mapping->status = iree_ok_status();
        iree_task_process_initialize(iree_hal_task_queue_map_execute,
                                     /*suspend_count=*/0, /*wake_budget=*/1,
                                     &mapping->process);
        mapping->process.user_data = mapping;
        mapping->process.release_fn = iree_hal_task_queue_map_release;
        status = iree_hal_task_queue_submit_op_finish(
            operation, waits, call.resource ? 1 : 0, &call.resource);
      } else {
        iree_hal_task_queue_op_discard(operation);
      }
    }
    iree_hal_resource_release(call.resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
