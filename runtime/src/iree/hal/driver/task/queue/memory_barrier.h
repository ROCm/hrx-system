// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_TASK_QUEUE_MEMORY_BARRIER_H_
#define IREE_HAL_DRIVER_TASK_QUEUE_MEMORY_BARRIER_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_task_queue_op_t iree_hal_task_queue_op_t;
typedef struct iree_hal_task_queue_memory_barriers_t
    iree_hal_task_queue_memory_barriers_t;

// Copies selected recipes into the operation arena and retains their buffers.
// NULL sides select conservative maintenance of the operation's actual
// operands. Coherent operations with no host recipes allocate no maintenance
// storage.
iree_status_t iree_hal_task_queue_memory_barriers_capture(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_queue_barriers_t* barriers);

// Executes after dependencies become ready on a Task worker. Native maintenance
// errors terminate the operation before its body can access memory.
iree_status_t iree_hal_task_queue_memory_barriers_before(
    iree_hal_task_queue_op_t* operation);

// Executes on a Task worker before terminal success is published.
iree_status_t iree_hal_task_queue_memory_barriers_after(
    iree_hal_task_queue_op_t* operation);

// True when successful file completion must return to a worker for cache work.
// Pure CPU fences alone can complete on the proactor thread.
bool iree_hal_task_queue_memory_barriers_file_needs_worker(
    const iree_hal_task_queue_op_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_TASK_QUEUE_MEMORY_BARRIER_H_
