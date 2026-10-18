// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TASK_ALLOCATION_QUEUE_H_
#define IREE_HAL_DRIVERS_TASK_ALLOCATION_QUEUE_H_

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A Task-private native allocation domain. Its exclusive process accepts only
// typed slab acquisition/release and runs on existing executor workers.
typedef struct iree_hal_task_allocation_queue_t
    iree_hal_task_allocation_queue_t;

// Creates a ref-counted allocation owner retaining the executor and selected
// progress entry. The entry keeps pool notifications usable through detached
// native retirement. No thread is created.
iree_status_t iree_hal_task_allocation_queue_create(
    iree_task_executor_t* executor,
    iree_async_proactor_pool_entry_t* progress_entry,
    iree_allocator_t host_allocator,
    iree_hal_task_allocation_queue_t** out_queue);

void iree_hal_task_allocation_queue_retain(
    iree_hal_task_allocation_queue_t* queue);
void iree_hal_task_allocation_queue_release(
    iree_hal_task_allocation_queue_t* queue);

// Drains accepted native batches and joins the process's final access. The
// owning device calls this from its outer teardown after all public queue and
// pool producers are quiescent, before releasing its queue reference. Provider
// references may be released on workers; they never initiate a shutdown wait.
void iree_hal_task_allocation_queue_close(
    iree_hal_task_allocation_queue_t* queue);

// Creates a provider routing native slab batches through |queue|. The native
// source is an inline CPU heap or shared-section provider; all other operations
// retain that source's execution semantics. Both source and queue are retained.
// Borrowed CPU imports have no native allocation to free and return inline.
iree_status_t iree_hal_task_slab_provider_create(
    iree_hal_task_allocation_queue_t* queue,
    iree_hal_slab_provider_t* native_provider, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TASK_ALLOCATION_QUEUE_H_
