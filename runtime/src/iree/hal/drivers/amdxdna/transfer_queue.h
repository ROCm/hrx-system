// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_TRANSFER_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_TRANSFER_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

typedef struct iree_hal_amdxdna_transfer_queue_t
    iree_hal_amdxdna_transfer_queue_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a queue with a dedicated worker for synchronous HAL file transfers.
// |block_pool| is borrowed and must outlive the queue.
iree_status_t iree_hal_amdxdna_transfer_queue_create(
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_amdxdna_transfer_queue_t** out_queue);

void iree_hal_amdxdna_transfer_queue_set_frontier(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_atomic_uint64_t* epoch_counter);

// Stops the worker, drains pending transfers, and frees the queue.
void iree_hal_amdxdna_transfer_queue_destroy(
    iree_hal_amdxdna_transfer_queue_t* queue);

iree_status_t iree_hal_amdxdna_transfer_queue_read(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

iree_status_t iree_hal_amdxdna_transfer_queue_write(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_TRANSFER_QUEUE_H_
