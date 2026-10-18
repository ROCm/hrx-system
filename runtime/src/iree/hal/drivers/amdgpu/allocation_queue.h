// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATION_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATION_QUEUE_H_

#include "iree/base/threading/affinity.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// A ROCr-private native allocation queue owned by one physical device.
// Only ready native slab acquisition and release execute on its worker.
typedef struct iree_hal_amdgpu_allocation_queue_t
    iree_hal_amdgpu_allocation_queue_t;

// Creates a ref-counted native allocation queue with the device's placement.
// The worker starts with |affinity| and sleeps when no native batch is ready.
// Failure publishes no output. The physical device retains its initial
// reference.
iree_status_t iree_hal_amdgpu_allocation_queue_create(
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_allocation_queue_t** out_queue);

void iree_hal_amdgpu_allocation_queue_retain(
    iree_hal_amdgpu_allocation_queue_t* queue);
void iree_hal_amdgpu_allocation_queue_release(
    iree_hal_amdgpu_allocation_queue_t* queue);

// Joins native work before the physical device releases its ROCr context.
// All producers have retired; this cold owner boundary is never a pool return
// or queue-operation wait. Calling from a native completion drains ready work
// inline and lets the worker's storage claim cover its final return.
void iree_hal_amdgpu_allocation_queue_close(
    iree_hal_amdgpu_allocation_queue_t* queue);

// Enqueues a prepared native slab batch. Its ROCr provider and complete
// request remain live through completion. Submission allocates no storage and
// performs no native calls. No dependency wait belongs in this queue.
void iree_hal_amdgpu_allocation_queue_enqueue(
    iree_hal_amdgpu_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATION_QUEUE_H_
