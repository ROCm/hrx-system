// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-owned queue allocation metadata and dependency registration. Native
// pool work runs on the existing service without occupying a queue's publisher.
// Accepted operations and returned transient roots retain the device, keeping
// this state alive independently of the caller's device reference.
typedef struct iree_hal_amd_gpu_allocation_context_t {
  // Borrowed parent, retained by individual operations and transient roots.
  iree_hal_device_t* device;
  // Borrowed diagnostic owner, dominated by device.
  const iree_hal_amd_gpu_context_t* context;
  // Borrowed thread-backed cold memory and continuation executor.
  iree_hal_memory_maintenance_t* service;
  // Reusable CPU storage for captures, wait records, and transient roots.
  iree_arena_block_pool_t blocks;
  // Dependency associations shared by this device's allocation operations.
  iree_async_semaphore_wait_context_t waits;
} iree_hal_amd_gpu_allocation_context_t;

// Initializes embedded state without creating a worker or native allocation.
void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, const iree_hal_amd_gpu_context_t* context,
    iree_hal_memory_maintenance_t* service, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context);

// Releases idle metadata after every operation/root and service callback joins.
void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context);

// Captures the complete transaction and returns uncommitted roots. The public
// queue boundary has validated lists, family grants, and request shape. Pool
// limits are admitted here. Failure leaves every output untouched; success
// transfers one root reference per request and completes through signals.
iree_status_t iree_hal_amd_gpu_queue_alloca(
    iree_hal_amd_gpu_allocation_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_pool_t* pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers);

// Captures roots transactionally, then decommits and returns their reservations
// after host-completed dependencies. Roots may originate from different pools
// or cooperating devices in the same sealed group. Pool/root grants and root
// capability are already checked by the public queue boundary.
iree_status_t iree_hal_amd_gpu_queue_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t count, iree_hal_buffer_t* const* buffers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
