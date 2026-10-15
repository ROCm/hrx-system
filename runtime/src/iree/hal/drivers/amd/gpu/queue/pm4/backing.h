// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Argument generations are FIFO by publication, including alignment and wrap.
// The service frees a replaced generation only after its final byte position
// has retired. Native readers of signals have a separate lifetime below.
typedef struct iree_hal_amd_gpu_pm4_argument_storage_t {
  // Older generation, or NULL at the end of the cold cleanup list.
  struct iree_hal_amd_gpu_pm4_argument_storage_t* next;
  // Native storage and persistent host view.
  iree_hal_amd_fabric_memory_t memory;
  // Address of byte zero in this queue's native domain.
  uint64_t address;
  // Monotonic first position belonging to this generation.
  uint64_t begin;
  // Exclusive final reserved position in this generation.
  uint64_t end;
  // Power-of-two usable byte capacity.
  uint64_t capacity;
} iree_hal_amd_gpu_pm4_argument_storage_t;

typedef struct iree_hal_amd_gpu_pm4_signal_storage_t {
  // Next backing, retained while active or referenced by native readers.
  struct iree_hal_amd_gpu_pm4_signal_storage_t* next;
  // Native cells and coherent CPU mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Finite bookkeeping over the following inline signal metadata.
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
} iree_hal_amd_gpu_pm4_signal_storage_t;

// Allocates coherent write-back storage for PM4 commands, arguments, or control
// cells under the caller's admitted memory policy. A non-NULL out_address asks
// for its GPU address at access ordinal zero. Failure may leave partial memory;
// the caller always deinitializes it. All backing operations run on service.
iree_status_t iree_hal_amd_gpu_pm4_backing_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address);

// Allocates an independent generation at the admitted monotonic begin position.
// The caller links and retires generations by their final reserved byte extent.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_pm4_argument_storage_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t** out_storage);

// Frees a generation after every published byte has retired. List ownership and
// the exact consumed-byte proof stay with the queue.
void iree_hal_amd_gpu_pm4_argument_storage_free(
    const iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t* storage);

// Allocates coherent native cells and inline host reference metadata. Capacity
// is already admitted by queue construction. No cell is initially acquired.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t** out_storage);

// Frees inactive backing after every cell reader has released its reference.
void iree_hal_amd_gpu_pm4_signal_storage_free(
    const iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_
