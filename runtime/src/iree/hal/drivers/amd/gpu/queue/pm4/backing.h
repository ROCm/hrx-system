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
// A replaced generation is released only after its final byte position has
// retired. Native readers of signals have a separate lifetime below.
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
  // Release descriptor acquired with this generation, never at retirement.
  iree_hal_amd_gpu_memory_release_t release;
  // Host allocator owning this metadata through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_pm4_argument_storage_t;

typedef struct iree_hal_amd_gpu_pm4_signal_storage_t {
  // Next backing, retained while active or referenced by native readers.
  struct iree_hal_amd_gpu_pm4_signal_storage_t* next;
  // Native cells and coherent CPU mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Finite extent attached to the queue's available signal pool.
  iree_hal_amd_gpu_pm4_signal_block_t block;
  // Number of native cells and corresponding host signal records.
  uint32_t capacity;
  // Native-domain frontier width for each signal record.
  uint32_t queue_count;
  // Prepared inline metadata, initialized after native backing arrives.
  iree_hal_amd_gpu_pm4_signal_t* signals;
  // Prepared inline frontier vectors, one per signal record.
  uint64_t* frontiers;
  // Release descriptor acquired with this storage, never at retirement.
  iree_hal_amd_gpu_memory_release_t release;
  // Host allocator owning this metadata through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_pm4_signal_storage_t;

// Allocates coherent write-back storage for PM4 commands, arguments, or control
// cells under the caller's admitted memory policy. A non-NULL out_address asks
// for its GPU address at access ordinal zero. Failure may leave partial memory;
// the caller always deinitializes it. This cold constructor waits for its typed
// native acquisition and is never called by publication or retirement.
iree_status_t iree_hal_amd_gpu_pm4_backing_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address);

// Allocates only host ownership and the native release descriptor. The caller
// acquires storage->memory asynchronously, then initializes its address and
// monotonic extent on the publication owner. Failure returns NULL storage.
iree_status_t iree_hal_amd_gpu_pm4_argument_storage_prepare(
    uint64_t capacity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t** out_storage);
iree_status_t iree_hal_amd_gpu_pm4_argument_storage_initialize(
    iree_hal_amd_gpu_context_t* context, uint64_t begin,
    iree_hal_amd_gpu_pm4_argument_storage_t* storage);

// Allocates an independent generation at the admitted monotonic begin position.
// The caller links and retires generations by their final reserved byte extent.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_pm4_argument_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t** out_storage);

// Transfers a retired or partially acquired generation to native release.
// List ownership and the exact consumed-byte proof stay with the queue. This
// performs no native call or wait, and storage cannot be accessed afterward.
void iree_hal_amd_gpu_pm4_argument_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_argument_storage_t* storage);

// Prepares host metadata and release ownership without acquiring native cells.
// After typed acquisition fills storage->memory, initialize publishes its
// already-admitted coherent mapping to the finite signal pool.
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_prepare(
    uint32_t capacity, uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t** out_storage);
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_initialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_signal_pool_t* pool,
    iree_hal_amd_gpu_pm4_signal_storage_t* storage);

// Allocates coherent native cells, inline host reference metadata, and a dense
// queue_count-entry frontier per cell. Both counts are already admitted by
// device/queue construction. No cell is initially acquired.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_signal_pool_t* pool,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t** out_storage);

// Transfers inactive backing after every cell reader has released its
// reference. Also accepts partially acquired storage. No native call or wait
// occurs here.
void iree_hal_amd_gpu_pm4_signal_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_signal_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_
