// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CAPTURE_POOL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CAPTURE_POOL_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_amd_gpu_capture_block_t;

// Queue-owned host storage for individually retained operation captures.
// Callers acquire independently of final releases on
// publisher/service/retirement threads. Returned blocks move to the acquisition
// list only when it is empty. Links occupy dead payload, so live captures may
// use the complete allocation. Neither list owns a reference to its queue; the
// queue joins every user before deinitialization. Acquire, release,
// preallocate, and trim are thread-safe; initialization and deinitialization
// are exclusive.
typedef struct iree_hal_amd_gpu_capture_pool_t {
  // Immutable usable bytes in every reusable allocation, at least
  // sizeof(void*).
  iree_host_size_t block_size;
  // Thread-safe allocator for empty-pool misses and oversized captures.
  iree_allocator_t host_allocator;
  // Caller-owned available blocks, isolated from concurrent returns.
  struct iree_alignas(64) {
    // Protects head and the transfer from returned, never system allocation.
    iree_slim_mutex_t mutex;
    // Unowned blocks available to callers, or NULL before the next transfer.
    struct iree_hal_amd_gpu_capture_block_t* head;
  } available;
  // Final releases accumulate here without modifying caller-owned metadata.
  struct iree_alignas(64) {
    // Protects head; nested locking always takes available.mutex first.
    iree_slim_mutex_t mutex;
    // Unowned blocks returned by any final-release thread.
    struct iree_hal_amd_gpu_capture_block_t* head;
  } returned;
} iree_hal_amd_gpu_capture_pool_t;

// Initializes embedded state using the queue's already-validated block size.
// The host allocator outlives this pool and every outstanding allocation.
void iree_hal_amd_gpu_capture_pool_initialize(
    iree_host_size_t block_size, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_capture_pool_t* out_pool);

// Frees all unused storage after every acquired allocation has been returned
// and all concurrent calls have joined. This does not join capture lifetimes.
void iree_hal_amd_gpu_capture_pool_deinitialize(
    iree_hal_amd_gpu_capture_pool_t* pool);

// Adds count reusable blocks. Allocation failure frees the partial batch and
// leaves the pool unchanged. Allocation and rollback run outside list locks.
iree_status_t iree_hal_amd_gpu_capture_pool_preallocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t count);

// Returns unowned blocks to the host allocator without touching live captures.
// Blocks released after detachment remain reusable until a subsequent trim.
// Detachment takes bounded work; system frees happen outside both list locks.
void iree_hal_amd_gpu_capture_pool_trim(iree_hal_amd_gpu_capture_pool_t* pool);

// Acquires at least minimum_size uninitialized bytes, with system allocation
// alignment. Ordinary captures receive one whole reusable block; oversized
// captures receive exactly minimum_size bytes. The returned extent must be
// preserved for release. Failure leaves out_storage empty.
iree_status_t iree_hal_amd_gpu_capture_pool_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t minimum_size,
    iree_byte_span_t* out_storage);

// Returns an allocation and its unchanged extent from acquire. Every user of
// these bytes must have joined; release immediately reuses the first word.
// Oversized allocations return directly to the host allocator.
void iree_hal_amd_gpu_capture_pool_release(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CAPTURE_POOL_H_
