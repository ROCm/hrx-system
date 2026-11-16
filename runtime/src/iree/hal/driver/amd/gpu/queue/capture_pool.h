// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_CAPTURE_POOL_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_CAPTURE_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_amd_gpu_capture_block_t;

enum { IREE_HAL_AMD_GPU_CAPTURE_CLASS_COUNT = sizeof(iree_host_size_t) * 8 };

// Queue-owned host storage for individually retained operation captures.
// Callers acquire independently of final releases on
// publisher/service/retirement threads. Returned blocks move to the acquisition
// list only when it is empty. Each size class uses backing slabs; links occupy
// dead payload, so live captures may use their complete block. Neither list
// owns a reference to its queue; the queue joins every user before
// deinitialization. Acquire, release, preallocate, and trim are thread-safe;
// initialization and deinitialization are exclusive.
typedef struct iree_hal_amd_gpu_capture_pool_t {
  // Immutable base-class extent, at least sizeof(void*). Larger classes use
  // power-of-two extents and preserve the whole extent through release.
  iree_host_size_t block_size;
  // Thread-safe allocator for backing slabs, called only during growth/trim.
  iree_allocator_t host_allocator;
  // Caller-owned available blocks, isolated from concurrent returns.
  struct iree_alignas(iree_hardware_destructive_interference_size) {
    // Protects head and the transfer from returned, never system allocation.
    iree_slim_mutex_t mutex;
    // Available head for each size class; class zero is the configured base.
    struct iree_hal_amd_gpu_capture_block_t*
        head[IREE_HAL_AMD_GPU_CAPTURE_CLASS_COUNT];
  } available;
  // Final releases accumulate here without modifying caller-owned metadata.
  struct iree_alignas(iree_hardware_destructive_interference_size) {
    // Protects head; nested locking always takes available.mutex first.
    iree_slim_mutex_t mutex;
    // Returned head for each size class, populated by final-release threads.
    struct iree_hal_amd_gpu_capture_block_t*
        head[IREE_HAL_AMD_GPU_CAPTURE_CLASS_COUNT];
  } returned;
  // Serializes detached-slab accounting between concurrent cold trims. Capture
  // acquisition and retirement never acquire this mutex.
  iree_slim_mutex_t trim_mutex;
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

// Adds count reusable base-class blocks in shared slabs. Allocation failure
// frees the batch and leaves the pool unchanged. Allocation and rollback run
// outside list locks.
iree_status_t iree_hal_amd_gpu_capture_pool_preallocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t count);

// Returns wholly unused slabs to the host allocator without touching live
// captures. Free blocks sharing a slab with live captures remain reusable.
// Blocks released after detachment participate in a subsequent trim. Detachment
// takes bounded work; slab accounting and frees happen outside both list locks.
void iree_hal_amd_gpu_capture_pool_trim(iree_hal_amd_gpu_capture_pool_t* pool);

// Acquires at least minimum_size uninitialized bytes, with system allocation
// alignment. An empty size class grows by a slab of reusable blocks. Requests
// up to block_size use the base class; larger requests use the next
// power-of-two extent. The complete returned extent must be preserved for
// release. Failure leaves out_storage empty.
iree_status_t iree_hal_amd_gpu_capture_pool_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t minimum_size,
    iree_byte_span_t* out_storage);

// Returns an allocation and its unchanged extent from acquire. Every user of
// these bytes must have joined; release immediately reuses the first word.
// All extents remain reusable until whole-slab trim reclaims their backing.
void iree_hal_amd_gpu_capture_pool_release(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_CAPTURE_POOL_H_
