// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_BYTE_STORAGE_POOL_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_BYTE_STORAGE_POOL_H_

#include "iree/hal/driver/amd/gpu/queue/backing.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independently owned byte generations for one queue use and memory policy.
// The exclusive publisher owns reservations and generation changes. The caller
// advances consumed only after its last native or host reader; native
// completion alone does not release bytes still needed by a host collector.
typedef struct iree_hal_amd_gpu_byte_storage_pool_t {
  // Current native generation, owned until all readers and acquisitions join.
  iree_hal_amd_gpu_byte_storage_t* current;
  // Replaced generations retained through their final consumer.
  iree_hal_amd_gpu_byte_storage_t* retired;
  // Monotonic byte prefix returned by the last consumer, including padding.
  iree_atomic_int64_t consumed;
  // One prepared KERNEL command's exact generation, or NULL.
  iree_hal_amd_gpu_byte_storage_t* pending;
  // Cursor before that unaccepted native command borrowed pending.
  uint64_t pending_begin;
} iree_hal_amd_gpu_byte_storage_pool_t;

// Cold setup prepares a power-of-two byte window with the caller's policy.
// Owner destruction joins native acquisitions and every last-reader return.
// Native releases use the existing allocation domain.
iree_status_t iree_hal_amd_gpu_byte_storage_pool_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* memory, uint64_t capacity,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_byte_storage_pool_t** out_storage);
void iree_hal_amd_gpu_byte_storage_pool_free(
    iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_byte_storage_pool_t* storage);

typedef enum iree_hal_amd_gpu_byte_reservation_e {
  IREE_HAL_AMD_GPU_BYTE_RESERVED,
  IREE_HAL_AMD_GPU_BYTE_CAPACITY,
  IREE_HAL_AMD_GPU_BYTE_GROW,
} iree_hal_amd_gpu_byte_reservation_t;

// Calculates a contiguous reservation without changing publication state.
// Length and alignment come from the consumer's trusted layout.
iree_hal_amd_gpu_byte_reservation_t iree_hal_amd_gpu_byte_storage_pool_reserve(
    iree_hal_amd_gpu_byte_storage_pool_t* storage, uint64_t length,
    uint64_t alignment, uint64_t* out_offset, uint64_t* out_end);

// Commits a prepared extent. KERNEL publication retains an exact-owner
// checkpoint until acceptance; USER publication hands ownership to retirement.
void iree_hal_amd_gpu_byte_storage_pool_commit(
    iree_hal_amd_gpu_byte_storage_pool_t* storage, uint64_t end, bool pending);
// NULL accepts metadata-only work with no native byte owner.
void iree_hal_amd_gpu_byte_storage_pool_accept(
    iree_hal_amd_gpu_byte_storage_pool_t* storage);
void iree_hal_amd_gpu_byte_storage_pool_rollback(
    iree_hal_amd_gpu_byte_storage_pool_t* storage);

// Installs an acquired generation whose begin is the prior current end.
void iree_hal_amd_gpu_byte_storage_pool_install(
    iree_hal_amd_gpu_byte_storage_pool_t* storage,
    iree_hal_amd_gpu_byte_storage_t* generation);
// Returns replaced generations only after their final consumer returns credit.
void iree_hal_amd_gpu_byte_storage_pool_trim(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_byte_storage_pool_t* storage);

// Returns a prefix after its final reader completes. Zero denotes work with
// no bytes in this owner and requires no native allocation.
static inline void iree_hal_amd_gpu_byte_storage_pool_retire(
    iree_hal_amd_gpu_byte_storage_pool_t* storage, uint64_t end) {
  if (end) {
    iree_atomic_store(&storage->consumed, end, iree_memory_order_release);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_BYTE_STORAGE_POOL_H_
