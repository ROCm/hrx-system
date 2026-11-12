// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SAMPLE_STORAGE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SAMPLE_STORAGE_H_

#include "iree/hal/driver/amd/gpu/queue/backing.h"

#ifdef __cplusplus
extern "C" {
#endif

// Timestamp storage owned by one queue's profiling session. Native completion
// permits collection, but only completed host collection returns sample credit.
// The exclusive publisher owns generations and reservations; retirement alone
// advances consumed. Unprofiled queues have no sample owner or native backing.
typedef struct iree_hal_amd_gpu_sample_storage_t {
  // Current native sample generation, owned through session teardown.
  iree_hal_amd_gpu_byte_storage_t* current;
  // Replaced generations retained through their final host sample reader.
  iree_hal_amd_gpu_byte_storage_t* retired;
  // Monotonic byte prefix collected by host retirement, including padding.
  iree_atomic_int64_t consumed;
  // One prepared KERNEL command's exact sample generation, or NULL.
  iree_hal_amd_gpu_byte_storage_t* pending;
  // Cursor before that unaccepted native command borrowed pending.
  uint64_t pending_begin;
} iree_hal_amd_gpu_sample_storage_t;

// Cold session setup prepares a power-of-two direct-dispatch sample window.
// Queue/session ownership joins native acquisitions and host retirement before
// destruction. Native releases use the existing allocation domain.
iree_status_t iree_hal_amd_gpu_sample_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* memory, uint64_t capacity,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sample_storage_t** out_storage);
void iree_hal_amd_gpu_sample_storage_free(
    iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sample_storage_t* storage);

typedef enum iree_hal_amd_gpu_sample_reservation_e {
  IREE_HAL_AMD_GPU_SAMPLE_RESERVED,
  IREE_HAL_AMD_GPU_SAMPLE_CAPACITY,
  IREE_HAL_AMD_GPU_SAMPLE_GROW,
} iree_hal_amd_gpu_sample_reservation_t;

// Calculates a contiguous reservation without changing publication state.
// Length and alignment come from the native protocol's trusted sample layout.
iree_hal_amd_gpu_sample_reservation_t iree_hal_amd_gpu_sample_storage_reserve(
    iree_hal_amd_gpu_sample_storage_t* storage, uint64_t length,
    uint64_t alignment, uint64_t* out_offset, uint64_t* out_end);

// Commits a prepared sample extent. KERNEL publication retains an exact-owner
// checkpoint until acceptance; USER publication hands ownership to retirement.
void iree_hal_amd_gpu_sample_storage_commit(
    iree_hal_amd_gpu_sample_storage_t* storage, uint64_t end, bool pending);
// NULL accepts metadata-only work with no native sample owner.
void iree_hal_amd_gpu_sample_storage_accept(
    iree_hal_amd_gpu_sample_storage_t* storage);
void iree_hal_amd_gpu_sample_storage_rollback(
    iree_hal_amd_gpu_sample_storage_t* storage);

// Installs an acquired generation whose begin is the prior current end.
void iree_hal_amd_gpu_sample_storage_install(
    iree_hal_amd_gpu_sample_storage_t* storage,
    iree_hal_amd_gpu_byte_storage_t* generation);
// Returns replaced generations only after the host has collected their samples.
void iree_hal_amd_gpu_sample_storage_trim(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_sample_storage_t* storage);

// Called after copying native samples into the bounded host profile stream.
// Zero denotes metadata-only work and requires no native sample owner.
static inline void iree_hal_amd_gpu_sample_storage_retire(
    iree_hal_amd_gpu_sample_storage_t* storage, uint64_t end) {
  if (end) {
    iree_atomic_store(&storage->consumed, end, iree_memory_order_release);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SAMPLE_STORAGE_H_
