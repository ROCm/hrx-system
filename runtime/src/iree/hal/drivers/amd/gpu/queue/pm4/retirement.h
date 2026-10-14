// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RETIREMENT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RETIREMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// PM4-produced host action, committed after the associated execution and cache
// visibility. Four records occupy one 64-byte line. Native qualification of the
// backing establishes coherent host reads and system-atomic generation stores;
// declaring the generation atomic does not establish those device properties.
typedef struct iree_hal_amd_gpu_pm4_retirement_record_t {
  // Opaque non-NULL host action pointer bits. The GPU never dereferences them.
  uint64_t action;
  // Generation release-written by the GPU after the complete action payload.
  iree_atomic_int32_t generation;
  // Unused padding; neither publication nor observation accesses this word.
  uint32_t reserved;
} iree_hal_amd_gpu_pm4_retirement_record_t;

// Immutable view over queue-owned control backing. The single publisher and
// completion consumer own their separate monotonic positions and capacity
// credit; this view carries no shared mutable cursor or reference count.
typedef struct iree_hal_amd_gpu_pm4_retirement_ring_t {
  // Borrowed CPU-cacheable coherent records, live through final native use.
  iree_hal_amd_gpu_pm4_retirement_record_t* records;
  // GPU address of the first record in the same backing.
  uint64_t device_address;
  // Power-of-two capacity minus one, used for constant-time slot selection.
  uint32_t capacity_mask;
  // Log2 of capacity, used to derive a generation without division.
  uint32_t capacity_shift;
} iree_hal_amd_gpu_pm4_retirement_ring_t;

// Initializes a newly idle ring with producer and consumer positions zero.
// Capacity is an admitted nonzero power of two; the CPU and GPU extents contain
// that many naturally aligned records. Storage is exclusively owned here and
// becomes shared only after initialization. No allocation or native call
// occurs.
void iree_hal_amd_gpu_pm4_retirement_ring_initialize(
    iree_hal_amd_gpu_pm4_retirement_record_t* records, uint64_t device_address,
    uint32_t capacity, iree_hal_amd_gpu_pm4_retirement_ring_t* out_ring);

// Emits fourteen DWORDs: confirmed WRITE_DATA for the opaque action pointer,
// then RELEASE_MEM committing the generation after preceding compute and the
// prepared release cache control. The caller reserved credit for position and
// initialized the action before publishing these commands. No subsequent host
// read of the action is implied: immediate completion can transfer it to the
// consumer before native publication returns.
//
// The returned exclusive command end is the exact consumed prefix established
// by this record. Later notification packets and padding require their own
// consumption join. Notifications and payload barriers are not inferred here.
uint32_t* iree_hal_amd_gpu_pm4_retirement_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_retirement_ring_t* ring,
    uint64_t position, void* action, uint32_t release_cache_control);

// Returns the committed action at position, or NULL while that generation is
// absent. A successful acquire joins the native action write and the release's
// execution/visibility scope. Only the single completion consumer calls this.
// It captures the returned action, advances its position, and finishes reading
// the slot before returning capacity to the publisher. It never clears slots.
// Capacity credit prevents overwriting a slot still being consumed. The caller
// stops at the first unready position; later readiness does not prove a prefix.
void* iree_hal_amd_gpu_pm4_retirement_poll(
    const iree_hal_amd_gpu_pm4_retirement_ring_t* ring, uint64_t position);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RETIREMENT_H_
