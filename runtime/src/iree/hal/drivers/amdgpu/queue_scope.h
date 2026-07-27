// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable identity for one logical AMDGPU host queue.
//
// Executable-load paths use this cold-path metadata to publish queue-specific
// globals without adding per-dispatch host bookkeeping.
typedef struct iree_hal_amdgpu_queue_scope_t {
  // One-bit HAL queue affinity selecting this queue.
  iree_hal_queue_affinity_t queue_affinity;

  // Flattened logical queue ordinal in the owning HAL device.
  iree_host_size_t queue_ordinal;

  // Physical GPU device ordinal owning this queue.
  iree_host_size_t physical_device_ordinal;

  // Queue ordinal relative to |physical_device_ordinal|.
  iree_host_size_t physical_queue_ordinal;

  // Host-observed base address of the HSA AQL packet ring. This is metadata,
  // not a device-addressing contract.
  uint64_t aql_ring_base;

  // Power-of-two packet-ring slot mask for host packet IDs.
  uint64_t aql_ring_mask;

  // Queue-owned TSAN state facts used for executable-global publication.
  struct {
    // Device-visible iree_hal_amdgpu_tsan_queue_state_t pointer, or zero.
    uint64_t queue_state_base;
    // Device-visible base of queue-local shadow storage, or zero.
    uint64_t shadow_base;
    // Byte length of |shadow_base|.
    uint64_t shadow_size;
    // Bytes reserved for one dispatch shadow slot.
    uint64_t dispatch_shadow_stride;
    // Bytes reserved for one workgroup inside a dispatch shadow slot.
    uint64_t workgroup_shadow_stride;
    // Maximum workgroup ordinals represented by one dispatch shadow slot.
    uint32_t workgroup_capacity;
    // Bytes in each shadow entry.
    uint32_t shadow_entry_size;
    // Log2 application memory bytes represented by one shadow entry.
    uint32_t memory_granule_shift;
    // Number of queue-local dispatch shadow slots available.
    uint32_t shadow_slot_count;
  } tsan;
} iree_hal_amdgpu_queue_scope_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
