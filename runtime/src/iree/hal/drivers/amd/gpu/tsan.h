// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
#define IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold geometry for the existing workgroup-race shadow ABI.
typedef struct iree_hal_amd_gpu_tsan_options_t {
  // Log2 application bytes represented by one eight-byte shadow entry.
  uint32_t memory_granule_shift;
  // Workgroup LDS coverage; zero uses the native device limit.
  uint32_t workgroup_local_memory_size;
  // Maximum workgroup ordinals represented by each dispatch slot.
  uint32_t workgroup_capacity;
  // Power-of-two dispatch window; publication joins before reusing its slots.
  uint32_t shadow_slot_count;
} iree_hal_amd_gpu_tsan_options_t;

// Initializes the existing four-byte granule, 256-workgroup, 16-dispatch
// layout.
void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options);

// Runtime-published TSAN configuration read by instrumented device code.
//
// Queue-scoped executable variants publish queue-specific shadow storage so
// device code can derive a queue-local dispatch slot from the dispatch ID
// without host per-dispatch mutation.
typedef struct iree_hal_amd_gpu_tsan_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Enabled (bit 0) and queue-state (bit 1) runtime facilities.
  uint32_t flags;
  // Log2 application memory bytes represented by one shadow entry.
  uint32_t memory_granule_shift;
  // Device-visible base of the queue or device shadow allocation.
  uint64_t shadow_base;
  // Total bytes in the shadow allocation.
  uint64_t shadow_size;
  // Bytes reserved for one dispatch slot.
  uint64_t dispatch_shadow_stride;
  // Bytes reserved for one workgroup inside a dispatch slot.
  uint64_t workgroup_shadow_stride;
  // Maximum workgroup ordinals represented by one dispatch slot.
  uint32_t workgroup_capacity;
  // Bytes in each shadow entry.
  uint32_t shadow_entry_size;
  // Host-observed base of the owning queue AQL ring, or 0 when unavailable.
  // This is metadata, not a device-addressing contract.
  uint64_t queue_aql_base;
  // Power-of-two AQL packet slot mask for host packet IDs.
  uint64_t queue_aql_slot_mask;
  // Device-visible iree_hal_amd_gpu_tsan_queue_state_t pointer, or zero.
  uint64_t queue_state_base;
  // Number of queue-local dispatch shadow slots available.
  uint32_t shadow_slot_count;
  // Reserved for future TSAN runtime state. Must be zero.
  uint32_t reserved0;
  // Reserved for future TSAN runtime state. Must be zero.
  uint64_t reserved1;
} iree_hal_amd_gpu_tsan_config_t;
static_assert(sizeof(iree_hal_amd_gpu_tsan_config_t) == 96,
              "TSAN config size is part of the device ABI");
static_assert(offsetof(iree_hal_amd_gpu_tsan_config_t, shadow_base) == 16,
              "TSAN config shadow fields must follow the fixed header");
static_assert(
    offsetof(iree_hal_amd_gpu_tsan_config_t, queue_state_base) == 72,
    "TSAN config queue state fields must follow the queue AQL fields");
static_assert(offsetof(iree_hal_amd_gpu_tsan_config_t, reserved1) == 88,
              "TSAN config reserved tail must keep the fixed layout");

// One queue's native shadow owner, retained by its parent device until every
// queue and executable cleanup has joined. Ordinary devices allocate no state.
typedef struct iree_hal_amd_gpu_tsan_t {
  // Native backing for the immutable queue header and mutable shadow entries.
  iree_hal_amd_fabric_memory_t memory;
  // Immutable shader configuration copied into this queue's executable variant.
  iree_hal_amd_gpu_tsan_config_t config;
} iree_hal_amd_gpu_tsan_t;

// Validates public geometry and computes the native-device layout without
// allocating. Addresses remain zero until initialize establishes backing.
iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    uint32_t workgroup_local_memory_size,
    iree_hal_amd_gpu_tsan_config_t* out_config);

// Allocates and publishes one queue's shadow on the existing device service.
// Requires physical-local, host-visible backing for initialization. Native
// accesses use this context's GPU address domain. Failure leaves
// deinitializable state; the parent joins GPU uses before deinitialization.
iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_tsan_options_t* options, uint32_t queue_ordinal,
    iree_hal_amd_gpu_tsan_t* out_tsan);

// Releases native backing after queue execution and executable cleanup join.
void iree_hal_amd_gpu_tsan_deinitialize(
    const iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_tsan_t* tsan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
