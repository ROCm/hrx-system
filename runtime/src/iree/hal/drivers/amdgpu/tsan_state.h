// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/queue_scope.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_tsan_state_t
//===----------------------------------------------------------------------===//

// Memory policy used for TSAN state dereferenced by device code.
typedef struct iree_hal_amdgpu_tsan_memory_policy_t {
  // HSA memory pool used for TSAN state and shadow storage.
  hsa_amd_memory_pool_t memory_pool;
  // Agents granted explicit access to TSAN storage.
  const hsa_agent_t* access_agents;
  // Number of entries in |access_agents|.
  iree_host_size_t access_agent_count;
} iree_hal_amdgpu_tsan_memory_policy_t;

// Per-physical-device TSAN layout state.
typedef struct iree_hal_amdgpu_tsan_device_state_t {
  // Physical device ordinal associated with this TSAN layout.
  iree_host_size_t physical_device_ordinal;

  // Device-visible config published into executable globals.
  iree_hal_amdgpu_tsan_config_t config;
} iree_hal_amdgpu_tsan_device_state_t;

// Logical-device TSAN state shared by executable and queue paths.
typedef struct iree_hal_amdgpu_tsan_state_t {
  // True when TSAN support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for state-owned allocations.
  iree_allocator_t host_allocator;

  // Number of entries in |device_states|.
  iree_host_size_t device_state_count;

  // Per-physical-device TSAN layout records.
  iree_hal_amdgpu_tsan_device_state_t* device_states;
} iree_hal_amdgpu_tsan_state_t;

// Initializes |out_state| from logical-device options.
//
// When TSAN is disabled this leaves |out_state| zeroed and returns OK.
iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_tsan_state_t* out_state);

// Releases all TSAN resources owned by |state|.
void iree_hal_amdgpu_tsan_state_deinitialize(
    iree_hal_amdgpu_tsan_state_t* state);

// Initializes queue-owned TSAN state after host queues have been assigned.
//
// When TSAN is disabled this returns OK without modifying queues. On failure,
// queue TSAN state initialized by the call remains attached to the owning queue
// and is released by normal queue teardown after in-flight initialization work
// drains.
iree_status_t iree_hal_amdgpu_tsan_state_assign_queues(
    iree_hal_amdgpu_tsan_state_t* state, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices);

// Returns true when |state| owns enabled TSAN resources.
bool iree_hal_amdgpu_tsan_state_is_enabled(
    const iree_hal_amdgpu_tsan_state_t* state);

// Populates |out_config| with the device-visible TSAN configuration.
//
// Callers must only call this when TSAN is enabled.
// |physical_device_ordinal| selects the physical device whose executable global
// is being assigned.
iree_status_t iree_hal_amdgpu_tsan_state_populate_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_tsan_config_t* out_config);

// Populates |out_config| with queue-specific TSAN configuration.
//
// |queue_scope| describes the owning queue's AQL ring and queue-owned TSAN
// state. The scope is host-owned metadata; this function never reads device
// memory.
iree_status_t iree_hal_amdgpu_tsan_state_populate_queue_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    const iree_hal_amdgpu_queue_scope_t* queue_scope,
    iree_hal_amdgpu_tsan_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_
