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

// Logical-device TSAN state shared by executable and queue paths.
typedef struct iree_hal_amdgpu_tsan_state_t {
  // True when TSAN support is enabled for the logical device.
  uint32_t is_enabled : 1;
  // Queue-local shadow arena size in bytes.
  iree_device_size_t shadow_size;
} iree_hal_amdgpu_tsan_state_t;

// Initializes |out_state| from logical-device options.
//
// When TSAN is disabled this leaves |out_state| zeroed and returns OK.
iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_tsan_state_t* out_state);

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

// Populates |out_config| with queue-specific TSAN configuration.
//
// The enabled state and initialized queue shadow are construction invariants.
void iree_hal_amdgpu_tsan_state_populate_queue_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    const iree_hal_amdgpu_queue_scope_t* queue_scope,
    iree_hal_amdgpu_tsan_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_
