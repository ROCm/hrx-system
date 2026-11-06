// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_SYSTEM_H_
#define IREE_HAL_DRIVER_AMDGPU_SYSTEM_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amdgpu/host_memory_pools.h"
#include "iree/hal/driver/amdgpu/util/agent_target.h"
#include "iree/hal/driver/amdgpu/util/device_clock.h"
#include "iree/hal/driver/amdgpu/util/device_library.h"
#include "iree/hal/driver/amdgpu/util/info.h"
#include "iree/hal/driver/amdgpu/util/libhsa.h"
#include "iree/hal/driver/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

// Options defining total system behavior.
typedef struct iree_hal_amdgpu_system_options_t {
  // Force queues to run one entry at a time instead of overlapping or
  // aggressively scheduling queue entries out-of-order.
  uint64_t exclusive_execution : 1;
} iree_hal_amdgpu_system_options_t;

// An initialized AMDGPU system topology and agent resources.
// This is the authoritative list of available devices and their configuration.
//
// Thread-safe; systems are immutable once initialized.
typedef struct iree_hal_amdgpu_system_t {
  // Allocator used for host operations.
  iree_allocator_t host_allocator;

  // HSA API handle.
  iree_hal_amdgpu_libhsa_t libhsa;

  // Platform source used for device/host clock-correlation sampling.
  iree_hal_amdgpu_device_clock_source_t device_clock_source;

  // System topology as visible to the HAL device. This may be a subset of
  // the devices available in the system.
  iree_hal_amdgpu_topology_t topology;

  // Immutable target identities corresponding to |topology.gpu_agents|.
  iree_hal_amdgpu_agent_target_t
      gpu_agent_targets[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

  // Cached system information queried from HSA or the platform.
  iree_hal_amdgpu_system_info_t info;

  // Options configuring system behavior.
  iree_hal_amdgpu_system_options_t options;

  // Loaded device library. Instantiated on all GPU devices in the topology.
  iree_hal_amdgpu_device_library_t device_library;

  // Memory pools for shared CPU/GPU agent access.
  iree_hal_amdgpu_host_memory_pools_t host_memory_pools[/*cpu_agent_count*/];
} iree_hal_amdgpu_system_t;

// Allocates a system in |out_system| with the given |topology|.
// The provided |libhsa| and |topology| will be copied and a reference to the
// HSA library will be retained for the lifetime of the system. Assumes that
// |topology| has been verified as valid.
iree_status_t iree_hal_amdgpu_system_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t** out_system);

// Frees |system| and releases any held resources.
void iree_hal_amdgpu_system_free(iree_hal_amdgpu_system_t* system);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_SYSTEM_H_
