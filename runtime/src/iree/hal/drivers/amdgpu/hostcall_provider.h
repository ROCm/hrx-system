// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOSTCALL_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOSTCALL_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_thread_t iree_thread_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// HAL-owned transport and lifecycle state for one opaque hostcall provider.
typedef struct iree_hal_amdgpu_hostcall_provider_state_t {
  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for this state.
  iree_allocator_t host_allocator;

  // Immutable provider callbacks copied from the creation extension.
  iree_hal_hostcall_provider_t provider;

  // Provider-owned context initialized over |shared_memory|.
  void* provider_context;

  // HAL-owned host pointer to the shared allocation.
  void* shared_memory;

  // Provider-visible byte length of |shared_memory|.
  iree_host_size_t shared_memory_size;

  // Stable device-visible address of |shared_memory|.
  uint64_t device_address;

  // HSA signal used by device producers to wake |service_thread|.
  hsa_signal_t notification_signal;

  // Thread exclusively servicing |provider_context|.
  iree_thread_t* service_thread;

  // True when the listener should exit after its final service pass.
  iree_atomic_int32_t stop_requested;
} iree_hal_amdgpu_hostcall_provider_state_t;

// Creates an eager physical-device hostcall provider state.
iree_status_t iree_hal_amdgpu_hostcall_provider_state_create(
    const iree_hal_hostcall_provider_t* provider,
    iree_hal_device_t* logical_device, const iree_hal_amdgpu_libhsa_t* libhsa,
    hsa_agent_t device_agent, hsa_amd_memory_pool_t shared_memory_pool,
    iree_host_size_t host_numa_node,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_hostcall_provider_state_t** out_state);

// Stops the listener and releases one provider state.
void iree_hal_amdgpu_hostcall_provider_state_destroy(
    iree_hal_amdgpu_hostcall_provider_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOSTCALL_PROVIDER_H_
