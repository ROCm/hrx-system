// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_IPC_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_IPC_MEMORY_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

// Allocator-owned dependencies borrowed for one typed IPC-memory operation.
typedef struct iree_hal_amdgpu_ipc_memory_allocator_context_t {
  // HSA dispatch table retained by the allocator's logical device.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Logical topology used to resolve the immutable IPC mapping-agent set.
  const iree_hal_amdgpu_topology_t* topology;

  // Host allocator used for attachment-lifetime metadata.
  iree_allocator_t host_allocator;
} iree_hal_amdgpu_ipc_memory_allocator_context_t;

// ROCr allocation facts required to create a typed IPC-memory descriptor.
typedef struct iree_hal_amdgpu_ipc_memory_export_range_t {
  // Base address of the requested shareable allocation in the exporting
  // agent's VA.
  void* agent_base;

  // Requested ROCr allocation extent used by IPC create and attach. Internal
  // physical padding is not included.
  iree_device_size_t allocation_size;

  // HSA memory-pool global flags reported for the allocation.
  uint32_t global_flags;
} iree_hal_amdgpu_ipc_memory_export_range_t;

// Validates |base_allocator| and returns its dependencies borrowed for the
// duration of the calling typed IPC-memory operation.
iree_status_t iree_hal_amdgpu_allocator_query_ipc_memory_context(
    iree_hal_allocator_t* base_allocator,
    iree_hal_amdgpu_ipc_memory_allocator_context_t* out_context);

// Validates that |params| resolve to an importable allocation placement.
iree_status_t iree_hal_amdgpu_allocator_validate_ipc_memory_import_params(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params);

// Resolves the device-coarse source pool selected by exactly one exporting
// queue family.
iree_status_t iree_hal_amdgpu_allocator_resolve_ipc_memory_export_pool(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    hsa_amd_memory_pool_t* out_memory_pool);

// Validates dedicated device-coarse allocator provenance and queries the
// complete ROCr allocation containing |device_ptr, size|.
iree_status_t iree_hal_amdgpu_allocator_query_ipc_memory_export_range(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* allocated_buffer,
    uint64_t device_ptr, iree_device_size_t size,
    iree_hal_amdgpu_ipc_memory_export_range_t* out_range);

// Wraps an already attached IPC mapping without changing its immutable ROCr
// agent-access set. When |owns_attachment| is true, the resulting buffer is
// tagged as the sole attachment owner for later alias validation.
iree_status_t iree_hal_amdgpu_allocator_wrap_attached_ipc_memory(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params,
    const iree_hal_external_buffer_t* external_buffer, bool owns_attachment,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

// Returns the attachment release callback stored by a live, sole-owner IPC
// buffer. Returns false for aliases and ordinary AMDGPU buffers.
bool iree_hal_amdgpu_allocator_query_ipc_memory_attachment(
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_release_callback_t* out_release_callback);

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_IPC_MEMORY_H_
