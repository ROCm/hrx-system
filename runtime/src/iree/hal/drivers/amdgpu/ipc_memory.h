// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_IPC_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_IPC_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Size of the process-independent ROCr IPC memory token.
#define IREE_HAL_AMDGPU_IPC_MEMORY_TOKEN_SIZE 32

// Opaque process-independent ROCr IPC memory token.
//
// The bytes are transported by value and must not be interpreted or rewritten.
// A token is only valid while the exporting process keeps the referenced ROCr
// allocation alive.
typedef struct iree_hal_amdgpu_ipc_memory_token_t {
  // Opaque ROCr hsa_amd_ipc_memory_t bytes.
  uint32_t data[IREE_HAL_AMDGPU_IPC_MEMORY_TOKEN_SIZE / sizeof(uint32_t)];
} iree_hal_amdgpu_ipc_memory_token_t;

// Process-independent description of a shareable AMDGPU allocation and the
// origin of an exported view within it. The view length is not encoded and
// must be transported separately by the caller.
typedef struct iree_hal_amdgpu_ipc_memory_descriptor_t {
  // ROCr token identifying the requested shareable allocation.
  iree_hal_amdgpu_ipc_memory_token_t token;
  // Requested allocation extent passed to IPC create and attach. Internal
  // physical padding is not included.
  uint64_t allocation_size;
  // Offset of the exported buffer view from the shared allocation base.
  uint64_t byte_offset;
} iree_hal_amdgpu_ipc_memory_descriptor_t;

// Exports |buffer| as a process-independent ROCr IPC memory descriptor. The
// descriptor identifies the view origin; callers can obtain its separate
// length with iree_hal_buffer_byte_length().
//
// |allocator| must be the AMDGPU allocator that created |buffer| as a
// dedicated, cached device-coarse allocation with
// IREE_HAL_BUFFER_USAGE_SHARING_EXPORT. Fine-grained, host-visible, uncached,
// pooled, and externally imported allocations cannot be exported. The source
// allocation must remain live while another process may import
// |out_descriptor|.
iree_status_t iree_hal_amdgpu_ipc_memory_export(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_ipc_memory_descriptor_t* out_descriptor);

// Attaches |descriptor| and imports its
// [byte_offset, byte_offset + view_size) view into |allocator|.
//
// |allocator| and |exporting_allocator| must be AMDGPU allocators in the same
// process. |exporting_queue_family_affinity| must select the one local device
// corresponding to the exported handle's device ordinal. The imported buffer
// owns the ROCr attachment and detaches it when released. The mapping is
// attached to the importing GPU agents and every visible GPU peer that can
// access the exporter's device-local memory, allowing compatible logical
// devices to create aliases without granting access to unrelated agents.
iree_status_t iree_hal_amdgpu_ipc_memory_import(
    iree_hal_allocator_t* allocator, iree_hal_allocator_t* exporting_allocator,
    iree_hal_queue_family_affinity_t exporting_queue_family_affinity,
    iree_hal_buffer_params_t params,
    const iree_hal_amdgpu_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

// Creates a destination-allocator alias for an existing attached IPC mapping.
//
// |attached_buffer| must have been returned by
// iree_hal_amdgpu_ipc_memory_import(). It remains the sole owner of the
// process-wide attachment and must outlive the returned alias. Releasing the
// alias performs per-device import bookkeeping only and never detaches the
// ROCr mapping. Every GPU selected by |params| must be in the original
// attachment's immutable mapping-agent set.
iree_status_t iree_hal_amdgpu_ipc_memory_import_alias(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* attached_buffer, iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_IPC_MEMORY_H_
