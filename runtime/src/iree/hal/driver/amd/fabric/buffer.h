// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_FABRIC_BUFFER_H_
#define IREE_HAL_DRIVER_AMD_FABRIC_BUFFER_H_

#include "iree/hal/api.h"
#include "iree/hal/driver/amd/fabric/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepared native storage borrowed by one materialized buffer view. The memory
// owner, mapping, and binding table must outlive the returned buffer.
typedef struct iree_hal_amd_fabric_buffer_storage_t {
  // Native allocation and optional host view used by public map calls.
  iree_hal_amd_fabric_memory_t* memory;
  // Complete native binding table for byte zero of the owning allocation.
  const iree_hal_buffer_native_binding_t* bindings;
  // Optional host execution slot (NONE when absent) in |bindings|.
  uint16_t host_binding_index;
  // Byte position of this view within the owning allocation.
  iree_device_size_t offset;
} iree_hal_amd_fabric_buffer_storage_t;

// Creates a HAL buffer view over provider-owned native storage. Destruction
// invokes |release_callback| and frees only the wrapper; the callback owner
// controls the storage lifetime. Failure leaves |out_buffer| NULL.
iree_status_t iree_hal_amd_fabric_buffer_wrap(
    iree_hal_buffer_placement_t placement, const amdf_api_t* api,
    iree_hal_amd_fabric_buffer_storage_t storage,
    iree_device_size_t allocation_size, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Acquires device writes for a prepared host execution binding. The caller has
// resolved producer dependencies and validated this buffer-relative range.
// Public mapping grants are independent of the implementation-owned binding.
// Cache maintenance may block and belongs on the host execution service.
iree_status_t iree_hal_amd_fabric_buffer_invalidate_host_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length);

// Publishes writes through a prepared host execution binding before signaling
// dependent device work. Uses the same range and ownership contract as acquire.
iree_status_t iree_hal_amd_fabric_buffer_flush_host_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_FABRIC_BUFFER_H_
