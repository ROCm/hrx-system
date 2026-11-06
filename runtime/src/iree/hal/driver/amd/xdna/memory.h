// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_MEMORY_H_
#define IREE_HAL_DRIVER_AMD_XDNA_MEMORY_H_

#include "iree/hal/driver/amd/fabric/memory.h"
#include "iree/hal/driver/amd/xdna/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// XDNA-private storage with the exact address consumed by firmware or shim DMA.
// The native context owns the source scope and outlives this storage.
typedef struct iree_hal_amd_xdna_memory_t {
  // Common backing and persistent host mapping.
  iree_hal_amd_fabric_memory_t storage;
  // Address in the source's selected execution domain.
  uint64_t device_address;
} iree_hal_amd_xdna_memory_t;

// Allocates native backing and establishes its map/address before publication.
// Failure leaves |out_memory| empty with no cleanup obligation.
iree_status_t iree_hal_amd_xdna_memory_allocate(
    iree_hal_amd_xdna_context_t* context,
    const iree_hal_amd_xdna_memory_source_t* source, uint64_t byte_length,
    uint64_t alignment, iree_hal_amd_xdna_memory_t* out_memory);

// Releases native ownership after all users retire. Cleanup failures are
// reported through the context's sink and never abort the application.
void iree_hal_amd_xdna_memory_deinitialize(iree_hal_amd_xdna_context_t* context,
                                           iree_hal_amd_xdna_memory_t* memory);

// Creates the allocation metadata facade used for heap discovery and VMM.
// The context is borrowed and outlives the allocator.
iree_status_t iree_hal_amd_xdna_allocator_create(
    iree_hal_amd_xdna_context_t* context, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Captures the stable shim-DMA slot prepared for |family|'s program site.
// Contract-backed buffers may come from any driver in the same memory domain.
// Every buffer must carry a prepared contract. This performs no native
// registration and does not read a potentially unpublished
// slot value.
iree_status_t iree_hal_amd_xdna_buffer_resolve_binding_slot(
    const iree_hal_queue_family_t* family, const iree_hal_buffer_t* buffer,
    iree_hal_buffer_native_binding_slot_t* out_slot);

// Loads the shim-DMA address after the caller's allocation prerequisites have
// resolved. The captured slot and binding table storage are stable for the
// allocation epoch; commitment publishes the payload before this read.
iree_status_t iree_hal_amd_xdna_buffer_load_device_address(
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_buffer_native_binding_slot_t slot, uint64_t* out_device_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_MEMORY_H_
