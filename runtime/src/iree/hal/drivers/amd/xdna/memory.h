// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_

#include "iree/hal/drivers/amd/xdna/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// One provider-owned allocation with a persistent host mapping and a prepared
// native address. The context outlives this owner and every device use.
typedef struct iree_hal_amd_xdna_memory_t {
  // Owned native storage, consumed at deinitialization.
  amdf_memory_t* handle;
  // Owned host view, destroyed before the native storage.
  amdf_host_mapping_t* mapping;
  // Borrowed host pointer and exact logical extent.
  iree_byte_span_t contents;
  // Address in the allocation source's declared native domain.
  uint64_t device_address;
  // Observed cache class of the persistent host mapping.
  amdf_host_cacheability_t host_cacheability;
} iree_hal_amd_xdna_memory_t;

// Prepared native storage borrowed by one materialized buffer view. The memory
// owner, mapping, and binding table must outlive the returned buffer.
typedef struct iree_hal_amd_xdna_buffer_storage_t {
  // Native allocation and persistent host mapping used by public map calls.
  iree_hal_amd_xdna_memory_t* memory;
  // Complete native binding table for byte zero of the owning allocation.
  const iree_hal_buffer_native_binding_t* bindings;
  // Private host execution slot in |bindings|.
  uint16_t host_binding_index;
  // Byte position of this view within the owning allocation.
  iree_device_size_t offset;
} iree_hal_amd_xdna_buffer_storage_t;

// Native slots published by direct XDNA allocator buffers. Contract-backed
// storage may assign the same interfaces to different slot indices.
enum iree_hal_amd_xdna_buffer_binding_index_e {
  // Address used by ordinary external buffer relocations.
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA = 0,
  // Private host execution address used by mappings and host queue operations.
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST = 1,
  // Number of generic native slots in a direct XDNA buffer.
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT = 2,
};

// Complete native table format published by direct XDNA allocator buffers.
const iree_hal_buffer_binding_layout_t* iree_hal_amd_xdna_buffer_binding_layout(
    void);

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

// Creates a HAL buffer view over provider-owned native storage. Destruction
// invokes |release_callback| and frees only the wrapper; the callback owner
// controls the storage lifetime. Failure leaves |out_buffer| NULL.
iree_status_t iree_hal_amd_xdna_buffer_wrap(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_buffer_storage_t storage,
    iree_device_size_t allocation_size, iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Creates the allocation facade used by direct HAL callers. The device and
// context are borrowed and outlive the allocator and its buffers.
iree_status_t iree_hal_amd_xdna_allocator_create(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

// Captures the stable shim-DMA slot prepared for |family|'s program site.
// Contract-backed buffers may come from any driver in the same memory domain.
// Unscoped buffers must be direct allocations from |context|. This performs no
// native registration and does not read a potentially unpublished slot value.
iree_status_t iree_hal_amd_xdna_buffer_resolve_binding_slot(
    iree_hal_amd_xdna_context_t* context, const iree_hal_queue_family_t* family,
    const iree_hal_buffer_t* buffer,
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

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_
