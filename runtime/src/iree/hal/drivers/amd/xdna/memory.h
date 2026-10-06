// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_

#include "iree/hal/drivers/amd/xdna/context.h"
#include "iree/hal/drivers/amd/xdna/executable_storage.h"

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

// Creates the allocation facade used by direct HAL callers. The context is
// borrowed and must outlive the allocator and buffers allocated through it.
iree_status_t iree_hal_amd_xdna_allocator_create(
    iree_hal_amd_xdna_context_t* context, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Resolves a public logical binding to already-prepared native storage. Foreign
// allocations are rejected; submission performs no implicit registration.
iree_status_t iree_hal_amd_xdna_buffer_resolve(
    iree_hal_amd_xdna_context_t* context, iree_hal_buffer_ref_t buffer_ref,
    iree_hal_amd_xdna_executable_binding_t* out_binding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_H_
