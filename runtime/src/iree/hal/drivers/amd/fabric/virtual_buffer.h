// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_VIRTUAL_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_VIRTUAL_BUFFER_H_

#include "amdf/amdf.h"
#include "iree/hal/memory_backend.h"
#include "iree/hal/virtual_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries complete-scope native address geometry without reserving addresses.
// Uses the same native consumer qualification as reservation creation.
iree_status_t iree_hal_amd_fabric_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities);

// Qualifies explicit native device address domains and creates the complete
// reservation. Unsupported scopes return OK with no output. Address views
// inherit no physical placement or visibility facts from future backing.
iree_status_t iree_hal_amd_fabric_virtual_buffer_create(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Resolves the mapped prefix of a nonempty range in a native AMDF view.
// The returned source offset is native-memory-relative; length stops at the
// next source boundary. The access ordinal names the exact consumer of
// scope_id, including when several consumers share one page table. The caller
// orders mapping changes against the range's use and keeps its physical sources
// live. No backing is retained. A short metadata lock protects the indexed
// lookup; native calls and payload access occur only after it returns. Holes
// and failed native mapping groups report errors instead of returning stale
// sources.
iree_status_t iree_hal_amd_fabric_virtual_buffer_query_source(
    iree_hal_virtual_buffer_view_t view, uint32_t scope_id,
    iree_device_size_t offset, iree_device_size_t length,
    amdf_memory_mapping_update_t* out_source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_VIRTUAL_BUFFER_H_
