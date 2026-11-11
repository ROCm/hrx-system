// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_VIRTUAL_BUFFER_H_
#define IREE_HAL_STREAMING_VIRTUAL_BUFFER_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// HIP/graph compatibility policy: all families of selected devices participate,
// and source classes come from their existing default backing owners. Native
// HRX callers instead construct their own exact scopes and source pools.
// Devices share one sealed group; arrays are borrowed only for construction.
// Host access describes virtual views, independently of backing placement.
typedef struct iree_hal_streaming_virtual_buffer_scope_t {
  // Nonzero number of distinct participating devices.
  size_t device_count;
  // Participating devices; the buffer retains their complete sealed group.
  const hrx_device_t* devices;
  // Required public host mapping access, or NONE for execution-only addresses.
  hrx_memory_access_t host_access;
} iree_hal_streaming_virtual_buffer_scope_t;

// One ordinary allocation class that may back a virtual reservation. Preparing
// its metadata does not allocate physical payload or change its placement.
typedef struct iree_hal_streaming_virtual_buffer_source_t {
  // Device whose allocator supplies this class, in the reservation's group.
  hrx_device_t device;
  // Memory type passed to MAPPING_SOURCE allocations from this device.
  hrx_memory_type_t memory_type;
} iree_hal_streaming_virtual_buffer_source_t;

// Compatibility source classes borrowed only through query or construction.
typedef struct iree_hal_streaming_virtual_buffer_sources_t {
  // Number of device/type classes.
  size_t source_count;
  // Selected allocation classes supplied by the frontend.
  const iree_hal_streaming_virtual_buffer_source_t* sources;
} iree_hal_streaming_virtual_buffer_sources_t;

// These construction calls select metadata only; no physical payload is
// acquired. Returned buffers retain their sealed group normally.
iree_status_t iree_hal_streaming_virtual_buffer_query_capabilities(
    iree_hal_streaming_virtual_buffer_scope_t scope,
    iree_hal_streaming_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_capabilities_t* out_capabilities);
iree_status_t iree_hal_streaming_virtual_buffer_create(
    iree_hal_streaming_virtual_buffer_scope_t scope,
    iree_hal_streaming_virtual_buffer_sources_t sources,
    const hrx_virtual_buffer_options_t* options, hrx_buffer_t* out_buffer);
iree_status_t iree_hal_streaming_virtual_buffer_query_source_requirements(
    hrx_device_t device, hrx_memory_type_t memory_type,
    hrx_memory_mapping_requirements_t* out_requirements);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_STREAMING_VIRTUAL_BUFFER_H_
