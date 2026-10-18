// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_SLAB_PROVIDER_H_

#include "iree/hal/drivers/amd/fabric/allocation_queue.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One native address to resolve into a prepared binding slot. The request at
// host_binding_index is unused: that slot receives the optional host pointer.
typedef struct iree_hal_amd_fabric_binding_request_t {
  // Prepared interface: an execution address or a mapping source.
  iree_hal_buffer_interface_t type;
  // Index into the complete native access set, or zero for a host source.
  uint32_t access_ordinal;
  // Exact native address interface, or HOST for a remappable host source.
  amdf_memory_address_kind_t address_kind;
  // Required native source offset alignment for MAPPING_SOURCE, otherwise zero.
  uint32_t mapping_granularity;
} iree_hal_amd_fabric_binding_request_t;

// One exact input class with its qualified native construction profile.
typedef struct iree_hal_amd_fabric_import_route_t {
  // Admitted input and correlated output classes.
  iree_hal_pool_import_class_t policy;
  // Profile qualified against the provider's complete device access set.
  amdf_memory_profile_t profile;
  // Correlated backing flags required for this route and its exports.
  amdf_memory_flags_t required_flags;
} iree_hal_amd_fabric_import_route_t;

// Cold native construction contract copied into one slab provider.
typedef struct iree_hal_amd_fabric_slab_provider_options_t {
  // Static physical domain prefix, independent of logical pool names.
  const char* trace_domain;
  // Native domain ordinal, or UINT32_MAX for unspecified system placement.
  uint32_t trace_ordinal;
  // Qualified construction policy; the provider copies the access array.
  iree_hal_amd_fabric_memory_options_t memory;
  // HAL owners parallel to memory.accesses, retained through native cleanup.
  iree_hal_device_t* const* access_devices;
  // Number of exact external storage routes.
  iree_host_size_t import_count;
  // Qualified input routes, including their export class arrays.
  const iree_hal_amd_fabric_import_route_t* imports;
  // Number of prepared slots, including the optional host slot.
  uint16_t binding_count;
  // Optional host slot, or IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE.
  uint16_t host_binding_index;
  // Native address requests in binding-slot order, copied by the provider.
  const iree_hal_amd_fabric_binding_request_t* bindings;
  // HAL memory properties promised by every materialized buffer.
  iree_hal_memory_type_t memory_type;
  // HAL usages implemented by every materialized buffer.
  iree_hal_buffer_usage_t supported_usage;
  // Smallest independently cache-maintained range across all access pairs.
  iree_device_size_t maintenance_alignment;
} iree_hal_amd_fabric_slab_provider_options_t;

// Creates a provider that obtains one fully attached libamdf allocation per
// slab. The provider copies the options arrays and retains |device|, every
// access owner, and |allocation_queue| through all native cleanup. These owners
// keep the API, native instance/scopes/devices, and progress services alive.
// Explicit external imports and their native teardown remain synchronous;
// owned allocation/free batches execute on the native allocation queue.
iree_status_t iree_hal_amd_fabric_slab_provider_create(
    iree_hal_device_t* device, const amdf_api_t* api,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_allocation_queue_t* allocation_queue,
    const iree_hal_amd_fabric_slab_provider_options_t* options,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

// Executes a prepared fabric-provider batch on its native allocation worker.
// The worker retains the provider through this call and its completion
// callback.
void iree_hal_amd_fabric_slab_provider_execute(
    iree_hal_slab_operation_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_SLAB_PROVIDER_H_
