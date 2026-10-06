// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_CONTEXT_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_CONTEXT_H_

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// A prepared native allocation contract. Scope and device are borrowed from
// the context owner. Selection happens once during device construction.
typedef struct iree_hal_amd_xdna_memory_source_t {
  // Scope accepting this construction contract.
  amdf_memory_scope_t* scope;
  // Selected provider allocation and host-mapping capabilities.
  amdf_memory_profile_t profile;
  // Device access established by every allocation from this source.
  amdf_memory_device_access_t access;
  // Selected device-access capabilities.
  amdf_memory_access_capabilities_t capabilities;
  // Address domain used by the consuming native program.
  amdf_memory_address_kind_t address_kind;
} iree_hal_amd_xdna_memory_source_t;

// Cold native ownership shared by the HAL device's buffers, executables, and
// queue. The HAL device dominates their public lifetimes. Native cleanup
// failure retains this owner and its driver instead of invalidating remaining
// children.
typedef struct iree_hal_amd_xdna_context_t {
  // Allocator owning this structure.
  iree_allocator_t host_allocator;
  // Retained driver keeping the native instance and API tables live.
  iree_hal_driver_t* driver;
  // Borrowed immutable common API table.
  const amdf_api_t* api;
  // Borrowed immutable XDNA extension table.
  const amdf_xdna_api_t* xdna;
  // Opened endpoint, released after the native device.
  amdf_endpoint_t* endpoint;
  // Ordinary device address domain, released after the execution context.
  amdf_device_t* device;
  // Exact schedulable context borrowed by the queue and command storage.
  amdf_xdna_context_t* handle;
  // Exact passive compiler identity.
  amdf_xdna_endpoint_info_t endpoint_info;
  // Activated instruction and geometry limits.
  amdf_xdna_device_info_t device_info;
  // Admitted immutable image contract.
  iree_hal_amd_xdna_aie2p_target_t target;
  // Endpoint-local native kernel-publication family.
  uint32_t queue_family_ordinal;
  // Ordinary host-mappable DMA storage contract.
  iree_hal_amd_xdna_memory_source_t data_source;
  // Context-private host-mappable instruction storage contract.
  iree_hal_amd_xdna_memory_source_t command_source;
  // Device creation's diagnostic sink, borrowed for the device lifetime.
  iree_hal_device_event_sink_t event_sink;
} iree_hal_amd_xdna_context_t;

// Opens an endpoint and admits an exact logical context. The instance and
// system scope belong to |driver|, which is retained on success. Failure
// publishes no output or cleanup obligation.
iree_status_t iree_hal_amd_xdna_context_create(
    iree_hal_driver_t* driver, const amdf_api_t* api,
    const amdf_xdna_api_t* xdna, amdf_instance_t* instance,
    amdf_memory_scope_t* system_scope, const amdf_endpoint_id_t* endpoint_id,
    uint32_t column_count, iree_hal_device_event_sink_t event_sink,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_context_t** out_context);

// Releases an idle context after all memory and queue users have been released.
// Native failure is diagnosed and the remaining ownership graph is left live.
void iree_hal_amd_xdna_context_destroy(iree_hal_amd_xdna_context_t* context);

// Publishes a native failure without allocating an IREE status or aborting.
void iree_hal_amd_xdna_context_report(
    const iree_hal_amd_xdna_context_t* context, amdf_status_t status,
    const char* operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_CONTEXT_H_
