// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_FABRIC_MEMORY_SPEC_H_
#define IREE_HAL_DRIVER_AMD_FABRIC_MEMORY_SPEC_H_

#include "amdf/amdf.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed construction facts for one physical device's memory metadata.
// These describe available choices, without creating pools or native payloads.
typedef struct iree_hal_amd_fabric_memory_spec_params_t {
  // Negotiated core API table, dominated by the device lifetime.
  const amdf_api_t* api;
  // Ordinary buffer address and permission requirements of the live device.
  amdf_memory_device_access_t access;
  // Native queue family whose memory atomic reach is qualified.
  uint32_t queue_family_ordinal;
  // Number of endpoint-local physical storage scopes.
  iree_host_size_t local_scope_count;
  // Borrowed endpoint-local scopes in native preference order.
  amdf_memory_scope_t* const* local_scopes;
  // Shared instance-owned system storage, also used by integrated devices.
  amdf_memory_scope_t* system_scope;
  // Ordinary buffer usages implemented by this device.
  iree_hal_buffer_usage_t supported_usage;
} iree_hal_amd_fabric_memory_spec_params_t;

// Adds physical heaps and available CREATE profiles to |builder|. Profiles
// sharing storage reference one heap; endpoint-local SYSTEM_BACKED views and
// the system scope reference the same physical RAM. All queries are cold and
// retain no native execution or allocation resources.
iree_status_t iree_hal_amd_fabric_memory_spec_populate(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder);

// Samples endpoint-local physical storage, or system RAM when the endpoint
// has no distinct local scope. Shared RAM is counted once. Availability is
// published only when every contributing scope supports it. Native errors
// leave |out_observation| unchanged; no submitted work is waited on.
iree_status_t iree_hal_amd_fabric_memory_spec_sample_observation(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    iree_hal_device_observation_t* out_observation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_FABRIC_MEMORY_SPEC_H_
