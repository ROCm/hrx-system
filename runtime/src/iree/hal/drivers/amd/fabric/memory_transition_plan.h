// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_PLAN_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_PLAN_H_

#include "iree/hal/drivers/amd/fabric/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native executor represented by one construction-time HAL site.
typedef enum iree_hal_amd_fabric_memory_site_kind_e {
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN = 0,
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST = 1,
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE = 2,
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_PROGRAM = 3,
} iree_hal_amd_fabric_memory_site_kind_t;

typedef struct iree_hal_amd_fabric_memory_transition_site_t {
  // Native interpretation of this exact HAL scope.
  iree_hal_amd_fabric_memory_site_kind_t kind;
  // Whether this exact queue requires coherent public host access.
  bool requires_host_coherent;
  // Queue execution can use PROGRAM recipes in addition to native CP atomics.
  bool program_atomics;
  // Admitted semantic operations for this exact site.
  iree_hal_atomic_operation_capabilities_t atomics;
  // CP subset selected before shaders for physical queue operations.
  iree_hal_atomic_operation_capabilities_t queue_atomics;
  // Prospective libamdf site for the exact cache executor.
  amdf_memory_profile_site_t native;
  // Native private copy family, or UINT32_MAX without a copy executor.
  uint32_t copy_family_ordinal;
} iree_hal_amd_fabric_memory_transition_site_t;

// Borrowed construction scratch. No native query state escapes into a pool.
typedef struct iree_hal_amd_fabric_memory_transition_query_t {
  // Borrowed native API used only during synchronous construction.
  const amdf_api_t* api;
  // Base construction facts completed with exact sites for each query.
  amdf_memory_profile_pair_query_t native;
  // Complete correlated construction routes covered by the pool's table.
  const iree_hal_amd_fabric_slab_provider_options_t* provider_options;
  // Borrowed dense construction-time site interpretations.
  const iree_hal_amd_fabric_memory_transition_site_t* sites;
  // Immutable per-device facts indexed by native profile, then access ordinal.
  const amdf_memory_access_capabilities_t* profile_capabilities;
  // Owned-route facts, which may include private sanitizer mapping constraints.
  const amdf_memory_access_capabilities_t* owned_capabilities;
  // Largest explicit range granularity observed across admitted exact pairs.
  iree_device_size_t maintenance_alignment;
  // Whether every admitted host/device direction requires no host cache action.
  bool host_coherent;
  // Whether every admitted host/queue pair has a complete native route.
  bool complete;
  // Qualifies reservation host aliases and fixed shader queue controls.
  bool reservation_alias;
} iree_hal_amd_fabric_memory_transition_query_t;

// Captures the CP subset for one queue across every acquisition route. A
// missing native relationship leaves zero operations without rejecting other
// executors or the pool's ordinary read/write access.
iree_status_t iree_hal_amd_fabric_memory_capture_queue_atomics(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_scope_id_t site_id,
    iree_hal_atomic_operation_capabilities_t* out_operations);

// Generic contract-builder callback. Captures the cache protocol and exact
// atomic relationships across all selected native acquisition routes.
iree_status_t iree_hal_amd_fabric_memory_query_transition(
    void* user_data, iree_hal_memory_scope_id_t producer_id,
    iree_hal_memory_scope_id_t consumer_id,
    iree_hal_memory_pair_info_t* out_info);

// Captures private copy-engine handoffs across every native acquisition route.
// Unsupported relations stay explicit without rejecting ordinary pool access.
iree_status_t iree_hal_amd_fabric_memory_capture_copy_protocols(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_contract_t* contract);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_PLAN_H_
