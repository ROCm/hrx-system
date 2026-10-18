// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_FABRIC_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_FABRIC_H_

#include "amdf/amdf.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/fabric/allocation_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_fabric_allocation_entry_t
    iree_hal_amd_fabric_allocation_entry_t;

// Shared native identity and allocation domains for AMD execution engines.
// The driver and every materialized device retain this owner independently.
// Discovery is immutable; node-local native allocation queues are acquired on
// demand. Engine resources and private scopes belong to the consuming device.
// Sharing native identity does not grant access across sealed HAL groups.
typedef struct iree_hal_amd_fabric_t {
  // References held by native owners; never modified directly by consumers.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this object and its discovery snapshot.
  iree_allocator_t host_allocator;
  // Borrowed negotiated core table, live for the complete fabric lifetime.
  const amdf_api_t* api;
  // Owned instance containing shared native provider connections.
  amdf_instance_t* instance;
  // Borrowed ordinary system storage scope, owned by the instance.
  amdf_memory_scope_t* system_scope;
  // Number of passive identities in the immutable discovery snapshot.
  uint32_t endpoint_count;
  // Owned snapshot including all native engine kinds, without activation.
  amdf_endpoint_summary_t* endpoints;
  // Protects cold allocation-domain construction, never native batch execution.
  iree_slim_mutex_t allocation_mutex;
  // Owned native allocation queues, one per requested NUMA node.
  iree_hal_amd_fabric_allocation_entry_t* allocations;
} iree_hal_amd_fabric_t;

// Creates an independent native instance and snapshots passive discovery.
// Engine extension negotiation and activation happen in engine constructors.
// The providing library and |api| must outlive the returned fabric. Failure
// publishes no output. The caller releases the returned reference.
// |native_lifetime| selects libamdf's native state lifetime bound,
// independently from HAL group identity and handle ownership. PROCESS may share
// Linux KFD state with foreign clients; the application must order their VM
// mutations against reservation updates and removal. INSTANCE supports only
// services that the native provider can reclaim at instance destruction.
iree_status_t iree_hal_amd_fabric_create(const amdf_api_t* api,
                                         amdf_native_lifetime_t native_lifetime,
                                         iree_allocator_t host_allocator,
                                         iree_hal_amd_fabric_t** out_fabric);

// Retains the shared owner. NULL is ignored.
void iree_hal_amd_fabric_retain(iree_hal_amd_fabric_t* fabric);

// Releases the shared owner after all native children are gone. A native
// teardown failure is diagnosed and preserves remaining ownership. NULL is
// ignored. This performs no implicit device wait.
void iree_hal_amd_fabric_release(iree_hal_amd_fabric_t* fabric);

// Acquires the shared native allocation queue for |node_id|, creating it on
// first use. The selected progress entry supplies the node; UINT32_MAX leaves
// placement unspecified. The caller releases the returned queue reference.
iree_status_t iree_hal_amd_fabric_acquire_allocation_queue(
    iree_hal_amd_fabric_t* fabric, uint32_t node_id,
    iree_hal_amd_fabric_allocation_queue_t** out_queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_FABRIC_H_
