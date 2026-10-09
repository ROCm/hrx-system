// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_BACKEND_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_BACKEND_H_

#include "iree/hal/drivers/amd/xdna/context.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed native and progress owners established during device creation.
// Cold construction factories qualify complete XDNA-family sets against these
// owners without creating queues, contexts, or allocation payload.
typedef struct iree_hal_amd_xdna_memory_backend_t {
  // Generic AMDF schema and published construction factories.
  iree_hal_memory_backend_t base;
  // Core libamdf storage factory.
  const iree_hal_slab_pool_factory_t* factories[1];
  // Existing HAL owner used for materialized buffer placement.
  iree_hal_device_t* device;
  // Device-owned native allocation and address namespace.
  iree_hal_amd_xdna_context_t* context;
  // Existing placement-local capacity notification.
  iree_async_notification_t* notification;
  // Existing independent cold allocation and retirement owner.
  iree_hal_memory_maintenance_t* maintenance;
  // Existing completion probe with a group-lifetime borrowed context.
  iree_hal_pool_epoch_query_t epoch_query;
} iree_hal_amd_xdna_memory_backend_t;

// Publishes the core libamdf factory after the caller supplies existing native
// and progress services. This creates no native object, payload, or policy
// owner.
void iree_hal_amd_xdna_memory_backend_initialize(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    iree_async_notification_t* notification,
    iree_hal_memory_maintenance_t* maintenance,
    iree_hal_pool_epoch_query_t epoch_query,
    iree_hal_amd_xdna_memory_backend_t* out_backend);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_BACKEND_H_
