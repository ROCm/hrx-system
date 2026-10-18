// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_HOST_MEMORY_BACKEND_H_
#define IREE_HAL_MEMORY_HOST_MEMORY_BACKEND_H_

#include "iree/hal/device_event.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/memory_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Already-established coherent CPU memory resources. The containing device
// owns these references and outlives every group pool borrowing the descriptor.
typedef struct iree_hal_host_memory_backend_t {
  // Generic native schema and shared host construction factory.
  iree_hal_memory_backend_t base;
  // Borrowed device diagnostic sink for native teardown faults.
  iree_hal_device_event_sink_t event_sink;
  // Existing coherent CPU allocation source.
  iree_hal_slab_provider_t* slab_provider;
  // Existing CPU shared-section source for remappable physical storage.
  iree_hal_slab_provider_t* mapping_provider;
  // Existing placement-local progress notification.
  iree_async_notification_t* notification;
  // Existing completion probe with a group-lifetime borrowed context.
  iree_hal_pool_epoch_query_t epoch_query;
} iree_hal_host_memory_backend_t;

// Captures borrowed owner views; creates no native object or payload.
void iree_hal_host_memory_backend_initialize(
    iree_hal_slab_provider_t* slab_provider,
    iree_hal_slab_provider_t* mapping_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_host_memory_backend_t* out_backend);

// Creates a process reservation from an already-qualified coherent contract.
// Used by the host backend after qualifying the complete group scope.
iree_status_t iree_hal_host_virtual_buffer_create(
    const iree_hal_host_memory_backend_t* backend,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_hal_memory_contract_t* contract,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_HOST_MEMORY_BACKEND_H_
