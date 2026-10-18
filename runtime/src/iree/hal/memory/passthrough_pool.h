// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_
#define IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_

#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/pool.h"
#include "iree/hal/slab_pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;

// Options for creating a pass-through HAL pool.
typedef struct iree_hal_passthrough_pool_options_t {
  // Qualified immutable memory contract retained by the pool, or NULL when
  // individual reservation parameters describe access.
  iree_hal_memory_contract_t* memory_contract;

  // Optional native completion probe inherited by child allocators. The
  // borrowed context remains alive with the pool's sealed device group.
  iree_hal_pool_epoch_query_t epoch_query;

  // ASAN policy used to shape hidden backing ranges for reservations.
  iree_hal_asan_pool_options_t asan;

  // Whether new provider-owned slabs may be acquired.
  iree_hal_slab_pool_owned_mode_t owned_mode;

  // Exact external storage classes admitted for native import. Arrays are
  // borrowed for this call and copied by the pool.
  struct {
    // Number of admitted classes.
    iree_host_size_t class_count;
    // Constructor-borrowed class descriptors.
    const iree_hal_pool_import_class_t* classes;
  } imports;

  // Optional named-memory trace identifier for logical reservations returned by
  // this pool. Empty uses a generic process-stable identifier.
  iree_string_view_t trace_name;
} iree_hal_passthrough_pool_options_t;

// Creates a pass-through pool that acquires one native slab per reservation.
// Acquisition captures native batch state and completes inline or later on the
// provider-selected context. All output reservations publish together; failed
// batches return their native storage before reporting failure.
//
// Returning the reservation token and every view registers its exact release
// frontier. Actual completion permits native release using descriptor storage
// secured during acquisition. A failed prerequisite quarantines the slab until
// caller-quiescent destruction and is reported by later acquisition. Native
// bytes remain charged until the provider completes their release.
//
// Final public release requires returned reservations/views and
// caller-quiescent execution. It cancels pending retirement records and submits
// native cleanup without waiting. Public metadata is freed before returning.
// Private retirement state uses the retained provider's host allocator and
// retains |notification| and |frontier_tracker| through final completion. The
// provider's native owner and allocator must remain valid through that final
// provider release; neither depends on the public pool's allocator.
//
// |host_allocator| owns public pool metadata. Native retirement records and
// source transactions use the provider's allocator, whose lifetime covers their
// terminal completion.
iree_status_t iree_hal_passthrough_pool_create(
    iree_hal_passthrough_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_
