// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/xdna/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates one finite execution owner. Context and proactor are borrowed from
// the device. Semaphore-ready admission and checked native completion run on
// the proactor; unresolved consumers occupy no native execution slot.
iree_status_t iree_hal_amd_xdna_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Assigns or clears an idle queue's causal domain. The tracker is retained.
// Topology assignment is externally serialized against submission and teardown.
iree_status_t iree_hal_amd_xdna_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_
