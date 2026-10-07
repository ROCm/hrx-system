// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_SEMAPHORE_H_

#include <stdbool.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/submitted_signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum number of independently progressing axes retained by an XDNA queue
// operation and semaphore. Exceeding this capacity preserves a conservative
// lower bound and disables device-side wait elision for the affected values.
#define IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY 64

// Creates an XDNA HAL semaphore backed by an embedded async semaphore.
//
// |device| identifies the exact creating device and is borrowed for the
// semaphore lifetime. |proactor| is likewise borrowed and must remain live
// until the semaphore is destroyed.
iree_status_t iree_hal_amd_xdna_semaphore_create(
    iree_hal_device_t* device, iree_async_proactor_t* proactor,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is an XDNA semaphore.
bool iree_hal_amd_xdna_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| is an XDNA semaphore created by |device|.
bool iree_hal_amd_xdna_semaphore_is_local(iree_hal_semaphore_t* semaphore,
                                          const iree_hal_device_t* device);

// Returns the creation flags. The caller must first verify the semaphore type.
iree_hal_semaphore_flags_t iree_hal_amd_xdna_semaphore_flags(
    iree_hal_semaphore_t* semaphore);

// Returns the complete queue family compatibility domain. The caller must
// first verify the semaphore type.
iree_hal_queue_family_affinity_t
iree_hal_amd_xdna_semaphore_queue_family_affinity(
    iree_hal_semaphore_t* semaphore);

// Returns the latest submitted-signal metadata. The caller must first verify
// the semaphore type.
iree_hal_submitted_signal_t* iree_hal_amd_xdna_semaphore_submitted_signal(
    iree_hal_semaphore_t* semaphore);

// Publishes submission-time causal state for one signal.
//
// |producer_frontier| is merged into the semaphore frontier under the same
// mutex that serializes submitted-signal writers. The cached producer is marked
// exact only when |producer_frontier_exact| is true and that frontier dominates
// all previously accumulated semaphore causality. Returns false if the merge
// exceeds the fixed frontier capacity; the previous lower bound is preserved
// and submitted-signal metadata is cleared.
bool iree_hal_amd_xdna_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier,
    bool producer_frontier_exact, uint64_t producer_epoch,
    uint64_t producer_value);

// Clears submitted-signal metadata while preserving timeline and frontier
// state. The caller must first verify the semaphore type.
void iree_hal_amd_xdna_semaphore_clear_submitted_signal(
    iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_SEMAPHORE_H_
