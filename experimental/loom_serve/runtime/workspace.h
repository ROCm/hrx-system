// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_WORKSPACE_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_WORKSPACE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared backing for private command allocations, independent of model
// ordering. HAL pools own suballocation and dependency-frontier reuse. This
// owner only registers slab geometry and preserves older pools while roots
// still use them.
typedef struct loom_serve_workspace_t loom_serve_workspace_t;

// Retains the device. No physical backing is acquired until acquire_pool.
// The device group must outlive this owner and all execution domains using it.
iree_status_t loom_serve_workspace_create(
    iree_hal_queue_t* dispatch_queue, loom_serve_workspace_t** out_workspace,
    iree_allocator_t host_allocator);
void loom_serve_workspace_retain(loom_serve_workspace_t* workspace);
void loom_serve_workspace_release(loom_serve_workspace_t* workspace);

// Registers a reflected command requirement without waiting for device work or
// committing memory. Larger requirements affect subsequent pool acquisitions;
// previously returned pools remain valid through their retained ownership.
iree_status_t loom_serve_workspace_reserve(loom_serve_workspace_t* workspace,
                                           iree_device_size_t length,
                                           iree_device_size_t alignment);

// Returns a retained pool satisfying all registered requirements. The caller
// keeps it through final queue/caller retirement of every root allocated from
// it, including failed submissions: transient roots borrow their source pool.
// Stable geometry reuses the same pool; allocation and frontier policy are
// HAL's.
iree_status_t loom_serve_workspace_acquire_pool(
    loom_serve_workspace_t* workspace, iree_hal_pool_t** out_pool);

// Combined live backing and cumulative allocation counters across generations.
// Device-side counters are observations, not an atomic whole-device snapshot.
iree_hal_pool_stats_t loom_serve_workspace_statistics(
    loom_serve_workspace_t* workspace);

// Releases idle backing and retired pool generations without waiting on work.
// Live roots and their source pools remain protected.
iree_status_t loom_serve_workspace_trim(loom_serve_workspace_t* workspace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_WORKSPACE_H_
