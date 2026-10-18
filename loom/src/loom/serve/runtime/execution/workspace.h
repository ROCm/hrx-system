// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_

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

// Retains the sealed group and an already qualified backing pool. The caller
// selects the pool's complete access scope and a sanitizer policy compatible
// with all of its consumers. The policy is copied and HAL qualifies it when
// constructing child allocators. No physical backing is acquired at creation.
// Sharing the workspace across execution domains establishes neither ordering
// nor visibility; each consumer must satisfy the backing pool's contract.
iree_status_t loom_serve_workspace_create(
    iree_hal_device_group_t* group, iree_hal_pool_t* backing_pool,
    const iree_hal_asan_pool_options_t* asan_options,
    iree_allocator_t host_allocator, loom_serve_workspace_t** out_workspace);

// Retains the shared workspace. NULL is accepted.
void loom_serve_workspace_retain(loom_serve_workspace_t* workspace);

// Final outer application teardown, after model consumers have retired. Joins
// any remaining queue allocation bookkeeping before releasing its source pools.
// NULL is accepted.
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

// Queue-owned allocation roots borrow their source pool. A scope records those
// two references independently of a model's payloads and logical lifetime.
// One execution owner serializes acquire/capture/retire; no queue work or
// custom suballocation is performed by this bookkeeping. Workspace scopes
// keep transient backing hot while their roots are live. Storage scopes only
// preserve lifetime: persistent model buffers do not inhibit workspace expiry.
typedef struct loom_serve_workspace_scope_t loom_serve_workspace_scope_t;
iree_status_t loom_serve_workspace_scope_allocate(
    loom_serve_workspace_t* workspace, bool tracks_workspace,
    loom_serve_workspace_scope_t** out_scope);

// Secures capture slots before submitting an allocation batch. The caller may
// then capture roots from any caller-selected pool without allocating metadata.
iree_status_t loom_serve_workspace_scope_reserve(
    loom_serve_workspace_scope_t* scope, iree_host_size_t count);

// Makes room for one capture and returns a retained current pool. Queue alloca
// follows. On queue rejection, the caller releases the pool; on acceptance,
// capture transfers that pool reference and retains the returned allocation.
iree_status_t loom_serve_workspace_scope_acquire(
    loom_serve_workspace_scope_t* scope, iree_hal_pool_t** out_pool);
void loom_serve_workspace_scope_capture(loom_serve_workspace_scope_t* scope,
                                        iree_hal_pool_t* pool,
                                        iree_hal_buffer_t* buffer);

// Relinquishes execution ownership without allocating or waiting. The shared
// workspace preserves raw root/pool references, never model pointers or
// callbacks. Acquisition, trim and statistics collect raw captures from both
// active and released scopes; final workspace teardown joins any remainder.
// NULL is accepted.
void loom_serve_workspace_scope_retire(loom_serve_workspace_scope_t* scope);

// Combined retained backing and logical reservation counters across
// generations. Backing includes idle cached slabs, but excludes detached native
// returns. Device-side counters are observations, not an atomic whole-device
// snapshot.
iree_hal_pool_stats_t loom_serve_workspace_statistics(
    loom_serve_workspace_t* workspace);

// Returns idle backing and retired pool generations without waiting on device
// work. Logical returns propagate directly through the backing cache; native
// freeing follows the source driver's allocation mechanism. Live roots and
// their source pools remain protected. Returns the bytes still retained by the
// logical pools after trim for budgeting further reclamation. Detached native
// returns are excluded from workspace accounting.
iree_device_size_t loom_serve_workspace_trim(loom_serve_workspace_t* workspace);

// Schedules cached backing trim after all workspace allocations have retired
// and remained idle for timeout nanoseconds. The caller supplies a positive
// timeout and its current time. Returns true while roots still await collection
// or an idle interval awaits expiry. Live workspace roots prevent expiry, and
// their retirement starts a new interval; persistent storage roots and repeated
// queries do not keep unused workspace storage hot.
// Native trim is queued without waiting for physical release.
bool loom_serve_workspace_trim_idle(loom_serve_workspace_t* workspace,
                                    iree_time_t now, iree_duration_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_
