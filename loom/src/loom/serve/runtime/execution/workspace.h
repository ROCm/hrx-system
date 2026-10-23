// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/storage/virtual_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared backing for private command allocations, independent of model
// ordering. HAL pools own suballocation and dependency-frontier reuse.
// Elastic backing preserves one logical allocator across every model lifetime;
// oversized requests use dedicated virtual ranges from the same source. Fixed
// backing retains exact native geometry caches until their raw roots retire.
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

// Uses a serialized physical owner instead of an ordinary native slab cache.
// The borrowed virtual_pool outlives this workspace. range selects the
// preferred TLSF size/alignment, independently of model requirements;
// larger requests use dedicated virtual ranges with the same physical slabs.
iree_status_t loom_serve_workspace_create_from_virtual(
    iree_hal_device_group_t* group, loom_serve_virtual_pool_t* virtual_pool,
    iree_hal_pool_reservation_request_t range, iree_allocator_t host_allocator,
    loom_serve_workspace_t** out_workspace);

// Retains the shared workspace. NULL is accepted.
void loom_serve_workspace_retain(loom_serve_workspace_t* workspace);

// Final outer application teardown, after model consumers have retired. Joins
// any remaining queue allocation bookkeeping before releasing its source pools.
// NULL is accepted.
void loom_serve_workspace_release(loom_serve_workspace_t* workspace);

// Registers a reflected command requirement without waiting for device work or
// committing memory. Fixed storage uses larger requirements to select a new
// native geometry without invalidating earlier users. Virtual backing uses
// TLSF's dedicated-range path and never replaces the shared logical pool.
iree_status_t loom_serve_workspace_reserve(loom_serve_workspace_t* workspace,
                                           iree_device_size_t length,
                                           iree_device_size_t alignment);

// Returns a retained pool satisfying all registered requirements. The caller
// keeps it through final queue/caller retirement of every root allocated from
// it, including failed submissions: transient roots borrow their source pool.
// Virtual backing always returns the same pool; allocation and frontier policy
// are HAL's.
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

// Makes room for count captures and returns a retained current pool. Queue
// alloca follows. On queue rejection, the caller releases the pool; on
// acceptance, each capture transfers a pool reference and retains its returned
// allocation. The caller retains additional pool references for a batch.
iree_status_t loom_serve_workspace_scope_acquire(
    loom_serve_workspace_scope_t* scope, iree_host_size_t count,
    iree_hal_pool_t** out_pool);
void loom_serve_workspace_scope_capture(loom_serve_workspace_scope_t* scope,
                                        iree_hal_pool_t* pool,
                                        iree_hal_buffer_t* buffer);

// Relinquishes execution ownership without allocating or waiting. The shared
// workspace preserves raw root/pool references, never model pointers or
// callbacks. Acquisition, trim and statistics collect raw captures from both
// active and released scopes; final workspace teardown joins any remainder.
// NULL is accepted.
void loom_serve_workspace_scope_retire(loom_serve_workspace_scope_t* scope);

// Retained backing and logical reservation counters, including older fixed
// generations. Backing includes idle cached slabs, but excludes detached native
// returns. Device-side counters are observations, not an atomic whole-device
// snapshot.
iree_hal_pool_stats_t loom_serve_workspace_statistics(
    loom_serve_workspace_t* workspace);

// Physical-owner ledger for virtual workspace ranges, without collecting or
// querying native work. Zero for fixed backing. Called on the serialized owner.
loom_serve_memory_usage_t loom_serve_workspace_memory_usage(
    const loom_serve_workspace_t* workspace);

// Rounds a declared simultaneous workspace peak to physical slab bytes for
// admission. Fixed backing leaves the bound unchanged. This uses immutable
// geometry, not allocator availability or currently retained backing.
uint64_t loom_serve_workspace_round_size(
    const loom_serve_workspace_t* workspace, uint64_t byte_length);

// Returns idle backing without waiting on device work. Logical returns
// propagate directly through the backing cache; native
// freeing follows the source driver's allocation mechanism. Live roots and
// their source pools remain protected. Reports the bytes still retained by the
// logical pools after trim for budgeting further reclamation. Detached native
// returns are excluded from workspace accounting.
// Starts virtual detaches on the serialized physical owner, without advancing
// acquisitions. Platform failure propagates while retaining cleanup ownership.
iree_status_t loom_serve_workspace_trim(loom_serve_workspace_t* workspace,
                                        iree_device_size_t* out_retained_bytes);

// Whether requested physical returns still await history or native completion.
// Cold acquisitions alone are not reclamation and cannot defer their own work.
bool loom_serve_workspace_is_reclaiming(loom_serve_workspace_t* workspace);

// Schedules cached backing trim after all workspace allocations have retired
// and remained idle for timeout nanoseconds. The caller supplies a positive
// timeout and its current time. Returns true while roots still await collection
// or an idle interval awaits expiry. Live workspace roots prevent expiry, and
// their retirement starts a new interval; persistent storage roots and repeated
// queries do not keep unused workspace storage hot.
// Native trim is queued without waiting for physical release.
bool loom_serve_workspace_trim_idle(loom_serve_workspace_t* workspace,
                                    iree_time_t now, iree_duration_t timeout);

// Physical-owner progress, serialized with model residency operations. Fixed
// backing has no serving-owned cold work. Queue callbacks only publish to this
// owner and wake it; they never commit or reclaim physical slabs.
void loom_serve_workspace_set_observer(loom_serve_workspace_t* workspace,
                                       loom_serve_wake_callback_t observer);
iree_status_t loom_serve_workspace_advance(loom_serve_workspace_t* workspace);
bool loom_serve_workspace_is_pending(loom_serve_workspace_t* workspace);

// Outer blocking caller equivalent to scheduler progress plus a timeline wait.
// The caller owns physical residency and excludes other owner operations.
iree_status_t loom_serve_workspace_wait(loom_serve_workspace_t* workspace,
                                        iree_hal_semaphore_t* semaphore,
                                        uint64_t value);

// Joins final consumer ownership while advancing cold backing even after a
// failed readiness timeline. Only an exclusive outer physical owner may wait;
// shared task workers instead keep advancing until retirement is idle.
iree_status_t loom_serve_workspace_await_retirement(
    loom_serve_workspace_t* workspace, loom_serve_retirement_t* retirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_WORKSPACE_H_
