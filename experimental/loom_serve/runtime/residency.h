// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_RESIDENCY_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_RESIDENCY_H_

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-host-owner parameter admission and LRU policy. Logical mutable state
// remains owned by its model; this cache only releases reloadable parameters.
// The common physical pool also invokes this policy during mutable-state
// growth.
typedef struct loom_serve_residency_cache_t loom_serve_residency_cache_t;
typedef struct loom_serve_residency_t loom_serve_residency_t;

// Borrows pool and execution. NULL pool supports fixed-backed consumers but
// cannot reclaim their storage. Installs the pool's sole pressure policy.
iree_status_t loom_serve_residency_cache_create(
    loom_serve_memory_pool_t* pool, loom_serve_execution_t* execution,
    loom_serve_residency_cache_t** out_cache, iree_allocator_t host_allocator);

// All registered residencies must have been destroyed. Removes pressure policy.
void loom_serve_residency_cache_destroy(loom_serve_residency_cache_t* cache);

// Best-effort trim toward total pool commitment of target_bytes. Pinned
// parameters and model-owned mutable state survive. The pool statistics report
// the reached extent; only real platform failures return a failing status.
iree_status_t loom_serve_residency_cache_trim(
    loom_serve_residency_cache_t* cache, uint64_t target_bytes);

// Registers the weight plans consumed together by one model invocation. Copies
// the pointer array, not the plans. Plans belong to this cache's physical pool,
// are initially inactive, belong exclusively to this registration, and outlive
// it. All calls on the cache and registered models are serialized by their
// shared host owner.
iree_status_t loom_serve_residency_create(
    loom_serve_residency_cache_t* cache, iree_host_size_t weight_count,
    loom_serve_weights_t* const* weights,
    loom_serve_residency_t** out_residency, iree_allocator_t host_allocator);

// Unregisters after all consumers have retired, including failed operations.
// Terminal teardown may remove pins whose failed work has now actually retired.
// Does not destroy borrowed plans or their backing. NULL is accepted.
void loom_serve_residency_destroy(loom_serve_residency_t* residency);

// Pins the entire group before admission/loading. Already active groups need
// no I/O. On success admitted=true owns one pin, released only after work
// retires or the caller relinquishes explicit retention. admitted=false owns no
// pin and reports ordinary backpressure, with no partial parameter load. Real
// loading failures are terminal and preserve protection until residency
// destruction.
iree_status_t loom_serve_residency_try_acquire(
    loom_serve_residency_t* residency, bool* out_admitted);

// Synchronous invocation admission: capacity denial is an execution error.
// Schedulers needing to queue on pressure use try_acquire instead.
iree_status_t loom_serve_residency_acquire(loom_serve_residency_t* residency);

// Releases one owned pin after consumption has retired. No allocation or wait.
void loom_serve_residency_release(loom_serve_residency_t* residency);

// Completes a synchronous model call holding one acquired pin. Success already
// includes retirement and only releases the pin. Error cleanup joins execution
// before releasing; failed readiness preserves the pin until terminal teardown.
// Takes ownership of status and returns it joined with cleanup failure.
iree_status_t loom_serve_residency_finish(loom_serve_residency_t* residency,
                                          iree_status_t status);

// Warmup without retaining a pin across calls. The next pressure event may
// evict it. Explicit deactivation rejects pinned or fixed-backed residency.
iree_status_t loom_serve_residency_activate(loom_serve_residency_t* residency);
iree_status_t loom_serve_residency_deactivate(
    loom_serve_residency_t* residency);

// Combined parameter accounting for the group's independently prepared plans.
loom_serve_memory_statistics_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_RESIDENCY_H_
