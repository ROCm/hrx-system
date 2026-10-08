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

// Borrows pool. NULL pool supports fixed-backed consumers but
// cannot reclaim their storage. Installs the pool's sole pressure policy.
iree_status_t loom_serve_residency_cache_create(
    loom_serve_memory_pool_t* pool, loom_serve_residency_cache_t** out_cache,
    iree_allocator_t host_allocator);

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
// it. Borrows the model's execution domain for activation dependencies and
// synchronous outer-thread error retirement; the shared cache never waits on
// unrelated model timelines. All calls on the cache and registered models are
// serialized by their shared host owner.
iree_status_t loom_serve_residency_create(
    loom_serve_residency_cache_t* cache, loom_serve_execution_t* execution,
    iree_host_size_t weight_count, loom_serve_weights_t* const* weights,
    loom_serve_residency_t** out_residency, iree_allocator_t host_allocator);

// Unregisters after all consumers have retired, including failed operations.
// Terminal teardown may remove pins whose failed work has now actually retired.
// Does not destroy borrowed plans or their backing. NULL is accepted.
void loom_serve_residency_destroy(loom_serve_residency_t* residency);

// Borrowed admission/LRU owner, outliving this registration. Equal cache
// identities share pressure; activity in another cache cannot release it.
loom_serve_residency_cache_t* loom_serve_residency_cache(
    const loom_serve_residency_t* residency);

// Pins the entire group before admission/loading. Already active groups need
// no I/O. Enqueues loading before consumption on the model timeline, without
// waiting on device or I/O completion. On success admitted=true owns one pin,
// released only after activation and consuming work retire (or after activation
// if the caller relinquishes explicit retention). admitted=false owns no
// pin and reports ordinary backpressure, with no partial parameter load. Real
// loading failures are terminal and preserve protection until residency
// destruction.
iree_status_t loom_serve_residency_try_acquire(
    loom_serve_residency_t* residency, bool* out_admitted);

typedef enum loom_serve_residency_admission_e {
  // Parameter backing cannot currently fit. No pin or partial load is owned.
  LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY = 0,
  // An older pressure waiter has priority within this physical domain.
  LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED,
  // One pin protects enqueued activation and subsequent consuming work.
  LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED,
} loom_serve_residency_admission_t;

// New-work admission with one outstanding interest per residency. An actual
// capacity denial starts a FIFO; subsequent weighted admissions join it, even
// for an already-active model. Accepted work uses acquire/try_acquire directly
// so nested execution pins never wait behind the work they must unblock.
// Parameter-free groups bypass weight pressure. Success removes the interest;
// abandoning a denied/deferred request requires cancel_admission. All calls
// share the cache's serialized owner. No callback or task waits are installed.
iree_status_t loom_serve_residency_try_admit(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission);

// Withdraws this model's pending admission, without changing any owned pin or
// queued device work. Idempotent; called on the owner when its prepared request
// is cancelled/rejected. A later request joins behind surviving interests.
void loom_serve_residency_cancel_admission(loom_serve_residency_t* residency);

// Invocation admission: capacity denial is an execution error. Like
// try_acquire this only enqueues activation and does not wait for readiness.
// New-work schedulers use try_admit; accepted consumers use acquire directly.
iree_status_t loom_serve_residency_acquire(loom_serve_residency_t* residency);

// Cold outer-thread activation join for a pinned group. Task services consume
// the model timeline asynchronously. Failure preserves the pin until teardown.
iree_status_t loom_serve_residency_await(loom_serve_residency_t* residency);

// Releases one owned pin after consumption has retired. No allocation or wait.
void loom_serve_residency_release(loom_serve_residency_t* residency);

// Completes a synchronous model call holding one acquired pin. Success already
// includes retirement and only releases the pin. Error cleanup joins execution
// before releasing; failed readiness preserves the pin until terminal teardown.
// Takes ownership of status and returns it joined with cleanup failure.
iree_status_t loom_serve_residency_finish(loom_serve_residency_t* residency,
                                          iree_status_t status);

// Synchronous outer-thread warmup without retaining a pin across calls. The
// next pressure event may evict it. Explicit deactivation rejects pinned or
// fixed-backed residency.
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
