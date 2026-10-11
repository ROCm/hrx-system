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

// Single-host-owner parameter admission and LRU policy. Mutable state remains
// model-owned; optional idle-cache owners can schedule reclamation when idle
// parameters alone cannot meet pressure. The common physical pool invokes this
// policy during both parameter activation and mutable-state growth.
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
// serialized by their shared host owner. A group's complete physical parameter
// footprint must fit the configured pool budget, independent of current usage;
// impossible groups fail before joining admission or claiming storage.
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

// Optional model-owned idle history, distinct from parameter pinning. Callbacks
// run on the shared host owner inside a physical-capacity request. They never
// allocate, submit device work, wait, or recurse into admission. The owner
// excludes live CPU/device borrowers and protects explicit retained endpoints.
typedef struct loom_serve_residency_reclaimer_t {
  // Borrowed owner, alive until this registration removes the callbacks.
  void* self;
  // Releases one eligible logical owner and schedules its normal maintenance
  // turn. True promises eventual progress through that owner's wake/retirement
  // path; false means no eligible owner can currently be released.
  bool (*request)(void* self);
  // Whether requested or ordinary maintenance can still release backing.
  // Completion/cancellation clears this through the same serialized owner.
  bool (*is_pending)(void* self);
} loom_serve_residency_reclaimer_t;

// Installs/removes borrowed callbacks on the serialized owner. An empty value
// removes them before service destruction; it never cancels accepted work.
void loom_serve_residency_set_reclaimer(
    loom_serve_residency_t* residency,
    loom_serve_residency_reclaimer_t reclaimer);

// Cold capacity-denial query. Pending cache retirement is actual scheduled
// work, so callers defer rather than reject while it can release backing.
// Queued inference alone is not pending reclamation.
bool loom_serve_residency_cache_is_reclaiming(
    const loom_serve_residency_cache_t* cache);

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
  // An older waiter has priority or idle-cache maintenance is releasing
  // backing.
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

// Captures the borrowed execution frontier after accepted activation. A caller
// owning a pin may join this through a proactor instead of blocking in await.
// The residency and its execution must outlive that asynchronous wait.
loom_serve_execution_frontier_t loom_serve_residency_frontier(
    const loom_serve_residency_t* residency);

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

// Copies cumulative submission counters across every registered weight plan.
// Called on the shared activation owner; no plan is activated or retained.
loom_serve_weight_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency);

// Copied on the serialized owner. Active means activation was enqueued, not
// that its device timeline has completed. Fixed backing has no virtual-pool
// accounting; its zero memory fields do not describe actual physical usage.
typedef struct loom_serve_residency_snapshot_t {
  // Every parameter plan has enqueued activation on the model timeline.
  bool active;
  // Physical parameter backing can be reclaimed without destroying the model.
  bool elastic;
  // Parameter-only virtual accounting, available only with elastic backing.
  loom_serve_memory_statistics_t memory;
  // Cumulative activation and preparation work, independent of current backing.
  loom_serve_weight_activity_t activity;
} loom_serve_residency_snapshot_t;

loom_serve_residency_snapshot_t loom_serve_residency_snapshot(
    const loom_serve_residency_t* residency);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_RESIDENCY_H_
