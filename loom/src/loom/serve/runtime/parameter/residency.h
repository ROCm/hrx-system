// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_RESIDENCY_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_RESIDENCY_H_

#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/parameter/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-host-owner admission of registered working sets and parameter LRU
// policy. Mutable state remains model-owned; optional idle-cache owners can
// schedule reclamation when idle parameters alone cannot meet pressure and
// pinned parameters do not already make the request impossible. The common
// physical pool invokes this policy during both parameter activation and
// mutable-state growth.
typedef struct loom_serve_residency_cache_t loom_serve_residency_cache_t;
typedef struct loom_serve_residency_group_t loom_serve_residency_group_t;
typedef struct loom_serve_residency_t loom_serve_residency_t;

// Borrows pool and its optional shared workspace. NULL pool supports
// fixed-backed consumers but cannot reclaim their storage. Installs the pool's
// sole pressure policy; idle workspace backing is reclaimed before model data.
iree_status_t loom_serve_residency_cache_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_workspace_t* workspace,
    iree_allocator_t host_allocator, loom_serve_residency_cache_t** out_cache);

// All residencies, groups, and downstream output views must have retired.
// Removes pressure policy.
void loom_serve_residency_cache_free(loom_serve_residency_cache_t* cache);

// Final output retirement wakes the serialized admission owner. Detachment
// joins callback access; observers never wait or reenter the cache.
void loom_serve_residency_cache_set_observer(
    loom_serve_residency_cache_t* cache, loom_serve_wake_callback_t observer);

// Accepted outputs still borrow this cache, independently of producer model
// removal or failed readiness. Final credit retirement wakes the owner.
bool loom_serve_residency_cache_is_pending(loom_serve_residency_cache_t* cache);

// Collects completed parameter releases on the serialized owner without
// starting new release work. Call on native-capacity wakes before advancing
// services, even when no cold activation is waiting. Native failure propagates
// to the owner for ordinary draining. With no releases this is constant-time.
iree_status_t loom_serve_residency_cache_query_status(
    loom_serve_residency_cache_t* cache);

// Best-effort trim toward total pool commitment of target_bytes. Pinned
// parameters and model-owned mutable state survive. The pool statistics report
// completed and releasing extents; only real platform failures return a failing
// status. Native release does not block the owner or grant capacity early.
iree_status_t loom_serve_residency_cache_trim(
    loom_serve_residency_cache_t* cache, uint64_t target_bytes);

// Schedules release of active, unpinned parameters idle for at least timeout
// nanoseconds. The caller supplies a positive timeout and its current time.
// Reports whether an eligible entry still awaits expiry. Active invocations,
// explicit pins, fixed backing, and model-owned history are untouched. Reads
// and repeated maintenance do not refresh age. Accepted releases finish through
// query_status and the shared capacity wake, without polling or blocking here.
iree_status_t loom_serve_residency_cache_trim_idle(
    loom_serve_residency_cache_t* cache, iree_time_t now,
    iree_duration_t timeout, bool* out_pending);

// Registers one independently evictable group of immutable weight plans.
// Copies the pointer array, not the plans. Plans belong to this cache's
// physical pool, are initially inactive, belong exclusively to this group, and
// outlive it. Multiple invocation residencies may borrow the same group. All
// cache, group, and residency operations share one serialized host owner.
iree_status_t loom_serve_residency_group_allocate(
    loom_serve_residency_cache_t* cache, iree_host_size_t weight_count,
    loom_serve_weights_t* const* weights, iree_allocator_t host_allocator,
    loom_serve_residency_group_t** out_group);

// Unregisters after every borrowing residency is destroyed and accepted weight
// operations have retired. Does not destroy borrowed plans. NULL is accepted.
void loom_serve_residency_group_free(loom_serve_residency_group_t* group);

// Storage and loading activity owned by the immutable parameter group,
// independent of invocation admission. Valid during source preparation and
// failed-load retirement. NULL reports zero before group registration.
loom_serve_memory_usage_t loom_serve_residency_group_statistics(
    const loom_serve_residency_group_t* group);
loom_serve_weight_activity_t loom_serve_residency_group_activity(
    const loom_serve_residency_group_t* group);

// Bank-owner retention, independent of invocation residency lifetimes. Applies
// to current and future activations without loading; destruction drops policy.
void loom_serve_residency_group_set_pinned(loom_serve_residency_group_t* group,
                                           bool pinned);
bool loom_serve_residency_group_is_pinned(
    const loom_serve_residency_group_t* group);

// Owner-side release of an unpinned elastic group. The caller excludes new
// admission and has checked the complete set before releasing its first group.
// Repeated calls collect accepted release; native failures preserve ownership.
iree_status_t loom_serve_residency_group_deactivate(
    loom_serve_residency_group_t* group, bool* out_complete);

// Registers one invocation/cohort working set of distinct groups in this cache.
// Copies the pointer array; groups outlive the residency. Borrows execution for
// readiness import and synchronous outer-thread error retirement, never waiting
// on unrelated consumer timelines. Parameters and their concurrent activation
// staging must fit the pool budget independent of current usage; impossible
// sets fail before admission. Weight source preparation precedes registration.
// A shared group contributes physical storage once regardless of its borrowers.
iree_status_t loom_serve_residency_allocate(
    loom_serve_residency_cache_t* cache, loom_serve_execution_t* execution,
    iree_host_size_t group_count, loom_serve_residency_group_t* const* groups,
    iree_allocator_t host_allocator, loom_serve_residency_t** out_residency);

// Unregisters after all consumers have retired, including failed operations.
// Terminal teardown may remove pins whose failed work has now actually retired.
// Does not destroy borrowed plans or their backing. NULL is accepted.
void loom_serve_residency_free(loom_serve_residency_t* residency);

// Checks whether one invocation can fit by itself, before acquiring any pins
// or submitting activation. Each extent describes additional simultaneously
// live workspace beyond the distinct parameter groups: cohort I/O and the peak
// of ordered private commands. Their aggregate is rounded to the workspace's
// physical slab geometry. This checks the configured serving ceiling, not
// available HAL memory, and neither reserves credit nor admits work against
// other invocations. Mutable state is accounted at admission. Fixed backing has
// no configured ceiling.
iree_status_t loom_serve_residency_check_working_set(
    const loom_serve_residency_t* residency, iree_host_size_t extent_count,
    const iree_device_size_t* extents);

// Sets the next serialized invocation's private peak and escaping output bound
// before admission. Private bytes include cohort input and ordered command
// scratch; output bytes survive independently of the producer after
// publication. Ordered activation staging shares the invocation's private
// high-water. The caller owns no invocation pin while replacing these prepared
// bounds.
iree_status_t loom_serve_residency_prepare_working_set(
    loom_serve_residency_t* residency, uint64_t private_bytes,
    uint64_t output_bytes);

// Checks a replacement command peak at a completed execution boundary. Live
// rows may retain parameter pins; an increase must fit alongside their state
// and accepted peers. This neither changes accounting nor reclaims history.
// Ordered activation staging remains part of the private high-water.
iree_status_t loom_serve_residency_check_private_peak(
    const loom_serve_residency_t* residency, uint64_t private_bytes);

// Publishes a checked replacement after its command binding succeeds, on the
// same serialized owner turn as check_private_peak. No work, parameter pin,
// escaping output, or retained state changes. Publication cannot fail.
void loom_serve_residency_set_private_peak(loom_serve_residency_t* residency,
                                           uint64_t private_bytes);

// Replaces an owned output reference with an independently retained credit
// view. The producer's pin and the final view retirement must both complete
// before its admission credit returns. The cache outlives every consumer;
// the view retains neither the producer model nor its parameter groups.
// Allocation failure leaves the input reference and producer credit unchanged.
iree_status_t loom_serve_residency_track_output(
    loom_serve_residency_t* residency, iree_hal_buffer_t** buffer);

// Physical parameter footprint plus concurrent activation lanes. The source
// plans are already prepared; this neither activates nor pins them. Shared
// groups count once in this residency, not across independent callers.
uint64_t loom_serve_residency_activation_footprint(
    const loom_serve_residency_t* residency);

// Borrowed admission/LRU owner, outliving this registration. Equal cache
// identities share pressure; activity in another cache cannot release it.
loom_serve_residency_cache_t* loom_serve_residency_cache(
    const loom_serve_residency_t* residency);

// Sets persistent parameter retention independently of operation-owned pins.
// Applies to current and subsequent activations without loading or unloading.
// Unpinning restores eviction eligibility only after all accepted consumers
// retire. Mutable state and its reclamation policy remain independent. Called
// on the serialized owner; destruction discards this policy with the residency.
void loom_serve_residency_set_pinned(loom_serve_residency_t* residency,
                                     bool pinned);

// Whether accepted consumers or persistent retention exclude policy eviction.
// Read on the serialized owner; this does not acquire or release protection.
bool loom_serve_residency_is_pinned(const loom_serve_residency_t* residency);

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

// Pins every group before admission/loading. Already active groups need no I/O;
// each execution imports their readiness once per activation. Enqueues loading
// before consumption without waiting on device or I/O completion.
// On success admitted=true owns one pin,
// released only after activation and consuming work retire (or after activation
// if the caller relinquishes explicit retention). admitted=false owns no
// pin and reports ordinary backpressure, with no partial parameter load. Real
// loading failures are terminal and preserve protection until residency
// destruction.
iree_status_t loom_serve_residency_try_acquire(
    loom_serve_residency_t* residency, bool* out_admitted);

typedef enum loom_serve_residency_admission_e {
  // Declared working set cannot fit. No pin or partial load is owned.
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
// Parameter-free working sets bypass weight pressure. Acquiring parameters
// preserves an existing interest: model-owned state may still need capacity.
// The caller withdraws it with cancel_admission only after the complete
// request/cohort is admitted or abandoned. All calls share the cache's
// serialized owner. No callback or task waits are installed.
iree_status_t loom_serve_residency_try_admit(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission);

// Admits parameters and the caller's complete mutable-state range union before
// submitting either. Ranges belong to this residency's physical owner and
// their logical borrowers exclude policy reclamation for the call. Denial
// retains neither a parameter pin nor new backing; FIFO interest remains.
// Success owns a parameter pin and publishes state backing on readiness/value,
// independently of parameter loading on the execution timeline. The caller
// owns the returned operation through actual native retirement, including
// terminal activation failure. A fixed-backed caller supplies no ranges or
// readiness. No task or device wait occurs.
iree_status_t loom_serve_residency_try_admit_state(
    loom_serve_residency_t* residency, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_residency_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation);

// Withdraws this model's pending admission, without changing any owned pin or
// queued device work. Idempotent; called on the owner when its prepared request
// is fully admitted, cancelled, or rejected. A later request joins behind
// surviving interests. Parameter readiness alone does not end a turn still
// waiting for mutable-state capacity.
void loom_serve_residency_cancel_admission(loom_serve_residency_t* residency);

// Invocation admission: capacity denial is an execution error. Like
// try_acquire this only enqueues activation and does not wait for readiness.
// New-work schedulers use try_admit; accepted consumers use acquire directly.
iree_status_t loom_serve_residency_acquire(loom_serve_residency_t* residency);

// Synchronous outer-thread invocation admission and readiness join. Waits for
// already-accepted native capacity operations before retrying admission. Live
// pins and model-owned history needing scheduler progress still deny capacity.
// Success owns one pin; activation failure preserves it until teardown. Shared
// task schedulers use try_admit/acquire and never call this function.
iree_status_t loom_serve_residency_acquire_wait(
    loom_serve_residency_t* residency);

// Cold outer-thread activation join for a pinned working set. Task services
// consume the model timeline asynchronously. Failure preserves the pin until
// teardown.
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
// next pressure event may evict it.
iree_status_t loom_serve_residency_activate(loom_serve_residency_t* residency);

// Starts/advances asynchronous release after consumers retire. New requests
// cannot reactivate this residency until native release completes. Rejects
// pinned/fixed backing before submission; pinning cannot cancel accepted
// release. The pool owner receives completion wakes; true joins all native
// release work.
iree_status_t loom_serve_residency_deactivate(loom_serve_residency_t* residency,
                                              bool* out_complete);

// Combined accounting for this working set. Shared groups appear in each
// borrower's snapshot; summing borrower snapshots is not physical accounting.
// Independent group peaks cannot establish a simultaneous working-set peak.
loom_serve_memory_usage_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency);

// Copies cumulative submission counters across every registered weight plan.
// Called on the shared activation owner; no plan is activated or retained.
loom_serve_weight_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency);

// Copied on the serialized owner. Active means activation was enqueued, not
// that its device timeline has completed. Fixed backing has no virtual-pool
// accounting; its zero memory fields do not describe actual physical usage.
typedef struct loom_serve_residency_snapshot_t {
  // This execution imported readiness for every group's current activation.
  bool active;
  // Physical parameter backing can be reclaimed without destroying the model.
  bool elastic;
  // Accepted-consumer and explicit-retention pins, excluding persistent policy.
  iree_host_size_t pin_count;
  // New-work FIFO turns, including joint parameter/state admission retries;
  // not total request queue latency or execution duration.
  struct {
    // One new-work interest is waiting for parameter admission.
    bool waiting;
    // Distinct wait episodes; retries of the same interest do not count again.
    uint64_t count;
    // Total nanoseconds waiting, including cancelled and currently open waits.
    uint64_t duration_ns;
    // Nanoseconds in the current wait, or zero when not waiting.
    uint64_t age_ns;
  } admission;
  // Parameter-only virtual accounting, available only with elastic backing.
  loom_serve_memory_usage_t memory;
  // Cumulative activation and preparation work, independent of current backing.
  loom_serve_weight_activity_t activity;
} loom_serve_residency_snapshot_t;

// Copies current ownership and cumulative work without advancing admission or
// retaining parameters. Reads the clock only while an admission wait is open.
loom_serve_residency_snapshot_t loom_serve_residency_snapshot(
    const loom_serve_residency_t* residency);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_RESIDENCY_H_
