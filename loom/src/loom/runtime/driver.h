// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_DRIVER_H_
#define LOOM_RUNTIME_DRIVER_H_

#include "loom/runtime/service.h"
#include "loom/runtime/service_schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared service progression on one serialized application owner. Admission,
// resource progress, and delivery surround each turn. The driver owns neither
// service storage nor an executor, and performs no allocation or waiting.
typedef struct loom_runtime_driver_t {
  // Borrowed stable slots through final collection. A NULL vtable is vacant.
  // The owner may publish or clear slots between calls or retire the current
  // slot during its advance; it must preserve accepted callback lifetimes.
  loom_runtime_service_t* services;
  // Immutable slot count, including vacant publication credit.
  iree_host_size_t capacity;
  // Rotating priority, advanced once per complete traversal.
  iree_host_size_t cursor;
  // Monotonic abort intent; accepted borrowers still require collection.
  bool stopping;
  // Every slot has received a traversal that began with abort intent.
  bool stopping_applied;
  // Owned terminal application failure, consumed after final collection.
  // Compositions retain model-local failures instead of returning them here.
  iree_status_t status;
} loom_runtime_driver_t;

// Initializes inert state over caller-owned slots. The caller later consumes
// driver.status. Services and their capacity owners outlive final collection.
void loom_runtime_driver_initialize(iree_host_size_t capacity,
                                    loom_runtime_service_t* services,
                                    loom_runtime_driver_t* out_driver);

// Installs or detaches the owner wake on published services. A later publisher
// installs the same wake before submitting asynchronous work. Detachment
// follows final collection, not merely a terminal status or readiness signal.
void loom_runtime_driver_set_observer(loom_runtime_driver_t* driver,
                                      loom_runtime_wake_callback_t observer);

// Another occupied slot in the same nonnull capacity domain owns activity that
// can release capacity. Queued requests alone never justify waiting on a peer.
loom_runtime_service_flags_t loom_runtime_driver_shared_activity(
    const loom_runtime_driver_t* driver, iree_host_size_t selected);

// Advances every occupied slot once in rotating order, including after failure.
// A returned failure latches stopping/status; accepted work still drains.
void loom_runtime_driver_advance(loom_runtime_driver_t* driver,
                                 bool* out_progress);

// Applies one decision through the existing stable slot, checking its lifetime
// before dispatch. Empty/reused slots and stopping return an unaccepted OK
// outcome. The live source validates its own offer/choice. No new table,
// allocation, policy invocation, or readiness wait occurs here.
iree_status_t loom_runtime_driver_try_schedule(
    loom_runtime_driver_t* driver,
    const loom_runtime_service_decision_t* decision,
    loom_runtime_service_schedule_outcome_t* out_outcome);

// Applies a bounded prefix after exclusive policy handoff. Every attempted
// record spends one decision, including stale or failed decisions. Resuming
// starts at applied_count; reset explicitly discards an unattempted suffix.
// Application failures latch driver stopping/status while the plan retains
// each source outcome. A remaining suffix reports progress to request another
// owner turn; it never loops around the cap or replays an earlier acceptance.
// Accepted service/resource obligations must progress independently of this
// new-work budget, including when the budget is zero.
void loom_runtime_driver_apply_plan(loom_runtime_driver_t* driver,
                                    loom_runtime_service_plan_t* plan,
                                    uint64_t* inout_decisions,
                                    bool* out_progress);

// Synchronous native policy. Reads controls through the existing stable slots;
// it neither invokes a source nor mutates that table. The previous plan is
// exhausted on entry: policy can consume its outcomes, then resets it before
// emitting independent decisions. Return transfers exclusive output ownership.
typedef struct loom_runtime_service_policy_t {
  // Borrowed policy state, retained for every invocation.
  void* self;
  // Bounded infallible selection over trusted controls and constructed storage.
  // Empty output returns to the application; timed deferral sets wake_time_ns.
  void (*select)(void* self, const loom_runtime_driver_t* driver,
                 loom_runtime_service_schedule_context_t* context,
                 loom_runtime_service_plan_t* plan);
} loom_runtime_service_policy_t;

// Traversal position for the native policy that selects each offered choice
// zero. A full output resumes at the first omitted offer on its next
// invocation. Zero-initialize before the first invocation. No per-service
// policy array.
typedef struct loom_runtime_service_default_policy_t {
  // Next coordinate in the existing stable dispatch table.
  iree_host_size_t service_ordinal;
  // Next offer within that service's immutable bounds.
  uint64_t offer_ordinal;
} loom_runtime_service_default_policy_t;

// Returns an explicit native policy over the caller-owned traversal state.
loom_runtime_service_policy_t loom_runtime_service_default_policy(
    loom_runtime_service_default_policy_t* state);

// Selects and applies bounded new work after the caller progresses accepted
// service/resource obligations. Resumes an unattempted suffix before invoking
// policy again. Every policy call spends an evaluation; every attempt spends a
// decision. An empty selection ends this call even if budget remains.
// context.now_ns is sampled once by the application for this turn. The output
// wake_time_ns is the earliest requested deadline, or now_ns when a cap leaves
// further scheduling work. The application integrates that deadline with its
// existing wait/timer and may wake earlier for actual service/resource events.
// An initially zero budget withholds that phase without requesting an immediate
// wake. The owner supplies its next scheduling opportunity; an unapplied plan
// remains retained. A positive budget exhausted by this call yields another
// turn. Native completion is the handoff; no plan count is a readiness signal.
void loom_runtime_driver_schedule(
    loom_runtime_driver_t* driver, loom_runtime_service_policy_t policy,
    loom_runtime_service_plan_t* plan,
    loom_runtime_service_schedule_budget_t* budget,
    loom_runtime_service_schedule_context_t* context, bool* out_progress);

// Collects pending ownership after the turn and any adapter delivery. A late
// abort requires a complete stopping turn before detachment, even when earlier
// services own queued requests but no asynchronous callbacks. Once that turn
// completes, pending services owe the next wake; collection never polls them.
bool loom_runtime_driver_collect(loom_runtime_driver_t* driver,
                                 bool* out_progress);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_DRIVER_H_
