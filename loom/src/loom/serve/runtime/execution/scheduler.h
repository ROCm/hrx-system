// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_SCHEDULER_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_SCHEDULER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_execution_permit_t loom_serve_execution_permit_t;

// Cooperative numerical permission on one serialized host owner. This object
// owns no source plans, native bindings, memory reservations, or callbacks.
// Every participant retains its prepared work and its actual device lifetime.
// Distinct controllers imply no claim about physical resource independence.
typedef struct loom_serve_execution_scheduler_t {
  // Maximum granted/accepted units; zero leaves submissions unrestricted.
  uint32_t concurrency;
  // Granted/accepted units, each retaining its own completion boundary.
  uint32_t outstanding_count;
  // First live permit, preserving offer order for equal virtual service.
  loom_serve_execution_permit_t* head;
  // Last live permit; NULL exactly when head is NULL.
  loom_serve_execution_permit_t* tail;
  // Monotonically advancing floor in weighted service nanoseconds.
  uint64_t minimum;
} loom_serve_execution_scheduler_t;

typedef enum loom_serve_execution_permit_phase_e {
  LOOM_SERVE_EXECUTION_PERMIT_IDLE = 0,
  LOOM_SERVE_EXECUTION_PERMIT_WAITING,
  LOOM_SERVE_EXECUTION_PERMIT_GRANTED,
  LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED,
} loom_serve_execution_permit_phase_t;

// Stable, embedded permission record for independently submitted numerical
// work. Reused across batches; never moved while offered or accepted. A service
// may own several records for independent executions. Prepared model state and
// output credit remain with the service, not this accounting record.
typedef struct loom_serve_execution_permit_t {
  // Borrowed controller through withdrawal/retirement; NULL is uncoordinated.
  loom_serve_execution_scheduler_t* scheduler;
  // Previous live record; linked through waiting, grant, and acceptance.
  loom_serve_execution_permit_t* previous;
  // Next live record; unlinked only by withdrawal or numerical completion.
  loom_serve_execution_permit_t* next;
  // Owner-published permission lifecycle, independent of model readiness.
  loom_serve_execution_permit_phase_t phase;
  // Positive relative share captured for this exact prepared numerical unit.
  uint32_t weight;
  // Accumulated weighted service nanoseconds, clipped to the floor on reentry.
  uint64_t service_time;
  // Host monotonic acceptance time, valid only in ACCEPTED phase.
  iree_time_t start_time;
} loom_serve_execution_permit_t;

// Initializes fixed owner storage before any offer. The controller outlives
// every permit; it has no asynchronous work or allocation to release.
void loom_serve_execution_scheduler_initialize(
    uint32_t concurrency, loom_serve_execution_scheduler_t* out_scheduler);
void loom_serve_execution_permit_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_permit_t* out_permit);

// Offers one source-prepared unit from IDLE. Weight is positive trusted owner
// policy, not a source geometry tag or a duration estimate. The service retains
// the exact plan until withdrawal or successful acceptance and retirement.
// Unrestricted/uncoordinated execution grants immediately. Offering is
// progress; repeatedly observing an ungranted offer is not.
void loom_serve_execution_permit_request(loom_serve_execution_permit_t* permit,
                                         uint32_t weight);

// Selects at most one ready unit within the outstanding budget after
// participants have progressed. Ready work cannot overtake less-served active
// peers merely because capacity is available. Accepted service includes elapsed
// time through now in the same monotonic domain as acceptance/completion.
// True means a new grant needs another owner turn. False needs an existing
// native/network/producer wake, never a polling timer. Continuously ready
// positive-weight participants in a finite set receive service when accepted
// units complete in finite time. Grants count against the budget before the
// owner takes them; numerical completion frees only that unit's slot.
bool loom_serve_execution_scheduler_advance(
    loom_serve_execution_scheduler_t* scheduler, iree_time_t now);

// Takes a granted offer immediately before the existing model submit call.
// False leaves the WAITING offer intact. The caller checks cancellation before
// taking permission; already accepted work cannot be revoked. now is the host
// monotonic observation used by the caller's submission accounting.
bool loom_serve_execution_permit_try_acquire(
    loom_serve_execution_permit_t* permit, iree_time_t now);

// Ends ACCEPTED permission at owner-observed device readiness, before CPU
// output interpretation/publication. The interval is observed service time, not
// device occupancy; overlapping units each include their observed interference.
// Real model buffers retain their independent retirement contract.
void loom_serve_execution_permit_complete(loom_serve_execution_permit_t* permit,
                                          iree_time_t now);

// Withdraws an unaccepted offer, including an untaken grant. IDLE is inert.
// The source owner separately releases its prepared plan without numerical
// mutation. Accepted units instead complete after readiness.
void loom_serve_execution_permit_cancel(loom_serve_execution_permit_t* permit);

// Detaches final service storage. No accepted work may remain unless the owner
// has stopped all new grants and will join failed native ownership separately.
// This removes accounting only; it never retires or releases model buffers.
void loom_serve_execution_permit_deinitialize(
    loom_serve_execution_permit_t* permit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_SCHEDULER_H_
