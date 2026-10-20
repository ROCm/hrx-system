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

// Immutable request policy, independent of residency and source compatibility.
typedef enum loom_serve_execution_qos_e {
  LOOM_SERVE_EXECUTION_QOS_BACKGROUND = 0,
  LOOM_SERVE_EXECUTION_QOS_REALTIME = 1,
} loom_serve_execution_qos_t;

// Indexed application resources, resolved before model construction. A bit
// denotes an actual bound resource, never a queue or model registration.
typedef uint64_t loom_serve_execution_resources_t;

typedef enum loom_serve_execution_coexistence_e {
  // Different classes take separate turns across this entire scope.
  LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE = 0,
  // Different classes may overlap subject to their individual resource limits.
  LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP = 1,
} loom_serve_execution_coexistence_t;

// Resources can interfere without sharing an admission count. Overlapping
// scopes compose their restrictions; an overlap scope never cancels exclusion.
typedef struct loom_serve_execution_scope_t {
  // Nonempty subset of the application's resolved resource namespace.
  loom_serve_execution_resources_t resources;
  // Cross-class interference policy, independent of per-resource limits.
  loom_serve_execution_coexistence_t coexistence;
} loom_serve_execution_scope_t;

// Cold application input, borrowed only during scheduler allocation. An
// explicit policy covers every resource. NULL policy selects one exclusion
// scope over all resources, with no additional software count limit.
typedef struct loom_serve_execution_policy_t {
  // Optional resource_count entries limiting accepted units on each resource.
  // Zero adds no count limit. These are claims, not cores, queues, or
  // occupancy.
  const uint32_t* max_active;
  // Number of immutable interference declarations.
  iree_host_size_t scope_count;
  // Borrowed declarations; missing resource coverage is an input error.
  const loom_serve_execution_scope_t* scopes;
} loom_serve_execution_policy_t;

typedef struct loom_serve_execution_scheduler_t
    loom_serve_execution_scheduler_t;
typedef struct loom_serve_execution_source_t loom_serve_execution_source_t;
typedef struct loom_serve_execution_permit_t loom_serve_execution_permit_t;

// One serialized application owner manages all resources and accepted claims.
// Allocation validates external policy once; hot consumers use resolved facts.
// resource_count is in [1, 64]. Failure returns no scheduler.
iree_status_t loom_serve_execution_scheduler_allocate(
    iree_host_size_t resource_count,
    const loom_serve_execution_policy_t* policy,
    iree_allocator_t host_allocator,
    loom_serve_execution_scheduler_t** out_scheduler);

// Every source is detached and every model has retired before destruction.
void loom_serve_execution_scheduler_free(
    loom_serve_execution_scheduler_t* scheduler);

// Current eligible realtime demand before source planning or numerical offer.
// Called only on the serialized owner. The callback reads owner-owned state;
// it never submits, allocates, mutates plans, or reads live CPU-task payloads.
// Empty input, absent output credit, and unavailable physical admission are not
// ready demand. Accepted work is accounted independently of this observation.
typedef loom_serve_execution_resources_t (*loom_serve_execution_demand_fn_t)(
    void* self);

// Stable service-owned observation, detached before service storage is freed.
// An accepted model claim survives detachment and retains its own lifetime.
struct loom_serve_execution_source_t {
  // Borrowed application owner; NULL denotes independently managed execution.
  loom_serve_execution_scheduler_t* scheduler;
  // Previous live observation on the serialized owner.
  loom_serve_execution_source_t* previous;
  // Next live observation on the serialized owner.
  loom_serve_execution_source_t* next;
  // Pure current-demand query, valid through detachment.
  loom_serve_execution_demand_fn_t query;
  // Borrowed service state, never accessed by the scheduler after detachment.
  void* self;
};

void loom_serve_execution_source_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_demand_fn_t query, void* self,
    loom_serve_execution_source_t* out_source);
void loom_serve_execution_source_deinitialize(
    loom_serve_execution_source_t* source);

typedef enum loom_serve_execution_permit_phase_e {
  LOOM_SERVE_EXECUTION_PERMIT_IDLE = 0,
  LOOM_SERVE_EXECUTION_PERMIT_WAITING,
  LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED,
} loom_serve_execution_permit_phase_t;

// Stable model-owned record, retained through actual native retirement. Each
// accepted unit claims all its bound resources atomically. Preparation,
// execution, and retirement keep their actual ownership; this object owns no
// plan, callback, or buffer. NULL permits explicitly bypass coordination.
struct loom_serve_execution_permit_t {
  // Borrowed application owner through withdrawal or actual model retirement.
  loom_serve_execution_scheduler_t* scheduler;
  // Actual resources used by this operation, resolved by its binding owner.
  loom_serve_execution_resources_t resources;
  // Previous live offer or accepted claim.
  loom_serve_execution_permit_t* previous;
  // Next live offer or accepted claim.
  loom_serve_execution_permit_t* next;
  // Owner-only lifecycle, independent of readiness notification.
  loom_serve_execution_permit_phase_t phase;
  // Immutable class for the current offer and its accepted execution.
  loom_serve_execution_qos_t qos;
};

void loom_serve_execution_permit_initialize(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_resources_t resources,
    loom_serve_execution_permit_t* out_permit);

// Offers one exact prepared unit from IDLE. Replacing a plan first withdraws
// its offer; neither operation spends source request priority or history.
void loom_serve_execution_permit_request(loom_serve_execution_permit_t* permit,
                                         loom_serve_execution_qos_t qos);

// Observes whether another owner turn can accept waiting work. This conveys no
// persistent grant: demand is checked again at actual acquisition. False needs
// an existing producer/native/network wake, never a polling timer.
bool loom_serve_execution_scheduler_is_ready(
    loom_serve_execution_scheduler_t* scheduler);

// Current cross-class pressure for planning. Resource count limits apply only
// at acceptance; they cannot require smaller executable shapes. A true result
// defers new conflicting work without cancelling requests or evicting weights.
bool loom_serve_execution_is_preempted(
    loom_serve_execution_scheduler_t* scheduler,
    loom_serve_execution_resources_t resources, loom_serve_execution_qos_t qos);

// Checks current demand and every resource together immediately before submit.
// False leaves WAITING intact. Accepted work cannot be revoked or resized.
bool loom_serve_execution_permit_try_acquire(
    loom_serve_execution_permit_t* permit);

// Releases ACCEPTED permission after successful numerical readiness, before CPU
// interpretation/publication. Failure notification alone cannot release it.
void loom_serve_execution_permit_complete(
    loom_serve_execution_permit_t* permit);

// Withdraws an unaccepted offer. IDLE and ACCEPTED are inert: cancelled client
// interest cannot revoke a native loan. The source separately releases plans.
void loom_serve_execution_permit_cancel(loom_serve_execution_permit_t* permit);

// Detaches final model storage after actual native retirement. Only the model
// owner may discharge an accepted claim here; service destruction cannot.
void loom_serve_execution_permit_deinitialize(
    loom_serve_execution_permit_t* permit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_SCHEDULER_H_
