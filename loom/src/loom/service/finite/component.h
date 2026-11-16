// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_COMPONENT_H_
#define LOOM_SERVICE_FINITE_COMPONENT_H_

#include "loom/runtime/service_control.h"
#include "loom/runtime/wake.h"
#include "loom/service/finite/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Deployment-owned optional finite model. Preparation and residency are
// independent of the service borrowing this handle. One serialized host owner
// makes every call; asynchronous callbacks only schedule that owner.
typedef struct loom_service_finite_component_t loom_service_finite_component_t;

typedef enum loom_service_finite_component_phase_e {
  LOOM_SERVICE_FINITE_COMPONENT_DECLARED = 0,
  LOOM_SERVICE_FINITE_COMPONENT_PREPARING,
  LOOM_SERVICE_FINITE_COMPONENT_READY,
  LOOM_SERVICE_FINITE_COMPONENT_FAILED,
  LOOM_SERVICE_FINITE_COMPONENT_RELEASING,
  LOOM_SERVICE_FINITE_COMPONENT_RETIRED,
} loom_service_finite_component_phase_t;

// Allocates only the dormant owner. No source/checkpoint access, compilation,
// model allocation, or parameter reservation occurs. Device, pool, options'
// views, package and observer remain borrowed through retirement and
// destruction. The observer must schedule the shared owner without reentering
// this object. Scheduling supplies one exclusive offer/choice and one operation
// with one device use each. Its storage and summary publisher survive actual
// component retirement. Reusing these slots for another component requires a
// cold control incarnation change after all readers and publishers join.
iree_status_t loom_service_finite_component_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* options,
    loom_runtime_wake_callback_t observer,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator,
    loom_service_finite_component_t** out_component);

// Copies the immutable attachment on the serialized owner. A composition uses
// these coordinates to dispatch component offers alongside its other owners.
loom_runtime_service_control_binding_t
loom_service_finite_component_scheduling_binding(
    const loom_service_finite_component_t* component);

// Publishes shared preparation demand without accessing sources or submitting
// work. Repeated demand shares the same offer/attempt. Failure and shutdown
// suppress further attempts; cancelling one request does not cancel this
// component's shared demand.
void loom_service_finite_component_request_preparation(
    loom_service_finite_component_t* component);

// Lends the prepared model only while READY. NULL means no available model;
// status distinguishes failed preparation from unaccepted or pending work.
// The borrower excludes trim and retirement until it relinquishes the model.
loom_service_finite_model_t* loom_service_finite_component_prepared_model(
    const loom_service_finite_component_t* component);

// Accepts the current preparation offer without waiting. Stale choices are
// unaccepted successes. Native errors remain component-owned and are reported
// independently of acceptance; the parent may continue unrelated requests.
loom_runtime_service_schedule_outcome_t
loom_service_finite_component_try_schedule(
    loom_service_finite_component_t* component,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal);

loom_service_finite_component_phase_t loom_service_finite_component_phase(
    const loom_service_finite_component_t* component);

// Cumulative preparation elapsed time, including the open attempt. Declared
// gaps, weight activation, invocation, and retirement add none. Fixed-backing
// re-preparation adds another interval; observation never starts preparation.
uint64_t loom_service_finite_component_preparation_duration_ns(
    const loom_service_finite_component_t* component);

// Explicitly grants the first preparation, or observes that same attempt.
// Success with *out_model == NULL means pending; the observer schedules further
// owner turns. Success with a model lends it until quiescent trim or
// retirement. No parameter activation occurs. A failed attempt retains its
// diagnostic and returns a clone to each consumer instead of retrying I/O/JIT
// automatically. Request cancellation relinquishes the caller's interest, not
// this shared preparation. The caller excludes concurrent trim and deployment
// shutdown.
iree_status_t loom_service_finite_component_prepare(
    loom_service_finite_component_t* component,
    loom_service_finite_model_t** out_model);

// Advances accepted preparation without starting a dormant component. Returns
// whether a terminal outcome was consumed. Failures remain component-owned;
// prepare returns them to affected requests without failing the parent service.
// stopping is irreversible deployment shutdown, never request cancellation.
// It cancels subsequent load phases and joins accepted work. NULL is inert.
bool loom_service_finite_component_advance(
    loom_service_finite_component_t* component, bool stopping);

// Accepted CPU/JIT work still needs owner turns. Does not include finite model
// invocations: the borrowing service owns and joins those separately.
bool loom_service_finite_component_is_pending(
    const loom_service_finite_component_t* component);

// Cumulative evidence from terminal preparation attempts. A cold declaration
// has an empty report; elastic reactivation adds no compilation. The pointer
// remains valid until destruction and is read only on the serialized owner.
const loom_runtime_compilation_report_t* loom_service_finite_component_report(
    const loom_service_finite_component_t* component);

// Borrows the current preparation diagnostic on the serialized owner. The
// component retains ownership; observation never starts or retries loading.
iree_status_t loom_service_finite_component_status(
    const loom_service_finite_component_t* component);

// Copies parameter state when a prepared model exists, including during final
// release. False means no residency exists; it is not an inactive fixed model.
bool loom_service_finite_component_parameters(
    const loom_service_finite_component_t* component,
    loom_runtime_residency_snapshot_t* out_parameters);

// Persistent eviction policy for present and subsequently prepared parameters.
// Does not start preparation, activate weights, or change invocation-owned
// pins.
void loom_service_finite_component_set_pinned(
    loom_service_finite_component_t* component, bool pinned);

// Quiescent policy boundary: all model CPU/device/variant borrowers have
// retired and no new consumer or pin-policy change may begin until
// *out_complete. Elastic backing releases only unpinned weights, retaining
// compiled code and virtual roots.
// Nonterminal release owes a pool-owner wake; subsequent calls join it.
// Fixed backing retires the finite model; the next prepare recreates it.
// Independent device results do not borrow the model or its parameters and
// remain usable through either release path.
// Pinned parameters remain unchanged; completion reports that the policy visit
// finished, not that the component's parameter commitment reached zero.
// Neither path changes another model's residency or retained state. A declared
// or failed component has no live parameter payload to trim. Preparation must
// first finish; this operation does not cancel shared demand.
iree_status_t loom_service_finite_component_trim(
    loom_service_finite_component_t* component, bool* out_complete);

// Irreversible removal after invocation borrowers retire. Cancels/joins source
// work, then retires the model's actual queue ownership without blocking the
// shared worker. Call until complete, then free. Cleanup failure preserves the
// remaining ownership for the caller; retained preparation errors do not
// prevent cleanup. NULL is already retired.
iree_status_t loom_service_finite_component_retire(
    loom_service_finite_component_t* component, bool* out_complete);

// Consumes a fully retired or still-dormant handle. NULL is accepted. The
// borrowed definition, source package, observer and device may be released
// after this call.
void loom_service_finite_component_free(
    loom_service_finite_component_t* component);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_FINITE_COMPONENT_H_
