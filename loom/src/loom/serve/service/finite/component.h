// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_COMPONENT_H_
#define LOOM_SERVE_SERVICE_FINITE_COMPONENT_H_

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/service/finite/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Deployment-owned optional finite model. Preparation and residency are
// independent of the service borrowing this handle. One serialized host owner
// makes every call; asynchronous callbacks only schedule that owner.
typedef struct loom_serve_finite_component_t loom_serve_finite_component_t;

typedef enum loom_serve_finite_component_phase_e {
  LOOM_SERVE_FINITE_COMPONENT_DECLARED = 0,
  LOOM_SERVE_FINITE_COMPONENT_PREPARING,
  LOOM_SERVE_FINITE_COMPONENT_READY,
  LOOM_SERVE_FINITE_COMPONENT_FAILED,
  LOOM_SERVE_FINITE_COMPONENT_RELEASING,
  LOOM_SERVE_FINITE_COMPONENT_RETIRED,
} loom_serve_finite_component_phase_t;

// Allocates only the dormant owner. No source/checkpoint access, compilation,
// model allocation, or parameter reservation occurs. Device, options' views,
// package and observer remain borrowed through retirement and destruction.
// The observer must schedule the shared owner without reentering this object.
iree_status_t loom_serve_finite_component_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_finite_component_t** out_component);

loom_serve_finite_component_phase_t loom_serve_finite_component_phase(
    const loom_serve_finite_component_t* component);

// Starts the first preparation on demand, or observes that same attempt.
// Success with *out_model == NULL means pending; the observer schedules further
// owner turns. Success with a model lends it until quiescent trim or
// retirement. No parameter activation occurs. A failed attempt retains its
// diagnostic and returns a clone to each consumer instead of retrying I/O/JIT
// automatically. Request cancellation relinquishes the caller's interest, not
// this shared preparation. The caller excludes concurrent trim and deployment
// shutdown.
iree_status_t loom_serve_finite_component_prepare(
    loom_serve_finite_component_t* component,
    loom_serve_finite_model_t** out_model);

// Advances accepted preparation without starting a dormant component. Returns
// whether a terminal outcome was consumed. Failures remain component-owned;
// prepare returns them to affected requests without failing the parent service.
// stopping is irreversible deployment shutdown, never request cancellation.
// It cancels subsequent load phases and joins accepted work. NULL is inert.
bool loom_serve_finite_component_advance(
    loom_serve_finite_component_t* component, bool stopping);

// Accepted CPU/JIT work still needs owner turns. Does not include finite model
// invocations: the borrowing service owns and joins those separately.
bool loom_serve_finite_component_is_pending(
    const loom_serve_finite_component_t* component);

// Cumulative evidence from terminal preparation attempts. A cold declaration
// has an empty report; elastic reactivation adds no compilation. The pointer
// remains valid until destruction and is read only on the serialized owner.
const loom_serve_compilation_report_t* loom_serve_finite_component_report(
    const loom_serve_finite_component_t* component);

// Quiescent policy boundary: all model CPU/device/variant borrowers have
// retired and no new consumer may begin until *out_complete. Elastic backing
// releases only unpinned weights, retaining compiled code and virtual roots.
// Fixed backing retires the finite model; the next prepare recreates it.
// Pinned parameters remain unchanged; completion reports that the policy visit
// finished, not that the component's parameter commitment reached zero.
// Neither path changes another model's residency or retained state. A declared
// or failed component has no live parameter payload to trim. Preparation must
// first finish; this operation does not cancel shared demand.
iree_status_t loom_serve_finite_component_trim(
    loom_serve_finite_component_t* component, bool* out_complete);

// Irreversible removal after invocation borrowers retire. Cancels/joins source
// work, then retires the model's actual queue ownership without blocking the
// shared worker. Call until complete, then free. Cleanup failure preserves the
// remaining ownership for the caller; retained preparation errors do not
// prevent cleanup. NULL is already retired.
iree_status_t loom_serve_finite_component_retire(
    loom_serve_finite_component_t* component, bool* out_complete);

// Consumes a fully retired or still-dormant handle. NULL is accepted. The
// borrowed definition, source package, observer and device may be released
// after this call.
void loom_serve_finite_component_free(loom_serve_finite_component_t* component);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_COMPONENT_H_
