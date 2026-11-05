// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_REGISTRATION_H_
#define LOOM_SERVICE_FINITE_REGISTRATION_H_

#include "loom/runtime/parameter/activation.h"
#include "loom/service/finite/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owns preparation, the finite service, and final model retirement. All calls
// use one serialized owner; callbacks only notify that owner. Request receipts
// have independent ownership and may outlive this registration.
typedef struct loom_service_finite_registration_t
    loom_service_finite_registration_t;

typedef struct loom_service_finite_registration_options_t {
  // Immutable modality contract, borrowed through retirement.
  const loom_service_finite_protocol_t* protocol;
  // Deployment identifier, borrowed through retirement.
  iree_string_view_t name;
  // Maximum outstanding requests, including the active cohort.
  iree_host_size_t capacity;
  // Retained execution choices; zero disables reconfiguration.
  iree_host_size_t variant_capacity;
  // Nonnegative observation interval; zero disables periodic samples.
  iree_duration_t heartbeat_interval;
  // Borrowed registration history, or NULL for service-owned history.
  loom_runtime_telemetry_stream_t* stream;
} loom_service_finite_registration_options_t;

// Creates an unstarted owner without source I/O, device allocation, or work
// submission. Copies option values; model option views remain borrowed through
// terminal preparation. Device, pool, protocol, name, stream, and observer
// payload survive retirement. Failure returns no owner.
iree_status_t loom_service_finite_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* model_options,
    const loom_service_finite_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_service_finite_registration_t** out_registration);

// Advances preparation or irrevocable cancellation without waiting. A
// nonterminal result owes a wake after accepted work retires. Terminal success
// lends the service until retire; terminal failure publishes no service but
// still requires retirement. Copies compilation evidence at terminal
// completion, including failure. Do not call again after terminal completion.
iree_status_t loom_service_finite_registration_prepare(
    loom_service_finite_registration_t* registration,
    loom_service_finite_load_flags_t flags, bool* out_complete,
    loom_service_finite_service_t** out_service,
    loom_runtime_compilation_report_t* out_report);

// Copies ready-model parameter accounting on the serialized owner.
loom_runtime_residency_snapshot_t loom_service_finite_registration_parameters(
    const loom_service_finite_registration_t* registration);

// Changes ready-model eviction policy without loading or changing admission.
void loom_service_finite_registration_set_pinned(
    loom_service_finite_registration_t* registration, bool pinned);

// Explicit warmup after admitted service work retires. Call until COMPLETE,
// including on cancellation, before permitting service work or retirement.
iree_status_t loom_service_finite_registration_activate(
    loom_service_finite_registration_t* registration,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase);

// Releases reloadable parameters after service activity and warmup retire.
// Nonterminal success owes an ownership wake; failure preserves ownership.
iree_status_t loom_service_finite_registration_deactivate(
    loom_service_finite_registration_t* registration, bool* out_complete);

// Begins after preparation terminates and service/tokenizer borrowers retire.
// Releases the service, then joins actual model ownership through the original
// observer, not readiness values. Nonterminal success owes a wake. A platform
// release failure preserves ownership for explicit retry. NULL is retired.
iree_status_t loom_service_finite_registration_retire(
    loom_service_finite_registration_t* registration, bool* out_complete);

// Frees a successfully retired registration. NULL is accepted. Borrowed
// definition/resource/observer storage can be released after this call.
void loom_service_finite_registration_free(
    loom_service_finite_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_FINITE_REGISTRATION_H_
