// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_REGISTRATION_H_
#define LOOM_SERVICE_FINITE_REGISTRATION_H_

#include "loom/runtime/parameter/activation.h"
#include "loom/runtime/service.h"
#include "loom/service/finite/model.h"
#include "loom/service/finite/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owns preparation, the finite service, and final model retirement. All calls
// use one serialized owner; callbacks only notify that owner. Request receipts
// have independent ownership and may outlive this registration.
typedef struct loom_service_finite_registration_t
    loom_service_finite_registration_t;

typedef enum loom_service_finite_registration_phase_e {
  LOOM_SERVICE_FINITE_REGISTRATION_DECLARED = 0,
  LOOM_SERVICE_FINITE_REGISTRATION_PREPARING,
  LOOM_SERVICE_FINITE_REGISTRATION_READY,
  LOOM_SERVICE_FINITE_REGISTRATION_FAILED,
  LOOM_SERVICE_FINITE_REGISTRATION_RETIRING,
  LOOM_SERVICE_FINITE_REGISTRATION_RETIRED,
  LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED,
} loom_service_finite_registration_phase_t;

typedef enum loom_service_finite_parameter_operation_e {
  LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE = 0,
  LOOM_SERVICE_FINITE_PARAMETER_OPERATION_ACTIVATE,
  LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE,
} loom_service_finite_parameter_operation_t;

// Owner-thread state, independent of concurrent scheduling observations. Every
// pointer is borrowed through the next owner mutation; nothing transfers
// resource or diagnostic ownership to the observer.
typedef struct loom_service_finite_registration_snapshot_t {
  // Native preparation/publication/retirement phase.
  loom_service_finite_registration_phase_t phase;
  // Requested or accepted parameter operation, independent of model phase.
  loom_service_finite_parameter_operation_t parameter_operation;
  // Ready engine; NULL in all other phases. Existing engine views remain
  // borrowed until the application's explicit retirement handoff.
  loom_service_finite_service_t* service;
  // Borrowed accumulated native failure; the registration owns the diagnostic.
  iree_status_t status;
  // Borrowed terminal preparation evidence, retained through model retirement.
  const loom_runtime_compilation_report_t* compilation;
} loom_service_finite_registration_snapshot_t;

typedef struct loom_service_finite_registration_options_t {
  // Nonzero composition-assigned lifetime, changed when a slot is reused.
  uint64_t incarnation;
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

loom_service_finite_registration_snapshot_t
loom_service_finite_registration_snapshot(
    const loom_service_finite_registration_t* registration);

// Requests one parameter operation without starting it. The application closes
// admission until this intent completes; already admitted requests drain before
// its offer becomes legal. Deactivation requires elastic backing and waits for
// persistent retention to be unpinned. NONE withdraws an unaccepted request.
// An accepted operation must complete before another can replace it. This call
// performs no source I/O, compilation, device submission, or readiness wait.
iree_status_t loom_service_finite_registration_request_parameters(
    loom_service_finite_registration_t* registration,
    loom_service_finite_parameter_operation_t operation);

enum loom_service_finite_retire_flag_bits_e {
  // Cancels queued work and joins accepted work instead of draining requests.
  LOOM_SERVICE_FINITE_RETIRE_FLAG_ABORT = 1u << 0,
};
typedef uint32_t loom_service_finite_retire_flags_t;

// Irreversible handoff after future admission stops and external service or
// tokenizer views join. Independent request receipts may remain alive. Normal
// removal offers release after admitted requests finish. ABORT cancels them;
// accepted cleanup then progresses even without new scheduling grants. Native
// failures also use that cleanup path after this handoff. Calling again after
// RELEASE_FAILED explicitly retries the retained platform release.
void loom_service_finite_registration_request_retirement(
    loom_service_finite_registration_t* registration,
    loom_service_finite_retire_flags_t flags);

// Consumes accepted native work without starting another offered unit. Reports
// actual progress; failures remain in snapshot.status and do not fail peers.
// STOPPING joins/cancels accepted work but cannot destroy an externally
// borrowed engine: request_retirement supplies that separate ownership handoff.
bool loom_service_finite_registration_progress(
    loom_service_finite_registration_t* registration,
    loom_runtime_service_flags_t flags);

// Claims one currently advertised lifecycle or engine choice without waiting.
// Stale choices return an unaccepted OK outcome. Native errors are retained in
// snapshot.status; acceptance remains independent of failure and actual
// cleanup. Direct prepare/activate/deactivate/retire calls below are explicit
// grants of those same native operations and share their ownership/progress
// paths.
loom_runtime_service_schedule_outcome_t
loom_service_finite_registration_try_schedule(
    loom_service_finite_registration_t* registration,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags);

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
// Failure can retain parameter protection; model retirement releases it.
iree_status_t loom_service_finite_registration_activate(
    loom_service_finite_registration_t* registration,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase);

// Releases reloadable parameters after service activity and warmup retire.
// Nonterminal success owes an ownership wake; failure preserves ownership.
iree_status_t loom_service_finite_registration_deactivate(
    loom_service_finite_registration_t* registration, bool* out_complete);

// Discards an unstarted owner, or begins after preparation terminates and
// service/tokenizer borrowers retire.
// Releases the service, then joins actual model ownership through the original
// observer, not readiness values. Nonterminal success owes a wake. A platform
// release failure preserves ownership for explicit retry. NULL is retired.
iree_status_t loom_service_finite_registration_retire(
    loom_service_finite_registration_t* registration, bool* out_complete);

// Frees a successfully retired registration. NULL is accepted. Borrowed
// definition/resource/observer storage can be released after this call.
void loom_service_finite_registration_free(
    loom_service_finite_registration_t* registration);

// Borrowed observations include preparation, parameter operations, ready-engine
// requests, and final model retirement. The registration publishes the summary;
// the engine contributes facts at its mutation boundaries. Failed activation
// and final retirement have independent operation slots because readiness
// failure does not release parameter protection. Observations survive engine
// and model retirement until free; reader/policy borrows must join before free.
const loom_runtime_service_control_t* loom_service_finite_registration_control(
    const loom_service_finite_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_FINITE_REGISTRATION_H_
