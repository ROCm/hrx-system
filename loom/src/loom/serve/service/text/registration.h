// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_
#define LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_

#include "loom/runtime/parameter/activation.h"
#include "loom/serve/service/text/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owns text preparation, its optional finite image component, the ready
// service, and final model retirement. Calls use one serialized owner;
// callbacks only notify that owner. Detached request results may outlive it.
//
// An embedder supplies its existing device/pool and a nonblocking wake target:
//   allocate -> prepare until complete -> borrow the ready service
//   -> stop and collect service borrowers -> retire until complete -> free.
// Preparation failure skips the ready-service phase but still requires retire.
// The ready service's interface supplies a loom_runtime_service_t for the
// shared driver; its typed try_open/try_accept calls provide admission. The
// embedder advances the device group alongside that driver and retains the
// registration's observer until final retirement. No HTTP owner is required.
typedef struct loom_serve_text_registration_t loom_serve_text_registration_t;

// Independently resident members of a text deployment. IMAGE is valid only
// when image model options were supplied at construction.
typedef enum loom_serve_text_component_e {
  LOOM_SERVE_TEXT_COMPONENT_PRIMARY = 0,
  LOOM_SERVE_TEXT_COMPONENT_IMAGE,
} loom_serve_text_component_t;

typedef struct loom_serve_text_registration_options_t {
  // Deployment identifier, borrowed through retirement.
  iree_string_view_t name;
  // Maximum resident request owners; zero imposes no count limit.
  iree_host_size_t max_sessions;
  // Maximum known input tokens admitted from one row per epoch.
  iree_host_size_t chunk_size;
  // Output bound when a request omits max_tokens.
  iree_host_size_t default_max_tokens;
  // Positive bound on requests waiting without a model row.
  iree_host_size_t pending_capacity;
  // Retained execution choices; zero disables reconfiguration.
  iree_host_size_t variant_capacity;
  // Successful overtakes allowed per blocked oldest request; zero is FIFO.
  iree_host_size_t admission_overtakes;
  // Nonnegative observation interval; zero disables periodic samples.
  iree_duration_t heartbeat_interval;
  // Execution choice within the shared admission and retirement lifecycle.
  loom_service_text_schedule_mode_t schedule_mode;
  // Whether prompt and decode inputs may share an epoch.
  loom_service_text_packing_mode_t packing_mode;
  // One or two device-fed epochs between owner observations.
  iree_host_size_t continuation_epochs;
  // Borrowed registration history, or NULL for service-owned history.
  loom_runtime_telemetry_stream_t* stream;
} loom_serve_text_registration_options_t;

// Creates an unstarted owner without source I/O, device allocation, or work
// submission. Copies option values and epoch shapes into owner storage; other
// text model views remain borrowed through terminal preparation. Optional
// image model views survive retirement because preparation follows demand.
// Device, pool, name, stream, and observer payload survive retirement.
// Failure returns no owner. Initial rows and cache capacities come from the
// prepared model, never a second independently configured service geometry.
iree_status_t loom_serve_text_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_serve_text_options_t* model_options,
    const loom_service_finite_model_options_t* image_options,
    const loom_serve_text_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_text_registration_t** out_registration);

// Advances preparation or irrevocable cancellation without waiting. A
// nonterminal result owes a wake after accepted work retires. Terminal success
// lends the service until retire; failure publishes no service but still
// requires retirement. Copies compilation evidence at terminal completion,
// including failure. Do not call again after terminal completion.
iree_status_t loom_serve_text_registration_prepare(
    loom_serve_text_registration_t* registration,
    loom_serve_text_load_flags_t flags, bool* out_complete,
    loom_serve_text_service_t** out_service,
    loom_runtime_compilation_report_t* out_report);

// Copies ready text-model parameter accounting on the serialized owner.
loom_runtime_residency_snapshot_t loom_serve_text_registration_parameters(
    const loom_serve_text_registration_t* registration);

// Borrows the optional component for read-only observations on the serialized
// owner. NULL means absent or retired. Reacquire after any retirement call;
// component destruction precedes final text-model retirement.
const loom_service_finite_component_t* loom_serve_text_registration_image(
    const loom_serve_text_registration_t* registration);

// Changes a declared member's eviction policy without loading or admitting
// work. Preparation has completed; IMAGE policy also applies while dormant.
void loom_serve_text_registration_set_pinned(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component, bool pinned);

// Explicit warmup after service work retires. IMAGE may start preparation;
// its source failure is returned separately in out_component_status without
// failing the text deployment. Call until COMPLETE, including cancellation,
// before permitting service work or retirement. The member remains fixed for
// that operation. Both returned statuses are owned by the caller.
iree_status_t loom_serve_text_registration_activate(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status);

// Releases a declared member after service activity and warmup retire. IMAGE
// first relinquishes the idle projector's borrow. Nonterminal success owes a
// wake; platform release failure preserves ownership for explicit retry.
iree_status_t loom_serve_text_registration_deactivate(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component, bool* out_complete);

// Begins after preparation terminates and service/tokenizer borrowers retire.
// Releases the ready service, joins image ownership, then joins actual text
// model ownership through the original observer. Nonterminal success owes a
// wake. Platform release failure preserves ownership for retry. NULL is
// retired.
iree_status_t loom_serve_text_registration_retire(
    loom_serve_text_registration_t* registration, bool* out_complete);

// Frees a successfully retired registration. NULL is accepted. Borrowed
// definition/resource/observer storage can be released after this call.
void loom_serve_text_registration_free(
    loom_serve_text_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_
