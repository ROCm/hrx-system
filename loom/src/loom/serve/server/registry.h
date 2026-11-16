// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REGISTRY_H_
#define LOOM_SERVE_SERVER_REGISTRY_H_

#include "loom/runtime/parameter/activation.h"
#include "loom/runtime/parameter/residency.h"
#include "loom/runtime/program/compilation.h"
#include "loom/runtime/wake.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_resolution.h"
#include "loom/serve/service/service.h"
#include "loom/service/registration.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_registry_t loom_serve_registry_t;

// Optional owner-side observation after service progress and before model
// retirement. It copies terminal observations without retaining model storage.
typedef struct loom_serve_registry_observer_t {
  // Borrowed adapter state for the synchronous service traversal.
  void* self;
  // Reports actual progress. Failure is an application failure.
  iree_status_t (*observe)(void* self, iree_host_size_t index, bool stopping,
                           bool* out_progress);
} loom_serve_registry_observer_t;

// HTTP publication of an independently owned native lifecycle. Native callers
// use the same owner without constructing a registry or HTTP dispatch loop.
typedef struct loom_serve_registration_t {
  // Owned native provider, retained through resource and observation
  // retirement.
  loom_service_registration_t* native;
  // Borrows the provider's typed HTTP facade after native preparation succeeds.
  loom_serve_service_t (*service)(loom_service_registration_t* registration);
} loom_serve_registration_t;

typedef struct loom_serve_registration_factory_t {
  // Borrowed application capabilities, live through registry destruction.
  void* self;
  // Supported configuration kinds as bit positions; zero disables management.
  uint32_t kinds;
  // Borrowed immutable group keys, live through registry destruction.
  iree_string_view_list_t device_names;
  // Number of immutable storage owners, independent of device count.
  iree_host_size_t pool_count;
  // Borrowed pool keys and qualified consumer membership through destruction.
  const loom_runtime_pool_binding_t* pools;
  // Shared cold-work sink for resolving local model cards before create.
  // An empty sink supports direct declarations only.
  loomc_task_sink_t resolution_sink;
  // Allocates an unstarted operation; performs no source I/O, model allocation
  // or submission. Failure returns no operation. Invoked after admission opens
  // or explicit activation, never for metadata-only registration alone.
  // Incarnation identifies this control binding, independently of route
  // identity. Replacing the cold control invalidates earlier decisions.
  iree_status_t (*create)(void* self,
                          const loom_serve_model_configuration_t* configuration,
                          uint64_t incarnation,
                          loom_runtime_wake_callback_t observer,
                          loom_runtime_telemetry_stream_t* stream,
                          iree_allocator_t host_allocator,
                          loom_serve_registration_t* out_registration);
  // Optional shared-resource failure query after a model operation fails.
  // Returns owned terminal failure if shared backing was disturbed, otherwise
  // OK so the registration retires independently. Never guesses from codes,
  // clears failure, waits or releases ownership. The factory owns this evidence
  // independently of any model that may have failed before service publication.
  iree_status_t (*query_status)(void* self);
} loom_serve_registration_factory_t;

typedef enum loom_serve_removal_e {
  LOOM_SERVE_REMOVAL_NONE = 0,
  LOOM_SERVE_REMOVAL_DRAIN,
  LOOM_SERVE_REMOVAL_ABORT,
} loom_serve_removal_t;

// Stable owner-only slot. Empty service.name marks an unused slot. A slot is
// reused only after every held claim, tokenizer task and model borrower
// retires. A ready service facade stays borrowed through its external users,
// including native failure. It is detached before native retirement; afterward
// the slot exposes immutable route metadata and copied transport observations.
// The opaque capacity owner survives native retirement.
typedef struct loom_serve_registered_model_t {
  // Owning stable table, including while this slot is vacant or retiring.
  loom_serve_registry_t* registry;
  // Borrowed ready interface or metadata-only description for an owned model.
  loom_serve_service_t service;
  // Owned declaration for managed entries; empty name denotes borrowed service.
  loom_serve_model_configuration_t configuration;
  // Owned native provider. It retains diagnostics after actual retirement.
  loom_serve_registration_t registration;
  // Cold declaration ownership before transfer to the native provider.
  struct {
    // Retained initial control, allocated with the registry and reused only
    // after all route and policy readers join.
    loom_runtime_service_control_t* control;
    // Legal card-resolution or provider-attachment choice.
    loom_runtime_service_offer_t offer;
    // Accepted cold task, independent of whether its choice remains legal.
    loom_runtime_service_operation_t operation;
    // Accepted card reader; joined before replacing configuration or control.
    loom_serve_model_resolution_t* resolution;
    // Owned construction diagnostic; native failures live in registration.
    iree_status_t status;
    // Deferred warmup transferred exactly once when the provider is attached.
    loom_service_parameter_operation_t parameter_operation;
    // Component selected by the deferred warmup.
    loom_service_model_component_t parameter_component;
    // Cold retention intent transferred before native preparation can start.
    uint32_t pinned_components;
    // Start of accepted card/attachment work in host monotonic nanoseconds.
    iree_time_t start_ns;
    // Completed cold intervals, excluding idle declaration time.
    uint64_t duration_ns;
    // Nonrecycled accepted identities within this cold control incarnation.
    uint64_t next_serial;
  } attachment;
  // Irreversible unregister intent; drain can be escalated explicitly to abort.
  loom_serve_removal_t removal;
  // Reversible admission gate, not a residency or cancellation flag.
  bool closed;
  // Requests parked outside service admission on transport-owned storage.
  iree_host_size_t held_count;
  // Held inference requests, excluding tokenizer and retained-state utilities.
  iree_host_size_t parameter_demand_count;
  // Independent tokenizer processes still borrowing the model.
  iree_host_size_t tokenizer_count;
  // Copied scheduler observation for one response; never a readiness signal.
  loom_service_snapshot_t snapshot;
  // Final work totals after service destruction, or zero before publication.
  loom_service_telemetry_counters_t counters;
  // Final duration distributions after service destruction, or zero before
  // publication. Separate from counters copied into compact event records.
  loom_service_telemetry_latency_t latency;
  // Owned history shared with the live service; identity survives preparation.
  loom_runtime_telemetry_stream_t* stream;
  // Monotonic registration identity; zero denotes an externally owned service.
  uint64_t incarnation;
  // Wall-clock Unix seconds at registration, fixed until deployment retirement.
  int64_t registered_unix_seconds;
} loom_serve_registered_model_t;

// Copies borrowed service interfaces into stable slots and reserves metadata
// capacity for managed registrations. Borrowed services remain externally owned
// and cannot be unregistered here. No source work or model allocation occurs.
iree_status_t loom_serve_registry_allocate(
    iree_host_size_t service_count, const loom_serve_service_t* services,
    iree_host_size_t managed_capacity,
    loom_serve_registration_factory_t factory, iree_allocator_t host_allocator,
    loom_serve_registry_t** out_registry);

// Requires the router to have retired. A failed model release preserves the
// entire registry and its borrowed factory/device owners; it is not permission
// to destroy their queues or pools. Successful destruction consumes registry.
iree_status_t loom_serve_registry_free(loom_serve_registry_t* registry);

iree_host_size_t loom_serve_registry_capacity(
    const loom_serve_registry_t* registry);
iree_host_size_t loom_serve_registry_count(
    const loom_serve_registry_t* registry);
// Monotonic epoch covering aggregate work, including retired registrations and
// borrowed services' earlier history. Fixed at construction: the earliest of
// registry creation and borrowed stream creation. Zero if any borrowed service
// lacks a counter epoch. This is not the start of an individual model's work.
iree_time_t loom_serve_registry_counters_since_ns(
    const loom_serve_registry_t* registry);
loom_serve_registered_model_t* loom_serve_registry_at(
    loom_serve_registry_t* registry, iree_host_size_t index);

// Borrows the compact runtime progression slots. Each occupied capability
// composes model service progress with this registry's preparation, parameter
// control, observation, and retirement; vacant slots have no vtable.
loom_runtime_service_t* loom_serve_registry_services(
    loom_serve_registry_t* registry);

// Supplies owner context for the next synchronous service traversal. The wake
// remains valid through accepted preparation/retirement; the observation is
// borrowed only during that traversal and never runs from asynchronous work.
void loom_serve_registry_set_advance_observer(
    loom_serve_registry_t* registry, loom_runtime_wake_callback_t wake,
    loom_serve_registry_observer_t observer);

// Snapshots service state only after publication and before retirement.
// Other states report their registration phase without calling a service.
// The registry owner serializes this query with model advancement/removal.
loom_service_snapshot_t loom_serve_registry_snapshot(
    const loom_serve_registered_model_t* model);

// Moves a validated declaration, including its owning allocator, into a
// stable slot on success, zeroing the caller's value. Failure changes neither.
// Resolves device binding before moving the declaration. Rejects unsupported
// kinds, ambiguous/missing devices, duplicate names and exhausted credit.
iree_status_t loom_serve_registry_register(
    loom_serve_registry_t* registry,
    loom_serve_model_configuration_t* configuration,
    iree_host_size_t* out_index);

// Resolves the current named speech registration and tries native admission
// on the serialized serving owner. Success with a stream moves the initialized
// request; NULL success is preparation/admission/queue pressure and leaves it
// with the caller. Waiting HTTP admissions retain priority. Failure leaves the
// request untouched. An accepted stream never rebinds to a later registration;
// its copied output/status can outlive model removal and registry destruction.
iree_status_t loom_serve_registry_try_open_speech(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_speech_request_t* request,
    loom_serve_speech_stream_t** out_stream);

// Uses the same speech admission and ownership policy for one-time uploads.
// The receipt publishes success only after the named voice is ready to use.
iree_status_t loom_serve_registry_try_upload_voice(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_speech_voice_request_t* request,
    loom_serve_speech_voice_upload_t** out_upload);

// Removes a published voice without activating a cold or drained model. Already
// accepted speech retains its captured voice through physical retirement.
iree_status_t loom_serve_registry_remove_voice(loom_serve_registry_t* registry,
                                               iree_string_view_t name,
                                               iree_string_view_t id);

// Resolves the current recognition registration under the same admission
// policy. A stream copies request options and accepts continuing normalized PCM
// with bounded frame credit. The caller owns its result until close; graceful
// removal waits for input completion/cancellation and physical retirement,
// while finalized documents can outlive the registration and registry.
iree_status_t loom_serve_registry_try_open_recognition(
    loom_serve_registry_t* registry, iree_string_view_t name,
    const loom_serve_recognition_request_t* request,
    loom_serve_recognition_stream_t** out_stream);

// Resolves the current text registration under the same admission policy.
// Success with a stream moves the owning JSON request and copies its selectors;
// pressure/failure leaves it untouched. Accepted requests use ordinary source
// policy, retained history, and packed execution. Ordered spans and validated
// final results belong to the caller until close, independently of removal.
iree_status_t loom_serve_registry_try_open_text(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_text_request_t* request, loom_serve_text_stream_t** out_stream);

// Resolves the current text registration and enqueues PIN, SUSPEND, or RELEASE
// against that incarnation's retained history. PIN/SUSPEND obey inference
// admission. RELEASE bypasses drain and held inference but uses the same
// bounded control slot; an unprepared model has no retained state and returns
// NOT_FOUND without preparing it. Success with a handle copies identifiers;
// NULL means preparation/queue pressure. Completion/status survive removal and
// same-name replacement. Named pins belong to their original registration,
// independently of control handles.
iree_status_t loom_serve_registry_try_open_text_checkpoint(
    loom_serve_registry_t* registry, iree_string_view_t name,
    const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control);

// Seals admission; callers resolve held claims before the next advance.
// Repeated removal is idempotent, with abort escalation and explicit
// release-failure retry.
iree_status_t loom_serve_registry_remove(loom_serve_registered_model_t* model,
                                         loom_serve_removal_t mode);

// Requests selected-component release while preserving primary mutable state
// and retained sessions/checkpoints. Requires closed admission and an unpinned
// component whose adapter supports release; a metadata-only declaration is
// already cold. Elastic primary backing releases weights without re-JIT;
// optional fixed-backed components may instead retire and prepare on demand.
// Repeated requests for the same component are idempotent. Advance completes
// after admitted work, native release, and component ownership retire, without
// blocking the owner or unrelated services.
iree_status_t loom_serve_registry_deactivate(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component);

// Warms a healthy managed registration while leaving admission closed. Declared
// models first prepare; ready models finish admitted work before activation.
// Primary activation does not prepare optional components. Image activation
// prepares that component lazily and retains source failures locally.
// Duplicates for the same component are idempotent. Other residency operations
// conflict; removal cancels intent but joins accepted work before retirement.
iree_status_t loom_serve_registry_activate(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component);

// Changes idempotent parameter-eviction policy on a healthy managed model,
// including metadata-only declarations. Does not load, allocate, alter
// admission or cancel accepted work. Pinning conflicts with deactivation;
// unregister discards policy only with the model's normal resource retirement.
iree_status_t loom_serve_registry_set_pinned(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component, bool pinned);

// Cold declaration queries. No source access is needed to name these roles.
const char* loom_serve_model_component_name(
    loom_service_model_component_t component);
bool loom_serve_registry_has_component(
    const loom_serve_registered_model_t* model,
    loom_service_model_component_t component);

// Returns STOPPING for failure, abort, or completed graceful work.
// The router still owns held claims and external tokenizer lifetimes.
loom_runtime_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model);

// Consumes a terminal operation error and seals this managed model's admission.
// Accepted service, tokenizer, and native borrowers retire through advance;
// the failed slot and diagnostic remain inspectable until explicit removal.
// Returns only factory-owned shared failure. Borrowed services have no managed
// retirement owner, so their error is returned to their external owner instead.
iree_status_t loom_serve_registry_fail(loom_serve_registry_t* registry,
                                       iree_host_size_t index,
                                       iree_status_t status);

// Progresses accepted cold/native work and observes the external-view handoff.
// The common service wrapper invokes transport observation before relinquishing
// a ready facade. New work requires explicit scheduling grants. Model-local
// errors stay in their owner; only factory-owned shared failure escapes.
iree_status_t loom_serve_registry_advance(loom_serve_registry_t* registry,
                                          iree_host_size_t index,
                                          loom_runtime_wake_callback_t observer,
                                          bool* out_progress);
// Copies native lifecycle facts, or the cold declaration before attachment.
// Status and report remain borrowed through the next owner mutation.
loom_service_registration_snapshot_t loom_serve_registry_state(
    const loom_serve_registered_model_t* model);

// Current policy binding, replaced only at a serialized cold handoff.
const loom_runtime_service_control_t* loom_serve_registry_control(
    const loom_serve_registered_model_t* model);

bool loom_serve_registry_pending(const loom_serve_registered_model_t* model);
const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model);

// Cumulative preparation elapsed time, including its open interval. Zero
// before preparation; declared idle gaps and later parameter activation add
// none. Called only by the serialized owner; observation never advances
// preparation.
uint64_t loom_serve_registry_preparation_duration_ns(
    const loom_serve_registered_model_t* model);

// Totals include removed incarnations. False means a borrowed service omitted
// counters; unavailable totals are never presented as a complete partial sum.
bool loom_serve_registry_counters(
    const loom_serve_registry_t* registry,
    loom_service_telemetry_counters_t* out_counters);

// Latency totals include removed incarnations. False means a borrowed
// service omitted latency observations; the partial result must not be exposed
// as a complete server distribution.
bool loom_serve_registry_latency(const loom_serve_registry_t* registry,
                                 loom_service_telemetry_latency_t* out_latency);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_REGISTRY_H_
