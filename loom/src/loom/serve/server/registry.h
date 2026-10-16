// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REGISTRY_H_
#define LOOM_SERVE_SERVER_REGISTRY_H_

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/runtime/parameter/activation.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/service/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independently controlled roles declared by the model document. Primary is
// the default for unqualified parameter controls. Image denotes a text input
// projector, not the primary computation of an image-generation service.
typedef enum loom_serve_model_component_e {
  LOOM_SERVE_MODEL_COMPONENT_PRIMARY = 0,
  LOOM_SERVE_MODEL_COMPONENT_IMAGE,
  LOOM_SERVE_MODEL_COMPONENT_COUNT,
} loom_serve_model_component_t;

// Owner-only observation of one component. Snapshots copy accounting; strings,
// diagnostics, and compilation evidence remain borrowed through this owner
// turn. Observation never prepares or activates a component.
typedef struct loom_serve_registration_parameters_t {
  // Static preparation/lifetime phase name, separate from weight activation.
  const char* phase;
  // A residency exists and parameters below describes it.
  bool prepared;
  // Quiescent release is supported, possibly by retiring an optional model.
  bool deactivatable;
  // Copied parameter-only accounting, meaningful only when prepared is true.
  loom_serve_residency_snapshot_t parameters;
  // Borrowed preparation failure retained by the component; never consumed.
  iree_status_t status;
  // Borrowed cumulative compilation evidence, or NULL for no separate report.
  const loom_serve_compilation_report_t* compilation;
} loom_serve_registration_parameters_t;

// A modality's owned preparation/retirement operation. All methods run on the
// serialized serving owner and never wait. Wake callbacks only schedule that
// owner. The declaration and factory remain borrowed through destruction.
typedef struct loom_serve_registration_t {
  // Owned modality operation, independent of any management connection.
  void* self;
  // Advances preparation or irrevocable cancellation. A nonterminal result
  // owes a wake after accepted work retires. Only terminal success publishes
  // a service using the supplied stream, cumulative counters and terminal
  // latency distributions; terminal failure still requires retire.
  // Copies compilation evidence on terminal completion, including failure.
  iree_status_t (*prepare)(void* self, bool cancel, bool* out_complete,
                           loom_serve_service_t* out_service,
                           loom_serve_compilation_report_t* out_report);
  // Called after preparation terminates and service/tokenizer borrowers retire.
  // Drops service consumers, then waits by ownership wake, not readiness value.
  // Terminal success releases all model resources. Terminal failure retains
  // quiescent ownership for an explicit retry; no callback remains on failure.
  iree_status_t (*retire)(void* self, bool* out_complete);
  // Optional copied parameter state, queried only after successful prepare and
  // before retire. The adapter owns its actual parameter groups; neither the
  // registry nor the HTTP client gains a borrowed model or pool handle.
  loom_serve_registration_parameters_t (*parameters)(
      void* self, loom_serve_model_component_t component);
  // Optional weight-only eviction policy. Applied on successful preparation
  // and ready policy changes, without allocation, submission or waiting.
  // Unpinning preserves operation-owned protection and all mutable state.
  void (*set_pinned)(void* self, loom_serve_model_component_t component,
                     bool pinned);
  // Advances explicit warmup after admitted service activity has retired.
  // Capacity waiting owns no device work; submitted readiness owes an owner
  // wake. Cancellation withdraws admission or joins accepted work. Terminal
  // native failure preserves resources until retirement and terminates shared
  // serving. A source preparation failure instead transfers a diagnostic to
  // out_component_status and completes without failing the parent service;
  // the component retains its own diagnostic for observation. The registry
  // checks the factory's shared failure channel before keeping service open.
  // Paired with parameters; no model handle escapes the adapter.
  iree_status_t (*activate)(void* self, loom_serve_model_component_t component,
                            loom_serve_activation_flags_t flags,
                            loom_serve_activation_phase_t* out_phase,
                            iree_status_t* out_component_status);
  // Releases reloadable parameter storage after admitted service activity has
  // retired. Preserves primary mutable state. Optional models may retire and
  // recreate fixed backing. Nonterminal success owes an owner wake after native
  // release or ownership retirement; repeated calls join that operation.
  // Called only when its snapshot permits deactivation. No wait.
  // Failure terminates shared serving through normal owner draining.
  iree_status_t (*deactivate)(void* self,
                              loom_serve_model_component_t component,
                              bool* out_complete);
  // Frees the operation after successful retirement; no model resources remain.
  void (*destroy)(void* self);
} loom_serve_registration_t;

typedef struct loom_serve_registration_factory_t {
  // Borrowed application capabilities, live through registry destruction.
  void* self;
  // Supported configuration kinds as bit positions; zero disables management.
  uint32_t kinds;
  // Immutable group cardinality used to resolve deployment device bindings.
  iree_host_size_t device_count;
  // Allocates an unstarted operation; performs no source I/O, model allocation
  // or submission. Failure returns no operation. Invoked after admission opens
  // or explicit activation, never for metadata-only registration alone.
  iree_status_t (*create)(void* self,
                          const loom_serve_model_configuration_t* configuration,
                          loom_serve_wake_callback_t observer,
                          loom_serve_telemetry_stream_t* stream,
                          iree_allocator_t host_allocator,
                          loom_serve_registration_t* out_registration);
  // Optional shared-resource failure query after preparation fails. Returns
  // owned terminal failure if preparation disturbed shared backing, otherwise
  // OK so source/package errors remain local. Never guesses from status codes,
  // clears failure, waits or releases ownership. The factory owns this evidence
  // independently of any model that may have failed before service publication.
  iree_status_t (*query_status)(void* self);
} loom_serve_registration_factory_t;

typedef enum loom_serve_registration_phase_e {
  LOOM_SERVE_REGISTRATION_DECLARED = 0,
  LOOM_SERVE_REGISTRATION_PREPARING,
  LOOM_SERVE_REGISTRATION_READY,
  LOOM_SERVE_REGISTRATION_RETIRING,
  LOOM_SERVE_REGISTRATION_FAILED,
  LOOM_SERVE_REGISTRATION_RELEASE_FAILED,
} loom_serve_registration_phase_t;

typedef enum loom_serve_removal_e {
  LOOM_SERVE_REMOVAL_NONE = 0,
  LOOM_SERVE_REMOVAL_DRAIN,
  LOOM_SERVE_REMOVAL_ABORT,
} loom_serve_removal_t;

typedef enum loom_serve_parameter_operation_e {
  LOOM_SERVE_PARAMETER_OPERATION_NONE = 0,
  // Explicit warmup waiting for preparation, admitted work or physical
  // capacity.
  LOOM_SERVE_PARAMETER_OPERATION_WAITING,
  // Warmup has submitted its asynchronous parameter-readiness wait.
  LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING,
  // Release waiting for admitted model work to retire.
  LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING,
} loom_serve_parameter_operation_t;

// Stable owner-only slot. Empty service.name marks an unused slot. A slot is
// reused only after every held claim, tokenizer task and model borrower
// retires. Service callbacks exist only while phase is READY. Other phases
// expose just immutable route/capacity-domain metadata and copied counters,
// never model pointers. The opaque capacity owner survives native retirement.
typedef struct loom_serve_registered_model_t {
  // Borrowed ready interface or metadata-only description for an owned model.
  loom_serve_service_t service;
  // Owned declaration for managed entries; empty name denotes borrowed service.
  loom_serve_model_configuration_t configuration;
  // Owned modality operation, created lazily by admission or explicit
  // activation.
  loom_serve_registration_t registration;
  // Preparation/publication/retirement state, independent of admission.
  loom_serve_registration_phase_t phase;
  // Irreversible unregister intent; drain can be escalated explicitly to abort.
  loom_serve_removal_t removal;
  // Reversible admission gate, not a residency or cancellation flag.
  bool closed;
  // Weight-only retention bits indexed by loom_serve_model_component_t.
  // Applied after preparation without activating parameters, retaining
  // sessions, or promising admission progress.
  uint8_t pinned_components;
  // loom_serve_model_component_t selected by the current or last operation.
  uint8_t parameter_component;
  // Mutually exclusive residency intent, independent of its HTTP connection.
  // Removal withdraws waiting admission and joins submitted activation first.
  loom_serve_parameter_operation_t parameter_operation;
  // Requests parked outside service admission on transport-owned storage.
  iree_host_size_t held_count;
  // Independent tokenizer processes still borrowing the model.
  iree_host_size_t tokenizer_count;
  // Copied scheduler observation for one response; never a readiness signal.
  loom_serve_service_snapshot_t snapshot;
  // Final work totals after service destruction, or zero before publication.
  loom_serve_telemetry_counters_t counters;
  // Final duration distributions after service destruction, or zero before
  // publication. Separate from counters copied into compact event records.
  loom_serve_telemetry_latency_t latency;
  // Owned history shared with the live service; identity survives preparation.
  loom_serve_telemetry_stream_t* stream;
  // Terminal load/release evidence, owned until removal or explicit retry.
  iree_status_t status;
  // Copied source diagnostics, independent of compiler and operation lifetimes.
  loom_serve_compilation_report_t compilation;
  // Monotonic registration identity; zero denotes an externally owned service.
  uint64_t incarnation;
  // Wall-clock Unix seconds at registration, fixed until deployment retirement.
  int64_t registered_unix_seconds;
} loom_serve_registered_model_t;

typedef struct loom_serve_registry_t loom_serve_registry_t;

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
loom_serve_registered_model_t* loom_serve_registry_at(
    loom_serve_registry_t* registry, iree_host_size_t index);

// Snapshots service state only after publication and before retirement.
// Other states report their registration phase without calling a service.
// The registry owner serializes this query with model advancement/removal.
loom_serve_service_snapshot_t loom_serve_registry_snapshot(
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
    loom_serve_model_component_t component);

// Warms a healthy managed registration while leaving admission closed. Declared
// models first prepare; ready models finish admitted work before activation.
// Primary activation does not prepare optional components. Image activation
// prepares that component lazily and retains source failures locally.
// Duplicates for the same component are idempotent. Other residency operations
// conflict; removal cancels intent but joins accepted work before retirement.
iree_status_t loom_serve_registry_activate(
    loom_serve_registered_model_t* model,
    loom_serve_model_component_t component);

// Changes idempotent parameter-eviction policy on a healthy managed model,
// including metadata-only declarations. Does not load, allocate, alter
// admission or cancel accepted work. Pinning conflicts with deactivation;
// unregister discards policy only with the model's normal resource retirement.
iree_status_t loom_serve_registry_set_pinned(
    loom_serve_registered_model_t* model,
    loom_serve_model_component_t component, bool pinned);

// Cold declaration queries. No source access is needed to name these roles.
const char* loom_serve_model_component_name(
    loom_serve_model_component_t component);
bool loom_serve_registry_has_component(
    const loom_serve_registered_model_t* model,
    loom_serve_model_component_t component);

// Returns STOPPING only for abort or after graceful admitted work has finished.
// The router still owns held claims and external tokenizer lifetimes.
loom_serve_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model);

// Advances one lifecycle after the router advances its ready service. Package
// errors remain in the slot unless the factory reports shared-resource failure.
// Accepted native parameter-operation errors terminate the serving owner, which
// must stop admission and drain every accepted borrower before releasing
// models. Reports actual progress; otherwise waits for an installed ownership
// wake.
iree_status_t loom_serve_registry_advance(loom_serve_registry_t* registry,
                                          iree_host_size_t index,
                                          loom_serve_wake_callback_t observer,
                                          bool* out_progress);
bool loom_serve_registry_pending(const loom_serve_registered_model_t* model);
const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model);

// Totals include removed incarnations. False means a borrowed service omitted
// counters; unavailable totals are never presented as a complete partial sum.
bool loom_serve_registry_counters(
    const loom_serve_registry_t* registry,
    loom_serve_telemetry_counters_t* out_counters);

// Latency totals include removed incarnations. False means a borrowed
// service omitted latency observations; the partial result must not be exposed
// as a complete server distribution.
bool loom_serve_registry_latency(const loom_serve_registry_t* registry,
                                 loom_serve_telemetry_latency_t* out_latency);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_REGISTRY_H_
