// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_REGISTRATION_H_
#define LOOM_SERVICE_REGISTRATION_H_

#include "loom/runtime/parameter/activation.h"
#include "loom/runtime/parameter/residency.h"
#include "loom/runtime/program/compilation.h"
#include "loom/runtime/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native preparation, ready execution, and actual resource retirement share
// one serialized owner. Providers embed this owner beside their typed model
// state; applications borrow it through their final observation and cleanup.
typedef struct loom_service_registration_t loom_service_registration_t;

// Independently managed roles in a model declaration. IMAGE is the optional
// text-input projector, not an image-generation service's primary computation.
typedef enum loom_service_model_component_e {
  LOOM_SERVICE_MODEL_COMPONENT_PRIMARY = 0,
  LOOM_SERVICE_MODEL_COMPONENT_IMAGE,
  LOOM_SERVICE_MODEL_COMPONENT_COUNT,
} loom_service_model_component_t;

typedef enum loom_service_registration_phase_e {
  LOOM_SERVICE_REGISTRATION_DECLARED = 0,
  LOOM_SERVICE_REGISTRATION_PREPARING,
  LOOM_SERVICE_REGISTRATION_READY,
  LOOM_SERVICE_REGISTRATION_FAILED,
  LOOM_SERVICE_REGISTRATION_RETIRING,
  LOOM_SERVICE_REGISTRATION_RETIRED,
  LOOM_SERVICE_REGISTRATION_RELEASE_FAILED,
} loom_service_registration_phase_t;

typedef enum loom_service_parameter_operation_e {
  LOOM_SERVICE_PARAMETER_OPERATION_NONE = 0,
  LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE,
  LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE,
} loom_service_parameter_operation_t;

// Owner-only component facts. Views remain borrowed until the next mutation;
// observing them never prepares, activates, or releases resources.
typedef struct loom_service_registration_parameters_t {
  // Static preparation/lifetime phase, independent of weight activation.
  const char* phase;
  // A residency exists and parameters below describe it.
  bool prepared;
  // Quiescent release is supported, including optional fixed model retirement.
  bool deactivatable;
  // Copied parameter accounting, meaningful when prepared is true.
  loom_runtime_residency_snapshot_t parameters;
  // Borrowed component-local failure, retained by its native owner.
  iree_status_t status;
  // Borrowed compilation evidence, or NULL without a separate component load.
  const loom_runtime_compilation_report_t* compilation;
  // Time spent preparing this component, in host monotonic nanoseconds.
  uint64_t preparation_duration_ns;
} loom_service_registration_parameters_t;

enum loom_service_registration_prepare_flag_bits_e {
  // Irreversibly cancels preparation while joining accepted ownership.
  LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL = 1u << 0,
  // Current inference or primary warmup permits early parameter acquisition.
  LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_DEMAND = 1u << 1,
};
typedef uint32_t loom_service_registration_prepare_flags_t;

enum loom_service_preparation_result_flag_bits_e {
  // This preparation attempt has terminated, including its asynchronous users.
  LOOM_SERVICE_PREPARATION_RESULT_COMPLETE = 1u << 0,
  // A failed attempt still owns model resources requiring actual retirement.
  LOOM_SERVICE_PREPARATION_RESULT_RETAINED = 1u << 1,
};
typedef uint32_t loom_service_preparation_result_flags_t;

typedef struct loom_service_preparation_result_t {
  // Completion and retained ownership are independent of the returned status.
  loom_service_preparation_result_flags_t flags;
  // Ready engine only on complete success, borrowed until retirement handoff.
  loom_runtime_service_t service;
} loom_service_preparation_result_t;

// Typed resource mechanics. These callbacks run on the serialized owner and
// never wait. A nonterminal accepted operation owes the owner's next wake.
// The provider knows its own resource retention; a missing ready engine alone
// cannot establish that failed preparation has released native resources.
typedef struct loom_service_registration_vtable_t {
  // Prepares the provider and copies its terminal compilation evidence.
  iree_status_t (*prepare)(void* self,
                           loom_service_registration_prepare_flags_t flags,
                           loom_service_preparation_result_t* out_result,
                           loom_runtime_compilation_report_t* out_report);
  // Consumes accepted ready-engine work without granting another source unit.
  iree_status_t (*progress)(void* self, loom_runtime_service_flags_t flags,
                            bool* out_progress);
  // Releases service consumers, then joins actual model ownership. Failure
  // retains quiescent ownership for explicit retry, without pending callbacks.
  iree_status_t (*retire)(void* self, bool* out_complete);
  // Copies facts for a supported component after successful preparation.
  loom_service_registration_parameters_t (*parameters)(
      void* self, loom_service_model_component_t component);
  // Changes weight retention only; does not acquire resources or submit work.
  void (*set_pinned)(void* self, loom_service_model_component_t component,
                     bool pinned);
  // Advances explicit warmup. Transfers an owned component-local diagnostic
  // separately from parent failure; the component retains its original.
  iree_status_t (*activate)(void* self,
                            loom_service_model_component_t component,
                            loom_runtime_activation_flags_t flags,
                            loom_runtime_activation_phase_t* out_phase,
                            iree_status_t* out_component_status);
  // Releases an idle component's parameters or optional prepared model.
  iree_status_t (*deactivate)(void* self,
                              loom_service_model_component_t component,
                              bool* out_complete);
  // Frees provider storage after successful retirement and observation joins.
  void (*destroy)(void* self);
} loom_service_registration_vtable_t;

typedef struct loom_service_registration_work_t {
  // Current legal choice, withdrawn independently of accepted work.
  loom_runtime_service_offer_t offer;
  // Accepted ownership retained through actual completion or cleanup.
  loom_runtime_service_operation_t operation;
} loom_service_registration_work_t;

// Embedded implementation state. Only registration functions mutate these
// fields. Providers own the surrounding allocation and retained control; source
// contributors publish through registration_binding, not directly to summary.
struct loom_service_registration_t {
  // Borrowed provider implementation, alive through destruction.
  const loom_service_registration_vtable_t* vtable;
  // Borrowed typed provider state containing this owner.
  void* self;
  // Owner wake target, retained until actual native retirement.
  loom_runtime_wake_callback_t observer;
  // Ready engine capability; detached before provider retirement destroys it.
  loom_runtime_service_t engine;
  // Retained observation bytes, independent of model and engine allocations.
  loom_runtime_service_control_t* control;
  // Native lifecycle, independent of route admission or parameter intent.
  loom_service_registration_phase_t phase;
  // Requested or accepted parameter work; admission remains caller-owned.
  loom_service_parameter_operation_t parameter_operation;
  // Component selected by the current parameter operation.
  loom_service_model_component_t parameter_component;
  // Immutable supported component roles as bit positions.
  uint32_t components;
  // Persistent weight retention installed on prepared components.
  uint32_t pinned_components;
  // Irreversible stop intent, without authority to destroy external views.
  bool stopping;
  // The application has relinquished its external engine/tokenizer views.
  bool retirement_requested;
  // Owned failure, retained after the engine and model are gone.
  iree_status_t status;
  // Copied preparation evidence, independent of compiler/source lifetimes.
  loom_runtime_compilation_report_t compilation;
  // Exact resource participation for this model's lifecycle work.
  uint64_t devices;
  // Composition-scoped instance shared by this registration's lifecycle work.
  uint64_t instance_ordinal;
  // Stable device binding; lifecycle work has no source-provided estimate.
  loom_runtime_service_use_t use;
  // Latest source contribution, updated only at producer mutation boundaries.
  loom_runtime_service_observation_t observation;
  // Preparation or parameter operation, serialized through one owner.
  loom_service_registration_work_t lifecycle;
  // Independent retirement: failed activation may still retain protection.
  loom_service_registration_work_t retirement;
  // Nonrecycled accepted identities across both lifecycle operation slots.
  uint64_t next_serial;
};

// Initializes zeroed provider-owned storage. Two consecutive exclusive offers
// and operations begin at binding's ordinals; all source detail precedes the
// lifecycle operations. Control, provider, and observer survive retirement and
// reader joins. This establishes trusted provider coordinates, not user input.
void loom_service_registration_initialize(
    const loom_service_registration_vtable_t* vtable, void* self,
    uint32_t components, uint64_t device_ordinal,
    loom_runtime_service_control_binding_t binding,
    loom_runtime_wake_callback_t observer,
    loom_service_registration_t* out_registration);

// Supplies one source contributor's exclusive detail slots and the native
// aggregate publisher. Multiple source contributors compose before this call.
loom_runtime_service_control_binding_t loom_service_registration_binding(
    loom_service_registration_t* registration, uint64_t offer_ordinal,
    uint64_t operation_ordinal, uint64_t instance_ordinal);

typedef struct loom_service_registration_snapshot_t {
  // Current native lifecycle phase.
  loom_service_registration_phase_t phase;
  // Requested or accepted parameter work, independent of phase.
  loom_service_parameter_operation_t parameter_operation;
  // Ready capability only in READY; external views require explicit handoff.
  loom_runtime_service_t service;
  // Borrowed retained native failure.
  iree_status_t status;
  // Borrowed copied compilation evidence.
  const loom_runtime_compilation_report_t* compilation;
} loom_service_registration_snapshot_t;

loom_service_registration_snapshot_t loom_service_registration_snapshot(
    const loom_service_registration_t* registration);

// Publishes parameter intent without starting work. Existing requests drain
// before its offer becomes legal. NONE withdraws unaccepted intent; accepted
// work must complete before another intent can replace it.
iree_status_t loom_service_registration_request_parameters(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_service_parameter_operation_t operation);

enum loom_service_retire_flag_bits_e {
  // Cancels queued work and joins accepted work instead of draining requests.
  LOOM_SERVICE_RETIRE_FLAG_ABORT = 1u << 0,
};
typedef uint32_t loom_service_retire_flags_t;

// Irreversible external-view handoff after future admission stops. Request
// receipts may remain independently owned. ABORT and native failure clean up
// accepted work with zero grants. Repeating after RELEASE_FAILED explicitly
// retries the retained platform release.
void loom_service_registration_request_retirement(
    loom_service_registration_t* registration,
    loom_service_retire_flags_t flags);

// Consumes accepted work only. Failures remain in the native snapshot; STOPPING
// cancels work but cannot destroy external views without retirement handoff.
bool loom_service_registration_progress(
    loom_service_registration_t* registration,
    loom_runtime_service_flags_t flags);

// Claims one advertised lifecycle or source unit without waiting. Errors stay
// in the owning native scope; acceptance is independent of status and cleanup.
loom_runtime_service_schedule_outcome_t loom_service_registration_try_schedule(
    loom_service_registration_t* registration,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags);

// Explicit preparation grant. Repeat nonterminal calls until completion; the
// snapshot lends the ready engine and report. Failure requires retirement.
iree_status_t loom_service_registration_prepare(
    loom_service_registration_t* registration,
    loom_service_registration_prepare_flags_t flags, bool* out_complete);
// Explicit warmup grant for a supported ready component after request activity
// retires. Repeat through COMPLETE, including cancellation, before retirement.
iree_status_t loom_service_registration_activate(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status);
// Explicit release grant for a supported, quiescent, deactivatable component.
// Nonterminal success owes an owner wake; failure preserves native ownership.
iree_status_t loom_service_registration_deactivate(
    loom_service_registration_t* registration,
    loom_service_model_component_t component, bool* out_complete);
// Retires after preparation and external views join, or discards an unstarted
// provider. Releases actual native ownership; NULL is already retired.
iree_status_t loom_service_registration_retire(
    loom_service_registration_t* registration, bool* out_complete);

// Copies a supported ready component's state without changing residency.
loom_service_registration_parameters_t loom_service_registration_parameters(
    const loom_service_registration_t* registration,
    loom_service_model_component_t component);
// Changes retention on a supported ready component without loading it.
void loom_service_registration_set_pinned(
    loom_service_registration_t* registration,
    loom_service_model_component_t component, bool pinned);

// Frees the provider after successful retirement and all policy readers join.
// NULL is accepted; declaration/options/resource borrows end here.
void loom_service_registration_free(loom_service_registration_t* registration);

const loom_runtime_service_control_t* loom_service_registration_control(
    const loom_service_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_REGISTRATION_H_
