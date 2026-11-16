// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/registration.h"

#include "loom/service/telemetry/record.h"
#include "loom/service/telemetry/tracing.h"

iree_status_t loom_service_registration_allocate_storage(
    iree_host_size_t header_length, iree_host_size_t offer_capacity,
    iree_host_size_t operation_capacity, uint64_t incarnation,
    iree_allocator_t host_allocator, void** out_registration,
    loom_runtime_service_control_t** out_control) {
  *out_registration = NULL;
  *out_control = NULL;
  iree_host_size_t capacity_length = 0, operation_uses_offset = 0,
                   offers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &capacity_length,
      IREE_STRUCT_FIELD(operation_capacity, iree_host_size_t,
                        &operation_uses_offset),
      IREE_STRUCT_FIELD(offer_capacity, loom_runtime_service_offer_capacity_t,
                        &offers_offset)));
  uint8_t* capacity_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, capacity_length,
                                             (void**)&capacity_storage));
  iree_host_size_t* operation_uses =
      (iree_host_size_t*)(capacity_storage + operation_uses_offset);
  for (iree_host_size_t i = 0; i < operation_capacity; ++i) {
    operation_uses[i] = 1;
  }
  const iree_host_size_t use_capacity = 1;
  loom_runtime_service_offer_capacity_t* offers =
      (loom_runtime_service_offer_capacity_t*)(capacity_storage +
                                               offers_offset);
  for (iree_host_size_t i = 0; i < offer_capacity; ++i) {
    offers[i] = (loom_runtime_service_offer_capacity_t){1, &use_capacity};
  }
  const loom_runtime_service_control_capacity_t capacity = {
      .offer_capacity = offer_capacity,
      .offers = offers,
      .operation_capacity = operation_capacity,
      .operation_use_capacities = operation_uses};
  iree_host_size_t control_length = 0;
  iree_status_t status =
      loom_runtime_service_control_size(&capacity, &control_length);
  iree_host_size_t allocation_length = 0, control_offset = 0;
  if (iree_status_is_ok(status)) {
    status =
        IREE_STRUCT_LAYOUT(header_length, &allocation_length,
                           IREE_STRUCT_FIELD_ALIGNED(control_length, uint8_t,
                                                     64, &control_offset));
  }
  void* registration = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_aligned(host_allocator, allocation_length,
                                           64, 0, (void**)&registration);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_service_control_initialize(
        &capacity, incarnation,
        iree_make_byte_span((uint8_t*)registration + control_offset,
                            control_length),
        out_control);
  }
  iree_allocator_free(host_allocator, capacity_storage);
  if (iree_status_is_ok(status)) {
    *out_registration = registration;
  } else {
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

static bool loom_service_registration_refresh_offers(
    loom_service_registration_t* registration);

static void loom_service_registration_set_parameter_intent(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_service_parameter_operation_t operation);

static void loom_service_registration_set_phase(
    loom_service_registration_t* registration,
    loom_service_registration_phase_t phase) {
  if (registration->phase == phase) {
    return;
  }
  if (registration->phase == LOOM_SERVICE_REGISTRATION_PREPARING) {
    registration->preparation_duration_ns =
        iree_time_now() - registration->preparation_start;
  } else if (phase == LOOM_SERVICE_REGISTRATION_PREPARING) {
    registration->preparation_start = iree_time_now();
  }
  IREE_TRACE({
    IREE_TRACE_SECTION_LEAVE(registration->trace.lifecycle);
    registration->trace.lifecycle = 0;
    if (phase == LOOM_SERVICE_REGISTRATION_PREPARING) {
      registration->trace.lifecycle = loom_service_trace_section_enter(
          LOOM_SERVICE_TRACE_SECTION_PREPARE, registration->name);
    } else if (phase == LOOM_SERVICE_REGISTRATION_RETIRING) {
      registration->trace.lifecycle = loom_service_trace_section_enter(
          LOOM_SERVICE_TRACE_SECTION_RETIRE, registration->name);
    }
  });
  registration->phase = phase;
}

static bool loom_service_registration_has_parameter_work(
    const loom_service_registration_t* registration) {
  return registration->lifecycle.operation.key.serial &&
         registration->lifecycle.operation.work !=
             LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
}

static loom_service_registration_prepare_flags_t
loom_service_registration_prepare_flags(
    const loom_service_registration_t* registration) {
  if (registration->stopping || registration->retirement_requested) {
    return LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL;
  }
  const bool inference = iree_any_bit_set(
      registration->demand, LOOM_SERVICE_REGISTRATION_DEMAND_PARAMETERS);
  const bool warmup =
      registration->parameter_operation ==
          LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE &&
      registration->parameter_component == LOOM_SERVICE_MODEL_COMPONENT_PRIMARY;
  return inference || warmup ? LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_DEMAND
                             : 0;
}

static void loom_service_registration_publish(
    loom_service_registration_t* registration) {
  loom_runtime_service_observation_t observation = registration->observation;
  const loom_service_registration_work_t* lifecycle = &registration->lifecycle;
  const loom_service_registration_work_t* retirement =
      &registration->retirement;
  observation.offer_count +=
      lifecycle->offer.choice_count + retirement->offer.choice_count;
  observation.pending_operation_count +=
      (lifecycle->operation.key.serial != 0) +
      (retirement->operation.key.serial != 0);
  if (lifecycle->offer.choice_count || retirement->offer.choice_count) {
    observation.offered_devices |= registration->devices;
  }
  observation.accepted_devices |=
      lifecycle->operation.devices | retirement->operation.devices;
  loom_runtime_service_publish_observation(registration->control, &observation);
}

static void loom_service_registration_publish_engine(
    void* self, const loom_runtime_service_observation_t* observation) {
  loom_service_registration_t* registration = self;
  registration->observation = *observation;
  // Engine activity affects lifecycle eligibility only while the application
  // wants parameter work or retirement. Lifecycle mutations refresh themselves.
  if (registration->parameter_operation !=
          LOOM_SERVICE_PARAMETER_OPERATION_NONE ||
      registration->retirement_requested) {
    loom_service_registration_refresh_offers(registration);
  }
  loom_service_registration_publish(registration);
}

static void loom_service_registration_publish_offer(
    loom_service_registration_t* registration,
    loom_service_registration_work_t* work) {
  const loom_runtime_service_choice_publication_t choice = {
      .choice = {.instance_ordinal = registration->instance_ordinal,
                 .devices = registration->devices,
                 .use_count = 1},
      .uses = &registration->use};
  loom_runtime_service_publish_offer(registration->control, &work->offer,
                                     &choice);
}

static void loom_service_registration_withdraw_offer(
    loom_service_registration_t* registration,
    loom_service_registration_work_t* work) {
  if (work->offer.choice_count) {
    IREE_ASSERT(work->offer.key.revision != UINT64_MAX);
    ++work->offer.key.revision;
    work->offer.choice_count = 0;
    loom_service_registration_publish_offer(registration, work);
  }
}

static void loom_service_registration_begin_work(
    loom_service_registration_t* registration,
    loom_service_registration_work_t* work,
    loom_runtime_service_work_flags_t kind) {
  if (work->operation.key.serial) {
    // Repeated explicit calls progress the same accepted unit. Its public
    // ownership contract excludes a different operation until it retires.
    IREE_ASSERT(work->operation.work == kind);
    return;
  }
  loom_runtime_service_offer_key_t origin = {.ordinal =
                                                 work->offer.key.ordinal};
  if (work->offer.choice_count && work->offer.work == kind) {
    origin = work->offer.key;
  }
  IREE_ASSERT(registration->next_serial != UINT64_MAX);
  work->operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = work->operation.key.ordinal,
              .serial = ++registration->next_serial},
      .origin = origin,
      .work = kind,
      .devices = registration->devices,
      .use_count = 1};
  loom_service_registration_withdraw_offer(registration, work);
  loom_runtime_service_publish_operation(registration->control,
                                         &work->operation, &registration->use);
  loom_service_registration_publish(registration);
}

static void loom_service_registration_end_work(
    loom_service_registration_t* registration,
    loom_service_registration_work_t* work) {
  if (!work->operation.key.serial) {
    return;
  }
  work->operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = work->operation.key.ordinal}};
  loom_runtime_service_publish_operation(registration->control,
                                         &work->operation, NULL);
  loom_service_registration_refresh_offers(registration);
  loom_service_registration_publish(registration);
}

static bool loom_service_registration_update_offer(
    loom_service_registration_t* registration,
    loom_service_registration_work_t* work,
    loom_runtime_service_work_flags_t kind) {
  const uint64_t choice_count = kind != 0;
  if (work->offer.choice_count == choice_count &&
      (!choice_count || work->offer.work == kind)) {
    return false;
  }
  IREE_ASSERT(work->offer.key.revision != UINT64_MAX);
  ++work->offer.key.revision;
  work->offer.choice_count = choice_count;
  if (kind) {
    work->offer.work = kind;
  }
  loom_service_registration_publish_offer(registration, work);
  return true;
}

static bool loom_service_registration_refresh_offers(
    loom_service_registration_t* registration) {
  const loom_runtime_service_t service = registration->engine;
  const bool needs_quiescence = registration->retirement_requested ||
                                registration->parameter_operation !=
                                    LOOM_SERVICE_PARAMETER_OPERATION_NONE;
  const bool requests = needs_quiescence && service.self &&
                        service.vtable->has_requests(service.self);
  const bool pending = needs_quiescence && service.self &&
                       service.vtable->is_pending(service.self);
  const bool active = needs_quiescence && service.self &&
                      service.vtable->is_active(service.self);
  loom_runtime_service_work_flags_t kind = 0;
  if (!registration->stopping && !registration->retirement_requested &&
      !registration->lifecycle.operation.key.serial) {
    if (registration->phase == LOOM_SERVICE_REGISTRATION_DECLARED &&
        (registration->demand ||
         registration->parameter_operation ==
             LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE)) {
      kind = LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
    } else if (registration->phase == LOOM_SERVICE_REGISTRATION_READY &&
               !requests && !active) {
      switch (registration->parameter_operation) {
        case LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE:
          kind = LOOM_RUNTIME_SERVICE_WORK_ACQUISITION;
          break;
        case LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE:
          if (!(registration->pinned_components &
                (1u << registration->parameter_component)) &&
              !registration->vtable
                   ->parameters(registration->self,
                                registration->parameter_component)
                   .parameters.pin_count) {
            kind = LOOM_RUNTIME_SERVICE_WORK_RELEASE;
          }
          break;
        case LOOM_SERVICE_PARAMETER_OPERATION_NONE:
          break;
      }
    }
  }
  bool changed = loom_service_registration_update_offer(
      registration, &registration->lifecycle, kind);
  const bool release =
      registration->retirement_requested &&
      registration->phase != LOOM_SERVICE_REGISTRATION_PREPARING &&
      registration->phase != LOOM_SERVICE_REGISTRATION_RETIRED &&
      registration->phase != LOOM_SERVICE_REGISTRATION_RELEASE_FAILED &&
      registration->parameter_operation ==
          LOOM_SERVICE_PARAMETER_OPERATION_NONE &&
      !registration->retirement.operation.key.serial && !requests && !pending;
  changed |= loom_service_registration_update_offer(
      registration, &registration->retirement,
      release ? LOOM_RUNTIME_SERVICE_WORK_RELEASE : 0);
  return changed;
}

static void loom_service_registration_record_error(
    loom_service_registration_t* registration, const iree_status_t status) {
  ++registration->diagnostic_revision;
  if (registration->stream) {
    loom_runtime_telemetry_record_t record = {
        .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_ERROR};
    loom_service_telemetry_diagnostic_t diagnostic = {0};
    loom_service_telemetry_diagnostic_capture(status, &diagnostic);
    memcpy(record.data.words, &diagnostic, sizeof(diagnostic));
    loom_runtime_telemetry_stream_append(registration->stream, &record);
  }
}

void loom_service_registration_fail(loom_service_registration_t* registration,
                                    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  if (iree_status_is_ok(registration->status) ||
      registration->phase == LOOM_SERVICE_REGISTRATION_RETIRING) {
    loom_service_registration_record_error(registration, status);
  }
  registration->status = iree_status_join(registration->status, status);
  registration->stopping = true;
  if (!loom_service_registration_has_parameter_work(registration)) {
    loom_service_registration_set_parameter_intent(
        registration, registration->parameter_component,
        LOOM_SERVICE_PARAMETER_OPERATION_NONE);
  }
  // An application failure cannot relinquish an accepted compiler/IO borrower.
  // Its terminal preparation result establishes the next lifecycle phase.
  if (registration->phase == LOOM_SERVICE_REGISTRATION_DECLARED ||
      registration->phase == LOOM_SERVICE_REGISTRATION_READY) {
    loom_service_registration_set_phase(registration,
                                        LOOM_SERVICE_REGISTRATION_FAILED);
  }
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
  }
}

static void loom_service_registration_record_failure(
    loom_service_registration_t* registration, iree_status_t status) {
  if (!iree_status_is_ok(status) &&
      registration->phase != LOOM_SERVICE_REGISTRATION_FAILED &&
      registration->phase != LOOM_SERVICE_REGISTRATION_RELEASE_FAILED) {
    loom_service_registration_fail(registration, iree_status_clone(status));
    loom_service_registration_set_phase(registration,
                                        LOOM_SERVICE_REGISTRATION_FAILED);
  }
}

void loom_service_registration_initialize(
    const loom_service_registration_vtable_t* vtable, void* self,
    uint32_t components, uint64_t device_ordinal, const void* capacity_domain,
    loom_runtime_service_control_binding_t binding,
    loom_runtime_wake_callback_t observer, iree_string_view_t name,
    loom_runtime_telemetry_stream_t* stream,
    loom_service_registration_t* registration) {
  registration->vtable = vtable;
  registration->self = self;
  registration->observer = observer;
  registration->name = name;
  registration->stream = stream;
  registration->engine.capacity_domain = capacity_domain;
  IREE_TRACE(loom_service_trace_initialize());
  registration->components = components;
  registration->demand = LOOM_SERVICE_REGISTRATION_DEMAND_PREPARATION;
  registration->control = binding.control;
  registration->devices = UINT64_C(1) << device_ordinal;
  registration->instance_ordinal = binding.instance_ordinal;
  registration->use.resource_ordinal = device_ordinal;
  registration->lifecycle.offer = (loom_runtime_service_offer_t){
      .key = {binding.offer_ordinal, 1},
      .work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION,
      .choice_count = 1};
  registration->retirement.offer.key.ordinal = binding.offer_ordinal + 1;
  registration->lifecycle.operation.key.ordinal = binding.operation_ordinal;
  registration->retirement.operation.key.ordinal =
      binding.operation_ordinal + 1;
  loom_service_registration_publish_offer(registration,
                                          &registration->lifecycle);
  loom_service_registration_publish(registration);
}

loom_runtime_service_control_binding_t loom_service_registration_binding(
    loom_service_registration_t* registration, uint64_t offer_ordinal,
    uint64_t operation_ordinal, uint64_t instance_ordinal) {
  return (loom_runtime_service_control_binding_t){
      .control = registration->control,
      .offer_ordinal = offer_ordinal,
      .operation_ordinal = operation_ordinal,
      .instance_ordinal = instance_ordinal,
      .publish_observation = loom_service_registration_publish_engine,
      .publisher = registration};
}

loom_service_registration_snapshot_t loom_service_registration_snapshot(
    const loom_service_registration_t* registration) {
  return (loom_service_registration_snapshot_t){
      .phase = registration->phase,
      .retirement_requested = registration->retirement_requested,
      .parameter_operation = registration->parameter_operation,
      .parameter_component = registration->parameter_component,
      .activation_phase = registration->activation_phase,
      .pinned_components = registration->pinned_components,
      .service = registration->phase == LOOM_SERVICE_REGISTRATION_READY
                     ? registration->engine
                     : (loom_runtime_service_t){.capacity_domain =
                                                    registration->engine
                                                        .capacity_domain},
      .status = registration->status,
      .diagnostic_revision = registration->diagnostic_revision,
      .compilation = &registration->compilation,
      .preparation_start = registration->preparation_start,
      .preparation_duration_ns = registration->preparation_duration_ns};
}

static bool loom_service_registration_parameters_pending(
    const loom_service_registration_t* registration) {
  return loom_service_registration_has_parameter_work(registration) &&
         (registration->parameter_operation ==
              LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE ||
          iree_any_bit_set(registration->lifecycle.operation.waits,
                           LOOM_RUNTIME_SERVICE_WAIT_CPU |
                               LOOM_RUNTIME_SERVICE_WAIT_DEVICE));
}

bool loom_service_registration_is_pending(
    const loom_service_registration_t* registration) {
  return registration->phase == LOOM_SERVICE_REGISTRATION_PREPARING ||
         registration->phase == LOOM_SERVICE_REGISTRATION_RETIRING ||
         loom_service_registration_parameters_pending(registration) ||
         (registration->engine.self &&
          registration->engine.vtable->is_pending(registration->engine.self));
}

bool loom_service_registration_is_active(
    const loom_service_registration_t* registration) {
  return registration->phase == LOOM_SERVICE_REGISTRATION_PREPARING ||
         registration->phase == LOOM_SERVICE_REGISTRATION_RETIRING ||
         loom_service_registration_parameters_pending(registration) ||
         (registration->engine.self &&
          registration->engine.vtable->is_active(registration->engine.self));
}

void loom_service_registration_set_demand(
    loom_service_registration_t* registration,
    loom_service_registration_demand_t demand) {
  if (registration->demand == demand) {
    return;
  }
  registration->demand = demand;
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
  }
}

static void loom_service_registration_set_parameter_intent(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_service_parameter_operation_t operation) {
  IREE_TRACE({
    if (registration->parameter_operation != operation) {
      IREE_TRACE_SECTION_LEAVE(registration->trace.parameters);
      registration->trace.parameters =
          operation != LOOM_SERVICE_PARAMETER_OPERATION_NONE
              ? loom_service_trace_section_enter(
                    operation == LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE
                        ? LOOM_SERVICE_TRACE_SECTION_WARMUP
                        : LOOM_SERVICE_TRACE_SECTION_DEACTIVATE,
                    registration->name)
              : 0;
    }
  });
  if (component != registration->parameter_component) {
    // An old decision cannot grant the same operation on a different model.
    loom_service_registration_withdraw_offer(registration,
                                             &registration->lifecycle);
  }
  registration->parameter_component = component;
  registration->parameter_operation = operation;
}

iree_status_t loom_service_registration_request_parameters(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_service_parameter_operation_t operation) {
  if ((uint32_t)component >= LOOM_SERVICE_MODEL_COMPONENT_COUNT ||
      !(registration->components & (1u << component))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "registration has no such component");
  }
  if ((uint32_t)operation > LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown parameter operation");
  }
  if (registration->phase > LOOM_SERVICE_REGISTRATION_READY ||
      registration->stopping || registration->retirement_requested) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter intent requires a healthy registration");
  }
  if (loom_service_registration_has_parameter_work(registration) &&
      (operation != registration->parameter_operation ||
       component != registration->parameter_component)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "accepted parameter work must complete first");
  }
  if (operation == LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE) {
    if (registration->phase == LOOM_SERVICE_REGISTRATION_DECLARED) {
      operation = LOOM_SERVICE_PARAMETER_OPERATION_NONE;
    } else if (registration->phase != LOOM_SERVICE_REGISTRATION_READY) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "registration has no prepared parameter residency");
    } else if (!registration->vtable->parameters(registration->self, component)
                    .deactivatable) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "component cannot release its parameter backing");
    }
  }
  if (!loom_service_registration_has_parameter_work(registration)) {
    registration->activation_phase = LOOM_RUNTIME_ACTIVATION_WAITING;
  }
  loom_service_registration_set_parameter_intent(registration, component,
                                                 operation);
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
  }
  return iree_ok_status();
}

void loom_service_registration_request_retirement(
    loom_service_registration_t* registration,
    loom_service_retire_flags_t flags) {
  if (registration->engine.self) {
    // The application may have detached its ready-engine traversal. Accepted
    // native borrowers now wake the retained lifecycle owner through cleanup.
    registration->engine.vtable->set_observer(registration->engine.self,
                                              registration->observer);
  }
  registration->retirement_requested = true;
  registration->stopping |=
      iree_any_bit_set(flags, LOOM_SERVICE_RETIRE_FLAG_ABORT);
  if (!loom_service_registration_has_parameter_work(registration)) {
    loom_service_registration_set_parameter_intent(
        registration, registration->parameter_component,
        LOOM_SERVICE_PARAMETER_OPERATION_NONE);
  }
  if (registration->phase == LOOM_SERVICE_REGISTRATION_RELEASE_FAILED) {
    loom_service_registration_set_phase(registration,
                                        LOOM_SERVICE_REGISTRATION_RETIRING);
  }
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
  }
}

bool loom_service_registration_progress(
    loom_service_registration_t* registration,
    loom_runtime_service_flags_t flags) {
  const loom_service_registration_phase_t previous_phase = registration->phase;
  const loom_service_parameter_operation_t previous_operation =
      registration->parameter_operation;
  const loom_runtime_activation_phase_t previous_activation =
      registration->activation_phase;
  const bool was_stopping = registration->stopping;
  registration->stopping |=
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  if (registration->stopping &&
      !loom_service_registration_has_parameter_work(registration)) {
    loom_service_registration_set_parameter_intent(
        registration, registration->parameter_component,
        LOOM_SERVICE_PARAMETER_OPERATION_NONE);
  }
  bool progress = false;
  if (registration->phase == LOOM_SERVICE_REGISTRATION_PREPARING) {
    bool complete = false;
    iree_status_free(
        loom_service_registration_prepare(registration, 0, &complete));
  }
  if (registration->engine.self &&
      registration->phase != LOOM_SERVICE_REGISTRATION_RETIRING &&
      registration->phase != LOOM_SERVICE_REGISTRATION_RELEASE_FAILED) {
    bool service_progress = false;
    iree_status_t status = registration->vtable->progress(
        registration->self,
        flags |
            (registration->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0),
        &service_progress);
    loom_service_registration_record_failure(registration, status);
    iree_status_free(status);
    progress |= service_progress;
  }
  if (loom_service_registration_has_parameter_work(registration)) {
    if (registration->parameter_operation ==
        LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE) {
      loom_runtime_activation_phase_t phase;
      iree_status_t component_status = iree_ok_status();
      iree_status_free(loom_service_registration_activate(
          registration, registration->parameter_component,
          registration->stopping || registration->retirement_requested
              ? LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL
              : 0,
          &phase, &component_status));
      iree_status_free(component_status);
    } else if (registration->parameter_operation ==
               LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE) {
      bool complete = false;
      iree_status_free(loom_service_registration_deactivate(
          registration, registration->parameter_component, &complete));
    }
  }
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
    progress = true;
  }
  if (registration->phase == LOOM_SERVICE_REGISTRATION_RETIRING ||
      (registration->stopping && registration->retirement_requested &&
       registration->retirement.offer.choice_count)) {
    // Cancellation/failure cleanup belongs to accepted native ownership, but
    // still requires the application's explicit external-borrow handoff.
    bool complete = false;
    iree_status_free(loom_service_registration_retire(registration, &complete));
  }
  return progress || registration->phase != previous_phase ||
         registration->parameter_operation != previous_operation ||
         registration->activation_phase != previous_activation ||
         registration->stopping != was_stopping;
}

loom_runtime_service_schedule_outcome_t loom_service_registration_try_schedule(
    loom_service_registration_t* registration,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags) {
  loom_runtime_service_schedule_outcome_t outcome = {0};
  if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    return outcome;
  }
  if (offer.ordinal != registration->lifecycle.offer.key.ordinal &&
      offer.ordinal != registration->retirement.offer.key.ordinal) {
    if (registration->phase != LOOM_SERVICE_REGISTRATION_READY ||
        registration->stopping ||
        registration->lifecycle.operation.key.serial) {
      return outcome;
    }
    iree_status_t status = registration->engine.vtable->try_schedule(
        registration->engine.self, offer, choice_ordinal, flags, &outcome);
    loom_service_registration_record_failure(registration, status);
    iree_status_free(status);
    if (loom_service_registration_refresh_offers(registration)) {
      loom_service_registration_publish(registration);
    }
    return outcome;
  }
  loom_service_registration_work_t* work =
      offer.ordinal == registration->lifecycle.offer.key.ordinal
          ? &registration->lifecycle
          : &registration->retirement;
  if (!work->offer.choice_count || choice_ordinal != 0 ||
      offer.revision != work->offer.key.revision) {
    return outcome;
  }
  const loom_runtime_service_work_flags_t kind = work->offer.work;
  loom_service_registration_begin_work(registration, work, kind);
  outcome.accepted = true;
  outcome.operation = work->operation.key;
  bool complete = false;
  iree_status_t status = iree_ok_status();
  if (offer.ordinal == registration->retirement.offer.key.ordinal) {
    status = loom_service_registration_retire(registration, &complete);
  } else if (kind == LOOM_RUNTIME_SERVICE_WORK_PREPARATION) {
    status = loom_service_registration_prepare(registration, 0, &complete);
  } else if (kind == LOOM_RUNTIME_SERVICE_WORK_ACQUISITION) {
    loom_runtime_activation_phase_t phase;
    iree_status_t component_status = iree_ok_status();
    status = loom_service_registration_activate(
        registration, registration->parameter_component, 0, &phase,
        &component_status);
    outcome.status_code = iree_status_code(component_status);
    iree_status_free(component_status);
  } else {
    status = loom_service_registration_deactivate(
        registration, registration->parameter_component, &complete);
  }
  if (!iree_status_is_ok(status)) {
    outcome.status_code = iree_status_code(status);
  }
  iree_status_free(status);
  return outcome;
}

iree_status_t loom_service_registration_prepare(
    loom_service_registration_t* registration,
    loom_service_registration_prepare_flags_t flags, bool* out_complete) {
  loom_service_registration_set_phase(registration,
                                      LOOM_SERVICE_REGISTRATION_PREPARING);
  loom_service_registration_begin_work(registration, &registration->lifecycle,
                                       LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  loom_service_preparation_result_t result = {0};
  iree_status_t status = registration->vtable->prepare(
      registration->self,
      flags | loom_service_registration_prepare_flags(registration), &result,
      &registration->compilation);
  *out_complete =
      iree_any_bit_set(result.flags, LOOM_SERVICE_PREPARATION_RESULT_COMPLETE);
  if (!*out_complete) {
    return status;
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t component = 0; component < LOOM_SERVICE_MODEL_COMPONENT_COUNT;
         ++component) {
      if (registration->pinned_components & (1u << component)) {
        registration->vtable->set_pinned(
            registration->self, (loom_service_model_component_t)component,
            true);
      }
    }
    registration->engine = result.service;
    registration->engine.vtable->set_observer(registration->engine.self,
                                              registration->observer);
    // A completed provider may race an application failure. Retain its engine
    // for cleanup without publishing a successful registration after failure.
    loom_service_registration_set_phase(registration,
                                        iree_status_is_ok(registration->status)
                                            ? LOOM_SERVICE_REGISTRATION_READY
                                            : LOOM_SERVICE_REGISTRATION_FAILED);
  }
  loom_service_registration_record_failure(registration, status);
  if (iree_status_is_ok(status) ||
      !iree_any_bit_set(result.flags,
                        LOOM_SERVICE_PREPARATION_RESULT_RETAINED)) {
    loom_service_registration_end_work(registration, &registration->lifecycle);
  }
  return status;
}

loom_service_registration_parameters_t loom_service_registration_parameters(
    const loom_service_registration_t* registration,
    loom_service_model_component_t component) {
  return registration->vtable->parameters(registration->self, component);
}

void loom_service_registration_set_pinned(
    loom_service_registration_t* registration,
    loom_service_model_component_t component, bool pinned) {
  if (pinned) {
    registration->pinned_components |= 1u << component;
  } else {
    registration->pinned_components &= ~(1u << component);
  }
  if (registration->phase == LOOM_SERVICE_REGISTRATION_READY) {
    registration->vtable->set_pinned(registration->self, component, pinned);
  }
  if (loom_service_registration_refresh_offers(registration)) {
    loom_service_registration_publish(registration);
  }
}

iree_status_t loom_service_registration_activate(
    loom_service_registration_t* registration,
    loom_service_model_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  loom_service_registration_work_t* work = &registration->lifecycle;
  loom_service_registration_set_parameter_intent(
      registration, component, LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE);
  loom_service_registration_begin_work(registration, work,
                                       LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  loom_service_activation_result_t result;
  iree_status_t status = registration->vtable->activate(
      registration->self, component, flags, &result);
  *out_phase = result.phase;
  *out_component_status = result.component_status;
  registration->activation_phase = result.phase;
  if (!iree_status_is_ok(*out_component_status)) {
    loom_service_registration_record_error(registration, *out_component_status);
  }
  if (*out_phase == LOOM_RUNTIME_ACTIVATION_COMPLETE) {
    loom_service_registration_set_parameter_intent(
        registration, registration->parameter_component,
        LOOM_SERVICE_PARAMETER_OPERATION_NONE);
    loom_service_registration_record_failure(registration, status);
    if (iree_status_is_ok(status)) {
      loom_service_registration_end_work(registration, work);
    }
  }
  if (work->operation.key.serial) {
    if (work->operation.waits != result.waits) {
      work->operation.waits = result.waits;
      loom_runtime_service_publish_operation(
          registration->control, &work->operation, &registration->use);
    }
  }
  return status;
}

iree_status_t loom_service_registration_deactivate(
    loom_service_registration_t* registration,
    loom_service_model_component_t component, bool* out_complete) {
  loom_service_registration_set_parameter_intent(
      registration, component, LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE);
  loom_service_registration_begin_work(registration, &registration->lifecycle,
                                       LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  iree_status_t status = registration->vtable->deactivate(
      registration->self, component, out_complete);
  if (*out_complete || !iree_status_is_ok(status)) {
    loom_service_registration_set_parameter_intent(
        registration, registration->parameter_component,
        LOOM_SERVICE_PARAMETER_OPERATION_NONE);
    loom_service_registration_record_failure(registration, status);
    if (iree_status_is_ok(status)) {
      loom_service_registration_end_work(registration,
                                         &registration->lifecycle);
    }
  }
  return status;
}

iree_status_t loom_service_registration_retire(
    loom_service_registration_t* registration, bool* out_complete) {
  *out_complete = true;
  if (!registration ||
      registration->phase == LOOM_SERVICE_REGISTRATION_RETIRED) {
    return iree_ok_status();
  }
  registration->retirement_requested = true;
  loom_service_registration_set_phase(registration,
                                      LOOM_SERVICE_REGISTRATION_RETIRING);
  loom_service_registration_withdraw_offer(registration,
                                           &registration->lifecycle);
  loom_service_registration_begin_work(registration, &registration->retirement,
                                       LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  // The provider may destroy its ready engine before model retirement joins.
  // Keep its copied ownership facts, but never call that engine again.
  registration->engine = (loom_runtime_service_t){
      .capacity_domain = registration->engine.capacity_domain};
  iree_status_t status =
      registration->vtable->retire(registration->self, out_complete);
  if (iree_status_is_ok(status) && !*out_complete) {
    return iree_ok_status();
  }
  if (!iree_status_is_ok(status)) {
    loom_service_registration_record_failure(registration, status);
    loom_service_registration_set_phase(
        registration, LOOM_SERVICE_REGISTRATION_RELEASE_FAILED);
    return status;
  }
  loom_service_registration_set_phase(registration,
                                      LOOM_SERVICE_REGISTRATION_RETIRED);
  for (uint64_t i = 0; i < registration->lifecycle.operation.key.ordinal; ++i) {
    const loom_runtime_service_operation_t operation = {.key.ordinal = i};
    loom_runtime_service_publish_operation(registration->control, &operation,
                                           NULL);
  }
  registration->observation = (loom_runtime_service_observation_t){0};
  loom_service_registration_end_work(registration, &registration->lifecycle);
  loom_service_registration_end_work(registration, &registration->retirement);
  return iree_ok_status();
}

void loom_service_registration_free(loom_service_registration_t* registration) {
  if (registration) {
    IREE_TRACE_SECTION_LEAVE(registration->trace.lifecycle);
    IREE_TRACE_SECTION_LEAVE(registration->trace.parameters);
    iree_status_free(registration->status);
    registration->vtable->destroy(registration->self);
  }
}

const loom_runtime_service_control_t* loom_service_registration_control(
    const loom_service_registration_t* registration) {
  return registration->control;
}
