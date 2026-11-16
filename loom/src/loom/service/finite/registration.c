// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/registration.h"

#include "iree/hal/api.h"

enum {
  LOOM_FINITE_REGISTRATION_COHORT = 0,
  LOOM_FINITE_REGISTRATION_LIFECYCLE = 1,
  LOOM_FINITE_REGISTRATION_RETIREMENT = 2,
  LOOM_FINITE_REGISTRATION_SLOT_COUNT = 3,
};

typedef struct loom_service_finite_registration_work_t {
  // Legal native operation, withdrawn when its explicit call accepts it.
  loom_runtime_service_offer_t offer;
  // Accepted ownership, independent of offer withdrawal or a terminal error.
  loom_runtime_service_operation_t operation;
} loom_service_finite_registration_work_t;

struct loom_service_finite_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed execution device, alive through retirement.
  loom_runtime_device_t* device;
  // Copied service configuration; its views retain their original ownership.
  loom_service_finite_registration_options_t options;
  // Nonblocking notification target, alive through retirement.
  loom_runtime_wake_callback_t observer;
  // Owned source load, consumed after successful preparation or during retire.
  loom_service_finite_load_t* load;
  // Owned prepared model, retained until actual consumer retirement.
  loom_service_finite_model_t* model;
  // Owned service, destroyed after all request and CPU borrowers retire.
  loom_service_finite_service_t* service;
  // Ready-engine capabilities established once at publication, cleared at free.
  loom_runtime_service_t engine;
  // Tail-allocated observations, retained independently of model residency.
  loom_runtime_service_control_t* control;
  // Explicit warmup, joined before model retirement.
  loom_runtime_activation_t* activation;
  // Native publication and release state, independent of parameter intent.
  loom_service_finite_registration_phase_t phase;
  // One requested or accepted parameter operation; admission is caller-owned.
  loom_service_finite_parameter_operation_t parameter_operation;
  // Owned native failure, retained independently of engine and model storage.
  iree_status_t status;
  // Irreversible cancellation of new work; does not authorize destruction.
  bool stopping;
  // The application has relinquished external service and tokenizer views.
  bool retirement_requested;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
  // Serialized offer legality and accepted identities, independent of
  // readiness.
  struct {
    // Exact placement retained by every model operation in this registration.
    uint64_t devices;
    // Stable device binding; lifecycle work has no source-provided estimate.
    loom_runtime_service_use_t use;
    // Latest ready-engine contribution, including admission outside progress.
    loom_runtime_service_observation_t engine;
    // Serialized preparation, activation, or deactivation ownership.
    loom_service_finite_registration_work_t lifecycle;
    // Separate teardown ownership: a failed activation can still retain a pin
    // while model retirement joins the actual borrowers that release it.
    loom_service_finite_registration_work_t retirement;
    // Nonrecycled accepted lifetime across both native operation slots.
    uint64_t next_serial;
  } scheduling;
  // Cold terminal preparation evidence, surviving load and model destruction.
  loom_runtime_compilation_report_t compilation;
};

static bool loom_service_finite_registration_refresh_offers(
    loom_service_finite_registration_t* registration);

static void loom_service_finite_registration_publish(
    loom_service_finite_registration_t* registration) {
  loom_runtime_service_observation_t observation =
      registration->scheduling.engine;
  const loom_service_finite_registration_work_t* lifecycle =
      &registration->scheduling.lifecycle;
  const loom_service_finite_registration_work_t* retirement =
      &registration->scheduling.retirement;
  observation.offer_count +=
      lifecycle->offer.choice_count + retirement->offer.choice_count;
  observation.pending_operation_count +=
      (lifecycle->operation.key.serial != 0) +
      (retirement->operation.key.serial != 0);
  if (lifecycle->offer.choice_count || retirement->offer.choice_count) {
    observation.offered_devices |= registration->scheduling.devices;
  }
  observation.accepted_devices |=
      lifecycle->operation.devices | retirement->operation.devices;
  loom_runtime_service_publish_observation(registration->control, &observation);
}

static void loom_service_finite_registration_publish_engine(
    void* self, const loom_runtime_service_observation_t* observation) {
  loom_service_finite_registration_t* registration = self;
  registration->scheduling.engine = *observation;
  loom_service_finite_registration_refresh_offers(registration);
  loom_service_finite_registration_publish(registration);
}

static void loom_service_finite_registration_publish_offer(
    loom_service_finite_registration_t* registration,
    loom_service_finite_registration_work_t* work) {
  const loom_runtime_service_choice_publication_t choice = {
      .choice = {.devices = registration->scheduling.devices, .use_count = 1},
      .uses = &registration->scheduling.use};
  loom_runtime_service_publish_offer(registration->control, &work->offer,
                                     &choice);
}

static void loom_service_finite_registration_withdraw_offer(
    loom_service_finite_registration_t* registration,
    loom_service_finite_registration_work_t* work) {
  if (work->offer.choice_count) {
    IREE_ASSERT(work->offer.key.revision != UINT64_MAX);
    ++work->offer.key.revision;
    work->offer.choice_count = 0;
    loom_service_finite_registration_publish_offer(registration, work);
  }
}

static void loom_service_finite_registration_begin_work(
    loom_service_finite_registration_t* registration,
    loom_service_finite_registration_work_t* work,
    loom_runtime_service_work_flags_t kind) {
  if (work->operation.key.serial) {
    // Repeated explicit calls progress the same accepted unit. Its public
    // ownership contract excludes a different operation until it retires.
    IREE_ASSERT(work->operation.work == kind);
    return;
  }
  if (work->offer.work != kind || !work->offer.choice_count) {
    IREE_ASSERT(work->offer.key.revision != UINT64_MAX);
    ++work->offer.key.revision;
    work->offer.work = kind;
    work->offer.choice_count = 1;
    loom_service_finite_registration_publish_offer(registration, work);
  }
  IREE_ASSERT(registration->scheduling.next_serial != UINT64_MAX);
  work->operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = work->operation.key.ordinal,
              .serial = ++registration->scheduling.next_serial},
      .origin = work->offer.key,
      .work = kind,
      .devices = registration->scheduling.devices,
      .use_count = 1};
  loom_service_finite_registration_withdraw_offer(registration, work);
  loom_runtime_service_publish_operation(
      registration->control, &work->operation, &registration->scheduling.use);
  loom_service_finite_registration_publish(registration);
}

static void loom_service_finite_registration_end_work(
    loom_service_finite_registration_t* registration,
    loom_service_finite_registration_work_t* work) {
  if (!work->operation.key.serial) {
    return;
  }
  work->operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = work->operation.key.ordinal}};
  loom_runtime_service_publish_operation(registration->control,
                                         &work->operation, NULL);
  loom_service_finite_registration_refresh_offers(registration);
  loom_service_finite_registration_publish(registration);
}

static bool loom_service_finite_registration_update_offer(
    loom_service_finite_registration_t* registration,
    loom_service_finite_registration_work_t* work,
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
  loom_service_finite_registration_publish_offer(registration, work);
  return true;
}

static bool loom_service_finite_registration_refresh_offers(
    loom_service_finite_registration_t* registration) {
  const loom_runtime_service_t service = registration->engine;
  const bool needs_quiescence =
      registration->retirement_requested ||
      registration->parameter_operation !=
          LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE;
  const bool requests = needs_quiescence && service.self &&
                        service.vtable->has_requests(service.self);
  const bool pending = needs_quiescence && service.self &&
                       service.vtable->is_pending(service.self);
  const bool active = needs_quiescence && service.self &&
                      service.vtable->is_active(service.self);
  loom_runtime_service_work_flags_t kind = 0;
  if (!registration->stopping && !registration->retirement_requested &&
      !registration->scheduling.lifecycle.operation.key.serial) {
    if (registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_DECLARED) {
      kind = LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
    } else if (registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_READY &&
               !requests && !active) {
      switch (registration->parameter_operation) {
        case LOOM_SERVICE_FINITE_PARAMETER_OPERATION_ACTIVATE:
          kind = LOOM_RUNTIME_SERVICE_WORK_ACQUISITION;
          break;
        case LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE:
          if (!loom_runtime_residency_is_pinned(
                  loom_service_finite_model_residency(registration->model))) {
            kind = LOOM_RUNTIME_SERVICE_WORK_RELEASE;
          }
          break;
        case LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE:
          break;
      }
    }
  }
  bool changed = loom_service_finite_registration_update_offer(
      registration, &registration->scheduling.lifecycle, kind);
  const bool release =
      registration->retirement_requested &&
      registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_PREPARING &&
      registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_RETIRED &&
      registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED &&
      registration->parameter_operation ==
          LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE &&
      !registration->scheduling.retirement.operation.key.serial && !requests &&
      !pending;
  changed |= loom_service_finite_registration_update_offer(
      registration, &registration->scheduling.retirement,
      release ? LOOM_RUNTIME_SERVICE_WORK_RELEASE : 0);
  return changed;
}

static void loom_service_finite_registration_record_failure(
    loom_service_finite_registration_t* registration, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    if (registration->phase ==
        LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED) {
      return;
    }
    if (registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_FAILED) {
      registration->status =
          iree_status_join(registration->status, iree_status_clone(status));
    }
    registration->stopping = true;
    registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_FAILED;
  }
}

static iree_status_t loom_service_finite_registration_allocate_storage(
    iree_host_size_t variant_capacity, uint64_t incarnation,
    iree_allocator_t host_allocator,
    loom_service_finite_registration_t** out_registration) {
  *out_registration = NULL;
  iree_host_size_t operation_capacity = 0;
  if (!iree_host_size_checked_add(LOOM_FINITE_REGISTRATION_SLOT_COUNT,
                                  variant_capacity, &operation_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "finite operation capacity is not representable");
  }
  iree_host_size_t* operation_use_capacities = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, operation_capacity, sizeof(*operation_use_capacities),
      (void**)&operation_use_capacities));
  for (iree_host_size_t i = 0; i < operation_capacity; ++i) {
    operation_use_capacities[i] = 1;
  }
  const iree_host_size_t use_capacity = 1;
  const loom_runtime_service_offer_capacity_t offers[] = {
      {1, &use_capacity}, {1, &use_capacity}, {1, &use_capacity}};
  const loom_runtime_service_control_capacity_t capacity = {
      .offer_capacity = LOOM_FINITE_REGISTRATION_SLOT_COUNT,
      .offers = offers,
      .operation_capacity = operation_capacity,
      .operation_use_capacities = operation_use_capacities};
  iree_host_size_t control_length = 0;
  iree_status_t status =
      loom_runtime_service_control_size(&capacity, &control_length);
  iree_host_size_t allocation_length = 0, control_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(loom_service_finite_registration_t), &allocation_length,
        IREE_STRUCT_FIELD_ALIGNED(control_length, uint8_t, 64,
                                  &control_offset));
  }
  loom_service_finite_registration_t* registration = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_aligned(host_allocator, allocation_length,
                                           64, 0, (void**)&registration);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_service_control_initialize(
        &capacity, incarnation,
        iree_make_byte_span((uint8_t*)registration + control_offset,
                            control_length),
        &registration->control);
  }
  iree_allocator_free(host_allocator, operation_use_capacities);
  if (iree_status_is_ok(status)) {
    *out_registration = registration;
  } else {
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

iree_status_t loom_service_finite_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* model_options,
    const loom_service_finite_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_service_finite_registration_t** out_registration) {
  *out_registration = NULL;
  loom_service_finite_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(loom_service_finite_registration_allocate_storage(
      options->variant_capacity, options->incarnation, host_allocator,
      &registration));
  registration->host_allocator = host_allocator;
  registration->device = device;
  registration->options = *options;
  registration->observer = observer;
  const uint64_t device_ordinal =
      iree_hal_device_topology_info(loom_runtime_device_handle(device))
          ->topology_index;
  registration->scheduling.devices = UINT64_C(1) << device_ordinal;
  registration->scheduling.use.resource_ordinal = device_ordinal;
  registration->scheduling.lifecycle.offer = (loom_runtime_service_offer_t){
      .key = {LOOM_FINITE_REGISTRATION_LIFECYCLE, 1},
      .work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION,
      .choice_count = 1};
  registration->scheduling.retirement.offer.key.ordinal =
      LOOM_FINITE_REGISTRATION_RETIREMENT;
  registration->scheduling.lifecycle.operation.key.ordinal =
      options->variant_capacity + LOOM_FINITE_REGISTRATION_LIFECYCLE;
  registration->scheduling.retirement.operation.key.ordinal =
      options->variant_capacity + LOOM_FINITE_REGISTRATION_RETIREMENT;
  iree_status_t status = loom_service_finite_load_allocate(
      device, pool, model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    loom_service_finite_registration_publish_offer(
        registration, &registration->scheduling.lifecycle);
    loom_service_finite_registration_publish(registration);
    *out_registration = registration;
  } else {
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

loom_service_finite_registration_snapshot_t
loom_service_finite_registration_snapshot(
    const loom_service_finite_registration_t* registration) {
  return (loom_service_finite_registration_snapshot_t){
      .phase = registration->phase,
      .parameter_operation = registration->parameter_operation,
      .service = registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_READY
                     ? registration->service
                     : NULL,
      .status = registration->status,
      .compilation = &registration->compilation};
}

iree_status_t loom_service_finite_registration_request_parameters(
    loom_service_finite_registration_t* registration,
    loom_service_finite_parameter_operation_t operation) {
  if ((uint32_t)operation >
      LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown finite parameter operation");
  }
  if (registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_READY ||
      registration->stopping || registration->retirement_requested) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter intent requires a ready registration");
  }
  if (registration->scheduling.lifecycle.operation.key.serial &&
      operation != registration->parameter_operation) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "accepted parameter work must complete first");
  }
  if (operation == LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE &&
      !loom_service_finite_model_parameters(registration->model).elastic) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires elastic parameters");
  }
  registration->parameter_operation = operation;
  if (loom_service_finite_registration_refresh_offers(registration)) {
    loom_service_finite_registration_publish(registration);
  }
  return iree_ok_status();
}

void loom_service_finite_registration_request_retirement(
    loom_service_finite_registration_t* registration,
    loom_service_finite_retire_flags_t flags) {
  if (registration->engine.self) {
    // The application may have detached its ready-engine traversal. Accepted
    // native borrowers now wake the retained lifecycle owner through cleanup.
    registration->engine.vtable->set_observer(registration->engine.self,
                                              registration->observer);
  }
  registration->retirement_requested = true;
  registration->stopping |=
      iree_any_bit_set(flags, LOOM_SERVICE_FINITE_RETIRE_FLAG_ABORT);
  if (!registration->scheduling.lifecycle.operation.key.serial) {
    registration->parameter_operation =
        LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE;
  }
  if (registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED) {
    registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_RETIRING;
  }
  if (loom_service_finite_registration_refresh_offers(registration)) {
    loom_service_finite_registration_publish(registration);
  }
}

bool loom_service_finite_registration_progress(
    loom_service_finite_registration_t* registration,
    loom_runtime_service_flags_t flags) {
  const loom_service_finite_registration_phase_t previous_phase =
      registration->phase;
  const loom_service_finite_parameter_operation_t previous_operation =
      registration->parameter_operation;
  const bool was_stopping = registration->stopping;
  registration->stopping |=
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  if (registration->stopping &&
      !registration->scheduling.lifecycle.operation.key.serial) {
    registration->parameter_operation =
        LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE;
  }
  bool progress = false;
  if (registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_PREPARING) {
    bool complete = false;
    loom_service_finite_service_t* service = NULL;
    iree_status_free(loom_service_finite_registration_prepare(
        registration,
        registration->stopping || registration->retirement_requested
            ? LOOM_SERVICE_FINITE_LOAD_FLAG_CANCEL
            : 0,
        &complete, &service, &registration->compilation));
  }
  if (registration->service) {
    bool service_progress = false;
    iree_status_t status = loom_service_finite_service_progress(
        registration->service,
        flags |
            (registration->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0),
        &service_progress);
    loom_service_finite_registration_record_failure(registration, status);
    iree_status_free(status);
    progress |= service_progress;
  }
  if (registration->scheduling.lifecycle.operation.key.serial) {
    if (registration->parameter_operation ==
        LOOM_SERVICE_FINITE_PARAMETER_OPERATION_ACTIVATE) {
      loom_runtime_activation_phase_t phase;
      iree_status_free(loom_service_finite_registration_activate(
          registration,
          registration->stopping || registration->retirement_requested
              ? LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL
              : 0,
          &phase));
    } else if (registration->parameter_operation ==
               LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE) {
      bool complete = false;
      iree_status_free(
          loom_service_finite_registration_deactivate(registration, &complete));
    }
  }
  if (loom_service_finite_registration_refresh_offers(registration)) {
    loom_service_finite_registration_publish(registration);
    progress = true;
  }
  if (registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_RETIRING ||
      (registration->stopping && registration->retirement_requested &&
       registration->scheduling.retirement.offer.choice_count)) {
    // Cancellation/failure cleanup belongs to accepted native ownership, but
    // still requires the application's explicit external-borrow handoff.
    bool complete = false;
    iree_status_free(
        loom_service_finite_registration_retire(registration, &complete));
  }
  return progress || registration->phase != previous_phase ||
         registration->parameter_operation != previous_operation ||
         registration->stopping != was_stopping;
}

loom_runtime_service_schedule_outcome_t
loom_service_finite_registration_try_schedule(
    loom_service_finite_registration_t* registration,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags) {
  loom_runtime_service_schedule_outcome_t outcome = {0};
  if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    return outcome;
  }
  if (offer.ordinal == LOOM_FINITE_REGISTRATION_COHORT) {
    if (registration->phase != LOOM_SERVICE_FINITE_REGISTRATION_READY ||
        registration->stopping ||
        registration->scheduling.lifecycle.operation.key.serial) {
      return outcome;
    }
    iree_status_t status = loom_service_finite_service_try_schedule(
        registration->service, offer, choice_ordinal, flags,
        &outcome.operation);
    outcome.accepted = outcome.operation.serial != 0;
    outcome.status_code = iree_status_code(status);
    loom_service_finite_registration_record_failure(registration, status);
    iree_status_free(status);
    if (loom_service_finite_registration_refresh_offers(registration)) {
      loom_service_finite_registration_publish(registration);
    }
    return outcome;
  }
  loom_service_finite_registration_work_t* work = NULL;
  if (offer.ordinal == LOOM_FINITE_REGISTRATION_LIFECYCLE) {
    work = &registration->scheduling.lifecycle;
  } else if (offer.ordinal == LOOM_FINITE_REGISTRATION_RETIREMENT) {
    work = &registration->scheduling.retirement;
  }
  if (!work || !work->offer.choice_count || choice_ordinal != 0 ||
      offer.revision != work->offer.key.revision) {
    return outcome;
  }
  const loom_runtime_service_work_flags_t kind = work->offer.work;
  loom_service_finite_registration_begin_work(registration, work, kind);
  outcome.accepted = true;
  outcome.operation = work->operation.key;
  bool complete = false;
  iree_status_t status = iree_ok_status();
  if (offer.ordinal == LOOM_FINITE_REGISTRATION_RETIREMENT) {
    status = loom_service_finite_registration_retire(registration, &complete);
  } else if (kind == LOOM_RUNTIME_SERVICE_WORK_PREPARATION) {
    loom_service_finite_service_t* service = NULL;
    status = loom_service_finite_registration_prepare(
        registration, 0, &complete, &service, &registration->compilation);
  } else if (kind == LOOM_RUNTIME_SERVICE_WORK_ACQUISITION) {
    loom_runtime_activation_phase_t phase;
    status = loom_service_finite_registration_activate(registration, 0, &phase);
  } else {
    status =
        loom_service_finite_registration_deactivate(registration, &complete);
  }
  outcome.status_code = iree_status_code(status);
  iree_status_free(status);
  return outcome;
}

iree_status_t loom_service_finite_registration_prepare(
    loom_service_finite_registration_t* registration,
    loom_service_finite_load_flags_t flags, bool* out_complete,
    loom_service_finite_service_t** out_service,
    loom_runtime_compilation_report_t* out_report) {
  *out_service = NULL;
  registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_PREPARING;
  loom_service_finite_registration_begin_work(
      registration, &registration->scheduling.lifecycle,
      LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  iree_status_t status =
      loom_service_finite_load_advance(registration->load, flags, out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_service_finite_load_report(registration->load);
  registration->compilation = *out_report;
  if (iree_status_is_ok(status)) {
    registration->model =
        loom_service_finite_load_take_model(registration->load);
    status = loom_service_finite_load_free(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_service_finite_registration_options_t* options =
        &registration->options;
    status = loom_service_finite_service_allocate(
        loom_runtime_device_host(registration->device),
        loom_service_finite_model_generator(registration->model),
        options->protocol, options->name, options->capacity,
        options->variant_capacity, options->heartbeat_interval, options->stream,
        (loom_runtime_service_control_binding_t){
            .control = registration->control,
            .offer_ordinal = LOOM_FINITE_REGISTRATION_COHORT,
            .operation_ordinal = LOOM_FINITE_REGISTRATION_COHORT,
            .publish_observation =
                loom_service_finite_registration_publish_engine,
            .publisher = registration},
        registration->host_allocator, &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = registration->service;
    registration->engine =
        loom_service_finite_service_interface(registration->service).runtime;
    registration->engine.vtable->set_observer(registration->engine.self,
                                              registration->observer);
    registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_READY;
  }
  loom_service_finite_registration_record_failure(registration, status);
  if (iree_status_is_ok(status) || !registration->model) {
    // Failed load completion has joined its own model retirement. Failure
    // after model transfer instead retains that ownership until retire.
    loom_service_finite_registration_end_work(
        registration, &registration->scheduling.lifecycle);
  }
  return status;
}

loom_runtime_residency_snapshot_t loom_service_finite_registration_parameters(
    const loom_service_finite_registration_t* registration) {
  return loom_service_finite_model_parameters(registration->model);
}

void loom_service_finite_registration_set_pinned(
    loom_service_finite_registration_t* registration, bool pinned) {
  loom_service_finite_model_set_pinned(registration->model, pinned);
  if (loom_service_finite_registration_refresh_offers(registration)) {
    loom_service_finite_registration_publish(registration);
  }
}

iree_status_t loom_service_finite_registration_activate(
    loom_service_finite_registration_t* registration,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase) {
  loom_service_finite_registration_work_t* work =
      &registration->scheduling.lifecycle;
  registration->parameter_operation =
      LOOM_SERVICE_FINITE_PARAMETER_OPERATION_ACTIVATE;
  loom_service_finite_registration_begin_work(
      registration, work, LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  iree_status_t status = loom_runtime_activation_poll(
      &registration->activation,
      loom_service_finite_model_residency(registration->model),
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(registration->device)),
      registration->observer.wake, registration->observer.user_data, flags,
      registration->host_allocator, out_phase);
  if (*out_phase == LOOM_RUNTIME_ACTIVATION_COMPLETE) {
    registration->parameter_operation =
        LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE;
    loom_service_finite_registration_record_failure(registration, status);
    if (iree_status_is_ok(status)) {
      loom_service_finite_registration_end_work(registration, work);
    }
  }
  if (work->operation.key.serial) {
    const loom_runtime_service_wait_flags_t waits =
        *out_phase == LOOM_RUNTIME_ACTIVATION_WAITING
            ? LOOM_RUNTIME_SERVICE_WAIT_CAPACITY
        : *out_phase == LOOM_RUNTIME_ACTIVATION_LOADING
            ? LOOM_RUNTIME_SERVICE_WAIT_DEVICE
            : 0;
    if (work->operation.waits != waits) {
      work->operation.waits = waits;
      loom_runtime_service_publish_operation(registration->control,
                                             &work->operation,
                                             &registration->scheduling.use);
    }
  }
  return status;
}

iree_status_t loom_service_finite_registration_deactivate(
    loom_service_finite_registration_t* registration, bool* out_complete) {
  registration->parameter_operation =
      LOOM_SERVICE_FINITE_PARAMETER_OPERATION_DEACTIVATE;
  loom_service_finite_registration_begin_work(
      registration, &registration->scheduling.lifecycle,
      LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  iree_status_t status =
      loom_service_finite_model_deactivate(registration->model, out_complete);
  if (*out_complete || !iree_status_is_ok(status)) {
    registration->parameter_operation =
        LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE;
    loom_service_finite_registration_record_failure(registration, status);
    if (iree_status_is_ok(status)) {
      loom_service_finite_registration_end_work(
          registration, &registration->scheduling.lifecycle);
    }
  }
  return status;
}

iree_status_t loom_service_finite_registration_retire(
    loom_service_finite_registration_t* registration, bool* out_complete) {
  *out_complete = true;
  if (!registration ||
      registration->phase == LOOM_SERVICE_FINITE_REGISTRATION_RETIRED) {
    return iree_ok_status();
  }
  registration->retirement_requested = true;
  registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_RETIRING;
  loom_service_finite_registration_withdraw_offer(
      registration, &registration->scheduling.lifecycle);
  loom_service_finite_registration_begin_work(
      registration, &registration->scheduling.retirement,
      LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  iree_status_t status = loom_service_finite_load_free(registration->load);
  if (iree_status_is_ok(status)) {
    registration->load = NULL;
    loom_service_finite_service_free(registration->service);
    registration->service = NULL;
    registration->engine = (loom_runtime_service_t){0};
  }
  if (iree_status_is_ok(status) && registration->model &&
      !registration->retiring) {
    loom_service_finite_model_retire(registration->model,
                                     registration->observer.wake,
                                     registration->observer.user_data);
    registration->retiring = true;
  }
  if (iree_status_is_ok(status) &&
      !loom_service_finite_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    status = loom_service_finite_model_free_retired(registration->model);
  }
  if (!iree_status_is_ok(status)) {
    loom_service_finite_registration_record_failure(registration, status);
    registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED;
    return status;
  }
  registration->model = NULL;
  registration->phase = LOOM_SERVICE_FINITE_REGISTRATION_RETIRED;
  const loom_runtime_service_operation_t operation = {0};
  loom_runtime_service_publish_operation(registration->control, &operation,
                                         NULL);
  registration->scheduling.engine = (loom_runtime_service_observation_t){0};
  loom_service_finite_registration_end_work(
      registration, &registration->scheduling.lifecycle);
  loom_service_finite_registration_end_work(
      registration, &registration->scheduling.retirement);
  return iree_ok_status();
}

void loom_service_finite_registration_free(
    loom_service_finite_registration_t* registration) {
  if (registration) {
    iree_status_free(registration->status);
    iree_allocator_free_aligned(registration->host_allocator, registration);
  }
}

const loom_runtime_service_control_t* loom_service_finite_registration_control(
    const loom_service_finite_registration_t* registration) {
  return registration->control;
}
