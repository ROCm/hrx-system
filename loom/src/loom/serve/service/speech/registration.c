// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/registration.h"

#include "iree/hal/api.h"
#include "loom/runtime/device/pool.h"
#include "loom/serve/service/speech/voice_load.h"

typedef struct loom_serve_speech_registration_t {
  // Shared native lifecycle and accepted ownership.
  loom_service_registration_t registration;
  // Resolved source namespace, borrowed through original voice preparation.
  loom_runtime_sources_t sources;
  // Copied source service options, whose views outlive this provider.
  loom_serve_speech_service_options_t options;
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_runtime_device_t* device;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_runtime_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_speech_load_t* load;
  // Owned prepared model, retained through actual consumer retirement.
  loom_serve_speech_model_t* model;
  // Owned source service, destroyed after CPU and external borrowers retire.
  loom_serve_speech_service_t* service;
  // Original card recordings, joined before publication or retirement.
  loom_serve_speech_voice_load_t* voices;
  // Explicit warmup, joined before service/model retirement.
  loom_runtime_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} loom_serve_speech_registration_t;

static iree_status_t loom_serve_speech_registration_prepare(
    void* self, loom_service_registration_prepare_flags_t flags,
    loom_service_preparation_result_t* out_result,
    loom_runtime_compilation_report_t* out_report) {
  loom_serve_speech_registration_t* registration = self;
  const bool cancel =
      iree_any_bit_set(flags, LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL);
  bool complete = false;
  iree_status_t status = iree_ok_status();
  if (registration->load) {
    status = loom_serve_speech_load_advance(
        registration->load, cancel ? LOOM_SERVE_SPEECH_LOAD_FLAG_CANCEL : 0,
        &complete);
    if (!complete) {
      return status;
    }
    *out_report = *loom_serve_speech_load_report(registration->load);
    if (iree_status_is_ok(status)) {
      registration->model =
          loom_serve_speech_load_take_model(registration->load);
    }
    iree_status_t cleanup = loom_serve_speech_load_free(registration->load);
    if (iree_status_is_ok(cleanup)) {
      registration->load = NULL;
    }
    status = iree_status_join(status, cleanup);
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_service_allocate(
          loom_runtime_device_host(registration->device),
          loom_serve_speech_model_generator(registration->model),
          &registration->options, registration->stream,
          loom_service_registration_binding(&registration->registration, 0, 0,
                                            0),
          registration->host_allocator, &registration->service);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_voice_load_allocate(
          loom_runtime_device_host(registration->device),
          &registration->sources, registration->service,
          registration->registration.observer, registration->host_allocator,
          &registration->voices);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_speech_voice_load_advance(registration->voices, cancel,
                                                  &complete);
    if (!complete) {
      return status;
    }
    loom_serve_speech_voice_load_free(registration->voices);
    registration->voices = NULL;
  }
  out_result->flags = LOOM_SERVICE_PREPARATION_RESULT_COMPLETE |
                      (registration->model || registration->load
                           ? LOOM_SERVICE_PREPARATION_RESULT_RETAINED
                           : 0);
  if (iree_status_is_ok(status)) {
    out_result->service =
        loom_serve_speech_service_interface(registration->service).runtime;
  }
  return status;
}

static iree_status_t loom_serve_speech_registration_progress(
    void* self, loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_serve_speech_registration_t* registration = self;
  return loom_serve_speech_service_progress(registration->service, flags,
                                            out_progress);
}

static iree_status_t loom_serve_speech_registration_retire(void* self,
                                                           bool* out_complete) {
  loom_serve_speech_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_speech_load_free(registration->load));
    registration->load = NULL;
  }
  loom_serve_speech_service_free(registration->service);
  registration->service = NULL;
  if (registration->model && !registration->retiring) {
    loom_serve_speech_model_retire(
        registration->model, registration->registration.observer.wake,
        registration->registration.observer.user_data);
    registration->retiring = true;
  }
  if (!loom_serve_speech_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_speech_model_free_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void loom_serve_speech_registration_destroy(void* self) {
  loom_serve_speech_registration_t* registration = self;
  iree_allocator_free_aligned(registration->host_allocator, registration);
}

static loom_service_registration_parameters_t
loom_serve_speech_registration_parameters(
    void* self, loom_service_model_component_t component) {
  loom_serve_speech_registration_t* registration = self;
  const loom_runtime_residency_snapshot_t parameters =
      loom_runtime_residency_snapshot(
          loom_serve_speech_model_residency(registration->model));
  return (loom_service_registration_parameters_t){
      .phase = "ready",
      .prepared = true,
      .deactivatable = parameters.elastic,
      .parameters = parameters};
}

static void loom_serve_speech_registration_set_pinned(
    void* self, loom_service_model_component_t component, bool pinned) {
  loom_serve_speech_registration_t* registration = self;
  loom_runtime_residency_set_pinned(
      loom_serve_speech_model_residency(registration->model), pinned);
}

static iree_status_t loom_serve_speech_registration_deactivate(
    void* self, loom_service_model_component_t component, bool* out_complete) {
  loom_serve_speech_registration_t* registration = self;
  return loom_runtime_residency_deactivate(
      loom_serve_speech_model_residency(registration->model), out_complete);
}

static iree_status_t loom_serve_speech_registration_activate(
    void* self, loom_service_model_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  loom_serve_speech_registration_t* registration = self;
  return loom_runtime_activation_poll(
      &registration->activation,
      loom_serve_speech_model_residency(registration->model),
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(registration->device)),
      registration->registration.observer.wake,
      registration->registration.observer.user_data, flags,
      registration->host_allocator, out_phase);
}

static const loom_service_registration_vtable_t
    loom_serve_speech_registration_vtable = {
        .prepare = loom_serve_speech_registration_prepare,
        .progress = loom_serve_speech_registration_progress,
        .retire = loom_serve_speech_registration_retire,
        .parameters = loom_serve_speech_registration_parameters,
        .set_pinned = loom_serve_speech_registration_set_pinned,
        .activate = loom_serve_speech_registration_activate,
        .deactivate = loom_serve_speech_registration_deactivate,
        .destroy = loom_serve_speech_registration_destroy};

iree_status_t loom_serve_speech_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_serve_speech_model_options_t* model_options,
    const loom_serve_speech_service_options_t* options, uint64_t incarnation,
    loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_service_registration_t** out_registration) {
  *out_registration = NULL;
  const iree_host_size_t operation_capacity = 4;
  loom_serve_speech_registration_t* registration = NULL;
  loom_runtime_service_control_t* control = NULL;
  IREE_RETURN_IF_ERROR(loom_service_registration_allocate_storage(
      sizeof(*registration), 3, operation_capacity, incarnation, host_allocator,
      (void**)&registration, &control));
  registration->host_allocator = host_allocator;
  registration->device = device;
  registration->options = *options;
  registration->stream = stream;
  registration->sources = model_options->sources;
  iree_status_t status = loom_serve_speech_load_allocate(
      device, pool, model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    loom_service_registration_initialize(
        &loom_serve_speech_registration_vtable, registration,
        1u << LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
        iree_hal_device_topology_info(loom_runtime_device_handle(device))
            ->topology_index,
        loom_runtime_pool_residency_cache(pool),
        loom_runtime_service_control_binding(control, 1, operation_capacity - 2,
                                             0),
        observer, options->name, stream, &registration->registration);
    *out_registration = &registration->registration;
  } else {
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

loom_serve_speech_service_t* loom_serve_speech_registration_service(
    loom_service_registration_t* registration) {
  return (loom_serve_speech_service_t*)loom_service_registration_snapshot(
             registration)
      .service.self;
}
