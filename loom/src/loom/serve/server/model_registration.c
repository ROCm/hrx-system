// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/model_registration.h"

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/service/finite/component.h"
#include "loom/serve/service/finite/decision/protocol.h"
#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/finite/service.h"
#include "loom/serve/service/image/model.h"
#include "loom/serve/service/image/service.h"
#include "loom/serve/service/recognition/http.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/service/speech/model.h"
#include "loom/serve/service/speech/service.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/serve/service/text/service.h"

// The adapter owns the native parameter group and its temporary readiness
// operation. Neither the registry nor its management connection borrows either.
static iree_status_t loom_serve_registration_activate(
    loom_serve_activation_t** operation, loom_serve_residency_t* residency,
    loom_serve_host_t* host, loom_serve_wake_callback_t observer,
    loom_serve_activation_flags_t flags, iree_allocator_t host_allocator,
    loom_serve_activation_phase_t* out_phase) {
  *out_phase = LOOM_SERVE_ACTIVATION_COMPLETE;
  if (!*operation) {
    if (iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_serve_activation_allocate(
        residency, loom_serve_host_proactor_pool(host), observer.wake,
        observer.user_data, host_allocator, operation));
  }
  iree_status_t status =
      loom_serve_activation_advance(*operation, flags, out_phase);
  if (*out_phase == LOOM_SERVE_ACTIVATION_COMPLETE) {
    loom_serve_activation_free(*operation);
    *operation = NULL;
  }
  return status;
}

static loom_serve_registration_parameters_t
loom_serve_registration_prepared_parameters(loom_serve_residency_t* residency) {
  const loom_serve_residency_snapshot_t parameters =
      loom_serve_residency_snapshot(residency);
  return (loom_serve_registration_parameters_t){
      .phase = "ready",
      .prepared = true,
      .deactivatable = parameters.elastic,
      .parameters = parameters};
}

typedef struct loom_serve_finite_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_serve_device_t* device;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_wake_callback_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_finite_load_t* load;
  // Owned prepared model, kept through actual consumer retirement.
  loom_serve_finite_model_t* model;
  // Owned HTTP service, destroyed only after external borrowers retire.
  loom_serve_finite_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} loom_serve_finite_registration_t;

static iree_status_t loom_serve_finite_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  loom_serve_finite_registration_t* registration = self;
  iree_status_t status = loom_serve_finite_load_advance(
      registration->load, cancel ? LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_finite_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model = loom_serve_finite_load_take_model(registration->load);
    status = loom_serve_finite_load_free(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    status = loom_serve_finite_service_allocate(
        loom_serve_device_host(registration->device), registration->model,
        configuration->kind == LOOM_SERVE_MODEL_KIND_EMBEDDING
            ? loom_serve_embedding_protocol()
            : loom_serve_decision_protocol(),
        configuration->name, configuration->pending_requests,
        configuration->variant_capacity,
        registration->options->heartbeat_interval, registration->stream,
        registration->host_allocator, &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_finite_service_interface(registration->service);
  }
  return status;
}

static iree_status_t loom_serve_finite_registration_retire(void* self,
                                                           bool* out_complete) {
  loom_serve_finite_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_finite_load_free(registration->load));
    registration->load = NULL;
  }
  loom_serve_finite_service_free(registration->service);
  registration->service = NULL;
  if (registration->model && !registration->retiring) {
    loom_serve_finite_model_retire(registration->model,
                                   registration->observer.wake,
                                   registration->observer.user_data);
    registration->retiring = true;
  }
  if (!loom_serve_finite_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_model_free_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void loom_serve_finite_registration_destroy(void* self) {
  loom_serve_finite_registration_t* registration = self;
  iree_allocator_free(registration->host_allocator, registration);
}

static loom_serve_registration_parameters_t
loom_serve_finite_registration_parameters(
    void* self, loom_serve_model_component_t component) {
  loom_serve_finite_registration_t* registration = self;
  return loom_serve_registration_prepared_parameters(
      loom_serve_finite_model_residency(registration->model));
}

static void loom_serve_finite_registration_set_pinned(
    void* self, loom_serve_model_component_t component, bool pinned) {
  loom_serve_finite_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_finite_model_residency(registration->model), pinned);
}

static iree_status_t loom_serve_finite_registration_deactivate(
    void* self, loom_serve_model_component_t component, bool* out_complete) {
  loom_serve_finite_registration_t* registration = self;
  return loom_serve_residency_deactivate(
      loom_serve_finite_model_residency(registration->model), out_complete);
}

static iree_status_t loom_serve_finite_registration_activate(
    void* self, loom_serve_model_component_t component,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  loom_serve_finite_registration_t* registration = self;
  return loom_serve_registration_activate(
      &registration->activation,
      loom_serve_finite_model_residency(registration->model),
      loom_serve_device_host(registration->device), registration->observer,
      flags, registration->host_allocator, out_phase);
}

static iree_status_t loom_serve_finite_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_finite_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*registration), (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->options = options;
  registration->device = loom_serve_device_group_at(
      options->device_group, configuration->device_index);
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_finite_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options,
      .specialization_cache = configuration->specialization_cache};
  iree_status_t status = loom_serve_finite_load_allocate(
      registration->device, &model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = loom_serve_finite_registration_prepare,
        .retire = loom_serve_finite_registration_retire,
        .parameters = loom_serve_finite_registration_parameters,
        .set_pinned = loom_serve_finite_registration_set_pinned,
        .activate = loom_serve_finite_registration_activate,
        .deactivate = loom_serve_finite_registration_deactivate,
        .destroy = loom_serve_finite_registration_destroy};
  } else {
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

typedef struct loom_serve_image_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_serve_device_t* device;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_wake_callback_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_image_load_t* load;
  // Owned prepared model, retained through actual consumer retirement.
  loom_serve_image_model_t* model;
  // Owned HTTP service, destroyed after CPU and external borrowers retire.
  loom_serve_image_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} loom_serve_image_registration_t;

static iree_status_t loom_serve_image_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  loom_serve_image_registration_t* registration = self;
  iree_status_t status = loom_serve_image_load_advance(
      registration->load, cancel ? LOOM_SERVE_IMAGE_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_image_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model = loom_serve_image_load_take_model(registration->load);
    status = loom_serve_image_load_free(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    const loom_serve_image_service_options_t service_options = {
        .name = configuration->name,
        .width = configuration->image.width,
        .height = configuration->image.height,
        .adapter_enabled = configuration->image.adapter.size != 0,
        .pending_capacity = configuration->pending_requests,
        .variant_capacity = configuration->variant_capacity,
        .heartbeat_interval = registration->options->heartbeat_interval};
    status = loom_serve_image_service_allocate(
        loom_serve_device_host(registration->device),
        loom_serve_image_model_generator(registration->model), &service_options,
        registration->stream, registration->host_allocator,
        &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_image_service_interface(registration->service);
  }
  return status;
}

static iree_status_t loom_serve_image_registration_retire(void* self,
                                                          bool* out_complete) {
  loom_serve_image_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_image_load_free(registration->load));
    registration->load = NULL;
  }
  loom_serve_image_service_free(registration->service);
  registration->service = NULL;
  if (registration->model && !registration->retiring) {
    loom_serve_image_model_retire(registration->model,
                                  registration->observer.wake,
                                  registration->observer.user_data);
    registration->retiring = true;
  }
  if (!loom_serve_image_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_image_model_free_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void loom_serve_image_registration_destroy(void* self) {
  loom_serve_image_registration_t* registration = self;
  iree_allocator_free(registration->host_allocator, registration);
}

static loom_serve_registration_parameters_t
loom_serve_image_registration_parameters(
    void* self, loom_serve_model_component_t component) {
  loom_serve_image_registration_t* registration = self;
  return loom_serve_registration_prepared_parameters(
      loom_serve_image_model_residency(registration->model));
}

static void loom_serve_image_registration_set_pinned(
    void* self, loom_serve_model_component_t component, bool pinned) {
  loom_serve_image_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_image_model_residency(registration->model), pinned);
}

static iree_status_t loom_serve_image_registration_deactivate(
    void* self, loom_serve_model_component_t component, bool* out_complete) {
  loom_serve_image_registration_t* registration = self;
  return loom_serve_image_model_deactivate(registration->model, out_complete);
}

static iree_status_t loom_serve_image_registration_activate(
    void* self, loom_serve_model_component_t component,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  loom_serve_image_registration_t* registration = self;
  return loom_serve_registration_activate(
      &registration->activation,
      loom_serve_image_model_residency(registration->model),
      loom_serve_device_host(registration->device), registration->observer,
      flags, registration->host_allocator, out_phase);
}

static iree_status_t loom_serve_image_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_image_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*registration), (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->options = options;
  registration->device = loom_serve_device_group_at(
      options->device_group, configuration->device_index);
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_image_model_options_t model_options = {
      .sources = configuration->sources,
      .checkpoint_directory = configuration->image.checkpoint,
      .adapter_path = configuration->image.adapter,
      .options = configuration->options,
      .height = configuration->image.height,
      .width = configuration->image.width,
      .text_tokens = configuration->image.text_tokens,
      .command_quantum = configuration->image.command_quantum,
      .specialization_cache = configuration->specialization_cache};
  iree_status_t status = loom_serve_image_load_allocate(
      registration->device, &model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = loom_serve_image_registration_prepare,
        .retire = loom_serve_image_registration_retire,
        .parameters = loom_serve_image_registration_parameters,
        .set_pinned = loom_serve_image_registration_set_pinned,
        .activate = loom_serve_image_registration_activate,
        .deactivate = loom_serve_image_registration_deactivate,
        .destroy = loom_serve_image_registration_destroy};
  } else {
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

typedef struct loom_serve_speech_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_serve_device_t* device;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_wake_callback_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_speech_load_t* load;
  // Owned prepared model, retained through actual consumer retirement.
  loom_serve_speech_model_t* model;
  // Owned HTTP service, destroyed after CPU and external borrowers retire.
  loom_serve_speech_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} loom_serve_speech_registration_t;

static iree_status_t loom_serve_speech_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  loom_serve_speech_registration_t* registration = self;
  iree_status_t status = loom_serve_speech_load_advance(
      registration->load, cancel ? LOOM_SERVE_SPEECH_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_speech_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model = loom_serve_speech_load_take_model(registration->load);
    status = loom_serve_speech_load_free(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    const loom_serve_speech_service_options_t service_options = {
        .name = configuration->name,
        .pending_capacity = configuration->pending_requests,
        .heartbeat_interval = registration->options->heartbeat_interval};
    status = loom_serve_speech_service_allocate(
        loom_serve_device_host(registration->device),
        loom_serve_speech_model_generator(registration->model),
        &service_options, registration->stream, registration->host_allocator,
        &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_speech_service_interface(registration->service);
  }
  return status;
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
    loom_serve_speech_model_retire(registration->model,
                                   registration->observer.wake,
                                   registration->observer.user_data);
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
  iree_allocator_free(registration->host_allocator, registration);
}

static loom_serve_registration_parameters_t
loom_serve_speech_registration_parameters(
    void* self, loom_serve_model_component_t component) {
  loom_serve_speech_registration_t* registration = self;
  return loom_serve_registration_prepared_parameters(
      loom_serve_speech_model_residency(registration->model));
}

static void loom_serve_speech_registration_set_pinned(
    void* self, loom_serve_model_component_t component, bool pinned) {
  loom_serve_speech_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_speech_model_residency(registration->model), pinned);
}

static iree_status_t loom_serve_speech_registration_deactivate(
    void* self, loom_serve_model_component_t component, bool* out_complete) {
  loom_serve_speech_registration_t* registration = self;
  return loom_serve_residency_deactivate(
      loom_serve_speech_model_residency(registration->model), out_complete);
}

static iree_status_t loom_serve_speech_registration_activate(
    void* self, loom_serve_model_component_t component,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  loom_serve_speech_registration_t* registration = self;
  return loom_serve_registration_activate(
      &registration->activation,
      loom_serve_speech_model_residency(registration->model),
      loom_serve_device_host(registration->device), registration->observer,
      flags, registration->host_allocator, out_phase);
}

static iree_status_t loom_serve_speech_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_speech_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*registration), (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->options = options;
  registration->device = loom_serve_device_group_at(
      options->device_group, configuration->device_index);
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_speech_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options};
  iree_status_t status = loom_serve_speech_load_allocate(
      registration->device, &model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = loom_serve_speech_registration_prepare,
        .retire = loom_serve_speech_registration_retire,
        .parameters = loom_serve_speech_registration_parameters,
        .set_pinned = loom_serve_speech_registration_set_pinned,
        .activate = loom_serve_speech_registration_activate,
        .deactivate = loom_serve_speech_registration_deactivate,
        .destroy = loom_serve_speech_registration_destroy};
  } else {
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

typedef struct loom_serve_recognition_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t host_allocator;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_serve_device_t* device;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_wake_callback_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_recognition_load_t* load;
  // Owned prepared model, retained through actual consumer retirement.
  loom_serve_recognition_model_t* model;
  // Owned HTTP service, destroyed after CPU and external borrowers retire.
  loom_serve_recognition_http_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} loom_serve_recognition_registration_t;

static iree_status_t loom_serve_recognition_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  loom_serve_recognition_registration_t* registration = self;
  iree_status_t status = loom_serve_recognition_load_advance(
      registration->load, cancel ? LOOM_SERVE_RECOGNITION_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_recognition_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model =
        loom_serve_recognition_load_take_model(registration->load);
    status = loom_serve_recognition_load_free(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    const loom_serve_recognition_http_options_t service_options = {
        .name = configuration->name,
        .pending_capacity = configuration->pending_requests,
        .request_limit = 64 * 1024,
        .document_limit = 1024 * 1024,
        .decode_frames = 4096};
    status = loom_serve_recognition_http_allocate(
        loom_serve_device_host(registration->device),
        loom_serve_recognition_model_generator(registration->model),
        &service_options, registration->stream, registration->host_allocator,
        &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_recognition_http_interface(registration->service);
  }
  return status;
}

static iree_status_t loom_serve_recognition_registration_retire(
    void* self, bool* out_complete) {
  loom_serve_recognition_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_recognition_load_free(registration->load));
    registration->load = NULL;
  }
  loom_serve_recognition_http_free(registration->service);
  registration->service = NULL;
  if (registration->model && !registration->retiring) {
    loom_serve_recognition_model_retire(registration->model,
                                        registration->observer.wake,
                                        registration->observer.user_data);
    registration->retiring = true;
  }
  if (!loom_serve_recognition_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_recognition_model_free_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void loom_serve_recognition_registration_destroy(void* self) {
  loom_serve_recognition_registration_t* registration = self;
  iree_allocator_free(registration->host_allocator, registration);
}

static loom_serve_registration_parameters_t
loom_serve_recognition_registration_parameters(
    void* self, loom_serve_model_component_t component) {
  loom_serve_recognition_registration_t* registration = self;
  return loom_serve_registration_prepared_parameters(
      loom_serve_recognition_model_residency(registration->model));
}

static void loom_serve_recognition_registration_set_pinned(
    void* self, loom_serve_model_component_t component, bool pinned) {
  loom_serve_recognition_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_recognition_model_residency(registration->model), pinned);
}

static iree_status_t loom_serve_recognition_registration_deactivate(
    void* self, loom_serve_model_component_t component, bool* out_complete) {
  loom_serve_recognition_registration_t* registration = self;
  return loom_serve_residency_deactivate(
      loom_serve_recognition_model_residency(registration->model),
      out_complete);
}

static iree_status_t loom_serve_recognition_registration_activate(
    void* self, loom_serve_model_component_t component,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  loom_serve_recognition_registration_t* registration = self;
  return loom_serve_registration_activate(
      &registration->activation,
      loom_serve_recognition_model_residency(registration->model),
      loom_serve_device_host(registration->device), registration->observer,
      flags, registration->host_allocator, out_phase);
}

static iree_status_t loom_serve_recognition_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_recognition_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*registration), (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->device = loom_serve_device_group_at(
      options->device_group, configuration->device_index);
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_recognition_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options};
  iree_status_t status = loom_serve_recognition_load_allocate(
      registration->device, &model_options, observer.wake, observer.user_data,
      host_allocator, &registration->load);
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = loom_serve_recognition_registration_prepare,
        .retire = loom_serve_recognition_registration_retire,
        .parameters = loom_serve_recognition_registration_parameters,
        .set_pinned = loom_serve_recognition_registration_set_pinned,
        .activate = loom_serve_recognition_registration_activate,
        .deactivate = loom_serve_recognition_registration_deactivate,
        .destroy = loom_serve_recognition_registration_destroy};
  } else {
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

typedef struct loom_serve_text_registration_t {
  // Allocator owning this operation and its stable shape table.
  iree_allocator_t host_allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_serve_device_t* device;
  // Immutable declaration retained by the stable registry slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback; no destruction occurs on its calling thread.
  loom_serve_wake_callback_t observer;
  // Registration-wide history retained beyond service retirement.
  loom_serve_telemetry_stream_t* stream;
  // Shape inputs borrowed by asynchronous preparation until publication.
  loom_serve_packing_shape_t shapes[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY];
  // Owned preparation operation, transferred on every terminal outcome.
  loom_serve_text_load_t* load;
  // Partial or prepared model retained through actual consumer retirement.
  loom_serve_text_model_t* model;
  // Owned dormant/on-demand image model, independent of text residency.
  loom_serve_finite_component_t* image;
  // Service borrowing model state until all protocol claims are resolved.
  loom_serve_text_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have relinquished views with an ownership wake installed.
  bool retiring;
} loom_serve_text_registration_t;

static iree_status_t loom_serve_text_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  loom_serve_text_registration_t* registration = self;
  iree_status_t status = loom_serve_text_load_advance(
      registration->load, cancel ? LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_text_load_report(registration->load);
  registration->model = loom_serve_text_load_take_model(registration->load);
  loom_serve_text_load_free(registration->load);
  registration->load = NULL;
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    const loom_serve_text_configuration_t* text = &configuration->text;
    const loom_serve_text_service_options_t options = {
        .name = configuration->name,
        .image = registration->image,
        .row_count = text->rows,
        .variant_capacity = configuration->variant_capacity,
        .chunk_size = text->prefill_capacity,
        .default_max_tokens = text->max_tokens,
        .pending_capacity = configuration->pending_requests,
        .session_cache_capacity = text->session_cache_capacity,
        .prefix_cache_capacity = text->prefix_cache_capacity,
        .admission_overtakes = text->admission_overtakes,
        .heartbeat_interval = registration->options->heartbeat_interval,
        .schedule_mode = LOOM_SERVE_TEXT_SCHEDULE_PACKED,
        .packing_mode = LOOM_SERVE_TEXT_PACKING_MIXED,
        .continuation_epochs = text->continuation_epochs};
    status = loom_serve_text_service_allocate(
        registration->model, &options, registration->stream,
        registration->host_allocator, &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_text_service_interface(registration->service);
  }
  return status;
}

static iree_status_t loom_serve_text_registration_retire(void* self,
                                                         bool* out_complete) {
  loom_serve_text_registration_t* registration = self;
  *out_complete = true;
  loom_serve_text_service_free(registration->service, IREE_STATUS_OK);
  registration->service = NULL;
  // Join one retirement domain before arming the other so a terminal cleanup
  // error leaves no outstanding callback on the retained registration.
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_component_retire(registration->image, out_complete));
  if (!*out_complete) {
    return iree_ok_status();
  }
  loom_serve_finite_component_free(registration->image);
  registration->image = NULL;
  if (registration->model && !registration->retiring) {
    loom_serve_text_model_retire(registration->model,
                                 registration->observer.wake,
                                 registration->observer.user_data);
    registration->retiring = true;
  }
  if (!loom_serve_text_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_free_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void loom_serve_text_registration_destroy(void* self) {
  loom_serve_text_registration_t* registration = self;
  iree_allocator_free(registration->host_allocator, registration);
}

static loom_serve_registration_parameters_t
loom_serve_text_registration_parameters(
    void* self, loom_serve_model_component_t component) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVE_MODEL_COMPONENT_IMAGE) {
    static const char* const phases[] = {"declared", "preparing", "ready",
                                         "failed",   "releasing", "retired"};
    loom_serve_registration_parameters_t state = {
        .phase = phases[loom_serve_finite_component_phase(registration->image)],
        .deactivatable = true,
        .status = loom_serve_finite_component_status(registration->image),
        .compilation = loom_serve_finite_component_report(registration->image)};
    state.prepared = loom_serve_finite_component_parameters(registration->image,
                                                            &state.parameters);
    return state;
  }
  return loom_serve_registration_prepared_parameters(
      loom_serve_text_model_residency(registration->model));
}

static void loom_serve_text_registration_set_pinned(
    void* self, loom_serve_model_component_t component, bool pinned) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVE_MODEL_COMPONENT_IMAGE) {
    loom_serve_finite_component_set_pinned(registration->image, pinned);
  } else {
    loom_serve_residency_set_pinned(
        loom_serve_text_model_residency(registration->model), pinned);
  }
}

static iree_status_t loom_serve_text_registration_deactivate(
    void* self, loom_serve_model_component_t component, bool* out_complete) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVE_MODEL_COMPONENT_IMAGE) {
    *out_complete = false;
    if (loom_serve_finite_component_is_pending(registration->image)) {
      return iree_ok_status();
    }
    loom_serve_text_service_release_projector(registration->service);
    return loom_serve_finite_component_trim(registration->image, out_complete);
  }
  return loom_serve_text_model_deactivate(registration->model, out_complete);
}

static iree_status_t loom_serve_text_registration_activate(
    void* self, loom_serve_model_component_t component,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  loom_serve_text_registration_t* registration = self;
  *out_component_status = iree_ok_status();
  *out_phase = LOOM_SERVE_ACTIVATION_COMPLETE;
  loom_serve_residency_t* residency = NULL;
  if (component == LOOM_SERVE_MODEL_COMPONENT_IMAGE &&
      !registration->activation) {
    if (iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL)) {
      return iree_ok_status();
    }
    loom_serve_finite_model_t* model = NULL;
    *out_component_status =
        loom_serve_finite_component_prepare(registration->image, &model);
    if (!iree_status_is_ok(*out_component_status)) {
      return iree_ok_status();
    }
    if (!model) {
      *out_phase = LOOM_SERVE_ACTIVATION_WAITING;
      return iree_ok_status();
    }
    residency = loom_serve_finite_model_residency(model);
  } else if (component == LOOM_SERVE_MODEL_COMPONENT_PRIMARY) {
    residency = loom_serve_text_model_residency(registration->model);
  }
  // An accepted image activation already owns its residency borrow. Joining
  // cancellation must not reenter component preparation during removal.
  return loom_serve_registration_activate(
      &registration->activation, residency,
      loom_serve_device_host(registration->device), registration->observer,
      flags, registration->host_allocator, out_phase);
}

static iree_status_t loom_serve_text_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_text_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*registration), (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->options = options;
  registration->device = loom_serve_device_group_at(
      options->device_group, configuration->device_index);
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_text_configuration_t* text = &configuration->text;
  const iree_host_size_t shape_count = loom_serve_text_default_shapes(
      text->rows, text->prefill_capacity, registration->shapes);
  const loom_serve_text_options_t model_options = {
      .sources = configuration->sources,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      .pool_capacity = text->pool_capacity,
      .epoch_count = shape_count,
      .epoch_shapes = registration->shapes,
      .draft_depth = text->draft_depth,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = text->rows,
      .checkpoint_capacity =
          text->checkpoint_capacity + text->prefix_cache_capacity +
          (text->session_cache_capacity ? text->session_cache_capacity + 1 : 0),
      .options = configuration->options};
  iree_status_t status = iree_ok_status();
  if (text->image.prepare.size) {
    loom_serve_finite_model_options_t image_options = {
        .sources = configuration->sources,
        .asset_directory = text->image.assets,
        .options = text->image.options,
        .specialization_cache = text->image.specialization_cache};
    image_options.sources.prepare = text->image.prepare;
    image_options.sources.control = text->image.control;
    status = loom_serve_finite_component_allocate(
        registration->device, &image_options, observer, host_allocator,
        &registration->image);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_load_allocate(registration->device, &model_options,
                                           observer.wake, observer.user_data,
                                           host_allocator, &registration->load);
  }
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = loom_serve_text_registration_prepare,
        .retire = loom_serve_text_registration_retire,
        .parameters = loom_serve_text_registration_parameters,
        .set_pinned = loom_serve_text_registration_set_pinned,
        .activate = loom_serve_text_registration_activate,
        .deactivate = loom_serve_text_registration_deactivate,
        .destroy = loom_serve_text_registration_destroy};
  } else {
    loom_serve_finite_component_free(registration->image);
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

static iree_status_t loom_serve_model_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    return loom_serve_text_registration_create(self, configuration, observer,
                                               stream, host_allocator,
                                               out_registration);
  }
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    return loom_serve_image_registration_create(self, configuration, observer,
                                                stream, host_allocator,
                                                out_registration);
  }
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_SPEECH) {
    return loom_serve_speech_registration_create(self, configuration, observer,
                                                 stream, host_allocator,
                                                 out_registration);
  }
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_RECOGNITION) {
    return loom_serve_recognition_registration_create(
        self, configuration, observer, stream, host_allocator,
        out_registration);
  }
  return loom_serve_finite_registration_create(
      self, configuration, observer, stream, host_allocator, out_registration);
}

static iree_status_t loom_serve_model_registration_query_status(void* self) {
  const loom_serve_model_registration_options_t* options = self;
  const iree_host_size_t device_count =
      loom_serve_device_group_count(options->device_group);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    loom_serve_memory_pool_t* pool = loom_serve_device_memory_pool(
        loom_serve_device_group_at(options->device_group, i));
    if (pool) {
      IREE_RETURN_IF_ERROR(loom_serve_memory_pool_query_status(pool));
    }
  }
  return iree_ok_status();
}

loom_serve_registration_factory_t loom_serve_model_registration_factory(
    const loom_serve_model_registration_options_t* options) {
  return (loom_serve_registration_factory_t){
      .self = (void*)options,
      .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
               (1u << LOOM_SERVE_MODEL_KIND_DECISION) |
               (1u << LOOM_SERVE_MODEL_KIND_IMAGE) |
               (1u << LOOM_SERVE_MODEL_KIND_SPEECH) |
               (1u << LOOM_SERVE_MODEL_KIND_RECOGNITION) |
               (1u << LOOM_SERVE_MODEL_KIND_TEXT),
      .device_count = loom_serve_device_group_count(options->device_group),
      .create = loom_serve_model_registration_create,
      .query_status = loom_serve_model_registration_query_status};
}
