// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/model_registration.h"

#include "experimental/loom_serve/decision/protocol.h"
#include "experimental/loom_serve/embedding/protocol.h"
#include "experimental/loom_serve/finite/model.h"
#include "experimental/loom_serve/finite/service.h"
#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/service.h"

typedef struct finite_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_http_observer_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_finite_load_t* load;
  // Owned prepared model, kept through actual consumer retirement.
  loom_serve_finite_model_t* model;
  // Owned HTTP service, destroyed only after external borrowers retire.
  loom_serve_finite_service_t* service;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} finite_registration_t;

static iree_status_t finite_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  finite_registration_t* registration = self;
  iree_status_t status = loom_serve_finite_load_advance(
      registration->load, cancel ? LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_finite_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model = loom_serve_finite_load_take_model(registration->load);
    status = loom_serve_finite_load_destroy(registration->load);
    if (iree_status_is_ok(status)) {
      registration->load = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    status = loom_serve_finite_service_create(
        loom_serve_device_host(registration->options->device),
        registration->model,
        configuration->kind == LOOM_SERVE_MODEL_KIND_EMBEDDING
            ? loom_serve_embedding_protocol()
            : loom_serve_decision_protocol(),
        configuration->name, configuration->pending_requests,
        registration->options->heartbeat_interval, registration->stream,
        &registration->service, registration->allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_finite_service_interface(registration->service);
  }
  return status;
}

static iree_status_t finite_registration_retire(void* self,
                                                bool* out_complete) {
  finite_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_finite_load_destroy(registration->load));
    registration->load = NULL;
  }
  loom_serve_finite_service_destroy(registration->service);
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
      loom_serve_finite_model_destroy_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void finite_registration_destroy(void* self) {
  finite_registration_t* registration = self;
  iree_allocator_free(registration->allocator, registration);
}

static iree_status_t finite_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_http_observer_t observer, loom_serve_telemetry_stream_t* stream,
    loom_serve_registration_t* out_registration, iree_allocator_t allocator) {
  const loom_serve_model_registration_options_t* options = self;
  finite_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*registration),
                                             (void**)&registration));
  registration->allocator = allocator;
  registration->options = options;
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_finite_model_options_t model_options = {
      .source_directory = configuration->source,
      .asset_directory = configuration->finite.assets,
      .options = configuration->finite.options};
  iree_status_t status = loom_serve_finite_load_create(
      options->device, &model_options, observer.wake, observer.user_data,
      &registration->load, allocator);
  if (iree_status_is_ok(status)) {
    *out_registration =
        (loom_serve_registration_t){.self = registration,
                                    .prepare = finite_registration_prepare,
                                    .retire = finite_registration_retire,
                                    .destroy = finite_registration_destroy};
  } else {
    iree_allocator_free(allocator, registration);
  }
  return status;
}

typedef struct image_registration_t {
  // Allocator owning this operation, independent of model allocations.
  iree_allocator_t allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Borrowed immutable declaration owned by the stable registration slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback, never an inline model operation.
  loom_serve_http_observer_t observer;
  // Borrowed registration-wide history, published only by this shared owner.
  loom_serve_telemetry_stream_t* stream;
  // Owned unstarted/live/terminal source load, consumed before publication.
  loom_serve_image_load_t* load;
  // Owned prepared model, retained through actual consumer retirement.
  loom_serve_image_model_t* model;
  // Owned HTTP service, destroyed after CPU and external borrowers retire.
  loom_serve_image_service_t* service;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
} image_registration_t;

static iree_status_t image_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  image_registration_t* registration = self;
  iree_status_t status = loom_serve_image_load_advance(
      registration->load, cancel ? LOOM_SERVE_IMAGE_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_image_load_report(registration->load);
  if (iree_status_is_ok(status)) {
    registration->model = loom_serve_image_load_take_model(registration->load);
    status = loom_serve_image_load_destroy(registration->load);
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
        .heartbeat_interval = registration->options->heartbeat_interval};
    status = loom_serve_image_service_create(
        loom_serve_device_host(registration->options->device),
        loom_serve_image_model_generator(registration->model), &service_options,
        registration->stream, &registration->service, registration->allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_image_service_interface(registration->service);
  }
  return status;
}

static iree_status_t image_registration_retire(void* self, bool* out_complete) {
  image_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_serve_image_load_destroy(registration->load));
    registration->load = NULL;
  }
  loom_serve_image_service_destroy(registration->service);
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
      loom_serve_image_model_destroy_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void image_registration_destroy(void* self) {
  image_registration_t* registration = self;
  iree_allocator_free(registration->allocator, registration);
}

static iree_status_t image_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_http_observer_t observer, loom_serve_telemetry_stream_t* stream,
    loom_serve_registration_t* out_registration, iree_allocator_t allocator) {
  const loom_serve_model_registration_options_t* options = self;
  image_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*registration),
                                             (void**)&registration));
  registration->allocator = allocator;
  registration->options = options;
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_image_model_options_t model_options = {
      .source_directory = configuration->source,
      .checkpoint_directory = configuration->image.checkpoint,
      .adapter_path = configuration->image.adapter,
      .height = configuration->image.height,
      .width = configuration->image.width,
      .text_tokens = configuration->image.text_tokens};
  iree_status_t status = loom_serve_image_load_create(
      options->device, &model_options, observer.wake, observer.user_data,
      &registration->load, allocator);
  if (iree_status_is_ok(status)) {
    *out_registration =
        (loom_serve_registration_t){.self = registration,
                                    .prepare = image_registration_prepare,
                                    .retire = image_registration_retire,
                                    .destroy = image_registration_destroy};
  } else {
    iree_allocator_free(allocator, registration);
  }
  return status;
}

static iree_status_t model_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_http_observer_t observer, loom_serve_telemetry_stream_t* stream,
    loom_serve_registration_t* out_registration, iree_allocator_t allocator) {
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    return image_registration_create(self, configuration, observer, stream,
                                     out_registration, allocator);
  }
  return finite_registration_create(self, configuration, observer, stream,
                                    out_registration, allocator);
}

loom_serve_registration_factory_t loom_serve_model_registration_factory(
    const loom_serve_model_registration_options_t* options) {
  return (loom_serve_registration_factory_t){
      .self = (void*)options,
      .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
               (1u << LOOM_SERVE_MODEL_KIND_DECISION) |
               (1u << LOOM_SERVE_MODEL_KIND_IMAGE),
      .create = model_registration_create};
}
