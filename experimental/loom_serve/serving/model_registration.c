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
#include "experimental/loom_serve/storage/memory.h"
#include "experimental/loom_serve/text/schedule.h"
#include "experimental/loom_serve/text/service.h"

// The adapter owns the native parameter group and its temporary readiness
// operation. Neither the registry nor its management connection borrows either.
static iree_status_t registration_activate(
    loom_serve_activation_t** operation, loom_serve_residency_t* residency,
    loom_serve_host_t* host, loom_serve_http_observer_t observer,
    loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase, iree_allocator_t allocator) {
  *out_phase = LOOM_SERVE_ACTIVATION_COMPLETE;
  if (!*operation) {
    if (iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_serve_activation_create(
        residency, loom_serve_host_proactor_pool(host), observer.wake,
        observer.user_data, operation, allocator));
  }
  iree_status_t status =
      loom_serve_activation_advance(*operation, flags, out_phase);
  if (*out_phase == LOOM_SERVE_ACTIVATION_COMPLETE) {
    loom_serve_activation_destroy(*operation);
    *operation = NULL;
  }
  return status;
}

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
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
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

static loom_serve_residency_snapshot_t finite_registration_parameters(
    void* self) {
  finite_registration_t* registration = self;
  return loom_serve_residency_snapshot(
      loom_serve_finite_model_residency(registration->model));
}

static void finite_registration_set_pinned(void* self, bool pinned) {
  finite_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_finite_model_residency(registration->model), pinned);
}

static iree_status_t finite_registration_deactivate(void* self) {
  finite_registration_t* registration = self;
  return loom_serve_residency_deactivate(
      loom_serve_finite_model_residency(registration->model));
}

static iree_status_t finite_registration_activate(
    void* self, loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase) {
  finite_registration_t* registration = self;
  return registration_activate(
      &registration->activation,
      loom_serve_finite_model_residency(registration->model),
      loom_serve_device_host(registration->options->device),
      registration->observer, flags, out_phase, registration->allocator);
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
    *out_registration = (loom_serve_registration_t){
        .self = registration,
        .prepare = finite_registration_prepare,
        .retire = finite_registration_retire,
        .parameters = finite_registration_parameters,
        .set_pinned = finite_registration_set_pinned,
        .activate = finite_registration_activate,
        .deactivate = finite_registration_deactivate,
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
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
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

static loom_serve_residency_snapshot_t image_registration_parameters(
    void* self) {
  image_registration_t* registration = self;
  return loom_serve_residency_snapshot(
      loom_serve_image_model_residency(registration->model));
}

static void image_registration_set_pinned(void* self, bool pinned) {
  image_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_image_model_residency(registration->model), pinned);
}

static iree_status_t image_registration_deactivate(void* self) {
  image_registration_t* registration = self;
  return loom_serve_image_model_deactivate(registration->model);
}

static iree_status_t image_registration_activate(
    void* self, loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase) {
  image_registration_t* registration = self;
  return registration_activate(
      &registration->activation,
      loom_serve_image_model_residency(registration->model),
      loom_serve_device_host(registration->options->device),
      registration->observer, flags, out_phase, registration->allocator);
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
                                    .parameters = image_registration_parameters,
                                    .set_pinned = image_registration_set_pinned,
                                    .activate = image_registration_activate,
                                    .deactivate = image_registration_deactivate,
                                    .destroy = image_registration_destroy};
  } else {
    iree_allocator_free(allocator, registration);
  }
  return status;
}

typedef struct text_registration_t {
  // Allocator owning this operation and its stable shape table.
  iree_allocator_t allocator;
  // Borrowed server capabilities and reporting policy.
  const loom_serve_model_registration_options_t* options;
  // Immutable declaration retained by the stable registry slot.
  const loom_serve_model_configuration_t* configuration;
  // Owner scheduling callback; no destruction occurs on its calling thread.
  loom_serve_http_observer_t observer;
  // Registration-wide history retained beyond service retirement.
  loom_serve_telemetry_stream_t* stream;
  // Shape inputs borrowed by asynchronous preparation until publication.
  loom_serve_packing_shape_t shapes[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY];
  // Owned preparation operation, transferred on every terminal outcome.
  loom_serve_text_load_t* load;
  // Partial or prepared model retained through actual consumer retirement.
  loom_serve_text_model_t* model;
  // Service borrowing model state until all protocol claims are resolved.
  loom_serve_text_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_serve_activation_t* activation;
  // Model consumers have relinquished views with an ownership wake installed.
  bool retiring;
} text_registration_t;

static iree_status_t text_registration_prepare(
    void* self, bool cancel, bool* out_complete,
    loom_serve_http_service_t* out_service,
    loom_serve_compilation_report_t* out_report) {
  text_registration_t* registration = self;
  iree_status_t status = loom_serve_text_load_advance(
      registration->load, cancel ? LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL : 0,
      out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_text_load_report(registration->load);
  registration->model = loom_serve_text_load_take_model(registration->load);
  loom_serve_text_load_destroy(registration->load);
  registration->load = NULL;
  if (iree_status_is_ok(status)) {
    const loom_serve_model_configuration_t* configuration =
        registration->configuration;
    const loom_serve_text_configuration_t* text = &configuration->text;
    const loom_serve_text_service_options_t options = {
        .name = configuration->name,
        .row_count = text->rows,
        .chunk_size = text->prefill_capacity,
        .default_max_tokens = text->max_tokens,
        .pending_capacity = configuration->pending_requests,
        .session_cache_capacity = text->session_cache_capacity,
        .admission_overtakes = text->admission_overtakes,
        .heartbeat_interval = registration->options->heartbeat_interval,
        .schedule_mode = LOOM_SERVE_TEXT_SCHEDULE_PACKED,
        .packing_mode = LOOM_SERVE_TEXT_PACKING_MIXED,
        .mtp_depth = text->mtp_depth,
        .continuation_epochs = text->continuation_epochs};
    status = loom_serve_text_service_create(
        registration->model, &options, registration->stream,
        &registration->service, registration->allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_service = loom_serve_text_service_interface(registration->service);
  }
  return status;
}

static iree_status_t text_registration_retire(void* self, bool* out_complete) {
  text_registration_t* registration = self;
  *out_complete = true;
  loom_serve_text_service_destroy(registration->service, IREE_STATUS_OK);
  registration->service = NULL;
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
  IREE_RETURN_IF_ERROR(
      loom_serve_text_model_destroy_retired(registration->model));
  registration->model = NULL;
  return iree_ok_status();
}

static void text_registration_destroy(void* self) {
  text_registration_t* registration = self;
  iree_allocator_free(registration->allocator, registration);
}

static loom_serve_residency_snapshot_t text_registration_parameters(
    void* self) {
  text_registration_t* registration = self;
  return loom_serve_residency_snapshot(
      loom_serve_text_model_residency(registration->model));
}

static void text_registration_set_pinned(void* self, bool pinned) {
  text_registration_t* registration = self;
  loom_serve_residency_set_pinned(
      loom_serve_text_model_residency(registration->model), pinned);
}

static iree_status_t text_registration_deactivate(void* self) {
  text_registration_t* registration = self;
  return loom_serve_text_model_deactivate(registration->model);
}

static iree_status_t text_registration_activate(
    void* self, loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase) {
  text_registration_t* registration = self;
  return registration_activate(
      &registration->activation,
      loom_serve_text_model_residency(registration->model),
      loom_serve_device_host(registration->options->device),
      registration->observer, flags, out_phase, registration->allocator);
}

static iree_status_t text_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_http_observer_t observer, loom_serve_telemetry_stream_t* stream,
    loom_serve_registration_t* out_registration, iree_allocator_t allocator) {
  const loom_serve_model_registration_options_t* options = self;
  text_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*registration),
                                             (void**)&registration));
  registration->allocator = allocator;
  registration->options = options;
  registration->configuration = configuration;
  registration->observer = observer;
  registration->stream = stream;
  const loom_serve_text_configuration_t* text = &configuration->text;
  const iree_host_size_t shape_count = loom_serve_text_default_shapes(
      text->rows, text->prefill_capacity, registration->shapes);
  const loom_serve_text_options_t model_options = {
      .source_directory = configuration->source,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      .pool_capacity = text->pool_capacity,
      .epoch_count = shape_count,
      .epoch_shapes = registration->shapes,
      .enable_mtp = text->mtp_depth != 0,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = text->rows,
      .checkpoint_capacity =
          text->checkpoint_capacity + (text->session_cache_capacity
                                           ? text->session_cache_capacity + 1
                                           : 0)};
  iree_status_t status = loom_serve_text_load_create(
      options->device, &model_options, observer.wake, observer.user_data,
      &registration->load, allocator);
  if (iree_status_is_ok(status)) {
    *out_registration =
        (loom_serve_registration_t){.self = registration,
                                    .prepare = text_registration_prepare,
                                    .retire = text_registration_retire,
                                    .parameters = text_registration_parameters,
                                    .set_pinned = text_registration_set_pinned,
                                    .activate = text_registration_activate,
                                    .deactivate = text_registration_deactivate,
                                    .destroy = text_registration_destroy};
  } else {
    iree_allocator_free(allocator, registration);
  }
  return status;
}

static iree_status_t model_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    loom_serve_http_observer_t observer, loom_serve_telemetry_stream_t* stream,
    loom_serve_registration_t* out_registration, iree_allocator_t allocator) {
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    return text_registration_create(self, configuration, observer, stream,
                                    out_registration, allocator);
  }
  if (configuration->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    return image_registration_create(self, configuration, observer, stream,
                                     out_registration, allocator);
  }
  return finite_registration_create(self, configuration, observer, stream,
                                    out_registration, allocator);
}

static iree_status_t model_registration_query_status(void* self) {
  const loom_serve_model_registration_options_t* options = self;
  loom_serve_memory_pool_t* pool =
      loom_serve_device_memory_pool(options->device);
  return pool ? loom_serve_memory_pool_query_status(pool) : iree_ok_status();
}

loom_serve_registration_factory_t loom_serve_model_registration_factory(
    const loom_serve_model_registration_options_t* options) {
  return (loom_serve_registration_factory_t){
      .self = (void*)options,
      .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
               (1u << LOOM_SERVE_MODEL_KIND_DECISION) |
               (1u << LOOM_SERVE_MODEL_KIND_IMAGE) |
               (1u << LOOM_SERVE_MODEL_KIND_TEXT),
      .create = model_registration_create,
      .query_status = model_registration_query_status};
}
