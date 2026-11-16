// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/registration.h"

#include <string.h>

#include "iree/hal/api.h"
#include "loom/service/finite/component.h"

typedef struct loom_serve_text_registration_t {
  // Shared native lifecycle, retained until both model domains retire.
  loom_service_registration_t registration;
  // Allocator owning this provider, copied shapes, and retained control.
  iree_allocator_t host_allocator;
  // Borrowed device through actual text and image retirement.
  loom_runtime_device_t* device;
  // Copied ready-engine options; views retain their caller-owned lifetime.
  loom_serve_text_service_options_t options;
  // Borrowed registration history, independent of engine lifetime.
  loom_runtime_telemetry_stream_t* stream;
  // Owned source preparation, transferred on every terminal result.
  loom_serve_text_load_t* load;
  // Partial or prepared primary model, joined by actual ownership.
  loom_serve_text_model_t* model;
  // Independent optional image model, allocated without starting preparation.
  loom_service_finite_component_t* image;
  // Ready engine, relinquished after external views and callbacks join.
  loom_serve_text_service_t* service;
  // Explicit parameter warmup, joined before model retirement.
  loom_runtime_activation_t* activation;
  // Primary model retirement was submitted with the owner's wake.
  bool retiring;
  // Latest aggregate engine contribution while its source is alive.
  loom_runtime_service_observation_t text_observation;
  // Independent component preparation and release contribution.
  loom_runtime_service_observation_t image_observation;
  // Split accepted source facts transferred before engine destruction.
  loom_runtime_service_observation_t primary_retirement;
} loom_serve_text_registration_t;

static void loom_serve_text_registration_publish(
    loom_serve_text_registration_t* registration) {
  loom_runtime_service_observation_t observation =
      registration->text_observation;
  observation.offer_count += registration->image_observation.offer_count;
  observation.pending_operation_count +=
      registration->image_observation.pending_operation_count;
  observation.offered_devices |=
      registration->image_observation.offered_devices;
  observation.accepted_devices |=
      registration->image_observation.accepted_devices;
  const loom_runtime_service_control_binding_t binding =
      loom_service_registration_binding(&registration->registration, 0, 0, 0);
  binding.publish_observation(binding.publisher, &observation);
}

static void loom_serve_text_registration_publish_text(
    void* self, const loom_runtime_service_observation_t* observation) {
  loom_serve_text_registration_t* registration = self;
  registration->text_observation = *observation;
  loom_serve_text_registration_publish(registration);
}

static void loom_serve_text_registration_publish_image(
    void* self, const loom_runtime_service_observation_t* observation) {
  loom_serve_text_registration_t* registration = self;
  registration->image_observation = *observation;
  loom_serve_text_registration_publish(registration);
}

static iree_status_t loom_serve_text_registration_prepare(
    void* self, loom_service_registration_prepare_flags_t flags,
    loom_service_preparation_result_t* out_result,
    loom_runtime_compilation_report_t* out_report) {
  loom_serve_text_registration_t* registration = self;
  loom_serve_text_load_flags_t load_flags = 0;
  if (iree_any_bit_set(flags, LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL)) {
    load_flags |= LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL;
  }
  if (iree_any_bit_set(flags, LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_DEMAND)) {
    load_flags |= LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND;
  }
  bool complete = false;
  iree_status_t status =
      loom_serve_text_load_advance(registration->load, load_flags, &complete);
  if (!complete) {
    return status;
  }
  *out_report = *loom_serve_text_load_report(registration->load);
  registration->model = loom_serve_text_load_take_model(registration->load);
  loom_serve_text_load_free(registration->load);
  registration->load = NULL;
  if (iree_status_is_ok(status)) {
    const loom_serve_text_cache_selection_t cache =
        loom_serve_text_model_cache_selection(registration->model);
    loom_serve_text_service_options_t options = registration->options;
    options.session_cache_capacity = cache.session_capacity;
    options.prefix_cache_capacity = cache.prefix_capacity;
    options.image = registration->image;
    options.scheduling =
        loom_service_registration_binding(&registration->registration, 0, 0, 0);
    if (registration->image) {
      options.scheduling.publish_observation =
          loom_serve_text_registration_publish_text;
      options.scheduling.publisher = registration;
    }
    status = loom_serve_text_service_allocate(
        registration->model, &options, registration->stream,
        registration->host_allocator, &registration->service);
  }
  out_result->flags =
      LOOM_SERVICE_PREPARATION_RESULT_COMPLETE |
      (registration->model ? LOOM_SERVICE_PREPARATION_RESULT_RETAINED : 0);
  if (iree_status_is_ok(status)) {
    out_result->service =
        loom_serve_text_service_interface(registration->service).runtime;
  }
  return status;
}

static iree_status_t loom_serve_text_registration_progress(
    void* self, loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_serve_text_registration_t* registration = self;
  return loom_serve_text_service_progress(registration->service, flags,
                                          out_progress);
}

static iree_status_t loom_serve_text_registration_retire(void* self,
                                                         bool* out_complete) {
  loom_serve_text_registration_t* registration = self;
  *out_complete = true;
  if (registration->load) {
    // Only an unstarted load reaches retirement without terminal preparation.
    // Cancellation transfers its partial model without reading source inputs.
    iree_status_t status = loom_serve_text_load_advance(
        registration->load, LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL, out_complete);
    if (!*out_complete) {
      return status;
    }
    registration->model = loom_serve_text_load_take_model(registration->load);
    loom_serve_text_load_free(registration->load);
    registration->load = NULL;
    if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
      return status;
    }
    iree_status_free(status);
  }
  if (registration->service) {
    registration->primary_retirement =
        loom_serve_text_service_free(registration->service, IREE_STATUS_OK)
            .primary;
    registration->service = NULL;
  }
  // Join one domain before arming the other. A failed platform cleanup then
  // retains quiescent ownership without an outstanding peer callback.
  IREE_RETURN_IF_ERROR(
      loom_service_finite_component_retire(registration->image, out_complete));
  if (!*out_complete) {
    return iree_ok_status();
  }
  if (registration->image) {
    loom_service_finite_component_free(registration->image);
    registration->image = NULL;
    const loom_runtime_service_operation_t projection = {
        .key.ordinal = 3 + registration->options.variant_capacity};
    loom_runtime_service_publish_operation(registration->registration.control,
                                           &projection, NULL);
    registration->text_observation = registration->primary_retirement;
    loom_serve_text_registration_publish(registration);
  }
  if (registration->model && !registration->retiring) {
    loom_serve_text_model_retire(registration->model,
                                 registration->registration.observer.wake,
                                 registration->registration.observer.user_data);
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
  iree_allocator_free_aligned(registration->host_allocator, registration);
}

static loom_service_registration_parameters_t
loom_serve_text_registration_parameters(
    void* self, loom_service_model_component_t component) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVICE_MODEL_COMPONENT_IMAGE) {
    static const char* const phases[] = {"declared", "preparing", "ready",
                                         "failed",   "releasing", "retired"};
    loom_service_registration_parameters_t state = {
        .phase =
            phases[loom_service_finite_component_phase(registration->image)],
        .deactivatable = true,
        .status = loom_service_finite_component_status(registration->image),
        .compilation =
            loom_service_finite_component_report(registration->image),
        .preparation_duration_ns =
            loom_service_finite_component_preparation_duration_ns(
                registration->image)};
    state.prepared = loom_service_finite_component_parameters(
        registration->image, &state.parameters);
    return state;
  }
  const loom_runtime_residency_snapshot_t parameters =
      loom_runtime_residency_snapshot(
          loom_serve_text_model_residency(registration->model));
  return (loom_service_registration_parameters_t){
      .phase = "ready",
      .prepared = true,
      .deactivatable = parameters.elastic,
      .parameters = parameters};
}

static void loom_serve_text_registration_set_pinned(
    void* self, loom_service_model_component_t component, bool pinned) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVICE_MODEL_COMPONENT_IMAGE) {
    loom_service_finite_component_set_pinned(registration->image, pinned);
  } else {
    loom_runtime_residency_set_pinned(
        loom_serve_text_model_residency(registration->model), pinned);
  }
}

static iree_status_t loom_serve_text_registration_deactivate(
    void* self, loom_service_model_component_t component, bool* out_complete) {
  loom_serve_text_registration_t* registration = self;
  if (component == LOOM_SERVICE_MODEL_COMPONENT_IMAGE) {
    return loom_serve_text_service_trim_image(registration->service,
                                              out_complete);
  }
  return loom_serve_text_model_deactivate(registration->model, out_complete);
}

static iree_status_t loom_serve_text_registration_activate(
    void* self, loom_service_model_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  loom_serve_text_registration_t* registration = self;
  *out_component_status = iree_ok_status();
  *out_phase = LOOM_RUNTIME_ACTIVATION_COMPLETE;
  loom_runtime_residency_t* residency = NULL;
  if (component == LOOM_SERVICE_MODEL_COMPONENT_IMAGE &&
      !registration->activation) {
    if (iree_any_bit_set(flags, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL)) {
      return iree_ok_status();
    }
    loom_service_finite_model_t* model = NULL;
    *out_component_status =
        loom_service_finite_component_prepare(registration->image, &model);
    if (!iree_status_is_ok(*out_component_status)) {
      return iree_ok_status();
    }
    if (!model) {
      *out_phase = LOOM_RUNTIME_ACTIVATION_WAITING;
      return iree_ok_status();
    }
    residency = loom_service_finite_model_residency(model);
  } else if (component == LOOM_SERVICE_MODEL_COMPONENT_PRIMARY) {
    residency = loom_serve_text_model_residency(registration->model);
  }
  // An accepted image activation already owns its residency borrow. Joining
  // cancellation must not reenter component preparation during removal.
  return loom_runtime_activation_poll(
      &registration->activation, residency,
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(registration->device)),
      registration->registration.observer.wake,
      registration->registration.observer.user_data, flags,
      registration->host_allocator, out_phase);
}

static const loom_service_registration_vtable_t
    loom_serve_text_registration_vtable = {
        .prepare = loom_serve_text_registration_prepare,
        .progress = loom_serve_text_registration_progress,
        .retire = loom_serve_text_registration_retire,
        .parameters = loom_serve_text_registration_parameters,
        .set_pinned = loom_serve_text_registration_set_pinned,
        .activate = loom_serve_text_registration_activate,
        .deactivate = loom_serve_text_registration_deactivate,
        .destroy = loom_serve_text_registration_destroy};

iree_status_t loom_serve_text_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_serve_text_options_t* model_options,
    const loom_serve_text_service_options_t* options,
    const loom_service_finite_model_options_t* image_options,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_service_registration_t** out_registration) {
  *out_registration = NULL;
  if (options->image || options->scheduling.control ||
      options->scheduling.publish_observation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "registration owns image and scheduling bindings");
  }
  const bool images = image_options != NULL;
  iree_host_size_t total = 0, shapes_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_registration_t), &total,
      IREE_STRUCT_FIELD(model_options->epoch_count, loom_serve_packing_shape_t,
                        &shapes_offset)));
  iree_host_size_t operation_capacity = 0;
  if (!iree_host_size_checked_add(5 + 2 * images, options->variant_capacity,
                                  &operation_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "text operation capacity is not representable");
  }
  loom_serve_text_registration_t* registration = NULL;
  loom_runtime_service_control_t* control = NULL;
  IREE_RETURN_IF_ERROR(loom_service_registration_allocate_storage(
      total, 3 + 2 * images, operation_capacity, incarnation, host_allocator,
      (void**)&registration, &control));
  registration->host_allocator = host_allocator;
  registration->device = device;
  registration->options = *options;
  registration->stream = stream;
  loom_service_registration_initialize(
      &loom_serve_text_registration_vtable, registration,
      (1u << LOOM_SERVICE_MODEL_COMPONENT_PRIMARY) |
          (images ? 1u << LOOM_SERVICE_MODEL_COMPONENT_IMAGE : 0),
      iree_hal_device_topology_info(loom_runtime_device_handle(device))
          ->topology_index,
      loom_runtime_service_control_binding(control, 1 + 2 * images,
                                           operation_capacity - 2, 0),
      observer, &registration->registration);
  loom_serve_text_options_t copied_options = *model_options;
  if (model_options->epoch_count) {
    loom_serve_packing_shape_t* shapes =
        (loom_serve_packing_shape_t*)((uint8_t*)registration + shapes_offset);
    memcpy(shapes, model_options->epoch_shapes,
           model_options->epoch_count * sizeof(*shapes));
    copied_options.epoch_shapes = shapes;
  }
  iree_status_t status = iree_ok_status();
  if (images) {
    status = loom_service_finite_component_allocate(
        device, pool, image_options, observer,
        (loom_runtime_service_control_binding_t){
            .control = control,
            .offer_ordinal = 2,
            .operation_ordinal = 4 + options->variant_capacity,
            .publish_observation = loom_serve_text_registration_publish_image,
            .publisher = registration},
        host_allocator, &registration->image);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_load_allocate(device, pool, &copied_options,
                                           observer.wake, observer.user_data,
                                           host_allocator, &registration->load);
  }
  if (iree_status_is_ok(status)) {
    *out_registration = &registration->registration;
  } else {
    loom_service_finite_component_free(registration->image);
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

loom_serve_text_service_t* loom_serve_text_registration_service(
    loom_service_registration_t* registration) {
  return (loom_serve_text_service_t*)loom_service_registration_snapshot(
             registration)
      .service.self;
}
