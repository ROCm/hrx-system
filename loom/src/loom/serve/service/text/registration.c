// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/registration.h"

struct loom_serve_text_registration_t {
  // Allocator owning this operation and its stable shape table.
  iree_allocator_t host_allocator;
  // Copied deployment policy; its views retain their original ownership.
  loom_serve_text_registration_options_t options;
  // Borrowed device resolved at declaration admission, alive through
  // retirement.
  loom_runtime_device_t* device;
  // Owner scheduling callback; no destruction occurs on its calling thread.
  loom_runtime_wake_callback_t observer;
  // Owned preparation operation, transferred on every terminal outcome.
  loom_serve_text_load_t* load;
  // Partial or prepared model retained through actual consumer retirement.
  loom_serve_text_model_t* model;
  // Owned dormant/on-demand image model, independent of text residency.
  loom_service_finite_component_t* image;
  // Service borrowing model state until all protocol claims are resolved.
  loom_serve_text_service_t* service;
  // Explicit warmup, joined before service/model retirement.
  loom_runtime_activation_t* activation;
  // Model consumers have relinquished views with an ownership wake installed.
  bool retiring;
};

iree_status_t loom_serve_text_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_serve_text_options_t* model_options,
    const loom_service_finite_model_options_t* image_options,
    const loom_serve_text_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_text_registration_t** out_registration) {
  *out_registration = NULL;
  iree_host_size_t total = 0, shapes_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_registration_t), &total,
      IREE_STRUCT_FIELD(model_options->epoch_count, loom_serve_packing_shape_t,
                        &shapes_offset)));
  loom_serve_text_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->device = device;
  registration->options = *options;
  registration->observer = observer;
  loom_serve_text_options_t model = *model_options;
  if (model.epoch_count) {
    loom_serve_packing_shape_t* shapes =
        (loom_serve_packing_shape_t*)((uint8_t*)registration + shapes_offset);
    memcpy(shapes, model.epoch_shapes, model.epoch_count * sizeof(*shapes));
    model.epoch_shapes = shapes;
  }
  iree_status_t status = iree_ok_status();
  if (image_options) {
    status = loom_service_finite_component_allocate(device, pool, image_options,
                                                    observer, host_allocator,
                                                    &registration->image);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_load_allocate(device, pool, &model, observer.wake,
                                           observer.user_data, host_allocator,
                                           &registration->load);
  }
  if (iree_status_is_ok(status)) {
    *out_registration = registration;
  } else {
    loom_service_finite_component_free(registration->image);
    iree_allocator_free(host_allocator, registration);
  }
  return status;
}

iree_status_t loom_serve_text_registration_prepare(
    loom_serve_text_registration_t* registration,
    loom_serve_text_load_flags_t flags, bool* out_complete,
    loom_serve_text_service_t** out_service,
    loom_runtime_compilation_report_t* out_report) {
  *out_service = NULL;
  iree_status_t status =
      loom_serve_text_load_advance(registration->load, flags, out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_serve_text_load_report(registration->load);
  registration->model = loom_serve_text_load_take_model(registration->load);
  loom_serve_text_load_free(registration->load);
  registration->load = NULL;
  if (iree_status_is_ok(status)) {
    const loom_serve_text_registration_options_t* policy =
        &registration->options;
    const loom_serve_text_cache_selection_t cache =
        loom_serve_text_model_cache_selection(registration->model);
    const loom_serve_text_service_options_t options = {
        .name = policy->name,
        .image = registration->image,
        .row_count = loom_serve_text_model_row_count(registration->model),
        .max_sessions = policy->max_sessions,
        .variant_capacity = policy->variant_capacity,
        .chunk_size = policy->chunk_size,
        .default_max_tokens = policy->default_max_tokens,
        .pending_capacity = policy->pending_capacity,
        .session_cache_capacity = cache.session_capacity,
        .prefix_cache_capacity = cache.prefix_capacity,
        .admission_overtakes = policy->admission_overtakes,
        .heartbeat_interval = policy->heartbeat_interval,
        .schedule_mode = policy->schedule_mode,
        .packing_mode = policy->packing_mode,
        .continuation_epochs = policy->continuation_epochs};
    status = loom_serve_text_service_allocate(
        registration->model, &options, policy->stream,
        registration->host_allocator, &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = registration->service;
  }
  return status;
}

loom_runtime_residency_snapshot_t loom_serve_text_registration_parameters(
    const loom_serve_text_registration_t* registration) {
  return loom_runtime_residency_snapshot(
      loom_serve_text_model_residency(registration->model));
}

const loom_service_finite_component_t* loom_serve_text_registration_image(
    const loom_serve_text_registration_t* registration) {
  return registration->image;
}

void loom_serve_text_registration_set_pinned(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component, bool pinned) {
  if (component == LOOM_SERVE_TEXT_COMPONENT_IMAGE) {
    loom_service_finite_component_set_pinned(registration->image, pinned);
  } else {
    loom_runtime_residency_set_pinned(
        loom_serve_text_model_residency(registration->model), pinned);
  }
}

iree_status_t loom_serve_text_registration_deactivate(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component, bool* out_complete) {
  if (component == LOOM_SERVE_TEXT_COMPONENT_IMAGE) {
    *out_complete = false;
    if (loom_service_finite_component_is_pending(registration->image)) {
      return iree_ok_status();
    }
    loom_serve_text_service_release_projector(registration->service);
    return loom_service_finite_component_trim(registration->image,
                                              out_complete);
  }
  return loom_serve_text_model_deactivate(registration->model, out_complete);
}

iree_status_t loom_serve_text_registration_activate(
    loom_serve_text_registration_t* registration,
    loom_serve_text_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  *out_component_status = iree_ok_status();
  *out_phase = LOOM_RUNTIME_ACTIVATION_COMPLETE;
  loom_runtime_residency_t* residency = NULL;
  if (component == LOOM_SERVE_TEXT_COMPONENT_IMAGE &&
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
  } else if (component == LOOM_SERVE_TEXT_COMPONENT_PRIMARY) {
    residency = loom_serve_text_model_residency(registration->model);
  }
  // An accepted image activation already owns its residency borrow. Joining
  // cancellation must not reenter component preparation during removal.
  return loom_runtime_activation_poll(
      &registration->activation, residency,
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(registration->device)),
      registration->observer.wake, registration->observer.user_data, flags,
      registration->host_allocator, out_phase);
}

iree_status_t loom_serve_text_registration_retire(
    loom_serve_text_registration_t* registration, bool* out_complete) {
  *out_complete = true;
  if (!registration) {
    return iree_ok_status();
  }
  loom_serve_text_service_free(registration->service, IREE_STATUS_OK);
  registration->service = NULL;
  // Join one retirement domain before arming the other so a terminal cleanup
  // error leaves no outstanding callback on the retained registration.
  IREE_RETURN_IF_ERROR(
      loom_service_finite_component_retire(registration->image, out_complete));
  if (!*out_complete) {
    return iree_ok_status();
  }
  loom_service_finite_component_free(registration->image);
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

void loom_serve_text_registration_free(
    loom_serve_text_registration_t* registration) {
  if (registration) {
    iree_allocator_free(registration->host_allocator, registration);
  }
}
