// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/registration.h"

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
  // Tail-allocated observations, retained independently of model residency.
  loom_runtime_service_control_t* control;
  // Explicit warmup, joined before model retirement.
  loom_runtime_activation_t* activation;
  // Model consumers have been relinquished with an ownership wake installed.
  bool retiring;
};

iree_status_t loom_service_finite_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* model_options,
    const loom_service_finite_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_service_finite_registration_t** out_registration) {
  *out_registration = NULL;
  const iree_host_size_t use_capacities[] = {1};
  const loom_runtime_service_offer_capacity_t offers[] = {{1, use_capacities}};
  const loom_runtime_service_control_capacity_t capacity = {
      .offer_capacity = 1,
      .offers = offers,
      .operation_capacity = 1,
      .operation_use_capacities = use_capacities};
  iree_host_size_t control_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_runtime_service_control_size(&capacity, &control_length));
  iree_host_size_t allocation_length = 0, control_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_service_finite_registration_t), &allocation_length,
      IREE_STRUCT_FIELD_ALIGNED(control_length, uint8_t, 64, &control_offset)));
  loom_service_finite_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, allocation_length, 64, 0, (void**)&registration));
  registration->host_allocator = host_allocator;
  registration->device = device;
  registration->options = *options;
  registration->observer = observer;
  iree_status_t status = loom_runtime_service_control_initialize(
      &capacity, options->incarnation,
      iree_make_byte_span((uint8_t*)registration + control_offset,
                          control_length),
      &registration->control);
  if (iree_status_is_ok(status)) {
    status = loom_service_finite_load_allocate(
        device, pool, model_options, observer.wake, observer.user_data,
        host_allocator, &registration->load);
  }
  if (iree_status_is_ok(status)) {
    *out_registration = registration;
  } else {
    iree_allocator_free_aligned(host_allocator, registration);
  }
  return status;
}

iree_status_t loom_service_finite_registration_prepare(
    loom_service_finite_registration_t* registration,
    loom_service_finite_load_flags_t flags, bool* out_complete,
    loom_service_finite_service_t** out_service,
    loom_runtime_compilation_report_t* out_report) {
  *out_service = NULL;
  iree_status_t status =
      loom_service_finite_load_advance(registration->load, flags, out_complete);
  if (!*out_complete) {
    return status;
  }
  *out_report = *loom_service_finite_load_report(registration->load);
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
        (loom_runtime_service_control_binding_t){.control =
                                                     registration->control},
        registration->host_allocator, &registration->service);
  }
  if (iree_status_is_ok(status)) {
    *out_service = registration->service;
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
}

iree_status_t loom_service_finite_registration_activate(
    loom_service_finite_registration_t* registration,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase) {
  return loom_runtime_activation_poll(
      &registration->activation,
      loom_service_finite_model_residency(registration->model),
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(registration->device)),
      registration->observer.wake, registration->observer.user_data, flags,
      registration->host_allocator, out_phase);
}

iree_status_t loom_service_finite_registration_deactivate(
    loom_service_finite_registration_t* registration, bool* out_complete) {
  return loom_service_finite_model_deactivate(registration->model,
                                              out_complete);
}

iree_status_t loom_service_finite_registration_retire(
    loom_service_finite_registration_t* registration, bool* out_complete) {
  *out_complete = true;
  if (!registration) {
    return iree_ok_status();
  }
  if (registration->load) {
    IREE_RETURN_IF_ERROR(loom_service_finite_load_free(registration->load));
    registration->load = NULL;
  }
  loom_service_finite_service_free(registration->service);
  registration->service = NULL;
  if (registration->model && !registration->retiring) {
    loom_service_finite_model_retire(registration->model,
                                     registration->observer.wake,
                                     registration->observer.user_data);
    registration->retiring = true;
  }
  if (!loom_service_finite_model_is_retired(registration->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_service_finite_model_free_retired(registration->model));
  registration->model = NULL;
  const loom_runtime_service_operation_t operation = {0};
  loom_runtime_service_publish_operation(registration->control, &operation,
                                         NULL);
  const loom_runtime_service_observation_t observation = {0};
  loom_runtime_service_publish_observation(registration->control, &observation);
  return iree_ok_status();
}

void loom_service_finite_registration_free(
    loom_service_finite_registration_t* registration) {
  if (registration) {
    iree_allocator_free_aligned(registration->host_allocator, registration);
  }
}

const loom_runtime_service_control_t* loom_service_finite_registration_control(
    const loom_service_finite_registration_t* registration) {
  return registration->control;
}
