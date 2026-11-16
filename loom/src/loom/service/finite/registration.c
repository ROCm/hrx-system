// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/registration.h"

#include "iree/hal/api.h"
#include "loom/runtime/device/pool.h"

// The native registration and its retained control share this one allocation.
// Source mechanics retain typed resource handles; lifecycle state has one
// owner.
typedef struct loom_service_finite_registration_t {
  // Shared native lifecycle, embedded through provider destruction.
  loom_service_registration_t registration;
  // Allocator owning this provider and its control tail.
  iree_allocator_t host_allocator;
  // Borrowed device, alive through actual model retirement.
  loom_runtime_device_t* device;
  // Copied source options; their views retain their original ownership.
  loom_service_finite_registration_options_t options;
  // Owned source load until preparation transfers or retires its model.
  loom_service_finite_load_t* load;
  // Owned model, released only after all actual native consumers retire.
  loom_service_finite_model_t* model;
  // Owned ready service, borrowing the model until the external-view handoff.
  loom_service_finite_service_t* service;
  // Accepted parameter activation, joined before model retirement.
  loom_runtime_activation_t* activation;
  // Model retirement was submitted with the native owner's wake target.
  bool retiring;
} loom_service_finite_registration_t;

static iree_status_t loom_service_finite_registration_prepare(
    void* self, loom_service_registration_prepare_flags_t flags,
    loom_service_preparation_result_t* out_result,
    loom_runtime_compilation_report_t* out_report) {
  loom_service_finite_registration_t* provider = self;
  bool complete = false;
  iree_status_t status = loom_service_finite_load_advance(
      provider->load,
      iree_any_bit_set(flags, LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL)
          ? LOOM_SERVICE_FINITE_LOAD_FLAG_CANCEL
          : 0,
      &complete);
  if (!complete) {
    return status;
  }
  *out_report = *loom_service_finite_load_report(provider->load);
  if (iree_status_is_ok(status)) {
    provider->model = loom_service_finite_load_take_model(provider->load);
  }
  // Terminal failure joins consumers, but releasing their native reservations
  // can still fail. The load then remains part of the accepted ownership.
  iree_status_t cleanup = loom_service_finite_load_free(provider->load);
  if (iree_status_is_ok(cleanup)) {
    provider->load = NULL;
  }
  status = iree_status_join(status, cleanup);
  if (iree_status_is_ok(status)) {
    const loom_service_finite_registration_options_t* options =
        &provider->options;
    status = loom_service_finite_service_allocate(
        loom_runtime_device_host(provider->device),
        loom_service_finite_model_generator(provider->model), options->protocol,
        options->name, options->capacity, options->variant_capacity,
        options->heartbeat_interval, options->stream,
        loom_service_registration_binding(&provider->registration, 0, 0, 0),
        provider->host_allocator, &provider->service);
  }
  out_result->flags = LOOM_SERVICE_PREPARATION_RESULT_COMPLETE |
                      (provider->model || provider->load
                           ? LOOM_SERVICE_PREPARATION_RESULT_RETAINED
                           : 0);
  if (iree_status_is_ok(status)) {
    out_result->service =
        loom_service_finite_service_interface(provider->service).runtime;
  }
  return status;
}

static iree_status_t loom_service_finite_registration_progress(
    void* self, loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_service_finite_registration_t* provider = self;
  return loom_service_finite_service_progress(provider->service, flags,
                                              out_progress);
}

static loom_service_registration_parameters_t
loom_service_finite_registration_parameters(
    void* self, loom_service_model_component_t component) {
  loom_service_finite_registration_t* provider = self;
  const loom_runtime_residency_snapshot_t parameters =
      loom_service_finite_model_parameters(provider->model);
  return (loom_service_registration_parameters_t){
      .phase = "ready",
      .prepared = true,
      .deactivatable = parameters.elastic,
      .parameters = parameters};
}

static void loom_service_finite_registration_set_pinned(
    void* self, loom_service_model_component_t component, bool pinned) {
  loom_service_finite_registration_t* provider = self;
  loom_service_finite_model_set_pinned(provider->model, pinned);
}

static iree_status_t loom_service_finite_registration_activate(
    void* self, loom_service_model_component_t component,
    loom_runtime_activation_flags_t flags,
    loom_runtime_activation_phase_t* out_phase,
    iree_status_t* out_component_status) {
  loom_service_finite_registration_t* provider = self;
  *out_component_status = iree_ok_status();
  return loom_runtime_activation_poll(
      &provider->activation,
      loom_service_finite_model_residency(provider->model),
      loom_runtime_host_proactor_pool(
          loom_runtime_device_host(provider->device)),
      provider->registration.observer.wake,
      provider->registration.observer.user_data, flags,
      provider->host_allocator, out_phase);
}

static iree_status_t loom_service_finite_registration_deactivate(
    void* self, loom_service_model_component_t component, bool* out_complete) {
  loom_service_finite_registration_t* provider = self;
  return loom_service_finite_model_deactivate(provider->model, out_complete);
}

static iree_status_t loom_service_finite_registration_retire(
    void* self, bool* out_complete) {
  loom_service_finite_registration_t* provider = self;
  *out_complete = true;
  IREE_RETURN_IF_ERROR(loom_service_finite_load_free(provider->load));
  provider->load = NULL;
  loom_service_finite_service_free(provider->service);
  provider->service = NULL;
  if (provider->model && !provider->retiring) {
    loom_service_finite_model_retire(provider->model,
                                     provider->registration.observer.wake,
                                     provider->registration.observer.user_data);
    provider->retiring = true;
  }
  if (!loom_service_finite_model_is_retired(provider->model)) {
    *out_complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_service_finite_model_free_retired(provider->model));
  provider->model = NULL;
  return iree_ok_status();
}

static void loom_service_finite_registration_destroy(void* self) {
  loom_service_finite_registration_t* provider = self;
  iree_allocator_free_aligned(provider->host_allocator, provider);
}

static const loom_service_registration_vtable_t
    loom_service_finite_registration_vtable = {
        .prepare = loom_service_finite_registration_prepare,
        .progress = loom_service_finite_registration_progress,
        .retire = loom_service_finite_registration_retire,
        .parameters = loom_service_finite_registration_parameters,
        .set_pinned = loom_service_finite_registration_set_pinned,
        .activate = loom_service_finite_registration_activate,
        .deactivate = loom_service_finite_registration_deactivate,
        .destroy = loom_service_finite_registration_destroy};

iree_status_t loom_service_finite_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* model_options,
    const loom_service_finite_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_service_registration_t** out_registration) {
  *out_registration = NULL;
  iree_host_size_t operation_capacity = 0;
  if (!iree_host_size_checked_add(3, options->variant_capacity,
                                  &operation_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "finite operation capacity is not representable");
  }
  loom_service_finite_registration_t* provider = NULL;
  loom_runtime_service_control_t* control = NULL;
  IREE_RETURN_IF_ERROR(loom_service_registration_allocate_storage(
      sizeof(*provider), 3, operation_capacity, options->incarnation,
      host_allocator, (void**)&provider, &control));
  provider->host_allocator = host_allocator;
  provider->device = device;
  provider->options = *options;
  iree_status_t status = loom_service_finite_load_allocate(
      device, pool, model_options, observer.wake, observer.user_data,
      host_allocator, &provider->load);
  if (iree_status_is_ok(status)) {
    const uint64_t device_ordinal =
        iree_hal_device_topology_info(loom_runtime_device_handle(device))
            ->topology_index;
    loom_service_registration_initialize(
        &loom_service_finite_registration_vtable, provider,
        1u << LOOM_SERVICE_MODEL_COMPONENT_PRIMARY, device_ordinal,
        loom_runtime_pool_residency_cache(pool),
        loom_runtime_service_control_binding(control, 1,
                                             options->variant_capacity + 1, 0),
        observer, options->name, options->stream, &provider->registration);
    *out_registration = &provider->registration;
  } else {
    iree_allocator_free_aligned(host_allocator, provider);
  }
  return status;
}

loom_service_finite_service_t* loom_service_finite_registration_service(
    loom_service_registration_t* registration) {
  const loom_service_registration_snapshot_t snapshot =
      loom_service_registration_snapshot(registration);
  return (loom_service_finite_service_t*)snapshot.service.self;
}
