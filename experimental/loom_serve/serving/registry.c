// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/registry.h"

struct loom_serve_registry_t {
  // Allocator owning the table and registration operations.
  iree_allocator_t allocator;
  // Model capabilities borrowed through final resource retirement.
  loom_serve_registration_factory_t factory;
  // Stable slots, never compacted while borrowers refer to their indices.
  loom_serve_registered_model_t* models;
  // Total borrowed and managed slot capacity.
  iree_host_size_t capacity;
  // Number of names still registered, including failed/removing entries.
  iree_host_size_t count;
  // Next identity assigned to a managed registration, independent of its name.
  uint64_t next_incarnation;
  // Work from removed incarnations, accumulated exactly once before slot reuse.
  loom_serve_telemetry_counters_t retired;
};

static void registry_add_counters(
    loom_serve_telemetry_counters_t* target,
    const loom_serve_telemetry_counters_t* source) {
  target->requests += source->requests;
  target->completed += source->completed;
  target->rejected += source->rejected;
  target->cancelled += source->cancelled;
  target->failed += source->failed;
  target->input_tokens += source->input_tokens;
  target->output_tokens += source->output_tokens;
  target->batches += source->batches;
}

iree_status_t loom_serve_registry_create(
    iree_host_size_t service_count, const loom_serve_http_service_t* services,
    iree_host_size_t managed_capacity,
    loom_serve_registration_factory_t factory,
    loom_serve_registry_t** out_registry, iree_allocator_t allocator) {
  *out_registry = NULL;
  if (managed_capacity > IREE_HOST_SIZE_MAX - service_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "registration capacity overflow");
  }
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    bool valid_name = services[i].name.size != 0;
    for (iree_host_size_t j = 0; j < services[i].name.size; ++j) {
      const uint8_t c = (uint8_t)services[i].name.data[j];
      valid_name &= c >= 0x21 && c <= 0x7e;
    }
    if (!valid_name) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "service names must be nonempty visible ASCII");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(services[i].name, services[j].name)) {
        return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "duplicate service name");
      }
    }
  }
  loom_serve_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*registry), (void**)&registry));
  registry->allocator = allocator;
  registry->factory = factory;
  registry->capacity = service_count + managed_capacity;
  registry->count = service_count;
  registry->next_incarnation = 1;
  iree_status_t status = iree_ok_status();
  if (registry->capacity) {
    status = iree_allocator_malloc_array(allocator, registry->capacity,
                                         sizeof(*registry->models),
                                         (void**)&registry->models);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < service_count; ++i) {
      registry->models[i].service = services[i];
      registry->models[i].phase = LOOM_SERVE_REGISTRATION_READY;
      registry->models[i].registration_time = iree_time_now();
    }
    *out_registry = registry;
  } else {
    iree_allocator_free(allocator, registry);
  }
  return status;
}

iree_status_t loom_serve_registry_destroy(loom_serve_registry_t* registry) {
  if (!registry) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    if (model->registration.self) {
      if (!iree_status_is_ok(model->status)) {
        return iree_status_clone(model->status);
      }
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "registration still owns model resources");
    }
  }
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    iree_status_free(model->status);
    loom_serve_telemetry_stream_destroy(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
  }
  iree_allocator_free(registry->allocator, registry->models);
  iree_allocator_free(registry->allocator, registry);
  return iree_ok_status();
}

iree_host_size_t loom_serve_registry_capacity(
    const loom_serve_registry_t* registry) {
  return registry->capacity;
}

iree_host_size_t loom_serve_registry_count(
    const loom_serve_registry_t* registry) {
  return registry->count;
}

loom_serve_registered_model_t* loom_serve_registry_at(
    loom_serve_registry_t* registry, iree_host_size_t index) {
  return &registry->models[index];
}

iree_status_t loom_serve_registry_register(
    loom_serve_registry_t* registry,
    loom_serve_model_configuration_t* configuration,
    iree_host_size_t* out_index) {
  if (!(registry->factory.kinds & (1u << configuration->kind))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "this modality has no asynchronous registration provider");
  }
  iree_host_size_t available = registry->capacity;
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    const iree_string_view_t name = registry->models[i].service.name;
    if (iree_string_view_equal(name, configuration->name)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "model name is already registered");
    }
    if (!name.size && available == registry->capacity) {
      available = i;
    }
  }
  if (available == registry->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "model registration capacity is full");
  }
  loom_serve_registered_model_t* model = &registry->models[available];
  IREE_RETURN_IF_ERROR(loom_serve_telemetry_stream_create(128, &model->stream,
                                                          registry->allocator));
  model->configuration = *configuration;
  memset(configuration, 0, sizeof(*configuration));
  model->closed = model->configuration.start_drained;
  model->incarnation = registry->next_incarnation++;
  model->registration_time = iree_time_now();
  model->service.name = model->configuration.name;
  // Protocol metadata is known before source preparation, without model access.
  switch (model->configuration.kind) {
    case LOOM_SERVE_MODEL_KIND_TEXT:
      model->service.kind = "text";
      model->service.endpoint = "/v1/chat/completions";
      break;
    case LOOM_SERVE_MODEL_KIND_IMAGE:
      model->service.kind = "image";
      model->service.endpoint = "/v1/images/generations";
      break;
    case LOOM_SERVE_MODEL_KIND_EMBEDDING:
      model->service.kind = "embedding";
      model->service.endpoint = "/v1/embeddings";
      break;
    case LOOM_SERVE_MODEL_KIND_DECISION:
      model->service.kind = "decision";
      model->service.endpoint = "/v1/systemone";
      break;
  }
  model->service.telemetry.counters = &model->counters;
  model->service.telemetry.stream = model->stream;
  const loom_serve_telemetry_record_t begin = {
      .time_ns = iree_time_now(),
      .event = &LOOM_SERVE_TELEMETRY_STREAM_BEGIN,
      .data.snapshot.phase = "registered"};
  loom_serve_telemetry_stream_append(model->stream, &begin);
  ++registry->count;
  *out_index = available;
  return iree_ok_status();
}

iree_status_t loom_serve_registry_remove(loom_serve_registered_model_t* model,
                                         loom_serve_removal_t mode) {
  if (!model->incarnation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "service lifetime belongs to its application, not this registry");
  }
  model->closed = true;
  if (model->parameter_operation ==
      LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING) {
    model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_NONE;
  }
  if (mode > model->removal) {
    model->removal = mode;
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_RELEASE_FAILED) {
    model->phase = LOOM_SERVE_REGISTRATION_RETIRING;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_registry_deactivate(
    loom_serve_registered_model_t* model) {
  if (!model->incarnation || !model->closed || model->removal ||
      model->pinned || !iree_status_is_ok(model->status) ||
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires a healthy, drained, "
                            "unpinned managed registration");
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_DECLARED) {
    return iree_ok_status();
  }
  if (model->phase != LOOM_SERVE_REGISTRATION_READY ||
      !model->registration.parameters) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "registration has no prepared parameter residency");
  }
  const loom_serve_residency_snapshot_t state =
      model->registration.parameters(model->registration.self);
  if (!state.elastic) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed parameter backing cannot deactivate");
  }
  model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING;
  return iree_ok_status();
}

iree_status_t loom_serve_registry_set_pinned(
    loom_serve_registered_model_t* model, bool pinned) {
  if (!model->incarnation || model->removal ||
      !iree_status_is_ok(model->status) ||
      (pinned && model->parameter_operation ==
                     LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pinning requires a healthy managed registration without pending "
        "deactivation");
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
    if (!model->registration.set_pinned) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "registration cannot pin parameters");
    }
    model->registration.set_pinned(model->registration.self, pinned);
  }
  model->pinned = pinned;
  return iree_ok_status();
}

iree_status_t loom_serve_registry_activate(
    loom_serve_registered_model_t* model) {
  if (!model->incarnation || !model->closed || model->removal ||
      !iree_status_is_ok(model->status) ||
      model->parameter_operation ==
          LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "activation requires a healthy, drained managed registration without "
        "pending deactivation");
  }
  if (model->phase != LOOM_SERVE_REGISTRATION_DECLARED &&
      model->phase != LOOM_SERVE_REGISTRATION_PREPARING &&
      (model->phase != LOOM_SERVE_REGISTRATION_READY ||
       !model->registration.activate)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "registration cannot activate parameters");
  }
  if (model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_NONE) {
    model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_WAITING;
  }
  return iree_ok_status();
}

loom_serve_http_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model) {
  if (model->removal == LOOM_SERVE_REMOVAL_ABORT) {
    return LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING;
  }
  if (model->removal == LOOM_SERVE_REMOVAL_DRAIN) {
    const loom_serve_http_service_snapshot_t snapshot =
        model->service.snapshot(model->service.self);
    if (!snapshot.queued_requests && !snapshot.active_requests &&
        !model->tokenizer_count) {
      return LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING;
    }
  }
  return 0;
}

static void registry_release_service(loom_serve_registered_model_t* model) {
  model->counters = *model->service.telemetry.counters;
  model->service = (loom_serve_http_service_t){
      .name = model->service.name,
      .kind = model->service.kind,
      .endpoint = model->service.endpoint,
      .capacity_domain = model->service.capacity_domain,
      .telemetry = {.stream = model->stream, .counters = &model->counters}};
}

static void registry_record_error(loom_serve_registered_model_t* model,
                                  iree_status_t status) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.error);
  loom_serve_telemetry_stream_append(model->stream, &record);
}

iree_status_t loom_serve_registry_advance(loom_serve_registry_t* registry,
                                          iree_host_size_t index,
                                          loom_serve_http_observer_t observer,
                                          bool* out_progress) {
  *out_progress = false;
  loom_serve_registered_model_t* model = &registry->models[index];
  if (!model->incarnation) {
    return iree_ok_status();
  }
  iree_status_t terminal_status = iree_ok_status();
  bool progress = false;
  const loom_serve_registration_phase_t previous_phase = model->phase;
  if (model->phase == LOOM_SERVE_REGISTRATION_DECLARED && !model->removal &&
      (!model->closed ||
       model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING)) {
    model->status = registry->factory.create(
        registry->factory.self, &model->configuration, observer, model->stream,
        &model->registration, registry->allocator);
    model->phase = iree_status_is_ok(model->status)
                       ? LOOM_SERVE_REGISTRATION_PREPARING
                       : LOOM_SERVE_REGISTRATION_FAILED;
    if (!iree_status_is_ok(model->status)) {
      model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_NONE;
      registry_record_error(model, model->status);
    }
    progress = true;
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_PREPARING) {
    bool complete = false;
    loom_serve_http_service_t service = {0};
    model->status = model->registration.prepare(
        model->registration.self, model->removal != LOOM_SERVE_REMOVAL_NONE,
        &complete, &service, &model->compilation);
    if (complete) {
      if (iree_status_is_ok(model->status) && model->pinned &&
          !model->registration.set_pinned) {
        model->status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                         "registration cannot pin parameters");
      }
      if (iree_status_is_ok(model->status) &&
          model->parameter_operation ==
              LOOM_SERVE_PARAMETER_OPERATION_WAITING &&
          !model->registration.activate) {
        model->status =
            iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "registration cannot activate parameters");
      }
      if (iree_status_is_ok(model->status)) {
        if (model->registration.set_pinned) {
          model->registration.set_pinned(model->registration.self,
                                         model->pinned);
        }
        model->service = service;
        service.set_observer(service.self, observer);
        model->phase = LOOM_SERVE_REGISTRATION_READY;
      } else {
        model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_NONE;
        registry_record_error(model, model->status);
        if (registry->factory.query_status) {
          terminal_status =
              registry->factory.query_status(registry->factory.self);
        }
        model->phase = LOOM_SERVE_REGISTRATION_RETIRING;
      }
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      (model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
       model->parameter_operation ==
           LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING)) {
    const loom_serve_http_service_snapshot_t snapshot =
        model->service.snapshot(model->service.self);
    if (model->removal ||
        (!snapshot.queued_requests && !snapshot.active_requests &&
         !model->service.is_active(model->service.self))) {
      loom_serve_activation_phase_t phase = LOOM_SERVE_ACTIVATION_COMPLETE;
      model->status = model->registration.activate(
          model->registration.self,
          model->removal ? LOOM_SERVE_ACTIVATION_FLAG_CANCEL : 0, &phase);
      const loom_serve_parameter_operation_t operation =
          phase == LOOM_SERVE_ACTIVATION_COMPLETE
              ? LOOM_SERVE_PARAMETER_OPERATION_NONE
          : phase == LOOM_SERVE_ACTIVATION_LOADING
              ? LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING
              : LOOM_SERVE_PARAMETER_OPERATION_WAITING;
      progress |= operation != model->parameter_operation;
      model->parameter_operation = operation;
      if (!iree_status_is_ok(model->status)) {
        registry_record_error(model, model->status);
        terminal_status = iree_status_clone(model->status);
      }
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY && model->removal &&
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_NONE) {
    const loom_serve_http_service_snapshot_t snapshot =
        model->service.snapshot(model->service.self);
    if (!snapshot.queued_requests && !snapshot.active_requests &&
        !model->tokenizer_count && !model->held_count &&
        !model->service.is_pending(model->service.self)) {
      model->service.set_observer(model->service.self,
                                  (loom_serve_http_observer_t){0});
      registry_release_service(model);
      model->phase = LOOM_SERVE_REGISTRATION_RETIRING;
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      model->parameter_operation ==
          LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING) {
    const loom_serve_http_service_snapshot_t snapshot =
        model->service.snapshot(model->service.self);
    if (!snapshot.queued_requests && !snapshot.active_requests &&
        !model->service.is_active(model->service.self)) {
      model->status = model->registration.deactivate(model->registration.self);
      model->parameter_operation = LOOM_SERVE_PARAMETER_OPERATION_NONE;
      if (!iree_status_is_ok(model->status)) {
        registry_record_error(model, model->status);
        terminal_status = iree_status_clone(model->status);
      }
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_RETIRING) {
    bool complete = false;
    iree_status_t status =
        model->registration.retire(model->registration.self, &complete);
    if (complete) {
      if (iree_status_is_ok(status)) {
        model->registration.destroy(model->registration.self);
        model->registration = (loom_serve_registration_t){0};
        model->phase = LOOM_SERVE_REGISTRATION_FAILED;
      } else {
        registry_record_error(model, status);
        model->status = iree_status_join(model->status, status);
        model->phase = LOOM_SERVE_REGISTRATION_RELEASE_FAILED;
      }
      progress = true;
    }
  }
  if (model->phase != previous_phase) {
    loom_serve_telemetry_record_t record = {
        .time_ns = iree_time_now(),
        .event = &LOOM_SERVE_TELEMETRY_SNAPSHOT,
        .data.snapshot.counters = model->counters};
    const char* phase = loom_serve_registry_phase(model);
    memcpy(record.data.snapshot.phase, phase, strlen(phase) + 1);
    loom_serve_telemetry_stream_append(model->stream, &record);
  }
  if (model->removal && !model->registration.self && !model->held_count &&
      !model->tokenizer_count) {
    registry_add_counters(&registry->retired, &model->counters);
    iree_status_free(model->status);
    loom_serve_telemetry_stream_destroy(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
    memset(model, 0, sizeof(*model));
    --registry->count;
    progress = true;
  }
  *out_progress = progress;
  return terminal_status;
}

bool loom_serve_registry_pending(const loom_serve_registered_model_t* model) {
  return model->parameter_operation != LOOM_SERVE_PARAMETER_OPERATION_NONE ||
         model->phase == LOOM_SERVE_REGISTRATION_PREPARING ||
         model->phase == LOOM_SERVE_REGISTRATION_RETIRING ||
         (model->phase == LOOM_SERVE_REGISTRATION_READY &&
          model->service.is_pending(model->service.self));
}

const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model) {
  static const char* const phases[] = {"registered", "preparing",
                                       "ready",      "retiring",
                                       "failed",     "release_failed"};
  return phases[model->phase];
}

bool loom_serve_registry_counters(
    const loom_serve_registry_t* registry,
    loom_serve_telemetry_counters_t* out_counters) {
  *out_counters = registry->retired;
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    const loom_serve_http_service_t* service = &registry->models[i].service;
    if (!service->name.size) {
      continue;
    }
    if (!service->telemetry.counters) {
      return false;
    }
    registry_add_counters(out_counters, service->telemetry.counters);
  }
  return true;
}
