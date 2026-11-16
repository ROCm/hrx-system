// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/registry.h"

#include "loom/runtime/host/clock.h"
#include "loom/runtime/wake.h"
#include "loom/serve/service/endpoint.h"
#include "loom/serve/service/text/checkpoint_control.h"
#include "loom/service/kind.h"
#include "loom/service/registration.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/telemetry/tracing.h"

struct loom_serve_registry_t {
  // Allocator owning the table and registration operations.
  iree_allocator_t host_allocator;
  // Model capabilities borrowed through final resource retirement.
  loom_serve_registration_factory_t factory;
  // Stable slots, never compacted while borrowers refer to their indices.
  loom_serve_registered_model_t* models;
  // Compact progression capabilities, published and retired with model slots.
  loom_runtime_service_t* services;
  // Serialized owner wake, borrowed through preparation and retirement.
  loom_runtime_wake_callback_t wake;
  // Synchronous observation between model progress and registration retirement.
  loom_serve_registry_observer_t observer;
  // Total borrowed and managed slot capacity.
  iree_host_size_t capacity;
  // Number of names still registered, including failed/removing entries.
  iree_host_size_t count;
  // Next identity assigned to a managed registration, independent of its name.
  uint64_t next_incarnation;
  // Next scheduling binding identity, independent of stable route identity.
  uint64_t next_control_incarnation;
  // Retained bytes per cold attachment control.
  iree_host_size_t attachment_control_length;
  // Aggregate counter epoch, including borrowed history; zero if unavailable.
  iree_time_t counters_since_ns;
  // Final observations accumulated exactly once before registration slot reuse.
  struct {
    // Completed work from removed incarnations.
    loom_service_telemetry_counters_t counters;
    // Terminal duration distributions from removed incarnations.
    loom_service_telemetry_latency_t latency;
  } retired;
};

static const loom_runtime_service_vtable_t loom_serve_registry_service_vtable;

static void loom_serve_registry_publish_service(loom_serve_registry_t* registry,
                                                iree_host_size_t index) {
  loom_serve_registered_model_t* model = &registry->models[index];
  if (!model->incarnation && !model->service.runtime.vtable) {
    registry->services[index] = model->service.runtime;
    return;
  }
  registry->services[index] = (loom_runtime_service_t){
      .self = model,
      .vtable = &loom_serve_registry_service_vtable,
      .capacity_domain =
          loom_serve_registry_state(model).service.capacity_domain,
      .control = loom_serve_registry_control(model),
  };
}

loom_service_registration_snapshot_t loom_serve_registry_state(
    const loom_serve_registered_model_t* model) {
  if (model->registration.native) {
    return loom_service_registration_snapshot(model->registration.native);
  }
  loom_service_registration_phase_t phase = LOOM_SERVICE_REGISTRATION_DECLARED;
  if (!model->incarnation) {
    phase = model->service.name.size ? LOOM_SERVICE_REGISTRATION_READY
                                     : LOOM_SERVICE_REGISTRATION_DECLARED;
  } else if (model->attachment.operation.key.serial) {
    phase = LOOM_SERVICE_REGISTRATION_PREPARING;
  } else if (!iree_status_is_ok(model->attachment.status)) {
    phase = LOOM_SERVICE_REGISTRATION_FAILED;
  } else if (model->removal) {
    phase = LOOM_SERVICE_REGISTRATION_RETIRED;
  }
  return (loom_service_registration_snapshot_t){
      .phase = phase,
      .parameter_operation = model->attachment.parameter_operation,
      .parameter_component = model->attachment.parameter_component,
      .pinned_components = model->attachment.pinned_components,
      .service = model->service.runtime,
      .status = model->attachment.status};
}

const loom_runtime_service_control_t* loom_serve_registry_control(
    const loom_serve_registered_model_t* model) {
  if (!model->incarnation) {
    return model->service.runtime.control;
  }
  return model->registration.native
             ? loom_service_registration_control(model->registration.native)
             : model->attachment.control;
}

static void loom_serve_registry_publish_attachment(
    loom_serve_registered_model_t* model) {
  const loom_runtime_service_choice_publication_t choice = {0};
  loom_runtime_service_publish_offer(model->attachment.control,
                                     &model->attachment.offer, &choice);
  loom_runtime_service_publish_operation(model->attachment.control,
                                         &model->attachment.operation, NULL);
  const loom_runtime_service_observation_t observation = {
      .offer_count = model->attachment.offer.choice_count,
      .pending_operation_count = model->attachment.operation.key.serial != 0};
  loom_runtime_service_publish_observation(model->attachment.control,
                                           &observation);
}

static bool loom_serve_registry_refresh_attachment(
    loom_serve_registered_model_t* model) {
  const bool available =
      !model->registration.native && !model->removal &&
      iree_status_is_ok(model->attachment.status) &&
      !model->attachment.operation.key.serial &&
      (model->configuration.sources.reference.path.size || !model->closed ||
       model->attachment.parameter_operation ==
           LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE);
  if (model->attachment.offer.choice_count == (uint64_t)available) {
    return false;
  }
  ++model->attachment.offer.key.revision;
  model->attachment.offer.choice_count = available;
  loom_serve_registry_publish_attachment(model);
  return true;
}

static void loom_serve_registry_update_demand(
    loom_serve_registered_model_t* model) {
  loom_service_registration_demand_t demand = 0;
  if (!model->closed) {
    demand |= LOOM_SERVICE_REGISTRATION_DEMAND_PREPARATION;
    if (model->parameter_demand_count) {
      demand |= LOOM_SERVICE_REGISTRATION_DEMAND_PARAMETERS;
    }
  }
  loom_service_registration_set_demand(model->registration.native, demand);
}

static bool loom_serve_registry_publish_ready(loom_serve_registry_t* registry,
                                              iree_host_size_t index) {
  loom_serve_registered_model_t* model = &registry->models[index];
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  if (state.phase != LOOM_SERVICE_REGISTRATION_READY ||
      state.retirement_requested || model->service.runtime.vtable) {
    return false;
  }
  model->service = model->registration.service(model->registration.native);
  registry->services[index].capacity_domain =
      model->service.runtime.capacity_domain;
  return true;
}

static loom_runtime_service_control_capacity_t
loom_serve_registry_attachment_capacity(void) {
  static const iree_host_size_t use_capacity = 0;
  static const loom_runtime_service_offer_capacity_t offer = {1, &use_capacity};
  return (loom_runtime_service_control_capacity_t){
      .offer_capacity = 1,
      .offers = &offer,
      .operation_capacity = 1,
      .operation_use_capacities = &use_capacity};
}

static void loom_serve_registry_add_counters(
    loom_service_telemetry_counters_t* target,
    const loom_service_telemetry_counters_t* source) {
  target->requests += source->requests;
  target->completed += source->completed;
  target->rejected += source->rejected;
  target->cancelled += source->cancelled;
  target->failed += source->failed;
  target->input_tokens += source->input_tokens;
  target->output_tokens += source->output_tokens;
  target->batches += source->batches;
}

iree_status_t loom_serve_registry_allocate(
    iree_host_size_t service_count, const loom_serve_service_t* services,
    iree_host_size_t managed_capacity,
    loom_serve_registration_factory_t factory, iree_allocator_t host_allocator,
    loom_serve_registry_t** out_registry) {
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
  const iree_host_size_t capacity = service_count + managed_capacity;
  const loom_runtime_service_control_capacity_t attachment_capacity =
      loom_serve_registry_attachment_capacity();
  iree_host_size_t control_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_runtime_service_control_size(&attachment_capacity, &control_length));
  iree_host_size_t controls_length = 0;
  if (!iree_host_size_checked_mul(managed_capacity, control_length,
                                  &controls_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "attachment control capacity overflow");
  }
  iree_host_size_t total_size = 0, models_offset = 0, services_offset = 0,
                   controls_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_registry_t), &total_size,
      IREE_STRUCT_FIELD(capacity, loom_serve_registered_model_t,
                        &models_offset),
      IREE_STRUCT_FIELD(capacity, loom_runtime_service_t, &services_offset),
      IREE_STRUCT_FIELD_ALIGNED(controls_length, uint8_t, 64,
                                &controls_offset)));
  loom_serve_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(host_allocator, total_size,
                                                     64, 0, (void**)&registry));
  registry->host_allocator = host_allocator;
  registry->factory = factory;
  registry->models =
      (loom_serve_registered_model_t*)((uint8_t*)registry + models_offset);
  registry->services =
      (loom_runtime_service_t*)((uint8_t*)registry + services_offset);
  registry->capacity = capacity;
  registry->count = service_count;
  registry->next_incarnation = 1;
  registry->next_control_incarnation = 1;
  registry->attachment_control_length = control_length;
  registry->counters_since_ns = iree_time_now();
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    registry->models[i].registry = registry;
    if (i >= service_count) {
      registry->models[i].attachment.control =
          (loom_runtime_service_control_t*)((uint8_t*)registry +
                                            controls_offset +
                                            (i - service_count) *
                                                control_length);
    }
  }
  IREE_TRACE(loom_service_trace_initialize());
  const int64_t registered_unix_seconds = loom_runtime_clock_now_unix_seconds();
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    registry->models[i].service = services[i];
    registry->models[i].registered_unix_seconds = registered_unix_seconds;
    loom_serve_registry_publish_service(registry, i);
    const iree_time_t since_ns = services[i].telemetry.stream
                                     ? loom_runtime_telemetry_stream_identity(
                                           services[i].telemetry.stream)
                                           .created_ns
                                     : 0;
    registry->counters_since_ns =
        iree_min(registry->counters_since_ns, since_ns);
  }
  *out_registry = registry;
  return iree_ok_status();
}

iree_status_t loom_serve_registry_free(loom_serve_registry_t* registry) {
  if (!registry) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    const loom_service_registration_snapshot_t state =
        loom_serve_registry_state(model);
    if (model->attachment.resolution ||
        (model->registration.native &&
         state.phase != LOOM_SERVICE_REGISTRATION_RETIRED)) {
      if (!iree_status_is_ok(state.status)) {
        return iree_status_clone(state.status);
      }
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "registration still owns model resources");
    }
  }
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    loom_service_registration_free(model->registration.native);
    iree_status_free(model->attachment.status);
    loom_runtime_telemetry_stream_free(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
  }
  iree_allocator_free_aligned(registry->host_allocator, registry);
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

iree_time_t loom_serve_registry_counters_since_ns(
    const loom_serve_registry_t* registry) {
  return registry->counters_since_ns;
}

loom_serve_registered_model_t* loom_serve_registry_at(
    loom_serve_registry_t* registry, iree_host_size_t index) {
  return &registry->models[index];
}

loom_runtime_service_t* loom_serve_registry_services(
    loom_serve_registry_t* registry) {
  return registry->services;
}

void loom_serve_registry_set_advance_observer(
    loom_serve_registry_t* registry, loom_runtime_wake_callback_t wake,
    loom_serve_registry_observer_t observer) {
  registry->wake = wake;
  registry->observer = observer;
}

loom_service_snapshot_t loom_serve_registry_snapshot(
    const loom_serve_registered_model_t* model) {
  return model->service.snapshot && loom_serve_registry_state(model).phase ==
                                        LOOM_SERVICE_REGISTRATION_READY
             ? model->service.snapshot(model->service.runtime.self)
             : (loom_service_snapshot_t){.phase =
                                             loom_serve_registry_phase(model)};
}

static iree_status_t loom_serve_registry_find_admission(
    loom_serve_registry_t* registry, iree_string_view_t name, uint32_t kinds,
    const char* operation, bool releasing, loom_serve_service_t** out_service) {
  *out_service = NULL;
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    if (!model->service.name.size ||
        !iree_string_view_equal(model->service.name, name)) {
      continue;
    }
    if (model->removal) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "model is being unregistered");
    }
    if (!iree_status_is_ok(loom_serve_registry_state(model).status)) {
      return iree_status_clone(loom_serve_registry_state(model).status);
    }
    if (!(kinds & (1u << model->service.kind))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected model does not provide %s", operation);
    }
    if (loom_serve_registry_state(model).phase !=
        LOOM_SERVICE_REGISTRATION_READY) {
      return releasing
                 ? iree_make_status(IREE_STATUS_NOT_FOUND,
                                    "unprepared model has no retained state")
                 : iree_ok_status();
    }
    if (!releasing && (model->closed || model->held_count)) {
      return iree_ok_status();
    }
    *out_service = &model->service;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "model is not registered");
}

iree_status_t loom_serve_registry_try_open_speech(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_speech_request_t* request,
    loom_serve_speech_stream_t** out_stream) {
  *out_stream = NULL;
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name, 1u << LOOM_SERVICE_KIND_SPEECH, "speech", false,
      &service));
  if (!service) {
    return iree_ok_status();
  }
  if (!service->try_open_speech) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "speech service has no native admission");
  }
  return service->try_open_speech(service->runtime.self, request, out_stream);
}

iree_status_t loom_serve_registry_try_upload_voice(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_speech_voice_request_t* request,
    loom_serve_speech_voice_upload_t** out_upload) {
  *out_upload = NULL;
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name, 1u << LOOM_SERVICE_KIND_SPEECH, "voice upload", false,
      &service));
  if (!service) {
    return iree_ok_status();
  }
  if (!service->try_upload_voice) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "speech model does not support voice uploads");
  }
  return service->try_upload_voice(service->runtime.self, request, out_upload);
}

iree_status_t loom_serve_registry_remove_voice(loom_serve_registry_t* registry,
                                               iree_string_view_t name,
                                               iree_string_view_t id) {
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name, 1u << LOOM_SERVICE_KIND_SPEECH, "voice removal", true,
      &service));
  if (!service->remove_voice) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "speech model does not support voice uploads");
  }
  return service->remove_voice(service->runtime.self, id);
}

iree_status_t loom_serve_registry_try_open_recognition(
    loom_serve_registry_t* registry, iree_string_view_t name,
    const loom_serve_recognition_request_t* request,
    loom_serve_recognition_stream_t** out_stream) {
  *out_stream = NULL;
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_registry_find_admission(registry, name,
                                         (1u << LOOM_SERVICE_KIND_RECOGNITION) |
                                             (1u << LOOM_SERVICE_KIND_ACTIVITY),
                                         "recognition", false, &service));
  if (!service) {
    return iree_ok_status();
  }
  if (!service->try_open_recognition) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "recognition service has no native admission");
  }
  return service->try_open_recognition(service->runtime.self, request,
                                       out_stream);
}

iree_status_t loom_serve_registry_try_open_text(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_text_request_t* request, loom_serve_text_stream_t** out_stream) {
  *out_stream = NULL;
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name, 1u << LOOM_SERVICE_KIND_TEXT, "text", false, &service));
  if (!service) {
    return iree_ok_status();
  }
  if (!service->try_open_text) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "text service has no native admission");
  }
  return service->try_open_text(service->runtime.self, request, out_stream);
}

iree_status_t loom_serve_registry_try_open_text_checkpoint(
    loom_serve_registry_t* registry, iree_string_view_t name,
    const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control) {
  *out_control = NULL;
  loom_serve_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name,
      (1u << LOOM_SERVICE_KIND_TEXT) | (1u << LOOM_SERVICE_KIND_SEQUENCE),
      "retained sequence",
      request->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE, &service));
  if (!service) {
    return iree_ok_status();
  }
  if (!service->try_open_text_checkpoint) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "text service has no native checkpoint admission");
  }
  return service->try_open_text_checkpoint(service->runtime.self, request,
                                           out_control);
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
  if (configuration->sources.reference.path.size &&
      !registry->factory.resolution_sink.submit) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "registration provider has no model-card "
                            "preparation worker");
  }
  iree_host_size_t device_index = 0;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_resolve_device(
      configuration, registry->factory.device_names, &device_index));
  iree_host_size_t pool_index = 0;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_resolve_pool(
      configuration, registry->factory.pool_count, registry->factory.pools,
      device_index, &pool_index));
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
  // Binding strings belong to the declaration; history belongs to the registry.
  // Keep both private until admission can publish the complete slot atomically.
  const iree_string_view_t device_name =
      registry->factory.device_names.values[device_index];
  const iree_string_view_t pool_name = registry->factory.pools[pool_index].name;
  char* device_storage = NULL;
  char* pool_storage = NULL;
  loom_runtime_telemetry_stream_t* stream = NULL;
  iree_status_t status = iree_ok_status();
  if (!configuration->device.size) {
    status = iree_allocator_clone(
        configuration->host_allocator,
        iree_make_const_byte_span(device_name.data, device_name.size),
        (void**)&device_storage);
  }
  if (iree_status_is_ok(status) && !configuration->pool.size) {
    status = iree_allocator_clone(
        configuration->host_allocator,
        iree_make_const_byte_span(pool_name.data, pool_name.size),
        (void**)&pool_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_telemetry_stream_allocate(
        128, registry->host_allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    const loom_runtime_service_control_capacity_t capacity =
        loom_serve_registry_attachment_capacity();
    loom_serve_registered_model_t* model = &registry->models[available];
    loom_runtime_service_control_t* control = NULL;
    status = loom_runtime_service_control_initialize(
        &capacity, registry->next_control_incarnation++,
        iree_make_byte_span(model->attachment.control,
                            registry->attachment_control_length),
        &control);
    if (iree_status_is_ok(status)) {
      model->attachment.control = control;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_serve_registered_model_t* model = &registry->models[available];
    model->configuration = *configuration;
    model->configuration.device_index = device_index;
    model->configuration.pool_index = pool_index;
    if (device_storage) {
      model->configuration.device =
          iree_make_string_view(device_storage, device_name.size);
    }
    if (pool_storage) {
      model->configuration.pool =
          iree_make_string_view(pool_storage, pool_name.size);
    }
    memset(configuration, 0, sizeof(*configuration));
    model->stream = stream;
    model->closed = model->configuration.start_drained;
    model->incarnation = registry->next_incarnation++;
    model->registered_unix_seconds = loom_runtime_clock_now_unix_seconds();
    model->service.name = model->configuration.name;
    // Protocol metadata is known before source preparation, without model
    // access.
    model->service.kind = model->configuration.kind;
    model->service.endpoint =
        loom_serve_service_endpoint(model->configuration.kind);
    model->service.telemetry.counters = &model->counters;
    model->service.telemetry.latency = &model->latency;
    model->service.telemetry.stream = model->stream;
    model->attachment.offer = (loom_runtime_service_offer_t){
        .work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION};
    loom_serve_registry_refresh_attachment(model);
    loom_serve_registry_publish_service(registry, available);
    loom_runtime_telemetry_record_t begin = {
        .time_ns = iree_time_now(),
        .event = &LOOM_SERVICE_TELEMETRY_STREAM_BEGIN};
    loom_service_telemetry_snapshot_record_t telemetry_payload = {
        .phase = "registered"};
    memcpy(begin.data.words, &telemetry_payload, sizeof(telemetry_payload));
    loom_runtime_telemetry_stream_append(model->stream, &begin);
    ++registry->count;
    *out_index = available;
  } else {
    loom_runtime_telemetry_stream_free(stream);
    iree_allocator_free(configuration->host_allocator, device_storage);
    iree_allocator_free(configuration->host_allocator, pool_storage);
  }
  return status;
}

iree_status_t loom_serve_registry_remove(loom_serve_registered_model_t* model,
                                         loom_serve_removal_t mode) {
  if (!model->incarnation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "service lifetime belongs to its application, not this registry");
  }
  model->closed = true;
  if (mode > model->removal) {
    model->removal = mode;
  }
  if (!model->registration.native) {
    model->attachment.parameter_operation =
        LOOM_SERVICE_PARAMETER_OPERATION_NONE;
    loom_serve_registry_refresh_attachment(model);
  } else if (loom_serve_registry_state(model).phase ==
             LOOM_SERVICE_REGISTRATION_RELEASE_FAILED) {
    loom_service_registration_request_retirement(
        model->registration.native, LOOM_SERVICE_RETIRE_FLAG_ABORT);
  }
  return iree_ok_status();
}

const char* loom_serve_model_component_name(
    loom_service_model_component_t component) {
  static const char* const names[] = {"primary", "image"};
  return names[component];
}

bool loom_serve_registry_has_component(
    const loom_serve_registered_model_t* model,
    loom_service_model_component_t component) {
  return component == LOOM_SERVICE_MODEL_COMPONENT_PRIMARY ||
         (model->configuration.kind == LOOM_SERVICE_KIND_TEXT &&
          (model->configuration.text.image.contract ||
           model->configuration.text.image.prepare.size != 0));
}

static iree_status_t loom_serve_registry_require_component(
    const loom_serve_registered_model_t* model,
    loom_service_model_component_t component) {
  if (component != LOOM_SERVICE_MODEL_COMPONENT_PRIMARY &&
      model->configuration.sources.reference.path.size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "model card is still resolving component metadata");
  }
  if (!loom_serve_registry_has_component(model, component)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "model has no %s component",
                            loom_serve_model_component_name(component));
  }
  return iree_ok_status();
}

iree_status_t loom_serve_registry_deactivate(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  if (!model->incarnation || !model->closed || model->removal ||
      (state.pinned_components & (1u << component)) ||
      !iree_status_is_ok(state.status) ||
      state.parameter_operation == LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE ||
      (state.parameter_operation != LOOM_SERVICE_PARAMETER_OPERATION_NONE &&
       state.parameter_component != component)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires a healthy, drained, "
                            "unpinned managed registration");
  }
  if (state.phase == LOOM_SERVICE_REGISTRATION_DECLARED) {
    return iree_ok_status();
  }
  if (!model->registration.native) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "registration has no prepared parameter residency");
  }
  return loom_service_registration_request_parameters(
      model->registration.native, component,
      LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE);
}

iree_status_t loom_serve_registry_set_pinned(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component, bool pinned) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  if (!model->incarnation || model->removal ||
      !iree_status_is_ok(state.status) ||
      (pinned && state.parameter_component == component &&
       state.parameter_operation ==
           LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pinning requires a healthy managed registration without pending "
        "deactivation");
  }
  if (model->registration.native) {
    loom_service_registration_set_pinned(model->registration.native, component,
                                         pinned);
  } else if (pinned) {
    model->attachment.pinned_components |= 1u << component;
  } else {
    model->attachment.pinned_components &= ~(1u << component);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_registry_activate(
    loom_serve_registered_model_t* model,
    loom_service_model_component_t component) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  if (!model->incarnation || !model->closed || model->removal ||
      !iree_status_is_ok(state.status) ||
      state.parameter_operation ==
          LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE ||
      (state.parameter_operation != LOOM_SERVICE_PARAMETER_OPERATION_NONE &&
       state.parameter_component != component)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "activation requires a healthy, drained managed registration without "
        "pending deactivation");
  }
  if (model->registration.native) {
    return loom_service_registration_request_parameters(
        model->registration.native, component,
        LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE);
  }
  model->attachment.parameter_component = component;
  model->attachment.parameter_operation =
      LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE;
  loom_serve_registry_refresh_attachment(model);
  return iree_ok_status();
}

loom_runtime_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model) {
  if (!iree_status_is_ok(loom_serve_registry_state(model).status) ||
      model->removal == LOOM_SERVE_REMOVAL_ABORT) {
    return LOOM_RUNTIME_SERVICE_FLAG_STOPPING;
  }
  if (model->removal == LOOM_SERVE_REMOVAL_DRAIN &&
      (!model->service.runtime.vtable ||
       !model->service.runtime.vtable->has_requests(
           model->service.runtime.self)) &&
      !model->tokenizer_count) {
    return LOOM_RUNTIME_SERVICE_FLAG_STOPPING;
  }
  return 0;
}

static void loom_serve_registry_release_service(
    loom_serve_registered_model_t* model) {
  model->counters = *model->service.telemetry.counters;
  model->latency = *model->service.telemetry.latency;
  model->service = (loom_serve_service_t){
      .name = model->service.name,
      .runtime = {.capacity_domain = model->service.runtime.capacity_domain},
      .kind = model->service.kind,
      .endpoint = model->service.endpoint,
      .telemetry = {.stream = model->stream,
                    .counters = &model->counters,
                    .latency = &model->latency}};
}

static void loom_serve_registry_record_error(
    loom_serve_registered_model_t* model, const iree_status_t status) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_ERROR};
  loom_service_telemetry_diagnostic_t telemetry_payload = {0};
  loom_service_telemetry_diagnostic_capture(status, &telemetry_payload);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(model->stream, &record);
}

iree_status_t loom_serve_registry_fail(loom_serve_registry_t* registry,
                                       iree_host_size_t index,
                                       iree_status_t status) {
  loom_serve_registered_model_t* model = &registry->models[index];
  if (!model->incarnation) {
    return status;
  }
  if (model->registration.native) {
    loom_service_registration_fail(model->registration.native, status);
  } else {
    if (iree_status_is_ok(model->attachment.status)) {
      loom_serve_registry_record_error(model, status);
    }
    model->attachment.status =
        iree_status_join(model->attachment.status, status);
    model->attachment.parameter_operation =
        LOOM_SERVICE_PARAMETER_OPERATION_NONE;
    loom_serve_registry_refresh_attachment(model);
  }
  return registry->factory.query_status
             ? registry->factory.query_status(registry->factory.self)
             : iree_ok_status();
}

// Actual cold completion retires the accepted operation before exposing a new
// choice or replacing this control. Decisions retain the old binding identity.
static void loom_serve_registry_complete_attachment(
    loom_serve_registered_model_t* model) {
  model->attachment.duration_ns += iree_time_now() - model->attachment.start_ns;
  model->attachment.start_ns = 0;
  model->attachment.operation = (loom_runtime_service_operation_t){0};
  loom_serve_registry_refresh_attachment(model);
  loom_serve_registry_publish_attachment(model);
}

static iree_status_t loom_serve_registry_progress(
    loom_serve_registry_t* registry, iree_host_size_t index,
    loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_serve_registered_model_t* model = &registry->models[index];
  if (!model->incarnation) {
    return model->service.runtime.vtable->advance(model->service.runtime.self,
                                                  flags, out_progress);
  }
  iree_status_t status = iree_ok_status();
  if (model->attachment.resolution) {
    loom_serve_model_configuration_t resolved = {0};
    iree_status_t resolution_status = iree_ok_status();
    if (loom_serve_model_resolution_take(&model->attachment.resolution,
                                         &resolution_status, &resolved)) {
      if (iree_status_is_ok(resolution_status) && !model->removal) {
        loom_serve_model_configuration_deinitialize(&model->configuration);
        model->configuration = resolved;
        resolved = (loom_serve_model_configuration_t){0};
        model->service.name = model->configuration.name;
      }
      loom_serve_model_configuration_deinitialize(&resolved);
      if (!iree_status_is_ok(resolution_status)) {
        status = loom_serve_registry_fail(registry, index, resolution_status);
      }
      loom_serve_registry_complete_attachment(model);
      *out_progress = true;
    }
  }
  if (model->registration.native) {
    const uint64_t previous_diagnostic =
        loom_serve_registry_state(model).diagnostic_revision;
    loom_serve_registry_update_demand(model);
    *out_progress |= loom_service_registration_progress(
        model->registration.native,
        flags | loom_serve_registry_service_flags(model));
    const loom_service_registration_snapshot_t state =
        loom_serve_registry_state(model);
    if (state.diagnostic_revision != previous_diagnostic &&
        registry->factory.query_status) {
      status = iree_status_join(
          status, registry->factory.query_status(registry->factory.self));
    }
    *out_progress |= loom_serve_registry_publish_ready(registry, index);
  } else {
    *out_progress |= loom_serve_registry_refresh_attachment(model);
  }
  return status;
}

static void loom_serve_registry_collect_model(loom_serve_registry_t* registry,
                                              iree_host_size_t index,
                                              bool* out_progress) {
  loom_serve_registered_model_t* model = &registry->models[index];
  if (!model->incarnation) {
    return;
  }
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  const loom_runtime_service_t service = model->service.runtime;
  const bool borrowed =
      model->held_count || model->tokenizer_count ||
      (service.vtable && (service.vtable->has_requests(service.self) ||
                          service.vtable->is_pending(service.self)));
  if ((model->removal || !iree_status_is_ok(state.status)) && !borrowed &&
      model->registration.native && !state.retirement_requested) {
    if (service.vtable) {
      service.vtable->set_observer(service.self,
                                   (loom_runtime_wake_callback_t){0});
      loom_serve_registry_release_service(model);
    }
    loom_service_registration_request_retirement(
        model->registration.native, loom_serve_registry_service_flags(model) &
                                            LOOM_RUNTIME_SERVICE_FLAG_STOPPING
                                        ? LOOM_SERVICE_RETIRE_FLAG_ABORT
                                        : 0);
    *out_progress = true;
  }
  const bool retired = !model->registration.native ||
                       state.phase == LOOM_SERVICE_REGISTRATION_RETIRED;
  if (model->removal && retired && !borrowed && !model->attachment.resolution) {
    loom_serve_registry_add_counters(&registry->retired.counters,
                                     &model->counters);
    loom_service_telemetry_latency_merge(&registry->retired.latency,
                                         &model->latency);
    loom_service_registration_free(model->registration.native);
    iree_status_free(model->attachment.status);
    loom_runtime_telemetry_stream_free(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
    loom_runtime_service_control_t* control = model->attachment.control;
    memset(model, 0, sizeof(*model));
    model->registry = registry;
    model->attachment.control = control;
    registry->services[index] = (loom_runtime_service_t){0};
    --registry->count;
    *out_progress = true;
  }
}

bool loom_serve_registry_pending(const loom_serve_registered_model_t* model) {
  if (model->registration.native) {
    return loom_service_registration_is_pending(model->registration.native);
  }
  return model->attachment.resolution ||
         (model->service.runtime.vtable &&
          model->service.runtime.vtable->is_pending(
              model->service.runtime.self));
}

static void loom_serve_registry_service_set_observer(
    void* self, loom_runtime_wake_callback_t observer) {
  loom_serve_registered_model_t* model = self;
  model->registry->wake = observer;
  if (model->service.runtime.vtable) {
    model->service.runtime.vtable->set_observer(model->service.runtime.self,
                                                observer);
  }
}

static bool loom_serve_registry_service_has_requests(void* self) {
  const loom_serve_registered_model_t* model = self;
  return model->held_count || model->tokenizer_count ||
         (model->service.runtime.vtable &&
          model->service.runtime.vtable->has_requests(
              model->service.runtime.self));
}

static bool loom_serve_registry_service_is_pending(void* self) {
  const loom_serve_registered_model_t* model = self;
  return loom_serve_registry_pending(model) ||
         (model->removal == LOOM_SERVE_REMOVAL_ABORT &&
          loom_serve_registry_state(model).phase !=
              LOOM_SERVICE_REGISTRATION_RELEASE_FAILED);
}

static bool loom_serve_registry_service_is_active(void* self) {
  const loom_serve_registered_model_t* model = self;
  if (model->registration.native) {
    return loom_service_registration_is_active(model->registration.native);
  }
  return model->service.runtime.vtable &&
         model->service.runtime.vtable->is_active(model->service.runtime.self);
}

static iree_status_t loom_serve_registry_service_advance(
    void* self, loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_serve_registered_model_t* model = self;
  loom_serve_registry_t* registry = model->registry;
  const iree_host_size_t index = model - registry->models;
  *out_progress = false;
  IREE_TRACE_ZONE_BEGIN_NAMED(z_model, "loom_serve_model_advance");
  IREE_TRACE_ZONE_APPEND_TEXT(z_model, model->service.name.data,
                              model->service.name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z_model, model->incarnation);
  const loom_service_registration_phase_t previous_phase =
      loom_serve_registry_state(model).phase;
  bool stopping = iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  iree_status_t status = iree_ok_status();
  if (stopping && model->incarnation &&
      model->removal != LOOM_SERVE_REMOVAL_ABORT) {
    status = loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT);
  }
  status = iree_status_join(status, loom_serve_registry_progress(
                                        registry, index, flags, out_progress));
  stopping |= !iree_status_is_ok(status);
  if (registry->observer.observe) {
    bool progress = false;
    status = iree_status_join(
        status, registry->observer.observe(
                    registry->observer.self, index,
                    stopping || !iree_status_is_ok(
                                    loom_serve_registry_state(model).status),
                    &progress));
    *out_progress |= progress;
  }
  if (model->incarnation &&
      loom_serve_registry_state(model).phase != previous_phase) {
    loom_runtime_telemetry_record_t record = {
        .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_SNAPSHOT};
    loom_service_telemetry_snapshot_record_t payload = {.counters =
                                                            model->counters};
    const char* phase = loom_serve_registry_phase(model);
    memcpy(payload.phase, phase, strlen(phase) + 1);
    memcpy(record.data.words, &payload, sizeof(payload));
    loom_runtime_telemetry_stream_append(model->stream, &record);
  }
  loom_serve_registry_collect_model(registry, index, out_progress);
  IREE_TRACE_ZONE_END(z_model);
  return status;
}

iree_status_t loom_serve_registry_advance(loom_serve_registry_t* registry,
                                          iree_host_size_t index,
                                          loom_runtime_wake_callback_t observer,
                                          bool* out_progress) {
  registry->wake = observer;
  return loom_serve_registry_service_advance(&registry->models[index], 0,
                                             out_progress);
}

static iree_status_t loom_serve_registry_service_try_schedule(
    void* self, loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags,
    loom_runtime_service_schedule_outcome_t* out_outcome) {
  loom_serve_registered_model_t* model = self;
  loom_serve_registry_t* registry = model->registry;
  const iree_host_size_t index = model - registry->models;
  *out_outcome = (loom_runtime_service_schedule_outcome_t){0};
  if (!model->incarnation) {
    return model->service.runtime.vtable->try_schedule(
        model->service.runtime.self, offer, choice_ordinal, flags, out_outcome);
  }
  flags |= loom_serve_registry_service_flags(model);
  if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    return iree_ok_status();
  }
  if (model->registration.native) {
    const uint64_t diagnostic =
        loom_serve_registry_state(model).diagnostic_revision;
    *out_outcome = loom_service_registration_try_schedule(
        model->registration.native, offer, choice_ordinal, flags);
    loom_serve_registry_publish_ready(registry, index);
    return loom_serve_registry_state(model).diagnostic_revision != diagnostic &&
                   registry->factory.query_status
               ? registry->factory.query_status(registry->factory.self)
               : iree_ok_status();
  }
  if (!model->attachment.offer.choice_count || choice_ordinal ||
      offer.ordinal || offer.revision != model->attachment.offer.key.revision) {
    return iree_ok_status();
  }
  model->attachment.operation = (loom_runtime_service_operation_t){
      .key = {.serial = ++model->attachment.next_serial},
      .origin = offer,
      .work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION,
      .waits = model->configuration.sources.reference.path.size
                   ? LOOM_RUNTIME_SERVICE_WAIT_CPU
                   : 0};
  model->attachment.start_ns = iree_time_now();
  out_outcome->accepted = true;
  out_outcome->operation = model->attachment.operation.key;
  loom_serve_registry_refresh_attachment(model);
  iree_status_t status = iree_ok_status();
  if (model->configuration.sources.reference.path.size) {
    status = loom_serve_model_resolution_start(
        registry->factory.resolution_sink, &model->configuration,
        registry->wake, registry->host_allocator,
        &model->attachment.resolution);
  } else {
    status = registry->factory.create(
        registry->factory.self, &model->configuration,
        registry->next_control_incarnation++, registry->wake, model->stream,
        registry->host_allocator, &model->registration);
    if (iree_status_is_ok(status)) {
      loom_serve_registry_update_demand(model);
      for (uint32_t component = 0;
           component < LOOM_SERVICE_MODEL_COMPONENT_COUNT; ++component) {
        if (model->attachment.pinned_components & (1u << component)) {
          loom_service_registration_set_pinned(
              model->registration.native,
              (loom_service_model_component_t)component, true);
        }
      }
      if (model->attachment.parameter_operation !=
          LOOM_SERVICE_PARAMETER_OPERATION_NONE) {
        status = loom_service_registration_request_parameters(
            model->registration.native, model->attachment.parameter_component,
            model->attachment.parameter_operation);
      }
      model->attachment.pinned_components = 0;
      model->attachment.parameter_component = 0;
      model->attachment.parameter_operation =
          LOOM_SERVICE_PARAMETER_OPERATION_NONE;
    }
  }
  iree_status_t shared_status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    out_outcome->status_code = iree_status_code(status);
    shared_status = loom_serve_registry_fail(registry, index, status);
  }
  if (!model->attachment.resolution) {
    loom_serve_registry_complete_attachment(model);
    if (model->registration.native) {
      loom_serve_registry_publish_service(registry, index);
    }
  }
  return shared_status;
}

static const loom_runtime_service_vtable_t loom_serve_registry_service_vtable =
    {.set_observer = loom_serve_registry_service_set_observer,
     .has_requests = loom_serve_registry_service_has_requests,
     .is_pending = loom_serve_registry_service_is_pending,
     .is_active = loom_serve_registry_service_is_active,
     .advance = loom_serve_registry_service_advance,
     .try_schedule = loom_serve_registry_service_try_schedule};

const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model) {
  static const char* const phases[] = {"registered",    "preparing", "ready",
                                       "failed",        "retiring",  "failed",
                                       "release_failed"};
  return phases[loom_serve_registry_state(model).phase];
}

uint64_t loom_serve_registry_preparation_duration_ns(
    const loom_serve_registered_model_t* model) {
  const loom_service_registration_snapshot_t state =
      loom_serve_registry_state(model);
  uint64_t duration =
      model->attachment.duration_ns + state.preparation_duration_ns;
  if (model->attachment.start_ns ||
      state.phase == LOOM_SERVICE_REGISTRATION_PREPARING) {
    const iree_time_t now = iree_time_now();
    if (model->attachment.start_ns) {
      duration += now - model->attachment.start_ns;
    } else if (state.preparation_start) {
      duration += now - state.preparation_start;
    }
  }
  return duration;
}

bool loom_serve_registry_counters(
    const loom_serve_registry_t* registry,
    loom_service_telemetry_counters_t* out_counters) {
  *out_counters = registry->retired.counters;
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    const loom_serve_service_t* service = &registry->models[i].service;
    if (!service->name.size) {
      continue;
    }
    if (!service->telemetry.counters) {
      return false;
    }
    loom_serve_registry_add_counters(out_counters, service->telemetry.counters);
  }
  return true;
}

bool loom_serve_registry_latency(
    const loom_serve_registry_t* registry,
    loom_service_telemetry_latency_t* out_latency) {
  *out_latency = registry->retired.latency;
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    const loom_serve_service_t* service = &registry->models[i].service;
    if (!service->name.size) {
      continue;
    }
    if (!service->telemetry.latency) {
      return false;
    }
    loom_service_telemetry_latency_merge(out_latency,
                                         service->telemetry.latency);
  }
  return true;
}
