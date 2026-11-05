// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/registry.h"

#include "loom/runtime/wake.h"
#include "loom/serve/runtime/host/clock.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/service/text/checkpoint_control.h"
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
  registry->services[index] = (loom_runtime_service_t){
      .self = model,
      .vtable = &loom_serve_registry_service_vtable,
      .capacity_domain = model->service.runtime.capacity_domain,
  };
}

static void loom_serve_registry_set_phase(
    loom_serve_registered_model_t* model,
    loom_serve_registration_phase_t phase) {
  if (model->phase != phase &&
      (model->phase == LOOM_SERVE_REGISTRATION_PREPARING ||
       phase == LOOM_SERVE_REGISTRATION_PREPARING)) {
    const iree_time_t now = iree_time_now();
    if (model->phase == LOOM_SERVE_REGISTRATION_PREPARING) {
      model->preparation_duration_ns += now - model->preparation_start;
    } else {
      model->preparation_start = now;
    }
  }
  IREE_TRACE({
    if (model->phase != phase) {
      IREE_TRACE_SECTION_LEAVE(model->trace.lifecycle);
      model->trace.lifecycle = 0;
      if (phase == LOOM_SERVE_REGISTRATION_PREPARING) {
        model->trace.lifecycle = loom_service_trace_section_enter(
            LOOM_SERVICE_TRACE_SECTION_PREPARE, model->service.name);
      } else if (phase == LOOM_SERVE_REGISTRATION_RETIRING) {
        model->trace.lifecycle = loom_service_trace_section_enter(
            LOOM_SERVICE_TRACE_SECTION_RETIRE, model->service.name);
      }
    }
  });
  model->phase = phase;
}

static void loom_serve_registry_set_parameter_operation(
    loom_serve_registered_model_t* model,
    loom_serve_parameter_operation_t operation) {
  IREE_TRACE({
    const bool was_warmup =
        model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
        model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING;
    const bool is_warmup =
        operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
        operation == LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING;
    // Waiting and loading are successive parts of the same explicit warmup.
    if (model->parameter_operation != operation && !(was_warmup && is_warmup)) {
      IREE_TRACE_SECTION_LEAVE(model->trace.parameters);
      model->trace.parameters =
          operation != LOOM_SERVE_PARAMETER_OPERATION_NONE
              ? loom_service_trace_section_enter(
                    is_warmup ? LOOM_SERVICE_TRACE_SECTION_WARMUP
                              : LOOM_SERVICE_TRACE_SECTION_DEACTIVATE,
                    model->service.name)
              : 0;
    }
  });
  model->parameter_operation = operation;
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
  iree_host_size_t total_size = 0, models_offset = 0, services_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_registry_t), &total_size,
      IREE_STRUCT_FIELD(capacity, loom_serve_registered_model_t,
                        &models_offset),
      IREE_STRUCT_FIELD(capacity, loom_runtime_service_t, &services_offset)));
  loom_serve_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&registry));
  registry->host_allocator = host_allocator;
  registry->factory = factory;
  registry->models =
      (loom_serve_registered_model_t*)((uint8_t*)registry + models_offset);
  registry->services =
      (loom_runtime_service_t*)((uint8_t*)registry + services_offset);
  registry->capacity = capacity;
  registry->count = service_count;
  registry->next_incarnation = 1;
  registry->counters_since_ns = iree_time_now();
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    registry->models[i].registry = registry;
  }
  IREE_TRACE(loom_service_trace_initialize());
  const int64_t registered_unix_seconds = loom_serve_clock_now_unix_seconds();
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    registry->models[i].service = services[i];
    registry->models[i].phase = LOOM_SERVE_REGISTRATION_READY;
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
    if (model->registration.self || model->resolution) {
      if (!iree_status_is_ok(model->status)) {
        return iree_status_clone(model->status);
      }
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "registration still owns model resources");
    }
  }
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    loom_serve_registered_model_t* model = &registry->models[i];
    IREE_TRACE_SECTION_LEAVE(model->trace.lifecycle);
    IREE_TRACE_SECTION_LEAVE(model->trace.parameters);
    iree_status_free(model->status);
    loom_runtime_telemetry_stream_free(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
  }
  iree_allocator_free(registry->host_allocator, registry);
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
  return model->phase == LOOM_SERVE_REGISTRATION_READY
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
    if (!iree_status_is_ok(model->status)) {
      return iree_status_clone(model->status);
    }
    if (!(kinds & (1u << model->service.kind))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected model does not provide %s", operation);
    }
    if (model->phase != LOOM_SERVE_REGISTRATION_READY) {
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
      registry, name, 1u << LOOM_SERVE_MODEL_KIND_SPEECH, "speech", false,
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
      registry, name, 1u << LOOM_SERVE_MODEL_KIND_SPEECH, "voice upload", false,
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
      registry, name, 1u << LOOM_SERVE_MODEL_KIND_SPEECH, "voice removal", true,
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
  IREE_RETURN_IF_ERROR(loom_serve_registry_find_admission(
      registry, name,
      (1u << LOOM_SERVE_MODEL_KIND_RECOGNITION) |
          (1u << LOOM_SERVE_MODEL_KIND_ACTIVITY),
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
      registry, name, 1u << LOOM_SERVE_MODEL_KIND_TEXT, "text", false,
      &service));
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
      (1u << LOOM_SERVE_MODEL_KIND_TEXT) |
          (1u << LOOM_SERVE_MODEL_KIND_SEQUENCE),
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
    model->registered_unix_seconds = loom_serve_clock_now_unix_seconds();
    model->service.name = model->configuration.name;
    // Protocol metadata is known before source preparation, without model
    // access.
    model->service.kind = model->configuration.kind;
    model->service.endpoint =
        loom_serve_model_kind_endpoint(model->configuration.kind);
    model->service.telemetry.counters = &model->counters;
    model->service.telemetry.latency = &model->latency;
    model->service.telemetry.stream = model->stream;
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
  if (model->parameter_operation ==
      LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING) {
    loom_serve_registry_set_parameter_operation(
        model, LOOM_SERVE_PARAMETER_OPERATION_NONE);
  }
  if (mode > model->removal) {
    model->removal = mode;
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_RELEASE_FAILED) {
    loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_RETIRING);
  }
  return iree_ok_status();
}

const char* loom_serve_model_component_name(
    loom_serve_model_component_t component) {
  static const char* const names[] = {"primary", "image"};
  return names[component];
}

bool loom_serve_registry_has_component(
    const loom_serve_registered_model_t* model,
    loom_serve_model_component_t component) {
  return component == LOOM_SERVE_MODEL_COMPONENT_PRIMARY ||
         (model->configuration.kind == LOOM_SERVE_MODEL_KIND_TEXT &&
          (model->configuration.text.image.contract ||
           model->configuration.text.image.prepare.size != 0));
}

static iree_status_t loom_serve_registry_require_component(
    const loom_serve_registered_model_t* model,
    loom_serve_model_component_t component) {
  if (component != LOOM_SERVE_MODEL_COMPONENT_PRIMARY &&
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
    loom_serve_model_component_t component) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  if (!model->incarnation || !model->closed || model->removal ||
      (model->pinned_components & (1u << component)) ||
      !iree_status_is_ok(model->status) ||
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING ||
      (model->parameter_operation != LOOM_SERVE_PARAMETER_OPERATION_NONE &&
       model->parameter_component != component)) {
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
  const loom_serve_registration_parameters_t state =
      model->registration.parameters(model->registration.self, component);
  if (!state.deactivatable) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "component cannot release its parameter backing");
  }
  model->parameter_component = component;
  loom_serve_registry_set_parameter_operation(
      model, LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING);
  return iree_ok_status();
}

iree_status_t loom_serve_registry_set_pinned(
    loom_serve_registered_model_t* model,
    loom_serve_model_component_t component, bool pinned) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  if (!model->incarnation || model->removal ||
      !iree_status_is_ok(model->status) ||
      (pinned && model->parameter_component == component &&
       model->parameter_operation ==
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
    model->registration.set_pinned(model->registration.self, component, pinned);
  }
  if (pinned) {
    model->pinned_components |= 1u << component;
  } else {
    model->pinned_components &= ~(1u << component);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_registry_activate(
    loom_serve_registered_model_t* model,
    loom_serve_model_component_t component) {
  IREE_RETURN_IF_ERROR(loom_serve_registry_require_component(model, component));
  if (!model->incarnation || !model->closed || model->removal ||
      !iree_status_is_ok(model->status) ||
      model->parameter_operation ==
          LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING ||
      (model->parameter_operation != LOOM_SERVE_PARAMETER_OPERATION_NONE &&
       model->parameter_component != component)) {
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
    model->parameter_component = component;
    loom_serve_registry_set_parameter_operation(
        model, LOOM_SERVE_PARAMETER_OPERATION_WAITING);
  }
  return iree_ok_status();
}

loom_runtime_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model) {
  if (!iree_status_is_ok(model->status) ||
      model->removal == LOOM_SERVE_REMOVAL_ABORT) {
    return LOOM_RUNTIME_SERVICE_FLAG_STOPPING;
  }
  if (model->removal == LOOM_SERVE_REMOVAL_DRAIN) {
    if (!model->service.runtime.vtable->has_requests(
            model->service.runtime.self) &&
        !model->tokenizer_count) {
      return LOOM_RUNTIME_SERVICE_FLAG_STOPPING;
    }
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
  if (iree_status_is_ok(model->status)) {
    loom_serve_registry_record_error(model, status);
  }
  model->status = iree_status_join(model->status, status);
  return registry->factory.query_status
             ? registry->factory.query_status(registry->factory.self)
             : iree_ok_status();
}

iree_status_t loom_serve_registry_advance(loom_serve_registry_t* registry,
                                          iree_host_size_t index,
                                          loom_runtime_wake_callback_t observer,
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
      (model->configuration.sources.reference.path.size || !model->closed ||
       model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING)) {
    loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_PREPARING);
    if (model->configuration.sources.reference.path.size) {
      model->status = loom_serve_model_resolution_start(
          registry->factory.resolution_sink, &model->configuration, observer,
          registry->host_allocator, &model->resolution);
    } else {
      model->status = registry->factory.create(
          registry->factory.self, &model->configuration, observer,
          model->stream, registry->host_allocator, &model->registration);
    }
    if (!iree_status_is_ok(model->status)) {
      loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_FAILED);
      loom_serve_registry_set_parameter_operation(
          model, LOOM_SERVE_PARAMETER_OPERATION_NONE);
      loom_serve_registry_record_error(model, model->status);
    }
    progress = true;
  }
  if (model->resolution) {
    loom_serve_model_configuration_t resolved = {0};
    if (loom_serve_model_resolution_take(&model->resolution, &model->status,
                                         &resolved)) {
      if (iree_status_is_ok(model->status) && !model->removal) {
        loom_serve_model_configuration_deinitialize(&model->configuration);
        model->configuration = resolved;
        resolved = (loom_serve_model_configuration_t){0};
        model->service.name = model->configuration.name;
        if (!model->closed || model->parameter_operation ==
                                  LOOM_SERVE_PARAMETER_OPERATION_WAITING) {
          model->status = registry->factory.create(
              registry->factory.self, &model->configuration, observer,
              model->stream, registry->host_allocator, &model->registration);
        } else {
          // Resolve model facts while drained without preparing programs or
          // parameters. Component management can then use the declared roles.
          loom_serve_registry_set_phase(model,
                                        LOOM_SERVE_REGISTRATION_DECLARED);
        }
      }
      loom_serve_model_configuration_deinitialize(&resolved);
      if (model->removal || !iree_status_is_ok(model->status)) {
        loom_serve_registry_set_parameter_operation(
            model, LOOM_SERVE_PARAMETER_OPERATION_NONE);
        loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_FAILED);
        if (!iree_status_is_ok(model->status)) {
          loom_serve_registry_record_error(model, model->status);
        }
      }
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_PREPARING && !model->resolution) {
    loom_serve_registration_prepare_flags_t flags = 0;
    if (model->removal) {
      flags = LOOM_SERVE_REGISTRATION_PREPARE_FLAG_CANCEL;
    } else if ((!model->closed && model->parameter_demand_count) ||
               (model->parameter_operation ==
                    LOOM_SERVE_PARAMETER_OPERATION_WAITING &&
                model->parameter_component ==
                    LOOM_SERVE_MODEL_COMPONENT_PRIMARY)) {
      flags = LOOM_SERVE_REGISTRATION_PREPARE_FLAG_DEMAND;
    }
    bool complete = false;
    loom_serve_service_t service = {0};
    model->status =
        model->registration.prepare(model->registration.self, flags, &complete,
                                    &service, &model->compilation);
    if (complete) {
      if (iree_status_is_ok(model->status) && model->pinned_components &&
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
          for (loom_serve_model_component_t component = 0;
               component < LOOM_SERVE_MODEL_COMPONENT_COUNT; ++component) {
            if (loom_serve_registry_has_component(model, component)) {
              model->registration.set_pinned(
                  model->registration.self, component,
                  (model->pinned_components & (1u << component)) != 0);
            }
          }
        }
        model->service = service;
        registry->services[index].capacity_domain =
            service.runtime.capacity_domain;
        service.runtime.vtable->set_observer(service.runtime.self, observer);
        loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_READY);
      } else {
        loom_serve_registry_set_parameter_operation(
            model, LOOM_SERVE_PARAMETER_OPERATION_NONE);
        loom_serve_registry_record_error(model, model->status);
        if (registry->factory.query_status) {
          terminal_status =
              registry->factory.query_status(registry->factory.self);
        }
        loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_RETIRING);
      }
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      (model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING ||
       model->parameter_operation ==
           LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING)) {
    if (model->removal || !iree_status_is_ok(model->status) ||
        (!model->service.runtime.vtable->has_requests(
             model->service.runtime.self) &&
         !model->service.runtime.vtable->is_active(
             model->service.runtime.self))) {
      loom_serve_activation_phase_t phase = LOOM_SERVE_ACTIVATION_COMPLETE;
      iree_status_t component_status = iree_ok_status();
      iree_status_t status = model->registration.activate(
          model->registration.self, model->parameter_component,
          model->removal || !iree_status_is_ok(model->status)
              ? LOOM_SERVE_ACTIVATION_FLAG_CANCEL
              : 0,
          &phase, &component_status);
      if (!iree_status_is_ok(component_status)) {
        // The component retains its diagnostic and remains inspectable. Only
        // its affected operation fails unless shared backing was disturbed.
        loom_serve_registry_record_error(model, component_status);
        iree_status_free(component_status);
        if (registry->factory.query_status) {
          status = iree_status_join(
              status, registry->factory.query_status(registry->factory.self));
        }
      }
      const loom_serve_parameter_operation_t operation =
          phase == LOOM_SERVE_ACTIVATION_COMPLETE
              ? LOOM_SERVE_PARAMETER_OPERATION_NONE
          : phase == LOOM_SERVE_ACTIVATION_LOADING
              ? LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING
              : LOOM_SERVE_PARAMETER_OPERATION_WAITING;
      progress |= operation != model->parameter_operation;
      loom_serve_registry_set_parameter_operation(model, operation);
      if (!iree_status_is_ok(status)) {
        terminal_status = loom_serve_registry_fail(registry, index, status);
      }
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      (model->removal || !iree_status_is_ok(model->status)) &&
      model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_NONE) {
    if (!model->service.runtime.vtable->has_requests(
            model->service.runtime.self) &&
        !model->tokenizer_count && !model->held_count &&
        !model->service.runtime.vtable->is_pending(
            model->service.runtime.self)) {
      model->service.runtime.vtable->set_observer(
          model->service.runtime.self, (loom_runtime_wake_callback_t){0});
      loom_serve_registry_release_service(model);
      loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_RETIRING);
      progress = true;
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      model->parameter_operation ==
          LOOM_SERVE_PARAMETER_OPERATION_DEACTIVATING) {
    if (!model->service.runtime.vtable->has_requests(
            model->service.runtime.self) &&
        !model->service.runtime.vtable->is_active(
            model->service.runtime.self)) {
      bool complete = false;
      iree_status_t status = model->registration.deactivate(
          model->registration.self, model->parameter_component, &complete);
      if (!iree_status_is_ok(status)) {
        terminal_status = loom_serve_registry_fail(registry, index, status);
        complete = true;
      }
      if (complete) {
        loom_serve_registry_set_parameter_operation(
            model, LOOM_SERVE_PARAMETER_OPERATION_NONE);
        progress = true;
      }
    }
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_RETIRING) {
    bool complete = false;
    iree_status_t status =
        model->registration.retire(model->registration.self, &complete);
    if (!iree_status_is_ok(status)) {
      loom_serve_registry_record_error(model, status);
      model->status = iree_status_join(model->status, status);
      loom_serve_registry_set_phase(model,
                                    LOOM_SERVE_REGISTRATION_RELEASE_FAILED);
      progress = true;
    } else if (complete) {
      model->registration.destroy(model->registration.self);
      model->registration = (loom_serve_registration_t){0};
      loom_serve_registry_set_phase(model, LOOM_SERVE_REGISTRATION_FAILED);
      progress = true;
    }
  }
  if (model->phase != previous_phase) {
    loom_runtime_telemetry_record_t record = {
        .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_SNAPSHOT};
    loom_service_telemetry_snapshot_record_t telemetry_payload = {
        .counters = model->counters};
    const char* phase = loom_serve_registry_phase(model);
    memcpy(telemetry_payload.phase, phase, strlen(phase) + 1);
    memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
    loom_runtime_telemetry_stream_append(model->stream, &record);
  }
  if (model->removal && !model->resolution && !model->registration.self &&
      !model->held_count && !model->tokenizer_count) {
    IREE_TRACE_SECTION_LEAVE(model->trace.lifecycle);
    IREE_TRACE_SECTION_LEAVE(model->trace.parameters);
    loom_serve_registry_add_counters(&registry->retired.counters,
                                     &model->counters);
    loom_service_telemetry_latency_merge(&registry->retired.latency,
                                         &model->latency);
    iree_status_free(model->status);
    loom_runtime_telemetry_stream_free(model->stream);
    loom_serve_model_configuration_deinitialize(&model->configuration);
    memset(model, 0, sizeof(*model));
    model->registry = registry;
    registry->services[index] = (loom_runtime_service_t){0};
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
          model->service.runtime.vtable->is_pending(
              model->service.runtime.self));
}

static void loom_serve_registry_service_set_observer(
    void* self, loom_runtime_wake_callback_t observer) {
  loom_serve_registered_model_t* model = self;
  model->registry->wake = observer;
  if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
    model->service.runtime.vtable->set_observer(model->service.runtime.self,
                                                observer);
  }
}

static bool loom_serve_registry_service_has_requests(void* self) {
  const loom_serve_registered_model_t* model = self;
  return model->held_count || model->tokenizer_count ||
         (model->phase == LOOM_SERVE_REGISTRATION_READY &&
          model->service.runtime.vtable->has_requests(
              model->service.runtime.self));
}

static bool loom_serve_registry_service_is_pending(void* self) {
  const loom_serve_registered_model_t* model = self;
  return loom_serve_registry_pending(model) ||
         (model->removal == LOOM_SERVE_REMOVAL_ABORT &&
          model->phase != LOOM_SERVE_REGISTRATION_RELEASE_FAILED);
}

static bool loom_serve_registry_service_is_active(void* self) {
  const loom_serve_registered_model_t* model = self;
  // Retirement can release backing after model callbacks are gone. A failed
  // release is quiescent and cannot supply capacity to a waiting peer.
  return model->phase == LOOM_SERVE_REGISTRATION_RETIRING ||
         (model->phase == LOOM_SERVE_REGISTRATION_READY &&
          (model->service.runtime.vtable->is_active(
               model->service.runtime.self) ||
           model->parameter_operation ==
               LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING));
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
  bool stopping = iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  iree_status_t status = iree_ok_status();
  if (stopping && model->incarnation &&
      model->removal != LOOM_SERVE_REMOVAL_ABORT) {
    status = loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT);
  }
  if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
    const loom_runtime_service_t* service = &model->service.runtime;
    bool progress = false;
    iree_status_t service_status = service->vtable->advance(
        service->self, flags | loom_serve_registry_service_flags(model),
        &progress);
    if (!iree_status_is_ok(service_status)) {
      // Local failure starts model retirement. Application failure is latched
      // by the shared driver, which schedules its one complete stopping turn.
      progress |= model->incarnation && iree_status_is_ok(model->status);
      service_status =
          loom_serve_registry_fail(registry, index, service_status);
    }
    status = iree_status_join(status, service_status);
    stopping |= !iree_status_is_ok(status);
    *out_progress |= progress;
  }
  if (registry->observer.observe) {
    bool progress = false;
    status = iree_status_join(
        status, registry->observer.observe(
                    registry->observer.self, index,
                    stopping || !iree_status_is_ok(model->status), &progress));
    *out_progress |= progress;
  }
  bool progress = false;
  status = iree_status_join(
      status,
      loom_serve_registry_advance(registry, index, registry->wake, &progress));
  *out_progress |= progress;
  IREE_TRACE_ZONE_END(z_model);
  return status;
}

static const loom_runtime_service_vtable_t loom_serve_registry_service_vtable =
    {
        .set_observer = loom_serve_registry_service_set_observer,
        .has_requests = loom_serve_registry_service_has_requests,
        .is_pending = loom_serve_registry_service_is_pending,
        .is_active = loom_serve_registry_service_is_active,
        .advance = loom_serve_registry_service_advance,
};

const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model) {
  static const char* const phases[] = {"registered", "preparing",
                                       "ready",      "retiring",
                                       "failed",     "release_failed"};
  return phases[model->phase];
}

uint64_t loom_serve_registry_preparation_duration_ns(
    const loom_serve_registered_model_t* model) {
  return model->preparation_duration_ns +
         (model->phase == LOOM_SERVE_REGISTRATION_PREPARING
              ? iree_time_now() - model->preparation_start
              : 0);
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
