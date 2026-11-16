// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/model_registration.h"

#include "loom/runtime/storage/memory.h"
#include "loom/serve/service/finite/http.h"
#include "loom/serve/service/image/registration.h"
#include "loom/serve/service/recognition/registration.h"
#include "loom/serve/service/speech/registration.h"
#include "loom/serve/service/text/cache_policy.h"
#include "loom/serve/service/text/registration.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/service/finite/decision/protocol.h"
#include "loom/service/finite/embedding/protocol.h"
#include "loom/service/finite/registration.h"

static loom_serve_service_t loom_serve_finite_registration_interface(
    loom_service_registration_t* registration) {
  return loom_serve_finite_http_interface(
      loom_service_finite_registration_service(registration));
}

static iree_status_t loom_serve_finite_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  const loom_service_finite_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options,
      .specialization_cache = configuration->specialization_cache};
  const loom_service_finite_registration_options_t service_options = {
      .incarnation = incarnation,
      .protocol = configuration->kind == LOOM_SERVICE_KIND_EMBEDDING
                      ? loom_service_embedding_protocol()
                      : loom_service_decision_protocol(),
      .name = configuration->name,
      .capacity = configuration->pending_requests,
      .variant_capacity = configuration->variant_capacity,
      .heartbeat_interval = options->heartbeat_interval,
      .stream = stream};
  loom_service_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(loom_service_finite_registration_allocate(
      loom_runtime_device_group_at(options->device_group,
                                   configuration->device_index),
      loom_runtime_device_group_pool_at(options->device_group,
                                        configuration->pool_index),
      &model_options, &service_options, observer, host_allocator,
      &registration));
  *out_registration = (loom_serve_registration_t){
      .native = registration,
      .service = loom_serve_finite_registration_interface};
  return iree_ok_status();
}

static loom_serve_service_t loom_serve_image_registration_interface(
    loom_service_registration_t* registration) {
  return loom_serve_image_service_interface(
      loom_serve_image_registration_service(registration));
}

static iree_status_t loom_serve_image_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
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
  const loom_serve_image_service_options_t service_options = {
      .name = configuration->name,
      .width = configuration->image.width,
      .height = configuration->image.height,
      .pending_capacity = configuration->pending_requests,
      .variant_capacity = configuration->variant_capacity,
      .heartbeat_interval = options->heartbeat_interval};
  loom_service_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_image_registration_allocate(
      loom_runtime_device_group_at(options->device_group,
                                   configuration->device_index),
      loom_runtime_device_group_pool_at(options->device_group,
                                        configuration->pool_index),
      &model_options, &service_options, incarnation, observer, stream,
      host_allocator, &registration));
  *out_registration = (loom_serve_registration_t){
      .native = registration,
      .service = loom_serve_image_registration_interface};
  return iree_ok_status();
}

static loom_serve_service_t loom_serve_speech_registration_interface(
    loom_service_registration_t* registration) {
  return loom_serve_speech_service_interface(
      loom_serve_speech_registration_service(registration));
}

static iree_status_t loom_serve_speech_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  const loom_serve_speech_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options};
  const loom_serve_speech_service_options_t service_options = {
      .name = configuration->name,
      .pending_capacity = configuration->pending_requests,
      .heartbeat_interval = options->heartbeat_interval};
  loom_service_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_speech_registration_allocate(
      loom_runtime_device_group_at(options->device_group,
                                   configuration->device_index),
      loom_runtime_device_group_pool_at(options->device_group,
                                        configuration->pool_index),
      &model_options, &service_options, incarnation, observer, stream,
      host_allocator, &registration));
  *out_registration = (loom_serve_registration_t){
      .native = registration,
      .service = loom_serve_speech_registration_interface};
  return iree_ok_status();
}

static loom_serve_service_t loom_serve_recognition_registration_interface(
    loom_service_registration_t* registration) {
  return loom_serve_recognition_service_interface(
      loom_serve_recognition_registration_service(registration));
}

static iree_status_t loom_serve_recognition_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  const loom_serve_recognition_model_options_t model_options = {
      .sources = configuration->sources,
      .asset_directory = configuration->assets,
      .options = configuration->options};
  const loom_serve_recognition_service_options_t service_options = {
      .name = configuration->name,
      .kind = configuration->kind,
      .pending_capacity = configuration->pending_requests,
      .request_limit = 64 * 1024,
      .document_limit = 1024 * 1024,
      .decode_frames = 4096,
      .conversion_limit = 8 * 1024 * 1024};
  loom_service_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_recognition_registration_allocate(
      loom_runtime_device_group_at(options->device_group,
                                   configuration->device_index),
      loom_runtime_device_group_pool_at(options->device_group,
                                        configuration->pool_index),
      &model_options, &service_options, incarnation, observer, stream,
      host_allocator, &registration));
  *out_registration = (loom_serve_registration_t){
      .native = registration,
      .service = loom_serve_recognition_registration_interface};
  return iree_ok_status();
}

static loom_serve_service_t loom_serve_text_registration_interface(
    loom_service_registration_t* registration) {
  return loom_serve_text_service_interface(
      loom_serve_text_registration_service(registration));
}

static iree_status_t loom_serve_text_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  const loom_serve_model_registration_options_t* options = self;
  const loom_serve_text_configuration_t* text = &configuration->text;
  loom_runtime_pool_t* pool = loom_runtime_device_group_pool_at(
      options->device_group, configuration->pool_index);
  loom_runtime_memory_pool_t* memory_pool = loom_runtime_pool_memory(pool);
  const uint64_t cache_limit_bytes =
      memory_pool ? loom_runtime_memory_pool_limit(memory_pool) : 0;
  const iree_host_size_t shape_count =
      loom_serve_text_default_shapes(text->prefill_capacity, NULL);
  loom_serve_packing_shape_t* shapes = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, shape_count, sizeof(*shapes), (void**)&shapes));
  loom_serve_text_default_shapes(text->prefill_capacity, shapes);
  const loom_serve_text_options_t model_options = {
      .mode = configuration->kind == LOOM_SERVICE_KIND_SEQUENCE
                  ? LOOM_SERVE_TEXT_MODE_ENCODE
                  : LOOM_SERVE_TEXT_MODE_GENERATE,
      .sources = configuration->sources,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      .pool_capacity = memory_pool ? 0 : text->context_capacity,
      .cache_limit_bytes = cache_limit_bytes,
      .epoch_count = shape_count,
      .epoch_shapes = shapes,
      .draft_depth = text->draft_depth,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = text->rows,
      .checkpoint_capacity = text->checkpoint_capacity,
      .cache = {.target_bytes =
                    text->cache_target_bytes == IREE_HOST_SIZE_MAX
                        ? loom_serve_text_cache_default_target(
                              memory_pool
                                  ? loom_runtime_memory_pool_limit(memory_pool)
                                  : 0)
                        : text->cache_target_bytes,
                .session_capacity = text->session_cache_capacity,
                .prefix_capacity = text->prefix_cache_capacity},
      .options = configuration->options};
  const loom_serve_text_service_options_t service_options = {
      .name = configuration->name,
      .row_count = text->rows,
      .max_sessions = text->max_sessions,
      .variant_capacity = configuration->variant_capacity,
      .chunk_size = text->prefill_capacity,
      .default_max_tokens = text->max_tokens,
      .pending_capacity = configuration->pending_requests,
      .session_cache_capacity = text->session_cache_capacity,
      .prefix_cache_capacity = text->prefix_cache_capacity,
      .admission_overtakes = text->admission_overtakes,
      .heartbeat_interval = options->heartbeat_interval,
      .schedule_mode = LOOM_SERVICE_TEXT_SCHEDULE_PACKED,
      .packing_mode = LOOM_SERVICE_TEXT_PACKING_MIXED,
      .continuation_epochs = text->continuation_epochs};
  const bool images = text->image.contract || text->image.prepare.size;
  loom_service_finite_model_options_t image_options = {
      .sources = configuration->sources,
      .asset_directory = text->image.assets,
      .options = text->image.options,
      .specialization_cache = text->image.specialization_cache};
  image_options.sources.component = text->image.contract;
  image_options.sources.prepare = text->image.prepare;
  image_options.sources.control = text->image.control;
  loom_service_registration_t* registration = NULL;
  iree_status_t status = loom_serve_text_registration_allocate(
      loom_runtime_device_group_at(options->device_group,
                                   configuration->device_index),
      pool, &model_options, &service_options, images ? &image_options : NULL,
      incarnation, observer, stream, host_allocator, &registration);
  iree_allocator_free(host_allocator, shapes);
  if (iree_status_is_ok(status)) {
    *out_registration = (loom_serve_registration_t){
        .native = registration,
        .service = loom_serve_text_registration_interface};
  }
  return status;
}

static iree_status_t loom_serve_model_registration_create(
    void* self, const loom_serve_model_configuration_t* configuration,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_registration_t* out_registration) {
  static iree_status_t (*const create[])(
      void*, const loom_serve_model_configuration_t*, uint64_t,
      loom_runtime_wake_callback_t, loom_runtime_telemetry_stream_t*,
      iree_allocator_t, loom_serve_registration_t*) = {
      [LOOM_SERVICE_KIND_TEXT] = loom_serve_text_registration_create,
      [LOOM_SERVICE_KIND_SEQUENCE] = loom_serve_text_registration_create,
      [LOOM_SERVICE_KIND_IMAGE] = loom_serve_image_registration_create,
      [LOOM_SERVICE_KIND_EMBEDDING] = loom_serve_finite_registration_create,
      [LOOM_SERVICE_KIND_DECISION] = loom_serve_finite_registration_create,
      [LOOM_SERVICE_KIND_SPEECH] = loom_serve_speech_registration_create,
      [LOOM_SERVICE_KIND_RECOGNITION] =
          loom_serve_recognition_registration_create,
      [LOOM_SERVICE_KIND_ACTIVITY] = loom_serve_recognition_registration_create,
  };
  return create[configuration->kind](self, configuration, incarnation, observer,
                                     stream, host_allocator, out_registration);
}

static iree_status_t loom_serve_model_registration_query_status(void* self) {
  const loom_serve_model_registration_options_t* options = self;
  const iree_host_size_t pool_count =
      loom_runtime_device_group_pool_count(options->device_group);
  for (iree_host_size_t i = 0; i < pool_count; ++i) {
    loom_runtime_memory_pool_t* pool = loom_runtime_pool_memory(
        loom_runtime_device_group_pool_at(options->device_group, i));
    if (pool) {
      IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_query_status(pool));
    }
  }
  return iree_ok_status();
}

loom_serve_registration_factory_t loom_serve_model_registration_factory(
    const loom_serve_model_registration_options_t* options) {
  return (loom_serve_registration_factory_t){
      .self = (void*)options,
      .kinds =
          (1u << LOOM_SERVICE_KIND_EMBEDDING) |
          (1u << LOOM_SERVICE_KIND_DECISION) | (1u << LOOM_SERVICE_KIND_IMAGE) |
          (1u << LOOM_SERVICE_KIND_SPEECH) |
          (1u << LOOM_SERVICE_KIND_RECOGNITION) |
          (1u << LOOM_SERVICE_KIND_ACTIVITY) |
          (1u << LOOM_SERVICE_KIND_SEQUENCE) | (1u << LOOM_SERVICE_KIND_TEXT),
      .device_names = loom_runtime_device_group_names(options->device_group),
      .pool_count = loom_runtime_device_group_pool_count(options->device_group),
      .pools = loom_runtime_device_group_pool_bindings(options->device_group),
      .resolution_sink =
          loom_runtime_host_compiler_sink(loom_runtime_device_host(
              loom_runtime_device_group_at(options->device_group, 0))),
      .create = loom_serve_model_registration_create,
      .query_status = loom_serve_model_registration_query_status};
}
