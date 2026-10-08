// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/service.h"
#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/host_flags.h"
#include "experimental/loom_serve/serving/configuration.h"
#include "experimental/loom_serve/text/schedule.h"
#include "experimental/loom_serve/text/service.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, models, "",
          "JSON deployment catalog with named source models.");
IREE_FLAG(int32_t, port, 8080,
          "Loopback TCP port; zero selects an ephemeral port.");
IREE_FLAG(int32_t, connections, 64, "Maximum simultaneous TCP connections.");
IREE_FLAG(int32_t, request_body_bytes, 8 * 1024 * 1024,
          "Maximum HTTP request body bytes per connection.");
IREE_FLAG(int32_t, heartbeat_ms, 1000,
          "Per-model JSONL heartbeat interval; zero disables reporting.");

// The corresponding catalog entry selects the initialized ownership pair.
typedef union serving_model_t {
  // Autoregressive deployment with model-local sessions and checkpoints.
  struct {
    // Owned source-JIT model borrowing the shared device.
    loom_serve_text_model_t* model;
    // Owned HTTP session state borrowing the model and deployment name.
    loom_serve_text_service_t* service;
  } text;
  // Finite image deployment with model-local request storage.
  struct {
    // Owned source-JIT model borrowing the shared device.
    loom_serve_image_model_t* model;
    // Owned HTTP admission borrowing the model and deployment name.
    loom_serve_image_service_t* service;
  } image;
} serving_model_t;

static iree_status_t serving_create_text(
    loom_serve_device_t* device,
    const loom_serve_model_configuration_t* configuration,
    serving_model_t* model, loom_serve_http_service_t* out_service,
    iree_allocator_t allocator) {
  const loom_serve_text_configuration_t* text = &configuration->text;
  loom_serve_packing_shape_t shapes[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY];
  const iree_host_size_t shape_count = loom_serve_text_default_shapes(
      text->rows, text->prefill_capacity, shapes);
  const loom_serve_text_options_t options = {
      .source_directory = configuration->source,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      .pool_capacity = text->pool_capacity,
      .epoch_count = shape_count,
      .epoch_shapes = shapes,
      .enable_mtp = text->mtp_depth != 0,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = text->rows,
      .checkpoint_capacity = text->checkpoint_capacity};
  IREE_RETURN_IF_ERROR(loom_serve_text_model_create(
      device, &options, &model->text.model, allocator));
  const loom_serve_text_service_options_t service_options = {
      .name = configuration->name,
      .row_count = text->rows,
      .chunk_size = text->prefill_capacity,
      .default_max_tokens = text->max_tokens,
      .pending_capacity = configuration->pending_requests,
      .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000,
      .schedule_mode = LOOM_SERVE_TEXT_SCHEDULE_PACKED,
      .packing_mode = LOOM_SERVE_TEXT_PACKING_MIXED,
      .mtp_depth = text->mtp_depth,
      .continuation_epochs = text->continuation_epochs};
  IREE_RETURN_IF_ERROR(loom_serve_text_service_create(
      model->text.model, &service_options, &model->text.service, allocator));
  *out_service = loom_serve_text_service_interface(model->text.service);
  return iree_ok_status();
}

static iree_status_t serving_create_image(
    loom_serve_device_t* device,
    const loom_serve_model_configuration_t* configuration,
    serving_model_t* model, loom_serve_http_service_t* out_service,
    iree_allocator_t allocator) {
  const loom_serve_image_configuration_t* image = &configuration->image;
  const loom_serve_image_model_options_t options = {
      .source_directory = configuration->source,
      .checkpoint_directory = image->checkpoint,
      .adapter_path = image->adapter,
      .height = image->height,
      .width = image->width,
      .text_tokens = image->text_tokens};
  IREE_RETURN_IF_ERROR(loom_serve_image_model_create(
      device, &options, &model->image.model, allocator));
  const loom_serve_image_service_options_t service_options = {
      .name = configuration->name,
      .width = image->width,
      .height = image->height,
      .adapter_enabled = image->adapter.size != 0,
      .pending_capacity = configuration->pending_requests,
      .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000};
  IREE_RETURN_IF_ERROR(loom_serve_image_service_create(
      loom_serve_device_host(device),
      loom_serve_image_model_generator(model->image.model), &service_options,
      &model->image.service, allocator));
  *out_service = loom_serve_image_service_interface(model->image.service);
  return iree_ok_status();
}

static iree_status_t serving_ready(loom_serve_http_server_t* server,
                                   iree_host_size_t model_count) {
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_format(loom_serve_http_server_address(server),
                                sizeof(storage), storage, &address));
  fprintf(stderr, "{\"event\":\"ready\",\"address\":\"%.*s\",\"models\":%zu}\n",
          (int)address.size, address.data, model_count);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!FLAG_models[0] || FLAG_port < 0 || FLAG_port > 65535 ||
      FLAG_connections < 1 || FLAG_request_body_bytes < 1 ||
      FLAG_heartbeat_ms < 0) {
    fprintf(stderr,
            "Provide --models, a valid port, and positive connection/body "
            "capacities.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  loom_serve_configuration_t configuration = {0};
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_models), allocator, &contents);
  if (iree_status_is_ok(status)) {
    const iree_const_byte_span_t bytes = contents->const_buffer;
    status = loom_serve_configuration_initialize(
        iree_make_string_view((const char*)bytes.data, bytes.data_length),
        &configuration, allocator);
  }
  iree_io_file_contents_free(contents);
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_block_default();
  }
  loom_serve_device_t* device = NULL;
  loom_serve_host_t* host = NULL;
  loom_serve_http_server_t* server = NULL;
  serving_model_t* models = NULL;
  loom_serve_http_service_t* services = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, configuration.model_count,
                                         sizeof(*models), (void**)&models);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, configuration.model_count,
                                         sizeof(*services), (void**)&services);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_host_create_from_flags(&host, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_create_from_flags(host, &device, allocator);
  }
  for (iree_host_size_t i = 0;
       i < configuration.model_count && iree_status_is_ok(status); ++i) {
    const loom_serve_model_configuration_t* entry = &configuration.models[i];
    status = entry->kind == LOOM_SERVE_MODEL_KIND_TEXT
                 ? serving_create_text(device, entry, &models[i], &services[i],
                                       allocator)
                 : serving_create_image(device, entry, &models[i], &services[i],
                                        allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_server_options_t options =
        loom_serve_http_server_options_default();
    options.port = (uint16_t)FLAG_port;
    options.connection_capacity = (iree_host_size_t)FLAG_connections;
    options.request_limits.body_byte_capacity =
        (iree_host_size_t)FLAG_request_body_bytes;
    status = loom_serve_http_server_create(&options, &server, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = serving_ready(server, configuration.model_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_router_run(
        loom_serve_host_executor(host), server, configuration.model_count,
        services, (iree_host_size_t)FLAG_connections, allocator);
  }
  const iree_status_code_t terminal_code = iree_status_code(status);
  for (iree_host_size_t i = 0; models && i < configuration.model_count; ++i) {
    if (configuration.models[i].kind == LOOM_SERVE_MODEL_KIND_TEXT) {
      loom_serve_text_service_destroy(models[i].text.service, terminal_code);
    } else {
      loom_serve_image_service_destroy(models[i].image.service);
    }
  }
  status = iree_status_join(status, loom_serve_http_server_destroy(server));
  for (iree_host_size_t i = 0; models && i < configuration.model_count; ++i) {
    status = iree_status_join(
        status, configuration.models[i].kind == LOOM_SERVE_MODEL_KIND_TEXT
                    ? loom_serve_text_model_destroy(models[i].text.model)
                    : loom_serve_image_model_destroy(models[i].image.model));
  }
  status = iree_status_join(status, loom_serve_device_destroy(device));
  loom_serve_host_release(host);
  iree_allocator_free(allocator, services);
  iree_allocator_free(allocator, models);
  loom_serve_configuration_deinitialize(&configuration);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
