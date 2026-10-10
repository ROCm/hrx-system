// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/service.h"
#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/host_flags.h"
#include "experimental/loom_serve/runtime/signals.h"
#include "experimental/loom_serve/serving/router.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, model, "", "Live source catalog directory.");
IREE_FLAG(string, checkpoint, "", "Checkpoint asset directory.");
IREE_FLAG(string, adapter, "", "Optional model-specific adapter asset.");
IREE_FLAG(int32_t, height, 384,
          "Output pixel height; source validates supported geometry.");
IREE_FLAG(int32_t, width, 384,
          "Output pixel width; source validates supported geometry.");
IREE_FLAG(int32_t, text_tokens, 512,
          "Text capacity requested from the source model.");
IREE_FLAG(int32_t, port, 8080,
          "Loopback port; zero selects an ephemeral port.");
IREE_FLAG(int32_t, connections, 64, "Maximum simultaneous TCP connections.");
IREE_FLAG(int32_t, pending_requests, 32,
          "Maximum requests waiting for the model.");
IREE_FLAG(int32_t, request_body_bytes, 65536,
          "Maximum JSON request body bytes.");
IREE_FLAG(int32_t, heartbeat_ms, 1000,
          "Periodic image-state report interval; zero disables heartbeats.");

// One cold CLI locator, separate from model telemetry and serving workers.
static iree_status_t image_ready(loom_serve_http_listener_t* listener) {
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_format(loom_serve_http_listener_address(listener),
                                sizeof(storage), storage, &address));
  fprintf(stderr, "{\"event\":\"ready\",\"address\":\"%.*s\",\"models\":1}\n",
          (int)address.size, address.data);
  return iree_ok_status();
}

static void image_shutdown_requested(void* user_data,
                                     iree_async_signal_t signal) {
  (void)signal;
  loom_serve_http_server_request_stop(user_data);
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!FLAG_model[0] || !FLAG_checkpoint[0] || FLAG_port < 0 ||
      FLAG_port > 65535 || FLAG_connections < 1 || FLAG_pending_requests < 1 ||
      FLAG_request_body_bytes < 1 || FLAG_heartbeat_ms < 0) {
    fprintf(stderr,
            "Provide --model, --checkpoint, a valid port and positive "
            "capacities.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_async_signal_block_default();
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_ignore_broken_pipe();
  }
  loom_serve_image_model_t* model = NULL;
  loom_serve_device_group_t* group = NULL;
  loom_serve_device_t* device = NULL;
  loom_serve_host_t* host = NULL;
  loom_serve_http_server_t* server = NULL;
  loom_serve_http_listener_t* listener = NULL;
  loom_serve_signals_t* signals = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_host_create_from_flags(&host, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_create_from_flags(host, &group, allocator);
  }
  if (iree_status_is_ok(status)) {
    device = loom_serve_device_group_at(group, 0);
    const loom_serve_image_model_options_t options = {
        .source_directory = iree_make_cstring_view(FLAG_model),
        .checkpoint_directory = iree_make_cstring_view(FLAG_checkpoint),
        .adapter_path = iree_make_cstring_view(FLAG_adapter),
        .height = (uint32_t)FLAG_height,
        .width = (uint32_t)FLAG_width,
        .text_tokens = (uint32_t)FLAG_text_tokens,
    };
    status = loom_serve_image_model_create(device, &options, &model, allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_server_options_t options =
        loom_serve_http_server_options_default();
    options.connection_capacity = (iree_host_size_t)FLAG_connections;
    options.request_limits.body_byte_capacity =
        (iree_host_size_t)FLAG_request_body_bytes;
    status = loom_serve_http_server_create(loom_serve_host_proactor_pool(host),
                                           &options, &server, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_signals_create(
        loom_serve_host_proactor_pool(host),
        (iree_async_signal_callback_t){image_shutdown_requested, server},
        &signals, allocator);
  }
  if (iree_status_is_ok(status)) {
    char address[32];
    snprintf(address, sizeof(address), "127.0.0.1:%d", FLAG_port);
    loom_serve_http_listener_options_t options =
        loom_serve_http_listener_options_default();
    options.address = iree_make_cstring_view(address);
    options.backlog = FLAG_connections;
    status =
        loom_serve_http_listener_create(server, &options, &listener, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = image_ready(listener);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_image_generator_t generator =
        loom_serve_image_model_generator(model);
    const loom_serve_image_service_options_t options = {
        .name = loom_serve_image_model_name(model),
        .width = (uint32_t)FLAG_width,
        .height = (uint32_t)FLAG_height,
        .adapter_enabled = FLAG_adapter[0] != 0,
        .pending_capacity = (iree_host_size_t)FLAG_pending_requests,
        .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000,
    };
    loom_serve_image_service_t* service = NULL;
    status = loom_serve_image_service_create(host, generator, &options, NULL,
                                             &service, allocator);
    if (iree_status_is_ok(status)) {
      const loom_serve_http_service_t interface =
          loom_serve_image_service_interface(service);
      loom_serve_registry_t* registry = NULL;
      status = loom_serve_registry_create(
          1, &interface, 0, (loom_serve_registration_factory_t){0}, &registry,
          allocator);
      if (iree_status_is_ok(status)) {
        status = loom_serve_router_run(
            loom_serve_host_executor(host), server, registry,
            (loom_serve_router_options_t){
                .request_quota = options.pending_capacity,
                .holding_capacity = (iree_host_size_t)FLAG_connections / 2},
            allocator);
      }
      status = iree_status_join(status, loom_serve_registry_destroy(registry));
    }
    loom_serve_image_service_destroy(service);
  }
  loom_serve_signals_destroy(signals);
  status = iree_status_join(status, loom_serve_http_listener_destroy(listener));
  loom_serve_http_server_destroy(server);
  status = iree_status_join(status, loom_serve_image_model_destroy(model));
  status = iree_status_join(status, loom_serve_device_group_destroy(group));
  loom_serve_host_release(host);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
