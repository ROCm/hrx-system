// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/host_flags.h"
#include "experimental/loom_serve/runtime/signals.h"
#include "experimental/loom_serve/serving/configuration.h"
#include "experimental/loom_serve/serving/model_registration.h"
#include "experimental/loom_serve/serving/router.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG(string, models, "",
          "Optional JSON deployment catalog; omitted starts with no models.");
IREE_FLAG(int32_t, model_capacity, 32,
          "Maximum registered models, including drained, failed and removing "
          "entries.");
IREE_FLAG_LIST(string, listen,
               "HTTP binding: IPv4 host:port, [IPv6]:port, or unix:path. "
               "Repeat for multiple listeners; default 127.0.0.1:8080.");
IREE_FLAG(int32_t, connections, 64,
          "Maximum simultaneous HTTP connections across all listeners.");
IREE_FLAG(int32_t, held_requests, -1,
          "Requests held behind drained model gates; -1 uses half the "
          "connection capacity. Must be below connection capacity.");
IREE_FLAG(int32_t, request_body_bytes, 8 * 1024 * 1024,
          "Maximum HTTP request body bytes per connection.");
IREE_FLAG(int32_t, heartbeat_ms, 1000,
          "Per-model JSONL heartbeat interval; zero disables reporting.");

static iree_status_t serving_ready(loom_serve_http_listener_t* listener,
                                   iree_host_size_t model_count) {
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_format(loom_serve_http_listener_address(listener),
                                sizeof(storage), storage, &address));
  loom_output_stream_t stream;
  loom_output_stream_for_file(stderr, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"event\":\"ready\",\"address\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, address));
  return loom_output_stream_write_format(&stream, ",\"models\":%zu}\n",
                                         model_count);
}

static void serving_shutdown_requested(void* user_data,
                                       iree_async_signal_t signal) {
  (void)signal;
  loom_serve_http_server_request_stop(user_data);
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_model_capacity < 1 || FLAG_connections < 1 ||
      FLAG_held_requests < -1 || FLAG_held_requests >= FLAG_connections ||
      FLAG_request_body_bytes < 1 || FLAG_heartbeat_ms < 0) {
    fprintf(
        stderr,
        "Provide positive model/connection/body capacities and a nonnegative "
        "heartbeat interval; held_requests must be -1 or below connection "
        "capacity.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  const iree_flag_string_list_t bindings = FLAG_listen_list();
  const iree_string_view_t default_binding = IREE_SV("127.0.0.1:8080");
  const iree_host_size_t listener_count = bindings.count ? bindings.count : 1;
  const iree_string_view_t* addresses =
      bindings.count ? bindings.values : &default_binding;
  loom_serve_configuration_t configuration = {.allocator = allocator};
  iree_status_t status = iree_ok_status();
  if (FLAG_models[0]) {
    iree_io_file_contents_t* contents = NULL;
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_models),
                                        allocator, &contents);
    if (iree_status_is_ok(status)) {
      const iree_const_byte_span_t bytes = contents->const_buffer;
      status = loom_serve_configuration_initialize(
          iree_make_string_view((const char*)bytes.data, bytes.data_length),
          &configuration, allocator);
    }
    iree_io_file_contents_free(contents);
  }
  if (iree_status_is_ok(status) &&
      configuration.model_count > (iree_host_size_t)FLAG_model_capacity) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "startup catalog exceeds model_capacity");
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_block_default();
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_ignore_broken_pipe();
  }
  loom_serve_device_group_t* group = NULL;
  loom_serve_device_t* device = NULL;
  loom_serve_host_t* host = NULL;
  loom_serve_http_server_t* server = NULL;
  loom_serve_http_listener_t** listeners = NULL;
  loom_serve_signals_t* signals = NULL;
  loom_serve_registry_t* registry = NULL;
  loom_serve_model_registration_options_t registration_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        allocator, listener_count, sizeof(*listeners), (void**)&listeners);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_host_create_from_flags(&host, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_create_from_flags(host, &group, allocator);
  }
  if (iree_status_is_ok(status)) {
    device = loom_serve_device_group_at(group, 0);
  }
  if (iree_status_is_ok(status)) {
    registration_options.device = device;
    registration_options.heartbeat_interval =
        (iree_duration_t)FLAG_heartbeat_ms * 1000000;
    status = loom_serve_registry_create(
        0, NULL, (iree_host_size_t)FLAG_model_capacity,
        loom_serve_model_registration_factory(&registration_options), &registry,
        allocator);
  }
  for (iree_host_size_t i = 0;
       i < configuration.model_count && iree_status_is_ok(status); ++i) {
    loom_serve_model_configuration_t* entry = &configuration.models[i];
    iree_host_size_t index = 0;
    status = loom_serve_registry_register(registry, entry, &index);
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
        (iree_async_signal_callback_t){serving_shutdown_requested, server},
        &signals, allocator);
  }
  for (iree_host_size_t i = 0; i < listener_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_http_listener_options_t options =
        loom_serve_http_listener_options_default();
    options.address = addresses[i];
    options.backlog = FLAG_connections;
    status = loom_serve_http_listener_create(server, &options, &listeners[i],
                                             allocator);
  }
  for (iree_host_size_t i = 0; i < listener_count && iree_status_is_ok(status);
       ++i) {
    status = serving_ready(listeners[i], configuration.model_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_router_run(
        loom_serve_host_executor(host), server, registry,
        (loom_serve_router_options_t){
            .request_quota = (iree_host_size_t)FLAG_connections,
            .resources = {.device_group = group,
                          .listener_count = listener_count,
                          .listeners = listeners},
            .holding_capacity = (iree_host_size_t)(FLAG_held_requests < 0
                                                       ? FLAG_connections / 2
                                                       : FLAG_held_requests)},
        allocator);
  }
  loom_serve_signals_destroy(signals);
  for (iree_host_size_t i = 0; listeners && i < listener_count; ++i) {
    status = iree_status_join(status,
                              loom_serve_http_listener_destroy(listeners[i]));
  }
  iree_allocator_free(allocator, listeners);
  loom_serve_http_server_destroy(server);
  iree_status_t registry_status = loom_serve_registry_destroy(registry);
  bool retained_resources = !iree_status_is_ok(registry_status);
  status = iree_status_join(status, registry_status);
  // A failed platform release retains its parents until process exit. It does
  // not authorize destroying a shared device or invalidating borrowed names.
  if (!retained_resources) {
    iree_status_t group_status = loom_serve_device_group_destroy(group);
    retained_resources = !iree_status_is_ok(group_status);
    status = iree_status_join(status, group_status);
  }
  if (!retained_resources) {
    loom_serve_host_release(host);
    loom_serve_configuration_deinitialize(&configuration);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
