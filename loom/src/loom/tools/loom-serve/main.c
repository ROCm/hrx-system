// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/base/tracing.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/tooling/cli/help.h"
#include "loom/tools/loom-serve/options.h"
#include "loom/tools/loom-serve/signals.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG_LIST(
    string, model,
    "Model JSON document; repeat to register multiple models. "
    "Omit to start empty. Relative arguments use the working directory; "
    "paths within each document use its directory.");
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

static iree_status_t loom_serve_main_ready(loom_serve_http_listener_t* listener,
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

static void loom_serve_main_shutdown_requested(void* user_data,
                                               iree_async_signal_t signal) {
  loom_serve_http_server_request_stop(user_data);
}

static int loom_serve_main_agents_markdown(void) {
  return fputs(
             "# loom-serve\n\n"
             "Runs source-defined text, image, embedding and decision models "
             "in one process. Numerical programs and parameter files are "
             "external inputs, not compiled-in model implementations.\n\n"
             "## Start and inspect\n\n"
             "```sh\n"
             "loom-serve --device=task --pool_backing=fixed "
             "--listen=127.0.0.1:8080\n"
             "curl http://127.0.0.1:8080/v1/status\n"
             "```\n\n"
             "At least one explicit `--device=URI` is required. The linked "
             "HAL drivers resolve URIs; the server adds no implicit device. "
             "Repeated devices form one ordered group. A model document "
             "needs a top-level `device` index when that group has more than "
             "one member. Fixed backing is suitable for the task and Vulkan "
             "drivers; elastic backing requires virtual-memory support.\n\n"
             "## Register models\n\n"
             "Repeated `--model=path/model.json` values register startup "
             "deployments. Omitting them starts an empty server. The same "
             "JSON object is accepted by `POST /v1/models/register`: "
             "`name`, `definition`, and optional `device`, `admission` and "
             "`base_directory`. Admission defaults to `open`; `drained` "
             "registers metadata without loading programs or weights.\n\n"
             "A definition names its modality, preparation/control programs, "
             "source providers, assets and capacities. Program inputs may "
             "be `.loom` or `.loombc`. File-document paths use the document "
             "directory; relative HTTP references need an explicit absolute "
             "`base_directory`. This is a source-model contract, not an "
             "automatic checkpoint converter.\n\n"
             "`/healthz` and `/readyz` report server availability. "
             "`/v1/status` reports registration progress and failures; "
             "`/v1/models` describes prepared deployments and capabilities. "
             "Registration's HTTP 202 acknowledges intent, not readiness. "
             "Inference can wait behind preparation using bounded admission "
             "credit. `/metrics`, `/logz` and `/logz/stream` expose "
             "counters and independent-reader telemetry.\n\n"
             "Drain/undrain controls admission independently of model "
             "residency. Unregister supports drain or abort; accepted device "
             "work retires before resources are released. Multiple TCP or "
             "`unix:path` listeners share the same server. HTTP has no "
             "built-in authentication: trusted clients and model programs "
             "are required, with deployment-owned isolation or tunnels.\n\n"
             "`--help` lists serving flags; `--help=all` adds linked runtime "
             "flags. `--flagfile=path` uses the ordinary newline-separated "
             "flag syntax and working-directory path resolution. There is "
             "no subcommand.\n\n"
             "## References\n\n"
             "- First output: "
             "https://rocm.github.io/hrx-system/loom/workflows/serving/"
             "serve-models/\n"
             "- Command options: "
             "https://rocm.github.io/hrx-system/loom/reference/loom-serve/\n"
             "- Model documents: "
             "https://rocm.github.io/hrx-system/loom/reference/serving-models/"
             "\n",
             stdout) == EOF
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}

static int loom_serve_main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-serve",
      "Serves source-defined models over HTTP with shared devices and "
      "workers.\n"
      "\n"
      "Usage:\n"
      "  loom-serve --device=URI [--model=path/model.json ...] [options]\n"
      "  loom-serve --device=task --pool_backing=fixed "
      "--listen=127.0.0.1:8080\n"
      "  loom-serve --flagfile=serve.flags\n"
      "  loom-serve --agents_md\n"
      "\n"
      "At least one --device is required; repeat it to form the device group.\n"
      "Omit --model to start empty and register models over HTTP. Startup and\n"
      "HTTP registration use the same JSON model document; .loom and .loombc\n"
      "programs are loaded at runtime. No model implementation is built in.\n"
      "Repeated --listen bindings share one server. /v1/status reports model\n"
      "readiness; /v1/models lists prepared deployments and capabilities.\n"
      "HTTP and model loading assume trusted clients and source packages.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      return loom_serve_main_agents_markdown();
    }
  }
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc > 1) {
    fprintf(stderr,
            "Unexpected positional argument '%s'; use --model=path or "
            "--flagfile=path.\n",
            argv[1]);
    return EXIT_FAILURE;
  }
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
  const iree_allocator_t host_allocator = iree_allocator_system();
  const iree_flag_string_list_t bindings = FLAG_listen_list();
  const iree_string_view_t default_binding = IREE_SV("127.0.0.1:8080");
  const iree_host_size_t listener_count = bindings.count ? bindings.count : 1;
  const iree_string_view_t* addresses =
      bindings.count ? bindings.values : &default_binding;
  const iree_flag_string_list_t models = FLAG_model_list();
  loom_serve_model_configuration_t* configurations = NULL;
  iree_status_t status = iree_ok_status();
  if (models.count > (iree_host_size_t)FLAG_model_capacity) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "startup models exceed model_capacity");
  }
  if (iree_status_is_ok(status) && models.count) {
    status = iree_allocator_malloc_array(host_allocator, models.count,
                                         sizeof(*configurations),
                                         (void**)&configurations);
  }
  for (iree_host_size_t i = 0; i < models.count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_model_configuration_load(
        models.values[i], host_allocator, &configurations[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_block_default();
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_ignore_broken_pipe();
  }
  loom_serve_device_group_t* group = NULL;
  loom_serve_host_t* host = NULL;
  loom_serve_http_server_t* server = NULL;
  loom_serve_http_listener_t** listeners = NULL;
  loom_serve_signals_t* signals = NULL;
  loom_serve_registry_t* registry = NULL;
  loom_serve_model_registration_options_t registration_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, listener_count, sizeof(*listeners), (void**)&listeners);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_host_create_from_flags(host_allocator, &host);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_allocate_from_flags(host, host_allocator,
                                                         &group);
  }
  if (iree_status_is_ok(status)) {
    registration_options.device_group = group;
    registration_options.heartbeat_interval =
        (iree_duration_t)FLAG_heartbeat_ms * 1000000;
    status = loom_serve_registry_allocate(
        0, NULL, (iree_host_size_t)FLAG_model_capacity,
        loom_serve_model_registration_factory(&registration_options),
        host_allocator, &registry);
  }
  for (iree_host_size_t i = 0; i < models.count && iree_status_is_ok(status);
       ++i) {
    loom_serve_model_configuration_t* entry = &configurations[i];
    iree_host_size_t index = 0;
    status = loom_serve_registry_register(registry, entry, &index);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_server_options_t options =
        loom_serve_http_server_options_default();
    options.connection_capacity = (iree_host_size_t)FLAG_connections;
    options.request_limits.body_byte_capacity =
        (iree_host_size_t)FLAG_request_body_bytes;
    status = loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host), &options, host_allocator, &server);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_signals_allocate(
        loom_serve_host_proactor_pool(host),
        (iree_async_signal_callback_t){loom_serve_main_shutdown_requested,
                                       server},
        host_allocator, &signals);
  }
  for (iree_host_size_t i = 0; i < listener_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_http_listener_options_t options =
        loom_serve_http_listener_options_default();
    options.address = addresses[i];
    options.backlog = FLAG_connections;
    status = loom_serve_http_listener_allocate(server, &options, host_allocator,
                                               &listeners[i]);
  }
  for (iree_host_size_t i = 0; i < listener_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_main_ready(listeners[i], models.count);
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
        host_allocator);
  }
  loom_serve_signals_free(signals);
  for (iree_host_size_t i = 0; listeners && i < listener_count; ++i) {
    status =
        iree_status_join(status, loom_serve_http_listener_free(listeners[i]));
  }
  iree_allocator_free(host_allocator, listeners);
  loom_serve_http_server_free(server);
  iree_status_t registry_status = loom_serve_registry_free(registry);
  bool retained_resources = !iree_status_is_ok(registry_status);
  status = iree_status_join(status, registry_status);
  // A failed platform release retains its parents until process exit. It does
  // not authorize destroying a shared device or invalidating borrowed names.
  if (!retained_resources) {
    iree_status_t group_status = loom_serve_device_group_free(group);
    retained_resources = !iree_status_is_ok(group_status);
    status = iree_status_join(status, group_status);
  }
  if (!retained_resources) {
    loom_serve_host_release(host);
  }
  for (iree_host_size_t i = 0; configurations && i < models.count; ++i) {
    loom_serve_model_configuration_deinitialize(&configurations[i]);
  }
  iree_allocator_free(host_allocator, configurations);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  int exit_code = loom_serve_main(argc, argv);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
