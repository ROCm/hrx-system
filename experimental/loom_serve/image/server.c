// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/service.h"
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

static iree_status_t image_generate(void* self,
                                    const loom_serve_image_request_t* request,
                                    iree_const_byte_span_t* out_rgb) {
  return loom_serve_image_model_generate(
      self, iree_string_builder_view(&request->prompt), request->seed,
      request->strength, out_rgb);
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
  loom_serve_image_model_t* model = NULL;
  loom_serve_http_server_t* server = NULL;
  if (iree_status_is_ok(status)) {
    const loom_serve_image_model_options_t options = {
        .source_directory = iree_make_cstring_view(FLAG_model),
        .checkpoint_directory = iree_make_cstring_view(FLAG_checkpoint),
        .adapter_path = iree_make_cstring_view(FLAG_adapter),
        .height = (uint32_t)FLAG_height,
        .width = (uint32_t)FLAG_width,
        .text_tokens = (uint32_t)FLAG_text_tokens,
    };
    status = loom_serve_image_model_create(&options, &model, allocator);
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
    const loom_serve_image_generator_t generator = {model, image_generate};
    const loom_serve_image_service_options_t options = {
        .model = loom_serve_image_model_name(model),
        .width = (uint32_t)FLAG_width,
        .height = (uint32_t)FLAG_height,
        .adapter_enabled = FLAG_adapter[0] != 0,
        .pending_capacity = (iree_host_size_t)FLAG_pending_requests,
        .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000,
    };
    status =
        loom_serve_image_service_run(generator, server, &options, allocator);
  }
  status = iree_status_join(status, loom_serve_http_server_destroy(server));
  status = iree_status_join(status, loom_serve_image_model_destroy(model));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
