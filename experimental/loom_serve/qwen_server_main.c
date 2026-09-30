// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/qwen_service.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, prefill, "", "Compiled prefill artifact directory.");
IREE_FLAG(string, decode, "", "Compiled decode artifact directory.");
IREE_FLAG(string, epoch, "", "Compiled packed epoch artifact directory.");
IREE_FLAG(string, scheduler, "packed",
          "packed, isolated, or matched (isolated with prefill decode math).");
IREE_FLAG(string, weights, "", "Canonical Qwen3.8-27B UD-Q5_K_XL GGUF path.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(int32_t, port, 8080,
          "Loopback TCP port; zero selects an ephemeral port.");
IREE_FLAG(int32_t, rows, 4, "Retained model rows (1-8).");
IREE_FLAG(int32_t, chunk_size, 0,
          "Prefill tokens per scheduling turn; zero uses compiled capacity.");
IREE_FLAG(int32_t, max_tokens, 512,
          "Default output token limit, including EOS.");
IREE_FLAG(int32_t, heartbeat_ms, 1000,
          "Periodic state report interval; zero disables heartbeats.");

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!FLAG_prefill[0] || !FLAG_decode[0] || !FLAG_weights[0] ||
      !FLAG_tokenizer[0] || FLAG_port < 0 || FLAG_port > 65535 ||
      FLAG_rows < 1 || FLAG_rows > 8 || FLAG_chunk_size < 0 ||
      FLAG_max_tokens < 1 || FLAG_max_tokens > 16384 || FLAG_heartbeat_ms < 0) {
    fprintf(stderr,
            "Provide model paths, 1-8 rows, a valid port and positive token "
            "limits.\n");
    return EXIT_FAILURE;
  }
  loom_serve_qwen_schedule_mode_t schedule_mode;
  if (!strcmp(FLAG_scheduler, "packed")) {
    schedule_mode = LOOM_SERVE_QWEN_SCHEDULE_PACKED;
  } else if (!strcmp(FLAG_scheduler, "isolated")) {
    schedule_mode = LOOM_SERVE_QWEN_SCHEDULE_ISOLATED;
  } else if (!strcmp(FLAG_scheduler, "matched")) {
    schedule_mode = LOOM_SERVE_QWEN_SCHEDULE_MATCHED;
  } else {
    fprintf(stderr, "scheduler must be packed, isolated, or matched.\n");
    return EXIT_FAILURE;
  }
  if (schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_PACKED && !FLAG_epoch[0]) {
    fprintf(stderr, "The packed scheduler requires --epoch.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_async_signal_block_default();
  const loom_serve_qwen_options_t options = {
      .prefill_directory = iree_make_cstring_view(FLAG_prefill),
      .decode_directory = iree_make_cstring_view(FLAG_decode),
      .epoch_directory = iree_make_cstring_view(FLAG_epoch),
      .weights_path = iree_make_cstring_view(FLAG_weights),
      .tokenizer_path = iree_make_cstring_view(FLAG_tokenizer),
      .row_count = (iree_host_size_t)FLAG_rows};
  loom_serve_qwen_model_t* model = NULL;
  loom_serve_http_server_t* server = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_qwen_model_create(&options, allocator, &model);
  }
  iree_host_size_t chunk_size = 0;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t capacity =
        schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_PACKED
            ? loom_serve_qwen_model_epoch_capacity(model)
            : loom_serve_qwen_model_prefill_capacity(model);
    chunk_size = FLAG_chunk_size ? (iree_host_size_t)FLAG_chunk_size : capacity;
    if (chunk_size > capacity) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "chunk_size exceeds compiled stage capacity");
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_http_server_create((uint16_t)FLAG_port, allocator, &server);
  }
  if (iree_status_is_ok(status)) {
    char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t address;
    status = iree_async_address_format(loom_serve_http_server_address(server),
                                       sizeof(storage), storage, &address);
    if (iree_status_is_ok(status)) {
      fprintf(stderr,
              "{\"event\":\"ready\",\"address\":\"%.*s\",\"rows\":%d,\"chunk_"
              "size\":%zu,\"scheduler\":\"%s\",\"epoch_capacity\":%zu,"
              "\"span_capacity\":%zu}\n",
              (int)address.size, address.data, FLAG_rows, chunk_size,
              FLAG_scheduler, loom_serve_qwen_model_epoch_capacity(model),
              loom_serve_qwen_model_span_capacity(model));
      const loom_serve_qwen_service_options_t service_options = {
          .row_count = (iree_host_size_t)FLAG_rows,
          .chunk_size = chunk_size,
          .default_max_tokens = (iree_host_size_t)FLAG_max_tokens,
          .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000,
          .schedule_mode = schedule_mode,
      };
      status = loom_serve_qwen_service_run(model, server, &service_options,
                                           allocator);
    }
  }
  status = iree_status_join(status, loom_serve_http_server_destroy(server));
  status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
