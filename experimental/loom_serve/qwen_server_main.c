// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/qwen_service.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, prefill, "", "Compiled prefill artifact directory.");
IREE_FLAG(string, decode, "", "Compiled decode artifact directory.");
IREE_FLAG(string, weights, "", "Canonical Qwen3.8-27B UD-Q5_K_XL GGUF path.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(int32_t, port, 8080,
          "Loopback TCP port; zero selects an ephemeral port.");
IREE_FLAG(int32_t, rows, 4, "Retained model rows (1-8).");
IREE_FLAG(int32_t, chunk_size, 0,
          "Prefill tokens per scheduling turn; zero uses compiled capacity.");
IREE_FLAG(int32_t, max_tokens, 512,
          "Default output token limit, including EOS.");

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!FLAG_prefill[0] || !FLAG_decode[0] || !FLAG_weights[0] ||
      !FLAG_tokenizer[0] || FLAG_port < 0 || FLAG_port > 65535 ||
      FLAG_rows < 1 || FLAG_rows > 8 || FLAG_chunk_size < 0 ||
      FLAG_max_tokens < 1 || FLAG_max_tokens > 16384) {
    fprintf(stderr,
            "Provide model paths, 1-8 rows, a valid port and positive token "
            "limits.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_async_signal_block_default();
  const loom_serve_qwen_options_t options = {
      .prefill_directory = iree_make_cstring_view(FLAG_prefill),
      .decode_directory = iree_make_cstring_view(FLAG_decode),
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
        loom_serve_qwen_model_prefill_capacity(model);
    chunk_size = FLAG_chunk_size ? (iree_host_size_t)FLAG_chunk_size : capacity;
    if (chunk_size > capacity) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "chunk_size exceeds compiled prefill capacity");
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
              "size\":%zu}\n",
              (int)address.size, address.data, FLAG_rows, chunk_size);
      status = loom_serve_qwen_service_run(
          model, server, (iree_host_size_t)FLAG_rows, chunk_size,
          (iree_host_size_t)FLAG_max_tokens, allocator);
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
