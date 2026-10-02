// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/qwen_flags.h"
#include "experimental/loom_serve/qwen_service.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(
    int32_t, mtp_depth, 0,
    "Proposal depth: 0 or 3. Depth 0 with --mtp measures warm target-only.");
IREE_FLAG(string, scheduler, "packed",
          "packed, isolated, or matched (isolated with prefill decode math).");
IREE_FLAG(string, packing, "mixed",
          "mixed or separate prompt/decode cohorts, with the same kernels.");
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
  const iree_host_size_t epoch_count = loom_serve_qwen_shape_count_from_flags();
  if (FLAG_port < 0 || FLAG_port > 65535 || FLAG_rows < 1 || FLAG_rows > 8 ||
      FLAG_chunk_size < 0 || FLAG_max_tokens < 1 || FLAG_max_tokens > 16384 ||
      FLAG_heartbeat_ms < 0) {
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
  if (schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_PACKED && !epoch_count) {
    fprintf(stderr, "The packed scheduler requires --epoch.\n");
    return EXIT_FAILURE;
  }
  if ((FLAG_mtp_depth != 0 && FLAG_mtp_depth != 3) ||
      (FLAG_mtp_depth && !loom_serve_qwen_mtp_from_flags()) ||
      (loom_serve_qwen_mtp_from_flags() &&
       schedule_mode != LOOM_SERVE_QWEN_SCHEDULE_PACKED)) {
    fprintf(stderr,
            "mtp_depth must be 0 or 3; MTP requires --mtp and packed "
            "scheduling.\n");
    return EXIT_FAILURE;
  }
  loom_serve_qwen_packing_mode_t packing_mode;
  if (!strcmp(FLAG_packing, "mixed")) {
    packing_mode = LOOM_SERVE_QWEN_PACKING_MIXED;
  } else if (!strcmp(FLAG_packing, "separate")) {
    packing_mode = LOOM_SERVE_QWEN_PACKING_SEPARATE;
  } else {
    fprintf(stderr, "packing must be mixed or separate.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_async_signal_block_default();
  loom_serve_qwen_model_t* model = NULL;
  loom_serve_http_server_t* server = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_qwen_model_create_from_flags(
        (iree_host_size_t)FLAG_rows, allocator, &model);
  }
  iree_host_size_t chunk_size = 0;
  if (iree_status_is_ok(status)) {
    iree_host_size_t capacity = loom_serve_qwen_model_prefill_capacity(model);
    if (schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_PACKED) {
      capacity = 0;
      const loom_serve_qwen_shape_t* shapes =
          loom_serve_qwen_model_shapes(model);
      for (iree_host_size_t i = 0; i < epoch_count; ++i) {
        capacity = iree_max(capacity, shapes[i].token_capacity);
      }
    }
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
              "size\":%zu,\"scheduler\":\"%s\",\"shape_count\":%zu,"
              "\"packing\":\"%s\",\"mtp_warm\":%s,\"mtp_depth\":%d}\n",
              (int)address.size, address.data, FLAG_rows, chunk_size,
              FLAG_scheduler, epoch_count, FLAG_packing,
              loom_serve_qwen_mtp_from_flags() ? "true" : "false",
              FLAG_mtp_depth);
      const loom_serve_qwen_service_options_t service_options = {
          .row_count = (iree_host_size_t)FLAG_rows,
          .chunk_size = chunk_size,
          .default_max_tokens = (iree_host_size_t)FLAG_max_tokens,
          .heartbeat_interval = (iree_duration_t)FLAG_heartbeat_ms * 1000000,
          .schedule_mode = schedule_mode,
          .packing_mode = packing_mode,
          .mtp_depth = (iree_host_size_t)FLAG_mtp_depth,
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
