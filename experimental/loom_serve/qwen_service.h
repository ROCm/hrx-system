// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_

#include "experimental/loom_serve/http_server.h"
#include "experimental/loom_serve/qwen_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_qwen_schedule_mode_e {
  // One shared weight traversal for all admitted prompt and decode spans.
  LOOM_SERVE_QWEN_SCHEDULE_PACKED = 0,
  // Same ready-span partition, executed with ordinary isolated model stages.
  LOOM_SERVE_QWEN_SCHEDULE_ISOLATED,
  // Isolated prefill math even for decode, for a same-math packed control.
  LOOM_SERVE_QWEN_SCHEDULE_MATCHED,
} loom_serve_qwen_schedule_mode_t;

typedef struct loom_serve_qwen_service_options_t {
  // Number of retained rows, matching the model residency.
  iree_host_size_t row_count;
  // Maximum known input tokens admitted from one row in an epoch.
  iree_host_size_t chunk_size;
  // Output bound when a request omits max_tokens.
  iree_host_size_t default_max_tokens;
  // Reporting interval in nanoseconds; zero disables periodic heartbeats.
  iree_duration_t heartbeat_interval;
  // Execution choice; the planner and HTTP lifecycle are shared by all modes.
  loom_serve_qwen_schedule_mode_t schedule_mode;
} loom_serve_qwen_service_options_t;

// Runs one application owner until transport shutdown or model failure. Model
// and transport are borrowed. Each epoch gathers credited ready rows, executes
// their known spans and commits outputs before reusing the shared workspace.
// Heartbeats observe a copied snapshot and continue during model waits.
// X-Loom-Session selects retained state; idle rows are an LRU prefix cache, not
// durable sessions. Busy named sessions reject concurrent requests. Untagged
// requests always replay. Peer cancellation discards its checkpoint at a
// completed stage boundary. Return relinquishes every connection view before
// the caller destroys transport/model. Capacities match the created model.
iree_status_t loom_serve_qwen_service_run(
    loom_serve_qwen_model_t* model, loom_serve_http_server_t* server,
    const loom_serve_qwen_service_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_
