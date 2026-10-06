// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SERVICE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SERVICE_H_

#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/models/qwen/model.h"

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

typedef enum loom_serve_qwen_packing_mode_e {
  // Fill one epoch with all ready prompt and decode spans.
  LOOM_SERVE_QWEN_PACKING_MIXED = 0,
  // Pack only the first ready row's phase, using the same rotating priority.
  LOOM_SERVE_QWEN_PACKING_SEPARATE,
} loom_serve_qwen_packing_mode_t;

typedef struct loom_serve_qwen_service_options_t {
  // Number of retained rows, matching the model residency.
  iree_host_size_t row_count;
  // Maximum known input tokens admitted from one row in an epoch.
  iree_host_size_t chunk_size;
  // Output bound when a request omits max_tokens.
  iree_host_size_t default_max_tokens;
  // Positive bound on validated requests waiting without a model row.
  iree_host_size_t pending_capacity;
  // Reporting interval in nanoseconds; zero disables periodic heartbeats.
  iree_duration_t heartbeat_interval;
  // Execution choice; the planner and HTTP lifecycle are shared by all modes.
  loom_serve_qwen_schedule_mode_t schedule_mode;
  // Whether prompt and decode inputs may share an epoch, independent of math.
  loom_serve_qwen_packing_mode_t packing_mode;
  // Proposal depth: zero or three. Three requires packed scheduling and an MTP
  // bundle on the borrowed model, with at least one epoch shape admitting four
  // tokens. Zero with a bundle measures warm target-only.
  iree_host_size_t mtp_depth;
  // One or two device-fed epochs between transport/admission observations.
  // Two requires MTP; output credit and speculative residency cover both.
  iree_host_size_t continuation_epochs;
} loom_serve_qwen_service_options_t;

// Runs one application owner until transport shutdown or model failure. Model
// and transport are borrowed. Each epoch gathers credited ready rows, executes
// their known/verifier spans and commits outputs before reusing the workspace.
// Heartbeats observe a copied snapshot and continue during model waits.
// Admission reserves completion capacity before assigning physical pages as
// execution grows. Excess work waits in a bounded FIFO; impossible requests
// reject before altering retained state. Idle cache yields to admitted work.
// X-Loom-Session selects retained state, not a durable session. Active or
// queued named sessions reject overlapping requests. Untagged requests always
// replay. Peer cancellation discards its checkpoint at a completed stage
// boundary. Return relinquishes every connection view before the caller
// destroys transport/model. Capacities match the created model.
iree_status_t loom_serve_qwen_service_run(
    loom_serve_qwen_model_t* model, loom_serve_http_server_t* server,
    const loom_serve_qwen_service_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SERVICE_H_
