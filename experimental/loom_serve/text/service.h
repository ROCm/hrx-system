// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_SERVICE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_SERVICE_H_

#include "experimental/loom_serve/http/router.h"
#include "experimental/loom_serve/text/model.h"
#include "experimental/loom_serve/text/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_service_options_t {
  // Borrowed deployment route name, outliving the service. Source identity and
  // chat policy remain owned by the model, independently of this HTTP name.
  iree_string_view_t name;
  // Number of retained rows, matching the model residency.
  iree_host_size_t row_count;
  // Maximum known input tokens admitted from one row in an epoch.
  iree_host_size_t chunk_size;
  // Output bound when a request omits max_tokens.
  iree_host_size_t default_max_tokens;
  // Positive bound on claimed requests waiting without a model row.
  iree_host_size_t pending_capacity;
  // Reporting interval in nanoseconds; zero disables periodic heartbeats.
  iree_duration_t heartbeat_interval;
  // Execution choice; the planner and HTTP lifecycle are shared by all modes.
  loom_serve_text_schedule_mode_t schedule_mode;
  // Whether prompt and decode inputs may share an epoch, independent of math.
  loom_serve_text_packing_mode_t packing_mode;
  // Proposal depth: zero or three. Three requires packed scheduling and an MTP
  // bundle on the borrowed model, with at least one epoch shape admitting four
  // tokens. Zero with a bundle measures warm target-only.
  iree_host_size_t mtp_depth;
  // One or two device-fed epochs between transport/admission observations.
  // Two requires MTP; output credit and speculative residency cover both.
  iree_host_size_t continuation_epochs;
} loom_serve_text_service_options_t;

typedef struct loom_serve_text_service_t loom_serve_text_service_t;

// Creates host session/admission state around a borrowed model. CPU work shares
// the host executor; reporting shares its proactor. No per-model thread is
// created. Capacities match the model. Calls use its one serialized host owner.
iree_status_t loom_serve_text_service_create(
    loom_serve_text_model_t* model,
    const loom_serve_text_service_options_t* options,
    loom_serve_text_service_t** out_service, iree_allocator_t host_allocator);

// Returns borrowed callbacks for the shared owner loop. They take connection
// claims, preserve model-local sessions/checkpoints, and advance one cohort.
loom_serve_http_service_t loom_serve_text_service_interface(
    loom_serve_text_service_t* service);

// Relinquishes every accepted connection after readiness retirement. On
// terminal failure, unresolved device reservations remain pinned for model
// destruction; on ordinary completed-boundary shutdown, unused reservations are
// released. terminal_code is the owner loop's result, including another model's
// failure. The owner has consumed all CPU, device and heartbeat callbacks,
// including timer cancellation receipts, before this call. Null is accepted.
// The borrowed model and transport both outlive this call.
void loom_serve_text_service_destroy(loom_serve_text_service_t* service,
                                     iree_status_code_t terminal_code);

// Runs one application owner until transport shutdown or model failure. Model
// and transport are borrowed. Each epoch gathers credited ready rows, executes
// their known/verifier spans asynchronously and commits outputs after readiness
// before reusing feedback. Other services can advance while an epoch executes.
// Heartbeats observe a copied snapshot and continue during model waits.
// Admission owns and physically backs the exact completion high-water, private
// recurrent writer and partial-tail COW while pinning required parameters.
// Shared prefixes are counted once. Excess work waits in a bounded FIFO;
// invalid requests reject before altering retained state. Idle cache yields to
// admission pressure, but explicit pins remain protected until deletion.
// Evicted/cancelled rows and unused completion credit trigger physical trim
// outside ordinary decoding. Compaction admits its temporary destination union
// before moving. state_trim events report copy, release
// and maintenance costs separately from model epochs.
// X-Loom-Session selects retained state, not a durable session. Active or
// queued named sessions reject overlapping requests. X-Loom-Checkpoint selects
// an explicit pinned endpoint only after full canonical-history validation.
// POST/DELETE /v1/checkpoints/NAME pin an idle completed X-Loom-Session or
// release that pin. Deletion refuses while queued requests borrow the endpoint.
// POST /v1/checkpoints/NAME/suspend explicitly parks the pin in DRAM on elastic
// backing without changing live branches or queued identity. Wake admission
// includes missing prefix pages and preserves the selected row until its
// replacement fits. Host image bytes are reported separately from device pages.
// Untagged requests without an explicit pin always replay. Peer cancellation
// discards its live row, not independently pinned endpoints, at a completed
// stage boundary. Return relinquishes every connection view before the caller
// destroys transport/model. Capacities match the created model.
iree_status_t loom_serve_text_service_run(
    loom_serve_text_model_t* model, loom_serve_http_server_t* server,
    const loom_serve_text_service_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_SERVICE_H_
