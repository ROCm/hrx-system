// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PENDING_QUEUE_H_
#define LOOM_SERVE_SERVICE_TEXT_PENDING_QUEUE_H_

#include "loom/serve/service/text/admission.h"
#include "loom/serve/service/text/consumer.h"
#include "loom/serve/service/text/pending_request.h"
#include "loom/serve/service/text/prefix_cache.h"
#include "loom/serve/service/text/session_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serialized admission ownership, independent of device rows and epoch spans.
// Request slots stay fixed while the FIFO pointer order changes. The scheduler
// owns readiness and identifies the exact request borrowed by CPU/device work.
typedef struct loom_serve_text_pending_queue_t {
  // Allocator owning stable slots and terminal request payloads.
  iree_allocator_t host_allocator;
  // Borrowed model whose selected admission intent is cleared on removal.
  loom_serve_text_model_t* model;
  // Borrowed owner-only counters and terminal request reporting.
  loom_serve_text_telemetry_t* telemetry;
  // One allocation holding stable request slots and their pointer order.
  uint8_t* storage;
  // Live FIFO prefix followed by reusable free slots. Removing another entry
  // never moves the payload of a request retained by a borrower.
  loom_serve_text_pending_request_t** values;
  // Number of claimed requests in FIFO order.
  iree_host_size_t count;
  // Maximum queued requests, allocated once during initialization.
  iree_host_size_t capacity;
  // Replaceable selection policy, independent of request/storage ownership.
  loom_serve_text_admission_t admission;
  // Scheduler-owned attempt retained through CPU preparation and reservation.
  // Removal adjusts its FIFO index or clears it after all borrowers retire.
  struct {
    // Stable selected request, independent of FIFO removal and slot reuse.
    loom_serve_text_pending_request_t* request;
    // FIFO position of this borrow, independent of admission lookahead.
    iree_host_size_t index;
    // Idle slot borrowed through this attempt's preparation/reservation.
    loom_serve_text_session_t* session;
    // CPU-selected row, published to session only after CPU readiness joins.
    loom_serve_text_session_t* resolved_session;
    // Idle logical history protected through CPU preparation/admission.
    loom_serve_text_session_cache_entry_t* cached;
    // Exact token prefix borrowed through this reservation attempt.
    loom_serve_text_prefix_entry_t* prefix;
    // Outgoing row pin awaiting readiness before logical ownership moves.
    loom_serve_text_session_cache_entry_t* detaching;
    // Cold image payload before the current reservation attempt.
    iree_host_size_t restored_bytes;
  } prepared;
} loom_serve_text_pending_queue_t;

// Allocates host records only. The model and telemetry outlive the queue.
// Partial initialization supports deinitialization without accepting work.
iree_status_t loom_serve_text_pending_queue_initialize(
    loom_serve_text_model_t* model, loom_serve_text_telemetry_t* telemetry,
    iree_host_size_t capacity, iree_host_size_t overtake_budget,
    iree_allocator_t host_allocator,
    loom_serve_text_pending_queue_t* out_queue);

// All request consumers and their CPU/device/media borrowers have retired.
void loom_serve_text_pending_queue_deinitialize(
    loom_serve_text_pending_queue_t* queue);

// Both ingress paths validate names against the actual rows, queued requests,
// and checkpoint registry before accepting input. The selected endpoint remains
// borrowed on the serialized owner until push acquires its pending reference.
iree_status_t loom_serve_text_pending_queue_select_request(
    const loom_serve_text_pending_queue_t* queue, iree_string_view_t name,
    iree_string_view_t checkpoint_name, iree_host_size_t row_count,
    loom_serve_text_session_t* sessions,
    loom_serve_text_checkpoint_registry_t* checkpoints,
    loom_serve_text_checkpoint_entry_t** out_checkpoint);

// Moves a validated request into an available stable queue slot. The caller
// establishes capacity and owns publication/scheduling; this only transfers
// input ownership and accounts the resolved checkpoint's pending reference.
void loom_serve_text_pending_queue_push(
    loom_serve_text_pending_queue_t* queue, loom_serve_text_consumer_t consumer,
    iree_string_view_t body, iree_string_view_t name,
    loom_serve_text_checkpoint_entry_t* checkpoint,
    loom_serve_text_receipt_t receipt);

// Resolves the selected request's exact completed frontier without acquiring
// native ownership. Row/cache contents stay protected through preparation.
loom_serve_text_input_prefix_t loom_serve_text_pending_queue_prefix(
    const loom_serve_text_pending_queue_t* queue,
    const loom_serve_text_session_t* session);

// Finite CPU preparation under the scheduler's exclusive source invocation
// loan. Renders once and publishes the selected idle row only after release.
// The supplied rows and logical geometry remain immutable during this loan;
// no concurrent physical-pool usage or admission state is read.
iree_status_t loom_serve_text_pending_queue_prepare(
    loom_serve_text_pending_queue_t* queue, iree_string_view_t name,
    iree_host_size_t default_max_tokens, iree_host_size_t row_count,
    loom_serve_text_session_t* sessions,
    const loom_serve_text_pool_usage_t* geometry);

// The caller has released or transferred request payloads and the consumer.
// Returns the stable slot to the free suffix, releases checkpoint accounting,
// and preserves other borrowers while adjusting selection/policy positions.
void loom_serve_text_pending_queue_remove(
    loom_serve_text_pending_queue_t* queue, iree_host_size_t index);

// Terminal cancellation after CPU/device loans have retired and the projector
// can detach. Releases request payloads, the consumer receipt, and the queue
// slot.
void loom_serve_text_pending_queue_cancel(
    loom_serve_text_pending_queue_t* queue, iree_host_size_t index,
    loom_serve_text_media_projector_t* projector);

// Removes closed consumers except borrowed, which the scheduler retains through
// readiness release. A media borrower may similarly prevent cancellation.
// Returns whether any request retired; it never waits for a borrower.
bool loom_serve_text_pending_queue_cancel_failed(
    loom_serve_text_pending_queue_t* queue,
    const loom_serve_text_pending_request_t* borrowed,
    loom_serve_text_media_projector_t* projector);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PENDING_QUEUE_H_
