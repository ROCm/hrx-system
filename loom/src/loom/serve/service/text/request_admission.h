// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_REQUEST_ADMISSION_H_
#define LOOM_SERVE_SERVICE_TEXT_REQUEST_ADMISSION_H_

#include "loom/serve/service/text/service_owner.h"

// Publishes queue-owned counters without reading CPU-borrowed row state.
void loom_serve_text_pending_observe(loom_serve_text_service_t* service);

// Cancels an unborrowed request and returns its queue slot and continuation.
void loom_serve_text_pending_cancel(loom_serve_text_service_t* service,
                                    iree_host_size_t index);

// Cancels disconnected peers while preserving the current CPU/device loan.
void loom_serve_text_pending_cancel_failed(loom_serve_text_service_t* service,
                                           bool* out_progress);

// Submits pending state maintenance at a completed model boundary.
iree_status_t loom_serve_text_submit_trim(loom_serve_text_service_t* service);

// Advances completed maintenance and publishes its final storage counters.
iree_status_t loom_serve_text_complete_trim(loom_serve_text_service_t* service);

// Whether idle-history release still requires native state maintenance.
bool loom_serve_text_cache_reclaim_pending(void* self);

// Reclaims one unborrowed idle history on shared residency pressure.
bool loom_serve_text_cache_reclaim_request(void* self);

// Projects retained native state using the same eligibility as reclamation.
// Explicit endpoints and currently borrowed state remain protected.
iree_status_t loom_serve_text_cache_forecast(
    void* self, const loom_runtime_residency_forecast_t* request,
    loom_runtime_residency_projection_t* out_projection);

// Lends an idle row to CPU preparation for the selected admission candidate.
bool loom_serve_text_begin_preparation(loom_serve_text_service_t* service);

// Publishes an outgoing row's cache endpoint after successful readiness.
void loom_serve_text_complete_retention(loom_serve_text_service_t* service);

// Joins CPU input preparation; request errors remain local to its receipt.
iree_status_t loom_serve_text_complete_preparation(
    loom_serve_text_service_t* service, loom_runtime_service_flags_t flags,
    iree_status_t status);

// Completes native state admission and transfers the request into its row.
iree_status_t loom_serve_text_complete_reservation(
    loom_serve_text_service_t* service, loom_runtime_service_flags_t flags);

// Admits ready text work or protects media consumers without worker waits.
iree_status_t loom_serve_text_admit_pending(loom_serve_text_service_t* service,
                                            loom_runtime_service_flags_t flags,
                                            bool allow_preparation);

// Transfers a validated receipt into a stable queue slot and offers
// preparation.
void loom_serve_text_pending_push(
    loom_serve_text_service_t* service, loom_serve_text_consumer_t consumer,
    iree_string_view_t body, iree_string_view_t name,
    loom_serve_text_checkpoint_entry_t* checkpoint,
    const loom_serve_text_sequence_t* sequence,
    loom_serve_text_receipt_t receipt);

// Gathers a bounded observed queue window without changing text admission
// order.
bool loom_serve_text_media_lookahead(loom_serve_text_service_t* service);

#endif  // LOOM_SERVE_SERVICE_TEXT_REQUEST_ADMISSION_H_
