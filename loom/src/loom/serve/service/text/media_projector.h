// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_

#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/media_input.h"
#include "loom/serve/service/text/readiness.h"

#ifdef __cplusplus
extern "C" {
#endif

// Request-owned device features, indexed by immutable chat image occurrence.
// Moves from the pending request to its admitted session without moving
// storage. This is not a persistent cache: only live requests retain projected
// roots.
typedef struct loom_serve_text_media_request_t loom_serve_text_media_request_t;
typedef struct loom_serve_text_media_projector_t
    loom_serve_text_media_projector_t;

// Borrows chat through projection, not through later text execution. Allocation
// is proportional to the checked image count; no pixels or device work occur.
iree_status_t loom_serve_text_media_request_allocate(
    const loom_serve_text_chat_t* chat, iree_allocator_t host_allocator,
    loom_serve_text_media_request_t** out_request);

// Rebinds checked input coordinates after history eviction/prefix lookup.
// Existing features remain reusable by image occurrence. Returns whether all
// remaining images have published results. Requests whose declared phases fit
// only sequentially publish after producer completion; others can compose
// directly with future-ready results. Does not start or wait for work.
bool loom_serve_text_media_request_bind(
    loom_serve_text_media_request_t* request,
    const loom_serve_text_input_t* input, iree_host_size_t consumed);

// Commits the bound input after admission and projector detachment. Drops
// projected images superseded by a retained prefix and ends the chat borrow.
void loom_serve_text_media_request_begin(
    loom_serve_text_media_request_t* request);

// Transfers any request-scoped preparation error to the response owner.
iree_status_t loom_serve_text_media_request_take_status(
    loom_serve_text_media_request_t* request);

// Borrowed through epoch submission, after bind reports complete. NULL is an
// empty text-only input. Consume releases roots only after consumer readiness.
const loom_serve_text_media_input_t* loom_serve_text_media_request_input(
    const loom_serve_text_media_request_t* request);
void loom_serve_text_media_request_consume(
    loom_serve_text_media_request_t* request, iree_host_size_t consumed);

// Request status has been consumed and projector_cancel has detached any jobs.
// NULL is accepted. Device queue captures preserve actual consumer lifetimes.
void loom_serve_text_media_request_free(
    loom_serve_text_media_request_t* request);

// One bounded host preparation window over an independently prepared component.
// capacity bounds queued image occurrences; the source batch limit further
// bounds the window. Decoded packet bytes target the source upload capacity;
// one larger checked image is allowed alone. Prepared windows drain before new
// decoding. CPU and producer readiness are independent of text's channels.
iree_status_t loom_serve_text_media_projector_allocate(
    loom_serve_finite_model_t* model, iree_host_size_t capacity,
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_text_media_projector_t** out_projector);

// Offers missing images in request order. False means the current window is
// sealed/full, not request failure. Each occurrence is selected at most once.
// consumer_bytes is the prepared language continuation's footprint, excluding
// features. It is captured before the CPU forecast loan; no consumer state or
// residency is accessed from that loan.
bool loom_serve_text_media_projector_offer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_request_t* request, uint64_t consumer_bytes);

// Seals a gathered window without a batching timer or waiting for new arrivals.
bool loom_serve_text_media_projector_start(
    loom_serve_text_media_projector_t* projector);

// A new preparation window can accept offers without borrowing caller storage.
bool loom_serve_text_media_projector_is_collecting(
    const loom_serve_text_media_projector_t* projector);

// False only while the codec CPU job borrows this request's HTTP-backed chat.
// Otherwise detaches the request immediately; selected cohort ordinals stay
// stable and already submitted device work is retired normally.
bool loom_serve_text_media_projector_cancel(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_request_t* request);

// Whether a codec job still borrows the request's immutable HTTP-backed chat.
bool loom_serve_text_media_projector_borrows(
    const loom_serve_text_media_projector_t* projector,
    const loom_serve_text_media_request_t* request);

bool loom_serve_text_media_projector_is_pending(
    const loom_serve_text_media_projector_t* projector);
// Replaces/detaches the owner's observer after the usual callback join.
void loom_serve_text_media_projector_set_observer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_wake_callback_t observer);
// Owner-published activity for service inspection; NULL means no active window.
const char* loom_serve_text_media_projector_phase(
    const loom_serve_text_media_projector_t* projector);
iree_status_t loom_serve_text_media_projector_advance(
    loom_serve_text_media_projector_t* projector,
    loom_serve_service_flags_t flags, bool* out_progress);

// All accepted callbacks have retired. Request owners have detached their jobs.
// Failed device work remains component-owned until actual model retirement.
void loom_serve_text_media_projector_free(
    loom_serve_text_media_projector_t* projector);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_
