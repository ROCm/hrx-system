// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_

#include "loom/serve/service/text/service.h"
#include "loom/serve/service/text/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// A CPU task prepares source/host storage without touching the shared pool.
// The serialized owner submits native growth, then joins its queue frontier.
// Public inspection sees the old directory until publication. Existing records
// remain stable throughout either loan.
typedef struct loom_serve_text_service_capacity_t {
  // Requested next capacity, independent of epoch shape and live KV extent.
  iree_host_size_t requested;
  // Initialized prefix of sessions; the old prefix is borrowed, not owned.
  iree_host_size_t count;
  // Owned replacement directory; the service arena owns its stable records.
  loom_serve_text_session_t** sessions;
  // Owned zeroed readiness entries for the next directory.
  loom_serve_ready_span_t* ready;
  // Owned source preparation, null when a model borrower deferred growth.
  loom_serve_text_resize_t* resize;
} loom_serve_text_service_capacity_t;

// Selects the next amortized capacity after no idle resident was found. Zero
// means a reached ceiling, accepted work, or an address-bound continuation
// prevents growth. Selection starts no work and changes no ownership.
iree_host_size_t loom_serve_text_service_capacity_select(
    const loom_serve_text_service_t* service);

// Starts CPU preparation for the selected capacity on the serialized owner.
void loom_serve_text_service_capacity_begin(loom_serve_text_service_t* service,
                                            iree_host_size_t requested);

// After CPU readiness, submits native growth on the shared pool's owner and
// imports its device frontier. Never waits for native completion.
iree_status_t loom_serve_text_service_capacity_await(
    loom_serve_text_service_t* service);

// After successful device readiness, atomically publishes the new directory
// on the owner. Normal request admission still reserves actual memory credit.
void loom_serve_text_service_capacity_publish(
    loom_serve_text_service_t* service);

// Releases unconsumed host preparation after all CPU/device borrowers retire.
void loom_serve_text_service_capacity_discard(
    loom_serve_text_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_
