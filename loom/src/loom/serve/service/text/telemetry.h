// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_TELEMETRY_H_
#define LOOM_SERVE_SERVICE_TEXT_TELEMETRY_H_

#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/service/text/epoch.h"
#include "loom/serve/service/text/heartbeat.h"
#include "loom/serve/service/text/session.h"
#include "loom/service/telemetry/latency.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/text/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_receipt_t {
  // Monotonic identity assigned exactly once at protocol receipt.
  uint64_t serial;
  // Host arrival time before validation, queuing or device admission.
  iree_time_t arrival_time;
} loom_serve_text_receipt_t;

// Updated only on serialized service turns, never by CPU borrowers or
// callbacks.
typedef struct loom_serve_text_telemetry_t {
  // Bounded history; consumers never pin or backpressure records.
  loom_runtime_telemetry_stream_t* stream;
  // False when the registration lends its history through service retirement.
  bool owns_stream;
  // Always-on model totals independent of heartbeat interval and readers.
  loom_service_telemetry_counters_t counters;
  // Request latency distributions, independent of event retention.
  loom_service_telemetry_latency_t latency;
  // Monotonic request identity independent of TCP connection identity.
  uint64_t next_serial;
  // Owned first model failure, retained through every request's retirement.
  iree_status_t failure;
} loom_serve_text_telemetry_t;

// Borrows an existing registration stream or creates a private bounded stream
// when NULL. Only a private stream publishes a start here.
iree_status_t loom_serve_text_telemetry_initialize(
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_text_telemetry_t* out_telemetry);

// Releases the retained failure and any owned stream after retiring the
// heartbeat and all producer/consumer uses. A borrowed stream remains live.
void loom_serve_text_telemetry_deinitialize(
    loom_serve_text_telemetry_t* telemetry);

// Counts a protocol receipt exactly once, before validation and admission.
loom_serve_text_receipt_t loom_serve_text_telemetry_begin(
    loom_serve_text_telemetry_t* telemetry);

// Records the final protocol outcome; status is borrowed, not consumed.
void loom_serve_text_telemetry_end(loom_serve_text_telemetry_t* telemetry,
                                   loom_serve_text_receipt_t receipt,
                                   loom_service_telemetry_outcome_t outcome,
                                   uint64_t input_tokens,
                                   const iree_status_t borrowed_status);

// Diagnostic ownership boundary: copies bounded detail and consumes status.
void loom_serve_text_telemetry_diagnose(loom_serve_text_telemetry_t* telemetry,
                                        uint64_t request, iree_status_t status);

// Retains the first model failure while its original status propagates
// normally. Later failures are recorded without replacing the drain's cause.
void loom_serve_text_telemetry_fail(loom_serve_text_telemetry_t* telemetry,
                                    const iree_status_t borrowed_status);

// Updates session/request outcomes and publishes lifetime snapshots before
// their host state is retired. Does not alter row, request or HTTP ownership.
void loom_serve_text_telemetry_request_end(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_session_t* session,
    loom_service_telemetry_outcome_t outcome);

// Publishes final session totals and clears its logical identity/accounting.
// No row or request storage is reached or changed.
void loom_serve_text_telemetry_session_end(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_session_lifetime_t* lifetime);

// Accounts completed model progress at every lifetime and publishes span/epoch
// records plus the heartbeat's completed-work snapshot. Model results remain
// immutable; request phase, output translation and retirement belong to the
// service. The caller publishes each successfully completed cohort once.
void loom_serve_text_telemetry_epoch(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_heartbeat_t* heartbeat,
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_t* cohort,
    loom_serve_text_session_t* const* sessions);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_TELEMETRY_H_
