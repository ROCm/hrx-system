// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_WIRE_H_
#define LOOM_SERVE_SERVER_REALTIME_WIRE_H_

#include "loom/serve/transport/websocket/connection.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_wire_message_t
    loom_serve_realtime_wire_message_t;
typedef struct loom_serve_realtime_wire_t {
  // Allocator owning queued event records and serialized payloads.
  iree_allocator_t host_allocator;
  // Borrowed framed connection; the session retains the HTTP claim.
  loom_serve_websocket_t* socket;
  // Borrowed HTTP owner for send credit and physical network retirement.
  loom_serve_http_connection_t* connection;
  // Session identity used to generate unique server event IDs.
  uint64_t session_id;
  // Last assigned event ID; cancelled publication never reuses IDs.
  uint64_t sequence;
  // Reusable owner-only event construction storage, outside queued byte credit.
  iree_string_builder_t scratch;
  // Oldest queued event, preserving command/response ordering under pressure.
  loom_serve_realtime_wire_message_t* first;
  // Tail for constant-time ordered insertion.
  loom_serve_realtime_wire_message_t* last;
  // Allocated payload capacity of queued events, bounded to 1 MiB.
  iree_host_size_t bytes;
  // Queued record count, bounded to 64 independently of byte credit.
  iree_host_size_t count;
} loom_serve_realtime_wire_t;

void loom_serve_realtime_wire_initialize(
    uint64_t session_id, loom_serve_websocket_t* socket,
    loom_serve_http_connection_t* connection, iree_allocator_t host_allocator,
    loom_serve_realtime_wire_t* wire);
void loom_serve_realtime_wire_deinitialize(loom_serve_realtime_wire_t* wire);
// Drops queued delivery, retaining only reusable construction storage.
void loom_serve_realtime_wire_clear(loom_serve_realtime_wire_t* wire);

// Rolls back events appended after a saved queue tail, or the entire queue
// when tail is NULL. The serialized command owner saves wire->last before
// construction; no flush, cancellation, or other queue mutation intervenes.
// Earlier queued events and their owned payloads remain unchanged.
void loom_serve_realtime_wire_rewind(loom_serve_realtime_wire_t* wire,
                                     loom_serve_realtime_wire_message_t* tail);

// Shared event prefix and exact per-message byte boundary, used by real queue
// admission and complete-call preflight. Prefix leaves the outer object open.
iree_status_t loom_serve_realtime_wire_prefix(uint64_t session_id,
                                              uint64_t sequence,
                                              const char* type,
                                              loom_output_stream_t* output);
iree_status_t loom_serve_realtime_wire_check_size(iree_host_size_t size);

// Begins one object with event_id and type, leaving the object open. The caller
// adds event-specific fields and closes it before commit. No queue mutation
// occurs until commit succeeds; failed command formatting can reset scratch.
iree_status_t loom_serve_realtime_wire_begin(loom_serve_realtime_wire_t* wire,
                                             const char* type);
// Transfers scratch to the bounded queue. publication tags unsent response
// data for cancellation. Queued events own their complete serialized payload;
// they borrow no conversation or model storage. An oversized individual event
// returns OUT_OF_RANGE and releases its construction storage. Aggregate queued
// byte/count pressure returns RESOURCE_EXHAUSTED.
iree_status_t loom_serve_realtime_wire_commit(loom_serve_realtime_wire_t* wire,
                                              uint64_t response,
                                              bool publication);
// Copies a control frame into the same ordered send owner.
iree_status_t loom_serve_realtime_wire_control(
    loom_serve_realtime_wire_t* wire, loom_serve_websocket_message_type_t type,
    iree_const_byte_span_t payload);
// Removes only unsent data tagged to this response. Accepted transport bytes
// remain in its ordinary completion lifetime; the client clears playback.
void loom_serve_realtime_wire_cancel(loom_serve_realtime_wire_t* wire,
                                     uint64_t response);
// A bounded owner turn sends at most one event. Network errors belong to this
// peer; the session cancels delivery while retaining accepted CPU borrowers.
iree_status_t loom_serve_realtime_wire_flush(loom_serve_realtime_wire_t* wire,
                                             bool* out_progress);
// Allows generation only when its next packet can reach transport admission.
bool loom_serve_realtime_wire_can_publish(
    const loom_serve_realtime_wire_t* wire);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_WIRE_H_
