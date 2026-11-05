// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_SERVICE_H_
#define LOOM_SERVICE_FINITE_SERVICE_H_

#include "loom/runtime/service.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/service/finite/call.h"
#include "loom/service/finite/consumer.h"
#include "loom/service/finite/model.h"
#include "loom/service/finite/protocol.h"
#include "loom/service/inspection.h"
#include "loom/service/reconfiguration.h"
#include "loom/service/telemetry/latency.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_service_finite_service_t loom_service_finite_service_t;

// Borrowed finite capabilities, live through service retirement. Progress and
// observation use the same serialized owner and runtime.self; no mutable task
// payload escapes. Frontends add their own transport metadata and admission.
typedef struct loom_service_finite_service_interface_t {
  // Borrowed deployment identifier, independent of source model identity.
  iree_string_view_t name;
  // Shared owner progression and physical-capacity relationship.
  loom_runtime_service_t runtime;
  // Immutable semantic role of this finite protocol.
  loom_service_kind_t kind;
  // Borrowed immutable input tokenizer, valid through service retirement.
  const iree_tokenizer_t* tokenizer;
  // Modality spelling borrowed from the retained protocol, not this view.
  const iree_string_view_t* output_modality;
  // Copies current ownership without reading in-flight task state.
  loom_service_snapshot_t (*snapshot)(void* self);
  // Bounded copied request metadata on the same serialized owner.
  loom_service_inspection_t inspection;
  // Service-owned executable choices and safe-point selection policy.
  loom_service_reconfiguration_t reconfiguration;
  // Read-only observation, borrowed through service retirement.
  struct {
    // Bounded record history with independent reader cursors.
    const loom_runtime_telemetry_stream_t* stream;
    // Cumulative work, independent of record retention.
    const loom_service_telemetry_counters_t* counters;
    // Cumulative terminal-request latency distributions.
    const loom_service_telemetry_latency_t* latency;
  } telemetry;
} loom_service_finite_service_interface_t;

// Bounded finite-vector admission. The borrowed source model, protocol and
// route name outlive the service. CPU parsing, source preparation, batch
// assembly and encoding share the host executor. Device ALL readiness uses the
// proactor; workers never wait. Compatible requests already waiting fill the
// oldest request's cohort up to its source-defined limit, without an arrival
// delay. capacity bounds all outstanding host requests, including the live
// cohort. variant_capacity bounds retained execution choices; zero disables
// reconfiguration. A nonnegative heartbeat_interval controls periodic
// snapshots; zero disables periodic samples, not counters or lifecycle events.
// Observation is a bounded overwriting stream, never a synchronous sink write
// on a worker. A non-NULL stream is borrowed through service destruction and
// shares the serialized owner's publication order. NULL creates a service-owned
// stream.
iree_status_t loom_service_finite_service_allocate(
    loom_runtime_host_t* host, loom_service_finite_model_t* model,
    const loom_service_finite_protocol_t* protocol, iree_string_view_t name,
    iree_host_size_t capacity, iree_host_size_t variant_capacity,
    iree_duration_t heartbeat_interval, loom_runtime_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_service_finite_service_t** out_service);

loom_service_finite_service_interface_t loom_service_finite_service_interface(
    loom_service_finite_service_t* service);

// Immutable semantic protocol role, also used by frontend route selection.
loom_service_kind_t loom_service_finite_service_kind(
    const loom_service_finite_service_t* service);

// Borrows body from a consumer receipt on the serialized owner. Acceptance
// transfers the receipt through consumer.end; pressure or failure preserves it.
// arrival_time is the frontend's monotonic receipt time, before its preflight.
// Source preparation, batching, and output use the same path as native calls.
iree_status_t loom_service_finite_service_try_accept(
    loom_service_finite_service_t* service, iree_string_view_t body,
    iree_time_t arrival_time, loom_service_finite_consumer_t consumer,
    bool* out_accepted);

// Records one frontend rejection that did not transfer a receipt to the
// service. Runs on the same serialized owner. status is a borrowed delivery or
// allocation failure; OK plus delivered distinguishes rejection from lost
// interest. Preserves cumulative request accounting without invoking model
// work.
void loom_service_finite_service_record_decline(
    loom_service_finite_service_t* service, iree_time_t arrival_time,
    bool delivered, const iree_status_t status);

// Native admission on the same serialized owner as advance. Acceptance moves
// the owning input builder and returns a call; success with NULL means bounded
// pressure. Both pressure and failure preserve input. The modality's normal
// source preparation, cohort packing, and residency policy are shared with
// HTTP. Device calls publish retained rows without downloading or encoding;
// the caller imports their dependency before device composition. JSON calls
// select the shared cohort download and publish owned semantic output after
// encoding. Compatible calls of either format can share a cohort.
iree_status_t loom_service_finite_service_try_open(
    loom_service_finite_service_t* service, iree_string_builder_t* body,
    loom_service_finite_output_format_t output_format,
    loom_service_finite_call_t** out_call);

// After owner retirement only. CPU/process release, not completion, releases
// borrowed inputs/results. Failed device work belongs to outer model teardown.
void loom_service_finite_service_free(loom_service_finite_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_FINITE_SERVICE_H_
