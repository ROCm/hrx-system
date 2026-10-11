// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_

#include "experimental/loom_serve/serving/registry.h"
#include "experimental/loom_serve/serving/resources.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_router_options_t {
  // Positive per-turn limit on new transport claims and, independently, on
  // previously held claims released into model admission and telemetry batches
  // formatted for live observers.
  iree_host_size_t request_quota;
  // Global requests parked behind closed model admission gates. Must be below
  // transport connection capacity so holding alone cannot exclude control.
  // Zero makes every such arrival an immediate overload rejection.
  iree_host_size_t holding_capacity;
  // Application resources borrowed until the outer routing call returns.
  loom_serve_resources_t resources;
} loom_serve_router_options_t;

// Routes JSON request bodies by their model field and checkpoint operations by
// X-Loom-Model. A checkpoint request may omit its selector with exactly one
// service. GET /v1/models lists deployment names and namespaced capabilities;
// GET /v1/status adds copied scheduler ownership and a server object containing
// actual worker/admission/framing bounds and published resources. No hardware
// polling or device synchronization is performed. /healthz and /readyz are
// model-free: HTTP is available, independently of model preparation/activation.
// Zero services is valid: model/status lists are empty, work counters are zero,
// and inference or tokenization requests receive model-not-registered errors.
// GET /metrics exposes the same copied ownership as Prometheus text gauges
// and cumulative model-protocol counters without invoking models/devices.
// Server counters include current and removed model incarnations, excluding
// router-handled utilities, observation and rejected routes. If a service has
// no counters, whole-server totals are unavailable rather than partial.
// GET /logz polls server diagnostics and current totals. X-Loom-Model instead
// selects a model service's stream; X-Loom-Cursor resumes that reader's
// previous poll. Each response copies records and reports gaps/current counters
// without retaining slots or slowing publication. See http/logs.h for cursor
// rules. GET /logz/stream carries the same bounded batches as SSE telemetry
// events, with the next cursor as id; Last-Event-ID resumes a connection.
// Each reader owns only a transport claim and copied identity/counters. Model
// removal never waits for observers; stream_end reports that lifetime's end
// once transport credit permits. Held requests plus observers remain below
// transport capacity, preserving one credit for other arrivals. Excess long-
// lived claims receive 429. Idle observers schedule no periodic work.
// POST /v1/models/register takes one catalog model object with optional
// admission="open" or "drained". A 202 acknowledges server-owned intent;
// /v1/status reports preparation/failure and copied source diagnostics. Closing
// the control connection does not cancel the operation. Drained registration
// stores metadata only; undrain starts preparation. Inference arriving before
// readiness uses the same bounded holding credit as drained admission.
// POST /v1/models/unregister takes {"model":"name","mode":"drain"|"abort"};
// omitted mode is drain. Its 202 seals admission irreversibly, resolves held
// arrivals with 410, then completes or cancels admitted work before retirement.
// A name remains reserved until actual resource retirement. Unsupported managed
// modalities return 501; borrowed application services cannot be unregistered.
// POST /v1/models/drain and /v1/models/undrain select {"model":"name"} and
// change admission without evicting weights or discarding retained state.
// Drain lets every already-admitted request finish, including service queues;
// later inference/checkpoint/tokenizer requests wait outside those services.
// Holding overload returns 429. Control and observation remain ungated. Status
// reports admission separately from the modality phase and held requests.
// Each turn claims at most request_quota new requests and releases at most that
// many held requests before advancing each service once, with rotating service
// priority on the borrowed executor. Owner turns return workers while waiting
// for network, CPU or device readiness. The outer
// caller waits only for final process release; this function is not a task.
// POST /tokenize and /detokenize use the selected service's optional tokenizer
// on independent finite CPU tasks. Transport connection credit bounds utility
// jobs; each retains its request claim until task release, even on
// cancellation. Shutdown/failure stops admission and drains accepted
// asynchronous callbacks before detaching their wake targets. Managed models
// retire here through nonblocking ownership wakes, independently of readiness
// success. Borrowed services still belong to the caller and are destroyed
// before the borrowed transport. The registry survives this call: a platform
// cleanup failure preserves its resources and factory/device owners for the
// outer application. No callback destroys a shared queue or device group.
iree_status_t loom_serve_router_run(iree_task_executor_t* executor,
                                    loom_serve_http_server_t* server,
                                    loom_serve_registry_t* registry,
                                    loom_serve_router_options_t options,
                                    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_
