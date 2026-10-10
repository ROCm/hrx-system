// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_

#include "experimental/loom_serve/http/service.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_router_options_t {
  // Positive per-turn limit on new transport claims and, independently, on
  // previously held claims released into model admission.
  iree_host_size_t request_quota;
  // Global requests parked behind closed model admission gates. Must be below
  // transport connection capacity so holding alone cannot exclude control.
  // Zero makes every such arrival an immediate overload rejection.
  iree_host_size_t holding_capacity;
} loom_serve_router_options_t;

// Routes JSON request bodies by their model field and checkpoint operations by
// X-Loom-Model. A checkpoint request may omit its selector with exactly one
// service. GET /v1/models lists deployment names and namespaced capabilities;
// GET /v1/status adds copied scheduler ownership. /healthz and /readyz are
// model-free: constructors have completed, not necessarily weight activation.
// Zero services is valid: model/status lists are empty, work counters are zero,
// and inference or tokenization requests receive model-not-registered errors.
// GET /metrics exposes the same copied ownership as Prometheus text gauges
// and cumulative model-protocol counters without invoking models/devices.
// Server counters sum the fixed catalog's live-through-return owners, excluding
// router-handled utilities, observation and rejected routes. If a service has
// no counters, whole-server totals are unavailable rather than partial.
// GET /logz polls server diagnostics and current totals. X-Loom-Model instead
// selects a model service's stream; X-Loom-Cursor resumes that reader's
// previous poll. Each response copies records and reports gaps/current counters
// without retaining slots or slowing publication. See http/logs.h for cursor
// rules.
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
// asynchronous callbacks before detaching their wake targets. Return leaves
// outstanding claims owned by services; the caller destroys them before the
// borrowed transport. Failed readiness is not queue retirement: model owners
// perform the outer device join, never a router callback.
iree_status_t loom_serve_router_run(iree_task_executor_t* executor,
                                    loom_serve_http_server_t* server,
                                    iree_host_size_t service_count,
                                    const loom_serve_http_service_t* services,
                                    loom_serve_router_options_t options,
                                    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_ROUTER_H_
