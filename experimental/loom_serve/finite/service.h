// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_FINITE_SERVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_FINITE_SERVICE_H_

#include "experimental/loom_serve/finite/model.h"
#include "experimental/loom_serve/finite/protocol.h"
#include "experimental/loom_serve/http/service.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_finite_service_t loom_serve_finite_service_t;

// Bounded finite-vector admission. The borrowed source model, protocol and
// route name outlive the service. CPU parsing, source preparation, batch
// assembly and encoding share the host executor. Device ALL readiness uses the
// proactor; workers never wait. Compatible requests already waiting fill the
// oldest request's cohort up to its source-defined limit, without an arrival
// delay. capacity bounds all outstanding host requests, including the live
// cohort. A nonnegative heartbeat_interval controls periodic counter snapshots;
// zero disables periodic samples, not counters or lifecycle events. Observation
// is a bounded overwriting stream exposed through the HTTP interface, never a
// synchronous stderr write on a worker.
iree_status_t loom_serve_finite_service_create(
    loom_serve_host_t* host, loom_serve_finite_model_t* model,
    const loom_serve_finite_protocol_t* protocol, iree_string_view_t name,
    iree_host_size_t capacity, iree_duration_t heartbeat_interval,
    loom_serve_finite_service_t** out_service, iree_allocator_t allocator);

loom_serve_http_service_t loom_serve_finite_service_interface(
    loom_serve_finite_service_t* service);

// After router retirement only. CPU/process release, not completion, releases
// borrowed inputs/results. Failed device work belongs to outer model teardown.
void loom_serve_finite_service_destroy(loom_serve_finite_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_FINITE_SERVICE_H_
