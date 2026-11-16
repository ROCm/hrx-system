// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_REGISTRATION_H_
#define LOOM_SERVICE_FINITE_REGISTRATION_H_

#include "loom/service/finite/model.h"
#include "loom/service/finite/service.h"
#include "loom/service/registration.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_service_finite_registration_options_t {
  // Nonzero composition-assigned lifetime, changed when a slot is reused.
  uint64_t incarnation;
  // Immutable modality contract, borrowed through retirement.
  const loom_service_finite_protocol_t* protocol;
  // Deployment identifier, borrowed through retirement.
  iree_string_view_t name;
  // Maximum outstanding requests, including the active cohort.
  iree_host_size_t capacity;
  // Retained execution choices; zero disables reconfiguration.
  iree_host_size_t variant_capacity;
  // Nonnegative observation interval; zero disables periodic samples.
  iree_duration_t heartbeat_interval;
  // Borrowed registration history, or NULL for service-owned history.
  loom_runtime_telemetry_stream_t* stream;
} loom_service_finite_registration_options_t;

// Creates an unstarted native owner without source I/O, model allocation, or
// work submission. Copies option values; their views, device, pool, protocol,
// stream, and observer remain borrowed through actual retirement. The returned
// registration owns the finite provider and its retained observation tail.
iree_status_t loom_service_finite_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* model_options,
    const loom_service_finite_registration_options_t* options,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_service_registration_t** out_registration);

// Lends the typed engine of a finite registration only in READY. The
// application joins this borrow before request_retirement; independent receipts
// may outlive the registration. The argument must come from the finite
// allocator above.
loom_service_finite_service_t* loom_service_finite_registration_service(
    loom_service_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_FINITE_REGISTRATION_H_
