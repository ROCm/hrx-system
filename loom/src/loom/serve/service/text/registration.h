// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_
#define LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_

#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/service.h"
#include "loom/service/finite/model.h"
#include "loom/service/registration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an unstarted native owner. Copies model shape inputs and option
// values; other views, device, pool, stream, and observer survive retirement.
// image_options optionally declares an independently prepared component.
// options.image and options.scheduling are empty: this owner supplies them.
// Resolved source cache capacities replace their service option counterparts.
iree_status_t loom_serve_text_registration_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_serve_text_options_t* model_options,
    const loom_serve_text_service_options_t* options,
    const loom_service_finite_model_options_t* image_options,
    uint64_t incarnation, loom_runtime_wake_callback_t observer,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_service_registration_t** out_registration);

// Borrows the typed engine only in READY. The argument comes from the factory
// above; the caller relinquishes this view before native retirement handoff.
loom_serve_text_service_t* loom_serve_text_registration_service(
    loom_service_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_REGISTRATION_H_
