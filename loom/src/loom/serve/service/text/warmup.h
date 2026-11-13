// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_WARMUP_H_
#define LOOM_SERVE_SERVICE_TEXT_WARMUP_H_

#include "loom/runtime/device/device.h"
#include "loom/runtime/parameter/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// Demand-driven parameter loading on an independent execution timeline. The
// model owns this operation across preparation and service publication; its
// parameter group outlives the operation, including failed native work.
typedef struct loom_serve_text_warmup_t loom_serve_text_warmup_t;

// Allocates unstarted ownership. No parameter I/O or capacity admission occurs.
iree_status_t loom_serve_text_warmup_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    loom_runtime_residency_group_t* group, void (*wake)(void* user_data),
    void* user_data, iree_allocator_t host_allocator,
    loom_serve_text_warmup_t** out_warmup);

// Attempts admission once, or joins a published completion without waiting.
// Capacity denial withdraws this startup opportunity; request admission owns
// subsequent waiting. Accepted work keeps its pin until readiness, including
// when the originating request disappears. Terminal errors remain queryable.
void loom_serve_text_warmup_poll(loom_serve_text_warmup_t* warmup);

// Transfers the scheduling payload at preparation/service/retirement
// boundaries, joining the previous observer's final access. NULL detaches the
// observer.
void loom_serve_text_warmup_set_observer(loom_serve_text_warmup_t* warmup,
                                         void (*wake)(void* user_data),
                                         void* user_data);

// The completion callback still owns operation storage. NULL is not pending.
bool loom_serve_text_warmup_is_pending(const loom_serve_text_warmup_t* warmup);

// Clones the terminal outcome; NULL or pending successful work returns OK.
iree_status_t loom_serve_text_warmup_status(
    const loom_serve_text_warmup_t* warmup);

// Outer-thread destruction helper. Joins accepted readiness and notification
// ownership without changing device or shared pool lifetimes.
void loom_serve_text_warmup_await(loom_serve_text_warmup_t* warmup);

// Releases terminal ownership after model consumers and weight I/O retire.
// The owner collects the operation outcome with warmup_status before release;
// destruction itself is infallible. NULL is accepted.
void loom_serve_text_warmup_free(loom_serve_text_warmup_t* warmup);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_WARMUP_H_
