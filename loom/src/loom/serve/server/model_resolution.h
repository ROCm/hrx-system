// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_MODEL_RESOLUTION_H_
#define LOOM_SERVE_SERVER_MODEL_RESOLUTION_H_

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/server/configuration.h"
#include "loomc/task.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold card I/O and selection, preceding native modality construction.
typedef struct loom_serve_model_resolution_t loom_serve_model_resolution_t;

// Borrows configuration and observer until take joins accepted work. Failure
// leaves out_resolution NULL and owes no notification. Accepted work owes one
// nonblocking wake after final publication, possibly before start returns.
iree_status_t loom_serve_model_resolution_start(
    loomc_task_sink_t sink,
    const loom_serve_model_configuration_t* configuration,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_model_resolution_t** out_resolution);

// False leaves every argument unchanged while work is pending. True joins the
// result into status, transfers any resolved declaration, and frees/clears the
// operation. Cancellation still joins accepted I/O before releasing inputs.
bool loom_serve_model_resolution_take(
    loom_serve_model_resolution_t** resolution, iree_status_t* status,
    loom_serve_model_configuration_t* out_configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_MODEL_RESOLUTION_H_
