// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_TELEMETRY_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_TELEMETRY_H_

#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/telemetry/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copies owner-side weight submissions into bounded history. time_ns lets
// related observations share one timestamp; formatting is consumer-only.
void loom_serve_telemetry_record_weights(
    loom_serve_telemetry_stream_t* stream, iree_time_t time_ns,
    const loom_serve_weight_activity_t* activity);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_RUNTIME_PARAMETER_TELEMETRY_H_
