// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_EVENT_OPERATIONS_H_
#define LIBHRX_SRC_BINDING_HIP_EVENT_OPERATIONS_H_

#include "binding/hip/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;

// Runs hipEventSynchronize after the public handle and current context have
// been validated. |event| must remain retained for the call. Returns through
// |out_captured_path| whether capture policy, including snapshot failure,
// produced the result.
hipError_t iree_hip_event_synchronize_retained(
    iree_hal_streaming_event_t* event, bool* out_captured_path);

// Runs hipEventQuery after the public handle and current context have been
// validated. |event| must remain retained for the call. Returns through
// |out_captured_path| whether capture policy, including snapshot failure,
// produced the result.
hipError_t iree_hip_event_query_retained(iree_hal_streaming_event_t* event,
                                         bool* out_captured_path);

// Runs hipEventElapsedTime after both public handles and |milliseconds| have
// been validated. Both events must remain retained for the call.
hipError_t iree_hip_event_elapsed_time_retained(
    float* milliseconds, iree_hal_streaming_event_t* start_event,
    iree_hal_streaming_event_t* stop_event);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_EVENT_OPERATIONS_H_
