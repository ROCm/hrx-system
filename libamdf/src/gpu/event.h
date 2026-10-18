// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_EVENT_H_
#define AMDF_SRC_GPU_EVENT_H_

#include "amdf/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates one native event borrowing its exact producer device.
amdf_status_t AMDF_CALL amdf_gpu_event_create(
    amdf_device_t* device, const amdf_gpu_event_create_info_t* create_info,
    amdf_gpu_event_t** out_event);

// Copies cached native producer operands without entering the driver.
amdf_status_t AMDF_CALL amdf_gpu_event_query_info(
    amdf_gpu_event_t* event, amdf_gpu_event_info_t* out_info);

// Publishes a coalescible host wake, independent of application predicates.
amdf_status_t AMDF_CALL amdf_gpu_event_signal(amdf_gpu_event_t* event);

// Waits on this caller without starting a library worker or active polling.
amdf_status_t AMDF_CALL amdf_gpu_event_wait(amdf_gpu_event_t* event,
                                            uint64_t timeout_nanoseconds);

// Releases an event after all native producers and host calls have ended.
amdf_status_t AMDF_CALL amdf_gpu_event_destroy(amdf_gpu_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_EVENT_H_
