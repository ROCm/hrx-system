// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_USER_QUEUE_H_
#define AMDF_SRC_GPU_USER_QUEUE_H_

#include "amdf/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Acquires one directly published native queue from a materialized GPU device.
amdf_status_t AMDF_CALL amdf_gpu_user_queue_create(
    amdf_device_t* device, const amdf_gpu_user_queue_create_info_t* create_info,
    amdf_user_queue_t** out_queue);

// Samples a firmware scratch request without native calls or allocation.
amdf_status_t AMDF_CALL amdf_gpu_user_queue_query_scratch(
    amdf_user_queue_t* queue, amdf_gpu_aql_scratch_request_t* out_request);

// Validates caller backing, installs it, and releases the firmware pause.
amdf_status_t AMDF_CALL amdf_gpu_user_queue_resume_scratch(
    amdf_user_queue_t* queue, const amdf_gpu_queue_scratch_t* scratch);

// Detaches idle scratch after the caller joins all accepted native uses.
amdf_status_t AMDF_CALL
amdf_gpu_user_queue_trim_scratch(amdf_user_queue_t* queue);

// Selects native dispatch timestamps after the caller joins all accepted uses.
amdf_status_t AMDF_CALL amdf_gpu_user_queue_set_dispatch_timestamps(
    amdf_user_queue_t* queue, uint32_t enabled);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_USER_QUEUE_H_
