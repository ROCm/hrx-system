// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_KERNEL_QUEUE_H_

#include "amdf/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Acquires one kernel-mediated native queue from a materialized GPU device.
amdf_status_t AMDF_CALL amdf_gpu_kernel_queue_create(
    amdf_device_t* device,
    const amdf_gpu_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue);

// Publishes one bounded array of already-materialized native command streams.
amdf_status_t AMDF_CALL amdf_gpu_kernel_queue_submit(
    amdf_kernel_queue_t* queue,
    const amdf_gpu_kernel_queue_submission_info_t* submission_info,
    uint64_t* out_submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_KERNEL_QUEUE_H_
