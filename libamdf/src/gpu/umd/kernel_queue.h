// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_UMD_KERNEL_QUEUE_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_umd_kernel_queue_t amdf_gpu_umd_kernel_queue_t;

// Acquires one native kernel-mediated queue accepting |command_type|.
// Cold path: command and completion resources are prepared before publication,
// including reusable native wait state. No first-submit or first-wait setup.
amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_gpu_umd_kernel_queue_t** out_queue);

// Publishes one immutable native command range and returns its progress value.
// Common code exclusively owns the submission slot. This hot path takes no
// provider lock and performs no allocation, initialization, command parsing,
// address lookup or native-submit retry. Native driver entry is permitted.
amdf_status_t amdf_gpu_umd_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t* out_native_submission);

// Samples the native progress fence with device-to-host acquire semantics.
// No locks, allocation, initialization, system calls or active polling.
uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Returns the observed terminal device failure without native queries. An
// operation error alone does not fail the queue or establish retirement.
// Atomic observation only; no locks, allocation, initialization or polling.
amdf_status_t amdf_gpu_umd_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Waits for one native progress value with caller-selected polling.
// Explicit synchronization path: clocks, polling, yielding, native waits and
// wait-event serialization are permitted within the same deadline. All reusable
// resources already exist; no provider allocation or lazy setup occurs here.
amdf_status_t amdf_gpu_umd_kernel_queue_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline);

// Releases an idle native GPU queue.
amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KERNEL_QUEUE_H_
