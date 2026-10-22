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
#include "libamdf/src/kernel_queue_dependency.h"
#include "libamdf/src/wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_umd_kernel_queue_t amdf_gpu_umd_kernel_queue_t;

// Maximum unretired prefix admitted before queue-ordered mapping publication,
// or zero when unavailable. Cached capability query without native work.
uint32_t amdf_gpu_umd_kernel_queue_mapping_capacity(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Supported native wake destinations. Cached query without native work.
amdf_native_event_types_t amdf_gpu_umd_kernel_queue_query_notification_types(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Requests one native-progress hint into a validated supported event. Zero
// means the shared owner already checked the public point and requires an
// immediate hint. No event descriptor or subscription is retained. Explicit
// native registration/signaling calls are allowed; library allocation, waits,
// locks, retries and lazy setup are not. Concurrent calls are independent of
// submission and the reusable synchronous-wait event.
amdf_status_t amdf_gpu_umd_kernel_queue_request_notification(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event);

// Acquires one native kernel-mediated queue accepting |command_type|.
// Cold path: command and completion resources are prepared before publication,
// including reusable native wait state. No first-submit or first-wait setup.
amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_kernel_queue_capabilities_t required_capabilities,
    amdf_gpu_umd_kernel_queue_t** out_queue);

// Borrows the exportable native progress identity, or NULL when not
// provisioned.
const amdf_kernel_queue_native_progress_t*
amdf_gpu_umd_kernel_queue_query_native_progress(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Imports a compatible native producer into this queue's consumer domain.
// Cold construction leaves out_native unchanged on failure.
amdf_status_t amdf_gpu_umd_kernel_queue_create_native_dependency(
    amdf_gpu_umd_kernel_queue_t* queue,
    const amdf_kernel_queue_native_progress_t* progress,
    amdf_kernel_queue_native_dependency_t** out_native);

// Publishes native dependency waits followed by one immutable command range
// with the supplied progress point. Common code assigns successive points
// starting at one, excludes UINT64_MAX, and exclusively owns publication.
// Rejection before any native operation leaves the point unaccepted. Reports
// the accepted wait prefix even when a later wait or command is rejected.
// A fresh native queue starts at progress zero. This hot path takes no
// provider lock and performs no allocation, initialization, command parsing,
// address lookup or native-submit retry. Native driver entry is permitted.
// Failure after acceptance returns OK and latches terminal queue failure;
// the assigned point remains live until independently checked retirement.
amdf_status_t amdf_gpu_umd_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t submission,
    uint32_t wait_count, const amdf_kernel_queue_dependency_wait_t* waits,
    uint32_t* out_accepted_wait_count);

// Samples the native progress fence with device-to-host acquire semantics.
// No locks, allocation, initialization, system calls or active polling.
uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue);

// Returns cached terminal queue/device failure without native queries. A
// rejected command alone does not fail the queue or establish retirement.
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

// Consumes an idle native GPU queue even when native cleanup fails. The shared
// layer has already established retirement. Native errors retain their domains;
// API-domain BUSY is reserved for the shared layer's precondition rejection.
amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KERNEL_QUEUE_H_
