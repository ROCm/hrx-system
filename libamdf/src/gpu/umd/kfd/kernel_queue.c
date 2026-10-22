// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kernel_queue.h"

#include <stddef.h>

// KFD exposes user-published queues, not this kernel-submission service.
uint32_t amdf_gpu_umd_kernel_queue_mapping_capacity(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return 0;
}

amdf_native_event_types_t amdf_gpu_umd_kernel_queue_query_notification_types(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return 0;
}

amdf_status_t amdf_gpu_umd_kernel_queue_request_notification(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  (void)queue;
  (void)native_submission;
  (void)event;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_kernel_queue_capabilities_t required_capabilities,
    amdf_gpu_umd_kernel_queue_t** out_queue) {
  (void)device;
  (void)command_type;
  (void)required_capabilities;
  (void)out_queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t submission,
    uint32_t wait_count, const amdf_kernel_queue_dependency_wait_t* waits,
    uint32_t* out_accepted_wait_count) {
  (void)queue;
  (void)command_buffer_address;
  (void)command_buffer_byte_length;
  (void)submission;
  (void)wait_count;
  (void)waits;
  *out_accepted_wait_count = 0;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

const amdf_kernel_queue_native_progress_t*
amdf_gpu_umd_kernel_queue_query_native_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return NULL;
}

amdf_status_t amdf_gpu_umd_kernel_queue_create_native_dependency(
    amdf_gpu_umd_kernel_queue_t* queue,
    const amdf_kernel_queue_native_progress_t* progress,
    amdf_kernel_queue_native_dependency_t** out_native) {
  (void)queue;
  (void)progress;
  (void)out_native;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  // No kernel queue can be created, so no submission can have progressed.
  return 0;
}

amdf_status_t amdf_gpu_umd_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_kernel_queue_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  (void)queue;
  (void)native_submission;
  (void)deadline;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}
