// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"

struct amdf_xdna_umd_kernel_queue_t {
  // Host allocator copied for queue teardown.
  amdf_allocator_t host_allocator;
  // Context execution state exclusively leased by this public queue.
  amdf_windows_xdna_kernel_execution_t* execution;
};

amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context, uint32_t capacity,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_kernel_queue_t* queue = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(amdf_xdna_umd_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->host_allocator = device->host_allocator;
  status = amdf_windows_xdna_kernel_execution_acquire_queue(
      context->kernel_execution, capacity);
  if (amdf_status_is_ok(status)) {
    queue->execution = context->kernel_execution;
    *out_queue = queue;
  } else {
    amdf_free(queue->host_allocator, queue);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_native_submission) {
  return amdf_windows_xdna_kernel_execution_submit(
      queue->execution, slot, instruction_address, instruction_byte_length,
      out_native_submission);
}

amdf_native_event_types_t amdf_xdna_umd_kernel_queue_query_notification_types(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  (void)queue;
  return AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
}

amdf_status_t amdf_xdna_umd_kernel_queue_request_notification(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  return amdf_windows_xdna_kernel_execution_request_notification(
      queue->execution, native_submission, event);
}

uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_query_progress(queue->execution);
}

amdf_status_t amdf_xdna_umd_kernel_queue_refresh_progress(
    amdf_xdna_umd_kernel_queue_t* queue) {
  // The progress reader already samples the mapped native fence.
  (void)queue;
  return AMDF_STATUS_OK;
}

void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot) {
  amdf_windows_xdna_kernel_execution_retire_command(queue->execution, slot);
}

amdf_status_t amdf_xdna_umd_kernel_queue_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  return amdf_windows_xdna_kernel_execution_wait(queue->execution,
                                                 native_submission, deadline);
}

amdf_status_t amdf_xdna_umd_kernel_queue_query_terminal_status(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_query_terminal_status(
      queue->execution);
}

amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue) {
  const amdf_status_t status =
      amdf_windows_xdna_kernel_execution_release_queue(queue->execution);
  amdf_free(queue->host_allocator, queue);
  return status;
}
