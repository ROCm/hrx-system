// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"

amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context, uint32_t capacity,
    amdf_xdna_instruction_features_t required_features,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  const amdf_status_t status = amdf_windows_xdna_kernel_execution_acquire_queue(
      context->kernel_execution, capacity, required_features);
  if (amdf_status_is_ok(status)) {
    // One public queue exclusively leases this context's native execution
    // state. The lease owns its packet storage; it needs no additional owner.
    *out_queue = (amdf_xdna_umd_kernel_queue_t*)context->kernel_execution;
  }
  return status;
}

uint32_t amdf_xdna_umd_kernel_queue_maximum_command_count(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_maximum_command_count(
      (const amdf_windows_xdna_kernel_execution_t*)queue);
}

bool amdf_xdna_umd_kernel_queue_has_capacity(
    const amdf_xdna_umd_kernel_queue_t* queue, uint32_t command_count) {
  return amdf_windows_xdna_kernel_execution_has_capacity(
      (const amdf_windows_xdna_kernel_execution_t*)queue, command_count);
}

amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot, uint32_t command_count,
    const amdf_xdna_umd_kernel_command_t* commands,
    uint64_t* out_native_submission) {
  return amdf_windows_xdna_kernel_execution_submit(
      (amdf_windows_xdna_kernel_execution_t*)queue, slot, command_count,
      commands, out_native_submission);
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
      (amdf_windows_xdna_kernel_execution_t*)queue, native_submission, event);
}

uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_query_progress(
      (const amdf_windows_xdna_kernel_execution_t*)queue);
}

amdf_status_t amdf_xdna_umd_kernel_queue_refresh_progress(
    amdf_xdna_umd_kernel_queue_t* queue) {
  // The progress reader already samples the mapped native fence.
  (void)queue;
  return AMDF_STATUS_OK;
}

void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot) {
  amdf_windows_xdna_kernel_execution_retire_command(
      (amdf_windows_xdna_kernel_execution_t*)queue, slot);
}

amdf_status_t amdf_xdna_umd_kernel_queue_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  return amdf_windows_xdna_kernel_execution_wait(
      (amdf_windows_xdna_kernel_execution_t*)queue, native_submission,
      deadline);
}

amdf_status_t amdf_xdna_umd_kernel_queue_query_terminal_status(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_query_terminal_status(
      (const amdf_windows_xdna_kernel_execution_t*)queue);
}

amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_windows_xdna_kernel_execution_release_queue(
      (amdf_windows_xdna_kernel_execution_t*)queue);
}
