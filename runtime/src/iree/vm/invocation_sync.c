// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/notification.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "iree/vm/program_storage.h"

// This file is a separate archive member so threadless hosts that use only the
// asynchronous core do not pull in notification-based blocking support.

static void iree_vm_sync_wake(void* user_data) {
  iree_notification_t* notification = (iree_notification_t*)user_data;
  iree_notification_post(notification, IREE_ALL_WAITERS);
}

IREE_API_EXPORT iree_status_t iree_vm_invoke(iree_vm_invocation_t* invocation,
                                             iree_vm_function_t function,
                                             iree_vm_variant_span_t arguments,
                                             iree_vm_variant_span_t results) {
  if (!iree_vm_program_target_may_yield(function.target_bits)) {
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const iree_vm_invocation_wake_callback_t wake_callback = {0};
    return iree_vm_invocation_start(invocation, function, arguments, results,
                                    wake_callback, &outcome);
  }

  iree_notification_t notification;
  iree_notification_initialize(&notification);
  iree_wait_token_t wait_token = iree_notification_prepare_wait(&notification);
  const iree_vm_invocation_wake_callback_t wake_callback = {
      .function = iree_vm_sync_wake,
      .user_data = &notification,
  };

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_start(
      invocation, function, arguments, results, wake_callback, &outcome);
  while (iree_status_is_ok(status) &&
         outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    iree_notification_commit_wait(&notification, wait_token, IREE_DURATION_ZERO,
                                  IREE_TIME_INFINITE_FUTURE);
    wait_token = iree_notification_prepare_wait(&notification);
    status = iree_vm_invocation_resume(invocation, results, &outcome);
  }
  iree_notification_cancel_wait(&notification);
  iree_notification_deinitialize(&notification);
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_process_create(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments, iree_allocator_t host_allocator,
    iree_vm_process_t** out_process) {
  if (!out_process) {
    if (arguments.data) iree_vm_variant_span_reset(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_process is required");
  }
  *out_process = NULL;

  if (!program ||
      !iree_vm_program_target_may_yield(program->initializer.target_bits)) {
    const iree_vm_invocation_wake_callback_t wake_callback = {0};
    iree_vm_process_create_outcome_t outcome = {0};
    iree_status_t status =
        iree_vm_process_create_start(program, invocation, arguments,
                                     wake_callback, host_allocator, &outcome);
    if (iree_status_is_ok(status)) *out_process = outcome.process;
    return status;
  }

  iree_notification_t notification;
  iree_notification_initialize(&notification);
  iree_wait_token_t wait_token = iree_notification_prepare_wait(&notification);
  const iree_vm_invocation_wake_callback_t wake_callback = {
      .function = iree_vm_sync_wake,
      .user_data = &notification,
  };

  iree_vm_process_create_outcome_t outcome = {0};
  iree_status_t status = iree_vm_process_create_start(
      program, invocation, arguments, wake_callback, host_allocator, &outcome);
  while (iree_status_is_ok(status) &&
         outcome.execution_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    iree_notification_commit_wait(&notification, wait_token, IREE_DURATION_ZERO,
                                  IREE_TIME_INFINITE_FUTURE);
    wait_token = iree_notification_prepare_wait(&notification);
    status = iree_vm_process_create_resume(invocation, &outcome);
  }
  iree_notification_cancel_wait(&notification);
  iree_notification_deinitialize(&notification);
  if (iree_status_is_ok(status)) *out_process = outcome.process;
  return status;
}
