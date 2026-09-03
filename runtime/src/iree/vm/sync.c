// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/sync.h"

#include "iree/base/threading/notification.h"
#include "iree/vm/program_storage.h"

// This file is a separate archive member so hosts using only the asynchronous
// core do not pull notification-based blocking support into their binaries.

typedef struct iree_vm_sync_wait_t {
  // Reusable process-local notification receiving provider wakes.
  iree_notification_t notification;
  // Prepared wait spanning one start or resume drive.
  iree_wait_token_t token;
} iree_vm_sync_wait_t;

static void iree_vm_sync_wake(void* user_data) {
  iree_vm_sync_wait_t* wait = (iree_vm_sync_wait_t*)user_data;
  iree_notification_post(&wait->notification, IREE_ALL_WAITERS);
}

static void iree_vm_sync_wait_initialize(iree_vm_sync_wait_t* out_wait) {
  iree_notification_initialize(&out_wait->notification);
  out_wait->token = iree_notification_prepare_wait(&out_wait->notification);
}

static void iree_vm_sync_wait_once(iree_vm_sync_wait_t* wait) {
  iree_notification_commit_wait(&wait->notification, wait->token,
                                IREE_DURATION_ZERO, IREE_TIME_INFINITE_FUTURE);
  wait->token = iree_notification_prepare_wait(&wait->notification);
}

static void iree_vm_sync_wait_deinitialize(iree_vm_sync_wait_t* wait) {
  iree_notification_cancel_wait(&wait->notification);
  iree_notification_deinitialize(&wait->notification);
}

IREE_API_EXPORT iree_status_t iree_vm_invoke(iree_vm_invocation_t* invocation,
                                             iree_vm_function_t function,
                                             iree_vm_variant_span_t arguments,
                                             iree_vm_variant_span_t results) {
  if (!iree_vm_program_target_may_yield(function.target_bits)) {
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    return iree_vm_invocation_start(invocation, function, arguments, results,
                                    (iree_vm_invocation_wake_callback_t){0},
                                    &outcome);
  }

  iree_vm_sync_wait_t wait;
  iree_vm_sync_wait_initialize(&wait);
  const iree_vm_invocation_wake_callback_t wake_callback = {
      iree_vm_sync_wake,
      &wait,
  };
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_start(
      invocation, function, arguments, results, wake_callback, &outcome);
  while (iree_status_is_ok(status) &&
         outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    iree_vm_sync_wait_once(&wait);
    status = iree_vm_invocation_resume(invocation, results, &outcome);
  }
  iree_vm_sync_wait_deinitialize(&wait);
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_process_create(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments, iree_allocator_t host_allocator,
    iree_vm_process_t** out_process) {
  if (!out_process) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_process is required");
  }
  *out_process = NULL;

  if (!program ||
      !iree_vm_program_target_may_yield(program->initializer.target_bits)) {
    iree_vm_process_create_outcome_t outcome = {0};
    iree_status_t status = iree_vm_process_create_start(
        program, invocation, arguments, (iree_vm_invocation_wake_callback_t){0},
        host_allocator, &outcome);
    if (iree_status_is_ok(status)) *out_process = outcome.process;
    return status;
  }

  iree_vm_sync_wait_t wait;
  iree_vm_sync_wait_initialize(&wait);
  const iree_vm_invocation_wake_callback_t wake_callback = {
      iree_vm_sync_wake,
      &wait,
  };
  iree_vm_process_create_outcome_t outcome = {0};
  iree_status_t status = iree_vm_process_create_start(
      program, invocation, arguments, wake_callback, host_allocator, &outcome);
  while (iree_status_is_ok(status) &&
         outcome.execution_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    iree_vm_sync_wait_once(&wait);
    status = iree_vm_process_create_resume(invocation, &outcome);
  }
  iree_vm_sync_wait_deinitialize(&wait);
  if (iree_status_is_ok(status)) *out_process = outcome.process;
  return status;
}
