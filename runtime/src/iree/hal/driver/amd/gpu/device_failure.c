// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/device_failure.h"

#include "iree/hal/driver/amd/status.h"

static void iree_hal_amd_gpu_device_failure_notify(void* user_data,
                                                   amdf_status_t status,
                                                   const char* operation) {
  iree_hal_amd_gpu_device_failure_t* failure = user_data;
  iree_slim_mutex_lock(&failure->mutex);
  const bool first = iree_hal_amd_gpu_failure_begin(&failure->state, status,
                                                    operation, iree_time_now());
  if (first) {
    // A concurrent observer must not return from notification before the first
    // fault has armed monitoring and published its terminal ownership request.
    iree_hal_amd_gpu_publication_watch_failure(failure->publication,
                                               &failure->state);
    iree_hal_amd_gpu_completion_request_failure(failure->completion, status);
    iree_notification_post(&failure->membership->notification,
                           IREE_ALL_WAITERS);
  }
  iree_slim_mutex_unlock(&failure->mutex);
  if (!first) {
    return;
  }
  // Independent reporting is armed before any native wake or stop can block.
  iree_hal_amd_gpu_context_report(failure->context, status, operation);
  const amdf_status_t wake_status =
      iree_hal_amd_gpu_completion_request_stop(failure->completion);
  if (!amdf_status_is_ok(wake_status)) {
    iree_hal_amd_gpu_context_report(failure->context, wake_status,
                                    "GPU terminal observer wake");
  }
}

static bool iree_hal_amd_gpu_device_failure_wake(void* user_data) {
  iree_hal_amd_gpu_device_failure_t* failure = user_data;
  return iree_hal_amd_gpu_failure_wake(&failure->state);
}

static bool iree_hal_amd_gpu_device_failure_step(
    iree_hal_amd_gpu_device_failure_t* failure,
    iree_hal_amd_gpu_failure_phase_t phase, iree_status_code_t code,
    amdf_status_t* out_status) {
  if (!iree_hal_amd_gpu_queue_membership_try_failure(failure->membership, phase,
                                                     code, out_status)) {
    return false;
  }
  if (!amdf_status_is_ok(*out_status)) {
    return true;
  }
  if (phase == IREE_HAL_AMD_GPU_FAILURE_INACTIVATING) {
    iree_hal_amd_gpu_failure_set_phase(&failure->state, phase, UINT32_MAX - 1,
                                       0, 1);
    if (!iree_hal_amd_gpu_sdma_queue_try_inactivate(failure->transfers,
                                                    out_status)) {
      return false;
    }
    if (!amdf_status_is_ok(*out_status)) {
      return true;
    }
    iree_hal_amd_gpu_failure_set_phase(&failure->state, phase, UINT32_MAX, 0,
                                       1);
    return iree_hal_amd_gpu_aql_code_try_inactivate(failure->code, out_status);
  }
  if (phase == IREE_HAL_AMD_GPU_FAILURE_RETIRING) {
    iree_hal_amd_gpu_failure_set_phase(&failure->state, phase, UINT32_MAX - 1,
                                       0, 1);
    return iree_hal_amd_gpu_sdma_queue_retire_failed(failure->transfers);
  }
  if (phase == IREE_HAL_AMD_GPU_FAILURE_JOINING) {
    iree_hal_amd_gpu_failure_set_phase(&failure->state, phase, UINT32_MAX - 1,
                                       0, 1);
    if (!iree_hal_amd_gpu_sdma_queue_is_host_idle(failure->transfers)) {
      return false;
    }
    iree_hal_amd_gpu_failure_set_phase(&failure->state, phase, UINT32_MAX, 0,
                                       1);
    return iree_hal_amd_gpu_aql_code_is_host_idle(failure->code);
  }
  return true;
}

static void iree_hal_amd_gpu_device_failure_join(void* user_data,
                                                 amdf_status_t native_error) {
  iree_hal_amd_gpu_device_failure_t* failure = user_data;
  // STOP_REQUESTED excludes every healthy helper. Completion's final failure
  // broadcast has joined before this callback; source-table setters do not
  // observe native state. Native joins need no observer mutex, so publishers
  // and cold acquired-queue releases can make their final host handoffs.
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_error, NULL);
  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  const amdf_status_t close_status =
      iree_hal_amd_gpu_aql_code_close(failure->code, native_error);
  if (!amdf_status_is_ok(close_status)) {
    iree_hal_amd_gpu_context_report(failure->context, close_status,
                                    "GPU code loader terminal wake");
  }
  amdf_status_t cleanup_status = AMDF_STATUS_OK;
  bool expired = false;
  for (iree_hal_amd_gpu_failure_phase_t phase =
           IREE_HAL_AMD_GPU_FAILURE_CLOSING;
       phase != IREE_HAL_AMD_GPU_FAILURE_JOINING &&
       amdf_status_is_ok(cleanup_status);
       ++phase) {
    bool ready = false;
    while (!ready && amdf_status_is_ok(cleanup_status)) {
      const iree_wait_token_t token =
          iree_notification_prepare_wait(&failure->state.changed);
      ready = iree_hal_amd_gpu_device_failure_step(failure, phase, code,
                                                   &cleanup_status);
      if (ready) {
        iree_notification_cancel_wait(&failure->state.changed);
      } else if (!iree_notification_commit_wait(&failure->state.changed, token,
                                                0, failure->state.deadline)) {
        cleanup_status =
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
        expired = true;
        iree_hal_amd_gpu_failure_poll(&failure->state, iree_time_now());
      }
    }
  }
  if (!amdf_status_is_ok(cleanup_status) && !expired) {
    iree_hal_amd_gpu_context_report(
        failure->context, cleanup_status,
        "GPU terminal native access unproven; preserving ownership");
  }
  // Native stop failure preserves issued captures, but cannot detach a live
  // CPU borrower. In particular, retaining GPU storage does not extend the
  // application's callback-data or allocator lifetime past final release.
  // After one deadline diagnostic, await real host handoffs without repeatedly
  // polling an expired deadline or retrying consumed native identifiers.
  iree_time_t deadline =
      expired ? IREE_TIME_INFINITE_FUTURE : failure->state.deadline;
  bool ready = false;
  while (!ready) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&failure->state.changed);
    amdf_status_t host_status = AMDF_STATUS_OK;
    ready = iree_hal_amd_gpu_device_failure_step(
        failure, IREE_HAL_AMD_GPU_FAILURE_JOINING, code, &host_status);
    if (ready) {
      iree_notification_cancel_wait(&failure->state.changed);
    } else if (!iree_notification_commit_wait(&failure->state.changed, token, 0,
                                              deadline)) {
      // Reporting a late CPU join does not revoke an established native stop.
      iree_hal_amd_gpu_failure_poll(&failure->state, iree_time_now());
      deadline = IREE_TIME_INFINITE_FUTURE;
    }
  }
  // An acquisition callback can enqueue a nested native release before its
  // final context access. Keep independent deadline reporting armed until
  // those callback tails join, including when native backing stays retained.
  iree_hal_amd_gpu_context_join_memory(failure->context);
  iree_hal_amd_gpu_failure_finish(&failure->state);
  iree_hal_amd_gpu_queue_membership_finish_failure(failure->membership,
                                                   cleanup_status);
}

void iree_hal_amd_gpu_device_failure_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    uint32_t timeout_ms, iree_hal_amd_gpu_publication_t* publication,
    iree_hal_amd_gpu_completion_t* completion,
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_sdma_queue_t* transfers, iree_hal_amd_gpu_aql_code_t* code,
    iree_hal_amd_gpu_device_failure_t* failure) {
  *failure = (iree_hal_amd_gpu_device_failure_t){
      .context = context,
      .publication = publication,
      .completion = completion,
      .membership = membership,
      .transfers = transfers,
      .code = code,
  };
  iree_hal_amd_gpu_failure_initialize(&context->endpoint_info, device,
                                      context->event_sink, timeout_ms,
                                      &failure->state);
  iree_slim_mutex_initialize(&failure->mutex);
  membership->failure = &failure->state;
  context->execution_failure.notify = iree_hal_amd_gpu_device_failure_notify;
  context->execution_failure.join = iree_hal_amd_gpu_device_failure_join;
  context->execution_failure.wake = iree_hal_amd_gpu_device_failure_wake;
  context->execution_failure.user_data = failure;
}

void iree_hal_amd_gpu_device_failure_deinitialize(
    iree_hal_amd_gpu_device_failure_t* failure) {
  if (!failure->context) {
    return;
  }
  memset(&failure->context->execution_failure, 0,
         sizeof(failure->context->execution_failure));
  iree_hal_amd_gpu_failure_deinitialize(&failure->state);
  iree_slim_mutex_deinitialize(&failure->mutex);
  failure->context = NULL;
}
