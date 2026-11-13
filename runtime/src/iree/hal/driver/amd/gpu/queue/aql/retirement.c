// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/retirement.h"

#include "iree/hal/driver/amd/gpu/queue/aql/state.h"

static void iree_hal_amd_gpu_aql_complete_download(void* user_data) {
  iree_hal_amd_gpu_download_t* download = user_data;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_download_complete");
  iree_hal_amd_gpu_download_complete(download);
  download->length = 0;
  IREE_TRACE_ZONE_END(z0);
}

// A physical completion covers a finite ordered run. Coalesce only the
// completed prefix sharing one native timeline; every logical signal keeps
// its own outcome and independently retained readers. The first action has
// already passed ordinary native and dependency readiness checks.
static uint64_t iree_hal_amd_gpu_aql_retire_run(
    iree_hal_amd_gpu_aql_queue_t* queue, uint64_t position, uint64_t limit,
    iree_hal_amd_gpu_completion_scope_t* scope) {
  const uint64_t mask = queue->options.retirement_capacity - 1;
  iree_hal_amd_gpu_aql_retirement_t* first = &queue->issued[position & mask];
  iree_hal_amd_gpu_aql_submission_t* first_submission = first->submission;
  if (first_submission->signals.count != 1 ||
      !iree_hal_amd_gpu_semaphore_isa(
          first_submission->signals.semaphores[0])) {
    return position;
  }
  iree_hal_semaphore_t* semaphore = first_submission->signals.semaphores[0];
  const uint64_t* native = first->signal->host_value;
  uint64_t value = 0;
  uint64_t end = position;
  for (; end < limit; ++end) {
    iree_hal_amd_gpu_aql_retirement_t* action = &queue->issued[end & mask];
    iree_hal_amd_gpu_aql_submission_t* submission = action->submission;
    if (action->signal->host_value != native || !action->terminal ||
        submission->entry.kind != IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION ||
        submission->profiled || submission->signals.count != 1 ||
        submission->signals.semaphores[0] != semaphore ||
        submission->signals.payload_values[0] < value ||
        iree_atomic_load(&submission->failure_code,
                         iree_memory_order_relaxed)) {
      break;
    }
    bool ready = true;
    for (iree_host_size_t i = 0; ready && i < submission->wait_count; ++i) {
      iree_hal_amd_gpu_signal_t* dependency = submission->waits[i].signal;
      if (dependency) {
        // Pending same-physical readers name earlier accepted actions in this
        // run. Their native effects are complete; this owner publishes their
        // logical outcomes after the shared public timeline advances.
        iree_status_code_t code;
        ready = iree_hal_amd_gpu_signal_query_retirement(dependency, &code)
                    ? code == IREE_STATUS_OK
                    : dependency->host_value == native;
      }
    }
    if (!ready) {
      break;
    }
    value = submission->signals.payload_values[0];
  }
  if (end - position < 2) {
    return position;
  }
  iree_hal_amd_gpu_completion_scope_unlock(scope);
  // All covered payload borrows end before any timeline callback can observe
  // success. Captures and logical readers remain owned until publication joins.
  for (uint64_t i = position; i < end; ++i) {
    iree_hal_amd_gpu_aql_submission_release_payload(
        queue->issued[i & mask].submission);
  }
  iree_async_single_frontier_t completed_frontier;
  const iree_async_frontier_t* frontier = NULL;
  if (queue->frontier.tracker) {
    iree_async_single_frontier_initialize(&completed_frontier,
                                          queue->frontier.axis, end);
    frontier =
        iree_async_single_frontier_as_const_frontier(&completed_frontier);
    iree_async_frontier_tracker_advance(queue->frontier.tracker,
                                        queue->frontier.axis, end);
  }
  const iree_status_t status =
      iree_hal_amd_gpu_semaphore_publish_completed(semaphore, value, frontier);
  const iree_status_code_t code = iree_status_code(status);
  for (uint64_t i = position; i < end; ++i) {
    iree_hal_amd_gpu_aql_retirement_t* action = &queue->issued[i & mask];
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_queue_fail_submission(
          &queue->base, action->submission, iree_status_clone(status));
    }
    iree_hal_amd_gpu_signal_retire(action->signal, code);
    iree_hal_amd_gpu_signal_release(action->signal);
    iree_hal_amd_gpu_aql_submission_release(action->submission);
  }
  iree_status_free(status);
  iree_atomic_store(&queue->completed.arguments,
                    queue->issued[(end - 1) & mask].arguments,
                    iree_memory_order_release);
  iree_atomic_store(&queue->completed.count, end, iree_memory_order_release);
  return end;
}

bool iree_hal_amd_gpu_aql_retirement_poll(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  iree_hal_amd_gpu_aql_queue_t* queue = user_data;
  uint64_t position = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                 iree_memory_order_relaxed);
  const uint64_t published = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_acquire);
  if (amdf_status_is_ok(wait_status) && position != published) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    wait_status = iree_hal_amd_gpu_aql_scratch_poll(&queue->scratch);
  }
  if (!amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    iree_hal_amd_gpu_aql_queue_fail_native(&queue->base, wait_status,
                                           "AQL completion/scratch");
  }
  const iree_status_code_t failure = (iree_status_code_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  if (failure != IREE_STATUS_OK) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    if (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      iree_hal_amd_gpu_aql_queue_fail_issued(queue, failure);
    }
    return false;
  }
  const iree_status_code_t feedback_failure =
      iree_hal_amd_gpu_aql_queue_feedback_failure(queue);
  if (feedback_failure != IREE_STATUS_OK &&
      !queue->feedback_failure_delivered) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    // Shader reports invalidate results without ending native borrows. This
    // consumer retains each capture until its accepted pieces complete.
    for (uint64_t i = position; i < published; ++i) {
      iree_hal_amd_gpu_aql_queue_fail_submission(
          &queue->base,
          queue->issued[i & (queue->options.retirement_capacity - 1)]
              .submission,
          iree_make_status(feedback_failure, "GPU sanitizer failure"));
    }
    queue->feedback_failure_delivered = true;
  }
  const uint64_t initial = position;
  const uint64_t limit = iree_min(published, position + 64);
  for (; position < limit; ++position) {
    iree_hal_amd_gpu_aql_retirement_t* action =
        &queue->issued[position & (queue->options.retirement_capacity - 1)];
    iree_hal_amd_gpu_aql_submission_t* submission = action->submission;
    if (iree_atomic_load((iree_atomic_int64_t*)action->signal->host_value,
                         iree_memory_order_acquire) != 0) {
      if ((uint64_t)iree_atomic_load(&queue->run.position,
                                     iree_memory_order_acquire) ==
              position + 1 &&
          !(iree_atomic_fetch_or(&queue->run.pending,
                                 IREE_HAL_AMD_GPU_AQL_SEAL_ENQUEUED,
                                 iree_memory_order_acq_rel) &
            IREE_HAL_AMD_GPU_AQL_SEAL_ENQUEUED)) {
        iree_hal_amd_gpu_completion_scope_unlock(scope);
        iree_hal_amd_gpu_aql_queue_enqueue_entry(queue, &queue->run.entry);
      }
      out_wait->address = action->signal->host_value;
      out_wait->value = 1;
      out_wait->signal = action->signal;
      out_wait->is_64_bit = true;
      out_wait->pending = true;
      break;
    }
    iree_status_code_t code;
    if (!iree_hal_amd_gpu_aql_submission_retirement_ready(submission, &code)) {
      break;
    }
    if (queue->options.feedback) {
      // Native completion makes this work's reports observable before its
      // public completion can wake a host consumer.
      iree_hal_amd_gpu_completion_scope_lock(scope);
      iree_hal_amd_gpu_feedback_drain(queue->options.feedback);
      if (code == IREE_STATUS_OK) {
        code = iree_hal_amd_gpu_aql_queue_feedback_failure(queue);
      }
    }
    if (code == IREE_STATUS_OK) {
      const uint64_t end =
          iree_hal_amd_gpu_aql_retire_run(queue, position, limit, scope);
      if (end != position) {
        position = end - 1;
        continue;
      }
    }
    const uint64_t arguments = action->arguments;
    // Native execution consumed the ABI. Download and profile collection retain
    // their own storage and cannot prolong this argument generation's lifetime.
    iree_atomic_store(&queue->completed.arguments, arguments,
                      iree_memory_order_release);
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE) {
      iree_hal_amd_gpu_byte_storage_pool_retire(&queue->values,
                                                action->execute.values_end);
    }
    if ((submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
         code == IREE_STATUS_OK && action->transfer.download.length) ||
        submission->profiled) {
      if (iree_atomic_exchange(&queue->completed.waiting, 0,
                               iree_memory_order_acq_rel)) {
        iree_hal_amd_gpu_aql_queue_capacity_available(queue);
      }
    }
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
        code == IREE_STATUS_OK && action->transfer.download.length) {
      if (out_copy) {
        *out_copy = (iree_hal_amd_gpu_completion_copy_t){
            .fn = iree_hal_amd_gpu_aql_complete_download,
            .user_data = &action->transfer.download};
      }
      break;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_aql_retire");
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    if (submission->profiled) {
      const iree_hal_amd_gpu_profile_operation_t* operation =
          iree_hal_amd_gpu_aql_submission_profile(submission);
      const iree_hal_amd_gpu_profile_completion_t* completion =
          submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE
              ? &action->execute.profile
              : &operation->completion;
      iree_hal_amd_gpu_profile_operation_complete(queue->profile, operation,
                                                  completion);
      iree_hal_amd_gpu_byte_storage_pool_retire(queue->samples,
                                                completion->samples_end);
    }
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
      iree_atomic_store(&queue->completed.transfers, action->transfer.end,
                        iree_memory_order_release);
    }
    if (action->terminal) {
      iree_hal_amd_gpu_aql_submission_release_payload(submission);
    }
    iree_async_single_frontier_t completed_frontier;
    const iree_async_frontier_t* frontier = NULL;
    if (queue->frontier.tracker) {
      iree_async_single_frontier_initialize(&completed_frontier,
                                            queue->frontier.axis, position + 1);
      frontier =
          iree_async_single_frontier_as_const_frontier(&completed_frontier);
      iree_async_frontier_tracker_advance(queue->frontier.tracker,
                                          queue->frontier.axis, position + 1);
      iree_hal_amd_gpu_aql_submission_set_frontier(submission,
                                                   &completed_frontier);
    }
    iree_status_t status = iree_status_from_code(code);
    for (iree_host_size_t i = 0;
         action->terminal && i < submission->signals.count &&
         iree_status_is_ok(status);
         ++i) {
      iree_hal_semaphore_t* semaphore = submission->signals.semaphores[i];
      const uint64_t value = submission->signals.payload_values[i];
      status = iree_hal_amd_gpu_semaphore_isa(semaphore)
                   ? iree_hal_amd_gpu_semaphore_publish_completed(
                         semaphore, value, frontier)
                   : iree_hal_semaphore_signal(semaphore, value, frontier);
    }
    code = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_queue_fail_submission(&queue->base, submission,
                                                 status);
    }
    iree_hal_amd_gpu_signal_retire(action->signal, code);
    iree_hal_amd_gpu_signal_release(action->signal);
    iree_hal_amd_gpu_aql_submission_release(submission);
    iree_atomic_store(&queue->completed.count, position + 1,
                      iree_memory_order_release);
    IREE_TRACE_ZONE_END(z0);
  }
  if (position != initial && iree_atomic_exchange(&queue->completed.waiting, 0,
                                                  iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_queue_capacity_available(queue);
  }
  if (position == published) {
    *out_wait =
        (iree_hal_amd_gpu_completion_wait_t){.address = &queue->producer.count,
                                             .value = (int64_t)published,
                                             .is_64_bit = true};
  }
  return position != initial;
}
