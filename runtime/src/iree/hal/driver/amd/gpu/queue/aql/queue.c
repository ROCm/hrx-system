// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/queue.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/executable.h"
#include "iree/hal/driver/amd/gpu/queue/aql/growth.h"
#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"
#include "iree/hal/driver/amd/gpu/queue/aql/ring.h"
#include "iree/hal/driver/amd/gpu/queue/aql/scratch.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/hal/driver/amd/gpu/queue/aql/transfer.h"
#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/control_program.h"
#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/host_call.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

// Reusable admission sidecar. The wait set joins failure cancellation with
// inline tracker completion before returning this storage to the capture.
typedef struct iree_hal_amd_gpu_aql_pending_t {
  // Queue cancellation membership and terminal capture ownership.
  iree_hal_amd_gpu_host_wait_entry_t entry;
  // Queue whose accepted capture dominates this sidecar's lifetime.
  iree_hal_amd_gpu_aql_queue_t* queue;
  // Owned captured operation awaiting admission.
  iree_hal_amd_gpu_aql_submission_t* submission;
  // Owned allocation bytes when the capture tail cannot hold this sidecar,
  // or zero for borrowed tail storage. Returned after tracker callbacks join.
  iree_host_size_t allocation_size;
  // Single tracker allocation in this block's trailing storage. Finalization
  // frees the entire allocation, resetting the arena for the next registration.
  iree_allocator_inline_storage_t tracker_storage;
  // Kind selected by admission; never READY while a sidecar exists.
  iree_hal_amd_gpu_aql_wait_kind_t kind;
  // Before begin, borrows the source from submission. After begin, owns the
  // registered completion or submission-availability timeline.
  iree_hal_semaphore_t* semaphore;
  // Captured value paired with semaphore.
  uint64_t value;
} iree_hal_amd_gpu_aql_pending_t;

// Stack-owned cold request; the queue only borrows it until completion.
typedef struct iree_hal_amd_gpu_aql_frontier_request_t {
  // Exact protocol tag and publication link.
  iree_hal_amd_gpu_aql_entry_t entry;
  // Queue borrowed by the serialized membership caller through every handoff.
  iree_hal_amd_gpu_aql_queue_t* queue;
  // Device-owned notification and caller-owned result.
  iree_hal_amd_gpu_frontier_request_t request;
} iree_hal_amd_gpu_aql_frontier_request_t;

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_aql_queue_vtable;

static iree_status_code_t iree_hal_amd_gpu_aql_feedback_failure(
    const iree_hal_amd_gpu_aql_queue_t* queue) {
  return queue->options.feedback ? (iree_status_code_t)iree_atomic_load(
                                       &queue->options.feedback->failure_code,
                                       iree_memory_order_acquire)
                                 : IREE_STATUS_OK;
}

static void iree_hal_amd_gpu_aql_wake_terminal(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire) ||
      iree_atomic_load(&queue->failure, iree_memory_order_acquire) ||
      iree_hal_amd_gpu_aql_feedback_failure(queue)) {
    if (iree_hal_amd_gpu_context_wake_failure(queue->context)) {
      return;
    }
    const amdf_status_t status =
        iree_hal_amd_gpu_context_signal(queue->context);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(queue->context, status,
                                      "AQL terminal wake");
    }
  }
}

static iree_status_t iree_hal_amd_gpu_aql_check_failure(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  const iree_status_code_t native_failure =
      (iree_status_code_t)iree_atomic_load(&queue->failure,
                                           iree_memory_order_acquire);
  return iree_status_from_code(
      native_failure != IREE_STATUS_OK
          ? native_failure
          : iree_hal_amd_gpu_aql_feedback_failure(queue));
}

static iree_status_t iree_hal_amd_gpu_aql_check_acceptance(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AQL queue closed");
  }
  return iree_hal_amd_gpu_aql_check_failure(queue);
}

// Records failure without releasing any accepted native or host ownership.
// Final capture release publishes failed signals after its payload joins.
void iree_hal_amd_gpu_aql_queue_fail_submission(
    iree_hal_queue_t* base, iree_hal_amd_gpu_aql_submission_t* submission,
    iree_status_t status) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
      false, 0);
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_amd_gpu_aql_submission_fail(submission, status);
  iree_status_free(status);
}

static void iree_hal_amd_gpu_aql_abandon(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t status) {
  const bool host_owner = submission->has_host_owner;
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
      false, 0);
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  // Accepted prefixes retain the capture independently. The last release
  // returns their payloads before publishing this rejection to the caller.
  iree_hal_amd_gpu_aql_submission_reject(submission, iree_status_code(status));
  iree_status_free(status);
  submission->has_host_owner = false;
  iree_hal_amd_gpu_aql_submission_release(submission);
  if (host_owner) {
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
}

// A failed unpublished operation still owns its causal join. Consume each
// resolved edge once, retaining only the remaining suffix across wakeups.
static iree_status_t iree_hal_amd_gpu_aql_admit_failed(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_aql_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_READY;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_failure(queue));
  while (submission->wait_count) {
    iree_hal_amd_gpu_aql_wait_t* wait = submission->waits;
    uint64_t value = 0;
    iree_status_t status = iree_hal_semaphore_query(wait->semaphore, &value);
    if (iree_status_is_ok(status) && value < wait->value) {
      *out_wait_index = 0;
      *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_COMPLETION;
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_submission_reject(submission,
                                             iree_status_code(status));
      iree_status_free(status);
    }
    iree_hal_semaphore_release(wait->semaphore);
    ++submission->waits;
    --submission->wait_count;
  }
  return iree_status_from_code((iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed));
}

static void iree_hal_amd_gpu_aql_pending_release(
    iree_hal_amd_gpu_aql_pending_t* pending) {
  if (pending->allocation_size) {
    iree_hal_amd_gpu_capture_pool_release(
        &pending->queue->captures,
        iree_make_byte_span(pending, pending->allocation_size));
  }
}

// The capture may retire inline or be abandoned before its resuming callback
// finishes using the queue. Its separate callback borrower includes the final
// shutdown notification; idle observation joins that notification under the
// same mutex before reclaiming the queue and native event.
static void iree_hal_amd_gpu_aql_pending_end(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                        iree_memory_order_release);
  iree_hal_amd_gpu_aql_wake_terminal(queue);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
}

// File I/O and host calls may finish on an external progress owner. Their
// callback tails participate in the same idle proof as ready publication.
static void iree_hal_amd_gpu_aql_host_begin(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
}

static void iree_hal_amd_gpu_aql_host_end(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_pending_end((iree_hal_amd_gpu_aql_queue_t*)base);
}

// Ready entries run on the caller when publication is idle; contended entries
// join the current owner's queue.
static void iree_hal_amd_gpu_aql_enqueue_entry(
    iree_hal_amd_gpu_aql_queue_t* queue, iree_hal_amd_gpu_aql_entry_t* entry);

static void iree_hal_amd_gpu_aql_allocation_ready(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_aql_submission_t* submission = user_data;
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)submission->allocation.queue;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  } else {
    iree_hal_amd_gpu_aql_abandon(queue, submission, status);
  }
  iree_hal_amd_gpu_aql_pending_end(queue);
}

// Wait completion reenters the publication owner only after registration and
// every cancellation borrower have relinquished the captured sidecar.
static void iree_hal_amd_gpu_aql_pending_resume(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_aql_pending_t* pending = user_data;
  iree_hal_amd_gpu_aql_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_aql_submission_t* submission = pending->submission;
  bool dependency_failed = false;
  if (!iree_status_is_ok(status) && pending->semaphore) {
    uint64_t value = 0;
    iree_status_t dependency_status =
        iree_hal_semaphore_query(pending->semaphore, &value);
    dependency_failed = !iree_status_is_ok(dependency_status);
    status = iree_status_join(status, dependency_status);
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_aql_pending_release(pending);
  if (dependency_failed) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    iree_hal_amd_gpu_aql_submission_reject(submission,
                                           iree_status_code(status));
    iree_status_free(status);
    status = iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  } else {
    iree_hal_amd_gpu_aql_abandon(queue, submission, status);
  }
  iree_hal_amd_gpu_aql_pending_end(queue);
}

static void iree_hal_amd_gpu_aql_pending_begin(
    iree_hal_amd_gpu_aql_pending_t* pending) {
  iree_hal_amd_gpu_aql_queue_t* queue = pending->queue;
  iree_hal_semaphore_t* source = pending->semaphore;
  pending->semaphore = NULL;
  iree_status_t status = iree_hal_amd_gpu_aql_check_failure(queue);
  if (iree_status_is_ok(status)) {
    if (pending->kind == IREE_HAL_AMD_GPU_AQL_WAIT_SUBMISSION) {
      status = iree_hal_amd_gpu_semaphore_acquire_submission_semaphore(
          source, &pending->semaphore);
    } else {
      pending->semaphore = source;
      iree_hal_semaphore_retain(source);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_host_wait_set_begin(
        &queue->pending.waits,
        (iree_hal_semaphore_list_t){1, &pending->semaphore, &pending->value},
        iree_hal_amd_gpu_aql_pending_resume, pending,
        iree_allocator_inline_arena(&pending->tracker_storage),
        &pending->entry);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  if (!pending->semaphore) {
    pending->semaphore = source;
    iree_hal_semaphore_retain(source);
  }
  iree_hal_amd_gpu_aql_pending_resume(pending, status);
}

static iree_status_t iree_hal_amd_gpu_aql_pending_allocate(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_host_size_t wait_index,
    iree_hal_amd_gpu_aql_wait_kind_t kind,
    iree_hal_amd_gpu_aql_pending_t** out_pending) {
  const iree_host_size_t tail_offset =
      iree_host_align(submission->capture_size, iree_max_align_t);
  iree_byte_span_t storage = iree_byte_span_empty();
  if (tail_offset < submission->allocation_size) {
    storage = iree_make_byte_span((uint8_t*)submission + tail_offset,
                                  submission->allocation_size - tail_offset);
  }
  iree_host_size_t allocation_size = 0;
  if (storage.data_length < queue->pending_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
        &queue->captures, queue->pending_size, &storage));
    allocation_size = storage.data_length;
  }
  iree_hal_amd_gpu_aql_pending_t* pending = (void*)storage.data;
  const iree_host_size_t prefix_size =
      iree_host_align(sizeof(*pending), iree_max_align_t);
  *pending = (iree_hal_amd_gpu_aql_pending_t){
      .queue = queue,
      .submission = submission,
      .allocation_size = allocation_size,
      .kind = kind,
      .semaphore = submission->waits[wait_index].semaphore,
      .value = submission->waits[wait_index].value,
      .tracker_storage =
          {
              .capacity = storage.data_length - prefix_size,
              .buffer = storage.data + prefix_size,
          },
  };
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  *out_pending = pending;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_aql_capacity_available(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  if (!iree_atomic_exchange(&queue->capacity.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_growth_enqueue(&queue->growth, &queue->capacity.entry);
  }
}

void iree_hal_amd_gpu_aql_queue_notify_failure(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_host_wait_set_cancel(&queue->pending.waits);
  // The one-time failure notification resumes parked captures through the
  // publication FIFO; native retirement remains independent.
  iree_hal_amd_gpu_aql_capacity_available(queue);
}

static void iree_hal_amd_gpu_aql_park_capacity(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, uint64_t completed) {
  iree_hal_amd_gpu_publication_entry_t* entry = &submission->entry.publication;
  iree_atomic_store(&entry->next, 0, iree_memory_order_relaxed);
  if (queue->capacity.last) {
    iree_atomic_store(&queue->capacity.last->next, (intptr_t)entry,
                      iree_memory_order_relaxed);
  } else {
    queue->capacity.first = entry;
  }
  queue->capacity.last = entry;
  iree_atomic_exchange(&queue->completed.waiting, 1, iree_memory_order_acq_rel);
  if ((uint64_t)iree_atomic_load(&queue->completed.count,
                                 iree_memory_order_acquire) != completed) {
    iree_hal_amd_gpu_aql_capacity_available(queue);
  }
}

static void iree_hal_amd_gpu_aql_process_submission(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission) {
  iree_status_t status = iree_hal_amd_gpu_aql_check_failure(queue);
  if (iree_status_is_ok(status) &&
      submission->native_wait_count == UINT32_MAX) {
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    const bool dependency_failed =
        iree_atomic_load(&submission->failure_code,
                         iree_memory_order_relaxed) != IREE_STATUS_OK;
    if (!dependency_failed) {
      iree_hal_amd_gpu_aql_wait_counts_t counts;
      status = iree_hal_amd_gpu_aql_submission_admit_waits(
          queue->context, &queue->options.signals->pool, submission->wait_count,
          submission->waits,
          !(submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
            submission->transfer.capture.host_wait_required) &&
              !(submission->entry.kind ==
                    IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION &&
                (!queue->frontier.tracker ||
                 iree_hal_amd_gpu_allocation_requires_host_wait(
                     submission->allocation.operation))) &&
              submission->wait_count <=
                  (iree_min(queue->producer.ring.capacity / 2, 64u) - 4) * 5,
          &counts, &wait_index, &kind);
      if (iree_status_is_ok(status) &&
          kind == IREE_HAL_AMD_GPU_AQL_WAIT_READY) {
        submission->native_wait_count = counts.native_count;
      }
    }
    if (IREE_UNLIKELY(dependency_failed || !iree_status_is_ok(status))) {
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_context_report_status(
            queue->context, queue->base.queue_family->device, status);
        iree_hal_amd_gpu_aql_submission_reject(submission,
                                               iree_status_code(status));
        iree_status_free(status);
      }
      status = iree_hal_amd_gpu_aql_admit_failed(queue, submission, &wait_index,
                                                 &kind);
    }
    if (iree_status_is_ok(status) && kind != IREE_HAL_AMD_GPU_AQL_WAIT_READY) {
      iree_hal_amd_gpu_profile_operation_defer(
          iree_hal_amd_gpu_aql_submission_profile(submission));
      iree_hal_amd_gpu_aql_pending_t* pending = NULL;
      status = iree_hal_amd_gpu_aql_pending_allocate(
          queue, submission, wait_index, kind, &pending);
      if (iree_status_is_ok(status)) {
        iree_hal_amd_gpu_aql_pending_begin(pending);
        return;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_abandon(queue, submission, status);
    return;
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION) {
    // Pool preparation consumes the prospective prefix without publishing it.
    // A cold acquisition may defer while other submissions make progress.
    uint64_t* frontier = queue->producer.frontier + queue->producer.queue_count;
    memcpy(frontier, queue->producer.frontier,
           queue->producer.queue_count * sizeof(uint64_t));
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(
            submission->waits[i].signal, queue->producer.queue_count, frontier);
      }
    }
    // Separate callback-tail ownership from capture/native retirement. Warm
    // preparation remains inline under the current publisher's queue borrow.
    iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                          iree_memory_order_relaxed);
    bool pending = false;
    status = iree_hal_amd_gpu_allocation_prepare(
        submission->allocation.operation,
        queue->frontier.tracker ? queue->frontier.axis : 0,
        queue->producer.queue_count, frontier,
        iree_hal_amd_gpu_aql_allocation_ready, submission, &pending);
    if (!iree_status_is_ok(status)) {
      iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                            iree_memory_order_release);
      iree_hal_amd_gpu_aql_abandon(queue, submission, status);
      return;
    }
    if (pending) {
      return;
    }
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  const bool execute =
      submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE;
  const iree_hal_amd_gpu_aql_dispatch_request_t empty = {0};
  const iree_hal_amd_gpu_aql_dispatch_request_t* request =
      submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION
          ? &empty
          : &submission->dispatch;
  iree_hal_amd_gpu_aql_publish_result_t result;
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    result = iree_hal_amd_gpu_aql_publish_transfer(queue, submission, &status);
  } else if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL) {
    result = iree_hal_amd_gpu_aql_publish_control(queue, submission, &status);
  } else if (execute) {
    result = iree_hal_amd_gpu_aql_publish_execute(queue, submission, &status);
  } else {
    uint64_t* addresses =
        request->function
            ? (uint64_t*)(request->bindings.values + request->bindings.count +
                          iree_hal_dispatch_uses_indirect_parameters(
                              request->flags))
            : NULL;
    result = iree_hal_amd_gpu_aql_publish(
        queue, request, submission->wait_count, submission->waits,
        (iree_hal_amd_gpu_aql_wait_counts_t){.native_count =
                                                 submission->native_wait_count},
        submission->signals, submission->before, submission->after, addresses,
        NULL, &submission, &status);
  }
  if (result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    iree_hal_amd_gpu_profile_operation_defer(
        iree_hal_amd_gpu_aql_submission_profile(submission));
  }
  if (result == IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE) {
    iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                       &submission->entry.publication);
  } else if (result == IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY) {
    iree_hal_amd_gpu_aql_park_capacity(queue, submission, completed);
  } else if (result == IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS ||
             result == IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS ||
             result == IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM) {
    uint64_t capacity = 0;
    const iree_hal_amd_gpu_profile_operation_t* operation =
        iree_hal_amd_gpu_aql_submission_profile(submission);
    const bool sampled = operation && operation->completion.sample_stride;
    if (result == IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS) {
      capacity = iree_math_round_up_to_pow2_u64(
          submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER
              ? iree_hal_amd_gpu_aql_transfer_argument_length(
                    &submission->transfer.capture, 0)
          : submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL
              ? IREE_HAL_AMD_GPU_AQL_CONTROL_BYTE_LENGTH
          : execute
              ? iree_hal_amd_gpu_aql_execute_argument_length(
                    &submission->execute, sampled)
              : iree_max(iree_hal_amd_gpu_aql_argument_length(
                             request,
                             queue->grid.memory
                                 ? IREE_HAL_AMD_GPU_AQL_GRID_SYNC_COMMAND_LENGTH
                                 : sizeof(iree_hal_amd_gpu_grid_sync_t),
                             sampled),
                         request->function->base.kernarg->alignment));
    }
    status = iree_hal_amd_gpu_aql_growth_begin(
        &queue->growth,
        result == IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS
            ? IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS
        : result == IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM
            ? IREE_HAL_AMD_GPU_AQL_ENTRY_PROGRAM
            : IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS,
        submission, capacity);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_abandon(queue, submission, status);
  }
}

static bool iree_hal_amd_gpu_aql_record_failure(
    iree_hal_amd_gpu_aql_queue_t* queue, iree_status_code_t code) {
  int32_t expected = 0;
  if (iree_atomic_compare_exchange_strong(&queue->failure, &expected, code,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_relaxed)) {
    iree_hal_amd_gpu_host_wait_set_cancel(&queue->pending.waits);
    iree_hal_amd_gpu_aql_capacity_available(queue);
    iree_hal_amd_gpu_aql_wake_terminal(queue);
    return true;
  }
  return false;
}

static void iree_hal_amd_gpu_aql_fail_queue(iree_hal_amd_gpu_aql_queue_t* queue,
                                            iree_status_t status) {
  if (iree_hal_amd_gpu_aql_record_failure(queue, iree_status_code(status))) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
  }
  iree_status_free(status);
}

void iree_hal_amd_gpu_aql_queue_fail_native(iree_hal_queue_t* base,
                                            amdf_status_t status,
                                            const char* operation) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_context_fail_execution(queue->context, status, operation);
  iree_status_t hal_status = IREE_HAL_AMD_STATUS_FROM_AMDF(status, NULL);
  iree_hal_amd_gpu_aql_record_failure(queue, iree_status_code(hal_status));
  iree_status_free(hal_status);
}

iree_status_t iree_hal_amd_gpu_aql_queue_set_profile(
    iree_hal_queue_t* base, iree_hal_amd_gpu_profile_t* profile) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_register_queue(
      profile, queue->options.queue_ordinal));
  const bool previous =
      iree_hal_amd_gpu_profile_timestamps_dispatches(queue->profile);
  const bool enabled = iree_hal_amd_gpu_profile_timestamps_dispatches(profile);
  // Public completion precedes the final host retirement and publisher
  // handoff. Begin/end already require externally joined work; join those
  // remaining host accesses before changing the borrowed session or native
  // queue descriptor. This cold yield adds no notification to ordinary work.
  iree_status_t status = iree_ok_status();
  while (!iree_hal_amd_gpu_aql_queue_is_idle(base) &&
         iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_aql_check_failure(queue);
    if (iree_status_is_ok(status)) {
      iree_thread_yield();
    }
  }
  // Clear borrows even if native teardown fails; the session still ends.
  queue->profile = NULL;
  queue->file_executor.profile = NULL;
  if (previous != enabled && iree_status_is_ok(status)) {
    // A completion write can precede the firmware's final read-index update.
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        queue->context->api->user_queue_wait_consumed(
            queue->storage.queue, queue->producer.ring.reserved,
            AMDF_TIMEOUT_INFINITE, 0),
        "join AQL packet consumption");
  }
  if (previous != enabled && iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        queue->context->gpu->user_queue_set_dispatch_timestamps(
            queue->storage.queue, enabled),
        "configure AQL dispatch timestamps");
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_fail_queue(queue, iree_status_clone(status));
    return status;
  }
  queue->profile = profile;
  queue->file_executor.profile = profile;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_aql_trim_storage(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  const uint64_t consumed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  iree_hal_amd_gpu_argument_storage_t** link =
      &queue->producer.retired_arguments;
  while (*link) {
    iree_hal_amd_gpu_argument_storage_t* storage = *link;
    if (storage->end <= consumed) {
      *link = storage->next;
      iree_hal_amd_gpu_argument_storage_free(queue->context, storage);
    } else {
      link = &storage->next;
    }
  }
  iree_hal_amd_gpu_signal_cache_trim(queue->context, 1, queue->options.signals);
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
}

static void iree_hal_amd_gpu_aql_expand_frontiers(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_frontier_request_t* request) {
  if (request->phase == IREE_HAL_AMD_GPU_FRONTIER_BEGIN && request->vectors) {
    uint64_t* frontier = request->vectors;
    memcpy(frontier, queue->producer.frontier,
           queue->producer.queue_count * sizeof(uint64_t));
    uint64_t* previous = queue->producer.frontier;
    queue->producer.frontier = frontier;
    queue->producer.queue_count = request->queue_count;
    request->vectors =
        previous == (void*)(queue->issued + queue->options.retirement_capacity)
            ? NULL
            : previous;
  }
  iree_hal_amd_gpu_aql_growth_process_frontiers(&queue->growth, request);
}

static void iree_hal_amd_gpu_aql_process_entry(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  iree_hal_amd_gpu_aql_entry_t* typed =
      iree_containerof(entry, iree_hal_amd_gpu_aql_entry_t, publication);
  switch (typed->kind) {
    case IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION:
      iree_hal_amd_gpu_aql_process_submission(
          queue,
          iree_containerof(typed, iree_hal_amd_gpu_aql_submission_t, entry));
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS:
    case IREE_HAL_AMD_GPU_AQL_ENTRY_PROGRAM: {
      iree_hal_amd_gpu_aql_growth_result_t result;
      iree_status_t status = iree_hal_amd_gpu_aql_growth_end(
          typed, queue->producer.arguments->end, &result);
      if (result.arguments) {
        queue->producer.arguments->next = queue->producer.retired_arguments;
        queue->producer.retired_arguments = queue->producer.arguments;
        queue->producer.arguments = result.arguments;
      }
      if (result.signals) {
        result.signals->next = queue->options.signals->storage;
        queue->options.signals->storage = result.signals;
      }
      if (iree_status_is_ok(status)) {
        if (result.first) {
          iree_hal_amd_gpu_publisher_enqueue_batch(&queue->publisher,
                                                   result.first, result.last);
        }
      } else {
        for (iree_hal_amd_gpu_publication_entry_t* current = result.first;
             current;) {
          iree_hal_amd_gpu_publication_entry_t* next = (void*)iree_atomic_load(
              &current->next, iree_memory_order_relaxed);
          iree_hal_amd_gpu_aql_submission_t* submission = iree_containerof(
              current, iree_hal_amd_gpu_aql_submission_t, entry.publication);
          iree_hal_amd_gpu_aql_abandon(queue, submission,
                                       iree_status_clone(status));
          current = next;
        }
        iree_status_free(status);
      }
      break;
    }
    case IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_PROGRAM:
      iree_hal_amd_gpu_aql_growth_reuse_program(typed);
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_SIGNALS:
      iree_hal_amd_gpu_aql_growth_reuse_signals(&queue->growth);
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_CAPACITY: {
      iree_atomic_store(&queue->capacity.pending, 0, iree_memory_order_release);
      iree_hal_amd_gpu_publication_entry_t* first = queue->capacity.first;
      iree_hal_amd_gpu_publication_entry_t* last = queue->capacity.last;
      queue->capacity.first = queue->capacity.last = NULL;
      if (first) {
        iree_hal_amd_gpu_publisher_enqueue_batch(&queue->publisher, first,
                                                 last);
      }
      break;
    }
    case IREE_HAL_AMD_GPU_AQL_ENTRY_SCRATCH: {
      if (iree_atomic_load(&queue->failure, iree_memory_order_acquire)) {
        iree_hal_amd_gpu_aql_scratch_abandon(&queue->scratch);
      } else {
        iree_status_t status =
            iree_hal_amd_gpu_aql_scratch_resume(&queue->scratch);
        if (!iree_status_is_ok(status)) {
          iree_hal_amd_gpu_aql_fail_queue(queue, status);
        }
      }
      break;
    }
    case IREE_HAL_AMD_GPU_AQL_ENTRY_FRONTIERS: {
      iree_hal_amd_gpu_aql_frontier_request_t* request = iree_containerof(
          typed, iree_hal_amd_gpu_aql_frontier_request_t, entry);
      iree_hal_amd_gpu_aql_expand_frontiers(queue, &request->request);
      break;
    }
    case IREE_HAL_AMD_GPU_AQL_ENTRY_TRIM:
      iree_atomic_store(&queue->trim.pending, 0, iree_memory_order_release);
      iree_hal_amd_gpu_aql_trim_storage(queue);
      break;
  }
}

static void iree_hal_amd_gpu_aql_enqueue_entry(
    iree_hal_amd_gpu_aql_queue_t* queue, iree_hal_amd_gpu_aql_entry_t* entry) {
  if (iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                         &entry->publication)) {
    iree_hal_amd_gpu_publication_entry_t* ready =
        iree_hal_amd_gpu_publisher_next(&queue->publisher);
    if (ready) {
      iree_hal_amd_gpu_aql_process_entry(queue, ready);
    }
    if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
      iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                          queue->options.source_ordinal);
    }
    iree_hal_amd_gpu_aql_wake_terminal(queue);
  }
}

static bool iree_hal_amd_gpu_aql_progress(void* user_data) {
  iree_hal_amd_gpu_aql_queue_t* queue = user_data;
  if (iree_hal_amd_gpu_signal_cache_take_ready(queue->options.signals)) {
    iree_hal_amd_gpu_aql_growth_signals_available(&queue->growth);
  }
  bool more = false;
  if (iree_hal_amd_gpu_publisher_acquire_service(&queue->publisher)) {
    for (uint32_t i = 0; i < 32; ++i) {
      iree_hal_amd_gpu_publication_entry_t* entry =
          iree_hal_amd_gpu_publisher_next(&queue->publisher);
      if (!entry) {
        break;
      }
      iree_hal_amd_gpu_aql_entry_t* typed =
          iree_containerof(entry, iree_hal_amd_gpu_aql_entry_t, publication);
      if (IREE_UNLIKELY(queue->profile) &&
          (typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION ||
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION ||
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE ||
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER ||
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL)) {
        iree_hal_amd_gpu_profile_operation_defer(
            iree_hal_amd_gpu_aql_submission_profile(iree_containerof(
                typed, iree_hal_amd_gpu_aql_submission_t, entry)));
      }
      iree_hal_amd_gpu_aql_process_entry(queue, entry);
    }
    more = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL);
  }
  // A native callback can return its final queue borrow after its result was
  // consumed. Its last readiness hint still needs to wake terminal observers.
  iree_hal_amd_gpu_aql_wake_terminal(queue);
  return more;
}

static void iree_hal_amd_gpu_aql_complete_download(void* user_data) {
  iree_hal_amd_gpu_download_t* download = user_data;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_download_complete");
  iree_hal_amd_gpu_download_complete(download);
  download->length = 0;
  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amd_gpu_aql_retire(
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
      iree_hal_amd_gpu_aql_feedback_failure(queue);
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
        code = iree_hal_amd_gpu_aql_feedback_failure(queue);
      }
    }
    const uint64_t arguments = action->arguments;
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
        code == IREE_STATUS_OK && action->download.length) {
      if (out_copy) {
        *out_copy = (iree_hal_amd_gpu_completion_copy_t){
            .fn = iree_hal_amd_gpu_aql_complete_download,
            .user_data = &action->download};
      }
      break;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_aql_retire");
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    if (submission->profiled) {
      const iree_hal_amd_gpu_profile_operation_t* operation =
          iree_hal_amd_gpu_aql_submission_profile(submission);
      iree_hal_amd_gpu_profile_operation_complete(
          queue->profile, operation,
          submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE
              ? &action->profile
              : &operation->completion);
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
    iree_atomic_store(&queue->completed.arguments, arguments,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.count, position + 1,
                      iree_memory_order_release);
    IREE_TRACE_ZONE_END(z0);
  }
  if (position != initial && iree_atomic_exchange(&queue->completed.waiting, 0,
                                                  iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_capacity_available(queue);
  }
  if (position == published) {
    *out_wait =
        (iree_hal_amd_gpu_completion_wait_t){.address = &queue->producer.count,
                                             .value = (int64_t)published,
                                             .is_64_bit = true};
  }
  return position != initial;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_aql_queue_publication_source(iree_hal_queue_t* queue) {
  return (iree_hal_amd_gpu_publication_source_t){iree_hal_amd_gpu_aql_progress,
                                                 queue};
}
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_aql_queue_completion_source(iree_hal_queue_t* queue) {
  return (iree_hal_amd_gpu_completion_source_t){iree_hal_amd_gpu_aql_retire,
                                                queue};
}

static iree_status_t iree_hal_amd_gpu_aql_submit_dispatch(
    iree_hal_amd_gpu_aql_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_profile_operation_t* profile) {
  const uint32_t before =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL);
  const uint32_t after =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL);
  enum { kScratchSize = 512 };
  const bool small =
      waits.count <= kScratchSize / sizeof(iree_hal_amd_gpu_aql_wait_t) &&
      request->bindings.count <= kScratchSize / sizeof(uint64_t) &&
      waits.count * sizeof(iree_hal_amd_gpu_aql_wait_t) +
              request->bindings.count * sizeof(uint64_t) <=
          kScratchSize;
  const bool publishing =
      small && iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher);
  iree_status_t status = iree_ok_status();
  iree_status_t dependency_failure = iree_ok_status();
  bool finished = false;
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  if (publishing) {
    iree_alignas(8) uint8_t storage[kScratchSize];
    iree_hal_amd_gpu_aql_wait_t* admitted = (void*)storage;
    uint64_t* addresses = (uint64_t*)(admitted + waits.count);
    for (iree_host_size_t i = 0; i < waits.count; ++i) {
      admitted[i] = (iree_hal_amd_gpu_aql_wait_t){
          .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    }
    iree_hal_amd_gpu_aql_wait_counts_t counts = {0};
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    status = iree_hal_amd_gpu_aql_check_failure(queue);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_aql_submission_admit_waits(
          queue->context, &queue->options.signals->pool, waits.count, admitted,
          true, &counts, &wait_index, &kind);
    }
    if (!iree_status_is_ok(status)) {
      dependency_failure = status;
      status = iree_ok_status();
    } else if (kind == IREE_HAL_AMD_GPU_AQL_WAIT_READY) {
      finished = iree_hal_amd_gpu_aql_publish(
                     queue, request, waits.count, admitted, counts, signals,
                     before, after, addresses, profile, &submission,
                     &status) == IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
    if (!submission) {
      for (iree_host_size_t i = 0; i < waits.count; ++i) {
        iree_hal_amd_gpu_signal_release(admitted[i].signal);
      }
    }
  }
  if (!finished && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_aql_submission_capture(
        &queue->captures, profile, request, waits, signals, before, after,
        &submission);
    if (iree_status_is_ok(status)) {
      if (!iree_status_is_ok(dependency_failure)) {
        iree_hal_amd_gpu_context_report_status(queue->context,
                                               queue->base.queue_family->device,
                                               dependency_failure);
        iree_hal_amd_gpu_aql_submission_reject(
            submission, iree_status_code(dependency_failure));
        iree_status_free(dependency_failure);
        dependency_failure = iree_ok_status();
      }
      submission->has_host_owner = true;
      iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                            iree_memory_order_relaxed);
      if (publishing) {
        iree_hal_amd_gpu_aql_process_submission(queue, submission);
      } else {
        iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
      }
    }
  }
  if (publishing) {
    if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
      iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                          queue->options.source_ordinal);
    }
    iree_hal_amd_gpu_aql_wake_terminal(queue);
  }
  return iree_status_join(status, dependency_failure);
}

static iree_status_t iree_hal_amd_gpu_aql_host_call_enter(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_retirement_callback_t callback,
    const iree_hal_queue_barriers_t* barriers,
    iree_host_size_t public_signal_count) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t operation;
  const bool profiled = queue->profile != NULL;
  if (profiled) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL, waits.count,
        public_signal_count, 1, &operation);
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_transfer(
      &queue->captures, profiled ? &operation : NULL, base->queue_family, waits,
      iree_hal_semaphore_list_empty(), 0, NULL,
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL),
      &submission));
  submission->transfer.callback = callback;
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t operation;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, waits.count, signals.count,
        0, &operation);
  }
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {0};
  return iree_hal_amd_gpu_aql_submit_dispatch(
      queue, waits, signals, &request, barriers,
      queue->profile ? &operation : NULL);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_host_call(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t arguments[4], iree_hal_host_call_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  return iree_hal_amd_gpu_host_call_submit(&queue->host_executor, waits,
                                           signals, call, arguments, flags);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_dispatch(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  const iree_hal_amd_gpu_aql_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_lookup_aql_for_queue(
      executable, token, queue->options.queue_ordinal, &function));
  iree_hal_amd_gpu_dispatch_geometry_t geometry;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_dispatch_prepare(
      function,
      (uint32_t)
          queue->context->target_info.compute.local_data_share_byte_length,
      config, constants, bindings.count, flags, &geometry));
  if (flags & IREE_HAL_DISPATCH_FLAG_COOPERATIVE) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_executable_validate_cooperative_dispatch(
            executable, token, &geometry, flags));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_dispatch_validate_direct_bindings(
      function->base.kernarg, config.workgroup_count_ref, bindings, flags));
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {
      .executable = executable,
      .function = function,
      .geometry = geometry,
      .parameters = config.workgroup_count_ref,
      .constants = constants,
      .bindings = bindings,
      .flags = flags};
  iree_hal_amd_gpu_profile_operation_t operation;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, waits.count, signals.count,
        1, &operation);
    if (iree_hal_amd_gpu_profile_captures_dispatch(
            queue->profile, queue->options.queue_ordinal, 0, 0,
            function->base.metadata->reflection_name) &&
        (iree_hal_dispatch_uses_indirect_parameters(flags) ||
         (geometry.workgroup_count[0] && geometry.workgroup_count[1] &&
          geometry.workgroup_count[2]))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_profile_dispatch(
          executable, token, &geometry, flags, &operation.dispatch));
      operation.completion.count = 1;
      operation.completion.sample_stride =
          iree_hal_amd_gpu_profile_timestamps_dispatches(queue->profile)
              ? sizeof(iree_hal_amd_gpu_aql_signal_t)
              : 0;
    }
  }
  return iree_hal_amd_gpu_aql_submit_dispatch(
      queue, waits, signals, &request, barriers,
      queue->profile ? &operation : NULL);
}

static void iree_hal_amd_gpu_aql_enqueue_frontiers(void* user_data) {
  iree_hal_amd_gpu_aql_frontier_request_t* request = user_data;
  iree_hal_amd_gpu_aql_queue_t* queue = request->queue;
  iree_hal_amd_gpu_publication_enqueue(
      queue->options.publication, queue->options.source_ordinal,
      &queue->publisher, &request->entry.publication);
}

iree_status_t iree_hal_amd_gpu_aql_queue_reserve_frontiers(
    iree_hal_queue_t* base, uint32_t queue_count,
    iree_notification_t* notification) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  // Only these serialized lifecycle requests change the installed extent.
  // Ordinary publication mutates vector contents, never this geometry.
  const uint32_t previous_count = queue->producer.queue_count;
  iree_hal_amd_gpu_aql_frontier_request_t request = {
      .entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_FRONTIERS,
      .queue = queue,
      .request = {.queue_count = iree_max(queue_count, previous_count),
                  .notification = notification},
  };
  if (queue_count > previous_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, queue_count, 2 * sizeof(uint64_t),
        (void**)&request.request.vectors));
  }
  return iree_hal_amd_gpu_frontier_request_run(
      &request.request, queue->host_allocator,
      iree_hal_amd_gpu_aql_enqueue_frontiers, &request);
}

void iree_hal_amd_gpu_aql_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (!iree_atomic_exchange(&queue->trim.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_enqueue_entry(queue, &queue->trim.entry);
  }
}

amdf_status_t iree_hal_amd_gpu_aql_queue_trim_scratch(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (!iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher)) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = AMDF_STATUS_OK;
  // Completion publishes count only after its scratch observation and payload
  // retirement. With publication excluded, equality also excludes a concurrent
  // nonempty scratch query. Accepted deferred work may resume after handoff.
  if (iree_atomic_load(&queue->producer.count, iree_memory_order_acquire) ==
          iree_atomic_load(&queue->completed.count,
                           iree_memory_order_acquire) &&
      !iree_hal_amd_gpu_aql_scratch_is_pending(&queue->scratch)) {
    status = queue->context->api->user_queue_wait_consumed(
        queue->storage.queue, queue->producer.ring.reserved,
        AMDF_TIMEOUT_INFINITE, 0);
    if (amdf_status_is_ok(status)) {
      status = iree_hal_amd_gpu_aql_scratch_trim(&queue->scratch);
    }
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_queue_fail_native(&queue->base, status,
                                             "AQL scratch trim");
    }
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.source_ordinal);
  }
  iree_hal_amd_gpu_aql_wake_terminal(queue);
  return status;
}

static iree_status_t iree_hal_amd_gpu_aql_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_hal_amd_gpu_executable_query_dispatch_concurrency(
      executable, function, queue->execution_resources.count, params,
      out_concurrency);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_control(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_amd_gpu_control_t command,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_profile_queue_event_type_t event_type) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  if (command.kind == IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP &&
      !base->queue_family->spec->timestamp_frequency_hz) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "queue family has no timestamp clock rate");
  }
  iree_hal_amd_gpu_profile_operation_t operation;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal, event_type, waits.count,
        signals.count, 1, &operation);
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_control(
      &queue->captures, queue->profile ? &operation : NULL, base->queue_family,
      waits, signals, target, offset, command,
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL),
      &submission));
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_aql_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_wait(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_WAIT);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_aql_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_store(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_STORE);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_aql_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_rmw(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_RMW);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_timestamp(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_hal_amd_gpu_aql_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_timestamp(), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TIMESTAMP);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_flush(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  // Inline submissions publish immediately. Deferred work owns its
  // readiness notification; flushing an idle publisher has nothing to advance.
  return iree_hal_amd_gpu_aql_check_acceptance(queue);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t profile;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA, waits.count, signals.count,
        count, &profile);
    for (iree_host_size_t i = 0; i < count; ++i) {
      profile.event.payload_length +=
          iree_min(requests[i].allocation_size,
                   UINT64_MAX - profile.event.payload_length);
    }
  }
  // The native point orders reservation reuse; allocation itself needs no
  // payload visibility transition or shader arguments.
  const iree_hal_amd_gpu_aql_dispatch_request_t empty = {0};
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_capture(
      &queue->captures, queue->profile ? &profile : NULL, &empty, waits,
      signals, 0, 0, &submission));
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_alloca(
      queue->options.allocations, pool, count, requests, out_buffers,
      &operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_submission_release(submission);
    return status;
  }
  submission->entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION;
  submission->allocation.operation = operation;
  submission->allocation.queue = base;
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t profile;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA, waits.count, signals.count,
        count, &profile);
    for (iree_host_size_t i = 0; i < count; ++i) {
      profile.event.payload_length +=
          iree_min(iree_hal_buffer_byte_length(buffers[i]),
                   UINT64_MAX - profile.event.payload_length);
    }
  }
  // The native point orders reservation reuse; allocation itself needs no
  // payload visibility transition or shader arguments.
  const iree_hal_amd_gpu_aql_dispatch_request_t empty = {0};
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_capture(
      &queue->captures, queue->profile ? &profile : NULL, &empty, waits,
      signals, 0, 0, &submission));
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_dealloca(
      queue->options.allocations, count, buffers, &operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_submission_release(submission);
    return status;
  }
  submission->entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION;
  submission->allocation.operation = operation;
  submission->allocation.queue = base;
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_map(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_map(queue->options.mappings, base,
                                    queue->options.queue_ordinal, waits,
                                    signals, mapping_count, mappings, flags);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_execute(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  const iree_hal_amd_gpu_aql_command_program_t* program =
      iree_hal_amd_gpu_aql_command_buffer_program(command_buffer);
  if (program->required_features & ~base->features) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "recorded cooperative dispatch requires a cooperative queue");
  }
  const bool capture =
      queue->profile && iree_hal_amd_gpu_profile_captures_commands(
                            queue->profile, queue->options.queue_ordinal,
                            iree_hal_command_buffer_profile_id(command_buffer));
  if (capture && program->dispatch_count && !program->profile) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dispatch capture requires command buffer profile metadata");
  }
  iree_hal_amd_gpu_profile_operation_t operation;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, waits.count, signals.count,
        program->operation_count, &operation);
    operation.event.command_buffer_id =
        iree_hal_command_buffer_profile_id(command_buffer);
    if (capture) {
      operation.commands = program->profile;
      operation.completion.count = program->dispatch_count;
      operation.completion.sample_stride =
          iree_hal_amd_gpu_profile_timestamps_dispatches(queue->profile)
              ? sizeof(iree_hal_amd_gpu_aql_signal_t)
              : 0;
    }
  }
  if (!program->block_count) {
    const iree_hal_barrier_list_t empty = {0};
    const iree_hal_queue_barriers_t barriers = {&empty, &empty};
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {0};
    return iree_hal_amd_gpu_aql_submit_dispatch(
        queue, waits, signals, &request, &barriers,
        queue->profile ? &operation : NULL);
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_execute(
      &queue->captures, queue->profile ? &operation : NULL, command_buffer,
      bindings, waits, signals, flags, &submission));
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t operation;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        count ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER
              : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
        waits.count, signals.count, count, &operation);
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_transfer(
      &queue->captures, queue->profile ? &operation : NULL, base->queue_family,
      waits, signals, count, operations,
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL),
      &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_t* captured =
        iree_hal_amd_gpu_aql_submission_profile(submission);
    iree_hal_amd_gpu_transfer_profile(&submission->transfer.capture,
                                      &captured->event);
  }
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_read(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* source,
    uint64_t source_offset, iree_hal_buffer_t* target,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  const uint32_t before =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL);
  const uint32_t after =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, source,
      IREE_HAL_MEMORY_ACCESS_READ, source_offset, target, target_offset, length,
      &before, &after);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_file_t* target,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  const uint32_t before =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->before : NULL);
  const uint32_t after =
      iree_hal_amd_gpu_aql_barrier_prepare(barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, target,
      IREE_HAL_MEMORY_ACCESS_WRITE, target_offset, source, source_offset,
      length, &before, &after);
}

static void iree_hal_amd_gpu_aql_file_add_effects(
    iree_hal_queue_t* base, iree_hal_memory_effects_t effects, void* barrier) {
  const iree_hal_barrier_t entry = {.effects = effects};
  const iree_hal_barrier_list_t list = {1, &entry};
  *(uint32_t*)barrier = iree_hal_amd_gpu_aql_barrier_join(
      *(uint32_t*)barrier, iree_hal_amd_gpu_aql_barrier_prepare(&list));
}

static iree_status_t iree_hal_amd_gpu_aql_file_submit(
    iree_hal_queue_t* base,
    const iree_hal_amd_gpu_transfer_operation_t* operation, const void* before,
    const void* after, iree_hal_amd_gpu_retirement_callback_t callback) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_file(
      &queue->captures, operation, before ? *(const uint32_t*)before : 0,
      after ? *(const uint32_t*)after : 0, callback, &submission));
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue_entry(queue, &submission->entry);
  return iree_ok_status();
}

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_aql_queue_vtable = {
    .destroy = iree_hal_amd_gpu_aql_queue_destroy,
    .barrier = iree_hal_amd_gpu_aql_queue_barrier,
    .execute = iree_hal_amd_gpu_aql_queue_execute,
    .host_call = iree_hal_amd_gpu_aql_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_gpu_aql_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_gpu_aql_queue_dispatch,
    .atomic_wait = iree_hal_amd_gpu_aql_queue_atomic_wait,
    .atomic_store = iree_hal_amd_gpu_aql_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_gpu_aql_queue_atomic_rmw,
    .timestamp = iree_hal_amd_gpu_aql_queue_timestamp,
    .flush = iree_hal_amd_gpu_aql_queue_flush,
    .alloca = iree_hal_amd_gpu_aql_queue_alloca,
    .dealloca = iree_hal_amd_gpu_aql_queue_dealloca,
    .map = iree_hal_amd_gpu_aql_queue_map,
    .transfer = iree_hal_amd_gpu_aql_queue_transfer,
    .read = iree_hal_amd_gpu_aql_queue_read,
    .write = iree_hal_amd_gpu_aql_queue_write,
};

amdf_queue_publication_modes_t iree_hal_amd_gpu_aql_queue_publication_modes(
    const amdf_queue_family_info_t* family) {
  if (family->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ||
      family->format_version != AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1 ||
      !iree_any_bit_set(family->roles, AMDF_QUEUE_ROLE_COMPUTE) ||
      !iree_any_bit_set(family->format_features,
                        AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE) ||
      !iree_all_bits_set(family->user_queue_capabilities,
                         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                             AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER)) {
    return 0;
  }
  return family->publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER;
}

iree_status_t iree_hal_amd_gpu_aql_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_aql_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  if (iree_cpu_architecture_host() != IREE_CPU_ARCHITECTURE_X86_64 ||
      !iree_atomic_int32_is_lock_free() || !iree_atomic_int64_is_lock_free()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AQL requires qualified x86-64 host publication");
  }
  if (context->gpu->extension_version < AMDF_GPU_EXTENSION_VERSION_4 ||
      options->family_ordinal >= context->endpoint_info.queue_family_count) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AQL requires native scratch service");
  }
  const amdf_queue_family_info_t* native_family =
      &context->families[options->family_ordinal];
  if (!iree_hal_amd_gpu_aql_queue_publication_modes(native_family) ||
      context->event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
      context->event_info.format_version != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL family lacks native publication, event, or scratch support");
  }
  if (!options->retirement_capacity ||
      (options->retirement_capacity & (options->retirement_capacity - 1)) ||
      options->argument_capacity < 8 ||
      (options->argument_capacity & (options->argument_capacity - 1)) ||
      options->command_byte_capacity < 4096 ||
      options->command_byte_capacity > UINT32_MAX ||
      (options->command_byte_capacity & (options->command_byte_capacity - 1))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AQL queue capacity");
  }
  iree_host_size_t tracker_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_wait_tracker_storage_size(1, &tracker_size));
  const iree_host_size_t pending_size =
      iree_host_align(sizeof(iree_hal_amd_gpu_aql_pending_t),
                      iree_max_align_t) +
      tracker_size;
  if (options->capture_block_size <
      iree_max(pending_size, sizeof(iree_hal_amd_gpu_aql_submission_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL capture block cannot hold wait ownership");
  }
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_aql_queue_t);
  const uint32_t queue_count = options->queue_count;
  if (!iree_host_size_checked_mul_add(total_size, options->retirement_capacity,
                                      sizeof(iree_hal_amd_gpu_aql_retirement_t),
                                      &total_size) ||
      !iree_host_size_checked_mul_add(total_size, queue_count,
                                      2 * sizeof(uint64_t), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, params->execution_resources.count,
          sizeof(iree_hal_queue_execution_resource_ordinal_t), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AQL queue allocation overflows");
  }
  iree_hal_amd_gpu_aql_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, iree_alignof(iree_hal_amd_gpu_aql_queue_t), 0,
      (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->storage.context = context;
  queue->options = *options;
  iree_atomic_store(&queue->producer.count, options->initial_epoch,
                    iree_memory_order_relaxed);
  iree_atomic_store(&queue->completed.count, options->initial_epoch,
                    iree_memory_order_relaxed);
  queue->pending_size = pending_size;
  queue->issued = (void*)(queue + 1);
  queue->producer.queue_count = queue_count;
  queue->producer.frontier =
      (void*)(queue->issued + options->retirement_capacity);
  queue->producer.frontier[options->queue_ordinal] = options->initial_epoch;
  iree_hal_queue_execution_resource_ordinal_t* ordinals =
      (void*)(queue->producer.frontier + 2 * queue_count);
  iree_hal_queue_params_t achieved = *params;
  if (params->execution_resources.count) {
    memcpy(ordinals, params->execution_resources.ordinals,
           params->execution_resources.count * sizeof(*ordinals));
    achieved.execution_resources.ordinals = ordinals;
  }
  iree_hal_queue_initialize(family, &achieved,
                            &iree_hal_amd_gpu_aql_queue_vtable, &queue->base);
  iree_hal_amd_gpu_publisher_initialize(&queue->publisher);
  iree_async_semaphore_wait_context_initialize(&queue->wait_context);
  iree_hal_amd_gpu_host_wait_set_initialize(&queue->wait_context,
                                            &queue->pending.waits);
  iree_slim_mutex_initialize(&queue->pending.completion_mutex);
  iree_hal_amd_gpu_capture_pool_initialize(options->capture_block_size,
                                           host_allocator, &queue->captures);
  queue->growth = (iree_hal_amd_gpu_aql_growth_context_t){
      .context = context,
      .memory = options->memory,
      .captures = &queue->captures,
      .publisher = &queue->publisher,
      .publication = options->publication,
      .queue_ordinal = options->queue_ordinal,
      .source_ordinal = options->source_ordinal,
      .queue_count = queue_count,
      .signal_capacity = options->retirement_capacity,
      .signal_cache = options->signals,
      .reuse.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_SIGNALS};
  queue->capacity.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_CAPACITY;
  queue->trim.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_TRIM;
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_aql_host_begin,
      .end = iree_hal_amd_gpu_aql_host_end,
      .enter = iree_hal_amd_gpu_aql_host_call_enter};
  queue->file_executor = (iree_hal_amd_gpu_file_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_aql_host_begin,
      .end = iree_hal_amd_gpu_aql_host_end,
      .staging = options->file_staging,
      .waits = &queue->wait_context,
      .queue_ordinal = options->queue_ordinal,
      .native_family_ordinal = options->family_ordinal,
      .transfer_site = IREE_HAL_MEMORY_SITE_PROGRAM,
      .barrier_size = sizeof(uint32_t),
      .add_effects = iree_hal_amd_gpu_aql_file_add_effects,
      .submit = iree_hal_amd_gpu_aql_file_submit};
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  const amdf_gpu_queue_execution_resource_selection_t selection = {
      .type = AMDF_STRUCTURE_TYPE_GPU_QUEUE_EXECUTION_RESOURCE_SELECTION,
      .structure_size = sizeof(selection),
      .resource_count = (uint32_t)params->execution_resources.count,
      .resource_ordinals = params->execution_resources.ordinals};
  const amdf_gpu_aql_scratch_service_info_t service = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_SERVICE_INFO,
      .structure_size = sizeof(service),
      .next = selection.resource_count ? &selection : NULL,
      .event = context->event};
  const amdf_gpu_user_queue_create_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(native),
      .next = &service,
      .queue_family_ordinal = options->family_ordinal,
      .priority = iree_hal_amd_gpu_user_queue_native_priority(params->priority),
      .placement = (amdf_queue_placement_t)params->placement,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .required_capabilities =
          AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
          AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER |
          ((params->features & IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)
               ? AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH
               : 0),
      .ring_byte_length = options->command_byte_capacity};
  amdf_user_queue_mapping_info_t host_mapping;
  amdf_user_queue_mapping_info_t device_mapping;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_user_queue_initialize(
        context, &native, &queue->storage, &host_mapping, &device_mapping);
  }
  if (iree_status_is_ok(status)) {
    queue->base.placement = queue->storage.placement;
    if (host_mapping.index_bits != 64 || host_mapping.doorbell_bits != 64 ||
        !host_mapping.read_index_address || !host_mapping.write_index_address ||
        !host_mapping.doorbell_address || !device_mapping.ring_address) {
      status =
          iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                           "AQL requires native 64-bit indices and doorbell");
    } else {
      iree_hal_amd_gpu_aql_ring_initialize(
          (void*)(uintptr_t)host_mapping.ring_address,
          device_mapping.ring_address,
          (uint32_t)(host_mapping.ring_byte_length /
                     sizeof(iree_hal_amd_gpu_aql_packet_t)),
          (void*)(uintptr_t)host_mapping.read_index_address,
          (void*)(uintptr_t)host_mapping.write_index_address,
          (void*)(uintptr_t)host_mapping.doorbell_address,
          queue->storage.host_publish_executor, &queue->producer.ring);
      iree_hal_amd_gpu_aql_scratch_initialize(
          &queue->growth, queue->storage.queue, options->scratch_memory,
          &queue->scratch);
      status = iree_hal_amd_gpu_argument_storage_allocate(
          context, options->memory, options->argument_capacity, 0,
          host_allocator, &queue->producer.arguments);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_signal_storage_t* signals = NULL;
    status = iree_hal_amd_gpu_aql_signal_storage_allocate(
        context, &queue->options.signals->pool, options->memory,
        options->retirement_capacity, queue_count, host_allocator, &signals);
    if (iree_status_is_ok(status)) {
      signals->next = queue->options.signals->storage;
      queue->options.signals->storage = signals;
    }
  }
  if (iree_status_is_ok(status) && !options->programs->grid_sync &&
      (params->features & IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)) {
    status = iree_hal_amd_gpu_grid_storage_initialize(
        context, options->scratch_memory, &queue->grid);
  }
  if (iree_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    const amdf_status_t cleanup =
        iree_hal_amd_gpu_aql_queue_shutdown(&queue->base);
    if (amdf_status_is_ok(cleanup)) {
      iree_hal_queue_release(&queue->base);
    } else {
      iree_hal_amd_gpu_context_report(context, cleanup,
                                      "AQL construction rollback");
      status = iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(cleanup, "AQL construction rollback"));
    }
  }
  return status;
}

uint32_t iree_hal_amd_gpu_aql_queue_maximum_program_packet_count(
    iree_hal_queue_t* base) {
  const iree_hal_amd_gpu_aql_queue_t* queue = (const void*)base;
  // Use the configured ring capacity to amortize each device materialization,
  // leaving half the ring for dependencies and publication packets.
  return queue->producer.ring.capacity / 2;
}
