// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/queue.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/dispatch.h"
#include "iree/hal/driver/amd/gpu/executable.h"
#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/capture_pool.h"
#include "iree/hal/driver/amd/gpu/queue/control_program.h"
#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/host_call.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/growth.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/publication.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/retirement.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/submission.h"
#include "iree/hal/driver/amd/gpu/queue/program.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

typedef enum iree_hal_amd_gpu_pm4_wait_kind_e {
  IREE_HAL_AMD_GPU_PM4_WAIT_READY,
  IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION,
  IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION,
} iree_hal_amd_gpu_pm4_wait_kind_t;

// Reusable admission sidecar. The wait set joins failure cancellation with
// inline tracker completion before returning this storage to the capture.
typedef struct iree_hal_amd_gpu_pm4_pending_t {
  // Queue cancellation membership and terminal capture ownership.
  iree_hal_amd_gpu_host_wait_entry_t entry;
  // Queue whose accepted capture dominates this sidecar's lifetime.
  iree_hal_amd_gpu_pm4_queue_t* queue;
  // Owned captured operation awaiting admission.
  iree_hal_amd_gpu_pm4_submission_t* submission;
  // Owned allocation bytes when the capture tail cannot hold this sidecar,
  // or zero for borrowed tail storage. Returned after tracker callbacks join.
  iree_host_size_t allocation_size;
  // Single tracker allocation in this block's trailing storage. Finalization
  // frees the entire allocation, resetting the arena for the next registration.
  iree_allocator_inline_storage_t tracker_storage;
  // Kind selected by admission; never READY while a sidecar exists.
  iree_hal_amd_gpu_pm4_wait_kind_t kind;
  // Before begin, borrows the source from submission. After begin, owns the
  // registered completion or submission-availability timeline.
  iree_hal_semaphore_t* semaphore;
  // Captured value paired with semaphore.
  uint64_t value;
} iree_hal_amd_gpu_pm4_pending_t;

// Stack-owned cold request; the queue only borrows it until completion.
typedef struct iree_hal_amd_gpu_pm4_frontier_request_t {
  // Exact protocol tag and publication link.
  iree_hal_amd_gpu_pm4_entry_t entry;
  // Queue borrowed by the serialized membership caller through every handoff.
  iree_hal_amd_gpu_pm4_queue_t* queue;
  // Device-owned notification and caller-owned result.
  iree_hal_amd_gpu_frontier_request_t request;
} iree_hal_amd_gpu_pm4_frontier_request_t;

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_queue_vtable;

static iree_status_t iree_hal_amd_gpu_pm4_file_submit(
    iree_hal_queue_t* base,
    const iree_hal_amd_gpu_transfer_operation_t* operation, const void* before,
    const void* after, iree_hal_amd_gpu_retirement_callback_t callback);

static void iree_hal_amd_gpu_pm4_file_add_effects(
    iree_hal_queue_t* base, iree_hal_memory_effects_t effects, void* value) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_pm4_barrier_t* barrier = value;
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM)) {
    barrier->release_cache_control |= queue->options.barriers.release;
    barrier->execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE |
                          IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  }
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM)) {
    barrier->acquire_cache_control |= queue->options.barriers.acquire;
  }
}

static void iree_hal_amd_gpu_pm4_wake_shutdown(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (IREE_UNLIKELY(
          iree_atomic_load(&queue->closed, iree_memory_order_acquire))) {
    if (iree_hal_amd_gpu_context_wake_failure(queue->context)) {
      return;
    }
    const amdf_status_t status =
        iree_hal_amd_gpu_context_signal(queue->context);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(queue->context, status,
                                      "GPU shutdown event_signal");
    }
  }
}

// Records failure and reports it without ending an issued operation's borrows.
// Final capture release publishes failed signals after all native pieces join.
void iree_hal_amd_gpu_pm4_fail_submission(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_amd_gpu_pm4_submission_fail(submission, status);
  iree_status_free(status);
}

void iree_hal_amd_gpu_pm4_abandon(iree_hal_amd_gpu_pm4_queue_t* queue,
                                  iree_hal_amd_gpu_pm4_submission_t* submission,
                                  iree_status_t status) {
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_pm4_submission_profile(submission),
      false, 0);
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_amd_gpu_pm4_submission_reject(submission, iree_status_code(status));
  iree_status_free(status);
  const bool has_host_owner = submission->has_host_owner;
  submission->has_host_owner = false;
  iree_hal_amd_gpu_pm4_submission_release(submission);
  if (has_host_owner) {
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
}

static iree_status_t iree_hal_amd_gpu_pm4_prepare_arguments(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t capacity) {
  return iree_hal_amd_gpu_byte_storage_allocate(
      queue->context, queue->options.memory, capacity, 0, queue->host_allocator,
      &queue->producer.arguments);
}

static iree_status_t iree_hal_amd_gpu_pm4_prepare_signals(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_signal_storage_allocate(
      queue->context, &queue->options.signals->pool, queue->options.memory,
      queue->options.retirement_capacity, queue->producer.frontier.count,
      queue->host_allocator, &storage));
  storage->next = queue->options.signals->storage;
  queue->options.signals->storage = storage;
  return iree_ok_status();
}

// The exact publisher retires only the ordered completed generation prefix.
// Returning native backing is asynchronous and already has release storage.
static void iree_hal_amd_gpu_pm4_trim_generations(
    iree_hal_amd_gpu_context_t* context, uint64_t completed,
    iree_hal_amd_gpu_byte_storage_t** first,
    iree_hal_amd_gpu_byte_storage_t** last) {
  while (*first && (*first)->end <= completed) {
    iree_hal_amd_gpu_byte_storage_t* storage = *first;
    *first = storage->next;
    iree_hal_amd_gpu_byte_storage_free(context, storage);
  }
  if (!*first) {
    *last = NULL;
  }
}

static void iree_hal_amd_gpu_pm4_trim_storage(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_hal_amd_gpu_sample_storage_trim(queue->context, queue->samples);
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
  iree_hal_amd_gpu_pm4_trim_generations(
      queue->context,
      (uint64_t)iree_atomic_load(&queue->completed.arguments,
                                 iree_memory_order_acquire),
      &queue->producer.retired_arguments,
      &queue->producer.retired_arguments_tail);
  iree_hal_amd_gpu_pm4_trim_generations(
      queue->context,
      (uint64_t)iree_atomic_load(&queue->completed.transfers,
                                 iree_memory_order_acquire),
      &queue->producer.transfers.retired, &queue->producer.transfers.tail);
  iree_hal_amd_gpu_pm4_scratch_trim(
      queue->context,
      (uint64_t)iree_atomic_load(&queue->completed.count,
                                 iree_memory_order_acquire),
      &queue->producer.scratch);
  iree_hal_amd_gpu_signal_cache_trim(queue->context, 1, queue->options.signals);
}

static iree_status_code_t iree_hal_amd_gpu_pm4_feedback_failure(
    const iree_hal_amd_gpu_pm4_queue_t* queue) {
  return queue->options.feedback ? (iree_status_code_t)iree_atomic_load(
                                       &queue->options.feedback->failure_code,
                                       iree_memory_order_acquire)
                                 : IREE_STATUS_OK;
}

static iree_status_t iree_hal_amd_gpu_pm4_check_failure(
    const iree_hal_amd_gpu_pm4_queue_t* queue) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      (amdf_status_t)iree_atomic_load(&queue->native_failure,
                                      iree_memory_order_acquire),
      "PM4 native completion"));
  iree_status_code_t code = iree_hal_amd_gpu_pm4_feedback_failure(queue);
  return code == IREE_STATUS_OK
             ? iree_ok_status()
             : iree_make_status(code, "GPU sanitizer failure");
}

iree_status_t iree_hal_amd_gpu_pm4_queue_set_profile(
    iree_hal_queue_t* base, iree_hal_amd_gpu_profile_t* profile) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_register_queue(
      profile, queue->options.queue_ordinal));
  // Public completion may precede the final host retirement access. Begin/end
  // already join user work; only this cold session change joins that tail.
  iree_status_t status = iree_ok_status();
  while (!iree_hal_amd_gpu_pm4_queue_is_idle(base) &&
         iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_check_failure(queue);
    if (iree_status_is_ok(status)) {
      iree_thread_yield();
    }
  }
  const bool enabled = iree_hal_amd_gpu_profile_timestamps_dispatches(profile);
  if (enabled && !queue->samples && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_sample_storage_allocate(
        queue->context, queue->options.memory,
        (uint64_t)queue->options.retirement_capacity *
            sizeof(iree_hal_amd_gpu_profile_sample_t),
        queue->host_allocator, &queue->samples);
  }
  if (!enabled && iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sample_storage_free(queue->context, queue->host_allocator,
                                         queue->samples);
    queue->samples = NULL;
  }
  queue->profile = iree_status_is_ok(status) ? profile : NULL;
  queue->file_executor.profile = queue->profile;
  return status;
}

// A candidate contains only waits emitted in this native piece. It becomes
// the accepted FIFO prefix after native acceptance; a rejected KERNEL piece
// cannot claim its dependencies for a later operation. Retained signals carry
// exact immutable snapshots, never a producer queue's newer mutable state.
static void iree_hal_amd_gpu_pm4_prepare_wait_frontier(
    iree_hal_amd_gpu_pm4_queue_t* queue, iree_host_size_t wait_count,
    const iree_hal_amd_gpu_pm4_submission_wait_t* waits,
    uint32_t native_wait_count) {
  queue->producer.frontier.pending = queue->producer.frontier.published;
  if (native_wait_count) {
    queue->producer.frontier.pending = queue->producer.frontier.scratch;
    memcpy(queue->producer.frontier.pending, queue->producer.frontier.published,
           queue->producer.frontier.count * sizeof(uint64_t));
    for (iree_host_size_t i = 0; i < wait_count; ++i) {
      if (waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(waits[i].signal,
                                              queue->producer.frontier.count,
                                              queue->producer.frontier.pending);
      }
    }
  }
}

static void iree_hal_amd_gpu_pm4_capacity_available(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (!iree_atomic_exchange(&queue->capacity.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_publication_enqueue(
        queue->options.publication, queue->options.source_ordinal,
        &queue->publisher, &queue->capacity.entry.publication);
  }
}

static void iree_hal_amd_gpu_pm4_await_capacity(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t completed) {
  iree_atomic_exchange(&queue->completed.waiting_for_capacity, 1,
                       iree_memory_order_acq_rel);
  if ((uint64_t)iree_atomic_load(&queue->completed.count,
                                 iree_memory_order_acquire) != completed ||
      iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) ||
      iree_hal_amd_gpu_pm4_feedback_failure(queue)) {
    iree_hal_amd_gpu_pm4_capacity_available(queue);
  }
}

static void iree_hal_amd_gpu_pm4_park_capacity(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  iree_atomic_store(&entry->next, 0, iree_memory_order_relaxed);
  if (queue->capacity.last) {
    iree_atomic_store(&queue->capacity.last->next, (intptr_t)entry,
                      iree_memory_order_relaxed);
  } else {
    iree_atomic_store(&queue->capacity.first, (intptr_t)entry,
                      iree_memory_order_release);
  }
  queue->capacity.last = entry;
}

void iree_hal_amd_gpu_pm4_queue_notify_failure(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                      queue->options.source_ordinal);
  if (iree_atomic_exchange(&queue->completed.waiting_for_capacity, 0,
                           iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_pm4_capacity_available(queue);
  }
}

static void iree_hal_amd_gpu_pm4_allocation_ready(void* user_data,
                                                  iree_status_t status);

// Publishes an owned capture. Immediate retirement may free it before return.
static iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_status_t failure = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (!iree_status_is_ok(failure)) {
    if (queue->kernel && queue->kernel->pending.byte_length) {
      queue->producer.ring.reserved_words = queue->kernel->rollback.words;
      if (queue->kernel->rollback.storage) {
        queue->kernel->rollback.storage->end = queue->kernel->rollback.bytes;
      }
      queue->kernel->pending.byte_length = 0;
      iree_hal_amd_gpu_sample_storage_rollback(queue->samples);
    }
    iree_hal_amd_gpu_pm4_abandon(queue, submission, failure);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (queue->kernel) {
    if (queue->kernel->pending.byte_length) {
      return iree_hal_amd_gpu_pm4_publish_kernel(queue, position);
    }
    queue->kernel->rollback.words = queue->producer.ring.reserved_words;
    queue->kernel->rollback.storage = queue->producer.arguments;
    queue->kernel->rollback.bytes = queue->producer.arguments->end;
  }
  const uint32_t scratch_required =
      iree_hal_amd_gpu_pm4_submission_scratch_required(submission);
  if (scratch_required &&
      (!queue->producer.scratch.current ||
       queue->producer.scratch.current->wave_byte_length < scratch_required)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SCRATCH;
  }
  const bool first =
      submission->entry.kind != IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER ||
      (!submission->transfer.cursor.index &&
       !submission->transfer.cursor.offset);
  iree_hal_amd_gpu_pm4_prepare_wait_frontier(
      queue, submission->wait_count, submission->waits,
      first ? submission->native_wait_count : 0);
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION) {
    // A deferred pool callback may enqueue and retire the capture before its
    // own tail returns. The inline path stays under this publisher's borrow.
    iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                          iree_memory_order_relaxed);
    bool pending = false;
    iree_status_t status = iree_hal_amd_gpu_allocation_prepare(
        submission->allocation.operation,
        queue->frontier.tracker ? queue->frontier.axis : 0,
        queue->producer.frontier.count, queue->producer.frontier.pending,
        iree_hal_amd_gpu_pm4_allocation_ready, submission, &pending);
    if (!iree_status_is_ok(status)) {
      iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                            iree_memory_order_release);
      iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
    if (pending) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
    return iree_hal_amd_gpu_pm4_publish_transfer(queue, submission, position);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    return iree_hal_amd_gpu_pm4_publish_execute(queue, submission, position);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_CONTROL) {
    return iree_hal_amd_gpu_pm4_publish_control(queue, submission, position);
  }
  iree_hal_amd_gpu_pm4_dispatch_request_t dispatch = {0};
  uint64_t* addresses = NULL;
  if (submission->entry.kind != IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION) {
    const bool indirect_parameters =
        iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags);
    const iree_host_size_t binding_count = submission->dispatch.bindings.count;
    dispatch = (iree_hal_amd_gpu_pm4_dispatch_request_t){
        .executable = submission->dispatch.executable,
        .function = submission->dispatch.function,
        .launch = submission->dispatch.launch,
        .parameters = indirect_parameters
                          ? submission->dispatch.bindings.values[binding_count]
                          : (iree_hal_buffer_ref_t){0},
        .constants = submission->dispatch.constants,
        .bindings = submission->dispatch.bindings,
        .flags = submission->dispatch.flags,
    };
    addresses = (uint64_t*)(submission->dispatch.bindings.values +
                            binding_count + indirect_parameters);
  }
  const iree_hal_amd_gpu_pm4_dispatch_publication_t request = {
      .dispatch = &dispatch,
      .waits = submission->waits,
      .wait_count = submission->wait_count,
      .native_wait_count = submission->native_wait_count,
      .signals = submission->signals,
      .before = submission->before,
      .after = submission->after,
      .profile = iree_hal_amd_gpu_pm4_submission_profile(submission),
  };
  iree_status_t status = iree_ok_status();
  const iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_publish_dispatch(queue, position, scratch_required,
                                            &request, addresses, &submission,
                                            &status);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  }
  return result;
}

// An unresolved edge leaves no native reader pinned. In particular, a consumer
// awaiting a future producer cannot exhaust that producer's signal backing.
static iree_status_t iree_hal_amd_gpu_pm4_admit_waits(
    iree_hal_amd_gpu_pm4_queue_t* queue, iree_host_size_t wait_count,
    iree_hal_amd_gpu_pm4_submission_wait_t* waits, bool native_eligible,
    uint32_t* out_native_wait_count, iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_READY;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_failure(queue));
  uint32_t native_wait_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < wait_count && *out_kind == IREE_HAL_AMD_GPU_PM4_WAIT_READY &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_pm4_submission_wait_t* wait = &waits[i];
    if (native_eligible && iree_hal_amd_gpu_semaphore_isa(wait->semaphore)) {
      iree_hal_amd_gpu_semaphore_wait_t acquired;
      status = iree_hal_amd_gpu_semaphore_acquire_wait(
          wait->semaphore, wait->value, queue->context, &acquired);
      if (iree_status_is_ok(status)) {
        if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED) {
          *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION;
        } else if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE) {
          wait->signal = acquired.signal;
          ++native_wait_count;
        } else if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST) {
          *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION;
        }
      }
    } else {
      uint64_t value = 0;
      status = iree_hal_semaphore_query(wait->semaphore, &value);
      if (iree_status_is_ok(status) && value < wait->value) {
        *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION;
      }
    }
    if (*out_kind != IREE_HAL_AMD_GPU_PM4_WAIT_READY) {
      *out_wait_index = i;
    }
  }
  if (*out_kind != IREE_HAL_AMD_GPU_PM4_WAIT_READY ||
      !iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < wait_count; ++i) {
      iree_hal_amd_gpu_signal_release(waits[i].signal);
      waits[i].signal = NULL;
    }
  } else {
    *out_native_wait_count = native_wait_count;
  }
  return status;
}

// An unpublished failed operation cannot execute, but its outputs still join
// every input. Consume completed edges in place so repeated wakeups traverse
// each dependency once and need no additional storage or native readers.
static iree_status_t iree_hal_amd_gpu_pm4_admit_failed(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_READY;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_failure(queue));
  while (submission->wait_count) {
    iree_hal_amd_gpu_pm4_submission_wait_t* wait = submission->waits;
    uint64_t value = 0;
    iree_status_t status = iree_hal_semaphore_query(wait->semaphore, &value);
    if (iree_status_is_ok(status) && value < wait->value) {
      *out_wait_index = 0;
      *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION;
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_submission_reject(submission,
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

static iree_status_t iree_hal_amd_gpu_pm4_admit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
  if (IREE_UNLIKELY(iree_atomic_load(&submission->failure_code,
                                     iree_memory_order_relaxed))) {
    return iree_hal_amd_gpu_pm4_admit_failed(queue, submission, out_wait_index,
                                             out_kind);
  }
  // A bounded direct publication must fit strictly inside half the ring.
  // Reserve a bounded launch plus 96 words for barriers, signal, retirement,
  // notification, and alignment. Large fan-in joins use completed host
  // timelines, leaving command capacity independent of the semaphore list.
  const uint32_t word_budget =
      iree_min(queue->producer.ring.capacity / 2, 16384u);
  const uint32_t reserved_words = IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS + 96;
  const bool native_eligible =
      submission->wait_count <= (word_budget - reserved_words) / 7 &&
      !(submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER &&
        submission->transfer.host_wait_required) &&
      !(submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION &&
        (!queue->frontier.tracker ||
         iree_hal_amd_gpu_allocation_requires_host_wait(
             submission->allocation.operation)));
  iree_status_t status = iree_hal_amd_gpu_pm4_admit_waits(
      queue, submission->wait_count, submission->waits, native_eligible,
      &submission->native_wait_count, out_wait_index, out_kind);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    iree_hal_amd_gpu_pm4_submission_reject(submission,
                                           iree_status_code(status));
    iree_status_free(status);
    status = iree_hal_amd_gpu_pm4_admit_failed(queue, submission,
                                               out_wait_index, out_kind);
  }
  return status;
}

static void iree_hal_amd_gpu_pm4_pending_release(
    iree_hal_amd_gpu_pm4_pending_t* pending) {
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
static void iree_hal_amd_gpu_pm4_pending_end(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                        iree_memory_order_release);
  iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
}

// File I/O and host calls may finish on an external progress owner. Keep their
// callback tails in the same idle proof as semaphore-ready publication.
static void iree_hal_amd_gpu_pm4_host_begin(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
}

static void iree_hal_amd_gpu_pm4_host_end(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_pending_end((iree_hal_amd_gpu_pm4_queue_t*)base);
}

// Wait completion reenters the publication owner only after registration and
// every cancellation borrower have relinquished the captured sidecar.
static void iree_hal_amd_gpu_pm4_enqueue(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission);

static void iree_hal_amd_gpu_pm4_allocation_ready(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_pm4_submission_t* submission = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)submission->allocation.queue;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_enqueue(queue, submission);
  } else {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  }
  iree_hal_amd_gpu_pm4_pending_end(queue);
}

static void iree_hal_amd_gpu_pm4_pending_resume(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_pm4_pending_t* pending = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
  bool dependency_failed = false;
  if (!iree_status_is_ok(status) && pending->semaphore) {
    uint64_t value = 0;
    iree_status_t dependency_status =
        iree_hal_semaphore_query(pending->semaphore, &value);
    dependency_failed = !iree_status_is_ok(dependency_status);
    status = iree_status_join(status, dependency_status);
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_pm4_pending_release(pending);
  if (dependency_failed) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    iree_hal_amd_gpu_pm4_submission_reject(submission,
                                           iree_status_code(status));
    iree_status_free(status);
    status = iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_enqueue(queue, submission);
  } else {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  }
  iree_hal_amd_gpu_pm4_pending_end(queue);
}

static void iree_hal_amd_gpu_pm4_pending_begin(
    iree_hal_amd_gpu_pm4_pending_t* pending) {
  iree_hal_amd_gpu_pm4_queue_t* queue = pending->queue;
  iree_hal_semaphore_t* source = pending->semaphore;
  pending->semaphore = NULL;
  iree_status_t status = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (iree_status_is_ok(status)) {
    if (pending->kind == IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION) {
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
        iree_hal_amd_gpu_pm4_pending_resume, pending,
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
  iree_hal_amd_gpu_pm4_pending_resume(pending, status);
}

static iree_status_t iree_hal_amd_gpu_pm4_pending_allocate(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_host_size_t wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t kind,
    iree_hal_amd_gpu_pm4_pending_t** out_pending) {
  iree_byte_span_t storage =
      iree_hal_amd_gpu_pm4_submission_tail_storage(submission);
  iree_host_size_t allocation_size = 0;
  if (storage.data_length < queue->pending_size) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
        &queue->captures, queue->pending_size, &storage));
    allocation_size = storage.data_length;
  }
  iree_hal_amd_gpu_pm4_pending_t* pending = (void*)storage.data;
  const iree_host_size_t prefix_size =
      iree_host_align(sizeof(*pending), iree_max_align_t);
  *pending = (iree_hal_amd_gpu_pm4_pending_t){
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

// The current publisher admits captured work immediately before publication.
// An unresolved edge leaves this list, allowing later producers to make
// progress. The admitted count and native readers survive capacity deferral
// unchanged.
static iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_process(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->native_wait_count == UINT32_MAX) {
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_pm4_wait_kind_t kind;
    iree_status_t status =
        iree_hal_amd_gpu_pm4_admit(queue, submission, &wait_index, &kind);
    if (!iree_status_is_ok(status) || kind != IREE_HAL_AMD_GPU_PM4_WAIT_READY) {
      iree_hal_amd_gpu_pm4_pending_t* pending = NULL;
      if (iree_status_is_ok(status)) {
        iree_hal_amd_gpu_profile_operation_defer(
            iree_hal_amd_gpu_pm4_submission_profile(submission));
        status = iree_hal_amd_gpu_pm4_pending_allocate(
            queue, submission, wait_index, kind, &pending);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_amd_gpu_pm4_pending_begin(pending);
      } else {
        iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
      }
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
  }
  const iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_publish(queue, submission);
  if (result >= IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY) {
    iree_hal_amd_gpu_profile_operation_defer(
        iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  return result;
}

static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_process_submission(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_process(queue, submission);
  if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE &&
      submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
    // Accepted transfer chunks are self-contained. Yield the retained suffix
    // to ready submissions instead of keeping a bulk transfer ahead of them
    // across every staging-credit wait. The final chunk alone signals users.
    iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                       &submission->entry.publication);
    result = IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (result < IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS) {
    return result;
  }
  // A parked invocation cannot reserve an unpublished mutable instance while
  // another ready invocation on this queue attempts to acquire the program.
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE &&
      submission->execute.instance) {
    iree_hal_amd_gpu_program_cancel(&submission->execute.program->resident,
                                    submission->execute.instance);
    submission->execute.instance = NULL;
  }
  iree_hal_amd_gpu_pm4_entry_kind_t kind =
      IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS;
  uint64_t capacity = 0;
  if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS;
    uint64_t required = 0;
    if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
      required = submission->execute.program->argument_length;
    } else if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_CONTROL) {
      required = sizeof(iree_hal_amd_gpu_control_program_arguments_t);
    } else {
      const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
          iree_hal_amd_gpu_pm4_dispatch_storage(submission->dispatch.function,
                                                submission->dispatch.flags);
      required = iree_max(storage.length, storage.alignment);
    }
    capacity = queue->producer.arguments->capacity;
    while (capacity < required) {
      capacity *= 2;
    }
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SAMPLES) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SAMPLES;
    const iree_hal_amd_gpu_profile_operation_t* profile =
        iree_hal_amd_gpu_pm4_submission_profile(submission);
    capacity =
        iree_math_round_up_to_pow2_u64((uint64_t)profile->completion.count *
                                       profile->completion.sample_stride);
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PACKETS) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_PACKETS;
    capacity = (uint64_t)queue->options.retirement_capacity *
               sizeof(iree_hal_amd_gpu_pm4_dispatch_packet_t);
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_TRANSFERS) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_TRANSFERS;
    capacity = iree_max(
        UINT64_C(8),
        iree_math_round_up_to_pow2_u64(submission->transfer.staging_capacity));
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SCRATCH) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SCRATCH;
    capacity = iree_hal_amd_gpu_pm4_submission_scratch_required(submission);
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM) {
    kind = IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_PROGRAM;
  }
  iree_status_t status = iree_hal_amd_gpu_pm4_growth_begin(
      &queue->growth, kind, submission, capacity);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  }
  return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
}

static void iree_hal_amd_gpu_pm4_resume_capacity(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_atomic_store(&queue->capacity.pending, 0, iree_memory_order_release);
  iree_hal_amd_gpu_publication_entry_t* entry = (void*)iree_atomic_load(
      &queue->capacity.first, iree_memory_order_relaxed);
  if (!entry) {
    return;
  }
  // Detach before acceptance can release/requeue the capture. Later native
  // work cannot prepare over a pending KERNEL span while metadata proceeds.
  iree_hal_amd_gpu_publication_entry_t* next =
      (void*)iree_atomic_load(&entry->next, iree_memory_order_relaxed);
  iree_atomic_store(&queue->capacity.first, (intptr_t)next,
                    iree_memory_order_release);
  if (!next) {
    queue->capacity.last = NULL;
  }
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  iree_hal_amd_gpu_pm4_submission_t* submission = iree_containerof(
      entry, iree_hal_amd_gpu_pm4_submission_t, entry.publication);
  const iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_process_submission(queue, submission);
  if (result != IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
    iree_atomic_store(&queue->capacity.first, (intptr_t)entry,
                      iree_memory_order_release);
    if (!next) {
      queue->capacity.last = entry;
    }
  }
  if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY) {
    iree_hal_amd_gpu_pm4_await_capacity(queue, completed);
  } else if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE || next) {
    iree_hal_amd_gpu_pm4_capacity_available(queue);
  }
}

static void iree_hal_amd_gpu_pm4_expand_frontiers(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_frontier_request_t* request) {
  if (request->phase == IREE_HAL_AMD_GPU_FRONTIER_BEGIN && request->vectors) {
    uint64_t* frontier = request->vectors;
    memcpy(frontier, queue->producer.frontier.published,
           queue->producer.frontier.count * sizeof(uint64_t));
    uint64_t* previous = iree_min(queue->producer.frontier.published,
                                  queue->producer.frontier.scratch);
    const bool pending_waits =
        queue->producer.frontier.pending != queue->producer.frontier.published;
    if (pending_waits) {
      memcpy(frontier + request->queue_count, queue->producer.frontier.pending,
             queue->producer.frontier.count * sizeof(uint64_t));
    }
    queue->producer.frontier.published = frontier;
    queue->producer.frontier.scratch = frontier + request->queue_count;
    queue->producer.frontier.pending =
        pending_waits ? queue->producer.frontier.scratch : frontier;
    queue->producer.frontier.count = request->queue_count;
    request->vectors =
        previous == (void*)(queue->issued + queue->options.retirement_capacity)
            ? NULL
            : previous;
  }
  iree_hal_amd_gpu_pm4_growth_process_frontiers(&queue->growth, request);
}

// The exact publisher installs acquired backing and requeues the waiting
// cohort. Native callbacks never enter this function or visit its captures.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_process_entry(iree_hal_amd_gpu_pm4_queue_t* queue,
                                   iree_hal_amd_gpu_publication_entry_t* link) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_pm4_publish");
  iree_hal_amd_gpu_pm4_entry_t* entry =
      iree_containerof(link, iree_hal_amd_gpu_pm4_entry_t, publication);
  if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_CAPACITY) {
    iree_hal_amd_gpu_pm4_resume_capacity(queue);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_FRONTIERS) {
    iree_hal_amd_gpu_pm4_frontier_request_t* request =
        iree_containerof(entry, iree_hal_amd_gpu_pm4_frontier_request_t, entry);
    iree_hal_amd_gpu_pm4_expand_frontiers(queue, &request->request);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRIM) {
    iree_hal_amd_gpu_pm4_trim_storage(queue);
    iree_atomic_store(&queue->trim.pending, 0, iree_memory_order_release);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS) {
    iree_hal_amd_gpu_pm4_growth_reuse_signals(&queue->growth);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_PROGRAM) {
    iree_hal_amd_gpu_pm4_growth_reuse_program(entry);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (entry->kind >= IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS) {
    const bool transfers =
        entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_TRANSFERS;
    const bool samples = entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SAMPLES;
    const bool packets = entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_PACKETS;
    iree_hal_amd_gpu_byte_storage_t** current =
        packets     ? &queue->packets
        : samples   ? &queue->samples->current
        : transfers ? &queue->producer.transfers.current
                    : &queue->producer.arguments;
    iree_hal_amd_gpu_byte_storage_t** first =
        transfers ? &queue->producer.transfers.retired
                  : &queue->producer.retired_arguments;
    iree_hal_amd_gpu_byte_storage_t** last =
        transfers ? &queue->producer.transfers.tail
                  : &queue->producer.retired_arguments_tail;
    iree_hal_amd_gpu_pm4_growth_result_t result;
    iree_status_t status = iree_hal_amd_gpu_pm4_growth_end(
        entry, *current ? (*current)->end : 0, &result);
    if (iree_status_is_ok(status)) {
      if (result.bytes && packets) {
        queue->packets = result.bytes;
      } else if (result.bytes && samples) {
        iree_hal_amd_gpu_sample_storage_install(queue->samples, result.bytes);
        iree_hal_amd_gpu_sample_storage_trim(queue->context, queue->samples);
      } else if (result.bytes) {
        if (*current) {
          if (*last) {
            (*last)->next = *current;
          } else {
            *first = *current;
          }
          *last = *current;
        }
        *current = result.bytes;
        iree_hal_amd_gpu_pm4_trim_generations(
            queue->context,
            (uint64_t)iree_atomic_load(transfers ? &queue->completed.transfers
                                                 : &queue->completed.arguments,
                                       iree_memory_order_acquire),
            first, last);
      }
      if (result.scratch) {
        iree_hal_amd_gpu_pm4_scratch_install(
            queue->context, result.scratch,
            (uint64_t)iree_atomic_load(&queue->completed.count,
                                       iree_memory_order_acquire),
            &queue->producer.scratch);
      }
      if (result.signals) {
        result.signals->next = queue->options.signals->storage;
        queue->options.signals->storage = result.signals;
      }
      if (result.first) {
        iree_hal_amd_gpu_publisher_enqueue_batch(&queue->publisher,
                                                 result.first, result.last);
      }
    } else {
      // Only the publication owner consumes a failed cohort. Every captured
      // operation keeps its own signal/resource ownership through abandonment.
      for (iree_hal_amd_gpu_publication_entry_t* current = result.first;
           current;) {
        iree_hal_amd_gpu_publication_entry_t* next =
            (void*)iree_atomic_load(&current->next, iree_memory_order_relaxed);
        iree_hal_amd_gpu_pm4_submission_t* submission = iree_containerof(
            current, iree_hal_amd_gpu_pm4_submission_t, entry.publication);
        iree_hal_amd_gpu_pm4_abandon(queue, submission,
                                     iree_status_clone(status));
        current = next;
      }
      iree_status_free(status);
    }
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (queue->capacity.last) {
    iree_hal_amd_gpu_pm4_park_capacity(queue, link);
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  iree_hal_amd_gpu_pm4_submission_t* submission =
      iree_containerof(entry, iree_hal_amd_gpu_pm4_submission_t, entry);
  const iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_process_submission(queue, submission);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Native capacity leaves the ready FIFO eligible for metadata. Retirement
// wakes one bounded continuation for the ordered native cohort.
static bool iree_hal_amd_gpu_pm4_handoff(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* remainder, bool capacity,
    uint64_t completed) {
  if (capacity && remainder) {
    iree_hal_amd_gpu_pm4_park_capacity(queue, remainder);
    iree_hal_amd_gpu_pm4_await_capacity(queue, completed);
    remainder = NULL;
  }
  return iree_hal_amd_gpu_publisher_handoff(&queue->publisher, remainder);
}

// Publication mode and whether dependencies are ready determine ownership;
// enqueue order does not supply a HAL execution dependency.
static void iree_hal_amd_gpu_pm4_enqueue_entry(
    iree_hal_amd_gpu_pm4_queue_t* queue, iree_hal_amd_gpu_pm4_entry_t* entry) {
  if (!iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                          &entry->publication)) {
    return;
  }
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  iree_hal_amd_gpu_publication_entry_t* remainder =
      iree_hal_amd_gpu_publisher_next(&queue->publisher);
  iree_hal_amd_gpu_pm4_publication_result_t result =
      remainder ? iree_hal_amd_gpu_pm4_process_entry(queue, remainder)
                : IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
    remainder = NULL;
  }
  if (iree_hal_amd_gpu_pm4_handoff(
          queue, remainder, result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY,
          completed) ||
      (iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) ||
       iree_hal_amd_gpu_pm4_feedback_failure(queue))) {
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.source_ordinal);
  }
  iree_hal_amd_gpu_pm4_wake_shutdown(queue);
}

static void iree_hal_amd_gpu_pm4_enqueue(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_pm4_enqueue_entry(queue, &submission->entry);
}

void iree_hal_amd_gpu_pm4_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  if (!iree_atomic_exchange(&queue->trim.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_pm4_enqueue_entry(queue, &queue->trim.entry);
  }
}

// Capture/validation have already succeeded. Admission errors after acceptance
// belong to signal timelines and the event sink, including on an inline turn.
static void iree_hal_amd_gpu_pm4_submit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_pm4_enqueue(queue, submission);
}

static bool iree_hal_amd_gpu_pm4_progress(void* user_data) {
  iree_hal_amd_gpu_pm4_queue_t* queue = user_data;
  if (iree_hal_amd_gpu_signal_cache_take_ready(queue->options.signals)) {
    iree_hal_amd_gpu_pm4_growth_signals_available(&queue->growth);
  }
  bool more = false;
  if (iree_hal_amd_gpu_publisher_acquire_service(&queue->publisher)) {
    const uint64_t completed = (uint64_t)iree_atomic_load(
        &queue->completed.count, iree_memory_order_acquire);
    bool capacity = false;
    iree_hal_amd_gpu_publication_entry_t* current = NULL;
    for (uint32_t i = 0; i < 32 && !capacity; ++i) {
      if (!current) {
        current = iree_hal_amd_gpu_publisher_next(&queue->publisher);
      }
      if (!current) {
        break;
      }
      const iree_hal_amd_gpu_pm4_publication_result_t result =
          iree_hal_amd_gpu_pm4_process_entry(queue, current);
      if (result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
        current = NULL;
      }
      capacity = result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
    more = iree_hal_amd_gpu_pm4_handoff(queue, current, capacity, completed);
  }
  // A completed native callback can return its final queue borrow after the
  // consumer drained its result. That hint must still wake shutdown.
  iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  iree_status_t failure = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (!iree_status_is_ok(failure)) {
    iree_hal_amd_gpu_host_wait_set_cancel(&queue->pending.waits);
  }
  iree_status_free(failure);
  return more;
}

void iree_hal_amd_gpu_pm4_fail_native(iree_hal_amd_gpu_pm4_queue_t* queue,
                                      amdf_status_t status,
                                      const char* operation) {
  int64_t expected = 0;
  if (iree_atomic_compare_exchange_strong(&queue->native_failure, &expected,
                                          status, iree_memory_order_release,
                                          iree_memory_order_relaxed)) {
    // A publishing caller hands off before its existing closed-queue wake.
    // Failure fanout then runs on completion, never on a competing observer.
    iree_hal_amd_gpu_pm4_queue_close(&queue->base);
    iree_hal_amd_gpu_context_fail_execution(queue->context, status, operation);
  }
  iree_hal_amd_gpu_pm4_queue_notify_failure(&queue->base);
}

static void iree_hal_amd_gpu_pm4_complete_download(void* user_data) {
  iree_hal_amd_gpu_download_t* download = user_data;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_download_complete");
  iree_hal_amd_gpu_download_complete(download);
  download->length = 0;
  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amd_gpu_pm4_retire(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  iree_hal_amd_gpu_pm4_queue_t* queue = user_data;
  if (!amdf_status_is_ok((amdf_status_t)iree_atomic_load(
          &queue->native_failure, iree_memory_order_relaxed)) ||
      !amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    if (!amdf_status_is_ok(wait_status)) {
      iree_hal_amd_gpu_pm4_fail_native(queue, wait_status,
                                       "GPU completion wait");
    }
    if (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      iree_status_t status = iree_hal_amd_gpu_pm4_check_failure(queue);
      iree_hal_amd_gpu_pm4_queue_fail_issued(queue, iree_status_code(status));
      iree_status_free(status);
    }
    return false;
  }
  uint64_t position = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                 iree_memory_order_relaxed);
  const uint64_t published = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_acquire);
  const iree_status_code_t feedback_failure =
      iree_hal_amd_gpu_pm4_feedback_failure(queue);
  if (feedback_failure != IREE_STATUS_OK &&
      !queue->feedback_failure_delivered) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    // Only this consumer releases published captures. Service cannot inspect
    // them during a logical failure because native retirement keeps running.
    for (uint64_t i = position; i != published; ++i) {
      iree_hal_amd_gpu_pm4_submission_t* submission =
          iree_hal_amd_gpu_pm4_retirement_submission(
              &queue->issued[i & queue->retirement.capacity_mask].action);
      iree_hal_amd_gpu_pm4_fail_submission(
          queue, submission,
          iree_make_status(feedback_failure, "GPU sanitizer failure"));
    }
    queue->feedback_failure_delivered = true;
  }
  uint64_t native_retired = 0;
  if (queue->kernel && position != published) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    const amdf_status_t status = iree_hal_amd_gpu_pm4_kmq_storage_refresh(
        queue->kernel, &native_retired);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_fail_native(queue, status,
                                       "kernel_queue_refresh_status");
      return false;
    }
  }
  const uint64_t limit = iree_min(published, position + 64);
  const uint64_t initial = position;
  bool waiting_for_producer = false;
  for (; position != limit; ++position) {
    if (queue->kernel &&
        queue->kernel->points[position & queue->retirement.capacity_mask] >
            native_retired) {
      const amdf_status_t status = iree_hal_amd_gpu_pm4_kmq_storage_arm(
          queue->kernel,
          queue->kernel->points[position & queue->retirement.capacity_mask]);
      if (!amdf_status_is_ok(status)) {
        iree_hal_amd_gpu_pm4_fail_native(queue, status,
                                         "kernel_queue_request_notification");
      }
      break;
    }
    iree_hal_amd_gpu_pm4_retirement_t* action =
        iree_hal_amd_gpu_pm4_retirement_poll(&queue->retirement, position);
    if (!action) {
      break;
    }
    iree_hal_amd_gpu_pm4_submission_t* submission =
        iree_hal_amd_gpu_pm4_retirement_submission(action);
    const bool terminal = iree_hal_amd_gpu_pm4_retirement_is_terminal(action);
    iree_status_code_t completion_code = IREE_STATUS_OK;
    if (terminal && !iree_hal_amd_gpu_pm4_submission_retirement_ready(
                        submission, &completion_code)) {
      waiting_for_producer = true;
      break;
    }
    if (queue->options.feedback) {
      // Native retirement makes this dispatch's reports observable. Drain the
      // ready shared prefix before waking host completion consumers.
      iree_hal_amd_gpu_completion_scope_lock(scope);
      iree_hal_amd_gpu_feedback_drain(queue->options.feedback);
      if (completion_code == IREE_STATUS_OK) {
        completion_code = iree_hal_amd_gpu_pm4_feedback_failure(queue);
      }
    }
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    iree_hal_amd_gpu_pm4_scratch_storage_free(queue->context, action->scratch);
    action->scratch = NULL;
    const uint64_t consumed_words = action->consumed_words;
    const uint64_t consumed_arguments = action->consumed_arguments;
    // Native consumption releases ABI storage independently of host sample
    // collection and transfer payload retirement.
    iree_atomic_store(&queue->completed.arguments, consumed_arguments,
                      iree_memory_order_release);
    if (action->download.length && feedback_failure == IREE_STATUS_OK &&
        completion_code == IREE_STATUS_OK) {
      if (out_copy) {
        *out_copy = (iree_hal_amd_gpu_completion_copy_t){
            .fn = iree_hal_amd_gpu_pm4_complete_download,
            .user_data = &action->download};
      }
      // The source keeps its complete FIFO ownership until the worker's CPU
      // copy returns. Helpers can retire other sources without touching it.
      break;
    }
    if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
      iree_atomic_store(
          &queue->completed.transfers,
          queue->issued[position & queue->retirement.capacity_mask]
              .consumed_transfers,
          iree_memory_order_release);
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_pm4_retire");
    if (terminal) {
      if (submission->profiled) {
        const iree_hal_amd_gpu_profile_operation_t* operation =
            iree_hal_amd_gpu_pm4_submission_profile(submission);
        iree_hal_amd_gpu_profile_operation_complete(queue->profile, operation,
                                                    &operation->completion);
        iree_hal_amd_gpu_sample_storage_retire(
            queue->samples, operation->completion.samples_end);
      }
      iree_hal_amd_gpu_pm4_submission_release_payload(submission);
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
    }
    iree_status_t status =
        completion_code == IREE_STATUS_OK
            ? iree_ok_status()
            : iree_make_status(completion_code, "GPU queue execution failed");
    for (iree_host_size_t i = 0;
         terminal && i < submission->signals.count && iree_status_is_ok(status);
         ++i) {
      iree_hal_semaphore_t* semaphore = submission->signals.semaphores[i];
      const uint64_t value = submission->signals.payload_values[i];
      status = iree_hal_amd_gpu_semaphore_isa(semaphore)
                   ? iree_hal_amd_gpu_semaphore_publish_completed(
                         semaphore, value, frontier)
                   : iree_hal_semaphore_signal(semaphore, value, frontier);
    }
    completion_code = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_fail_submission(queue, submission, status);
    }
    if (terminal) {
      iree_hal_amd_gpu_signal_retire(submission->signal, completion_code);
      iree_hal_amd_gpu_pm4_submission_set_frontier(
          submission, frontier ? &completed_frontier : NULL);
    }
    iree_hal_amd_gpu_pm4_submission_release(submission);
    iree_atomic_store(&queue->completed.words, consumed_words,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.count, position + 1,
                      iree_memory_order_release);
    IREE_TRACE_ZONE_END(z0);
  }
  // Participate even when no publisher appears enrolled: a later enrollment
  // must acquire the completion credit preceding this exchange.
  if (position != initial &&
      iree_atomic_exchange(&queue->completed.waiting_for_capacity, 0,
                           iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_pm4_capacity_available(queue);
  }
  if (queue->kernel && position == published) {
    // Enrollment precedes the predicate recheck. Either this observer sees a
    // racing publication or that publisher sees wake_required and signals.
    iree_atomic_exchange(&queue->kernel->wake_required, 1,
                         iree_memory_order_acq_rel);
    if ((uint64_t)iree_atomic_load(&queue->producer.count,
                                   iree_memory_order_acquire) != published) {
      iree_atomic_exchange(&queue->kernel->wake_required, 0,
                           iree_memory_order_acq_rel);
      return true;
    }
  }
  if (!queue->kernel && position == published) {
    // A publication on this idle queue must interrupt observation of other
    // queues. This passive hint alone never keeps the worker awake.
    *out_wait =
        (iree_hal_amd_gpu_completion_wait_t){.address = &queue->producer.count,
                                             .value = (int64_t)published,
                                             .is_64_bit = true};
  } else if (!queue->kernel && !waiting_for_producer) {
    // Capacity credit prevents record reuse until this consumer advances it.
    out_wait->address =
        &queue->retirement.records[position & queue->retirement.capacity_mask]
             .generation;
    out_wait->value = (int32_t)(position >> queue->retirement.capacity_shift);
    out_wait->pending = true;
  }
  // A later source can retire this source's native producer. One more round
  // after actual progress joins those causal payload lifetimes without a
  // topological queue walk or polling merely outstanding native work.
  return position != initial;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_pm4_queue_publication_source(iree_hal_queue_t* base) {
  return (iree_hal_amd_gpu_publication_source_t){iree_hal_amd_gpu_pm4_progress,
                                                 base};
}

iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_pm4_queue_completion_source(iree_hal_queue_t* base) {
  return (iree_hal_amd_gpu_completion_source_t){iree_hal_amd_gpu_pm4_retire,
                                                base};
}

static void iree_hal_amd_gpu_pm4_enqueue_frontiers(void* user_data) {
  iree_hal_amd_gpu_pm4_frontier_request_t* request = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = request->queue;
  iree_hal_amd_gpu_publication_enqueue(
      queue->options.publication, queue->options.source_ordinal,
      &queue->publisher, &request->entry.publication);
}

iree_status_t iree_hal_amd_gpu_pm4_queue_reserve_frontiers(
    iree_hal_queue_t* base, uint32_t queue_count,
    iree_notification_t* notification) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  // Only these serialized lifecycle requests change the installed extent.
  // Ordinary publication mutates vector contents, never this geometry.
  const uint32_t previous_count = queue->producer.frontier.count;
  iree_hal_amd_gpu_pm4_frontier_request_t request = {
      .entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_FRONTIERS,
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
      iree_hal_amd_gpu_pm4_enqueue_frontiers, &request);
}

static iree_status_t iree_hal_amd_gpu_pm4_check_acceptance(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "GPU queue is closed");
  }
  return iree_hal_amd_gpu_pm4_check_failure(queue);
}

static iree_status_t iree_hal_amd_gpu_pm4_host_call_enter(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_retirement_callback_t callback,
    const iree_hal_queue_barriers_t* barriers,
    iree_host_size_t public_signal_count) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const bool profiled = queue->profile != NULL;
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_transfer(
      &queue->captures, profiled, base->queue_family, waits,
      iree_hal_semaphore_list_empty(), 0, NULL, &queue->options.barriers,
      barriers, callback, &submission));
  if (profiled) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL, waits.count,
        public_signal_count, 1,
        iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, queue->profile != NULL, waits, signals, NULL, NULL,
      (iree_hal_amd_gpu_pm4_dispatch_t){0}, (iree_hal_buffer_ref_t){0},
      iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
      &queue->options.barriers, barriers, 0, &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, waits.count, signals.count,
        0, iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

// Captures only when publication must outlive this call. The bounded local
// scratch holds native-reader admission and resolved addresses, not arguments.
// Larger fan-in/binding tables use the same owned capture and packet writer.
static iree_status_t iree_hal_amd_gpu_pm4_submit_dispatch(
    iree_hal_amd_gpu_pm4_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_amd_gpu_pm4_dispatch_request_t* dispatch,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_profile_operation_t* profile) {
  enum { IREE_HAL_AMD_GPU_PM4_INLINE_SCRATCH_SIZE = 512 };
  const bool inline_storage =
      waits.count <= IREE_HAL_AMD_GPU_PM4_INLINE_SCRATCH_SIZE /
                         sizeof(iree_hal_amd_gpu_pm4_submission_wait_t) &&
      dispatch->bindings.count <=
          IREE_HAL_AMD_GPU_PM4_INLINE_SCRATCH_SIZE / sizeof(uint64_t) &&
      waits.count * sizeof(iree_hal_amd_gpu_pm4_submission_wait_t) +
              dispatch->bindings.count * sizeof(uint64_t) <=
          IREE_HAL_AMD_GPU_PM4_INLINE_SCRATCH_SIZE;
  const bool publishing =
      inline_storage &&
      iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher);
  uint64_t completed = 0;
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  iree_hal_amd_gpu_pm4_publication_result_t result =
      IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  iree_status_t status = iree_ok_status();
  iree_status_t dependency_failure = iree_ok_status();
  if (publishing) {
    completed = (uint64_t)iree_atomic_load(&queue->completed.count,
                                           iree_memory_order_acquire);
    const uint64_t position = (uint64_t)iree_atomic_load(
        &queue->producer.count, iree_memory_order_relaxed);
    const bool executes =
        dispatch->function &&
        (iree_hal_dispatch_uses_indirect_parameters(dispatch->flags) ||
         (dispatch->launch.geometry.workgroup_count[0] &&
          dispatch->launch.geometry.workgroup_count[1] &&
          dispatch->launch.geometry.workgroup_count[2]));
    const uint32_t scratch_required =
        executes ? dispatch->launch.scratch_wave_byte_length : 0;
    if (!queue->capacity.last &&
        position - completed < queue->options.retirement_capacity &&
        (!scratch_required ||
         (queue->producer.scratch.current &&
          queue->producer.scratch.current->wave_byte_length >=
              scratch_required))) {
      iree_alignas(8) uint8_t storage[IREE_HAL_AMD_GPU_PM4_INLINE_SCRATCH_SIZE];
      iree_hal_amd_gpu_pm4_submission_wait_t* admitted = (void*)storage;
      uint64_t* addresses = (uint64_t*)(admitted + waits.count);
      for (iree_host_size_t i = 0; i < waits.count; ++i) {
        admitted[i] = (iree_hal_amd_gpu_pm4_submission_wait_t){
            .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
      }
      const uint32_t word_budget =
          iree_min(queue->producer.ring.capacity / 2, 16384u);
      const uint32_t reserved_words =
          IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS + 96;
      uint32_t native_wait_count = 0;
      iree_host_size_t wait_index = 0;
      iree_hal_amd_gpu_pm4_wait_kind_t kind;
      status = iree_hal_amd_gpu_pm4_admit_waits(
          queue, waits.count, admitted,
          waits.count <= (word_budget - reserved_words) / 7, &native_wait_count,
          &wait_index, &kind);
      if (!iree_status_is_ok(status)) {
        // Capture the remaining join even when its first failed input was
        // observed inline. Failing the outputs here could precede an earlier
        // producer on the same timeline.
        dependency_failure = status;
        status = iree_ok_status();
        kind = IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION;
      }
      if (result != IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED &&
          kind == IREE_HAL_AMD_GPU_PM4_WAIT_READY) {
        const iree_hal_amd_gpu_pm4_dispatch_publication_t request = {
            .dispatch = dispatch,
            .waits = admitted,
            .wait_count = waits.count,
            .native_wait_count = native_wait_count,
            .signals = signals,
            .before = iree_hal_amd_gpu_pm4_barrier_prepare(
                &queue->options.barriers, barriers ? barriers->before : NULL),
            .after = iree_hal_amd_gpu_pm4_barrier_prepare(
                &queue->options.barriers, barriers ? barriers->after : NULL),
            .profile = profile,
        };
        if (queue->kernel) {
          queue->kernel->rollback.words = queue->producer.ring.reserved_words;
          queue->kernel->rollback.storage = queue->producer.arguments;
          queue->kernel->rollback.bytes = queue->producer.arguments->end;
        }
        iree_hal_amd_gpu_pm4_prepare_wait_frontier(queue, waits.count, admitted,
                                                   native_wait_count);
        result = iree_hal_amd_gpu_pm4_publish_dispatch(
            queue, position, scratch_required, &request, addresses, &submission,
            &status);
        if (submission && result != IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
          // KERNEL capacity can defer after the complete native piece is
          // prepared. It needs retirement ownership, not another argument copy.
          iree_hal_amd_gpu_profile_operation_defer(
              iree_hal_amd_gpu_pm4_submission_profile(submission));
        }
      }
      if (!submission) {
        for (iree_host_size_t i = 0; i < waits.count; ++i) {
          iree_hal_amd_gpu_signal_release(admitted[i].signal);
        }
      }
    }
  }
  if (!submission && iree_status_is_ok(status) &&
      result != IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
    status = iree_hal_amd_gpu_pm4_submission_create(
        &queue->captures, profile != NULL, waits, signals, dispatch->executable,
        dispatch->function, dispatch->launch, dispatch->parameters,
        dispatch->constants, dispatch->bindings, &queue->options.barriers,
        barriers, dispatch->flags, &submission);
    if (iree_status_is_ok(status)) {
      if (!iree_status_is_ok(dependency_failure)) {
        iree_hal_amd_gpu_context_report_status(queue->context,
                                               queue->base.queue_family->device,
                                               dependency_failure);
        iree_hal_amd_gpu_pm4_submission_reject(
            submission, iree_status_code(dependency_failure));
        iree_status_free(dependency_failure);
        dependency_failure = iree_ok_status();
      }
      if (profile) {
        *iree_hal_amd_gpu_pm4_submission_profile(submission) = *profile;
      }
      if (publishing) {
        submission->has_host_owner = true;
        iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                              iree_memory_order_relaxed);
        result = iree_hal_amd_gpu_pm4_process_entry(
            queue, &submission->entry.publication);
      } else {
        iree_hal_amd_gpu_pm4_submit(queue, submission);
      }
    }
  }
  if (publishing) {
    iree_hal_amd_gpu_publication_entry_t* remainder =
        iree_status_is_ok(status) &&
                result != IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED
            ? &submission->entry.publication
            : NULL;
    if (iree_hal_amd_gpu_pm4_handoff(
            queue, remainder,
            result == IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY, completed) ||
        iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) ||
        iree_hal_amd_gpu_pm4_feedback_failure(queue)) {
      iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                          queue->options.source_ordinal);
    }
    iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  }
  return iree_status_join(status, dependency_failure);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_dispatch(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_lookup_pm4_for_queue(
      executable, token, queue->options.queue_ordinal, &function));
  iree_hal_amd_gpu_pm4_dispatch_t dispatch;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_dispatch_prepare(
      function,
      (uint32_t)
          queue->context->target_info.compute.local_data_share_byte_length,
      queue->options.scratch_profile.maximum_wave_byte_length, config,
      constants, bindings.count, flags, &dispatch));
  if (flags & IREE_HAL_DISPATCH_FLAG_COOPERATIVE) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_executable_validate_cooperative_dispatch(
            executable, token, &dispatch.geometry, flags));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_dispatch_validate_direct_bindings(
      function->base.kernarg, config.workgroup_count_ref, bindings, flags));
  iree_hal_amd_gpu_profile_operation_t profile_storage;
  iree_hal_amd_gpu_profile_operation_t* operation = NULL;
  if (queue->profile) {
    operation = &profile_storage;
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, waits.count, signals.count,
        1, operation);
    if (iree_hal_amd_gpu_profile_captures_dispatch(
            queue->profile, queue->options.queue_ordinal, 0, 0,
            function->base.metadata->reflection_name) &&
        (iree_hal_dispatch_uses_indirect_parameters(flags) ||
         (dispatch.geometry.workgroup_count[0] &&
          dispatch.geometry.workgroup_count[1] &&
          dispatch.geometry.workgroup_count[2]))) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_profile_dispatch(
          executable, token, &dispatch.geometry, flags, &operation->dispatch));
      operation->completion.count = 1;
      operation->completion.sample_stride =
          iree_hal_amd_gpu_profile_timestamps_dispatches(queue->profile)
              ? sizeof(iree_hal_amd_gpu_profile_sample_t)
              : 0;
    }
  }
  const iree_hal_amd_gpu_pm4_dispatch_request_t request = {
      .executable = executable,
      .function = function,
      .launch = dispatch,
      .parameters = config.workgroup_count_ref,
      .constants = constants,
      .bindings = bindings,
      .flags = flags,
  };
  return iree_hal_amd_gpu_pm4_submit_dispatch(queue, waits, signals, &request,
                                              barriers, operation);
}

//===----------------------------------------------------------------------===//
// Cold native construction and joined teardown
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amd_gpu_pm4_prepare_native(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (queue->options.publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_kmq_storage_allocate(
        queue->context, queue->options.family_ordinal,
        queue->options.command_byte_capacity,
        queue->options.retirement_capacity, queue->options.memory,
        queue->host_allocator, &queue->kernel, &queue->producer.ring));
  } else {
    const amdf_gpu_queue_execution_resource_selection_t selection = {
        .type = AMDF_STRUCTURE_TYPE_GPU_QUEUE_EXECUTION_RESOURCE_SELECTION,
        .structure_size = sizeof(selection),
        .resource_count = (uint32_t)queue->base.execution_resources.count,
        .resource_ordinals = queue->base.execution_resources.ordinals};
    const amdf_gpu_user_queue_create_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(native),
        .next = selection.resource_count ? &selection : NULL,
        .queue_family_ordinal = queue->options.family_ordinal,
        .priority =
            iree_hal_amd_gpu_user_queue_native_priority(queue->base.priority),
        .placement = (amdf_queue_placement_t)queue->base.placement,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities =
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
            ((queue->base.features &
              IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)
                 ? AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH
                 : 0),
        .ring_byte_length = queue->options.command_byte_capacity,
    };
    amdf_user_queue_mapping_info_t mapping;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_user_queue_initialize(
        queue->context, &native, &queue->storage, &mapping, NULL));
    queue->base.placement = queue->storage.placement;
    iree_hal_amd_gpu_pm4_ring_initialize(
        (uint32_t*)(uintptr_t)mapping.ring_address,
        (uint32_t)(mapping.ring_byte_length / sizeof(uint32_t)),
        queue->storage.host_publish_executor, &queue->producer.ring);
    if (mapping.index_bits != 64 || mapping.doorbell_bits != 64 ||
        !mapping.write_index_address || !mapping.doorbell_address) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "PM4 USER publication requires the format-1 "
                              "64-bit index and doorbell mapping");
    }
    queue->producer.write_index =
        (iree_atomic_int64_t*)(uintptr_t)mapping.write_index_address;
    queue->producer.doorbell =
        (volatile uint64_t*)(uintptr_t)mapping.doorbell_address;
  }
  uint64_t control_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_backing_allocate(
      queue->context, queue->options.memory,
      (uint64_t)queue->options.retirement_capacity *
          sizeof(iree_hal_amd_gpu_pm4_retirement_record_t),
      64, &queue->control, &control_address));
  amdf_memory_access_info_t access = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(access),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->memory_query_access_info(queue->control.handle, 0,
                                                    &access),
      "memory_query_access_info"));
  const amdf_memory_site_t device_site = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(device_site),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device = {.memory = queue->control.handle,
                       .access_ordinal = 0,
                       .queue_family_ordinal = queue->options.family_ordinal},
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(queue->context->api->memory_query_pair_info(
                                        &device_site, &device_site, &pair),
                                    "memory_query_pair_info"));
  if (!iree_all_bits_set(access.flags, AMDF_MEMORY_FLAG_HOST_COHERENT |
                                           AMDF_MEMORY_FLAG_DEVICE_ADDRESS) ||
      !iree_all_bits_set(
          access.atomic_operations_32,
          AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE) ||
      pair.atomic_reach.scope_32 < AMDF_ATOMIC_SCOPE_DEVICE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 control backing on '%s' lacks coherent host access or qualified "
        "32-bit native signals: profile=%u, family=%u, access flags=0x%" PRIx64
        ", atomic operations=0x%" PRIx64 ", atomic scope=%u",
        queue->context->endpoint_info.name,
        queue->options.memory->profile.ordinal, queue->options.family_ordinal,
        access.flags, access.atomic_operations_32, pair.atomic_reach.scope_32);
  }
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(
      (iree_hal_amd_gpu_pm4_retirement_record_t*)queue->control.contents.data,
      control_address, queue->options.retirement_capacity, &queue->retirement);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_prepare_arguments(
      queue, queue->options.argument_capacity));
  return iree_hal_amd_gpu_pm4_prepare_signals(queue);
}

amdf_queue_publication_modes_t iree_hal_amd_gpu_pm4_queue_publication_modes(
    const amdf_queue_family_info_t* native) {
  const uint64_t required_cache = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
  if (native->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ||
      native->format_version != AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 ||
      !iree_all_bits_set(native->roles, AMDF_QUEUE_ROLE_COMPUTE |
                                            AMDF_QUEUE_ROLE_TRANSFER |
                                            AMDF_QUEUE_ROLE_ATOMIC |
                                            AMDF_QUEUE_ROLE_CACHE_CONTROL) ||
      !iree_any_bit_set(native->format_features,
                        AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) ||
      !iree_all_bits_set(native->cache_operations, required_cache) ||
      !iree_any_bit_set(native->cache_transition_kinds,
                        AMDF_CACHE_TRANSITION_KINDS_GLOBAL) ||
      !iree_all_bits_set(
          native->atomic_capabilities.operations_without_dispatch_32,
          AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT) ||
      !iree_any_bit_set(native->atomic_capabilities.wait_conditions_32,
                        AMDF_ATOMIC_WAIT_CONDITION_EQUAL)) {
    return 0;
  }
  amdf_queue_publication_modes_t modes =
      native->publication_modes &
      (AMDF_QUEUE_PUBLICATION_MODE_USER | AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  if (!iree_any_bit_set(native->user_queue_capabilities,
                        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
    modes &= ~AMDF_QUEUE_PUBLICATION_MODE_USER;
  }
  return modes;
}

bool iree_hal_amd_gpu_pm4_queue_host_is_supported(void) {
  return iree_cpu_architecture_host() == IREE_CPU_ARCHITECTURE_X86_64 &&
         iree_atomic_int32_is_lock_free() && iree_atomic_int64_is_lock_free();
}

iree_status_t iree_hal_amd_gpu_pm4_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_pm4_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  if (!iree_hal_amd_gpu_pm4_queue_host_is_supported()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 publication requires the qualified "
                            "x86-64 host store-fence and atomic contract");
  }
  if (options->publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER &&
      (context->event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
       context->event_info.format_version != 1 ||
       !iree_any_bit_set(context->event_info.publication_modes,
                         AMDF_QUEUE_PUBLICATION_MODE_USER))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 USER completion requires a qualified native event producer");
  }
  iree_host_size_t tracker_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_wait_tracker_storage_size(1, &tracker_size));
  const iree_host_size_t pending_size =
      iree_host_align(sizeof(iree_hal_amd_gpu_pm4_pending_t),
                      iree_max_align_t) +
      tracker_size;
  const iree_host_size_t minimum_capture_size =
      iree_max(pending_size, sizeof(iree_hal_amd_gpu_pm4_submission_t));
  if (!options->retirement_capacity ||
      (options->retirement_capacity & (options->retirement_capacity - 1)) ||
      options->argument_capacity < 8 ||
      (options->argument_capacity & (options->argument_capacity - 1)) ||
      options->command_byte_capacity < 2048 ||
      options->command_byte_capacity > UINT32_MAX ||
      (options->command_byte_capacity & (options->command_byte_capacity - 1))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid PM4 queue capacity or producer options");
  }
  if (options->capture_block_size < minimum_capture_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 capture blocks require at least %" PRIhsz
                            " bytes for submission and wait metadata",
                            minimum_capture_size);
  }
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_pm4_queue_t);
  const uint32_t queue_count = options->queue_count;
  if (!iree_host_size_checked_mul_add(
          total_size, options->retirement_capacity,
          sizeof(iree_hal_amd_gpu_pm4_retirement_slot_t), &total_size) ||
      !iree_host_size_checked_mul_add(total_size, 2 * queue_count,
                                      sizeof(uint64_t), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, params->execution_resources.count,
          sizeof(iree_hal_queue_execution_resource_ordinal_t), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 queue allocation overflows");
  }
  iree_hal_amd_gpu_pm4_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, iree_alignof(iree_hal_amd_gpu_pm4_queue_t), 0,
      (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->storage.context = context;
  queue->options = *options;
  iree_atomic_store(&queue->producer.count, options->initial_epoch,
                    iree_memory_order_relaxed);
  iree_atomic_store(&queue->completed.count, options->initial_epoch,
                    iree_memory_order_relaxed);
  queue->producer.scratch.profile = options->scratch_profile;
  queue->pending_size = pending_size;
  queue->trim.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_TRIM;
  queue->capacity.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_CAPACITY;
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_pm4_host_begin,
      .end = iree_hal_amd_gpu_pm4_host_end,
      .enter = iree_hal_amd_gpu_pm4_host_call_enter};
  queue->file_executor = (iree_hal_amd_gpu_file_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_pm4_host_begin,
      .end = iree_hal_amd_gpu_pm4_host_end,
      .staging = options->file_staging,
      .waits = &queue->wait_context,
      .queue_ordinal = options->queue_ordinal,
      .native_family_ordinal = options->family_ordinal,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(iree_hal_amd_gpu_pm4_barrier_t),
      .add_effects = iree_hal_amd_gpu_pm4_file_add_effects,
      .submit = iree_hal_amd_gpu_pm4_file_submit};
  queue->issued = (iree_hal_amd_gpu_pm4_retirement_slot_t*)(queue + 1);
  queue->producer.frontier.count = queue_count;
  queue->producer.frontier.published =
      (uint64_t*)(queue->issued + options->retirement_capacity);
  queue->producer.frontier.published[options->queue_ordinal] =
      options->initial_epoch;
  queue->producer.frontier.pending = queue->producer.frontier.published;
  queue->producer.frontier.scratch =
      queue->producer.frontier.published + queue_count;
  iree_hal_queue_params_t achieved = *params;
  iree_hal_queue_execution_resource_ordinal_t* ordinals =
      (iree_hal_queue_execution_resource_ordinal_t*)(queue->producer.frontier
                                                         .scratch +
                                                     queue_count);
  if (params->execution_resources.count) {
    memcpy(ordinals, params->execution_resources.ordinals,
           params->execution_resources.count * sizeof(*ordinals));
    achieved.execution_resources.ordinals = ordinals;
  }
  iree_hal_queue_initialize(family, &achieved,
                            &iree_hal_amd_gpu_pm4_queue_vtable, &queue->base);
  iree_hal_amd_gpu_publisher_initialize(&queue->publisher);
  iree_async_semaphore_wait_context_initialize(&queue->wait_context);
  iree_hal_amd_gpu_host_wait_set_initialize(&queue->wait_context,
                                            &queue->pending.waits);
  iree_slim_mutex_initialize(&queue->pending.completion_mutex);
  queue->growth = (iree_hal_amd_gpu_pm4_growth_context_t){
      .context = context,
      .memory = options->memory,
      .transfer_memory = options->transfer_memory,
      .scratch_memory = options->scratch_memory,
      .scratch_profile = &queue->producer.scratch.profile,
      .captures = &queue->captures,
      .publisher = &queue->publisher,
      .publication = options->publication,
      .queue_ordinal = options->queue_ordinal,
      .source_ordinal = options->source_ordinal,
      .queue_count = queue_count,
      .signal_capacity = options->retirement_capacity,
      .signal_cache = options->signals,
      .reuse.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS,
  };
  iree_hal_amd_gpu_capture_pool_initialize(options->capture_block_size,
                                           host_allocator, &queue->captures);
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_prepare_native(queue);
  }
  if (iree_status_is_ok(status) &&
      (params->features & IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)) {
    status = iree_hal_amd_gpu_grid_storage_initialize(
        context, options->scratch_memory, &queue->grid);
  }
  if (iree_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    const amdf_status_t cleanup =
        iree_hal_amd_gpu_pm4_queue_shutdown(&queue->base);
    if (amdf_status_is_ok(cleanup)) {
      iree_hal_queue_release(&queue->base);
    } else {
      iree_hal_amd_gpu_context_report(context, cleanup,
                                      "PM4 construction rollback");
      status = iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(cleanup, "PM4 construction rollback"));
    }
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Operation families requiring their own native admission
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amd_gpu_pm4_queue_execute(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_execute(
      &queue->captures, queue->profile != NULL, waits, signals, command_buffer,
      bindings, &queue->options.barriers, flags, &submission));
  if (submission->execute.program->required_features & ~base->features) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "recorded cooperative dispatch requires a cooperative queue");
  }
  const bool capture =
      queue->profile && iree_hal_amd_gpu_profile_captures_commands(
                            queue->profile, queue->options.queue_ordinal,
                            iree_hal_command_buffer_profile_id(command_buffer));
  if (capture && submission->execute.program->dispatch_count &&
      !submission->execute.program->profile) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dispatch capture requires command buffer profile metadata");
  }
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_t* operation =
        iree_hal_amd_gpu_pm4_submission_profile(submission);
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, waits.count, signals.count,
        iree_hal_amd_gpu_pm4_command_buffer_operation_count(command_buffer),
        operation);
    operation->event.command_buffer_id =
        iree_hal_command_buffer_profile_id(command_buffer);
    if (capture) {
      operation->commands = submission->execute.program->profile;
      operation->completion.count =
          iree_hal_amd_gpu_profile_commands_count(operation->commands);
      operation->completion.sample_stride =
          iree_hal_amd_gpu_profile_timestamps_dispatches(queue->profile)
              ? sizeof(iree_hal_amd_gpu_profile_sample_t)
              : 0;
    }
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_host_call(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t arguments[4], iree_hal_host_call_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  return iree_hal_amd_gpu_host_call_submit(&queue->host_executor, waits,
                                           signals, call, arguments, flags);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_map(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_map(queue->options.mappings, base,
                                    queue->options.queue_ordinal, waits,
                                    signals, mapping_count, mappings, flags);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_hal_amd_gpu_executable_query_dispatch_concurrency(
      executable, function, queue->execution_resources.count, params,
      out_concurrency);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_control(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_amd_gpu_control_t command,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_profile_queue_event_type_t event_type) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  if (command.kind == IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP &&
      !base->queue_family->spec->timestamp_frequency_hz) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "queue family has no timestamp clock rate");
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_control(
      &queue->captures, queue->profile != NULL, base->queue_family, waits,
      signals, target, offset, command, &queue->options.barriers, barriers,
      &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal, event_type, waits.count,
        signals.count, 1, iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_pm4_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_wait(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_WAIT);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_pm4_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_store(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_STORE);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_hal_amd_gpu_pm4_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_rmw(params), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_RMW);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_timestamp(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_hal_amd_gpu_pm4_queue_control(
      queue, waits, signals, target, offset,
      iree_hal_amd_gpu_control_timestamp(), barriers,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TIMESTAMP);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_flush(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  // Inline submissions publish immediately. Deferred work owns its
  // readiness notification; flushing an idle publisher has nothing to advance.
  return iree_hal_amd_gpu_pm4_check_acceptance(queue);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  // Allocation changes ownership, not payload visibility. Its native point
  // orders reuse without adding a global data-cache transition.
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, queue->profile != NULL, waits, signals, NULL, NULL,
      (iree_hal_amd_gpu_pm4_dispatch_t){0}, (iree_hal_buffer_ref_t){0},
      iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
      &queue->options.barriers, &barriers, 0, &submission));
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_alloca(
      queue->options.allocations, pool, count, requests, out_buffers,
      &operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
    return status;
  }
  submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION;
  submission->allocation.operation = operation;
  submission->allocation.queue = base;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_t* profile =
        iree_hal_amd_gpu_pm4_submission_profile(submission);
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA, waits.count, signals.count,
        count, profile);
    for (iree_host_size_t i = 0; i < count; ++i) {
      profile->event.payload_length +=
          iree_min(requests[i].allocation_size,
                   UINT64_MAX - profile->event.payload_length);
    }
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  // Allocation changes ownership, not payload visibility. Its native point
  // orders reuse without adding a global data-cache transition.
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, queue->profile != NULL, waits, signals, NULL, NULL,
      (iree_hal_amd_gpu_pm4_dispatch_t){0}, (iree_hal_buffer_ref_t){0},
      iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
      &queue->options.barriers, &barriers, 0, &submission));
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_gpu_allocation_capture_dealloca(
      queue->options.allocations, count, buffers, &operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
    return status;
  }
  submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION;
  submission->allocation.operation = operation;
  submission->allocation.queue = base;
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_t* profile =
        iree_hal_amd_gpu_pm4_submission_profile(submission);
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA, waits.count, signals.count,
        count, profile);
    for (iree_host_size_t i = 0; i < count; ++i) {
      profile->event.payload_length +=
          iree_min(iree_hal_buffer_byte_length(buffers[i]),
                   UINT64_MAX - profile->event.payload_length);
    }
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_transfer(
      &queue->captures, queue->profile != NULL, base->queue_family, waits,
      signals, count, operations, &queue->options.barriers, barriers,
      (iree_hal_amd_gpu_retirement_callback_t){0}, &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_pm4_submission_profile_transfer(
        submission, queue->profile, queue->options.queue_ordinal);
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

// File continuations already own dependency resolution and prepared barriers.
// Reuse ordinary bounded transfer publication without another HAL replay layer.
static iree_status_t iree_hal_amd_gpu_pm4_file_submit(
    iree_hal_queue_t* base,
    const iree_hal_amd_gpu_transfer_operation_t* operation, const void* before,
    const void* after, iree_hal_amd_gpu_retirement_callback_t callback) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_file(
      &queue->captures, operation, &queue->options.barriers, before, after,
      callback, &submission));
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_read(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* source,
    uint64_t source_offset, iree_hal_buffer_t* target,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const iree_hal_amd_gpu_pm4_barrier_t before =
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->before : NULL);
  const iree_hal_amd_gpu_pm4_barrier_t after =
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, source,
      IREE_HAL_MEMORY_ACCESS_READ, source_offset, target, target_offset, length,
      &before, &after);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_file_t* target,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const iree_hal_amd_gpu_pm4_barrier_t before =
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->before : NULL);
  const iree_hal_amd_gpu_pm4_barrier_t after =
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->after : NULL);
  return iree_hal_amd_gpu_file_transfer_submit(
      &queue->file_executor, waits, signals, target,
      IREE_HAL_MEMORY_ACCESS_WRITE, target_offset, source, source_offset,
      length, &before, &after);
}

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_queue_vtable = {
    .destroy = iree_hal_amd_gpu_pm4_queue_destroy,
    .barrier = iree_hal_amd_gpu_pm4_queue_barrier,
    .execute = iree_hal_amd_gpu_pm4_queue_execute,
    .host_call = iree_hal_amd_gpu_pm4_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_gpu_pm4_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_gpu_pm4_queue_dispatch,
    .atomic_wait = iree_hal_amd_gpu_pm4_queue_atomic_wait,
    .atomic_store = iree_hal_amd_gpu_pm4_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_gpu_pm4_queue_atomic_rmw,
    .timestamp = iree_hal_amd_gpu_pm4_queue_timestamp,
    .flush = iree_hal_amd_gpu_pm4_queue_flush,
    .alloca = iree_hal_amd_gpu_pm4_queue_alloca,
    .dealloca = iree_hal_amd_gpu_pm4_queue_dealloca,
    .map = iree_hal_amd_gpu_pm4_queue_map,
    .transfer = iree_hal_amd_gpu_pm4_queue_transfer,
    .read = iree_hal_amd_gpu_pm4_queue_read,
    .write = iree_hal_amd_gpu_pm4_queue_write,
};
