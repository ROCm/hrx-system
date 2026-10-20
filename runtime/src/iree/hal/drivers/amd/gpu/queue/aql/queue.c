// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/queue.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/growth.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/scratch.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/file_transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/host_call.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"
#include "iree/hal/drivers/amd/status.h"

typedef struct iree_hal_amd_gpu_aql_queue_t iree_hal_amd_gpu_aql_queue_t;

typedef enum iree_hal_amd_gpu_aql_wait_kind_e {
  IREE_HAL_AMD_GPU_AQL_WAIT_READY,
  IREE_HAL_AMD_GPU_AQL_WAIT_COMPLETION,
  IREE_HAL_AMD_GPU_AQL_WAIT_SUBMISSION,
} iree_hal_amd_gpu_aql_wait_kind_t;

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

// Retirement credit owns both the native point and the inline host capture.
// Firmware consumption alone never returns this storage.
typedef struct iree_hal_amd_gpu_aql_retirement_t {
  // Capture dominated by the native point, immutable after publication.
  iree_alignas(64) iree_hal_amd_gpu_aql_submission_t* submission;
  // Independent native reader/completion point for this chunk.
  iree_hal_amd_gpu_signal_t* signal;
  // Argument prefix retired with this native completion.
  uint64_t arguments;
  // Final chunk releases invocation payload and publishes public completion.
  bool terminal;
  union {
    // Ready direct calls retain resources without another pool access.
    iree_alignas(8) uint8_t capture[480];
    // Pooled transfers instead own a host copy before returning staging credit.
    iree_hal_amd_gpu_download_t download;
    // Recorded chunks own sample ranges independently of the shared capture.
    iree_hal_amd_gpu_profile_completion_t profile;
  };
} iree_hal_amd_gpu_aql_retirement_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_retirement_t) == 512,
              "AQL retirement has a power-of-two stride");

struct iree_hal_amd_gpu_aql_queue_t {
  // Refcounted HAL queue; family/device are borrowed.
  iree_hal_queue_t base;
  // Allocator owning this wrapper and cold captures.
  iree_allocator_t host_allocator;
  // Native device/address/event owner.
  iree_hal_amd_gpu_context_t* context;
  // Immutable construction policies.
  iree_hal_amd_gpu_aql_queue_options_t options;
  // Native queue and independently mapped host/device operands.
  iree_hal_amd_gpu_user_queue_t storage;
  // Group-owned completion facts, retained until final queue cleanup.
  struct {
    // Retained tracker or NULL before group assignment.
    iree_async_frontier_tracker_t* tracker;
    // Exact queue completion axis.
    iree_async_axis_t axis;
  } frontier;
  // Current publication owner; producers and the service executor share it.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_hal_amd_gpu_publisher_t publisher;
  // Deferred captures and typed acquisition result ownership.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Native boundary executor borrowed by accepted host callbacks.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Native barriers/transfers borrowed by the shared direct-I/O/SDMA pipeline.
  iree_hal_amd_gpu_file_executor_t file_executor;
  // Device-owned optional session, changed only with idle queues.
  iree_hal_amd_gpu_profile_t* profile;
  // Fixed sidecar extent computed at construction.
  iree_host_size_t pending_size;
  // Native backing acquisition, independent of execution publication.
  iree_hal_amd_gpu_aql_growth_context_t growth;
  // Demand-driven firmware private memory.
  iree_hal_amd_gpu_aql_scratch_t scratch;
  // Independent native reader ownership.
  iree_hal_amd_gpu_signal_pool_t signals;
  // Publisher-owned native storage and accepted causal prefix.
  struct {
    // Native slot credit and write-only packet/doorbell operands.
    iree_hal_amd_gpu_aql_ring_t ring;
    // Current argument generation.
    iree_hal_amd_gpu_argument_storage_t* arguments;
    // Replaced generations, newest first; visited only at explicit trim.
    iree_hal_amd_gpu_argument_storage_t* retired_arguments;
    // Native descriptor generations, visited only during trim/shutdown.
    iree_hal_amd_gpu_signal_storage_t* signal_storage;
    // Dense immutable-domain causal prefix, updated only at acceptance.
    uint64_t* frontier;
    // Frontier width captured at construction.
    uint32_t queue_count;
    // Accepted retirement count published before any terminal header is valid.
    iree_atomic_int64_t count;
  } producer;
  // Completion credit is isolated from ordinary publisher writes.
  struct {
    // Fully retired native prefix, including final resource release.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_atomic_int64_t count;
    // Retired argument prefix including alignment/wrap padding.
    iree_atomic_int64_t arguments;
    // Publisher enrolled for actual completion credit.
    iree_atomic_int32_t waiting;
  } completed;
  // Captures waiting for execution credit, outside the ready FIFO so native
  // scratch results cannot get stuck behind the work requiring that scratch.
  struct {
    // First parked capture, exclusively owned by the publisher.
    iree_hal_amd_gpu_publication_entry_t* first;
    // Last parked capture for constant-time append/requeue.
    iree_hal_amd_gpu_publication_entry_t* last;
    // Stable coalesced capacity-return entry.
    iree_hal_amd_gpu_aql_entry_t entry;
    // One while entry is linked or being consumed.
    iree_atomic_int32_t pending;
  } capacity;
  // Coalesced explicit trim request.
  struct {
    // Stable trim entry.
    iree_hal_amd_gpu_aql_entry_t entry;
    // One while entry is linked or being consumed.
    iree_atomic_int32_t pending;
  } trim;
  // Wait tracker associations used only on dependency deferral.
  iree_async_semaphore_wait_context_t wait_context;
  // Independent callback-return and accepted-host ownership.
  struct {
    // Deferred work plus callback borrowers, separate from native retirement.
    iree_atomic_int64_t host_owners;
    // Failure-cancellable dependency registrations.
    iree_hal_amd_gpu_host_wait_set_t waits;
    // Joins a callback's final count drop and shutdown notification.
    iree_slim_mutex_t completion_mutex;
  } pending;
  // Public acceptance has closed; previously accepted waits remain live.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t closed;
  // Sticky terminal failure; it never supplies resource-reclamation credit.
  iree_atomic_int32_t failure;
  // Completion owner has reported failure to all accepted native captures.
  bool failure_delivered;
  // Logical shader failure has been reported without stopping retirement.
  bool feedback_failure_delivered;
  // Host metadata parallel to the finite native execution credit.
  iree_hal_amd_gpu_aql_retirement_t* issued;
};

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

static void iree_hal_amd_gpu_aql_fail_submission(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
      false, 0);
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_amd_gpu_aql_submission_fail(submission, status);
  iree_hal_semaphore_list_fail(submission->signals, status);
}

static void iree_hal_amd_gpu_aql_abandon(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t status) {
  const bool host_owner = submission->has_host_owner;
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
      false, 0);
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    // A prior piece can still be copying into borrowed host memory. Its last
    // retirement reference, not this rejected suffix, reports completion.
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    int32_t expected = IREE_STATUS_OK;
    iree_atomic_compare_exchange_strong(
        &submission->transfer.failure, &expected, iree_status_code(status),
        iree_memory_order_relaxed, iree_memory_order_relaxed);
    iree_status_free(status);
  } else {
    iree_hal_amd_gpu_aql_fail_submission(queue, submission, status);
  }
  iree_hal_amd_gpu_aql_submission_release(submission);
  if (host_owner) {
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
}

static iree_status_t iree_hal_amd_gpu_aql_admit_waits(
    iree_hal_amd_gpu_aql_queue_t* queue, iree_host_size_t wait_count,
    iree_hal_amd_gpu_aql_wait_t* waits, bool native_eligible,
    uint32_t* out_native_wait_count, iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_aql_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_READY;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_failure(queue));
  uint32_t native_wait_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < wait_count && *out_kind == IREE_HAL_AMD_GPU_AQL_WAIT_READY &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_aql_wait_t* wait = &waits[i];
    if (native_eligible && iree_hal_amd_gpu_semaphore_isa(wait->semaphore)) {
      iree_hal_amd_gpu_semaphore_wait_t acquired;
      status = iree_hal_amd_gpu_semaphore_acquire_wait(
          wait->semaphore, wait->value, queue->context, &acquired);
      if (iree_status_is_ok(status)) {
        if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED) {
          *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_SUBMISSION;
        } else if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE) {
          wait->signal = acquired.signal;
          ++native_wait_count;
        } else if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST) {
          *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_COMPLETION;
        }
      }
    } else {
      uint64_t value = 0;
      status = iree_hal_semaphore_query(wait->semaphore, &value);
      if (iree_status_is_ok(status) && value < wait->value) {
        *out_kind = IREE_HAL_AMD_GPU_AQL_WAIT_COMPLETION;
      }
    }
    if (*out_kind != IREE_HAL_AMD_GPU_AQL_WAIT_READY) {
      *out_wait_index = i;
    }
  }
  if (*out_kind != IREE_HAL_AMD_GPU_AQL_WAIT_READY ||
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

// Wait completion reenters the publication owner only after registration and
// every cancellation borrower have relinquished the captured sidecar.
static void iree_hal_amd_gpu_aql_enqueue(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission);

static void iree_hal_amd_gpu_aql_pending_resume(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_aql_pending_t* pending = user_data;
  iree_hal_amd_gpu_aql_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_aql_submission_t* submission = pending->submission;
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_aql_pending_release(pending);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_enqueue(queue, submission);
  } else {
    iree_hal_amd_gpu_aql_abandon(queue, submission, status);
  }
  iree_hal_amd_gpu_aql_pending_end(queue);
}

static void iree_hal_amd_gpu_aql_pending_begin(
    iree_hal_amd_gpu_aql_pending_t* pending) {
  iree_hal_amd_gpu_aql_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_aql_submission_t* submission = pending->submission;
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
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_aql_pending_release(pending);
  iree_hal_amd_gpu_aql_abandon(queue, submission, status);
  iree_hal_amd_gpu_aql_pending_end(queue);
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

// Every fallible preparation precedes native reservation. Once reserved, all
// ownership is published before firmware can observe the first valid header.
typedef enum iree_hal_amd_gpu_aql_publish_result_e {
  IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS,
} iree_hal_amd_gpu_aql_publish_result_t;

static void iree_hal_amd_gpu_aql_write_packet(
    iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    iree_hal_amd_gpu_aql_packet_t packet) {
  iree_hal_amd_gpu_aql_ring_write_body(ring, position, &packet);
  iree_hal_amd_gpu_aql_ring_publish_header(ring, position, packet.words[0]);
}

static uint64_t iree_hal_amd_gpu_aql_argument_length(
    const iree_hal_amd_gpu_aql_dispatch_request_t* request, bool sampled) {
  if (!request->function) {
    return 0;
  }
  const uint64_t length =
      iree_hal_dispatch_uses_indirect_arguments(request->flags)
          ? 0
          : request->function->base.kernarg->byte_length;
  const uint64_t arguments =
      iree_hal_dispatch_uses_indirect_parameters(request->flags)
          ? iree_align_uint64(length, 8) +
                sizeof(iree_hal_amd_gpu_aql_indirect_arguments_t)
          : length;
  return sampled ? iree_align_uint64(arguments, 64) +
                       sizeof(iree_hal_amd_gpu_aql_signal_t)
                 : arguments;
}

// A reservation calculation changes no ring state. Completion prefixes include
// wrap/alignment padding; successful publication alone commits the end.
static iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_arguments(
    iree_hal_amd_gpu_aql_queue_t* queue, uint64_t length, uint64_t alignment,
    uint64_t* out_offset, uint64_t* out_end) {
  iree_hal_amd_gpu_argument_storage_t* storage = queue->producer.arguments;
  *out_offset = 0;
  *out_end = storage->end;
  if (!length) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  if (length > storage->capacity || alignment > storage->capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS;
  }
  uint64_t relative =
      iree_align_uint64(storage->end - storage->begin, alignment);
  uint64_t offset = relative & (storage->capacity - 1);
  if (offset + length > storage->capacity) {
    relative += storage->capacity - offset;
    offset = 0;
  }
  const uint64_t end = storage->begin + relative + length;
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  const uint64_t reusable = completed >= storage->end
                                ? storage->begin + relative
                                : iree_max(completed, storage->begin);
  if (end - reusable > storage->capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  *out_offset = offset;
  *out_end = end;
  return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
}

// Installs ownership before firmware can observe the terminal valid header.
// All fallible preparation and native reservation have already succeeded.
static void iree_hal_amd_gpu_aql_accept(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission,
    iree_hal_amd_gpu_signal_t* signal, uint64_t position, uint64_t argument_end,
    bool terminal, bool native_signals) {
  if (IREE_UNLIKELY(submission->profiled)) {
    iree_hal_amd_gpu_profile_operation_record(
        queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
        true, (submission->native_wait_count + 4) / 5);
  }
  queue->producer.frontier[queue->options.queue_ordinal] = position + 1;
  iree_hal_amd_gpu_signal_publish_frontier(signal, queue->producer.frontier);
  iree_hal_amd_gpu_aql_retirement_t* action =
      &queue->issued[position & (queue->options.retirement_capacity - 1)];
  action->submission = submission;
  action->signal = signal;
  action->arguments = argument_end;
  action->terminal = terminal;
  if (!terminal) {
    iree_atomic_ref_count_inc(submission->entry.kind ==
                                      IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE
                                  ? &submission->execute.ref_count
                                  : &submission->transfer.ref_count);
  } else {
    for (iree_host_size_t i = 0;
         native_signals && i < submission->signals.count; ++i) {
      if (iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i])) {
        iree_status_t status = iree_hal_amd_gpu_semaphore_publish_signal(
            submission->signals.semaphores[i],
            submission->signals.payload_values[i], signal);
        if (!iree_status_is_ok(status)) {
          iree_hal_amd_gpu_aql_fail_submission(queue, submission, status);
        }
      }
    }
    if (submission->has_host_owner) {
      submission->has_host_owner = false;
      iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                            iree_memory_order_release);
    }
  }
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
}

static uint64_t iree_hal_amd_gpu_aql_emit_waits(
    iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    iree_host_size_t wait_count, const iree_hal_amd_gpu_aql_wait_t* waits) {
  uint64_t dependencies[5];
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    if (waits[i].signal) {
      dependencies[count++] = waits[i].signal->device_address;
    }
    if (count == 5 || (i + 1 == wait_count && count)) {
      iree_hal_amd_gpu_aql_write_packet(
          ring, position++,
          iree_hal_amd_gpu_aql_packet_barrier(
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
              count, dependencies, 0));
      count = 0;
    }
  }
  return position;
}

static iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish(
    iree_hal_amd_gpu_aql_queue_t* queue,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, iree_hal_amd_gpu_aql_wait_t* waits,
    uint32_t native_wait_count, iree_hal_semaphore_list_t signals,
    uint32_t before, uint32_t after, uint64_t* addresses,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_aql_submission_t** inout_submission,
    iree_status_t* out_status) {
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  if (position - completed == queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  const bool indirect =
      iree_hal_dispatch_uses_indirect_parameters(request->flags);
  const bool dispatch =
      request->function && (indirect || (request->geometry.workgroup_count[0] &&
                                         request->geometry.workgroup_count[1] &&
                                         request->geometry.workgroup_count[2]));
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(request->flags);
  const iree_hal_amd_gpu_kernarg_layout_t* layout =
      dispatch ? request->function->base.kernarg : NULL;
  const uint64_t kernarg_length =
      dispatch && !indirect_arguments ? layout->byte_length : 0;
  const iree_hal_amd_gpu_profile_operation_t* operation =
      *inout_submission
          ? iree_hal_amd_gpu_aql_submission_profile(*inout_submission)
          : profile;
  const bool sampled = operation && operation->completion.sample_stride;
  const uint64_t argument_length =
      dispatch ? iree_hal_amd_gpu_aql_argument_length(request, sampled) : 0;
  const uint64_t argument_alignment =
      dispatch ? iree_max(layout->alignment, sampled    ? 64u
                                             : indirect ? 8u
                                                        : 1u)
               : 1;
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  uint64_t argument_offset = 0, argument_end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t argument_result =
      iree_hal_amd_gpu_aql_arguments(queue, argument_length, argument_alignment,
                                     &argument_offset, &argument_end);
  if (argument_result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return argument_result;
  }
  uint64_t argument_address =
      argument_length ? arguments->address + argument_offset : 0;
  const bool prefix = dispatch && (before & (3u << 11));
  const bool suffix = !dispatch || sampled || (after & (3u << 9));
  const uint32_t packet_count =
      (native_wait_count + 4) / 5 + prefix + indirect + dispatch + suffix;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < request->bindings.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        request->bindings.values[i],
        indirect_arguments ? IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS
                           : IREE_HAL_BUFFER_USAGE_STORAGE |
                                 IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
        indirect_arguments ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (dispatch && indirect_arguments && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (layout->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AQL argument address is misaligned");
    }
  }
  uint64_t counts_address = 0;
  if (indirect && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        request->parameters, IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        IREE_HAL_MEMORY_ACCESS_READ, &counts_address);
  }
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  iree_hal_amd_gpu_signal_t* signal =
      iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  iree_hal_amd_gpu_aql_retirement_t* action =
      &queue->issued[position & (queue->options.retirement_capacity - 1)];
  iree_hal_amd_gpu_aql_submission_t* submission = *inout_submission;
  if (!submission) {
    status = iree_hal_amd_gpu_aql_submission_retirement(
        &queue->captures,
        iree_make_byte_span(action->capture, sizeof(action->capture)), profile,
        request, wait_count, waits, native_wait_count, signals, &submission);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_signal_release(signal);
      *out_status = status;
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
    *inout_submission = submission;
  }
  if (kernarg_length) {
    iree_hal_amd_gpu_dispatch_emplace_arguments(
        &request->function->base, &request->geometry, request->constants,
        addresses, request->flags,
        arguments->memory.contents.data + argument_offset);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    action->download = (iree_hal_amd_gpu_download_t){0};
  }
  uint64_t sample_address = 0;
  if (sampled) {
    const uint64_t offset = argument_offset + argument_length -
                            sizeof(iree_hal_amd_gpu_aql_signal_t);
    iree_hal_amd_gpu_aql_signal_t* sample =
        (void*)(arguments->memory.contents.data + offset);
    *sample = (iree_hal_amd_gpu_aql_signal_t){.kind = 1, .value = 1};
    iree_hal_amd_gpu_aql_submission_profile(submission)->completion.samples =
        (const void*)&sample->start_timestamp;
    sample_address = arguments->address + offset;
  }
  // Prepare the terminal packet before a retirement slot can be consumed.
  // A ready request may borrow the same capture which native retirement frees.
  const uint32_t dispatch_header = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                                   IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                                   (after & (3u << 11));
  iree_hal_amd_gpu_aql_packet_t payload =
      dispatch ? iree_hal_amd_gpu_aql_dispatch_packet(
                     request->function, &request->geometry, dispatch_header,
                     argument_address,
                     sampled  ? sample_address
                     : suffix ? 0
                              : signal->device_address,
                     request->flags)
               : iree_hal_amd_gpu_aql_packet_barrier(
                     IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                         IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                         iree_hal_amd_gpu_aql_barrier_join(before, after),
                     0, NULL, signal->device_address);
  iree_hal_amd_gpu_aql_packet_t patch = {0};
  if (indirect) {
    const uint64_t patch_offset =
        argument_offset + iree_align_uint64(kernarg_length, 8);
    const uint64_t target_position =
        ring->reserved + (native_wait_count + 4) / 5 + prefix + 1;
    iree_hal_amd_gpu_aql_indirect_prepare(
        layout, &request->geometry, request->flags,
        iree_hal_amd_gpu_aql_ring_packet_address(ring, target_position),
        counts_address, argument_address, payload.words[0],
        (void*)(arguments->memory.contents.data + patch_offset));
    patch = iree_hal_amd_gpu_aql_indirect_packet(
        queue->options.programs->indirect, arguments->address + patch_offset);
  }
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, packet_count, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  arguments->end = argument_end;
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    if (waits[i].signal) {
      iree_hal_amd_gpu_signal_join_frontier(waits[i].signal,
                                            queue->producer.frontier);
    }
  }
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position, argument_end,
                              true, true);
  packet_position =
      iree_hal_amd_gpu_aql_emit_waits(ring, packet_position, wait_count, waits);
  if (prefix) {
    iree_hal_amd_gpu_aql_write_packet(
        ring, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | (before & (3u << 11)) |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
            0, NULL, 0));
  }
  if (indirect) {
    // The producer may run as soon as its header becomes valid, before the
    // doorbell. Complete the target body first; only the GPU publishes it.
    iree_hal_amd_gpu_aql_ring_write_body(ring, packet_position + 1, &payload);
    iree_hal_amd_gpu_aql_write_packet(ring, packet_position, patch);
    packet_position += 2;
  } else {
    iree_hal_amd_gpu_aql_write_packet(ring, packet_position++, payload);
  }
  if (dispatch && suffix) {
    iree_hal_amd_gpu_aql_write_packet(
        ring, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | (after & (3u << 9)) |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
            0, NULL, signal->device_address));
  }
  // No capture/signal/request access follows the terminal header publication.
  iree_hal_amd_gpu_aql_ring_notify(ring);
  return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
}

static uint64_t iree_hal_amd_gpu_aql_execute_argument_length(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation, bool sampled) {
  uint64_t length = invocation->block_ordinal
                        ? sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t)
                        : sizeof(iree_hal_amd_gpu_aql_program_arguments_t) +
                              (uint64_t)invocation->program->value_count * 8;
  if (invocation->program->profile) {
    length += sizeof(iree_hal_amd_gpu_aql_materialize_profile_arguments_t) -
              sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t);
  }
  return sampled ? iree_align_uint64(length, 64) +
                       (uint64_t)invocation->program
                               ->blocks[invocation->block_ordinal]
                               .dispatches.count *
                           sizeof(iree_hal_amd_gpu_aql_signal_t)
                 : length;
}

static iree_hal_amd_gpu_aql_publish_result_t
iree_hal_amd_gpu_aql_publish_execute(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  iree_hal_amd_gpu_aql_command_invocation_t* invocation = &submission->execute;
  iree_hal_amd_gpu_program_t* program = &invocation->program->resident;
  const bool first = !invocation->block_ordinal;
  const bool terminal =
      invocation->block_ordinal + 1 == invocation->program->block_count;
  const iree_hal_amd_gpu_profile_operation_t* operation =
      iree_hal_amd_gpu_aql_submission_profile(submission);
  const bool sampled = operation && operation->completion.sample_stride;
  const uint64_t argument_length =
      iree_hal_amd_gpu_aql_execute_argument_length(invocation, sampled);
  uint64_t argument_offset = 0, argument_end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t argument_result =
      iree_hal_amd_gpu_aql_arguments(queue, argument_length, sampled ? 64 : 8,
                                     &argument_offset, &argument_end);
  if (argument_result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return argument_result;
  }
  // Admission candidates include only waits this chunk will emit. Failed
  // preparation must not add another queue's causal proof to the FIFO prefix.
  uint64_t* frontier = queue->producer.frontier;
  if (first && submission->native_wait_count) {
    frontier += queue->producer.queue_count;
    memcpy(frontier, queue->producer.frontier,
           queue->producer.queue_count * sizeof(uint64_t));
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(submission->waits[i].signal,
                                              frontier);
      }
    }
  }
  if (!invocation->instance) {
    invocation->instance = iree_hal_amd_gpu_program_try_acquire(
        program, queue->options.queue_ordinal, frontier);
    if (!invocation->instance) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM;
    }
  }
  const iree_hal_amd_gpu_aql_program_block_t block =
      invocation->program->blocks[invocation->block_ordinal];
  const uint32_t waits = first ? (submission->native_wait_count + 4) / 5 : 0;
  const bool initialize = first && !invocation->instance->initialized;
  const uint32_t prefix = iree_hal_amd_gpu_aql_program_prefix_count(
      program, invocation->instance, first);
  const uint32_t packet_count = waits + prefix + block.packet_count + 1;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  void* upload = arguments->memory.contents.data + argument_offset;
  if (first) {
    *out_status = iree_hal_amd_gpu_aql_command_invocation_write(
        invocation, queue->options.queue_ordinal,
        (uint64_t*)((uint8_t*)upload +
                    sizeof(iree_hal_amd_gpu_aql_program_arguments_t) +
                    (invocation->program->profile
                         ? sizeof(
                               iree_hal_amd_gpu_aql_materialize_profile_arguments_t) -
                               sizeof(
                                   iree_hal_amd_gpu_aql_materialize_arguments_t)
                         : 0)));
    if (!iree_status_is_ok(*out_status)) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
  }
  iree_hal_amd_gpu_signal_t* signal =
      iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, packet_count, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  if (frontier != queue->producer.frontier) {
    memcpy(queue->producer.frontier, frontier,
           queue->producer.queue_count * sizeof(uint64_t));
  }
  arguments->end = argument_end;
  if (terminal) {
    // Install the reusable writer coordinate before native completion.
    invocation->published = true;
    iree_hal_amd_gpu_program_publish(program, invocation->instance,
                                     queue->options.queue_ordinal,
                                     position + 1);
  }
  uint64_t sample_address = 0;
  if (operation) {
    iree_hal_amd_gpu_profile_completion_t* completion =
        &queue->issued[position & (queue->options.retirement_capacity - 1)]
             .profile;
    *completion = (iree_hal_amd_gpu_profile_completion_t){
        .first = block.dispatches.first,
        .count = operation->commands ? block.dispatches.count : 0,
        .sample_stride = operation->completion.sample_stride,
    };
    if (sampled && completion->count) {
      const uint64_t offset =
          argument_offset + argument_length -
          (uint64_t)completion->count * sizeof(iree_hal_amd_gpu_aql_signal_t);
      completion->samples =
          (const void*)(arguments->memory.contents.data + offset +
                        offsetof(iree_hal_amd_gpu_aql_signal_t,
                                 start_timestamp));
      sample_address = arguments->address + offset;
    }
  }
  const iree_hal_amd_gpu_aql_program_flags_t program_flags =
      (first ? IREE_HAL_AMD_GPU_AQL_PROGRAM_FIRST : 0) |
      (initialize ? IREE_HAL_AMD_GPU_AQL_PROGRAM_INITIALIZE : 0) |
      (invocation->program->profile ? IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE : 0);
  ++invocation->block_ordinal;
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position, argument_end,
                              terminal, true);
  packet_position = iree_hal_amd_gpu_aql_emit_waits(
      ring, packet_position, first ? submission->wait_count : 0,
      submission->waits);
  // publish() changes initialization state, so native prelude emission must
  // consume the pre-publication facts captured above.
  iree_hal_amd_gpu_aql_program_emit(queue->options.programs, program,
                                    invocation->instance, block.offset,
                                    block.packet_count, ring, packet_position,
                                    arguments->address + argument_offset,
                                    sample_address, upload, program_flags);
  packet_position += prefix + block.packet_count;
  iree_hal_amd_gpu_aql_write_packet(
      ring, packet_position,
      iree_hal_amd_gpu_aql_packet_barrier(
          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
          0, NULL, signal->device_address));
  // No invocation access follows the terminal header: it may retire now.
  iree_hal_amd_gpu_aql_ring_notify(ring);
  return terminal ? IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE
                  : IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE;
}

static uint64_t iree_hal_amd_gpu_aql_transfer_argument_length(
    const iree_hal_amd_gpu_transfer_t* transfer) {
  return IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS *
             sizeof(iree_hal_amd_gpu_blit_arguments_t) +
         transfer->staging_capacity;
}

static iree_hal_amd_gpu_aql_publish_result_t
iree_hal_amd_gpu_aql_publish_transfer(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  if (!submission->transfer.capture.count) {
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {0};
    return iree_hal_amd_gpu_aql_publish(
        queue, &request, submission->wait_count, submission->waits,
        submission->native_wait_count, submission->signals, submission->before,
        submission->after, NULL, NULL, &submission, out_status);
  }
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  iree_hal_amd_gpu_transfer_t* transfer = &submission->transfer.capture;
  const bool first = !transfer->cursor.index && !transfer->cursor.offset;
  const uint64_t length =
      iree_hal_amd_gpu_aql_transfer_argument_length(transfer);
  uint64_t offset = 0, end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t prepared =
      iree_hal_amd_gpu_aql_arguments(queue, length, 8, &offset, &end);
  if (prepared != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return prepared;
  }
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  const uint32_t waits = first ? (submission->native_wait_count + 4) / 5 : 0;
  const uint32_t capacity =
      iree_min(iree_min(IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS,
                        transfer->count - transfer->cursor.index),
               ring->capacity / 2 - waits - 2);
  const uint32_t before = first ? submission->before : 0;
  const bool prefix = (before & (3u << 11)) != 0;
  // Reserve enough credit for the bounded candidate before touching staging.
  const uint32_t maximum = waits + prefix + capacity + 1;
  if (ring->reserved - ring->consumed + maximum > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + maximum > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
  *out_status = iree_hal_amd_gpu_aql_transfer_prepare_chunk(
      transfer, queue->options.programs->blit, capacity,
      iree_make_byte_span(arguments->memory.contents.data + offset, length),
      arguments->address + offset, &chunk);
  if (!iree_status_is_ok(*out_status)) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  iree_hal_amd_gpu_signal_t* signal =
      iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  const bool terminal = chunk.next.index == transfer->count;
  const uint32_t after = iree_hal_amd_gpu_aql_barrier_join(
      terminal ? submission->after : 0,
      chunk.download.length ? IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM : 0);
  // All host arguments precede the first acquire. Sibling transfers have no
  // implicit dependencies; the ordered terminal packet joins the entire group.
  chunk.packets[0].words[0] = chunk.packets[0].words[0] |
                              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                              IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
  const iree_hal_amd_gpu_aql_packet_t completion =
      iree_hal_amd_gpu_aql_packet_barrier(
          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | after,
          0, NULL, signal->device_address);
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, waits + prefix + chunk.packet_count + 1, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  arguments->end = end - length + chunk.argument_length;
  if (first) {
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(submission->waits[i].signal,
                                              queue->producer.frontier);
      }
    }
  }
  transfer->cursor = chunk.next;
  queue->issued[position & (queue->options.retirement_capacity - 1)].download =
      chunk.download;
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position,
                              arguments->end, terminal,
                              !transfer->host_completion_required);
  packet_position = iree_hal_amd_gpu_aql_emit_waits(
      ring, packet_position, first ? submission->wait_count : 0,
      submission->waits);
  if (prefix) {
    iree_hal_amd_gpu_aql_write_packet(
        ring, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (before & (3u << 11)),
            0, NULL, 0));
  }
  for (uint32_t i = 0; i < chunk.packet_count; ++i) {
    iree_hal_amd_gpu_aql_write_packet(ring, packet_position++,
                                      chunk.packets[i]);
  }
  iree_hal_amd_gpu_aql_write_packet(ring, packet_position, completion);
  iree_hal_amd_gpu_aql_ring_notify(ring);
  return terminal ? IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE
                  : IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE;
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
    status = iree_hal_amd_gpu_aql_admit_waits(
        queue, submission->wait_count, submission->waits,
        !(submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
          submission->transfer.capture.host_wait_required) &&
            submission->wait_count <=
                (iree_min(queue->producer.ring.capacity / 2, 64u) - 4) * 5,
        &submission->native_wait_count, &wait_index, &kind);
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
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  const bool execute =
      submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE;
  const iree_hal_amd_gpu_aql_dispatch_request_t* request =
      &submission->dispatch;
  iree_hal_amd_gpu_aql_publish_result_t result;
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    result = iree_hal_amd_gpu_aql_publish_transfer(queue, submission, &status);
  } else if (execute) {
    result = iree_hal_amd_gpu_aql_publish_execute(queue, submission, &status);
  } else {
    uint64_t* addresses =
        (uint64_t*)(request->bindings.values + request->bindings.count +
                    iree_hal_dispatch_uses_indirect_parameters(request->flags));
    result = iree_hal_amd_gpu_aql_publish(
        queue, request, submission->wait_count, submission->waits,
        submission->native_wait_count, submission->signals, submission->before,
        submission->after, addresses, NULL, &submission, &status);
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
                    &submission->transfer.capture)
          : execute
              ? iree_hal_amd_gpu_aql_execute_argument_length(
                    &submission->execute, sampled)
              : iree_max(iree_hal_amd_gpu_aql_argument_length(request, sampled),
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

static void iree_hal_amd_gpu_aql_fail_queue(iree_hal_amd_gpu_aql_queue_t* queue,
                                            iree_status_t status) {
  int32_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &queue->failure, &expected, iree_status_code(status),
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_hal_amd_gpu_context_report_status(
        queue->context, queue->base.queue_family->device, status);
    iree_hal_amd_gpu_host_wait_set_cancel(&queue->pending.waits);
    iree_hal_amd_gpu_aql_capacity_available(queue);
    const amdf_status_t event_status =
        iree_hal_amd_gpu_context_signal(queue->context);
    if (!amdf_status_is_ok(event_status)) {
      iree_hal_amd_gpu_context_report(queue->context, event_status,
                                      "AQL failure wake");
    }
  }
  iree_status_free(status);
}

iree_status_t iree_hal_amd_gpu_aql_queue_set_profile(
    iree_hal_queue_t* base, iree_hal_amd_gpu_profile_t* profile) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
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
  iree_hal_amd_gpu_signal_storage_t** signal_link =
      &queue->producer.signal_storage;
  while (*signal_link && (*signal_link)->next) {
    iree_hal_amd_gpu_signal_storage_t* storage = *signal_link;
    if (iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block)) {
      *signal_link = storage->next;
      iree_hal_amd_gpu_signal_storage_free(queue->context, storage);
    } else {
      signal_link = &storage->next;
    }
  }
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
}

static void iree_hal_amd_gpu_aql_process_entry(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  iree_hal_amd_gpu_aql_entry_t* typed =
      iree_containerof(entry, iree_hal_amd_gpu_aql_entry_t, publication);
  switch (typed->kind) {
    case IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER:
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
        result.signals->next = queue->producer.signal_storage;
        queue->producer.signal_storage = result.signals;
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
      iree_status_t status =
          iree_hal_amd_gpu_aql_scratch_resume(&queue->scratch);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_aql_fail_queue(queue, status);
      }
      break;
    }
    case IREE_HAL_AMD_GPU_AQL_ENTRY_TRIM:
      iree_atomic_store(&queue->trim.pending, 0, iree_memory_order_release);
      iree_hal_amd_gpu_aql_trim_storage(queue);
      break;
  }
}

static void iree_hal_amd_gpu_aql_enqueue(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission) {
  if (iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                         &submission->entry.publication)) {
    iree_hal_amd_gpu_publication_entry_t* entry =
        iree_hal_amd_gpu_publisher_next(&queue->publisher);
    if (entry) {
      iree_hal_amd_gpu_aql_process_entry(queue, entry);
    }
    if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
      iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                          queue->options.queue_ordinal);
    }
    iree_hal_amd_gpu_aql_wake_terminal(queue);
  }
}

static bool iree_hal_amd_gpu_aql_progress(void* user_data) {
  iree_hal_amd_gpu_aql_queue_t* queue = user_data;
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
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE ||
           typed->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER)) {
        iree_hal_amd_gpu_profile_operation_defer(
            iree_hal_amd_gpu_aql_submission_profile(iree_containerof(
                typed, iree_hal_amd_gpu_aql_submission_t, entry)));
      }
      iree_hal_amd_gpu_aql_process_entry(queue, entry);
    }
    more = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL);
    iree_hal_amd_gpu_aql_wake_terminal(queue);
  }
  return more;
}

static bool iree_hal_amd_gpu_aql_retire(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_wait_t* out_wait) {
  iree_hal_amd_gpu_aql_queue_t* queue = user_data;
  uint64_t position = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                 iree_memory_order_relaxed);
  const uint64_t published = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_acquire);
  if (amdf_status_is_ok(wait_status) && position != published) {
    wait_status = iree_hal_amd_gpu_aql_scratch_poll(&queue->scratch);
  }
  if (!amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_aql_fail_queue(
        queue,
        IREE_HAL_AMD_STATUS_FROM_AMDF(wait_status, "AQL completion/scratch"));
  }
  const iree_status_code_t failure = (iree_status_code_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  if (failure != IREE_STATUS_OK) {
    if (!queue->failure_delivered &&
        iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      // No further acceptance can add a native user after the publisher joins.
      const uint64_t end = (uint64_t)iree_atomic_load(
          &queue->producer.count, iree_memory_order_acquire);
      for (uint64_t i = position; i < end; ++i) {
        iree_hal_amd_gpu_aql_fail_submission(
            queue,
            queue->issued[i & (queue->options.retirement_capacity - 1)]
                .submission,
            iree_make_status(failure, "AQL queue failed"));
      }
      if (end && queue->frontier.tracker) {
        iree_async_frontier_tracker_fail_axis(
            queue->frontier.tracker, queue->frontier.axis,
            iree_make_status(failure, "AQL queue failed"));
      }
      queue->failure_delivered = true;
    }
    return false;
  }
  const iree_status_code_t feedback_failure =
      iree_hal_amd_gpu_aql_feedback_failure(queue);
  if (feedback_failure != IREE_STATUS_OK &&
      !queue->feedback_failure_delivered) {
    // This consumer owns published captures until native completion, even
    // after their public timelines fail from a shader report.
    for (uint64_t i = position; i < published; ++i) {
      iree_hal_amd_gpu_aql_fail_submission(
          queue,
          queue->issued[i & (queue->options.retirement_capacity - 1)]
              .submission,
          iree_make_status(feedback_failure, "GPU sanitizer failure"));
    }
    if (published && queue->frontier.tracker) {
      iree_async_frontier_tracker_fail_axis(
          queue->frontier.tracker, queue->frontier.axis,
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
      out_wait->address = (iree_atomic_int32_t*)action->signal->host_value;
      out_wait->value = 1;
      break;
    }
    iree_status_code_t code;
    if (!iree_hal_amd_gpu_aql_submission_retirement_ready(submission, &code)) {
      break;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_aql_retire");
    if (queue->options.feedback) {
      // Native completion makes this work's reports observable before its
      // public completion can wake a host consumer.
      iree_hal_amd_gpu_feedback_drain(queue->options.feedback);
      if (code == IREE_STATUS_OK) {
        code = iree_hal_amd_gpu_aql_feedback_failure(queue);
      }
    }
    const uint64_t arguments = action->arguments;
    if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
        code == IREE_STATUS_OK && action->download.length) {
      memcpy(action->download.target, action->download.source,
             action->download.length);
    }
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
      iree_hal_amd_gpu_aql_fail_submission(queue, submission, status);
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
    uint32_t native_count = 0;
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    status = iree_hal_amd_gpu_aql_admit_waits(
        queue, waits.count, admitted, true, &native_count, &wait_index, &kind);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_profile_operation_record(queue->profile, profile, false,
                                                0);
      iree_hal_amd_gpu_context_report_status(
          queue->context, queue->base.queue_family->device, status);
      iree_hal_semaphore_list_fail(signals, status);
      status = iree_ok_status();
      finished = true;
    } else if (kind == IREE_HAL_AMD_GPU_AQL_WAIT_READY) {
      finished = iree_hal_amd_gpu_aql_publish(
                     queue, request, waits.count, admitted, native_count,
                     signals, before, after, addresses, profile, &submission,
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
      submission->has_host_owner = true;
      iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                            iree_memory_order_relaxed);
      if (publishing) {
        iree_hal_amd_gpu_aql_process_submission(queue, submission);
      } else {
        iree_hal_amd_gpu_aql_enqueue(queue, submission);
      }
    }
  }
  if (publishing) {
    if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
      iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                          queue->options.queue_ordinal);
    }
    iree_hal_amd_gpu_aql_wake_terminal(queue);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_aql_host_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t operation_count,
    iree_host_size_t public_signal_count) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_profile_operation_t operation;
  const bool profiled =
      queue->profile && type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  if (profiled) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal, type, waits.count,
        public_signal_count, operation_count, &operation);
  }
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {0};
  return iree_hal_amd_gpu_aql_submit_dispatch(
      queue, waits, signals, &request, barriers, profiled ? &operation : NULL);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_amd_gpu_aql_host_barrier(
      base, waits, signals, barriers, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
      0, signals.count);
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
    if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile) &&
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

void iree_hal_amd_gpu_aql_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_aql_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_aql_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  const bool idle =
      !iree_atomic_load(&queue->pending.host_owners,
                        iree_memory_order_acquire) &&
      !iree_atomic_load(&queue->growth.pending_count,
                        iree_memory_order_acquire) &&
      !iree_hal_amd_gpu_aql_scratch_is_pending(&queue->scratch) &&
      iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
      iree_atomic_load(&queue->producer.count, iree_memory_order_acquire) ==
          iree_atomic_load(&queue->completed.count, iree_memory_order_acquire);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
  return idle;
}

void iree_hal_amd_gpu_aql_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (!iree_atomic_exchange(&queue->trim.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_growth_enqueue(&queue->growth, &queue->trim.entry);
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
      iree_hal_amd_gpu_aql_fail_queue(
          queue, IREE_HAL_AMD_STATUS_FROM_AMDF(status, "AQL scratch trim"));
    }
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.queue_ordinal);
  }
  iree_hal_amd_gpu_aql_wake_terminal(queue);
  return status;
}

amdf_status_t iree_hal_amd_gpu_aql_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_aql_queue_is_idle(base),
              "native shutdown requires joined accepted work");
  amdf_status_t status = iree_hal_amd_gpu_user_queue_deinitialize(
      &queue->storage, queue->producer.ring.reserved);
  if (!amdf_status_is_ok(status)) {
    queue->context->cleanup_blocked = true;
    return status;
  }
  if (queue->scratch.growth) {
    iree_hal_amd_gpu_aql_scratch_deinitialize(&queue->scratch);
  }
  while (queue->producer.signal_storage) {
    iree_hal_amd_gpu_signal_storage_t* storage = queue->producer.signal_storage;
    queue->producer.signal_storage = storage->next;
    iree_hal_amd_gpu_signal_storage_free(queue->context, storage);
  }
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  if (arguments) {
    arguments->next = queue->producer.retired_arguments;
  } else {
    arguments = queue->producer.retired_arguments;
  }
  queue->producer.arguments = queue->producer.retired_arguments = NULL;
  while (arguments) {
    iree_hal_amd_gpu_argument_storage_t* next = arguments->next;
    iree_hal_amd_gpu_argument_storage_free(queue->context, arguments);
    arguments = next;
  }
  return AMDF_STATUS_OK;
}

static void iree_hal_amd_gpu_aql_clear_frontier(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  if (queue->frontier.tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier.tracker, queue->frontier.axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->frontier.tracker);
  }
  queue->frontier.tracker = NULL;
  queue->frontier.axis = 0;
}

iree_status_t iree_hal_amd_gpu_aql_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (tracker) {
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_tracker_register_axis(tracker, axis, NULL));
    iree_async_frontier_tracker_retain(tracker);
  }
  iree_hal_amd_gpu_aql_clear_frontier(queue);
  queue->frontier.tracker = tracker;
  queue->frontier.axis = axis;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_aql_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_ASSERT(!queue->storage.queue,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_hal_amd_gpu_signal_pool_deinitialize(&queue->signals);
  iree_hal_amd_gpu_host_wait_set_deinitialize(&queue->pending.waits);
  iree_slim_mutex_deinitialize(&queue->pending.completion_mutex);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_hal_amd_gpu_capture_pool_deinitialize(&queue->captures);
  iree_hal_amd_gpu_aql_clear_frontier(queue);
  iree_allocator_free_aligned(queue->host_allocator, queue);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AQL dispatch residency calculation is not implemented");
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL public atomic wait is not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL public atomic store is not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL public atomic RMW is not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_queue_timestamp(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL public timestamp is not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_queue_flush(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                      queue->options.queue_ordinal);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_alloca(
      queue->options.allocations, queue->options.queue_ordinal, waits, signals,
      pool, count, requests, out_buffers);
}

static iree_status_t iree_hal_amd_gpu_aql_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_dealloca(queue->options.allocations,
                                         queue->options.queue_ordinal, waits,
                                         signals, count, buffers);
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
  if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile) &&
      program->dispatch_count && !program->profile) {
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
    if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile)) {
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
  iree_hal_amd_gpu_aql_enqueue(queue, submission);
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
  iree_hal_amd_gpu_aql_enqueue(queue, submission);
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
    iree_hal_queue_t* base, const iree_hal_transfer_operation_t* operation,
    const void* before, const void* after,
    iree_hal_amd_gpu_retirement_callback_t callback) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_check_acceptance(queue));
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_transfer(
      &queue->captures, NULL, base->queue_family,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      operation ? 1 : 0, operation, before ? *(const uint32_t*)before : 0,
      after ? *(const uint32_t*)after : 0, &submission));
  submission->transfer.callback = callback;
  submission->has_host_owner = true;
  iree_atomic_fetch_add(&queue->pending.host_owners, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_enqueue(queue, submission);
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
  if (native_family->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ||
      !(native_family->publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ||
      native_family->format_version != AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1 ||
      !(native_family->format_features &
        AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE) ||
      !iree_all_bits_set(native_family->user_queue_capabilities,
                         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                             AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) ||
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
  const uint32_t queue_count = family->spec->provisioned_queue_count;
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
  queue->pending_size = pending_size;
  queue->issued = (void*)(queue + 1);
  queue->producer.queue_count = queue_count;
  queue->producer.frontier =
      (void*)(queue->issued + options->retirement_capacity);
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
      .queue_count = queue_count,
      .signal_capacity = options->retirement_capacity,
      .signal_pool = &queue->signals,
      .reuse.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_SIGNALS};
  iree_hal_amd_gpu_signal_pool_initialize(
      (iree_hal_amd_gpu_signal_pool_notification_t){
          .fn = iree_hal_amd_gpu_aql_growth_signals_available,
          .user_data = &queue->growth},
      &queue->signals);
  queue->capacity.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_CAPACITY;
  queue->trim.entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_TRIM;
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .barrier = iree_hal_amd_gpu_aql_host_barrier};
  queue->file_executor = (iree_hal_amd_gpu_file_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .staging = options->file_staging,
      .waits = &queue->wait_context,
      .queue_ordinal = options->queue_ordinal,
      .native_family_ordinal = options->family_ordinal,
      .barrier_size = sizeof(uint32_t),
      .add_effects = iree_hal_amd_gpu_aql_file_add_effects,
      .submit = iree_hal_amd_gpu_aql_file_submit};
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  const amdf_gpu_aql_scratch_service_info_t service = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_SERVICE_INFO,
      .structure_size = sizeof(service),
      .event = context->event};
  const amdf_gpu_user_queue_create_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(native),
      .next = &service,
      .queue_family_ordinal = options->family_ordinal,
      .priority = AMDF_QUEUE_PRIORITY_NORMAL,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                               AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
      .ring_byte_length = options->command_byte_capacity};
  amdf_user_queue_mapping_info_t host_mapping;
  amdf_user_queue_mapping_info_t device_mapping;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_user_queue_initialize(
        context, &native, &queue->storage, &host_mapping, &device_mapping);
  }
  if (iree_status_is_ok(status)) {
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
          &queue->producer.ring);
      iree_hal_amd_gpu_aql_scratch_initialize(
          &queue->growth, queue->storage.queue, options->scratch_memory,
          &queue->scratch);
      status = iree_hal_amd_gpu_argument_storage_allocate(
          context, options->memory, options->argument_capacity, 0,
          host_allocator, &queue->producer.arguments);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_aql_signal_storage_allocate(
        context, &queue->signals, options->memory, options->retirement_capacity,
        queue_count, host_allocator, &queue->producer.signal_storage);
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
  return iree_min(1024u, queue->producer.ring.capacity / 2);
}
