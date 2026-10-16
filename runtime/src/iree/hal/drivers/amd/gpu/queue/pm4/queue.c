// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/continuation.h"
#include "iree/hal/drivers/amd/gpu/queue/host_call.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/file_transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"
#include "iree/hal/drivers/amd/status.h"

typedef struct iree_hal_amd_gpu_pm4_queue_t iree_hal_amd_gpu_pm4_queue_t;

typedef enum iree_hal_amd_gpu_pm4_wait_kind_e {
  IREE_HAL_AMD_GPU_PM4_WAIT_READY,
  IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION,
  IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION,
} iree_hal_amd_gpu_pm4_wait_kind_t;

// Reusable admission sidecar. Its list links and tracker acceptance are owned
// by service, while tracker callbacks only enqueue finalization there.
typedef struct iree_hal_amd_gpu_pm4_pending_t {
  // Entry used to begin admission on service before tracker registration.
  iree_hal_memory_maintenance_entry_t entry;
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
  // Next registered wait, accessed only by service.
  struct iree_hal_amd_gpu_pm4_pending_t* next;
  // Previous registered wait, or NULL at the head.
  struct iree_hal_amd_gpu_pm4_pending_t* previous;
  // Existing asynchronous wait lifecycle; no additional thread is created.
  iree_hal_amd_gpu_host_wait_t wait;
  // Kind selected by admission; never READY while a sidecar exists.
  iree_hal_amd_gpu_pm4_wait_kind_t kind;
  // Before begin, borrows the source from submission. After begin, owns the
  // registered completion or submission-availability timeline.
  iree_hal_semaphore_t* semaphore;
  // Captured value paired with semaphore.
  uint64_t value;
} iree_hal_amd_gpu_pm4_pending_t;

// One cache line of host retirement data per native publication. A capture
// can publish several pieces without overwriting a still-live piece's credit.
typedef struct iree_hal_amd_gpu_pm4_retirement_t {
  // Owned capture reference, distinct from any unpublished suffix.
  iree_alignas(64) iree_hal_amd_gpu_pm4_submission_t* submission;
  // Reclaimable command prefix: USER release end or KERNEL chunk end.
  uint64_t consumed_words;
  // Exclusive argument/staging prefix reusable after this host action.
  uint64_t consumed_arguments;
  // Optional bounded host copy, completed before returning staging credit.
  iree_hal_amd_gpu_pm4_download_t download;
  // This publication completes the entire captured operation.
  bool terminal;
} iree_hal_amd_gpu_pm4_retirement_t;

struct iree_hal_amd_gpu_pm4_queue_t {
  // Refcounted exact HAL queue; its family/device remain borrowed.
  iree_hal_queue_t base;
  // Allocator for the wrapper, cold backing owners, and oversized captures.
  iree_allocator_t host_allocator;
  // Borrowed native domain and event owner.
  iree_hal_amd_gpu_context_t* context;
  // Immutable native construction and memory facts.
  iree_hal_amd_gpu_pm4_queue_options_t options;
  // Optional device-owned session, changed only under idle begin/end.
  iree_hal_amd_gpu_profile_t* profile;
  // Persistent services borrowed by host calls and mappings.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Persistent services borrowed by file transfers.
  iree_hal_amd_gpu_pm4_file_executor_t file_executor;
  // Group-assigned completed progress; immutable after the first submission.
  struct {
    // Retained tracker, or NULL before group construction.
    iree_async_frontier_tracker_t* tracker;
    // Registered queue identity; epochs are native publication positions + 1.
    iree_async_axis_t axis;
  } frontier;
  // Sole caller/service native publication owner.
  iree_alignas(64) iree_hal_amd_gpu_publisher_t publisher;
  // Reusable host captures, independent of native ring capacity.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Required sidecar bytes, including one tracker, computed at construction.
  iree_host_size_t pending_size;
  // Deferred turns on the existing device service worker.
  iree_hal_amd_gpu_continuation_t continuation;
  // Accepted operations awaiting execution retirement or abandonment. The
  // publisher independently owns any trailing host publication bookkeeping.
  iree_atomic_int64_t outstanding;
  // Parent has closed public acceptance; accepted dependencies remain live.
  iree_atomic_int32_t closed;
  // Sticky raw native failure; failure never supplies reclamation credit.
  iree_atomic_int64_t native_failure;
  // Tracker associations shared by cold waits on this queue.
  iree_async_semaphore_wait_context_t wait_context;
  // Registered waits; only service accesses these links.
  iree_hal_amd_gpu_pm4_pending_t* pending;
  // Service has failed every published operation after native failure and
  // the last publishing caller's handoff. Their GPU ownership stays retained.
  bool issued_failure_delivered;
  // Completion-owned delivery of logical failure while native retirement runs.
  bool feedback_failure_delivered;
  // Native queue and mapping, joined after the final notification tail.
  iree_hal_amd_gpu_pm4_ring_storage_t storage;
  // Finite command owner, present only for KERNEL publication.
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel;
  // Exclusive native publication state and MMIO operands.
  struct {
    // Primary command ring, accessed only by the current publisher.
    iree_alignas(64) iree_hal_amd_gpu_pm4_ring_t ring;
    // Naturally aligned native monotonic DWORD index.
    iree_atomic_int64_t* write_index;
    // Write-only 64-bit MMIO doorbell.
    volatile uint64_t* doorbell;
    // Current argument generation, owned by the publisher.
    iree_hal_amd_gpu_pm4_argument_storage_t* arguments;
    // Service-owned replaced generations awaiting their completed byte prefix.
    iree_hal_amd_gpu_pm4_argument_storage_t* retired_arguments;
    // Active finite signal pool; exhaustion moves growth to service.
    iree_hal_amd_gpu_pm4_signal_storage_t* signals;
    // All independently retained signal backings, including the active one.
    iree_hal_amd_gpu_pm4_signal_storage_t* signal_storage;
    // Accepted native publication count, release-published for completion.
    iree_atomic_int64_t count;
    // Next native dispatch ID, independent of publication chunk boundaries.
    uint64_t dispatch_count;
    // Dense causal epochs over the device's immutable native queue set.
    struct {
      // Number of provisioned queues in this context's signal domain.
      uint32_t count;
      // Accepted FIFO prefix, indexed by native queue ordinal.
      uint64_t* published;
      // Current candidate, aliasing published when no waits are added.
      uint64_t* pending;
      // Second inline vector for candidates containing new native waits.
      uint64_t* scratch;
    } frontier;
  } producer;
  // Single completion observer's release-published capacity credit.
  struct {
    // Number of consumed retirement records.
    iree_alignas(64) iree_atomic_int64_t count;
    // Reclaimable command prefix established by the last retired action.
    iree_atomic_int64_t words;
    // Last consumed argument position, including wrap/alignment.
    iree_atomic_int64_t arguments;
  } completed;
  // Native host-visible records, immutable after preparation.
  iree_hal_amd_fabric_memory_t control;
  // Indexed native record geometry shared by publisher and observer.
  iree_hal_amd_gpu_pm4_retirement_ring_t retirement;
  // Fixed host actions parallel to native slots; no per-piece allocation.
  // GPU-carried pointers name these actions directly on normal retirement.
  iree_hal_amd_gpu_pm4_retirement_t* issued;
};

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_queue_vtable;

static iree_status_t iree_hal_amd_gpu_pm4_file_submit(
    iree_hal_queue_t* base, const iree_hal_transfer_operation_t* operation,
    iree_hal_semaphore_list_t signals, iree_hal_amd_gpu_pm4_barrier_t before,
    iree_hal_amd_gpu_pm4_barrier_t after);

void iree_hal_amd_gpu_pm4_queue_set_profile(
    iree_hal_queue_t* base, iree_hal_amd_gpu_profile_t* profile) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  queue->profile = profile;
  queue->file_executor.profile = profile;
}

static void iree_hal_amd_gpu_pm4_wake_shutdown(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (IREE_UNLIKELY(
          iree_atomic_load(&queue->closed, iree_memory_order_acquire))) {
    const amdf_status_t status =
        iree_hal_amd_gpu_context_signal(queue->context);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(queue->context, status,
                                      "GPU shutdown event_signal");
    }
  }
}

static void iree_hal_amd_gpu_pm4_complete(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_pm4_submission_release(submission);
  iree_atomic_fetch_sub(&queue->outstanding, 1, iree_memory_order_release);
  iree_hal_amd_gpu_pm4_wake_shutdown(queue);
}

// Consumes an operation failure through its public signals and diagnostic sink.
// This is also the terminal diagnostic boundary for a call without signals.
static void iree_hal_amd_gpu_pm4_fail_signals(
    iree_hal_amd_gpu_pm4_queue_t* queue, iree_hal_semaphore_list_t signals,
    iree_status_t status) {
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_semaphore_list_fail(signals, status);
}

static void iree_hal_amd_gpu_pm4_abandon(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_pm4_submission_profile(submission),
      false, 0);
  iree_hal_amd_gpu_pm4_fail_signals(queue, submission->signals, status);
  iree_hal_amd_gpu_pm4_complete(queue, submission);
}

static iree_status_t iree_hal_amd_gpu_pm4_grow_arguments(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t required_capacity) {
  iree_hal_amd_gpu_pm4_argument_storage_t* previous = queue->producer.arguments;
  uint64_t capacity =
      previous ? previous->capacity : queue->options.argument_capacity;
  while (capacity < required_capacity) {
    capacity *= 2;
  }
  iree_hal_amd_gpu_pm4_argument_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_argument_storage_allocate(
      queue->context, queue->options.memory, capacity,
      previous ? previous->end : 0, queue->host_allocator, &storage));
  if (previous) {
    previous->next = queue->producer.retired_arguments;
    queue->producer.retired_arguments = previous;
  }
  queue->producer.arguments = storage;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_grow_signals(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_hal_amd_gpu_pm4_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_signal_storage_allocate(
      queue->context, queue->options.memory, queue->options.retirement_capacity,
      queue->producer.frontier.count, queue->host_allocator, &storage));
  storage->next = queue->producer.signal_storage;
  queue->producer.signal_storage = storage;
  queue->producer.signals = storage;
  return iree_ok_status();
}

// Replaced argument backing has no independently retained readers. All cold
// reclamation stays on service, after the exact final argument position.
static void iree_hal_amd_gpu_pm4_trim_arguments(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  iree_hal_amd_gpu_pm4_argument_storage_t** link =
      &queue->producer.retired_arguments;
  while (*link) {
    iree_hal_amd_gpu_pm4_argument_storage_t* storage = *link;
    if (storage->end <= completed) {
      *link = storage->next;
      iree_hal_amd_gpu_pm4_argument_storage_free(
          queue->context, queue->host_allocator, storage);
    } else {
      link = &storage->next;
    }
  }
}

void iree_hal_amd_gpu_pm4_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
  iree_hal_amd_gpu_pm4_trim_arguments(queue);
  iree_hal_amd_gpu_pm4_signal_storage_t** link =
      &queue->producer.signal_storage;
  while (*link) {
    iree_hal_amd_gpu_pm4_signal_storage_t* storage = *link;
    // Only service selects the active pool. Inline publishers acquire from
    // that pool; inactive pools can only lose readers during this turn.
    bool unused = false;
    if (storage != queue->producer.signals) {
      iree_slim_mutex_lock(&storage->pool.mutex);
      unused = storage->pool.live_count == 0;
      iree_slim_mutex_unlock(&storage->pool.mutex);
    }
    if (unused) {
      *link = storage->next;
      iree_hal_amd_gpu_pm4_signal_storage_free(queue->context,
                                               queue->host_allocator, storage);
    } else {
      link = &storage->next;
    }
  }
}

typedef enum iree_hal_amd_gpu_pm4_publication_result_e {
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM,
} iree_hal_amd_gpu_pm4_publication_result_t;

// Acceptance publishes host retirement state before either waking the observer
// or ringing MMIO. The retained capture pins the publication tail even if the
// GPU and completion worker retire the entire piece before this caller returns.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_accepted(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position,
    bool terminal) {
  if (queue->producer.frontier.pending != queue->producer.frontier.published) {
    queue->producer.frontier.scratch = queue->producer.frontier.published;
    queue->producer.frontier.published = queue->producer.frontier.pending;
  }
  queue->producer.frontier.published[queue->options.queue_ordinal] =
      position + 1;
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_EXECUTE) {
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation =
        &submission->execute;
    iree_hal_amd_gpu_pm4_program_publish(
        &invocation->program->resident, invocation->instance,
        queue->options.queue_ordinal, position + 1);
    invocation->published = true;
  }
  if (submission->signal) {
    iree_hal_amd_gpu_pm4_signal_publish_frontier(
        submission->signal, queue->producer.frontier.published);
  }
  if (submission->profiled) {
    iree_hal_amd_gpu_profile_operation_record(
        queue->profile, iree_hal_amd_gpu_pm4_submission_profile(submission),
        true, submission->native_wait_count);
  }
  iree_hal_amd_gpu_pm4_submission_retain(submission);
  if (!terminal) {
    iree_atomic_fetch_add(&queue->outstanding, 1, iree_memory_order_relaxed);
  }
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
  if (queue->kernel) {
    if (iree_atomic_exchange(&queue->kernel->wake_required, 0,
                             iree_memory_order_acq_rel)) {
      iree_async_event_native_set(&queue->context->kernel_event);
    }
  } else {
    const uint64_t published_words = queue->producer.ring.reserved_words;
    iree_atomic_store(queue->producer.write_index, published_words,
                      iree_memory_order_release);
    const iree_hal_memory_transition_recipe_info_t publication_fence = {
        .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
    };
    iree_hal_memory_transition_execute_host_direct(&publication_fence, NULL, 1);
    *queue->producer.doorbell = published_words;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       terminal && submission->signal && i < submission->signals.count &&
       iree_status_is_ok(status);
       ++i) {
    if (iree_hal_amd_gpu_pm4_semaphore_isa(submission->signals.semaphores[i])) {
      status = iree_hal_amd_gpu_pm4_semaphore_publish_signal(
          submission->signals.semaphores[i],
          submission->signals.payload_values[i],
          (iree_async_frontier_entry_t){
              .axis = queue->frontier.axis,
              .epoch = queue->frontier.tracker ? position + 1 : 0,
          },
          submission->signal);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_fail_signals(queue, submission->signals, status);
  }
  if (terminal) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return terminal ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED
                  : IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE;
}

// The exclusive publisher keeps one prepared, unaccepted KERNEL piece. Native
// capacity leaves its bytes/cursors intact. Rejection rolls back storage only;
// accepted earlier pieces retain their independent references and retirement.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_kernel(iree_hal_amd_gpu_pm4_queue_t* queue,
                                    uint64_t position) {
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel = queue->kernel;
  iree_hal_amd_gpu_pm4_retirement_t* action =
      &queue->issued[position & queue->retirement.capacity_mask];
  const amdf_status_t status =
      iree_hal_amd_gpu_pm4_kmq_storage_submit(kernel, position);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY) &&
      position != (uint64_t)iree_atomic_load(&queue->completed.count,
                                             iree_memory_order_acquire)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (!amdf_status_is_ok(status)) {
    queue->producer.ring.reserved_words = kernel->rollback.words;
    queue->producer.arguments->end = kernel->rollback.arguments;
    kernel->pending.byte_length = 0;
    iree_hal_amd_gpu_pm4_abandon(
        queue, action->submission,
        IREE_HAL_AMD_STATUS_FROM_AMDF(status, "gpu.kernel_queue_submit"));
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  return iree_hal_amd_gpu_pm4_publish_accepted(queue, action->submission,
                                               position, action->terminal);
}

// Encodes one complete piece before native publication. USER publication is
// infallible; KERNEL acceptance can defer without rebuilding or replaying it.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_commit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position,
    const iree_hal_amd_gpu_pm4_ring_reservation_t* reservation,
    uint32_t body_words, uint32_t entry_word_count, uint32_t* words,
    iree_hal_amd_gpu_pm4_barrier_t after, uint64_t argument_end,
    iree_hal_amd_gpu_pm4_download_t download, bool terminal) {
  iree_hal_amd_gpu_pm4_retirement_t* action =
      &queue->issued[position & queue->retirement.capacity_mask];
  *action = (iree_hal_amd_gpu_pm4_retirement_t){
      .submission = submission,
      .consumed_arguments = argument_end,
      .download = download,
      .terminal = terminal,
  };
  uint32_t release_control;
  words =
      iree_hal_amd_gpu_pm4_barrier_emit_tail(words, after, &release_control);
  if (submission->signal) {
    words = iree_hal_amd_gpu_pm4_emit_release32(
        words, release_control, submission->signal->device_address, 1);
    release_control = 0;
  }
  words = iree_hal_amd_gpu_pm4_retirement_emit(
      words, &queue->retirement, position, action, release_control);
  if (queue->kernel) {
    if (entry_word_count) {
      // The resident IB chains into this fixed 32-DWORD continuation. Root
      // padding precedes its terminal CHAIN; only continuation padding is here.
      words = iree_hal_amd_gpu_pm4_emit_nop(
          words, body_words - (uint32_t)(words - reservation->words));
    }
    IREE_ASSERT_EQ(words - reservation->words, body_words);
    action->consumed_words = queue->producer.ring.reserved_words;
    queue->kernel->pending = (amdf_gpu_kernel_command_t){
        .memory = queue->kernel->memory.handle,
        .byte_offset = (reservation->words - queue->producer.ring.words) * 4,
        .byte_length = entry_word_count ? entry_word_count * 4
                                        : (queue->producer.ring.reserved_words -
                                           reservation->position) *
                                              4,
    };
    return iree_hal_amd_gpu_pm4_publish_kernel(queue, position);
  }
  action->consumed_words = reservation->position + (words - reservation->words);
  words = iree_hal_amd_gpu_pm4_emit_notify32(
      words, 0, queue->context->event_info.mailbox_address,
      queue->context->event_info.trigger_data,
      queue->context->event_info.interrupt_context);
  IREE_ASSERT_EQ(words - reservation->words, body_words);
  return iree_hal_amd_gpu_pm4_publish_accepted(queue, submission, position,
                                               terminal);
}

static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_transfer(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  iree_hal_amd_gpu_pm4_transfer_t* transfer = &submission->transfer;
  const bool first = !transfer->cursor.index && !transfer->cursor.offset;
  iree_hal_amd_gpu_pm4_argument_storage_t* arguments =
      queue->producer.arguments;
  if (arguments->capacity < transfer->staging_capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
  }
  const uint64_t relative =
      iree_align_uint64(arguments->end - arguments->begin, 8);
  const uint64_t offset = relative & (arguments->capacity - 1);
  const uint64_t start = arguments->begin + relative;
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  const uint64_t reusable = completed >= arguments->end
                                ? start
                                : iree_max(completed, arguments->begin);
  const uint64_t used = start - reusable;
  const uint64_t available =
      used >= arguments->capacity
          ? 0
          : iree_min(arguments->capacity - used, arguments->capacity - offset);
  const uint32_t word_budget =
      iree_min(queue->producer.ring.capacity / 2, 16384u);
  const uint32_t wait_words = first ? submission->native_wait_count * 7 : 0;
  // Admission reserves launch capacity beyond native waits. Keep 96 words for
  // signals, retirement, notification, and alignment, leaving room for DMA
  // packets even at the maximum admitted native fan-in.
  iree_hal_amd_gpu_pm4_transfer_chunk_t chunk;
  iree_status_t status = iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
      transfer,
      iree_min(IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_WORDS,
               word_budget - wait_words - 96),
      iree_make_byte_span(
          arguments->memory.contents.data + offset,
          iree_min(available,
                   IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_STAGING_LENGTH)),
      arguments->address + offset, &chunk);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (!chunk.word_count) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  const bool terminal = chunk.next.index == transfer->count;
  iree_hal_amd_gpu_pm4_barrier_t before =
      first ? submission->before : (iree_hal_amd_gpu_pm4_barrier_t){0};
  if (chunk.staging_length > chunk.download.length) {
    // Host-written transport bytes need visibility even with empty payload
    // barriers. This is the same mandatory publication as native kernargs.
    before.cache_control |= queue->options.barriers.acquire;
  }
  iree_hal_amd_gpu_pm4_barrier_t after =
      terminal ? submission->after : (iree_hal_amd_gpu_pm4_barrier_t){0};
  after.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  if (chunk.download.length) {
    after.cache_control |= queue->options.barriers.release;
  }
  bool native_signal = false;
  for (iree_host_size_t i = 0;
       terminal && !transfer->host_completion_required &&
       i < submission->signals.count;
       ++i) {
    native_signal |=
        iree_hal_amd_gpu_pm4_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words = wait_words + chunk.word_count +
                              iree_hal_amd_gpu_pm4_barrier_word_count(before) +
                              7 + (native_signal ? 8 : 0) + 14 +
                              (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
        &queue->producer.signals->pool);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  if (chunk.staging_length) {
    arguments->end = start + chunk.staging_length;
  }
  queue->producer.ring.reserved_words = candidate.reserved_words;
  transfer->cursor = chunk.next;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; first && i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  memcpy(words, chunk.words, chunk.word_count * sizeof(*words));
  words += chunk.word_count;
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      arguments->end, chunk.download, terminal);
}

// One bounded primary sequence invokes the entire resident program. Native
// copy/fixup work scales with program size; host work scales only with its
// dense input table and semaphore edges. No recorded operation is visited here.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_execute(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  iree_hal_amd_gpu_pm4_command_invocation_t* invocation = &submission->execute;
  iree_hal_amd_gpu_pm4_command_program_t* command = invocation->program;
  iree_hal_amd_gpu_pm4_program_t* program = &command->resident;
  iree_hal_amd_gpu_pm4_argument_storage_t* arguments =
      queue->producer.arguments;
  if (command->argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
  }
  uint64_t relative = iree_align_uint64(arguments->end - arguments->begin, 8);
  uint64_t offset = relative & (arguments->capacity - 1);
  if (command->argument_length > arguments->capacity - offset) {
    relative += arguments->capacity - offset;
    offset = 0;
  }
  const uint64_t start = arguments->begin + relative;
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  const uint64_t reusable = completed >= arguments->end
                                ? start
                                : iree_max(completed, arguments->begin);
  if (start - reusable + command->argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (!invocation->instance) {
    invocation->instance = iree_hal_amd_gpu_pm4_program_try_acquire(
        program, queue->options.queue_ordinal,
        queue->producer.frontier.pending);
    if (!invocation->instance) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM;
    }
  }
  iree_hal_amd_gpu_pm4_program_instance_t* instance = invocation->instance;
  const uint32_t program_words = iree_hal_amd_gpu_pm4_program_word_count(
      program, instance, &queue->options.barriers);
  const uint32_t prefix_words =
      submission->native_wait_count * 7 + program_words - 4;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_pm4_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t tail_words =
      (command->pending_dma ? 7 : 0) + (native_signal ? 8 : 0) + 14;
  uint32_t root_padding = 0;
  if (queue->kernel) {
    root_padding = (0u - prefix_words - 4) & 7u;
    if (root_padding == 1) {
      root_padding += 8;
    }
  }
  const uint32_t entry_words =
      queue->kernel ? prefix_words + root_padding + 4 : 0;
  const uint32_t body_words =
      queue->kernel ? entry_words + 32 : prefix_words + 4 + tail_words + 8;
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
        &queue->producer.signals->pool);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  const uint64_t dispatch_id =
      iree_align_uint64(queue->producer.dispatch_count,
                        iree_max(command->tsan_shadow_slot_count, 1u));
  const uint64_t continuation =
      queue->kernel
          ? queue->kernel->address +
                (reservation.words - queue->producer.ring.words + entry_words) *
                    4
          : 0;
  uint8_t* upload = arguments->memory.contents.data + offset;
  iree_status_t status = iree_hal_amd_gpu_pm4_command_invocation_write(
      invocation, queue->options.queue_ordinal, dispatch_id, continuation,
      upload);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  arguments->end = start + command->argument_length;
  queue->producer.ring.reserved_words = candidate.reserved_words;
  queue->producer.dispatch_count = dispatch_id + command->dispatch_count;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_program_emit(
      program, instance, command->entry, &queue->options.barriers,
      arguments->address + offset, (void*)upload, root_padding, words);
  const iree_hal_amd_gpu_pm4_barrier_t after = {
      .execution =
          command->pending_dma ? IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA : 0,
  };
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, entry_words, words,
      after, arguments->end, (iree_hal_amd_gpu_pm4_download_t){0}, true);
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

// A candidate contains only waits emitted in this native piece. It becomes
// the accepted FIFO prefix after native acceptance; a rejected KERNEL piece
// cannot claim its dependencies for a later operation. Retained signals carry
// exact immutable snapshots, never a producer queue's newer mutable state.
static void iree_hal_amd_gpu_pm4_prepare_frontier(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    const iree_hal_amd_gpu_pm4_submission_t* submission) {
  queue->producer.frontier.pending = queue->producer.frontier.published;
  bool first = true;
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER) {
    first = !submission->transfer.cursor.index &&
            !submission->transfer.cursor.offset;
  }
  if (first && submission->native_wait_count) {
    queue->producer.frontier.pending = queue->producer.frontier.scratch;
    memcpy(queue->producer.frontier.pending, queue->producer.frontier.published,
           queue->producer.frontier.count * sizeof(uint64_t));
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_pm4_signal_join_frontier(
            submission->waits[i].signal, queue->producer.frontier.pending);
      }
    }
  }
}

void iree_hal_amd_gpu_pm4_queue_notify_failure(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_continuation_notify(&queue->continuation);
}

// The sole publisher prepares one bounded piece without host allocation.
// Native KERNEL submission is the only fallible publication boundary. The
// caller detached the entry before calling; immediate retirement may free it.
static iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_status_t failure = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (!iree_status_is_ok(failure)) {
    if (queue->kernel && queue->kernel->pending.byte_length) {
      queue->producer.ring.reserved_words = queue->kernel->rollback.words;
      queue->producer.arguments->end = queue->kernel->rollback.arguments;
      queue->kernel->pending.byte_length = 0;
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
    queue->kernel->rollback.arguments = queue->producer.arguments->end;
  }
  iree_hal_amd_gpu_pm4_prepare_frontier(queue, submission);
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER) {
    return iree_hal_amd_gpu_pm4_publish_transfer(queue, submission, position);
  }
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_EXECUTE) {
    return iree_hal_amd_gpu_pm4_publish_execute(queue, submission, position);
  }
  const iree_hal_amd_gpu_pm4_function_t* function =
      submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH
          ? submission->dispatch.function
          : NULL;
  const bool indirect_parameters =
      function &&
      iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags);
  const bool dispatch =
      function && (indirect_parameters ||
                   (submission->dispatch.geometry.workgroup_count[0] &&
                    submission->dispatch.geometry.workgroup_count[1] &&
                    submission->dispatch.geometry.workgroup_count[2]));
  const bool indirect =
      iree_hal_dispatch_uses_indirect_arguments(submission->dispatch.flags);
  const iree_hal_amd_gpu_pm4_dispatch_storage_t argument_storage =
      dispatch ? iree_hal_amd_gpu_pm4_dispatch_storage(
                     function, submission->dispatch.flags)
               : (iree_hal_amd_gpu_pm4_dispatch_storage_t){0};
  const uint32_t argument_length = argument_storage.length;
  iree_hal_amd_gpu_pm4_argument_storage_t* arguments =
      queue->producer.arguments;
  uint64_t argument_offset = 0;
  uint64_t argument_end = arguments->end;
  uint64_t argument_address = 0;
  if (argument_length) {
    const uint64_t alignment = argument_storage.alignment;
    if (argument_length > arguments->capacity ||
        alignment > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
    }
    uint64_t relative =
        iree_align_uint64(arguments->end - arguments->begin, alignment);
    argument_offset = relative & (arguments->capacity - 1);
    if (argument_offset + argument_length > arguments->capacity) {
      relative += arguments->capacity - argument_offset;
      argument_offset = 0;
    }
    argument_end = arguments->begin + relative + argument_length;
    const uint64_t completed = (uint64_t)iree_atomic_load(
        &queue->completed.arguments, iree_memory_order_acquire);
    const uint64_t reusable = completed >= arguments->end
                                  ? arguments->begin + relative
                                  : iree_max(completed, arguments->begin);
    if (argument_end - reusable > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
    argument_address = arguments->address + argument_offset;
  }
  iree_hal_amd_gpu_pm4_barrier_t before = submission->before;
  if (dispatch) {
    // Instruction upload and host argument publication remain mandatory even
    // when the caller supplies an empty optional payload boundary.
    before.cache_control |= 1u | queue->options.barriers.acquire;
    if (queue->options.tsan_shadow_slot_count &&
        !(queue->producer.dispatch_count &
          (queue->options.tsan_shadow_slot_count - 1))) {
      before.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
      before.cache_control |= queue->options.barriers.shader;
    }
  }
  const uint16_t indirect_copy_count =
      dispatch && indirect_parameters && !indirect
          ? function->kernarg->indirect_copy_count
          : 0;
  const bool prepare_packet =
      dispatch && function->packet && indirect_parameters;
  const uint32_t launch_words =
      dispatch ? (prepare_packet ? IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS
                                 : 0u) +
                     (function->launch.user_data_count
                          ? 24 + function->launch.user_data_count
                          : 22) +
                     (indirect_parameters
                          ? iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(
                                indirect_copy_count)
                          : 5)
               : 0;
  const uint32_t tail_words =
      (submission->after.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA)
          ? 7
          : 0;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_pm4_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words = submission->native_wait_count * 7 + launch_words +
                              iree_hal_amd_gpu_pm4_barrier_word_count(before) +
                              tail_words + (native_signal ? 8 : 0) + 14 +
                              (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  const iree_host_size_t binding_count = submission->dispatch.bindings.count;
  uint64_t* addresses = (uint64_t*)(submission->dispatch.bindings.values +
                                    binding_count + indirect_parameters);
  uint64_t parameters_address = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < submission->dispatch.bindings.count &&
       iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_pm4_submission_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        submission->dispatch.bindings.values[i],
        indirect ? IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS
                 : IREE_HAL_BUFFER_USAGE_STORAGE |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
        indirect ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (indirect_parameters && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_submission_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_QUEUE,
        submission->dispatch.bindings.values[binding_count],
        IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        IREE_HAL_MEMORY_ACCESS_READ, &parameters_address);
    if (iree_status_is_ok(status) && (parameters_address & 3)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect parameter address is misaligned");
    }
  }
  if (dispatch && indirect && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (function->kernarg->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect kernarg address is misaligned");
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
        &queue->producer.signals->pool);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  arguments->end = argument_end;
  // Reservation commits only the advancing position; ring geometry is fixed.
  queue->producer.ring.reserved_words = candidate.reserved_words;
  if (argument_length) {
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        function, &submission->dispatch.geometry,
        submission->dispatch.constants, addresses, submission->dispatch.flags,
        arguments->memory.contents.data + argument_offset);
  }
  uint64_t packet_address = 0;
  if (dispatch && function->packet) {
    const uint64_t packet_offset =
        argument_offset + argument_storage.packet_offset;
    packet_address = arguments->address + packet_offset;
    iree_hal_amd_gpu_pm4_dispatch_packet_t* packet =
        (void*)(arguments->memory.contents.data + packet_offset);
    packet->kernarg_address = argument_address;
    if (prepare_packet) {
      uint64_t* inputs = (uint64_t*)(packet + 1);
      inputs[0] = parameters_address;
      inputs[1] = packet_address;
    }
  }
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  if (dispatch) {
    if (prepare_packet) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
          words, function->packet->prepare, packet_address + 64,
          queue->options.barriers.shader);
    }
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
        .dispatch_pointer = packet_address,
        .kernarg_pointer = argument_address,
        .dispatch_id = queue->producer.dispatch_count++};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch, submission->dispatch.geometry.workgroup_size,
        submission->dispatch.geometry.partial_size,
        submission->dispatch.geometry.group_segment_byte_length, &inputs);
    const uint32_t initiator =
        function->launch.dispatch_initiator |
        (submission->dispatch.geometry.partial_size[0] ? UINT32_C(1) << 6 : 0);
    if (indirect_parameters) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
          words, function->kernarg, indirect_copy_count, parameters_address,
          argument_address,
          queue->options.barriers.command_processor_bypasses_l2
              ? queue->options.barriers.acquire
              : queue->options.barriers.shader,
          initiator);
    } else {
      words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
          words, submission->dispatch.geometry.workgroup_count, initiator);
    }
  }
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words,
      submission->after, argument_end, (iree_hal_amd_gpu_pm4_download_t){0},
      true);
}

// An unresolved edge leaves no native reader pinned. In particular, a consumer
// awaiting a future producer cannot exhaust that producer's signal backing.
static iree_status_t iree_hal_amd_gpu_pm4_admit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_READY;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_failure(queue));
  // A bounded direct publication must fit strictly inside half the ring.
  // Reserve a bounded launch plus 96 words for barriers, signal, retirement,
  // notification, and alignment. Large fan-in joins use completed host
  // timelines, leaving command capacity independent of the semaphore list.
  const uint32_t word_budget =
      iree_min(queue->producer.ring.capacity / 2, 16384u);
  const uint32_t reserved_words = IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS + 96;
  const bool native_eligible =
      submission->wait_count <= (word_budget - reserved_words) / 7 &&
      !(submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER &&
        submission->transfer.host_wait_required);
  uint32_t native_wait_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < submission->wait_count &&
                               *out_kind == IREE_HAL_AMD_GPU_PM4_WAIT_READY &&
                               iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_pm4_submission_wait_t* wait = &submission->waits[i];
    if (native_eligible &&
        iree_hal_amd_gpu_pm4_semaphore_isa(wait->semaphore)) {
      iree_hal_amd_gpu_pm4_semaphore_wait_t acquired;
      status = iree_hal_amd_gpu_pm4_semaphore_acquire_wait(
          wait->semaphore, wait->value, &acquired);
      if (iree_status_is_ok(status)) {
        if (acquired.state == IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_DEFERRED) {
          *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION;
        } else if (acquired.state ==
                   IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_NATIVE) {
          if (acquired.signal->pool->domain == queue->context) {
            wait->signal = acquired.signal;
            ++native_wait_count;
          } else {
            iree_hal_amd_gpu_pm4_signal_release(acquired.signal);
            *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION;
          }
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
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      iree_hal_amd_gpu_pm4_signal_release(submission->waits[i].signal);
      submission->waits[i].signal = NULL;
    }
  } else {
    submission->native_wait_count = native_wait_count;
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

// Wait completion reenters the same publication owner after tracker
// finalization has joined every callback. Registration itself always runs on
// service.
static void iree_hal_amd_gpu_pm4_enqueue(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission);

static void iree_hal_amd_gpu_pm4_pending_resume(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_pm4_pending_t* pending = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
  if (pending->previous) {
    pending->previous->next = pending->next;
  } else {
    queue->pending = pending->next;
  }
  if (pending->next) {
    pending->next->previous = pending->previous;
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_pm4_pending_release(pending);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_enqueue(queue, submission);
  } else {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  }
}

static void iree_hal_amd_gpu_pm4_pending_begin(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_pm4_pending_t* pending = (void*)entry;
  iree_hal_amd_gpu_pm4_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
  iree_hal_semaphore_t* source = pending->semaphore;
  pending->semaphore = NULL;
  iree_status_t status = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (iree_status_is_ok(status)) {
    if (pending->kind == IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION) {
      status = iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
          source, &pending->semaphore);
    } else {
      pending->semaphore = source;
      iree_hal_semaphore_retain(source);
    }
  }
  if (iree_status_is_ok(status)) {
    pending->previous = NULL;
    pending->next = queue->pending;
    if (pending->next) {
      pending->next->previous = pending;
    }
    queue->pending = pending;
    status = iree_hal_amd_gpu_host_wait_begin(
        &queue->wait_context, queue->options.service,
        (iree_hal_semaphore_list_t){1, &pending->semaphore, &pending->value},
        iree_hal_amd_gpu_pm4_pending_resume, pending,
        iree_allocator_inline_arena(&pending->tracker_storage), &pending->wait);
    if (iree_status_is_ok(status)) {
      return;
    }
    queue->pending = pending->next;
    if (pending->next) {
      pending->next->previous = NULL;
    }
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_pm4_pending_release(pending);
  iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
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
      .entry = {.fn = iree_hal_amd_gpu_pm4_pending_begin},
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
        iree_hal_memory_maintenance_enqueue(queue->options.service,
                                            &pending->entry);
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

// Publication mode and whether dependencies are ready determine ownership;
// enqueue order does not supply a HAL execution dependency.
static void iree_hal_amd_gpu_pm4_enqueue(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (!iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                          &submission->publication)) {
    return;
  }
  iree_hal_amd_gpu_publication_entry_t* remainder =
      iree_hal_amd_gpu_publisher_next(&queue->publisher);
  if (remainder &&
      iree_hal_amd_gpu_pm4_process(
          queue, iree_containerof(remainder, iree_hal_amd_gpu_pm4_submission_t,
                                  publication)) ==
          IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
    remainder = NULL;
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, remainder) ||
      (iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) ||
       iree_hal_amd_gpu_pm4_feedback_failure(queue))) {
    iree_hal_amd_gpu_continuation_notify(&queue->continuation);
  }
  iree_hal_amd_gpu_pm4_wake_shutdown(queue);
}

// Capture/validation have already succeeded. Admission errors after acceptance
// belong to signal timelines and the event sink, including on an inline turn.
static void iree_hal_amd_gpu_pm4_submit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_atomic_fetch_add(&queue->outstanding, 1, iree_memory_order_relaxed);
  iree_hal_amd_gpu_pm4_enqueue(queue, submission);
}

static bool iree_hal_amd_gpu_pm4_progress(void* user_data) {
  iree_hal_amd_gpu_pm4_queue_t* queue = user_data;
  // Growth and this retired list belong exclusively to service. Completion
  // can return their final credit without another submission arriving.
  iree_hal_amd_gpu_pm4_trim_arguments(queue);
  bool more = false;
  if (iree_hal_amd_gpu_publisher_acquire_service(&queue->publisher)) {
    bool capacity = false;
    iree_hal_amd_gpu_publication_entry_t* current = NULL;
    for (uint32_t i = 0; i < 32 && !capacity; ++i) {
      if (!current) {
        current = iree_hal_amd_gpu_publisher_next(&queue->publisher);
      }
      if (!current) {
        break;
      }
      iree_hal_amd_gpu_pm4_submission_t* submission = iree_containerof(
          current, iree_hal_amd_gpu_pm4_submission_t, publication);
      const iree_hal_amd_gpu_pm4_publication_result_t result =
          iree_hal_amd_gpu_pm4_process(queue, submission);
      iree_status_t status = iree_ok_status();
      switch (result) {
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED:
          current = NULL;
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE:
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY:
          capacity = true;
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS: {
          uint64_t capacity = 0;
          if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_EXECUTE) {
            capacity = submission->execute.program->argument_length;
          } else if (submission->kind ==
                     IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER) {
            capacity = submission->transfer.staging_capacity;
          } else {
            const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
                iree_hal_amd_gpu_pm4_dispatch_storage(
                    submission->dispatch.function, submission->dispatch.flags);
            capacity = iree_max(storage.length, storage.alignment);
          }
          status = iree_hal_amd_gpu_pm4_grow_arguments(queue, capacity);
          break;
        }
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM:
          status = iree_hal_amd_gpu_pm4_program_grow(
              &submission->execute.program->resident);
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS:
          // Reuse idle earlier backing before requesting more native memory.
          for (iree_hal_amd_gpu_pm4_signal_storage_t* storage =
                   queue->producer.signal_storage;
               storage; storage = storage->next) {
            iree_hal_amd_gpu_pm4_signal_t* signal =
                iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage->pool);
            if (signal) {
              queue->producer.signals = storage;
              iree_hal_amd_gpu_pm4_signal_release(signal);
              break;
            }
          }
          // All currently free slots are stable while this publisher owns
          // acquisition; other threads only return references.
          iree_slim_mutex_lock(&queue->producer.signals->pool.mutex);
          const bool available =
              queue->producer.signals->pool.free_head != NULL;
          iree_slim_mutex_unlock(&queue->producer.signals->pool.mutex);
          if (!available) {
            status = iree_hal_amd_gpu_pm4_grow_signals(queue);
          }
          break;
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
        current = NULL;
      }
    }
    more = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, current) &&
           !capacity;
    iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  }
  iree_status_t failure = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (!iree_status_is_ok(failure)) {
    for (iree_hal_amd_gpu_pm4_pending_t* pending = queue->pending; pending;
         pending = pending->next) {
      iree_hal_amd_gpu_host_wait_cancel(&pending->wait);
    }
    if (!queue->issued_failure_delivered &&
        iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) &&
        iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      const uint64_t begin = (uint64_t)iree_atomic_load(
          &queue->completed.count, iree_memory_order_acquire);
      const uint64_t end = (uint64_t)iree_atomic_load(
          &queue->producer.count, iree_memory_order_acquire);
      for (uint64_t i = begin; i != end; ++i) {
        iree_hal_amd_gpu_pm4_submission_t* submission =
            queue->issued[i & queue->retirement.capacity_mask].submission;
        iree_hal_amd_gpu_pm4_fail_signals(queue, submission->signals,
                                          iree_status_clone(failure));
      }
      // A published operation establishes that group assignment completed.
      // An idle observer must not race construction by reading frontier state.
      if (end && queue->frontier.tracker) {
        iree_async_frontier_tracker_fail_axis(queue->frontier.tracker,
                                              queue->frontier.axis,
                                              iree_status_clone(failure));
      }
      queue->issued_failure_delivered = true;
    }
  }
  iree_status_free(failure);
  return more;
}

static void iree_hal_amd_gpu_pm4_fail_native(
    iree_hal_amd_gpu_pm4_queue_t* queue, amdf_status_t status,
    const char* operation) {
  iree_atomic_store(&queue->native_failure, status, iree_memory_order_release);
  iree_hal_amd_gpu_context_report(queue->context, status, operation);
  iree_hal_amd_gpu_continuation_notify(&queue->continuation);
}

static bool iree_hal_amd_gpu_pm4_retire(void* user_data,
                                        amdf_status_t wait_status) {
  iree_hal_amd_gpu_pm4_queue_t* queue = user_data;
  if (!amdf_status_is_ok((amdf_status_t)iree_atomic_load(
          &queue->native_failure, iree_memory_order_relaxed))) {
    return false;
  }
  if (!amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_pm4_fail_native(queue, wait_status, "GPU completion wait");
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
    // Only this consumer releases published captures. Service cannot inspect
    // them during a logical failure because native retirement keeps running.
    for (uint64_t i = position; i != published; ++i) {
      iree_hal_amd_gpu_pm4_submission_t* submission =
          queue->issued[i & queue->retirement.capacity_mask].submission;
      iree_hal_amd_gpu_pm4_fail_signals(
          queue, submission->signals,
          iree_make_status(feedback_failure, "GPU sanitizer failure"));
    }
    if (published && queue->frontier.tracker) {
      iree_async_frontier_tracker_fail_axis(
          queue->frontier.tracker, queue->frontier.axis,
          iree_make_status(feedback_failure, "GPU sanitizer failure"));
    }
    queue->feedback_failure_delivered = true;
  }
  uint64_t native_retired = 0;
  if (queue->kernel && position != published) {
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
    iree_hal_amd_gpu_pm4_submission_t* submission = action->submission;
    iree_status_code_t completion_code = IREE_STATUS_OK;
    if (action->terminal && !iree_hal_amd_gpu_pm4_submission_retirement_ready(
                                submission, &completion_code)) {
      break;
    }
    if (queue->options.feedback) {
      // Native retirement makes this dispatch's reports observable. Drain the
      // ready shared prefix before waking host completion consumers.
      iree_hal_amd_gpu_feedback_drain(queue->options.feedback);
      if (completion_code == IREE_STATUS_OK) {
        completion_code = iree_hal_amd_gpu_pm4_feedback_failure(queue);
      }
    }
    const uint64_t consumed_words = action->consumed_words;
    const uint64_t consumed_arguments = action->consumed_arguments;
    if (action->download.length) {
      memcpy(action->download.target, action->download.source,
             action->download.length);
    }
    if (action->terminal) {
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
         action->terminal && i < submission->signals.count &&
         iree_status_is_ok(status);
         ++i) {
      iree_hal_semaphore_t* semaphore = submission->signals.semaphores[i];
      const uint64_t value = submission->signals.payload_values[i];
      status = iree_hal_amd_gpu_pm4_semaphore_isa(semaphore)
                   ? iree_hal_amd_gpu_pm4_semaphore_publish_completed(
                         semaphore, value, frontier)
                   : iree_hal_semaphore_signal(semaphore, value, frontier);
    }
    completion_code = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_fail_signals(queue, submission->signals, status);
    }
    if (action->terminal) {
      iree_hal_amd_gpu_pm4_signal_retire(submission->signal, completion_code);
    }
    iree_hal_amd_gpu_pm4_complete(queue, submission);
    iree_atomic_store(&queue->completed.words, consumed_words,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.arguments, consumed_arguments,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.count, position + 1,
                      iree_memory_order_release);
  }
  if (position != initial) {
    iree_hal_amd_gpu_continuation_notify(&queue->continuation);
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
  // A later source can retire this source's native producer. One more round
  // after actual progress joins those causal payload lifetimes without a
  // topological queue walk or polling merely outstanding native work.
  return position != initial;
}

iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_pm4_queue_completion_source(iree_hal_queue_t* base) {
  return (iree_hal_amd_gpu_completion_source_t){iree_hal_amd_gpu_pm4_retire,
                                                base};
}

void iree_hal_amd_gpu_pm4_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_pm4_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_pm4_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  return !iree_atomic_load(&queue->outstanding, iree_memory_order_acquire) &&
         iree_hal_amd_gpu_publisher_is_idle(&queue->publisher);
}

static void iree_hal_amd_gpu_pm4_clear_frontier(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (queue->frontier.tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier.tracker, queue->frontier.axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->frontier.tracker);
  }
  queue->frontier.tracker = NULL;
  queue->frontier.axis = 0;
}

iree_status_t iree_hal_amd_gpu_pm4_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  if (tracker) {
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_tracker_register_axis(tracker, axis, NULL));
    iree_async_frontier_tracker_retain(tracker);
  }
  iree_hal_amd_gpu_pm4_clear_frontier(queue);
  queue->frontier.tracker = tracker;
  queue->frontier.axis = axis;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_check_acceptance(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "GPU queue is closed");
  }
  return iree_hal_amd_gpu_pm4_check_failure(queue);
}

static iree_status_t iree_hal_amd_gpu_pm4_host_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t operation_count,
    iree_host_size_t public_signal_count) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const bool profiled =
      queue->profile && type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, profiled, waits, signals, NULL, NULL,
      (iree_hal_amd_gpu_pm4_dispatch_t){0}, (iree_hal_buffer_ref_t){0},
      iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
      &queue->options.barriers, barriers, 0, &submission));
  if (profiled) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal, type, waits.count,
        public_signal_count, operation_count,
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
  return iree_hal_amd_gpu_pm4_host_barrier(
      base, waits, signals, barriers, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
      0, signals.count);
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
      config, constants, bindings.count, flags, &dispatch));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_dispatch_validate_direct_bindings(
      function->kernarg, config.workgroup_count_ref, bindings, flags));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, queue->profile != NULL, waits, signals, executable,
      function, dispatch, config.workgroup_count_ref, constants, bindings,
      &queue->options.barriers, barriers, flags, &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, waits.count, signals.count,
        1, iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cold native construction and joined teardown
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_pm4_prepare_t {
  // Queue under exclusive construction ownership.
  iree_hal_amd_gpu_pm4_queue_t* queue;
  // Construction status transferred to the joining caller.
  iree_status_t status;
} iree_hal_amd_gpu_pm4_prepare_t;

static iree_status_t iree_hal_amd_gpu_pm4_prepare_native(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  if (queue->options.publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_kmq_storage_allocate(
        queue->context, queue->options.family_ordinal,
        queue->options.command_byte_capacity,
        queue->options.retirement_capacity, queue->options.memory,
        queue->host_allocator, &queue->kernel, &queue->producer.ring));
  } else {
    const amdf_gpu_user_queue_create_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(native),
        .queue_family_ordinal = queue->options.family_ordinal,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        .ring_byte_length = queue->options.command_byte_capacity,
    };
    amdf_user_queue_mapping_info_t mapping;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_ring_storage_initialize(
        queue->context, &native, &queue->storage, &queue->producer.ring,
        &mapping));
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
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_backing_allocate(
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
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 control backing lacks coherent host access "
                            "or qualified 32-bit native signals");
  }
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(
      (iree_hal_amd_gpu_pm4_retirement_record_t*)queue->control.contents.data,
      control_address, queue->options.retirement_capacity, &queue->retirement);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_grow_arguments(
      queue, queue->options.argument_capacity));
  return iree_hal_amd_gpu_pm4_grow_signals(queue);
}

static void iree_hal_amd_gpu_pm4_prepare(void* user_data) {
  iree_hal_amd_gpu_pm4_prepare_t* prepare = user_data;
  prepare->status = iree_hal_amd_gpu_pm4_prepare_native(prepare->queue);
}

typedef struct iree_hal_amd_gpu_pm4_cleanup_t {
  // Joined queue whose native handles may be removed.
  iree_hal_amd_gpu_pm4_queue_t* queue;
  // Raw native join/removal result, transferred to the parent.
  amdf_status_t status;
} iree_hal_amd_gpu_pm4_cleanup_t;

static void iree_hal_amd_gpu_pm4_cleanup(void* user_data) {
  iree_hal_amd_gpu_pm4_cleanup_t* cleanup = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = cleanup->queue;
  cleanup->status =
      queue->kernel ? iree_hal_amd_gpu_pm4_kmq_storage_free(queue->kernel)
                    : iree_hal_amd_gpu_pm4_ring_storage_deinitialize(
                          &queue->storage, queue->producer.ring.reserved_words);
  if (!amdf_status_is_ok(cleanup->status)) {
    queue->context->cleanup_blocked = true;
    return;
  }
  queue->kernel = NULL;
  while (queue->producer.signal_storage) {
    iree_hal_amd_gpu_pm4_signal_storage_t* storage =
        queue->producer.signal_storage;
    queue->producer.signal_storage = storage->next;
    iree_hal_amd_gpu_pm4_signal_storage_free(queue->context,
                                             queue->host_allocator, storage);
  }
  queue->producer.signals = NULL;
  iree_hal_amd_gpu_pm4_argument_storage_t* arguments =
      queue->producer.arguments;
  if (arguments) {
    arguments->next = queue->producer.retired_arguments;
  } else {
    arguments = queue->producer.retired_arguments;
  }
  queue->producer.arguments = NULL;
  queue->producer.retired_arguments = NULL;
  while (arguments) {
    iree_hal_amd_gpu_pm4_argument_storage_t* next = arguments->next;
    iree_hal_amd_gpu_pm4_argument_storage_free(
        queue->context, queue->host_allocator, arguments);
    arguments = next;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->control);
}

amdf_status_t iree_hal_amd_gpu_pm4_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_pm4_queue_is_idle(base),
              "shutdown requires joined accepted work");
  iree_hal_amd_gpu_continuation_deinitialize(&queue->continuation);
  iree_hal_amd_gpu_pm4_cleanup_t cleanup = {.queue = queue};
  iree_hal_memory_maintenance_call(queue->options.service,
                                   iree_hal_amd_gpu_pm4_cleanup, &cleanup);
  return cleanup.status;
}

static void iree_hal_amd_gpu_pm4_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_ASSERT(!queue->storage.queue && !queue->kernel && !queue->control.handle,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_hal_amd_gpu_capture_pool_deinitialize(&queue->captures);
  iree_hal_amd_gpu_pm4_clear_frontier(queue);
  iree_allocator_free_aligned(queue->host_allocator, queue);
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

iree_status_t iree_hal_amd_gpu_pm4_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_pm4_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  if (iree_cpu_architecture_host() != IREE_CPU_ARCHITECTURE_X86_64 ||
      !iree_atomic_int32_is_lock_free() || !iree_atomic_int64_is_lock_free()) {
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
  const uint32_t queue_count = family->spec->provisioned_queue_count;
  if (!iree_host_size_checked_mul_add(total_size, options->retirement_capacity,
                                      sizeof(iree_hal_amd_gpu_pm4_retirement_t),
                                      &total_size) ||
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
      host_allocator, total_size, iree_alignof(*queue), 0, (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->storage.context = context;
  queue->options = *options;
  queue->pending_size = pending_size;
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .service = options->service,
      .captures = &queue->captures,
      .barrier = iree_hal_amd_gpu_pm4_host_barrier};
  queue->file_executor = (iree_hal_amd_gpu_pm4_file_executor_t){
      .queue = &queue->base,
      .context = context,
      .service = options->service,
      .captures = &queue->captures,
      .waits = &queue->wait_context,
      .queue_ordinal = options->queue_ordinal,
      .submit = iree_hal_amd_gpu_pm4_file_submit};
  queue->issued = (iree_hal_amd_gpu_pm4_retirement_t*)(queue + 1);
  queue->producer.frontier.count = queue_count;
  queue->producer.frontier.published =
      (uint64_t*)(queue->issued + options->retirement_capacity);
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
  iree_hal_amd_gpu_continuation_initialize(options->service,
                                           iree_hal_amd_gpu_pm4_progress, queue,
                                           &queue->continuation);
  iree_hal_amd_gpu_capture_pool_initialize(options->capture_block_size,
                                           host_allocator, &queue->captures);
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_prepare_t prepare = {.queue = queue};
    iree_hal_memory_maintenance_call(options->service,
                                     iree_hal_amd_gpu_pm4_prepare, &prepare);
    status = prepare.status;
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
  return iree_hal_amd_gpu_memory_mapping_submit(
      &queue->host_executor, waits, signals, mapping_count, mappings, flags);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PM4 dispatch residency calculation is not implemented");
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic wait is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic store is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic RMW is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_timestamp(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public timestamp is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_flush(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_continuation_notify(&queue->continuation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_alloca(
      queue->options.allocations, queue->options.queue_ordinal, waits, signals,
      pool, count, requests, out_buffers);
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  return iree_hal_amd_gpu_queue_dealloca(queue->options.allocations,
                                         queue->options.queue_ordinal, waits,
                                         signals, count, buffers);
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
      &submission));
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
    iree_hal_queue_t* base, const iree_hal_transfer_operation_t* operation,
    iree_hal_semaphore_list_t signals, iree_hal_amd_gpu_pm4_barrier_t before,
    iree_hal_amd_gpu_pm4_barrier_t after) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_transfer(
      &queue->captures, false, base->queue_family,
      iree_hal_semaphore_list_empty(), signals, operation ? 1 : 0, operation,
      &queue->options.barriers, &barriers, &submission));
  submission->before = before;
  submission->after = after;
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
  return iree_hal_amd_gpu_pm4_file_transfer_submit(
      &queue->file_executor, waits, signals, source,
      IREE_HAL_MEMORY_ACCESS_READ, source_offset, target, target_offset, length,
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->after : NULL));
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_file_t* target,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  return iree_hal_amd_gpu_pm4_file_transfer_submit(
      &queue->file_executor, waits, signals, target,
      IREE_HAL_MEMORY_ACCESS_WRITE, target_offset, source, source_offset,
      length,
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_pm4_barrier_prepare(&queue->options.barriers,
                                           barriers ? barriers->after : NULL));
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
