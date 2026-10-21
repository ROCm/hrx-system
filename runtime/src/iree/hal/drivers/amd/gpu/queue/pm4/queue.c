// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/gpu/dispatch.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/file_transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/host_call.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/growth.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"
#include "iree/hal/drivers/amd/gpu/queue/transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"
#include "iree/hal/drivers/amd/status.h"

typedef struct iree_hal_amd_gpu_pm4_queue_t iree_hal_amd_gpu_pm4_queue_t;

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
  iree_hal_amd_gpu_download_t download;
  // Replaced scratch generations dominated by this FIFO completion.
  iree_hal_amd_gpu_pm4_scratch_storage_t* scratch;
  // This publication completes the entire captured operation.
  bool terminal;
} iree_hal_amd_gpu_pm4_retirement_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_retirement_t) == 64,
              "retirement records occupy one cache line");

// A ready single-piece dispatch borrows its capture from the reserved native
// retirement slot. Deferred or larger captures retain independent pool storage.
// Reuse follows completed.count, published only after final resource release.
typedef struct iree_hal_amd_gpu_pm4_retirement_slot_t {
  // Native retirement action, initialized without overwriting capture storage.
  iree_hal_amd_gpu_pm4_retirement_t action;
  // Exclusively held through publication, including KERNEL capacity deferral,
  // and final retirement. The action's alignment also aligns this span.
  uint8_t capture[448];
} iree_hal_amd_gpu_pm4_retirement_slot_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_retirement_slot_t) == 512,
              "retirement slots have a power-of-two stride");
static_assert(offsetof(iree_hal_amd_gpu_pm4_retirement_slot_t, capture) %
                      iree_alignof(iree_hal_amd_gpu_pm4_submission_t) ==
                  0,
              "retirement actions preserve capture alignment");

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
  // Persistent native boundary submission borrowed by host calls.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Persistent services borrowed by file transfers.
  iree_hal_amd_gpu_file_executor_t file_executor;
  // Group-assigned completed progress; immutable after the first submission.
  struct {
    // Retained tracker, or NULL before group construction.
    iree_async_frontier_tracker_t* tracker;
    // Registered queue identity; epochs are native publication positions + 1.
    iree_async_axis_t axis;
  } frontier;
  // Sole caller/deferred native publication owner.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_hal_amd_gpu_publisher_t publisher;
  // Reusable host captures, independent of native ring capacity.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Required sidecar bytes, including one tracker, computed at construction.
  iree_host_size_t pending_size;
  // Typed native acquisition and coalesced waiting captures.
  iree_hal_amd_gpu_pm4_growth_context_t growth;
  // Constant-time available cells from every completed native signal block.
  iree_hal_amd_gpu_signal_pool_t signals;
  // Coalesced explicit trim request, consumed under publication ownership.
  struct {
    // Stable typed record; only the first requester links it into the FIFO.
    iree_hal_amd_gpu_pm4_entry_t entry;
    // One while the trim entry is enqueued or executing.
    iree_atomic_int32_t pending;
  } trim;
  // Parent has closed public acceptance; accepted dependencies remain live.
  // Admission facts stay separate from publication, retirement, and callbacks.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t closed;
  // Sticky raw native failure; failure never supplies reclamation credit.
  iree_atomic_int64_t native_failure;
  // Tracker associations shared by cold waits on this queue.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_async_semaphore_wait_context_t wait_context;
  // Deferred admission and its independent callback-return lifetime.
  struct {
    // Deferred unpublished captures and admission callbacks. Native pieces
    // use the published/completed prefixes instead of modifying this count.
    iree_atomic_int64_t host_owners;
    // Failure-cancellable waits, independent of publication ownership.
    iree_hal_amd_gpu_host_wait_set_t waits;
    // Joins each callback's final count drop and shutdown notification. Neither
    // wait registration nor resumed publication holds this cold mutex.
    iree_slim_mutex_t completion_mutex;
  } pending;
  // Service has failed every published operation after native failure and
  // the last publishing caller's handoff. Their GPU ownership stays retained.
  bool issued_failure_delivered;
  // Completion-owned delivery of logical failure while native retirement runs.
  bool feedback_failure_delivered;
  // Native queue and mapping, joined after the final notification tail.
  iree_hal_amd_gpu_user_queue_t storage;
  // Finite command owner, present only for KERNEL publication.
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel;
  // Exclusive native publication state and MMIO operands.
  struct {
    // Primary command ring, accessed only by the current publisher.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_hal_amd_gpu_pm4_ring_t ring;
    // Naturally aligned native monotonic DWORD index.
    iree_atomic_int64_t* write_index;
    // Write-only 64-bit MMIO doorbell.
    volatile uint64_t* doorbell;
    // Current argument generation, owned by the publisher.
    iree_hal_amd_gpu_argument_storage_t* arguments;
    // Replaced generations ordered by their final consumed byte position.
    iree_hal_amd_gpu_argument_storage_t* retired_arguments;
    // Constant-time append boundary; NULL when no generation awaits retirement.
    iree_hal_amd_gpu_argument_storage_t* retired_arguments_tail;
    // All attached signal backings, visited only by cold trim and shutdown.
    iree_hal_amd_gpu_signal_storage_t* signal_storage;
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
    // Lazy private storage; only the exact publisher mutates active ownership.
    iree_hal_amd_gpu_pm4_scratch_t scratch;
  } producer;
  // Single completion observer's release-published capacity credit.
  struct {
    // Number of consumed retirement records.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_atomic_int64_t count;
    // Reclaimable command prefix established by the last retired action.
    iree_atomic_int64_t words;
    // Last consumed argument position, including wrap/alignment.
    iree_atomic_int64_t arguments;
    // Publisher enrolled for capacity return. Only a capacity-blocked
    // publisher writes this observer-owned line; retirement consumes it once.
    iree_atomic_int32_t waiting_for_capacity;
  } completed;
  // Native host-visible records, immutable after preparation.
  iree_hal_amd_fabric_memory_t control;
  // Indexed native record geometry shared by publisher and observer.
  iree_hal_amd_gpu_pm4_retirement_ring_t retirement;
  // Fixed host actions and inline captures parallel to native slots.
  // GPU-carried pointers name these actions directly on normal retirement.
  iree_hal_amd_gpu_pm4_retirement_slot_t* issued;
};

static_assert(
    sizeof(((iree_hal_amd_gpu_pm4_queue_t*)0)->completed) ==
        iree_hardware_destructive_interference_size,
    "completion credit and enrollment occupy one observer cache line");

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_queue_vtable;

static iree_status_t iree_hal_amd_gpu_pm4_file_submit(
    iree_hal_queue_t* base, const iree_hal_transfer_operation_t* operation,
    const void* before, const void* after,
    iree_hal_amd_gpu_retirement_callback_t callback);

static void iree_hal_amd_gpu_pm4_file_add_effects(
    iree_hal_queue_t* base, iree_hal_memory_effects_t effects, void* value) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_pm4_barrier_t* barrier = value;
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM)) {
    barrier->cache_control |= queue->options.barriers.release;
    barrier->execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE |
                          IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  }
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM)) {
    barrier->cache_control |= queue->options.barriers.acquire;
  }
}

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

// Consumes an operation failure through its public signals and diagnostic sink.
// This is also the terminal diagnostic boundary for a call without signals.
static void iree_hal_amd_gpu_pm4_fail_submission(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_context_report_status(
      queue->context, queue->base.queue_family->device, status);
  iree_hal_amd_gpu_pm4_submission_fail(submission, status);
  iree_hal_semaphore_list_fail(submission->signals, status);
}

static void iree_hal_amd_gpu_pm4_abandon(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
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
  return iree_hal_amd_gpu_argument_storage_allocate(
      queue->context, queue->options.memory, capacity, 0, queue->host_allocator,
      &queue->producer.arguments);
}

static iree_status_t iree_hal_amd_gpu_pm4_prepare_signals(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_signal_storage_allocate(
      queue->context, &queue->signals, queue->options.memory,
      queue->options.retirement_capacity, queue->producer.frontier.count,
      queue->host_allocator, &storage));
  storage->next = queue->producer.signal_storage;
  queue->producer.signal_storage = storage;
  return iree_ok_status();
}

// The exact publisher retires only the ordered completed generation prefix.
// Returning native backing is asynchronous and already has release storage.
static void iree_hal_amd_gpu_pm4_trim_arguments(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  while (queue->producer.retired_arguments &&
         queue->producer.retired_arguments->end <= completed) {
    iree_hal_amd_gpu_argument_storage_t* storage =
        queue->producer.retired_arguments;
    queue->producer.retired_arguments = storage->next;
    iree_hal_amd_gpu_argument_storage_free(queue->context, storage);
  }
  if (!queue->producer.retired_arguments) {
    queue->producer.retired_arguments_tail = NULL;
  }
}

static void iree_hal_amd_gpu_pm4_trim_storage(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  iree_hal_amd_gpu_capture_pool_trim(&queue->captures);
  iree_hal_amd_gpu_pm4_trim_arguments(queue);
  iree_hal_amd_gpu_pm4_scratch_trim(
      queue->context,
      (uint64_t)iree_atomic_load(&queue->completed.count,
                                 iree_memory_order_acquire),
      &queue->producer.scratch);
  iree_hal_amd_gpu_signal_storage_t** link = &queue->producer.signal_storage;
  while (*link) {
    iree_hal_amd_gpu_signal_storage_t* storage = *link;
    // Keep one provisioned block. Detachment joins the last reader and removes
    // unused capacity from acquisition before its native release is submitted.
    if (storage != queue->producer.signal_storage &&
        iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block)) {
      *link = storage->next;
      iree_hal_amd_gpu_signal_storage_free(queue->context, storage);
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
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SCRATCH,
} iree_hal_amd_gpu_pm4_publication_result_t;

// Native acceptance is irrevocable here. Finish every capture access before
// exposing the retirement record, transferring the final publication reference
// directly to completion. Only a piece with an unpublished suffix needs another
// reference. No capture access follows count publication or native
// notification.
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
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation =
        &submission->execute;
    iree_hal_amd_gpu_program_publish(
        &invocation->program->resident, invocation->instance,
        queue->options.queue_ordinal, position + 1);
    invocation->published = true;
  }
  if (submission->signal) {
    iree_hal_amd_gpu_signal_publish_frontier(
        submission->signal, queue->producer.frontier.published);
  }
  if (submission->profiled) {
    iree_hal_amd_gpu_profile_operation_record(
        queue->profile, iree_hal_amd_gpu_pm4_submission_profile(submission),
        true, submission->native_wait_count);
  }
  if (!terminal) {
    iree_hal_amd_gpu_pm4_submission_retain(submission);
  }
  queue->issued[position & queue->retirement.capacity_mask].action.scratch =
      iree_hal_amd_gpu_pm4_scratch_publish(
          &queue->producer.scratch, position + 1,
          iree_hal_amd_gpu_pm4_submission_scratch_required(submission));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       terminal && submission->signal && i < submission->signals.count &&
       iree_status_is_ok(status);
       ++i) {
    if (iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i])) {
      status = iree_hal_amd_gpu_semaphore_publish_signal(
          submission->signals.semaphores[i],
          submission->signals.payload_values[i], submission->signal);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_fail_submission(queue, submission, status);
  }
  const bool has_host_owner = terminal && submission->has_host_owner;
  if (has_host_owner) {
    submission->has_host_owner = false;
  }
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
  // Native ownership becomes observable before deferred host ownership ends.
  // The publishing caller's trailing accesses remain under publisher ownership.
  if (has_host_owner) {
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
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
      &queue->issued[position & queue->retirement.capacity_mask].action;
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
    iree_hal_amd_gpu_download_t download, bool terminal) {
  iree_hal_amd_gpu_pm4_retirement_t* action =
      &queue->issued[position & queue->retirement.capacity_mask].action;
  *action = (iree_hal_amd_gpu_pm4_retirement_t){
      .submission = submission,
      .consumed_arguments = argument_end,
      .download = download,
      .terminal = terminal,
  };
  words = iree_hal_amd_gpu_pm4_barrier_emit_tail(words, after);
  if (submission->signal) {
    words = iree_hal_amd_gpu_pm4_emit_release32(
        words, 0, submission->signal->device_address, 1);
  }
  words = iree_hal_amd_gpu_pm4_retirement_emit(words, &queue->retirement,
                                               position, action, 0);
  if (queue->kernel) {
    if (entry_word_count) {
      // The resident IB chains into this fixed 32-DWORD continuation. Root
      // padding precedes its terminal CHAIN; only continuation padding is here.
      const uint32_t padding =
          body_words - (uint32_t)(words - reservation->words);
      if (padding) {
        words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
      }
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
  iree_hal_amd_gpu_transfer_t* transfer = &submission->transfer;
  const bool first = !transfer->cursor.index && !transfer->cursor.offset;
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
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
  // Every preceding publication drains its own CP DMA before retirement.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
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
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words =
      wait_words + chunk.word_count +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
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
    submission->signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 0);
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
  iree_hal_amd_gpu_program_t* program = &command->resident;
  iree_hal_amd_gpu_profile_operation_t* profile =
      iree_hal_amd_gpu_pm4_submission_profile(submission);
  const uint64_t sample_length = profile ? (uint64_t)profile->completion.count *
                                               profile->completion.sample_stride
                                         : 0;
  const uint64_t argument_length = command->argument_length + sample_length;
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  if (argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
  }
  uint64_t relative = iree_align_uint64(arguments->end - arguments->begin, 8);
  uint64_t offset = relative & (arguments->capacity - 1);
  if (argument_length > arguments->capacity - offset) {
    relative += arguments->capacity - offset;
    offset = 0;
  }
  const uint64_t start = arguments->begin + relative;
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  const uint64_t reusable = completed >= arguments->end
                                ? start
                                : iree_max(completed, arguments->begin);
  if (start - reusable + argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (!invocation->instance) {
    invocation->instance = iree_hal_amd_gpu_program_try_acquire(
        program, queue->options.queue_ordinal,
        queue->producer.frontier.pending);
    if (!invocation->instance) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM;
    }
  }
  iree_hal_amd_gpu_program_instance_t* instance = invocation->instance;
  const uint32_t program_words = iree_hal_amd_gpu_pm4_program_word_count(
      program, instance, &queue->options.barriers);
  const uint32_t scratch_words = iree_hal_amd_gpu_pm4_scratch_word_count(
      &queue->producer.scratch, command->scratch_wave_byte_length,
      queue->kernel != NULL);
  const uint32_t prefix_words =
      submission->native_wait_count * 7 + scratch_words + program_words - 4;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const iree_hal_amd_gpu_pm4_barrier_t after = {
      .cache_control = sample_length ? queue->options.barriers.release : 0,
  };
  const uint32_t tail_words =
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      (native_signal ? 8 : 0) + 14;
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
    submission->signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 0);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  const uint64_t dispatch_id = queue->producer.dispatch_count;
  const uint64_t continuation =
      queue->kernel
          ? queue->kernel->address +
                (reservation.words - queue->producer.ring.words + entry_words) *
                    4
          : 0;
  uint8_t* upload = arguments->memory.contents.data + offset;
  const uint64_t sample_address =
      sample_length ? arguments->address + offset + command->argument_length
                    : 0;
  if (sample_length) {
    profile->completion.samples = (void*)(upload + command->argument_length);
  }
  iree_status_t status = iree_hal_amd_gpu_pm4_command_invocation_write(
      invocation, queue->options.queue_ordinal, dispatch_id, continuation,
      sample_address, upload);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  arguments->end = start + argument_length;
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
  words = iree_hal_amd_gpu_pm4_scratch_emit(words, &queue->producer.scratch,
                                            scratch_words);
  words = iree_hal_amd_gpu_pm4_program_emit(
      program, instance, command->entry, &queue->options.barriers,
      arguments->address + offset, (void*)upload, root_padding, words);
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, entry_words, words,
      after, arguments->end, (iree_hal_amd_gpu_download_t){0}, true);
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
                                              queue->producer.frontier.pending);
      }
    }
  }
}

void iree_hal_amd_gpu_pm4_queue_notify_failure(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                      queue->options.queue_ordinal);
}

// Borrowed inputs to one dispatch/barrier publication attempt. Existing
// captures and inline API calls use the same packet writer. The writer creates
// retirement ownership only after dependencies, addresses, and capacity permit
// publishing a complete piece.
typedef struct iree_hal_amd_gpu_pm4_dispatch_publication_t {
  // Borrowed argument/launch inputs, valid until this attempt returns.
  const iree_hal_amd_gpu_pm4_dispatch_request_t* dispatch;
  // Admitted dependencies; native readers transfer only on successful capture.
  const iree_hal_amd_gpu_pm4_submission_wait_t* waits;
  // Number of entries in waits, including already-completed dependencies.
  iree_host_size_t wait_count;
  // Number of native readers requiring a device wait.
  uint32_t native_wait_count;
  // Borrowed output timelines and values, retained before native acceptance.
  iree_hal_semaphore_list_t signals;
  // Prepared payload acquisition boundary.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared terminal payload release boundary.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // Optional attribution, copied into retirement ownership before emission.
  iree_hal_amd_gpu_profile_operation_t* profile;
} iree_hal_amd_gpu_pm4_dispatch_publication_t;

static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_dispatch(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t position,
    uint32_t scratch_required,
    const iree_hal_amd_gpu_pm4_dispatch_publication_t* request,
    uint64_t* addresses, iree_hal_amd_gpu_pm4_submission_t** inout_submission,
    iree_status_t* out_status) {
  const iree_hal_amd_gpu_pm4_dispatch_request_t* dispatch_request =
      request->dispatch;
  const iree_hal_amd_gpu_pm4_function_t* function = dispatch_request->function;
  const bool indirect_parameters =
      function &&
      iree_hal_dispatch_uses_indirect_parameters(dispatch_request->flags);
  const bool dispatch =
      function && (indirect_parameters ||
                   (dispatch_request->launch.geometry.workgroup_count[0] &&
                    dispatch_request->launch.geometry.workgroup_count[1] &&
                    dispatch_request->launch.geometry.workgroup_count[2]));
  const bool indirect =
      iree_hal_dispatch_uses_indirect_arguments(dispatch_request->flags);
  const iree_hal_amd_gpu_pm4_dispatch_storage_t argument_storage =
      dispatch ? iree_hal_amd_gpu_pm4_dispatch_storage(function,
                                                       dispatch_request->flags)
               : (iree_hal_amd_gpu_pm4_dispatch_storage_t){0};
  iree_hal_amd_gpu_profile_operation_t* profile = request->profile;
  const uint64_t sample_length = profile ? (uint64_t)profile->completion.count *
                                               profile->completion.sample_stride
                                         : 0;
  const uint64_t sample_offset =
      sample_length ? iree_align_uint64(argument_storage.length, 8) : 0;
  const uint64_t argument_length =
      sample_length ? sample_offset + sample_length : argument_storage.length;
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  uint64_t argument_offset = 0;
  uint64_t argument_end = arguments->end;
  uint64_t argument_address = 0;
  if (argument_length) {
    const uint64_t alignment = sample_length
                                   ? iree_max(argument_storage.alignment, 8u)
                                   : argument_storage.alignment;
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
  iree_hal_amd_gpu_pm4_barrier_t before = request->before;
  // Transfer chunks and resident-program exits join their own CP DMA. Direct
  // dispatch setup also joins its copies before launching, so neither boundary
  // of this publication has outstanding CP DMA to drain.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  iree_hal_amd_gpu_pm4_barrier_t after = request->after;
  after.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  if (sample_length) {
    after.cache_control |= queue->options.barriers.release;
  }
  if (dispatch) {
    // Instruction upload and host argument publication remain mandatory even
    // when the caller supplies an empty optional payload boundary.
    before.cache_control |= 1u | queue->options.barriers.acquire;
    if (function->launch.requirements &
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_TSAN) {
      before.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
      before.cache_control |= queue->options.barriers.shader;
    }
  }
  const uint16_t indirect_copy_count =
      dispatch && indirect_parameters && !indirect
          ? function->base.kernarg->indirect_copy_count
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
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < request->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(request->signals.semaphores[i]);
  }
  const uint32_t scratch_words = iree_hal_amd_gpu_pm4_scratch_word_count(
      &queue->producer.scratch, scratch_required, queue->kernel != NULL);
  const uint32_t body_words =
      request->native_wait_count * 7 + launch_words + scratch_words +
      (sample_length ? 16 : 0) +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  uint64_t parameters_address = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < dispatch_request->bindings.count &&
       iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        dispatch_request->bindings.values[i],
        indirect ? IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS
                 : IREE_HAL_BUFFER_USAGE_STORAGE |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
        indirect ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (indirect_parameters && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_QUEUE,
        dispatch_request->parameters,
        IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        IREE_HAL_MEMORY_ACCESS_READ, &parameters_address);
    if (iree_status_is_ok(status) && (parameters_address & 3)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect parameter address is misaligned");
    }
  }
  if (dispatch && indirect && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (function->base.kernarg->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect kernarg address is misaligned");
    }
  }
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  iree_hal_amd_gpu_signal_t* signal = NULL;
  if (native_signal) {
    signal = iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 0);
    if (!signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = *inout_submission;
  if (!submission) {
    iree_hal_amd_gpu_pm4_retirement_slot_t* slot =
        &queue->issued[position & queue->retirement.capacity_mask];
    status = iree_hal_amd_gpu_pm4_submission_create_retirement(
        &queue->captures,
        iree_make_byte_span(slot->capture, sizeof(slot->capture)),
        dispatch_request, request->wait_count, request->waits,
        request->native_wait_count, request->signals, request->before,
        request->after, profile, &submission);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_signal_release(signal);
      *out_status = status;
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
    *inout_submission = submission;
    profile = iree_hal_amd_gpu_pm4_submission_profile(submission);
  }
  submission->signal = signal;
  arguments->end = argument_end;
  // Reservation commits only the advancing position; ring geometry is fixed.
  queue->producer.ring.reserved_words = candidate.reserved_words;
  if (argument_storage.length) {
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        function, &dispatch_request->launch, dispatch_request->constants,
        addresses, dispatch_request->flags,
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
  for (iree_host_size_t i = 0; i < request->wait_count; ++i) {
    if (request->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, request->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  words = iree_hal_amd_gpu_pm4_scratch_emit(words, &queue->producer.scratch,
                                            scratch_words);
  if (dispatch) {
    if (prepare_packet) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
          words, function->packet->prepare, packet_address + 64,
          queue->options.barriers.shader);
    }
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
        .dispatch_pointer = packet_address,
        .kernarg_pointer = argument_address,
        .dispatch_id = queue->producer.dispatch_count++,
        .private_segment_size =
            (function->base.metadata->private_segment_fixed_size + 3u) & ~3u};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch,
        dispatch_request->launch.geometry.workgroup_size,
        dispatch_request->launch.geometry.partial_size,
        dispatch_request->launch.geometry.group_segment_byte_length, &inputs);
    const uint32_t initiator =
        function->launch.dispatch_initiator |
        (dispatch_request->launch.geometry.partial_size[0] ? UINT32_C(1) << 6
                                                           : 0);
    const uint64_t sample_address =
        arguments->address + argument_offset + sample_offset;
    if (sample_length) {
      profile->completion.samples = (void*)(arguments->memory.contents.data +
                                            argument_offset + sample_offset);
      words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
      words = iree_hal_amd_gpu_pm4_emit_clock(words, sample_address);
    }
    if (indirect_parameters) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
          words, function->base.kernarg, indirect_copy_count,
          parameters_address, argument_address,
          queue->options.barriers.command_processor_bypasses_l2
              ? queue->options.barriers.acquire
              : queue->options.barriers.shader,
          initiator);
    } else {
      words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
          words, dispatch_request->launch.geometry.workgroup_count, initiator);
    }
    if (sample_length) {
      words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
      words = iree_hal_amd_gpu_pm4_emit_clock(words, sample_address + 8);
    }
  }
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      argument_end, (iree_hal_amd_gpu_download_t){0}, true);
}

static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_timestamp(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  uint64_t target = 0;
  iree_status_t status = iree_hal_amd_gpu_buffer_reference_resolve(
      &submission->timestamp, &target);
  if (iree_status_is_ok(status) && (target & 7u)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "timestamp GPU address must be 8-byte aligned");
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  iree_hal_amd_gpu_pm4_barrier_t before = submission->before;
  iree_hal_amd_gpu_pm4_barrier_t after = submission->after;
  // Every earlier publication joins its own DMA before leaving the CP.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  after.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words =
      submission->native_wait_count * 7 + 6 +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
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
    submission->signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&queue->signals, 0);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  queue->producer.ring.reserved_words = candidate.reserved_words;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  words = iree_hal_amd_gpu_pm4_emit_clock(words, target);
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      queue->producer.arguments->end, (iree_hal_amd_gpu_download_t){0}, true);
}

// Publishes an owned capture. Immediate retirement may free it before return.
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
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
    return iree_hal_amd_gpu_pm4_publish_transfer(queue, submission, position);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    return iree_hal_amd_gpu_pm4_publish_execute(queue, submission, position);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TIMESTAMP) {
    return iree_hal_amd_gpu_pm4_publish_timestamp(queue, submission, position);
  }
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags);
  const iree_host_size_t binding_count = submission->dispatch.bindings.count;
  const iree_hal_amd_gpu_pm4_dispatch_request_t dispatch = {
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
  uint64_t* addresses = (uint64_t*)(submission->dispatch.bindings.values +
                                    binding_count + indirect_parameters);
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

static iree_status_t iree_hal_amd_gpu_pm4_admit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
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
        submission->transfer.host_wait_required);
  return iree_hal_amd_gpu_pm4_admit_waits(
      queue, submission->wait_count, submission->waits, native_eligible,
      &submission->native_wait_count, out_wait_index, out_kind);
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

static void iree_hal_amd_gpu_pm4_pending_resume(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_pm4_pending_t* pending = user_data;
  iree_hal_amd_gpu_pm4_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_pm4_pending_release(pending);
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
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
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
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_amd_gpu_pm4_pending_release(pending);
  iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
  iree_hal_amd_gpu_pm4_pending_end(queue);
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

// The exact publisher installs acquired backing and requeues the waiting
// cohort. Native callbacks never enter this function or visit its captures.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_process_entry(iree_hal_amd_gpu_pm4_queue_t* queue,
                                   iree_hal_amd_gpu_publication_entry_t* link) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_pm4_publish");
  iree_hal_amd_gpu_pm4_entry_t* entry =
      iree_containerof(link, iree_hal_amd_gpu_pm4_entry_t, publication);
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
    iree_hal_amd_gpu_pm4_growth_result_t result;
    iree_status_t status = iree_hal_amd_gpu_pm4_growth_end(
        entry, queue->producer.arguments->end, &result);
    if (iree_status_is_ok(status)) {
      if (result.arguments) {
        iree_hal_amd_gpu_argument_storage_t* previous =
            queue->producer.arguments;
        if (queue->producer.retired_arguments_tail) {
          queue->producer.retired_arguments_tail->next = previous;
        } else {
          queue->producer.retired_arguments = previous;
        }
        queue->producer.retired_arguments_tail = previous;
        queue->producer.arguments = result.arguments;
        iree_hal_amd_gpu_pm4_trim_arguments(queue);
      }
      if (result.scratch) {
        iree_hal_amd_gpu_pm4_scratch_install(
            queue->context, result.scratch,
            (uint64_t)iree_atomic_load(&queue->completed.count,
                                       iree_memory_order_acquire),
            &queue->producer.scratch);
      }
      if (result.signals) {
        result.signals->next = queue->producer.signal_storage;
        queue->producer.signal_storage = result.signals;
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
  iree_hal_amd_gpu_pm4_submission_t* submission =
      iree_containerof(entry, iree_hal_amd_gpu_pm4_submission_t, entry);
  iree_hal_amd_gpu_pm4_publication_result_t result =
      iree_hal_amd_gpu_pm4_process(queue, submission);
  if (result < IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS) {
    IREE_TRACE_ZONE_END(z0);
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
    } else if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
      required = submission->transfer.staging_capacity;
    } else {
      const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
          iree_hal_amd_gpu_pm4_dispatch_storage(submission->dispatch.function,
                                                submission->dispatch.flags);
      required = iree_max(storage.length, storage.alignment);
    }
    iree_hal_amd_gpu_profile_operation_t* profile =
        iree_hal_amd_gpu_pm4_submission_profile(submission);
    if (profile && profile->completion.sample_stride) {
      required = iree_align_uint64(required, 8) +
                 (uint64_t)profile->completion.count *
                     profile->completion.sample_stride;
    }
    capacity = queue->producer.arguments->capacity;
    while (capacity < required) {
      capacity *= 2;
    }
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
  IREE_TRACE_ZONE_END(z0);
  return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
}

// Enrollment follows the final publication access. The exchanges order this
// enrollment against retirement: either retirement observes the enrollment, or
// enrollment acquires its returned credit before rechecking the predicate.
static bool iree_hal_amd_gpu_pm4_handoff(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* remainder, bool capacity,
    uint64_t completed) {
  bool more = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, remainder);
  if (more && capacity) {
    iree_atomic_exchange(&queue->completed.waiting_for_capacity, 1,
                         iree_memory_order_acq_rel);
    more = (uint64_t)iree_atomic_load(&queue->completed.count,
                                      iree_memory_order_acquire) != completed;
    // Only retirement consumes enrollment. Publication ownership has already
    // transferred, so clearing here could erase the next publisher's wait.
  }
  return more;
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
                                        queue->options.queue_ordinal);
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
    iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  }
  iree_status_t failure = iree_hal_amd_gpu_pm4_check_failure(queue);
  if (!iree_status_is_ok(failure)) {
    iree_hal_amd_gpu_host_wait_set_cancel(&queue->pending.waits);
    if (!queue->issued_failure_delivered &&
        iree_atomic_load(&queue->native_failure, iree_memory_order_acquire) &&
        iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      const uint64_t begin = (uint64_t)iree_atomic_load(
          &queue->completed.count, iree_memory_order_acquire);
      const uint64_t end = (uint64_t)iree_atomic_load(
          &queue->producer.count, iree_memory_order_acquire);
      for (uint64_t i = begin; i != end; ++i) {
        iree_hal_amd_gpu_pm4_submission_t* submission =
            queue->issued[i & queue->retirement.capacity_mask]
                .action.submission;
        iree_hal_amd_gpu_pm4_fail_submission(queue, submission,
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
  iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                      queue->options.queue_ordinal);
}

static bool iree_hal_amd_gpu_pm4_retire(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_wait_t* out_wait) {
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
          queue->issued[i & queue->retirement.capacity_mask].action.submission;
      iree_hal_amd_gpu_pm4_fail_submission(
          queue, submission,
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
    iree_hal_amd_gpu_pm4_submission_t* submission = action->submission;
    iree_status_code_t completion_code = IREE_STATUS_OK;
    if (action->terminal && !iree_hal_amd_gpu_pm4_submission_retirement_ready(
                                submission, &completion_code)) {
      waiting_for_producer = true;
      break;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_gpu_pm4_retire");
    if (queue->options.feedback) {
      // Native retirement makes this dispatch's reports observable. Drain the
      // ready shared prefix before waking host completion consumers.
      iree_hal_amd_gpu_feedback_drain(queue->options.feedback);
      if (completion_code == IREE_STATUS_OK) {
        completion_code = iree_hal_amd_gpu_pm4_feedback_failure(queue);
      }
    }
    iree_hal_amd_gpu_pm4_scratch_storage_free(queue->context, action->scratch);
    action->scratch = NULL;
    const uint64_t consumed_words = action->consumed_words;
    const uint64_t consumed_arguments = action->consumed_arguments;
    if (action->download.length && feedback_failure == IREE_STATUS_OK &&
        completion_code == IREE_STATUS_OK) {
      memcpy(action->download.target, action->download.source,
             action->download.length);
    }
    if (action->terminal) {
      if (submission->profiled) {
        const iree_hal_amd_gpu_profile_operation_t* operation =
            iree_hal_amd_gpu_pm4_submission_profile(submission);
        iree_hal_amd_gpu_profile_operation_complete(queue->profile, operation,
                                                    &operation->completion);
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
    completion_code = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_fail_submission(queue, submission, status);
    }
    if (action->terminal) {
      iree_hal_amd_gpu_signal_retire(submission->signal, completion_code);
      iree_hal_amd_gpu_pm4_submission_set_frontier(
          submission, frontier ? &completed_frontier : NULL);
    }
    iree_hal_amd_gpu_pm4_submission_release(submission);
    iree_atomic_store(&queue->completed.words, consumed_words,
                      iree_memory_order_release);
    iree_atomic_store(&queue->completed.arguments, consumed_arguments,
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
    iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                        queue->options.queue_ordinal);
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
  if (!queue->kernel && position != published && !waiting_for_producer) {
    // Observe already-published work without touching the publisher's count
    // on every active-wait iteration. An idle queue waits on the native event;
    // a later publication signals that event when it completes. Capacity credit
    // prevents record reuse until this consumer advances it.
    out_wait->address =
        &queue->retirement.records[position & queue->retirement.capacity_mask]
             .generation;
    out_wait->value = (int32_t)(position >> queue->retirement.capacity_shift);
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

void iree_hal_amd_gpu_pm4_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_pm4_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_pm4_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  // Deferred work stays counted until terminal publication or abandonment.
  // Acquire the final publisher handoff before observing native prefixes.
  const bool idle =
      !iree_atomic_load(&queue->pending.host_owners,
                        iree_memory_order_acquire) &&
      !iree_atomic_load(&queue->growth.pending_count,
                        iree_memory_order_acquire) &&
      iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
      iree_atomic_load(&queue->producer.count, iree_memory_order_acquire) ==
          iree_atomic_load(&queue->completed.count, iree_memory_order_acquire);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
  return idle;
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
    if (position - completed < queue->options.retirement_capacity &&
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
        // Dependency failures complete accepted operations through their
        // signals, including when admission runs in the calling thread.
        iree_hal_amd_gpu_profile_operation_record(queue->profile, profile,
                                                  false, 0);
        iree_hal_amd_gpu_context_report_status(
            queue->context, queue->base.queue_family->device, status);
        iree_hal_semaphore_list_fail(signals, status);
        status = iree_ok_status();
        result = IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
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
          queue->kernel->rollback.arguments = queue->producer.arguments->end;
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
                                          queue->options.queue_ordinal);
    }
    iree_hal_amd_gpu_pm4_wake_shutdown(queue);
  }
  return status;
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
    if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile) &&
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
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_user_queue_initialize(
        queue->context, &native, &queue->storage, &mapping, NULL));
    iree_hal_amd_gpu_pm4_ring_initialize(
        (uint32_t*)(uintptr_t)mapping.ring_address,
        (uint32_t)(mapping.ring_byte_length / sizeof(uint32_t)),
        &queue->producer.ring);
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

static amdf_status_t iree_hal_amd_gpu_pm4_cleanup(
    iree_hal_amd_gpu_pm4_queue_t* queue) {
  const amdf_status_t status =
      queue->kernel ? iree_hal_amd_gpu_pm4_kmq_storage_free(queue->kernel)
                    : iree_hal_amd_gpu_user_queue_deinitialize(
                          &queue->storage, queue->producer.ring.reserved_words);
  if (!amdf_status_is_ok(status)) {
    queue->context->cleanup_blocked = true;
    return status;
  }
  queue->kernel = NULL;
  iree_hal_amd_gpu_pm4_scratch_trim(queue->context, UINT64_MAX,
                                    &queue->producer.scratch);
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
  queue->producer.arguments = NULL;
  queue->producer.retired_arguments = NULL;
  queue->producer.retired_arguments_tail = NULL;
  while (arguments) {
    iree_hal_amd_gpu_argument_storage_t* next = arguments->next;
    iree_hal_amd_gpu_argument_storage_free(queue->context, arguments);
    arguments = next;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->control);
  return AMDF_STATUS_OK;
}

amdf_status_t iree_hal_amd_gpu_pm4_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_pm4_queue_is_idle(base),
              "shutdown requires joined accepted work");
  return iree_hal_amd_gpu_pm4_cleanup(queue);
}

static void iree_hal_amd_gpu_pm4_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_ASSERT(!queue->storage.queue && !queue->kernel && !queue->control.handle,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_hal_amd_gpu_signal_pool_deinitialize(&queue->signals);
  iree_hal_amd_gpu_host_wait_set_deinitialize(&queue->pending.waits);
  iree_slim_mutex_deinitialize(&queue->pending.completion_mutex);
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
  const uint32_t queue_count = family->spec->provisioned_queue_count;
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
  queue->producer.scratch.profile = options->scratch_profile;
  queue->pending_size = pending_size;
  queue->trim.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_TRIM;
  iree_hal_amd_gpu_signal_pool_initialize(
      (iree_hal_amd_gpu_signal_pool_notification_t){
          .fn = iree_hal_amd_gpu_pm4_growth_signals_available,
          .user_data = &queue->growth},
      &queue->signals);
  queue->host_executor = (iree_hal_amd_gpu_host_call_executor_t){
      .queue = &queue->base,
      .context = context,
      .captures = &queue->captures,
      .begin = iree_hal_amd_gpu_pm4_host_begin,
      .end = iree_hal_amd_gpu_pm4_host_end,
      .barrier = iree_hal_amd_gpu_pm4_host_barrier};
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
      .barrier_size = sizeof(iree_hal_amd_gpu_pm4_barrier_t),
      .add_effects = iree_hal_amd_gpu_pm4_file_add_effects,
      .submit = iree_hal_amd_gpu_pm4_file_submit};
  queue->issued = (iree_hal_amd_gpu_pm4_retirement_slot_t*)(queue + 1);
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
  iree_hal_amd_gpu_host_wait_set_initialize(&queue->wait_context,
                                            &queue->pending.waits);
  iree_slim_mutex_initialize(&queue->pending.completion_mutex);
  queue->growth = (iree_hal_amd_gpu_pm4_growth_context_t){
      .context = context,
      .memory = options->memory,
      .scratch_memory = options->scratch_memory,
      .scratch_profile = &queue->producer.scratch.profile,
      .captures = &queue->captures,
      .publisher = &queue->publisher,
      .publication = options->publication,
      .queue_ordinal = options->queue_ordinal,
      .queue_count = queue_count,
      .signal_capacity = options->retirement_capacity,
      .signal_pool = &queue->signals,
      .reuse.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS,
  };
  iree_hal_amd_gpu_capture_pool_initialize(options->capture_block_size,
                                           host_allocator, &queue->captures);
  iree_status_t status = iree_hal_amd_gpu_capture_pool_preallocate(
      &queue->captures, options->capture_block_count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_prepare_native(queue);
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
  if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile) &&
      submission->execute.program->dispatch_count &&
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
    if (iree_hal_amd_gpu_profile_captures_dispatches(queue->profile)) {
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
  const iree_hal_amd_gpu_pm4_queue_t* pm4_queue = (void*)queue;
  const amdf_queue_format_features_t features =
      pm4_queue->context->families[pm4_queue->options.family_ordinal]
          .format_features;
  const uint32_t scalar_reservation =
      features & AMDF_GPU_PM4_FORMAT_FEATURE_COMPUTE_TRAP_PRESENT ? 16 : 0;
  return iree_hal_amd_gpu_executable_query_dispatch_concurrency(
      executable, function, scalar_reservation, params, out_concurrency);
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
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  if (!base->queue_family->spec->timestamp_frequency_hz) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "queue family has no timestamp clock rate");
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_timestamp(
      &queue->captures, queue->profile != NULL, base->queue_family, waits,
      signals, target, offset, &queue->options.barriers, barriers,
      &submission));
  if (queue->profile) {
    iree_hal_amd_gpu_profile_operation_initialize(
        queue->profile, queue->options.queue_ordinal,
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TIMESTAMP, waits.count, signals.count,
        1, iree_hal_amd_gpu_pm4_submission_profile(submission));
  }
  iree_hal_amd_gpu_pm4_submit(queue, submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_flush(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_publication_notify(queue->options.publication,
                                      queue->options.queue_ordinal);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
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
    const void* before, const void* after,
    iree_hal_amd_gpu_retirement_callback_t callback) {
  iree_hal_amd_gpu_pm4_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create_file(
      &queue->captures, base->queue_family, operation, before, after, callback,
      &submission));
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
