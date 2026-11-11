// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/native_queue.h"

#include "iree/hal/driver/amd/fabric/memory_transition.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

#define IREE_HAL_AMD_GPU_SDMA_CAPACITY IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY
#define IREE_HAL_AMD_GPU_SDMA_RING_LENGTH 65536u
#define IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE 1024u
#define IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS \
  (IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS + 4u + 6u)

enum {
  IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE = 1u << 0,
  IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED = 1u << 1,
};

amdf_queue_publication_modes_t
iree_hal_amd_gpu_sdma_native_queue_publication_modes(
    const amdf_queue_family_info_t* family) {
  if (family->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA ||
      family->format_version != AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 ||
      !(family->roles & AMDF_QUEUE_ROLE_TRANSFER)) {
    return 0;
  }
  amdf_queue_publication_modes_t modes =
      family->publication_modes &
      (AMDF_QUEUE_PUBLICATION_MODE_USER | AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  if (!(family->user_queue_capabilities &
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
    modes &= ~AMDF_QUEUE_PUBLICATION_MODE_USER;
  }
  return modes;
}

// At most CAPACITY publications can remain live. A complete wrap also leaves
// room for the last retired publication's notification tail and wrap padding.
// Thus publication never needs a read-pointer syscall or a delayed writeback
// wake to reclaim primary-ring storage. KERNEL commands have separate slots
// retained through their native submission points.
static_assert(
    (IREE_HAL_AMD_GPU_SDMA_CAPACITY + 2u) *
            IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS * sizeof(uint32_t) <
        IREE_HAL_AMD_GPU_SDMA_RING_LENGTH,
    "the primary ring must dominate the complete live command window");
static_assert(IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS * sizeof(uint32_t) <=
                  IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE,
              "a complete native span must fit its immutable KERNEL slot");

typedef struct iree_hal_amd_gpu_sdma_record_t {
  // Per-span retirement owner, independent of reusable publication storage.
  iree_hal_amd_gpu_sdma_completion_t completion;
  // Opaque KERNEL retirement point, or zero for USER publication.
  uint64_t point;
} iree_hal_amd_gpu_sdma_record_t;

struct iree_hal_amd_gpu_sdma_native_queue_t {
  // Parent-owned native domain, live through final native removal.
  iree_hal_amd_gpu_context_t* context;
  // Allocator for this cold owner and its fixed retirement array.
  iree_allocator_t host_allocator;
  // Existing device worker accepting this exact source's readiness bit.
  iree_hal_amd_gpu_publication_t* publication;
  // Stable source index assigned by device construction.
  uint32_t publication_ordinal;
  // Exact family encoding features, captured before queue creation.
  amdf_queue_format_features_t features;
  // First native error; it does not establish quiescence of accepted work.
  iree_atomic_int64_t failure;
  // The deferred publication source has reported every unretired submission
  // once.
  bool failure_delivered;
  // Sole caller/deferred native producer, isolated from completion writes.
  iree_hal_amd_gpu_publisher_t publisher;
  // Native waiters cannot prevent metadata from reaching the publisher. Their
  // order preserves the first prepared KERNEL span until native acceptance.
  struct {
    // First waiter; only the publisher mutates it, cold idle queries observe
    // it.
    iree_atomic_intptr_t first;
    // Last waiter for constant-time append under publication ownership.
    iree_hal_amd_gpu_publication_entry_t* last;
    // Coalesced retirement/failure continuation on the existing publisher.
    iree_hal_amd_gpu_sdma_metadata_t resume;
    // One while the resume entry is linked or executing.
    iree_atomic_int32_t pending;
  } capacity;
  // Release-published retirement records, independent of native progress.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int64_t published;
  // Completion-owned prefix returning both command and record capacity.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int64_t retired;
  // Publisher parks only after enrolling this native-capacity return.
  iree_atomic_int32_t waiting;
  // Joins the current completion scan before publication reports native
  // failure. Shares the completion writer's row, not the failure reader's.
  iree_atomic_int32_t observer;
  // Constant-size SPSC records, indexed by publication sequence. Publication
  // of a new record does not invalidate the completion counter's row.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_hal_amd_gpu_sdma_record_t records[IREE_HAL_AMD_GPU_SDMA_CAPACITY];
  // Qualified control backing and KERNEL command storage, never payloads.
  iree_hal_amd_fabric_memory_t memory;
  // Stable GPU address of the completion DWORD at the end of control storage.
  uint64_t completion_address;
  // Coherent GPU-written 32-bit publication sequence.
  iree_atomic_int32_t* completion;
  // Command base: mapped primary ring or owned KERNEL command backing.
  uint32_t* commands;
  // USER native ownership and direct producer operands.
  struct {
    // Shared native ownership and host publication policy.
    iree_hal_amd_gpu_user_queue_t storage;
    // Borrowed native monotonic byte-count index.
    iree_atomic_int64_t* write_index;
    // Borrowed write-only 64-bit MMIO doorbell.
    volatile uint64_t* doorbell;
    // Last published byte frontier, joined including its notification tail.
    uint64_t position;
  } user;
  // KERNEL ownership and immutable command retirement.
  struct {
    // Owned native queue, absent for USER transport.
    amdf_kernel_queue_t* queue;
    // Last accepted opaque submission point, joined during cold teardown.
    uint64_t last_submission;
    // Native command admission limit, bounded by the local record window.
    uint32_t capacity;
  } kernel;
  // Sticky removal error after a native destroy consumed its queue handle.
  amdf_status_t removal_status;
  // Successful native stop replaces consumption/submission waits at teardown.
  bool inactivated;
};

static void iree_hal_amd_gpu_sdma_capacity_available(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  if (!iree_atomic_exchange(&queue->capacity.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_sdma_native_queue_defer_metadata(queue,
                                                      &queue->capacity.resume);
  }
}

amdf_status_t iree_hal_amd_gpu_sdma_native_queue_query_status(
    const iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return (amdf_status_t)iree_atomic_load(&queue->failure,
                                         iree_memory_order_acquire);
}

void iree_hal_amd_gpu_sdma_native_queue_fail(
    iree_hal_amd_gpu_sdma_native_queue_t* queue, amdf_status_t status) {
  int64_t expected = 0;
  const bool first = iree_atomic_compare_exchange_strong(
      &queue->failure, &expected, (int64_t)status, iree_memory_order_release,
      iree_memory_order_relaxed);
  iree_atomic_fetch_or(&queue->observer, IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED,
                       iree_memory_order_acq_rel);
  iree_hal_amd_gpu_publication_notify(queue->publication,
                                      queue->publication_ordinal);
  if (first) {
    if (iree_atomic_exchange(&queue->waiting, 0, iree_memory_order_acq_rel)) {
      iree_hal_amd_gpu_sdma_capacity_available(queue);
    }
    iree_hal_amd_gpu_context_fail_execution(queue->context, status,
                                            "SDMA execution/publication");
  }
}

// Only the deferred source reports failures after the completion owner stops.
// It owns publication or has observed its idle handoff after the failure;
// subsequent callers can only reject commands and cannot mutate these records.
// Failed notification delivery therefore does not depend on another native
// wake.
static void iree_hal_amd_gpu_sdma_native_queue_fail_submissions(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  int32_t expected = IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED;
  if (!iree_atomic_compare_exchange_strong(
          &queue->observer, &expected,
          IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED |
              IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
          iree_memory_order_acquire, iree_memory_order_relaxed)) {
    return;
  }
  const amdf_status_t status = (amdf_status_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  const uint64_t begin =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  const uint64_t end =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_relaxed);
  for (uint64_t i = begin; !queue->failure_delivered && i != end; ++i) {
    const iree_hal_amd_gpu_sdma_completion_t completion =
        queue->records[i & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1)].completion;
    completion.notify(completion.user_data, status,
                      IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
  }
  queue->failure_delivered = true;
  iree_atomic_fetch_and(&queue->observer,
                        ~IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
                        iree_memory_order_release);
  iree_hal_amd_gpu_context_wake_failure(queue->context);
}

static uint32_t iree_hal_amd_gpu_sdma_encode(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    const iree_hal_amd_gpu_sdma_submission_t* submission, uint32_t value,
    uint32_t* words) {
  if (submission->word_count) {
    memcpy(words, submission->words, submission->word_count * sizeof(*words));
  }
  uint32_t* cursor = words + submission->word_count;
  cursor = iree_hal_amd_gpu_sdma_emit_fence(cursor, queue->features,
                                            queue->completion_address, value);
  if (queue->user.storage.queue) {
    const amdf_gpu_event_info_t* event = &queue->context->event_info;
    cursor = iree_hal_amd_gpu_sdma_emit_fence(
        cursor, queue->features, event->mailbox_address, event->trigger_data);
    *cursor++ = 6;
    *cursor++ = event->interrupt_context;
  }
  return (uint32_t)(cursor - words);
}

// False retains this exact span for native capacity. Preparation owns bounded
// packet/staging work under this publisher and runs once across native retries.
static bool iree_hal_amd_gpu_sdma_native_queue_try_publish(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_submission_t* submission) {
  amdf_status_t status = (amdf_status_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  if (!amdf_status_is_ok(status)) {
    submission->completion.notify(submission->completion.user_data, status,
                                  IREE_HAL_AMD_GPU_SDMA_RETIRED);
    return true;
  }
  const uint64_t published =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_relaxed);
  const uint64_t retired =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  const uint32_t capacity = queue->kernel.queue
                                ? queue->kernel.capacity
                                : IREE_HAL_AMD_GPU_SDMA_CAPACITY;
  if (published - retired == capacity) {
    return false;
  }
  if (submission->prepare) {
    if (!submission->prepare(submission->user_data, published)) {
      return true;
    }
    submission->prepare = NULL;
  }
  const iree_hal_amd_gpu_sdma_completion_t completion = submission->completion;
  void (*accepted)(void*) = submission->accepted;
  void* accepted_data = submission->user_data;
  const uint32_t index = published & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1);
  uint64_t point = 0;
  uint32_t words[IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS];
  const uint32_t word_count = iree_hal_amd_gpu_sdma_encode(
      queue, submission, (uint32_t)(published + 1), words);
  if (amdf_status_is_ok(status) && queue->kernel.queue) {
    const uint64_t offset = index * IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE;
    memcpy((uint8_t*)queue->commands + offset, words, word_count * 4u);
    const amdf_gpu_kernel_command_t command = {
        .memory = queue->memory.handle,
        .byte_offset = offset,
        .byte_length = word_count * 4u,
    };
    const amdf_gpu_kernel_queue_submission_info_t native_submission = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(native_submission),
        .command_count = 1,
        .commands = &command,
    };
    status = queue->context->gpu->kernel_queue_submit(
        queue->kernel.queue, &native_submission, &point);
    // Capacity can be deferred only while an accepted submission owns a wake.
    // BUSY on an empty, exclusively owned queue has no progress source.
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY) &&
        published != retired) {
      return false;
    }
    if (amdf_status_is_ok(status)) {
      queue->kernel.last_submission = point;
    }
  }
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_native_queue_fail(queue, status);
    completion.notify(completion.user_data, status,
                      IREE_HAL_AMD_GPU_SDMA_RETIRED);
    return true;
  }
  queue->records[index] = (iree_hal_amd_gpu_sdma_record_t){
      .completion = completion, .point = point};
  if (queue->kernel.queue) {
    // Execution may already be complete. Publish producer metadata before
    // exposing this record to its retirement observer.
    if (accepted) {
      accepted(accepted_data);
    }
    iree_atomic_store(&queue->published, (int64_t)(published + 1),
                      iree_memory_order_release);
    status = queue->context->api->kernel_queue_request_notification(
        queue->kernel.queue, point, &queue->context->kernel_notification);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_native_queue_fail(queue, status);
    }
  } else {
    const uint32_t length = word_count * 4u;
    uint32_t offset =
        queue->user.position & (IREE_HAL_AMD_GPU_SDMA_RING_LENGTH - 1);
    if (offset + length > IREE_HAL_AMD_GPU_SDMA_RING_LENGTH) {
      const uint32_t padding = IREE_HAL_AMD_GPU_SDMA_RING_LENGTH - offset;
      memset((uint8_t*)queue->commands + offset, 0, padding);
      queue->user.position += padding;
      offset = 0;
    }
    memcpy((uint8_t*)queue->commands + offset, words, length);
    queue->user.position += length;
    status = iree_hal_amd_gpu_user_queue_flush(&queue->user.storage);
    if (IREE_UNLIKELY(!amdf_status_is_ok(status))) {
      // Failed release retains the record until checked native cessation.
      iree_atomic_store(&queue->published, (int64_t)(published + 1),
                        iree_memory_order_release);
      iree_hal_amd_gpu_sdma_native_queue_fail(queue, status);
      return true;
    }
    // Every fallible release step has succeeded. The remaining index and
    // doorbell stores commit this complete stream without another native call.
    if (accepted) {
      accepted(accepted_data);
    }
    // Native completion alone does not publish the CPU acceptance metadata.
    // The retirement observer acquires this record before reading that state.
    iree_atomic_store(&queue->published, (int64_t)(published + 1),
                      iree_memory_order_release);
    iree_atomic_store(queue->user.write_index, (int64_t)queue->user.position,
                      iree_memory_order_release);
    const iree_hal_memory_transition_recipe_info_t fence = {
        .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE}};
    iree_hal_memory_transition_execute_host_direct(&fence, NULL, 1);
    *queue->user.doorbell = queue->user.position;
  }
  return true;
}

static void iree_hal_amd_gpu_sdma_await_capacity(
    iree_hal_amd_gpu_sdma_native_queue_t* queue, uint64_t completed) {
  // Either retirement observes enrollment or this recheck acquires its
  // returned credit. The publisher owns the waiting cohort throughout.
  iree_atomic_exchange(&queue->waiting, 1, iree_memory_order_acq_rel);
  if (iree_atomic_load(&queue->failure, iree_memory_order_acquire) ||
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire) !=
          completed) {
    iree_hal_amd_gpu_sdma_capacity_available(queue);
  }
}

static void iree_hal_amd_gpu_sdma_park_capacity(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
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

static void iree_hal_amd_gpu_sdma_resume_capacity(void* user_data,
                                                  amdf_status_t status) {
  iree_hal_amd_gpu_sdma_native_queue_t* queue = user_data;
  iree_atomic_store(&queue->capacity.pending, 0, iree_memory_order_release);
  iree_hal_amd_gpu_publication_entry_t* entry = (void*)iree_atomic_load(
      &queue->capacity.first, iree_memory_order_relaxed);
  if (!entry) {
    return;
  }
  // Detach before acceptance can reuse/free this record. Later native entries
  // stay behind it until this turn finishes; metadata never touches its slot.
  iree_hal_amd_gpu_publication_entry_t* next =
      (void*)iree_atomic_load(&entry->next, iree_memory_order_relaxed);
  iree_atomic_store(&queue->capacity.first, (intptr_t)next,
                    iree_memory_order_release);
  if (!next) {
    queue->capacity.last = NULL;
  }
  const uint64_t completed =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  iree_hal_amd_gpu_sdma_submission_t* submission = iree_containerof(
      entry, iree_hal_amd_gpu_sdma_submission_t, entry.publication);
  if (!iree_hal_amd_gpu_sdma_native_queue_try_publish(queue, submission)) {
    iree_atomic_store(&queue->capacity.first, (intptr_t)entry,
                      iree_memory_order_release);
    if (!next) {
      queue->capacity.last = entry;
    }
    iree_hal_amd_gpu_sdma_await_capacity(queue, completed);
  } else if (next) {
    iree_hal_amd_gpu_sdma_capacity_available(queue);
  }
}

static bool iree_hal_amd_gpu_sdma_native_queue_publish_one(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  iree_hal_amd_gpu_publication_entry_t* link =
      iree_hal_amd_gpu_publisher_next(&queue->publisher);
  if (!link) {
    return false;
  }
  iree_hal_amd_gpu_sdma_entry_t* entry =
      iree_containerof(link, iree_hal_amd_gpu_sdma_entry_t, publication);
  if (entry->kind == IREE_HAL_AMD_GPU_SDMA_ENTRY_METADATA) {
    iree_hal_amd_gpu_sdma_metadata_t* metadata =
        iree_containerof(entry, iree_hal_amd_gpu_sdma_metadata_t, entry);
    metadata->process(metadata->user_data,
                      iree_hal_amd_gpu_sdma_native_queue_query_status(queue));
  } else if (queue->capacity.last) {
    iree_hal_amd_gpu_sdma_park_capacity(queue, link);
  } else {
    const uint64_t completed =
        (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
    iree_hal_amd_gpu_sdma_submission_t* submission =
        iree_containerof(entry, iree_hal_amd_gpu_sdma_submission_t, entry);
    if (!iree_hal_amd_gpu_sdma_native_queue_try_publish(queue, submission)) {
      iree_hal_amd_gpu_sdma_park_capacity(queue, link);
      iree_hal_amd_gpu_sdma_await_capacity(queue, completed);
    }
  }
  return true;
}

static bool iree_hal_amd_gpu_sdma_native_queue_publish(void* user_data) {
  iree_hal_amd_gpu_sdma_native_queue_t* queue = user_data;
  if (!iree_hal_amd_gpu_publisher_acquire_service(&queue->publisher)) {
    if (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      iree_hal_amd_gpu_sdma_native_queue_fail_submissions(queue);
    }
    return false;
  }
  for (uint32_t i = 0;
       i < 8 && iree_hal_amd_gpu_sdma_native_queue_publish_one(queue); ++i) {
  }
  iree_hal_amd_gpu_sdma_native_queue_fail_submissions(queue);
  const bool ready =
      iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL);
  if (iree_atomic_load(&queue->failure, iree_memory_order_acquire)) {
    iree_hal_amd_gpu_context_wake_failure(queue->context);
  }
  return ready;
}

void iree_hal_amd_gpu_sdma_native_queue_submit(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_submission_t* submission) {
  if (!iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                          &submission->entry.publication)) {
    return;
  }
  iree_hal_amd_gpu_sdma_native_queue_publish_one(queue);
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL) ||
      iree_atomic_load(&queue->failure, iree_memory_order_acquire)) {
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
}

void iree_hal_amd_gpu_sdma_native_queue_defer(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_submission_t* submission) {
  iree_hal_amd_gpu_publication_enqueue(
      queue->publication, queue->publication_ordinal, &queue->publisher,
      &submission->entry.publication);
}

void iree_hal_amd_gpu_sdma_native_queue_defer_metadata(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_metadata_t* metadata) {
  metadata->entry.kind = IREE_HAL_AMD_GPU_SDMA_ENTRY_METADATA;
  iree_hal_amd_gpu_publication_enqueue(
      queue->publication, queue->publication_ordinal, &queue->publisher,
      &metadata->entry.publication);
}

static bool iree_hal_amd_gpu_sdma_native_queue_scan(
    iree_hal_amd_gpu_sdma_native_queue_t* queue, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  if (!amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    iree_hal_amd_gpu_sdma_native_queue_fail(queue, wait_status);
    return false;
  }
  uint64_t retired =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_relaxed);
  const uint64_t published =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_acquire);
  if (retired == published) {
    return false;
  }
  uint64_t native_point = 0;
  if (queue->kernel.queue) {
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    amdf_kernel_queue_status_t snapshot = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(snapshot)};
    amdf_status_t status = queue->context->api->kernel_queue_refresh_status(
        queue->kernel.queue, &snapshot);
    if (amdf_status_is_ok(status)) {
      status = snapshot.terminal_status;
    }
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_native_queue_fail(queue, status);
      return false;
    }
    native_point = snapshot.retired_submission;
  }
  const uint32_t completed =
      (uint32_t)iree_atomic_load(queue->completion, iree_memory_order_acquire);
  bool progress = false;
  for (uint32_t i = 0; i < 8 && retired < published; ++i) {
    const iree_hal_amd_gpu_sdma_record_t record =
        queue->records[retired & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1)];
    if (queue->kernel.queue
            ? record.point > native_point
            : (int32_t)(completed - (uint32_t)(retired + 1)) < 0) {
      break;
    }
    iree_hal_amd_gpu_completion_scope_unlock(scope);
    if (record.completion.ready &&
        !record.completion.ready(record.completion.user_data, out_copy)) {
      out_wait->address = NULL;
      break;
    }
    record.completion.notify(record.completion.user_data, AMDF_STATUS_OK,
                             IREE_HAL_AMD_GPU_SDMA_RETIRED);
    ++retired;
    // Return capacity only after the callback's final access. Its caller may
    // keep per-span metadata in a matching fixed slot; native completion alone
    // cannot let another publisher overwrite that callback's operands.
    iree_atomic_store(&queue->retired, (int64_t)retired,
                      iree_memory_order_release);
    if (iree_atomic_exchange(&queue->waiting, 0, iree_memory_order_acq_rel)) {
      iree_hal_amd_gpu_sdma_capacity_available(queue);
    }
    progress = true;
  }
  return progress;
}

static bool iree_hal_amd_gpu_sdma_native_queue_poll(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  iree_hal_amd_gpu_sdma_native_queue_t* queue = user_data;
  // Other engines share the observer. Passive publication observation notices
  // this queue becoming active without keeping an idle device awake or adding
  // a publisher-side wake operation.
  const uint64_t published =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_acquire);
  if (amdf_status_is_ok(wait_status) &&
      published == (uint64_t)iree_atomic_load(&queue->retired,
                                              iree_memory_order_relaxed)) {
    if (!queue->kernel.queue) {
      *out_wait =
          (iree_hal_amd_gpu_completion_wait_t){.address = &queue->published,
                                               .value = (int64_t)published,
                                               .is_64_bit = true};
    }
    return false;
  }
  if (!queue->kernel.queue && amdf_status_is_ok(wait_status) &&
      !iree_atomic_load(&queue->failure, iree_memory_order_acquire)) {
    out_wait->address = queue->completion;
    out_wait->value =
        (int32_t)iree_atomic_load(&queue->retired, iree_memory_order_relaxed);
    out_wait->pending = true;
  }
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &queue->observer, &expected, IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
          iree_memory_order_acquire, iree_memory_order_relaxed)) {
    return false;
  }
  const bool progress = iree_hal_amd_gpu_sdma_native_queue_scan(
      queue, wait_status, scope, out_wait, out_copy);
  const int32_t previous = iree_atomic_fetch_and(
      &queue->observer, ~IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
      iree_memory_order_acq_rel);
  if (previous & IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED) {
    out_wait->address = NULL;
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
  out_wait->value =
      (int32_t)iree_atomic_load(&queue->retired, iree_memory_order_relaxed);
  out_wait->pending = out_wait->address != NULL;
  const int64_t retired =
      iree_atomic_load(&queue->retired, iree_memory_order_relaxed);
  const int64_t next_published =
      iree_atomic_load(&queue->published, iree_memory_order_acquire);
  if (retired == next_published) {
    *out_wait = (iree_hal_amd_gpu_completion_wait_t){0};
    if (!queue->kernel.queue) {
      *out_wait =
          (iree_hal_amd_gpu_completion_wait_t){.address = &queue->published,
                                               .value = next_published,
                                               .is_64_bit = true};
    }
  }
  return progress;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_sdma_native_queue_publication_source(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return (iree_hal_amd_gpu_publication_source_t){
      iree_hal_amd_gpu_sdma_native_queue_publish, queue};
}

iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_sdma_native_queue_completion_source(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return (iree_hal_amd_gpu_completion_source_t){
      iree_hal_amd_gpu_sdma_native_queue_poll, queue};
}

bool iree_hal_amd_gpu_sdma_native_queue_is_idle(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return !queue ||
         (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
          !iree_atomic_load(&queue->capacity.first,
                            iree_memory_order_acquire) &&
          iree_atomic_load(&queue->published, iree_memory_order_acquire) ==
              iree_atomic_load(&queue->retired, iree_memory_order_acquire));
}

bool iree_hal_amd_gpu_sdma_native_queue_is_host_idle(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return !queue ||
         (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
          !iree_atomic_load(&queue->capacity.first,
                            iree_memory_order_acquire) &&
          !(iree_atomic_load(&queue->observer, iree_memory_order_acquire) &
            IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE));
}

bool iree_hal_amd_gpu_sdma_native_queue_try_inactivate(
    iree_hal_amd_gpu_sdma_native_queue_t* queue, amdf_status_t* out_status) {
  if (!queue) {
    *out_status = AMDF_STATUS_OK;
    return true;
  }
  if (!iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher)) {
    return false;
  }
  amdf_status_t status = queue->removal_status;
  if (amdf_status_is_ok(status) && !queue->inactivated) {
    status =
        queue->user.storage.queue
            ? iree_hal_amd_gpu_user_queue_inactivate(&queue->user.storage)
            : queue->context->gpu->kernel_queue_inactivate(queue->kernel.queue);
    if (amdf_status_is_ok(status)) {
      queue->inactivated = true;
    } else {
      queue->removal_status = status;
    }
  }
  *out_status = status;
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
  return true;
}

bool iree_hal_amd_gpu_sdma_native_queue_retire_failed(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  if (!queue) {
    return true;
  }
  IREE_ASSERT(queue->inactivated,
              "failed command retirement requires native access cessation");
  if (!iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher)) {
    return false;
  }
  int32_t expected = IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED;
  const bool acquired = iree_atomic_compare_exchange_strong(
      &queue->observer, &expected,
      IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED |
          IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
      iree_memory_order_acquire, iree_memory_order_relaxed);
  if (acquired) {
    const amdf_status_t status = (amdf_status_t)iree_atomic_load(
        &queue->failure, iree_memory_order_acquire);
    const uint64_t end = (uint64_t)iree_atomic_load(&queue->published,
                                                    iree_memory_order_acquire);
    uint64_t position =
        (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_relaxed);
    for (; position != end; ++position) {
      const iree_hal_amd_gpu_sdma_completion_t completion =
          queue->records[position & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1)]
              .completion;
      iree_atomic_store(&queue->retired, position + 1,
                        iree_memory_order_release);
      completion.notify(completion.user_data, status,
                        IREE_HAL_AMD_GPU_SDMA_RETIRED);
    }
    queue->failure_delivered = true;
    iree_atomic_fetch_and(&queue->observer,
                          ~IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
                          iree_memory_order_release);
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, NULL)) {
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
  return acquired;
}

amdf_status_t iree_hal_amd_gpu_sdma_native_queue_free(
    iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  if (!queue) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(queue->removal_status)) {
    return queue->removal_status;
  }
  const amdf_api_t* api = queue->context->api;
  amdf_status_t status = AMDF_STATUS_OK;
  if (queue->user.storage.context) {
    status = iree_hal_amd_gpu_user_queue_deinitialize(&queue->user.storage,
                                                      queue->user.position);
  } else if (!queue->inactivated && queue->kernel.queue &&
             queue->kernel.last_submission) {
    status = api->kernel_queue_wait(queue->kernel.queue,
                                    queue->kernel.last_submission,
                                    AMDF_TIMEOUT_INFINITE, 0);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (queue->kernel.queue) {
    status = api->kernel_queue_destroy(queue->kernel.queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      return status;
    }
    queue->kernel.queue = NULL;
  }
  queue->removal_status = status;
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->memory);
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_allocator_free_aligned(queue->host_allocator, queue);
  return AMDF_STATUS_OK;
}

iree_status_t iree_hal_amd_gpu_sdma_native_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    amdf_queue_priority_t priority, amdf_queue_placement_t placement,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sdma_native_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_gpu_sdma_native_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*queue),
      iree_alignof(iree_hal_amd_gpu_sdma_native_queue_t), 0, (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  queue->context = context;
  queue->host_allocator = host_allocator;
  queue->publication = publication;
  queue->publication_ordinal = publication_ordinal;
  queue->features = context->families[family_ordinal].format_features;
  iree_hal_amd_gpu_publisher_initialize(&queue->publisher);
  queue->capacity.resume = (iree_hal_amd_gpu_sdma_metadata_t){
      .process = iree_hal_amd_gpu_sdma_resume_capacity,
      .user_data = queue,
  };
  const uint64_t command_length =
      publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL
          ? IREE_HAL_AMD_GPU_SDMA_CAPACITY *
                IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE
          : 0;
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, control, command_length + 64, 64, context->event_sink,
      &queue->memory, NULL);
  uint64_t address = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->memory_query_address(queue->memory.handle, 0,
                                           AMDF_MEMORY_ADDRESS_GPU, &address),
        "memory_query_address");
  }
  if (iree_status_is_ok(status)) {
    queue->completion_address = address + command_length;
    queue->completion = (void*)(queue->memory.contents.data + command_length);
    iree_atomic_store(queue->completion, 0, iree_memory_order_relaxed);
    if (publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
      const amdf_gpu_kernel_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = family_ordinal,
          .maximum_pending_submission_count = IREE_HAL_AMD_GPU_SDMA_CAPACITY};
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          context->gpu->kernel_queue_create(context->device, &create,
                                            &queue->kernel.queue),
          "gpu.kernel_queue_create(SDMA)");
      amdf_kernel_queue_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
          .structure_size = sizeof(info)};
      if (iree_status_is_ok(status)) {
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            context->api->kernel_queue_query_info(queue->kernel.queue, &info),
            "kernel_queue_query_info");
      }
      if (iree_status_is_ok(status) &&
          !(info.notification_types &
            (UINT64_C(1) << context->kernel_notification.type))) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "SDMA KERNEL queue has no compatible wake");
      }
      queue->kernel.capacity = iree_min(IREE_HAL_AMD_GPU_SDMA_CAPACITY,
                                        info.maximum_pending_submission_count);
      queue->commands = (void*)queue->memory.contents.data;
    } else {
      const amdf_gpu_user_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = family_ordinal,
          .priority = priority,
          .placement = placement,
          .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
          .ring_byte_length = IREE_HAL_AMD_GPU_SDMA_RING_LENGTH};
      amdf_user_queue_mapping_info_t info;
      status = iree_hal_amd_gpu_user_queue_initialize(
          context, &create, &queue->user.storage, &info, NULL);
      if (iree_status_is_ok(status) &&
          (info.index_bits != 64 || info.doorbell_bits != 64 ||
           info.ring_byte_length != IREE_HAL_AMD_GPU_SDMA_RING_LENGTH ||
           !info.ring_address || !info.write_index_address ||
           !info.doorbell_address)) {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "SDMA USER queue did not provide its requested ring mapping");
      }
      if (iree_status_is_ok(status)) {
        queue->commands = (void*)(uintptr_t)info.ring_address;
        queue->user.write_index = (void*)(uintptr_t)info.write_index_address;
        queue->user.doorbell = (void*)(uintptr_t)info.doorbell_address;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    const amdf_status_t cleanup =
        iree_hal_amd_gpu_sdma_native_queue_free(queue);
    if (!amdf_status_is_ok(cleanup)) {
      iree_atomic_store(&context->cleanup_blocked, 1,
                        iree_memory_order_relaxed);
      iree_hal_amd_gpu_context_report(context, cleanup, "SDMA queue rollback");
    }
  }
  return status;
}

iree_hal_queue_placement_t iree_hal_amd_gpu_sdma_native_queue_placement(
    const iree_hal_amd_gpu_sdma_native_queue_t* queue) {
  return queue->user.storage.placement;
}
