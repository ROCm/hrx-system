// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"

#include "iree/hal/drivers/amd/status.h"

#define IREE_HAL_AMD_GPU_SDMA_CAPACITY 64u
#define IREE_HAL_AMD_GPU_SDMA_RING_LENGTH 65536u
#define IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE 1024u
// The common 22-bit form also covers engines with wider count fields. Keep
// the established 256-byte headroom without deriving SDMA IP from shader ISA.
#define IREE_HAL_AMD_GPU_SDMA_PACKET_LENGTH 0x3fff00u
#define IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS (17u * 7u + 5u + 5u + 4u + 6u)

enum {
  IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE = 1u << 0,
  IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED = 1u << 1,
};

amdf_queue_publication_modes_t iree_hal_amd_gpu_sdma_queue_publication_modes(
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
static_assert((IREE_HAL_AMD_GPU_SDMA_CAPACITY + 2u) *
                      IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS * sizeof(uint32_t) <
                  IREE_HAL_AMD_GPU_SDMA_RING_LENGTH,
              "the primary ring must dominate the complete live copy window");

typedef struct iree_hal_amd_gpu_sdma_record_t {
  // Borrowed placed copy, returned exactly once by the completion observer.
  iree_hal_amd_gpu_sdma_copy_t* copy;
  // Opaque KERNEL retirement point, or zero for USER publication.
  uint64_t point;
} iree_hal_amd_gpu_sdma_record_t;

struct iree_hal_amd_gpu_sdma_queue_t {
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
  // Sole caller/deferred native producer.
  iree_alignas(64) iree_hal_amd_gpu_publisher_t publisher;
  // Publisher parks only after enrolling this native-capacity return.
  iree_atomic_int32_t waiting;
  // First native error; it does not establish quiescence of accepted work.
  iree_atomic_int64_t failure;
  // Joins the current completion scan before publication reports native
  // failure.
  iree_atomic_int32_t observer;
  // The deferred publication source has reported every unretired copy once.
  bool failure_delivered;
  // Release-published retirement records, independent of native progress.
  iree_alignas(64) iree_atomic_int64_t published;
  // Completion-owned prefix returning both command and record capacity.
  iree_alignas(64) iree_atomic_int64_t retired;
  // Constant-size SPSC records, indexed by publication sequence.
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
    // Owned directly published queue, absent for KERNEL transport.
    amdf_user_queue_t* queue;
    // Owned host mapping, destroyed before its queue.
    amdf_user_queue_mapping_t* mapping;
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
};

static void iree_hal_amd_gpu_sdma_queue_fail(
    iree_hal_amd_gpu_sdma_queue_t* queue, amdf_status_t status) {
  int64_t expected = 0;
  iree_atomic_compare_exchange_strong(
      &queue->failure, &expected, (int64_t)status, iree_memory_order_release,
      iree_memory_order_relaxed);
  iree_atomic_fetch_or(&queue->observer, IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED,
                       iree_memory_order_acq_rel);
  iree_hal_amd_gpu_publication_notify(queue->publication,
                                      queue->publication_ordinal);
}

// Only the deferred source reports failures after the completion owner stops.
// It owns publication or has observed its idle handoff after the failure;
// subsequent callers can only reject copies and cannot mutate these records.
// Failed notification delivery therefore does not depend on another native
// wake.
static void iree_hal_amd_gpu_sdma_queue_fail_copies(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (queue->failure_delivered ||
      iree_atomic_load(&queue->observer, iree_memory_order_acquire) !=
          IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED) {
    return;
  }
  const amdf_status_t status = (amdf_status_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  const uint64_t begin =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  const uint64_t end =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_relaxed);
  for (uint64_t i = begin; i != end; ++i) {
    iree_hal_amd_gpu_sdma_copy_t* copy =
        queue->records[i & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1)].copy;
    copy->notify(copy->user_data, status, IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
  }
  queue->failure_delivered = true;
}

static uint32_t* iree_hal_amd_gpu_sdma_fence(
    uint32_t* words, amdf_queue_format_features_t features, uint64_t address,
    uint32_t value) {
  uint32_t header = 5;
  if (features & (AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
                  AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM)) {
    header |= 3u << 16;
  }
  if (features & AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM) {
    header |= 1u << 20;
  }
  if (features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) {
    header |= 3u << 24;
  }
  *words++ = header;
  *words++ = (uint32_t)address;
  *words++ = (uint32_t)(address >> 32);
  *words++ = value;
  return words;
}

static uint32_t* iree_hal_amd_gpu_sdma_cache(uint32_t* words,
                                             uint32_t control) {
  *words++ = 17u | (1u << 8);
  *words++ = 0;
  *words++ = control << 16;
  *words++ = 0;
  *words++ = 0;
  return words;
}

static uint32_t iree_hal_amd_gpu_sdma_encode(
    iree_hal_amd_gpu_sdma_queue_t* queue,
    const iree_hal_amd_gpu_sdma_copy_t* copy, uint32_t value, uint32_t* words) {
  uint32_t* cursor = words;
  if (copy->effects.bits & IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM) {
    cursor = iree_hal_amd_gpu_sdma_cache(cursor, 0xc3c0u);
  }
  const uint32_t scopes =
      queue->features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE
          ? (3u << 18) | (3u << 26)
          : 0;
  for (uint32_t offset = 0; offset < copy->length;) {
    const uint32_t length =
        iree_min(copy->length - offset, IREE_HAL_AMD_GPU_SDMA_PACKET_LENGTH);
    const uint64_t source = copy->source + offset;
    const uint64_t target = copy->target + offset;
    *cursor++ = 1;
    *cursor++ = length - 1;
    *cursor++ = scopes;
    *cursor++ = (uint32_t)source;
    *cursor++ = (uint32_t)(source >> 32);
    *cursor++ = (uint32_t)target;
    *cursor++ = (uint32_t)(target >> 32);
    offset += length;
  }
  if (copy->effects.bits & IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM) {
    cursor = iree_hal_amd_gpu_sdma_cache(cursor, 0x8040u);
  }
  cursor = iree_hal_amd_gpu_sdma_fence(cursor, queue->features,
                                       queue->completion_address, value);
  if (queue->user.queue) {
    const amdf_gpu_event_info_t* event = &queue->context->event_info;
    cursor = iree_hal_amd_gpu_sdma_fence(
        cursor, queue->features, event->mailbox_address, event->trigger_data);
    *cursor++ = 6;
    *cursor++ = event->interrupt_context;
  }
  return (uint32_t)(cursor - words);
}

// Returns the entry only when native capacity has not accepted it. No payload
// access, file submission, or completion callback occurs on this owner.
static iree_hal_amd_gpu_publication_entry_t*
iree_hal_amd_gpu_sdma_queue_publish_one(iree_hal_amd_gpu_sdma_queue_t* queue) {
  iree_hal_amd_gpu_publication_entry_t* entry =
      iree_hal_amd_gpu_publisher_next(&queue->publisher);
  if (!entry) {
    return NULL;
  }
  iree_hal_amd_gpu_sdma_copy_t* copy =
      iree_containerof(entry, iree_hal_amd_gpu_sdma_copy_t, entry);
  amdf_status_t status = (amdf_status_t)iree_atomic_load(
      &queue->failure, iree_memory_order_acquire);
  if (!amdf_status_is_ok(status)) {
    copy->notify(copy->user_data, status, IREE_HAL_AMD_GPU_SDMA_RETIRED);
    return NULL;
  }
  const uint64_t published =
      (uint64_t)iree_atomic_load(&queue->published, iree_memory_order_relaxed);
  const uint64_t retired =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  const uint32_t capacity = queue->kernel.queue
                                ? queue->kernel.capacity
                                : IREE_HAL_AMD_GPU_SDMA_CAPACITY;
  if (published - retired == capacity) {
    return entry;
  }
  const uint32_t index = published & (IREE_HAL_AMD_GPU_SDMA_CAPACITY - 1);
  uint64_t point = 0;
  uint32_t words[IREE_HAL_AMD_GPU_SDMA_COMMAND_WORDS];
  const uint32_t word_count = iree_hal_amd_gpu_sdma_encode(
      queue, copy, (uint32_t)(published + 1), words);
  if (amdf_status_is_ok(status) && queue->kernel.queue) {
    const uint64_t offset = index * IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE;
    memcpy((uint8_t*)queue->commands + offset, words, word_count * 4u);
    const amdf_gpu_kernel_command_t command = {
        .memory = queue->memory.handle,
        .byte_offset = offset,
        .byte_length = word_count * 4u,
    };
    const amdf_gpu_kernel_queue_submission_info_t submission = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submission),
        .command_count = 1,
        .commands = &command,
    };
    status = queue->context->gpu->kernel_queue_submit(queue->kernel.queue,
                                                      &submission, &point);
    // Capacity can be deferred only while an accepted submission owns a wake.
    // BUSY on an empty, exclusively owned queue has no progress source.
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY) &&
        published != retired) {
      return entry;
    }
    if (amdf_status_is_ok(status)) {
      queue->kernel.last_submission = point;
    }
  }
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_queue_fail(queue, status);
    copy->notify(copy->user_data, status, IREE_HAL_AMD_GPU_SDMA_RETIRED);
    return NULL;
  }
  queue->records[index] =
      (iree_hal_amd_gpu_sdma_record_t){.copy = copy, .point = point};
  iree_atomic_store(&queue->published, (int64_t)(published + 1),
                    iree_memory_order_release);
  if (queue->kernel.queue) {
    status = queue->context->api->kernel_queue_request_notification(
        queue->kernel.queue, point, &queue->context->kernel_notification);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_queue_fail(queue, status);
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
    iree_atomic_store(queue->user.write_index, (int64_t)queue->user.position,
                      iree_memory_order_release);
    const iree_hal_memory_transition_recipe_info_t fence = {
        .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE}};
    iree_hal_memory_transition_execute_host_direct(&fence, NULL, 1);
    *queue->user.doorbell = queue->user.position;
  }
  return NULL;
}

static bool iree_hal_amd_gpu_sdma_queue_handoff(
    iree_hal_amd_gpu_sdma_queue_t* queue,
    iree_hal_amd_gpu_publication_entry_t* remainder, uint64_t completed) {
  bool ready = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, remainder);
  if (ready && remainder) {
    // Enrollment follows ownership handoff. A completion between the failed
    // capacity check and enrollment is observed by this acquired recheck.
    iree_atomic_exchange(&queue->waiting, 1, iree_memory_order_acq_rel);
    ready = iree_atomic_load(&queue->failure, iree_memory_order_acquire) ||
            (uint64_t)iree_atomic_load(&queue->retired,
                                       iree_memory_order_acquire) != completed;
  }
  return ready;
}

static bool iree_hal_amd_gpu_sdma_queue_publish(void* user_data) {
  iree_hal_amd_gpu_sdma_queue_t* queue = user_data;
  if (!iree_hal_amd_gpu_publisher_acquire_service(&queue->publisher)) {
    if (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      iree_hal_amd_gpu_sdma_queue_fail_copies(queue);
    }
    return false;
  }
  const uint64_t completed =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  iree_hal_amd_gpu_publication_entry_t* remainder = NULL;
  for (uint32_t i = 0; i < 8 && !remainder; ++i) {
    const int64_t published =
        iree_atomic_load(&queue->published, iree_memory_order_relaxed);
    remainder = iree_hal_amd_gpu_sdma_queue_publish_one(queue);
    if (iree_atomic_load(&queue->published, iree_memory_order_relaxed) ==
        published) {
      break;
    }
  }
  iree_hal_amd_gpu_sdma_queue_fail_copies(queue);
  return iree_hal_amd_gpu_sdma_queue_handoff(queue, remainder, completed);
}

void iree_hal_amd_gpu_sdma_queue_submit(iree_hal_amd_gpu_sdma_queue_t* queue,
                                        iree_hal_amd_gpu_sdma_copy_t* copy) {
  if (!iree_hal_amd_gpu_publisher_enqueue(&queue->publisher, &copy->entry)) {
    return;
  }
  const uint64_t completed =
      (uint64_t)iree_atomic_load(&queue->retired, iree_memory_order_acquire);
  iree_hal_amd_gpu_publication_entry_t* remainder =
      iree_hal_amd_gpu_sdma_queue_publish_one(queue);
  if (iree_hal_amd_gpu_sdma_queue_handoff(queue, remainder, completed) ||
      iree_atomic_load(&queue->failure, iree_memory_order_acquire)) {
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
}

static bool iree_hal_amd_gpu_sdma_queue_scan(
    iree_hal_amd_gpu_sdma_queue_t* queue, amdf_status_t wait_status) {
  if (!amdf_status_is_ok(wait_status)) {
    iree_hal_amd_gpu_sdma_queue_fail(queue, wait_status);
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
    amdf_kernel_queue_status_t snapshot = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(snapshot)};
    amdf_status_t status = queue->context->api->kernel_queue_refresh_status(
        queue->kernel.queue, &snapshot);
    if (amdf_status_is_ok(status)) {
      status = snapshot.terminal_status;
    }
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_queue_fail(queue, status);
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
    ++retired;
    // Detach metadata before its callback can immediately reuse the copy or
    // publish another slot. KERNEL point retirement also releases command
    // bytes.
    iree_atomic_store(&queue->retired, (int64_t)retired,
                      iree_memory_order_release);
    if (iree_atomic_exchange(&queue->waiting, 0, iree_memory_order_acq_rel)) {
      iree_hal_amd_gpu_publication_notify(queue->publication,
                                          queue->publication_ordinal);
    }
    record.copy->notify(record.copy->user_data, AMDF_STATUS_OK,
                        IREE_HAL_AMD_GPU_SDMA_RETIRED);
    progress = true;
  }
  return progress;
}

static bool iree_hal_amd_gpu_sdma_queue_poll(void* user_data,
                                             amdf_status_t wait_status) {
  iree_hal_amd_gpu_sdma_queue_t* queue = user_data;
  // Other engines share the observer. An idle copy source performs no RMW and
  // does not contend with its publisher merely because a PM4 queue woke up.
  if (amdf_status_is_ok(wait_status) &&
      iree_atomic_load(&queue->published, iree_memory_order_acquire) ==
          iree_atomic_load(&queue->retired, iree_memory_order_relaxed)) {
    return false;
  }
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &queue->observer, &expected, IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
          iree_memory_order_acquire, iree_memory_order_relaxed)) {
    return false;
  }
  const bool progress = iree_hal_amd_gpu_sdma_queue_scan(queue, wait_status);
  const int32_t previous = iree_atomic_fetch_and(
      &queue->observer, ~IREE_HAL_AMD_GPU_SDMA_OBSERVER_ACTIVE,
      iree_memory_order_acq_rel);
  if (previous & IREE_HAL_AMD_GPU_SDMA_OBSERVER_STOPPED) {
    iree_hal_amd_gpu_publication_notify(queue->publication,
                                        queue->publication_ordinal);
  }
  return progress;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_sdma_queue_publication_source(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  return (iree_hal_amd_gpu_publication_source_t){
      iree_hal_amd_gpu_sdma_queue_publish, queue};
}

iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_sdma_queue_completion_source(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  return (iree_hal_amd_gpu_completion_source_t){
      iree_hal_amd_gpu_sdma_queue_poll, queue};
}

bool iree_hal_amd_gpu_sdma_queue_is_idle(iree_hal_amd_gpu_sdma_queue_t* queue) {
  return !queue ||
         (iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
          iree_atomic_load(&queue->published, iree_memory_order_acquire) ==
              iree_atomic_load(&queue->retired, iree_memory_order_acquire));
}

amdf_status_t iree_hal_amd_gpu_sdma_queue_free(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (!queue) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(queue->removal_status)) {
    return queue->removal_status;
  }
  const amdf_api_t* api = queue->context->api;
  amdf_status_t status = AMDF_STATUS_OK;
  if (queue->user.queue && queue->user.position) {
    status = api->user_queue_wait_consumed(
        queue->user.queue, queue->user.position, AMDF_TIMEOUT_INFINITE, 0);
  } else if (queue->kernel.queue && queue->kernel.last_submission) {
    status = api->kernel_queue_wait(queue->kernel.queue,
                                    queue->kernel.last_submission,
                                    AMDF_TIMEOUT_INFINITE, 0);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (queue->user.mapping) {
    status = api->user_queue_mapping_destroy(queue->user.mapping);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    queue->user.mapping = NULL;
  }
  if (queue->user.queue) {
    status = api->user_queue_destroy(queue->user.queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      return status;
    }
    queue->user.queue = NULL;
  } else if (queue->kernel.queue) {
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
  iree_hal_amd_gpu_context_release_memory_sync(queue->context, 1,
                                               &queue->memory);
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_allocator_free_aligned(queue->host_allocator, queue);
  return AMDF_STATUS_OK;
}

iree_status_t iree_hal_amd_gpu_sdma_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sdma_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_gpu_sdma_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*queue), iree_alignof(*queue), 0, (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  queue->context = context;
  queue->host_allocator = host_allocator;
  queue->publication = publication;
  queue->publication_ordinal = publication_ordinal;
  queue->features = context->families[family_ordinal].format_features;
  iree_hal_amd_gpu_publisher_initialize(&queue->publisher);
  const uint64_t command_length =
      publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL
          ? IREE_HAL_AMD_GPU_SDMA_CAPACITY *
                IREE_HAL_AMD_GPU_SDMA_COMMAND_STRIDE
          : 0;
  const iree_hal_amd_fabric_memory_request_t request = {
      .options = control, .byte_length = command_length + 64, .alignment = 64};
  iree_status_t status = iree_hal_amd_gpu_context_acquire_memory_sync(
      context, 1, &request, &queue->memory);
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
          .priority = AMDF_QUEUE_PRIORITY_NORMAL,
          .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
          .ring_byte_length = IREE_HAL_AMD_GPU_SDMA_RING_LENGTH};
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          context->gpu->user_queue_create(context->device, &create,
                                          &queue->user.queue),
          "gpu.user_queue_create(SDMA)");
      if (iree_status_is_ok(status)) {
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            context->api->user_queue_map(queue->user.queue, NULL,
                                         &queue->user.mapping),
            "user_queue_map");
      }
      amdf_user_queue_mapping_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
          .structure_size = sizeof(info)};
      if (iree_status_is_ok(status)) {
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            context->api->user_queue_mapping_query_info(queue->user.mapping,
                                                        &info),
            "user_queue_mapping_query_info");
      }
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
    const amdf_status_t cleanup = iree_hal_amd_gpu_sdma_queue_free(queue);
    if (!amdf_status_is_ok(cleanup)) {
      context->cleanup_blocked = true;
      iree_hal_amd_gpu_context_report(context, cleanup, "SDMA queue rollback");
    }
  }
  return status;
}
