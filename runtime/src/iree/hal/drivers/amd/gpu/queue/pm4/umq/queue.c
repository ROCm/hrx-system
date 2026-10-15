// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/queue.h"

#include "iree/base/cpu_data.h"
#include "iree/hal/drivers/amd/gpu/queue/continuation.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"
#include "iree/hal/drivers/amd/status.h"

// Argument generations are FIFO by publication, including alignment and wrap.
// The service frees a replaced generation only after its final byte position
// has retired. Native readers of signals have a separate lifetime below.
typedef struct iree_hal_amd_gpu_pm4_argument_storage_t {
  // Older generation, or NULL at the end of the cold cleanup list.
  struct iree_hal_amd_gpu_pm4_argument_storage_t* next;
  // Native storage and persistent host view.
  iree_hal_amd_fabric_memory_t memory;
  // Address of byte zero in this queue's native domain.
  uint64_t address;
  // Monotonic first position belonging to this generation.
  uint64_t begin;
  // Exclusive final reserved position in this generation.
  uint64_t end;
  // Power-of-two usable byte capacity.
  uint64_t capacity;
} iree_hal_amd_gpu_pm4_argument_storage_t;

typedef struct iree_hal_amd_gpu_pm4_signal_storage_t {
  // Next cold backing, retained until all device queues have joined.
  struct iree_hal_amd_gpu_pm4_signal_storage_t* next;
  // Native cells and coherent CPU mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Finite bookkeeping over the following inline signal metadata.
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
} iree_hal_amd_gpu_pm4_signal_storage_t;

typedef struct iree_hal_amd_gpu_pm4_umq_queue_t
    iree_hal_amd_gpu_pm4_umq_queue_t;

// Cold sidecar. Its list links and tracker acceptance are owned by service,
// while tracker callbacks only enqueue finalization onto that same service.
typedef struct iree_hal_amd_gpu_pm4_pending_t {
  // Entry used to begin admission on service before tracker registration.
  iree_hal_memory_maintenance_entry_t entry;
  // Queue whose accepted capture dominates this sidecar's lifetime.
  iree_hal_amd_gpu_pm4_umq_queue_t* queue;
  // Owned captured operation awaiting admission.
  iree_hal_amd_gpu_pm4_submission_t* submission;
  // Next registered wait, accessed only by service.
  struct iree_hal_amd_gpu_pm4_pending_t* next;
  // Previous registered wait, or NULL at the head.
  struct iree_hal_amd_gpu_pm4_pending_t* previous;
  // Existing asynchronous wait lifecycle; no additional thread is created.
  iree_hal_amd_gpu_host_wait_t wait;
  // Owned timeline selected for this registration, including availability.
  iree_hal_semaphore_t* semaphore;
  // Captured value paired with semaphore.
  uint64_t value;
} iree_hal_amd_gpu_pm4_pending_t;

struct iree_hal_amd_gpu_pm4_umq_queue_t {
  // Refcounted exact HAL queue; its family/device remain borrowed.
  iree_hal_queue_t base;
  // Allocator for the wrapper, cold backing owners, and oversized captures.
  iree_allocator_t host_allocator;
  // Borrowed native domain and event owner.
  const iree_hal_amd_gpu_context_t* context;
  // Immutable native construction and memory facts.
  iree_hal_amd_gpu_pm4_umq_queue_options_t options;
  // Sole caller/service native publication owner.
  iree_alignas(64) iree_hal_amd_gpu_publisher_t publisher;
  // Reusable host captures, independent of native ring capacity.
  iree_arena_block_pool_t captures;
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
  // Native queue and mapping, joined after the final notification tail.
  iree_hal_amd_gpu_pm4_ring_storage_t storage;
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
    // Replaced argument generations awaiting their completed byte prefix.
    iree_hal_amd_gpu_pm4_argument_storage_t* retired_arguments;
    // Active finite signal pool; exhaustion moves growth to service.
    iree_hal_amd_gpu_pm4_signal_storage_t* signals;
    // All independently retained signal backings, including the active one.
    iree_hal_amd_gpu_pm4_signal_storage_t* signal_storage;
    // Number of accepted native publications, release-published before MMIO.
    iree_atomic_int64_t count;
  } producer;
  // Single completion observer's release-published capacity credit.
  struct {
    // Number of consumed retirement records.
    iree_alignas(64) iree_atomic_int64_t count;
    // Exact primary command prefix through the last consumed release.
    iree_atomic_int64_t words;
    // Last consumed argument position, including wrap/alignment.
    iree_atomic_int64_t arguments;
  } completed;
  // Native host-visible records, immutable after preparation.
  iree_hal_amd_fabric_memory_t control;
  // Indexed native record geometry shared by publisher and observer.
  iree_hal_amd_gpu_pm4_retirement_ring_t retirement;
  // Host pointers parallel to record slots, read only on terminal native
  // failure after publication joins. Normal retirement uses GPU-carried bits.
  iree_hal_amd_gpu_pm4_submission_t** issued;
};

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_umq_queue_vtable;

static void iree_hal_amd_gpu_pm4_umq_wake_shutdown(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue) {
  if (IREE_UNLIKELY(
          iree_atomic_load(&queue->closed, iree_memory_order_acquire))) {
    const amdf_status_t status =
        queue->context->gpu->event_signal(queue->context->event);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(queue->context, status,
                                      "GPU shutdown event_signal");
    }
  }
}

static void iree_hal_amd_gpu_pm4_umq_complete(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_pm4_submission_release(submission);
  iree_atomic_fetch_sub(&queue->outstanding, 1, iree_memory_order_release);
  iree_hal_amd_gpu_pm4_umq_wake_shutdown(queue);
}

// Consumes an operation failure through its public signals and diagnostic sink.
// This is also the terminal diagnostic boundary for a call without signals.
static void iree_hal_amd_gpu_pm4_umq_fail_signals(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue, iree_hal_semaphore_list_t signals,
    iree_status_t status) {
  char message[1024] = {0};
  iree_host_size_t message_length = 0;
  iree_status_format(status, sizeof(message), message, &message_length);
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = iree_status_code(status),
      .message = iree_make_string_view(
          message, iree_min(message_length, sizeof(message) - 1)),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.device = queue->base.queue_family->device;
  event.source.driver_id = IREE_SV("amd");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(queue->context->event_sink, &event);
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    iree_hal_semaphore_fail(signals.semaphores[i], iree_status_clone(status));
  }
  iree_status_free(status);
}

static void iree_hal_amd_gpu_pm4_umq_abandon(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_pm4_umq_fail_signals(queue, submission->signals, status);
  iree_hal_amd_gpu_pm4_umq_complete(queue, submission);
}

// Cold allocation runs only on service. Coherent host access is a hard input
// requirement; ordinary user payload memory need not share this control policy.
static iree_status_t iree_hal_amd_gpu_pm4_umq_allocate_memory(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      queue->context->api, queue->options.memory, length, alignment,
      queue->context->event_sink, memory));
  if (memory->host_cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 control storage requires a coherent "
                            "write-back host mapping");
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->memory_query_address(
          memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, out_address),
      "memory_query_address");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_grow_arguments(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue, uint64_t required_capacity) {
  iree_hal_amd_gpu_pm4_argument_storage_t* previous = queue->producer.arguments;
  uint64_t capacity =
      previous ? previous->capacity : queue->options.argument_capacity;
  while (capacity < required_capacity) {
    capacity *= 2;
  }
  iree_hal_amd_gpu_pm4_argument_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*storage), (void**)&storage));
  storage->capacity = capacity;
  storage->begin = previous ? previous->end : 0;
  storage->end = storage->begin;
  iree_status_t status = iree_hal_amd_gpu_pm4_umq_allocate_memory(
      queue, capacity, capacity, &storage->memory, &storage->address);
  if (iree_status_is_ok(status)) {
    if (previous) {
      previous->next = queue->producer.retired_arguments;
      queue->producer.retired_arguments = previous;
    }
    queue->producer.arguments = storage;
  } else {
    iree_hal_amd_fabric_memory_deinitialize(
        queue->context->api, queue->context->event_sink, &storage->memory);
    iree_allocator_free(queue->host_allocator, storage);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_grow_signals(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue) {
  const uint32_t capacity = queue->options.retirement_capacity;
  iree_hal_amd_gpu_pm4_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator,
      sizeof(*storage) + capacity * sizeof(iree_hal_amd_gpu_pm4_signal_t),
      (void**)&storage));
  uint64_t address = 0;
  iree_status_t status = iree_hal_amd_gpu_pm4_umq_allocate_memory(
      queue, capacity * sizeof(uint64_t), 8, &storage->memory, &address);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_signal_pool_initialize(
        queue->context, (uint64_t*)storage->memory.contents.data, address,
        (iree_hal_amd_gpu_pm4_signal_t*)(storage + 1), capacity,
        (iree_hal_amd_gpu_pm4_signal_pool_notification_t){0}, &storage->pool);
    storage->next = queue->producer.signal_storage;
    queue->producer.signal_storage = storage;
    queue->producer.signals = storage;
  } else {
    iree_hal_amd_fabric_memory_deinitialize(
        queue->context->api, queue->context->event_sink, &storage->memory);
    iree_allocator_free(queue->host_allocator, storage);
  }
  return status;
}

// Replaced argument backing has no independently retained readers. All cold
// reclamation stays on service, after the exact final argument position.
static void iree_hal_amd_gpu_pm4_umq_trim_arguments(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue) {
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  iree_hal_amd_gpu_pm4_argument_storage_t** link =
      &queue->producer.retired_arguments;
  while (*link) {
    iree_hal_amd_gpu_pm4_argument_storage_t* storage = *link;
    if (storage->end <= completed) {
      *link = storage->next;
      iree_hal_amd_fabric_memory_deinitialize(
          queue->context->api, queue->context->event_sink, &storage->memory);
      iree_allocator_free(queue->host_allocator, storage);
    } else {
      link = &storage->next;
    }
  }
}

typedef enum iree_hal_amd_gpu_pm4_publication_result_e {
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS,
} iree_hal_amd_gpu_pm4_publication_result_t;

// The sole publisher performs no native call or host allocation here. Every
// address/error check precedes committing either ring reservation. The caller
// captured next before entry; immediate native retirement may free submission.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_umq_publish(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  const amdf_status_t failure = (amdf_status_t)iree_atomic_load(
      &queue->native_failure, iree_memory_order_acquire);
  if (!amdf_status_is_ok(failure)) {
    iree_hal_amd_gpu_pm4_umq_abandon(
        queue, submission,
        IREE_HAL_AMD_STATUS_FROM_AMDF(failure, "PM4 native completion"));
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  const iree_hal_amd_gpu_pm4_function_t* function = submission->function;
  const bool dispatch = function && submission->dispatch.workgroup_count[0] &&
                        submission->dispatch.workgroup_count[1] &&
                        submission->dispatch.workgroup_count[2];
  const bool indirect =
      iree_hal_dispatch_uses_indirect_arguments(submission->flags);
  const uint32_t argument_length =
      dispatch && !indirect ? function->kernarg->byte_length : 0;
  iree_hal_amd_gpu_pm4_argument_storage_t* arguments =
      queue->producer.arguments;
  uint64_t argument_offset = 0;
  uint64_t argument_end = arguments->end;
  uint64_t argument_address = 0;
  if (argument_length) {
    const uint64_t alignment = function->kernarg->alignment;
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
  uint32_t native_wait_count = 0;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    native_wait_count += submission->waits[i].signal != NULL;
  }
  iree_hal_amd_gpu_pm4_barrier_t before = submission->before;
  if (dispatch) {
    // Instruction upload and host argument publication remain mandatory even
    // when the caller supplies an empty optional payload boundary.
    before.cache_control |= 1u | queue->options.barriers.acquire;
  }
  const uint32_t launch_words =
      dispatch ? (function->launch.user_data_count
                      ? 24 + function->launch.user_data_count
                      : 22) +
                     5
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
  const uint32_t body_words = native_wait_count * 7 + launch_words +
                              iree_hal_amd_gpu_pm4_barrier_word_count(before) +
                              tail_words + (native_signal ? 8 : 0) + 14 + 8;
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  uint64_t* addresses =
      (uint64_t*)(submission->bindings.values + submission->bindings.count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < submission->bindings.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_pm4_submission_resolve_address(
        queue->base.queue_family, submission->bindings.values[i],
        indirect ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (dispatch && indirect && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (function->kernarg->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect kernarg address is misaligned");
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_umq_abandon(queue, submission, status);
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
  // Admission concurrently reads immutable ring geometry. Commit only the
  // publisher-owned position, preserving those read-only fields in place.
  queue->producer.ring.reserved_words = candidate.reserved_words;
  submission->consumed_arguments = argument_end;
  if (argument_length) {
    iree_hal_amd_gpu_pm4_submission_emplace_arguments(
        submission, addresses,
        arguments->memory.contents.data + argument_offset);
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
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
        .kernarg_pointer = argument_address, .dispatch_id = position};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch, submission->dispatch.workgroup_size,
        submission->dispatch.partial_size,
        submission->dispatch.group_segment_byte_length, &inputs);
    const uint32_t initiator =
        function->launch.dispatch_initiator |
        (submission->dispatch.partial_size[0] ? UINT32_C(1) << 6 : 0);
    words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
        words, submission->dispatch.workgroup_count, initiator);
  }
  uint32_t release_control;
  words = iree_hal_amd_gpu_pm4_barrier_emit_tail(words, submission->after,
                                                 &release_control);
  if (native_signal) {
    words = iree_hal_amd_gpu_pm4_emit_release32(
        words, release_control, submission->signal->device_address, 1);
    release_control = 0;
  }
  words = iree_hal_amd_gpu_pm4_retirement_emit(
      words, &queue->retirement, position, submission, release_control);
  submission->consumed_words =
      reservation.position + (words - reservation.words);
  words = iree_hal_amd_gpu_pm4_emit_notify32(
      words, 0, queue->context->event_info.mailbox_address,
      queue->context->event_info.trigger_data,
      queue->context->event_info.interrupt_context);
  IREE_ASSERT_EQ(words - reservation.words, body_words);
  queue->issued[position & queue->retirement.capacity_mask] = submission;
  iree_hal_amd_gpu_pm4_submission_retain(submission);
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
  const uint64_t published_words = queue->producer.ring.reserved_words;
  iree_atomic_store(queue->producer.write_index, published_words,
                    iree_memory_order_release);
  const iree_hal_memory_transition_recipe_info_t publication_fence = {
      .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
  };
  iree_hal_memory_transition_execute_host_direct(&publication_fence, NULL, 1);
  *queue->producer.doorbell = published_words;
  for (iree_host_size_t i = 0; native_signal && i < submission->signals.count &&
                               iree_status_is_ok(status);
       ++i) {
    if (iree_hal_amd_gpu_pm4_semaphore_isa(submission->signals.semaphores[i])) {
      status = iree_hal_amd_gpu_pm4_semaphore_publish_signal(
          submission->signals.semaphores[i],
          submission->signals.payload_values[i],
          (iree_async_frontier_entry_t){0}, submission->signal);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_umq_fail_signals(queue, submission->signals, status);
  }
  iree_hal_amd_gpu_pm4_submission_release(submission);
  return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
}

// Publication mode and whether dependencies are ready determine ownership;
// enqueue order does not supply a HAL execution dependency.
static void iree_hal_amd_gpu_pm4_umq_enqueue_ready(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (!iree_hal_amd_gpu_publisher_enqueue(&queue->publisher,
                                          &submission->publication)) {
    return;
  }
  iree_hal_amd_gpu_publication_list_t remainder = {0};
  if (iree_hal_amd_gpu_pm4_umq_publish(queue, submission) !=
      IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED) {
    remainder.head = remainder.tail = &submission->publication;
  }
  if (iree_hal_amd_gpu_publisher_handoff(&queue->publisher, remainder) ||
      iree_atomic_load(&queue->native_failure, iree_memory_order_acquire)) {
    iree_hal_amd_gpu_continuation_notify(&queue->continuation);
  }
  iree_hal_amd_gpu_pm4_umq_wake_shutdown(queue);
}

typedef enum iree_hal_amd_gpu_pm4_wait_kind_e {
  IREE_HAL_AMD_GPU_PM4_WAIT_READY,
  IREE_HAL_AMD_GPU_PM4_WAIT_COMPLETION,
  IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION,
} iree_hal_amd_gpu_pm4_wait_kind_t;

// An unresolved edge leaves no native reader pinned. In particular, a consumer
// awaiting a future producer cannot exhaust that producer's signal backing.
static iree_status_t iree_hal_amd_gpu_pm4_umq_admit(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_host_size_t* out_wait_index,
    iree_hal_amd_gpu_pm4_wait_kind_t* out_kind) {
  *out_kind = IREE_HAL_AMD_GPU_PM4_WAIT_READY;
  // A bounded direct publication must fit strictly inside half the ring.
  // Large fan-in joins use completed host timelines, leaving native command
  // capacity independent of an unbounded public semaphore list.
  const uint32_t word_budget =
      iree_min(queue->producer.ring.capacity / 2, 16384u);
  const bool native_eligible =
      submission->wait_count <= (word_budget - 128) / 7;
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
  }
  return status;
}

// The sidecar registration continuation is recursive through the service
// queue: immediate semaphore completion never recursively admits more work.
static void iree_hal_amd_gpu_pm4_umq_pending_resume(void* user_data,
                                                    iree_status_t status);

static void iree_hal_amd_gpu_pm4_umq_pending_begin(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_pm4_pending_t* pending = (void*)entry;
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = pending->queue;
  iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
  const amdf_status_t failure = (amdf_status_t)iree_atomic_load(
      &queue->native_failure, iree_memory_order_acquire);
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(failure, "PM4 native completion");
  iree_host_size_t wait_index = 0;
  iree_hal_amd_gpu_pm4_wait_kind_t kind = IREE_HAL_AMD_GPU_PM4_WAIT_READY;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_gpu_pm4_umq_admit(queue, submission, &wait_index, &kind);
  }
  if (iree_status_is_ok(status) && kind != IREE_HAL_AMD_GPU_PM4_WAIT_READY) {
    const iree_hal_amd_gpu_pm4_submission_wait_t* wait =
        &submission->waits[wait_index];
    pending->value = wait->value;
    if (kind == IREE_HAL_AMD_GPU_PM4_WAIT_SUBMISSION) {
      status = iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
          wait->semaphore, &pending->semaphore);
    } else {
      pending->semaphore = wait->semaphore;
      iree_hal_semaphore_retain(pending->semaphore);
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
          iree_hal_amd_gpu_pm4_umq_pending_resume, pending,
          queue->host_allocator, &pending->wait);
      if (iree_status_is_ok(status)) {
        return;
      }
      queue->pending = pending->next;
      if (pending->next) {
        pending->next->previous = NULL;
      }
    }
    iree_hal_semaphore_release(pending->semaphore);
  }
  iree_allocator_free(queue->host_allocator, pending);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_umq_enqueue_ready(queue, submission);
  } else {
    iree_hal_amd_gpu_pm4_umq_abandon(queue, submission, status);
  }
}

static void iree_hal_amd_gpu_pm4_umq_pending_resume(void* user_data,
                                                    iree_status_t status) {
  iree_hal_amd_gpu_pm4_pending_t* pending = user_data;
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = pending->queue;
  if (pending->previous) {
    pending->previous->next = pending->next;
  } else {
    queue->pending = pending->next;
  }
  if (pending->next) {
    pending->next->previous = pending->previous;
  }
  iree_hal_semaphore_release(pending->semaphore);
  pending->semaphore = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_umq_pending_begin(&pending->entry);
  } else {
    iree_hal_amd_gpu_pm4_submission_t* submission = pending->submission;
    iree_allocator_free(queue->host_allocator, pending);
    iree_hal_amd_gpu_pm4_umq_abandon(queue, submission, status);
  }
}

// Returns synchronous capture/admission failures before acceptance. Once linked
// or registered, terminal completion belongs to signals and the event sink.
static iree_status_t iree_hal_amd_gpu_pm4_umq_submit(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_host_size_t wait_index = 0;
  iree_hal_amd_gpu_pm4_wait_kind_t kind;
  iree_status_t status =
      iree_hal_amd_gpu_pm4_umq_admit(queue, submission, &wait_index, &kind);
  iree_hal_amd_gpu_pm4_pending_t* pending = NULL;
  if (iree_status_is_ok(status) && kind != IREE_HAL_AMD_GPU_PM4_WAIT_READY) {
    status = iree_allocator_malloc(queue->host_allocator, sizeof(*pending),
                                   (void**)&pending);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_submission_release(submission);
    return status;
  }
  iree_atomic_fetch_add(&queue->outstanding, 1, iree_memory_order_relaxed);
  if (pending) {
    pending->queue = queue;
    pending->submission = submission;
    pending->entry.fn = iree_hal_amd_gpu_pm4_umq_pending_begin;
    iree_hal_memory_maintenance_enqueue(queue->options.service,
                                        &pending->entry);
  } else {
    iree_hal_amd_gpu_pm4_umq_enqueue_ready(queue, submission);
  }
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_pm4_umq_progress(void* user_data) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = user_data;
  iree_hal_amd_gpu_publication_list_t work =
      iree_hal_amd_gpu_publisher_take(&queue->publisher);
  bool more = false;
  if (work.head) {
    iree_hal_amd_gpu_pm4_umq_trim_arguments(queue);
    bool capacity = false;
    for (uint32_t i = 0; work.head && i < 32 && !capacity; ++i) {
      iree_hal_amd_gpu_publication_entry_t* next = work.head->next;
      iree_hal_amd_gpu_pm4_submission_t* submission = iree_containerof(
          work.head, iree_hal_amd_gpu_pm4_submission_t, publication);
      const iree_hal_amd_gpu_pm4_publication_result_t result =
          iree_hal_amd_gpu_pm4_umq_publish(queue, submission);
      iree_status_t status = iree_ok_status();
      switch (result) {
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED:
          work.head = next;
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY:
          capacity = true;
          break;
        case IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS:
          status = iree_hal_amd_gpu_pm4_umq_grow_arguments(
              queue, iree_max(submission->function->kernarg->byte_length,
                              submission->function->kernarg->alignment));
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
            status = iree_hal_amd_gpu_pm4_umq_grow_signals(queue);
          }
          break;
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_pm4_umq_abandon(queue, submission, status);
        work.head = next;
      }
    }
    more = iree_hal_amd_gpu_publisher_handoff(&queue->publisher, work) &&
           !capacity;
    iree_hal_amd_gpu_pm4_umq_wake_shutdown(queue);
  }
  const amdf_status_t failure = (amdf_status_t)iree_atomic_load(
      &queue->native_failure, iree_memory_order_acquire);
  if (!amdf_status_is_ok(failure)) {
    for (iree_hal_amd_gpu_pm4_pending_t* pending = queue->pending; pending;
         pending = pending->next) {
      iree_hal_amd_gpu_host_wait_cancel(&pending->wait);
    }
    if (!queue->issued_failure_delivered &&
        iree_hal_amd_gpu_publisher_is_idle(&queue->publisher)) {
      const uint64_t begin = (uint64_t)iree_atomic_load(
          &queue->completed.count, iree_memory_order_acquire);
      const uint64_t end = (uint64_t)iree_atomic_load(
          &queue->producer.count, iree_memory_order_acquire);
      for (uint64_t i = begin; i != end; ++i) {
        iree_hal_amd_gpu_pm4_submission_t* submission =
            queue->issued[i & queue->retirement.capacity_mask];
        iree_hal_amd_gpu_pm4_umq_fail_signals(
            queue, submission->signals,
            IREE_HAL_AMD_STATUS_FROM_AMDF(failure, "PM4 native completion"));
      }
      queue->issued_failure_delivered = true;
    }
  }
  return more;
}

static bool iree_hal_amd_gpu_pm4_umq_retire(void* user_data,
                                            amdf_status_t wait_status) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = user_data;
  if (!amdf_status_is_ok(wait_status)) {
    iree_atomic_store(&queue->native_failure, wait_status,
                      iree_memory_order_release);
    iree_hal_amd_gpu_context_report(queue->context, wait_status,
                                    "GPU completion event_wait");
    iree_hal_amd_gpu_continuation_notify(&queue->continuation);
    return false;
  }
  uint64_t position = (uint64_t)iree_atomic_load(&queue->completed.count,
                                                 iree_memory_order_relaxed);
  const uint64_t published = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_acquire);
  const uint64_t limit = iree_min(published, position + 64);
  const uint64_t initial = position;
  for (; position != limit; ++position) {
    iree_hal_amd_gpu_pm4_submission_t* submission =
        iree_hal_amd_gpu_pm4_retirement_poll(&queue->retirement, position);
    if (!submission) {
      break;
    }
    const uint64_t consumed_words = submission->consumed_words;
    const uint64_t consumed_arguments = submission->consumed_arguments;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < submission->signals.count && iree_status_is_ok(status); ++i) {
      iree_hal_semaphore_t* semaphore = submission->signals.semaphores[i];
      const uint64_t value = submission->signals.payload_values[i];
      status = iree_hal_amd_gpu_pm4_semaphore_isa(semaphore)
                   ? iree_hal_amd_gpu_pm4_semaphore_publish_completed(
                         semaphore, value, NULL)
                   : iree_hal_semaphore_signal(semaphore, value, NULL);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_umq_fail_signals(queue, submission->signals, status);
    }
    iree_hal_amd_gpu_pm4_umq_complete(queue, submission);
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
  return position != published && iree_hal_amd_gpu_pm4_retirement_poll(
                                      &queue->retirement, position) != NULL;
}

iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_pm4_umq_queue_completion_source(iree_hal_queue_t* base) {
  return (iree_hal_amd_gpu_completion_source_t){iree_hal_amd_gpu_pm4_umq_retire,
                                                base};
}

void iree_hal_amd_gpu_pm4_umq_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_pm4_umq_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_pm4_umq_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  return !iree_atomic_load(&queue->outstanding, iree_memory_order_acquire) &&
         iree_hal_amd_gpu_publisher_is_idle(&queue->publisher);
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_check_acceptance(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue) {
  if (iree_atomic_load(&queue->closed, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "GPU queue is closed");
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      (amdf_status_t)iree_atomic_load(&queue->native_failure,
                                      iree_memory_order_acquire),
      "PM4 native completion");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_umq_check_acceptance(queue));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, waits, signals, NULL, NULL,
      (iree_hal_amd_gpu_pm4_dispatch_t){0}, iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_empty(), &queue->options.barriers, barriers, 0,
      &submission));
  return iree_hal_amd_gpu_pm4_umq_submit(queue, submission);
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_dispatch(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_umq_check_acceptance(queue));
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_pm4(executable, token, &function));
  iree_hal_amd_gpu_pm4_dispatch_t dispatch;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_dispatch_prepare(
      function,
      (uint32_t)
          queue->context->target_info.compute.local_data_share_byte_length,
      config, constants, bindings, flags, &dispatch));
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, waits, signals, executable, function, dispatch,
      constants, bindings, &queue->options.barriers, barriers, flags,
      &submission));
  return iree_hal_amd_gpu_pm4_umq_submit(queue, submission);
}

//===----------------------------------------------------------------------===//
// Cold native construction and joined teardown
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_pm4_umq_prepare_t {
  // Queue under exclusive construction ownership.
  iree_hal_amd_gpu_pm4_umq_queue_t* queue;
  // Construction status transferred to the joining caller.
  iree_status_t status;
} iree_hal_amd_gpu_pm4_umq_prepare_t;

static iree_status_t iree_hal_amd_gpu_pm4_umq_prepare_native(
    iree_hal_amd_gpu_pm4_umq_queue_t* queue) {
  amdf_user_queue_mapping_info_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_ring_storage_initialize(
      queue->context, &queue->options.native, &queue->storage,
      &queue->producer.ring, &mapping));
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
  uint64_t control_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_umq_allocate_memory(
      queue,
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
                       .queue_family_ordinal =
                           queue->options.native.queue_family_ordinal},
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
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_umq_grow_arguments(
      queue, queue->options.argument_capacity));
  return iree_hal_amd_gpu_pm4_umq_grow_signals(queue);
}

static void iree_hal_amd_gpu_pm4_umq_prepare(void* user_data) {
  iree_hal_amd_gpu_pm4_umq_prepare_t* prepare = user_data;
  prepare->status = iree_hal_amd_gpu_pm4_umq_prepare_native(prepare->queue);
}

typedef struct iree_hal_amd_gpu_pm4_umq_cleanup_t {
  // Joined queue whose native handles may be removed.
  iree_hal_amd_gpu_pm4_umq_queue_t* queue;
  // Raw native join/removal result, transferred to the parent.
  amdf_status_t status;
} iree_hal_amd_gpu_pm4_umq_cleanup_t;

static void iree_hal_amd_gpu_pm4_umq_cleanup(void* user_data) {
  iree_hal_amd_gpu_pm4_umq_cleanup_t* cleanup = user_data;
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = cleanup->queue;
  cleanup->status = iree_hal_amd_gpu_pm4_ring_storage_deinitialize(
      &queue->storage, queue->producer.ring.reserved_words);
  if (!amdf_status_is_ok(cleanup->status)) {
    return;
  }
  while (queue->producer.signal_storage) {
    iree_hal_amd_gpu_pm4_signal_storage_t* storage =
        queue->producer.signal_storage;
    queue->producer.signal_storage = storage->next;
    iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&storage->pool);
    iree_hal_amd_fabric_memory_deinitialize(
        queue->context->api, queue->context->event_sink, &storage->memory);
    iree_allocator_free(queue->host_allocator, storage);
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
    iree_hal_amd_fabric_memory_deinitialize(
        queue->context->api, queue->context->event_sink, &arguments->memory);
    iree_allocator_free(queue->host_allocator, arguments);
    arguments = next;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      queue->context->api, queue->context->event_sink, &queue->control);
}

amdf_status_t iree_hal_amd_gpu_pm4_umq_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_pm4_umq_queue_is_idle(base),
              "shutdown requires joined accepted work");
  iree_hal_amd_gpu_continuation_deinitialize(&queue->continuation);
  iree_hal_amd_gpu_pm4_umq_cleanup_t cleanup = {.queue = queue};
  iree_hal_memory_maintenance_call(queue->options.service,
                                   iree_hal_amd_gpu_pm4_umq_cleanup, &cleanup);
  return cleanup.status;
}

static void iree_hal_amd_gpu_pm4_umq_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  IREE_ASSERT(!queue->storage.queue && !queue->control.handle,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_arena_block_pool_deinitialize(&queue->captures);
  iree_allocator_free_aligned(queue->host_allocator, queue);
}

iree_status_t iree_hal_amd_gpu_pm4_umq_queue_create(
    const iree_hal_queue_family_t* family,
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_pm4_umq_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  if (iree_cpu_architecture_host() != IREE_CPU_ARCHITECTURE_X86_64 ||
      !iree_atomic_int32_is_lock_free() || !iree_atomic_int64_is_lock_free()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 USER publication requires the qualified "
                            "x86-64 host store-fence and atomic contract");
  }
  const amdf_queue_family_info_t* native =
      &context->families[options->native.queue_family_ordinal];
  const uint64_t required_cache = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
  if (native->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ||
      native->format_version != AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 ||
      !iree_all_bits_set(native->roles, AMDF_QUEUE_ROLE_COMPUTE |
                                            AMDF_QUEUE_ROLE_ATOMIC |
                                            AMDF_QUEUE_ROLE_CACHE_CONTROL) ||
      !iree_any_bit_set(native->publication_modes,
                        AMDF_QUEUE_PUBLICATION_MODE_USER) ||
      !iree_any_bit_set(native->user_queue_capabilities,
                        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) ||
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
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native family lacks the PM4 USER dispatch, cache, "
                            "or completion-point contract");
  }
  if (context->event_info.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
      context->event_info.format_version != 1 ||
      !iree_any_bit_set(context->event_info.publication_modes,
                        AMDF_QUEUE_PUBLICATION_MODE_USER)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 USER completion requires a qualified native event producer");
  }
  if (!options->retirement_capacity ||
      (options->retirement_capacity & (options->retirement_capacity - 1)) ||
      !options->argument_capacity ||
      (options->argument_capacity & (options->argument_capacity - 1)) ||
      options->native.ring_byte_length < 2048 ||
      options->native.producer_mode != AMDF_QUEUE_PRODUCER_MODE_SINGLE ||
      !iree_arena_block_pool_is_valid_total_size(options->capture_block_size) ||
      options->capture_block_size < sizeof(iree_hal_amd_gpu_pm4_submission_t) +
                                        sizeof(iree_arena_block_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid PM4 queue capacity or producer options");
  }
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_pm4_umq_queue_t);
  if (!iree_host_size_checked_mul_add(
          total_size, options->retirement_capacity,
          sizeof(iree_hal_amd_gpu_pm4_submission_t*), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, params->execution_resources.count,
          sizeof(iree_hal_queue_execution_resource_ordinal_t), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 queue allocation overflows");
  }
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, iree_alignof(*queue), 0, (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->storage.context = context;
  queue->options = *options;
  queue->issued = (iree_hal_amd_gpu_pm4_submission_t**)(queue + 1);
  iree_hal_queue_params_t achieved = *params;
  iree_hal_queue_execution_resource_ordinal_t* ordinals =
      (iree_hal_queue_execution_resource_ordinal_t*)(queue->issued +
                                                     options
                                                         ->retirement_capacity);
  if (params->execution_resources.count) {
    memcpy(ordinals, params->execution_resources.ordinals,
           params->execution_resources.count * sizeof(*ordinals));
    achieved.execution_resources.ordinals = ordinals;
  }
  iree_hal_queue_initialize(
      family, &achieved, &iree_hal_amd_gpu_pm4_umq_queue_vtable, &queue->base);
  iree_hal_amd_gpu_publisher_initialize(&queue->publisher);
  iree_async_semaphore_wait_context_initialize(&queue->wait_context);
  iree_hal_amd_gpu_continuation_initialize(options->service,
                                           iree_hal_amd_gpu_pm4_umq_progress,
                                           queue, &queue->continuation);
  iree_arena_block_pool_initialize(options->capture_block_size, host_allocator,
                                   &queue->captures);
  iree_status_t status = iree_arena_block_pool_preallocate(
      &queue->captures, options->capture_block_count);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_umq_prepare_t prepare = {.queue = queue};
    iree_hal_memory_maintenance_call(
        options->service, iree_hal_amd_gpu_pm4_umq_prepare, &prepare);
    status = prepare.status;
  }
  if (iree_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    const amdf_status_t cleanup =
        iree_hal_amd_gpu_pm4_umq_queue_shutdown(&queue->base);
    if (amdf_status_is_ok(cleanup)) {
      iree_hal_queue_release(&queue->base);
    } else {
      iree_hal_amd_gpu_context_report(context, cleanup,
                                      "PM4 USER construction rollback");
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

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_execute(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer execution is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_host_call(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t arguments[4], iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 host-call retirement is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PM4 dispatch residency calculation is not implemented");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_atomic_wait(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic wait is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_atomic_store(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic store is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_atomic_rmw(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public atomic RMW is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_timestamp(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public timestamp is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_flush(
    iree_hal_queue_t* base) {
  iree_hal_amd_gpu_pm4_umq_queue_t* queue = (void*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_umq_check_acceptance(queue));
  iree_hal_amd_gpu_continuation_notify(&queue->continuation);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_alloca(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 queue allocation is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_dealloca(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 queue deallocation is not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_transfer(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 public transfers are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_read(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* source,
    uint64_t source_offset, iree_hal_buffer_t* target,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 file reads are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_umq_queue_write(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source,
    iree_device_size_t source_offset, iree_hal_file_t* target,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 file writes are not admitted");
}

static const iree_hal_queue_vtable_t iree_hal_amd_gpu_pm4_umq_queue_vtable = {
    .destroy = iree_hal_amd_gpu_pm4_umq_queue_destroy,
    .barrier = iree_hal_amd_gpu_pm4_umq_queue_barrier,
    .execute = iree_hal_amd_gpu_pm4_umq_queue_execute,
    .host_call = iree_hal_amd_gpu_pm4_umq_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_gpu_pm4_umq_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_gpu_pm4_umq_queue_dispatch,
    .atomic_wait = iree_hal_amd_gpu_pm4_umq_queue_atomic_wait,
    .atomic_store = iree_hal_amd_gpu_pm4_umq_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_gpu_pm4_umq_queue_atomic_rmw,
    .timestamp = iree_hal_amd_gpu_pm4_umq_queue_timestamp,
    .flush = iree_hal_amd_gpu_pm4_umq_queue_flush,
    .alloca = iree_hal_amd_gpu_pm4_umq_queue_alloca,
    .dealloca = iree_hal_amd_gpu_pm4_umq_queue_dealloca,
    .transfer = iree_hal_amd_gpu_pm4_umq_queue_transfer,
    .read = iree_hal_amd_gpu_pm4_umq_queue_read,
    .write = iree_hal_amd_gpu_pm4_umq_queue_write,
};
