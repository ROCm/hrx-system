// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/allocation.h"

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/amd/fabric/transient_buffer.h"

typedef enum iree_hal_amd_gpu_allocation_phase_e {
  IREE_HAL_AMD_GPU_ALLOCATION_ACQUIRE,
  IREE_HAL_AMD_GPU_ALLOCATION_MATERIALIZE,
  IREE_HAL_AMD_GPU_ALLOCATION_RELEASE,
} iree_hal_amd_gpu_allocation_phase_t;

#define IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY 8
IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_amd_gpu_allocation_frontier_t,
                               IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY);

// Captured pool transaction. Only genuinely deferred native acquisition
// results require a publisher handoff; warm acquisition and return are inline.
typedef struct iree_hal_amd_gpu_allocation_t {
  // Typed completed-acquisition link, unused for warm pool hits.
  iree_atomic_slist_entry_t entry;
  // Borrowed embedded state owned by the application-held device.
  iree_hal_amd_gpu_allocation_context_t* context;
  // Borrowed parent for diagnostics and transient-root placement.
  iree_hal_device_t* device;
  // Owns this object and every variable-length capture allocation.
  iree_arena_allocator_t arena;
  // Copied and retained input dependencies, joined before native pool work.
  iree_hal_semaphore_list_t waits;
  // Copied and retained terminal signals.
  iree_hal_semaphore_list_t signals;
  // Exclusive phase after initial semaphore admission.
  iree_hal_amd_gpu_allocation_phase_t phase;
  // Owned acquisition result consumed after the submit/callback join.
  iree_status_t status;
  // Joins the pool's submitting call and its potentially inline callback.
  iree_atomic_int32_t pending;
  // Exact observed prerequisite history, when representable.
  iree_hal_amd_gpu_allocation_frontier_t frontier;
  // False when bounded or tainted history cannot prove early reuse.
  bool frontier_exact;
  // Complete transaction size; arrays contain count entries.
  iree_host_size_t count;
  // Initialized retained roots; NULL entries represent capture rollback.
  iree_hal_buffer_t** buffers;
  // Reserved allocation outputs or transferred deallocation inputs.
  iree_hal_pool_reservation_t* reservations;
  // Completed dependency registration, using metadata from arena.
  iree_hal_amd_gpu_host_wait_t wait;
  union {
    // Allocation acquisition and complete backing publication.
    struct {
      // Borrowed source, live through the returned roots and this transaction.
      iree_hal_pool_t* pool;
      // Result returned by the whole pool acquisition transaction.
      iree_hal_pool_acquire_result_t result;
      // Copied requested extents and alignment.
      iree_hal_pool_reservation_request_t* requests;
      // Pool-owned reuse frontiers, borrowed while reservations are held.
      iree_hal_pool_acquire_info_t* infos;
      // Prepared backing views, committed only after the whole batch succeeds.
      iree_hal_pool_reservation_view_t* views;
      // Temporary materializations for pools without direct reservation views.
      iree_hal_buffer_t** materialized;
      // True until all acquired reservations transfer to their roots.
      bool held;
      // First reuse frontier not yet joined; each is visited only once.
      iree_host_size_t frontier_index;
      // Active borrowed frontier wait, joined before further materialization.
      iree_async_frontier_waiter_t frontier_wait;
      // Lazily allocated capacity observation, reused across attempts.
      iree_hal_pool_wait_t* capacity_wait;
    } acquire;
    // Captured logical deallocation epochs.
    struct {
      // Borrowed common source, live through every captured allocation root.
      iree_hal_pool_t* pool;
      // Prefix of exclusive deallocation marks to abort on failure.
      iree_host_size_t marked_count;
    } release;
  };
} iree_hal_amd_gpu_allocation_t;

void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, const iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context) {
  *out_context = (iree_hal_amd_gpu_allocation_context_t){
      .device = device,
      .context = context,
      .publication = publication,
      .publication_ordinal = publication_ordinal};
  iree_atomic_slist_initialize(&out_context->ready);
  iree_arena_block_pool_initialize(1024, host_allocator, &out_context->blocks);
  iree_async_semaphore_wait_context_initialize(&out_context->waits);
}

void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context) {
  iree_atomic_slist_deinitialize(&context->ready);
  iree_async_semaphore_wait_context_deinitialize(&context->waits);
  iree_arena_block_pool_deinitialize(&context->blocks);
}

// Returns only reservations still owned by a failed allocation transaction.
// Their original prerequisites remain attached when another requester reuses
// them; failure must never promote unjoined storage to immediately reusable.
static void iree_hal_amd_gpu_allocation_rollback(
    iree_hal_amd_gpu_allocation_t* operation) {
  if (operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_RELEASE) {
    for (iree_host_size_t i = 0; i < operation->release.marked_count; ++i) {
      iree_hal_buffer_allocation_abort_dealloca(operation->buffers[i]);
    }
    operation->release.marked_count = 0;
  } else if (operation->acquire.held) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_pool_release_reservations(
          operation->acquire.pool, 1, &operation->reservations[i],
          operation->acquire.infos[i].reuse_frontier);
    }
    operation->acquire.held = false;
  }
}

// Releases all buffer/pool uses before any terminal signal becomes visible.
static void iree_hal_amd_gpu_allocation_retire(
    iree_hal_amd_gpu_allocation_t* operation) {
  iree_hal_amd_gpu_allocation_rollback(operation);
  if (operation->phase != IREE_HAL_AMD_GPU_ALLOCATION_RELEASE) {
    iree_hal_pool_wait_free(operation->acquire.capacity_wait);
    if (operation->acquire.materialized) {
      for (iree_host_size_t i = 0; i < operation->count; ++i) {
        iree_hal_buffer_release(operation->acquire.materialized[i]);
      }
    }
  }
  if (operation->buffers) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_buffer_release(operation->buffers[i]);
    }
  }
}

static void iree_hal_amd_gpu_allocation_free(
    iree_hal_amd_gpu_allocation_t* operation) {
  iree_arena_allocator_t arena = operation->arena;
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_arena_deinitialize(&arena);
}

static void iree_hal_amd_gpu_allocation_complete(
    iree_hal_amd_gpu_allocation_t* operation, iree_status_t status) {
  if (operation->context->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->context->profile,
                                              (void*)(operation + 1), false, 0);
  }
  iree_hal_amd_gpu_allocation_retire(operation);
  if (iree_status_is_ok(status) && operation->context->failure) {
    const iree_status_code_t code = (iree_status_code_t)iree_atomic_load(
        operation->context->failure, iree_memory_order_acquire);
    if (code != IREE_STATUS_OK) {
      status =
          iree_make_status(code, "GPU allocation completed on failed device");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        operation->frontier_exact
            ? iree_async_fixed_frontier_as_const_frontier(&operation->frontier)
            : NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(operation->context->context,
                                           operation->device, status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_allocation_free(operation);
}

static iree_status_t iree_hal_amd_gpu_allocation_materialize(
    iree_hal_amd_gpu_allocation_t* operation) {
  iree_hal_pool_t* pool = operation->acquire.pool;
  if (!iree_hal_pool_query_reservation_views(pool, operation->count,
                                             operation->reservations,
                                             operation->acquire.views)) {
    IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
        pool, operation->count, operation->acquire.requests,
        operation->reservations, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        operation->acquire.materialized));
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_buffer_t* buffer = operation->acquire.materialized[i];
      operation->acquire.views[i] = (iree_hal_pool_reservation_view_t){
          .buffer = buffer,
          .byte_offset = iree_hal_buffer_byte_offset(buffer),
          .byte_length = iree_hal_buffer_byte_length(buffer),
          .memory = iree_hal_buffer_memory_view(buffer),
      };
    }
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < operation->count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_fabric_transient_buffer_stage_backing(
        operation->buffers[i], &operation->acquire.views[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  iree_hal_pool_advise_asan_reservations(
      pool, operation->count, operation->reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
  if (operation->context->failure) {
    const iree_status_code_t code = (iree_status_code_t)iree_atomic_load(
        operation->context->failure, iree_memory_order_acquire);
    if (code != IREE_STATUS_OK) {
      return iree_make_status(code, "GPU allocation shadow publication failed");
    }
  }
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_amd_fabric_transient_buffer_attach_reservation(
        operation->buffers[i], pool, &operation->reservations[i]);
  }
  operation->acquire.held = false;
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_amd_fabric_transient_buffer_commit(operation->buffers[i]);
  }
  return iree_ok_status();
}

static void iree_hal_amd_gpu_allocation_release(
    iree_hal_amd_gpu_allocation_t* operation) {
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_pool_t* pool = NULL;
    iree_hal_buffer_allocation_take_dealloca_reservation(
        operation->buffers[i], &pool, &operation->reservations[i]);
    iree_hal_buffer_allocation_decommit(operation->buffers[i]);
  }
  operation->release.marked_count = 0;
  iree_hal_pool_advise_asan_reservations(
      operation->release.pool, operation->count, operation->reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  iree_hal_pool_release_reservations(
      operation->release.pool, operation->count, operation->reservations,
      operation->frontier_exact
          ? iree_async_fixed_frontier_as_const_frontier(&operation->frontier)
          : NULL);
}

static void iree_hal_amd_gpu_allocation_acquire(
    iree_hal_amd_gpu_allocation_t* operation);
static void iree_hal_amd_gpu_allocation_reuse(
    iree_hal_amd_gpu_allocation_t* operation);

static void iree_hal_amd_gpu_allocation_resume(void* user_data,
                                               iree_status_t status) {
  iree_hal_amd_gpu_allocation_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_allocation_complete(operation, status);
  } else if (operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_ACQUIRE) {
    iree_hal_amd_gpu_allocation_acquire(operation);
  } else {
    iree_hal_amd_gpu_allocation_reuse(operation);
  }
}

static void iree_hal_amd_gpu_allocation_reuse(
    iree_hal_amd_gpu_allocation_t* operation) {
  iree_status_t status = iree_ok_status();
  // Visit each exact reuse frontier once. The compact signal history can
  // overflow without losing the actual dependencies that guard materialization.
  while (operation->acquire.frontier_index < operation->count &&
         iree_status_is_ok(status)) {
    const iree_hal_pool_acquire_info_t* info =
        &operation->acquire.infos[operation->acquire.frontier_index++];
    if (operation->frontier_exact && info->reuse_frontier &&
        !iree_async_frontier_merge(
            iree_async_fixed_frontier_as_frontier(&operation->frontier),
            IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY,
            info->reuse_frontier)) {
      operation->frontier_exact = false;
    }
    if (info->result != IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      continue;
    }
    iree_async_frontier_tracker_t* tracker =
        operation->acquire.pool->frontier_tracker;
    bool satisfied = false;
    status = iree_async_frontier_tracker_query(tracker, info->reuse_frontier,
                                               &satisfied);
    if (iree_status_is_ok(status) && !satisfied) {
      status = iree_async_frontier_tracker_wait(
          tracker, info->reuse_frontier, iree_hal_amd_gpu_allocation_resume,
          operation, &operation->acquire.frontier_wait);
      if (iree_status_is_ok(status)) {
        return;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_allocation_materialize(operation);
  }
  iree_hal_amd_gpu_allocation_complete(operation, status);
}

static void iree_hal_amd_gpu_allocation_finish_acquire(
    iree_hal_amd_gpu_allocation_t* operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = operation->status;
  operation->status = iree_ok_status();
  iree_hal_pool_wait_t* wait = operation->acquire.capacity_wait;
  if (iree_status_is_ok(status) &&
      (operation->acquire.result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
       operation->acquire.result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET)) {
    if (wait) {
      iree_hal_pool_wait_commit(wait, iree_infinite_timeout(),
                                (iree_hal_pool_wait_callback_t){
                                    .fn = iree_hal_amd_gpu_allocation_resume,
                                    .user_data = operation});
    } else {
      // Enroll observation only after a real miss, then retry after observing
      // capacity so a concurrent return cannot be lost.
      status = iree_hal_pool_wait_allocate(
          operation->acquire.pool, iree_arena_allocator(&operation->arena),
          &operation->acquire.capacity_wait);
      iree_hal_amd_gpu_allocation_resume(operation, status);
    }
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  if (wait) {
    iree_hal_pool_wait_abort(wait);
  }
  if (iree_status_is_ok(status)) {
    operation->acquire.held = true;
    operation->phase = IREE_HAL_AMD_GPU_ALLOCATION_MATERIALIZE;
    iree_hal_amd_gpu_allocation_reuse(operation);
  } else {
    iree_hal_amd_gpu_allocation_complete(operation, status);
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amd_gpu_allocation_acquired(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_amd_gpu_allocation_t* operation = user_data;
  operation->status = status;
  operation->acquire.result = result;
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  // A genuinely deferred pool callback can run on the native allocation
  // owner. Publish only its typed result there: committing buffers and issuing
  // dependent GPU work belong to the device publisher. Warm hits finish on
  // their submitting caller after the other borrow is returned below.
  iree_hal_amd_gpu_allocation_context_t* context = operation->context;
  iree_atomic_slist_push(&context->ready, &operation->entry);
  iree_hal_amd_gpu_publication_notify(context->publication,
                                      context->publication_ordinal);
}

static void iree_hal_amd_gpu_allocation_acquire(
    iree_hal_amd_gpu_allocation_t* operation) {
  if (operation->acquire.capacity_wait) {
    iree_hal_pool_wait_prepare(operation->acquire.capacity_wait);
  }
  iree_atomic_store(&operation->pending, 2, iree_memory_order_relaxed);
  iree_status_t status = iree_hal_pool_acquire_reservations(
      operation->acquire.pool, operation->count, operation->acquire.requests,
      operation->frontier_exact
          ? iree_async_fixed_frontier_as_const_frontier(&operation->frontier)
          : NULL,
      operation->reservations, operation->acquire.infos,
      (iree_hal_pool_acquire_callback_t){
          .fn = iree_hal_amd_gpu_allocation_acquired, .user_data = operation},
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER);
  if (!iree_status_is_ok(status)) {
    if (operation->acquire.capacity_wait) {
      iree_hal_pool_wait_abort(operation->acquire.capacity_wait);
    }
    iree_hal_amd_gpu_allocation_complete(operation, status);
    return;
  }
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_amd_gpu_allocation_finish_acquire(operation);
  }
}

static bool iree_hal_amd_gpu_allocation_publish(void* user_data) {
  iree_hal_amd_gpu_allocation_context_t* context = user_data;
  if (!context->ready_head) {
    iree_atomic_slist_flush(&context->ready,
                            IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                            &context->ready_head, NULL);
  }
  for (uint32_t i = 0; i < 32 && context->ready_head; ++i) {
    iree_atomic_slist_entry_t* entry = context->ready_head;
    context->ready_head = entry->next;
    iree_hal_amd_gpu_allocation_finish_acquire(
        iree_containerof(entry, iree_hal_amd_gpu_allocation_t, entry));
  }
  return context->ready_head != NULL;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_allocation_context_publication_source(
    iree_hal_amd_gpu_allocation_context_t* context) {
  return (iree_hal_amd_gpu_publication_source_t){
      .publish = iree_hal_amd_gpu_allocation_publish, .user_data = context};
}

static void iree_hal_amd_gpu_allocation_resolve_frontier(
    iree_hal_amd_gpu_allocation_t* operation) {
  operation->frontier_exact = true;
  iree_async_frontier_t* target =
      iree_async_fixed_frontier_as_frontier(&operation->frontier);
  iree_async_frontier_initialize(target, 0);
  for (iree_host_size_t i = 0; i < operation->waits.count; ++i) {
    iree_async_semaphore_t* semaphore =
        (iree_async_semaphore_t*)operation->waits.semaphores[i];
    iree_hal_amd_gpu_allocation_frontier_t source_storage;
    iree_async_frontier_t* source =
        iree_async_fixed_frontier_as_frontier(&source_storage);
    const uint8_t count = iree_async_semaphore_query_frontier(
        semaphore, source, IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY);
    if (count > IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY ||
        iree_async_semaphore_is_value_tainted(
            semaphore, operation->waits.payload_values[i]) ||
        !iree_async_frontier_merge(
            target, IREE_HAL_AMD_GPU_ALLOCATION_FRONTIER_CAPACITY, source)) {
      operation->frontier_exact = false;
      break;
    }
  }
}

static void iree_hal_amd_gpu_allocation_ready(void* user_data,
                                              iree_status_t status) {
  iree_hal_amd_gpu_allocation_t* operation = user_data;
  if (operation->context->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->context->profile,
                                              (void*)(operation + 1),
                                              iree_status_is_ok(status), 0);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_allocation_complete(operation, status);
    return;
  }
  iree_hal_amd_gpu_allocation_resolve_frontier(operation);
  if (operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_RELEASE) {
    iree_hal_amd_gpu_allocation_release(operation);
    iree_hal_amd_gpu_allocation_complete(operation, iree_ok_status());
  } else {
    iree_hal_amd_gpu_allocation_acquire(operation);
  }
}

static void iree_hal_amd_gpu_allocation_begin(
    iree_hal_amd_gpu_allocation_t* operation) {
  if (!operation->waits.count) {
    iree_hal_amd_gpu_allocation_ready(operation, iree_ok_status());
    return;
  }
  iree_status_t status = iree_hal_amd_gpu_host_wait_begin(
      &operation->context->waits, operation->waits,
      iree_hal_amd_gpu_allocation_ready, operation,
      iree_arena_allocator(&operation->arena), &operation->wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_allocation_complete(operation, status);
  }
}

static iree_status_t iree_hal_amd_gpu_allocation_allocate(
    iree_hal_amd_gpu_allocation_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_amd_gpu_allocation_phase_t phase, iree_host_size_t count,
    iree_hal_amd_gpu_allocation_t** out_operation) {
  *out_operation = NULL;
  iree_arena_allocator_t arena;
  iree_arena_initialize(&context->blocks, &arena);
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status = iree_arena_allocate(
      &arena,
      sizeof(*operation) +
          (context->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0),
      (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  *operation = (iree_hal_amd_gpu_allocation_t){
      .context = context,
      .device = context->device,
      .arena = arena,
      .phase = phase,
      .count = count,
  };
  if (phase == IREE_HAL_AMD_GPU_ALLOCATION_RELEASE) {
    // The default acquire member has padding after its 32-bit result where
    // release has the upper half of this count. Initialize the selected member
    // before any fallible capture can invoke rollback.
    operation->release.marked_count = 0;
  }
  const iree_allocator_t allocator = iree_arena_allocator(&operation->arena);
  status = iree_hal_semaphore_list_clone(&waits, allocator, &operation->waits);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_list_clone(&signals, allocator, &operation->signals);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->buffers),
        (void**)&operation->buffers);
    if (iree_status_is_ok(status)) {
      memset(operation->buffers, 0, count * sizeof(*operation->buffers));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->reservations),
        (void**)&operation->reservations);
  }
  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    iree_hal_amd_gpu_allocation_retire(operation);
    iree_hal_amd_gpu_allocation_free(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_alloca(
    iree_hal_amd_gpu_allocation_context_t* context, uint32_t queue_ordinal,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_pool_t* pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (capabilities.max_allocation_size &&
        requests[i].allocation_size > capabilities.max_allocation_size) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "allocation request %" PRIhsz " exceeds the pool maximum", i);
    }
    if (requests[i].min_alignment > capabilities.max_allocation_alignment) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocation request %" PRIhsz " exceeds the pool alignment", i);
    }
  }
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_allocation_allocate(
      context, waits, signals, IREE_HAL_AMD_GPU_ALLOCATION_ACQUIRE, count,
      &operation));
  operation->acquire.pool = pool;
  const iree_allocator_t allocator = iree_arena_allocator(&operation->arena);
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      allocator, count, sizeof(*operation->acquire.requests),
      (void**)&operation->acquire.requests);
  if (iree_status_is_ok(status)) {
    memcpy(operation->acquire.requests, requests, count * sizeof(*requests));
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->acquire.infos),
        (void**)&operation->acquire.infos);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->acquire.views),
        (void**)&operation->acquire.views);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->acquire.materialized),
        (void**)&operation->acquire.materialized);
    if (iree_status_is_ok(status)) {
      memset(operation->acquire.materialized, 0,
             count * sizeof(*operation->acquire.materialized));
    }
  }
  const iree_hal_buffer_placement_t placement = {
      .device = context->device,
      .queue_family_affinity = capabilities.queue_family_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_fabric_transient_buffer_create(
        placement, capabilities.memory_type,
        pool->memory_contract ? pool->memory_contract->allowed_access
                              : capabilities.allowed_access,
        pool->memory_contract ? pool->memory_contract->allowed_usage
                              : capabilities.supported_usage,
        requests[i].allocation_size, requests[i].allocation_size, pool,
        &context->blocks, &operation->buffers[i]);
  }
  if (iree_status_is_ok(status)) {
    // Admission cannot fail. Publish caller references before it may retire
    // the captured list, so there is no post-submission access to its storage.
    for (iree_host_size_t i = 0; i < count; ++i) {
      out_buffers[i] = operation->buffers[i];
      iree_hal_buffer_retain(out_buffers[i]);
    }
    if (context->profile) {
      iree_hal_amd_gpu_profile_operation_t* profile_operation =
          (void*)(operation + 1);
      iree_hal_amd_gpu_profile_operation_initialize(
          context->profile, queue_ordinal,
          IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA, waits.count, signals.count,
          count, profile_operation);
      for (iree_host_size_t i = 0; i < count; ++i) {
        profile_operation->event.payload_length +=
            iree_min(requests[i].allocation_size,
                     UINT64_MAX - profile_operation->event.payload_length);
      }
      iree_hal_amd_gpu_profile_operation_defer(profile_operation);
    }
    iree_hal_amd_gpu_allocation_begin(operation);
  } else {
    iree_hal_amd_gpu_allocation_retire(operation);
    iree_hal_amd_gpu_allocation_free(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context, uint32_t queue_ordinal,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t count, iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_allocation_allocate(
      context, waits, signals, IREE_HAL_AMD_GPU_ALLOCATION_RELEASE, count,
      &operation));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_pool_t* source_pool = NULL;
    status =
        iree_hal_buffer_allocation_begin_dealloca(buffers[i], &source_pool);
    if (iree_status_is_ok(status)) {
      operation->buffers[i] = buffers[i];
      iree_hal_buffer_retain(buffers[i]);
      ++operation->release.marked_count;
      if (i == 0) {
        operation->release.pool = source_pool;
      } else if (source_pool != operation->release.pool) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "GPU deallocation spans multiple source pools");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    if (context->profile) {
      iree_hal_amd_gpu_profile_operation_t* profile_operation =
          (void*)(operation + 1);
      iree_hal_amd_gpu_profile_operation_initialize(
          context->profile, queue_ordinal,
          IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA, waits.count,
          signals.count, count, profile_operation);
      for (iree_host_size_t i = 0; i < count; ++i) {
        profile_operation->event.payload_length +=
            iree_min(iree_hal_buffer_byte_length(buffers[i]),
                     UINT64_MAX - profile_operation->event.payload_length);
      }
      iree_hal_amd_gpu_profile_operation_defer(profile_operation);
    }
    iree_hal_amd_gpu_allocation_begin(operation);
  } else {
    iree_hal_amd_gpu_allocation_retire(operation);
    iree_hal_amd_gpu_allocation_free(operation);
  }
  return status;
}
