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
  IREE_HAL_AMD_GPU_ALLOCATION_PREPARED,
  IREE_HAL_AMD_GPU_ALLOCATION_ACCEPTED,
} iree_hal_amd_gpu_allocation_phase_t;

// Captured pool transaction. Only genuinely deferred native acquisition
// results require a publisher handoff; warm acquisition and return are inline.
struct iree_hal_amd_gpu_allocation_t {
  // Typed completed-acquisition link, unused for warm pool hits.
  iree_atomic_slist_entry_t entry;
  // Borrowed embedded state owned by the application-held device.
  iree_hal_amd_gpu_allocation_context_t* context;
  // Owns this object and every variable-length capture allocation.
  iree_arena_allocator_t arena;
  // Borrowed source; the application keeps it live through roots and
  // operations.
  iree_hal_pool_t* pool;
  // Pool progress and the native acceptance boundary.
  iree_hal_amd_gpu_allocation_phase_t phase;
  // Selects captured deallocation marks instead of acquisition state.
  bool is_deallocation;
  // Owned acquisition result consumed after the submit/callback join.
  iree_status_t status;
  // Joins the pool's submitting call and its potentially inline callback.
  iree_atomic_int32_t pending;
  // Owned canonical snapshot of the admitted native dependency prefix.
  iree_async_frontier_t* requester;
  // Whole preparation joins its caller separately from individual pool calls.
  struct {
    // Two borrowers: prepare() and its terminal result.
    iree_atomic_int32_t pending;
    // Terminal result transferred to the caller or ready callback.
    iree_status_t status;
    // Resumes the exact queue only when preparation actually deferred.
    void (*ready)(void* user_data, iree_status_t status);
    // Queue-owned capture, live through the final ready callback access.
    void* user_data;
  } preparation;
  // Complete transaction size; arrays contain count entries.
  iree_host_size_t count;
  // Initialized retained roots; NULL entries represent capture rollback.
  iree_hal_buffer_t** buffers;
  // Reserved allocation outputs or transferred deallocation inputs.
  iree_hal_pool_reservation_t* reservations;
  union {
    // Allocation acquisition and complete backing publication.
    struct {
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
      // Prefix of exclusive deallocation marks to abort on failure.
      iree_host_size_t marked_count;
    } release;
  };
};

void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, iree_hal_amd_gpu_publication_t* publication,
    uint32_t publication_ordinal, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context) {
  *out_context = (iree_hal_amd_gpu_allocation_context_t){
      .device = device,
      .publication = publication,
      .publication_ordinal = publication_ordinal};
  iree_atomic_slist_initialize(&out_context->ready);
  iree_arena_block_pool_initialize(1024, host_allocator, &out_context->blocks);
}

void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context) {
  iree_atomic_slist_deinitialize(&context->ready);
  iree_arena_block_pool_deinitialize(&context->blocks);
}

void iree_hal_amd_gpu_allocation_destroy(
    iree_hal_amd_gpu_allocation_t* operation) {
  if (!operation) {
    return;
  }
  if (operation->is_deallocation) {
    if (operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_ACCEPTED) {
      for (iree_host_size_t i = 0; i < operation->count; ++i) {
        iree_hal_buffer_allocation_decommit(operation->buffers[i]);
      }
    } else {
      for (iree_host_size_t i = 0; i < operation->release.marked_count; ++i) {
        iree_hal_buffer_allocation_abort_dealloca(operation->buffers[i]);
      }
    }
  } else {
    iree_hal_pool_wait_free(operation->acquire.capacity_wait);
    if (operation->acquire.materialized) {
      for (iree_host_size_t i = 0; i < operation->count; ++i) {
        iree_hal_buffer_release(operation->acquire.materialized[i]);
      }
    }
    if (operation->acquire.held) {
      // Drop staged views before returning unused tokens with their original,
      // unjoined prerequisites. No accepted native use owns these roots yet.
      for (iree_host_size_t i = 0; i < operation->count; ++i) {
        iree_hal_buffer_allocation_decommit(operation->buffers[i]);
        iree_hal_pool_release_reservations(
            operation->pool, 1, &operation->reservations[i],
            operation->acquire.infos[i].reuse_frontier);
      }
    }
  }
  if (operation->buffers) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_buffer_release(operation->buffers[i]);
    }
  }
  iree_arena_allocator_t arena = operation->arena;
  iree_arena_deinitialize(&arena);
}

static void iree_hal_amd_gpu_allocation_complete(
    iree_hal_amd_gpu_allocation_t* operation, iree_status_t status) {
  operation->phase = IREE_HAL_AMD_GPU_ALLOCATION_PREPARED;
  operation->preparation.status = status;
  if (iree_atomic_fetch_sub(&operation->preparation.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    void* user_data = operation->preparation.user_data;
    void (*ready)(void*, iree_status_t) = operation->preparation.ready;
    operation->preparation.status = iree_ok_status();
    ready(user_data, status);
  }
}

static iree_status_t iree_hal_amd_gpu_allocation_materialize(
    iree_hal_amd_gpu_allocation_t* operation) {
  iree_hal_pool_t* pool = operation->pool;
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
  return status;
}

bool iree_hal_amd_gpu_allocation_requires_host_wait(
    const iree_hal_amd_gpu_allocation_t* operation) {
  return iree_hal_pool_requires_asan_advice(operation->pool);
}

void iree_hal_amd_gpu_allocation_accept(
    iree_hal_amd_gpu_allocation_t* operation,
    const iree_async_frontier_t* death_frontier) {
  if (operation->is_deallocation) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_pool_t* pool = NULL;
      iree_hal_buffer_allocation_take_dealloca_reservation(
          operation->buffers[i], &pool, &operation->reservations[i]);
    }
    operation->release.marked_count = 0;
    iree_hal_pool_advise_asan_reservations(
        operation->pool, operation->count, operation->reservations,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    iree_hal_pool_release_reservations(operation->pool, operation->count,
                                       operation->reservations, death_frontier);
  } else {
    iree_hal_pool_advise_asan_reservations(
        operation->pool, operation->count, operation->reservations,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_amd_fabric_transient_buffer_attach_reservation(
          operation->buffers[i], operation->pool, &operation->reservations[i]);
      iree_hal_amd_fabric_transient_buffer_commit(operation->buffers[i]);
    }
    operation->acquire.held = false;
  }
  operation->phase = IREE_HAL_AMD_GPU_ALLOCATION_ACCEPTED;
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
  const bool guarded = iree_hal_pool_requires_asan_advice(operation->pool);
  while (operation->acquire.frontier_index < operation->count &&
         iree_status_is_ok(status)) {
    const iree_hal_pool_acquire_info_t* info =
        &operation->acquire.infos[operation->acquire.frontier_index++];
    // Dominated native reuse needs no host retirement. Host poisoning still
    // requires actual completion, including a dominated history. An uncovered
    // reuse dependency retains its explicit completion join.
    if (!info->reuse_frontier ||
        (!guarded && info->result != IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT)) {
      continue;
    }
    iree_async_frontier_tracker_t* tracker = operation->pool->frontier_tracker;
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
          operation->pool, iree_arena_allocator(&operation->arena),
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
      operation->pool, operation->count, operation->acquire.requests,
      operation->requester, operation->reservations, operation->acquire.infos,
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

// Converts admitted native epochs once. A fully populated 256-coordinate
// domain exceeds the canonical frontier's uint8 count; that case uses the
// pool's explicit reuse prerequisites instead of truncating the causal proof.
static iree_status_t iree_hal_amd_gpu_allocation_capture_frontier(
    iree_hal_amd_gpu_allocation_t* operation, iree_async_axis_t axis,
    uint32_t queue_count, const uint64_t* epochs) {
  if (!axis) {
    return iree_ok_status();
  }
  uint32_t count = 0;
  for (uint32_t i = 0; i < queue_count; ++i) {
    count += epochs[i] != 0;
  }
  if (!count || count > UINT8_MAX) {
    return iree_ok_status();
  }
  const iree_host_size_t size = sizeof(iree_async_frontier_t) +
                                count * sizeof(iree_async_frontier_entry_t);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&operation->arena, size,
                                           (void**)&operation->requester));
  iree_async_frontier_initialize(operation->requester, 0);
  const iree_async_axis_t base_axis = axis & ~(UINT64_C(0xFF) << 24);
  for (uint32_t i = 0; i < queue_count; ++i) {
    if (epochs[i]) {
      operation->requester->entries[operation->requester->entry_count++] =
          (iree_async_frontier_entry_t){base_axis | ((uint64_t)i << 24),
                                        epochs[i]};
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_allocation_prepare(
    iree_hal_amd_gpu_allocation_t* operation, iree_async_axis_t axis,
    uint32_t queue_count, const uint64_t* frontier,
    void (*ready)(void* user_data, iree_status_t status), void* user_data,
    bool* out_pending) {
  *out_pending = false;
  if (operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_PREPARED) {
    return iree_ok_status();
  }
  if (operation->is_deallocation) {
    operation->phase = IREE_HAL_AMD_GPU_ALLOCATION_PREPARED;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_allocation_capture_frontier(
      operation, axis, queue_count, frontier));
  operation->preparation.ready = ready;
  operation->preparation.user_data = user_data;
  iree_atomic_store(&operation->preparation.pending, 2,
                    iree_memory_order_relaxed);
  iree_hal_amd_gpu_allocation_acquire(operation);
  if (iree_atomic_fetch_sub(&operation->preparation.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_status_t status = operation->preparation.status;
    operation->preparation.status = iree_ok_status();
    return status;
  }
  *out_pending = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_allocation_allocate(
    iree_hal_amd_gpu_allocation_context_t* context, bool is_deallocation,
    iree_host_size_t count, iree_hal_amd_gpu_allocation_t** out_operation) {
  *out_operation = NULL;
  iree_arena_allocator_t arena;
  iree_arena_initialize(&context->blocks, &arena);
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*operation), (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  *operation = (iree_hal_amd_gpu_allocation_t){
      .context = context,
      .arena = arena,
      .is_deallocation = is_deallocation,
      .count = count,
  };
  if (is_deallocation) {
    operation->release.marked_count = 0;
  }
  const iree_allocator_t allocator = iree_arena_allocator(&operation->arena);
  status = iree_allocator_malloc_array_uninitialized(
      allocator, count, sizeof(*operation->buffers),
      (void**)&operation->buffers);
  if (iree_status_is_ok(status)) {
    memset(operation->buffers, 0, count * sizeof(*operation->buffers));
    status = iree_allocator_malloc_array_uninitialized(
        allocator, count, sizeof(*operation->reservations),
        (void**)&operation->reservations);
  }
  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    iree_hal_amd_gpu_allocation_destroy(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_allocation_capture_alloca(
    iree_hal_amd_gpu_allocation_context_t* context, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers,
    iree_hal_amd_gpu_allocation_t** out_operation) {
  *out_operation = NULL;
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
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_allocation_allocate(context, false, count, &operation));
  operation->pool = pool;
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
    for (iree_host_size_t i = 0; i < count; ++i) {
      out_buffers[i] = operation->buffers[i];
      iree_hal_buffer_retain(out_buffers[i]);
    }
    *out_operation = operation;
  } else {
    iree_hal_amd_gpu_allocation_destroy(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_allocation_capture_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers,
    iree_hal_amd_gpu_allocation_t** out_operation) {
  *out_operation = NULL;
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_allocation_allocate(context, true, count, &operation));
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
        operation->pool = source_pool;
      } else if (source_pool != operation->pool) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "GPU deallocation spans multiple source pools");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    iree_hal_amd_gpu_allocation_destroy(operation);
  }
  return status;
}
