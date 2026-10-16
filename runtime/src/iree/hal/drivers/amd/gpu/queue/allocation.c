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

// Service-owned transaction. Callbacks hand ownership back to service; the
// capture never occupies native publication or the native completion worker.
typedef struct iree_hal_amd_gpu_allocation_t {
  // One-shot initial admission or resumed pool/frontier work.
  iree_hal_memory_maintenance_entry_t entry;
  // Borrowed embedded state dominated by the operation's device reference.
  iree_hal_amd_gpu_allocation_context_t* context;
  // Retained parent, released after every capture and root access finishes.
  iree_hal_device_t* device;
  // Owns this object and every variable-length capture allocation.
  iree_arena_allocator_t arena;
  // Copied and retained input dependencies, joined before native pool work.
  iree_hal_semaphore_list_t waits;
  // Copied and retained terminal signals.
  iree_hal_semaphore_list_t signals;
  // Exclusive service phase after initial semaphore admission.
  iree_hal_amd_gpu_allocation_phase_t phase;
  // Terminal callback result consumed by the next service turn.
  iree_status_t status;
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
      // Active borrowed frontier wait, joined before the next service turn.
      iree_async_frontier_waiter_t frontier_wait;
      // Lazily allocated capacity observation, reused across attempts.
      iree_hal_pool_wait_t* capacity_wait;
    } acquire;
    // Captured logical deallocation epochs.
    struct {
      // Borrowed source per root, allowing one transaction across pools.
      iree_hal_pool_t** pools;
      // Prefix of exclusive deallocation marks to abort on failure.
      iree_host_size_t marked_count;
    } release;
  };
} iree_hal_amd_gpu_allocation_t;

void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, const iree_hal_amd_gpu_context_t* context,
    iree_hal_memory_maintenance_t* service, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context) {
  *out_context = (iree_hal_amd_gpu_allocation_context_t){
      .device = device, .context = context, .service = service};
  iree_arena_block_pool_initialize(1024, host_allocator, &out_context->blocks);
  iree_async_semaphore_wait_context_initialize(&out_context->waits);
}

void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context) {
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
  iree_hal_device_t* device = operation->device;
  iree_arena_allocator_t arena = operation->arena;
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_arena_deinitialize(&arena);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_allocation_complete(
    iree_hal_amd_gpu_allocation_t* operation, iree_status_t status) {
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
    status = iree_hal_semaphore_list_signal(operation->signals, NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(operation->context->context,
                                           operation->device, status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_allocation_free(operation);
}

// Both pool and frontier callbacks finish all access at this queue handoff.
static void iree_hal_amd_gpu_allocation_resume(void* user_data,
                                               iree_status_t status) {
  iree_hal_amd_gpu_allocation_t* operation = user_data;
  operation->status = status;
  iree_hal_memory_maintenance_enqueue(operation->context->service,
                                      &operation->entry);
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
    iree_hal_amd_fabric_transient_buffer_stage_backing(
        operation->buffers[i], &operation->acquire.views[i]);
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
  for (iree_host_size_t i = 0; i < operation->count;) {
    iree_hal_pool_t* pool = operation->release.pools[i];
    iree_host_size_t end = i + 1;
    while (end < operation->count && operation->release.pools[end] == pool) {
      ++end;
    }
    iree_hal_pool_advise_asan_reservations(
        pool, end - i, operation->reservations + i,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    iree_hal_pool_release_reservations(pool, end - i,
                                       operation->reservations + i, NULL);
    i = end;
  }
}

static void iree_hal_amd_gpu_allocation_progress(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_allocation_t* operation = (void*)entry;
  iree_status_t status = operation->status;
  operation->status = iree_ok_status();
  if (iree_status_is_ok(status) &&
      operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_RELEASE) {
    iree_hal_amd_gpu_allocation_release(operation);
  }
  while (iree_status_is_ok(status) &&
         operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_ACQUIRE) {
    iree_hal_pool_wait_t* wait = operation->acquire.capacity_wait;
    if (wait) {
      iree_hal_pool_wait_prepare(wait);
    }
    iree_hal_pool_acquire_result_t result;
    status = iree_hal_pool_acquire_reservations(
        operation->acquire.pool, operation->count, operation->acquire.requests,
        NULL, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
        operation->reservations, operation->acquire.infos, &result);
    if (!iree_status_is_ok(status)) {
      if (wait) {
        iree_hal_pool_wait_abort(wait);
      }
      break;
    }
    if (result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
        result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
      if (wait) {
        iree_hal_pool_wait_commit(wait, iree_infinite_timeout(),
                                  (iree_hal_pool_wait_callback_t){
                                      .fn = iree_hal_amd_gpu_allocation_resume,
                                      .user_data = operation});
        return;
      }
      // Allocate observations only after a real miss, then retry with tokens
      // captured before the attempt so a concurrent release cannot be lost.
      status = iree_hal_pool_wait_allocate(
          operation->acquire.pool, iree_arena_allocator(&operation->arena),
          &operation->acquire.capacity_wait);
      continue;
    }
    if (wait) {
      iree_hal_pool_wait_abort(wait);
    }
    operation->acquire.held = true;
    operation->phase = IREE_HAL_AMD_GPU_ALLOCATION_MATERIALIZE;
  }
  if (iree_status_is_ok(status) &&
      operation->phase == IREE_HAL_AMD_GPU_ALLOCATION_MATERIALIZE) {
    // Each borrowed reservation frontier is already normalized by its pool.
    // Visit it directly instead of merging the batch into a bounded union.
    while (operation->acquire.frontier_index < operation->count &&
           iree_status_is_ok(status)) {
      const iree_hal_pool_acquire_info_t* info =
          &operation->acquire.infos[operation->acquire.frontier_index++];
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
  }
  iree_hal_amd_gpu_allocation_complete(operation, status);
}

static void iree_hal_amd_gpu_allocation_ready(void* user_data,
                                              iree_status_t status) {
  iree_hal_amd_gpu_allocation_t* operation = user_data;
  operation->status = status;
  iree_hal_amd_gpu_allocation_progress(&operation->entry);
}

static void iree_hal_amd_gpu_allocation_begin(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_allocation_t* operation = (void*)entry;
  operation->entry.fn = iree_hal_amd_gpu_allocation_progress;
  if (!operation->waits.count) {
    iree_hal_amd_gpu_allocation_ready(operation, iree_ok_status());
    return;
  }
  iree_status_t status = iree_hal_amd_gpu_host_wait_begin(
      &operation->context->waits, operation->context->service, operation->waits,
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
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*operation), (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  *operation = (iree_hal_amd_gpu_allocation_t){
      .entry = {.fn = iree_hal_amd_gpu_allocation_begin},
      .context = context,
      .device = context->device,
      .arena = arena,
      .phase = phase,
      .count = count,
  };
  iree_hal_device_retain(operation->device);
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
    iree_hal_amd_gpu_allocation_context_t* context,
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
    // Enqueue cannot fail. Publish caller references before service may retire
    // the captured list, so there is no post-submission access to its storage.
    for (iree_host_size_t i = 0; i < count; ++i) {
      out_buffers[i] = operation->buffers[i];
      iree_hal_buffer_retain(out_buffers[i]);
    }
    iree_hal_memory_maintenance_enqueue(context->service, &operation->entry);
  } else {
    iree_hal_amd_gpu_allocation_retire(operation);
    iree_hal_amd_gpu_allocation_free(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_queue_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t count, iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_gpu_allocation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_allocation_allocate(
      context, waits, signals, IREE_HAL_AMD_GPU_ALLOCATION_RELEASE, count,
      &operation));
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      iree_arena_allocator(&operation->arena), count,
      sizeof(*operation->release.pools), (void**)&operation->release.pools);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_hal_buffer_allocation_begin_dealloca(
        buffers[i], &operation->release.pools[i]);
    if (iree_status_is_ok(status)) {
      operation->buffers[i] = buffers[i];
      iree_hal_buffer_retain(buffers[i]);
      ++operation->release.marked_count;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_memory_maintenance_enqueue(context->service, &operation->entry);
  } else {
    iree_hal_amd_gpu_allocation_retire(operation);
    iree_hal_amd_gpu_allocation_free(operation);
  }
  return status;
}
