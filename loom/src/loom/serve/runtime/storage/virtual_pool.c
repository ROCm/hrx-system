// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/virtual_pool.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/buffer_range.h"

IREE_ASYNC_FIXED_FRONTIER_TYPE(loom_serve_virtual_pool_frontier_t, UINT8_MAX);

typedef enum loom_serve_virtual_pool_entry_state_e {
  // Idle addresses with no physical backing.
  VIRTUAL_POOL_ENTRY_EMPTY,
  // Idle mapped range, eligible for ordered logical reuse.
  VIRTUAL_POOL_ENTRY_READY,
  // Exclusively owned by an acquisition or its published reservation.
  VIRTUAL_POOL_ENTRY_ACQUIRED,
  // Excluded from reuse while joining history or detaching physical storage.
  VIRTUAL_POOL_ENTRY_TRIMMING,
  // Failed native edit; only checked outer teardown can return its backing.
  VIRTUAL_POOL_ENTRY_FAILED,
} loom_serve_virtual_pool_entry_state_t;

typedef struct loom_serve_virtual_pool_entry_t {
  // Borrowed pool, outliving reservations and materialized views.
  loom_serve_virtual_pool_t* pool;
  // Stable virtual storage owned independently of physical commitment.
  loom_serve_virtual_buffer_t* storage;
  // Retained prepared range supplying all child views.
  iree_hal_pool_buffer_range_t range;
  // Exact last-use prerequisite, immutable while acquired or trimming.
  loom_serve_virtual_pool_frontier_t frontier;
  // Protected by pool mutex; the owner alone changes physical backing.
  loom_serve_virtual_pool_entry_state_t state;
  // Materialization transferred the explicit reservation into an owning view.
  bool ownership_transferred;
  // Owner-only accepted physical return, or NULL before/after detachment.
  loom_serve_memory_operation_t* operation;
  // Registered observation of actual completion, not requester dominance.
  iree_async_frontier_waiter_t waiter;
  // Submission and callback join before trim may consume history_status.
  iree_atomic_int32_t history_pending;
  // Owned completion result published before the last history claim.
  iree_status_t history_status;
  // All retained addresses, including empty and failed ranges.
  struct loom_serve_virtual_pool_entry_t* next;
} loom_serve_virtual_pool_entry_t;

typedef struct loom_serve_virtual_pool_element_t {
  // Privately acquired range, published only when the entire batch succeeds.
  loom_serve_virtual_pool_entry_t* entry;
  // This range was mapped before the transaction; rollback preserves it.
  bool ready;
} loom_serve_virtual_pool_element_t;

typedef struct loom_serve_virtual_pool_acquire_t {
  // Accepted request arrays borrowed through callback completion.
  const iree_hal_pool_reservation_request_t* requests;
  // Borrowed requester history, used only for logical reuse classification.
  const iree_async_frontier_t* requester;
  // Caller result slots, untouched on failed acquisition.
  iree_hal_pool_reservation_t* reservations;
  // Caller reuse-result slots, published with reservations.
  iree_hal_pool_acquire_info_t* infos;
  // Exactly one completion for this accepted acquisition.
  iree_hal_pool_acquire_callback_t completion;
  // Request flags captured at submission.
  iree_hal_pool_reserve_flags_t flags;
  // Total request/element count.
  iree_host_size_t count;
  // Prefix privately acquired before owner publication.
  iree_host_size_t acquired_count;
  // Accepted union commitment, owned until result publication.
  loom_serve_memory_operation_t* operation;
  // Whether owner-side commitment has been accepted.
  bool started;
  // Owner queue link, protected by pool mutex.
  struct loom_serve_virtual_pool_acquire_t* next;
  // Stable range selections trailing this transaction.
  loom_serve_virtual_pool_element_t elements[];
} loom_serve_virtual_pool_acquire_t;

typedef struct loom_serve_virtual_pool_source_t {
  // Refcounted HAL interface borrowed by logical allocators and queue work.
  iree_hal_pool_t base;
  // Unique physical owner; outer teardown joins every interface borrower.
  loom_serve_virtual_pool_t* owner;
} loom_serve_virtual_pool_source_t;

struct loom_serve_virtual_pool_t {
  // Owned HAL interface in trailing storage, distinct from owner lifetime.
  loom_serve_virtual_pool_source_t* source;
  // Allocator for cold metadata; warm reuse uses caller/stack storage.
  iree_allocator_t host_allocator;
  // Borrowed exclusive physical-accounting owner.
  loom_serve_memory_pool_t* memory_pool;
  // Qualified immutable native slab geometry.
  loom_serve_memory_geometry_t geometry;
  // Prepared virtual-target capabilities, independent of physical availability.
  iree_hal_pool_capabilities_t capabilities;
  // Owner-only physical usage for these ranges, within memory_pool's totals.
  loom_serve_memory_statistics_t memory;
  // Protects logical ownership and callback access, never native operations.
  iree_slim_mutex_t mutex;
  // Notification for blocking outer callers driving the same owner.
  iree_notification_t notification;
  // Borrowed asynchronous scheduler wake, protected by mutex.
  loom_serve_wake_callback_t observer;
  // Stable range metadata; only owner progress adds/removes entries.
  loom_serve_virtual_pool_entry_t* entries;
  // FIFO of cold acquisitions captured by queue callbacks.
  loom_serve_virtual_pool_acquire_t* first;
  // Tail insertion link, avoiding a scan on off-owner cold submission.
  loom_serve_virtual_pool_acquire_t** tail;
  // Idle physical bytes to retain after the current trim request.
  iree_device_size_t trim_floor;
  // Requested trim remains active until eligible returns have completed.
  bool trim_requested;
  // A newer off-owner trim cannot be cleared by an older owner pass.
  uint64_t trim_generation;
  // Owner-side trim failure, retained through outer cleanup.
  iree_status_t failure;
  // Logical statistics under mutex; physical snapshots publish on the owner.
  iree_hal_pool_stats_t statistics;
};

static const iree_async_frontier_t* loom_serve_virtual_pool_frontier(
    const loom_serve_virtual_pool_entry_t* entry) {
  return entry->frontier.entry_count
             ? iree_async_fixed_frontier_as_const_frontier(&entry->frontier)
             : NULL;
}

// Called with the metadata mutex held. The callback only schedules its owner.
static void loom_serve_virtual_pool_wake_locked(
    loom_serve_virtual_pool_t* pool) {
  iree_notification_post(&pool->notification, IREE_ALL_WAITERS);
  if (pool->observer.wake) {
    pool->observer.wake(pool->observer.user_data);
  }
}

static void loom_serve_virtual_pool_wake(void* user_data) {
  loom_serve_virtual_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->mutex);
  loom_serve_virtual_pool_wake_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
}

void loom_serve_virtual_pool_set_observer(loom_serve_virtual_pool_t* pool,
                                          loom_serve_wake_callback_t observer) {
  iree_slim_mutex_lock(&pool->mutex);
  pool->observer = observer;
  iree_slim_mutex_unlock(&pool->mutex);
}

static bool loom_serve_virtual_pool_frontier_is_satisfied(
    const loom_serve_virtual_pool_t* pool,
    const iree_async_frontier_t* requester,
    const iree_async_frontier_t* frontier) {
  if (!frontier) {
    return true;
  }
  iree_host_size_t requester_index = 0;
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t entry = frontier->entries[i];
    while (requester && requester_index < requester->entry_count &&
           requester->entries[requester_index].axis < entry.axis) {
      ++requester_index;
    }
    if (requester && requester_index < requester->entry_count &&
        requester->entries[requester_index].axis == entry.axis &&
        requester->entries[requester_index].epoch >= entry.epoch) {
      continue;
    }
    if (!iree_async_frontier_tracker_query_epoch(
            pool->source->base.frontier_tracker, entry.axis, entry.epoch)) {
      return false;
    }
  }
  return true;
}

// Selection only observes local completion facts. It never calls a native
// allocator, changes physical accounting, or invokes reclamation.
static loom_serve_virtual_pool_entry_t* loom_serve_virtual_pool_take(
    loom_serve_virtual_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_async_frontier_t* requester, iree_hal_pool_reserve_flags_t flags,
    loom_serve_virtual_pool_entry_state_t state) {
  loom_serve_virtual_pool_entry_t* selected = NULL;
  bool selected_ready = false;
  for (loom_serve_virtual_pool_entry_t* entry = pool->entries; entry;
       entry = entry->next) {
    if (entry->state != state ||
        entry->range.length < request->allocation_size) {
      continue;
    }
    const bool ready = loom_serve_virtual_pool_frontier_is_satisfied(
        pool, requester, loom_serve_virtual_pool_frontier(entry));
    if (!ready && !iree_any_bit_set(
                      flags, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER)) {
      continue;
    }
    if (!selected || (ready && !selected_ready) ||
        (ready == selected_ready &&
         entry->range.length < selected->range.length)) {
      selected = entry;
      selected_ready = ready;
    }
  }
  if (selected) {
    selected->state = VIRTUAL_POOL_ENTRY_ACQUIRED;
  }
  return selected;
}

static iree_hal_pool_acquire_result_t loom_serve_virtual_pool_publish(
    loom_serve_virtual_pool_t* pool, iree_host_size_t count,
    const loom_serve_virtual_pool_element_t* elements,
    const iree_async_frontier_t* requester,
    iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_acquire_info_t* infos) {
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_virtual_pool_entry_t* entry = elements[i].entry;
    const iree_async_frontier_t* frontier =
        loom_serve_virtual_pool_frontier(entry);
    const iree_hal_pool_acquire_result_t entry_result =
        !frontier ? IREE_HAL_POOL_ACQUIRE_OK_FRESH
        : loom_serve_virtual_pool_frontier_is_satisfied(pool, requester,
                                                        frontier)
            ? IREE_HAL_POOL_ACQUIRE_OK
            : IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
    reservations[i] = (iree_hal_pool_reservation_t){
        .byte_length = entry->range.length,
        .block_handle = (uint64_t)(uintptr_t)entry,
    };
    infos[i] = (iree_hal_pool_acquire_info_t){.reuse_frontier = frontier,
                                              .result = entry_result};
    if (entry_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      result = entry_result;
      ++pool->statistics.wait_count;
    } else if (entry_result == IREE_HAL_POOL_ACQUIRE_OK) {
      if (result == IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
        result = entry_result;
      }
      ++pool->statistics.reuse_count;
    } else {
      ++pool->statistics.fresh_count;
    }
    pool->statistics.bytes_reserved += entry->range.length;
    ++pool->statistics.reservation_count;
    ++pool->statistics.reserve_count;
  }
  return result;
}

static void loom_serve_virtual_pool_restore(
    iree_host_size_t count, const loom_serve_virtual_pool_element_t* elements,
    bool failed_commit) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    elements[i].entry->state = elements[i].ready ? VIRTUAL_POOL_ENTRY_READY
                               : failed_commit   ? VIRTUAL_POOL_ENTRY_FAILED
                                                 : VIRTUAL_POOL_ENTRY_EMPTY;
  }
}

static iree_status_t loom_serve_virtual_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester,
    iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_acquire_info_t* infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  loom_serve_virtual_pool_t* pool =
      ((loom_serve_virtual_pool_source_t*)base_pool)->owner;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_device_size_t alignment = iree_max(requests[i].min_alignment, 1);
    if (!requests[i].allocation_size ||
        requests[i].allocation_size > INT64_MAX - pool->geometry.slab_size ||
        !iree_device_size_is_valid_alignment(alignment) ||
        alignment > pool->capabilities.max_allocation_alignment) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "virtual backing request exceeds its geometry");
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_serve_virtual_pool_element_t inline_elements[8] = {0};
  loom_serve_virtual_pool_acquire_t* acquire = NULL;
  loom_serve_virtual_pool_element_t* elements = inline_elements;
  if (count > IREE_ARRAYSIZE(inline_elements)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
        pool->host_allocator, sizeof(*acquire), count, sizeof(*elements),
        (void**)&acquire));
    elements = acquire->elements;
  }
  iree_host_size_t acquired_count = 0;
  iree_slim_mutex_lock(&pool->mutex);
  while (acquired_count < count) {
    loom_serve_virtual_pool_entry_t* entry =
        loom_serve_virtual_pool_take(pool, &requests[acquired_count], requester,
                                     flags, VIRTUAL_POOL_ENTRY_READY);
    if (!entry) {
      break;
    }
    elements[acquired_count++] =
        (loom_serve_virtual_pool_element_t){entry, true};
  }
  if (acquired_count == count) {
    const iree_hal_pool_acquire_result_t result =
        loom_serve_virtual_pool_publish(pool, count, elements, requester,
                                        reservations, infos);
    iree_slim_mutex_unlock(&pool->mutex);
    iree_allocator_free(pool->host_allocator, acquire);
    completion.fn(completion.user_data, iree_ok_status(), result);
    return iree_ok_status();
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (!acquire) {
    status = iree_allocator_malloc_struct_array(
        pool->host_allocator, sizeof(*acquire), count, sizeof(*elements),
        (void**)&acquire);
    if (iree_status_is_ok(status)) {
      memcpy(acquire->elements, elements, acquired_count * sizeof(*elements));
    }
  }
  iree_slim_mutex_lock(&pool->mutex);
  if (iree_status_is_ok(status)) {
    acquire->requests = requests;
    acquire->requester = requester;
    acquire->reservations = reservations;
    acquire->infos = infos;
    acquire->completion = completion;
    acquire->flags = flags;
    acquire->count = count;
    acquire->acquired_count = acquired_count;
    *pool->tail = acquire;
    pool->tail = &acquire->next;
    ++pool->statistics.reuse_miss_count;
    loom_serve_virtual_pool_wake_locked(pool);
  } else {
    loom_serve_virtual_pool_restore(acquired_count, elements, false);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

static void loom_serve_virtual_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* frontier) {
  loom_serve_virtual_pool_t* pool =
      ((loom_serve_virtual_pool_source_t*)base_pool)->owner;
  iree_slim_mutex_lock(&pool->mutex);
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_virtual_pool_entry_t* entry =
        (loom_serve_virtual_pool_entry_t*)(uintptr_t)reservations[i]
            .block_handle;
    if (frontier) {
      // Returning an unused token may alias this very storage.
      memmove(&entry->frontier, frontier,
              sizeof(*frontier) +
                  frontier->entry_count * sizeof(iree_async_frontier_entry_t));
    } else {
      entry->frontier.entry_count = 0;
    }
    entry->ownership_transferred = false;
    entry->state = VIRTUAL_POOL_ENTRY_READY;
    pool->statistics.bytes_reserved -= entry->range.length;
    --pool->statistics.reservation_count;
    ++pool->statistics.release_count;
  }
  loom_serve_virtual_pool_wake_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
  iree_async_notification_signal(pool->source->base.notification, 1);
}

static void loom_serve_virtual_pool_query_reservation_views(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* views) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_virtual_pool_entry_t* entry =
        (const loom_serve_virtual_pool_entry_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_pool_buffer_range_query_reservation_view(
        &entry->range, 0, reservations[i].byte_length,
        loom_serve_virtual_pool_frontier(entry), &views[i]);
  }
}

static void loom_serve_virtual_pool_buffer_release(void* user_data,
                                                   iree_hal_buffer_t* buffer) {
  loom_serve_virtual_pool_entry_t* entry = user_data;
  if (!entry->ownership_transferred) {
    return;
  }
  const iree_hal_pool_reservation_t reservation = {
      .byte_length = entry->range.length,
      .block_handle = (uint64_t)(uintptr_t)entry,
  };
  loom_serve_virtual_pool_release_reservations(&entry->pool->source->base, 1,
                                               &reservation, NULL);
}

static iree_status_t loom_serve_virtual_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  loom_serve_virtual_pool_t* pool =
      ((loom_serve_virtual_pool_source_t*)base_pool)->owner;
  iree_hal_buffer_t* inline_buffers[8];
  iree_hal_buffer_t** buffers = inline_buffers;
  if (count > IREE_ARRAYSIZE(inline_buffers)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        pool->host_allocator, count, sizeof(*buffers), (void**)&buffers));
  }
  const bool transfer = iree_any_bit_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_status_t status = iree_ok_status();
  iree_host_size_t materialized = 0;
  while (materialized < count && iree_status_is_ok(status)) {
    loom_serve_virtual_pool_entry_t* entry =
        (loom_serve_virtual_pool_entry_t*)(uintptr_t)reservations[materialized]
            .block_handle;
    const iree_hal_buffer_release_callback_t callback = {
        .fn = transfer ? loom_serve_virtual_pool_buffer_release : NULL,
        .user_data = entry,
    };
    status = iree_hal_pool_buffer_range_materialize(
        &entry->range, 0, reservations[materialized].byte_length,
        loom_serve_virtual_pool_frontier(entry), callback, pool->host_allocator,
        &buffers[materialized]);
    if (iree_status_is_ok(status)) {
      ++materialized;
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_serve_virtual_pool_entry_t* entry =
          (loom_serve_virtual_pool_entry_t*)(uintptr_t)reservations[i]
              .block_handle;
      entry->ownership_transferred = transfer;
      out_buffers[i] = buffers[i];
    }
  } else {
    for (iree_host_size_t i = 0; i < materialized; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  if (buffers != inline_buffers) {
    iree_allocator_free(pool->host_allocator, buffers);
  }
  return status;
}

static void loom_serve_virtual_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  *out_capabilities =
      ((const loom_serve_virtual_pool_source_t*)base_pool)->owner->capabilities;
}

static void loom_serve_virtual_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_statistics) {
  loom_serve_virtual_pool_t* pool =
      ((const loom_serve_virtual_pool_source_t*)base_pool)->owner;
  iree_slim_mutex_lock(&pool->mutex);
  *out_statistics = pool->statistics;
  out_statistics->bytes_free =
      out_statistics->bytes_committed > out_statistics->bytes_reserved
          ? out_statistics->bytes_committed - out_statistics->bytes_reserved
          : 0;
  iree_slim_mutex_unlock(&pool->mutex);
}

static void loom_serve_virtual_pool_trim(iree_hal_pool_t* base_pool,
                                         iree_hal_pool_trim_flags_t flags,
                                         iree_device_size_t min_bytes_to_keep) {
  loom_serve_virtual_pool_t* pool =
      ((loom_serve_virtual_pool_source_t*)base_pool)->owner;
  iree_slim_mutex_lock(&pool->mutex);
  pool->trim_floor = min_bytes_to_keep;
  pool->trim_requested = true;
  ++pool->trim_generation;
  loom_serve_virtual_pool_wake_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
}

static void loom_serve_virtual_pool_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_pool_deinitialize(base_pool);
}

static const iree_hal_pool_vtable_t loom_serve_virtual_pool_vtable = {
    .destroy = loom_serve_virtual_pool_destroy,
    .acquire_reservations = loom_serve_virtual_pool_acquire_reservations,
    .release_reservations = loom_serve_virtual_pool_release_reservations,
    .query_reservation_views = loom_serve_virtual_pool_query_reservation_views,
    .materialize_reservations =
        loom_serve_virtual_pool_materialize_reservations,
    .query_capabilities = loom_serve_virtual_pool_query_capabilities,
    .query_stats = loom_serve_virtual_pool_query_stats,
    .trim = loom_serve_virtual_pool_trim,
};

// Owner-only address creation; publication into the shared list is the only
// mutex-protected part. Rounded physical slabs are reusable across all roles.
static iree_status_t loom_serve_virtual_pool_create_entry(
    loom_serve_virtual_pool_t* pool, iree_device_size_t length,
    loom_serve_virtual_pool_entry_t** out_entry) {
  *out_entry = NULL;
  loom_serve_virtual_pool_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*entry), (void**)&entry));
  entry->pool = pool;
  const iree_device_size_t aligned_length =
      loom_serve_virtual_pool_round_size(pool, length);
  iree_status_t status = loom_serve_virtual_buffer_allocate(
      pool->memory_pool, pool->geometry, aligned_length,
      pool->geometry.granularity, &pool->memory, &entry->storage);
  if (iree_status_is_ok(status)) {
    const iree_hal_asan_pool_options_t asan = {0};
    status = iree_hal_pool_buffer_range_initialize(
        loom_serve_virtual_buffer_handle(entry->storage), 0,
        IREE_HAL_WHOLE_BUFFER, pool->geometry.granularity, &asan,
        &entry->range);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&pool->mutex);
    entry->state = VIRTUAL_POOL_ENTRY_ACQUIRED;
    entry->next = pool->entries;
    pool->entries = entry;
    ++pool->statistics.slab_count;
    iree_slim_mutex_unlock(&pool->mutex);
    *out_entry = entry;
  } else {
    status = iree_status_join(status,
                              loom_serve_virtual_buffer_free(entry->storage));
    iree_allocator_free(pool->host_allocator, entry);
  }
  return status;
}

iree_status_t loom_serve_virtual_pool_allocate(
    loom_serve_memory_pool_t* memory_pool,
    loom_serve_memory_geometry_t geometry, iree_allocator_t host_allocator,
    loom_serve_virtual_pool_t** out_pool) {
  *out_pool = NULL;
  loom_serve_virtual_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*pool) + sizeof(*pool->source), (void**)&pool));
  pool->source = (loom_serve_virtual_pool_source_t*)(pool + 1);
  pool->source->owner = pool;
  pool->host_allocator = host_allocator;
  pool->memory_pool = memory_pool;
  pool->geometry = geometry;
  pool->tail = &pool->first;
  iree_slim_mutex_initialize(&pool->mutex);
  iree_notification_initialize(&pool->notification);
  loom_serve_virtual_pool_entry_t* entry = NULL;
  iree_status_t status =
      loom_serve_virtual_pool_create_entry(pool, geometry.slab_size, &entry);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_buffer_range_t* range = &entry->range;
    pool->capabilities = (iree_hal_pool_capabilities_t){
        .memory_type = iree_hal_buffer_memory_type(range->buffer),
        .allowed_access = iree_hal_buffer_allowed_access(range->buffer),
        .supported_usage = iree_hal_buffer_allowed_usage(range->buffer),
        .queue_family_affinity =
            iree_hal_buffer_allocation_placement(range->buffer)
                .queue_family_affinity,
        .atomic_operations = range->memory.backing->atomic_operations,
        .max_allocation_alignment = geometry.granularity,
        .maintenance_alignment = range->memory.backing->maintenance_alignment,
    };
    entry->state = VIRTUAL_POOL_ENTRY_EMPTY;
    status = iree_hal_pool_initialize(
        &loom_serve_virtual_pool_vtable, range->memory.contract,
        range->memory.backing->notification,
        (iree_hal_pool_wait_source_list_t){0}, range->memory.backing->tracker,
        host_allocator, &pool->source->base);
  }
  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    if (entry) {
      iree_hal_pool_buffer_range_deinitialize(&entry->range);
      status = iree_status_join(status,
                                loom_serve_virtual_buffer_free(entry->storage));
      iree_allocator_free(host_allocator, entry);
    }
    iree_notification_deinitialize(&pool->notification);
    iree_slim_mutex_deinitialize(&pool->mutex);
    iree_allocator_free(host_allocator, pool);
  }
  return status;
}

iree_hal_pool_t* loom_serve_virtual_pool_handle(
    loom_serve_virtual_pool_t* pool) {
  return &pool->source->base;
}

uint64_t loom_serve_virtual_pool_round_size(
    const loom_serve_virtual_pool_t* pool, uint64_t byte_length) {
  const uint64_t padding = pool->geometry.slab_size - 1;
  return byte_length > UINT64_MAX - padding
             ? UINT64_MAX
             : iree_device_align(byte_length, pool->geometry.slab_size);
}

loom_serve_memory_usage_t loom_serve_virtual_pool_memory_usage(
    const loom_serve_virtual_pool_t* pool) {
  return (loom_serve_memory_usage_t){
      .reserved_bytes = pool->memory.reserved_bytes,
      .committed_bytes = pool->memory.committed_bytes,
      .pending_bytes = pool->memory.pending_bytes,
      .releasing_bytes = pool->memory.releasing_bytes,
      .released_bytes = pool->memory.released_bytes,
  };
}

static void loom_serve_virtual_pool_finish_acquire(
    loom_serve_virtual_pool_t* pool, loom_serve_virtual_pool_acquire_t* acquire,
    iree_status_t status, bool admitted) {
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
  iree_slim_mutex_lock(&pool->mutex);
  pool->first = acquire->next;
  if (!pool->first) {
    pool->tail = &pool->first;
  }
  if (iree_status_is_ok(status) && admitted) {
    result = loom_serve_virtual_pool_publish(
        pool, acquire->count, acquire->elements, acquire->requester,
        acquire->reservations, acquire->infos);
  } else {
    loom_serve_virtual_pool_restore(
        acquire->acquired_count, acquire->elements,
        acquire->started && !iree_status_is_ok(status));
    if (iree_status_is_ok(status)) {
      ++pool->statistics.over_budget_count;
      memset(acquire->infos, 0, acquire->count * sizeof(*acquire->infos));
      if (acquire->count) {
        acquire->infos[0].result = result;
      }
    }
  }
  iree_slim_mutex_unlock(&pool->mutex);
  const iree_hal_pool_acquire_callback_t completion = acquire->completion;
  iree_allocator_free(pool->host_allocator, acquire);
  completion.fn(completion.user_data, status, result);
}

// Cold transactions publish as a whole. A returned capacity outcome rolls
// back the selected prefix; no caller receives partially reserved storage.
static void loom_serve_virtual_pool_advance_acquisitions(
    loom_serve_virtual_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  loom_serve_virtual_pool_acquire_t* acquire = pool->first;
  iree_slim_mutex_unlock(&pool->mutex);
  while (acquire) {
    iree_status_t status = iree_ok_status();
    bool admitted = acquire->started;
    loom_serve_memory_admission_t admission =
        LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
    while (!acquire->started && acquire->acquired_count < acquire->count &&
           iree_status_is_ok(status)) {
      const iree_host_size_t index = acquire->acquired_count;
      const iree_hal_pool_reservation_request_t* request =
          &acquire->requests[index];
      iree_slim_mutex_lock(&pool->mutex);
      loom_serve_virtual_pool_entry_t* entry = loom_serve_virtual_pool_take(
          pool, request, acquire->requester, acquire->flags,
          VIRTUAL_POOL_ENTRY_READY);
      const bool ready = entry != NULL;
      if (!entry) {
        entry = loom_serve_virtual_pool_take(pool, request, NULL,
                                             IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                             VIRTUAL_POOL_ENTRY_EMPTY);
      }
      iree_slim_mutex_unlock(&pool->mutex);
      if (!entry) {
        status = loom_serve_virtual_pool_create_entry(
            pool, request->allocation_size, &entry);
      }
      if (iree_status_is_ok(status)) {
        acquire->elements[index] =
            (loom_serve_virtual_pool_element_t){entry, ready};
        ++acquire->acquired_count;
      }
    }
    if (!acquire->started && iree_status_is_ok(status)) {
      loom_serve_memory_range_t inline_ranges[8];
      loom_serve_memory_range_t* ranges = inline_ranges;
      if (acquire->count > IREE_ARRAYSIZE(inline_ranges)) {
        status =
            iree_allocator_malloc_array(pool->host_allocator, acquire->count,
                                        sizeof(*ranges), (void**)&ranges);
      }
      if (iree_status_is_ok(status)) {
        for (iree_host_size_t i = 0; i < acquire->count; ++i) {
          const loom_serve_virtual_pool_entry_t* entry =
              acquire->elements[i].entry;
          ranges[i] = (loom_serve_memory_range_t){entry->storage, 0,
                                                  entry->range.length};
        }
        status = loom_serve_memory_pool_commit_async(
            pool->memory_pool, acquire->count, ranges, NULL, 0, &admission,
            &acquire->operation);
        admitted = admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
        acquire->started = acquire->operation != NULL || admitted;
        loom_serve_memory_operation_set_observer(
            acquire->operation, loom_serve_virtual_pool_wake, pool);
      }
      if (ranges != inline_ranges) {
        iree_allocator_free(pool->host_allocator, ranges);
      }
    }
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_MEMORY_ADMISSION_DEFERRED) {
      // Preserve this accepted queue acquisition while native capacity returns.
      // A later owner turn retries the same selected ranges; completion may
      // already have published its wake, so do not re-query pending state here.
      break;
    }
    if (acquire->operation &&
        !loom_serve_memory_operation_is_complete(acquire->operation)) {
      break;
    }
    status = iree_status_join(
        status, loom_serve_memory_operation_query_status(acquire->operation));
    loom_serve_memory_operation_free(acquire->operation);
    acquire->operation = NULL;
    loom_serve_virtual_pool_finish_acquire(pool, acquire, status, admitted);
    iree_slim_mutex_lock(&pool->mutex);
    acquire = pool->first;
    iree_slim_mutex_unlock(&pool->mutex);
  }
}

static void loom_serve_virtual_pool_history_complete(void* user_data,
                                                     iree_status_t status) {
  loom_serve_virtual_pool_entry_t* entry = user_data;
  loom_serve_virtual_pool_t* pool = entry->pool;
  iree_slim_mutex_lock(&pool->mutex);
  entry->history_status = status;
  iree_atomic_fetch_sub(&entry->history_pending, 1, iree_memory_order_release);
  loom_serve_virtual_pool_wake_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
}

static void loom_serve_virtual_pool_advance_trim(
    loom_serve_virtual_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  const bool requested = pool->trim_requested;
  const uint64_t generation = pool->trim_generation;
  iree_slim_mutex_unlock(&pool->mutex);
  if (!requested) {
    return;
  }
  // Snapshot the immutable chain head; only this serialized owner adds nodes.
  // Logical callbacks only change states under the mutex.
  iree_device_size_t retained =
      pool->memory.committed_bytes - pool->memory.releasing_bytes;
  bool pending = false;
  for (loom_serve_virtual_pool_entry_t* entry = pool->entries; entry;
       entry = entry->next) {
    bool start = false;
    iree_slim_mutex_lock(&pool->mutex);
    if (pool->trim_requested && entry->state == VIRTUAL_POOL_ENTRY_READY &&
        retained > pool->trim_floor) {
      entry->state = VIRTUAL_POOL_ENTRY_TRIMMING;
      retained -= entry->range.length;
      iree_atomic_store(&entry->history_pending, 2, iree_memory_order_relaxed);
      start = true;
    }
    const bool trimming = entry->state == VIRTUAL_POOL_ENTRY_TRIMMING;
    iree_slim_mutex_unlock(&pool->mutex);
    if (!trimming) {
      continue;
    }
    if (start) {
      const iree_async_frontier_t* frontier =
          loom_serve_virtual_pool_frontier(entry);
      iree_status_t status = iree_ok_status();
      if (frontier) {
        status = iree_async_frontier_tracker_wait(
            pool->source->base.frontier_tracker, frontier,
            loom_serve_virtual_pool_history_complete, entry, &entry->waiter);
      } else {
        loom_serve_virtual_pool_history_complete(entry, iree_ok_status());
      }
      if (!iree_status_is_ok(status)) {
        loom_serve_virtual_pool_history_complete(entry, status);
      }
      iree_atomic_fetch_sub(&entry->history_pending, 1,
                            iree_memory_order_acq_rel);
    }
    if (iree_atomic_load(&entry->history_pending, iree_memory_order_acquire)) {
      pending = true;
      continue;
    }
    // Acquire the mutex once after the final history claim to join the
    // callback's final wake/access, not merely its result publication.
    iree_slim_mutex_lock(&pool->mutex);
    iree_status_t status = entry->history_status;
    entry->history_status = iree_ok_status();
    iree_slim_mutex_unlock(&pool->mutex);
    if (!entry->operation && iree_status_is_ok(status)) {
      loom_serve_virtual_buffer_begin_trim(entry->storage);
      status = loom_serve_memory_pool_trim_async(
          pool->memory_pool, 1, &entry->storage, NULL, 0, &entry->operation);
      loom_serve_memory_operation_set_observer(
          entry->operation, loom_serve_virtual_pool_wake, pool);
    }
    if (entry->operation &&
        !loom_serve_memory_operation_is_complete(entry->operation)) {
      pending = true;
      continue;
    }
    status = iree_status_join(
        status, loom_serve_memory_operation_query_status(entry->operation));
    loom_serve_memory_operation_free(entry->operation);
    entry->operation = NULL;
    iree_slim_mutex_lock(&pool->mutex);
    if (iree_status_is_ok(status)) {
      // Successful actual completion and detach make old history irrelevant.
      entry->frontier.entry_count = 0;
      entry->state = VIRTUAL_POOL_ENTRY_EMPTY;
    } else {
      entry->state = VIRTUAL_POOL_ENTRY_FAILED;
      pool->failure = iree_status_join(pool->failure, status);
    }
    iree_slim_mutex_unlock(&pool->mutex);
  }
  iree_slim_mutex_lock(&pool->mutex);
  pool->trim_requested = pending || pool->trim_generation != generation;
  iree_slim_mutex_unlock(&pool->mutex);
}

iree_status_t loom_serve_virtual_pool_advance(loom_serve_virtual_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_virtual_pool_advance_trim(pool);
  loom_serve_virtual_pool_advance_acquisitions(pool);
  iree_slim_mutex_lock(&pool->mutex);
  pool->statistics.bytes_committed = pool->memory.committed_bytes;
  iree_status_t status = iree_status_clone(pool->failure);
  iree_slim_mutex_unlock(&pool->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool loom_serve_virtual_pool_is_pending(loom_serve_virtual_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  const bool pending = pool->first != NULL || pool->trim_requested;
  iree_slim_mutex_unlock(&pool->mutex);
  return pending;
}

iree_status_t loom_serve_virtual_pool_reclaim(loom_serve_virtual_pool_t* pool) {
  loom_serve_virtual_pool_advance_trim(pool);
  return iree_status_clone(pool->failure);
}

bool loom_serve_virtual_pool_is_reclaiming(loom_serve_virtual_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  const bool pending = pool->trim_requested;
  iree_slim_mutex_unlock(&pool->mutex);
  return pending;
}

typedef struct loom_serve_virtual_pool_wait_t {
  // Borrowed physical owner, also providing this wait's notification.
  loom_serve_virtual_pool_t* pool;
  // Owned terminal semaphore result, protected by the pool mutex.
  iree_status_t status;
  // Callback completion and its final access join under the pool mutex.
  bool complete;
} loom_serve_virtual_pool_wait_t;

static void loom_serve_virtual_pool_wait_complete(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_virtual_pool_wait_t* wait = user_data;
  loom_serve_virtual_pool_t* pool = wait->pool;
  iree_slim_mutex_lock(&pool->mutex);
  wait->status = status;
  wait->complete = true;
  loom_serve_virtual_pool_wake_locked(pool);
  iree_slim_mutex_unlock(&pool->mutex);
}

iree_status_t loom_serve_virtual_pool_wait(loom_serve_virtual_pool_t* pool,
                                           iree_hal_semaphore_t* semaphore,
                                           uint64_t value) {
  loom_serve_virtual_pool_wait_t wait = {.pool = pool};
  iree_async_semaphore_timepoint_t timepoint = {
      .callback = loom_serve_virtual_pool_wait_complete, .user_data = &wait};
  IREE_RETURN_IF_ERROR(iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)semaphore, value, &timepoint));
  // The exclusive outer owner waits on both execution and shared capacity
  // notifications. Joining all memory work would stall an independent warm
  // execution behind a cold acquisition whose native return is still pending.
  const loom_serve_wake_callback_t previous =
      loom_serve_memory_pool_set_observer(pool->memory_pool,
                                          loom_serve_virtual_pool_wake, pool);
  iree_status_t status = iree_ok_status();
  bool complete = false;
  bool cancelled = false;
  while (!complete) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&pool->notification);
    if (iree_status_is_ok(status)) {
      status = loom_serve_virtual_pool_advance(pool);
    }
    if (!iree_status_is_ok(status) && !cancelled) {
      cancelled = true;
      complete = iree_async_semaphore_cancel_timepoint(
          (iree_async_semaphore_t*)semaphore, &timepoint);
    }
    iree_slim_mutex_lock(&pool->mutex);
    complete |= wait.complete;
    iree_slim_mutex_unlock(&pool->mutex);
    if (complete) {
      iree_notification_cancel_wait(&pool->notification);
    } else {
      iree_notification_commit_wait(&pool->notification, token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  loom_serve_memory_pool_set_observer(pool->memory_pool, previous.wake,
                                      previous.user_data);
  return iree_status_join(status, wait.status);
}

static void loom_serve_virtual_pool_notify_waiter(void* user_data) {
  iree_notification_post(user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_virtual_pool_await(loom_serve_virtual_pool_t* pool,
                                            iree_notification_t* notification,
                                            iree_condition_fn_t condition,
                                            void* user_data) {
  iree_slim_mutex_lock(&pool->mutex);
  const loom_serve_wake_callback_t previous = pool->observer;
  pool->observer = (loom_serve_wake_callback_t){
      loom_serve_virtual_pool_notify_waiter, notification};
  iree_slim_mutex_unlock(&pool->mutex);
  const loom_serve_wake_callback_t previous_capacity =
      loom_serve_memory_pool_set_observer(pool->memory_pool,
                                          loom_serve_virtual_pool_notify_waiter,
                                          notification);
  iree_status_t status = iree_ok_status();
  bool complete = false;
  while (!complete) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(notification);
    status = iree_status_join(status, loom_serve_virtual_pool_advance(pool));
    complete = condition(user_data);
    if (complete) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  loom_serve_virtual_pool_set_observer(pool, previous);
  loom_serve_memory_pool_set_observer(pool->memory_pool, previous_capacity.wake,
                                      previous_capacity.user_data);
  return status;
}

iree_status_t loom_serve_virtual_pool_free(loom_serve_virtual_pool_t* pool) {
  if (!pool) {
    return iree_ok_status();
  }
  if (iree_atomic_ref_count_load(&pool->source->base.resource.ref_count) != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "virtual backing still has interface borrowers");
  }
  iree_slim_mutex_lock(&pool->mutex);
  const bool acquired =
      pool->first != NULL || pool->statistics.reservation_count != 0;
  iree_slim_mutex_unlock(&pool->mutex);
  if (acquired) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "virtual backing still has acquired reservations");
  }
  iree_status_t status = iree_ok_status();
  while (pool->entries && iree_status_is_ok(status)) {
    loom_serve_virtual_pool_entry_t* entry = pool->entries;
    iree_slim_mutex_lock(&pool->mutex);
    const bool pending = iree_atomic_load(&entry->history_pending,
                                          iree_memory_order_acquire) != 0;
    iree_slim_mutex_unlock(&pool->mutex);
    if (pending) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "virtual backing still has live consumers");
    } else {
      // Joining accepted native operations here is final outer teardown only.
      loom_serve_memory_operation_await(entry->operation);
      loom_serve_memory_operation_free(entry->operation);
      entry->operation = NULL;
      status = loom_serve_virtual_buffer_free(entry->storage);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_pool_buffer_range_deinitialize(&entry->range);
      pool->entries = entry->next;
      iree_status_ignore(entry->history_status);
      iree_allocator_free(pool->host_allocator, entry);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_pool_release(&pool->source->base);
    iree_status_ignore(pool->failure);
    iree_notification_deinitialize(&pool->notification);
    iree_slim_mutex_deinitialize(&pool->mutex);
    iree_allocator_free(pool->host_allocator, pool);
  }
  return status;
}
