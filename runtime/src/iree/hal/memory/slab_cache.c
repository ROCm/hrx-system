// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_cache.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/tracing.h"

// Cache entries are whole backing ranges, so one exact representable frontier
// is sufficient. Allocate its storage with the entry, never on release.
IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_slab_cache_frontier_t, UINT8_MAX);

typedef struct iree_hal_slab_cache_t iree_hal_slab_cache_t;

typedef struct iree_hal_slab_cache_entry_t {
  // Intrusive idle or pending-return link, protected by the cache mutex.
  struct iree_hal_slab_cache_entry_t* next;
  // Borrowed cache owning the explicit reservation epoch.
  iree_hal_slab_cache_t* cache;
  // Exact reservation held in the backing pool until eviction.
  iree_hal_pool_reservation_t reservation;
  // Full prepared parent view; no alignment margins are discarded.
  iree_hal_pool_buffer_range_t range;
  // Exact prerequisite, immutable while checked out.
  iree_hal_slab_cache_frontier_t frontier;
  // Most recent return time used by the reuse interval diagnostic.
  iree_time_t return_time;
  // Whether the entry belongs to the configured reusable class.
  bool cacheable;
  // Set only after an owning materialization transaction commits.
  bool ownership_transferred;
} iree_hal_slab_cache_entry_t;

typedef struct iree_hal_slab_cache_t {
  // Ordinary reservation/materialization interface.
  iree_hal_pool_t base;
  // Retained source of storage and captured progress services.
  iree_hal_pool_t* backing_pool;
  // Allocator for cold entry and transaction metadata.
  iree_allocator_t host_allocator;
  // Immutable class acquired for compatible requests.
  iree_hal_pool_reservation_request_t slab;
  // Captured source capabilities for validation and bypass routing.
  iree_hal_pool_capabilities_t capabilities;
  // Stable logical allocation trace identity.
  iree_hal_memory_trace_t trace;
  // Protects only lists, retention policy and maintenance admission.
  iree_slim_mutex_t mutex;
  // Entries available for ordered reuse.
  iree_hal_slab_cache_entry_t* idle_head;
  // Detached entries waiting for cold return to the backing pool.
  iree_hal_slab_cache_entry_t* return_head;
  // Number of idle entries, including pending prerequisites.
  uint32_t idle_count;
  // Backing still retained by live/private/idle entries, excluding queued
  // returns.
  iree_device_size_t retained_bytes;
  // Requested number of immediately ready idle entries.
  uint32_t target_count;
  // Maximum number of retained idle entries.
  uint32_t max_count;
  // Invalidates preparation admitted before explicit trimming.
  uint64_t generation;
  // Whether current policy requests proactive preparation.
  bool refill_requested;
  // Whether the embedded maintenance entry is queued or executing.
  bool maintenance_pending;
  // Ordered second sweep catches returns already queued by child allocators.
  struct {
    // Dedicated entry ordered after child maintenance, even during refill.
    iree_hal_memory_maintenance_entry_t entry;
    // Whether the second sweep is queued or executing.
    bool pending;
    // Trim generation observed when this entry was last enqueued.
    uint64_t generation;
    // Strongest trim requested while the entry is pending.
    iree_hal_pool_trim_flags_t flags;
    // Smallest retained-byte floor requested by the pending trims.
    iree_device_size_t min_bytes_to_keep;
  } trim;
  // Set during caller-quiescent destruction.
  bool shutting_down;
  // First unconsumed asynchronous refill failure, owned under the mutex.
  iree_status_t pending_error;
  // One reusable task on the shared placement-local executor.
  iree_hal_memory_maintenance_entry_t maintenance_entry;
  // Joins this cache's maintenance tail during final destruction.
  iree_notification_t maintenance_notification;
  // Common accounting and cache diagnostics updated without walking parents.
  struct {
    // Bytes retained by private, live and idle entries.
    iree_atomic_int64_t committed;
    // Parent-visible bytes checked out to clients.
    iree_atomic_int64_t reserved;
    // Parent-visible idle bytes eligible for ordered reuse.
    iree_atomic_int64_t free;
    // Entries retained by private, live and idle ownership.
    iree_atomic_int32_t slabs;
    // Live client reservation tokens.
    iree_atomic_int32_t reservations;
    // Successful reservation count.
    iree_atomic_int64_t acquisitions;
    // Returned reservation count.
    iree_atomic_int64_t releases;
    // Successful idle-cache hits.
    iree_atomic_int64_t hits;
    // Compatible requests requiring cold preparation.
    iree_atomic_int64_t misses;
    // Requests outside the reusable class.
    iree_atomic_int64_t bypasses;
    // Reservation attempts deferred without growing.
    iree_atomic_int64_t exhausted;
    // Successful fresh reservations.
    iree_atomic_int64_t fresh;
    // Successful reservations whose prerequisite is already covered.
    iree_atomic_int64_t reused;
    // Parent budget refusals.
    iree_atomic_int64_t over_budget;
    // Successful reservations requiring a dependency.
    iree_atomic_int64_t waits;
    // EMA reuse interval in nanoseconds, updated under the mutex.
    iree_atomic_int64_t reuse_nanoseconds;
    // Cumulative retired-range preparation time in nanoseconds.
    iree_atomic_int64_t prefault_nanoseconds;
  } counters;
} iree_hal_slab_cache_t;

static const iree_hal_pool_vtable_t iree_hal_slab_cache_vtable;

static const iree_async_frontier_t* iree_hal_slab_cache_entry_frontier(
    const iree_hal_slab_cache_entry_t* entry) {
  return entry->frontier.entry_count
             ? iree_async_fixed_frontier_as_const_frontier(&entry->frontier)
             : NULL;
}

static bool iree_hal_slab_cache_frontier_is_satisfied(
    const iree_hal_slab_cache_t* cache, const iree_async_frontier_t* requester,
    const iree_async_frontier_t* frontier) {
  if (!frontier) {
    return true;
  }
  if (requester) {
    const iree_async_frontier_comparison_t comparison =
        iree_async_frontier_compare(requester, frontier);
    if (comparison == IREE_ASYNC_FRONTIER_AFTER ||
        comparison == IREE_ASYNC_FRONTIER_EQUAL) {
      return true;
    }
  }
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t entry = frontier->entries[i];
    if (cache->base.epoch_query.fn) {
      if (!cache->base.epoch_query.fn(cache->base.epoch_query.user_data,
                                      entry.axis, entry.epoch)) {
        return false;
      }
    } else if (!iree_async_frontier_tracker_query_epoch(
                   cache->base.frontier_tracker, entry.axis, entry.epoch)) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_slab_cache_request_is_cacheable(
    const iree_hal_slab_cache_t* cache,
    const iree_hal_pool_reservation_request_t* request) {
  return request->allocation_size <= cache->slab.allocation_size &&
         request->params.min_alignment <= cache->slab.params.min_alignment;
}

static iree_status_t iree_hal_slab_cache_validate_request(
    const iree_hal_slab_cache_t* cache,
    const iree_hal_pool_reservation_request_t* request) {
  if (!request->allocation_size ||
      !iree_device_size_is_valid_alignment(request->params.min_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache request requires a nonzero size and "
                            "power-of-two alignment");
  }
  if ((cache->capabilities.max_allocation_size &&
       request->allocation_size > cache->capabilities.max_allocation_size) ||
      request->params.min_alignment >
          cache->capabilities.max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "cache request exceeds backing pool geometry");
  }
  iree_hal_buffer_params_t params = request->params;
  if (cache->base.memory_contract) {
    params = cache->base.memory_contract->buffer_params;
  } else {
    iree_hal_buffer_params_canonicalize(&params);
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      cache->capabilities.memory_type,
      params.type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      cache->capabilities.allowed_access, params.access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      cache->capabilities.supported_usage, params.usage));
  if (!iree_hal_queue_family_affinity_is_any(params.queue_family_affinity) &&
      !iree_all_bits_set(cache->capabilities.queue_family_affinity,
                         params.queue_family_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache request exceeds backing queue families");
  }
  return iree_ok_status();
}

static void iree_hal_slab_cache_destroy_entry(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t* entry) {
  iree_hal_pool_release_reservations(cache->backing_pool, 1,
                                     &entry->reservation,
                                     iree_hal_slab_cache_entry_frontier(entry));
  iree_hal_pool_buffer_range_deinitialize(&entry->range);
  iree_allocator_free(cache->host_allocator, entry);
}

// Cold preparation always runs on the captured owner. The requested epoch is
// held privately until transaction commit or exact-history rollback.
typedef struct iree_hal_slab_cache_prepare_t {
  // Cache borrowing all other fields for this joined call.
  iree_hal_slab_cache_t* cache;
  // Parent geometry and permissions, normalized to the class for cache hits.
  iree_hal_pool_reservation_request_t request;
  // Requester's causal position, borrowed until preparation returns.
  const iree_async_frontier_t* requester;
  // Whether the caller can accept a pending reuse dependency.
  iree_hal_pool_reserve_flags_t flags;
  // Prepared entry, or NULL on failure/transient exhaustion.
  iree_hal_slab_cache_entry_t* entry;
  // Whether the cold acquisition found capacity returned before it ran.
  bool from_idle;
  // Exact parent acquisition outcome.
  iree_hal_pool_acquire_result_t result;
  // Owned terminal infrastructure failure.
  iree_status_t status;
} iree_hal_slab_cache_prepare_t;

static void iree_hal_slab_cache_prepare(void* user_data) {
  iree_hal_slab_cache_prepare_t* call = user_data;
  iree_hal_slab_cache_t* cache = call->cache;
  iree_hal_slab_cache_entry_t* entry = NULL;
  call->status = iree_allocator_malloc(cache->host_allocator, sizeof(*entry),
                                       (void**)&entry);
  if (!iree_status_is_ok(call->status)) {
    return;
  }
  entry->cache = cache;
  entry->cacheable =
      iree_hal_slab_cache_request_is_cacheable(cache, &call->request);
  iree_hal_pool_acquire_info_t info = {0};
  call->status = iree_hal_pool_acquire_reservations(
      cache->backing_pool, 1, &call->request, call->requester, call->flags,
      &entry->reservation, &info, &call->result);
  const bool acquired = iree_status_is_ok(call->status) &&
                        (call->result == IREE_HAL_POOL_ACQUIRE_OK ||
                         call->result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
                         call->result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  if (acquired) {
    if (info.reuse_frontier) {
      memcpy(&entry->frontier, info.reuse_frontier,
             sizeof(*info.reuse_frontier) +
                 info.reuse_frontier->entry_count *
                     sizeof(info.reuse_frontier->entries[0]));
    }
    call->status = iree_hal_pool_materialize_reservations(
        cache->backing_pool, 1, &call->request, &entry->reservation,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &entry->range.buffer);
    if (iree_status_is_ok(call->status)) {
      entry->range.length = iree_hal_buffer_byte_length(entry->range.buffer);
      entry->range.memory = iree_hal_buffer_memory_view(entry->range.buffer);
      entry->range.memory.reuse_frontier =
          iree_hal_slab_cache_entry_frontier(entry);
      if (!entry->range.memory.backing ||
          !entry->range.memory.backing->advice ||
          !entry->range.memory.backing->advice->prefault) {
        call->status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "backing pool has no prepared range advice");
      }
    }
    bool retired = !info.reuse_frontier;
    if (iree_status_is_ok(call->status) && info.reuse_frontier) {
      call->status = iree_async_frontier_tracker_query(
          cache->base.frontier_tracker, info.reuse_frontier, &retired);
    }
    if (iree_status_is_ok(call->status) && retired) {
      const iree_hal_buffer_range_advice_t* advice =
          entry->range.memory.backing->advice;
      const iree_time_t begin = iree_time_now();
      advice->prefault(advice->user_data, entry->range.memory.offset,
                       entry->range.length);
      iree_atomic_fetch_add(&cache->counters.prefault_nanoseconds,
                            iree_time_now() - begin, iree_memory_order_relaxed);
    }
  }
  if (acquired && iree_status_is_ok(call->status)) {
    iree_atomic_fetch_add(&cache->counters.committed,
                          (int64_t)entry->reservation.byte_length,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_add(&cache->counters.slabs, 1, iree_memory_order_relaxed);
    iree_slim_mutex_lock(&cache->mutex);
    cache->retained_bytes += entry->reservation.byte_length;
    iree_slim_mutex_unlock(&cache->mutex);
    call->entry = entry;
  } else {
    if (acquired) {
      iree_hal_pool_release_reservations(
          cache->backing_pool, 1, &entry->reservation, info.reuse_frontier);
      iree_hal_pool_buffer_range_deinitialize(&entry->range);
    }
    iree_allocator_free(cache->host_allocator, entry);
  }
}

// Reclamation relinquishes retained capacity immediately. Parent return and
// physical retirement remain cold work and do not inflate this policy's floor.
static void iree_hal_slab_cache_drop_retention(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t* entry) {
  cache->retained_bytes -= entry->reservation.byte_length;
  iree_atomic_fetch_sub(&cache->counters.committed,
                        (int64_t)entry->reservation.byte_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_sub(&cache->counters.slabs, 1, iree_memory_order_relaxed);
}

// These list helpers only move already-prepared entries under the mutex.
static void iree_hal_slab_cache_insert_idle(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t* entry) {
  entry->next = cache->idle_head;
  cache->idle_head = entry;
  ++cache->idle_count;
  iree_atomic_fetch_add(&cache->counters.free,
                        (int64_t)entry->reservation.byte_length,
                        iree_memory_order_relaxed);
}

static void iree_hal_slab_cache_remove_idle(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t** link) {
  iree_hal_slab_cache_entry_t* entry = *link;
  *link = entry->next;
  entry->next = NULL;
  --cache->idle_count;
  iree_atomic_fetch_sub(&cache->counters.free,
                        (int64_t)entry->reservation.byte_length,
                        iree_memory_order_relaxed);
}

static uint32_t iree_hal_slab_cache_ready_count(iree_hal_slab_cache_t* cache) {
  uint32_t count = 0;
  for (iree_hal_slab_cache_entry_t* entry = cache->idle_head; entry;
       entry = entry->next) {
    count += iree_hal_slab_cache_frontier_is_satisfied(
        cache, NULL, iree_hal_slab_cache_entry_frontier(entry));
  }
  return count;
}

// Claims the embedded work entry under the metadata mutex. Enqueue happens
// after unlocking; destruction observes this claim and joins publication.
static bool iree_hal_slab_cache_schedule(iree_hal_slab_cache_t* cache) {
  if (cache->maintenance_pending) {
    return false;
  }
  if (!cache->return_head && !cache->refill_requested) {
    return false;
  }
  cache->maintenance_pending = true;
  return true;
}

static void iree_hal_slab_cache_trim_idle(
    iree_hal_slab_cache_t* cache, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  const uint32_t target = iree_any_bit_set(flags, IREE_HAL_POOL_TRIM_FLAG_ALL)
                              ? 0
                              : cache->target_count;
  iree_device_size_t committed = cache->retained_bytes;
  iree_hal_slab_cache_entry_t** link = &cache->idle_head;
  while (*link && cache->idle_count > target) {
    iree_hal_slab_cache_entry_t* entry = *link;
    const iree_device_size_t length = entry->reservation.byte_length;
    if (committed >= min_bytes_to_keep &&
        length <= committed - min_bytes_to_keep) {
      iree_hal_slab_cache_remove_idle(cache, link);
      iree_hal_slab_cache_drop_retention(cache, entry);
      entry->next = cache->return_head;
      cache->return_head = entry;
      committed -= length;
    } else {
      link = &entry->next;
    }
  }
}

// An entry queued before a later trim may precede that trim's child returns.
// Requeue once at the current generation to put the sweep after those returns.
static void iree_hal_slab_cache_trim_on_owner(
    iree_hal_memory_maintenance_entry_t* work) {
  iree_hal_slab_cache_t* cache =
      (iree_hal_slab_cache_t*)((uint8_t*)work -
                               offsetof(iree_hal_slab_cache_t, trim.entry));
  iree_slim_mutex_lock(&cache->mutex);
  if (cache->trim.generation != cache->generation) {
    cache->trim.generation = cache->generation;
    iree_slim_mutex_unlock(&cache->mutex);
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance, work);
    return;
  }
  cache->refill_requested = false;
  iree_hal_slab_cache_trim_idle(cache, cache->trim.flags,
                                cache->trim.min_bytes_to_keep);
  const bool schedule = iree_hal_slab_cache_schedule(cache);
  cache->trim.pending = false;
  iree_notification_post(&cache->maintenance_notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&cache->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->maintenance_entry);
  }
}

static void iree_hal_slab_cache_maintain(
    iree_hal_memory_maintenance_entry_t* work) {
  iree_hal_slab_cache_t* cache =
      (iree_hal_slab_cache_t*)((uint8_t*)work - offsetof(iree_hal_slab_cache_t,
                                                         maintenance_entry));
  while (true) {
    iree_slim_mutex_lock(&cache->mutex);
    iree_hal_slab_cache_entry_t* entry = cache->return_head;
    const uint64_t generation = cache->generation;
    bool refill = false;
    if (entry) {
      cache->return_head = entry->next;
    } else if (cache->refill_requested && !cache->shutting_down &&
               iree_status_is_ok(cache->pending_error) &&
               cache->idle_count < cache->max_count &&
               iree_hal_slab_cache_ready_count(cache) < cache->target_count) {
      refill = true;
    } else {
      cache->refill_requested = false;
      cache->maintenance_pending = false;
      iree_notification_post(&cache->maintenance_notification,
                             IREE_ALL_WAITERS);
      iree_slim_mutex_unlock(&cache->mutex);
      return;
    }
    iree_slim_mutex_unlock(&cache->mutex);
    if (entry) {
      iree_hal_slab_cache_destroy_entry(cache, entry);
      continue;
    }
    if (refill) {
      iree_hal_slab_cache_prepare_t call = {
          .cache = cache,
          .request = cache->slab,
          .flags = IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      };
      iree_hal_slab_cache_prepare(&call);
      bool published = false;
      iree_slim_mutex_lock(&cache->mutex);
      if (generation == cache->generation && !cache->shutting_down &&
          call.entry && cache->idle_count < cache->max_count &&
          iree_hal_slab_cache_ready_count(cache) < cache->target_count) {
        call.entry->return_time = iree_time_now();
        iree_hal_slab_cache_insert_idle(cache, call.entry);
        call.entry = NULL;
        published = true;
      }
      if (!iree_status_is_ok(call.status)) {
        if (iree_status_is_ok(cache->pending_error)) {
          cache->pending_error = call.status;
          call.status = iree_ok_status();
        }
        published = true;
      }
      if (!call.entry && !published && generation == cache->generation) {
        // Transient parent exhaustion is not a progress event. A later caller
        // retry, target update or release can request another refill.
        cache->refill_requested = false;
      }
      if (call.entry) {
        iree_hal_slab_cache_drop_retention(cache, call.entry);
      }
      iree_slim_mutex_unlock(&cache->mutex);
      // The first pending diagnostic owns the failure; any later duplicate is
      // disposed outside the metadata lock.
      iree_status_free(call.status);
      if (call.entry) {
        iree_hal_slab_cache_destroy_entry(cache, call.entry);
      }
      if (published) {
        iree_async_notification_signal_if_observed(cache->base.notification,
                                                   INT32_MAX);
      }
    }
  }
}

static bool iree_hal_slab_cache_maintenance_is_complete(void* user_data) {
  iree_hal_slab_cache_t* cache = user_data;
  iree_slim_mutex_lock(&cache->mutex);
  const bool complete = !cache->maintenance_pending && !cache->trim.pending;
  iree_slim_mutex_unlock(&cache->mutex);
  return complete;
}

static void iree_hal_slab_cache_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  IREE_ASSERT(iree_atomic_load(&cache->counters.reservations,
                               iree_memory_order_relaxed) == 0);
  iree_slim_mutex_lock(&cache->mutex);
  cache->shutting_down = true;
  cache->refill_requested = false;
  while (cache->idle_head) {
    iree_hal_slab_cache_entry_t* entry = cache->idle_head;
    iree_hal_slab_cache_remove_idle(cache, &cache->idle_head);
    iree_hal_slab_cache_drop_retention(cache, entry);
    entry->next = cache->return_head;
    cache->return_head = entry;
  }
  const bool schedule = iree_hal_slab_cache_schedule(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->maintenance_entry);
  }
  iree_notification_await(&cache->maintenance_notification,
                          iree_hal_slab_cache_maintenance_is_complete, cache,
                          iree_infinite_timeout());
  // Final destruction consumes an undelivered diagnostic; no caller remains.
  iree_status_free(cache->pending_error);
  iree_hal_pool_release(cache->backing_pool);
  iree_hal_memory_trace_deinitialize(&cache->trace);
  iree_hal_pool_deinitialize(base_pool);
  iree_notification_deinitialize(&cache->maintenance_notification);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
}

void iree_hal_slab_cache_options_initialize(
    iree_hal_slab_cache_options_t* options) {
  memset(options, 0, sizeof(*options));
  options->max_count = 4;
}

iree_status_t iree_hal_slab_cache_create(
    iree_hal_pool_t* backing_pool, const iree_hal_slab_cache_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  if (!backing_pool->maintenance) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "backing pool has no captured memory owner");
  }
  if (options->target_count > options->max_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache target_count exceeds max_count");
  }
  iree_hal_slab_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  iree_async_notification_t* notification = NULL;
  iree_status_t status = iree_async_notification_create(
      backing_pool->notification->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
      &notification);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_slab_cache_vtable, backing_pool->memory_contract,
        notification, backing_pool->wait_sources,
        backing_pool->frontier_tracker, host_allocator, &cache->base);
  }
  iree_async_notification_release(notification);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, cache);
    return status;
  }
  cache->base.maintenance = backing_pool->maintenance;
  cache->base.epoch_query = backing_pool->epoch_query;
  cache->backing_pool = backing_pool;
  iree_hal_pool_retain(backing_pool);
  cache->host_allocator = host_allocator;
  cache->base.asan_enabled = iree_hal_pool_requires_asan_advice(backing_pool);
  cache->slab = options->slab;
  cache->slab.params.min_alignment =
      iree_max(1, cache->slab.params.min_alignment);
  iree_hal_buffer_params_canonicalize(&cache->slab.params);
  cache->target_count = options->target_count;
  cache->max_count = options->max_count;
  cache->maintenance_entry.fn = iree_hal_slab_cache_maintain;
  cache->trim.entry.fn = iree_hal_slab_cache_trim_on_owner;
  iree_slim_mutex_initialize(&cache->mutex);
  iree_notification_initialize(&cache->maintenance_notification);
  iree_hal_pool_query_capabilities(backing_pool, &cache->capabilities);
  status = iree_hal_slab_cache_validate_request(cache, &cache->slab);
  if (iree_status_is_ok(status)) {
    // Cached views must serve every request accepted by the cache. The class
    // inherits the source's complete access scope; per-request views narrow it.
    cache->slab.params.type = cache->capabilities.memory_type;
    cache->slab.params.access = cache->capabilities.allowed_access;
    cache->slab.params.usage = cache->capabilities.supported_usage;
    cache->slab.params.queue_family_affinity =
        cache->capabilities.queue_family_affinity;
    status = iree_hal_memory_trace_initialize_pool(
        options->trace_name, "iree-hal-slab-cache", host_allocator,
        &cache->trace);
  }
  if (iree_status_is_ok(status)) {
    cache->refill_requested = cache->target_count != 0;
    if (iree_hal_slab_cache_schedule(cache)) {
      iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                          &cache->maintenance_entry);
    }
    *out_pool = &cache->base;
  } else {
    iree_hal_slab_cache_destroy(&cache->base);
  }
  return status;
}

iree_status_t iree_hal_slab_cache_set_target(iree_hal_pool_t* base_pool,
                                             uint32_t target_count) {
  if (base_pool->resource.vtable != &iree_hal_slab_cache_vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool is not a slab cache");
  }
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  if (target_count > cache->max_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache target_count exceeds max_count");
  }
  iree_slim_mutex_lock(&cache->mutex);
  cache->target_count = target_count;
  cache->refill_requested = target_count != 0;
  const bool schedule = iree_hal_slab_cache_schedule(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->maintenance_entry);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Reservation transactions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_slab_cache_acquire_element_t {
  // Tentative entry selected while holding the metadata lock.
  iree_hal_slab_cache_entry_t* entry;
  // Private cold preparation, held only by this transaction between retries.
  iree_hal_slab_cache_entry_t* prepared;
  // Whether private preparation took an already cached entry.
  bool prepared_from_idle;
  // Exact result staged until the whole transaction commits.
  iree_hal_pool_acquire_info_t info;
} iree_hal_slab_cache_acquire_element_t;

// Prefers ready capacity to a pending entry. No native or parent query occurs
// during selection; completion probes use the captured local progress facts.
static iree_hal_slab_cache_entry_t* iree_hal_slab_cache_take_entry(
    iree_hal_slab_cache_t* cache, const iree_async_frontier_t* requester,
    iree_hal_pool_reserve_flags_t flags) {
  iree_hal_slab_cache_entry_t** selected = NULL;
  for (iree_hal_slab_cache_entry_t** link = &cache->idle_head; *link;
       link = &(*link)->next) {
    if (iree_hal_slab_cache_frontier_is_satisfied(
            cache, requester, iree_hal_slab_cache_entry_frontier(*link))) {
      selected = link;
      break;
    }
    if (!selected &&
        iree_any_bit_set(flags,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER)) {
      selected = link;
    }
  }
  if (!selected) {
    return NULL;
  }
  iree_hal_slab_cache_entry_t* entry = *selected;
  iree_hal_slab_cache_remove_idle(cache, selected);
  return entry;
}

// A queued child return may have populated the cache while this miss waited
// for the captured owner. Recheck there before acquiring native backing.
static void iree_hal_slab_cache_prepare_acquisition(void* user_data) {
  iree_hal_slab_cache_prepare_t* call = user_data;
  iree_hal_slab_cache_t* cache = call->cache;
  const bool cacheable =
      iree_hal_slab_cache_request_is_cacheable(cache, &call->request);
  if (cacheable) {
    iree_slim_mutex_lock(&cache->mutex);
    call->entry =
        iree_hal_slab_cache_take_entry(cache, call->requester, call->flags);
    iree_slim_mutex_unlock(&cache->mutex);
  }
  if (call->entry) {
    call->from_idle = true;
  } else {
    iree_atomic_fetch_add(
        cacheable ? &cache->counters.misses : &cache->counters.bypasses, 1,
        iree_memory_order_relaxed);
    iree_hal_slab_cache_prepare(call);
  }
}

static iree_hal_pool_acquire_info_t iree_hal_slab_cache_acquire_info(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t* entry,
    const iree_async_frontier_t* requester) {
  const iree_async_frontier_t* frontier =
      iree_hal_slab_cache_entry_frontier(entry);
  return (iree_hal_pool_acquire_info_t){
      .reuse_frontier = frontier,
      .result = !frontier ? IREE_HAL_POOL_ACQUIRE_OK_FRESH
                : iree_hal_slab_cache_frontier_is_satisfied(cache, requester,
                                                            frontier)
                    ? IREE_HAL_POOL_ACQUIRE_OK
                    : IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT,
  };
}

static iree_status_t iree_hal_slab_cache_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester, iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_slab_cache_validate_request(cache, &requests[i]));
  }
  iree_slim_mutex_lock(&cache->mutex);
  iree_status_t status = cache->pending_error;
  cache->pending_error = iree_ok_status();
  iree_slim_mutex_unlock(&cache->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_slab_cache_acquire_element_t inline_elements[8];
  const bool growth_allowed =
      !iree_any_bit_set(flags, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH);
  if (request_count > IREE_ARRAYSIZE(inline_elements) && !growth_allowed) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = (iree_hal_pool_acquire_info_t){
          .result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED,
          .flags = IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED,
      };
    }
    *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    iree_atomic_fetch_add(&cache->counters.exhausted, 1,
                          iree_memory_order_relaxed);
    return iree_ok_status();
  }
  iree_hal_slab_cache_acquire_element_t* elements = inline_elements;
  if (request_count > IREE_ARRAYSIZE(inline_elements)) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(cache->host_allocator, request_count,
                                    sizeof(*elements), (void**)&elements));
  }
  memset(elements, 0, request_count * sizeof(*elements));
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  bool committed = false;
  bool retry = true;
  while (iree_status_is_ok(status) && retry) {
    iree_host_size_t selected = 0;
    result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    iree_slim_mutex_lock(&cache->mutex);
    for (; selected < request_count; ++selected) {
      iree_hal_slab_cache_acquire_element_t* element = &elements[selected];
      element->entry = element->prepared;
      if (!element->entry && iree_hal_slab_cache_request_is_cacheable(
                                 cache, &requests[selected])) {
        element->entry =
            iree_hal_slab_cache_take_entry(cache, requester, flags);
      }
      if (!element->entry) {
        break;
      }
      element->info =
          iree_hal_slab_cache_acquire_info(cache, element->entry, requester);
      if (element->info.result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
        result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
      } else if (element->info.result == IREE_HAL_POOL_ACQUIRE_OK &&
                 result == IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
        result = IREE_HAL_POOL_ACQUIRE_OK;
      }
    }
    committed = selected == request_count;
    if (committed) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        iree_hal_slab_cache_entry_t* entry = elements[i].entry;
        entry->ownership_transferred = false;
        if (!elements[i].prepared || elements[i].prepared_from_idle) {
          iree_atomic_fetch_add(&cache->counters.hits, 1,
                                iree_memory_order_relaxed);
          const int64_t interval =
              iree_max(0, iree_time_now() - entry->return_time);
          const int64_t previous = iree_atomic_load(
              &cache->counters.reuse_nanoseconds, iree_memory_order_relaxed);
          iree_atomic_store(&cache->counters.reuse_nanoseconds,
                            previous - previous / 8 + interval / 8,
                            iree_memory_order_relaxed);
        }
        iree_atomic_fetch_add(&cache->counters.reserved,
                              (int64_t)entry->reservation.byte_length,
                              iree_memory_order_relaxed);
        iree_atomic_int64_t* counter = &cache->counters.fresh;
        if (elements[i].info.result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
          counter = &cache->counters.waits;
        } else if (elements[i].info.result == IREE_HAL_POOL_ACQUIRE_OK) {
          counter = &cache->counters.reused;
        }
        iree_atomic_fetch_add(counter, 1, iree_memory_order_relaxed);
      }
      iree_atomic_fetch_add(&cache->counters.reservations,
                            (int32_t)request_count, iree_memory_order_relaxed);
      iree_atomic_fetch_add(&cache->counters.acquisitions,
                            (int64_t)request_count, iree_memory_order_relaxed);
      cache->refill_requested = cache->target_count != 0;
    } else {
      // Restore the tentative prefix before cold work. Entries obtained on
      // the owner remain transaction-owned across retries, including returns
      // found by its final cache check. No rollback-only wake is published.
      for (iree_host_size_t i = 0; i < selected; ++i) {
        if (!elements[i].prepared) {
          iree_hal_slab_cache_insert_idle(cache, elements[i].entry);
        }
        elements[i].entry = NULL;
        memset(&elements[i].info, 0, sizeof(elements[i].info));
      }
    }
    const bool schedule = iree_hal_slab_cache_schedule(cache);
    iree_slim_mutex_unlock(&cache->mutex);
    if (schedule) {
      iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                          &cache->maintenance_entry);
    }
    retry = !committed && growth_allowed;
    if (!committed && !growth_allowed) {
      result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      elements[selected].info = (iree_hal_pool_acquire_info_t){
          .result = result,
          .flags = IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED,
      };
    } else if (retry) {
      const bool cacheable =
          iree_hal_slab_cache_request_is_cacheable(cache, &requests[selected]);
      iree_hal_slab_cache_prepare_t call = {
          .cache = cache,
          .request = cacheable ? cache->slab : requests[selected],
          .requester = requester,
          .flags = flags,
      };
      iree_hal_memory_maintenance_call(cache->base.maintenance,
                                       iree_hal_slab_cache_prepare_acquisition,
                                       &call);
      status = call.status;
      elements[selected].prepared = call.entry;
      elements[selected].prepared_from_idle = call.from_idle;
      if (iree_status_is_ok(status) && !call.entry) {
        result = call.result;
        elements[selected].info.result = result;
        retry = false;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    if (result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED) {
      iree_atomic_fetch_add(&cache->counters.exhausted, 1,
                            iree_memory_order_relaxed);
    }
    if (result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
      iree_atomic_fetch_add(&cache->counters.over_budget, 1,
                            iree_memory_order_relaxed);
    }
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = elements[i].info;
      if (committed) {
        iree_hal_slab_cache_entry_t* entry = elements[i].entry;
        out_reservations[i] = (iree_hal_pool_reservation_t){
            .byte_length = entry->range.length,
            .block_handle = (uint64_t)(uintptr_t)entry,
        };
        iree_hal_memory_trace_alloc(&cache->trace, entry, entry->range.length);
      }
    }
    *out_result = result;
  }
  if (!committed) {
    bool restored_capacity = false;
    iree_slim_mutex_lock(&cache->mutex);
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_slab_cache_entry_t* entry = elements[i].prepared;
      if (entry) {
        if (elements[i].prepared_from_idle &&
            cache->idle_count < cache->max_count) {
          iree_hal_slab_cache_insert_idle(cache, entry);
          restored_capacity = true;
        } else {
          iree_hal_slab_cache_drop_retention(cache, entry);
          entry->next = cache->return_head;
          cache->return_head = entry;
        }
      }
    }
    const bool schedule = iree_hal_slab_cache_schedule(cache);
    iree_slim_mutex_unlock(&cache->mutex);
    if (schedule) {
      iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                          &cache->maintenance_entry);
    }
    if (restored_capacity) {
      iree_async_notification_signal_if_observed(cache->base.notification,
                                                 INT32_MAX);
    }
  }
  if (elements != inline_elements) {
    iree_allocator_free(cache->host_allocator, elements);
  }
  return status;
}

static void iree_hal_slab_cache_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  // Copy and trace while the caller still owns each token. Publication below
  // only swaps prepared links and updates counters under the metadata lock.
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_slab_cache_entry_t* entry =
        (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[i].block_handle;
    if (death_frontier) {
      if (death_frontier != iree_hal_slab_cache_entry_frontier(entry)) {
        memcpy(
            &entry->frontier, death_frontier,
            sizeof(*death_frontier) + death_frontier->entry_count *
                                          sizeof(death_frontier->entries[0]));
      }
    } else {
      entry->frontier.entry_count = 0;
    }
    entry->return_time = iree_time_now();
    iree_hal_memory_trace_free(&cache->trace, entry);
  }
  iree_slim_mutex_lock(&cache->mutex);
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_slab_cache_entry_t* entry =
        (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[i].block_handle;
    iree_atomic_fetch_sub(&cache->counters.reserved,
                          (int64_t)entry->reservation.byte_length,
                          iree_memory_order_relaxed);
    if (entry->cacheable && cache->idle_count < cache->max_count) {
      iree_hal_slab_cache_insert_idle(cache, entry);
    } else {
      iree_hal_slab_cache_drop_retention(cache, entry);
      entry->next = cache->return_head;
      cache->return_head = entry;
    }
  }
  iree_atomic_fetch_sub(&cache->counters.reservations, (int32_t)count,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&cache->counters.releases, (int64_t)count,
                        iree_memory_order_relaxed);
  cache->refill_requested = cache->target_count != 0;
  const bool schedule = iree_hal_slab_cache_schedule(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->maintenance_entry);
  }
  iree_async_notification_signal_if_observed(base_pool->notification,
                                             INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Materialization, retention and diagnostics
//===----------------------------------------------------------------------===//

static void iree_hal_slab_cache_advise_asan_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_slab_cache_entry_t* entry =
        (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[i].block_handle;
    iree_hal_pool_advise_asan_reservations(cache->backing_pool, 1,
                                           &entry->reservation, flags);
  }
}

static void iree_hal_slab_cache_buffer_release(void* user_data,
                                               iree_hal_buffer_t* buffer) {
  iree_hal_slab_cache_entry_t* entry = user_data;
  if (!entry->ownership_transferred) {
    return;
  }
  const iree_hal_pool_reservation_t reservation = {
      .byte_length = entry->range.length,
      .block_handle = (uint64_t)(uintptr_t)entry,
  };
  iree_hal_pool_advise_asan_reservations(
      &entry->cache->base, 1, &reservation,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  iree_hal_slab_cache_release_reservations(&entry->cache->base, 1, &reservation,
                                           NULL);
}

static void iree_hal_slab_cache_query_reservation_views(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* out_views) {
  (void)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_slab_cache_entry_t* entry =
        (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[i].block_handle;
    iree_hal_pool_buffer_range_query_reservation_view(
        &entry->range, 0, reservations[i].byte_length,
        iree_hal_slab_cache_entry_frontier(entry), &out_views[i]);
  }
}

static iree_status_t iree_hal_slab_cache_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!reservations[i].block_handle ||
        requests[i].allocation_size > reservations[i].byte_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid cache materialization range");
    }
  }
  iree_hal_buffer_t* inline_buffers[8];
  iree_hal_buffer_t** buffers = inline_buffers;
  if (count > IREE_ARRAYSIZE(inline_buffers)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        cache->host_allocator, count, sizeof(*buffers), (void**)&buffers));
  }
  const bool transfer = iree_any_bit_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_status_t status = iree_ok_status();
  iree_host_size_t materialized = 0;
  while (materialized < count && iree_status_is_ok(status)) {
    iree_hal_slab_cache_entry_t* entry =
        (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[materialized]
            .block_handle;
    const iree_hal_buffer_release_callback_t callback = {
        .fn = transfer ? iree_hal_slab_cache_buffer_release : NULL,
        .user_data = entry,
    };
    status = iree_hal_pool_buffer_range_materialize(
        &entry->range, 0, reservations[materialized].byte_length,
        requests[materialized].params,
        iree_hal_slab_cache_entry_frontier(entry), callback,
        cache->host_allocator, &buffers[materialized]);
    if (iree_status_is_ok(status)) {
      ++materialized;
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_slab_cache_entry_t* entry =
          (iree_hal_slab_cache_entry_t*)(uintptr_t)reservations[i].block_handle;
      if (transfer) {
        entry->ownership_transferred = true;
      }
      out_buffers[i] = buffers[i];
    }
  } else {
    for (iree_host_size_t i = 0; i < materialized; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  if (buffers != inline_buffers) {
    iree_allocator_free(cache->host_allocator, buffers);
  }
  return status;
}

static void iree_hal_slab_cache_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  *out_capabilities = ((const iree_hal_slab_cache_t*)base_pool)->capabilities;
}

static iree_status_t iree_hal_slab_cache_validate_asan(
    const iree_hal_pool_t* base_pool,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_slab_cache_t* cache = (const iree_hal_slab_cache_t*)base_pool;
  return iree_hal_pool_validate_asan_options(cache->backing_pool, options);
}

static void iree_hal_slab_cache_query_pool_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_slab_cache_t* cache = (const iree_hal_slab_cache_t*)base_pool;
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &cache->counters.committed, iree_memory_order_relaxed);
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &cache->counters.reserved, iree_memory_order_relaxed);
  out_stats->bytes_free = (iree_device_size_t)iree_atomic_load(
      &cache->counters.free, iree_memory_order_relaxed);
  out_stats->slab_count = (uint32_t)iree_atomic_load(&cache->counters.slabs,
                                                     iree_memory_order_relaxed);
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &cache->counters.reservations, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &cache->counters.acquisitions, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &cache->counters.releases, iree_memory_order_relaxed);
  out_stats->exhausted_count = (uint64_t)iree_atomic_load(
      &cache->counters.exhausted, iree_memory_order_relaxed);
  out_stats->fresh_count = (uint64_t)iree_atomic_load(
      &cache->counters.fresh, iree_memory_order_relaxed);
  out_stats->reuse_count = (uint64_t)iree_atomic_load(
      &cache->counters.reused, iree_memory_order_relaxed);
  out_stats->over_budget_count = (uint64_t)iree_atomic_load(
      &cache->counters.over_budget, iree_memory_order_relaxed);
  out_stats->wait_count = (uint64_t)iree_atomic_load(&cache->counters.waits,
                                                     iree_memory_order_relaxed);
}

iree_status_t iree_hal_slab_cache_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_slab_cache_stats_t* out_stats) {
  if (base_pool->resource.vtable != &iree_hal_slab_cache_vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool is not a slab cache");
  }
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  iree_slim_mutex_lock(&cache->mutex);
  out_stats->ready_count = iree_hal_slab_cache_ready_count(cache);
  out_stats->pending_count = cache->idle_count - out_stats->ready_count;
  iree_slim_mutex_unlock(&cache->mutex);
  out_stats->hit_count = (uint64_t)iree_atomic_load(&cache->counters.hits,
                                                    iree_memory_order_relaxed);
  out_stats->miss_count = (uint64_t)iree_atomic_load(&cache->counters.misses,
                                                     iree_memory_order_relaxed);
  out_stats->bypass_count = (uint64_t)iree_atomic_load(
      &cache->counters.bypasses, iree_memory_order_relaxed);
  out_stats->ema_reuse_interval_nanoseconds = (uint64_t)iree_atomic_load(
      &cache->counters.reuse_nanoseconds, iree_memory_order_relaxed);
  out_stats->prefault_time_nanoseconds = (uint64_t)iree_atomic_load(
      &cache->counters.prefault_nanoseconds, iree_memory_order_relaxed);
  return iree_ok_status();
}

static void iree_hal_slab_cache_trim(iree_hal_pool_t* base_pool,
                                     iree_hal_pool_trim_flags_t flags,
                                     iree_device_size_t min_bytes_to_keep) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  iree_slim_mutex_lock(&cache->mutex);
  ++cache->generation;
  cache->refill_requested = false;
  iree_hal_slab_cache_trim_idle(cache, flags, min_bytes_to_keep);
  const bool schedule = iree_hal_slab_cache_schedule(cache);
  const bool schedule_trim = !cache->trim.pending;
  if (schedule_trim) {
    cache->trim.pending = true;
    cache->trim.generation = cache->generation;
    cache->trim.flags = flags;
    cache->trim.min_bytes_to_keep = min_bytes_to_keep;
  } else {
    cache->trim.flags |= flags;
    cache->trim.min_bytes_to_keep =
        iree_min(cache->trim.min_bytes_to_keep, min_bytes_to_keep);
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->maintenance_entry);
  }
  if (schedule_trim) {
    iree_hal_memory_maintenance_enqueue(cache->base.maintenance,
                                        &cache->trim.entry);
  }
}

static const iree_hal_pool_vtable_t iree_hal_slab_cache_vtable = {
    .destroy = iree_hal_slab_cache_destroy,
    .acquire_reservations = iree_hal_slab_cache_acquire_reservations,
    .release_reservations = iree_hal_slab_cache_release_reservations,
    .query_reservation_views = iree_hal_slab_cache_query_reservation_views,
    .materialize_reservations = iree_hal_slab_cache_materialize_reservations,
    .query_capabilities = iree_hal_slab_cache_query_capabilities,
    .validate_asan = iree_hal_slab_cache_validate_asan,
    .query_stats = iree_hal_slab_cache_query_pool_stats,
    .trim = iree_hal_slab_cache_trim,
    .advise_asan_reservations = iree_hal_slab_cache_advise_asan_reservations,
};
