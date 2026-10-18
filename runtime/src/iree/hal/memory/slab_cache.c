// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_cache.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/tracing.h"

// Cache entries are whole backing ranges, so one exact representable frontier
// is sufficient. Allocate its storage with the entry, never on release.
IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_slab_cache_frontier_t, UINT8_MAX);

typedef struct iree_hal_slab_cache_t iree_hal_slab_cache_t;

typedef struct iree_hal_slab_cache_entry_t {
  // Intrusive idle link or exclusively owned detached-return link.
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

// Stable parent acquisition storage. The submitter and callback each own one
// join reference, so inline completion cannot recurse through the next miss.
typedef struct iree_hal_slab_cache_parent_t {
  // Parent geometry retained through submission and completion.
  iree_hal_pool_reservation_request_t request;
  // Acquired token consumed by entry preparation or returned on failure.
  iree_hal_pool_reservation_t reservation;
  // Exact history borrowed from the acquired parent token.
  iree_hal_pool_acquire_info_t info;
  // Parent capacity outcome, published before the completion handoff.
  iree_hal_pool_acquire_result_t result;
  // Owned infrastructure failure, consumed by the continuation.
  iree_status_t status;
  // Submission and callback must finish before the continuation resumes.
  iree_atomic_int32_t pending;
} iree_hal_slab_cache_parent_t;

typedef struct iree_hal_slab_cache_t {
  // Ordinary reservation/materialization interface.
  iree_hal_pool_t base;
  // Retained source of storage.
  iree_hal_pool_t* backing_pool;
  // Allocator for cold entry and transaction metadata.
  iree_allocator_t host_allocator;
  // Immutable class acquired for compatible requests.
  iree_hal_pool_reservation_request_t slab;
  // Captured source capabilities for validation and bypass routing.
  iree_hal_pool_capabilities_t capabilities;
  // Stable logical allocation trace identity.
  iree_hal_memory_trace_t trace;
  // Protects only lists, retention policy, and refill admission.
  iree_slim_mutex_t mutex;
  // Entries available for ordered reuse.
  iree_hal_slab_cache_entry_t* idle_head;
  // Number of idle entries, including pending prerequisites.
  uint32_t idle_count;
  // Backing still retained by live/private/idle entries, excluding returns.
  iree_device_size_t retained_bytes;
  // Requested number of immediately ready idle entries.
  uint32_t target_count;
  // Maximum number of retained idle entries.
  uint32_t max_count;
  // Invalidates preparation admitted before explicit trimming.
  uint64_t generation;
  // Whether current policy requests proactive preparation.
  bool refill_requested;
  // One in-flight refill, owning a cache reference until terminal completion.
  bool refill_pending;
  // Embedded cold refill state, exclusively used by the active continuation.
  struct {
    // Parent request and result kept stable through its completion.
    iree_hal_slab_cache_parent_t parent;
    // Trim generation captured at admission.
    uint64_t generation;
    // Whether a completed parent result must be consumed on resumption.
    bool has_result;
  } refill;
  // First unconsumed asynchronous refill failure, owned under the mutex.
  iree_status_t pending_error;
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
    // Cache selection only reads the captured completion table. It cannot
    // invoke a backend query while holding the cache metadata lock.
    if (!cache->base.frontier_tracker ||
        !iree_async_frontier_tracker_query_epoch(cache->base.frontier_tracker,
                                                 entry.axis, entry.epoch)) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_slab_cache_request_is_cacheable(
    const iree_hal_slab_cache_t* cache,
    const iree_hal_pool_reservation_request_t* request) {
  return request->allocation_size <= cache->slab.allocation_size &&
         request->min_alignment <= cache->slab.min_alignment;
}

static iree_status_t iree_hal_slab_cache_validate_request(
    const iree_hal_slab_cache_t* cache,
    const iree_hal_pool_reservation_request_t* request) {
  if (!request->allocation_size ||
      !iree_device_size_is_valid_alignment(request->min_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache request requires a nonzero size and "
                            "power-of-two alignment");
  }
  if ((cache->capabilities.max_allocation_size &&
       request->allocation_size > cache->capabilities.max_allocation_size) ||
      request->min_alignment > cache->capabilities.max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "cache request exceeds backing pool geometry");
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

// Detached entries belong to this caller. Propagate exact histories through
// parent pools without holding a child lock or waiting for native allocation.
static void iree_hal_slab_cache_return_entries(
    iree_hal_slab_cache_t* cache, iree_hal_slab_cache_entry_t* entries) {
  while (entries) {
    iree_hal_slab_cache_entry_t* entry = entries;
    entries = entry->next;
    iree_hal_slab_cache_destroy_entry(cache, entry);
  }
}

// Consumes a completed parent reservation. Only metadata and a view are
// prepared here; touching pages belongs to native allocation, not reuse policy.
static iree_status_t iree_hal_slab_cache_prepare_entry(
    iree_hal_slab_cache_t* cache, const iree_hal_slab_cache_parent_t* parent,
    iree_hal_slab_cache_entry_t** out_entry) {
  *out_entry = NULL;
  iree_hal_slab_cache_entry_t* entry = NULL;
  iree_status_t status = iree_allocator_malloc(cache->host_allocator,
                                               sizeof(*entry), (void**)&entry);
  if (iree_status_is_ok(status)) {
    entry->cache = cache;
    entry->cacheable =
        iree_hal_slab_cache_request_is_cacheable(cache, &parent->request);
    entry->reservation = parent->reservation;
    if (parent->info.reuse_frontier) {
      memcpy(&entry->frontier, parent->info.reuse_frontier,
             sizeof(*parent->info.reuse_frontier) +
                 parent->info.reuse_frontier->entry_count *
                     sizeof(parent->info.reuse_frontier->entries[0]));
    }
    status = iree_hal_pool_materialize_reservations(
        cache->backing_pool, 1, &parent->request, &entry->reservation,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &entry->range.buffer);
  }
  if (iree_status_is_ok(status)) {
    entry->range.length = iree_hal_buffer_byte_length(entry->range.buffer);
    entry->range.memory = iree_hal_buffer_memory_view(entry->range.buffer);
    entry->range.memory.reuse_frontier =
        iree_hal_slab_cache_entry_frontier(entry);
    iree_atomic_fetch_add(&cache->counters.committed,
                          (int64_t)entry->reservation.byte_length,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_add(&cache->counters.slabs, 1, iree_memory_order_relaxed);
    iree_slim_mutex_lock(&cache->mutex);
    cache->retained_bytes += entry->reservation.byte_length;
    iree_slim_mutex_unlock(&cache->mutex);
    *out_entry = entry;
  } else {
    iree_hal_pool_release_reservations(cache->backing_pool, 1,
                                       &parent->reservation,
                                       parent->info.reuse_frontier);
    if (entry) {
      iree_hal_pool_buffer_range_deinitialize(&entry->range);
    }
    iree_allocator_free(cache->host_allocator, entry);
  }
  return status;
}

// Detachment relinquishes retained capacity immediately. Neither parent return
// nor physical retirement inflates this policy's retained-byte floor.
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

// Claims refill under the metadata lock. The reference covers parent capture,
// completion, publication, and final cleanup without a destruction join.
static bool iree_hal_slab_cache_claim_refill(iree_hal_slab_cache_t* cache) {
  if (cache->refill_pending || !cache->refill_requested ||
      !iree_status_is_ok(cache->pending_error) ||
      cache->idle_count >= cache->max_count ||
      iree_hal_slab_cache_ready_count(cache) >= cache->target_count) {
    return false;
  }
  cache->refill_pending = true;
  iree_hal_pool_retain(&cache->base);
  return true;
}

static iree_hal_slab_cache_entry_t* iree_hal_slab_cache_trim_idle(
    iree_hal_slab_cache_t* cache, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  iree_hal_slab_cache_entry_t* returns = NULL;
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
      entry->next = returns;
      returns = entry;
      committed -= length;
    } else {
      link = &entry->next;
    }
  }
  return returns;
}

static void iree_hal_slab_cache_refill_progress(iree_hal_slab_cache_t* cache);

static void iree_hal_slab_cache_refill_complete(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_slab_cache_t* cache = user_data;
  cache->refill.parent.status = status;
  cache->refill.parent.result = result;
  if (iree_atomic_fetch_sub(&cache->refill.parent.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_slab_cache_refill_progress(cache);
  }
}

static void iree_hal_slab_cache_refill_progress(iree_hal_slab_cache_t* cache) {
  while (true) {
    if (cache->refill.has_result) {
      cache->refill.has_result = false;
      iree_hal_slab_cache_parent_t* parent = &cache->refill.parent;
      iree_status_t status = parent->status;
      parent->status = iree_ok_status();
      iree_hal_slab_cache_entry_t* entry = NULL;
      if (iree_status_is_ok(status) &&
          parent->result != IREE_HAL_POOL_ACQUIRE_EXHAUSTED &&
          parent->result != IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        status = iree_hal_slab_cache_prepare_entry(cache, parent, &entry);
      }
      bool published = false;
      iree_slim_mutex_lock(&cache->mutex);
      if (cache->refill.generation == cache->generation &&
          cache->refill_requested && entry &&
          cache->idle_count < cache->max_count &&
          iree_hal_slab_cache_ready_count(cache) < cache->target_count) {
        entry->return_time = iree_time_now();
        iree_hal_slab_cache_insert_idle(cache, entry);
        entry = NULL;
        published = true;
      }
      if (!iree_status_is_ok(status)) {
        if (iree_status_is_ok(cache->pending_error)) {
          cache->pending_error = status;
          status = iree_ok_status();
        }
        cache->refill_requested = false;
        published = true;
      }
      if (!entry && !published &&
          cache->refill.generation == cache->generation) {
        // Exhaustion is not a progress event. A later acquisition, return, or
        // explicit target update can ask for another attempt.
        cache->refill_requested = false;
      }
      if (entry) {
        iree_hal_slab_cache_drop_retention(cache, entry);
      }
      iree_slim_mutex_unlock(&cache->mutex);
      iree_status_free(status);
      if (entry) {
        iree_hal_slab_cache_destroy_entry(cache, entry);
      }
      if (published) {
        iree_async_notification_signal_if_observed(cache->base.notification,
                                                   INT32_MAX);
      }
    }
    iree_slim_mutex_lock(&cache->mutex);
    const bool done =
        !cache->refill_requested || !iree_status_is_ok(cache->pending_error) ||
        cache->idle_count >= cache->max_count ||
        iree_hal_slab_cache_ready_count(cache) >= cache->target_count;
    if (done) {
      cache->refill_pending = false;
      iree_slim_mutex_unlock(&cache->mutex);
      iree_hal_pool_release(&cache->base);
      return;
    }
    cache->refill.generation = cache->generation;
    iree_slim_mutex_unlock(&cache->mutex);
    cache->refill.has_result = true;
    cache->refill.parent.request = cache->slab;
    cache->refill.parent.info = (iree_hal_pool_acquire_info_t){0};
    iree_atomic_store(&cache->refill.parent.pending, 2,
                      iree_memory_order_relaxed);
    iree_status_t status = iree_hal_pool_acquire_reservations(
        cache->backing_pool, 1, &cache->refill.parent.request, NULL,
        &cache->refill.parent.reservation, &cache->refill.parent.info,
        (iree_hal_pool_acquire_callback_t){
            .fn = iree_hal_slab_cache_refill_complete,
            .user_data = cache,
        },
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER);
    if (!iree_status_is_ok(status)) {
      iree_hal_slab_cache_refill_complete(cache, status,
                                          IREE_HAL_POOL_ACQUIRE_NONE);
    }
    if (iree_atomic_fetch_sub(&cache->refill.parent.pending, 1,
                              iree_memory_order_acq_rel) != 1) {
      return;
    }
  }
}

static void iree_hal_slab_cache_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  IREE_ASSERT(iree_atomic_load(&cache->counters.reservations,
                               iree_memory_order_relaxed) == 0);
  iree_hal_slab_cache_entry_t* returns =
      iree_hal_slab_cache_trim_idle(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_slab_cache_return_entries(cache, returns);
  // Final destruction consumes an undelivered diagnostic; no caller remains.
  iree_status_free(cache->pending_error);
  iree_hal_pool_release(cache->backing_pool);
  iree_hal_memory_trace_deinitialize(&cache->trace);
  iree_hal_pool_deinitialize(base_pool);
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
  cache->base.epoch_query = backing_pool->epoch_query;
  cache->backing_pool = backing_pool;
  iree_hal_pool_retain(backing_pool);
  cache->host_allocator = host_allocator;
  cache->base.asan_enabled = iree_hal_pool_requires_asan_advice(backing_pool);
  cache->base.allocation_interceptor = backing_pool->allocation_interceptor;
  cache->slab = options->slab;
  cache->slab.min_alignment = iree_max(1, cache->slab.min_alignment);
  cache->target_count = options->target_count;
  cache->max_count = options->max_count;
  iree_slim_mutex_initialize(&cache->mutex);
  iree_hal_pool_query_capabilities(backing_pool, &cache->capabilities);
  status = iree_hal_slab_cache_validate_request(cache, &cache->slab);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_trace_initialize_pool(
        options->trace_name, "iree-hal-slab-cache", &cache->trace);
  }
  if (iree_status_is_ok(status)) {
    cache->refill_requested = cache->target_count != 0;
    if (iree_hal_slab_cache_claim_refill(cache)) {
      iree_hal_slab_cache_refill_progress(cache);
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
  const bool refill = iree_hal_slab_cache_claim_refill(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  if (refill) {
    iree_hal_slab_cache_refill_progress(cache);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Reservation transactions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_slab_cache_acquire_element_t {
  // Entry held privately until whole-batch publication or exact rollback.
  iree_hal_slab_cache_entry_t* entry;
  // Whether rollback restores the entry to the idle cache.
  bool from_idle;
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

// Ready hits use stack storage. A true parent miss promotes the transaction
// once, preserving its prefix and original histories through completion.
typedef struct iree_hal_slab_cache_acquire_t {
  // Borrowed through the submitting caller's completion contract.
  iree_hal_slab_cache_t* cache;
  // Captured public transaction arguments.
  struct {
    // Number of borrowed request and output elements.
    iree_host_size_t count;
    // Validated request geometry.
    const iree_hal_pool_reservation_request_t* requests;
    // Requester ordering proof, independent of physical completion.
    const iree_async_frontier_t* frontier;
    // Reservations published only after the whole batch succeeds.
    iree_hal_pool_reservation_t* reservations;
    // Per-item outcomes and exact prior-use histories.
    iree_hal_pool_acquire_info_t* infos;
    // Exactly one terminal completion for the accepted batch.
    iree_hal_pool_acquire_callback_t completion;
    // Public acquisition behavior.
    iree_hal_pool_reserve_flags_t flags;
  } call;
  // Private entry prefix, including history and original ownership.
  iree_hal_slab_cache_acquire_element_t* elements;
  // Number of entries held privately by this transaction.
  iree_host_size_t acquired_count;
  // Index of the first unavailable request in a declined transaction.
  iree_host_size_t unavailable_index;
  // Whether this object owns its trailing element storage.
  bool allocated;
  // Whether the prefix was withheld across an unlocked parent preparation.
  bool exposed_prefix;
  // Whether the next progression step consumes a completed parent result.
  bool has_parent_result;
  // Stable request and output storage for the current parent miss.
  iree_hal_slab_cache_parent_t parent;
  // Terminal failure transferred to the completion.
  iree_status_t status;
  // Aggregate successful outcome or terminal capacity refusal.
  iree_hal_pool_acquire_result_t result;
} iree_hal_slab_cache_acquire_t;

static iree_status_t iree_hal_slab_cache_acquire_promote(
    iree_hal_slab_cache_acquire_t** inout_acquire) {
  iree_hal_slab_cache_acquire_t* source = *inout_acquire;
  iree_host_size_t total_size = 0;
  iree_host_size_t elements_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*source), &total_size,
      IREE_STRUCT_FIELD(source->call.count,
                        iree_hal_slab_cache_acquire_element_t,
                        &elements_offset)));
  iree_hal_slab_cache_acquire_t* acquire = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(source->cache->host_allocator,
                                             total_size, (void**)&acquire));
  *acquire = *source;
  acquire->allocated = true;
  acquire->elements =
      (iree_hal_slab_cache_acquire_element_t*)((uint8_t*)acquire +
                                               elements_offset);
  if (source->elements) {
    memcpy(acquire->elements, source->elements,
           source->call.count * sizeof(*source->elements));
  }
  *inout_acquire = acquire;
  return iree_ok_status();
}

static void iree_hal_slab_cache_acquire_finish(
    iree_hal_slab_cache_acquire_t* acquire) {
  iree_hal_slab_cache_t* cache = acquire->cache;
  const bool committed = iree_status_is_ok(acquire->status) &&
                         acquire->acquired_count == acquire->call.count;
  bool restored_capacity = false;
  iree_hal_slab_cache_entry_t* returns = NULL;
  iree_slim_mutex_lock(&cache->mutex);
  if (committed) {
    acquire->result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    for (iree_host_size_t i = 0; i < acquire->call.count; ++i) {
      const iree_hal_slab_cache_acquire_element_t* element =
          &acquire->elements[i];
      iree_hal_slab_cache_entry_t* entry = element->entry;
      entry->ownership_transferred = false;
      if (element->from_idle) {
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
      if (element->info.result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
        counter = &cache->counters.waits;
        acquire->result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
      } else if (element->info.result == IREE_HAL_POOL_ACQUIRE_OK) {
        counter = &cache->counters.reused;
        if (acquire->result == IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
          acquire->result = IREE_HAL_POOL_ACQUIRE_OK;
        }
      }
      iree_atomic_fetch_add(counter, 1, iree_memory_order_relaxed);
    }
    iree_atomic_fetch_add(&cache->counters.reservations,
                          (int32_t)acquire->call.count,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_add(&cache->counters.acquisitions,
                          (int64_t)acquire->call.count,
                          iree_memory_order_relaxed);
    cache->refill_requested = cache->target_count != 0;
  } else {
    for (iree_host_size_t i = 0; i < acquire->acquired_count; ++i) {
      iree_hal_slab_cache_acquire_element_t* element = &acquire->elements[i];
      iree_hal_slab_cache_entry_t* entry = element->entry;
      if (element->from_idle && cache->idle_count < cache->max_count) {
        iree_hal_slab_cache_insert_idle(cache, entry);
        restored_capacity = true;
      } else {
        iree_hal_slab_cache_drop_retention(cache, entry);
        entry->next = returns;
        returns = entry;
      }
    }
  }
  const bool refill = iree_hal_slab_cache_claim_refill(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  iree_hal_slab_cache_return_entries(cache, returns);
  if (restored_capacity && acquire->exposed_prefix) {
    iree_async_notification_signal_if_observed(cache->base.notification,
                                               INT32_MAX);
  }
  if (iree_status_is_ok(acquire->status)) {
    if (committed) {
      for (iree_host_size_t i = 0; i < acquire->call.count; ++i) {
        iree_hal_slab_cache_entry_t* entry = acquire->elements[i].entry;
        acquire->call.reservations[i] = (iree_hal_pool_reservation_t){
            .byte_length = entry->range.length,
            .block_handle = (uint64_t)(uintptr_t)entry,
        };
        acquire->call.infos[i] = acquire->elements[i].info;
        iree_hal_memory_trace_alloc(&cache->trace, entry, entry->range.length);
      }
    } else {
      iree_atomic_fetch_add(acquire->result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED
                                ? &cache->counters.exhausted
                                : &cache->counters.over_budget,
                            1, iree_memory_order_relaxed);
      memset(acquire->call.infos, 0,
             acquire->call.count * sizeof(*acquire->call.infos));
      acquire->call.infos[acquire->unavailable_index] =
          (iree_hal_pool_acquire_info_t){
              .result = acquire->result,
          };
    }
  }
  const iree_hal_pool_acquire_callback_t completion = acquire->call.completion;
  const iree_status_t status = acquire->status;
  const iree_hal_pool_acquire_result_t result = acquire->result;
  if (acquire->allocated) {
    iree_allocator_free(cache->host_allocator, acquire);
  }
  if (refill) {
    iree_hal_slab_cache_refill_progress(cache);
  }
  completion.fn(completion.user_data, status, result);
}

static void iree_hal_slab_cache_acquire_progress(
    iree_hal_slab_cache_acquire_t* acquire);

static void iree_hal_slab_cache_acquire_parent_complete(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_slab_cache_acquire_t* acquire = user_data;
  acquire->parent.status = status;
  acquire->parent.result = result;
  if (iree_atomic_fetch_sub(&acquire->parent.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_slab_cache_acquire_progress(acquire);
  }
}

static void iree_hal_slab_cache_acquire_progress(
    iree_hal_slab_cache_acquire_t* acquire) {
  iree_hal_slab_cache_t* cache = acquire->cache;
  while (iree_status_is_ok(acquire->status)) {
    if (acquire->has_parent_result) {
      acquire->has_parent_result = false;
      acquire->status = acquire->parent.status;
      acquire->parent.status = iree_ok_status();
      acquire->result = acquire->parent.result;
      if (!iree_status_is_ok(acquire->status) ||
          acquire->result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
          acquire->result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        break;
      }
      iree_hal_slab_cache_acquire_element_t* element =
          &acquire->elements[acquire->acquired_count];
      acquire->status = iree_hal_slab_cache_prepare_entry(
          cache, &acquire->parent, &element->entry);
      if (!iree_status_is_ok(acquire->status)) {
        break;
      }
      element->info = iree_hal_slab_cache_acquire_info(cache, element->entry,
                                                       acquire->call.frontier);
      ++acquire->acquired_count;
    }
    iree_slim_mutex_lock(&cache->mutex);
    while (acquire->acquired_count < acquire->call.count) {
      const iree_host_size_t index = acquire->acquired_count;
      if (!iree_hal_slab_cache_request_is_cacheable(
              cache, &acquire->call.requests[index])) {
        break;
      }
      iree_hal_slab_cache_acquire_element_t* element =
          &acquire->elements[index];
      element->entry = iree_hal_slab_cache_take_entry(
          cache, acquire->call.frontier, acquire->call.flags);
      if (!element->entry) {
        break;
      }
      element->from_idle = true;
      element->info = iree_hal_slab_cache_acquire_info(cache, element->entry,
                                                       acquire->call.frontier);
      ++acquire->acquired_count;
    }
    iree_slim_mutex_unlock(&cache->mutex);
    if (acquire->acquired_count == acquire->call.count) {
      break;
    }
    acquire->unavailable_index = acquire->acquired_count;
    acquire->exposed_prefix |= acquire->acquired_count != 0;
    if (!acquire->allocated) {
      iree_status_t status = iree_hal_slab_cache_acquire_promote(&acquire);
      acquire->status = status;
      if (!iree_status_is_ok(acquire->status)) {
        break;
      }
    }
    const iree_hal_pool_reservation_request_t* request =
        &acquire->call.requests[acquire->acquired_count];
    const bool cacheable =
        iree_hal_slab_cache_request_is_cacheable(cache, request);
    iree_atomic_fetch_add(
        cacheable ? &cache->counters.misses : &cache->counters.bypasses, 1,
        iree_memory_order_relaxed);
    acquire->has_parent_result = true;
    acquire->parent.request = cacheable ? cache->slab : *request;
    acquire->parent.info = (iree_hal_pool_acquire_info_t){0};
    iree_atomic_store(&acquire->parent.pending, 2, iree_memory_order_relaxed);
    iree_status_t status = iree_hal_pool_acquire_reservations(
        cache->backing_pool, 1, &acquire->parent.request,
        acquire->call.frontier, &acquire->parent.reservation,
        &acquire->parent.info,
        (iree_hal_pool_acquire_callback_t){
            .fn = iree_hal_slab_cache_acquire_parent_complete,
            .user_data = acquire,
        },
        acquire->call.flags);
    if (!iree_status_is_ok(status)) {
      iree_hal_slab_cache_acquire_parent_complete(acquire, status,
                                                  IREE_HAL_POOL_ACQUIRE_NONE);
    }
    if (iree_atomic_fetch_sub(&acquire->parent.pending, 1,
                              iree_memory_order_acq_rel) != 1) {
      return;
    }
  }
  iree_hal_slab_cache_acquire_finish(acquire);
}

static iree_status_t iree_hal_slab_cache_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
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
  const bool needs_storage = request_count > IREE_ARRAYSIZE(inline_elements);
  iree_hal_slab_cache_acquire_t inline_acquire = {
      .cache = cache,
      .call =
          {
              .count = request_count,
              .requests = requests,
              .frontier = requester,
              .reservations = out_reservations,
              .infos = out_infos,
              .completion = completion,
              .flags = flags,
          },
      .elements = needs_storage ? NULL : inline_elements,
  };
  iree_hal_slab_cache_acquire_t* acquire = &inline_acquire;
  if (needs_storage) {
    IREE_RETURN_IF_ERROR(iree_hal_slab_cache_acquire_promote(&acquire));
  }
  memset(acquire->elements, 0, request_count * sizeof(*acquire->elements));
  iree_hal_slab_cache_acquire_progress(acquire);
  return iree_ok_status();
}

static void iree_hal_slab_cache_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  iree_hal_slab_cache_entry_t* returns = NULL;
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
      entry->next = returns;
      returns = entry;
    }
  }
  iree_atomic_fetch_sub(&cache->counters.reservations, (int32_t)count,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&cache->counters.releases, (int64_t)count,
                        iree_memory_order_relaxed);
  cache->refill_requested = cache->target_count != 0;
  const bool refill = iree_hal_slab_cache_claim_refill(cache);
  iree_slim_mutex_unlock(&cache->mutex);
  iree_hal_slab_cache_return_entries(cache, returns);
  if (refill) {
    iree_hal_slab_cache_refill_progress(cache);
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
  return iree_ok_status();
}

static void iree_hal_slab_cache_trim(iree_hal_pool_t* base_pool,
                                     iree_hal_pool_trim_flags_t flags,
                                     iree_device_size_t min_bytes_to_keep) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_pool;
  iree_slim_mutex_lock(&cache->mutex);
  ++cache->generation;
  cache->refill_requested = false;
  iree_hal_slab_cache_entry_t* returns =
      iree_hal_slab_cache_trim_idle(cache, flags, min_bytes_to_keep);
  iree_slim_mutex_unlock(&cache->mutex);
  iree_hal_slab_cache_return_entries(cache, returns);
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
