// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/residency.h"

struct loom_serve_residency_cache_t {
  // Allocation policy for the cache metadata.
  iree_allocator_t allocator;
  // Borrowed physical budget; NULL for explicitly fixed allocations.
  loom_serve_memory_pool_t* pool;
  // Least recently used registration, including inactive and pinned entries.
  loom_serve_residency_t* oldest;
  // Most recently used registration.
  loom_serve_residency_t* newest;
  // Explicit new-work interests after a physical weight-capacity denial.
  struct {
    // Oldest waiting service, retried by the shared host owner.
    loom_serve_residency_t* first;
    // Newest waiting service.
    loom_serve_residency_t* last;
  } admission;
};

struct loom_serve_residency_t {
  // Allocation policy for the registration and copied plan pointer array.
  iree_allocator_t allocator;
  // Borrowed cache outliving this registration.
  loom_serve_residency_cache_t* cache;
  // Borrowed model frontiers for cold error retirement, never other models.
  loom_serve_execution_t* execution;
  // Previous registration in least-to-most-recent order.
  loom_serve_residency_t* previous;
  // Next registration in least-to-most-recent order.
  loom_serve_residency_t* next;
  // Accepted consumers and explicit retention owners excluding eviction.
  iree_host_size_t pin_count;
  // Persistent weight-only retention, independent of accepted operation pins.
  bool pinned;
  // Every checkpoint domain has enqueued readiness on the model timeline.
  bool active;
  // Borrowed model idle-cache owner, independent of parameter pins.
  loom_serve_residency_reclaimer_t reclaimer;
  // Independent of LRU membership and all already-accepted execution pins.
  struct {
    // Older interest, or NULL at the head/not queued.
    loom_serve_residency_t* previous;
    // Newer interest, or NULL at the tail/not queued.
    loom_serve_residency_t* next;
  } admission;
  // Number of borrowed checkpoint plans in the trailing array.
  iree_host_size_t weight_count;
  // Immutable checkpoint plans required together by each invocation.
  loom_serve_weights_t* weights[];
};

loom_serve_residency_cache_t* loom_serve_residency_cache(
    const loom_serve_residency_t* residency) {
  return residency->cache;
}

void loom_serve_residency_set_pinned(loom_serve_residency_t* residency,
                                     bool pinned) {
  residency->pinned = pinned;
}

void loom_serve_residency_set_reclaimer(
    loom_serve_residency_t* residency,
    loom_serve_residency_reclaimer_t reclaimer) {
  residency->reclaimer = reclaimer;
}

bool loom_serve_residency_cache_is_reclaiming(
    const loom_serve_residency_cache_t* cache) {
  for (const loom_serve_residency_t* entry = cache->oldest; entry;
       entry = entry->next) {
    if (entry->reclaimer.is_pending &&
        entry->reclaimer.is_pending(entry->reclaimer.self)) {
      return true;
    }
  }
  return false;
}

loom_serve_memory_statistics_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency) {
  loom_serve_memory_statistics_t total = {0};
  for (iree_host_size_t i = 0; i < residency->weight_count; ++i) {
    const loom_serve_memory_statistics_t domain =
        loom_serve_weights_statistics(residency->weights[i]);
    total.reserved_bytes += domain.reserved_bytes;
    total.committed_bytes += domain.committed_bytes;
    total.peak_bytes += domain.peak_bytes;
    total.released_bytes += domain.released_bytes;
  }
  return total;
}

loom_serve_weight_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency) {
  loom_serve_weight_activity_t total = {0};
  for (iree_host_size_t i = 0; i < residency->weight_count; ++i) {
    const loom_serve_weight_activity_t domain =
        loom_serve_weights_activity(residency->weights[i]);
    total.activations += domain.activations;
    total.deactivations += domain.deactivations;
    total.read_bytes += domain.read_bytes;
    total.prepared_bytes += domain.prepared_bytes;
    total.groups += domain.groups;
  }
  return total;
}

loom_serve_residency_snapshot_t loom_serve_residency_snapshot(
    const loom_serve_residency_t* residency) {
  return (loom_serve_residency_snapshot_t){
      .active = residency->active,
      .elastic = residency->cache->pool != NULL,
      .memory = loom_serve_residency_statistics(residency),
      .activity = loom_serve_residency_activity(residency)};
}

static void residency_unlink(loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (residency->previous) {
    residency->previous->next = residency->next;
  } else {
    cache->oldest = residency->next;
  }
  if (residency->next) {
    residency->next->previous = residency->previous;
  } else {
    cache->newest = residency->previous;
  }
}

static void residency_append(loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  residency->previous = cache->newest;
  residency->next = NULL;
  if (cache->newest) {
    cache->newest->next = residency;
  } else {
    cache->oldest = residency;
  }
  cache->newest = residency;
}

iree_status_t loom_serve_residency_deactivate(
    loom_serve_residency_t* residency) {
  if (residency->pin_count || residency->pinned || !residency->cache->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires unpinned elastic residency");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < residency->weight_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weights_deactivate(residency->weights[i]);
  }
  if (iree_status_is_ok(status)) {
    residency->active = false;
  }
  return status;
}

iree_status_t loom_serve_residency_cache_trim(
    loom_serve_residency_cache_t* cache, uint64_t target_bytes) {
  if (!cache->pool) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (loom_serve_residency_t* entry = cache->oldest;
       entry && iree_status_is_ok(status) &&
       loom_serve_memory_pool_statistics(cache->pool).committed_bytes >
           target_bytes;
       entry = entry->next) {
    if (!entry->pin_count && !entry->pinned &&
        loom_serve_residency_statistics(entry).committed_bytes) {
      status = loom_serve_residency_deactivate(entry);
    }
  }
  return status;
}

static iree_status_t residency_reclaim(void* user_data, uint64_t target_bytes) {
  loom_serve_residency_cache_t* cache = user_data;
  uint64_t remaining =
      loom_serve_memory_pool_statistics(cache->pool).committed_bytes;
  uint64_t pinned_bytes = 0;
  for (loom_serve_residency_t* entry = cache->oldest;
       entry && remaining > target_bytes; entry = entry->next) {
    const uint64_t committed_bytes =
        loom_serve_residency_statistics(entry).committed_bytes;
    if (entry->pin_count || entry->pinned) {
      pinned_bytes += committed_bytes;
    } else {
      remaining -= committed_bytes;
    }
  }
  if (remaining <= target_bytes) {
    return loom_serve_residency_cache_trim(cache, target_bytes);
  }
  // No idle history can make room while pinned parameters alone exceed the
  // target. Preserve it until a pin retires and reclamation can help admission.
  if (pinned_bytes > target_bytes) {
    return iree_ok_status();
  }
  // Keep useful weights when their release alone cannot admit the request.
  // Release at most one idle logical owner, then let its normal asynchronous
  // maintenance retire before asking for more. A physical trim here could
  // recurse into admission for compaction destinations.
  if (!loom_serve_residency_cache_is_reclaiming(cache)) {
    for (loom_serve_residency_t* entry = cache->oldest; entry;
         entry = entry->next) {
      if (entry->reclaimer.request &&
          entry->reclaimer.request(entry->reclaimer.self)) {
        break;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_residency_cache_create(
    loom_serve_memory_pool_t* pool, loom_serve_residency_cache_t** out_cache,
    iree_allocator_t host_allocator) {
  *out_cache = NULL;
  loom_serve_residency_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->allocator = host_allocator;
  cache->pool = pool;
  if (pool) {
    loom_serve_memory_pool_set_reclaimer(
        pool, (loom_serve_memory_reclaimer_t){cache, residency_reclaim});
  }
  *out_cache = cache;
  return iree_ok_status();
}

void loom_serve_residency_cache_destroy(loom_serve_residency_cache_t* cache) {
  if (!cache) {
    return;
  }
  IREE_ASSERT(!cache->oldest);
  if (cache->pool) {
    loom_serve_memory_pool_set_reclaimer(cache->pool,
                                         (loom_serve_memory_reclaimer_t){0});
  }
  iree_allocator_free(cache->allocator, cache);
}

iree_status_t loom_serve_residency_create(
    loom_serve_residency_cache_t* cache, loom_serve_execution_t* execution,
    iree_host_size_t weight_count, loom_serve_weights_t* const* weights,
    loom_serve_residency_t** out_residency, iree_allocator_t host_allocator) {
  *out_residency = NULL;
  if (cache->pool) {
    uint64_t remaining = loom_serve_memory_pool_limit(cache->pool);
    for (iree_host_size_t i = 0; i < weight_count; ++i) {
      const uint64_t required =
          loom_serve_weights_statistics(weights[i]).reserved_bytes;
      if (required > remaining) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "model parameters exceed the physical pool budget of %" PRIu64
            " bytes",
            loom_serve_memory_pool_limit(cache->pool));
      }
      remaining -= required;
    }
  }
  loom_serve_residency_t* residency = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*residency) + weight_count * sizeof(*weights),
      (void**)&residency));
  residency->allocator = host_allocator;
  residency->cache = cache;
  residency->execution = execution;
  residency->weight_count = weight_count;
  for (iree_host_size_t i = 0; i < weight_count; ++i) {
    residency->weights[i] = weights[i];
  }
  residency_append(residency);
  *out_residency = residency;
  return iree_ok_status();
}

void loom_serve_residency_cancel_admission(loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (!residency->admission.previous && cache->admission.first != residency) {
    return;
  }
  if (residency->admission.previous) {
    residency->admission.previous->admission.next = residency->admission.next;
  } else {
    cache->admission.first = residency->admission.next;
  }
  if (residency->admission.next) {
    residency->admission.next->admission.previous =
        residency->admission.previous;
  } else {
    cache->admission.last = residency->admission.previous;
  }
  residency->admission.previous = NULL;
  residency->admission.next = NULL;
}

static void residency_queue_admission(loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (residency->admission.previous || cache->admission.first == residency) {
    return;
  }
  residency->admission.previous = cache->admission.last;
  if (cache->admission.last) {
    cache->admission.last->admission.next = residency;
  } else {
    cache->admission.first = residency;
  }
  cache->admission.last = residency;
}

void loom_serve_residency_destroy(loom_serve_residency_t* residency) {
  if (!residency) {
    return;
  }
  loom_serve_residency_cancel_admission(residency);
  residency_unlink(residency);
  iree_allocator_free(residency->allocator, residency);
}

iree_status_t loom_serve_residency_try_acquire(
    loom_serve_residency_t* residency, bool* out_admitted) {
  *out_admitted = false;
  ++residency->pin_count;
  if (!residency->active) {
    if (residency->cache->pool) {
      const loom_serve_memory_statistics_t memory =
          loom_serve_residency_statistics(residency);
      bool admitted = false;
      IREE_RETURN_IF_ERROR(loom_serve_memory_pool_prepare(
          residency->cache->pool,
          memory.reserved_bytes - memory.committed_bytes, &admitted));
      if (!admitted) {
        --residency->pin_count;
        return iree_ok_status();
      }
    }
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < residency->weight_count && iree_status_is_ok(status); ++i) {
      status = loom_serve_weights_activate(residency->weights[i],
                                           residency->execution);
    }
    IREE_RETURN_IF_ERROR(status);
    residency->active = true;
  }
  *out_admitted = true;
  return iree_ok_status();
}

iree_status_t loom_serve_residency_try_admit(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  loom_serve_residency_cache_t* cache = residency->cache;
  if (cache->admission.first && cache->admission.first != residency &&
      loom_serve_residency_statistics(residency).reserved_bytes) {
    residency_queue_admission(residency);
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_try_acquire(residency, &admitted));
  if (admitted) {
    loom_serve_residency_cancel_admission(residency);
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  } else {
    residency_queue_admission(residency);
    if (loom_serve_residency_cache_is_reclaiming(cache)) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_residency_acquire(loom_serve_residency_t* residency) {
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_try_acquire(residency, &admitted));
  return admitted ? iree_ok_status()
                  : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                     "model residency is waiting for capacity");
}

iree_status_t loom_serve_residency_await(loom_serve_residency_t* residency) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < residency->weight_count; ++i) {
    status = iree_status_join(status,
                              loom_serve_weights_await(residency->weights[i]));
  }
  return status;
}

loom_serve_execution_frontier_t loom_serve_residency_frontier(
    const loom_serve_residency_t* residency) {
  return loom_serve_execution_frontier(residency->execution);
}

void loom_serve_residency_release(loom_serve_residency_t* residency) {
  IREE_ASSERT(residency->pin_count);
  if (!--residency->pin_count) {
    residency_unlink(residency);
    residency_append(residency);
  }
}

iree_status_t loom_serve_residency_finish(loom_serve_residency_t* residency,
                                          iree_status_t status) {
  iree_status_t retirement = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    retirement = loom_serve_execution_drain(residency->execution);
  }
  if (iree_status_is_ok(retirement)) {
    loom_serve_residency_release(residency);
  }
  return iree_status_join(status, retirement);
}

iree_status_t loom_serve_residency_activate(loom_serve_residency_t* residency) {
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire(residency));
  IREE_RETURN_IF_ERROR(loom_serve_residency_await(residency));
  loom_serve_residency_release(residency);
  return iree_ok_status();
}
