// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/residency.h"

struct loom_serve_residency_cache_t {
  // Allocation policy for the cache metadata.
  iree_allocator_t allocator;
  // Borrowed physical budget; NULL for explicitly fixed allocations.
  loom_serve_memory_pool_t* pool;
  // Borrowed queues and accepted-work frontiers for cold error retirement.
  loom_serve_execution_t* execution;
  // Least recently used registration, including inactive and pinned entries.
  loom_serve_residency_t* oldest;
  // Most recently used registration.
  loom_serve_residency_t* newest;
};

struct loom_serve_residency_t {
  // Allocation policy for the registration and copied plan pointer array.
  iree_allocator_t allocator;
  // Borrowed cache outliving this registration.
  loom_serve_residency_cache_t* cache;
  // Previous registration in least-to-most-recent order.
  loom_serve_residency_t* previous;
  // Next registration in least-to-most-recent order.
  loom_serve_residency_t* next;
  // Accepted consumers and explicit retention owners excluding eviction.
  iree_host_size_t pin_count;
  // Every checkpoint domain has completed activation.
  bool active;
  // Number of borrowed checkpoint plans in the trailing array.
  iree_host_size_t weight_count;
  // Immutable checkpoint plans required together by each invocation.
  loom_serve_weights_t* weights[];
};

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
  if (residency->pin_count || !residency->cache->pool) {
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
    if (!entry->pin_count &&
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
  for (loom_serve_residency_t* entry = cache->oldest;
       entry && remaining > target_bytes; entry = entry->next) {
    if (!entry->pin_count) {
      remaining -= loom_serve_residency_statistics(entry).committed_bytes;
    }
  }
  // Insufficient eligible backing cannot admit this request. Preserve useful
  // idle weights instead of evicting them for a request that still cannot fit.
  return remaining <= target_bytes
             ? loom_serve_residency_cache_trim(cache, target_bytes)
             : iree_ok_status();
}

iree_status_t loom_serve_residency_cache_create(
    loom_serve_memory_pool_t* pool, loom_serve_execution_t* execution,
    loom_serve_residency_cache_t** out_cache, iree_allocator_t host_allocator) {
  *out_cache = NULL;
  loom_serve_residency_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->allocator = host_allocator;
  cache->pool = pool;
  cache->execution = execution;
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
    loom_serve_residency_cache_t* cache, iree_host_size_t weight_count,
    loom_serve_weights_t* const* weights,
    loom_serve_residency_t** out_residency, iree_allocator_t host_allocator) {
  *out_residency = NULL;
  loom_serve_residency_t* residency = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*residency) + weight_count * sizeof(*weights),
      (void**)&residency));
  residency->allocator = host_allocator;
  residency->cache = cache;
  residency->weight_count = weight_count;
  for (iree_host_size_t i = 0; i < weight_count; ++i) {
    residency->weights[i] = weights[i];
  }
  residency_append(residency);
  *out_residency = residency;
  return iree_ok_status();
}

void loom_serve_residency_destroy(loom_serve_residency_t* residency) {
  if (!residency) {
    return;
  }
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
      status = loom_serve_weights_activate(residency->weights[i]);
    }
    IREE_RETURN_IF_ERROR(status);
    residency->active = true;
  }
  *out_admitted = true;
  return iree_ok_status();
}

iree_status_t loom_serve_residency_acquire(loom_serve_residency_t* residency) {
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_try_acquire(residency, &admitted));
  return admitted ? iree_ok_status()
                  : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                     "model residency is waiting for capacity");
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
    retirement = loom_serve_execution_drain(residency->cache->execution);
  }
  if (iree_status_is_ok(retirement)) {
    loom_serve_residency_release(residency);
  }
  return iree_status_join(status, retirement);
}

iree_status_t loom_serve_residency_activate(loom_serve_residency_t* residency) {
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire(residency));
  loom_serve_residency_release(residency);
  return iree_ok_status();
}
