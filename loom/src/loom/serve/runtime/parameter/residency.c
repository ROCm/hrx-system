// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/residency.h"

struct loom_serve_residency_cache_t {
  // Allocation policy for the cache metadata.
  iree_allocator_t host_allocator;
  // Borrowed physical budget; NULL for explicitly fixed allocations.
  loom_serve_memory_pool_t* pool;
  // Least recently used model history owner, independent of shared weights.
  loom_serve_residency_t* oldest;
  // Most recently used model history owner.
  loom_serve_residency_t* newest;
  // Physical groups, each counted once regardless of borrowing residencies.
  struct {
    // Least recently used group, including inactive and pinned entries.
    loom_serve_residency_group_t* oldest;
    // Most recently used group.
    loom_serve_residency_group_t* newest;
  } groups;
  // Physical groups with release interest awaiting owner collection.
  iree_host_size_t releasing_count;
  // Explicit new-work interests after a physical weight-capacity denial.
  struct {
    // Oldest waiting service, retried by the shared host owner.
    loom_serve_residency_t* first;
    // Newest waiting service.
    loom_serve_residency_t* last;
  } admission;
};

struct loom_serve_residency_group_t {
  // Allocator owning this registration and its copied plan pointers.
  iree_allocator_t host_allocator;
  // Borrowed physical policy owner, outliving this group.
  loom_serve_residency_cache_t* cache;
  // Previous physical group in least-to-most-recent order.
  loom_serve_residency_group_t* previous;
  // Next physical group in least-to-most-recent order.
  loom_serve_residency_group_t* next;
  // Residencies with accepted consumers, counted once per nested pin chain.
  iree_host_size_t pin_count;
  // Residencies requesting persistent retention, independent of consumers.
  iree_host_size_t retention_count;
  // Readiness identity, advanced on eviction before subsequent reactivation.
  uint64_t generation;
  // Every plan has submitted its immutable loading/preparation work.
  bool active;
  // Logical release interest retained until owner collection.
  bool releasing;
  // Last final consumer release or explicit unpin, not observation time.
  iree_time_t idle_since;
  // Number of exclusive borrowed plans in the trailing array.
  iree_host_size_t weight_count;
  // Plans whose backing is activated and evicted together.
  loom_serve_weights_t* weights[];
};

typedef struct loom_serve_residency_entry_t {
  // Borrowed immutable resource group, outliving this working set.
  loom_serve_residency_group_t* group;
  // Group generation whose readiness was imported into this execution.
  uint64_t generation;
} loom_serve_residency_entry_t;

struct loom_serve_residency_t {
  // Allocation policy for the registration and copied working-set entries.
  iree_allocator_t host_allocator;
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
  // Last acquisition submitted all imports; groups may since be evicted.
  bool active;
  // Borrowed model idle-cache owner, independent of parameter pins.
  loom_serve_residency_reclaimer_t reclaimer;
  // Independent of LRU membership and all already-accepted execution pins.
  struct {
    // Older interest, or NULL at the head/not queued.
    loom_serve_residency_t* previous;
    // Newer interest, or NULL at the tail/not queued.
    loom_serve_residency_t* next;
    // Number of distinct wait episodes, independent of repeated admission
    // polls.
    uint64_t count;
    // Accumulated completed waits, including withdrawals, in nanoseconds.
    uint64_t duration_ns;
    // Monotonic start of the current wait; meaningful only while queued.
    iree_time_t start_time;
  } admission;
  // Number of distinct physical groups in the trailing working set.
  iree_host_size_t group_count;
  // Shared immutable groups acquired together by each invocation/cohort.
  loom_serve_residency_entry_t entries[];
};

static bool loom_serve_residency_is_waiting(
    const loom_serve_residency_t* residency) {
  return residency->admission.previous ||
         residency->cache->admission.first == residency;
}

loom_serve_residency_cache_t* loom_serve_residency_cache(
    const loom_serve_residency_t* residency) {
  return residency->cache;
}

void loom_serve_residency_set_pinned(loom_serve_residency_t* residency,
                                     bool pinned) {
  if (residency->pinned == pinned) {
    return;
  }
  const iree_time_t now = pinned ? 0 : iree_time_now();
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    loom_serve_residency_group_t* group = residency->entries[i].group;
    if (pinned) {
      ++group->retention_count;
    } else if (!--group->retention_count) {
      group->idle_since = now;
    }
  }
  residency->pinned = pinned;
}

bool loom_serve_residency_is_pinned(const loom_serve_residency_t* residency) {
  return residency->pin_count || residency->pinned;
}

void loom_serve_residency_set_reclaimer(
    loom_serve_residency_t* residency,
    loom_serve_residency_reclaimer_t reclaimer) {
  residency->reclaimer = reclaimer;
}

bool loom_serve_residency_cache_is_reclaiming(
    const loom_serve_residency_cache_t* cache) {
  if (cache->releasing_count ||
      (cache->pool && loom_serve_memory_pool_is_pending(cache->pool))) {
    return true;
  }
  for (const loom_serve_residency_t* entry = cache->oldest; entry;
       entry = entry->next) {
    if (entry->reclaimer.is_pending &&
        entry->reclaimer.is_pending(entry->reclaimer.self)) {
      return true;
    }
  }
  return false;
}

static loom_serve_memory_statistics_t loom_serve_residency_group_statistics(
    const loom_serve_residency_group_t* group) {
  loom_serve_memory_statistics_t total = {0};
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_serve_memory_statistics_t domain =
        loom_serve_weights_statistics(group->weights[i]);
    total.reserved_bytes += domain.reserved_bytes;
    total.committed_bytes += domain.committed_bytes;
    total.pending_bytes += domain.pending_bytes;
    total.releasing_bytes += domain.releasing_bytes;
    total.peak_bytes += domain.peak_bytes;
    total.released_bytes += domain.released_bytes;
  }
  return total;
}

loom_serve_memory_statistics_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency) {
  loom_serve_memory_statistics_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_memory_statistics_t group =
        loom_serve_residency_group_statistics(residency->entries[i].group);
    total.reserved_bytes += group.reserved_bytes;
    total.committed_bytes += group.committed_bytes;
    total.pending_bytes += group.pending_bytes;
    total.releasing_bytes += group.releasing_bytes;
    total.peak_bytes += group.peak_bytes;
    total.released_bytes += group.released_bytes;
  }
  return total;
}

loom_serve_weight_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency) {
  loom_serve_weight_activity_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
      const loom_serve_weight_activity_t domain =
          loom_serve_weights_activity(group->weights[j]);
      total.activations += domain.activations;
      total.deactivations += domain.deactivations;
      total.read_bytes += domain.read_bytes;
      total.prepared_bytes += domain.prepared_bytes;
      total.groups += domain.groups;
    }
  }
  return total;
}

loom_serve_residency_snapshot_t loom_serve_residency_snapshot(
    const loom_serve_residency_t* residency) {
  const bool waiting = loom_serve_residency_is_waiting(residency);
  const uint64_t age_ns =
      waiting ? iree_time_now() - residency->admission.start_time : 0;
  bool active = residency->active;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_entry_t* entry = &residency->entries[i];
    active &=
        entry->group->active && entry->generation == entry->group->generation;
  }
  return (loom_serve_residency_snapshot_t){
      .active = active,
      .elastic = residency->cache->pool != NULL,
      .pin_count = residency->pin_count,
      .admission = {.waiting = waiting,
                    .count = residency->admission.count,
                    .duration_ns = residency->admission.duration_ns + age_ns,
                    .age_ns = age_ns},
      .memory = loom_serve_residency_statistics(residency),
      .activity = loom_serve_residency_activity(residency)};
}

static void loom_serve_residency_unlink(loom_serve_residency_t* residency) {
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

static void loom_serve_residency_append(loom_serve_residency_t* residency) {
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

static void loom_serve_residency_group_unlink(
    loom_serve_residency_group_t* group) {
  loom_serve_residency_cache_t* cache = group->cache;
  if (group->previous) {
    group->previous->next = group->next;
  } else {
    cache->groups.oldest = group->next;
  }
  if (group->next) {
    group->next->previous = group->previous;
  } else {
    cache->groups.newest = group->previous;
  }
}

static void loom_serve_residency_group_append(
    loom_serve_residency_group_t* group) {
  loom_serve_residency_cache_t* cache = group->cache;
  group->previous = cache->groups.newest;
  group->next = NULL;
  if (cache->groups.newest) {
    cache->groups.newest->next = group;
  } else {
    cache->groups.oldest = group;
  }
  cache->groups.newest = group;
}

static iree_status_t loom_serve_residency_group_deactivate(
    loom_serve_residency_group_t* group, bool* out_complete) {
  *out_complete = false;
  if (group->active) {
    group->active = false;
    ++group->generation;
  }
  if (!group->releasing) {
    ++group->cache->releasing_count;
    group->releasing = true;
  }
  iree_status_t status = iree_ok_status();
  bool complete = true;
  for (iree_host_size_t i = 0;
       i < group->weight_count && iree_status_is_ok(status); ++i) {
    bool weight_complete = false;
    status = loom_serve_weights_deactivate(group->weights[i], &weight_complete);
    complete &= weight_complete;
  }
  if (iree_status_is_ok(status)) {
    if (complete) {
      --group->cache->releasing_count;
      group->releasing = false;
    }
    *out_complete = complete;
  }
  return status;
}

iree_status_t loom_serve_residency_deactivate(loom_serve_residency_t* residency,
                                              bool* out_complete) {
  *out_complete = false;
  bool pinned = residency->pin_count || residency->pinned;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    pinned |= !group->releasing && (group->pin_count || group->retention_count);
  }
  if (pinned || !residency->cache->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires unpinned elastic residency");
  }
  iree_status_t status = iree_ok_status();
  bool complete = true;
  for (iree_host_size_t i = 0;
       i < residency->group_count && iree_status_is_ok(status); ++i) {
    bool group_complete = false;
    status = loom_serve_residency_group_deactivate(residency->entries[i].group,
                                                   &group_complete);
    complete &= group_complete;
  }
  residency->active = false;
  *out_complete = iree_status_is_ok(status) && complete;
  return status;
}

iree_status_t loom_serve_residency_cache_query_status(
    loom_serve_residency_cache_t* cache) {
  if (!cache->releasing_count) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (loom_serve_residency_group_t* entry = cache->groups.oldest; entry;
       entry = entry->next) {
    if (entry->releasing) {
      bool complete = false;
      status = iree_status_join(
          status, loom_serve_residency_group_deactivate(entry, &complete));
    }
  }
  return status;
}

static uint64_t loom_serve_residency_cache_projected_bytes(
    loom_serve_residency_cache_t* cache) {
  const loom_serve_memory_statistics_t memory =
      loom_serve_memory_pool_statistics(cache->pool);
  return memory.committed_bytes - memory.releasing_bytes;
}

iree_status_t loom_serve_residency_cache_trim_idle(
    loom_serve_residency_cache_t* cache, iree_time_t now,
    iree_duration_t timeout, bool* out_pending) {
  *out_pending = false;
  if (!cache->pool) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (loom_serve_residency_group_t* entry = cache->groups.oldest;
       entry && iree_status_is_ok(status); entry = entry->next) {
    if (!entry->active || !entry->weight_count || entry->pin_count ||
        entry->retention_count) {
      continue;
    }
    if (now - entry->idle_since < timeout) {
      *out_pending = true;
    } else {
      bool complete = false;
      status = loom_serve_residency_group_deactivate(entry, &complete);
    }
  }
  return status;
}

iree_status_t loom_serve_residency_cache_trim(
    loom_serve_residency_cache_t* cache, uint64_t target_bytes) {
  if (!cache->pool) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (loom_serve_residency_group_t* entry = cache->groups.oldest;
       entry && iree_status_is_ok(status) &&
       loom_serve_residency_cache_projected_bytes(cache) > target_bytes;
       entry = entry->next) {
    if (entry->active && !entry->pin_count && !entry->retention_count &&
        !entry->releasing &&
        loom_serve_residency_group_statistics(entry).committed_bytes) {
      bool complete = false;
      status = loom_serve_residency_group_deactivate(entry, &complete);
    }
  }
  return status;
}

static iree_status_t loom_serve_residency_reclaim(void* user_data,
                                                  uint64_t target_bytes) {
  loom_serve_residency_cache_t* cache = user_data;
  uint64_t remaining = loom_serve_residency_cache_projected_bytes(cache);
  uint64_t pinned_bytes = 0;
  for (loom_serve_residency_group_t* entry = cache->groups.oldest;
       entry && remaining > target_bytes; entry = entry->next) {
    const loom_serve_memory_statistics_t memory =
        loom_serve_residency_group_statistics(entry);
    const uint64_t committed_bytes =
        memory.committed_bytes - memory.releasing_bytes;
    if (entry->pin_count || entry->retention_count || !entry->active) {
      pinned_bytes += committed_bytes;
    } else {
      remaining -= committed_bytes;
    }
  }
  if (remaining <= target_bytes) {
    return loom_serve_residency_cache_trim(cache, target_bytes);
  }
  // Pinned parameters and terminal partial loads cannot be evicted. No idle
  // history can help while that protected backing alone exceeds the target.
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

iree_status_t loom_serve_residency_cache_allocate(
    loom_serve_memory_pool_t* pool, iree_allocator_t host_allocator,
    loom_serve_residency_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_residency_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  cache->pool = pool;
  if (pool) {
    loom_serve_memory_pool_set_reclaimer(
        pool,
        (loom_serve_memory_reclaimer_t){cache, loom_serve_residency_reclaim});
  }
  *out_cache = cache;
  return iree_ok_status();
}

void loom_serve_residency_cache_free(loom_serve_residency_cache_t* cache) {
  if (!cache) {
    return;
  }
  IREE_ASSERT(!cache->oldest);
  IREE_ASSERT(!cache->groups.oldest);
  if (cache->pool) {
    loom_serve_memory_pool_set_reclaimer(cache->pool,
                                         (loom_serve_memory_reclaimer_t){0});
  }
  iree_allocator_free(cache->host_allocator, cache);
}

iree_status_t loom_serve_residency_group_allocate(
    loom_serve_residency_cache_t* cache, iree_host_size_t weight_count,
    loom_serve_weights_t* const* weights, iree_allocator_t host_allocator,
    loom_serve_residency_group_t** out_group) {
  *out_group = NULL;
  loom_serve_residency_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*group) + weight_count * sizeof(*weights),
      (void**)&group));
  group->host_allocator = host_allocator;
  group->cache = cache;
  group->generation = 1;
  group->weight_count = weight_count;
  for (iree_host_size_t i = 0; i < weight_count; ++i) {
    group->weights[i] = weights[i];
  }
  loom_serve_residency_group_append(group);
  *out_group = group;
  return iree_ok_status();
}

void loom_serve_residency_group_free(loom_serve_residency_group_t* group) {
  if (!group) {
    return;
  }
  loom_serve_residency_group_unlink(group);
  if (group->releasing) {
    --group->cache->releasing_count;
  }
  iree_allocator_free(group->host_allocator, group);
}

iree_status_t loom_serve_residency_allocate(
    loom_serve_residency_cache_t* cache, loom_serve_execution_t* execution,
    iree_host_size_t group_count, loom_serve_residency_group_t* const* groups,
    iree_allocator_t host_allocator, loom_serve_residency_t** out_residency) {
  *out_residency = NULL;
  if (cache->pool) {
    uint64_t remaining = loom_serve_memory_pool_limit(cache->pool);
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      const uint64_t required =
          loom_serve_residency_group_statistics(groups[i]).reserved_bytes;
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
      host_allocator,
      sizeof(*residency) + group_count * sizeof(*residency->entries),
      (void**)&residency));
  residency->host_allocator = host_allocator;
  residency->cache = cache;
  residency->execution = execution;
  residency->group_count = group_count;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    residency->entries[i].group = groups[i];
  }
  loom_serve_residency_append(residency);
  *out_residency = residency;
  return iree_ok_status();
}

void loom_serve_residency_cancel_admission(loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (!loom_serve_residency_is_waiting(residency)) {
    return;
  }
  residency->admission.duration_ns +=
      iree_time_now() - residency->admission.start_time;
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

static void loom_serve_residency_queue_admission(
    loom_serve_residency_t* residency) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (loom_serve_residency_is_waiting(residency)) {
    return;
  }
  ++residency->admission.count;
  residency->admission.start_time = iree_time_now();
  residency->admission.previous = cache->admission.last;
  if (cache->admission.last) {
    cache->admission.last->admission.next = residency;
  } else {
    cache->admission.first = residency;
  }
  cache->admission.last = residency;
}

void loom_serve_residency_free(loom_serve_residency_t* residency) {
  if (!residency) {
    return;
  }
  loom_serve_residency_cancel_admission(residency);
  loom_serve_residency_set_pinned(residency, false);
  if (residency->pin_count) {
    residency->pin_count = 1;
    loom_serve_residency_release(residency);
  }
  loom_serve_residency_unlink(residency);
  iree_allocator_free(residency->host_allocator, residency);
}

static iree_status_t loom_serve_residency_acquire_parameters(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  // Nested accepted work already protects the complete working set.
  if (residency->pin_count && residency->active) {
    ++residency->pin_count;
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_cache_query_status(residency->cache));
  bool active = true;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    if (residency->entries[i].group->releasing) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      return iree_ok_status();
    }
    active &= residency->entries[i].group->active;
  }
  residency->active = false;
  ++residency->pin_count;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    ++residency->entries[i].group->pin_count;
  }
  if (!active && residency->cache->pool) {
    uint64_t missing_bytes = 0;
    for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
      const loom_serve_residency_group_t* group = residency->entries[i].group;
      if (!group->active) {
        const loom_serve_memory_statistics_t memory =
            loom_serve_residency_group_statistics(group);
        missing_bytes += memory.reserved_bytes - memory.committed_bytes -
                         memory.pending_bytes;
      }
    }
    loom_serve_memory_admission_t admission;
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_prepare(
        residency->cache->pool, missing_bytes, &admission));
    if (admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
      --residency->pin_count;
      for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
        --residency->entries[i].group->pin_count;
      }
      if (admission == LOOM_SERVE_MEMORY_ADMISSION_DEFERRED) {
        *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      }
      return iree_ok_status();
    }
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < residency->group_count && iree_status_is_ok(status); ++i) {
    loom_serve_residency_entry_t* entry = &residency->entries[i];
    loom_serve_residency_group_t* group = entry->group;
    if (group->active && entry->generation == group->generation) {
      continue;
    }
    if (!group->active) {
      for (iree_host_size_t j = 0;
           j < group->weight_count && iree_status_is_ok(status); ++j) {
        status = loom_serve_weights_activate(group->weights[j]);
      }
      group->active = iree_status_is_ok(status);
    }
    for (iree_host_size_t j = 0;
         j < group->weight_count && iree_status_is_ok(status); ++j) {
      status = loom_serve_weights_import_readiness(group->weights[j],
                                                   residency->execution);
    }
    if (iree_status_is_ok(status)) {
      entry->generation = group->generation;
    }
  }
  IREE_RETURN_IF_ERROR(status);
  residency->active = true;
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  return iree_ok_status();
}

iree_status_t loom_serve_residency_try_acquire(
    loom_serve_residency_t* residency, bool* out_admitted) {
  loom_serve_residency_admission_t admission;
  iree_status_t status =
      loom_serve_residency_acquire_parameters(residency, &admission);
  *out_admitted = admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  return status;
}

iree_status_t loom_serve_residency_try_admit(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  loom_serve_residency_cache_t* cache = residency->cache;
  // Mutable-state denial may follow even a warm parameter acquisition. Retire
  // completed peer releases before that attempt so its backpressure cannot
  // wait on logical interest whose native completion wake already happened.
  IREE_RETURN_IF_ERROR(loom_serve_residency_cache_query_status(cache));
  if (cache->admission.first && cache->admission.first != residency &&
      loom_serve_residency_statistics(residency).reserved_bytes) {
    loom_serve_residency_queue_admission(residency);
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_acquire_parameters(residency, out_admission));
  if (*out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_residency_queue_admission(residency);
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

iree_status_t loom_serve_residency_acquire_wait(
    loom_serve_residency_t* residency) {
  bool admitted = false;
  loom_serve_memory_pool_t* pool = residency->cache->pool;
  iree_status_t status = loom_serve_residency_try_acquire(residency, &admitted);
  while (iree_status_is_ok(status) && !admitted && pool) {
    if (!loom_serve_memory_pool_is_pending(pool) &&
        !residency->cache->releasing_count) {
      break;
    }
    status = loom_serve_memory_pool_await(pool);
    if (iree_status_is_ok(status)) {
      status = loom_serve_residency_try_acquire(residency, &admitted);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (!admitted) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "model residency is waiting for capacity");
  }
  return loom_serve_residency_await(residency);
}

iree_status_t loom_serve_residency_await(loom_serve_residency_t* residency) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
      status =
          iree_status_join(status, loom_serve_weights_await(group->weights[j]));
    }
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
    const iree_time_t now = iree_time_now();
    for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
      loom_serve_residency_group_t* group = residency->entries[i].group;
      if (!--group->pin_count) {
        group->idle_since = now;
        loom_serve_residency_group_unlink(group);
        loom_serve_residency_group_append(group);
      }
    }
    loom_serve_residency_unlink(residency);
    loom_serve_residency_append(residency);
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
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire_wait(residency));
  loom_serve_residency_release(residency);
  return iree_ok_status();
}
