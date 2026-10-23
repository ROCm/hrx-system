// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/residency.h"

#include "iree/base/threading/mutex.h"

struct loom_serve_residency_cache_t {
  // Allocation policy for the cache metadata.
  iree_allocator_t host_allocator;
  // Borrowed physical budget; NULL for explicitly fixed allocations.
  loom_serve_memory_pool_t* pool;
  // Borrowed logical workspace sharing pool, or NULL without shared backing.
  loom_serve_workspace_t* workspace;
  // Serializes final output-credit returns with owner observation/detachment.
  iree_slim_mutex_t output_mutex;
  // Accepted output bounds, including independent downstream queue captures.
  uint64_t output_bytes;
  // Nonblocking owner wake after final output retirement, under output_mutex.
  loom_serve_wake_callback_t observer;
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

typedef struct loom_serve_residency_output_t {
  // One producer obligation plus one published output-view obligation.
  iree_atomic_ref_count_t ref_count;
  // Borrowed device admission owner, outliving every external consumer.
  loom_serve_residency_cache_t* cache;
  // Accepted source output bound, not allocator-reported availability.
  uint64_t bytes;
} loom_serve_residency_output_t;

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
  // Bank owner's persistent retention, independent of borrowing residencies.
  bool pinned;
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
  // Prepared peak for ordered commands and cohort-private input storage.
  uint64_t private_bytes;
  // Prepared escaping output bound for the next invocation.
  uint64_t output_bytes;
  // Producer obligation for the accepted output, independent after publication.
  loom_serve_residency_output_t* output;
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

void loom_serve_residency_cache_set_observer(
    loom_serve_residency_cache_t* cache, loom_serve_wake_callback_t observer) {
  iree_slim_mutex_lock(&cache->output_mutex);
  cache->observer = observer;
  iree_slim_mutex_unlock(&cache->output_mutex);
}

bool loom_serve_residency_cache_is_pending(
    loom_serve_residency_cache_t* cache) {
  iree_slim_mutex_lock(&cache->output_mutex);
  const bool pending = cache->output_bytes != 0;
  iree_slim_mutex_unlock(&cache->output_mutex);
  return pending;
}

static void loom_serve_residency_output_release(
    loom_serve_residency_output_t* output) {
  if (output && iree_atomic_ref_count_dec(&output->ref_count) == 1) {
    loom_serve_residency_cache_t* cache = output->cache;
    const uint64_t bytes = output->bytes;
    iree_allocator_free(cache->host_allocator, output);
    iree_slim_mutex_lock(&cache->output_mutex);
    cache->output_bytes -= bytes;
    if (cache->observer.wake) {
      cache->observer.wake(cache->observer.user_data);
    }
    iree_slim_mutex_unlock(&cache->output_mutex);
  }
}

static void loom_serve_residency_output_retired(void* user_data,
                                                iree_hal_buffer_t* buffer) {
  loom_serve_residency_output_release(user_data);
}

iree_status_t loom_serve_residency_track_output(
    loom_serve_residency_t* residency, iree_hal_buffer_t** buffer) {
  if (!residency->output) {
    return iree_ok_status();
  }
  iree_hal_buffer_t* view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_subspan_buffer_create_with_callback(
      *buffer, iree_hal_buffer_byte_offset(*buffer),
      iree_hal_buffer_byte_length(*buffer),
      (iree_hal_buffer_release_callback_t){loom_serve_residency_output_retired,
                                           residency->output},
      residency->host_allocator, &view));
  iree_atomic_ref_count_inc(&residency->output->ref_count);
  iree_hal_buffer_release(*buffer);
  *buffer = view;
  return iree_ok_status();
}

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

void loom_serve_residency_group_set_pinned(loom_serve_residency_group_t* group,
                                           bool pinned) {
  if (group->pinned != pinned) {
    group->pinned = pinned;
    if (!pinned) {
      group->idle_since = iree_time_now();
    }
  }
}

bool loom_serve_residency_group_is_pinned(
    const loom_serve_residency_group_t* group) {
  return group->pinned || group->pin_count || group->retention_count;
}

void loom_serve_residency_set_reclaimer(
    loom_serve_residency_t* residency,
    loom_serve_residency_reclaimer_t reclaimer) {
  residency->reclaimer = reclaimer;
}

bool loom_serve_residency_cache_is_reclaiming(
    const loom_serve_residency_cache_t* cache) {
  if (cache->releasing_count ||
      (cache->workspace &&
       loom_serve_workspace_is_reclaiming(cache->workspace)) ||
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

loom_serve_memory_usage_t loom_serve_residency_group_statistics(
    const loom_serve_residency_group_t* group) {
  loom_serve_memory_usage_t total = {0};
  if (!group) {
    return total;
  }
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_serve_memory_statistics_t domain =
        loom_serve_weights_statistics(group->weights[i]);
    total.reserved_bytes += domain.reserved_bytes;
    total.committed_bytes += domain.committed_bytes;
    total.pending_bytes += domain.pending_bytes;
    total.releasing_bytes += domain.releasing_bytes;
    total.released_bytes += domain.released_bytes;
  }
  return total;
}

loom_serve_weight_activity_t loom_serve_residency_group_activity(
    const loom_serve_residency_group_t* group) {
  loom_serve_weight_activity_t total = {0};
  if (!group) {
    return total;
  }
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_serve_weight_activity_t domain =
        loom_serve_weights_activity(group->weights[i]);
    total.activations += domain.activations;
    total.deactivations += domain.deactivations;
    total.read_bytes += domain.read_bytes;
    total.prepared_bytes += domain.prepared_bytes;
    total.groups += domain.groups;
  }
  return total;
}

loom_serve_memory_usage_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency) {
  loom_serve_memory_usage_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_memory_usage_t group =
        loom_serve_residency_group_statistics(residency->entries[i].group);
    total.reserved_bytes += group.reserved_bytes;
    total.committed_bytes += group.committed_bytes;
    total.pending_bytes += group.pending_bytes;
    total.releasing_bytes += group.releasing_bytes;
    total.released_bytes += group.released_bytes;
  }
  return total;
}

loom_serve_weight_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency) {
  loom_serve_weight_activity_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_weight_activity_t group =
        loom_serve_residency_group_activity(residency->entries[i].group);
    total.activations += group.activations;
    total.deactivations += group.deactivations;
    total.read_bytes += group.read_bytes;
    total.prepared_bytes += group.prepared_bytes;
    total.groups += group.groups;
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

iree_status_t loom_serve_residency_group_deactivate(
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
    pinned |= !group->releasing && loom_serve_residency_group_is_pinned(group);
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
  return memory.committed_bytes - memory.releasing_bytes - memory.cached_bytes;
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
    if (!entry->active || !entry->weight_count ||
        loom_serve_residency_group_is_pinned(entry)) {
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
    if (entry->active && !loom_serve_residency_group_is_pinned(entry) &&
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
  if (cache->workspace) {
    iree_device_size_t retained = 0;
    IREE_RETURN_IF_ERROR(
        loom_serve_workspace_trim(cache->workspace, &retained));
    if (loom_serve_workspace_is_reclaiming(cache->workspace)) {
      // Keep model data while idle workspace is returning. Cold acquisitions
      // progress outside this reclaimer, avoiding recursive physical admission.
      return iree_ok_status();
    }
  }
  uint64_t remaining = loom_serve_residency_cache_projected_bytes(cache);
  uint64_t pinned_bytes = 0;
  for (loom_serve_residency_group_t* entry = cache->groups.oldest;
       entry && remaining > target_bytes; entry = entry->next) {
    const loom_serve_memory_usage_t memory =
        loom_serve_residency_group_statistics(entry);
    const uint64_t committed_bytes =
        memory.committed_bytes - memory.releasing_bytes;
    if (loom_serve_residency_group_is_pinned(entry) || !entry->active) {
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

static uint64_t loom_serve_residency_workspace_footprint(
    const loom_serve_residency_cache_t* cache, uint64_t byte_length) {
  return cache->workspace
             ? loom_serve_workspace_round_size(cache->workspace, byte_length)
             : byte_length;
}

static uint64_t loom_serve_residency_activation_workspace(
    const loom_serve_residency_t* residency) {
  uint64_t bytes = 0;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
      bytes += loom_serve_weights_activation_workspace(group->weights[j])
                   .allocation_size;
    }
  }
  return bytes;
}

iree_status_t loom_serve_residency_cache_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_workspace_t* workspace,
    iree_allocator_t host_allocator, loom_serve_residency_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_residency_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  cache->pool = pool;
  cache->workspace = workspace;
  iree_slim_mutex_initialize(&cache->output_mutex);
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
  iree_slim_mutex_lock(&cache->output_mutex);
  IREE_ASSERT(!cache->output_bytes);
  iree_slim_mutex_unlock(&cache->output_mutex);
  iree_slim_mutex_deinitialize(&cache->output_mutex);
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
    const uint64_t staging_capacity = remaining;
    // Independent weight plans stream concurrently. Their input lanes overlap
    // all final parameter roots even though consumers can reuse those lanes'
    // storage after importing activation readiness.
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      const loom_serve_residency_group_t* group = groups[i];
      for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
        const uint64_t required =
            loom_serve_weights_activation_workspace(group->weights[j])
                .allocation_size;
        if (required > remaining) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "model parameters and activation staging exceed the serving "
              "memory limit of %" PRIu64 " bytes",
              loom_serve_memory_pool_limit(cache->pool));
        }
        remaining -= required;
      }
    }
    if (loom_serve_residency_workspace_footprint(
            cache, staging_capacity - remaining) > staging_capacity) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "model activation staging slabs exceed the serving memory limit");
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
  residency->private_bytes =
      loom_serve_residency_activation_workspace(residency);
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

uint64_t loom_serve_residency_activation_footprint(
    const loom_serve_residency_t* residency) {
  uint64_t bytes = 0;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    bytes += loom_serve_residency_group_statistics(group).reserved_bytes;
  }
  return bytes + loom_serve_residency_workspace_footprint(
                     residency->cache,
                     loom_serve_residency_activation_workspace(residency));
}

iree_status_t loom_serve_residency_check_working_set(
    const loom_serve_residency_t* residency, iree_host_size_t extent_count,
    const iree_device_size_t* extents) {
  const uint64_t limit =
      residency->cache->pool
          ? loom_serve_memory_pool_limit(residency->cache->pool)
          : UINT64_MAX;
  uint64_t remaining = limit;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const uint64_t required =
        loom_serve_residency_group_statistics(residency->entries[i].group)
            .reserved_bytes;
    // Registration established that the distinct parameter groups fit alone.
    remaining -= required;
  }
  const uint64_t workspace_capacity = remaining;
  for (iree_host_size_t i = 0; i < extent_count; ++i) {
    if (extents[i] > remaining) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "model parameters and invocation working set exceed the serving "
          "memory limit of %" PRIu64 " bytes",
          limit);
    }
    remaining -= extents[i];
  }
  if (loom_serve_residency_workspace_footprint(residency->cache,
                                               workspace_capacity - remaining) >
      workspace_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "model invocation workspace slabs exceed the serving memory limit of "
        "%" PRIu64 " bytes",
        limit);
  }
  return iree_ok_status();
}

static uint64_t loom_serve_residency_private_peak(
    const loom_serve_residency_t* residency, uint64_t private_bytes,
    uint64_t output_bytes) {
  // Activation staging and execution are ordered on this invocation's
  // timeline. Reserve their maximum, retaining output credit independently
  // after producer completion. Shared-group borrowers conservatively reserve
  // their own activation peak until their invocation releases its final pin.
  const uint64_t activation =
      loom_serve_residency_activation_workspace(residency);
  return iree_max(private_bytes,
                  activation > output_bytes ? activation - output_bytes : 0);
}

iree_status_t loom_serve_residency_prepare_working_set(
    loom_serve_residency_t* residency, uint64_t private_bytes,
    uint64_t output_bytes) {
  private_bytes =
      loom_serve_residency_private_peak(residency, private_bytes, output_bytes);
  const iree_device_size_t extents[] = {private_bytes, output_bytes};
  IREE_RETURN_IF_ERROR(loom_serve_residency_check_working_set(
      residency, IREE_ARRAYSIZE(extents), extents));
  IREE_ASSERT(!residency->pin_count);
  residency->private_bytes = private_bytes;
  residency->output_bytes = output_bytes;
  return iree_ok_status();
}

static bool loom_serve_residency_contains_group(
    const loom_serve_residency_t* residency,
    const loom_serve_residency_group_t* group) {
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    if (residency->entries[i].group == group) {
      return true;
    }
  }
  return false;
}

// Forecast against service-owned requirements and live reservation accounting.
// Unpinned weight groups are reclaimable; shared pinned groups count once.
// Native allocator availability and idle slab caches are not scheduling inputs.
static bool loom_serve_residency_admit_working_set(
    const loom_serve_residency_t* residency, uint64_t additional_state,
    uint64_t private_bytes, bool reclaim_history) {
  loom_serve_residency_cache_t* cache = residency->cache;
  if (!cache->pool || loom_serve_memory_pool_limit(cache->pool) == UINT64_MAX) {
    return true;
  }
  uint64_t remaining = loom_serve_memory_pool_limit(cache->pool);
  uint64_t parameter_bytes = 0;
  for (const loom_serve_residency_group_t* group = cache->groups.oldest; group;
       group = group->next) {
    const bool needed = loom_serve_residency_group_is_pinned(group) ||
                        loom_serve_residency_contains_group(residency, group);
    for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
      const loom_serve_memory_usage_t memory =
          loom_serve_weights_memory_usage(group->weights[i]);
      parameter_bytes += memory.committed_bytes + memory.pending_bytes;
      if (needed) {
        if (memory.reserved_bytes > remaining) {
          return false;
        }
        remaining -= memory.reserved_bytes;
      }
    }
  }
  // No observation above collects native operations: every counter belongs to
  // the same serialized owner snapshot, including pending range commitments.
  // Workspace backing already fulfills the private/output bounds below; it
  // must not be charged a second time as model-owned mutable state.
  const loom_serve_memory_usage_t workspace =
      cache->workspace ? loom_serve_workspace_memory_usage(cache->workspace)
                       : (loom_serve_memory_usage_t){0};
  const uint64_t mutable_bytes =
      loom_serve_memory_pool_live_bytes(cache->pool) - parameter_bytes -
      workspace.committed_bytes - workspace.pending_bytes;
  const uint64_t workspace_capacity = remaining;
  iree_slim_mutex_lock(&cache->output_mutex);
  const uint64_t output_bytes = cache->output_bytes;
  iree_slim_mutex_unlock(&cache->output_mutex);
  if (output_bytes > remaining) {
    return false;
  }
  remaining -= output_bytes;
  for (const loom_serve_residency_t* entry = cache->oldest; entry;
       entry = entry->next) {
    if (entry == residency || entry->pin_count) {
      const uint64_t required =
          entry == residency ? private_bytes : entry->private_bytes;
      if (required > remaining) {
        return false;
      }
      remaining -= required;
    }
  }
  const uint64_t output = residency->pin_count ? 0 : residency->output_bytes;
  if (output > remaining) {
    return false;
  }
  remaining -= output;
  // Logical peers can pack into the same slab; round their aggregate once.
  // Even a small cold request needs a complete slab before its queue can run.
  const uint64_t workspace_bytes = loom_serve_residency_workspace_footprint(
      cache, workspace_capacity - remaining);
  if (workspace_bytes > workspace_capacity ||
      additional_state > workspace_capacity - workspace_bytes) {
    return false;
  }
  remaining = workspace_capacity - workspace_bytes - additional_state;
  if (mutable_bytes <= remaining) {
    return true;
  }
  // Protected work fits; idle mutable history is the only remaining pressure.
  // Ask one eligible owner to retire its history, preserving every accepted
  // consumer and borrowed preparation. Native retirement wakes the owner;
  // this admission neither waits nor probes allocator space.
  if (reclaim_history && !loom_serve_residency_cache_is_reclaiming(cache)) {
    for (loom_serve_residency_t* entry = cache->oldest; entry;
         entry = entry->next) {
      if (!entry->pin_count && entry->reclaimer.request &&
          entry->reclaimer.request(entry->reclaimer.self)) {
        break;
      }
    }
  }
  return false;
}

iree_status_t loom_serve_residency_check_private_peak(
    const loom_serve_residency_t* residency, uint64_t private_bytes) {
  private_bytes = loom_serve_residency_private_peak(residency, private_bytes,
                                                    residency->output_bytes);
  const iree_device_size_t extents[] = {private_bytes, residency->output_bytes};
  IREE_RETURN_IF_ERROR(loom_serve_residency_check_working_set(
      residency, IREE_ARRAYSIZE(extents), extents));
  if (residency->pin_count && private_bytes > residency->private_bytes &&
      !loom_serve_residency_admit_working_set(residency, 0, private_bytes,
                                              false)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "replacement command workspace exceeds available serving capacity");
  }
  return iree_ok_status();
}

void loom_serve_residency_set_private_peak(loom_serve_residency_t* residency,
                                           uint64_t private_bytes) {
  residency->private_bytes = loom_serve_residency_private_peak(
      residency, private_bytes, residency->output_bytes);
}

// Borrowed for one serialized admission, before either state or parameters
// submit work. The state owner retains the resulting native operation.
typedef struct loom_serve_residency_state_t {
  // Exact mutable-state ranges produced by the caller's logical planner.
  const loom_serve_memory_range_t* ranges;
  // Number of ranges, including zero for fixed backing.
  iree_host_size_t range_count;
  // State-owned mapping timeline, or NULL for fixed backing.
  iree_hal_semaphore_t* readiness;
  // State readiness value to publish after successful backing.
  uint64_t value;
  // Caller-owned operation slot, preserved on terminal activation failure.
  loom_serve_memory_operation_t** operation;
} loom_serve_residency_state_t;

static iree_status_t loom_serve_residency_commit_state(
    loom_serve_residency_t* residency,
    const loom_serve_residency_state_t* state,
    loom_serve_memory_demand_t immediate,
    loom_serve_memory_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  if (!residency->cache->pool) {
    return iree_ok_status();
  }
  return loom_serve_memory_pool_commit_with_demand_async(
      residency->cache->pool, state->range_count, state->ranges, immediate,
      state->readiness, state->value, out_admission, state->operation);
}

static iree_status_t loom_serve_residency_acquire_parameters(
    loom_serve_residency_t* residency,
    const loom_serve_residency_state_t* state,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_cache_query_status(residency->cache));
  const uint64_t additional_state = state
                                        ? loom_serve_memory_ranges_footprint(
                                              state->range_count, state->ranges)
                                              .missing_bytes
                                        : 0;
  if (!loom_serve_residency_admit_working_set(residency, additional_state,
                                              residency->private_bytes, true)) {
    return iree_ok_status();
  }
  // Nested execution shares parameter protection. A new row still admits its
  // complete state growth before borrowing another pin.
  if (residency->pin_count && residency->active) {
    if (state) {
      loom_serve_memory_admission_t admission;
      IREE_RETURN_IF_ERROR(loom_serve_residency_commit_state(
          residency, state, (loom_serve_memory_demand_t){0}, &admission));
      if (admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
        if (admission == LOOM_SERVE_MEMORY_ADMISSION_DEFERRED) {
          *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
        }
        return iree_ok_status();
      }
    }
    ++residency->pin_count;
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
    return iree_ok_status();
  }
  bool active = true;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    if (residency->entries[i].group->releasing) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      return iree_ok_status();
    }
    active &= residency->entries[i].group->active;
  }
  residency->active = false;
  if (residency->output_bytes && residency->cache->pool &&
      loom_serve_memory_pool_limit(residency->cache->pool) != UINT64_MAX) {
    loom_serve_residency_output_t* output = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        residency->cache->host_allocator, sizeof(*output), (void**)&output));
    iree_atomic_ref_count_init(&output->ref_count);
    output->cache = residency->cache;
    output->bytes = residency->output_bytes;
    iree_slim_mutex_lock(&output->cache->output_mutex);
    output->cache->output_bytes += output->bytes;
    iree_slim_mutex_unlock(&output->cache->output_mutex);
    residency->output = output;
  }
  ++residency->pin_count;
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    ++residency->entries[i].group->pin_count;
  }
  if ((!active || state) && residency->cache->pool) {
    loom_serve_memory_demand_t demand = {0};
    for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
      const loom_serve_residency_group_t* group = residency->entries[i].group;
      if (!group->active) {
        for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
          loom_serve_memory_demand_add(
              &demand, loom_serve_weights_missing_demand(group->weights[j]));
        }
      }
    }
    loom_serve_memory_admission_t admission;
    iree_status_t status =
        state ? loom_serve_residency_commit_state(residency, state, demand,
                                                  &admission)
              : loom_serve_memory_pool_prepare(residency->cache->pool, demand,
                                               &admission);
    if (!iree_status_is_ok(status) ||
        admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
      // No parameter activation was accepted. Undo this attempt's pin even
      // when a native state failure leaves its caller-owned operation to join.
      --residency->pin_count;
      loom_serve_residency_output_release(residency->output);
      residency->output = NULL;
      for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
        --residency->entries[i].group->pin_count;
      }
      if (admission == LOOM_SERVE_MEMORY_ADMISSION_DEFERRED) {
        *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      }
      return status;
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
      loom_serve_residency_acquire_parameters(residency, NULL, &admission);
  *out_admitted = admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  return status;
}

static iree_status_t loom_serve_residency_admit(
    loom_serve_residency_t* residency,
    const loom_serve_residency_state_t* state,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  loom_serve_residency_cache_t* cache = residency->cache;
  // Mutable-state growth may need capacity even with warm parameters. Collect
  // completed peer releases before admission so backpressure cannot wait on
  // logical interest whose native completion wake already happened.
  IREE_RETURN_IF_ERROR(loom_serve_residency_cache_query_status(cache));
  if (cache->admission.first && cache->admission.first != residency &&
      loom_serve_residency_statistics(residency).reserved_bytes) {
    loom_serve_residency_queue_admission(residency);
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_acquire_parameters(residency, state, out_admission));
  if (*out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_residency_queue_admission(residency);
    if (loom_serve_residency_cache_is_reclaiming(cache)) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_residency_try_admit(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission) {
  return loom_serve_residency_admit(residency, NULL, out_admission);
}

iree_status_t loom_serve_residency_try_admit_state(
    loom_serve_residency_t* residency, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_residency_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation) {
  *out_operation = NULL;
  const loom_serve_residency_state_t state = {
      .ranges = ranges,
      .range_count = range_count,
      .readiness = readiness,
      .value = value,
      .operation = out_operation,
  };
  return loom_serve_residency_admit(residency, &state, out_admission);
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
    loom_serve_residency_output_release(residency->output);
    residency->output = NULL;
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
