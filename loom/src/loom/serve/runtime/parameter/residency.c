// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/residency.h"

#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/storage/virtual_pool.h"

typedef struct loom_serve_residency_entry_t loom_serve_residency_entry_t;

struct loom_serve_residency_cache_t {
  // Allocation policy for the cache metadata.
  iree_allocator_t host_allocator;
  // Borrowed physical budget; NULL for explicitly fixed allocations.
  loom_serve_memory_pool_t* pool;
  // Borrowed logical workspace sharing pool, or NULL without shared backing.
  loom_serve_workspace_t* workspace;
  // Borrowed permanent virtual source, distinct from reclaimable model history.
  // NULL when permanent allocations do not share this physical budget.
  iree_hal_pool_t* permanent_source;
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
  // Accepted staging deallocations whose credit still awaits completion.
  loom_serve_residency_group_t* activating;
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
  // Cold or accepted-but-uncompleted captured-input staging requirement.
  uint64_t activation_bytes;
  // Working-set memberships whose aggregate staging bound includes this group.
  loom_serve_residency_entry_t* borrowers;
  // Previous pending-list link, or NULL without accepted staging work.
  loom_serve_residency_group_t** activation_previous;
  // Next group with accepted staging awaiting owner collection.
  loom_serve_residency_group_t* activation_next;
  // Plans whose backing is activated and evicted together.
  loom_serve_weights_t* weights[];
};

struct loom_serve_residency_entry_t {
  // Borrowed immutable resource group, outliving this working set.
  loom_serve_residency_group_t* group;
  // Group generation whose readiness was imported into this execution.
  uint64_t generation;
  // Borrowing invocation whose aggregate changes with group activation.
  loom_serve_residency_t* residency;
  // Previous borrower-list link, including the group's head.
  loom_serve_residency_entry_t** previous;
  // Next working set borrowing the same immutable group.
  loom_serve_residency_entry_t* next;
};

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
  // Sum of current cold/in-flight staging requirements across borrowed groups.
  uint64_t activation_bytes;
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
  if (cache->activating || cache->releasing_count ||
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

loom_runtime_storage_usage_t loom_serve_residency_group_statistics(
    const loom_serve_residency_group_t* group) {
  loom_runtime_storage_usage_t total = {0};
  if (!group) {
    return total;
  }
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_runtime_storage_statistics_t domain =
        loom_serve_weights_statistics(group->weights[i]);
    total.reserved_bytes += domain.reserved_bytes;
    total.committed_bytes += domain.committed_bytes;
    total.pending_bytes += domain.pending_bytes;
    total.releasing_bytes += domain.releasing_bytes;
    total.released_bytes += domain.released_bytes;
  }
  return total;
}

loom_runtime_parameter_activity_t loom_serve_residency_group_activity(
    const loom_serve_residency_group_t* group) {
  loom_runtime_parameter_activity_t total = {0};
  if (!group) {
    return total;
  }
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_runtime_parameter_activity_t domain =
        loom_serve_weights_activity(group->weights[i]);
    total.activations += domain.activations;
    total.deactivations += domain.deactivations;
    total.read_bytes += domain.read_bytes;
    total.prepared_bytes += domain.prepared_bytes;
    total.groups += domain.groups;
  }
  return total;
}

loom_runtime_storage_usage_t loom_serve_residency_statistics(
    const loom_serve_residency_t* residency) {
  loom_runtime_storage_usage_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_runtime_storage_usage_t group =
        loom_serve_residency_group_statistics(residency->entries[i].group);
    total.reserved_bytes += group.reserved_bytes;
    total.committed_bytes += group.committed_bytes;
    total.pending_bytes += group.pending_bytes;
    total.releasing_bytes += group.releasing_bytes;
    total.released_bytes += group.released_bytes;
  }
  return total;
}

loom_runtime_parameter_activity_t loom_serve_residency_activity(
    const loom_serve_residency_t* residency) {
  loom_runtime_parameter_activity_t total = {0};
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_runtime_parameter_activity_t group =
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

static void loom_serve_residency_group_set_activation_bytes(
    loom_serve_residency_group_t* group, uint64_t bytes) {
  for (loom_serve_residency_entry_t* entry = group->borrowers; entry;
       entry = entry->next) {
    entry->residency->activation_bytes -= group->activation_bytes;
    entry->residency->activation_bytes += bytes;
  }
  group->activation_bytes = bytes;
}

static void loom_serve_residency_group_unlink_activation(
    loom_serve_residency_group_t* group) {
  if (group->activation_previous) {
    *group->activation_previous = group->activation_next;
    if (group->activation_next) {
      group->activation_next->activation_previous = group->activation_previous;
    }
    group->activation_previous = NULL;
    group->activation_next = NULL;
  }
}

iree_status_t loom_serve_residency_group_deactivate(
    loom_serve_residency_group_t* group, bool* out_complete) {
  *out_complete = false;
  if (group->active) {
    uint64_t bytes = 0;
    for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
      bytes += loom_serve_weights_activation_workspace(group->weights[i])
                   .allocation_size;
    }
    loom_serve_residency_group_set_activation_bytes(group, bytes);
    loom_serve_residency_group_unlink_activation(group);
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

iree_status_t loom_serve_residency_deactivate(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, bool* out_complete) {
  *out_complete = false;
  bool deactivatable = true;
  for (iree_host_size_t i = 0; i < residency_count; ++i) {
    const loom_serve_residency_t* residency = residencies[i];
    deactivatable &= !residency->pin_count && !residency->pinned &&
                     residency->cache->pool != NULL;
    for (iree_host_size_t j = 0; j < residency->group_count; ++j) {
      const loom_serve_residency_group_t* group = residency->entries[j].group;
      deactivatable &=
          group->releasing || !loom_serve_residency_group_is_pinned(group);
    }
  }
  if (!deactivatable) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "deactivation requires unpinned elastic residency");
  }
  iree_status_t status = iree_ok_status();
  bool complete = true;
  for (iree_host_size_t i = 0; i < residency_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_residency_t* residency = residencies[i];
    for (iree_host_size_t j = 0;
         j < residency->group_count && iree_status_is_ok(status); ++j) {
      bool group_complete = false;
      status = loom_serve_residency_group_deactivate(
          residency->entries[j].group, &group_complete);
      complete &= group_complete;
    }
    residency->active = false;
  }
  *out_complete = iree_status_is_ok(status) && complete;
  return status;
}

iree_status_t loom_serve_residency_cache_query_status(
    loom_serve_residency_cache_t* cache) {
  for (loom_serve_residency_group_t* group = cache->activating; group;) {
    loom_serve_residency_group_t* next = group->activation_next;
    uint64_t bytes = 0;
    bool pending = false;
    for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
      uint64_t weight_bytes = 0;
      pending |= loom_serve_weights_query_activation_workspace(
          group->weights[i], &weight_bytes);
      bytes += weight_bytes;
    }
    if (bytes != group->activation_bytes) {
      loom_serve_residency_group_set_activation_bytes(group, bytes);
    }
    if (!pending) {
      loom_serve_residency_group_unlink_activation(group);
    }
    group = next;
  }
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

static uint64_t loom_serve_residency_group_live_bytes(
    const loom_serve_residency_group_t* group) {
  uint64_t bytes = 0;
  for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
    const loom_runtime_storage_usage_t memory =
        loom_serve_weights_memory_usage(group->weights[i]);
    bytes +=
        memory.committed_bytes + memory.pending_bytes - memory.releasing_bytes;
  }
  return bytes;
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
       entry && iree_status_is_ok(status); entry = entry->next) {
    const loom_runtime_storage_statistics_t memory =
        loom_serve_memory_pool_statistics(cache->pool);
    const uint64_t remaining = memory.committed_bytes + memory.pending_bytes -
                               memory.releasing_bytes - memory.cached_bytes;
    if (remaining <= target_bytes) {
      break;
    }
    if (entry->active && !loom_serve_residency_group_is_pinned(entry) &&
        !entry->releasing && loom_serve_residency_group_live_bytes(entry)) {
      bool complete = false;
      status = loom_serve_residency_group_deactivate(entry, &complete);
    }
  }
  return status;
}

static uint64_t loom_serve_residency_mutable_bytes(
    const loom_serve_residency_cache_t* cache, uint64_t parameter_bytes) {
  const loom_runtime_storage_usage_t workspace =
      cache->workspace ? loom_serve_workspace_memory_usage(cache->workspace)
                       : (loom_runtime_storage_usage_t){0};
  const loom_runtime_storage_usage_t permanent =
      cache->permanent_source
          ? loom_serve_virtual_pool_source_memory_usage(cache->permanent_source)
          : (loom_runtime_storage_usage_t){0};
  return loom_serve_memory_pool_live_bytes(cache->pool) - parameter_bytes -
         workspace.committed_bytes - workspace.pending_bytes -
         permanent.committed_bytes - permanent.pending_bytes;
}

// Replace each known owner's actual charge with its necessary state bound.
// Unknown owners remain fully protected; no completion collection may split
// this ledger snapshot. Request demand is included by its producer query.
static iree_status_t loom_serve_residency_project_state(
    const loom_serve_residency_cache_t* cache,
    const loom_serve_residency_t* requester,
    const loom_serve_residency_forecast_t* request, uint64_t resident_bytes,
    uint64_t* out_minimum_bytes) {
  uint64_t minimum_bytes = 0;
  iree_status_t status = iree_ok_status();
  for (loom_serve_residency_t* entry = cache->oldest;
       entry && iree_status_is_ok(status); entry = entry->next) {
    const loom_serve_residency_forecast_t* incoming =
        entry == requester ? request : NULL;
    loom_serve_residency_projection_t projection = {0};
    if (entry->reclaimer.forecast) {
      status = entry->reclaimer.forecast(entry->reclaimer.self, incoming,
                                         &projection);
    } else if (incoming) {
      status = incoming->query(incoming->self, &projection);
    }
    if (iree_status_is_ok(status)) {
      resident_bytes -= projection.resident_bytes;
      minimum_bytes +=
          iree_min(projection.minimum_bytes, UINT64_MAX - minimum_bytes);
    }
  }
  *out_minimum_bytes =
      minimum_bytes + iree_min(resident_bytes, UINT64_MAX - minimum_bytes);
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
  // Readiness may collect completions. Classify the resulting owner ledger
  // without further collection: a failed acquisition can lower both aggregate
  // and group charges, and mixing those epochs invents reclaimable history.
  const bool reclaiming = loom_serve_residency_cache_is_reclaiming(cache);
  const loom_runtime_storage_statistics_t memory =
      loom_serve_memory_pool_snapshot(cache->pool);
  uint64_t remaining = memory.committed_bytes + memory.pending_bytes -
                       memory.releasing_bytes + memory.abandoned_bytes;
  uint64_t pinned_bytes = memory.abandoned_bytes;
  if (cache->permanent_source) {
    pinned_bytes +=
        loom_serve_virtual_pool_source_protected_bytes(cache->permanent_source);
  }
  if (cache->workspace) {
    pinned_bytes += loom_serve_workspace_protected_bytes(cache->workspace);
  }
  for (loom_serve_residency_group_t* entry = cache->groups.oldest;
       entry && remaining > target_bytes; entry = entry->next) {
    const uint64_t live_bytes = loom_serve_residency_group_live_bytes(entry);
    if (loom_serve_residency_group_is_pinned(entry) || !entry->active) {
      pinned_bytes += live_bytes;
    } else {
      remaining -= live_bytes;
    }
  }
  if (remaining <= target_bytes) {
    return loom_serve_residency_cache_trim(cache, target_bytes);
  }
  // Live scratch, permanent roots, pinned parameters, partial loads, and lost
  // native backing cannot be evicted. No idle history can help while that
  // backing exceeds the target.
  if (pinned_bytes > target_bytes) {
    return iree_ok_status();
  }
  // Keep useful weights when their release alone cannot admit the request.
  // Release at most one idle logical owner, then let its normal asynchronous
  // maintenance retire before asking for more. A physical trim here could
  // recurse into admission for compaction destinations.
  if (!reclaiming) {
    uint64_t parameter_bytes = 0;
    for (loom_serve_residency_group_t* entry = cache->groups.oldest; entry;
         entry = entry->next) {
      const loom_runtime_storage_usage_t usage =
          loom_serve_residency_group_statistics(entry);
      parameter_bytes += usage.committed_bytes + usage.pending_bytes;
    }
    uint64_t state_bytes = 0;
    IREE_RETURN_IF_ERROR(loom_serve_residency_project_state(
        cache, NULL, NULL,
        loom_serve_residency_mutable_bytes(cache, parameter_bytes),
        &state_bytes));
    if (state_bytes > target_bytes - pinned_bytes) {
      return iree_ok_status();
    }
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
    iree_hal_pool_t* permanent_source, iree_allocator_t host_allocator,
    loom_serve_residency_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_residency_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  cache->pool = pool;
  cache->workspace = workspace;
  cache->permanent_source = permanent_source;
  iree_slim_mutex_initialize(&cache->output_mutex);
  if (pool) {
    loom_serve_memory_pool_set_reclaimer(
        pool,
        (loom_serve_memory_reclaimer_t){cache, loom_serve_residency_reclaim});
  }
  *out_cache = cache;
  return iree_ok_status();
}

bool loom_serve_residency_cache_is_empty(loom_serve_residency_cache_t* cache) {
  return !cache || (!cache->oldest && !cache->groups.oldest &&
                    !loom_serve_residency_cache_is_pending(cache));
}

void loom_serve_residency_cache_free(loom_serve_residency_cache_t* cache) {
  if (!cache) {
    return;
  }
  IREE_ASSERT(loom_serve_residency_cache_is_empty(cache));
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
  loom_serve_residency_group_unlink_activation(group);
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
    loom_serve_residency_entry_t* entry = &residency->entries[i];
    loom_serve_residency_group_t* group = groups[i];
    // Banks register groups before source preparation. The first invocation
    // binds only after preparation has established the cold lane geometry.
    if (!group->borrowers && !group->active) {
      group->activation_bytes = 0;
      for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
        group->activation_bytes +=
            loom_serve_weights_activation_workspace(group->weights[j])
                .allocation_size;
      }
    }
    entry->group = group;
    entry->residency = residency;
    entry->previous = &group->borrowers;
    entry->next = group->borrowers;
    if (entry->next) {
      entry->next->previous = &entry->next;
    }
    group->borrowers = entry;
    residency->activation_bytes += group->activation_bytes;
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
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    loom_serve_residency_entry_t* entry = &residency->entries[i];
    *entry->previous = entry->next;
    if (entry->next) {
      entry->next->previous = entry->previous;
    }
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

static uint64_t loom_serve_residency_private_peak(uint64_t activation_bytes,
                                                  uint64_t private_bytes,
                                                  uint64_t output_bytes) {
  // Activation staging and execution are ordered on this invocation's
  // timeline. Reserve their maximum, retaining output credit independently
  // after producer completion. Shared-group borrowers conservatively reserve
  // their own activation peak until the group's staging deallocation completes.
  return iree_max(private_bytes, activation_bytes > output_bytes
                                     ? activation_bytes - output_bytes
                                     : 0);
}

iree_status_t loom_serve_residency_prepare_working_set(
    loom_serve_residency_t* residency, uint64_t private_bytes,
    uint64_t output_bytes) {
  const iree_device_size_t extents[] = {
      loom_serve_residency_private_peak(
          loom_serve_residency_activation_workspace(residency), private_bytes,
          output_bytes),
      output_bytes};
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
static bool loom_serve_residency_state_budget(
    const loom_serve_residency_t* residency, uint64_t private_bytes,
    uint64_t* out_resident_bytes, uint64_t* out_capacity) {
  loom_serve_residency_cache_t* cache = residency->cache;
  *out_resident_bytes = 0;
  *out_capacity = UINT64_MAX;
  if (!cache->pool) {
    return true;
  }
  uint64_t remaining = loom_serve_memory_pool_limit(cache->pool);
  if (cache->permanent_source) {
    const uint64_t protected_bytes =
        loom_serve_virtual_pool_source_protected_bytes(cache->permanent_source);
    if (protected_bytes > remaining) {
      return false;
    }
    remaining -= protected_bytes;
  }
  uint64_t parameter_bytes = 0;
  for (const loom_serve_residency_group_t* group = cache->groups.oldest; group;
       group = group->next) {
    const bool needed = loom_serve_residency_group_is_pinned(group) ||
                        loom_serve_residency_contains_group(residency, group);
    for (iree_host_size_t i = 0; i < group->weight_count; ++i) {
      const loom_runtime_storage_usage_t memory =
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
  *out_resident_bytes =
      loom_serve_residency_mutable_bytes(cache, parameter_bytes);
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
      const uint64_t required = loom_serve_residency_private_peak(
          entry->activation_bytes,
          entry == residency ? private_bytes : entry->private_bytes,
          entry->output_bytes);
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
  if (workspace_bytes > workspace_capacity) {
    return false;
  }
  *out_capacity = workspace_capacity - workspace_bytes;
  return true;
}

static bool loom_serve_residency_fits(const loom_serve_residency_t* residency,
                                      uint64_t additional_state,
                                      uint64_t private_bytes) {
  uint64_t resident_bytes = 0, capacity = 0;
  if (!loom_serve_residency_state_budget(residency, private_bytes,
                                         &resident_bytes, &capacity) ||
      additional_state > capacity) {
    return false;
  }
  return resident_bytes <= capacity - additional_state;
}

iree_status_t loom_serve_residency_check_private_peak(
    const loom_serve_residency_t* residency, uint64_t private_bytes) {
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_cache_query_status(residency->cache));
  const iree_device_size_t extents[] = {
      loom_serve_residency_private_peak(
          loom_serve_residency_activation_workspace(residency), private_bytes,
          residency->output_bytes),
      residency->output_bytes};
  IREE_RETURN_IF_ERROR(loom_serve_residency_check_working_set(
      residency, IREE_ARRAYSIZE(extents), extents));
  if (residency->pin_count && private_bytes > residency->private_bytes &&
      !loom_serve_residency_fits(residency, 0, private_bytes)) {
    // Completed failures may still be charged. Refresh only under pressure;
    // replacing workspace never requests logical history eviction.
    IREE_RETURN_IF_ERROR(
        loom_serve_memory_pool_query_status(residency->cache->pool));
    if (!loom_serve_residency_fits(residency, 0, private_bytes)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "replacement command workspace exceeds available serving capacity");
    }
  }
  return iree_ok_status();
}

void loom_serve_residency_set_private_peak(loom_serve_residency_t* residency,
                                           uint64_t private_bytes) {
  residency->private_bytes = private_bytes;
}

// Borrowed for one serialized admission, before either state or parameters
// submit work. The state owner retains the resulting native operation.
typedef struct loom_serve_residency_state_t {
  // Exact mutable-state ranges produced by the caller's logical planner.
  const loom_serve_memory_range_t* ranges;
  // Number of ranges, including zero for fixed backing.
  iree_host_size_t range_count;
  // Optional producer projection including legal re-placement after denial.
  const loom_serve_residency_forecast_t* forecast;
  // State-owned mapping timeline, or NULL for fixed backing.
  iree_hal_semaphore_t* readiness;
  // State readiness value to publish after successful backing.
  uint64_t value;
  // Caller-owned operation slot, preserved on terminal activation failure.
  loom_serve_memory_operation_t** operation;
} loom_serve_residency_state_t;

// A cold necessary bound, not a reservation or a claim that compaction will
// succeed. Unknown mutable owners stay protected. Known owners replace only
// their own actual charge with source-derived retained/incoming requirements.
static iree_status_t loom_serve_residency_can_reclaim(
    loom_serve_residency_t* residency,
    const loom_serve_residency_state_t* state, uint64_t additional_state,
    bool* out_possible) {
  *out_possible = false;
  uint64_t resident_bytes = 0, capacity = 0;
  if (!loom_serve_residency_state_budget(residency, residency->private_bytes,
                                         &resident_bytes, &capacity)) {
    return iree_ok_status();
  }
  const loom_serve_residency_forecast_t* request =
      state ? state->forecast : NULL;
  uint64_t minimum_bytes = 0;
  IREE_RETURN_IF_ERROR(loom_serve_residency_project_state(
      residency->cache, residency, request, resident_bytes, &minimum_bytes));
  if (minimum_bytes > capacity ||
      (!request && additional_state > capacity - minimum_bytes)) {
    return iree_ok_status();
  }
  *out_possible = true;
  return iree_ok_status();
}

// Every physical owner has passed its necessary bound before this action.
static void loom_serve_residency_request_history(
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission) {
  // Parameter pins protect weights, not every history of that model. Each
  // history owner excludes accepted consumers and borrowed source state.
  for (loom_serve_residency_t* entry = residency->cache->oldest; entry;
       entry = entry->next) {
    if (entry->reclaimer.request &&
        entry->reclaimer.request(entry->reclaimer.self)) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      break;
    }
  }
}

static iree_status_t loom_serve_residency_protect(
    loom_serve_residency_t* residency) {
  if (!residency->pin_count) {
    if (residency->output_bytes && residency->cache->pool) {
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
    for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
      ++residency->entries[i].group->pin_count;
    }
  }
  ++residency->pin_count;
  return iree_ok_status();
}

// Undoes an attempt before parameter activation. This is not consumption and
// must not refresh the group's eviction age or disturb an older consumer pin.
static void loom_serve_residency_unprotect(loom_serve_residency_t* residency) {
  if (!--residency->pin_count) {
    loom_serve_residency_output_release(residency->output);
    residency->output = NULL;
    for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
      --residency->entries[i].group->pin_count;
    }
  }
}

static loom_serve_memory_demand_t loom_serve_residency_missing_demand(
    const loom_serve_residency_t* residency) {
  loom_serve_memory_demand_t demand = {0};
  if (residency->pin_count > 1) {
    return demand;
  }
  for (iree_host_size_t i = 0; i < residency->group_count; ++i) {
    const loom_serve_residency_group_t* group = residency->entries[i].group;
    if (!group->active) {
      for (iree_host_size_t j = 0; j < group->weight_count; ++j) {
        loom_serve_memory_demand_add(
            &demand, loom_serve_weights_missing_demand(group->weights[j]));
      }
    }
  }
  return demand;
}

static iree_status_t loom_serve_residency_activate_parameters(
    loom_serve_residency_t* residency) {
  if (residency->pin_count > 1) {
    return iree_ok_status();
  }
  residency->active = false;
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
      if (group->active && group->activation_bytes) {
        group->activation_previous = &group->cache->activating;
        group->activation_next = group->cache->activating;
        if (group->activation_next) {
          group->activation_next->activation_previous = &group->activation_next;
        }
        group->cache->activating = group;
      }
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
  residency->active = iree_status_is_ok(status);
  return status;
}

// Fitting admission reads only owner facts. Under pressure, completed native
// ownership is collected before evaluating the entire set's retained-state
// bounds. One impossible peer must not discard another owner's useful history.
static iree_status_t loom_serve_residency_preflight(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, iree_host_size_t state_owner,
    const loom_serve_residency_state_t* state, uint64_t additional_state,
    bool* out_available, loom_serve_residency_admission_t* out_admission) {
  bool available = true;
  bool possible = true;
  iree_host_size_t history_owner = residency_count;
  for (iree_host_size_t i = 0; i < residency_count; ++i) {
    loom_serve_residency_t* residency = residencies[i];
    const uint64_t additional = i == state_owner ? additional_state : 0;
    if (!loom_serve_residency_fits(residency, additional,
                                   residency->private_bytes)) {
      loom_serve_residency_cache_t* cache = residency->cache;
      const bool reclaiming = loom_serve_residency_cache_is_reclaiming(cache);
      IREE_RETURN_IF_ERROR(loom_serve_memory_pool_query_status(cache->pool));
      if (!loom_serve_residency_fits(residency, additional,
                                     residency->private_bytes)) {
        available = false;
        if (reclaiming) {
          // Keep this observation even if its completion races the query.
          *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
        } else {
          bool owner_possible = false;
          IREE_RETURN_IF_ERROR(loom_serve_residency_can_reclaim(
              residency, i == state_owner ? state : NULL, additional,
              &owner_possible));
          possible &= owner_possible;
          if (owner_possible && history_owner == residency_count) {
            history_owner = i;
          }
        }
      }
    }
    for (iree_host_size_t j = 0;
         !residency->pin_count && j < residency->group_count; ++j) {
      if (residency->entries[j].group->releasing) {
        available = false;
        *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      }
    }
  }
  if (!available && possible && history_owner != residency_count &&
      *out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED) {
    loom_serve_residency_request_history(residencies[history_owner],
                                         out_admission);
  }
  *out_available = available;
  return iree_ok_status();
}

static iree_status_t loom_serve_residency_acquire_parameters(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, iree_host_size_t state_owner,
    const loom_serve_residency_state_t* state,
    loom_serve_residency_admission_t* out_admission) {
  loom_serve_memory_footprint_t footprint = {0};
  if (state) {
    IREE_RETURN_IF_ERROR(loom_serve_memory_ranges_footprint(
        state->range_count, state->ranges, &footprint));
  }
  bool available = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_preflight(
      residency_count, residencies, state_owner, state, footprint.missing_bytes,
      &available, out_admission));
  if (!available) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  iree_host_size_t protected_count = 0;
  while (protected_count < residency_count && iree_status_is_ok(status)) {
    status = loom_serve_residency_protect(residencies[protected_count]);
    if (iree_status_is_ok(status)) {
      ++protected_count;
    }
  }
  // Distinct pools share this serialized owner turn. A successful prepare
  // remains valid until activation; no other admission can spend its capacity.
  loom_serve_memory_admission_t admission =
      LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  for (iree_host_size_t i = 0;
       i < residency_count && iree_status_is_ok(status) &&
       admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
       ++i) {
    loom_serve_residency_t* residency = residencies[i];
    if (residency->pin_count == 1 && residency->cache->pool &&
        (!state || i != state_owner)) {
      const loom_serve_memory_demand_t demand =
          loom_serve_residency_missing_demand(residency);
      if (demand.weight_bytes || demand.packed_bytes ||
          demand.dedicated_bytes) {
        status = loom_serve_memory_pool_prepare(residency->cache->pool, demand,
                                                &admission);
      }
    }
  }
  const bool preparing_state =
      iree_status_is_ok(status) &&
      admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE && state;
  if (preparing_state && residencies[state_owner]->cache->pool) {
    loom_serve_memory_pool_t* pool = residencies[state_owner]->cache->pool;
    loom_serve_memory_demand_t demand =
        loom_serve_residency_missing_demand(residencies[state_owner]);
    loom_serve_memory_demand_add(&demand, footprint.missing_demand);
    status = loom_serve_memory_pool_prepare(pool, demand, &admission);
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
      status = loom_serve_memory_pool_commit_admitted_async(
          pool, state->range_count, state->ranges, state->readiness,
          state->value, state->operation);
    }
  }
  if (!iree_status_is_ok(status) ||
      admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
    // No weight activation has been accepted. A partially accepted native
    // state operation still belongs to its caller and must retire there.
    for (iree_host_size_t i = 0; i < protected_count; ++i) {
      loom_serve_residency_unprotect(residencies[i]);
    }
    if (admission == LOOM_SERVE_MEMORY_ADMISSION_DEFERRED) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    } else if (iree_status_is_ok(status) && preparing_state &&
               state->forecast) {
      // Exact placement can fail while source-guided relocation remains legal.
      bool possible = false;
      status = loom_serve_residency_can_reclaim(
          residencies[state_owner], state, footprint.missing_bytes, &possible);
      if (iree_status_is_ok(status) && possible) {
        loom_serve_residency_request_history(residencies[state_owner],
                                             out_admission);
      }
    }
    return status;
  }
  for (iree_host_size_t i = 0; i < residency_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_residency_activate_parameters(residencies[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  }
  return status;
}

static iree_status_t loom_serve_residency_admit(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, bool new_work,
    iree_host_size_t state_owner, const loom_serve_residency_state_t* state,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  iree_status_t status = iree_ok_status();
  // Collect every owner's completed returns before checking interest: their
  // capacity wakes may already have happened, including with warm weights.
  for (iree_host_size_t i = 0; i < residency_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_residency_t* residency = residencies[i];
    loom_serve_residency_cache_t* cache = residency->cache;
    status = loom_serve_residency_cache_query_status(cache);
    if (new_work && cache->admission.first &&
        cache->admission.first != residency &&
        loom_serve_residency_statistics(residency).reserved_bytes) {
      *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
    }
  }
  if (iree_status_is_ok(status) &&
      *out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED) {
    status = loom_serve_residency_acquire_parameters(
        residency_count, residencies, state_owner, state, out_admission);
  }
  if (iree_status_is_ok(status) && new_work &&
      *out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    // Queue the complete immutable membership in one owner turn. Overlapping
    // sets therefore cannot establish opposite orders in different caches.
    for (iree_host_size_t i = 0; i < residency_count; ++i) {
      loom_serve_residency_queue_admission(residencies[i]);
      if (loom_serve_residency_cache_is_reclaiming(residencies[i]->cache)) {
        *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
      }
    }
  }
  return status;
}

iree_status_t loom_serve_residency_try_acquire(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, bool* out_admitted) {
  loom_serve_residency_admission_t admission;
  iree_status_t status = loom_serve_residency_admit(
      residency_count, residencies, false, 0, NULL, &admission);
  *out_admitted = admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  return status;
}

iree_status_t loom_serve_residency_try_admit(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies,
    loom_serve_residency_admission_t* out_admission) {
  return loom_serve_residency_admit(residency_count, residencies, true, 0, NULL,
                                    out_admission);
}

iree_status_t loom_serve_residency_try_admit_state(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies, iree_host_size_t state_owner,
    iree_host_size_t range_count, const loom_serve_memory_range_t* ranges,
    const loom_serve_residency_forecast_t* forecast,
    iree_hal_semaphore_t* readiness, uint64_t value,
    loom_serve_residency_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation) {
  *out_operation = NULL;
  const loom_serve_residency_state_t state = {
      .ranges = ranges,
      .range_count = range_count,
      .forecast = forecast,
      .readiness = readiness,
      .value = value,
      .operation = out_operation,
  };
  return loom_serve_residency_admit(residency_count, residencies, true,
                                    state_owner, &state, out_admission);
}

iree_status_t loom_serve_residency_acquire(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies) {
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_residency_try_acquire(
      residency_count, residencies, &admitted));
  return admitted ? iree_ok_status()
                  : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                     "model residency is waiting for capacity");
}

iree_status_t loom_serve_residency_acquire_wait(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies) {
  bool admitted = false;
  iree_status_t status =
      loom_serve_residency_try_acquire(residency_count, residencies, &admitted);
  while (iree_status_is_ok(status) && !admitted) {
    bool pending = false;
    for (iree_host_size_t i = 0;
         i < residency_count && iree_status_is_ok(status); ++i) {
      loom_serve_residency_cache_t* cache = residencies[i]->cache;
      if (cache->pool && (loom_serve_memory_pool_is_pending(cache->pool) ||
                          cache->releasing_count)) {
        pending = true;
        status = loom_serve_memory_pool_await(cache->pool);
      }
    }
    if (!pending) {
      break;
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_residency_try_acquire(residency_count, residencies,
                                                &admitted);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (!admitted) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "model residency is waiting for capacity");
  }
  for (iree_host_size_t i = 0; i < residency_count; ++i) {
    status =
        iree_status_join(status, loom_serve_residency_await(residencies[i]));
  }
  return status;
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

iree_status_t loom_serve_residency_activate(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies) {
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_acquire_wait(residency_count, residencies));
  for (iree_host_size_t i = 0; i < residency_count; ++i) {
    loom_serve_residency_release(residencies[i]);
  }
  return iree_ok_status();
}
