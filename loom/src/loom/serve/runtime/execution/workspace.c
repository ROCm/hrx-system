// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/workspace.h"

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

typedef struct loom_serve_workspace_generation_t {
  // Owned logical pool, retained independently by queue allocation roots.
  iree_hal_pool_t* pool;
  // Owned exact native cache for fixed backing; null for virtual backing.
  iree_hal_pool_t* cache;
  // Preferred range size for this allocator.
  iree_device_size_t length;
  // Regular range alignment.
  iree_device_size_t alignment;
  // Earlier fixed geometry still retained by raw roots; null for virtual.
  struct loom_serve_workspace_generation_t* next;
} loom_serve_workspace_generation_t;

typedef struct loom_serve_workspace_root_t {
  // Owned allocation root until every caller and queue capture releases it.
  iree_hal_buffer_t* buffer;
  // Retained source pool borrowed by the root, including during destruction.
  iree_hal_pool_t* pool;
} loom_serve_workspace_root_t;

struct loom_serve_workspace_scope_t {
  // Borrowed shared owner, retained by the execution until scope retirement.
  loom_serve_workspace_t* workspace;
  // Owned raw allocation records; never contains model payloads or callbacks.
  loom_serve_workspace_root_t* roots;
  // Reusable ring capacity, grown before queue submission.
  iree_host_size_t capacity;
  // First occupied ring entry.
  iree_host_size_t head;
  // Occupied entries awaiting final caller/queue root release.
  iree_host_size_t count;
  // True after execution releases its ownership; collection then frees storage.
  bool released;
  // Whether this scope's roots contribute to transient workspace activity.
  bool tracks_workspace;
  // Shared-owner scope chain, including active and released execution domains.
  struct loom_serve_workspace_scope_t* next;
};

struct loom_serve_workspace_t {
  // Ownership shared by resource bindings and independent execution domains.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for this owner and raw capture metadata.
  iree_allocator_t host_allocator;
  // Retained group enclosing native owners and captured memory scopes.
  iree_hal_device_group_t* group;
  // Immutable allocation-shaping policy selected for all consumers.
  iree_hal_asan_pool_options_t asan;
  // Retained prepared source, native for fixed storage or a virtual source.
  iree_hal_pool_t* backing_pool;
  // Borrowed serialized physical owner, present only for elastic backing.
  loom_serve_virtual_pool_t* virtual_pool;
  // Protects pool construction and raw captures, never device waits.
  iree_slim_mutex_t mutex;
  // Regular range size; virtual backing fixes this independently of models.
  iree_device_size_t length;
  // Regular range alignment, independent of later dedicated reservations.
  iree_device_size_t alignment;
  // One persistent virtual-backed allocator, or fixed geometry generations.
  loom_serve_workspace_generation_t* current;
  // Counters of retired fixed generations; virtual backing retains its pool.
  iree_hal_pool_stats_t retired;
  // Raw allocation bookkeeping shared with active and released executions.
  loom_serve_workspace_scope_t* scopes;
  // Idle backing becomes eligible only after the last raw root retires.
  struct {
    // Observation time of the most recently retired root.
    iree_time_t since;
    // An idle trim has already been scheduled since the last retirement.
    bool trimmed;
  } idle;
};

typedef enum loom_serve_workspace_collection_mode_e {
  // Submission fast path stops at the first live root.
  WORKSPACE_COLLECTION_PREFIX,
  // Maintenance also collects retired roots behind live escaping buffers.
  WORKSPACE_COLLECTION_COMPLETE,
} loom_serve_workspace_collection_mode_t;

// The scope owns one reference. Uniqueness proves the queue's allocation and
// deallocation records have retired, independently of semaphore readiness.
static bool loom_serve_workspace_root_release(
    loom_serve_workspace_root_t root) {
  if (iree_atomic_ref_count_load(
          &((iree_hal_resource_t*)root.buffer)->ref_count) != 1) {
    return false;
  }
  iree_hal_buffer_release(root.buffer);
  iree_hal_pool_release(root.pool);
  return true;
}

static void loom_serve_workspace_scope_collect(
    loom_serve_workspace_scope_t* scope,
    loom_serve_workspace_collection_mode_t mode) {
  const iree_host_size_t previous_count = scope->count;
  if (mode == WORKSPACE_COLLECTION_PREFIX) {
    while (scope->count &&
           loom_serve_workspace_root_release(scope->roots[scope->head])) {
      scope->head = (scope->head + 1) % scope->capacity;
      --scope->count;
    }
  } else {
    iree_host_size_t retained = 0;
    for (iree_host_size_t i = 0; i < scope->count; ++i) {
      const loom_serve_workspace_root_t root =
          scope->roots[(scope->head + i) % scope->capacity];
      if (!loom_serve_workspace_root_release(root)) {
        scope->roots[(scope->head + retained++) % scope->capacity] = root;
      }
    }
    scope->count = retained;
  }
  if (scope->tracks_workspace && scope->count != previous_count) {
    scope->workspace->idle.since = iree_time_now();
    scope->workspace->idle.trimmed = false;
  }
}

static void loom_serve_workspace_add_counters(
    iree_hal_pool_stats_t* total, const iree_hal_pool_stats_t* source) {
  total->reserve_count += source->reserve_count;
  total->release_count += source->release_count;
  total->reuse_count += source->reuse_count;
  total->reuse_miss_count += source->reuse_miss_count;
  total->fresh_count += source->fresh_count;
  total->exhausted_count += source->exhausted_count;
  total->over_budget_count += source->over_budget_count;
  total->wait_count += source->wait_count;
  total->quarantine_eviction_count += source->quarantine_eviction_count;
}

static void loom_serve_workspace_generation_destroy(
    loom_serve_workspace_generation_t* generation,
    iree_allocator_t host_allocator) {
  if (!generation) {
    return;
  }
  iree_hal_pool_release(generation->pool);
  iree_hal_pool_release(generation->cache);
  iree_allocator_free(host_allocator, generation);
}

// Execution records retain the pool until the last root/queue user retires.
// Active scopes are collected too: idle models must not require another
// submission to enable backing trim.
static bool loom_serve_workspace_collect(
    loom_serve_workspace_t* workspace,
    loom_serve_workspace_collection_mode_t mode) {
  bool live = false;
  loom_serve_workspace_scope_t** scope_link = &workspace->scopes;
  while (*scope_link) {
    loom_serve_workspace_scope_t* scope = *scope_link;
    loom_serve_workspace_scope_collect(scope, mode);
    live |= scope->tracks_workspace && scope->count != 0;
    if (scope->count || !scope->released) {
      scope_link = &scope->next;
    } else {
      *scope_link = scope->next;
      iree_allocator_free(workspace->host_allocator, scope->roots);
      iree_allocator_free(workspace->host_allocator, scope);
    }
  }
  if (!workspace->current) {
    return live;
  }
  loom_serve_workspace_generation_t** link = &workspace->current->next;
  while (*link) {
    loom_serve_workspace_generation_t* generation = *link;
    if (iree_atomic_ref_count_load(
            &((iree_hal_resource_t*)generation->pool)->ref_count) != 1) {
      link = &generation->next;
      continue;
    }
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    loom_serve_workspace_add_counters(&workspace->retired, &statistics);
    *link = generation->next;
    loom_serve_workspace_generation_destroy(generation,
                                            workspace->host_allocator);
  }
  return live;
}

void loom_serve_workspace_retain(loom_serve_workspace_t* workspace) {
  if (workspace) {
    iree_atomic_ref_count_inc(&workspace->ref_count);
  }
}

void loom_serve_workspace_release(loom_serve_workspace_t* workspace) {
  if (workspace && iree_atomic_ref_count_dec(&workspace->ref_count) == 1) {
    // Only final application teardown joins raw queue ownership. Live model
    // removal relinquishes its scope without waiting for queue bookkeeping.
    loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
    while (workspace->scopes) {
      iree_thread_yield();
      loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
    }
    loom_serve_workspace_generation_t* generation = workspace->current;
    while (generation) {
      loom_serve_workspace_generation_t* next = generation->next;
      loom_serve_workspace_generation_destroy(generation,
                                              workspace->host_allocator);
      generation = next;
    }
    iree_slim_mutex_deinitialize(&workspace->mutex);
    iree_hal_pool_release(workspace->backing_pool);
    iree_hal_device_group_release(workspace->group);
    iree_allocator_free(workspace->host_allocator, workspace);
  }
}

iree_status_t loom_serve_workspace_create(
    iree_hal_device_group_t* group, iree_hal_pool_t* backing_pool,
    const iree_hal_asan_pool_options_t* asan_options,
    iree_allocator_t host_allocator, loom_serve_workspace_t** out_workspace) {
  *out_workspace = NULL;
  loom_serve_workspace_t* workspace = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*workspace),
                                             (void**)&workspace));
  iree_atomic_ref_count_init(&workspace->ref_count);
  iree_slim_mutex_initialize(&workspace->mutex);
  workspace->host_allocator = host_allocator;
  workspace->group = group;
  iree_hal_device_group_retain(workspace->group);
  workspace->asan = *asan_options;
  workspace->backing_pool = backing_pool;
  iree_hal_pool_retain(workspace->backing_pool);
  *out_workspace = workspace;
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_create_from_virtual(
    iree_hal_device_group_t* group, loom_serve_virtual_pool_t* virtual_pool,
    iree_hal_pool_reservation_request_t range, iree_allocator_t host_allocator,
    loom_serve_workspace_t** out_workspace) {
  const iree_hal_asan_pool_options_t asan = {0};
  IREE_RETURN_IF_ERROR(loom_serve_workspace_create(
      group, loom_serve_virtual_pool_handle(virtual_pool), &asan,
      host_allocator, out_workspace));
  loom_serve_workspace_t* workspace = *out_workspace;
  workspace->virtual_pool = virtual_pool;
  workspace->length = range.allocation_size;
  workspace->alignment = range.min_alignment;
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_reserve(loom_serve_workspace_t* workspace,
                                           iree_device_size_t length,
                                           iree_device_size_t alignment) {
  iree_slim_mutex_lock(&workspace->mutex);
  if (workspace->virtual_pool || !length ||
      (length <= workspace->length && alignment <= workspace->alignment)) {
    iree_slim_mutex_unlock(&workspace->mutex);
    return iree_ok_status();
  }
  length = iree_max(length, workspace->length);
  alignment = iree_max(iree_max(alignment, workspace->alignment),
                       IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT);
  // TLSF rounds a maximum-size search up to the next size class. Round slab
  // geometry the same way; command requests keep their exact reflected sizes.
  const uint32_t first_level = 63 - iree_math_count_leading_zeros_u64(length);
  const uint32_t shift = first_level > IREE_HAL_MEMORY_TLSF_SL_LOG2
                             ? first_level - IREE_HAL_MEMORY_TLSF_SL_LOG2
                             : 0;
  const iree_device_size_t quantum =
      iree_max((iree_device_size_t)1 << shift, alignment);
  if (length > IREE_DEVICE_SIZE_MAX - (quantum - 1)) {
    iree_slim_mutex_unlock(&workspace->mutex);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workspace slab size overflows device range");
  }
  workspace->length = iree_device_align(length, quantum);
  workspace->alignment = alignment;
  iree_slim_mutex_unlock(&workspace->mutex);
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_acquire_pool(
    loom_serve_workspace_t* workspace, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  iree_slim_mutex_lock(&workspace->mutex);
  loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_PREFIX);
  loom_serve_workspace_generation_t* current = workspace->current;
  iree_status_t status = iree_ok_status();
  if (!current || (!workspace->virtual_pool &&
                   (current->length < workspace->length ||
                    current->alignment < workspace->alignment))) {
    loom_serve_workspace_generation_t* generation = NULL;
    status = iree_allocator_malloc(workspace->host_allocator,
                                   sizeof(*generation), (void**)&generation);
    const iree_hal_tlsf_pool_options_t options = {
        .tlsf_options = {.range_length = workspace->length,
                         .alignment = workspace->alignment},
        .asan = workspace->asan,
        .trace_name = IREE_SVL("loom-serve-workspace"),
    };
    iree_hal_pool_t* source = workspace->backing_pool;
    if (iree_status_is_ok(status) && !workspace->virtual_pool) {
      iree_hal_slab_cache_options_t cache_options = {
          .max_count = UINT32_MAX,
          .trace_name = IREE_SVL("loom-serve-workspace-cache"),
      };
      status = iree_hal_tlsf_pool_query_backing_request(
          workspace->backing_pool, &options, &cache_options.slab);
      if (iree_status_is_ok(status)) {
        status = iree_hal_slab_cache_create(
            workspace->backing_pool, &cache_options, workspace->host_allocator,
            &generation->cache);
      }
      source = generation->cache;
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_tlsf_pool_create(
          source, &options, workspace->host_allocator, &generation->pool);
    }
    if (iree_status_is_ok(status)) {
      generation->length = workspace->length;
      generation->alignment = workspace->alignment;
      generation->next = current;
      workspace->current = generation;
      loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
    } else {
      loom_serve_workspace_generation_destroy(generation,
                                              workspace->host_allocator);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_pool = workspace->current->pool;
    iree_hal_pool_retain(*out_pool);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return status;
}

iree_hal_pool_stats_t loom_serve_workspace_statistics(
    loom_serve_workspace_t* workspace) {
  iree_slim_mutex_lock(&workspace->mutex);
  loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
  iree_hal_pool_stats_t total = workspace->retired;
  for (loom_serve_workspace_generation_t* generation = workspace->current;
       generation; generation = generation->next) {
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    iree_hal_pool_stats_t backing;
    iree_hal_pool_query_stats(
        generation->cache ? generation->cache : workspace->backing_pool,
        &backing);
    total.bytes_reserved += statistics.bytes_reserved;
    total.bytes_free += statistics.bytes_free + backing.bytes_free;
    total.bytes_committed +=
        workspace->virtual_pool
            ? loom_serve_virtual_pool_memory_usage(workspace->virtual_pool)
                  .committed_bytes
            : backing.bytes_committed;
    total.bytes_quarantined += statistics.bytes_quarantined;
    total.reservation_count += statistics.reservation_count;
    total.slab_count += backing.slab_count;
    loom_serve_workspace_add_counters(&total, &statistics);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return total;
}

iree_status_t loom_serve_workspace_scope_allocate(
    loom_serve_workspace_t* workspace, bool tracks_workspace,
    loom_serve_workspace_scope_t** out_scope) {
  *out_scope = NULL;
  loom_serve_workspace_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(workspace->host_allocator,
                                             sizeof(*scope), (void**)&scope));
  scope->workspace = workspace;
  scope->tracks_workspace = tracks_workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  scope->next = workspace->scopes;
  workspace->scopes = scope;
  iree_slim_mutex_unlock(&workspace->mutex);
  *out_scope = scope;
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_scope_reserve(
    loom_serve_workspace_scope_t* scope, iree_host_size_t count) {
  loom_serve_workspace_t* workspace = scope->workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  loom_serve_workspace_scope_collect(scope, WORKSPACE_COLLECTION_PREFIX);
  iree_status_t status = iree_ok_status();
  if (count > scope->capacity - scope->count) {
    loom_serve_workspace_scope_collect(scope, WORKSPACE_COLLECTION_COMPLETE);
    // Reclaim at least half the ring before reusing a full allocation.
    // Otherwise geometric growth amortizes the scan over subsequent captures,
    // including when nearly every root remains live across many submissions.
    if (count > scope->capacity - scope->count ||
        scope->count > scope->capacity / 2) {
      const iree_host_size_t capacity =
          iree_max(iree_max(16, scope->capacity * 2), scope->count + count);
      loom_serve_workspace_root_t* roots = NULL;
      status = iree_allocator_malloc_array(workspace->host_allocator, capacity,
                                           sizeof(*roots), (void**)&roots);
      if (iree_status_is_ok(status)) {
        for (iree_host_size_t i = 0; i < scope->count; ++i) {
          roots[i] = scope->roots[(scope->head + i) % scope->capacity];
        }
        iree_allocator_free(workspace->host_allocator, scope->roots);
        scope->roots = roots;
        scope->capacity = capacity;
        scope->head = 0;
      }
    }
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return status;
}

iree_status_t loom_serve_workspace_scope_acquire(
    loom_serve_workspace_scope_t* scope, iree_host_size_t count,
    iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_workspace_scope_reserve(scope, count));
  return loom_serve_workspace_acquire_pool(scope->workspace, out_pool);
}

void loom_serve_workspace_scope_capture(loom_serve_workspace_scope_t* scope,
                                        iree_hal_pool_t* pool,
                                        iree_hal_buffer_t* buffer) {
  iree_slim_mutex_lock(&scope->workspace->mutex);
  const iree_host_size_t tail =
      (scope->head + scope->count++) % scope->capacity;
  scope->roots[tail] = (loom_serve_workspace_root_t){buffer, pool};
  iree_hal_buffer_retain(buffer);
  if (scope->tracks_workspace) {
    scope->workspace->idle.trimmed = false;
  }
  iree_slim_mutex_unlock(&scope->workspace->mutex);
}

void loom_serve_workspace_scope_retire(loom_serve_workspace_scope_t* scope) {
  if (!scope) {
    return;
  }
  loom_serve_workspace_t* workspace = scope->workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  scope->released = true;
  loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
  iree_slim_mutex_unlock(&workspace->mutex);
}

static iree_device_size_t loom_serve_workspace_trim_locked(
    loom_serve_workspace_t* workspace) {
  iree_device_size_t retained_bytes = 0;
  for (loom_serve_workspace_generation_t* generation = workspace->current;
       generation; generation = generation->next) {
    iree_hal_pool_trim(generation->pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    iree_hal_pool_trim(
        generation->cache ? generation->cache : workspace->backing_pool,
        IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    // Child returns reach the cache synchronously. After trimming both layers,
    // only backing still retained by the child contributes to reclamation.
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    retained_bytes += statistics.bytes_committed;
  }
  workspace->idle.trimmed = true;
  return retained_bytes;
}

iree_status_t loom_serve_workspace_trim(
    loom_serve_workspace_t* workspace, iree_device_size_t* out_retained_bytes) {
  iree_slim_mutex_lock(&workspace->mutex);
  loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
  *out_retained_bytes = loom_serve_workspace_trim_locked(workspace);
  iree_slim_mutex_unlock(&workspace->mutex);
  return workspace->virtual_pool
             ? loom_serve_virtual_pool_reclaim(workspace->virtual_pool)
             : iree_ok_status();
}

bool loom_serve_workspace_is_reclaiming(loom_serve_workspace_t* workspace) {
  return workspace->virtual_pool &&
         loom_serve_virtual_pool_is_reclaiming(workspace->virtual_pool);
}

bool loom_serve_workspace_trim_idle(loom_serve_workspace_t* workspace,
                                    iree_time_t now, iree_duration_t timeout) {
  iree_slim_mutex_lock(&workspace->mutex);
  const bool live =
      loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
  bool pending = live || (workspace->idle.since && !workspace->idle.trimmed);
  if (!live && pending && now - workspace->idle.since >= timeout) {
    loom_serve_workspace_trim_locked(workspace);
    pending = false;
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return pending;
}

void loom_serve_workspace_set_observer(loom_serve_workspace_t* workspace,
                                       loom_serve_wake_callback_t observer) {
  if (workspace->virtual_pool) {
    loom_serve_virtual_pool_set_observer(workspace->virtual_pool, observer);
  }
}

loom_serve_memory_usage_t loom_serve_workspace_memory_usage(
    const loom_serve_workspace_t* workspace) {
  return workspace->virtual_pool
             ? loom_serve_virtual_pool_memory_usage(workspace->virtual_pool)
             : (loom_serve_memory_usage_t){0};
}

uint64_t loom_serve_workspace_round_size(
    const loom_serve_workspace_t* workspace, uint64_t byte_length) {
  return workspace->virtual_pool ? loom_serve_virtual_pool_round_size(
                                       workspace->virtual_pool, byte_length)
                                 : byte_length;
}

iree_status_t loom_serve_workspace_advance(loom_serve_workspace_t* workspace) {
  iree_slim_mutex_lock(&workspace->mutex);
  loom_serve_workspace_collect(workspace, WORKSPACE_COLLECTION_COMPLETE);
  iree_slim_mutex_unlock(&workspace->mutex);
  return workspace->virtual_pool
             ? loom_serve_virtual_pool_advance(workspace->virtual_pool)
             : iree_ok_status();
}

bool loom_serve_workspace_is_pending(loom_serve_workspace_t* workspace) {
  return workspace->virtual_pool &&
         loom_serve_virtual_pool_is_pending(workspace->virtual_pool);
}

iree_status_t loom_serve_workspace_wait(loom_serve_workspace_t* workspace,
                                        iree_hal_semaphore_t* semaphore,
                                        uint64_t value) {
  return workspace->virtual_pool
             ? loom_serve_virtual_pool_wait(workspace->virtual_pool, semaphore,
                                            value)
             : iree_hal_semaphore_wait(semaphore, value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
}

static bool loom_serve_workspace_is_retired(void* user_data) {
  return loom_serve_retirement_is_idle(user_data);
}

iree_status_t loom_serve_workspace_await_retirement(
    loom_serve_workspace_t* workspace, loom_serve_retirement_t* retirement) {
  if (!workspace->virtual_pool) {
    loom_serve_retirement_await(retirement);
    return iree_ok_status();
  }
  return loom_serve_virtual_pool_await(
      workspace->virtual_pool, &retirement->notification,
      loom_serve_workspace_is_retired, retirement);
}
