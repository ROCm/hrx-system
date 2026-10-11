// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/workspace.h"

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

typedef struct workspace_generation_t {
  // Owned pool, also retained by each outstanding execution root record.
  iree_hal_pool_t* pool;
  // Owned backing-class cache retaining slabs returned by the logical pool.
  iree_hal_pool_t* cache;
  // Maximum reservation and slab bytes for this immutable pool geometry.
  iree_device_size_t length;
  // Minimum base alignment of this pool's slabs, in bytes.
  iree_device_size_t alignment;
  // Older generation still accounted until its roots and queue captures retire.
  struct workspace_generation_t* next;
} workspace_generation_t;

typedef struct workspace_root_t {
  // Owned allocation root until every caller and queue capture releases it.
  iree_hal_buffer_t* buffer;
  // Retained source pool borrowed by the root, including during destruction.
  iree_hal_pool_t* pool;
} workspace_root_t;

struct loom_serve_workspace_scope_t {
  // Borrowed shared owner, retained by the execution until scope retirement.
  loom_serve_workspace_t* workspace;
  // Owned raw allocation records; never contains model payloads or callbacks.
  workspace_root_t* roots;
  // Reusable ring capacity, grown before queue submission.
  iree_host_size_t capacity;
  // First occupied ring entry.
  iree_host_size_t head;
  // Occupied entries awaiting final caller/queue root release.
  iree_host_size_t count;
  // True after execution releases its ownership; collection then frees storage.
  bool released;
  // Shared-owner scope chain, including active and released execution domains.
  struct loom_serve_workspace_scope_t* next;
};

struct loom_serve_workspace_t {
  // Ownership shared by resource bindings and independent execution domains.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for this owner and cold generation metadata.
  iree_allocator_t allocator;
  // Retained group enclosing native owners and captured memory scopes.
  iree_hal_device_group_t* group;
  // Immutable allocation-shaping policy selected for all consumers.
  iree_hal_asan_pool_options_t asan;
  // Owned native backing source shared by all geometry generations.
  iree_hal_pool_t* backing_pool;
  // Protects geometry, generations and archived statistics, never device waits.
  iree_slim_mutex_t mutex;
  // Largest registered searchable slab size, in bytes.
  iree_device_size_t length;
  // Largest registered slab base alignment, in bytes.
  iree_device_size_t alignment;
  // Newest allocated geometry followed by still-retained older pools.
  workspace_generation_t* current;
  // Cumulative counters preserved when an idle old generation is destroyed.
  iree_hal_pool_stats_t retired;
  // Raw allocation bookkeeping shared with active and released executions.
  loom_serve_workspace_scope_t* scopes;
};

// The scope owns one reference. Uniqueness proves the queue's allocation and
// deallocation records have retired, independently of semaphore readiness.
static void workspace_scope_collect(loom_serve_workspace_scope_t* scope) {
  while (scope->count) {
    workspace_root_t* root = &scope->roots[scope->head];
    if (iree_atomic_ref_count_load(
            &((iree_hal_resource_t*)root->buffer)->ref_count) != 1) {
      break;
    }
    iree_hal_buffer_release(root->buffer);
    iree_hal_pool_release(root->pool);
    scope->head = (scope->head + 1) % scope->capacity;
    --scope->count;
  }
}

static void workspace_add_counters(iree_hal_pool_stats_t* total,
                                   const iree_hal_pool_stats_t* source) {
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

static void workspace_generation_destroy(workspace_generation_t* generation,
                                         iree_allocator_t allocator) {
  if (!generation) {
    return;
  }
  iree_hal_pool_release(generation->pool);
  iree_hal_pool_release(generation->cache);
  iree_allocator_free(allocator, generation);
}

// Execution records retain the pool until the last root/queue user retires.
// The owner's sole reference therefore proves an old generation can be freed,
// independently of semaphore success or failure. Active scopes are collected
// too: idle models must not require another submission to enable backing trim.
static void workspace_collect(loom_serve_workspace_t* workspace) {
  loom_serve_workspace_scope_t** scope_link = &workspace->scopes;
  while (*scope_link) {
    loom_serve_workspace_scope_t* scope = *scope_link;
    workspace_scope_collect(scope);
    if (scope->count || !scope->released) {
      scope_link = &scope->next;
    } else {
      *scope_link = scope->next;
      iree_allocator_free(workspace->allocator, scope->roots);
      iree_allocator_free(workspace->allocator, scope);
    }
  }
  if (!workspace->current) {
    return;
  }
  workspace_generation_t** link = &workspace->current->next;
  while (*link) {
    workspace_generation_t* generation = *link;
    if (iree_atomic_ref_count_load(
            &((iree_hal_resource_t*)generation->pool)->ref_count) != 1) {
      link = &generation->next;
      continue;
    }
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    workspace_add_counters(&workspace->retired, &statistics);
    *link = generation->next;
    workspace_generation_destroy(generation, workspace->allocator);
  }
}

void loom_serve_workspace_retain(loom_serve_workspace_t* workspace) {
  iree_atomic_ref_count_inc(&workspace->ref_count);
}

void loom_serve_workspace_release(loom_serve_workspace_t* workspace) {
  if (!workspace || iree_atomic_ref_count_dec(&workspace->ref_count) != 1) {
    return;
  }
  // Only final application teardown joins raw queue ownership. Live model
  // removal relinquishes its scope without waiting for queue bookkeeping.
  workspace_collect(workspace);
  while (workspace->scopes) {
    iree_thread_yield();
    workspace_collect(workspace);
  }
  workspace_generation_t* generation = workspace->current;
  while (generation) {
    workspace_generation_t* next = generation->next;
    workspace_generation_destroy(generation, workspace->allocator);
    generation = next;
  }
  iree_slim_mutex_deinitialize(&workspace->mutex);
  iree_hal_pool_release(workspace->backing_pool);
  iree_hal_device_group_release(workspace->group);
  iree_allocator_free(workspace->allocator, workspace);
}

iree_status_t loom_serve_workspace_create(
    iree_hal_device_group_t* group, iree_hal_pool_t* backing_pool,
    const iree_hal_asan_pool_options_t* asan_options,
    loom_serve_workspace_t** out_workspace, iree_allocator_t host_allocator) {
  *out_workspace = NULL;
  loom_serve_workspace_t* workspace = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*workspace),
                                             (void**)&workspace));
  iree_atomic_ref_count_init(&workspace->ref_count);
  iree_slim_mutex_initialize(&workspace->mutex);
  workspace->allocator = host_allocator;
  workspace->group = group;
  iree_hal_device_group_retain(workspace->group);
  workspace->asan = *asan_options;
  workspace->backing_pool = backing_pool;
  iree_hal_pool_retain(workspace->backing_pool);
  *out_workspace = workspace;
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_reserve(loom_serve_workspace_t* workspace,
                                           iree_device_size_t length,
                                           iree_device_size_t alignment) {
  iree_slim_mutex_lock(&workspace->mutex);
  if (!length ||
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
  workspace_collect(workspace);
  workspace_generation_t* current = workspace->current;
  iree_status_t status = iree_ok_status();
  if (!current || current->length < workspace->length ||
      current->alignment < workspace->alignment) {
    workspace_generation_t* generation = NULL;
    status = iree_allocator_malloc(workspace->allocator, sizeof(*generation),
                                   (void**)&generation);
    if (iree_status_is_ok(status)) {
      generation->length = workspace->length;
      generation->alignment = workspace->alignment;
      const iree_hal_tlsf_pool_options_t options = {
          .tlsf_options = {.range_length = generation->length,
                           .alignment = generation->alignment},
          .asan = workspace->asan,
          .trace_name = IREE_SVL("loom_serve_workspace"),
      };
      iree_hal_slab_cache_options_t cache_options = {
          .max_count = UINT32_MAX,
          .trace_name = IREE_SVL("loom_serve_workspace_cache"),
      };
      status = iree_hal_tlsf_pool_query_backing_request(
          workspace->backing_pool, &options, &cache_options.slab);
      if (iree_status_is_ok(status)) {
        status = iree_hal_slab_cache_create(
            workspace->backing_pool, &cache_options, workspace->allocator,
            &generation->cache);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_hal_tlsf_pool_create(generation->cache, &options,
                                      workspace->allocator, &generation->pool);
      }
    }
    if (iree_status_is_ok(status)) {
      generation->next = current;
      workspace->current = generation;
      workspace_collect(workspace);
    } else {
      workspace_generation_destroy(generation, workspace->allocator);
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
  workspace_collect(workspace);
  iree_hal_pool_stats_t total = workspace->retired;
  for (workspace_generation_t* generation = workspace->current; generation;
       generation = generation->next) {
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    iree_hal_pool_stats_t backing;
    iree_hal_pool_query_stats(generation->cache, &backing);
    total.bytes_reserved += statistics.bytes_reserved;
    total.bytes_free += statistics.bytes_free + backing.bytes_free;
    total.bytes_committed += backing.bytes_committed;
    total.bytes_quarantined += statistics.bytes_quarantined;
    total.reservation_count += statistics.reservation_count;
    total.slab_count += backing.slab_count;
    workspace_add_counters(&total, &statistics);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return total;
}

iree_status_t loom_serve_workspace_scope_create(
    loom_serve_workspace_t* workspace,
    loom_serve_workspace_scope_t** out_scope) {
  *out_scope = NULL;
  loom_serve_workspace_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(workspace->allocator,
                                             sizeof(*scope), (void**)&scope));
  scope->workspace = workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  scope->next = workspace->scopes;
  workspace->scopes = scope;
  iree_slim_mutex_unlock(&workspace->mutex);
  *out_scope = scope;
  return iree_ok_status();
}

iree_status_t loom_serve_workspace_scope_acquire(
    loom_serve_workspace_scope_t* scope, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  loom_serve_workspace_t* workspace = scope->workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  workspace_scope_collect(scope);
  iree_status_t status = iree_ok_status();
  if (scope->count == scope->capacity) {
    const iree_host_size_t capacity = iree_max(16, scope->capacity * 2);
    workspace_root_t* roots = NULL;
    status = iree_allocator_malloc_array(workspace->allocator, capacity,
                                         sizeof(*roots), (void**)&roots);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < scope->count; ++i) {
        roots[i] = scope->roots[(scope->head + i) % scope->capacity];
      }
      iree_allocator_free(workspace->allocator, scope->roots);
      scope->roots = roots;
      scope->capacity = capacity;
      scope->head = 0;
    }
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  if (iree_status_is_ok(status)) {
    status = loom_serve_workspace_acquire_pool(workspace, out_pool);
  }
  return status;
}

void loom_serve_workspace_scope_capture(loom_serve_workspace_scope_t* scope,
                                        iree_hal_pool_t* pool,
                                        iree_hal_buffer_t* buffer) {
  iree_slim_mutex_lock(&scope->workspace->mutex);
  const iree_host_size_t tail =
      (scope->head + scope->count++) % scope->capacity;
  scope->roots[tail] = (workspace_root_t){buffer, pool};
  iree_hal_buffer_retain(buffer);
  iree_slim_mutex_unlock(&scope->workspace->mutex);
}

void loom_serve_workspace_scope_retire(loom_serve_workspace_scope_t* scope) {
  if (!scope) {
    return;
  }
  loom_serve_workspace_t* workspace = scope->workspace;
  iree_slim_mutex_lock(&workspace->mutex);
  scope->released = true;
  workspace_collect(workspace);
  iree_slim_mutex_unlock(&workspace->mutex);
}

void loom_serve_workspace_trim(loom_serve_workspace_t* workspace) {
  iree_slim_mutex_lock(&workspace->mutex);
  workspace_collect(workspace);
  for (workspace_generation_t* generation = workspace->current; generation;
       generation = generation->next) {
    iree_hal_pool_trim(generation->pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    iree_hal_pool_trim(generation->cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
}
