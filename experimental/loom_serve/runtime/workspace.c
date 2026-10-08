// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/workspace.h"

#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/tlsf_pool.h"

typedef struct workspace_generation_t {
  // Owned pool, also retained by each outstanding execution root record.
  iree_hal_pool_t* pool;
  // Maximum reservation and slab bytes for this immutable pool geometry.
  iree_device_size_t length;
  // Minimum base alignment of this pool's slabs, in bytes.
  iree_device_size_t alignment;
  // Older generation still accounted until its roots and queue captures retire.
  struct workspace_generation_t* next;
} workspace_generation_t;

struct loom_serve_workspace_t {
  // Ownership shared by the device and its independent execution domains.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for this owner and cold generation metadata.
  iree_allocator_t allocator;
  // Retained device enclosing backend pool ingredients.
  iree_hal_device_t* device;
  // Device-owned slab provider, completion query and notification services.
  iree_hal_queue_pool_backend_t backend;
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
};

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
}

// Execution records retain the pool until the last root/queue user retires.
// The owner's sole reference therefore proves an old generation can be freed,
// independently of semaphore success or failure. Stable geometry has no walk.
static void workspace_collect(loom_serve_workspace_t* workspace) {
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
    iree_hal_pool_release(generation->pool);
    iree_allocator_free(workspace->allocator, generation);
  }
}

void loom_serve_workspace_retain(loom_serve_workspace_t* workspace) {
  iree_atomic_ref_count_inc(&workspace->ref_count);
}

void loom_serve_workspace_release(loom_serve_workspace_t* workspace) {
  if (!workspace || iree_atomic_ref_count_dec(&workspace->ref_count) != 1) {
    return;
  }
  workspace_generation_t* generation = workspace->current;
  while (generation) {
    workspace_generation_t* next = generation->next;
    iree_hal_pool_release(generation->pool);
    iree_allocator_free(workspace->allocator, generation);
    generation = next;
  }
  iree_slim_mutex_deinitialize(&workspace->mutex);
  iree_hal_device_release(workspace->device);
  iree_allocator_free(workspace->allocator, workspace);
}

iree_status_t loom_serve_workspace_create(
    iree_hal_queue_t* dispatch_queue, loom_serve_workspace_t** out_workspace,
    iree_allocator_t host_allocator) {
  *out_workspace = NULL;
  loom_serve_workspace_t* workspace = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*workspace),
                                             (void**)&workspace));
  iree_atomic_ref_count_init(&workspace->ref_count);
  iree_slim_mutex_initialize(&workspace->mutex);
  workspace->allocator = host_allocator;
  workspace->device =
      iree_hal_queue_family_device(iree_hal_queue_family(dispatch_queue));
  iree_hal_device_retain(workspace->device);
  iree_status_t status = iree_hal_device_query_queue_pool_backend(
      workspace->device, iree_hal_queue_family(dispatch_queue),
      &workspace->backend);
  if (iree_status_is_ok(status)) {
    *out_workspace = workspace;
  } else {
    loom_serve_workspace_release(workspace);
  }
  return status;
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
          .asan = workspace->backend.asan,
          .trace_name = IREE_SVL("loom_serve_workspace"),
      };
      status = iree_hal_tlsf_pool_create(
          options, workspace->backend.slab_provider,
          workspace->backend.notification, workspace->backend.epoch_query,
          workspace->allocator, &generation->pool);
    }
    if (iree_status_is_ok(status)) {
      generation->next = current;
      workspace->current = generation;
      workspace_collect(workspace);
    } else {
      iree_allocator_free(workspace->allocator, generation);
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
  iree_hal_pool_stats_t total = workspace->retired;
  for (workspace_generation_t* generation = workspace->current; generation;
       generation = generation->next) {
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(generation->pool, &statistics);
    total.bytes_reserved += statistics.bytes_reserved;
    total.bytes_free += statistics.bytes_free;
    total.bytes_committed += statistics.bytes_committed;
    total.reservation_count += statistics.reservation_count;
    total.slab_count += statistics.slab_count;
    workspace_add_counters(&total, &statistics);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return total;
}

iree_status_t loom_serve_workspace_trim(loom_serve_workspace_t* workspace) {
  iree_slim_mutex_lock(&workspace->mutex);
  workspace_collect(workspace);
  iree_status_t status = iree_ok_status();
  for (workspace_generation_t* generation = workspace->current;
       generation && iree_status_is_ok(status); generation = generation->next) {
    status = iree_hal_pool_trim(generation->pool);
  }
  iree_slim_mutex_unlock(&workspace->mutex);
  return status;
}
