// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/device/pool.h"

#include "iree/hal/memory/tlsf_pool.h"

struct loom_serve_pool_t {
  // Allocator owning this pool's metadata.
  iree_allocator_t host_allocator;
  bool uncached;
  iree_host_size_t live_family_count;
  const iree_hal_queue_family_t** live_families;
  // Persistent buffers and their independently accounted elastic source.
  struct {
    // Owned allocator, native when fixed or TLSF when elastic.
    iree_hal_pool_t* allocator;
    // Borrowed source within virtual_pool; NULL for fixed native backing.
    iree_hal_pool_t* source;
  } storage;
  // Owned shared command backing, independent of model execution timelines.
  loom_serve_workspace_t* workspace;
  // Owned stable sources for workspace and persistent ranges; NULL when fixed.
  loom_serve_virtual_pool_t* virtual_pool;
  // Owned elastic physical budget; NULL when fixed.
  loom_serve_memory_pool_t* memory;
  // Qualified bulk parameter and workspace geometry.
  loom_serve_memory_geometry_t weight_geometry;
  // Qualified sparse mutable-state geometry.
  loom_serve_memory_geometry_t state_geometry;
  // Owned parameter admission and eviction policy borrowing memory/workspace.
  loom_serve_residency_cache_t* residency_cache;
};

static iree_status_t loom_serve_pool_resolve_placement(
    iree_hal_device_group_t* group, loom_serve_pool_placement_t requested,
    iree_hal_pool_placement_t* out_placement) {
  *out_placement = (iree_hal_pool_placement_t){0};
  if (requested.kind == LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC) {
    return iree_ok_status();
  }
  if ((uint32_t)requested.kind > LOOM_SERVE_POOL_PLACEMENT_HOST_NUMA ||
      (requested.kind == LOOM_SERVE_POOL_PLACEMENT_DEVICE &&
       requested.ordinal >= iree_hal_device_group_device_count(group))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool placement is outside the device group");
  }
  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  uint32_t match_count = 0;
  uint32_t selected_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  for (uint32_t i = 0; i < iree_hal_topology_node_count(topology); ++i) {
    const iree_hal_topology_node_t* node =
        iree_hal_topology_node_at(topology, i);
    const bool matches =
        requested.kind == LOOM_SERVE_POOL_PLACEMENT_DEVICE
            ? node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
                  node->device_ordinal == requested.ordinal
            : node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
                  node->local_ordinal == requested.ordinal;
    if (matches) {
      ++match_count;
      selected_node = node->ordinal;
    }
  }
  if (!match_count) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "pool placement %s %u is absent from the topology",
                            requested.kind == LOOM_SERVE_POOL_PLACEMENT_DEVICE
                                ? "device"
                                : "host NUMA node",
                            requested.ordinal);
  }
  if (match_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool placement names more than one physical "
                            "domain; select a device with one physical owner");
  }
  *out_placement = (iree_hal_pool_placement_t){
      .mode = IREE_HAL_POOL_PLACEMENT_REQUIRED,
      .node = selected_node,
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_pool_initialize_elastic(
    loom_serve_pool_t* pool, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_hal_queue_t* allocation_queue,
    const loom_serve_pool_options_t* options,
    iree_hal_pool_placement_t placement) {
  iree_hal_pool_family_access_t* source_families = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      pool->host_allocator,
      iree_make_const_byte_span(scope.families,
                                scope.family_count * sizeof(*scope.families)),
      (void**)&source_families));
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    source_families[i].interfaces |=
        UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  iree_hal_pool_scope_t source_scope = scope;
  source_scope.families = source_families;
  const iree_hal_slab_pool_options_t source_options = {
      .placement = placement,
      .trace_name = IREE_SVL("loom-serve-elastic-backing"),
  };
  iree_hal_pool_t* source_pool = NULL;
  iree_status_t status = iree_hal_slab_pool_create(
      group, source_scope, &source_options, pool->host_allocator, &source_pool);
  iree_allocator_free(pool->host_allocator, source_families);
  iree_hal_virtual_buffer_capabilities_t capabilities;
  if (iree_status_is_ok(status)) {
    status = iree_hal_virtual_buffer_query_capabilities(
        group, scope, (iree_hal_virtual_buffer_sources_t){1, &source_pool},
        pool->host_allocator, &capabilities);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_memory_pool_query_geometry(&capabilities, source_pool,
                                                   options->weight_slab_size,
                                                   &pool->weight_geometry);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_memory_pool_query_geometry(&capabilities, source_pool,
                                                   options->state_slab_size,
                                                   &pool->state_geometry);
    if (iree_status_is_ok(status)) {
      pool->state_geometry.mapping_size = pool->state_geometry.minimum_size;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_memory_pool_allocate(
        group, scope, source_pool, allocation_queue, pool->weight_geometry,
        pool->state_geometry, options->memory_limit, pool->host_allocator,
        &pool->memory);
  }
  iree_hal_pool_release(source_pool);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_virtual_pool_allocate(
      pool->memory, pool->weight_geometry, pool->host_allocator,
      &pool->virtual_pool));
  IREE_RETURN_IF_ERROR(loom_serve_virtual_pool_create_source(
      pool->virtual_pool, &pool->storage.source));
  const iree_hal_tlsf_pool_options_t storage_options = {
      .tlsf_options = {.range_length = pool->weight_geometry.mapping_size,
                       .alignment = 256},
      .trace_name = IREE_SVL("loom-serve-storage"),
  };
  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_create(
      pool->storage.source, &storage_options, pool->host_allocator,
      &pool->storage.allocator));
  // Small command allocations pack into a logical slab. Larger requirements
  // use dedicated virtual ranges drawing from the same physical owner.
  const iree_hal_pool_reservation_request_t range = {
      .allocation_size = pool->weight_geometry.mapping_size,
      .min_alignment = 256,
  };
  return loom_serve_workspace_create_from_virtual(
      group, pool->virtual_pool, range, pool->host_allocator, &pool->workspace);
}

iree_status_t loom_serve_pool_allocate(iree_hal_device_group_t* group,
                                       iree_hal_pool_scope_t scope,
                                       iree_hal_queue_t* allocation_queue,
                                       const loom_serve_pool_options_t* options,
                                       iree_allocator_t host_allocator,
                                       loom_serve_pool_t** out_pool) {
  *out_pool = NULL;
  if (!options->elastic && options->memory_limit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a physical budget requires elastic backing");
  }
  if (options->elastic &&
      iree_hal_asan_pool_options_is_enabled(&options->asan)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "address-sanitized pools require fixed backing");
  }
  iree_hal_pool_placement_t placement;
  IREE_RETURN_IF_ERROR(
      loom_serve_pool_resolve_placement(group, options->placement, &placement));
  loom_serve_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  pool->host_allocator = host_allocator;
  pool->uncached = options->uncached;
  const iree_hal_slab_pool_options_t storage_options = {
      .placement = placement,
      .trace_name = IREE_SVL("loom-serve-storage-backing"),
  };
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, scope.family_count, sizeof(*pool->live_families),
      (void**)&pool->live_families);
  for (iree_host_size_t i = 0;
       i < scope.family_count && iree_status_is_ok(status); ++i) {
    if (iree_any_bit_set(scope.families[i].requirements,
                         IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED |
                             IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST)) {
      pool->live_families[pool->live_family_count++] = scope.families[i].family;
    }
  }
  if (iree_status_is_ok(status) && options->elastic) {
    status = loom_serve_pool_initialize_elastic(
        pool, group, scope, allocation_queue, options, placement);
  } else if (iree_status_is_ok(status)) {
    status =
        iree_hal_slab_pool_create(group, scope, &storage_options,
                                  host_allocator, &pool->storage.allocator);
    if (iree_status_is_ok(status)) {
      status = loom_serve_workspace_create(group, pool->storage.allocator,
                                           &options->asan, host_allocator,
                                           &pool->workspace);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_cache_allocate(
        pool->memory, pool->workspace, pool->storage.source, host_allocator,
        &pool->residency_cache);
  }
  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    status = iree_status_join(status, loom_serve_pool_free(pool));
  }
  return status;
}

loom_serve_pool_statistics_t loom_serve_pool_statistics(
    const loom_serve_pool_t* pool) {
  loom_serve_pool_statistics_t statistics = {
      .workspace = loom_serve_workspace_statistics(pool->workspace)};
  iree_hal_pool_query_stats(
      pool->storage.source ? pool->storage.source : pool->storage.allocator,
      &statistics.storage);
  if (pool->memory) {
    statistics.retained = loom_serve_memory_pool_statistics(pool->memory);
  }
  return statistics;
}

iree_status_t loom_serve_pool_trim(loom_serve_pool_t* pool,
                                   uint64_t target_bytes) {
  const loom_serve_pool_statistics_t memory = loom_serve_pool_statistics(pool);
  const uint64_t committed = pool->memory ? memory.retained.committed_bytes
                                          : memory.workspace.bytes_committed;
  if (committed <= target_bytes) {
    return iree_ok_status();
  }
  iree_device_size_t workspace_bytes = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_workspace_trim(pool->workspace, &workspace_bytes));
  if (pool->memory) {
    IREE_RETURN_IF_ERROR(
        loom_serve_residency_cache_trim(pool->residency_cache, target_bytes));
    loom_serve_memory_pool_trim(pool->memory, target_bytes);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_pool_trim_idle(loom_serve_pool_t* pool,
                                        loom_serve_pool_idle_policy_t policy,
                                        iree_time_t now, bool* out_pending) {
  *out_pending = false;
  if (policy.workspace_timeout) {
    *out_pending = loom_serve_workspace_trim_idle(pool->workspace, now,
                                                  policy.workspace_timeout);
  }
  if (policy.parameter_timeout) {
    bool pending = false;
    IREE_RETURN_IF_ERROR(loom_serve_residency_cache_trim_idle(
        pool->residency_cache, now, policy.parameter_timeout, &pending));
    *out_pending |= pending;
  }
  return iree_ok_status();
}

bool loom_serve_pool_is_in_use(const loom_serve_pool_t* pool) {
  if (!pool) {
    return false;
  }
  if (!loom_serve_residency_cache_is_empty(pool->residency_cache)) {
    return true;
  }
  if (!pool->memory) {
    return false;
  }
  if (pool->storage.allocator) {
    iree_hal_pool_stats_t storage;
    iree_hal_pool_query_stats(pool->storage.allocator, &storage);
    if (storage.reservation_count) {
      return true;
    }
  }
  const uint64_t allocator_reserved =
      pool->virtual_pool
          ? loom_serve_virtual_pool_memory_usage(pool->virtual_pool)
                .reserved_bytes
          : 0;
  return loom_serve_memory_pool_statistics(pool->memory).reserved_bytes !=
         allocator_reserved;
}

iree_status_t loom_serve_pool_free(loom_serve_pool_t* pool) {
  if (!pool) {
    return iree_ok_status();
  }
  if (loom_serve_pool_is_in_use(pool)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "serving pool still has live consumers");
  }
  loom_serve_residency_cache_free(pool->residency_cache);
  pool->residency_cache = NULL;
  loom_serve_workspace_release(pool->workspace);
  pool->workspace = NULL;
  iree_hal_pool_release(pool->storage.allocator);
  pool->storage.allocator = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_virtual_pool_free(pool->virtual_pool));
  pool->virtual_pool = NULL;
  pool->storage.source = NULL;
  loom_serve_memory_pool_free(pool->memory);
  iree_allocator_free(pool->host_allocator, pool->live_families);
  iree_allocator_free(pool->host_allocator, pool);
  return iree_ok_status();
}

bool loom_serve_pool_is_uncached(const loom_serve_pool_t* pool) {
  return pool->uncached;
}

bool loom_serve_pool_has_live_access(const loom_serve_pool_t* pool,
                                     const iree_hal_queue_family_t* family) {
  for (iree_host_size_t i = 0; i < pool->live_family_count; ++i) {
    if (pool->live_families[i] == family) {
      return true;
    }
  }
  return false;
}

iree_hal_pool_t* loom_serve_pool_storage(const loom_serve_pool_t* pool) {
  return pool->storage.allocator;
}

loom_serve_workspace_t* loom_serve_pool_workspace(
    const loom_serve_pool_t* pool) {
  return pool->workspace;
}

loom_serve_memory_pool_t* loom_serve_pool_memory(
    const loom_serve_pool_t* pool) {
  return pool->memory;
}

loom_serve_memory_geometry_t loom_serve_pool_weight_geometry(
    const loom_serve_pool_t* pool) {
  return pool->weight_geometry;
}

loom_serve_memory_geometry_t loom_serve_pool_state_geometry(
    const loom_serve_pool_t* pool) {
  return pool->state_geometry;
}

loom_serve_residency_cache_t* loom_serve_pool_residency_cache(
    const loom_serve_pool_t* pool) {
  return pool->residency_cache;
}
