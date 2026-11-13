// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/construction_storage.h"

#include "loom/analysis/pipeline/invocations.h"
#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_storage_request_t {
  // Canonical backing identity, independent of occurrence-local pool indices.
  loom_source_storage_packing_t* packing;
  // Placement receiving this request's final offset.
  loom_pipeline_storage_allocation_t* allocation;
  // Construction owning the requested storage.
  iree_host_size_t owner;
  // Dense backing index assigned after grouping requests by canonical identity.
  iree_host_size_t pool;
} loom_pipeline_storage_request_t;

static bool loom_pipeline_storage_request_less(
    const loom_pipeline_storage_request_t* left,
    const loom_pipeline_storage_request_t* right) {
  if (left->packing != right->packing) {
    return (uintptr_t)left->packing < (uintptr_t)right->packing;
  }
  if (left->owner != right->owner) {
    return left->owner < right->owner;
  }
  return left->allocation->source->construction_order <
         right->allocation->source->construction_order;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_storage_sort_requests,
                          loom_pipeline_storage_request_t,
                          loom_pipeline_storage_request_less)

iree_status_t loom_pipeline_construction_storage_build(
    const loom_pipeline_invocations_t* invocations,
    iree_arena_allocator_t* arena,
    const loom_pipeline_storage_allocation_t*** out_allocations) {
  const loom_pipeline_storage_allocation_t** owners = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->count, sizeof(*owners), (void**)&owners));
  *out_allocations = owners;
  iree_host_size_t allocation_count = 0;
  for (iree_host_size_t i = 0; i < invocations->count; ++i) {
    allocation_count += invocations->entries[i]->resources->allocation_count;
  }
  loom_pipeline_storage_allocation_t* allocations = NULL;
  loom_pipeline_storage_request_t* requests = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, allocation_count, sizeof(*allocations), (void**)&allocations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, allocation_count, sizeof(*requests), (void**)&requests));
  iree_host_size_t next = 0;
  for (iree_host_size_t i = 0; i < invocations->count; ++i) {
    const loom_pipeline_resources_t* resources =
        invocations->entries[i]->resources;
    owners[i] = resources->allocation_count ? &allocations[next] : NULL;
    for (iree_host_size_t a = 0; a < resources->allocation_count; ++a) {
      const loom_pipeline_resource_allocation_t* source =
          &resources->allocations[a];
      allocations[next + a] =
          (loom_pipeline_storage_allocation_t){.source = source};
      requests[next + source->construction_order] =
          (loom_pipeline_storage_request_t){
              .packing = resources->pools[source->pool_index].packing,
              .allocation = &allocations[next + a],
              .owner = i};
    }
    next += resources->allocation_count;
  }
  if (!allocation_count) {
    return iree_ok_status();
  }
  // Without serial lifetimes, preserve the existing individual first-fit
  // placements, including holes between fixed transport/service reservations.
  // There is no subtree sizing or placement work to perform in this case.
  if (!invocations->lifetimes.has_serial) {
    for (iree_host_size_t a = 0; a < allocation_count; ++a) {
      loom_pipeline_storage_allocation_t* allocation = requests[a].allocation;
      IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
          requests[a].packing, allocation->source->byte_length,
          allocation->source->byte_alignment, NULL, 0,
          &allocation->byte_offset));
    }
    return iree_ok_status();
  }
  loom_pipeline_storage_sort_requests(requests, allocation_count);
  iree_host_size_t pool_count = 0;
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    if (!a || requests[a].packing != requests[a - 1].packing) {
      ++pool_count;
    }
    requests[a].pool = pool_count - 1;
  }

  loom_pipeline_lifetime_request_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, allocation_count, sizeof(*ranges), (void**)&ranges));
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    const loom_pipeline_storage_request_t* request = &requests[a];
    const loom_pipeline_resource_allocation_t* source =
        request->allocation->source;
    ranges[a] = (loom_pipeline_lifetime_request_t){
        .lifetime = invocations->lifetimes.owners[request->owner],
        .pool = request->pool,
        .length = source->byte_length,
        .alignment = source->byte_alignment};
  }
  const loom_pipeline_lifetime_extent_t* extents = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout(
      &invocations->lifetimes, ranges, allocation_count, pool_count, arena,
      &extents));
  uint64_t pool_offset = 0;
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    const loom_pipeline_storage_request_t* request = &requests[a];
    if (!a || request->pool != requests[a - 1].pool) {
      const loom_pipeline_lifetime_extent_t* extent = &extents[request->pool];
      IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
          request->packing, extent->length, extent->alignment, NULL, 0,
          &pool_offset));
    }
    request->allocation->byte_offset = pool_offset + ranges[a].offset;
  }
  return iree_ok_status();
}
