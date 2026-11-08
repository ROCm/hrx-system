// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/construction_storage.h"

#include "loom/analysis/pipeline/invocations.h"
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

typedef struct loom_pipeline_storage_segment_t {
  // Complete extent of this lifetime subtree in one backing store.
  uint64_t byte_length;
  // Strongest allocation alignment in this subtree, or one when empty.
  uint64_t byte_alignment;
  // Offset relative to the parent during measurement, absolute after placement.
  uint64_t byte_offset;
} loom_pipeline_storage_segment_t;

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

static iree_status_t loom_pipeline_storage_append(
    loom_pipeline_storage_segment_t* segment, uint64_t byte_length,
    uint64_t byte_alignment, uint64_t* out_offset) {
  if (!byte_length) {
    *out_offset = 0;
    segment->byte_alignment = iree_max(segment->byte_alignment, byte_alignment);
    return iree_ok_status();
  }
  if (!iree_checked_align_u64(segment->byte_length, byte_alignment,
                              out_offset) ||
      !iree_checked_add_u64(*out_offset, byte_length, &segment->byte_length) ||
      segment->byte_length > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline storage extent exceeds INT64_MAX");
  }
  segment->byte_alignment = iree_max(segment->byte_alignment, byte_alignment);
  return iree_ok_status();
}

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

  const loom_pipeline_lifetimes_t* lifetimes = &invocations->lifetimes;
  iree_host_size_t* owner_nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->count, sizeof(*owner_nodes), (void**)&owner_nodes));
  loom_pipeline_storage_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, lifetimes->count,
                                                 pool_count * sizeof(*segments),
                                                 (void**)&segments));
  for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
    if (lifetimes->nodes[i].kind == LOOM_PIPELINE_LIFETIME_OWNER) {
      owner_nodes[lifetimes->nodes[i].owner] = i;
    }
    for (iree_host_size_t p = 0; p < pool_count; ++p) {
      segments[i * pool_count + p] =
          (loom_pipeline_storage_segment_t){.byte_alignment = 1};
    }
  }
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    const loom_pipeline_storage_request_t* request = &requests[a];
    loom_pipeline_storage_allocation_t* allocation = request->allocation;
    IREE_RETURN_IF_ERROR(loom_pipeline_storage_append(
        &segments[owner_nodes[request->owner] * pool_count + request->pool],
        allocation->source->byte_length, allocation->source->byte_alignment,
        &allocation->byte_offset));
  }
  // Children precede their parents in this reverse fold. Relative offsets are
  // retained now; final placement only translates them, without repacking.
  for (iree_host_size_t i = lifetimes->count; i-- > 0;) {
    const loom_pipeline_lifetime_t* node = &lifetimes->nodes[i];
    if (node->kind == LOOM_PIPELINE_LIFETIME_OWNER) {
      continue;
    }
    for (iree_host_size_t c = node->first_child; c != IREE_HOST_SIZE_MAX;
         c = lifetimes->nodes[c].next_sibling) {
      for (iree_host_size_t p = 0; p < pool_count; ++p) {
        loom_pipeline_storage_segment_t* parent = &segments[i * pool_count + p];
        loom_pipeline_storage_segment_t* child = &segments[c * pool_count + p];
        if (node->kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
          parent->byte_length =
              iree_max(parent->byte_length, child->byte_length);
          parent->byte_alignment =
              iree_max(parent->byte_alignment, child->byte_alignment);
        } else {
          IREE_RETURN_IF_ERROR(loom_pipeline_storage_append(
              parent, child->byte_length, child->byte_alignment,
              &child->byte_offset));
        }
      }
    }
  }
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    if (a && requests[a].pool == requests[a - 1].pool) {
      continue;
    }
    loom_pipeline_storage_segment_t* root = &segments[requests[a].pool];
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        requests[a].packing, root->byte_length, root->byte_alignment, NULL, 0,
        &root->byte_offset));
  }
  for (iree_host_size_t i = 1; i < lifetimes->count; ++i) {
    const iree_host_size_t parent = lifetimes->nodes[i].parent;
    for (iree_host_size_t p = 0; p < pool_count; ++p) {
      segments[i * pool_count + p].byte_offset +=
          segments[parent * pool_count + p].byte_offset;
    }
  }
  for (iree_host_size_t a = 0; a < allocation_count; ++a) {
    const loom_pipeline_storage_request_t* request = &requests[a];
    request->allocation->byte_offset +=
        segments[owner_nodes[request->owner] * pool_count + request->pool]
            .byte_offset;
  }
  return iree_ok_status();
}
