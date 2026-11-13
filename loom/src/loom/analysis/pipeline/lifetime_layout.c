// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_layout.h"

#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_lifetime_segment_t {
  // Original lifetime represented by this compressed subtree.
  iree_host_size_t lifetime;
  // Parent in the compressed pool tree, or IREE_HOST_SIZE_MAX at its root.
  iree_host_size_t parent;
  // Complete extent of this lifetime subtree in one resource pool.
  loom_pipeline_lifetime_extent_t extent;
  // Offset relative to the parent during measurement, pool-relative afterward.
  uint64_t offset;
} loom_pipeline_lifetime_segment_t;

static iree_status_t loom_pipeline_lifetime_append_range(
    loom_pipeline_lifetime_extent_t* extent, uint64_t length,
    uint64_t alignment, uint64_t* out_offset) {
  if (!length) {
    *out_offset = 0;
    extent->alignment = iree_max(extent->alignment, alignment);
    return iree_ok_status();
  }
  if (!iree_checked_align_u64(extent->length, alignment, out_offset) ||
      !iree_checked_add_u64(*out_offset, length, &extent->length) ||
      extent->length > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline resource extent exceeds INT64_MAX");
  }
  extent->alignment = iree_max(extent->alignment, alignment);
  return iree_ok_status();
}

typedef struct loom_pipeline_lifetime_order_t {
  // Indexed source order established by lifetime construction.
  const loom_pipeline_lifetime_ancestry_t* ancestry;
  // Requests whose input positions break ties within one owner.
  const loom_pipeline_lifetime_request_t* requests;
} loom_pipeline_lifetime_order_t;

static bool loom_pipeline_lifetime_request_less(
    const loom_pipeline_lifetime_order_t* context, const iree_host_size_t* a,
    const iree_host_size_t* b) {
  const iree_host_size_t first =
      context->ancestry[context->requests[*a].lifetime].preorder;
  const iree_host_size_t second =
      context->ancestry[context->requests[*b].lifetime].preorder;
  return first != second ? first < second : *a < *b;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_pipeline_lifetime_sort_requests,
                                       iree_host_size_t,
                                       const loom_pipeline_lifetime_order_t*,
                                       loom_pipeline_lifetime_request_less)

// Finalizes one completed subtree in source order. The parent already contains
// all preceding siblings. Unary source paths have no extra demand and need no
// segment: the indexed common ancestor is the next relevant boundary.
static iree_status_t loom_pipeline_lifetime_fold(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_segment_t* segments, iree_host_size_t child) {
  loom_pipeline_lifetime_segment_t* current = &segments[child];
  loom_pipeline_lifetime_segment_t* parent = &segments[current->parent];
  if (lifetimes->nodes[parent->lifetime].kind ==
      LOOM_PIPELINE_LIFETIME_SERIAL) {
    parent->extent.length =
        iree_max(parent->extent.length, current->extent.length);
    parent->extent.alignment =
        iree_max(parent->extent.alignment, current->extent.alignment);
    return iree_ok_status();
  }
  return loom_pipeline_lifetime_append_range(
      &parent->extent, current->extent.length, current->extent.alignment,
      &current->offset);
}

static iree_status_t loom_pipeline_lifetime_project(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_request_t* requests, const iree_host_size_t* order,
    iree_host_size_t count, iree_host_size_t* placements,
    loom_pipeline_lifetime_segment_t* segments,
    loom_pipeline_lifetime_extent_t* out_extent) {
  iree_host_size_t segment_count = 0;
  iree_host_size_t previous = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < count;) {
    const iree_host_size_t lifetime = requests[order[i]].lifetime;
    iree_host_size_t parent = IREE_HOST_SIZE_MAX;
    if (previous != IREE_HOST_SIZE_MAX) {
      const iree_host_size_t ancestor = loom_pipeline_lifetimes_common_ancestor(
          lifetimes, segments[previous].lifetime, lifetime);
      const iree_host_size_t preorder = lifetimes->ancestry[ancestor].preorder;
      iree_host_size_t child = previous;
      parent = segments[child].parent;
      while (parent != IREE_HOST_SIZE_MAX) {
        const loom_pipeline_lifetime_ancestry_t* candidate =
            &lifetimes->ancestry[segments[parent].lifetime];
        if (preorder >= candidate->preorder &&
            preorder < candidate->preorder + candidate->subtree_count) {
          break;
        }
        IREE_RETURN_IF_ERROR(
            loom_pipeline_lifetime_fold(lifetimes, segments, child));
        child = parent;
        parent = segments[child].parent;
      }
      if (parent == IREE_HOST_SIZE_MAX ||
          segments[parent].lifetime != ancestor) {
        const iree_host_size_t branch = segment_count++;
        segments[branch] = (loom_pipeline_lifetime_segment_t){
            .lifetime = ancestor, .parent = parent, .extent = {.alignment = 1}};
        segments[child].parent = branch;
        parent = branch;
      }
      IREE_RETURN_IF_ERROR(
          loom_pipeline_lifetime_fold(lifetimes, segments, child));
    }
    previous = segment_count++;
    segments[previous] = (loom_pipeline_lifetime_segment_t){
        .lifetime = lifetime, .parent = parent, .extent = {.alignment = 1}};
    do {
      loom_pipeline_lifetime_request_t* request = &requests[order[i]];
      placements[i] = previous;
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
          &segments[previous].extent, request->length, request->alignment,
          &request->offset));
      ++i;
    } while (i < count && requests[order[i]].lifetime == lifetime);
  }
  while (segments[previous].parent != IREE_HOST_SIZE_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_pipeline_lifetime_fold(lifetimes, segments, previous));
    previous = segments[previous].parent;
  }
  *out_extent = segments[previous].extent;
  // Resolve each relative placement once. Path compression retains absolute
  // offsets at detached roots; repeated owners and deep source nesting do not
  // repeat ancestor work.
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t first = placements[i];
    uint64_t offset = 0;
    iree_host_size_t node = first;
    do {
      offset += segments[node].offset;
      node = segments[node].parent;
    } while (node != IREE_HOST_SIZE_MAX);
    requests[order[i]].offset += offset;
    node = first;
    while (segments[node].parent != IREE_HOST_SIZE_MAX) {
      const iree_host_size_t parent = segments[node].parent;
      const uint64_t relative = segments[node].offset;
      segments[node].offset = offset;
      segments[node].parent = IREE_HOST_SIZE_MAX;
      offset -= relative;
      node = parent;
    }
  }
  return iree_ok_status();
}

typedef struct loom_pipeline_lifetime_column_t {
  // One pool's complete subtree footprint at every source lifetime.
  loom_pipeline_lifetime_extent_t extent;
  // Offset in the parent, translated to a pool offset after the fold.
  uint64_t offset;
} loom_pipeline_lifetime_column_t;

// Dense demand already has work proportional to the source tree. A bounded
// group of columns amortizes topology visits and preserves vectorization;
// scratch storage is independent of the complete resource-pool domain. Keep
// the dense working set out of the sparse planner's register-allocation scope.
// NULL offsets selects the complete compact domain in original request order.
static IREE_ATTRIBUTE_NOINLINE iree_status_t
loom_pipeline_lifetime_layout_columns(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_request_t* requests, const iree_host_size_t* order,
    const iree_host_size_t* offsets, const iree_host_size_t* pools,
    iree_host_size_t width, iree_host_size_t request_count,
    loom_pipeline_lifetime_column_t* columns,
    loom_pipeline_lifetime_extent_t* extents) {
  const iree_host_size_t column_count = lifetimes->count * width;
  for (iree_host_size_t i = 0; i < column_count; ++i) {
    columns[i] = (loom_pipeline_lifetime_column_t){.extent = {.alignment = 1}};
  }
  if (!offsets) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      loom_pipeline_lifetime_request_t* request = &requests[i];
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
          &columns[request->lifetime * width + request->pool].extent,
          request->length, request->alignment, &request->offset));
    }
  } else {
    for (iree_host_size_t p = 0; p < width; ++p) {
      for (iree_host_size_t i = offsets[pools[p]]; i < offsets[pools[p] + 1];
           ++i) {
        loom_pipeline_lifetime_request_t* request = &requests[order[i]];
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
            &columns[request->lifetime * width + p].extent, request->length,
            request->alignment, &request->offset));
      }
    }
  }
  for (iree_host_size_t i = lifetimes->count; i-- > 0;) {
    const loom_pipeline_lifetime_t* node = &lifetimes->nodes[i];
    for (iree_host_size_t child = node->first_child;
         child != IREE_HOST_SIZE_MAX;
         child = lifetimes->nodes[child].next_sibling) {
      for (iree_host_size_t p = 0; p < width; ++p) {
        loom_pipeline_lifetime_column_t* parent = &columns[i * width + p];
        loom_pipeline_lifetime_column_t* current = &columns[child * width + p];
        if (node->kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
          parent->extent.length =
              iree_max(parent->extent.length, current->extent.length);
          parent->extent.alignment =
              iree_max(parent->extent.alignment, current->extent.alignment);
        } else {
          IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
              &parent->extent, current->extent.length,
              current->extent.alignment, &current->offset));
        }
      }
    }
  }
  for (iree_host_size_t p = 0; p < width; ++p) {
    extents[pools ? pools[p] : p] = columns[p].extent;
  }
  for (iree_host_size_t i = 1; i < lifetimes->count; ++i) {
    const iree_host_size_t parent = lifetimes->nodes[i].parent;
    for (iree_host_size_t p = 0; p < width; ++p) {
      columns[i * width + p].offset += columns[parent * width + p].offset;
    }
  }
  if (!offsets) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      loom_pipeline_lifetime_request_t* request = &requests[i];
      request->offset +=
          columns[request->lifetime * width + request->pool].offset;
    }
  } else {
    for (iree_host_size_t p = 0; p < width; ++p) {
      for (iree_host_size_t i = offsets[pools[p]]; i < offsets[pools[p] + 1];
           ++i) {
        loom_pipeline_lifetime_request_t* request = &requests[order[i]];
        request->offset += columns[request->lifetime * width + p].offset;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_lifetime_layout(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_request_t* requests, iree_host_size_t request_count,
    iree_host_size_t pool_count, iree_arena_allocator_t* arena,
    const loom_pipeline_lifetime_extent_t** out_extents) {
  loom_pipeline_lifetime_extent_t* extents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pool_count, sizeof(*extents), (void**)&extents));
  *out_extents = extents;
  for (iree_host_size_t i = 0; i < pool_count; ++i) {
    extents[i] = (loom_pipeline_lifetime_extent_t){.alignment = 1};
  }
  if (!request_count) {
    return iree_ok_status();
  }
  // A compact, heavily used domain fits one bounded scratch block. Keep its
  // requests in their original stream: regrouping them would discard locality.
  // The fixed width and demand bound keep storage/work linear in the inputs.
  if (pool_count <= 32 &&
      request_count / pool_count >=
          lifetimes->count / 4 + (lifetimes->count % 4 != 0)) {
    loom_pipeline_lifetime_column_t* columns = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, lifetimes->count, pool_count * sizeof(*columns),
        (void**)&columns));
    return loom_pipeline_lifetime_layout_columns(
        lifetimes, requests, NULL, NULL, NULL, pool_count, request_count,
        columns, extents);
  }
  // Stable counting groups each pool's actual requests without sorting dense
  // workloads. Only sparse projections need owner-preorder sorting below.
  iree_host_size_t* offsets = NULL;
  iree_host_size_t* cursors = NULL;
  iree_host_size_t* order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pool_count + 1, sizeof(*offsets), (void**)&offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pool_count, sizeof(*cursors), (void**)&cursors));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, request_count, sizeof(*order), (void**)&order));
  memset(offsets, 0, (pool_count + 1) * sizeof(*offsets));
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    ++offsets[requests[i].pool + 1];
  }
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    offsets[p + 1] += offsets[p];
    cursors[p] = offsets[p];
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    order[cursors[requests[i].pool]++] = i;
  }
  // Bound a full-column visit by four source nodes per request. Dense columns
  // therefore still cost O(request count); sparse pools use compressed paths.
  const iree_host_size_t dense_count =
      lifetimes->count / 4 + (lifetimes->count % 4 != 0);
  iree_host_size_t capacity = 0, dense_pool_count = 0;
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    const iree_host_size_t count = offsets[p + 1] - offsets[p];
    if (count >= dense_count) {
      cursors[dense_pool_count++] = p;
    } else {
      capacity = iree_max(capacity, count);
    }
  }
  loom_pipeline_lifetime_segment_t* segments = NULL;
  iree_host_size_t* placements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, 2 * sizeof(*segments), (void**)&segments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*placements), (void**)&placements));
  const loom_pipeline_lifetime_order_t context = {lifetimes->ancestry,
                                                  requests};
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    const iree_host_size_t count = offsets[p + 1] - offsets[p];
    if (!count || count >= dense_count) {
      continue;
    }
    loom_pipeline_lifetime_sort_requests(&context, order + offsets[p], count);
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_project(
        lifetimes, requests, order + offsets[p], count, placements, segments,
        &extents[p]));
  }
  const iree_host_size_t width =
      iree_min((iree_host_size_t)8, dense_pool_count);
  loom_pipeline_lifetime_column_t* columns = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lifetimes->count, width * sizeof(*columns), (void**)&columns));
  for (iree_host_size_t p = 0; p < dense_pool_count; p += width) {
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout_columns(
        lifetimes, requests, order, offsets, cursors + p,
        iree_min(width, dense_pool_count - p), 0, columns, extents));
  }
  return iree_ok_status();
}
