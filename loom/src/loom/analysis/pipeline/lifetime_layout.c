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
  // First request and child-list head during projection; parent-relative and
  // then absolute pool offset during each numeric evaluation.
  uint64_t offset;
} loom_pipeline_lifetime_segment_t;

struct loom_pipeline_lifetime_projection_t {
  // Indexed source lifetimes whose kinds remain fixed through evaluation.
  const loom_pipeline_lifetimes_t* lifetimes;
  // Mutable numeric workspace attached to the immutable projected parents.
  loom_pipeline_lifetime_segment_t* segments;
  // Number of occupied projected nodes, bounded by twice the request count.
  iree_host_size_t segment_count;
  // Requests in owner preorder, preserving input order within each owner.
  const iree_host_size_t* request_order;
  // Projected node for each entry in request_order.
  iree_host_size_t* placements;
  // Number of requests in the projection.
  iree_host_size_t request_count;
  // Children before parents, with concurrent siblings in first-request order.
  iree_host_size_t* postorder;
};

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

static bool loom_pipeline_lifetime_segment_later(
    const loom_pipeline_lifetime_segment_t* segments, const iree_host_size_t* a,
    const iree_host_size_t* b) {
  return segments[*a].offset > segments[*b].offset;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_pipeline_lifetime_sort_segments,
                                       iree_host_size_t,
                                       const loom_pipeline_lifetime_segment_t*,
                                       loom_pipeline_lifetime_segment_later)

// Orders each projected sibling set by its earliest request. The temporary
// reverse priority order builds child lists; its consumed storage then holds
// a reusable postorder. Neither evaluation nor offset translation severs edges.
static void loom_pipeline_lifetime_order_projection(
    loom_pipeline_lifetime_segment_t* segments, iree_host_size_t* order,
    iree_host_size_t count, iree_host_size_t root,
    iree_host_size_t* next_siblings) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    segments[order[i]].offset = IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t child = order[i];
    const iree_host_size_t parent = segments[child].parent;
    if (parent != IREE_HOST_SIZE_MAX) {
      next_siblings[child] = segments[parent].offset;
      segments[parent].offset = child;
    }
  }
  iree_host_size_t node = root;
  iree_host_size_t position = 0;
  bool descend = true;
  while (true) {
    if (descend) {
      while (segments[node].offset != IREE_HOST_SIZE_MAX) {
        node = segments[node].offset;
      }
    }
    order[position++] = node;
    const iree_host_size_t parent = segments[node].parent;
    if (parent == IREE_HOST_SIZE_MAX) {
      break;
    }
    const iree_host_size_t next = next_siblings[node];
    descend = next != IREE_HOST_SIZE_MAX;
    node = descend ? next : parent;
  }
}

// Source-ordered requests already finish in placement order. Reordered input
// retains each subtree's earliest request until its sibling set is complete.
static void loom_pipeline_lifetime_finish_segment(
    loom_pipeline_lifetime_projection_t* projection, iree_host_size_t child,
    bool request_order, iree_host_size_t* position) {
  if (request_order) {
    loom_pipeline_lifetime_segment_t* current = &projection->segments[child];
    loom_pipeline_lifetime_segment_t* parent =
        &projection->segments[current->parent];
    parent->offset = iree_min(parent->offset, current->offset);
  } else {
    projection->postorder[(*position)++] = child;
  }
}

static void loom_pipeline_lifetime_project(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_pipeline_lifetime_request_t* requests,
    const iree_host_size_t* order, iree_host_size_t count,
    iree_host_size_t* placements, loom_pipeline_lifetime_segment_t* segments,
    iree_host_size_t* postorder, bool request_order,
    iree_host_size_t* next_siblings,
    loom_pipeline_lifetime_projection_t* out_projection) {
  *out_projection = (loom_pipeline_lifetime_projection_t){
      .lifetimes = lifetimes,
      .segments = segments,
      .request_order = order,
      .placements = placements,
      .request_count = count,
      .postorder = postorder,
  };
  iree_host_size_t position = 0;
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
        loom_pipeline_lifetime_finish_segment(out_projection, child,
                                              request_order, &position);
        child = parent;
        parent = segments[child].parent;
      }
      if (parent == IREE_HOST_SIZE_MAX ||
          segments[parent].lifetime != ancestor) {
        const iree_host_size_t branch = segment_count++;
        segments[branch] = (loom_pipeline_lifetime_segment_t){
            .lifetime = ancestor,
            .parent = parent,
            .extent = {.alignment = 1},
            .offset = request_order ? IREE_HOST_SIZE_MAX : 0};
        segments[child].parent = branch;
        parent = branch;
      }
      loom_pipeline_lifetime_finish_segment(out_projection, child,
                                            request_order, &position);
    }
    previous = segment_count++;
    segments[previous] = (loom_pipeline_lifetime_segment_t){
        .lifetime = lifetime,
        .parent = parent,
        .extent = {.alignment = 1},
        .offset = request_order ? order[i] : 0};
    do {
      placements[i] = previous;
      ++i;
    } while (i < count && requests[order[i]].lifetime == lifetime);
  }
  while (segments[previous].parent != IREE_HOST_SIZE_MAX) {
    loom_pipeline_lifetime_finish_segment(out_projection, previous,
                                          request_order, &position);
    previous = segments[previous].parent;
  }
  if (request_order) {
    for (iree_host_size_t i = 0; i < segment_count; ++i) {
      postorder[i] = i;
    }
    loom_pipeline_lifetime_sort_segments(segments, postorder, segment_count);
    loom_pipeline_lifetime_order_projection(segments, postorder, segment_count,
                                            previous, next_siblings);
  } else {
    postorder[position] = previous;
  }
  out_projection->segment_count = segment_count;
}

iree_status_t loom_pipeline_lifetime_projection_evaluate(
    loom_pipeline_lifetime_projection_t* projection,
    loom_pipeline_lifetime_request_t* requests,
    loom_pipeline_lifetime_extent_t* out_extent) {
  *out_extent = (loom_pipeline_lifetime_extent_t){.alignment = 1};
  if (!projection->request_count) {
    return iree_ok_status();
  }
  loom_pipeline_lifetime_segment_t* segments = projection->segments;
  for (iree_host_size_t i = 0; i < projection->segment_count; ++i) {
    segments[i].extent = (loom_pipeline_lifetime_extent_t){.alignment = 1};
    segments[i].offset = 0;
  }
  for (iree_host_size_t i = 0; i < projection->request_count; ++i) {
    loom_pipeline_lifetime_request_t* request =
        &requests[projection->request_order[i]];
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
        &segments[projection->placements[i]].extent, request->length,
        request->alignment, &request->offset));
  }
  const iree_host_size_t last = projection->segment_count - 1;
  for (iree_host_size_t i = 0; i < last; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_fold(
        projection->lifetimes, segments, projection->postorder[i]));
  }
  *out_extent = segments[projection->postorder[last]].extent;
  for (iree_host_size_t i = last; i-- > 0;) {
    loom_pipeline_lifetime_segment_t* node =
        &segments[projection->postorder[i]];
    node->offset += segments[node->parent].offset;
  }
  for (iree_host_size_t i = 0; i < projection->request_count; ++i) {
    requests[projection->request_order[i]].offset +=
        segments[projection->placements[i]].offset;
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_lifetime_projection_prepare(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_pipeline_lifetime_request_t* requests,
    iree_host_size_t request_count, iree_arena_allocator_t* arena,
    loom_pipeline_lifetime_projection_t** out_projection) {
  loom_pipeline_lifetime_projection_t* projection = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*projection), (void**)&projection));
  if (!request_count) {
    *projection = (loom_pipeline_lifetime_projection_t){.lifetimes = lifetimes};
    *out_projection = projection;
    return iree_ok_status();
  }
  loom_pipeline_lifetime_segment_t* segments = NULL;
  iree_host_size_t* indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, request_count, 2 * sizeof(*segments), (void**)&segments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, request_count, 6 * sizeof(*indices), (void**)&indices));
  iree_host_size_t* order = indices;
  iree_host_size_t* placements = indices + request_count;
  iree_host_size_t* postorder = placements + request_count;
  iree_host_size_t* siblings = postorder + 2 * request_count;
  bool reordered = false;
  iree_host_size_t previous = 0;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    order[i] = i;
    const iree_host_size_t preorder =
        lifetimes->ancestry[requests[i].lifetime].preorder;
    reordered |= preorder < previous;
    previous = preorder;
  }
  if (reordered) {
    const loom_pipeline_lifetime_order_t context = {lifetimes->ancestry,
                                                    requests};
    loom_pipeline_lifetime_sort_requests(&context, order, request_count);
  }
  loom_pipeline_lifetime_project(lifetimes, requests, order, request_count,
                                 placements, segments, postorder, reordered,
                                 siblings, projection);
  *out_projection = projection;
  return iree_ok_status();
}

typedef struct loom_pipeline_lifetime_column_t {
  union {
    // One pool's complete subtree footprint after its children are folded.
    loom_pipeline_lifetime_extent_t extent;
    struct {
      // First requested child, encoded as its lifetime index plus one.
      uint64_t first;
      // Last requested child, used to append in first-request order.
      uint64_t last;
    } children;
  };
  // Next sibling while collecting ordered children, then a parent-relative
  // offset and finally an absolute pool offset. UINT64_MAX marks unvisited.
  uint64_t offset;
} loom_pipeline_lifetime_column_t;

// The first request in each subtree fixes its sibling position. Every edge
// is attached once; later requests stop at their already-attached owner.
// Branches reuse their as-yet-unmeasured extent for the child list.
static void loom_pipeline_lifetime_column_attach(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_column_t* columns, iree_host_size_t width,
    iree_host_size_t pool, iree_host_size_t node) {
  while (node != IREE_HOST_SIZE_MAX &&
         columns[node * width + pool].offset == UINT64_MAX) {
    columns[node * width + pool].offset = 0;
    const iree_host_size_t parent = lifetimes->nodes[node].parent;
    if (parent != IREE_HOST_SIZE_MAX) {
      loom_pipeline_lifetime_column_t* column = &columns[parent * width + pool];
      if (column->children.last) {
        columns[(column->children.last - 1) * width + pool].offset = node + 1;
      } else {
        column->children.first = node + 1;
      }
      column->children.last = node + 1;
    }
    node = parent;
  }
}

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
    iree_host_size_t width, iree_host_size_t request_count, bool request_order,
    loom_pipeline_lifetime_column_t* columns,
    loom_pipeline_lifetime_extent_t* extents) {
  const iree_host_size_t column_count = lifetimes->count * width;
  const bool collect_order = request_order;
  if (collect_order) {
    for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
      const loom_pipeline_lifetime_column_t initial = {
          .extent = {.alignment = lifetimes->nodes[i].kind ==
                                  LOOM_PIPELINE_LIFETIME_OWNER},
          .offset = UINT64_MAX};
      for (iree_host_size_t p = 0; p < width; ++p) {
        columns[i * width + p] = initial;
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < column_count; ++i) {
      columns[i] =
          (loom_pipeline_lifetime_column_t){.extent = {.alignment = 1}};
    }
  }
  if (!offsets) {
    iree_host_size_t previous[32] = {0};
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      loom_pipeline_lifetime_request_t* request = &requests[i];
      const iree_host_size_t preorder =
          lifetimes->ancestry[request->lifetime].preorder;
      request_order |= preorder < previous[request->pool];
      previous[request->pool] = preorder;
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
          &columns[request->lifetime * width + request->pool].extent,
          request->length, request->alignment, &request->offset));
    }
  } else {
    for (iree_host_size_t p = 0; p < width; ++p) {
      for (iree_host_size_t i = offsets[pools[p]]; i < offsets[pools[p] + 1];
           ++i) {
        loom_pipeline_lifetime_request_t* request = &requests[order[i]];
        if (collect_order) {
          loom_pipeline_lifetime_column_attach(lifetimes, columns, width, p,
                                               request->lifetime);
        }
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
            &columns[request->lifetime * width + p].extent, request->length,
            request->alignment, &request->offset));
      }
    }
  }
  if (request_order) {
    if (!collect_order) {
      // Owner extents are complete. Unmeasured branch extents now hold child
      // lists; no additional per-lifetime scratch is needed for ordering.
      for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
        for (iree_host_size_t p = 0; p < width; ++p) {
          loom_pipeline_lifetime_column_t* column = &columns[i * width + p];
          if (lifetimes->nodes[i].kind != LOOM_PIPELINE_LIFETIME_OWNER) {
            column->children.first = column->children.last = 0;
          }
          column->offset = UINT64_MAX;
        }
      }
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        loom_pipeline_lifetime_column_attach(
            lifetimes, columns, width, requests[i].pool, requests[i].lifetime);
      }
    }
    for (iree_host_size_t i = lifetimes->count; i-- > 0;) {
      const loom_pipeline_lifetime_t* node = &lifetimes->nodes[i];
      if (node->kind == LOOM_PIPELINE_LIFETIME_OWNER) {
        continue;
      }
      for (iree_host_size_t p = 0; p < width; ++p) {
        loom_pipeline_lifetime_column_t* parent = &columns[i * width + p];
        uint64_t child = parent->children.first;
        parent->extent = (loom_pipeline_lifetime_extent_t){.alignment = 1};
        while (child) {
          loom_pipeline_lifetime_column_t* current =
              &columns[(child - 1) * width + p];
          child = current->offset;
          current->offset = 0;
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
    // Unrequested lifetimes have no range to translate in the forward pass.
    for (iree_host_size_t i = 0; i < column_count; ++i) {
      if (columns[i].offset == UINT64_MAX) {
        columns[i].offset = 0;
      }
    }
  } else {
    for (iree_host_size_t i = lifetimes->count; i-- > 0;) {
      const loom_pipeline_lifetime_t* node = &lifetimes->nodes[i];
      for (iree_host_size_t child = node->first_child;
           child != IREE_HOST_SIZE_MAX;
           child = lifetimes->nodes[child].next_sibling) {
        for (iree_host_size_t p = 0; p < width; ++p) {
          loom_pipeline_lifetime_column_t* parent = &columns[i * width + p];
          loom_pipeline_lifetime_column_t* current =
              &columns[child * width + p];
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
  }
  for (iree_host_size_t p = 0; p < width; ++p) {
    extents[pools ? pools[p] : p] = columns[p].extent;
    columns[p].offset = 0;
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
        lifetimes, requests, NULL, NULL, NULL, pool_count, request_count, false,
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
  enum {
    LOOM_LIFETIME_PROJECT_SOURCE_ORDER,
    LOOM_LIFETIME_PROJECT_REQUEST_ORDER,
    LOOM_LIFETIME_DENSE_REQUEST_ORDER,
    LOOM_LIFETIME_DENSE_COLUMNS,
  };
  iree_host_size_t capacity = 0, priority_capacity = 0, dense_count_all = 0;
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    const iree_host_size_t count = offsets[p + 1] - offsets[p];
    bool reordered = false;
    iree_host_size_t previous = 0;
    for (iree_host_size_t i = offsets[p]; count > 1 && i < offsets[p + 1];
         ++i) {
      const iree_host_size_t preorder =
          lifetimes->ancestry[requests[order[i]].lifetime].preorder;
      reordered |= preorder < previous;
      previous = preorder;
    }
    if (reordered && count >= dense_count) {
      cursors[p] = LOOM_LIFETIME_DENSE_REQUEST_ORDER;
      ++dense_count_all;
    } else if (reordered) {
      cursors[p] = LOOM_LIFETIME_PROJECT_REQUEST_ORDER;
      priority_capacity = iree_max(priority_capacity, count);
      capacity = iree_max(capacity, count);
    } else if (count && count >= dense_count) {
      cursors[p] = LOOM_LIFETIME_DENSE_COLUMNS;
      ++dense_count_all;
    } else {
      cursors[p] = LOOM_LIFETIME_PROJECT_SOURCE_ORDER;
      capacity = iree_max(capacity, count);
    }
  }
  loom_pipeline_lifetime_segment_t* segments = NULL;
  iree_host_size_t* placements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, 2 * sizeof(*segments), (void**)&segments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*placements), (void**)&placements));
  iree_host_size_t* postorder = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, capacity + priority_capacity,
                                2 * sizeof(*postorder), (void**)&postorder));
  iree_host_size_t* next_siblings =
      priority_capacity ? postorder + 2 * capacity : NULL;
  const loom_pipeline_lifetime_order_t context = {lifetimes->ancestry,
                                                  requests};
  const iree_host_size_t width = iree_min((iree_host_size_t)8, dense_count_all);
  loom_pipeline_lifetime_column_t* columns = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lifetimes->count, width * sizeof(*columns), (void**)&columns));
  iree_host_size_t request_pools[8];
  iree_host_size_t request_pool_count = 0, dense_pool_count = 0;
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    const iree_host_size_t count = offsets[p + 1] - offsets[p];
    const iree_host_size_t mode = cursors[p];
    if (mode == LOOM_LIFETIME_DENSE_COLUMNS) {
      // Compact finalized modes into the dense-pool list. This cursor never
      // overtakes p, so unread modes remain intact.
      cursors[dense_pool_count++] = p;
      continue;
    }
    if (!count) {
      continue;
    }
    // A lone request has no coexistence or preference edges to project.
    if (count == 1) {
      loom_pipeline_lifetime_request_t* request = &requests[order[offsets[p]]];
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
          &extents[p], request->length, request->alignment, &request->offset));
      continue;
    }
    if (mode == LOOM_LIFETIME_DENSE_REQUEST_ORDER) {
      request_pools[request_pool_count++] = p;
      if (request_pool_count == width) {
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout_columns(
            lifetimes, requests, order, offsets, request_pools, width, 0, true,
            columns, extents));
        request_pool_count = 0;
      }
      continue;
    }
    const bool request_order = mode == LOOM_LIFETIME_PROJECT_REQUEST_ORDER;
    if (request_order) {
      loom_pipeline_lifetime_sort_requests(&context, order + offsets[p], count);
    }
    loom_pipeline_lifetime_projection_t projection;
    loom_pipeline_lifetime_project(lifetimes, requests, order + offsets[p],
                                   count, placements, segments, postorder,
                                   request_order, next_siblings, &projection);
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_projection_evaluate(
        &projection, requests, &extents[p]));
  }
  if (request_pool_count) {
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout_columns(
        lifetimes, requests, order, offsets, request_pools, request_pool_count,
        0, true, columns, extents));
  }
  for (iree_host_size_t p = 0; p < dense_pool_count; p += width) {
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout_columns(
        lifetimes, requests, order, offsets, cursors + p,
        iree_min(width, dense_pool_count - p), 0, false, columns, extents));
  }
  return iree_ok_status();
}
