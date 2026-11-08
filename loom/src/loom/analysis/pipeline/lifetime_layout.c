// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_layout.h"

typedef struct loom_pipeline_lifetime_segment_t {
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

iree_status_t loom_pipeline_lifetime_layout(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_request_t* requests, iree_host_size_t request_count,
    iree_host_size_t pool_count, iree_arena_allocator_t* arena,
    const loom_pipeline_lifetime_extent_t** out_extents) {
  loom_pipeline_lifetime_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, lifetimes->count,
                                                 pool_count * sizeof(*segments),
                                                 (void**)&segments));
  for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
    for (iree_host_size_t p = 0; p < pool_count; ++p) {
      segments[i * pool_count + p] =
          (loom_pipeline_lifetime_segment_t){.extent = {.alignment = 1}};
    }
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    loom_pipeline_lifetime_request_t* request = &requests[i];
    IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
        &segments[request->lifetime * pool_count + request->pool].extent,
        request->length, request->alignment, &request->offset));
  }
  // Children precede parents in this reverse fold. Retain their relative
  // offsets now; final placement only translates them, without repacking.
  for (iree_host_size_t i = lifetimes->count; i-- > 0;) {
    const loom_pipeline_lifetime_t* node = &lifetimes->nodes[i];
    for (iree_host_size_t c = node->first_child; c != IREE_HOST_SIZE_MAX;
         c = lifetimes->nodes[c].next_sibling) {
      for (iree_host_size_t p = 0; p < pool_count; ++p) {
        loom_pipeline_lifetime_segment_t* parent =
            &segments[i * pool_count + p];
        loom_pipeline_lifetime_segment_t* child = &segments[c * pool_count + p];
        if (node->kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
          parent->extent.length =
              iree_max(parent->extent.length, child->extent.length);
          parent->extent.alignment =
              iree_max(parent->extent.alignment, child->extent.alignment);
        } else {
          IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append_range(
              &parent->extent, child->extent.length, child->extent.alignment,
              &child->offset));
        }
      }
    }
  }
  loom_pipeline_lifetime_extent_t* extents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pool_count, sizeof(*extents), (void**)&extents));
  *out_extents = extents;
  for (iree_host_size_t p = 0; p < pool_count; ++p) {
    extents[p] = segments[p].extent;
  }
  for (iree_host_size_t i = 1; i < lifetimes->count; ++i) {
    const iree_host_size_t parent = lifetimes->nodes[i].parent;
    for (iree_host_size_t p = 0; p < pool_count; ++p) {
      segments[i * pool_count + p].offset +=
          segments[parent * pool_count + p].offset;
    }
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    loom_pipeline_lifetime_request_t* request = &requests[i];
    request->offset +=
        segments[request->lifetime * pool_count + request->pool].offset;
  }
  return iree_ok_status();
}
