// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_retention.h"

#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_lifetime_retention_candidate_t {
  // Stable input identity, also the tie-break for equal removed volume.
  iree_host_size_t item;
  // Total allocation units removed from nonresident groups.
  uint64_t saving;
  // Whether the original membership already includes the resident group.
  bool resident;
} loom_pipeline_lifetime_retention_candidate_t;

static bool loom_pipeline_lifetime_retention_less(
    const loom_pipeline_lifetime_retention_candidate_t* a,
    const loom_pipeline_lifetime_retention_candidate_t* b) {
  return a->saving != b->saving ? a->saving > b->saving : a->item < b->item;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_lifetime_retention_sort,
                          loom_pipeline_lifetime_retention_candidate_t,
                          loom_pipeline_lifetime_retention_less)

iree_status_t loom_pipeline_lifetime_retain(
    loom_pipeline_lifetime_projection_t* projection,
    loom_pipeline_lifetime_request_t* requests, iree_host_size_t request_count,
    const loom_pipeline_lifetime_retention_item_t* items,
    iree_host_size_t item_count,
    const loom_pipeline_lifetime_retention_policy_t* policy,
    iree_arena_allocator_t* arena, bool* out_retained) {
  loom_pipeline_lifetime_retention_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, item_count, sizeof(*candidates), (void**)&candidates));
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    requests[i].length = 0;
    requests[i].alignment = 1;
  }
  for (iree_host_size_t i = 0; i < item_count; ++i) {
    out_retained[i] = false;
    candidates[i] = (loom_pipeline_lifetime_retention_candidate_t){.item = i};
    const loom_pipeline_lifetime_retention_item_t* item = &items[i];
    for (iree_host_size_t j = 0; j < item->group_count; ++j) {
      const iree_host_size_t group = item->groups[j];
      requests[group].length += item->length;
      candidates[i].resident |= group == policy->resident_group;
      candidates[i].saving +=
          group == policy->resident_group ? 0 : item->length;
    }
  }
  loom_pipeline_lifetime_retention_sort(candidates, item_count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < item_count && iree_status_is_ok(status);
       ++i) {
    const loom_pipeline_lifetime_retention_candidate_t* candidate =
        &candidates[i];
    if (!candidate->saving) {
      break;
    }
    const loom_pipeline_lifetime_retention_item_t* item =
        &items[candidate->item];
    bool preserves_groups = true;
    for (iree_host_size_t j = 0; j < item->group_count; ++j) {
      const iree_host_size_t group = item->groups[j];
      preserves_groups &=
          group == policy->resident_group ||
          requests[group].length - item->length >= policy->minimum_group_length;
    }
    if (!preserves_groups) {
      continue;
    }
    if (!candidate->resident) {
      requests[policy->resident_group].length += item->length;
    }
    for (iree_host_size_t j = 0; j < item->group_count; ++j) {
      const iree_host_size_t group = item->groups[j];
      if (group != policy->resident_group) {
        requests[group].length -= item->length;
      }
    }
    loom_pipeline_lifetime_extent_t extent;
    status = loom_pipeline_lifetime_projection_evaluate(projection, requests,
                                                        &extent);
    const bool accepted =
        iree_status_is_ok(status) && extent.length <= policy->capacity;
    out_retained[candidate->item] = accepted;
    if (!accepted) {
      if (!candidate->resident) {
        requests[policy->resident_group].length -= item->length;
      }
      for (iree_host_size_t j = 0; j < item->group_count; ++j) {
        const iree_host_size_t group = item->groups[j];
        if (group != policy->resident_group) {
          requests[group].length += item->length;
        }
      }
    }
  }
  return status;
}
