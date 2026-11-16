// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"

#include <string.h>

#include "loom/analysis/pipeline/lifetime_retention.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"
#include "loom/util/adaptive_sort.h"

static bool loom_aie2p_program_layout_owner_less(const iree_host_size_t* owners,
                                                 const iree_host_size_t* a,
                                                 const iree_host_size_t* b) {
  if (owners[*a] == owners[*b]) {
    return *a < *b;
  }
  if (owners[*a] == owners[0]) {
    return true;
  }
  if (owners[*b] == owners[0]) {
    return false;
  }
  return owners[*a] < owners[*b];
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_aie2p_program_layout_sort_owners,
                                       iree_host_size_t,
                                       const iree_host_size_t*,
                                       loom_aie2p_program_layout_owner_less)

iree_status_t loom_aie2p_program_layout_initialize(
    const loom_pipeline_lifetimes_t* lifetimes,
    const iree_host_size_t* function_owners, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_layout_t* out_layout) {
  *out_layout = (loom_aie2p_program_layout_t){0};
  iree_host_size_t* order = NULL;
  iree_host_size_t* function_groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, function_count, sizeof(*order), (void**)&order));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, function_count,
                                                 sizeof(*function_groups),
                                                 (void**)&function_groups));
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    order[i] = i;
  }
  loom_aie2p_program_layout_sort_owners(function_owners, order, function_count);
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    if (i == 0 || function_owners[order[i]] != function_owners[order[i - 1]]) {
      ++group_count;
    }
    function_groups[order[i]] = group_count - 1;
  }
  loom_aie2p_program_layout_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*groups), (void**)&groups));
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    groups[function_groups[i]] = (loom_aie2p_program_layout_group_t){
        .owner = function_owners[i], .request_index = function_groups[i]};
  }
  out_layout->function_groups = function_groups;
  out_layout->groups = groups;
  out_layout->group_count = group_count;
  out_layout->request_count = group_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*out_layout->requests),
      (void**)&out_layout->requests));
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    out_layout->requests[i] = (loom_pipeline_lifetime_request_t){
        .lifetime = lifetimes->owners[groups[i].owner], .alignment = 1};
  }
  return loom_pipeline_lifetime_projection_prepare(
      lifetimes, out_layout->requests, group_count, arena,
      &out_layout->projection);
}

iree_status_t loom_aie2p_program_layout_copy(
    const loom_aie2p_program_layout_t* source, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* out_layout) {
  *out_layout = *source;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function_count, sizeof(*out_layout->function_groups),
      (void**)&out_layout->function_groups));
  memcpy(out_layout->function_groups, source->function_groups,
         function_count * sizeof(*out_layout->function_groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, source->group_count,
                                                 sizeof(*out_layout->groups),
                                                 (void**)&out_layout->groups));
  memcpy(out_layout->groups, source->groups,
         source->group_count * sizeof(*out_layout->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->request_count, sizeof(*out_layout->requests),
      (void**)&out_layout->requests));
  memcpy(out_layout->requests, source->requests,
         source->request_count * sizeof(*out_layout->requests));
  return iree_ok_status();
}

typedef struct loom_aie2p_program_membership_t {
  // Native equivalence identity from the prepared function inventory.
  iree_host_size_t canonical;
  // Original construction group containing this logical use.
  iree_host_size_t group;
} loom_aie2p_program_membership_t;

static bool loom_aie2p_program_membership_less(
    const loom_aie2p_program_membership_t* a,
    const loom_aie2p_program_membership_t* b) {
  return a->canonical != b->canonical ? a->canonical < b->canonical
                                      : a->group < b->group;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_program_sort_memberships,
                          loom_aie2p_program_membership_t,
                          loom_aie2p_program_membership_less)

iree_status_t loom_aie2p_program_layout_retain(
    const loom_aie2p_program_native_t* native, uint64_t minimum_group_length,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* layout) {
  const iree_host_size_t count = native->function_count;
  loom_aie2p_program_membership_t* memberships = NULL;
  loom_pipeline_lifetime_retention_item_t* items = NULL;
  iree_host_size_t* uses = NULL;
  bool* retained = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*memberships), (void**)&memberships));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*items), (void**)&items));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*uses), (void**)&uses));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*retained), (void**)&retained));
  for (iree_host_size_t i = 0; i < count; ++i) {
    memberships[i] = (loom_aie2p_program_membership_t){
        .canonical = native->equivalent_indices[i],
        .group = layout->function_groups[i],
    };
    items[i] = (loom_pipeline_lifetime_retention_item_t){0};
  }
  loom_aie2p_program_sort_memberships(memberships, count);
  // Native emission gives every callable the same entry alignment. Complete
  // aligned units bound storage before composition removes each group's tail
  // padding; exact native placement remains the final capacity authority.
  const uint64_t alignment =
      native->contributions[0]->realization.code.minimum_alignment;
  iree_host_size_t use_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i && memberships[i].canonical == memberships[i - 1].canonical &&
        memberships[i].group == memberships[i - 1].group) {
      continue;
    }
    const iree_host_size_t canonical = memberships[i].canonical;
    loom_pipeline_lifetime_retention_item_t* item = &items[canonical];
    if (!item->group_count) {
      item->groups = uses + use_count;
      item->length =
          iree_host_align(
              native->contributions[canonical]->realization.code.byte_length,
              alignment) /
          alignment;
    }
    uses[use_count++] = memberships[i].group;
    ++item->group_count;
  }
  const loom_pipeline_lifetime_retention_policy_t policy = {
      .resident_group = 0,
      .minimum_group_length = minimum_group_length,
      .capacity = LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE / alignment,
  };
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_retain(
      layout->projection, layout->requests, layout->group_count, items, count,
      &policy, arena, retained));
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (retained[native->equivalent_indices[i]]) {
      layout->function_groups[i] = 0;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_layout_compact(
    iree_host_size_t function_count, iree_arena_allocator_t* arena,
    loom_aie2p_program_layout_t* layout, iree_host_size_t** out_group_indices) {
  iree_host_size_t* indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, layout->group_count, sizeof(*indices), (void**)&indices));
  indices[0] = 0;
  iree_host_size_t count = 1;
  for (iree_host_size_t i = 1; i < layout->group_count; ++i) {
    indices[i] = 0;
    if (layout->requests[layout->groups[i].request_index].length) {
      indices[i] = count;
      layout->groups[count++] = layout->groups[i];
    }
  }
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    layout->function_groups[i] = indices[layout->function_groups[i]];
  }
  layout->group_count = count;
  *out_group_indices = indices;
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_layout_finish(
    const loom_aie2p_program_code_group_t* native_groups,
    loom_aie2p_program_layout_t* layout) {
  const iree_host_size_t group_count = layout->group_count;
  loom_aie2p_program_layout_group_t* groups = layout->groups;
  loom_pipeline_lifetime_request_t* requests = layout->requests;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    groups[i].native = native_groups[i];
    loom_pipeline_lifetime_request_t* request =
        &requests[groups[i].request_index];
    request->length = iree_host_align(native_groups[i].code.byte_length, 4);
    request->alignment = native_groups[i].code.minimum_alignment;
  }
  loom_pipeline_lifetime_extent_t extent;
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_projection_evaluate(
      layout->projection, requests, &extent));
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    groups[i].offset = requests[groups[i].request_index].offset;
  }
  layout->byte_length = extent.length;
  return iree_ok_status();
}
