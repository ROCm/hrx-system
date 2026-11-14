// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"

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
    groups[function_groups[i]] =
        (loom_aie2p_program_layout_group_t){.owner = function_owners[i]};
  }
  out_layout->function_groups = function_groups;
  out_layout->groups = groups;
  out_layout->group_count = group_count;
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_layout_finish(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_aie2p_program_code_group_t* native_groups,
    iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_layout_t* layout) {
  const iree_host_size_t group_count = layout->group_count;
  loom_aie2p_program_layout_group_t* groups = layout->groups;
  loom_pipeline_lifetime_request_t* requests = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, group_count, sizeof(*requests), (void**)&requests));
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    groups[i].native = native_groups[i];
    requests[i] = (loom_pipeline_lifetime_request_t){
        .lifetime = lifetimes->owners[groups[i].owner],
        .length = iree_host_align(native_groups[i].code.byte_length, 4),
        .alignment = native_groups[i].code.minimum_alignment,
    };
  }
  const loom_pipeline_lifetime_extent_t* extents = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout(
      lifetimes, requests, group_count, 1, scratch_arena, &extents));
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    groups[i].offset = requests[i].offset;
  }
  layout->byte_length = extents[0].length;
  return iree_ok_status();
}
