// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/entries.h"

#include <string.h>

const loom_pipeline_entry_scope_type_t loom_pipeline_entry_scope_type = {
    .size = sizeof(loom_pipeline_entry_scope_t),
};

iree_status_t loom_pipeline_entry_scope_allocate(
    const loom_pipeline_entry_scope_type_t* type,
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t owner_count,
    iree_arena_allocator_t* arena, loom_pipeline_entry_scope_t** out_scope) {
  *out_scope = NULL;
  loom_pipeline_entry_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, type->size, (void**)&scope));
  loom_pipeline_lifetime_t* nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lifetimes->count, sizeof(*nodes), (void**)&nodes));
  memcpy(nodes, lifetimes->nodes, lifetimes->count * sizeof(*nodes));
  loom_pipeline_lifetime_ancestry_t* ancestry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lifetimes->count, sizeof(*ancestry), (void**)&ancestry));
  memcpy(ancestry, lifetimes->ancestry, lifetimes->count * sizeof(*ancestry));
  iree_host_size_t* owners = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, owner_count, sizeof(*owners), (void**)&owners));
  memcpy(owners, lifetimes->owners, owner_count * sizeof(*owners));
  uint32_t* loop_depths = NULL;
  if (lifetimes->loop_depths) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, owner_count, sizeof(*loop_depths), (void**)&loop_depths));
    memcpy(loop_depths, lifetimes->loop_depths,
           owner_count * sizeof(*loop_depths));
  }
  *scope = (loom_pipeline_entry_scope_t){
      .type = type,
      .lifetimes = {.nodes = nodes,
                    .count = lifetimes->count,
                    .owners = owners,
                    .ancestry = ancestry,
                    .has_serial = lifetimes->has_serial,
                    .loop_depths = loop_depths},
      .owner_count = owner_count,
  };
  *out_scope = scope;
  return iree_ok_status();
}

iree_status_t loom_pipeline_entry_scope_clone(
    const loom_pipeline_entry_scope_t* source, iree_arena_allocator_t* arena,
    const loom_pipeline_entry_scope_t** out_scope) {
  *out_scope = NULL;
  loom_pipeline_entry_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_entry_scope_allocate(
      source->type, &source->lifetimes, source->owner_count, arena, &scope));
  if (source->type->clone) {
    IREE_RETURN_IF_ERROR(source->type->clone(source, arena, scope));
  }
  *out_scope = scope;
  return iree_ok_status();
}

void loom_pipeline_entry_owners_project(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_pipeline_entry_calls_t* entries, iree_host_size_t entry_count,
    iree_host_size_t* out_owners) {
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    out_owners[i] = entries[i].owner;
  }
  for (iree_host_size_t remaining = entry_count; remaining != 0; --remaining) {
    const iree_host_size_t caller = remaining - 1;
    for (iree_host_size_t i = 0; i < entries[caller].callee_count; ++i) {
      const iree_host_size_t callee = entries[caller].callees[i];
      if (entries[callee].owner != IREE_HOST_SIZE_MAX) {
        continue;
      }
      out_owners[callee] =
          out_owners[callee] == IREE_HOST_SIZE_MAX
              ? out_owners[caller]
              : loom_pipeline_lifetimes_common_owner(
                    lifetimes, out_owners[callee], out_owners[caller]);
    }
  }
}
