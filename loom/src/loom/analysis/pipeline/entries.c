// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/entries.h"

#include <string.h>

iree_status_t loom_pipeline_entry_scope_clone(
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t owner_count,
    iree_arena_allocator_t* arena,
    const loom_pipeline_entry_scope_t** out_scope) {
  *out_scope = NULL;
  loom_pipeline_entry_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
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
  *scope = (loom_pipeline_entry_scope_t){
      .lifetimes = {.nodes = nodes,
                    .count = lifetimes->count,
                    .owners = owners,
                    .ancestry = ancestry,
                    .has_serial = lifetimes->has_serial},
      .owner_count = owner_count,
  };
  *out_scope = scope;
  return iree_ok_status();
}
