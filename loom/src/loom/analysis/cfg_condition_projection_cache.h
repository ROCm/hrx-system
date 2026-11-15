// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Memoized condition-set images for one immutable CFG projection domain.

#ifndef LOOM_ANALYSIS_CFG_CONDITION_PROJECTION_CACHE_H_
#define LOOM_ANALYSIS_CFG_CONDITION_PROJECTION_CACHE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/condition_relation_set.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cfg_condition_projection_cache_slot_t
    loom_cfg_condition_projection_cache_slot_t;

// Zero-initialize with the caller's scratch arena. Edges, set IDs, and their
// mappings remain immutable for the cache lifetime; the arena owns all storage.
typedef struct loom_cfg_condition_projection_cache_t {
  // Scratch arena owning every slot-table generation.
  iree_arena_allocator_t* arena;

  // Open-addressed cache slots.
  loom_cfg_condition_projection_cache_slot_t* slots;

  // Power-of-two number of entries in slots.
  uint32_t capacity;

  // Number of populated entries.
  uint32_t count;
} loom_cfg_condition_projection_cache_t;

// Looks up a previously computed image. An empty target set is a valid hit.
bool loom_cfg_condition_projection_cache_lookup(
    const loom_cfg_condition_projection_cache_t* cache,
    loom_cfg_edge_index_t edge, loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t* out_target);

// Retains the image of an absent (edge, source) pair after a lookup miss.
// Entries remain valid through table growth; existing mappings never change.
iree_status_t loom_cfg_condition_projection_cache_insert(
    loom_cfg_condition_projection_cache_t* cache, loom_cfg_edge_index_t edge,
    loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CFG_CONDITION_PROJECTION_CACHE_H_
