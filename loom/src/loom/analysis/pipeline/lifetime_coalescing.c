// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_coalescing.h"

#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_coalescing_scope_t {
  // Enclosing scope after flattening consecutive serial choices.
  iree_host_size_t parent;
  // First child after flattening, or IREE_HOST_SIZE_MAX.
  iree_host_size_t first_child;
  // Next alternative or concurrent sibling, or IREE_HOST_SIZE_MAX.
  iree_host_size_t next_sibling;
  // Minimum simultaneous demand initially; selected peak after placement.
  uint64_t peak;
  // Payload retained at this scope after selecting its children.
  uint64_t retained;
  // Serial fanout during indexing, assigned capacity during selection,
  // and the following alternatives' maximum peak after selection.
  uint64_t budget;
} loom_pipeline_coalescing_scope_t;

typedef struct loom_pipeline_coalescing_order_t {
  // Construction ordinals and loop nesting from the execution schedule.
  const loom_pipeline_lifetimes_t* lifetimes;
  // Already selected peak and resident payload for each alternative.
  const loom_pipeline_coalescing_scope_t* scopes;
} loom_pipeline_coalescing_order_t;

static bool loom_pipeline_alternative_precedes(
    const loom_pipeline_coalescing_order_t* order, const iree_host_size_t* a,
    const iree_host_size_t* b) {
  const iree_host_size_t lhs = order->lifetimes->nodes[*a].owner;
  const iree_host_size_t rhs = order->lifetimes->nodes[*b].owner;
  const uint32_t* depths = order->lifetimes->loop_depths;
  if (depths && depths[lhs] != depths[rhs]) {
    return depths[lhs] > depths[rhs];
  }
  const uint64_t left = order->scopes[*a].retained;
  const uint64_t right = order->scopes[*b].retained;
  return left != right ? left > right : lhs < rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_pipeline_sort_alternatives,
                                       iree_host_size_t,
                                       const loom_pipeline_coalescing_order_t*,
                                       loom_pipeline_alternative_precedes)

static void loom_pipeline_coalesce_serial(
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t index,
    loom_pipeline_coalescing_scope_t* scopes, iree_host_size_t* order,
    iree_host_size_t* owners) {
  loom_pipeline_coalescing_scope_t* scope = &scopes[index];
  iree_host_size_t count = 0;
  for (iree_host_size_t c = scope->first_child; c != IREE_HOST_SIZE_MAX;
       c = scopes[c].next_sibling) {
    order[count++] = c;
  }
  const loom_pipeline_coalescing_order_t priority = {lifetimes, scopes};
  loom_pipeline_sort_alternatives(&priority, order, count);
  uint64_t suffix = 0;
  for (iree_host_size_t i = count; i > 0; --i) {
    loom_pipeline_coalescing_scope_t* child = &scopes[order[i - 1]];
    child->budget = suffix;
    suffix = iree_max(suffix, child->peak);
  }
  // Promoting a child makes its resident portion coexist with every sibling.
  // The prefix and suffix give the exact worst remaining alternative in O(1).
  uint64_t retained = 0;
  uint64_t remaining = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_pipeline_coalescing_scope_t* child = &scopes[order[i]];
    uint64_t residual = child->peak - child->retained;
    const uint64_t other = iree_max(remaining, child->budget);
    if (retained + child->retained + iree_max(residual, other) <=
        scope->budget) {
      owners[lifetimes->nodes[order[i]].owner] = lifetimes->nodes[index].owner;
      retained += child->retained;
    } else {
      residual = child->peak;
    }
    remaining = iree_max(remaining, residual);
  }
  scope->retained = retained;
  scope->peak = retained + remaining;
}

iree_status_t loom_pipeline_lifetimes_coalesce(
    const loom_pipeline_lifetimes_t* lifetimes, const uint64_t* owner_lengths,
    uint64_t capacity, iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_owners) {
  const loom_pipeline_lifetime_t* nodes = lifetimes->nodes;
  const iree_host_size_t count = lifetimes->count;
  loom_pipeline_coalescing_scope_t* scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*scopes), (void**)&scopes));
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_host_size_t parent = nodes[i].parent;
    if (parent != IREE_HOST_SIZE_MAX &&
        nodes[parent].kind == LOOM_PIPELINE_LIFETIME_SERIAL &&
        scopes[parent].parent != IREE_HOST_SIZE_MAX &&
        nodes[scopes[parent].parent].kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
      parent = scopes[parent].parent;
    }
    uint64_t length = 0;
    if (nodes[i].kind == LOOM_PIPELINE_LIFETIME_OWNER) {
      out_owners[nodes[i].owner] = nodes[i].owner;
      length = owner_lengths[nodes[i].owner];
    }
    scopes[i] =
        (loom_pipeline_coalescing_scope_t){.parent = parent,
                                           .first_child = IREE_HOST_SIZE_MAX,
                                           .peak = length,
                                           .retained = length};
  }
  iree_host_size_t widest = 0;
  for (iree_host_size_t i = count; --i > 0;) {
    loom_pipeline_coalescing_scope_t* child = &scopes[i];
    const iree_host_size_t p = child->parent;
    const bool serial = nodes[p].kind == LOOM_PIPELINE_LIFETIME_SERIAL;
    if (serial && nodes[i].kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
      continue;
    }
    loom_pipeline_coalescing_scope_t* parent = &scopes[p];
    child->next_sibling = parent->first_child;
    parent->first_child = i;
    if (serial) {
      ++parent->budget;
      widest = iree_max(widest, (iree_host_size_t)parent->budget);
    }
    parent->peak = serial ? iree_max(parent->peak, child->peak)
                          : parent->peak + child->peak;
  }
  if (scopes[0].peak > capacity) {
    return iree_ok_status();
  }
  iree_host_size_t* order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, widest, sizeof(*order), (void**)&order));
  // Walk the flattened tree without recursive stack growth. A concurrent
  // scope reserves every sibling's minimum on entry and charges its actual
  // selected peak on return. Unspent headroom reaches the next sibling.
  scopes[0].budget = capacity;
  iree_host_size_t index = 0;
  while (true) {
    if (nodes[index].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT) {
      scopes[index].budget -= scopes[index].peak;
      scopes[index].retained = scopes[index].peak = 0;
    }
    iree_host_size_t child = scopes[index].first_child;
    while (child == IREE_HOST_SIZE_MAX) {
      if (nodes[index].kind == LOOM_PIPELINE_LIFETIME_SERIAL) {
        loom_pipeline_coalesce_serial(lifetimes, index, scopes, order,
                                      out_owners);
      }
      const iree_host_size_t parent = scopes[index].parent;
      if (parent == IREE_HOST_SIZE_MAX) {
        break;
      }
      if (nodes[parent].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT) {
        scopes[parent].budget -= scopes[index].peak;
        scopes[parent].peak += scopes[index].peak;
        scopes[parent].retained += scopes[index].retained;
        if (nodes[index].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT) {
          out_owners[nodes[index].owner] = nodes[parent].owner;
        }
      }
      child = scopes[index].next_sibling;
      index = parent;
    }
    if (child == IREE_HOST_SIZE_MAX) {
      break;
    }
    if (nodes[index].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT) {
      scopes[index].budget += scopes[child].peak;
    }
    scopes[child].budget = scopes[index].budget;
    index = child;
  }
  // The producer numbers lifetime nodes parent first. Compress accepted
  // ancestor chains once so every caller can map a source owner in O(1).
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (nodes[i].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT) {
      const iree_host_size_t owner = nodes[i].owner;
      out_owners[owner] = out_owners[out_owners[owner]];
    }
  }
  return iree_ok_status();
}
