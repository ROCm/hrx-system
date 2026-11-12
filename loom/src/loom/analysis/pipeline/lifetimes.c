// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetimes.h"

#include "loom/analysis/pipeline/invocations.h"

iree_status_t loom_pipeline_lifetimes_index(
    iree_arena_allocator_t* arena, loom_pipeline_lifetimes_t* lifetimes) {
  loom_pipeline_lifetime_ancestry_t* ancestry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lifetimes->count, sizeof(*ancestry), (void**)&ancestry));
  const loom_pipeline_lifetime_t* nodes = lifetimes->nodes;
  for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
    ancestry[i] = (loom_pipeline_lifetime_ancestry_t){.subtree_count = 1,
                                                      .chain_head = i};
  }
  for (iree_host_size_t i = lifetimes->count; --i > 0;) {
    ancestry[nodes[i].parent].subtree_count += ancestry[i].subtree_count;
  }
  for (iree_host_size_t i = 0; i < lifetimes->count; ++i) {
    iree_host_size_t largest = nodes[i].first_child;
    if (largest == IREE_HOST_SIZE_MAX) {
      continue;
    }
    for (iree_host_size_t child = nodes[largest].next_sibling;
         child != IREE_HOST_SIZE_MAX; child = nodes[child].next_sibling) {
      if (ancestry[child].subtree_count > ancestry[largest].subtree_count) {
        largest = child;
      }
    }
    ancestry[largest].chain_head = ancestry[i].chain_head;
  }
  iree_host_size_t node = 0;
  for (iree_host_size_t preorder = 0; preorder < lifetimes->count; ++preorder) {
    ancestry[node].preorder = preorder;
    if (nodes[node].first_child != IREE_HOST_SIZE_MAX) {
      node = nodes[node].first_child;
    } else {
      while (node && nodes[node].next_sibling == IREE_HOST_SIZE_MAX) {
        node = nodes[node].parent;
      }
      node = nodes[node].next_sibling;
    }
  }
  lifetimes->ancestry = ancestry;
  return iree_ok_status();
}

iree_host_size_t loom_pipeline_lifetimes_common_ancestor(
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t first,
    iree_host_size_t second) {
  const loom_pipeline_lifetime_ancestry_t* ancestry = lifetimes->ancestry;
  while (ancestry[first].chain_head != ancestry[second].chain_head) {
    const iree_host_size_t first_head = ancestry[first].chain_head;
    const iree_host_size_t second_head = ancestry[second].chain_head;
    if (ancestry[first_head].preorder > ancestry[second_head].preorder) {
      first = lifetimes->nodes[first_head].parent;
    } else {
      second = lifetimes->nodes[second_head].parent;
    }
  }
  return ancestry[first].preorder < ancestry[second].preorder ? first : second;
}

typedef struct loom_pipeline_lifetime_pending_t {
  // Node awaiting expansion in the parent-before-child worklist.
  iree_host_size_t node;
  // Construction for a concurrent node; NULL for a serial node.
  const loom_pipeline_invocation_t* invocation;
  // Bound control for a serial node; NULL for a concurrent node.
  const loom_pipeline_invocation_schedule_t* schedule;
} loom_pipeline_lifetime_pending_t;

typedef struct loom_pipeline_lifetime_builder_t {
  // Arena owning the final tree and construction worklist.
  iree_arena_allocator_t* arena;
  // Nodes already discovered, in parent-before-child order.
  loom_pipeline_lifetime_t* nodes;
  // OWNER node indexed by the original construction ordinal.
  iree_host_size_t* owners;
  // Number of discovered nodes.
  iree_host_size_t count;
  // Allocated node capacity.
  iree_host_size_t capacity;
  // Non-leaf nodes awaiting expansion.
  loom_pipeline_lifetime_pending_t* pending;
  // Number of pending rows, including previously expanded rows.
  iree_host_size_t pending_count;
  // Allocated worklist capacity.
  iree_host_size_t pending_capacity;
  // Retained proof that the tree contains a serial composition.
  bool has_serial;
} loom_pipeline_lifetime_builder_t;

static iree_status_t loom_pipeline_lifetime_append(
    loom_pipeline_lifetime_builder_t* builder, iree_host_size_t parent,
    iree_host_size_t* last_child, loom_pipeline_lifetime_kind_t kind,
    const loom_pipeline_invocation_t* invocation,
    const loom_pipeline_invocation_schedule_t* schedule) {
  if (builder->count == builder->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->count, builder->count + 1,
        sizeof(*builder->nodes), &builder->capacity, (void**)&builder->nodes));
  }
  const iree_host_size_t index = builder->count++;
  builder->has_serial |= kind == LOOM_PIPELINE_LIFETIME_SERIAL;
  builder->nodes[index] =
      (loom_pipeline_lifetime_t){.kind = kind,
                                 .owner = invocation ? invocation->ordinal : 0,
                                 .parent = parent,
                                 .first_child = IREE_HOST_SIZE_MAX,
                                 .next_sibling = IREE_HOST_SIZE_MAX};
  if (kind == LOOM_PIPELINE_LIFETIME_OWNER) {
    builder->owners[invocation->ordinal] = index;
  }
  if (parent != IREE_HOST_SIZE_MAX) {
    if (*last_child == IREE_HOST_SIZE_MAX) {
      builder->nodes[parent].first_child = index;
    } else {
      builder->nodes[*last_child].next_sibling = index;
    }
    *last_child = index;
  }
  if (kind != LOOM_PIPELINE_LIFETIME_OWNER) {
    if (builder->pending_count == builder->pending_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          builder->arena, builder->pending_count, builder->pending_count + 1,
          sizeof(*builder->pending), &builder->pending_capacity,
          (void**)&builder->pending));
    }
    builder->pending[builder->pending_count++] =
        (loom_pipeline_lifetime_pending_t){index, invocation, schedule};
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_lifetimes_build(
    const loom_pipeline_invocations_t* invocations,
    iree_arena_allocator_t* arena, loom_pipeline_lifetimes_t* out_lifetimes) {
  loom_pipeline_lifetime_builder_t builder = {.arena = arena};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, invocations->count,
                                                 sizeof(*builder.owners),
                                                 (void**)&builder.owners));
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
      &builder, IREE_HOST_SIZE_MAX, NULL, LOOM_PIPELINE_LIFETIME_CONCURRENT,
      invocations->entries[0], NULL));
  for (iree_host_size_t i = 0; i < builder.pending_count; ++i) {
    const loom_pipeline_lifetime_pending_t pending = builder.pending[i];
    iree_host_size_t last = IREE_HOST_SIZE_MAX;
    if (pending.invocation) {
      const loom_pipeline_invocation_t* invocation = pending.invocation;
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
          &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_OWNER,
          invocation, NULL));
      for (iree_host_size_t c = 0; c < invocation->resources->composition_count;
           ++c) {
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
            &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_CONCURRENT,
            invocation->compositions[c], NULL));
      }
      for (iree_host_size_t s = 0; s < invocation->resources->schedule_count;
           ++s) {
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
            &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_SERIAL, NULL,
            &invocation->schedules[s]));
      }
    } else {
      const loom_pipeline_invocation_schedule_t* schedule = pending.schedule;
      const loom_pipeline_schedule_t* source = schedule->source;
      for (iree_host_size_t b = 0; b < source->control->graph.block_count;
           ++b) {
        const loom_pipeline_schedule_block_t* block = &source->blocks[b];
        for (iree_host_size_t s = 0; s < block->step_count; ++s) {
          const loom_pipeline_invocation_step_t binding =
              schedule->blocks[b][s];
          if (block->steps[s].kind == LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION) {
            IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
                &builder, pending.node, &last,
                LOOM_PIPELINE_LIFETIME_CONCURRENT, binding.child, NULL));
          } else if (block->steps[s].kind ==
                     LOOM_PIPELINE_SCHEDULE_STEP_SERIAL) {
            IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
                &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_SERIAL,
                NULL, binding.nested));
          }
        }
      }
    }
  }
  *out_lifetimes =
      (loom_pipeline_lifetimes_t){.nodes = builder.nodes,
                                  .count = builder.count,
                                  .owners = builder.owners,
                                  .has_serial = builder.has_serial};
  return loom_pipeline_lifetimes_index(arena, out_lifetimes);
}
