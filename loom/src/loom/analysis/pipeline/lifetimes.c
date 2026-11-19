// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetimes.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/pipeline/invocations.h"
#include "loom/util/cfg_loop_nest.h"

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
    // Child subtree sizes assign both depth-first orders without a second
    // tree walk. Only concurrent scopes reverse their sibling intervals.
    iree_host_size_t offset = 1;
    for (iree_host_size_t child = largest; child != IREE_HOST_SIZE_MAX;
         child = nodes[child].next_sibling) {
      const iree_host_size_t length = ancestry[child].subtree_count;
      ancestry[child].preorder = ancestry[i].preorder + offset;
      ancestry[child].concurrent_preorder =
          ancestry[i].concurrent_preorder +
          (nodes[i].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT
               ? ancestry[i].subtree_count - offset - length + 1
               : offset);
      offset += length;
      if (length > ancestry[largest].subtree_count) {
        largest = child;
      }
    }
    ancestry[largest].chain_head = ancestry[i].chain_head;
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

iree_host_size_t loom_pipeline_lifetimes_common_owner(
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t first,
    iree_host_size_t second) {
  const iree_host_size_t ancestor = loom_pipeline_lifetimes_common_ancestor(
      lifetimes, lifetimes->owners[first], lifetimes->owners[second]);
  return lifetimes->nodes[ancestor].owner;
}

typedef struct loom_pipeline_lifetime_pending_t {
  // Node awaiting expansion in the parent-before-child worklist.
  iree_host_size_t node;
  // Construction for a concurrent node; NULL for a serial node.
  const loom_pipeline_invocation_t* invocation;
  // Bound control for a serial node; NULL for a concurrent node.
  const loom_pipeline_invocation_schedule_t* schedule;
  // Loop nesting inherited from the site entering this scope.
  uint32_t loop_depth;
} loom_pipeline_lifetime_pending_t;

typedef struct loom_pipeline_lifetime_builder_t {
  // Arena owning the final tree and construction worklist.
  iree_arena_allocator_t* arena;
  // Nodes already discovered, in parent-before-child order.
  loom_pipeline_lifetime_t* nodes;
  // OWNER node indexed by the original construction ordinal.
  iree_host_size_t* owners;
  // Complete construction inventory, including entries outside all loops.
  iree_host_size_t owner_count;
  // Populated on the first construction reached inside a natural loop.
  uint32_t* loop_depths;
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
    const loom_pipeline_invocation_schedule_t* schedule, uint32_t loop_depth) {
  if (builder->count == builder->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->count, builder->count + 1,
        sizeof(*builder->nodes), &builder->capacity, (void**)&builder->nodes));
  }
  const iree_host_size_t index = builder->count++;
  builder->has_serial |= kind == LOOM_PIPELINE_LIFETIME_SERIAL;
  builder->nodes[index] = (loom_pipeline_lifetime_t){
      .kind = kind,
      .owner = invocation ? invocation->ordinal : builder->nodes[parent].owner,
      .parent = parent,
      .first_child = IREE_HOST_SIZE_MAX,
      .next_sibling = IREE_HOST_SIZE_MAX,
  };
  if (kind == LOOM_PIPELINE_LIFETIME_OWNER) {
    builder->owners[invocation->ordinal] = index;
    if (loop_depth) {
      if (!builder->loop_depths) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            builder->arena, builder->owner_count, sizeof(*builder->loop_depths),
            (void**)&builder->loop_depths));
        memset(builder->loop_depths, 0,
               builder->owner_count * sizeof(*builder->loop_depths));
      }
      builder->loop_depths[invocation->ordinal] = loop_depth;
    }
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
        (loom_pipeline_lifetime_pending_t){index, invocation, schedule,
                                           loop_depth};
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_lifetimes_build(
    const loom_pipeline_invocations_t* invocations,
    iree_arena_allocator_t* arena, loom_pipeline_lifetimes_t* out_lifetimes) {
  loom_pipeline_lifetime_builder_t builder = {
      .arena = arena, .owner_count = invocations->count};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, invocations->count,
                                                 sizeof(*builder.owners),
                                                 (void**)&builder.owners));
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
      &builder, IREE_HOST_SIZE_MAX, NULL, LOOM_PIPELINE_LIFETIME_CONCURRENT,
      invocations->entries[0], NULL, 0));
  for (iree_host_size_t i = 0; i < builder.pending_count; ++i) {
    const loom_pipeline_lifetime_pending_t pending = builder.pending[i];
    iree_host_size_t last = IREE_HOST_SIZE_MAX;
    if (pending.invocation) {
      const loom_pipeline_invocation_t* invocation = pending.invocation;
      IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
          &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_OWNER,
          invocation, NULL, pending.loop_depth));
      for (iree_host_size_t c = 0; c < invocation->resources->composition_count;
           ++c) {
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
            &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_CONCURRENT,
            invocation->compositions[c], NULL, pending.loop_depth));
      }
      for (iree_host_size_t s = 0; s < invocation->resources->schedule_count;
           ++s) {
        IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
            &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_SERIAL, NULL,
            &invocation->schedules[s], pending.loop_depth));
      }
    } else {
      const loom_pipeline_invocation_schedule_t* schedule = pending.schedule;
      const loom_pipeline_schedule_t* source = schedule->source;
      for (iree_host_size_t b = 0; b < source->control->graph.block_count;
           ++b) {
        const loom_pipeline_schedule_block_t* block = &source->blocks[b];
        const loom_cfg_loop_nest_t* loops = &source->control->loops;
        const uint16_t loop = loom_cfg_loop_nest_innermost(loops, (uint16_t)b);
        const uint32_t depth =
            loop == LOOM_CFG_LOOP_NEST_NONE
                ? pending.loop_depth
                : iree_math_saturating_add_u32(pending.loop_depth,
                                               loops->loops[loop].depth);
        for (iree_host_size_t s = 0; s < block->step_count; ++s) {
          const loom_pipeline_invocation_step_t binding =
              schedule->blocks[b][s];
          if (block->steps[s].kind == LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION) {
            IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
                &builder, pending.node, &last,
                LOOM_PIPELINE_LIFETIME_CONCURRENT, binding.child, NULL, depth));
          } else if (block->steps[s].kind ==
                     LOOM_PIPELINE_SCHEDULE_STEP_SERIAL) {
            IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_append(
                &builder, pending.node, &last, LOOM_PIPELINE_LIFETIME_SERIAL,
                NULL, binding.nested, depth));
          }
        }
      }
    }
  }
  *out_lifetimes =
      (loom_pipeline_lifetimes_t){.nodes = builder.nodes,
                                  .count = builder.count,
                                  .owners = builder.owners,
                                  .has_serial = builder.has_serial,
                                  .loop_depths = builder.loop_depths};
  return loom_pipeline_lifetimes_index(arena, out_lifetimes);
}
