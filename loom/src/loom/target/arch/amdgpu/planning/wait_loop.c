// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_loop.h"

#include <string.h>

#include "loom/ops/low/ops.h"
#include "loom/util/cfg_graph.h"

// Number of binary-ancestor levels needed for uint16_t CFG block indices.
#define LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT 16u

// AMDGPU eligibility layered over one generic canonical loop interval.
struct loom_amdgpu_wait_loop_t {
  // Nearest eligible enclosing loop, or LOOM_CFG_LOOP_NONE.
  uint32_t valid_parent_loop_index;
  // Whether both loop edges have the required target-low branch shape.
  bool is_valid;
};

static bool loom_amdgpu_wait_loop_interval_is_supported(
    const loom_low_schedule_table_t* schedule,
    const loom_cfg_loop_interval_t* interval) {
  if (!interval->is_canonical) return false;
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  const loom_cfg_edge_info_t* entry_edge =
      loom_cfg_graph_edge(graph, interval->entry_edge_index);
  const loom_cfg_edge_info_t* backedge =
      loom_cfg_graph_edge(graph, interval->backedge_edge_index);
  const loom_block_t* header = graph->blocks[interval->header_index].block;
  if (entry_edge == NULL || backedge == NULL ||
      entry_edge->terminator == NULL || backedge->terminator == NULL ||
      !loom_low_br_isa(entry_edge->terminator) ||
      loom_low_br_dest(entry_edge->terminator) != header ||
      !loom_low_br_isa(backedge->terminator) ||
      loom_low_br_dest(backedge->terminator) != header ||
      backedge->successor_index != 0) {
    return false;
  }
  const loom_cfg_block_index_span_t entry_successors =
      loom_cfg_graph_successors(graph, entry_edge->source_block_index);
  if (entry_successors.count != 1 || entry_edge->successor_index != 0) {
    return false;
  }

  const loom_low_schedule_block_t* preheader =
      &schedule->blocks[interval->entry_predecessor_index];
  if (preheader->node_count == 0) return false;
  const uint32_t insertion_node =
      preheader->node_start + preheader->node_count - 1;
  IREE_ASSERT_LT(insertion_node, schedule->node_count);
  return schedule->nodes[insertion_node].op == entry_edge->terminator;
}

static iree_host_size_t loom_amdgpu_wait_loop_ancestor_slot_index(
    uint32_t level, iree_host_size_t loop_count, uint32_t loop_index) {
  IREE_ASSERT_LT(level, LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT);
  IREE_ASSERT_LT(loop_index, loop_count);
  return (iree_host_size_t)level * loop_count + loop_index;
}

iree_status_t loom_amdgpu_wait_loop_analysis_initialize(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_loop_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_analysis);
  *out_analysis = (loom_amdgpu_wait_loop_analysis_t){
      .schedule = schedule,
  };
  const loom_cfg_loop_forest_t* forest = &schedule->loop_forest;
  if (forest->interval_count == 0) return iree_ok_status();

  loom_amdgpu_wait_loop_t* loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, forest->interval_count, sizeof(*loops), (void**)&loops));
  for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
    const loom_cfg_loop_interval_t* interval = &forest->intervals[i];
    const uint32_t parent_loop = interval->parent_loop_index;
    uint32_t valid_parent_loop = LOOM_CFG_LOOP_NONE;
    if (parent_loop != LOOM_CFG_LOOP_NONE) {
      valid_parent_loop = loops[parent_loop].is_valid
                              ? parent_loop
                              : loops[parent_loop].valid_parent_loop_index;
    }
    loops[i] = (loom_amdgpu_wait_loop_t){
        .valid_parent_loop_index = valid_parent_loop,
        .is_valid =
            loom_amdgpu_wait_loop_interval_is_supported(schedule, interval),
    };
  }

  // Sixteen rows cover every ancestor distance representable by uint16_t block
  // indices and make each dependency query a fixed number of probes.
  const iree_host_size_t ancestor_slot_count =
      forest->interval_count * LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT;
  uint32_t* valid_ancestor_loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, ancestor_slot_count, sizeof(*valid_ancestor_loops),
      (void**)&valid_ancestor_loops));
  for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
    valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
        /*level=*/0, forest->interval_count, (uint32_t)i)] =
        loops[i].is_valid ? loops[i].valid_parent_loop_index
                          : LOOM_CFG_LOOP_NONE;
  }
  for (uint32_t level = 1; level < LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT;
       ++level) {
    for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
      const uint32_t prior_ancestor =
          valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
              level - 1, forest->interval_count, (uint32_t)i)];
      valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
          level, forest->interval_count, (uint32_t)i)] =
          prior_ancestor == LOOM_CFG_LOOP_NONE
              ? LOOM_CFG_LOOP_NONE
              : valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
                    level - 1, forest->interval_count, prior_ancestor)];
    }
  }

  *out_analysis = (loom_amdgpu_wait_loop_analysis_t){
      .schedule = schedule,
      .loops = loops,
      .valid_ancestor_loops = valid_ancestor_loops,
      .loop_count = forest->interval_count,
  };
  return iree_ok_status();
}

uint16_t loom_amdgpu_wait_loop_analysis_preheader(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node) {
  if (analysis->loop_count == 0) return UINT16_MAX;
  const loom_cfg_loop_forest_t* forest = &analysis->schedule->loop_forest;
  const uint16_t producer_block =
      analysis->schedule->nodes[producer_node].block_index;
  const uint16_t consumer_block =
      analysis->schedule->nodes[consumer_node].block_index;
  const uint32_t innermost_loop =
      forest->innermost_loop_indices[consumer_block];
  if (innermost_loop == LOOM_CFG_LOOP_NONE) return UINT16_MAX;

  uint32_t loop_index =
      analysis->loops[innermost_loop].is_valid
          ? innermost_loop
          : analysis->loops[innermost_loop].valid_parent_loop_index;
  if (loop_index == LOOM_CFG_LOOP_NONE ||
      forest->intervals[loop_index].header_index <= producer_block) {
    return UINT16_MAX;
  }

  for (uint32_t i = LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT; i > 0; --i) {
    const uint32_t level = i - 1;
    const uint32_t ancestor_loop =
        analysis
            ->valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
                level, analysis->loop_count, loop_index)];
    if (ancestor_loop != LOOM_CFG_LOOP_NONE &&
        forest->intervals[ancestor_loop].header_index > producer_block) {
      loop_index = ancestor_loop;
    }
  }
  return forest->intervals[loop_index].entry_predecessor_index;
}

const loom_cfg_loop_interval_t* loom_amdgpu_wait_loop_analysis_cyclic_interval(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node) {
  if (analysis->loop_count == 0 ||
      producer_node >= analysis->schedule->node_count ||
      consumer_node >= analysis->schedule->node_count) {
    return NULL;
  }
  const loom_low_schedule_node_t* producer =
      &analysis->schedule->nodes[producer_node];
  const loom_low_schedule_node_t* consumer =
      &analysis->schedule->nodes[consumer_node];
  if (producer->block_index < consumer->block_index ||
      (producer->block_index == consumer->block_index &&
       producer->scheduled_ordinal <= consumer->scheduled_ordinal)) {
    return NULL;
  }

  const loom_cfg_loop_forest_t* forest = &analysis->schedule->loop_forest;
  const uint32_t innermost_loop =
      forest->innermost_loop_indices[consumer->block_index];
  if (innermost_loop == LOOM_CFG_LOOP_NONE) return NULL;
  const uint32_t loop_index =
      analysis->loops[innermost_loop].is_valid
          ? innermost_loop
          : analysis->loops[innermost_loop].valid_parent_loop_index;
  if (loop_index == LOOM_CFG_LOOP_NONE) return NULL;
  const loom_cfg_loop_interval_t* interval = &forest->intervals[loop_index];
  return producer->block_index >= interval->header_index &&
                 producer->block_index <= interval->latch_index
             ? interval
             : NULL;
}

static bool loom_amdgpu_wait_loop_counter_supports_cyclic_frontier(
    uint16_t counter_id) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      return true;
    default:
      return false;
  }
}

static iree_host_size_t loom_amdgpu_wait_loop_frontier_index(
    iree_host_size_t block_index, uint32_t counter_slot) {
  IREE_ASSERT_LT(counter_slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return block_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT + counter_slot;
}

typedef enum loom_amdgpu_wait_loop_transparency_flag_bits_e {
  // Dependencies observe a counter epoch proven empty before the block.
  LOOM_AMDGPU_WAIT_LOOP_TRANSPARENCY_FLAG_COMPLETED_DEPENDENCIES = 1u << 0,
} loom_amdgpu_wait_loop_transparency_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_loop_transparency_flags_t;

static bool loom_amdgpu_wait_loop_blocks_are_counter_transparent(
    const loom_amdgpu_wait_loop_analysis_t* analysis,
    const loom_amdgpu_wait_loop_node_t* nodes,
    const uint32_t* first_dependency_by_consumer,
    const loom_amdgpu_wait_loop_dependency_t* dependencies,
    iree_host_size_t dependency_count, const loom_cfg_loop_interval_t* interval,
    uint16_t frontier_block_index, uint32_t counter_mask,
    loom_amdgpu_wait_loop_transparency_flags_t flags) {
  const loom_low_schedule_table_t* schedule = analysis->schedule;
  for (uint32_t block_index = interval->header_index;
       block_index <= interval->latch_index; ++block_index) {
    if (block_index == frontier_block_index) continue;
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      const loom_amdgpu_wait_loop_node_t* node = &nodes[node_index];
      if (((node->producer_counter_mask | node->reset_counter_mask |
            node->hazard_counter_mask) &
           counter_mask) != 0) {
        return false;
      }
      if (!iree_any_bit_set(
              flags,
              LOOM_AMDGPU_WAIT_LOOP_TRANSPARENCY_FLAG_COMPLETED_DEPENDENCIES)) {
        for (uint32_t dependency_index =
                 first_dependency_by_consumer[node_index];
             dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
             dependency_index =
                 dependencies[dependency_index].next_dependency) {
          IREE_ASSERT_LT(dependency_index, dependency_count);
          if ((dependencies[dependency_index].counter_mask & counter_mask) !=
              0) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

static uint32_t loom_amdgpu_wait_loop_last_guaranteed_reset_ordinal(
    const loom_amdgpu_wait_loop_analysis_t* analysis,
    const loom_amdgpu_wait_loop_node_t* nodes,
    const uint32_t* first_dependency_by_consumer,
    const loom_amdgpu_wait_loop_dependency_t* dependencies,
    iree_host_size_t dependency_count, const loom_low_schedule_block_t* block,
    uint32_t counter_mask) {
  const loom_low_schedule_table_t* schedule = analysis->schedule;
  uint32_t last_reset_ordinal = UINT32_MAX;
  uint32_t last_producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  bool has_outstanding_workgroup_write = false;
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    const uint32_t node_index = schedule->scheduled_node_indices[packet_index];
    const loom_amdgpu_wait_loop_node_t* node = &nodes[node_index];
    bool guarantees_reset = (node->reset_counter_mask & counter_mask) != 0;
    if (!guarantees_reset && has_outstanding_workgroup_write &&
        (node->workgroup_barrier_counter_mask & counter_mask) != 0) {
      guarantees_reset = true;
    }
    if (!guarantees_reset &&
        last_producer_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
      for (uint32_t dependency_index = first_dependency_by_consumer[node_index];
           dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
           dependency_index = dependencies[dependency_index].next_dependency) {
        IREE_ASSERT_LT(dependency_index, dependency_count);
        const loom_amdgpu_wait_loop_dependency_t* dependency =
            &dependencies[dependency_index];
        if ((dependency->counter_mask & counter_mask) != 0 &&
            dependency->producer_node == last_producer_node) {
          guarantees_reset = true;
          break;
        }
      }
    }
    if (guarantees_reset) {
      last_reset_ordinal = i;
      last_producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      has_outstanding_workgroup_write = false;
    }
    if ((node->producer_counter_mask & counter_mask) != 0) {
      last_producer_node = node_index;
      if ((node->workgroup_write_counter_mask & counter_mask) != 0) {
        has_outstanding_workgroup_write = true;
      }
    }
  }
  return last_reset_ordinal;
}

static bool loom_amdgpu_wait_loop_analyze_cyclic_frontier(
    const loom_amdgpu_wait_loop_analysis_t* analysis,
    const loom_amdgpu_wait_loop_node_t* nodes,
    const uint32_t* first_dependency_by_consumer,
    const loom_amdgpu_wait_loop_dependency_t* dependencies,
    iree_host_size_t dependency_count, uint16_t block_index,
    uint32_t counter_slot,
    loom_amdgpu_wait_loop_cyclic_frontier_t* out_frontier) {
  *out_frontier = (loom_amdgpu_wait_loop_cyclic_frontier_t){0};
  const uint16_t counter_id =
      loom_amdgpu_wait_counter_id_from_slot(counter_slot);
  if (!loom_amdgpu_wait_loop_counter_supports_cyclic_frontier(counter_id)) {
    return false;
  }

  const loom_low_schedule_table_t* schedule = analysis->schedule;
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  const uint32_t counter_mask =
      loom_amdgpu_wait_counter_mask_from_slot(counter_slot);
  const loom_cfg_loop_interval_t* cyclic_interval = NULL;
  uint32_t first_required_producer_ordinal = UINT32_MAX;
  uint32_t last_required_producer_ordinal = 0;
  uint32_t last_consumer_ordinal = 0;
  bool has_cyclic_dependency = false;
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    const uint32_t consumer_node =
        schedule->scheduled_node_indices[packet_index];
    const loom_low_schedule_node_t* consumer = &schedule->nodes[consumer_node];
    for (uint32_t dependency_index =
             first_dependency_by_consumer[consumer_node];
         dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
         dependency_index = dependencies[dependency_index].next_dependency) {
      IREE_ASSERT_LT(dependency_index, dependency_count);
      const loom_amdgpu_wait_loop_dependency_t* dependency =
          &dependencies[dependency_index];
      if ((dependency->counter_mask & counter_mask) == 0) continue;
      const loom_low_schedule_node_t* producer =
          &schedule->nodes[dependency->producer_node];
      if (producer->block_index != block_index ||
          producer->scheduled_ordinal <= consumer->scheduled_ordinal) {
        continue;
      }
      const loom_cfg_loop_interval_t* dependency_interval =
          loom_amdgpu_wait_loop_analysis_cyclic_interval(
              analysis, dependency->producer_node, consumer_node);
      if (!iree_any_bit_set(dependency->flags,
                            LOOM_AMDGPU_WAIT_LOOP_DEPENDENCY_FLAG_SSA_USE) ||
          dependency_interval == NULL ||
          (cyclic_interval != NULL && dependency_interval != cyclic_interval)) {
        return false;
      }
      cyclic_interval = dependency_interval;
      has_cyclic_dependency = true;
      first_required_producer_ordinal = iree_min(
          first_required_producer_ordinal, producer->scheduled_ordinal);
      last_required_producer_ordinal =
          iree_max(last_required_producer_ordinal, producer->scheduled_ordinal);
      last_consumer_ordinal =
          iree_max(last_consumer_ordinal, consumer->scheduled_ordinal);
    }
  }
  if (!has_cyclic_dependency) return false;

  const uint32_t last_reset_ordinal =
      loom_amdgpu_wait_loop_last_guaranteed_reset_ordinal(
          analysis, nodes, first_dependency_by_consumer, dependencies,
          dependency_count, block, counter_mask);
  uint32_t first_producer_ordinal = UINT32_MAX;
  uint32_t last_producer_ordinal = 0;
  uint32_t outstanding_count = 0;
  uint32_t outstanding_write_count = 0;
  uint32_t outstanding_workgroup_write_count = 0;
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    if (last_reset_ordinal != UINT32_MAX && i < last_reset_ordinal) continue;
    const uint32_t packet_index = block->scheduled_node_start + i;
    const uint32_t node_index = schedule->scheduled_node_indices[packet_index];
    const loom_amdgpu_wait_loop_node_t* node = &nodes[node_index];
    if ((node->producer_counter_mask & counter_mask) == 0) continue;
    first_producer_ordinal = iree_min(first_producer_ordinal, i);
    last_producer_ordinal = i;
    ++outstanding_count;
    if ((node->write_counter_mask & counter_mask) != 0) {
      ++outstanding_write_count;
    }
    if ((node->workgroup_write_counter_mask & counter_mask) != 0) {
      ++outstanding_workgroup_write_count;
    }
  }
  const bool reset_establishes_epoch =
      last_reset_ordinal != UINT32_MAX &&
      last_consumer_ordinal < last_reset_ordinal;
  if (outstanding_count == 0) {
    if (!reset_establishes_epoch ||
        !loom_amdgpu_wait_loop_blocks_are_counter_transparent(
            analysis, nodes, first_dependency_by_consumer, dependencies,
            dependency_count, cyclic_interval, block_index, counter_mask,
            LOOM_AMDGPU_WAIT_LOOP_TRANSPARENCY_FLAG_COMPLETED_DEPENDENCIES)) {
      return false;
    }
    *out_frontier = (loom_amdgpu_wait_loop_cyclic_frontier_t){
        .flags = LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID,
        .producer_start_ordinal = block->scheduled_node_count,
    };
    return true;
  }
  if (first_required_producer_ordinal < first_producer_ordinal ||
      last_required_producer_ordinal > last_producer_ordinal ||
      last_consumer_ordinal >= first_required_producer_ordinal ||
      !loom_amdgpu_wait_loop_blocks_are_counter_transparent(
          analysis, nodes, first_dependency_by_consumer, dependencies,
          dependency_count, cyclic_interval, block_index, counter_mask,
          /*flags=*/0)) {
    return false;
  }

  const bool consumers_cover_epoch =
      last_required_producer_ordinal == last_producer_ordinal;
  if (!reset_establishes_epoch && !consumers_cover_epoch) return false;

  // Once the trailing epoch begins, every progress operation must be part of
  // that stable epoch. A later reset or dependency needs a more general
  // counter dataflow model and therefore keeps conservative wait placement.
  for (uint32_t i = first_producer_ordinal; i < block->scheduled_node_count;
       ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    const uint32_t node_index = schedule->scheduled_node_indices[packet_index];
    const loom_amdgpu_wait_loop_node_t* node = &nodes[node_index];
    if ((node->reset_counter_mask & counter_mask) != 0) return false;
    for (uint32_t dependency_index = first_dependency_by_consumer[node_index];
         dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
         dependency_index = dependencies[dependency_index].next_dependency) {
      IREE_ASSERT_LT(dependency_index, dependency_count);
      if ((dependencies[dependency_index].counter_mask & counter_mask) != 0) {
        return false;
      }
    }
  }

  *out_frontier = (loom_amdgpu_wait_loop_cyclic_frontier_t){
      .flags = LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID,
      .producer_start_ordinal = first_producer_ordinal,
      .outstanding_count = outstanding_count,
      .outstanding_write_count = outstanding_write_count,
      .outstanding_workgroup_write_count = outstanding_workgroup_write_count,
  };
  return true;
}

iree_status_t loom_amdgpu_wait_loop_analysis_build_cyclic_frontiers(
    const loom_amdgpu_wait_loop_analysis_t* analysis,
    const loom_amdgpu_wait_loop_node_t* nodes,
    const uint32_t* first_dependency_by_consumer,
    const loom_amdgpu_wait_loop_dependency_t* dependencies,
    iree_host_size_t dependency_count, iree_arena_allocator_t* arena,
    const loom_amdgpu_wait_loop_cyclic_frontier_t** out_frontiers) {
  IREE_ASSERT_ARGUMENT(analysis);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontiers);
  *out_frontiers = NULL;
  if (analysis->loop_count == 0 || dependency_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(nodes);
  IREE_ASSERT_ARGUMENT(first_dependency_by_consumer);
  IREE_ASSERT_ARGUMENT(dependencies);
  const loom_low_schedule_table_t* schedule = analysis->schedule;
  if (schedule->block_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU wait-loop block count exceeds uint16_t");
  }
  if (schedule->block_count >
      IREE_HOST_SIZE_MAX / LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU cyclic wait-frontier count exceeds host size");
  }

  uint32_t* candidate_counter_masks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*candidate_counter_masks),
      (void**)&candidate_counter_masks));
  memset(candidate_counter_masks, 0,
         schedule->block_count * sizeof(*candidate_counter_masks));
  bool has_candidate = false;
  for (iree_host_size_t i = 0; i < dependency_count; ++i) {
    const loom_amdgpu_wait_loop_dependency_t* dependency = &dependencies[i];
    if (!iree_any_bit_set(dependency->flags,
                          LOOM_AMDGPU_WAIT_LOOP_DEPENDENCY_FLAG_SSA_USE)) {
      continue;
    }
    IREE_ASSERT_LT(dependency->producer_node, schedule->node_count);
    IREE_ASSERT_LT(dependency->consumer_node, schedule->node_count);
    const loom_low_schedule_node_t* producer =
        &schedule->nodes[dependency->producer_node];
    const loom_low_schedule_node_t* consumer =
        &schedule->nodes[dependency->consumer_node];
    if (producer->block_index != consumer->block_index ||
        producer->scheduled_ordinal <= consumer->scheduled_ordinal ||
        loom_amdgpu_wait_loop_analysis_cyclic_interval(
            analysis, dependency->producer_node, dependency->consumer_node) ==
            NULL) {
      continue;
    }
    uint32_t counter_mask = dependency->counter_mask;
    while (counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
      const uint16_t counter_id =
          loom_amdgpu_wait_counter_id_from_slot(counter_slot);
      if (loom_amdgpu_wait_loop_counter_supports_cyclic_frontier(counter_id)) {
        candidate_counter_masks[consumer->block_index] |=
            loom_amdgpu_wait_counter_mask_from_slot(counter_slot);
        has_candidate = true;
      }
      counter_mask &= counter_mask - 1;
    }
  }
  if (!has_candidate) return iree_ok_status();

  const iree_host_size_t frontier_count =
      schedule->block_count * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
  loom_amdgpu_wait_loop_cyclic_frontier_t* frontiers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, frontier_count, sizeof(*frontiers), (void**)&frontiers));
  memset(frontiers, 0, frontier_count * sizeof(*frontiers));
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    uint32_t counter_mask = candidate_counter_masks[block_index];
    while (counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
      loom_amdgpu_wait_loop_cyclic_frontier_t frontier = {0};
      if (loom_amdgpu_wait_loop_analyze_cyclic_frontier(
              analysis, nodes, first_dependency_by_consumer, dependencies,
              dependency_count, (uint16_t)block_index, counter_slot,
              &frontier)) {
        frontiers[loom_amdgpu_wait_loop_frontier_index(
            block_index, counter_slot)] = frontier;
      }
      counter_mask &= counter_mask - 1;
    }
  }
  *out_frontiers = frontiers;
  return iree_ok_status();
}
