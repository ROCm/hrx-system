// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/guarded_motion.h"

#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static bool loom_low_guarded_motion_narrows_mask(
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_schedule_node_t* node) {
  if (node->descriptor == NULL) {
    return false;
  }
  const loom_low_descriptor_t* descriptor = node->descriptor;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptors->operands[descriptor->operand_start + i];
    if (iree_any_bit_set(operand->flags,
                         LOOM_LOW_OPERAND_FLAG_NARROWS_EXECUTION_MASK)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_guarded_motion_can_cross_mask(
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_descriptor_t* candidate,
    const loom_low_descriptor_t* boundary) {
  if (boundary == NULL) {
    return true;
  }
  for (uint16_t i = 0; i < candidate->operand_count; ++i) {
    const loom_low_operand_t* read =
        &descriptors->operands[candidate->operand_start + i];
    if (!iree_any_bit_set(read->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      continue;
    }
    const uint16_t read_class =
        descriptors->reg_class_alts[read->reg_class_alt_start].reg_class_id;
    for (uint16_t j = 0; j < boundary->operand_count; ++j) {
      const loom_low_operand_t* write =
          &descriptors->operands[boundary->operand_start + j];
      if (!iree_any_bit_set(write->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
        continue;
      }
      const uint16_t write_class =
          descriptors->reg_class_alts[write->reg_class_alt_start].reg_class_id;
      if (read_class == write_class &&
          (!iree_any_bit_set(read->flags,
                             LOOM_LOW_OPERAND_FLAG_EXECUTION_MASK) ||
           !iree_any_bit_set(write->flags,
                             LOOM_LOW_OPERAND_FLAG_NARROWS_EXECUTION_MASK))) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_low_guarded_motion_operands_available(
    const loom_module_t* module, const loom_op_t* candidate,
    const loom_op_t* destination) {
  const loom_value_id_t* operands = loom_op_const_operands(candidate);
  for (uint16_t i = 0; i < candidate->operand_count; ++i) {
    const loom_value_t* value = loom_module_value(module, operands[i]);
    if (loom_value_is_block_arg(value)) {
      if (loom_value_def_block(value) == candidate->parent_block) {
        return false;
      }
      continue;
    }
    const loom_op_t* producer = loom_value_def_op(value);
    // Verified SSA plus the unique incoming edge proves all other captures
    // dominate the predecessor. Earlier prefix members move as one bundle.
    if (producer->parent_block == destination->parent_block &&
        producer->block_ordinal >= destination->block_ordinal) {
      return false;
    }
  }
  return true;
}

static uint32_t loom_low_guarded_motion_block_extent(
    const loom_low_schedule_table_t* schedule, uint32_t block_index) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  return block->issue_group_count == 0
             ? 0
             : schedule
                   ->issue_groups[block->issue_group_start +
                                  block->issue_group_count - 1]
                   .issue_cycle;
}

iree_status_t loom_low_guarded_motion_plan(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_low_guarded_motion_plan_t* out_plan) {
  *out_plan = (loom_low_guarded_motion_plan_t){.nodes = schedule->nodes};
  if (schedule->block_count < 2 || schedule->scopes.scope_count != 0 ||
      loom_low_function_schedule(schedule->function_op) ==
          LOOM_LOW_SCHEDULE_LOCKED ||
      loom_low_function_allocation(schedule->function_op) ==
          LOOM_LOW_ALLOCATION_FIXED) {
    return iree_ok_status();
  }
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  const loom_low_descriptor_set_t* descriptors =
      schedule->target.descriptor_set;
  iree_status_t status = iree_ok_status();
  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < schedule->block_count;
       ++block_index) {
    const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
    if (!block_info->reachable || block_info->predecessor_count != 1) {
      continue;
    }
    const uint16_t parent_index =
        graph->predecessor_indices[block_info->predecessor_start];
    if (parent_index == block_index) {
      continue;
    }
    const loom_low_schedule_block_t* parent = &schedule->blocks[parent_index];
    const loom_low_schedule_node_t* terminator =
        &schedule->nodes[parent->node_start + parent->node_count - 1];
    if (terminator->op->kind != LOOM_OP_LOW_COND_BR) {
      continue;
    }
    const loom_low_schedule_node_t* boundary = NULL;
    if (parent->node_count > 1) {
      const loom_low_schedule_node_t* previous = terminator - 1;
      if (loom_low_guarded_motion_narrows_mask(descriptors, previous)) {
        boundary = previous;
      }
    }
    const loom_op_t* destination = boundary ? boundary->op : terminator->op;
    const uint32_t available_cycles =
        boundary ? boundary->issue_cycle : terminator->issue_cycle;
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    uint32_t prefix_count = 0;
    uint64_t prefix_ready_cycle = 0;
    while (prefix_count < block->node_count) {
      const loom_low_schedule_node_t* node =
          &schedule->nodes[block->node_start + prefix_count];
      if (node->descriptor == NULL ||
          iree_any_bit_set(node->traits, LOOM_TRAIT_OBSERVABLE_EFFECT) ||
          !iree_any_bit_set(node->descriptor->flags,
                            LOOM_LOW_DESCRIPTOR_FLAG_SAFE_TO_SPECULATE) ||
          !loom_low_guarded_motion_can_cross_mask(
              descriptors, node->descriptor,
              boundary ? boundary->descriptor : NULL) ||
          !loom_low_guarded_motion_operands_available(schedule->module,
                                                      node->op, destination)) {
        break;
      }
      const uint64_t ready_cycle =
          (uint64_t)node->issue_cycle + node->schedule_class->latency_cycles;
      prefix_ready_cycle = iree_max(prefix_ready_cycle, ready_cycle);
      ++prefix_count;
    }
    if (prefix_count == 0 || prefix_ready_cycle > available_cycles) {
      continue;
    }
    // The first guarded memory access defines the bundle's consumer boundary.
    const loom_low_schedule_node_t* access =
        &schedule->nodes[block->node_start + prefix_count];
    if (!iree_any_bit_set(access->traits,
                          LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY)) {
      continue;
    }
    if (out_plan->regions == NULL) {
      status = iree_arena_allocate_array(arena, schedule->block_count,
                                         sizeof(*out_plan->regions),
                                         (void**)&out_plan->regions);
      if (iree_status_is_ok(status)) {
        const iree_host_size_t word_count =
            iree_bitmap_calculate_words(schedule->block_count);
        status = iree_arena_allocate_array(
            arena, word_count, sizeof(*out_plan->changed_blocks.words),
            (void**)&out_plan->changed_blocks.words);
        if (iree_status_is_ok(status)) {
          memset(out_plan->changed_blocks.words, 0,
                 word_count * sizeof(*out_plan->changed_blocks.words));
          out_plan->changed_blocks.bit_count = schedule->block_count;
        }
      }
    }
    if (iree_status_is_ok(status)) {
      out_plan->regions[out_plan->region_count++] =
          (loom_low_guarded_motion_region_t){
              .node_start = block->node_start,
              .node_count = prefix_count,
              .destination = (loom_op_t*)destination,
              .source =
                  (loom_op_t*)schedule->nodes[block->node_start + prefix_count]
                      .op,
          };
      iree_bitmap_set(out_plan->changed_blocks, parent_index);
      iree_bitmap_set(out_plan->changed_blocks, block_index);
    }
  }
  return status;
}

iree_status_t loom_low_guarded_motion_apply(
    loom_rewriter_t* rewriter, const loom_low_guarded_motion_plan_t* plan) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < plan->region_count;
       ++i) {
    const loom_low_guarded_motion_region_t* region = &plan->regions[i];
    for (uint32_t j = 0; iree_status_is_ok(status) && j < region->node_count;
         ++j) {
      status = loom_rewriter_move_before(
          rewriter, (loom_op_t*)plan->nodes[region->node_start + j].op,
          region->destination);
    }
  }
  return status;
}

iree_status_t loom_low_guarded_motion_rollback(
    loom_rewriter_t* rewriter, const loom_low_guarded_motion_plan_t* plan) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < plan->region_count;
       ++i) {
    const loom_low_guarded_motion_region_t* region = &plan->regions[i];
    for (uint32_t j = 0; iree_status_is_ok(status) && j < region->node_count;
         ++j) {
      status = loom_rewriter_move_before(
          rewriter, (loom_op_t*)plan->nodes[region->node_start + j].op,
          region->source);
    }
  }
  return status;
}

bool loom_low_guarded_motion_improves_schedule(
    const loom_low_schedule_table_t* baseline,
    const loom_low_schedule_table_t* trial) {
  bool improved = false;
  for (uint32_t i = 0; i < baseline->block_count; ++i) {
    const uint32_t before = loom_low_guarded_motion_block_extent(baseline, i);
    const uint32_t after = loom_low_guarded_motion_block_extent(trial, i);
    if (after > before) {
      return false;
    }
    improved |= after < before;
  }
  return improved;
}
