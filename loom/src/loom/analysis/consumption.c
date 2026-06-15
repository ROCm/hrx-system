// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/consumption.h"

#include <string.h>

#include "loom/ir/module.h"

static void loom_consumption_bitset_set(uint64_t* bits,
                                        iree_host_size_t word_count,
                                        iree_host_size_t bit_index) {
  const iree_host_size_t word_index = bit_index / 64u;
  IREE_ASSERT_LT(word_index, word_count);
  bits[word_index] |= ((uint64_t)1) << (bit_index % 64u);
}

static bool loom_consumption_bitset_test(const uint64_t* bits,
                                         iree_host_size_t word_count,
                                         iree_host_size_t bit_index) {
  const iree_host_size_t word_index = bit_index / 64u;
  IREE_ASSERT_LT(word_index, word_count);
  return (bits[word_index] & (((uint64_t)1) << (bit_index % 64u))) != 0;
}

static iree_host_size_t loom_consumption_bitset_word_count(
    iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static bool loom_consumption_region_is_cfg(const loom_region_t* region) {
  return region &&
         iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG);
}

iree_status_t loom_consumption_region_query_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query) {
  if (!module || !region || !arena || !out_query) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "consumption query requires a module, region, arena, and output query");
  }
  memset(out_query, 0, sizeof(*out_query));
  out_query->module = module;
  out_query->region = region;
  out_query->arena = arena;
  return iree_ok_status();
}

static iree_status_t loom_consumption_region_query_cfg_graph(
    loom_consumption_region_query_t* query,
    const loom_cfg_graph_t** out_graph) {
  *out_graph = NULL;
  if (!loom_consumption_region_is_cfg(query->region)) {
    return iree_ok_status();
  }
  if (!query->cfg_graph_ready) {
    IREE_RETURN_IF_ERROR(loom_cfg_graph_build(query->module, query->region,
                                              query->arena, &query->cfg_graph));
    query->cfg_graph_ready = true;
  }
  *out_graph = &query->cfg_graph;
  return iree_ok_status();
}

static iree_status_t loom_consumption_region_query_prepare_cfg_search(
    loom_consumption_region_query_t* query, iree_host_size_t block_count,
    iree_host_size_t* out_visited_word_count) {
  iree_host_size_t visited_word_count =
      loom_consumption_bitset_word_count(block_count);
  if (visited_word_count > query->visited_word_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, visited_word_count, sizeof(*query->visited_bits),
        &query->visited_word_capacity, (void**)&query->visited_bits));
  }
  if (block_count > query->block_stack_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, block_count, sizeof(*query->block_stack),
        &query->block_stack_capacity, (void**)&query->block_stack));
  }
  if (visited_word_count > 0) {
    memset(query->visited_bits, 0,
           visited_word_count * sizeof(query->visited_bits[0]));
  }
  *out_visited_word_count = visited_word_count;
  return iree_ok_status();
}

static bool loom_consumption_op_uses_value(const loom_op_t* op,
                                           loom_value_id_t value_id,
                                           uint16_t* out_operand_index) {
  if (!op) return false;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == value_id) {
      *out_operand_index = i;
      return true;
    }
  }
  return false;
}

static bool loom_consumption_region_uses_value(const loom_region_t* region,
                                               loom_value_id_t value_id,
                                               loom_consumption_use_t* out_use);

static bool loom_consumption_op_or_nested_region_uses_value(
    const loom_op_t* op, loom_value_id_t value_id,
    loom_consumption_use_t* out_use) {
  uint16_t operand_index = 0;
  if (loom_consumption_op_uses_value(op, value_id, &operand_index)) {
    *out_use = (loom_consumption_use_t){
        .op = op,
        .operand_index = operand_index,
    };
    return true;
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_consumption_region_uses_value(regions[i], value_id, out_use)) {
      return true;
    }
  }
  return false;
}

static bool loom_consumption_region_uses_value(
    const loom_region_t* region, loom_value_id_t value_id,
    loom_consumption_use_t* out_use) {
  if (!region) return false;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_consumption_op_or_nested_region_uses_value(op, value_id,
                                                          out_use)) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_consumption_block_arg_defines_value(const loom_block_t* block,
                                                     loom_value_id_t value_id) {
  if (!block) return false;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (loom_block_arg_id(block, i) == value_id) return true;
  }
  return false;
}

static bool loom_consumption_find_block_use(const loom_block_t* block,
                                            loom_value_id_t value_id,
                                            loom_consumption_use_t* out_use) {
  if (!block) return false;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_consumption_op_or_nested_region_uses_value(op, value_id,
                                                        out_use)) {
      return true;
    }
  }
  return false;
}

static bool loom_consumption_find_same_block_use_after(
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    loom_consumption_use_t* out_use) {
  const loom_op_t* op = consuming_op ? consuming_op->next_op : NULL;
  while (op) {
    if (loom_consumption_op_or_nested_region_uses_value(op, value_id,
                                                        out_use)) {
      return true;
    }
    op = op->next_op;
  }
  return false;
}

// A CFG backedge to the consuming block executes block arguments and earlier
// op definitions before it reaches the consuming op again. Those values are
// fresh dynamic storage for the next block entry, not later uses of the storage
// consumed by the previous dynamic execution.
static bool loom_consumption_value_is_recreated_before_op_on_reentry(
    const loom_module_t* module, const loom_block_t* block,
    const loom_op_t* consuming_op, loom_value_id_t value_id) {
  if (!block || !consuming_op) {
    return false;
  }
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value) == block;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op && defining_op->parent_block == block &&
         defining_op->block_ordinal < consuming_op->block_ordinal;
}

static void loom_consumption_cfg_search_push(
    const loom_cfg_graph_t* graph, uint16_t block_index, uint64_t* visited_bits,
    iree_host_size_t visited_word_count, uint16_t* stack,
    iree_host_size_t* inout_stack_count) {
  if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return;
  }
  if (loom_consumption_bitset_test(visited_bits, visited_word_count,
                                   block_index)) {
    return;
  }
  loom_consumption_bitset_set(visited_bits, visited_word_count, block_index);
  stack[(*inout_stack_count)++] = block_index;
}

static iree_status_t loom_consumption_find_cfg_use_after(
    loom_consumption_region_query_t* query, const loom_cfg_graph_t* graph,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    loom_consumption_use_t* out_use, bool* out_found) {
  *out_found = false;
  if (!graph || graph->malformed) {
    return iree_ok_status();
  }
  if (graph->region != consuming_op->parent_block->parent_region) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consumption CFG graph must describe the consuming op region");
  }
  iree_host_size_t consuming_block_index =
      loom_cfg_graph_block_index(graph, consuming_op->parent_block);
  if (consuming_block_index == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }

  iree_host_size_t visited_word_count = 0;
  IREE_RETURN_IF_ERROR(loom_consumption_region_query_prepare_cfg_search(
      query, graph->block_count, &visited_word_count));
  if (loom_consumption_value_is_recreated_before_op_on_reentry(
          query->module, consuming_op->parent_block, consuming_op, value_id)) {
    loom_consumption_bitset_set(query->visited_bits, visited_word_count,
                                (uint16_t)consuming_block_index);
  }

  iree_host_size_t stack_count = 0;
  loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(graph, (uint16_t)consuming_block_index);
  for (iree_host_size_t i = 0; i < successors.count; ++i) {
    loom_consumption_cfg_search_push(graph, successors.values[i],
                                     query->visited_bits, visited_word_count,
                                     query->block_stack, &stack_count);
  }

  while (stack_count > 0) {
    uint16_t block_index = query->block_stack[--stack_count];
    const loom_block_t* block = graph->blocks[block_index].block;
    if (loom_consumption_block_arg_defines_value(block, value_id)) {
      continue;
    }
    if (loom_consumption_find_block_use(block, value_id, out_use)) {
      *out_found = true;
      return iree_ok_status();
    }

    successors = loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      loom_consumption_cfg_search_push(graph, successors.values[i],
                                       query->visited_bits, visited_word_count,
                                       query->block_stack, &stack_count);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_consumption_find_use_after(
    loom_consumption_region_query_t* query, const loom_op_t* consuming_op,
    loom_value_id_t value_id, loom_consumption_use_t* out_use,
    bool* out_found) {
  if (!out_found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "consumption query requires an output flag");
  }
  *out_found = false;
  if (!query || !consuming_op || !out_use) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "consumption query requires query, consuming op, and output use");
  }
  if (!query->module || !query->region || !query->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "consumption query is not initialized");
  }
  if (!consuming_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "consuming op must belong to a block");
  }
  if (consuming_op->parent_block->parent_region != query->region) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consumption query must describe the consuming op region");
  }
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  if (loom_consumption_find_same_block_use_after(consuming_op, value_id,
                                                 out_use)) {
    *out_found = true;
    return iree_ok_status();
  }

  const loom_region_t* region = consuming_op->parent_block->parent_region;
  if (!loom_consumption_region_is_cfg(region)) {
    return iree_ok_status();
  }
  const loom_cfg_graph_t* cfg_graph = NULL;
  IREE_RETURN_IF_ERROR(
      loom_consumption_region_query_cfg_graph(query, &cfg_graph));
  return loom_consumption_find_cfg_use_after(query, cfg_graph, consuming_op,
                                             value_id, out_use, out_found);
}
