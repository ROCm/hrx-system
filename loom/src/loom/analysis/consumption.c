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

void loom_consumption_region_query_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_query);
  memset(out_query, 0, sizeof(*out_query));
  out_query->module = module;
  out_query->region = region;
  out_query->arena = arena;
}

void loom_consumption_region_query_initialize_with_cfg_graph(
    const loom_module_t* module, const loom_region_t* region,
    const loom_cfg_graph_t* cfg_graph, iree_arena_allocator_t* arena,
    loom_consumption_region_query_t* out_query) {
  loom_consumption_region_query_initialize(module, region, arena, out_query);
  IREE_ASSERT_ARGUMENT(cfg_graph);
  IREE_ASSERT(cfg_graph->module == module);
  IREE_ASSERT(cfg_graph->region == region);
  IREE_ASSERT_EQ(cfg_graph->block_count, region->block_count);
  out_query->cfg_graph = *cfg_graph;
  out_query->cfg_graph_ready = true;
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
  if (visited_word_count > query->reachable_word_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, visited_word_count, sizeof(*query->reachable_bits),
        &query->reachable_word_capacity, (void**)&query->reachable_bits));
  }
  if (block_count > query->block_stack_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, block_count, sizeof(*query->block_stack),
        &query->block_stack_capacity, (void**)&query->block_stack));
  }
  if (visited_word_count > 0) {
    memset(query->visited_bits, 0,
           visited_word_count * sizeof(query->visited_bits[0]));
    memset(query->reachable_bits, 0,
           visited_word_count * sizeof(query->reachable_bits[0]));
  }
  *out_visited_word_count = visited_word_count;
  return iree_ok_status();
}

static bool loom_consumption_block_arg_defines_value(const loom_block_t* block,
                                                     loom_value_id_t value_id) {
  if (!block) return false;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (loom_block_arg_id(block, i) == value_id) return true;
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

static iree_status_t loom_consumption_prepare_cfg_reachability(
    loom_consumption_region_query_t* query, const loom_cfg_graph_t* graph,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    iree_host_size_t* out_reachable_word_count) {
  *out_reachable_word_count = 0;
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
  *out_reachable_word_count = visited_word_count;
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
    loom_consumption_bitset_set(query->reachable_bits, visited_word_count,
                                block_index);

    successors = loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      loom_consumption_cfg_search_push(graph, successors.values[i],
                                       query->visited_bits, visited_word_count,
                                       query->block_stack, &stack_count);
    }
  }
  return iree_ok_status();
}

static const loom_op_t* loom_consumption_region_anchor_op(
    const loom_region_t* region, const loom_op_t* op) {
  const loom_op_t* anchor_op = op;
  while (anchor_op != NULL && anchor_op->parent_block != NULL &&
         anchor_op->parent_block->parent_region != region) {
    anchor_op = anchor_op->parent_op;
  }
  if (anchor_op == NULL || anchor_op->parent_block == NULL ||
      anchor_op->parent_block->parent_region != region) {
    return NULL;
  }
  return anchor_op;
}

iree_status_t loom_consumption_use_after_query_prepare(
    loom_consumption_region_query_t* region_query,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    loom_consumption_use_after_query_t* out_query) {
  if (!region_query || !consuming_op || !out_query) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "consumption use-after query requires region query, operation, and "
        "output query");
  }
  *out_query = (loom_consumption_use_after_query_t){
      .region_query = region_query,
      .consuming_op = consuming_op,
      .value_id = value_id,
  };
  if (!region_query->module || !region_query->region || !region_query->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "consumption region query is not initialized");
  }
  if (!consuming_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "consuming op must belong to a block");
  }
  if (consuming_op->parent_block->parent_region != region_query->region) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consumption query must describe the consuming op region");
  }
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= region_query->module->values.count ||
      !loom_consumption_region_is_cfg(region_query->region)) {
    return iree_ok_status();
  }
  const loom_cfg_graph_t* cfg_graph = NULL;
  IREE_RETURN_IF_ERROR(
      loom_consumption_region_query_cfg_graph(region_query, &cfg_graph));
  return loom_consumption_prepare_cfg_reachability(
      region_query, cfg_graph, consuming_op, value_id,
      &out_query->reachable_word_count);
}

bool loom_consumption_use_after_query_contains(
    const loom_consumption_use_after_query_t* query, loom_use_t use) {
  IREE_ASSERT_ARGUMENT(query);
  IREE_ASSERT_ARGUMENT(query->region_query);
  IREE_ASSERT_ARGUMENT(query->consuming_op);
  const loom_consumption_region_query_t* region_query = query->region_query;
  const loom_op_t* use_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  IREE_ASSERT_LT(operand_index, use_op->operand_count);
  IREE_ASSERT_EQ(loom_op_const_operands(use_op)[operand_index],
                 query->value_id);
  const loom_op_t* anchor_op =
      loom_consumption_region_anchor_op(region_query->region, use_op);
  if (anchor_op == NULL) {
    return false;
  }
  const loom_block_t* anchor_block = anchor_op->parent_block;
  const loom_block_t* consuming_block = query->consuming_op->parent_block;
  if (anchor_block == consuming_block &&
      anchor_op->block_ordinal > query->consuming_op->block_ordinal) {
    return true;
  }
  if (query->reachable_word_count == 0) {
    return false;
  }
  const loom_cfg_graph_t* graph = &region_query->cfg_graph;
  const iree_host_size_t block_index =
      loom_cfg_graph_block_index(graph, anchor_block);
  return block_index != IREE_HOST_SIZE_MAX &&
         loom_consumption_bitset_test(region_query->reachable_bits,
                                      query->reachable_word_count, block_index);
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
  loom_consumption_use_after_query_t use_after_query = {0};
  IREE_RETURN_IF_ERROR(loom_consumption_use_after_query_prepare(
      query, consuming_op, value_id, &use_after_query));
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(query->module, value_id);
  const loom_use_t* use_ptr = NULL;
  loom_value_for_each_use(value, use_ptr) {
    if (!loom_consumption_use_after_query_contains(&use_after_query,
                                                   *use_ptr)) {
      continue;
    }
    *out_use = (loom_consumption_use_t){
        .op = loom_use_user_op(*use_ptr),
        .operand_index = loom_use_operand_index(*use_ptr),
    };
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}
