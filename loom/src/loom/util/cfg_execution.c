// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_execution.h"

#include <string.h>

static iree_status_t loom_cfg_execution_allocate_bitmap(
    iree_host_size_t bit_count, iree_arena_allocator_t* arena,
    iree_bitmap_t* out_bitmap) {
  *out_bitmap = (iree_bitmap_t){.bit_count = bit_count};
  const iree_host_size_t word_count = iree_bitmap_calculate_words(bit_count);
  if (word_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, word_count,
                                                 sizeof(*out_bitmap->words),
                                                 (void**)&out_bitmap->words));
  memset(out_bitmap->words, 0, word_count * sizeof(*out_bitmap->words));
  return iree_ok_status();
}

iree_status_t loom_cfg_execution_classify_unmodeled_blocks(
    const loom_cfg_control_t* control,
    loom_cfg_execution_selector_model_t selector_model,
    iree_arena_allocator_t* arena, iree_bitmap_t* out_unmodeled_blocks) {
  const loom_cfg_graph_t* graph = control->graph;
  IREE_RETURN_IF_ERROR(loom_cfg_execution_allocate_bitmap(
      graph->block_count, arena, out_unmodeled_blocks));
  if (!control->available) {
    for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
      if (graph->blocks[i].reachable) {
        iree_bitmap_set(*out_unmodeled_blocks, i);
      }
    }
    return iree_ok_status();
  }
  if (control->component_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  bool* unmodeled_components = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, control->component_count, sizeof(*unmodeled_components),
      (void**)&unmodeled_components);
  if (iree_status_is_ok(status)) {
    memset(unmodeled_components, 0,
           control->component_count * sizeof(*unmodeled_components));

    // Direct selector inputs establish uncertainty at their controlled path
    // components. Propagation inputs are handled in component order below.
    for (uint32_t i = 0; i < control->input_count; ++i) {
      const loom_cfg_control_input_t* input = &control->inputs[i];
      if (input->edge == LOOM_CFG_EDGE_INDEX_INVALID) {
        continue;
      }
      const uint16_t selector = graph->edges[input->edge].source_block_index;
      const bool modeled =
          selector_model.is_modeled != NULL &&
          selector_model.is_modeled(selector_model.user_data, selector);
      if (!modeled) {
        unmodeled_components[input->target_component] = true;
      }
    }

    // SCCs are retained in successor-before-predecessor order. Walking them
    // backward visits each controller before its controlled successors.
    for (uint32_t source = control->component_count; source > 0; --source) {
      const uint32_t source_component = source - 1;
      if (!unmodeled_components[source_component]) {
        continue;
      }
      for (uint32_t input_index =
               control->components[source_component].outgoing_head;
           input_index != LOOM_CFG_CONTROL_INVALID;
           input_index = control->inputs[input_index].next_outgoing) {
        const uint32_t target_component =
            control->inputs[input_index].target_component;
        IREE_ASSERT_LT(target_component, source_component);
        unmodeled_components[target_component] = true;
      }
    }

    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      const uint32_t node = control->blocks[block_index].node;
      if (node != LOOM_CFG_CONTROL_INVALID &&
          unmodeled_components[control->nodes[node].component]) {
        iree_bitmap_set(*out_unmodeled_blocks, block_index);
      }
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static bool loom_cfg_execution_loop_selector_is_modeled(const void* user_data,
                                                        uint16_t block_index) {
  const loom_cfg_loop_nest_t* loops = user_data;
  const uint16_t loop = loom_cfg_loop_nest_innermost(loops, block_index);
  return loop != LOOM_CFG_LOOP_NEST_NONE &&
         loops->loops[loop].header_index == block_index;
}

iree_status_t loom_cfg_execution_counts_calculate(
    const loom_cfg_loop_nest_t* loops, const uint64_t* trip_counts,
    const loom_cfg_control_t* control, iree_arena_allocator_t* arena,
    uint64_t* storage, loom_cfg_execution_counts_t* out_counts) {
  *out_counts = (loom_cfg_execution_counts_t){0};
  bool single_path = false;
  const bool bounded = loom_cfg_loop_nest_calculate_bounds(
      loops, trip_counts, storage, &single_path);
  if (bounded && single_path) {
    out_counts->single_path = true;
    out_counts->values = storage;
    return iree_ok_status();
  }
  if (!control || !bounded) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_execution_classify_unmodeled_blocks(
      control,
      (loom_cfg_execution_selector_model_t){
          .is_modeled = loom_cfg_execution_loop_selector_is_modeled,
          .user_data = loops,
      },
      arena, &out_counts->unmodeled_blocks));
  out_counts->values = storage;
  return iree_ok_status();
}
