// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_cfg.h"

#include <string.h>

#include "loom/util/fact_table.h"

static void loom_value_fact_cfg_seed_selector(
    const loom_value_fact_table_t* table,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index) {
  if (!region->control_structure.node_count ||
      !region->control_structure.blocks[block_index].binding_count) {
    return;
  }
  const loom_value_id_t selector =
      loom_cfg_graph_refresh_selector(&region->graph, block_index);
  loom_value_facts_t facts = loom_value_fact_table_lookup(table, selector);
  if (selector != LOOM_VALUE_ID_INVALID &&
      !loom_value_fact_table_has_entry(table, selector)) {
    loom_value_facts_mark_cluster_uniform(&facts);
  }
  loom_value_fact_control_set_selector(region->control, block_index, facts);
}

bool loom_value_fact_cfg_update_control(
    const loom_value_fact_table_t* table,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index) {
  loom_value_fact_cfg_seed_selector(table, region, block_index);
  return loom_value_fact_control_settle(region->control);
}

void loom_value_fact_cfg_seed_control(
    const loom_value_fact_table_t* table,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component) {
  const iree_host_size_t count =
      component ? component->node_count : region->graph.block_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_value_fact_cfg_seed_selector(table, region,
                                      component ? component->nodes[i] : i);
  }
  loom_value_fact_control_settle(region->control);
}

iree_host_size_t loom_value_fact_cfg_region_argument_index(
    const loom_value_fact_cfg_region_t* region, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(region->graph.module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return IREE_HOST_SIZE_MAX;
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&region->graph, loom_value_def_block(value));
  if (block_index == IREE_HOST_SIZE_MAX || block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(&region->graph, block_index)) {
    return IREE_HOST_SIZE_MAX;
  }
  return region->argument_offsets[block_index] + loom_value_def_index(value);
}

typedef struct loom_value_fact_cfg_forwarding_graph_t {
  // Region owning the indexed arguments and current branch payloads.
  const loom_value_fact_cfg_region_t* region;
  // Control-flow component's contiguous argument partition.
  const loom_value_fact_cfg_forwarding_t* partition;
} loom_value_fact_cfg_forwarding_graph_t;

static iree_status_t loom_value_fact_cfg_visit_forwarded_arguments(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_value_fact_cfg_forwarding_graph_t* forwarding_graph = user_data;
  const loom_value_fact_cfg_region_t* region = forwarding_graph->region;
  const loom_value_fact_cfg_forwarding_t* partition =
      forwarding_graph->partition;
  const loom_value_fact_cfg_argument_t* argument =
      &region->arguments[partition->argument_offset + node];
  const loom_cfg_graph_t* graph = &region->graph;
  const loom_block_t* block = graph->blocks[argument->block_index].block;
  loom_cfg_edge_index_span_t incoming =
      loom_cfg_graph_predecessor_edges(graph, argument->block_index);
  for (iree_host_size_t i = 0; i < incoming.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(graph, incoming.values[i]);
    if (!loom_cfg_graph_block_is_reachable(graph, edge->source_block_index)) {
      continue;
    }
    const loom_value_id_t* sources = NULL;
    uint16_t count = 0;
    if (!loom_cfg_terminator_payload_for_successor(edge->terminator, block,
                                                   &sources, &count) ||
        argument->argument_index >= count) {
      continue;
    }
    iree_host_size_t source = loom_value_fact_cfg_region_argument_index(
        region, sources[argument->argument_index]);
    if (source >= partition->argument_offset &&
        source - partition->argument_offset < partition->argument_count) {
      IREE_RETURN_IF_ERROR(successor.fn(successor.user_data,
                                        source - partition->argument_offset));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_cfg_update_forwarding(
    const loom_value_fact_cfg_region_t* region,
    iree_host_size_t component_index, iree_arena_allocator_t* scratch_arena) {
  loom_value_fact_cfg_forwarding_t* partition =
      &region->control_flow.forwarding[component_index];
  if (!partition->dirty) {
    return iree_ok_status();
  }
  loom_value_fact_cfg_forwarding_graph_t forwarding_graph = {
      .region = region,
      .partition = partition,
  };
  const loom_scc_graph_t graph = {
      .node_count = partition->argument_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_value_fact_cfg_visit_forwarded_arguments, &forwarding_graph),
  };
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_scc_list_t components = {0};
  iree_status_t status =
      loom_scc_compute(&graph, NULL, scratch_arena, &components);
  if (iree_status_is_ok(status)) {
    iree_host_size_t member_offset = partition->argument_offset;
    for (iree_host_size_t i = 0; i < components.count; ++i) {
      const loom_scc_t* component = &components.values[i];
      iree_host_size_t retained_index = partition->argument_offset + i;
      region->components[retained_index] = (loom_scc_t){
          .nodes = region->component_nodes + member_offset,
          .node_count = component->node_count,
          .is_cycle = component->is_cycle,
      };
      for (iree_host_size_t j = 0; j < component->node_count; ++j) {
        iree_host_size_t argument_index =
            partition->argument_offset + component->nodes[j];
        region->component_nodes[member_offset++] = argument_index;
        region->argument_components[argument_index] = retained_index;
      }
    }
    partition->component_count = components.count;
    partition->dirty = false;
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

// Group graph-owned component IDs into the member spans used by fact solves.
// The entry reaches every retained component, so its ordinal is the largest.
static iree_status_t loom_value_fact_cfg_group_control_flow(
    const loom_cfg_graph_t* graph, iree_arena_allocator_t* arena,
    loom_scc_list_t* out_components) {
  const iree_host_size_t component_count = graph->blocks[0].component + 1;
  loom_scc_t* components = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, component_count, sizeof(*components), (void**)&components));
  memset(components, 0, component_count * sizeof(*components));
  iree_host_size_t* nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->reverse_postorder.count, sizeof(*nodes), (void**)&nodes));
  for (uint16_t i = 0; i < graph->block_count; ++i) {
    const loom_cfg_block_info_t* block = &graph->blocks[i];
    if (!block->reachable) {
      continue;
    }
    ++components[block->component].node_count;
    components[block->component].is_cycle = block->component_is_cyclic;
  }
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < component_count; ++i) {
    components[i].nodes = nodes + offset;
    offset += components[i].node_count;
    components[i].node_count = 0;
  }
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    const uint16_t block_index = graph->reverse_postorder.values[i];
    const loom_cfg_block_info_t* block = &graph->blocks[block_index];
    loom_scc_t* component = &components[block->component];
    nodes[(component->nodes - nodes) + component->node_count++] = block_index;
  }
  *out_components = (loom_scc_list_t){
      .values = components,
      .count = component_count,
  };
  return iree_ok_status();
}

iree_status_t loom_value_fact_cfg_region_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_value_fact_cfg_region_t* out_region) {
  *out_region = (loom_value_fact_cfg_region_t){0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(module, region, arena, &out_region->graph));
  IREE_RETURN_IF_ERROR(loom_cfg_dominance_build(&out_region->graph, arena,
                                                &out_region->dominance));
  IREE_RETURN_IF_ERROR(loom_cfg_regions_build(
      &out_region->graph, &out_region->dominance, arena, &out_region->regions));
  IREE_RETURN_IF_ERROR(loom_cfg_loop_nest_build(
      &out_region->graph, &out_region->dominance, arena, &out_region->loops));
  if (out_region->loops.loop_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_region->loops.loop_count, sizeof(*out_region->inductions),
        (void**)&out_region->inductions));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_region->loops.loop_count, sizeof(*out_region->recurrences),
        (void**)&out_region->recurrences));
    memset(out_region->recurrences, 0,
           out_region->loops.loop_count * sizeof(*out_region->recurrences));
    for (iree_host_size_t i = 0; i < out_region->loops.loop_count; ++i) {
      out_region->inductions[i] = (loom_value_fact_induction_t){
          .value = LOOM_VALUE_ID_INVALID,
      };
    }
  }
  IREE_RETURN_IF_ERROR(loom_cfg_control_build(&out_region->graph, arena,
                                              &out_region->control_structure));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*out_region->control),
                                           (void**)&out_region->control));
  IREE_RETURN_IF_ERROR(loom_value_fact_control_initialize(
      &out_region->control_structure, arena, out_region->control));
  out_region->control_flow.dirty.bit_count =
      out_region->graph.block_count
          ? (iree_host_size_t)out_region->graph.blocks[0].component + 1
          : 0;
  const iree_host_size_t dirty_word_count =
      iree_bitmap_calculate_words(out_region->control_flow.dirty.bit_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, dirty_word_count, sizeof(*out_region->control_flow.dirty.words),
      (void**)&out_region->control_flow.dirty.words));
  iree_bitmap_reset_all(out_region->control_flow.dirty);
  if (!out_region->graph.has_cycles) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_group_control_flow(
      &out_region->graph, arena, &out_region->control_flow.components));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_region->control_flow.components.count,
      sizeof(*out_region->control_flow.anchors),
      (void**)&out_region->control_flow.anchors));
  memset(out_region->control_flow.anchors, 0,
         out_region->control_flow.components.count *
             sizeof(*out_region->control_flow.anchors));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_region->control_flow.components.count,
      sizeof(*out_region->control_flow.forwarding),
      (void**)&out_region->control_flow.forwarding));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, region->block_count, sizeof(*out_region->argument_offsets),
      (void**)&out_region->argument_offsets));
  for (iree_host_size_t i = 0; i < out_region->control_flow.components.count;
       ++i) {
    const loom_scc_t* component =
        &out_region->control_flow.components.values[i];
    loom_value_fact_cfg_forwarding_t* partition =
        &out_region->control_flow.forwarding[i];
    *partition = (loom_value_fact_cfg_forwarding_t){
        .argument_offset = out_region->argument_count,
    };
    if (component->is_cycle) {
      // Every cycle owns a summary even without carried arguments: repeated
      // observations can change facts when its execution scope changes.
      out_region->control_flow.anchors[i] =
          out_region->graph.blocks[component->nodes[0]].block->last_op;
    }
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      iree_host_size_t block_index = component->nodes[j];
      const loom_block_t* block = out_region->graph.blocks[block_index].block;
      out_region->argument_offsets[block_index] = out_region->argument_count;
      if (block_index != 0) {
        out_region->argument_count += block->arg_count;
      }
    }
    partition->argument_count =
        out_region->argument_count - partition->argument_offset;
    partition->component_count = partition->argument_count;
    partition->dirty = component->is_cycle && partition->argument_count != 0;
  }
  if (out_region->argument_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_region->argument_count, sizeof(*out_region->arguments),
      (void**)&out_region->arguments));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, out_region->argument_count,
                                sizeof(*out_region->argument_components),
                                (void**)&out_region->argument_components));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_region->argument_count, sizeof(*out_region->components),
      (void**)&out_region->components));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_region->argument_count, sizeof(*out_region->component_nodes),
      (void**)&out_region->component_nodes));
  for (iree_host_size_t i = 0; i < out_region->argument_count; ++i) {
    out_region->argument_components[i] = i;
    out_region->component_nodes[i] = i;
    out_region->components[i] = (loom_scc_t){
        .nodes = out_region->component_nodes + i,
        .node_count = 1,
    };
  }
  for (uint16_t i = 1; i < region->block_count; ++i) {
    if (!loom_cfg_graph_block_is_reachable(&out_region->graph, i)) {
      continue;
    }
    const loom_block_t* block = out_region->graph.blocks[i].block;
    for (uint16_t j = 0; j < block->arg_count; ++j) {
      out_region->arguments[out_region->argument_offsets[i] + j] =
          (loom_value_fact_cfg_argument_t){
              .value_id = loom_block_arg_id(block, j),
              .block_index = i,
              .argument_index = j,
          };
    }
  }
  for (iree_host_size_t i = 0; i < out_region->control_flow.components.count;
       ++i) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_cfg_update_forwarding(out_region, i, arena));
  }
  return iree_ok_status();
}

// Retained arrays in construction order, shared by sizing and copying. Control
// state follows its header so its member pointers can be assigned directly.
// Nested member spans and snapshot-local pointers are rebound after copying.
#define LOOM_CFG_PACK_FIELDS(F)                                              \
  F(graph.blocks, source->graph.block_count)                                 \
  F(graph.edges, source->graph.edge_count)                                   \
  F(graph.successor_indices, source->graph.edge_count)                       \
  F(graph.successor_edge_indices, source->graph.edge_count)                  \
  F(graph.predecessor_indices, source->graph.edge_count)                     \
  F(graph.predecessor_edge_indices, source->graph.edge_count)                \
  F(graph.reverse_postorder.values, source->graph.reverse_postorder.count)   \
  F(dominance.immediate_dominators, source->graph.block_count)               \
  F(dominance.intervals, source->graph.block_count)                          \
  F(dominance.entry_predecessors, source->graph.block_count)                 \
  F(dominance.preorder.values, source->dominance.preorder.count)             \
  F(regions.blocks, source->graph.block_count)                               \
  F(regions.incoming_edges, source->graph.edge_count)                        \
  F(regions.incoming_offsets, source->graph.block_count + 1)                 \
  F(loops.loops, source->loops.loop_count)                                   \
  F(loops.innermost_loop_indices, source->graph.block_count)                 \
  F(inductions, source->loops.loop_count)                                    \
  F(recurrences, source->loops.loop_count)                                   \
  F(control_structure.postdominance.nodes, source->graph.block_count + 1)    \
  F(control_structure.blocks, source->graph.block_count)                     \
  F(control_structure.nodes, source->control_structure.node_count)           \
  F(control_structure.bindings, source->control_structure.binding_count)     \
  F(control_structure.inputs, source->control_structure.input_count)         \
  F(control_structure.components, source->control_structure.component_count) \
  F(control, 1)                                                              \
  F(control->inputs, source->control_structure.input_count)                  \
  F(control->components, source->control_structure.component_count)          \
  F(control->blocks, source->graph.block_count)                              \
  F(control->worklist, source->control_structure.component_count)            \
  F(control->pending_blocks, source->graph.block_count)                      \
  F(argument_offsets, source->graph.block_count)                             \
  F(arguments, source->argument_count)                                       \
  F(components, source->argument_count)                                      \
  F(component_nodes, source->argument_count)                                 \
  F(argument_components, source->argument_count)                             \
  F(control_flow.components.values, source->control_flow.components.count)   \
  F(control_flow.forwarding, source->control_flow.components.count)          \
  F(control_flow.anchors, source->control_flow.components.count)             \
  F(control_flow.dirty.words,                                                \
    iree_bitmap_calculate_words(source->control_flow.dirty.bit_count))

static iree_host_size_t loom_value_fact_cfg_aligned_size(
    iree_host_size_t size) {
  return iree_host_align(size, iree_alignof(iree_max_align_t));
}

static void* loom_value_fact_cfg_pack_array(uint8_t** cursor,
                                            const void* source,
                                            iree_host_size_t size) {
  if (!size) {
    return NULL;
  }
  void* result = *cursor;
  memcpy(result, source, size);
  *cursor += loom_value_fact_cfg_aligned_size(size);
  return result;
}

iree_status_t loom_value_fact_cfg_region_create_packed(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* scratch, iree_allocator_t allocator,
    loom_value_fact_cfg_region_t** out_region) {
  *out_region = NULL;
  loom_value_fact_cfg_region_t constructed;
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_region_initialize(
      module, region, scratch, &constructed));
  // Construction precedes numeric publication. Recurrence slots therefore
  // contain no borrowed solve storage, and control worklists are empty.
  const loom_value_fact_cfg_region_t* source = &constructed;
  iree_host_size_t size = loom_value_fact_cfg_aligned_size(sizeof(*source));
#define LOOM_CFG_PACK_SIZE(member, count)                         \
  size += source->member ? loom_value_fact_cfg_aligned_size(      \
                               (count) * sizeof(*source->member)) \
                         : 0;
  LOOM_CFG_PACK_FIELDS(LOOM_CFG_PACK_SIZE)
#undef LOOM_CFG_PACK_SIZE
  const iree_host_size_t member_count =
      source->control_flow.components.count
          ? source->graph.reverse_postorder.count
          : 0;
  size +=
      loom_value_fact_cfg_aligned_size(member_count * sizeof(iree_host_size_t));
  loom_value_fact_cfg_region_t* result = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_uninitialized(allocator, size, (void**)&result));
  *result = *source;
  uint8_t* cursor =
      (uint8_t*)result + loom_value_fact_cfg_aligned_size(sizeof(*result));
#define LOOM_CFG_PACK_COPY(member, count)          \
  result->member = loom_value_fact_cfg_pack_array( \
      &cursor, source->member,                     \
      source->member ? (count) * sizeof(*source->member) : 0);
  LOOM_CFG_PACK_FIELDS(LOOM_CFG_PACK_COPY)
#undef LOOM_CFG_PACK_COPY
  const iree_host_size_t* source_members =
      member_count ? source->control_flow.components.values[0].nodes : NULL;
  const iree_host_size_t* result_members = loom_value_fact_cfg_pack_array(
      &cursor, source_members, member_count * sizeof(*source_members));

  // External IR identities remain borrowed. Only aliases within the completed
  // snapshot move with its retained arrays.
  result->loops.graph = &result->graph;
  result->control_structure.graph = &result->graph;
  result->control->structure = &result->control_structure;
  for (iree_host_size_t i = 0;
       result->regions.blocks && i < result->graph.block_count; ++i) {
    if (source->regions.blocks[i].exit_edges.values) {
      result->regions.blocks[i].exit_edges.values =
          result->regions.incoming_edges +
          (source->regions.blocks[i].exit_edges.values -
           source->regions.incoming_edges);
    }
  }
  for (iree_host_size_t i = 0; i < result->argument_count; ++i) {
    result->components[i].nodes =
        result->component_nodes +
        (source->components[i].nodes - source->component_nodes);
  }
  loom_scc_t* components = (loom_scc_t*)result->control_flow.components.values;
  for (iree_host_size_t i = 0; i < result->control_flow.components.count; ++i) {
    components[i].nodes =
        result_members +
        (source->control_flow.components.values[i].nodes - source_members);
  }
  *out_region = result;
  return iree_ok_status();
}

#undef LOOM_CFG_PACK_FIELDS
