// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_value_identity.h"

#include <string.h>

#include "loom/ir/types.h"

#define LOOM_CFG_VALUE_IDENTITY_INVALID UINT32_MAX

// A call-scoped projection of branch payloads onto retained CFG structure.
// Payloads are copied once because the refinement repeatedly indexes rows.
typedef struct loom_cfg_value_identity_projection_t {
  // Retained CFG and forwarding components borrowed from the fact scope.
  const loom_value_fact_cfg_region_t* region;

  // Source value rows indexed through edge_offsets.
  loom_value_id_t* sources;

  // Source row starts indexed by CFG edge, or INVALID for ignored edges.
  uint32_t* edge_offsets;

  // Argument row starts for acyclic blocks.
  uint32_t* block_offsets;

  // Argument descriptors synthesized for an acyclic CFG.
  loom_value_fact_cfg_argument_t* acyclic_arguments;

  // Block index keyed by the acyclic graph's component order.
  uint16_t* component_blocks;

  // Number of reachable non-entry block arguments.
  uint32_t argument_count;

  // Number of incoming source occurrences.
  uint32_t source_count;
} loom_cfg_value_identity_projection_t;

static const loom_value_fact_cfg_argument_t*
loom_cfg_value_identity_projection_argument(
    const loom_cfg_value_identity_projection_t* projection, uint32_t argument) {
  return projection->region->argument_count != 0
             ? &projection->region->arguments[argument]
             : &projection->acyclic_arguments[argument];
}

static uint32_t loom_cfg_value_identity_projection_block_offset(
    const loom_cfg_value_identity_projection_t* projection,
    uint16_t block_index) {
  return projection->region->argument_count != 0
             ? (uint32_t)projection->region->argument_offsets[block_index]
             : projection->block_offsets[block_index];
}

static uint32_t loom_cfg_value_identity_projection_argument_ordinal(
    const loom_cfg_value_identity_projection_t* projection,
    loom_value_id_t value_id) {
  const loom_cfg_graph_t* graph = &projection->region->graph;
  const loom_value_t* value = loom_module_value(graph->module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return LOOM_CFG_VALUE_IDENTITY_INVALID;
  }
  const loom_block_t* block = loom_value_def_block(value);
  if (block->parent_region != graph->region || block->region_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block->region_index)) {
    return LOOM_CFG_VALUE_IDENTITY_INVALID;
  }
  return loom_cfg_value_identity_projection_block_offset(projection,
                                                         block->region_index) +
         loom_value_def_index(value);
}

static iree_status_t loom_cfg_value_identity_count_arguments(
    const loom_value_fact_cfg_region_t* region, uint32_t* out_count) {
  if (region->argument_count != 0) {
    if (region->argument_count >= LOOM_CFG_VALUE_IDENTITY_INVALID) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "CFG argument domain exceeds compact index");
    }
    *out_count = (uint32_t)region->argument_count;
    return iree_ok_status();
  }

  uint32_t count = 0;
  const loom_cfg_graph_t* graph = &region->graph;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    const uint32_t block_argument_count =
        graph->blocks[block_index].block->arg_count;
    if (block_argument_count > LOOM_CFG_VALUE_IDENTITY_INVALID - 1 - count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "CFG argument domain exceeds compact index");
    }
    count += block_argument_count;
  }
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_cfg_value_identity_count_sources(
    const loom_cfg_graph_t* graph, uint32_t* out_count, bool* out_supported) {
  *out_count = 0;
  *out_supported = true;
  for (iree_host_size_t i = 0; i < graph->edge_count; ++i) {
    const loom_cfg_edge_info_t* edge = &graph->edges[i];
    if (!loom_cfg_graph_block_is_reachable(graph, edge->source_block_index) ||
        edge->target_block_index == 0) {
      continue;
    }
    const loom_block_t* target = graph->blocks[edge->target_block_index].block;
    if (target->arg_count == 0) {
      continue;
    }
    const loom_value_id_t* sources = NULL;
    uint16_t source_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(edge->terminator, target,
                                                   &sources, &source_count)) {
      *out_supported = false;
      return iree_ok_status();
    }
    if ((uint32_t)source_count >
        LOOM_CFG_VALUE_IDENTITY_INVALID - 1 - *out_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "CFG payload domain exceeds compact index");
    }
    *out_count += source_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_value_identity_projection_initialize(
    const loom_value_fact_cfg_region_t* region, iree_arena_allocator_t* arena,
    loom_cfg_value_identity_projection_t* out_projection, bool* out_supported) {
  *out_projection = (loom_cfg_value_identity_projection_t){
      .region = region,
  };
  *out_supported = true;
  const loom_cfg_graph_t* graph = &region->graph;
  if (graph->malformed || graph->block_count == 0) {
    *out_supported = false;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_count_arguments(
      region, &out_projection->argument_count));
  if (out_projection->argument_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_count_sources(
      graph, &out_projection->source_count, out_supported));
  if (!*out_supported) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->edge_count, sizeof(*out_projection->edge_offsets),
      (void**)&out_projection->edge_offsets));
  memset(out_projection->edge_offsets, 0xFF,
         graph->edge_count * sizeof(*out_projection->edge_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_projection->source_count, sizeof(*out_projection->sources),
      (void**)&out_projection->sources));

  if (region->argument_count == 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->block_count, sizeof(*out_projection->block_offsets),
        (void**)&out_projection->block_offsets));
    memset(out_projection->block_offsets, 0xFF,
           graph->block_count * sizeof(*out_projection->block_offsets));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_projection->argument_count,
                                  sizeof(*out_projection->acyclic_arguments),
                                  (void**)&out_projection->acyclic_arguments));
    const uint32_t component_count = graph->blocks[0].component + 1;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, component_count, sizeof(*out_projection->component_blocks),
        (void**)&out_projection->component_blocks));
    memset(out_projection->component_blocks, 0,
           component_count * sizeof(*out_projection->component_blocks));
    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      if (loom_cfg_graph_block_is_reachable(graph, block_index)) {
        out_projection->component_blocks[graph->blocks[block_index].component] =
            block_index;
      }
    }
    uint32_t argument = 0;
    for (uint32_t component = 0; component < component_count; ++component) {
      const uint16_t block_index = out_projection->component_blocks[component];
      out_projection->block_offsets[block_index] = argument;
      if (block_index == 0) {
        continue;
      }
      const loom_block_t* block = graph->blocks[block_index].block;
      for (uint16_t block_argument = 0; block_argument < block->arg_count;
           ++block_argument) {
        out_projection->acyclic_arguments[argument++] =
            (loom_value_fact_cfg_argument_t){
                .value_id = loom_block_arg_id(block, block_argument),
                .block_index = block_index,
                .argument_index = block_argument,
            };
      }
    }
    IREE_ASSERT_EQ(argument, out_projection->argument_count);
  }

  uint32_t source_offset = 0;
  for (iree_host_size_t i = 0; i < graph->edge_count; ++i) {
    const loom_cfg_edge_info_t* edge = &graph->edges[i];
    if (!loom_cfg_graph_block_is_reachable(graph, edge->source_block_index) ||
        edge->target_block_index == 0) {
      continue;
    }
    const loom_block_t* target = graph->blocks[edge->target_block_index].block;
    if (target->arg_count == 0) {
      continue;
    }
    const loom_value_id_t* sources = NULL;
    uint16_t source_count = 0;
    const bool has_payload = loom_cfg_terminator_payload_for_successor(
        edge->terminator, target, &sources, &source_count);
    IREE_ASSERT(has_payload);
    out_projection->edge_offsets[i] = source_offset;
    memcpy(out_projection->sources + source_offset, sources,
           source_count * sizeof(*sources));
    source_offset += source_count;
  }
  IREE_ASSERT_EQ(source_offset, out_projection->source_count);
  return iree_ok_status();
}

static void loom_cfg_value_identity_projection_assumption_span(
    const loom_cfg_value_identity_projection_t* projection,
    const uint32_t* assumption_components, uint32_t assumption,
    iree_host_size_t* singleton, const iree_host_size_t** out_members,
    iree_host_size_t* out_member_count) {
  if (projection->region->argument_count != 0) {
    const loom_scc_t* component =
        &projection->region->components[assumption_components[assumption]];
    *out_members = component->nodes;
    *out_member_count = component->node_count;
  } else {
    *singleton = assumption;
    *out_members = singleton;
    *out_member_count = 1;
  }
}

static bool loom_cfg_value_identity_projection_source_is_internal(
    const loom_cfg_value_identity_projection_t* projection,
    loom_value_id_t source, uint32_t member) {
  const uint32_t ordinal =
      loom_cfg_value_identity_projection_argument_ordinal(projection, source);
  return projection->region->argument_count != 0 &&
         ordinal != LOOM_CFG_VALUE_IDENTITY_INVALID &&
         projection->region->argument_components[ordinal] ==
             projection->region->argument_components[member];
}

static bool loom_cfg_value_identity_projection_value_available(
    const loom_cfg_value_identity_projection_t* projection,
    const loom_dominance_info_t* dominance, loom_value_id_t value_id,
    const iree_host_size_t* members, iree_host_size_t member_count) {
  const loom_module_t* module = projection->region->graph.module;
  const loom_value_t* value = loom_module_value(module, value_id);
  for (iree_host_size_t i = 0; i < member_count; ++i) {
    const loom_value_fact_cfg_argument_t* argument =
        loom_cfg_value_identity_projection_argument(projection,
                                                    (uint32_t)members[i]);
    const loom_block_t* block =
        projection->region->graph.blocks[argument->block_index].block;
    if ((loom_value_is_block_arg(value) &&
         loom_value_def_block(value) == block) ||
        !loom_value_is_available_before_op(dominance, value_id,
                                           block->first_op) ||
        !loom_value_type_is_available_before_op(dominance, value_id,
                                                block->first_op)) {
      return false;
    }
  }
  return true;
}

typedef struct loom_cfg_value_identity_value_t {
  // SSA value represented by this node.
  loom_value_id_t value_id;

  // Provisional forest parent, or INVALID for a root.
  uint32_t parent;

  // Current forest component and partition state.
  uint32_t component;

  // First incident provisional forest edge.
  uint32_t first_edge;

  // Previous member in the current forest component.
  uint32_t previous_member;

  // Next member in the current forest component.
  uint32_t next_member;

  // First equation occurrence using this value.
  uint32_t first_occurrence;

  // First retained obligation endpoint using this value.
  uint32_t first_endpoint;
} loom_cfg_value_identity_value_t;

typedef struct loom_cfg_value_identity_component_t {
  // Root value determining this component's active equation or leaf label.
  uint32_t root;

  // First value in the component member list.
  uint32_t first_member;

  // Number of values in the component.
  uint32_t member_count;
} loom_cfg_value_identity_component_t;

typedef struct loom_cfg_value_identity_forest_edge_t {
  // Value node at the other end of this undirected edge.
  uint32_t target;

  // Previous edge in the source value's incident list.
  uint32_t previous;

  // Next edge in the source value's incident list.
  uint32_t next;
} loom_cfg_value_identity_forest_edge_t;

typedef struct loom_cfg_value_identity_visit_t {
  // Value node reached by the forest walk.
  uint32_t node;

  // Value node from which node was reached.
  uint32_t parent;
} loom_cfg_value_identity_visit_t;

typedef struct loom_cfg_value_identity_search_t {
  // Storage for values reached from one side of a cut.
  loom_cfg_value_identity_visit_t* visits;

  // Number of reached values.
  uint32_t count;

  // Current breadth-first position.
  uint32_t position;

  // Next incident edge at the current position.
  uint32_t edge;
} loom_cfg_value_identity_search_t;

typedef struct loom_cfg_value_identity_state_t {
  // Current stable partition group.
  uint32_t group;

  // Previous state in the partition group.
  uint32_t previous;

  // Next state in the partition group.
  uint32_t next;

  // First active source occurrence resolved to this state.
  uint32_t first_inverse;

  // First obligation endpoint resolved to this state.
  uint32_t first_endpoint;

  // Hash of the active equation's label and transitions.
  uint64_t signature;
} loom_cfg_value_identity_state_t;

typedef struct loom_cfg_value_identity_group_t {
  // First state in this stable partition group.
  uint32_t first_state;

  // Number of states in the group.
  uint32_t state_count;

  // First temporarily marked state while processing a splitter.
  uint32_t first_marked;

  // Number of temporarily marked states.
  uint32_t marked_count;
} loom_cfg_value_identity_group_t;

typedef struct loom_cfg_value_identity_occurrence_t {
  // Value node supplying the transition.
  uint32_t source_value;

  // Argument value node whose active equation owns the transition.
  uint32_t argument;

  // CFG edge distinguishing this transition position.
  uint32_t edge;

  // Next occurrence sourced from the same value node.
  uint32_t value_next;

  // Previous occurrence in the resolved source state's inverse list.
  uint32_t inverse_previous;

  // Next occurrence in the resolved source state's inverse list.
  uint32_t inverse_next;
} loom_cfg_value_identity_occurrence_t;

typedef struct loom_cfg_value_identity_assumption_t {
  // Value node for source.
  uint32_t candidate_value;

  // First external incoming endpoint that must equal the candidate.
  uint32_t first_endpoint;

  // True while the provisional equality remains in the forest.
  bool active;

  // True while this assumption has a pending obligation check.
  bool queued;
} loom_cfg_value_identity_assumption_t;

typedef struct loom_cfg_value_identity_endpoint_t {
  // Assumption owning this endpoint.
  uint32_t assumption;

  // Value node followed by this endpoint.
  uint32_t value;

  // Next endpoint following the same value node.
  uint32_t value_next;

  // Previous endpoint in the resolved state's list.
  uint32_t state_previous;

  // Next endpoint in the resolved state's list.
  uint32_t state_next;

  // Next external endpoint owned by the same assumption.
  uint32_t assumption_next;
} loom_cfg_value_identity_endpoint_t;

typedef struct loom_cfg_value_identity_refinement_t {
  // Branch payload projection being refined.
  const loom_cfg_value_identity_projection_t* projection;

  // Function-wide direct representative table.
  const loom_cfg_value_identity_table_t* table;

  // Dominance information used to qualify publishable representatives.
  const loom_dominance_info_t* dominance;

  // Arena owning all call-scoped refinement storage.
  iree_arena_allocator_t* arena;

  // Existing forwarding component selected for each assumption.
  const uint32_t* assumption_components;

  // Number of provisional assumptions.
  uint32_t assumption_count;

  // Number of reachable non-entry block arguments.
  uint32_t argument_count;

  // Number of distinct values in equations and obligations.
  uint32_t value_count;

  // Power-of-two capacity of the initial label hash table.
  uint32_t group_bucket_count;

  // Value nodes indexed by compact refinement value.
  loom_cfg_value_identity_value_t* values;

  // Forest components, each of which is one partition state.
  loom_cfg_value_identity_component_t* components;

  // Undirected forest edges, two entries per provisional argument link.
  loom_cfg_value_identity_forest_edge_t* forest_edges;

  // Two disjoint workspaces for simultaneous cut-side searches.
  loom_cfg_value_identity_visit_t* visits;

  // Number of live forest components and partition states.
  uint32_t component_count;

  // Stable partition states indexed identically to components.
  loom_cfg_value_identity_state_t* states;

  // Stable partition groups.
  loom_cfg_value_identity_group_t* groups;

  // Open-addressed initial label table containing representative states.
  uint32_t* group_buckets;

  // Number of live stable partition groups.
  uint32_t group_count;

  // Number of groups already used as partition splitters.
  uint32_t processed_group_count;

  // Active equation source occurrences.
  loom_cfg_value_identity_occurrence_t* occurrences;

  // Occurrence row starts indexed by argument ordinal.
  uint32_t* occurrence_offsets;

  // Number of populated source occurrences.
  uint32_t occurrence_count;

  // Temporary occurrence rows indexed by CFG edge.
  uint32_t* row_heads;

  // Temporary next links for edge occurrence rows.
  uint32_t* row_next;

  // CFG edges touched by the current splitter.
  uint32_t* touched_edges;

  // Partition groups touched by the current edge row.
  uint32_t* touched_groups;

  // Temporary links between marked states in one group.
  uint32_t* marked_next;

  // Temporary membership bits for marked states.
  bool* marked;

  // Provisional equality assumptions.
  loom_cfg_value_identity_assumption_t* assumptions;

  // Retained candidate and external-source obligation endpoints.
  loom_cfg_value_identity_endpoint_t* endpoints;

  // Number of populated endpoints.
  uint32_t endpoint_count;

  // Allocated endpoint capacity.
  uint32_t endpoint_capacity;

  // Circular queue of assumptions requiring an obligation check.
  uint32_t* assumption_queue;

  // Monotonic queue read position.
  uint64_t queue_head;

  // Monotonic queue write position.
  uint64_t queue_tail;

  // True after endpoints and the assumption queue are initialized.
  bool endpoints_ready;
} loom_cfg_value_identity_refinement_t;

static uint64_t loom_cfg_value_identity_mix64(uint64_t value) {
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static uint32_t loom_cfg_value_identity_refinement_argument_ordinal(
    const loom_cfg_value_identity_refinement_t* refinement,
    uint32_t value_node) {
  return value_node < refinement->argument_count
             ? value_node
             : LOOM_CFG_VALUE_IDENTITY_INVALID;
}

static loom_value_id_t loom_cfg_value_identity_refinement_mapped_value(
    const loom_cfg_value_identity_refinement_t* refinement,
    loom_value_id_t value_id) {
  const uint32_t ordinal = loom_cfg_value_identity_projection_argument_ordinal(
      refinement->projection, value_id);
  return ordinal != LOOM_CFG_VALUE_IDENTITY_INVALID
             ? value_id
             : loom_cfg_value_identity_table_lookup(refinement->table,
                                                    value_id);
}

static uint32_t loom_cfg_value_identity_refinement_value_node(
    const loom_cfg_value_identity_refinement_t* refinement,
    const uint32_t* nodes_by_ordinal, loom_value_id_t value_id) {
  value_id =
      loom_cfg_value_identity_refinement_mapped_value(refinement, value_id);
  const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
      refinement->table->value_domain, value_id);
  const uint32_t node = nodes_by_ordinal[ordinal];
  IREE_ASSERT_NE(node, LOOM_CFG_VALUE_IDENTITY_INVALID);
  return node;
}

static void loom_cfg_value_identity_refinement_link_forest_edge(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t source,
    uint32_t target, uint32_t edge) {
  refinement->forest_edges[edge] = (loom_cfg_value_identity_forest_edge_t){
      .target = target,
      .previous = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .next = refinement->values[source].first_edge,
  };
  if (refinement->forest_edges[edge].next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->forest_edges[refinement->forest_edges[edge].next].previous =
        edge;
  }
  refinement->values[source].first_edge = edge;
}

static void loom_cfg_value_identity_refinement_unlink_forest_edge(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t source,
    uint32_t edge) {
  loom_cfg_value_identity_forest_edge_t* item = &refinement->forest_edges[edge];
  if (item->previous == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->values[source].first_edge = item->next;
  } else {
    refinement->forest_edges[item->previous].next = item->next;
  }
  if (item->next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->forest_edges[item->next].previous = item->previous;
  }
}

static void loom_cfg_value_identity_refinement_add_component_member(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t node,
    uint32_t component) {
  loom_cfg_value_identity_value_t* value = &refinement->values[node];
  loom_cfg_value_identity_component_t* destination =
      &refinement->components[component];
  value->component = component;
  value->previous_member = LOOM_CFG_VALUE_IDENTITY_INVALID;
  value->next_member = destination->first_member;
  if (value->next_member != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->values[value->next_member].previous_member = node;
  }
  destination->first_member = node;
  ++destination->member_count;
}

static void loom_cfg_value_identity_refinement_remove_component_member(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t node,
    uint32_t component) {
  loom_cfg_value_identity_value_t* value = &refinement->values[node];
  if (value->previous_member == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->components[component].first_member = value->next_member;
  } else {
    refinement->values[value->previous_member].next_member = value->next_member;
  }
  if (value->next_member != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->values[value->next_member].previous_member =
        value->previous_member;
  }
  --refinement->components[component].member_count;
}

static uint64_t loom_cfg_value_identity_refinement_label_hash(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t state) {
  const uint32_t root = refinement->components[state].root;
  const uint32_t argument =
      loom_cfg_value_identity_refinement_argument_ordinal(refinement, root);
  if (argument == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return loom_cfg_value_identity_mix64(UINT64_C(0xe17a1465) ^
                                         refinement->values[root].value_id);
  }
  const loom_value_fact_cfg_argument_t* descriptor =
      loom_cfg_value_identity_projection_argument(refinement->projection,
                                                  argument);
  const loom_module_t* module = refinement->projection->region->graph.module;
  const uint64_t type_hash = loom_type_hash(
      loom_module_value_type(module, refinement->values[root].value_id));
  return loom_cfg_value_identity_mix64(
      ((uint64_t)descriptor->block_index << 32) ^ type_hash ^
      UINT64_C(0x612b0da9));
}

static bool loom_cfg_value_identity_refinement_same_label(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t left,
    uint32_t right) {
  const uint32_t left_root = refinement->components[left].root;
  const uint32_t right_root = refinement->components[right].root;
  const uint32_t left_argument =
      loom_cfg_value_identity_refinement_argument_ordinal(refinement,
                                                          left_root);
  const uint32_t right_argument =
      loom_cfg_value_identity_refinement_argument_ordinal(refinement,
                                                          right_root);
  if (left_argument == LOOM_CFG_VALUE_IDENTITY_INVALID ||
      right_argument == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return left_argument == right_argument &&
           refinement->values[left_root].value_id ==
               refinement->values[right_root].value_id;
  }
  const loom_value_fact_cfg_argument_t* left_descriptor =
      loom_cfg_value_identity_projection_argument(refinement->projection,
                                                  left_argument);
  const loom_value_fact_cfg_argument_t* right_descriptor =
      loom_cfg_value_identity_projection_argument(refinement->projection,
                                                  right_argument);
  const loom_module_t* module = refinement->projection->region->graph.module;
  return left_descriptor->block_index == right_descriptor->block_index &&
         loom_type_equal(loom_module_value_type(
                             module, refinement->values[left_root].value_id),
                         loom_module_value_type(
                             module, refinement->values[right_root].value_id));
}

static uint64_t loom_cfg_value_identity_refinement_transition_hash(
    uint32_t edge, uint32_t group) {
  return loom_cfg_value_identity_mix64(((uint64_t)edge << 32) ^ group ^
                                       UINT64_C(0xd6e8feb8));
}

// Signature computation is the per-state refinement hot path.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline uint64_t
loom_cfg_value_identity_refinement_compute_signature(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t state) {
  uint64_t signature =
      loom_cfg_value_identity_refinement_label_hash(refinement, state);
  const uint32_t argument = loom_cfg_value_identity_refinement_argument_ordinal(
      refinement, refinement->components[state].root);
  if (argument == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return signature;
  }
  for (uint32_t i = refinement->occurrence_offsets[argument];
       i < refinement->occurrence_offsets[argument + 1]; ++i) {
    const loom_cfg_value_identity_occurrence_t* occurrence =
        &refinement->occurrences[i];
    const uint32_t source_state =
        refinement->values[occurrence->source_value].component;
    signature ^= loom_cfg_value_identity_refinement_transition_hash(
        occurrence->edge, refinement->states[source_state].group);
  }
  return signature;
}

static bool loom_cfg_value_identity_refinement_same_signature(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t left,
    uint32_t right) {
  if (refinement->states[left].signature !=
          refinement->states[right].signature ||
      !loom_cfg_value_identity_refinement_same_label(refinement, left, right)) {
    return false;
  }
  const uint32_t left_argument =
      loom_cfg_value_identity_refinement_argument_ordinal(
          refinement, refinement->components[left].root);
  const uint32_t right_argument =
      loom_cfg_value_identity_refinement_argument_ordinal(
          refinement, refinement->components[right].root);
  if (left_argument == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return true;
  }
  uint32_t left_position = refinement->occurrence_offsets[left_argument];
  const uint32_t left_end = refinement->occurrence_offsets[left_argument + 1];
  uint32_t right_position = refinement->occurrence_offsets[right_argument];
  const uint32_t right_end = refinement->occurrence_offsets[right_argument + 1];
  if (left_end - left_position != right_end - right_position) {
    return false;
  }
  for (; left_position < left_end; ++left_position, ++right_position) {
    const loom_cfg_value_identity_occurrence_t* left_occurrence =
        &refinement->occurrences[left_position];
    const loom_cfg_value_identity_occurrence_t* right_occurrence =
        &refinement->occurrences[right_position];
    if (left_occurrence->edge != right_occurrence->edge ||
        refinement
                ->states[refinement->values[left_occurrence->source_value]
                             .component]
                .group !=
            refinement
                ->states[refinement->values[right_occurrence->source_value]
                             .component]
                .group) {
      return false;
    }
  }
  return true;
}

static uint32_t loom_cfg_value_identity_refinement_create_group(
    loom_cfg_value_identity_refinement_t* refinement) {
  const uint32_t group = refinement->group_count++;
  refinement->groups[group] = (loom_cfg_value_identity_group_t){
      .first_state = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_marked = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  return group;
}

static void loom_cfg_value_identity_refinement_insert_state(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t state,
    uint32_t group) {
  refinement->states[state].group = group;
  refinement->states[state].previous = LOOM_CFG_VALUE_IDENTITY_INVALID;
  refinement->states[state].next = refinement->groups[group].first_state;
  if (refinement->states[state].next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->states[refinement->states[state].next].previous = state;
  }
  refinement->groups[group].first_state = state;
  ++refinement->groups[group].state_count;
}

static void loom_cfg_value_identity_refinement_queue_assumption(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t assumption) {
  loom_cfg_value_identity_assumption_t* item =
      &refinement->assumptions[assumption];
  if (!refinement->endpoints_ready || !item->active || item->queued) {
    return;
  }
  item->queued = true;
  refinement->assumption_queue[refinement->queue_tail++ %
                               refinement->assumption_count] = assumption;
}

static void loom_cfg_value_identity_refinement_queue_state_endpoints(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t state) {
  if (!refinement->endpoints_ready) {
    return;
  }
  for (uint32_t endpoint = refinement->states[state].first_endpoint;
       endpoint != LOOM_CFG_VALUE_IDENTITY_INVALID;
       endpoint = refinement->endpoints[endpoint].state_next) {
    loom_cfg_value_identity_refinement_queue_assumption(
        refinement, refinement->endpoints[endpoint].assumption);
  }
}

static void loom_cfg_value_identity_refinement_move_state(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t state,
    uint32_t destination) {
  loom_cfg_value_identity_state_t* node = &refinement->states[state];
  const uint32_t source = node->group;
  if (node->previous == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->groups[source].first_state = node->next;
  } else {
    refinement->states[node->previous].next = node->next;
  }
  if (node->next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->states[node->next].previous = node->previous;
  }
  --refinement->groups[source].state_count;
  loom_cfg_value_identity_refinement_insert_state(refinement, state,
                                                  destination);

  for (uint32_t occurrence = node->first_inverse;
       occurrence != LOOM_CFG_VALUE_IDENTITY_INVALID;
       occurrence = refinement->occurrences[occurrence].inverse_next) {
    const uint32_t target =
        refinement->values[refinement->occurrences[occurrence].argument]
            .component;
    refinement->states[target].signature ^=
        loom_cfg_value_identity_refinement_transition_hash(
            refinement->occurrences[occurrence].edge, source) ^
        loom_cfg_value_identity_refinement_transition_hash(
            refinement->occurrences[occurrence].edge, destination);
  }
  loom_cfg_value_identity_refinement_queue_state_endpoints(refinement, state);
}

static void loom_cfg_value_identity_refinement_link_inverse(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t occurrence,
    uint32_t state) {
  loom_cfg_value_identity_occurrence_t* item =
      &refinement->occurrences[occurrence];
  item->inverse_previous = LOOM_CFG_VALUE_IDENTITY_INVALID;
  item->inverse_next = refinement->states[state].first_inverse;
  if (item->inverse_next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->occurrences[item->inverse_next].inverse_previous = occurrence;
  }
  refinement->states[state].first_inverse = occurrence;
}

static void loom_cfg_value_identity_refinement_unlink_inverse(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t occurrence,
    uint32_t state) {
  loom_cfg_value_identity_occurrence_t* item =
      &refinement->occurrences[occurrence];
  if (item->inverse_previous == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->states[state].first_inverse = item->inverse_next;
  } else {
    refinement->occurrences[item->inverse_previous].inverse_next =
        item->inverse_next;
  }
  if (item->inverse_next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->occurrences[item->inverse_next].inverse_previous =
        item->inverse_previous;
  }
}

static void loom_cfg_value_identity_refinement_link_endpoint(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t endpoint,
    uint32_t state) {
  loom_cfg_value_identity_endpoint_t* item = &refinement->endpoints[endpoint];
  item->state_previous = LOOM_CFG_VALUE_IDENTITY_INVALID;
  item->state_next = refinement->states[state].first_endpoint;
  if (item->state_next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->endpoints[item->state_next].state_previous = endpoint;
  }
  refinement->states[state].first_endpoint = endpoint;
}

static void loom_cfg_value_identity_refinement_unlink_endpoint(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t endpoint,
    uint32_t state) {
  loom_cfg_value_identity_endpoint_t* item = &refinement->endpoints[endpoint];
  if (item->state_previous == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->states[state].first_endpoint = item->state_next;
  } else {
    refinement->endpoints[item->state_previous].state_next = item->state_next;
  }
  if (item->state_next != LOOM_CFG_VALUE_IDENTITY_INVALID) {
    refinement->endpoints[item->state_next].state_previous =
        item->state_previous;
  }
}

static void loom_cfg_value_identity_refinement_activate_argument(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t argument) {
  for (uint32_t i = refinement->occurrence_offsets[argument];
       i < refinement->occurrence_offsets[argument + 1]; ++i) {
    loom_cfg_value_identity_refinement_link_inverse(
        refinement, i,
        refinement->values[refinement->occurrences[i].source_value].component);
  }
}

static void loom_cfg_value_identity_refinement_split_marked_group(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t original) {
  loom_cfg_value_identity_group_t* group = &refinement->groups[original];
  if (group->marked_count == 0 || group->marked_count == group->state_count) {
    return;
  }
  const uint32_t created =
      loom_cfg_value_identity_refinement_create_group(refinement);
  if (group->marked_count <= group->state_count / 2) {
    for (uint32_t state = group->first_marked;
         state != LOOM_CFG_VALUE_IDENTITY_INVALID;
         state = refinement->marked_next[state]) {
      loom_cfg_value_identity_refinement_move_state(refinement, state, created);
    }
  } else {
    uint32_t state = group->first_state;
    while (state != LOOM_CFG_VALUE_IDENTITY_INVALID) {
      const uint32_t next = refinement->states[state].next;
      if (!refinement->marked[state]) {
        loom_cfg_value_identity_refinement_move_state(refinement, state,
                                                      created);
      }
      state = next;
    }
  }
}

static void loom_cfg_value_identity_refinement_process_splitter(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t splitter) {
  uint32_t edge_count = 0;
  for (uint32_t state = refinement->groups[splitter].first_state;
       state != LOOM_CFG_VALUE_IDENTITY_INVALID;
       state = refinement->states[state].next) {
    for (uint32_t occurrence = refinement->states[state].first_inverse;
         occurrence != LOOM_CFG_VALUE_IDENTITY_INVALID;
         occurrence = refinement->occurrences[occurrence].inverse_next) {
      const uint32_t edge = refinement->occurrences[occurrence].edge;
      if (refinement->row_heads[edge] == LOOM_CFG_VALUE_IDENTITY_INVALID) {
        refinement->touched_edges[edge_count++] = edge;
      }
      refinement->row_next[occurrence] = refinement->row_heads[edge];
      refinement->row_heads[edge] = occurrence;
    }
  }

  for (uint32_t edge_index = 0; edge_index < edge_count; ++edge_index) {
    const uint32_t edge = refinement->touched_edges[edge_index];
    uint32_t affected_count = 0;
    for (uint32_t occurrence = refinement->row_heads[edge];
         occurrence != LOOM_CFG_VALUE_IDENTITY_INVALID;
         occurrence = refinement->row_next[occurrence]) {
      const uint32_t target =
          refinement->values[refinement->occurrences[occurrence].argument]
              .component;
      const uint32_t group = refinement->states[target].group;
      if (refinement->groups[group].marked_count == 0) {
        refinement->touched_groups[affected_count++] = group;
      }
      refinement->marked[target] = true;
      refinement->marked_next[target] = refinement->groups[group].first_marked;
      refinement->groups[group].first_marked = target;
      ++refinement->groups[group].marked_count;
    }
    for (uint32_t i = 0; i < affected_count; ++i) {
      loom_cfg_value_identity_refinement_split_marked_group(
          refinement, refinement->touched_groups[i]);
    }
    for (uint32_t occurrence = refinement->row_heads[edge];
         occurrence != LOOM_CFG_VALUE_IDENTITY_INVALID;
         occurrence = refinement->row_next[occurrence]) {
      const uint32_t target =
          refinement->values[refinement->occurrences[occurrence].argument]
              .component;
      refinement->marked[target] = false;
    }
    for (uint32_t i = 0; i < affected_count; ++i) {
      loom_cfg_value_identity_group_t* group =
          &refinement->groups[refinement->touched_groups[i]];
      group->first_marked = LOOM_CFG_VALUE_IDENTITY_INVALID;
      group->marked_count = 0;
    }
    refinement->row_heads[edge] = LOOM_CFG_VALUE_IDENTITY_INVALID;
  }
}

static void loom_cfg_value_identity_refinement_stabilize(
    loom_cfg_value_identity_refinement_t* refinement) {
  while (refinement->processed_group_count < refinement->group_count) {
    loom_cfg_value_identity_refinement_process_splitter(
        refinement, refinement->processed_group_count++);
  }
}

static void loom_cfg_value_identity_refinement_split_changed_state(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t changed,
    uint32_t reference) {
  if (loom_cfg_value_identity_refinement_same_signature(refinement, changed,
                                                        reference)) {
    return;
  }
  IREE_ASSERT_EQ(refinement->states[changed].group,
                 refinement->states[reference].group);
  const uint32_t created =
      loom_cfg_value_identity_refinement_create_group(refinement);
  loom_cfg_value_identity_refinement_move_state(refinement, changed, created);
}

static bool loom_cfg_value_identity_refinement_search_step(
    loom_cfg_value_identity_refinement_t* refinement,
    loom_cfg_value_identity_search_t* search) {
  if (search->position == search->count) {
    return false;
  }
  if (search->edge == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    ++search->position;
    if (search->position < search->count) {
      search->edge =
          refinement->values[search->visits[search->position].node].first_edge;
    }
    return true;
  }
  const uint32_t target = refinement->forest_edges[search->edge].target;
  search->edge = refinement->forest_edges[search->edge].next;
  if (target != search->visits[search->position].parent) {
    search->visits[search->count++] = (loom_cfg_value_identity_visit_t){
        .node = target,
        .parent = search->visits[search->position].node,
    };
  }
  return true;
}

static void loom_cfg_value_identity_refinement_relabel_value(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t value,
    uint32_t source, uint32_t destination) {
  IREE_ASSERT_EQ(refinement->states[source].group,
                 refinement->states[destination].group);
  for (uint32_t occurrence = refinement->values[value].first_occurrence;
       occurrence != LOOM_CFG_VALUE_IDENTITY_INVALID;
       occurrence = refinement->occurrences[occurrence].value_next) {
    const uint32_t argument = refinement->occurrences[occurrence].argument;
    if (refinement->values[argument].parent !=
        LOOM_CFG_VALUE_IDENTITY_INVALID) {
      continue;
    }
    loom_cfg_value_identity_refinement_unlink_inverse(refinement, occurrence,
                                                      source);
    loom_cfg_value_identity_refinement_link_inverse(refinement, occurrence,
                                                    destination);
  }
  if (!refinement->endpoints_ready) {
    return;
  }
  for (uint32_t endpoint = refinement->values[value].first_endpoint;
       endpoint != LOOM_CFG_VALUE_IDENTITY_INVALID;
       endpoint = refinement->endpoints[endpoint].value_next) {
    loom_cfg_value_identity_refinement_unlink_endpoint(refinement, endpoint,
                                                       source);
    loom_cfg_value_identity_refinement_link_endpoint(refinement, endpoint,
                                                     destination);
    loom_cfg_value_identity_refinement_queue_assumption(
        refinement, refinement->endpoints[endpoint].assumption);
  }
}

// A cut runs only after a provisional equality has been disproven. Keep its
// mutation machinery outside the successful refinement path.
#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static void loom_cfg_value_identity_refinement_cut(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t child) {
  const uint32_t parent = refinement->values[child].parent;
  IREE_ASSERT_NE(parent, LOOM_CFG_VALUE_IDENTITY_INVALID);
  const uint32_t old_component = refinement->values[child].component;
  const uint32_t old_root = refinement->components[old_component].root;
  const uint64_t old_signature = refinement->states[old_component].signature;
  const uint32_t old_group = refinement->states[old_component].group;

  loom_cfg_value_identity_refinement_unlink_forest_edge(refinement, child,
                                                        child * 2);
  loom_cfg_value_identity_refinement_unlink_forest_edge(refinement, parent,
                                                        child * 2 + 1);
  loom_cfg_value_identity_search_t searches[2] = {
      {
          .visits = refinement->visits,
          .count = 1,
          .edge = refinement->values[child].first_edge,
      },
      {
          .visits = refinement->visits + refinement->value_count,
          .count = 1,
          .edge = refinement->values[parent].first_edge,
      },
  };
  searches[0].visits[0] = (loom_cfg_value_identity_visit_t){
      .node = child,
      .parent = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  searches[1].visits[0] = (loom_cfg_value_identity_visit_t){
      .node = parent,
      .parent = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  uint32_t smaller = 0;
  while (true) {
    if (!loom_cfg_value_identity_refinement_search_step(refinement,
                                                        &searches[0])) {
      smaller = 0;
      break;
    }
    if (!loom_cfg_value_identity_refinement_search_step(refinement,
                                                        &searches[1])) {
      smaller = 1;
      break;
    }
  }

  const uint32_t created = refinement->component_count++;
  refinement->components[created] = (loom_cfg_value_identity_component_t){
      .root = smaller != 0 ? old_root : child,
      .first_member = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  refinement->components[old_component].root = smaller != 0 ? child : old_root;
  refinement->states[created] = (loom_cfg_value_identity_state_t){
      .group = old_group,
      .previous = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .next = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_inverse = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_endpoint = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  loom_cfg_value_identity_refinement_insert_state(refinement, created,
                                                  old_group);
  for (uint32_t i = 0; i < searches[smaller].count; ++i) {
    const uint32_t member = searches[smaller].visits[i].node;
    loom_cfg_value_identity_refinement_remove_component_member(
        refinement, member, old_component);
    loom_cfg_value_identity_refinement_add_component_member(refinement, member,
                                                            created);
    loom_cfg_value_identity_refinement_relabel_value(refinement, member,
                                                     old_component, created);
  }

  const uint32_t old_root_component = refinement->values[old_root].component;
  const uint32_t child_component = refinement->values[child].component;
  refinement->states[old_root_component].signature = old_signature;
  // Relabel the old active equation before exposing and activating child.
  refinement->values[child].parent = LOOM_CFG_VALUE_IDENTITY_INVALID;
  loom_cfg_value_identity_refinement_activate_argument(refinement, child);
  refinement->states[child_component].signature =
      loom_cfg_value_identity_refinement_compute_signature(refinement,
                                                           child_component);
  loom_cfg_value_identity_refinement_split_changed_state(
      refinement, child_component, old_root_component);
  loom_cfg_value_identity_refinement_queue_state_endpoints(refinement,
                                                           old_root_component);
  loom_cfg_value_identity_refinement_queue_state_endpoints(refinement,
                                                           child_component);
  loom_cfg_value_identity_refinement_stabilize(refinement);
}

static iree_status_t loom_cfg_value_identity_refinement_allocate_storage(
    loom_cfg_value_identity_refinement_t* refinement) {
  if (refinement->value_count > (UINT32_C(1) << 30)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CFG identity partition exceeds compact index");
  }
  refinement->group_bucket_count = 1;
  while (refinement->group_bucket_count < refinement->value_count * 2u) {
    if (refinement->group_bucket_count > UINT32_MAX / 2u) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "CFG identity partition exceeds compact index");
    }
    refinement->group_bucket_count *= 2u;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count, sizeof(*refinement->values),
      (void**)&refinement->values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count,
      sizeof(*refinement->components), (void**)&refinement->components));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, (iree_host_size_t)refinement->argument_count * 2,
      sizeof(*refinement->forest_edges), (void**)&refinement->forest_edges));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, (iree_host_size_t)refinement->value_count * 2,
      sizeof(*refinement->visits), (void**)&refinement->visits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count, sizeof(*refinement->states),
      (void**)&refinement->states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count, sizeof(*refinement->groups),
      (void**)&refinement->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->group_bucket_count,
      sizeof(*refinement->group_buckets), (void**)&refinement->group_buckets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->projection->source_count,
      sizeof(*refinement->occurrences), (void**)&refinement->occurrences));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, (iree_host_size_t)refinement->argument_count + 1,
      sizeof(*refinement->occurrence_offsets),
      (void**)&refinement->occurrence_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->projection->region->graph.edge_count,
      sizeof(*refinement->row_heads), (void**)&refinement->row_heads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->projection->source_count,
      sizeof(*refinement->row_next), (void**)&refinement->row_next));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->projection->region->graph.edge_count,
      sizeof(*refinement->touched_edges), (void**)&refinement->touched_edges));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(refinement->arena, refinement->value_count,
                                sizeof(*refinement->touched_groups),
                                (void**)&refinement->touched_groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count,
      sizeof(*refinement->marked_next), (void**)&refinement->marked_next));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->value_count, sizeof(*refinement->marked),
      (void**)&refinement->marked));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      refinement->arena, refinement->assumption_count,
      sizeof(*refinement->assumptions), (void**)&refinement->assumptions));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(refinement->arena, refinement->assumption_count,
                                sizeof(*refinement->assumption_queue),
                                (void**)&refinement->assumption_queue));

  memset(refinement->values, 0xFF,
         refinement->value_count * sizeof(*refinement->values));
  memset(refinement->group_buckets, 0xFF,
         refinement->group_bucket_count * sizeof(*refinement->group_buckets));
  memset(refinement->row_heads, 0xFF,
         refinement->projection->region->graph.edge_count *
             sizeof(*refinement->row_heads));
  memset(refinement->marked, 0,
         refinement->value_count * sizeof(*refinement->marked));
  return iree_ok_status();
}

static uint32_t loom_cfg_value_identity_register_value_node(
    const loom_cfg_value_identity_refinement_t* refinement,
    uint32_t* nodes_by_ordinal, uint32_t* inout_value_count,
    loom_value_id_t value_id) {
  value_id =
      loom_cfg_value_identity_refinement_mapped_value(refinement, value_id);
  const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
      refinement->table->value_domain, value_id);
  uint32_t node = nodes_by_ordinal[ordinal];
  if (node == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    node = (*inout_value_count)++;
    nodes_by_ordinal[ordinal] = node;
  }
  return node;
}

static void loom_cfg_value_identity_refinement_initialize_value(
    loom_cfg_value_identity_refinement_t* refinement,
    const uint32_t* nodes_by_ordinal, loom_value_id_t value_id) {
  value_id =
      loom_cfg_value_identity_refinement_mapped_value(refinement, value_id);
  const uint32_t node = loom_cfg_value_identity_refinement_value_node(
      refinement, nodes_by_ordinal, value_id);
  if (refinement->values[node].value_id != LOOM_VALUE_ID_INVALID) {
    return;
  }
  refinement->values[node] = (loom_cfg_value_identity_value_t){
      .value_id = value_id,
      .parent = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .component = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_edge = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .previous_member = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .next_member = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_occurrence = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .first_endpoint = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
}

static iree_status_t loom_cfg_value_identity_refinement_initialize(
    loom_cfg_value_identity_refinement_t* refinement,
    const loom_value_id_t* targets, const uint32_t* nodes_by_ordinal) {
  IREE_RETURN_IF_ERROR(
      loom_cfg_value_identity_refinement_allocate_storage(refinement));
  for (uint32_t i = 0; i < refinement->argument_count; ++i) {
    const loom_value_id_t value_id =
        loom_cfg_value_identity_projection_argument(refinement->projection, i)
            ->value_id;
    loom_cfg_value_identity_refinement_initialize_value(
        refinement, nodes_by_ordinal, value_id);
    IREE_ASSERT_EQ(loom_cfg_value_identity_refinement_value_node(
                       refinement, nodes_by_ordinal, value_id),
                   i);
  }
  for (uint32_t i = 0; i < refinement->projection->source_count; ++i) {
    loom_cfg_value_identity_refinement_initialize_value(
        refinement, nodes_by_ordinal, refinement->projection->sources[i]);
  }

  for (uint32_t i = 0; i < refinement->argument_count; ++i) {
    if (targets[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t parent = loom_cfg_value_identity_refinement_value_node(
        refinement, nodes_by_ordinal, targets[i]);
    refinement->values[i].parent = parent;
    loom_cfg_value_identity_refinement_link_forest_edge(refinement, i, parent,
                                                        i * 2);
    loom_cfg_value_identity_refinement_link_forest_edge(refinement, parent, i,
                                                        i * 2 + 1);
  }
  for (uint32_t root = 0; root < refinement->value_count; ++root) {
    if (refinement->values[root].parent != LOOM_CFG_VALUE_IDENTITY_INVALID) {
      continue;
    }
    const uint32_t component = refinement->component_count++;
    refinement->components[component] = (loom_cfg_value_identity_component_t){
        .root = root,
        .first_member = LOOM_CFG_VALUE_IDENTITY_INVALID,
    };
    uint32_t visit_count = 1;
    refinement->visits[0] = (loom_cfg_value_identity_visit_t){
        .node = root,
        .parent = LOOM_CFG_VALUE_IDENTITY_INVALID,
    };
    for (uint32_t position = 0; position < visit_count; ++position) {
      const loom_cfg_value_identity_visit_t current =
          refinement->visits[position];
      IREE_ASSERT_EQ(refinement->values[current.node].component,
                     LOOM_CFG_VALUE_IDENTITY_INVALID);
      loom_cfg_value_identity_refinement_add_component_member(
          refinement, current.node, component);
      for (uint32_t edge = refinement->values[current.node].first_edge;
           edge != LOOM_CFG_VALUE_IDENTITY_INVALID;
           edge = refinement->forest_edges[edge].next) {
        if (refinement->forest_edges[edge].target != current.parent) {
          refinement->visits[visit_count++] = (loom_cfg_value_identity_visit_t){
              .node = refinement->forest_edges[edge].target,
              .parent = current.node,
          };
        }
      }
    }
  }
  for (uint32_t i = 0; i < refinement->value_count; ++i) {
    IREE_ASSERT_NE(refinement->values[i].component,
                   LOOM_CFG_VALUE_IDENTITY_INVALID);
  }

  const loom_cfg_graph_t* graph = &refinement->projection->region->graph;
  for (uint32_t argument = 0; argument < refinement->argument_count;
       ++argument) {
    refinement->occurrence_offsets[argument] = refinement->occurrence_count;
    const loom_value_fact_cfg_argument_t* descriptor =
        loom_cfg_value_identity_projection_argument(refinement->projection,
                                                    argument);
    const loom_cfg_edge_index_span_t incoming =
        loom_cfg_graph_predecessor_edges(graph, descriptor->block_index);
    for (iree_host_size_t i = 0; i < incoming.count; ++i) {
      const uint32_t edge = incoming.values[i];
      const uint32_t row = refinement->projection->edge_offsets[edge];
      if (row == LOOM_CFG_VALUE_IDENTITY_INVALID) {
        continue;
      }
      const uint32_t source = loom_cfg_value_identity_refinement_value_node(
          refinement, nodes_by_ordinal,
          refinement->projection->sources[row + descriptor->argument_index]);
      const uint32_t occurrence = refinement->occurrence_count++;
      refinement->occurrences[occurrence] =
          (loom_cfg_value_identity_occurrence_t){
              .source_value = source,
              .argument = argument,
              .edge = edge,
              .value_next = refinement->values[source].first_occurrence,
              .inverse_previous = LOOM_CFG_VALUE_IDENTITY_INVALID,
              .inverse_next = LOOM_CFG_VALUE_IDENTITY_INVALID,
          };
      refinement->values[source].first_occurrence = occurrence;
    }
  }
  refinement->occurrence_offsets[refinement->argument_count] =
      refinement->occurrence_count;
  IREE_ASSERT_LE(refinement->occurrence_count,
                 refinement->projection->source_count);

  for (uint32_t state = 0; state < refinement->component_count; ++state) {
    refinement->states[state] = (loom_cfg_value_identity_state_t){
        .group = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .previous = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .next = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .first_inverse = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .first_endpoint = LOOM_CFG_VALUE_IDENTITY_INVALID,
    };
    uint32_t bucket = (uint32_t)loom_cfg_value_identity_refinement_label_hash(
                          refinement, state) &
                      (refinement->group_bucket_count - 1);
    while (refinement->group_buckets[bucket] !=
               LOOM_CFG_VALUE_IDENTITY_INVALID &&
           !loom_cfg_value_identity_refinement_same_label(
               refinement, state, refinement->group_buckets[bucket])) {
      bucket = (bucket + 1) & (refinement->group_bucket_count - 1);
    }
    uint32_t group = LOOM_CFG_VALUE_IDENTITY_INVALID;
    if (refinement->group_buckets[bucket] == LOOM_CFG_VALUE_IDENTITY_INVALID) {
      group = loom_cfg_value_identity_refinement_create_group(refinement);
      refinement->group_buckets[bucket] = state;
    } else {
      group = refinement->states[refinement->group_buckets[bucket]].group;
    }
    loom_cfg_value_identity_refinement_insert_state(refinement, state, group);
  }
  for (uint32_t state = 0; state < refinement->component_count; ++state) {
    const uint32_t argument =
        loom_cfg_value_identity_refinement_argument_ordinal(
            refinement, refinement->components[state].root);
    if (argument != LOOM_CFG_VALUE_IDENTITY_INVALID) {
      loom_cfg_value_identity_refinement_activate_argument(refinement,
                                                           argument);
    }
  }
  for (uint32_t state = 0; state < refinement->component_count; ++state) {
    refinement->states[state].signature =
        loom_cfg_value_identity_refinement_compute_signature(refinement, state);
  }
  loom_cfg_value_identity_refinement_stabilize(refinement);
  return iree_ok_status();
}

static uint32_t loom_cfg_value_identity_refinement_add_endpoint(
    loom_cfg_value_identity_refinement_t* refinement, uint32_t assumption,
    uint32_t value, bool candidate) {
  IREE_ASSERT_LT(refinement->endpoint_count, refinement->endpoint_capacity);
  const uint32_t endpoint = refinement->endpoint_count++;
  refinement->endpoints[endpoint] = (loom_cfg_value_identity_endpoint_t){
      .assumption = assumption,
      .value = value,
      .value_next = refinement->values[value].first_endpoint,
      .state_previous = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .state_next = LOOM_CFG_VALUE_IDENTITY_INVALID,
      .assumption_next = LOOM_CFG_VALUE_IDENTITY_INVALID,
  };
  refinement->values[value].first_endpoint = endpoint;
  loom_cfg_value_identity_refinement_link_endpoint(
      refinement, endpoint, refinement->values[value].component);
  if (!candidate) {
    refinement->endpoints[endpoint].assumption_next =
        refinement->assumptions[assumption].first_endpoint;
    refinement->assumptions[assumption].first_endpoint = endpoint;
  }
  return endpoint;
}

static iree_status_t loom_cfg_value_identity_refinement_build_obligations(
    loom_cfg_value_identity_refinement_t* refinement,
    const uint32_t* nodes_by_ordinal,
    const loom_value_id_t* assumption_sources) {
  const loom_cfg_graph_t* graph = &refinement->projection->region->graph;
  uint64_t endpoint_capacity = 0;
  for (uint32_t i = 0; i < refinement->assumption_count; ++i) {
    refinement->assumptions[i] = (loom_cfg_value_identity_assumption_t){
        .candidate_value = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .first_endpoint = LOOM_CFG_VALUE_IDENTITY_INVALID,
        .active = assumption_sources[i] != LOOM_VALUE_ID_INVALID,
    };
    if (!refinement->assumptions[i].active) {
      continue;
    }
    ++endpoint_capacity;
    iree_host_size_t singleton = 0;
    const iree_host_size_t* members = NULL;
    iree_host_size_t member_count = 0;
    loom_cfg_value_identity_projection_assumption_span(
        refinement->projection, refinement->assumption_components, i,
        &singleton, &members, &member_count);
    for (iree_host_size_t member_index = 0; member_index < member_count;
         ++member_index) {
      const uint32_t member = (uint32_t)members[member_index];
      const loom_value_fact_cfg_argument_t* argument =
          loom_cfg_value_identity_projection_argument(refinement->projection,
                                                      member);
      const loom_cfg_edge_index_span_t incoming =
          loom_cfg_graph_predecessor_edges(graph, argument->block_index);
      for (iree_host_size_t edge_index = 0; edge_index < incoming.count;
           ++edge_index) {
        const uint32_t row =
            refinement->projection->edge_offsets[incoming.values[edge_index]];
        if (row == LOOM_CFG_VALUE_IDENTITY_INVALID) {
          continue;
        }
        const loom_value_id_t source =
            refinement->projection->sources[row + argument->argument_index];
        if (!loom_cfg_value_identity_projection_source_is_internal(
                refinement->projection, source, member)) {
          ++endpoint_capacity;
        }
      }
    }
  }
  if (endpoint_capacity >= LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CFG identity obligations exceed compact index");
  }
  refinement->endpoint_capacity = (uint32_t)endpoint_capacity;
  if (refinement->endpoint_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        refinement->arena, refinement->endpoint_capacity,
        sizeof(*refinement->endpoints), (void**)&refinement->endpoints));
  }

  for (uint32_t i = 0; i < refinement->assumption_count; ++i) {
    if (!refinement->assumptions[i].active) {
      continue;
    }
    const uint32_t candidate = loom_cfg_value_identity_refinement_value_node(
        refinement, nodes_by_ordinal, assumption_sources[i]);
    refinement->assumptions[i].candidate_value = candidate;
    (void)loom_cfg_value_identity_refinement_add_endpoint(refinement, i,
                                                          candidate, true);
    iree_host_size_t singleton = 0;
    const iree_host_size_t* members = NULL;
    iree_host_size_t member_count = 0;
    loom_cfg_value_identity_projection_assumption_span(
        refinement->projection, refinement->assumption_components, i,
        &singleton, &members, &member_count);
    for (iree_host_size_t member_index = 0; member_index < member_count;
         ++member_index) {
      const uint32_t member = (uint32_t)members[member_index];
      const loom_value_fact_cfg_argument_t* argument =
          loom_cfg_value_identity_projection_argument(refinement->projection,
                                                      member);
      const loom_cfg_edge_index_span_t incoming =
          loom_cfg_graph_predecessor_edges(graph, argument->block_index);
      for (iree_host_size_t edge_index = 0; edge_index < incoming.count;
           ++edge_index) {
        const uint32_t row =
            refinement->projection->edge_offsets[incoming.values[edge_index]];
        if (row == LOOM_CFG_VALUE_IDENTITY_INVALID) {
          continue;
        }
        const loom_value_id_t source =
            refinement->projection->sources[row + argument->argument_index];
        if (loom_cfg_value_identity_projection_source_is_internal(
                refinement->projection, source, member)) {
          continue;
        }
        const uint32_t source_node =
            loom_cfg_value_identity_refinement_value_node(
                refinement, nodes_by_ordinal, source);
        (void)loom_cfg_value_identity_refinement_add_endpoint(
            refinement, i, source_node, false);
      }
    }
  }
  IREE_ASSERT_EQ(refinement->endpoint_count, refinement->endpoint_capacity);
  refinement->endpoints_ready = true;
  for (uint32_t i = 0; i < refinement->assumption_count; ++i) {
    loom_cfg_value_identity_refinement_queue_assumption(refinement, i);
  }
  return iree_ok_status();
}

static loom_value_id_t loom_cfg_value_identity_refinement_group_representative(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t group) {
  const uint32_t state = refinement->groups[group].first_state;
  return refinement->values[refinement->components[state].root].value_id;
}

static bool loom_cfg_value_identity_refinement_assumption_valid(
    const loom_cfg_value_identity_refinement_t* refinement, uint32_t index) {
  const loom_cfg_value_identity_assumption_t* assumption =
      &refinement->assumptions[index];
  const uint32_t candidate_state =
      refinement->values[assumption->candidate_value].component;
  const uint32_t candidate_group = refinement->states[candidate_state].group;
  const loom_value_id_t candidate =
      loom_cfg_value_identity_refinement_group_representative(refinement,
                                                              candidate_group);
  iree_host_size_t singleton = 0;
  const iree_host_size_t* members = NULL;
  iree_host_size_t member_count = 0;
  loom_cfg_value_identity_projection_assumption_span(
      refinement->projection, refinement->assumption_components, index,
      &singleton, &members, &member_count);
  if (!loom_cfg_value_identity_projection_value_available(
          refinement->projection, refinement->dominance, candidate, members,
          member_count)) {
    return false;
  }
  for (uint32_t endpoint = assumption->first_endpoint;
       endpoint != LOOM_CFG_VALUE_IDENTITY_INVALID;
       endpoint = refinement->endpoints[endpoint].assumption_next) {
    const uint32_t source_state =
        refinement->values[refinement->endpoints[endpoint].value].component;
    if (refinement->states[source_state].group != candidate_group) {
      return false;
    }
  }
  return true;
}

static void loom_cfg_value_identity_refinement_resolve_obligations(
    loom_cfg_value_identity_refinement_t* refinement) {
  while (refinement->queue_head != refinement->queue_tail) {
    const uint32_t index =
        refinement->assumption_queue[refinement->queue_head++ %
                                     refinement->assumption_count];
    refinement->assumptions[index].queued = false;
    if (!refinement->assumptions[index].active ||
        loom_cfg_value_identity_refinement_assumption_valid(refinement,
                                                            index)) {
      continue;
    }
    refinement->assumptions[index].active = false;
    iree_host_size_t singleton = 0;
    const iree_host_size_t* members = NULL;
    iree_host_size_t member_count = 0;
    loom_cfg_value_identity_projection_assumption_span(
        refinement->projection, refinement->assumption_components, index,
        &singleton, &members, &member_count);
    for (iree_host_size_t i = 0; i < member_count; ++i) {
      loom_cfg_value_identity_refinement_cut(refinement, (uint32_t)members[i]);
    }
  }
}

static void loom_cfg_value_identity_refinement_publish(
    const loom_cfg_value_identity_refinement_t* refinement,
    loom_value_id_t* roots) {
  for (uint32_t i = 0; i < refinement->argument_count; ++i) {
    const uint32_t state = refinement->values[i].component;
    roots[i] = loom_cfg_value_identity_refinement_group_representative(
        refinement, refinement->states[state].group);
  }
}

typedef struct loom_cfg_value_identity_selection_t {
  // Branch payload projection being selected.
  const loom_cfg_value_identity_projection_t* projection;

  // Representatives previously published for enclosing CFG regions.
  const loom_cfg_value_identity_table_t* table;

  // Provisional direct roots indexed by local CFG argument.
  loom_value_id_t* roots;
} loom_cfg_value_identity_selection_t;

static loom_value_id_t loom_cfg_value_identity_selection_canonical(
    const loom_cfg_value_identity_selection_t* selection,
    loom_value_id_t value_id) {
  const uint32_t ordinal = loom_cfg_value_identity_projection_argument_ordinal(
      selection->projection, value_id);
  return ordinal != LOOM_CFG_VALUE_IDENTITY_INVALID
             ? selection->roots[ordinal]
             : loom_cfg_value_identity_table_lookup(selection->table, value_id);
}

static bool loom_cfg_value_identity_selection_coarse_equal(
    const loom_cfg_value_identity_selection_t* selection, loom_value_id_t left,
    loom_value_id_t right) {
  left = loom_cfg_value_identity_selection_canonical(selection, left);
  right = loom_cfg_value_identity_selection_canonical(selection, right);
  if (left == right) {
    return true;
  }
  const uint32_t left_ordinal =
      loom_cfg_value_identity_projection_argument_ordinal(selection->projection,
                                                          left);
  const uint32_t right_ordinal =
      loom_cfg_value_identity_projection_argument_ordinal(selection->projection,
                                                          right);
  if (left_ordinal == LOOM_CFG_VALUE_IDENTITY_INVALID ||
      right_ordinal == LOOM_CFG_VALUE_IDENTITY_INVALID) {
    return false;
  }
  const loom_value_fact_cfg_argument_t* left_argument =
      loom_cfg_value_identity_projection_argument(selection->projection,
                                                  left_ordinal);
  const loom_value_fact_cfg_argument_t* right_argument =
      loom_cfg_value_identity_projection_argument(selection->projection,
                                                  right_ordinal);
  const loom_module_t* module = selection->projection->region->graph.module;
  return left_argument->block_index == right_argument->block_index &&
         loom_type_equal(loom_module_value_type(module, left),
                         loom_module_value_type(module, right));
}

static iree_status_t loom_cfg_value_identity_collect_assumptions(
    const loom_cfg_value_identity_projection_t* projection,
    iree_arena_allocator_t* arena, uint32_t** out_components,
    uint32_t* out_count) {
  uint32_t* components = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, projection->argument_count,
                                sizeof(*components), (void**)&components));
  if (projection->region->argument_count == 0) {
    for (uint32_t i = 0; i < projection->argument_count; ++i) {
      components[i] = i;
    }
    *out_components = components;
    *out_count = projection->argument_count;
    return iree_ok_status();
  }

  bool* seen = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, projection->argument_count, sizeof(*seen), (void**)&seen));
  memset(seen, 0, projection->argument_count * sizeof(*seen));
  for (uint32_t i = 0; i < projection->argument_count; ++i) {
    const iree_host_size_t component =
        projection->region->argument_components[i];
    IREE_ASSERT_LT(component, projection->argument_count);
    seen[component] = true;
  }
  uint32_t count = 0;
  for (uint32_t component = 0; component < projection->argument_count;
       ++component) {
    if (seen[component]) {
      components[count++] = component;
    }
  }
  *out_components = components;
  *out_count = count;
  return iree_ok_status();
}

static void loom_cfg_value_identity_select_initial_assumptions(
    const loom_cfg_value_identity_projection_t* projection,
    const loom_cfg_value_identity_table_t* table,
    const loom_dominance_info_t* dominance,
    const uint32_t* assumption_components, uint32_t assumption_count,
    loom_value_id_t* assumption_sources, loom_value_id_t* targets,
    loom_value_id_t* roots) {
  memset(assumption_sources, 0xFF,
         assumption_count * sizeof(*assumption_sources));
  memset(targets, 0xFF, projection->argument_count * sizeof(*targets));
  for (uint32_t i = 0; i < projection->argument_count; ++i) {
    roots[i] =
        loom_cfg_value_identity_projection_argument(projection, i)->value_id;
  }

  const loom_cfg_value_identity_selection_t selection = {
      .projection = projection,
      .table = table,
      .roots = roots,
  };
  const loom_cfg_graph_t* graph = &projection->region->graph;
  for (uint32_t assumption = 0; assumption < assumption_count; ++assumption) {
    iree_host_size_t singleton = 0;
    const iree_host_size_t* members = NULL;
    iree_host_size_t member_count = 0;
    loom_cfg_value_identity_projection_assumption_span(
        projection, assumption_components, assumption, &singleton, &members,
        &member_count);
    bool plausible = true;
    for (iree_host_size_t i = 0; i < member_count; ++i) {
      const uint32_t member = (uint32_t)members[i];
      const loom_value_fact_cfg_argument_t* argument =
          loom_cfg_value_identity_projection_argument(projection, member);
      const loom_cfg_edge_index_span_t incoming =
          loom_cfg_graph_predecessor_edges(graph, argument->block_index);
      for (iree_host_size_t edge_index = 0; edge_index < incoming.count;
           ++edge_index) {
        const uint32_t row =
            projection->edge_offsets[incoming.values[edge_index]];
        if (row == LOOM_CFG_VALUE_IDENTITY_INVALID) {
          continue;
        }
        const loom_value_id_t source =
            projection->sources[row + argument->argument_index];
        if (loom_cfg_value_identity_projection_source_is_internal(
                projection, source, member)) {
          continue;
        }
        if (assumption_sources[assumption] == LOOM_VALUE_ID_INVALID) {
          assumption_sources[assumption] = source;
        } else if (!loom_cfg_value_identity_selection_coarse_equal(
                       &selection, source, assumption_sources[assumption])) {
          plausible = false;
        }
      }
    }
    if (assumption_sources[assumption] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_id_t root = loom_cfg_value_identity_selection_canonical(
        &selection, assumption_sources[assumption]);
    if (!plausible || !loom_cfg_value_identity_projection_value_available(
                          projection, dominance, root, members, member_count)) {
      assumption_sources[assumption] = LOOM_VALUE_ID_INVALID;
      continue;
    }
    for (iree_host_size_t i = 0; i < member_count; ++i) {
      const uint32_t member = (uint32_t)members[i];
      targets[member] = assumption_sources[assumption];
      roots[member] = root;
    }
  }
}

static iree_status_t loom_cfg_value_identity_refine_projection(
    loom_cfg_value_identity_table_t* table,
    const loom_cfg_value_identity_projection_t* projection,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena) {
  uint32_t* assumption_components = NULL;
  uint32_t assumption_count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_collect_assumptions(
      projection, arena, &assumption_components, &assumption_count));

  loom_value_id_t* assumption_sources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, assumption_count,
                                                 sizeof(*assumption_sources),
                                                 (void**)&assumption_sources));
  loom_value_id_t* targets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, projection->argument_count, sizeof(*targets), (void**)&targets));
  loom_value_id_t* roots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, projection->argument_count, sizeof(*roots), (void**)&roots));
  loom_cfg_value_identity_select_initial_assumptions(
      projection, table, dominance, assumption_components, assumption_count,
      assumption_sources, targets, roots);

  uint32_t* nodes_by_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->representative_count, sizeof(*nodes_by_ordinal),
      (void**)&nodes_by_ordinal));
  memset(nodes_by_ordinal, 0xFF,
         table->representative_count * sizeof(*nodes_by_ordinal));
  loom_cfg_value_identity_refinement_t refinement = {
      .projection = projection,
      .table = table,
      .dominance = dominance,
      .arena = arena,
      .assumption_components = assumption_components,
      .assumption_count = assumption_count,
      .argument_count = projection->argument_count,
  };
  for (uint32_t i = 0; i < projection->argument_count; ++i) {
    const uint32_t node = loom_cfg_value_identity_register_value_node(
        &refinement, nodes_by_ordinal, &refinement.value_count,
        loom_cfg_value_identity_projection_argument(projection, i)->value_id);
    IREE_ASSERT_EQ(node, i);
  }
  for (uint32_t i = 0; i < projection->source_count; ++i) {
    (void)loom_cfg_value_identity_register_value_node(
        &refinement, nodes_by_ordinal, &refinement.value_count,
        projection->sources[i]);
  }

  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_refinement_initialize(
      &refinement, targets, nodes_by_ordinal));
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_refinement_build_obligations(
      &refinement, nodes_by_ordinal, assumption_sources));
  loom_cfg_value_identity_refinement_resolve_obligations(&refinement);
  loom_cfg_value_identity_refinement_publish(&refinement, roots);

  for (uint32_t i = 0; i < projection->argument_count; ++i) {
    const loom_value_id_t argument =
        loom_cfg_value_identity_projection_argument(projection, i)->value_id;
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_ordinal(table->value_domain, argument);
    table->representatives[ordinal] = roots[i];
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_value_identity_table_initialize(
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_cfg_value_identity_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  *out_table = (loom_cfg_value_identity_table_t){
      .value_domain = value_domain,
      .representative_count = value_domain->value_count,
  };
  if (value_domain->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_domain->value_count, sizeof(*out_table->representatives),
      (void**)&out_table->representatives));
  memset(out_table->representatives, 0xFF,
         value_domain->value_count * sizeof(*out_table->representatives));
  return iree_ok_status();
}

iree_status_t loom_cfg_value_identity_table_update(
    loom_cfg_value_identity_table_t* table,
    const loom_value_fact_cfg_region_t* region,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(dominance);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_EQ(table->value_domain->module, region->graph.module);
  IREE_ASSERT_EQ(table->representative_count, table->value_domain->value_count);

  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  loom_cfg_value_identity_projection_t projection = {0};
  bool supported = false;
  iree_status_t status = loom_cfg_value_identity_projection_initialize(
      region, arena, &projection, &supported);
  if (iree_status_is_ok(status) && supported &&
      projection.argument_count != 0) {
    status = loom_cfg_value_identity_refine_projection(table, &projection,
                                                       dominance, arena);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
