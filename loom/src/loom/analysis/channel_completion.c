// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/channel_completion.h"

#include <string.h>

#include "loom/analysis/view_regions.h"
#include "loom/ir/types.h"
#include "loom/ops/channel/ops.h"
#include "loom/util/cfg_dataflow.h"

typedef struct loom_channel_completion_endpoint_t {
  // Protocol identity, independent of payload addresses.
  const loom_channel_identity_t* channel;
  // True for the producer's publication frontier; false for reclamation.
  bool producer;
  // Upper bound on admissions over reachable execution paths.
  uint32_t maximum_admissions;
} loom_channel_completion_endpoint_t;

typedef struct loom_channel_completion_record_t {
  // Endpoint ordinal; records are grouped by endpoint and FIFO admission order.
  uint32_t endpoint;
  // Outstanding owned value, or INVALID for a completed hole in the prefix.
  loom_value_id_t value;
} loom_channel_completion_record_t;

typedef struct loom_channel_completion_state_t {
  // Ordered live records and completed holes, excluding retired prefixes.
  loom_channel_completion_record_t* records;
  // Number of records in the snapshot.
  iree_host_size_t count;
} loom_channel_completion_state_t;

typedef struct loom_channel_completion_block_t {
  // Retained action interval in the canonical source plan.
  struct {
    // First action ordinal in the block.
    iree_host_size_t begin;
    // Exclusive end action ordinal.
    iree_host_size_t end;
  } actions;
  // First incoming state; later incoming edges must agree after renaming.
  loom_channel_completion_state_t input;
  // Whether an entry-reachable edge has established this input.
  bool reached;
} loom_channel_completion_block_t;

static void loom_channel_completion_require(
    loom_channel_completion_t* completion,
    loom_channel_completion_requirement_t requirement, const loom_op_t* op) {
  completion->requirement = requirement;
  completion->op = op;
}

static const loom_channel_payload_record_t
    loom_channel_payload_record_unavailable = {
        .value_id = LOOM_VALUE_ID_INVALID,
        .maximum_admissions = UINT32_MAX,
};

// Joins one path's payload/record correlation into a finite-height lattice:
// unknown becomes one exact pair, and absence or disagreement becomes
// permanently unavailable.
static bool loom_channel_completion_observe_payload_record(
    const loom_local_value_domain_t* value_domain,
    loom_channel_payload_record_t* payload_records, loom_value_id_t payload,
    loom_channel_payload_record_t record) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(value_domain, payload);
  loom_channel_payload_record_t* current = &payload_records[ordinal];
  const bool unobserved = current->value_id == LOOM_VALUE_ID_INVALID &&
                          current->maximum_admissions == 0;
  if (unobserved) {
    *current = record;
    return true;
  } else if (current->value_id != record.value_id ||
             current->maximum_admissions != record.maximum_admissions) {
    if (current->value_id != LOOM_VALUE_ID_INVALID) {
      *current = loom_channel_payload_record_unavailable;
      return true;
    }
  }
  return false;
}

bool loom_channel_completion_try_lookup_payload_record(
    const loom_channel_completion_t* completion, loom_value_id_t payload,
    loom_channel_payload_record_t* out_record) {
  if (!completion->value_domain || !completion->payload_records) {
    return false;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(completion->value_domain, payload);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      ordinal >= completion->payload_record_count ||
      completion->payload_records[ordinal].value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_record = completion->payload_records[ordinal];
  return true;
}

typedef struct loom_channel_payload_dataflow_t {
  // Source channel plan and its local value domain.
  const loom_channel_plan_t* plan;
  // CFG whose edge arguments project correlated payload/record pairs.
  const loom_cfg_graph_t* graph;
  // Canonical view bases used to inherit associations through projections.
  const loom_view_region_table_t* view_regions;
  // Mutable value-indexed facts joined by the solver.
  loom_channel_payload_record_t* payload_records;
} loom_channel_payload_dataflow_t;

// Returns the fact for a payload's canonical base. NULL means the base has not
// received a fact yet; an invalid returned record is known to be unavailable.
static const loom_channel_payload_record_t*
loom_channel_completion_payload_record_fact(
    const loom_channel_payload_dataflow_t* dataflow, loom_value_id_t payload) {
  const loom_view_region_t* region = NULL;
  const bool has_region = loom_view_region_table_try_lookup(
      dataflow->view_regions, payload, &region);
  IREE_ASSERT(has_region && region);
  const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
      dataflow->plan->value_domain, region->base_view_value_id);
  // View analysis can extend the shared domain with captured backing roots
  // after the channel plan snapshots its prefix. Such roots cannot be payload
  // producers in this plan.
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      ordinal >= dataflow->plan->value_count) {
    return &loom_channel_payload_record_unavailable;
  }
  const loom_channel_payload_record_t* record =
      &dataflow->payload_records[ordinal];
  if (record->value_id != LOOM_VALUE_ID_INVALID ||
      record->maximum_admissions != 0) {
    return record;
  }
  const loom_value_t* base = loom_module_value(
      dataflow->plan->value_domain->module, region->base_view_value_id);
  return loom_value_is_block_arg(base) &&
                 loom_value_def_block(base) !=
                     loom_region_const_entry_block(dataflow->graph->region)
             ? NULL
             : &loom_channel_payload_record_unavailable;
}

static iree_status_t loom_channel_completion_transfer_payload_records(
    void* user_data, uint16_t source_block_index, uint16_t target_block_index,
    bool* out_changed) {
  loom_channel_payload_dataflow_t* dataflow =
      (loom_channel_payload_dataflow_t*)user_data;
  *out_changed = false;
  const loom_block_t* destination =
      dataflow->graph->blocks[target_block_index].block;
  if (!destination->arg_count) {
    return iree_ok_status();
  }
  const loom_cfg_edge_index_span_t successors =
      loom_cfg_graph_successor_edges(dataflow->graph, source_block_index);
  for (iree_host_size_t i = 0; i < successors.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &dataflow->graph->edges[successors.values[i]];
    if (edge->target_block_index != target_block_index) {
      continue;
    }
    const loom_value_id_t* arguments = NULL;
    uint16_t argument_count = 0;
    const bool has_payload = loom_cfg_terminator_payload_for_successor(
        edge->terminator, destination, &arguments, &argument_count);
    IREE_ASSERT(has_payload);
    for (uint16_t a = 0; a < argument_count; ++a) {
      const loom_value_id_t destination_value = destination->arg_ids[a];
      if (!loom_type_is_view(loom_module_value_type(
              dataflow->plan->value_domain->module, destination_value))) {
        continue;
      }
      const loom_channel_payload_record_t* source_record =
          loom_channel_completion_payload_record_fact(dataflow, arguments[a]);
      if (!source_record) {
        continue;
      }
      loom_channel_payload_record_t destination_record = *source_record;
      if (destination_record.value_id != LOOM_VALUE_ID_INVALID) {
        for (uint16_t r = 0; r < argument_count; ++r) {
          if (arguments[r] == destination_record.value_id) {
            destination_record.value_id = destination->arg_ids[r];
            break;
          }
        }
      }
      *out_changed |= loom_channel_completion_observe_payload_record(
          dataflow->plan->value_domain, dataflow->payload_records,
          destination_value, destination_record);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_channel_completion_analyze_payload_records(
    const loom_channel_plan_t* plan, const loom_cfg_graph_t* graph,
    const loom_view_region_table_t* view_regions,
    const uint32_t* action_endpoints,
    const loom_channel_completion_endpoint_t* endpoints,
    loom_channel_payload_record_t* payload_records,
    iree_arena_allocator_t* arena) {
  // Each direct producer establishes the initial payload/owned-record pair.
  for (iree_host_size_t i = 0; i < plan->action_count; ++i) {
    const loom_op_t* op = plan->actions[i].op;
    loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
    loom_value_id_t record = LOOM_VALUE_ID_INVALID;
    if (loom_channel_reserve_isa(op)) {
      payload = loom_channel_reserve_view(op);
      record = loom_channel_reserve_write(op);
    } else if (loom_channel_acquire_isa(op)) {
      payload = loom_channel_acquire_view(op);
      record = loom_channel_acquire_read(op);
    } else if (loom_channel_wait_isa(op)) {
      payload = loom_channel_wait_view(op);
      record = loom_channel_wait_read(op);
    }
    if (payload != LOOM_VALUE_ID_INVALID) {
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_ordinal(plan->value_domain, payload);
      payload_records[ordinal] = (loom_channel_payload_record_t){
          .value_id = record,
          .maximum_admissions =
              endpoints[action_endpoints[i]].maximum_admissions,
      };
    }
  }

  loom_channel_payload_dataflow_t dataflow = {
      .plan = plan,
      .graph = graph,
      .view_regions = view_regions,
      .payload_records = payload_records,
  };
  IREE_RETURN_IF_ERROR(loom_cfg_dataflow_solve(
      graph, loom_channel_completion_transfer_payload_records, &dataflow,
      arena));

  // Publish a direct fact for every derived view so consumers need only one
  // value-domain lookup and never repeat canonical-base resolution.
  for (loom_value_ordinal_t i = 0; i < plan->value_count; ++i) {
    const loom_value_id_t value = plan->value_domain->value_ids[i];
    if (!loom_type_is_view(
            loom_module_value_type(plan->value_domain->module, value))) {
      continue;
    }
    const loom_view_region_t* region = NULL;
    const bool has_region =
        loom_view_region_table_try_lookup(view_regions, value, &region);
    IREE_ASSERT(has_region && region);
    if (region->base_view_value_id == value) {
      continue;
    }
    const loom_value_ordinal_t base_ordinal =
        loom_local_value_domain_try_ordinal(plan->value_domain,
                                            region->base_view_value_id);
    if (base_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        base_ordinal >= plan->value_count) {
      continue;
    }
    payload_records[i] = payload_records[base_ordinal];
  }
  return iree_ok_status();
}

static bool loom_channel_completion_apply(
    const loom_channel_plan_action_t* action, uint32_t endpoint,
    const loom_channel_completion_endpoint_t* endpoints,
    loom_channel_completion_state_t* state, uint32_t* credits,
    loom_channel_completion_t* completion) {
  const loom_op_t* op = action->op;
  if (loom_channel_reserve_isa(op) || loom_channel_acquire_isa(op)) {
    for (iree_host_size_t i = 0; i < state->count; ++i) {
      if (state->records[i].value == LOOM_VALUE_ID_INVALID &&
          !endpoints[state->records[i].endpoint].producer) {
        loom_channel_completion_require(
            completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_REUSE, op);
        return false;
      }
    }
    iree_host_size_t position = 0;
    while (position < state->count &&
           state->records[position].endpoint <= endpoint) {
      ++position;
    }
    memmove(&state->records[position + 1], &state->records[position],
            (state->count - position) * sizeof(*state->records));
    state->records[position] = (loom_channel_completion_record_t){
        .endpoint = endpoint, .value = loom_op_results(op)[0]};
    ++state->count;
    return true;
  }
  const loom_value_id_t value = loom_op_operands(op)[0];
  iree_host_size_t position = 0;
  while (position < state->count && state->records[position].value != value) {
    ++position;
  }
  if (position == state->count) {
    loom_channel_completion_require(
        completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_IDENTITY, op);
    return false;
  }
  if (loom_channel_wait_isa(op)) {
    // Every admitted read originates at acquire, so readiness is already
    // established and the same obligation remains live after borrowing.
    return true;
  }
  state->records[position].value = LOOM_VALUE_ID_INVALID;
  while (position && state->records[position - 1].endpoint == endpoint) {
    --position;
  }
  iree_host_size_t end = position;
  while (end < state->count && state->records[end].endpoint == endpoint &&
         state->records[end].value == LOOM_VALUE_ID_INVALID) {
    ++end;
  }
  *credits = (uint32_t)(end - position);
  memmove(&state->records[position], &state->records[end],
          (state->count - end) * sizeof(*state->records));
  state->count -= end - position;
  return true;
}

iree_status_t loom_channel_completion_analyze(
    const loom_channel_plan_t* plan, const loom_cfg_graph_t* graph,
    const loom_view_region_table_t* view_regions,
    const uint64_t* block_execution_counts, iree_arena_allocator_t* arena,
    loom_channel_completion_t* out_completion) {
  *out_completion = (loom_channel_completion_t){
      .value_domain = plan->value_domain,
  };
  if (!plan->action_count) {
    return iree_ok_status();
  }
  IREE_ASSERT(!view_regions ||
              view_regions->value_domain == plan->value_domain);
  loom_channel_completion_action_t* actions = NULL;
  uint32_t* action_endpoints = NULL;
  loom_channel_completion_endpoint_t* endpoints = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->action_count, sizeof(*actions), (void**)&actions));
  memset(actions, 0, plan->action_count * sizeof(*actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->action_count,
                                                 sizeof(*action_endpoints),
                                                 (void**)&action_endpoints));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->action_count, sizeof(*endpoints), (void**)&endpoints));
  uint32_t endpoint_count = 0;
  iree_host_size_t record_capacity = 0;
  for (iree_host_size_t i = 0; i < plan->action_count; ++i) {
    const loom_channel_plan_action_t* action = &plan->actions[i];
    const loom_op_t* op = action->op;
    if (!loom_channel_reserve_isa(op) && !loom_channel_acquire_isa(op) &&
        !loom_channel_publish_isa(op) && !loom_channel_release_isa(op) &&
        !loom_channel_wait_isa(op)) {
      loom_channel_completion_require(
          out_completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_ACTION, op);
      return iree_ok_status();
    }
    record_capacity +=
        loom_channel_reserve_isa(op) || loom_channel_acquire_isa(op);
    const bool producer =
        loom_channel_reserve_isa(op) || loom_channel_publish_isa(op);
    uint32_t endpoint = 0;
    while (endpoint < endpoint_count &&
           (endpoints[endpoint].channel != action->channel ||
            endpoints[endpoint].producer != producer)) {
      ++endpoint;
    }
    if (endpoint == endpoint_count) {
      endpoints[endpoint_count++] = (loom_channel_completion_endpoint_t){
          .channel = action->channel, .producer = producer};
    }
    action_endpoints[i] = endpoint;
  }
  loom_channel_completion_record_t* current_records = NULL;
  loom_channel_completion_record_t* edge_records = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, record_capacity,
                                                 sizeof(*current_records),
                                                 (void**)&current_records));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_capacity, sizeof(*edge_records), (void**)&edge_records));
  loom_channel_completion_block_t* blocks = NULL;
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*blocks), (void**)&blocks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*worklist), (void**)&worklist));
  iree_host_size_t action_index = 0;
  for (iree_host_size_t b = 0; b < graph->block_count; ++b) {
    blocks[b] =
        (loom_channel_completion_block_t){.actions.begin = action_index};
    while (action_index < plan->action_count &&
           plan->actions[action_index].op->parent_block ==
               graph->blocks[b].block) {
      const loom_op_t* op = plan->actions[action_index].op;
      if (graph->blocks[b].reachable &&
          (loom_channel_reserve_isa(op) || loom_channel_acquire_isa(op))) {
        uint32_t* bound =
            &endpoints[action_endpoints[action_index]].maximum_admissions;
        if (block_execution_counts) {
          const uint64_t count = block_execution_counts[b];
          *bound = count >= UINT32_MAX - *bound ? UINT32_MAX
                                                : *bound + (uint32_t)count;
        } else if (graph->blocks[b].component_is_cyclic) {
          *bound = UINT32_MAX;
        } else if (*bound != UINT32_MAX) {
          ++*bound;
        }
      }
      ++action_index;
    }
    blocks[b].actions.end = action_index;
  }
  iree_host_size_t read_position = 0, write_position = 1;
  worklist[0] = 0;
  blocks[0].reached = true;
  while (read_position < write_position) {
    const uint16_t index = worklist[read_position++];
    const loom_channel_completion_block_t* block = &blocks[index];
    loom_channel_completion_state_t current = {.records = current_records,
                                               .count = block->input.count};
    if (current.count) {
      memcpy(current_records, block->input.records,
             current.count * sizeof(*current_records));
    }
    for (iree_host_size_t i = block->actions.begin; i < block->actions.end;
         ++i) {
      if (!loom_channel_completion_apply(&plan->actions[i], action_endpoints[i],
                                         endpoints, &current,
                                         &actions[i].credits, out_completion)) {
        return iree_ok_status();
      }
    }
    const loom_cfg_edge_index_span_t successors =
        loom_cfg_graph_successor_edges(graph, index);
    if (!successors.count && current.count) {
      loom_channel_completion_require(
          out_completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_IDENTITY,
          graph->blocks[index].block->last_op);
      return iree_ok_status();
    }
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const loom_cfg_edge_info_t* edge = &graph->edges[successors.values[i]];
      const loom_block_t* destination =
          graph->blocks[edge->target_block_index].block;
      if (current.count) {
        memcpy(edge_records, current_records,
               current.count * sizeof(*edge_records));
      }
      const loom_value_id_t* arguments = NULL;
      uint16_t argument_count = 0;
      if (destination->arg_count &&
          !loom_cfg_terminator_payload_for_successor(
              edge->terminator, destination, &arguments, &argument_count)) {
        loom_channel_completion_require(
            out_completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_CONTROL,
            edge->terminator);
        return iree_ok_status();
      }
      // Edge transfers are simultaneous, including rotating iterated reads.
      // Completed holes have no owned SSA name to rename.
      for (iree_host_size_t r = 0; r < current.count; ++r) {
        if (current_records[r].value == LOOM_VALUE_ID_INVALID) {
          continue;
        }
        for (uint16_t a = 0; a < argument_count; ++a) {
          if (arguments[a] == current_records[r].value) {
            edge_records[r].value = destination->arg_ids[a];
          }
        }
      }
      loom_channel_completion_block_t* target =
          &blocks[edge->target_block_index];
      if (target->reached) {
        if (target->input.count != current.count ||
            (current.count &&
             memcmp(target->input.records, edge_records,
                    current.count * sizeof(*edge_records)) != 0)) {
          loom_channel_completion_require(
              out_completion, LOOM_CHANNEL_COMPLETION_REQUIREMENT_CONTROL,
              edge->terminator);
          return iree_ok_status();
        }
      } else {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            arena, current.count, sizeof(*edge_records),
            (void**)&target->input.records));
        if (current.count) {
          memcpy(target->input.records, edge_records,
                 current.count * sizeof(*edge_records));
        }
        target->input.count = current.count;
        target->reached = true;
        worklist[write_position++] = edge->target_block_index;
      }
    }
  }
  for (iree_host_size_t i = 0; i < plan->action_count; ++i) {
    actions[i].maximum_admissions =
        endpoints[action_endpoints[i]].maximum_admissions;
  }
  if (view_regions) {
    loom_channel_payload_record_t* payload_records = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->value_count,
                                                   sizeof(*payload_records),
                                                   (void**)&payload_records));
    for (loom_value_ordinal_t i = 0; i < plan->value_count; ++i) {
      payload_records[i] = (loom_channel_payload_record_t){
          .value_id = LOOM_VALUE_ID_INVALID,
      };
    }
    IREE_RETURN_IF_ERROR(loom_channel_completion_analyze_payload_records(
        plan, graph, view_regions, action_endpoints, endpoints, payload_records,
        arena));
    out_completion->payload_records = payload_records;
    out_completion->payload_record_count = plan->value_count;
  }
  out_completion->actions = actions;
  return iree_ok_status();
}
