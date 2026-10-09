// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/private_storage.h"

#include <stdlib.h>
#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/walk.h"

// The planner owns two traversals with distinct results: use classification
// retains the access/control tree, and value flow interprets that tree once.
// Neither the solver nor the rewrite rediscovers IR aliases or store order.
typedef struct loom_private_storage_region_t loom_private_storage_region_t;

typedef struct loom_private_storage_event_t {
  // Operation whose memory or control semantics this event represents.
  loom_op_t* op;
  // Access record for a memory operation, or NULL.
  loom_private_storage_access_t* access;
  // Allocation reset at this program point, or NULL.
  loom_private_storage_allocation_t* allocation;
  // Nested control regions in operation region order, or NULL.
  loom_private_storage_region_t** regions;
  // Explicit alternatives, including omitted optional fallthrough regions.
  uint8_t region_count;
  // Next event in the same block.
  struct loom_private_storage_event_t* next;
} loom_private_storage_event_t;

struct loom_private_storage_region_t {
  // Source region, whose block order indexes events.
  loom_region_t* region;
  // Shared fact-owned CFG snapshot, or NULL for a linear region.
  const loom_cfg_graph_t* graph;
  // First event for each block.
  loom_private_storage_event_t** events;
};

typedef struct loom_private_storage_interval_t {
  // Access awaiting byte-cell partitioning.
  loom_private_storage_access_t* access;
  // Allocation root supplied by shared region facts.
  loom_private_storage_allocation_t* allocation;
  // Exact inclusive root-relative byte offset.
  int64_t begin;
  // Exact exclusive root-relative byte offset.
  int64_t end;
  // Accessed scalar representation.
  loom_type_t type;
} loom_private_storage_interval_t;

typedef struct loom_private_storage_builder_t {
  // Result representation owned by the pass arena.
  loom_private_storage_plan_t* plan;
  // Shared function facts, borrowed until planning ends.
  const loom_value_fact_table_t* facts;
  // Shared byte-region analysis used during access classification.
  loom_view_region_table_t regions;
  // Availability for folding transport values without crossing lexical scopes.
  loom_dominance_info_t dominance;
  // Allocation by local value ordinal, including typed projections.
  loom_private_storage_allocation_t** allocations;
  // Collected intervals sorted once to create exact typed cells.
  loom_private_storage_interval_t* intervals;
  // Number of collected intervals.
  iree_host_size_t interval_count;
  // Allocated interval capacity.
  iree_host_size_t interval_capacity;
  // Capacity of the plan's projection list.
  iree_host_size_t projection_capacity;
  // All flow nodes in dense solver order.
  loom_private_storage_value_t** nodes;
  // Number of flow nodes.
  iree_host_size_t node_count;
  // Allocated flow-node pointer capacity.
  iree_host_size_t node_capacity;
  // Last signature, for child-before-parent appending.
  loom_private_storage_signature_t* signature_tail;
  // Signature owning each transport node, indexed by node ordinal.
  loom_op_t** anchors;
} loom_private_storage_builder_t;

static iree_status_t loom_private_storage_collect_allocation(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_private_storage_plan_t* plan = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_buffer_alloca_isa(op) || loom_buffer_alloca_memory_space(op) !=
                                         LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return iree_ok_status();
  }
  loom_private_storage_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*allocation),
                                           (void**)&allocation));
  *allocation = (loom_private_storage_allocation_t){
      .op = op, .next = plan->allocations, .selected = true};
  plan->allocations = allocation;
  return iree_ok_status();
}

static loom_private_storage_allocation_t* loom_private_storage_allocation(
    const loom_private_storage_builder_t* builder, loom_value_id_t value) {
  loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(&builder->plan->domain, value);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID ? NULL
                                               : builder->allocations[ordinal];
}

static iree_status_t loom_private_storage_collect_access(
    loom_private_storage_builder_t* builder, loom_op_t* op,
    loom_private_storage_allocation_t* allocation,
    loom_private_storage_access_t** out_access) {
  const bool load = loom_view_load_isa(op);
  loom_view_region_t region;
  bool derived = false;
  IREE_RETURN_IF_ERROR(loom_view_region_table_derive_element_region(
      &builder->regions,
      load ? loom_view_load_view(op) : loom_view_store_view(op),
      load ? loom_view_load_static_indices(op)
           : loom_view_store_static_indices(op),
      load ? loom_view_load_indices(op) : loom_view_store_indices(op), &region,
      &derived));
  loom_type_t type = loom_module_value_type(
      builder->plan->module,
      load ? loom_view_load_result(op) : loom_view_store_value(op));
  if (!derived || !loom_type_is_scalar(type) ||
      !loom_symbolic_expr_is_constant(&region.begin_byte_offset) ||
      !loom_symbolic_expr_is_constant(&region.end_byte_offset) ||
      region.begin_byte_offset.constant < 0 ||
      region.end_byte_offset.constant > allocation->byte_count ||
      region.end_byte_offset.constant <= region.begin_byte_offset.constant ||
      iree_any_bit_set(loom_op_effective_traits(builder->plan->module, op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    allocation->selected = false;
    return iree_ok_status();
  }
  loom_private_storage_access_t* access = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(builder->plan->arena,
                                           sizeof(*access), (void**)&access));
  *access = (loom_private_storage_access_t){.op = op,
                                            .next = builder->plan->accesses};
  builder->plan->accesses = access;
  if (builder->interval_count == builder->interval_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->plan->arena, builder->interval_count,
        builder->interval_count + 1, sizeof(*builder->intervals),
        &builder->interval_capacity, (void**)&builder->intervals));
  }
  builder->intervals[builder->interval_count++] =
      (loom_private_storage_interval_t){
          .access = access,
          .allocation = allocation,
          .begin = region.begin_byte_offset.constant,
          .end = region.end_byte_offset.constant,
          .type = type};
  *out_access = access;
  return iree_ok_status();
}

static iree_status_t loom_private_storage_collect_region(
    loom_private_storage_builder_t* builder, loom_region_t* region,
    bool supported, loom_private_storage_region_t** out_region) {
  loom_private_storage_plan_t* plan = builder->plan;
  loom_private_storage_region_t* result = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*result), (void**)&result));
  *result = (loom_private_storage_region_t){
      .region = region,
      .graph = loom_value_fact_table_lookup_cfg_graph(builder->facts, region)};
  if (result->graph) {
    // The rewrite extends High branch payloads. Other successor conventions
    // retain storage until their transport semantics participate in the plan.
    for (iree_host_size_t e = 0; e < result->graph->edge_count; ++e) {
      const loom_op_t* terminator = result->graph->edges[e].terminator;
      supported &=
          loom_cfg_br_isa(terminator) || loom_cfg_cond_br_isa(terminator);
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, region->block_count, sizeof(*result->events),
      (void**)&result->events));
  memset(result->events, 0, region->block_count * sizeof(*result->events));
  for (uint16_t b = 0; b < region->block_count; ++b) {
    loom_private_storage_event_t** tail = &result->events[b];
    loom_op_t* op = NULL;
    loom_block_for_each_op(loom_region_block(region, b), op) {
      loom_private_storage_access_t* access = NULL;
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        loom_private_storage_allocation_t* allocation =
            loom_private_storage_allocation(builder, operands[i]);
        if (!allocation) {
          continue;
        }
        if (!supported) {
          allocation->selected = false;
        }
        if (loom_buffer_view_isa(op) && i == 0) {
          if (plan->projection_count == builder->projection_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                plan->arena, plan->projection_count, plan->projection_count + 1,
                sizeof(*plan->projections), &builder->projection_capacity,
                (void**)&plan->projections));
          }
          plan->projections[plan->projection_count++] = op;
        } else if ((loom_view_load_isa(op) && i == 0) ||
                   (loom_view_store_isa(op) && i == 1)) {
          IREE_RETURN_IF_ERROR(loom_private_storage_collect_access(
              builder, op, allocation, &access));
        } else {
          // The entire allocation stays observable through an unmodeled use.
          allocation->selected = false;
        }
      }
      loom_private_storage_allocation_t* allocation =
          loom_buffer_alloca_isa(op)
              ? loom_private_storage_allocation(builder,
                                                loom_buffer_alloca_result(op))
              : NULL;
      if (!access && !allocation && !op->region_count) {
        continue;
      }
      loom_private_storage_event_t* event = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(plan->arena, sizeof(*event), (void**)&event));
      *event = (loom_private_storage_event_t){
          .op = op, .access = access, .allocation = allocation};
      *tail = event;
      tail = &event->next;
      if (!op->region_count) {
        continue;
      }
      event->region_count = op->region_count;
      const loom_op_vtable_t* vtable = loom_op_vtable(plan->module, op);
      while (event->region_count < vtable->region_count &&
             iree_any_bit_set(
                 vtable->region_descriptors[event->region_count].flags,
                 LOOM_REGION_OPTIONAL)) {
        ++event->region_count;
      }
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, event->region_count, sizeof(*event->regions),
          (void**)&event->regions));
      memset(event->regions, 0, event->region_count * sizeof(*event->regions));
      const bool control =
          loom_loop_like_isa(loom_loop_like_cast(plan->module, op)) ||
          loom_region_branch_isa(loom_region_branch_cast(plan->module, op));
      for (uint8_t r = 0; r < op->region_count; ++r) {
        loom_region_t* child = loom_op_regions(op)[r];
        if (!child) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_private_storage_collect_region(
            builder, child, supported && control && child->block_count == 1,
            &event->regions[r]));
      }
    }
  }
  *out_region = result;
  return iree_ok_status();
}

static int loom_private_storage_compare_intervals(const void* left_pointer,
                                                  const void* right_pointer) {
  const loom_private_storage_interval_t* left = left_pointer;
  const loom_private_storage_interval_t* right = right_pointer;
  if (left->allocation != right->allocation) {
    return (uintptr_t)left->allocation < (uintptr_t)right->allocation ? -1 : 1;
  }
  if (left->begin != right->begin) {
    return left->begin < right->begin ? -1 : 1;
  }
  if (left->end != right->end) {
    return left->end < right->end ? -1 : 1;
  }
  return 0;
}

static iree_status_t loom_private_storage_partition(
    loom_private_storage_builder_t* builder) {
  loom_private_storage_plan_t* plan = builder->plan;
  if (!builder->interval_count) {
    return iree_ok_status();
  }
  qsort(builder->intervals, builder->interval_count,
        sizeof(*builder->intervals), loom_private_storage_compare_intervals);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, builder->interval_count,
                                sizeof(*plan->cells), (void**)&plan->cells));
  loom_private_storage_cell_t* previous = NULL;
  for (iree_host_size_t i = 0; i < builder->interval_count; ++i) {
    const loom_private_storage_interval_t* interval = &builder->intervals[i];
    if (!interval->allocation->selected) {
      continue;
    }
    loom_private_storage_cell_t* cell = previous;
    if (previous && previous->allocation == interval->allocation &&
        interval->begin < previous->end) {
      if (interval->begin != previous->begin ||
          interval->end != previous->end ||
          !loom_type_equal(interval->type, previous->type)) {
        interval->allocation->selected = false;
        continue;
      }
    } else {
      cell = &plan->cells[plan->cell_count];
      *cell = (loom_private_storage_cell_t){.allocation = interval->allocation,
                                            .begin = interval->begin,
                                            .end = interval->end,
                                            .type = interval->type,
                                            .index = plan->cell_count++};
      previous = cell;
    }
    interval->access->cell = cell;
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_new_value(
    loom_private_storage_builder_t* builder, loom_value_id_t source,
    iree_host_size_t input_count, loom_private_storage_transport_t* transport,
    loom_private_storage_value_t** out_value) {
  loom_private_storage_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(builder->plan->arena, sizeof(*value),
                                           (void**)&value));
  *value = (loom_private_storage_value_t){.source = source,
                                          .replacement = LOOM_VALUE_ID_INVALID,
                                          .representative = value,
                                          .input_count = input_count,
                                          .transport = transport,
                                          .ordinal = builder->node_count};
  if (input_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->plan->arena, input_count, sizeof(*value->inputs),
        (void**)&value->inputs));
    memset(value->inputs, 0, input_count * sizeof(*value->inputs));
  }
  if (builder->node_count == builder->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->plan->arena, builder->node_count, builder->node_count + 1,
        sizeof(*builder->nodes), &builder->node_capacity,
        (void**)&builder->nodes));
  }
  builder->nodes[builder->node_count++] = value;
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_private_storage_source_value(
    loom_private_storage_builder_t* builder, loom_value_id_t source,
    loom_private_storage_value_t** out_value) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&builder->plan->domain, source);
  if (!builder->plan->values[ordinal]) {
    IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
        builder, source, 0, NULL, &builder->plan->values[ordinal]));
  }
  *out_value = builder->plan->values[ordinal];
  return iree_ok_status();
}

static iree_status_t loom_private_storage_copy_state(
    loom_private_storage_builder_t* builder,
    loom_private_storage_value_t** state,
    loom_private_storage_value_t*** out_state) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->plan->arena, builder->plan->cell_count,
                                sizeof(**out_state), (void**)out_state));
  memcpy(*out_state, state, builder->plan->cell_count * sizeof(**out_state));
  return iree_ok_status();
}

static iree_status_t loom_private_storage_new_signature(
    loom_private_storage_builder_t* builder, loom_op_t* op, loom_block_t* block,
    loom_private_storage_signature_t** out_signature) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      builder->plan->arena, sizeof(**out_signature), (void**)out_signature));
  **out_signature =
      (loom_private_storage_signature_t){.op = op, .block = block};
  return iree_ok_status();
}

static void loom_private_storage_append_signature(
    loom_private_storage_builder_t* builder,
    loom_private_storage_signature_t* signature) {
  if (builder->signature_tail) {
    builder->signature_tail->next = signature;
  } else {
    builder->plan->signatures = signature;
  }
  builder->signature_tail = signature;
}

static iree_status_t loom_private_storage_new_transport(
    loom_private_storage_builder_t* builder, loom_private_storage_cell_t* cell,
    iree_host_size_t outgoing_count,
    loom_private_storage_transport_t** out_transport) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      builder->plan->arena, sizeof(**out_transport), (void**)out_transport));
  **out_transport = (loom_private_storage_transport_t){
      .cell = cell, .outgoing_count = outgoing_count};
  if (outgoing_count) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->plan->arena, outgoing_count,
                                  sizeof(*(*out_transport)->outgoing),
                                  (void**)&(*out_transport)->outgoing));
    memset((*out_transport)->outgoing, 0,
           outgoing_count * sizeof(*(*out_transport)->outgoing));
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_flow_region(
    loom_private_storage_builder_t* builder,
    const loom_private_storage_region_t* region,
    loom_private_storage_value_t** state);

static iree_status_t loom_private_storage_flow_branch(
    loom_private_storage_builder_t* builder,
    const loom_private_storage_event_t* event,
    loom_private_storage_value_t** state) {
  loom_op_t* op = event->op;
  const uint8_t region_count = event->region_count;
  loom_private_storage_value_t*** alternatives = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->plan->arena, region_count,
                                sizeof(*alternatives), (void**)&alternatives));
  for (uint8_t r = 0; r < region_count; ++r) {
    IREE_RETURN_IF_ERROR(
        loom_private_storage_copy_state(builder, state, &alternatives[r]));
    if (event->regions[r]) {
      IREE_RETURN_IF_ERROR(loom_private_storage_flow_region(
          builder, event->regions[r], alternatives[r]));
    }
  }
  loom_private_storage_signature_t* signature = NULL;
  IREE_RETURN_IF_ERROR(
      loom_private_storage_new_signature(builder, op, NULL, &signature));
  signature->region_count = region_count;
  loom_private_storage_transport_t** tail = &signature->transports;
  for (iree_host_size_t c = 0; c < builder->plan->cell_count; ++c) {
    if (!builder->plan->cells[c].allocation->selected) {
      continue;
    }
    bool identical = true;
    for (uint8_t r = 1; r < region_count; ++r) {
      identical &= alternatives[r][c] == alternatives[0][c];
    }
    if (identical) {
      state[c] = alternatives[0][c];
      continue;
    }
    loom_private_storage_transport_t* transport = NULL;
    IREE_RETURN_IF_ERROR(loom_private_storage_new_transport(
        builder, &builder->plan->cells[c], region_count, &transport));
    IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
        builder, LOOM_VALUE_ID_INVALID, region_count, transport,
        &transport->result));
    for (uint8_t r = 0; r < region_count; ++r) {
      transport->outgoing[r] = alternatives[r][c];
      transport->result->inputs[r] = alternatives[r][c];
    }
    *tail = transport;
    tail = &transport->next;
    state[c] = transport->result;
  }
  loom_private_storage_append_signature(builder, signature);
  return iree_ok_status();
}

static iree_status_t loom_private_storage_flow_loop(
    loom_private_storage_builder_t* builder,
    const loom_private_storage_event_t* event, loom_loop_like_t loop,
    loom_private_storage_value_t** state) {
  loom_private_storage_signature_t* signature = NULL;
  IREE_RETURN_IF_ERROR(
      loom_private_storage_new_signature(builder, event->op, NULL, &signature));
  signature->region_count = event->region_count;
  loom_private_storage_value_t** recurring = NULL;
  IREE_RETURN_IF_ERROR(
      loom_private_storage_copy_state(builder, state, &recurring));
  const bool conditional = loom_loop_like_condition_region(loop) != NULL;
  loom_private_storage_transport_t** tail = &signature->transports;
  for (iree_host_size_t c = 0; c < builder->plan->cell_count; ++c) {
    if (!builder->plan->cells[c].allocation->selected) {
      continue;
    }
    loom_private_storage_transport_t* transport = NULL;
    IREE_RETURN_IF_ERROR(loom_private_storage_new_transport(
        builder, &builder->plan->cells[c], event->op->region_count,
        &transport));
    transport->initial = state[c];
    IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
        builder, LOOM_VALUE_ID_INVALID, 2, transport, &transport->entry));
    transport->entry->inputs[0] = state[c];
    recurring[c] = transport->entry;
    if (conditional) {
      IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
          builder, LOOM_VALUE_ID_INVALID, 1, transport, &transport->body));
    }
    IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
        builder, LOOM_VALUE_ID_INVALID, 1, transport, &transport->result));
    *tail = transport;
    tail = &transport->next;
  }
  if (conditional) {
    const uint8_t condition_index = loop.vtable->condition_region_index;
    IREE_RETURN_IF_ERROR(loom_private_storage_flow_region(
        builder, event->regions[condition_index], recurring));
    for (loom_private_storage_transport_t* t = signature->transports; t;
         t = t->next) {
      t->outgoing[condition_index] = recurring[t->cell->index];
      t->body->inputs[0] = recurring[t->cell->index];
      t->result->inputs[0] = recurring[t->cell->index];
      recurring[t->cell->index] = t->body;
    }
  }
  const uint8_t body_index = loop.vtable->body_region_index;
  IREE_RETURN_IF_ERROR(loom_private_storage_flow_region(
      builder, event->regions[body_index], recurring));
  for (loom_private_storage_transport_t* t = signature->transports; t;
       t = t->next) {
    t->outgoing[body_index] = recurring[t->cell->index];
    t->entry->inputs[1] = recurring[t->cell->index];
    if (!conditional) {
      t->result->inputs[0] = t->entry;
    }
    state[t->cell->index] = t->result;
  }
  loom_private_storage_append_signature(builder, signature);
  return iree_ok_status();
}

static iree_status_t loom_private_storage_flow_events(
    loom_private_storage_builder_t* builder,
    loom_private_storage_event_t* event, loom_private_storage_value_t** state) {
  for (; event; event = event->next) {
    if (event->allocation) {
      for (iree_host_size_t c = 0; c < builder->plan->cell_count; ++c) {
        if (builder->plan->cells[c].allocation == event->allocation) {
          state[c] = NULL;
        }
      }
    }
    loom_private_storage_access_t* access = event->access;
    if (access && access->cell && access->cell->allocation->selected) {
      const iree_host_size_t cell = access->cell->index;
      if (loom_view_load_isa(access->op)) {
        access->value = state[cell];
        const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
            &builder->plan->domain, loom_view_load_result(access->op));
        builder->plan->loads[ordinal] = access;
      } else {
        IREE_RETURN_IF_ERROR(loom_private_storage_source_value(
            builder, loom_view_store_value(access->op), &access->value));
        state[cell] = access->value;
      }
    }
    if (!event->regions) {
      continue;
    }
    loom_loop_like_t loop =
        loom_loop_like_cast(builder->plan->module, event->op);
    if (loom_loop_like_isa(loop)) {
      IREE_RETURN_IF_ERROR(
          loom_private_storage_flow_loop(builder, event, loop, state));
    } else if (loom_region_branch_isa(
                   loom_region_branch_cast(builder->plan->module, event->op))) {
      IREE_RETURN_IF_ERROR(
          loom_private_storage_flow_branch(builder, event, state));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_flow_region(
    loom_private_storage_builder_t* builder,
    const loom_private_storage_region_t* region,
    loom_private_storage_value_t** state) {
  if (region->region->block_count == 1 &&
      (!region->graph || region->graph->edge_count == 0)) {
    return loom_private_storage_flow_events(builder, region->events[0], state);
  }
  if (!region->graph) {
    // A function can contain disconnected blocks without any CFG edges. Each
    // block starts with independent state; definitions do not cross between
    // those blocks merely because they are adjacent in the region's storage.
    for (uint16_t b = 0; b < region->region->block_count; ++b) {
      loom_private_storage_value_t** block_state = NULL;
      IREE_RETURN_IF_ERROR(
          loom_private_storage_copy_state(builder, state, &block_state));
      IREE_RETURN_IF_ERROR(loom_private_storage_flow_events(
          builder, region->events[b], block_state));
    }
    return iree_ok_status();
  }
  const loom_cfg_graph_t* graph = region->graph;
  loom_private_storage_value_t*** exits = NULL;
  loom_private_storage_signature_t** signatures = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->plan->arena, graph->block_count,
                                sizeof(*exits), (void**)&exits));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->plan->arena, graph->block_count,
                                sizeof(*signatures), (void**)&signatures));
  memset(signatures, 0, graph->block_count * sizeof(*signatures));
  for (uint16_t b = 0; b < graph->block_count; ++b) {
    IREE_RETURN_IF_ERROR(
        loom_private_storage_copy_state(builder, state, &exits[b]));
    if (b == 0) {
      continue;
    }
    loom_private_storage_signature_t* signature = NULL;
    IREE_RETURN_IF_ERROR(loom_private_storage_new_signature(
        builder, NULL, loom_region_block(region->region, b), &signature));
    signatures[b] = signature;
    const loom_cfg_edge_index_span_t predecessors =
        loom_cfg_graph_predecessor_edges(graph, b);
    signature->predecessor_count = predecessors.count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->plan->arena, predecessors.count,
        sizeof(*signature->predecessors), (void**)&signature->predecessors));
    for (iree_host_size_t p = 0; p < predecessors.count; ++p) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, predecessors.values[p]);
      signature->predecessors[p] = (loom_private_storage_predecessor_t){
          .terminator = (loom_op_t*)edge->terminator,
          .successor_index = edge->successor_index};
    }
    loom_private_storage_transport_t** tail = &signature->transports;
    for (iree_host_size_t c = 0; c < builder->plan->cell_count; ++c) {
      if (!builder->plan->cells[c].allocation->selected) {
        continue;
      }
      loom_private_storage_transport_t* transport = NULL;
      IREE_RETURN_IF_ERROR(loom_private_storage_new_transport(
          builder, &builder->plan->cells[c], predecessors.count, &transport));
      IREE_RETURN_IF_ERROR(loom_private_storage_new_value(
          builder, LOOM_VALUE_ID_INVALID, predecessors.count, transport,
          &transport->entry));
      exits[b][c] = transport->entry;
      *tail = transport;
      tail = &transport->next;
    }
  }
  // Block order is not dominance order. All entry definitions exist before any
  // block is interpreted; backedges are attached only after every exit exists.
  for (uint16_t b = 0; b < graph->block_count; ++b) {
    IREE_RETURN_IF_ERROR(
        loom_private_storage_flow_events(builder, region->events[b], exits[b]));
  }
  for (uint16_t b = 1; b < graph->block_count; ++b) {
    loom_private_storage_signature_t* signature = signatures[b];
    const loom_cfg_edge_index_span_t predecessors =
        loom_cfg_graph_predecessor_edges(graph, b);
    for (loom_private_storage_transport_t* t = signature->transports; t;
         t = t->next) {
      for (iree_host_size_t p = 0; p < predecessors.count; ++p) {
        const loom_cfg_edge_info_t* edge =
            loom_cfg_graph_edge(graph, predecessors.values[p]);
        t->outgoing[p] = exits[edge->source_block_index][t->cell->index];
        t->entry->inputs[p] = t->outgoing[p];
      }
    }
    loom_private_storage_append_signature(builder, signature);
  }
  return iree_ok_status();
}

loom_private_storage_value_t* loom_private_storage_value_resolve(
    loom_private_storage_value_t* value) {
  if (!value) {
    return NULL;
  }
  loom_private_storage_value_t* representative = value;
  while (representative->representative != representative) {
    representative = representative->representative;
  }
  while (value->representative != representative) {
    loom_private_storage_value_t* next = value->representative;
    value->representative = representative;
    value = next;
  }
  return representative;
}

static iree_status_t loom_private_storage_visit_inputs(
    void* user_data, iree_host_size_t ordinal,
    loom_scc_successor_callback_t successor) {
  loom_private_storage_builder_t* builder = user_data;
  const loom_private_storage_value_t* value = builder->nodes[ordinal];
  for (iree_host_size_t i = 0; i < value->input_count; ++i) {
    if (value->inputs[i]) {
      IREE_RETURN_IF_ERROR(
          successor.fn(successor.user_data, value->inputs[i]->ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_solve(
    loom_private_storage_builder_t* builder) {
  loom_private_storage_plan_t* plan = builder->plan;
  if (!builder->node_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, builder->node_count, sizeof(*builder->anchors),
      (void**)&builder->anchors));
  memset(builder->anchors, 0, builder->node_count * sizeof(*builder->anchors));
  for (loom_private_storage_signature_t* s = plan->signatures; s; s = s->next) {
    loom_op_t* anchor = s->op ? s->op : s->block->first_op;
    for (loom_private_storage_transport_t* t = s->transports; t; t = t->next) {
      if (t->entry) {
        builder->anchors[t->entry->ordinal] = anchor;
      }
      if (t->body) {
        builder->anchors[t->body->ordinal] = anchor;
      }
      if (t->result) {
        builder->anchors[t->result->ordinal] = anchor;
      }
    }
  }
  loom_scc_list_t components;
  const loom_scc_graph_t graph = {
      .node_count = builder->node_count,
      .visit_successors = {loom_private_storage_visit_inputs, builder}};
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, NULL, plan->arena, &components));
  iree_host_size_t* component_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, builder->node_count, sizeof(*component_indices),
      (void**)&component_indices));
  for (iree_host_size_t c = 0; c < components.count; ++c) {
    for (iree_host_size_t n = 0; n < components.values[c].node_count; ++n) {
      component_indices[components.values[c].nodes[n]] = c;
    }
  }
  for (iree_host_size_t c = 0; c < components.count; ++c) {
    const loom_scc_t* component = &components.values[c];
    bool initialized = true;
    bool has_external = false;
    bool same = true;
    loom_private_storage_value_t* candidate = NULL;
    for (iree_host_size_t n = 0; n < component->node_count; ++n) {
      loom_private_storage_value_t* value = builder->nodes[component->nodes[n]];
      if (value->source != LOOM_VALUE_ID_INVALID) {
        has_external = true;
        candidate = value;
        continue;
      }
      if (value->input_count == 0) {
        initialized = false;
      }
      for (iree_host_size_t i = 0; i < value->input_count; ++i) {
        loom_private_storage_value_t* input = value->inputs[i];
        if (!input) {
          initialized = false;
          continue;
        }
        if (component_indices[input->ordinal] == c) {
          continue;
        }
        has_external = true;
        input = loom_private_storage_value_resolve(input);
        initialized &= input->initialized;
        if (!candidate) {
          candidate = input;
        }
        same &= candidate == input;
      }
    }
    initialized &= has_external;
    bool fold = initialized && same && candidate &&
                candidate->source != LOOM_VALUE_ID_INVALID;
    for (iree_host_size_t n = 0; n < component->node_count && fold; ++n) {
      loom_op_t* anchor = builder->anchors[component->nodes[n]];
      if (anchor) {
        fold = loom_value_is_available_before_op(&builder->dominance,
                                                 candidate->source, anchor);
      }
    }
    for (iree_host_size_t n = 0; n < component->node_count; ++n) {
      loom_private_storage_value_t* value = builder->nodes[component->nodes[n]];
      value->initialized = initialized;
      if (fold) {
        value->representative = candidate;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_select_live(
    loom_private_storage_builder_t* builder) {
  loom_private_storage_plan_t* plan = builder->plan;
  loom_private_storage_value_t** pending = NULL;
  bool* visited = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, builder->node_count, sizeof(*pending), (void**)&pending));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, builder->node_count, sizeof(*visited), (void**)&visited));
  if (builder->node_count) {
    memset(visited, 0, builder->node_count * sizeof(*visited));
  }
  iree_host_size_t count = 0;
  for (loom_private_storage_access_t* a = plan->accesses; a; a = a->next) {
    if (!a->cell || !a->cell->allocation->selected ||
        !loom_view_load_isa(a->op)) {
      continue;
    }
    loom_private_storage_value_t* value =
        loom_private_storage_value_resolve(a->value);
    if (!value || !value->initialized) {
      a->cell->allocation->selected = false;
    } else if (!visited[value->ordinal]) {
      visited[value->ordinal] = true;
      pending[count++] = value;
    }
  }
  while (count) {
    loom_private_storage_value_t* value = pending[--count];
    loom_private_storage_transport_t* transport = value->transport;
    if (!transport || transport->live ||
        !transport->cell->allocation->selected) {
      continue;
    }
    transport->live = true;
    // A loop slot is a complete recurrence tuple even if only its result is
    // read. Admission includes its initial, condition, and backedge values.
    loom_private_storage_value_t* definitions[] = {
        transport->entry, transport->body, transport->result};
    for (iree_host_size_t d = 0; d < IREE_ARRAYSIZE(definitions); ++d) {
      loom_private_storage_value_t* definition = definitions[d];
      if (!definition) {
        continue;
      }
      if (!definition->initialized) {
        transport->cell->allocation->selected = false;
      }
      for (iree_host_size_t i = 0; i < definition->input_count; ++i) {
        loom_private_storage_value_t* input =
            loom_private_storage_value_resolve(definition->inputs[i]);
        if (input && !visited[input->ordinal]) {
          visited[input->ordinal] = true;
          pending[count++] = input;
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_add_cfg_graph(
    void* user_data, const loom_cfg_graph_t* graph) {
  return loom_dominance_info_add_cfg_graph(user_data, graph, NULL);
}

static iree_status_t loom_private_storage_analyze(
    loom_pass_t* pass, loom_func_like_t function,
    loom_private_storage_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      plan->module, loom_func_like_body(function), plan->arena, &plan->domain));
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, plan->module, loom_pass_value_fact_scope_function(function),
      &facts));
  loom_private_storage_builder_t builder = {
      .plan = plan,
      .facts = facts,
      .dominance = {.module = plan->module, .arena = plan->arena}};
  IREE_RETURN_IF_ERROR(loom_value_fact_table_enumerate_cfg_graphs(
      facts, (loom_value_fact_cfg_graph_callback_t){
                 .user_data = &builder.dominance,
                 .fn = loom_private_storage_add_cfg_graph}));
  const iree_host_size_t value_count = plan->domain.value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, value_count,
                                                 sizeof(*builder.allocations),
                                                 (void**)&builder.allocations));
  memset(builder.allocations, 0, value_count * sizeof(*builder.allocations));
  for (loom_private_storage_allocation_t* a = plan->allocations; a;
       a = a->next) {
    const loom_value_id_t root = loom_buffer_alloca_result(a->op);
    builder.allocations[loom_local_value_domain_ordinal(&plan->domain, root)] =
        a;
    const loom_value_facts_t size = loom_value_fact_table_lookup(
        facts, loom_buffer_alloca_byte_length(a->op));
    a->selected = loom_value_facts_is_exact(size) && size.range_lo > 0;
    a->byte_count = a->selected ? size.range_lo : 0;
  }
  loom_symbolic_expr_context_t expressions;
  loom_symbolic_expr_context_initialize(plan->module, &plan->domain, facts,
                                        plan->arena, &expressions);
  IREE_RETURN_IF_ERROR(loom_view_region_table_initialize(
      &plan->domain, &expressions, &builder.regions));
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze(&builder.regions));
  for (iree_host_size_t i = 0; i < builder.regions.region_count; ++i) {
    const loom_view_region_t* region = &builder.regions.regions[i];
    loom_private_storage_allocation_t* allocation =
        loom_private_storage_allocation(&builder, region->root_value_id);
    builder.allocations[loom_local_value_domain_ordinal(
        &plan->domain, region->view_value_id)] = allocation;
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_private_storage_allocation_t* allocation = builder.allocations[i];
    if (!allocation) {
      continue;
    }
    loom_value_id_t value = plan->domain.value_ids[i];
    if (loom_value_has_attribute_uses(loom_module_value(plan->module, value)) ||
        loom_module_value_has_type_uses(plan->module, value)) {
      allocation->selected = false;
    }
  }
  loom_private_storage_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_private_storage_collect_region(
      &builder, loom_func_like_body(function), true, &region));
  IREE_RETURN_IF_ERROR(loom_private_storage_partition(&builder));
  if (!plan->cell_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, value_count, sizeof(*plan->values), (void**)&plan->values));
  memset(plan->values, 0, value_count * sizeof(*plan->values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, value_count, sizeof(*plan->loads), (void**)&plan->loads));
  memset(plan->loads, 0, value_count * sizeof(*plan->loads));
  loom_private_storage_value_t** state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->cell_count, sizeof(*state), (void**)&state));
  memset(state, 0, plan->cell_count * sizeof(*state));
  IREE_RETURN_IF_ERROR(
      loom_private_storage_flow_region(&builder, region, state));
  IREE_RETURN_IF_ERROR(loom_private_storage_solve(&builder));
  return loom_private_storage_select_live(&builder);
}

iree_status_t loom_private_storage_plan_build(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_private_storage_plan_t* out_plan) {
  *out_plan =
      (loom_private_storage_plan_t){.module = module, .arena = pass->arena};
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_private_storage_collect_allocation, out_plan},
      &result));
  if (!out_plan->allocations) {
    return iree_ok_status();
  }
  return loom_private_storage_analyze(pass, function, out_plan);
}

void loom_private_storage_plan_release(loom_private_storage_plan_t* plan) {
  if (loom_local_value_domain_is_acquired(&plan->domain)) {
    loom_local_value_domain_release(&plan->domain);
  }
}
