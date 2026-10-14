// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_barrier_lifetime.h"

#include "iree/base/internal/arena.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_kernel_barrier_flow_state_t {
  // Active arrival result, or LOOM_VALUE_ID_INVALID when no phase is active.
  loom_value_id_t phase_id;
  // Phase required at exits from this connected control-flow component.
  loom_value_id_t exit_phase_id;
  // True after an incoming path has assigned this state.
  bool assigned;
} loom_kernel_barrier_flow_state_t;

typedef struct loom_kernel_barrier_region_request_t {
  // Region whose retained CFG must be checked.
  loom_region_t* region;
  // Phase active when control enters the region.
  loom_value_id_t entry_phase_id;
} loom_kernel_barrier_region_request_t;

typedef struct loom_kernel_barrier_lifetime_state_t {
  // Module owning the function and SSA values.
  const loom_module_t* module;
  // Caller-owned analysis options.
  const loom_kernel_barrier_lifetime_options_t* options;
  // Call-scoped arena owning queues and block states.
  iree_arena_allocator_t* arena;
  // Result receiving the emitted diagnostic count.
  loom_kernel_barrier_lifetime_result_t* result;
  // Regions waiting to be checked in parent-before-child order.
  loom_kernel_barrier_region_request_t* regions;
  // Index of the next region request to check.
  iree_host_size_t next_region;
  // Number of populated region requests.
  iree_host_size_t region_count;
  // Allocated region request capacity.
  iree_host_size_t region_capacity;
  // True after an authored lifetime diagnostic has been emitted.
  bool failed;
} loom_kernel_barrier_lifetime_state_t;

static iree_string_view_t loom_kernel_barrier_lifetime_op_name(
    const loom_kernel_barrier_lifetime_state_t* state, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  return vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
}

static iree_string_view_t loom_kernel_barrier_lifetime_phase_name(
    const loom_kernel_barrier_lifetime_state_t* state) {
  return iree_string_view_is_empty(state->options->phase_name)
             ? IREE_SV("kernel-barrier-lifetime")
             : state->options->phase_name;
}

static const loom_op_t* loom_kernel_barrier_lifetime_arrival(
    const loom_kernel_barrier_lifetime_state_t* state,
    loom_value_id_t phase_id) {
  if (phase_id == LOOM_VALUE_ID_INVALID ||
      phase_id >= state->module->values.count) {
    return NULL;
  }
  const loom_value_t* phase = loom_module_value(state->module, phase_id);
  if (loom_value_is_block_arg(phase)) {
    return NULL;
  }
  const loom_op_t* defining_op = loom_value_def_op(phase);
  return defining_op && loom_kernel_barrier_arrive_isa(defining_op)
             ? defining_op
             : NULL;
}

static iree_status_t loom_kernel_barrier_lifetime_fail(
    loom_kernel_barrier_lifetime_state_t* state, const loom_op_t* op,
    loom_value_id_t active_phase_id, iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_barrier_lifetime_phase_name(state)),
      loom_param_string(loom_kernel_barrier_lifetime_op_name(state, op)),
      loom_param_string(reason),
  };
  const loom_op_t* arrival =
      loom_kernel_barrier_lifetime_arrival(state, active_phase_id);
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("active phase arrived here"),
      .op = arrival,
  }};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_056,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = arrival ? related : NULL,
      .related_op_count = arrival ? IREE_ARRAYSIZE(related) : 0,
  };
  state->failed = true;
  ++state->result->error_count;
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_kernel_barrier_lifetime_push_region(
    loom_kernel_barrier_lifetime_state_t* state, loom_region_t* region,
    loom_value_id_t entry_phase_id) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  if (state->region_count == state->region_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->region_count, state->region_count + 1,
        sizeof(*state->regions), &state->region_capacity,
        (void**)&state->regions));
  }
  state->regions[state->region_count++] =
      (loom_kernel_barrier_region_request_t){
          .region = region,
          .entry_phase_id = entry_phase_id,
      };
  return iree_ok_status();
}

static iree_status_t loom_kernel_barrier_lifetime_check_op(
    loom_kernel_barrier_lifetime_state_t* state, loom_op_t* op,
    loom_value_id_t* phase_id) {
  if (loom_kernel_barrier_arrive_isa(op)) {
    if (*phase_id != LOOM_VALUE_ID_INVALID) {
      return loom_kernel_barrier_lifetime_fail(
          state, op, *phase_id,
          IREE_SV("another barrier arrives before the active phase is waited"));
    }
    *phase_id = loom_kernel_barrier_arrive_phase(op);
  } else if (loom_kernel_barrier_wait_isa(op)) {
    const loom_value_id_t waited_phase_id = loom_kernel_barrier_wait_phase(op);
    if (*phase_id != waited_phase_id) {
      return loom_kernel_barrier_lifetime_fail(
          state, op, *phase_id,
          IREE_SV("the wait does not consume the phase active on this path"));
    }
    *phase_id = LOOM_VALUE_ID_INVALID;
  } else if (loom_kernel_barrier_isa(op) &&
             *phase_id != LOOM_VALUE_ID_INVALID) {
    return loom_kernel_barrier_lifetime_fail(
        state, op, *phase_id,
        IREE_SV(
            "a complete barrier executes before the active phase is waited"));
  } else if (loom_call_like_isa(loom_call_like_const_cast(state->module, op)) &&
             *phase_id != LOOM_VALUE_ID_INVALID &&
             !iree_any_bit_set(loom_op_effective_traits(state->module, op),
                               LOOM_TRAIT_PURE)) {
    return loom_kernel_barrier_lifetime_fail(
        state, op, *phase_id,
        IREE_SV(
            "an impure or unresolved call executes while a phase is active"));
  } else if (*phase_id != LOOM_VALUE_ID_INVALID &&
             !iree_any_bit_set(loom_op_effective_traits(state->module, op),
                               LOOM_TRAIT_TERMINATOR) &&
             iree_any_bit_set(loom_op_effective_traits(state->module, op),
                              LOOM_TRAIT_CONVERGENT)) {
    return loom_kernel_barrier_lifetime_fail(
        state, op, *phase_id,
        IREE_SV(
            "another convergent operation executes while a phase is active"));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_barrier_lifetime_push_region(state, regions[i], *phase_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_barrier_lifetime_merge(
    loom_kernel_barrier_lifetime_state_t* state,
    loom_kernel_barrier_flow_state_t* target,
    loom_kernel_barrier_flow_state_t incoming, const loom_op_t* edge_op,
    uint16_t target_index, uint16_t* queue, iree_host_size_t* queue_count) {
  if (!target->assigned) {
    *target = incoming;
    target->assigned = true;
    queue[(*queue_count)++] = target_index;
    return iree_ok_status();
  }
  if (target->phase_id == incoming.phase_id &&
      target->exit_phase_id == incoming.exit_phase_id) {
    return iree_ok_status();
  }
  const loom_value_id_t active_phase_id =
      target->phase_id != LOOM_VALUE_ID_INVALID ? target->phase_id
                                                : incoming.phase_id;
  return loom_kernel_barrier_lifetime_fail(
      state, edge_op, active_phase_id,
      IREE_SV("incoming control-flow paths carry different active phases"));
}

static iree_status_t loom_kernel_barrier_lifetime_check_region(
    loom_kernel_barrier_lifetime_state_t* state,
    loom_kernel_barrier_region_request_t request) {
  const loom_cfg_graph_t* graph = loom_value_fact_table_lookup_cfg_graph(
      state->options->fact_table, request.region);
  if (!graph) {
    IREE_ASSERT_EQ(request.region->block_count, 1,
                   "multi-block regions require retained CFG facts");
    loom_block_t* block = loom_region_entry_block(request.region);
    loom_value_id_t phase_id = request.entry_phase_id;
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_barrier_lifetime_check_op(state, op, &phase_id));
      if (state->failed) {
        return iree_ok_status();
      }
    }
    if (phase_id != request.entry_phase_id) {
      return loom_kernel_barrier_lifetime_fail(
          state, block->last_op, phase_id,
          IREE_SV("a structured region changes its surrounding active phase"));
    }
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(graph->block_count, request.region->block_count);

  loom_kernel_barrier_flow_state_t* block_states = NULL;
  uint16_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, graph->block_count,
                                sizeof(*block_states), (void**)&block_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, graph->block_count, sizeof(*queue), (void**)&queue));
  memset(block_states, 0, graph->block_count * sizeof(*block_states));

  iree_host_size_t queue_head = 0;
  iree_host_size_t queue_count = 0;
  for (uint16_t seed_index = 0; seed_index < graph->block_count; ++seed_index) {
    if (block_states[seed_index].assigned) {
      continue;
    }
    const loom_value_id_t seed_phase_id =
        seed_index == 0 ? request.entry_phase_id : LOOM_VALUE_ID_INVALID;
    block_states[seed_index] = (loom_kernel_barrier_flow_state_t){
        .phase_id = seed_phase_id,
        .exit_phase_id = seed_phase_id,
        .assigned = true,
    };
    queue[queue_count++] = seed_index;

    while (queue_head < queue_count) {
      const uint16_t block_index = queue[queue_head++];
      loom_kernel_barrier_flow_state_t flow = block_states[block_index];
      loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(
            loom_kernel_barrier_lifetime_check_op(state, op, &flow.phase_id));
        if (state->failed) {
          return iree_ok_status();
        }
      }

      const loom_cfg_edge_index_span_t successors =
          loom_cfg_graph_successor_edges(graph, block_index);
      if (successors.count == 0 && flow.phase_id != flow.exit_phase_id) {
        return loom_kernel_barrier_lifetime_fail(
            state, block->last_op, flow.phase_id,
            IREE_SV("a control-flow exit leaves a split-barrier phase active"));
      }
      for (iree_host_size_t i = 0; i < successors.count; ++i) {
        const loom_cfg_edge_info_t* edge =
            loom_cfg_graph_edge(graph, successors.values[i]);
        IREE_RETURN_IF_ERROR(loom_kernel_barrier_lifetime_merge(
            state, &block_states[edge->target_block_index], flow,
            edge->terminator, edge->target_block_index, queue, &queue_count));
        if (state->failed) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

// Returns true when the function-local definition index contains a split
// barrier arrival. The dense value-domain scan keeps the common no-barrier path
// allocation-free and avoids rediscovering operations through the region tree.
static bool loom_kernel_barrier_lifetime_has_arrival(
    const loom_module_t* module,
    const loom_local_value_domain_t* value_domain) {
  IREE_ASSERT_EQ(value_domain->module, module);
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  for (loom_value_ordinal_t i = 0; i < value_domain->definition_count; ++i) {
    const loom_value_t* value =
        loom_module_value(module, value_domain->value_ids[i]);
    if (!loom_value_is_block_arg(value) &&
        loom_kernel_barrier_arrive_isa(loom_value_def_op(value))) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_kernel_barrier_lifetime_verify_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_kernel_barrier_lifetime_options_t* options,
    loom_kernel_barrier_lifetime_result_t* out_result) {
  *out_result = (loom_kernel_barrier_lifetime_result_t){0};
  loom_region_t* body = loom_func_like_body(function);
  if (!body || body->block_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(options->value_domain->region, body);
  if (!loom_kernel_barrier_lifetime_has_arrival(module,
                                                options->value_domain)) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_kernel_barrier_lifetime_state_t state = {
      .module = module,
      .options = options,
      .arena = &arena,
      .result = out_result,
  };
  iree_status_t status = loom_kernel_barrier_lifetime_push_region(
      &state, body, LOOM_VALUE_ID_INVALID);
  while (iree_status_is_ok(status) && state.next_region < state.region_count &&
         !state.failed) {
    status = loom_kernel_barrier_lifetime_check_region(
        &state, state.regions[state.next_region++]);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
