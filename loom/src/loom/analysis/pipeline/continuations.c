// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/continuations.h"

#include <string.h>

#include "loom/util/adaptive_sort.h"
#include "loom/util/cfg_loop_nest.h"
#include "loom/util/fact_cfg.h"

typedef struct loom_pipeline_continuation_candidate_t {
  // Existing instruction worker whose retained read supplies this observation.
  uint32_t worker;
  // Source phase whose record ordinal this observation preserves.
  uint32_t phase;
  // Exact read action, independent of the phase's record count.
  uint32_t action;
  // Structural execution position of the read's block.
  uint16_t position;
} loom_pipeline_continuation_candidate_t;

typedef struct loom_pipeline_continuation_control_t {
  // Borrowed loop and CFG facts from the original worker analysis.
  const loom_value_fact_cfg_region_t* structure;
  // Structural position by block ordinal, computed once for this selection.
  uint16_t* positions;
} loom_pipeline_continuation_control_t;

static bool loom_pipeline_continuation_less(
    const loom_pipeline_continuation_candidate_t* lhs,
    const loom_pipeline_continuation_candidate_t* rhs) {
  if (lhs->worker != rhs->worker) {
    return lhs->worker < rhs->worker;
  }
  if (lhs->phase != rhs->phase) {
    return lhs->phase < rhs->phase;
  }
  if (lhs->position != rhs->position) {
    return lhs->position < rhs->position;
  }
  return lhs->action < rhs->action;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_continuation_sort,
                          loom_pipeline_continuation_candidate_t,
                          loom_pipeline_continuation_less)

static iree_status_t loom_pipeline_continuation_control(
    const loom_pipeline_worker_t* worker, iree_arena_allocator_t* arena,
    loom_pipeline_continuation_control_t* control) {
  if (control->positions) {
    return iree_ok_status();
  }
  control->structure = loom_value_fact_table_lookup_cfg_region(
      &worker->facts, loom_func_like_body(worker->function));
  const loom_cfg_graph_t* graph = &control->structure->graph;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->block_count,
                                                 sizeof(*control->positions),
                                                 (void**)&control->positions));
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    control->positions[worker->block_order[i]] = (uint16_t)i;
  }
  return iree_ok_status();
}

// Finds the enclosing consumer lifetime, allowing several repeated phase reads
// within one source lease. A single phase can need an outer loop above its
// innermost shared scope; address-period loops are not source lifetimes.
static bool loom_pipeline_continuation_scope(
    const loom_pipeline_transport_input_t* input,
    const loom_pipeline_worker_t* worker, const loom_cfg_loop_nest_t* loops,
    uint16_t* scope) {
  while (*scope != LOOM_CFG_LOOP_NEST_NONE) {
    const uint16_t header = loops->loops[*scope].header_index;
    const uint64_t trips = worker->loop_trip_counts[*scope];
    const uint64_t headers = worker->execution_counts.values[header];
    if (loom_cfg_execution_count_is_exact(&worker->execution_counts, header) &&
        trips && trips != UINT64_MAX && headers % (trips + 1) == 0 &&
        input->admissions % trips == 0 &&
        headers / (trips + 1) == input->admissions / trips) {
      return true;
    }
    *scope = loops->loops[*scope].parent_loop_index;
  }
  return input->admissions == 1;
}

iree_status_t loom_pipeline_continuations_select(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* workers,
    const loom_pipeline_progress_observations_t* observations,
    const loom_pipeline_progress_site_t* sites, const uint32_t* inputs,
    iree_host_size_t input_count, loom_pipeline_progress_callback_t callback,
    iree_arena_allocator_t* arena,
    loom_pipeline_continuation_t** out_selections) {
  *out_selections = NULL;
  if (!input_count) {
    return iree_ok_status();
  }
  loom_pipeline_continuation_t* selections;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, input_count, sizeof(*selections), (void**)&selections));
  loom_pipeline_continuation_control_t* controls;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->worker_count, sizeof(*controls), (void**)&controls));
  memset(controls, 0, invocations->worker_count * sizeof(*controls));
  for (iree_host_size_t request = 0; request < input_count; ++request) {
    loom_pipeline_continuation_t* selection = &selections[request];
    *selection = (loom_pipeline_continuation_t){.worker = UINT32_MAX};
    const loom_pipeline_transport_input_t* input =
        workers[inputs[request]].transport.input;
    iree_host_size_t capacity = 0;
    for (iree_host_size_t phase = 0; phase < input->phase_count; ++phase) {
      const iree_host_size_t channel = loom_pipeline_invocations_channel_index(
          invocations, input->phases[phase].reservation->channel);
      capacity += observations[channel].count;
    }
    loom_pipeline_continuation_candidate_t* candidates;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*candidates), (void**)&candidates));
    iree_host_size_t count = 0;
    for (iree_host_size_t phase = 0; phase < input->phase_count; ++phase) {
      const iree_host_size_t channel = loom_pipeline_invocations_channel_index(
          invocations, input->phases[phase].reservation->channel);
      const loom_pipeline_progress_observations_t* population =
          &observations[channel];
      for (iree_host_size_t i = 0; i < population->count; ++i) {
        const loom_pipeline_progress_site_t* site =
            &sites[population->channels[i]];
        const loom_pipeline_worker_t* worker = &workers[site->worker];
        if (!worker->block_order) {
          continue;
        }
        const uint16_t block = worker->channels.actions[site->action]
                                   .op->parent_block->region_index;
        if (!loom_cfg_execution_count_is_exact(&worker->execution_counts,
                                               block)) {
          continue;
        }
        const uint64_t reads = worker->execution_counts.values[block];
        if (!input->admissions || reads % input->admissions ||
            reads / input->admissions != input->phases[phase].transfers) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_pipeline_continuation_control(
            worker, arena, &controls[site->worker]));
        candidates[count++] = (loom_pipeline_continuation_candidate_t){
            .worker = site->worker,
            .phase = (uint32_t)phase,
            .action = site->action,
            .position = controls[site->worker].positions[block],
        };
      }
    }
    loom_pipeline_continuation_sort(candidates, count);
    uint64_t best_cost = UINT64_MAX;
    iree_host_size_t best = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t begin = 0; begin < count;) {
      const uint32_t owner = candidates[begin].worker;
      const loom_pipeline_worker_t* worker = &workers[owner];
      const loom_cfg_loop_nest_t* loops = &controls[owner].structure->loops;
      iree_host_size_t end = begin, phases = 0, first = begin;
      uint16_t scope = LOOM_CFG_LOOP_NEST_NONE;
      while (end < count && candidates[end].worker == owner) {
        const loom_pipeline_continuation_candidate_t* candidate =
            &candidates[end];
        if (candidate->phase != phases) {
          break;
        }
        const uint16_t block = worker->channels.actions[candidate->action]
                                   .op->parent_block->region_index;
        if (!phases) {
          scope = loom_cfg_loop_nest_innermost(loops, block);
        } else {
          while (scope != LOOM_CFG_LOOP_NEST_NONE &&
                 !loom_cfg_loop_nest_contains(loops, scope, block)) {
            scope = loops->loops[scope].parent_loop_index;
          }
          if (candidate->position < candidates[first].position ||
              (candidate->position == candidates[first].position &&
               candidate->action < candidates[first].action)) {
            first = end;
          }
        }
        ++phases;
        do {
          ++end;
        } while (end < count && candidates[end].worker == owner &&
                 candidates[end].phase == candidate->phase);
      }
      if (phases == input->phase_count &&
          loom_pipeline_continuation_scope(input, worker, loops, &scope)) {
        const uint64_t cost =
            callback.cost(callback.user_data, inputs[request], owner);
        if (best == IREE_HOST_SIZE_MAX || cost < best_cost) {
          best = begin;
          best_cost = cost;
          selection->worker = owner;
          selection->first_phase = candidates[first].phase;
          selection->scope = scope;
        }
      }
      while (end < count && candidates[end].worker == owner) {
        ++end;
      }
      begin = end;
    }
    if (best != IREE_HOST_SIZE_MAX) {
      uint32_t* actions;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, input->phase_count, sizeof(*actions), (void**)&actions));
      for (iree_host_size_t phase = 0; phase < input->phase_count; ++phase) {
        actions[phase] = candidates[best].action;
        do {
          ++best;
        } while (best < count && candidates[best].worker == selection->worker &&
                 candidates[best].phase == phase);
      }
      selection->phase_actions = actions;
    }
  }
  *out_selections = selections;
  return iree_ok_status();
}
