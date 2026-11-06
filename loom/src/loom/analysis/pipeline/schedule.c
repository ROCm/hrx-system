// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/schedule.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/pipeline/ops.h"

typedef struct loom_pipeline_schedule_pending_t {
  // Serial scope awaiting classification of its own region.
  loom_pipeline_schedule_t schedule;
  // Next nested scope discovered during the owning construction traversal.
  struct loom_pipeline_schedule_pending_t* next;
} loom_pipeline_schedule_pending_t;

static iree_status_t loom_pipeline_schedule_reject(
    const loom_op_t* operation, iree_string_view_t requirement,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_diagnostic_param_t parameters[] = {loom_param_string(requirement)};
  return iree_diagnostic_emit(diagnostic_emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = operation,
                                  .error = LOOM_ERR_LOWERING_066,
                                  .params = parameters,
                                  .param_count = IREE_ARRAYSIZE(parameters),
                              });
}

// Pure schedule dataflow is stable across a child's execution. An acyclic
// choice with no intermediate completion or resource lifetime can therefore
// select the same child independently for start and retirement projections.
static iree_status_t loom_pipeline_schedule_classify_choice(
    loom_pipeline_schedule_t* schedule, iree_arena_allocator_t* arena) {
  const loom_cfg_graph_t* graph = &schedule->control->graph;
  if (graph->has_cycles) {
    return iree_ok_status();
  }
  uint8_t* visits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*visits), (void**)&visits));
  memset(visits, 0, graph->block_count * sizeof(*visits));
  for (iree_host_size_t r = 0; r < graph->reverse_postorder.count; ++r) {
    const uint16_t b = graph->reverse_postorder.values[r];
    const loom_pipeline_schedule_block_t* block = &schedule->blocks[b];
    uint8_t count = visits[b];
    for (iree_host_size_t s = 0; s < block->step_count; ++s) {
      switch (block->steps[s].kind) {
        case LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION:
          if (++count > 1) {
            return iree_ok_status();
          }
          break;
        case LOOM_PIPELINE_SCHEDULE_STEP_DATAFLOW:
        case LOOM_PIPELINE_SCHEDULE_STEP_BRANCH:
        case LOOM_PIPELINE_SCHEDULE_STEP_YIELD:
          break;
        default:
          return iree_ok_status();
      }
    }
    const loom_cfg_block_info_t* info = &graph->blocks[b];
    for (iree_host_size_t s = 0; s < info->successor_count; ++s) {
      const uint16_t successor =
          graph->successor_indices[info->successor_start + s];
      if (count > visits[successor]) {
        visits[successor] = count;
      }
    }
  }
  schedule->single_child_choice = true;
  return iree_ok_status();
}

iree_status_t loom_pipeline_schedule_build(
    loom_module_t* module, loom_op_t* declaration,
    loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_schedule_t** out_schedule, bool* out_valid) {
  *out_schedule = NULL;
  *out_valid = false;
  loom_pipeline_schedule_pending_t* first = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*first), (void**)&first));
  *first = (loom_pipeline_schedule_pending_t){
      .schedule = {.declaration = declaration},
  };
  loom_pipeline_schedule_pending_t* last = first;
  for (loom_pipeline_schedule_pending_t* pending = first; pending;
       pending = pending->next) {
    loom_pipeline_schedule_t* schedule = &pending->schedule;
    const loom_region_t* body =
        loom_pipeline_serial_body(schedule->declaration);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
        facts, module, body, &schedule->control));
    loom_pipeline_schedule_block_t* blocks = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, body->block_count, sizeof(*blocks), (void**)&blocks));
    schedule->blocks = blocks;
    for (uint16_t b = 0; b < body->block_count; ++b) {
      const loom_block_t* block = body->blocks[b];
      loom_pipeline_schedule_step_t* steps = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, block->op_count, sizeof(*steps), (void**)&steps));
      blocks[b] = (loom_pipeline_schedule_block_t){
          .steps = steps,
          .step_count = block->op_count,
      };
      iree_host_size_t step_index = 0;
      for (loom_op_t* operation = block->first_op; operation;
           operation = operation->next_op) {
        loom_pipeline_schedule_step_t step = {.operation = operation};
        switch (operation->kind) {
          case LOOM_OP_PIPELINE_COMPOSE:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
            break;
          case LOOM_OP_PIPELINE_SERIAL: {
            loom_pipeline_schedule_pending_t* nested = NULL;
            IREE_RETURN_IF_ERROR(
                iree_arena_allocate(arena, sizeof(*nested), (void**)&nested));
            *nested = (loom_pipeline_schedule_pending_t){
                .schedule = {.declaration = operation},
            };
            last->next = nested;
            last = nested;
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_SERIAL;
            step.nested = &nested->schedule;
            break;
          }
          case LOOM_OP_PIPELINE_MEMORY:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_MEMORY;
            break;
          case LOOM_OP_BUFFER_ALLOCA:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_ALLOCATION;
            break;
          case LOOM_OP_CHANNEL_BIND:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_CHANNEL;
            break;
          case LOOM_OP_PIPELINE_YIELD:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
            break;
          default: {
            if (operation->region_count) {
              return loom_pipeline_schedule_reject(
                  operation,
                  IREE_SV("resident schedule control lowered to CFG"),
                  diagnostic_emitter);
            }
            if (operation->successor_count != 0 &&
                iree_any_bit_set(operation->traits, LOOM_TRAIT_TERMINATOR)) {
              step.kind = LOOM_PIPELINE_SCHEDULE_STEP_BRANCH;
              break;
            }
            const loom_trait_flags_t traits =
                loom_op_effective_traits(module, operation);
            if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
                loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
                iree_any_bit_set(traits, LOOM_TRAIT_OBSERVABLE_EFFECT |
                                             LOOM_TRAIT_CONVERGENT)) {
              return loom_pipeline_schedule_reject(
                  operation,
                  IREE_SV("an execution placement for construction effects"),
                  diagnostic_emitter);
            }
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_DATAFLOW;
            break;
          }
        }
        steps[step_index++] = step;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_pipeline_schedule_classify_choice(schedule, arena));
  }
  *out_schedule = &first->schedule;
  *out_valid = true;
  return iree_ok_status();
}
