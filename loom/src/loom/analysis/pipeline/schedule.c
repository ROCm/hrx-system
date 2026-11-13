// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/schedule.h"

#include <string.h>

#include "loom/analysis/value_relation.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/view/ops.h"

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

typedef struct loom_pipeline_collective_dependencies_t {
  // Borrowed function-local value map for this schedule.
  const loom_local_value_domain_t* domain;
  // Values depending on the current record, indexed by local ordinal.
  uint8_t* values;
  // Whether the operation being classified consumes a record-dependent value.
  bool dependent;
} loom_pipeline_collective_dependencies_t;

static iree_status_t loom_pipeline_collective_dependency(
    loom_value_id_t provider, loom_value_id_t type_carrier, void* user_data) {
  (void)type_carrier;
  loom_pipeline_collective_dependencies_t* dependencies = user_data;
  dependencies->dependent |=
      dependencies->values[loom_local_value_domain_ordinal(dependencies->domain,
                                                           provider)] != 0;
  return iree_ok_status();
}

// A closed borrow can execute once and distribute its escaping values without
// replicating the FIFO cursor or exposing an address in another worker's
// domain. The retained span owns all effects; external memory reads and escaped
// payload references still require a different realization.
static iree_status_t loom_pipeline_schedule_collectives(
    loom_module_t* module, loom_pipeline_schedule_t* schedule,
    const loom_local_value_domain_t* domain,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid) {
  *out_valid = false;
  uint8_t* provenance = NULL;
  uint8_t* dependent_values = NULL;
  uint32_t* internal_uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain->value_count, sizeof(*provenance), (void**)&provenance));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain->value_count,
                                                 sizeof(*dependent_values),
                                                 (void**)&dependent_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain->value_count,
                                                 sizeof(*internal_uses),
                                                 (void**)&internal_uses));
  memset(provenance, 0, domain->value_count * sizeof(*provenance));
  memset(dependent_values, 0, domain->value_count * sizeof(*dependent_values));
  memset(internal_uses, 0, domain->value_count * sizeof(*internal_uses));
  for (iree_host_size_t b = 0; b < schedule->control->graph.block_count; ++b) {
    const loom_pipeline_schedule_block_t* block = &schedule->blocks[b];
    loom_pipeline_schedule_step_t* steps =
        (loom_pipeline_schedule_step_t*)block->steps;
    for (iree_host_size_t begin = 0; begin < block->step_count; ++begin) {
      if (steps[begin].kind == LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE_MEMBER) {
        return loom_pipeline_schedule_reject(
            steps[begin].operation,
            IREE_SV("an immutable collective borrow owning this access"),
            diagnostic_emitter);
      }
      if (steps[begin].kind != LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE) {
        continue;
      }
      loom_op_t* acquire = steps[begin].operation;
      if (loom_channel_acquire_mode(acquire)) {
        return loom_pipeline_schedule_reject(
            acquire, IREE_SV("immutable collective channel consumption"),
            diagnostic_emitter);
      }
      const loom_value_id_t read = loom_channel_acquire_read(acquire);
      provenance[loom_local_value_domain_ordinal(
          domain, loom_channel_acquire_view(acquire))] = 1;
      dependent_values[loom_local_value_domain_ordinal(
          domain, loom_channel_acquire_view(acquire))] = 1;
      dependent_values[loom_local_value_domain_ordinal(domain, read)] = 1;
      iree_host_size_t end = begin + 1;
      iree_host_size_t operation_count = 2;
      iree_host_size_t result_capacity = 0;
      for (; end < block->step_count; ++end) {
        loom_op_t* operation = steps[end].operation;
        if (loom_channel_release_isa(operation) &&
            loom_channel_release_read(operation) == read) {
          break;
        }
        if (loom_view_load_isa(operation)) {
          if (provenance[loom_local_value_domain_ordinal(
                  domain, loom_view_load_view(operation))] != 1) {
            return loom_pipeline_schedule_reject(
                operation, IREE_SV("a load from this immutable channel record"),
                diagnostic_emitter);
          }
        } else if (steps[end].kind != LOOM_PIPELINE_SCHEDULE_STEP_DATAFLOW) {
          return loom_pipeline_schedule_reject(
              operation,
              IREE_SV(
                  "a closed immutable borrow before child or control transfer"),
              diagnostic_emitter);
        }
        const loom_value_relation_mask_t mask =
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REFERENCE_SOURCE) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD);
        loom_value_relation_iterator_t iterator;
        loom_value_relation_iterator_initialize(module, operation, mask,
                                                &iterator);
        loom_value_relation_t relation;
        while (loom_value_relation_iterator_next(&iterator, &relation)) {
          const loom_type_t type =
              loom_module_value_type(module, relation.destination_value_id);
          if (!loom_type_is_view(type) && !loom_type_is_buffer(type)) {
            continue;
          }
          const loom_value_ordinal_t source =
              loom_local_value_domain_ordinal(domain, relation.source_value_id);
          const loom_value_ordinal_t destination =
              loom_local_value_domain_ordinal(domain,
                                              relation.destination_value_id);
          provenance[destination] =
              provenance[destination] == 2 || provenance[source] != 1 ? 2 : 1;
        }
        loom_pipeline_collective_dependencies_t dependencies = {
            .domain = domain, .values = dependent_values};
        IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
            module, operation, loom_pipeline_collective_dependency,
            &dependencies));
        if (dependencies.dependent) {
          steps[end].kind = LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE_MEMBER;
          ++operation_count;
          result_capacity += operation->result_count;
          for (uint16_t result = 0; result < operation->result_count;
               ++result) {
            dependent_values[loom_local_value_domain_ordinal(
                domain, loom_op_results(operation)[result])] = 1;
          }
        }
      }
      if (end == block->step_count) {
        return loom_pipeline_schedule_reject(
            acquire, IREE_SV("a release closing the immutable record read"),
            diagnostic_emitter);
      }
      loom_pipeline_collective_t* collective = NULL;
      loom_op_t** operations = NULL;
      loom_pipeline_collective_result_t* results = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*collective), (void**)&collective));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, operation_count, sizeof(*operations), (void**)&operations));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, result_capacity, sizeof(*results), (void**)&results));
      *collective =
          (loom_pipeline_collective_t){.operations = operations,
                                       .operation_count = operation_count,
                                       .results = results};
      iree_host_size_t operation_index = 0;
      for (iree_host_size_t i = begin; i <= end; ++i) {
        if (steps[i].kind == LOOM_PIPELINE_SCHEDULE_STEP_DATAFLOW) {
          continue;
        }
        loom_op_t* operation = operations[operation_index++] =
            steps[i].operation;
        for (uint16_t operand = 0; operand < operation->operand_count;
             ++operand) {
          ++internal_uses[loom_local_value_domain_ordinal(
              domain, loom_op_operands(operation)[operand])];
        }
      }
      for (iree_host_size_t i = 0; i < operation_count; ++i) {
        loom_op_t* operation = operations[i];
        for (uint16_t result = 0; result < operation->result_count; ++result) {
          const loom_value_id_t id = loom_op_results(operation)[result];
          dependent_values[loom_local_value_domain_ordinal(domain, id)] = 0;
          const loom_value_t* value = loom_module_value(module, id);
          if (value->use_count ==
                  internal_uses[loom_local_value_domain_ordinal(domain, id)] &&
              !loom_value_has_attribute_uses(value) &&
              !loom_module_value_has_type_uses(module, id)) {
            continue;
          }
          const loom_type_t type = loom_module_value_type(module, id);
          if (!loom_type_is_scalar(type) && !loom_type_is_vector(type)) {
            return loom_pipeline_schedule_reject(
                operation,
                IREE_SV(
                    "values independent of the released collective payload"),
                diagnostic_emitter);
          }
          results[collective->result_count++] =
              (loom_pipeline_collective_result_t){.operation = (uint32_t)i,
                                                  .result = result};
        }
      }
      steps[begin].collective = collective;
      begin = end;
    }
  }
  *out_valid = true;
  return iree_ok_status();
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
    bool has_collectives = false;
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
          case LOOM_OP_CHANNEL_ACQUIRE:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE;
            has_collectives = true;
            break;
          case LOOM_OP_CHANNEL_RELEASE:
          case LOOM_OP_VIEW_LOAD:
            step.kind = LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE_MEMBER;
            has_collectives = true;
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
    if (has_collectives) {
      loom_local_value_domain_t domain;
      IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
          module, body, arena, &domain));
      bool valid = false;
      iree_status_t status = loom_pipeline_schedule_collectives(
          module, schedule, &domain, diagnostic_emitter, arena, &valid);
      loom_local_value_domain_release(&domain);
      IREE_RETURN_IF_ERROR(status);
      if (!valid) {
        return iree_ok_status();
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_pipeline_schedule_classify_choice(schedule, arena));
  }
  *out_schedule = &first->schedule;
  *out_valid = true;
  return iree_ok_status();
}
