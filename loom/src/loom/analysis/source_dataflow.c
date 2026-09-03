// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_dataflow.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"

typedef struct loom_source_dataflow_runtime_rule_t {
  // First source ordinal in the solver source pool.
  uint32_t source_start;
  // First target ordinal in the solver target pool.
  uint32_t target_start;
  // Number of source ordinals.
  uint32_t source_count;
  // Number of target ordinals.
  uint32_t target_count;
  // Evidence tested or copied from sources.
  loom_source_dataflow_bits_t source_bits;
  // Fixed evidence added to targets for ANY and ALL rules.
  loom_source_dataflow_bits_t target_bits;
  // Normalized transfer kind; seeds are applied during graph construction.
  loom_source_dataflow_rule_kind_t kind;
} loom_source_dataflow_runtime_rule_t;

typedef struct loom_source_dataflow_build_state_t {
  // Static transfer provider being instantiated.
  const loom_source_dataflow_provider_t* provider;
  // Immutable solve environment.
  const loom_source_dataflow_environment_t* environment;
  // Arena owning all runtime graph and result storage.
  iree_arena_allocator_t* arena;
  // Result receiving dense states and statistics.
  loom_source_dataflow_result_t* result;
  // Normalized runtime transfer equations.
  loom_source_dataflow_runtime_rule_t* rules;
  // Number of populated runtime rules.
  uint32_t rule_count;
  // Allocated capacity of rules.
  iree_host_size_t rule_capacity;
  // Flattened source ordinals referenced by rules.
  loom_value_ordinal_t* sources;
  // Number of populated source ordinals.
  uint32_t source_count;
  // Allocated capacity of sources.
  iree_host_size_t source_capacity;
  // Flattened target ordinals referenced by rules.
  loom_value_ordinal_t* targets;
  // Number of populated target ordinals.
  uint32_t target_count;
  // Allocated capacity of targets.
  iree_host_size_t target_capacity;
} loom_source_dataflow_build_state_t;

static iree_status_t loom_source_dataflow_invalid_provider(
    const loom_source_dataflow_provider_t* provider, const char* detail) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "source dataflow provider '%.*s' %s",
      (int)provider->name.size, provider->name.data, detail);
}

static const loom_source_dataflow_operation_t*
loom_source_dataflow_lookup_operation(
    const loom_source_dataflow_provider_t* provider, loom_op_kind_t kind) {
  const uint8_t dialect_id = loom_op_dialect_id(kind);
  if (dialect_id < provider->dialect_base_id) return NULL;
  const uint8_t dialect_index = dialect_id - provider->dialect_base_id;
  if (dialect_index >= provider->dialect_count) return NULL;
  const loom_source_dataflow_dialect_table_t* dialect =
      &provider->dialects[dialect_index];
  const uint8_t operation_ordinal = loom_op_dialect_index(kind);
  if (operation_ordinal >= dialect->operation_count) return NULL;
  const uint16_t operation_index =
      dialect->operation_indices[operation_ordinal];
  return operation_index == LOOM_SOURCE_DATAFLOW_OPERATION_INDEX_NONE
             ? NULL
             : &provider->operations[operation_index - 1];
}

static iree_status_t loom_source_dataflow_append_ordinal(
    iree_arena_allocator_t* arena, loom_value_ordinal_t ordinal,
    uint32_t slice_start, loom_value_ordinal_t** values, uint32_t* value_count,
    iree_host_size_t* value_capacity) {
  for (uint32_t i = slice_start; i < *value_count; ++i) {
    if ((*values)[i] == ordinal) return iree_ok_status();
  }
  if (*value_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source dataflow graph exceeds ordinal range");
  }
  const iree_host_size_t minimum_capacity = (iree_host_size_t)*value_count + 1;
  if (minimum_capacity > *value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, *value_count, minimum_capacity, sizeof(**values), value_capacity,
        (void**)values));
  }
  (*values)[(*value_count)++] = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_append_source(
    loom_source_dataflow_build_state_t* state, uint32_t slice_start,
    loom_value_ordinal_t ordinal) {
  return loom_source_dataflow_append_ordinal(
      state->arena, ordinal, slice_start, &state->sources, &state->source_count,
      &state->source_capacity);
}

static iree_status_t loom_source_dataflow_append_target(
    loom_source_dataflow_build_state_t* state, uint32_t slice_start,
    loom_value_ordinal_t ordinal) {
  return loom_source_dataflow_append_ordinal(
      state->arena, ordinal, slice_start, &state->targets, &state->target_count,
      &state->target_capacity);
}

static iree_status_t loom_source_dataflow_append_rule(
    loom_source_dataflow_build_state_t* state,
    loom_source_dataflow_runtime_rule_t rule) {
  if (state->rule_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source dataflow graph exceeds rule range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)state->rule_count + 1;
  if (minimum_capacity > state->rule_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->rule_count, minimum_capacity,
        sizeof(*state->rules), &state->rule_capacity, (void**)&state->rules));
  }
  state->rules[state->rule_count++] = rule;
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_resolve_port(
    loom_source_dataflow_build_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_source_dataflow_port_t port,
    bool append_source, uint32_t slice_start) {
  loom_value_slice_t values = {0};
  if (port.kind == LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD) {
    const uint8_t field_count = loom_op_vtable_operand_descriptor_count(vtable);
    if (port.field_index >= field_count) {
      return loom_source_dataflow_invalid_provider(
          state->provider, "references an absent operand field");
    }
    values = loom_op_operand_field_span(vtable, op, port.field_index);
  } else {
    const uint8_t field_count =
        (uint8_t)(vtable->fixed_result_count +
                  (iree_any_bit_set(vtable->vtable_flags,
                                    LOOM_OP_VTABLE_VARIADIC_RESULTS)
                       ? 1
                       : 0));
    if (port.field_index >= field_count) {
      return loom_source_dataflow_invalid_provider(
          state->provider, "references an absent result field");
    }
    values = loom_op_result_field_span(vtable, op, port.field_index);
  }
  const loom_local_value_domain_t* value_domain =
      state->environment->program->value_domain;
  for (uint16_t i = 0; i < values.count; ++i) {
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_try_ordinal(value_domain, values.values[i]);
    if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source dataflow operation port escapes the indexed value domain");
    }
    if (append_source) {
      IREE_RETURN_IF_ERROR(
          loom_source_dataflow_append_source(state, slice_start, ordinal));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_source_dataflow_append_target(state, slice_start, ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_resolve_ports(
    loom_source_dataflow_build_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable,
    const loom_source_dataflow_operation_t* operation, uint32_t port_mask,
    bool append_source, uint32_t slice_start) {
  const loom_source_dataflow_port_t* ports =
      state->provider->ports + operation->port_start;
  for (uint8_t i = 0; i < operation->port_count; ++i) {
    if ((port_mask & ((uint32_t)1u << i)) == 0) continue;
    IREE_RETURN_IF_ERROR(loom_source_dataflow_resolve_port(
        state, op, vtable, ports[i], append_source, slice_start));
  }
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_apply_seed(
    loom_source_dataflow_build_state_t* state, uint32_t target_start,
    loom_source_dataflow_bits_t bits) {
  for (uint32_t i = target_start; i < state->target_count; ++i) {
    state->result->states[state->targets[i]] |= bits;
  }
  state->target_count = target_start;
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_build_rule(
    loom_source_dataflow_build_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable,
    const loom_source_dataflow_operation_t* operation,
    const loom_source_dataflow_rule_t* rule) {
  const uint32_t source_start = state->source_count;
  const uint32_t target_start = state->target_count;
  IREE_RETURN_IF_ERROR(loom_source_dataflow_resolve_ports(
      state, op, vtable, operation, rule->source_port_mask,
      /*append_source=*/true, source_start));
  IREE_RETURN_IF_ERROR(loom_source_dataflow_resolve_ports(
      state, op, vtable, operation, rule->target_port_mask,
      /*append_source=*/false, target_start));
  const uint32_t source_count = state->source_count - source_start;
  const uint32_t target_count = state->target_count - target_start;
  if (target_count == 0 ||
      (rule->kind != LOOM_SOURCE_DATAFLOW_RULE_SEED && source_count == 0)) {
    state->source_count = source_start;
    state->target_count = target_start;
    return iree_ok_status();
  }
  if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_SEED) {
    return loom_source_dataflow_apply_seed(state, target_start,
                                           rule->target_bits);
  }
  if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_COPY) {
    if (source_count != target_count) {
      return loom_source_dataflow_invalid_provider(
          state->provider, "has a copy rule with unequal resolved spans");
    }
    for (uint32_t i = 0; i < source_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_dataflow_append_rule(
          state, (loom_source_dataflow_runtime_rule_t){
                     .source_start = source_start + i,
                     .target_start = target_start + i,
                     .source_count = 1,
                     .target_count = 1,
                     .source_bits = rule->source_bits,
                     .target_bits = rule->target_bits,
                     .kind = rule->kind,
                 }));
    }
    return iree_ok_status();
  }
  return loom_source_dataflow_append_rule(state,
                                          (loom_source_dataflow_runtime_rule_t){
                                              .source_start = source_start,
                                              .target_start = target_start,
                                              .source_count = source_count,
                                              .target_count = target_count,
                                              .source_bits = rule->source_bits,
                                              .target_bits = rule->target_bits,
                                              .kind = rule->kind,
                                          });
}

static iree_status_t loom_source_dataflow_build_operation(
    loom_source_dataflow_build_state_t* state, const loom_op_t* op,
    const loom_source_dataflow_operation_t* operation) {
  if (operation->rule_count == 0) return iree_ok_status();
  const loom_op_vtable_t* vtable =
      loom_op_vtable(state->environment->program->module, op);
  if (vtable == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source dataflow operation has no registered vtable");
  }
  const loom_source_dataflow_rule_t* rules =
      state->provider->rules + operation->rule_start;
  uint64_t required_predicates = 0;
  for (uint8_t i = 0; i < operation->rule_count; ++i) {
    const loom_source_dataflow_rule_t* rule = &rules[i];
    const uint32_t valid_port_mask =
        operation->port_count == 32
            ? UINT32_MAX
            : (((uint32_t)1u << operation->port_count) - 1);
    if (((rule->source_port_mask | rule->target_port_mask) &
         ~valid_port_mask) != 0) {
      return loom_source_dataflow_invalid_provider(
          state->provider, "has a rule that selects an absent operation port");
    }
    if (rule->predicate_index_plus_one != 0) {
      required_predicates |= (uint64_t)1u
                             << (rule->predicate_index_plus_one - 1);
    }
  }

  uint64_t matching_predicates = 0;
  while (required_predicates != 0) {
    const uint32_t predicate_index =
        (uint32_t)iree_math_count_trailing_zeros_u64(required_predicates);
    required_predicates &= required_predicates - 1;
    bool matches = false;
    const loom_source_dataflow_predicate_t predicate =
        state->provider->predicates[predicate_index];
    IREE_RETURN_IF_ERROR(
        predicate.fn(predicate.user_data, state->environment, op, &matches));
    ++state->result->statistics.predicate_invocation_count;
    if (matches) matching_predicates |= (uint64_t)1u << predicate_index;
  }

  for (uint8_t i = 0; i < operation->rule_count; ++i) {
    const loom_source_dataflow_rule_t* rule = &rules[i];
    if (rule->predicate_index_plus_one != 0 &&
        (matching_predicates &
         ((uint64_t)1u << (rule->predicate_index_plus_one - 1))) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_source_dataflow_build_rule(state, op, vtable, operation, rule));
  }
  return iree_ok_status();
}

static iree_status_t loom_source_dataflow_build_structural_copy(
    loom_source_dataflow_build_state_t* state, loom_value_ordinal_t source,
    loom_value_ordinal_t target) {
  const uint32_t source_start = state->source_count;
  const uint32_t target_start = state->target_count;
  IREE_RETURN_IF_ERROR(
      loom_source_dataflow_append_source(state, source_start, source));
  IREE_RETURN_IF_ERROR(
      loom_source_dataflow_append_target(state, target_start, target));
  return loom_source_dataflow_append_rule(
      state, (loom_source_dataflow_runtime_rule_t){
                 .source_start = source_start,
                 .target_start = target_start,
                 .source_count = 1,
                 .target_count = 1,
                 .source_bits = state->provider->structural_copy_bits,
                 .target_bits = state->provider->structural_copy_bits,
                 .kind = LOOM_SOURCE_DATAFLOW_RULE_COPY,
             });
}

static iree_status_t loom_source_dataflow_build_graph(
    loom_source_dataflow_build_state_t* state) {
  const loom_source_program_t* program = state->environment->program;
  if (state->provider->seed_value.fn != NULL) {
    for (loom_value_ordinal_t i = 0; i < program->value_domain->value_count;
         ++i) {
      loom_source_dataflow_bits_t bits = 0;
      IREE_RETURN_IF_ERROR(state->provider->seed_value.fn(
          state->provider->seed_value.user_data, state->environment,
          program->value_domain->value_ids[i], &bits));
      ++state->result->statistics.value_seed_invocation_count;
      if ((bits & ~state->provider->valid_bits) != 0) {
        return loom_source_dataflow_invalid_provider(
            state->provider, "direct value seed produced unknown bits");
      }
      state->result->states[i] |= bits;
    }
  }

  if (state->provider->structural_copy_bits != 0) {
    for (uint32_t i = 0; i < program->value_relation_count; ++i) {
      const loom_source_program_value_relation_t relation =
          program->value_relations[i];
      IREE_RETURN_IF_ERROR(loom_source_dataflow_build_structural_copy(
          state, relation.lhs, relation.rhs));
      IREE_RETURN_IF_ERROR(loom_source_dataflow_build_structural_copy(
          state, relation.rhs, relation.lhs));
    }
  }

  for (loom_source_program_node_ordinal_t i = 0; i < program->node_count; ++i) {
    const loom_source_program_node_t* node = &program->nodes[i];
    if (node->kind != LOOM_SOURCE_PROGRAM_NODE_OPERATION) continue;
    const loom_op_t* op = loom_source_program_node_operation(node);
    const loom_source_dataflow_operation_t* operation =
        loom_source_dataflow_lookup_operation(state->provider, op->kind);
    if (operation == NULL) continue;
    IREE_RETURN_IF_ERROR(
        loom_source_dataflow_build_operation(state, op, operation));
  }
  return iree_ok_status();
}

typedef struct loom_source_dataflow_worklist_t {
  // Circular queue of pending runtime rule indices.
  uint32_t* entries;
  // Number of allocated queue entries.
  uint32_t capacity;
  // First pending queue position.
  uint32_t head;
  // Position receiving the next enqueued rule.
  uint32_t tail;
  // Number of pending queue entries.
  uint32_t count;
  // One byte per runtime rule indicating queue membership.
  uint8_t* pending;
} loom_source_dataflow_worklist_t;

static void loom_source_dataflow_worklist_push(
    loom_source_dataflow_worklist_t* worklist, uint32_t rule_index) {
  if (worklist->pending[rule_index]) return;
  IREE_ASSERT_LT(worklist->count, worklist->capacity);
  worklist->entries[worklist->tail] = rule_index;
  worklist->tail = (worklist->tail + 1) % worklist->capacity;
  ++worklist->count;
  worklist->pending[rule_index] = 1;
}

static uint32_t loom_source_dataflow_worklist_pop(
    loom_source_dataflow_worklist_t* worklist) {
  IREE_ASSERT_GT(worklist->count, 0);
  const uint32_t rule_index = worklist->entries[worklist->head];
  worklist->head = (worklist->head + 1) % worklist->capacity;
  --worklist->count;
  worklist->pending[rule_index] = 0;
  return rule_index;
}

static bool loom_source_dataflow_rule_matches(
    const loom_source_dataflow_build_state_t* state,
    const loom_source_dataflow_runtime_rule_t* rule) {
  const loom_source_dataflow_bits_t* states = state->result->states;
  if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_COPY) {
    return true;
  }
  if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_ANY) {
    for (uint32_t i = 0; i < rule->source_count; ++i) {
      if ((states[state->sources[rule->source_start + i]] &
           rule->source_bits) == rule->source_bits) {
        return true;
      }
    }
    return false;
  }
  IREE_ASSERT_EQ(rule->kind, LOOM_SOURCE_DATAFLOW_RULE_ALL);
  for (uint32_t i = 0; i < rule->source_count; ++i) {
    if ((states[state->sources[rule->source_start + i]] & rule->source_bits) !=
        rule->source_bits) {
      return false;
    }
  }
  return true;
}

static void loom_source_dataflow_evaluate_rule(
    const loom_source_dataflow_build_state_t* state,
    const loom_source_dataflow_runtime_rule_t* rule,
    const uint32_t* watch_offsets, const uint32_t* watch_rules,
    loom_source_dataflow_worklist_t* worklist) {
  if (!loom_source_dataflow_rule_matches(state, rule)) return;
  loom_source_dataflow_bits_t bits = rule->target_bits;
  if (rule->kind == LOOM_SOURCE_DATAFLOW_RULE_COPY) {
    bits = state->result->states[state->sources[rule->source_start]] &
           rule->source_bits;
  }
  if (bits == 0) return;
  for (uint32_t i = 0; i < rule->target_count; ++i) {
    const loom_value_ordinal_t target = state->targets[rule->target_start + i];
    loom_source_dataflow_bits_t* target_state = &state->result->states[target];
    const loom_source_dataflow_bits_t added_bits = bits & ~*target_state;
    if (added_bits == 0) continue;
    *target_state |= added_bits;
    for (uint32_t watch_index = watch_offsets[target];
         watch_index < watch_offsets[target + 1]; ++watch_index) {
      loom_source_dataflow_worklist_push(worklist, watch_rules[watch_index]);
    }
  }
}

static iree_status_t loom_source_dataflow_run(
    loom_source_dataflow_build_state_t* state) {
  if (state->rule_count == 0) return iree_ok_status();
  const loom_value_ordinal_t value_count = state->result->state_count;
  uint32_t* watch_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, (iree_host_size_t)value_count + 1, sizeof(*watch_offsets),
      (void**)&watch_offsets));
  memset(watch_offsets, 0,
         ((iree_host_size_t)value_count + 1) * sizeof(*watch_offsets));
  uint64_t watch_count = 0;
  for (uint32_t i = 0; i < state->rule_count; ++i) {
    const loom_source_dataflow_runtime_rule_t* rule = &state->rules[i];
    watch_count += rule->source_count;
    if (watch_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source dataflow graph exceeds watcher range");
    }
    for (uint32_t j = 0; j < rule->source_count; ++j) {
      ++watch_offsets[state->sources[rule->source_start + j] + 1];
    }
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    watch_offsets[i + 1] += watch_offsets[i];
  }
  uint32_t* watch_rules = NULL;
  if (watch_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, (iree_host_size_t)watch_count,
                                  sizeof(*watch_rules), (void**)&watch_rules));
  }
  uint32_t* watch_cursors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, value_count,
                                                 sizeof(*watch_cursors),
                                                 (void**)&watch_cursors));
  memcpy(watch_cursors, watch_offsets,
         (iree_host_size_t)value_count * sizeof(*watch_cursors));
  for (uint32_t i = 0; i < state->rule_count; ++i) {
    const loom_source_dataflow_runtime_rule_t* rule = &state->rules[i];
    for (uint32_t j = 0; j < rule->source_count; ++j) {
      const loom_value_ordinal_t source =
          state->sources[rule->source_start + j];
      watch_rules[watch_cursors[source]++] = i;
    }
  }

  loom_source_dataflow_worklist_t worklist = {
      .capacity = state->rule_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, worklist.capacity, sizeof(*worklist.entries),
      (void**)&worklist.entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, worklist.capacity, sizeof(*worklist.pending),
      (void**)&worklist.pending));
  memset(worklist.pending, 0, worklist.capacity * sizeof(*worklist.pending));
  for (uint32_t i = 0; i < state->rule_count; ++i) {
    loom_source_dataflow_worklist_push(&worklist, i);
  }
  while (worklist.count != 0) {
    const uint32_t rule_index = loom_source_dataflow_worklist_pop(&worklist);
    ++state->result->statistics.rule_evaluation_count;
    loom_source_dataflow_evaluate_rule(state, &state->rules[rule_index],
                                       watch_offsets, watch_rules, &worklist);
  }
  return iree_ok_status();
}

iree_status_t loom_source_dataflow_solve(
    const loom_source_dataflow_provider_t* provider,
    const loom_source_dataflow_environment_t* environment,
    iree_arena_allocator_t* arena, loom_source_dataflow_result_t* out_result) {
  if (out_result != NULL) {
    *out_result = (loom_source_dataflow_result_t){0};
  }
  if (provider == NULL || environment == NULL || environment->program == NULL ||
      arena == NULL || out_result == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source dataflow solve requires all arguments");
  }
  const loom_source_program_t* program = environment->program;
  if (program->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(program->value_domain)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source dataflow program has no acquired value domain");
  }
  *out_result = (loom_source_dataflow_result_t){
      .provider = provider,
      .value_domain = program->value_domain,
      .state_count = program->value_domain->value_count,
  };
  if (out_result->state_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_result->state_count, sizeof(*out_result->states),
        (void**)&out_result->states));
    memset(out_result->states, 0,
           (iree_host_size_t)out_result->state_count *
               sizeof(*out_result->states));
  }

  loom_source_dataflow_build_state_t state = {
      .provider = provider,
      .environment = environment,
      .arena = arena,
      .result = out_result,
  };
  iree_status_t status = loom_source_dataflow_build_graph(&state);
  if (iree_status_is_ok(status)) {
    status = loom_source_dataflow_run(&state);
  }
  if (!iree_status_is_ok(status)) {
    *out_result = (loom_source_dataflow_result_t){0};
  }
  return status;
}

loom_source_dataflow_bits_t loom_source_dataflow_result_lookup(
    const loom_source_dataflow_result_t* result, loom_value_id_t value_id) {
  if (result == NULL || result->value_domain == NULL ||
      result->states == NULL) {
    return 0;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(result->value_domain, value_id);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID ? 0 : result->states[ordinal];
}
