// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_representation.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"

// Candidate eligibility evaluated once for one source operation occurrence.
enum loom_low_source_representation_match_e {
  LOOM_LOW_SOURCE_REPRESENTATION_MATCH_REJECTED = 0,
  LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE = 1,
};
typedef uint8_t loom_low_source_representation_match_t;

typedef struct loom_low_source_representation_domain_state_t {
  // Sorted provider-local representation indices in the current intersection.
  uint16_t* representation_indices;
  // Number of entries in representation_indices.
  uint16_t representation_count;
  // Unique canonical representation in every intersected constraint.
  uint16_t canonical_representation_index;
  // Final selected representation, or NONE before selection.
  uint16_t selected_representation_index;
  // True after the first finite-domain constraint is applied.
  bool constrained;
} loom_low_source_representation_domain_state_t;

typedef struct loom_low_source_representation_runtime_group_t {
  // Source operation carrying this candidate group occurrence.
  const loom_op_t* source_op;
  // Source-program operation node carrying source_op.
  loom_source_program_node_ordinal_t source_node;
  // Provider-local candidate group index.
  uint16_t group_index;
  // First candidate eligibility byte in the runtime match pool.
  uint32_t candidate_match_start;
  // First group-local component value ordinal in the runtime value pool.
  uint32_t component_value_start;
  // Sole component root varied by candidates, or INVALID when all are fixed.
  loom_value_ordinal_t varying_component_root;
  // Next runtime group varying the same component, or UINT32_MAX.
  uint32_t next_component_group;
  // True when an optional group had no capability-matching candidates.
  bool skipped;
} loom_low_source_representation_runtime_group_t;

typedef struct loom_low_source_representation_state_t {
  const loom_low_source_representation_provider_t* provider;
  const loom_source_program_t* program;
  const loom_low_source_representation_environment_t* environment;
  iree_arena_allocator_t* arena;
  loom_low_source_representation_plan_t* plan;

  loom_value_ordinal_t* component_parents;
  uint8_t* component_ranks;
  loom_low_source_representation_domain_state_t* component_domains;
  uint32_t* component_group_heads;
  uint32_t* component_group_tails;

  loom_low_source_representation_runtime_group_t* runtime_groups;
  uint32_t runtime_group_count;
  iree_host_size_t runtime_group_capacity;
  uint32_t planned_operation_count;

  loom_low_source_representation_match_t* candidate_matches;
  uint32_t candidate_match_count;
  iree_host_size_t candidate_match_capacity;

  loom_value_ordinal_t* component_values;
  uint32_t component_value_count;
  iree_host_size_t component_value_capacity;
} loom_low_source_representation_state_t;

static iree_status_t loom_low_source_representation_invalid_provider(
    const loom_low_source_representation_provider_t* provider,
    const char* detail) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "source representation provider '%.*s' %s",
      (int)provider->name.size, provider->name.data, detail);
}

static const loom_low_source_representation_operation_t*
loom_low_source_representation_lookup_operation(
    const loom_low_source_representation_provider_t* provider,
    loom_op_kind_t kind) {
  const uint8_t dialect_id = loom_op_dialect_id(kind);
  if (dialect_id < provider->dialect_base_id) return NULL;
  const uint8_t dialect_index = dialect_id - provider->dialect_base_id;
  if (dialect_index >= provider->dialect_count) return NULL;
  const loom_low_source_representation_dialect_table_t* dialect =
      &provider->dialects[dialect_index];
  const uint8_t operation_ordinal = loom_op_dialect_index(kind);
  if (operation_ordinal >= dialect->operation_count) return NULL;
  const uint16_t operation_index =
      dialect->operation_indices[operation_ordinal];
  return operation_index == LOOM_LOW_SOURCE_REPRESENTATION_OPERATION_INDEX_NONE
             ? NULL
             : &provider->operations[operation_index - 1];
}

static loom_value_ordinal_t loom_low_source_representation_find_component(
    loom_low_source_representation_state_t* state, loom_value_ordinal_t value) {
  loom_value_ordinal_t root = value;
  while (state->component_parents[root] != root) {
    root = state->component_parents[root];
  }
  while (state->component_parents[value] != value) {
    const loom_value_ordinal_t parent = state->component_parents[value];
    state->component_parents[value] = root;
    value = parent;
  }
  return root;
}

static void loom_low_source_representation_union_components(
    loom_low_source_representation_state_t* state, loom_value_ordinal_t left,
    loom_value_ordinal_t right) {
  left = loom_low_source_representation_find_component(state, left);
  right = loom_low_source_representation_find_component(state, right);
  if (left == right) return;
  if (state->component_ranks[left] < state->component_ranks[right]) {
    state->component_parents[left] = right;
  } else if (state->component_ranks[left] > state->component_ranks[right]) {
    state->component_parents[right] = left;
  } else {
    state->component_parents[right] = left;
    ++state->component_ranks[left];
  }
}

static iree_status_t loom_low_source_representation_append_runtime_group(
    loom_low_source_representation_state_t* state,
    loom_low_source_representation_runtime_group_t group) {
  if (state->runtime_group_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source representation group count overflows");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)state->runtime_group_count + 1;
  if (minimum_capacity > state->runtime_group_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->runtime_group_count, minimum_capacity,
        sizeof(*state->runtime_groups), &state->runtime_group_capacity,
        (void**)&state->runtime_groups));
  }
  state->runtime_groups[state->runtime_group_count++] = group;
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_append_candidate_match(
    loom_low_source_representation_state_t* state,
    loom_low_source_representation_match_t match) {
  if (state->candidate_match_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source representation candidate count overflows");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)state->candidate_match_count + 1;
  if (minimum_capacity > state->candidate_match_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->candidate_match_count, minimum_capacity,
        sizeof(*state->candidate_matches), &state->candidate_match_capacity,
        (void**)&state->candidate_matches));
  }
  state->candidate_matches[state->candidate_match_count++] = match;
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_append_component_value(
    loom_low_source_representation_state_t* state, loom_value_ordinal_t value) {
  if (state->component_value_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source representation component count overflows");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)state->component_value_count + 1;
  if (minimum_capacity > state->component_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->component_value_count, minimum_capacity,
        sizeof(*state->component_values), &state->component_value_capacity,
        (void**)&state->component_values));
  }
  state->component_values[state->component_value_count++] = value;
  return iree_ok_status();
}

static void loom_low_source_representation_set_problem(
    loom_low_source_representation_state_t* state,
    loom_low_source_representation_problem_kind_t kind,
    const loom_op_t* source_op, loom_value_ordinal_t component_root,
    uint16_t group_index) {
  if (state->plan->problem.kind !=
      LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
    return;
  }
  loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
  const loom_local_value_domain_t* value_domain = state->program->value_domain;
  if (component_root != LOOM_VALUE_ORDINAL_INVALID &&
      component_root < value_domain->value_count) {
    source_value_id = value_domain->value_ids[component_root];
  }
  state->plan->problem = (loom_low_source_representation_problem_t){
      .kind = kind,
      .source_op = source_op,
      .source_value_id = source_value_id,
      .group_index = group_index,
  };
}

static iree_status_t loom_low_source_representation_candidate_matches(
    loom_low_source_representation_state_t* state, const loom_op_t* source_op,
    const loom_low_source_representation_candidate_t* candidate,
    uint8_t* predicate_states, bool* out_matches) {
  *out_matches = true;
  if (candidate->predicate_index_plus_one == 0) return iree_ok_status();
  const uint8_t predicate_index = candidate->predicate_index_plus_one - 1;
  uint8_t* predicate_state = &predicate_states[predicate_index];
  if (*predicate_state == 0) {
    bool matches = false;
    const loom_low_source_representation_predicate_t predicate =
        state->provider->predicates[predicate_index];
    IREE_RETURN_IF_ERROR(predicate.fn(predicate.user_data, state->environment,
                                      source_op, &matches));
    *predicate_state = matches ? 2 : 1;
    ++state->plan->statistics.predicate_invocation_count;
  }
  *out_matches = *predicate_state == 2;
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_resolve_port(
    loom_low_source_representation_state_t* state, const loom_op_t* source_op,
    const loom_op_vtable_t* vtable,
    const loom_low_source_representation_port_t* port,
    loom_value_ordinal_t* component_values) {
  loom_value_slice_t values = {0};
  if (port->kind == LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD) {
    if (port->field_index >= loom_op_vtable_operand_descriptor_count(vtable)) {
      return loom_low_source_representation_invalid_provider(
          state->provider, "references an absent operand field");
    }
    values = loom_op_operand_field_span(vtable, source_op, port->field_index);
  } else {
    const uint8_t result_field_count =
        (uint8_t)(vtable->fixed_result_count +
                  (iree_any_bit_set(vtable->vtable_flags,
                                    LOOM_OP_VTABLE_VARIADIC_RESULTS)
                       ? 1
                       : 0));
    if (port->field_index >= result_field_count) {
      return loom_low_source_representation_invalid_provider(
          state->provider, "references an absent result field");
    }
    values = loom_op_result_field_span(vtable, source_op, port->field_index);
  }
  uint16_t element_start = 0;
  uint16_t element_count = values.count;
  if (port->element_index != LOOM_LOW_SOURCE_REPRESENTATION_PORT_ALL_ELEMENTS) {
    if (port->element_index >= values.count) {
      return loom_low_source_representation_invalid_provider(
          state->provider, "references an absent source field element");
    }
    element_start = port->element_index;
    element_count = 1;
  }
  const loom_local_value_domain_t* value_domain = state->program->value_domain;
  for (uint16_t i = 0; i < element_count; ++i) {
    const loom_value_id_t value_id = values.values[element_start + i];
    const loom_value_ordinal_t value =
        loom_local_value_domain_try_ordinal(value_domain, value_id);
    if (value == LOOM_VALUE_ORDINAL_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source representation port escapes the indexed value domain");
    }
    loom_value_ordinal_t* component_value =
        &component_values[port->component_index];
    if (*component_value == LOOM_VALUE_ORDINAL_INVALID) {
      *component_value = value;
    } else {
      loom_low_source_representation_union_components(state, *component_value,
                                                      value);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_build_group(
    loom_low_source_representation_state_t* state, const loom_op_t* source_op,
    loom_source_program_node_ordinal_t source_node, uint16_t group_index,
    uint8_t* predicate_states) {
  const loom_low_source_representation_group_t* group =
      &state->provider->groups[group_index];
  const uint32_t candidate_match_start = state->candidate_match_count;
  uint8_t eligible_count = 0;
  for (uint8_t i = 0; i < group->candidate_count; ++i) {
    const loom_low_source_representation_candidate_t* candidate =
        &state->provider->candidates[group->candidate_start + i];
    bool matches = false;
    IREE_RETURN_IF_ERROR(loom_low_source_representation_candidate_matches(
        state, source_op, candidate, predicate_states, &matches));
    IREE_RETURN_IF_ERROR(loom_low_source_representation_append_candidate_match(
        state, matches ? LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE
                       : LOOM_LOW_SOURCE_REPRESENTATION_MATCH_REJECTED));
    eligible_count += matches;
  }

  const uint32_t component_value_start = state->component_value_count;
  for (uint8_t i = 0; i < group->component_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_source_representation_append_component_value(
        state, LOOM_VALUE_ORDINAL_INVALID));
  }
  loom_low_source_representation_runtime_group_t runtime_group = {
      .source_op = source_op,
      .source_node = source_node,
      .group_index = group_index,
      .candidate_match_start = candidate_match_start,
      .component_value_start = component_value_start,
      .varying_component_root = LOOM_VALUE_ORDINAL_INVALID,
      .next_component_group = UINT32_MAX,
  };
  if (eligible_count == 0) {
    if (iree_any_bit_set(group->flags,
                         LOOM_LOW_SOURCE_REPRESENTATION_GROUP_OPTIONAL)) {
      runtime_group.skipped = true;
      return loom_low_source_representation_append_runtime_group(state,
                                                                 runtime_group);
    }
    loom_low_source_representation_set_problem(
        state, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_UNAVAILABLE_GROUP,
        source_op, LOOM_VALUE_ORDINAL_INVALID, group_index);
    return loom_low_source_representation_append_runtime_group(state,
                                                               runtime_group);
  }

  const loom_op_vtable_t* vtable =
      loom_op_vtable(state->program->module, source_op);
  if (vtable == NULL) {
    return loom_low_source_representation_invalid_provider(
        state->provider, "references an unregistered source operation");
  }
  loom_value_ordinal_t* component_values =
      &state->component_values[component_value_start];
  for (uint8_t i = 0; i < group->port_count; ++i) {
    const loom_low_source_representation_port_t* port =
        &state->provider->ports[group->port_start + i];
    IREE_RETURN_IF_ERROR(loom_low_source_representation_resolve_port(
        state, source_op, vtable, port, component_values));
  }
  for (uint8_t i = 0; i < group->component_count; ++i) {
    if (component_values[i] == LOOM_VALUE_ORDINAL_INVALID) {
      return loom_low_source_representation_invalid_provider(
          state->provider,
          "resolves a candidate component without source values");
    }
  }
  return loom_low_source_representation_append_runtime_group(state,
                                                             runtime_group);
}

static iree_status_t loom_low_source_representation_build_groups(
    loom_low_source_representation_state_t* state) {
  const loom_source_program_t* program = state->program;
  for (loom_source_program_node_ordinal_t node_index = 0;
       node_index < program->node_count; ++node_index) {
    const loom_source_program_node_t* node = &program->nodes[node_index];
    if (node->kind != LOOM_SOURCE_PROGRAM_NODE_OPERATION) continue;
    const loom_op_t* source_op = loom_source_program_node_operation(node);
    const loom_low_source_representation_operation_t* operation =
        loom_low_source_representation_lookup_operation(state->provider,
                                                        source_op->kind);
    if (operation == NULL) continue;
    const uint32_t group_start = state->runtime_group_count;
    uint8_t predicate_states[UINT8_MAX + 1] = {0};
    for (uint8_t i = 0; i < operation->group_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_source_representation_build_group(
          state, source_op, node_index, (uint16_t)(operation->group_start + i),
          predicate_states));
      if (state->plan->problem.kind !=
          LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
        break;
      }
    }
    const uint32_t group_count = state->runtime_group_count - group_start;
    IREE_ASSERT_LE(group_count, UINT8_MAX);
    state->plan->node_selections[node_index] =
        (loom_low_source_representation_node_selection_t){
            .group_start = group_start,
            .group_count = (uint8_t)group_count,
        };
    state->planned_operation_count += group_count != 0;
    state->plan->statistics.candidate_group_count += group_count;
    if (state->plan->problem.kind !=
        LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
      break;
    }
  }
  return iree_ok_status();
}

static void loom_low_source_representation_sort_indices(uint16_t* values,
                                                        uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    const uint16_t value = values[i];
    uint16_t j = i;
    while (j != 0 && values[j - 1] > value) {
      values[j] = values[j - 1];
      --j;
    }
    values[j] = value;
  }
}

static iree_status_t loom_low_source_representation_apply_domain(
    loom_low_source_representation_state_t* state,
    loom_value_ordinal_t component_root, const uint16_t* representation_indices,
    uint16_t representation_count, uint16_t canonical_representation_index,
    const loom_op_t* source_op, uint16_t group_index) {
  if (representation_count == 0) {
    loom_low_source_representation_set_problem(
        state, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_EMPTY_DOMAIN, source_op,
        component_root, group_index);
    return iree_ok_status();
  }
  if (representation_indices == NULL) {
    return loom_low_source_representation_invalid_provider(
        state->provider, "returns a domain without representation rows");
  }
  bool contains_canonical = false;
  for (uint16_t i = 0; i < representation_count; ++i) {
    if (representation_indices[i] >= state->provider->representation_count) {
      return loom_low_source_representation_invalid_provider(
          state->provider,
          "returns a domain with an absent representation index");
    }
    if (i != 0 && representation_indices[i - 1] >= representation_indices[i]) {
      return loom_low_source_representation_invalid_provider(
          state->provider,
          "returns a domain whose representations are not strictly ordered");
    }
    contains_canonical |=
        representation_indices[i] == canonical_representation_index;
  }
  if (!contains_canonical) {
    return loom_low_source_representation_invalid_provider(
        state->provider,
        "returns a domain without its declared canonical representation");
  }

  loom_low_source_representation_domain_state_t* domain =
      &state->component_domains[component_root];
  if (!domain->constrained) {
    uint16_t* retained_indices = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, representation_count, sizeof(*retained_indices),
        (void**)&retained_indices));
    memcpy(retained_indices, representation_indices,
           representation_count * sizeof(*retained_indices));
    *domain = (loom_low_source_representation_domain_state_t){
        .representation_indices = retained_indices,
        .representation_count = representation_count,
        .canonical_representation_index = canonical_representation_index,
        .selected_representation_index =
            LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE,
        .constrained = true,
    };
    return iree_ok_status();
  }
  if (domain->canonical_representation_index !=
      canonical_representation_index) {
    return loom_low_source_representation_invalid_provider(
        state->provider,
        "declares inconsistent canonical representations for one component");
  }

  uint16_t left = 0;
  uint16_t right = 0;
  uint16_t result_count = 0;
  while (left < domain->representation_count && right < representation_count) {
    const uint16_t left_value = domain->representation_indices[left];
    const uint16_t right_value = representation_indices[right];
    if (left_value < right_value) {
      ++left;
    } else if (right_value < left_value) {
      ++right;
    } else {
      domain->representation_indices[result_count++] = left_value;
      ++left;
      ++right;
    }
  }
  domain->representation_count = result_count;
  if (result_count == 0) {
    loom_low_source_representation_set_problem(
        state, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_EMPTY_DOMAIN, source_op,
        component_root, group_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_apply_value_domains(
    loom_low_source_representation_state_t* state) {
  if (state->provider->seed_value.fn == NULL) return iree_ok_status();
  const loom_local_value_domain_t* value_domain = state->program->value_domain;
  for (loom_value_ordinal_t value = 0; value < value_domain->value_count;
       ++value) {
    loom_low_source_representation_domain_t domain = {
        .canonical_representation_index =
            LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE,
    };
    IREE_RETURN_IF_ERROR(state->provider->seed_value.fn(
        state->provider->seed_value.user_data, state->environment,
        value_domain->value_ids[value], &domain));
    ++state->plan->statistics.value_seed_invocation_count;
    if (!domain.constrained) {
      if (domain.representation_indices != NULL ||
          domain.representation_count != 0 ||
          domain.canonical_representation_index !=
              LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE) {
        return loom_low_source_representation_invalid_provider(
            state->provider,
            "returns rows for an unconstrained source value domain");
      }
      continue;
    }
    const loom_value_ordinal_t component_root =
        loom_low_source_representation_find_component(state, value);
    IREE_RETURN_IF_ERROR(loom_low_source_representation_apply_domain(
        state, component_root, domain.representation_indices,
        domain.representation_count, domain.canonical_representation_index,
        /*source_op=*/NULL, UINT16_MAX));
    if (state->plan->problem.kind !=
        LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_source_representation_candidate_bindings_match(
    const loom_low_source_representation_state_t* state,
    const loom_low_source_representation_runtime_group_t* runtime_group,
    uint8_t left_candidate_ordinal, uint8_t right_candidate_ordinal) {
  const loom_low_source_representation_group_t* group =
      &state->provider->groups[runtime_group->group_index];
  const loom_low_source_representation_candidate_t* left =
      &state->provider
           ->candidates[group->candidate_start + left_candidate_ordinal];
  const loom_low_source_representation_candidate_t* right =
      &state->provider
           ->candidates[group->candidate_start + right_candidate_ordinal];
  for (uint8_t i = 0; i < group->component_count; ++i) {
    if (state->provider->bindings[left->binding_start + i]
            .representation_index !=
        state->provider->bindings[right->binding_start + i]
            .representation_index) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_source_representation_analyze_group(
    loom_low_source_representation_state_t* state,
    uint32_t runtime_group_index) {
  loom_low_source_representation_runtime_group_t* runtime_group =
      &state->runtime_groups[runtime_group_index];
  if (runtime_group->skipped) return iree_ok_status();
  const loom_low_source_representation_group_t* group =
      &state->provider->groups[runtime_group->group_index];
  loom_low_source_representation_match_t* candidate_matches =
      &state->candidate_matches[runtime_group->candidate_match_start];
  loom_value_ordinal_t* component_values =
      &state->component_values[runtime_group->component_value_start];
  loom_value_ordinal_t
      component_roots[LOOM_LOW_SOURCE_REPRESENTATION_MAX_COMPONENT_COUNT];
  for (uint8_t i = 0; i < group->component_count; ++i) {
    component_roots[i] = loom_low_source_representation_find_component(
        state, component_values[i]);
  }

  uint8_t eligible_count = 0;
  for (uint8_t candidate_ordinal = 0;
       candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
    if (candidate_matches[candidate_ordinal] !=
        LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE) {
      continue;
    }
    const loom_low_source_representation_candidate_t* candidate =
        &state->provider
             ->candidates[group->candidate_start + candidate_ordinal];
    bool compatible = true;
    for (uint8_t left = 0; left < group->component_count && compatible;
         ++left) {
      const loom_low_source_representation_binding_t left_binding =
          state->provider->bindings[candidate->binding_start + left];
      for (uint8_t right = left + 1; right < group->component_count; ++right) {
        if (component_roots[left] != component_roots[right]) continue;
        const loom_low_source_representation_binding_t right_binding =
            state->provider->bindings[candidate->binding_start + right];
        if (left_binding.representation_index !=
                right_binding.representation_index ||
            left_binding.flags != right_binding.flags) {
          compatible = false;
          break;
        }
      }
    }
    if (!compatible) {
      candidate_matches[candidate_ordinal] =
          LOOM_LOW_SOURCE_REPRESENTATION_MATCH_REJECTED;
      continue;
    }
    for (uint8_t prior = 0; prior < candidate_ordinal; ++prior) {
      if (candidate_matches[prior] ==
              LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE &&
          loom_low_source_representation_candidate_bindings_match(
              state, runtime_group, prior, candidate_ordinal)) {
        return loom_low_source_representation_invalid_provider(
            state->provider,
            "produces duplicate eligible representation tuples");
      }
    }
    ++eligible_count;
  }
  if (eligible_count == 0) {
    if (iree_any_bit_set(group->flags,
                         LOOM_LOW_SOURCE_REPRESENTATION_GROUP_OPTIONAL)) {
      runtime_group->skipped = true;
      return iree_ok_status();
    }
    loom_low_source_representation_set_problem(
        state, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_UNAVAILABLE_GROUP,
        runtime_group->source_op, component_roots[0],
        runtime_group->group_index);
    return iree_ok_status();
  }

  loom_value_ordinal_t
      distinct_roots[LOOM_LOW_SOURCE_REPRESENTATION_MAX_COMPONENT_COUNT];
  uint8_t distinct_root_count = 0;
  for (uint8_t i = 0; i < group->component_count; ++i) {
    bool present = false;
    for (uint8_t j = 0; j < distinct_root_count; ++j) {
      present |= distinct_roots[j] == component_roots[i];
    }
    if (!present) distinct_roots[distinct_root_count++] = component_roots[i];
  }

  uint8_t varying_component_count = 0;
  for (uint8_t root_ordinal = 0; root_ordinal < distinct_root_count;
       ++root_ordinal) {
    const loom_value_ordinal_t component_root = distinct_roots[root_ordinal];
    uint16_t representation_indices[UINT8_MAX] = {0};
    bool canonical_flags[UINT8_MAX] = {false};
    uint8_t representation_count = 0;
    for (uint8_t candidate_ordinal = 0;
         candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
      if (candidate_matches[candidate_ordinal] !=
          LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE) {
        continue;
      }
      const loom_low_source_representation_candidate_t* candidate =
          &state->provider
               ->candidates[group->candidate_start + candidate_ordinal];
      for (uint8_t slot = 0; slot < group->component_count; ++slot) {
        if (component_roots[slot] != component_root) continue;
        const loom_low_source_representation_binding_t binding =
            state->provider->bindings[candidate->binding_start + slot];
        const bool is_canonical = iree_any_bit_set(
            binding.flags, LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL);
        uint8_t representation_ordinal = 0;
        for (; representation_ordinal < representation_count;
             ++representation_ordinal) {
          if (representation_indices[representation_ordinal] ==
              binding.representation_index) {
            if (canonical_flags[representation_ordinal] != is_canonical) {
              return loom_low_source_representation_invalid_provider(
                  state->provider,
                  "classifies one eligible representation inconsistently as "
                  "canonical");
            }
            break;
          }
        }
        if (representation_ordinal == representation_count) {
          representation_indices[representation_count] =
              binding.representation_index;
          canonical_flags[representation_count] = is_canonical;
          ++representation_count;
        }
      }
    }
    uint16_t canonical_representation_index =
        LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE;
    for (uint8_t i = 0; i < representation_count; ++i) {
      if (!canonical_flags[i]) continue;
      if (canonical_representation_index !=
              LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE &&
          canonical_representation_index != representation_indices[i]) {
        return loom_low_source_representation_invalid_provider(
            state->provider,
            "has multiple eligible canonical representations in a domain");
      }
      canonical_representation_index = representation_indices[i];
    }
    if (canonical_representation_index ==
        LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE) {
      return loom_low_source_representation_invalid_provider(
          state->provider,
          "has no eligible canonical representation in a domain");
    }
    loom_low_source_representation_sort_indices(representation_indices,
                                                representation_count);
    IREE_RETURN_IF_ERROR(loom_low_source_representation_apply_domain(
        state, component_root, representation_indices, representation_count,
        canonical_representation_index, runtime_group->source_op,
        runtime_group->group_index));
    if (state->plan->problem.kind !=
        LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
      return iree_ok_status();
    }
    if (representation_count > 1) {
      runtime_group->varying_component_root = component_root;
      ++varying_component_count;
    }
  }
  if (varying_component_count > 1) {
    return loom_low_source_representation_invalid_provider(
        state->provider,
        "couples alternatives across multiple independent components");
  }
  if (varying_component_count == 1) {
    const loom_value_ordinal_t component_root =
        runtime_group->varying_component_root;
    uint32_t* component_group_head =
        &state->component_group_heads[component_root];
    uint32_t* component_group_tail =
        &state->component_group_tails[component_root];
    if (*component_group_head == UINT32_MAX) {
      *component_group_head = runtime_group_index;
    } else {
      state->runtime_groups[*component_group_tail].next_component_group =
          runtime_group_index;
    }
    *component_group_tail = runtime_group_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_analyze_groups(
    loom_low_source_representation_state_t* state) {
  for (uint32_t i = 0; i < state->runtime_group_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_source_representation_analyze_group(state, i));
    if (state->plan->problem.kind !=
        LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static uint16_t loom_low_source_representation_group_candidate_for(
    const loom_low_source_representation_state_t* state,
    const loom_low_source_representation_runtime_group_t* runtime_group,
    loom_value_ordinal_t component_root, uint16_t representation_index) {
  const loom_low_source_representation_group_t* group =
      &state->provider->groups[runtime_group->group_index];
  const loom_low_source_representation_match_t* candidate_matches =
      &state->candidate_matches[runtime_group->candidate_match_start];
  const loom_value_ordinal_t* component_values =
      &state->component_values[runtime_group->component_value_start];
  for (uint8_t candidate_ordinal = 0;
       candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
    if (candidate_matches[candidate_ordinal] !=
        LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE) {
      continue;
    }
    const uint16_t candidate_index =
        (uint16_t)(group->candidate_start + candidate_ordinal);
    const loom_low_source_representation_candidate_t* candidate =
        &state->provider->candidates[candidate_index];
    for (uint8_t slot = 0; slot < group->component_count; ++slot) {
      if (state->component_parents[component_values[slot]] != component_root) {
        continue;
      }
      if (state->provider->bindings[candidate->binding_start + slot]
              .representation_index == representation_index) {
        return candidate_index;
      }
      break;
    }
  }
  return LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE;
}

static iree_host_size_t loom_low_source_representation_component_recipe_count(
    const loom_low_source_representation_state_t* state,
    loom_value_ordinal_t component_root) {
  iree_host_size_t recipe_count = 0;
  for (uint32_t i = state->component_group_heads[component_root];
       i != UINT32_MAX; i = state->runtime_groups[i].next_component_group) {
    ++recipe_count;
  }
  return recipe_count;
}

static iree_status_t loom_low_source_representation_build_component_recipes(
    const loom_low_source_representation_state_t* state,
    loom_value_ordinal_t component_root, uint16_t representation_index,
    iree_host_size_t recipe_capacity,
    loom_low_descriptor_recipe_t* out_recipes) {
  iree_host_size_t recipe_count = 0;
  for (uint32_t i = state->component_group_heads[component_root];
       i != UINT32_MAX; i = state->runtime_groups[i].next_component_group) {
    const loom_low_source_representation_runtime_group_t* runtime_group =
        &state->runtime_groups[i];
    const uint16_t candidate_index =
        loom_low_source_representation_group_candidate_for(
            state, runtime_group, component_root, representation_index);
    if (candidate_index ==
        LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source representation domain lacks a matching realization");
    }
    const loom_low_source_representation_candidate_t* candidate =
        &state->provider->candidates[candidate_index];
    IREE_ASSERT_LT(recipe_count, recipe_capacity);
    out_recipes[recipe_count++] = (loom_low_descriptor_recipe_t){
        .entries = candidate->recipe_entry_count == 0
                       ? NULL
                       : &state->provider
                              ->recipe_entries[candidate->recipe_entry_start],
        .entry_count = candidate->recipe_entry_count,
        .dependencies = candidate->recipe_dependency_count == 0
                            ? NULL
                            : &state->provider->recipe_dependencies
                                   [candidate->recipe_dependency_start],
        .dependency_count = candidate->recipe_dependency_count,
        .durable_pressure_deltas =
            candidate->durable_pressure_delta_count == 0
                ? NULL
                : &state->provider->durable_pressure_deltas
                       [candidate->durable_pressure_delta_start],
        .durable_pressure_delta_count = candidate->durable_pressure_delta_count,
    };
  }
  IREE_ASSERT_EQ(recipe_count, recipe_capacity);
  return iree_ok_status();
}

static iree_status_t loom_low_source_representation_select_components(
    loom_low_source_representation_state_t* state) {
  const loom_value_ordinal_t value_count =
      state->program->value_domain->value_count;
  for (loom_value_ordinal_t component_root = 0; component_root < value_count;
       ++component_root) {
    loom_low_source_representation_domain_state_t* domain =
        &state->component_domains[component_root];
    if (!domain->constrained) continue;
    IREE_ASSERT_EQ(state->component_parents[component_root], component_root);
    loom_low_descriptor_cost_t* costs = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, domain->representation_count,
                                  sizeof(*costs), (void**)&costs));
    const iree_host_size_t recipe_count =
        loom_low_source_representation_component_recipe_count(state,
                                                              component_root);
    loom_low_descriptor_recipe_t* recipes = NULL;
    if (recipe_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, recipe_count, sizeof(*recipes), (void**)&recipes));
    }
    uint16_t selected_ordinal = 0;
    for (uint16_t i = 0; i < domain->representation_count; ++i) {
      const uint16_t representation_index = domain->representation_indices[i];
      IREE_RETURN_IF_ERROR(
          loom_low_source_representation_build_component_recipes(
              state, component_root, representation_index, recipe_count,
              recipes));
      IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_compute_independent(
          state->environment->descriptor_set, recipes, recipe_count,
          state->arena, &costs[i]));
      if (i == 0) continue;
      const uint16_t selected_representation_index =
          domain->representation_indices[selected_ordinal];
      const loom_low_descriptor_cost_candidate_t selected = {
          .cost = &costs[selected_ordinal],
          .stable_key =
              state->provider->representations[selected_representation_index]
                  .stable_key,
          .is_canonical = selected_representation_index ==
                          domain->canonical_representation_index,
      };
      const loom_low_descriptor_cost_candidate_t candidate = {
          .cost = &costs[i],
          .stable_key =
              state->provider->representations[representation_index].stable_key,
          .is_canonical =
              representation_index == domain->canonical_representation_index,
      };
      if (loom_low_descriptor_cost_compare(&candidate, &selected) ==
          LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT) {
        selected_ordinal = i;
      }
    }
    domain->selected_representation_index =
        domain->representation_indices[selected_ordinal];
    state->plan->component_costs[component_root] = &costs[selected_ordinal];
    ++state->plan->statistics.selected_component_count;
  }

  for (loom_value_ordinal_t value = 0; value < value_count; ++value) {
    const loom_value_ordinal_t component_root =
        loom_low_source_representation_find_component(state, value);
    state->plan->value_component_roots[value] = component_root;
    const loom_low_source_representation_domain_state_t* domain =
        &state->component_domains[component_root];
    if (domain->constrained) {
      state->plan->value_representation_indices[value] =
          domain->selected_representation_index;
    }
  }
  return iree_ok_status();
}

static bool loom_low_source_representation_candidate_is_selected(
    const loom_low_source_representation_state_t* state,
    const loom_low_source_representation_runtime_group_t* runtime_group,
    const loom_low_source_representation_candidate_t* candidate) {
  const loom_low_source_representation_group_t* group =
      &state->provider->groups[runtime_group->group_index];
  const loom_value_ordinal_t* component_values =
      &state->component_values[runtime_group->component_value_start];
  for (uint8_t slot = 0; slot < group->component_count; ++slot) {
    const loom_value_ordinal_t component_root =
        state->component_parents[component_values[slot]];
    const uint16_t selected_representation =
        state->component_domains[component_root].selected_representation_index;
    if (state->provider->bindings[candidate->binding_start + slot]
            .representation_index != selected_representation) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_source_representation_select_groups(
    loom_low_source_representation_state_t* state) {
  if (state->runtime_group_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, state->runtime_group_count,
                                  sizeof(*state->plan->selected_groups),
                                  (void**)&state->plan->selected_groups));
  }
  state->plan->selected_group_count = state->runtime_group_count;
  for (uint32_t i = 0; i < state->runtime_group_count; ++i) {
    const loom_low_source_representation_runtime_group_t* runtime_group =
        &state->runtime_groups[i];
    loom_low_source_representation_group_selection_t* selection =
        &state->plan->selected_groups[i];
    *selection = (loom_low_source_representation_group_selection_t){
        .group_index = runtime_group->group_index,
        .candidate_index = LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE,
    };
    if (runtime_group->skipped) continue;
    const loom_low_source_representation_group_t* group =
        &state->provider->groups[runtime_group->group_index];
    const loom_low_source_representation_match_t* candidate_matches =
        &state->candidate_matches[runtime_group->candidate_match_start];
    for (uint8_t candidate_ordinal = 0;
         candidate_ordinal < group->candidate_count; ++candidate_ordinal) {
      if (candidate_matches[candidate_ordinal] !=
          LOOM_LOW_SOURCE_REPRESENTATION_MATCH_ELIGIBLE) {
        continue;
      }
      const uint16_t candidate_index =
          (uint16_t)(group->candidate_start + candidate_ordinal);
      const loom_low_source_representation_candidate_t* candidate =
          &state->provider->candidates[candidate_index];
      if (!loom_low_source_representation_candidate_is_selected(
              state, runtime_group, candidate)) {
        continue;
      }
      if (selection->candidate_index !=
          LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE) {
        return loom_low_source_representation_invalid_provider(
            state->provider,
            "selects multiple realizations for one representation tuple");
      }
      selection->candidate_index = candidate_index;
    }
    if (selection->candidate_index ==
        LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source representation selection has no retained realization");
    }
  }
  return iree_ok_status();
}

static uint32_t loom_low_source_representation_hash_pointer(const void* ptr) {
  uint64_t value = (uint64_t)(uintptr_t)ptr;
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return (uint32_t)value;
}

static iree_status_t loom_low_source_representation_build_operation_lookup(
    loom_low_source_representation_state_t* state) {
  const loom_source_program_t* program = state->program;
  if (state->planned_operation_count == 0) return iree_ok_status();
  if (state->planned_operation_count > UINT32_MAX / 2u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "source representation operation lookup size overflows");
  }
  uint32_t slot_count = 1;
  while (slot_count < state->planned_operation_count * 2u) {
    if (slot_count > UINT32_MAX / 2u) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "source representation operation lookup size overflows");
    }
    slot_count *= 2u;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, slot_count, sizeof(*state->plan->operation_lookup_slots),
      (void**)&state->plan->operation_lookup_slots));
  state->plan->operation_lookup_slot_count = slot_count;
  for (uint32_t i = 0; i < slot_count; ++i) {
    state->plan->operation_lookup_slots[i] =
        LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
  }
  const uint32_t slot_mask = slot_count - 1;
  for (loom_source_program_node_ordinal_t node_index = 0;
       node_index < program->node_count; ++node_index) {
    const loom_source_program_node_t* node = &program->nodes[node_index];
    if (node->kind != LOOM_SOURCE_PROGRAM_NODE_OPERATION) continue;
    if (state->plan->node_selections[node_index].group_count == 0) continue;
    const loom_op_t* source_op = loom_source_program_node_operation(node);
    uint32_t slot =
        loom_low_source_representation_hash_pointer(source_op) & slot_mask;
    while (state->plan->operation_lookup_slots[slot] !=
           LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID) {
      slot = (slot + 1) & slot_mask;
    }
    state->plan->operation_lookup_slots[slot] = node_index;
  }
  return iree_ok_status();
}

iree_status_t loom_low_source_representation_plan(
    const loom_low_source_representation_provider_t* provider,
    const loom_source_program_t* program,
    const loom_low_source_representation_environment_t* environment,
    iree_arena_allocator_t* arena,
    loom_low_source_representation_plan_t* out_plan) {
  if (out_plan != NULL) *out_plan = (loom_low_source_representation_plan_t){0};
  if (provider == NULL || program == NULL || program->module == NULL ||
      program->value_domain == NULL || environment == NULL ||
      environment->module != program->module ||
      environment->descriptor_set == NULL || arena == NULL ||
      out_plan == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source representation planning requires complete inputs");
  }
  *out_plan = (loom_low_source_representation_plan_t){
      .provider = provider,
      .program = program,
      .value_domain = program->value_domain,
      .problem =
          {
              .group_index = UINT16_MAX,
              .source_value_id = LOOM_VALUE_ID_INVALID,
          },
  };
  loom_low_source_representation_state_t state = {
      .provider = provider,
      .program = program,
      .environment = environment,
      .arena = arena,
      .plan = out_plan,
  };
  const loom_value_ordinal_t value_count = program->value_domain->value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*state.component_parents),
        (void**)&state.component_parents));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*state.component_ranks),
        (void**)&state.component_ranks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*state.component_domains),
        (void**)&state.component_domains));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*state.component_group_heads),
        (void**)&state.component_group_heads));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*state.component_group_tails),
        (void**)&state.component_group_tails));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*out_plan->value_representation_indices),
        (void**)&out_plan->value_representation_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*out_plan->value_component_roots),
        (void**)&out_plan->value_component_roots));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*out_plan->component_costs),
        (void**)&out_plan->component_costs));
    memset(state.component_ranks, 0,
           value_count * sizeof(*state.component_ranks));
    memset(state.component_domains, 0,
           value_count * sizeof(*state.component_domains));
    memset(out_plan->component_costs, 0,
           value_count * sizeof(*out_plan->component_costs));
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      state.component_parents[i] = i;
      state.component_domains[i].canonical_representation_index =
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE;
      state.component_domains[i].selected_representation_index =
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE;
      state.component_group_heads[i] = UINT32_MAX;
      state.component_group_tails[i] = UINT32_MAX;
      out_plan->value_representation_indices[i] =
          LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE;
      out_plan->value_component_roots[i] = i;
    }
  }
  if (program->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, program->node_count, sizeof(*out_plan->node_selections),
        (void**)&out_plan->node_selections));
    memset(out_plan->node_selections, 0,
           program->node_count * sizeof(*out_plan->node_selections));
  }

  const loom_source_program_value_flow_kinds_t preserving_flow_kinds =
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD |
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT |
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY |
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS |
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY |
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_REGION_YIELD;
  for (uint32_t i = 0; i < program->value_flow_count; ++i) {
    const loom_source_program_value_flow_t flow = program->value_flows[i];
    if (!iree_any_bit_set(flow.kinds, preserving_flow_kinds)) continue;
    loom_low_source_representation_union_components(&state, flow.source,
                                                    flow.target);
    ++out_plan->statistics.preserving_flow_count;
  }

  iree_status_t status = loom_low_source_representation_build_groups(&state);
  if (iree_status_is_ok(status) &&
      out_plan->problem.kind == LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      state.component_parents[i] =
          loom_low_source_representation_find_component(&state, i);
    }
    status = loom_low_source_representation_apply_value_domains(&state);
  }
  if (iree_status_is_ok(status) &&
      out_plan->problem.kind == LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
    status = loom_low_source_representation_analyze_groups(&state);
  }
  if (iree_status_is_ok(status) &&
      out_plan->problem.kind == LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
    status = loom_low_source_representation_select_components(&state);
  }
  if (iree_status_is_ok(status) &&
      out_plan->problem.kind == LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE) {
    status = loom_low_source_representation_select_groups(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_source_representation_build_operation_lookup(&state);
  }
  if (!iree_status_is_ok(status)) {
    *out_plan = (loom_low_source_representation_plan_t){0};
  }
  return status;
}

static iree_string_view_t loom_low_source_representation_name(
    const loom_low_source_representation_provider_t* provider,
    loom_bstring_table_offset_t offset) {
  loom_bstring_t name = NULL;
  return loom_bstring_table_try_get(&provider->string_table, offset, &name)
             ? loom_bstring_view(name)
             : iree_string_view_empty();
}

loom_low_source_representation_value_view_t
loom_low_source_representation_plan_lookup_value(
    const loom_low_source_representation_plan_t* plan,
    loom_value_id_t value_id) {
  loom_low_source_representation_value_view_t view = {
      .component_ordinal = LOOM_VALUE_ORDINAL_INVALID,
  };
  if (plan == NULL || plan->provider == NULL || plan->value_domain == NULL) {
    return view;
  }
  const loom_value_ordinal_t value =
      loom_local_value_domain_try_ordinal(plan->value_domain, value_id);
  if (value == LOOM_VALUE_ORDINAL_INVALID) return view;
  const uint16_t representation_index =
      plan->value_representation_indices[value];
  if (representation_index == LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE) {
    return view;
  }
  const loom_value_ordinal_t component_root =
      plan->value_component_roots[value];
  const loom_low_source_representation_t* representation =
      &plan->provider->representations[representation_index];
  view = (loom_low_source_representation_value_view_t){
      .selected = true,
      .component_ordinal = component_root,
      .representation = representation,
      .representation_name = loom_low_source_representation_name(
          plan->provider, representation->name_string_offset),
      .cost = plan->component_costs[component_root],
  };
  return view;
}

static loom_source_program_node_ordinal_t
loom_low_source_representation_plan_find_operation_node(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op) {
  if (plan == NULL || plan->provider == NULL || source_op == NULL ||
      plan->operation_lookup_slot_count == 0) {
    return LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
  }
  const uint32_t slot_mask = plan->operation_lookup_slot_count - 1;
  uint32_t slot =
      loom_low_source_representation_hash_pointer(source_op) & slot_mask;
  for (uint32_t probe = 0; probe < plan->operation_lookup_slot_count; ++probe) {
    const loom_source_program_node_ordinal_t node_index =
        plan->operation_lookup_slots[slot];
    if (node_index == LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID) {
      return LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
    }
    const loom_source_program_node_t* node = &plan->program->nodes[node_index];
    if (node->kind == LOOM_SOURCE_PROGRAM_NODE_OPERATION &&
        loom_source_program_node_operation(node) == source_op) {
      return node_index;
    }
    slot = (slot + 1) & slot_mask;
  }
  return LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
}

iree_host_size_t loom_low_source_representation_plan_candidate_count(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op) {
  const loom_source_program_node_ordinal_t node_index =
      loom_low_source_representation_plan_find_operation_node(plan, source_op);
  return node_index == LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID
             ? 0
             : plan->node_selections[node_index].group_count;
}

loom_low_source_representation_candidate_view_t
loom_low_source_representation_plan_candidate_view(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op, iree_host_size_t group_ordinal) {
  loom_low_source_representation_candidate_view_t view = {0};
  const loom_source_program_node_ordinal_t node_index =
      loom_low_source_representation_plan_find_operation_node(plan, source_op);
  if (node_index == LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID) return view;
  const loom_low_source_representation_node_selection_t node_selection =
      plan->node_selections[node_index];
  if (group_ordinal >= node_selection.group_count) return view;
  const loom_low_source_representation_group_selection_t selection =
      plan->selected_groups[node_selection.group_start + group_ordinal];
  const loom_low_source_representation_group_t* group =
      &plan->provider->groups[selection.group_index];
  view.group = group;
  view.group_name = loom_low_source_representation_name(
      plan->provider, group->name_string_offset);
  if (selection.candidate_index ==
      LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE) {
    return view;
  }
  const loom_low_source_representation_candidate_t* candidate =
      &plan->provider->candidates[selection.candidate_index];
  view.selected = true;
  view.candidate = candidate;
  view.candidate_name = loom_low_source_representation_name(
      plan->provider, candidate->name_string_offset);
  if (candidate->target_data_ordinal !=
      LOOM_LOW_SOURCE_REPRESENTATION_TARGET_DATA_ORDINAL_NONE) {
    view.target_data = plan->provider->target_data +
                       (iree_host_size_t)candidate->target_data_ordinal *
                           plan->provider->target_data_stride;
  }
  return view;
}

bool loom_low_source_representation_plan_find_candidate(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op, uint64_t group_key,
    loom_low_source_representation_candidate_view_t* out_view) {
  if (out_view != NULL) {
    *out_view = (loom_low_source_representation_candidate_view_t){0};
  }
  const iree_host_size_t group_count =
      loom_low_source_representation_plan_candidate_count(plan, source_op);
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    const loom_low_source_representation_candidate_view_t view =
        loom_low_source_representation_plan_candidate_view(plan, source_op, i);
    if (view.group != NULL && view.group->stable_key == group_key) {
      if (out_view != NULL) *out_view = view;
      return true;
    }
  }
  return false;
}
