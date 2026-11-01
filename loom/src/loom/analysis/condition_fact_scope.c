// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_fact_scope.h"

void loom_condition_fact_scope_initialize_local(
    const loom_condition_fact_scope_t* parent,
    const loom_condition_derivation_t* derivation,
    loom_condition_fact_scope_t* out_scope) {
  *out_scope = (loom_condition_fact_scope_t){
      .parent = parent,
      .local_derivation = derivation,
  };
}

void loom_condition_fact_scope_initialize_projected(
    const loom_condition_fact_scope_t* parent,
    const loom_condition_edge_projection_t* projection,
    loom_condition_fact_scope_t* out_scope) {
  *out_scope = (loom_condition_fact_scope_t){
      .parent = parent,
      .edge_projection = projection,
  };
}

iree_status_t loom_condition_fact_scope_extend_region(
    const loom_value_fact_table_t* fact_table, const loom_region_t* region,
    const loom_condition_fact_scope_t* parent, iree_arena_allocator_t* arena,
    const loom_condition_fact_scope_t** out_scope) {
  *out_scope = parent;
  const loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_region_condition_projection(fact_table,
                                                               region);
  if (!projection || loom_condition_edge_projection_is_empty(projection)) {
    return iree_ok_status();
  }
  loom_condition_fact_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
  loom_condition_fact_scope_initialize_projected(parent, projection, scope);
  *out_scope = scope;
  return iree_ok_status();
}

void loom_condition_fact_scope_initialize_indexed(
    const loom_condition_fact_scope_t* parent,
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    loom_condition_fact_scope_t* out_scope) {
  *out_scope = (loom_condition_fact_scope_t){
      .parent = parent,
      .relation_table = table,
      .relation_view = view,
  };
}

bool loom_condition_fact_scope_has_integer_relations(
    const loom_condition_fact_scope_t* scope) {
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    if ((current->local_derivation != NULL &&
         current->local_derivation->integer_facts.integer_relation_count !=
             0) ||
        (current->edge_projection != NULL &&
         current->edge_projection->visible_integer_relation_count != 0) ||
        (current->relation_view != NULL &&
         current->relation_view->integer_relations.entry_count != 0)) {
      return true;
    }
  }
  return false;
}

typedef struct loom_condition_fact_scope_apply_state_t {
  // Ambient facts used to resolve exact integer operands.
  const loom_value_fact_table_t* fact_table;

  // Anchored value being refined.
  loom_value_id_t value_id;

  // Mutable scalar facts receiving every applicable relation.
  loom_value_facts_t* facts;

  // True when at least one relation was applicable.
  bool applied;
} loom_condition_fact_scope_apply_state_t;

static bool loom_condition_fact_scope_apply_relation(
    void* user_data, const loom_condition_integer_relation_t* relation) {
  loom_condition_fact_scope_apply_state_t* state =
      (loom_condition_fact_scope_apply_state_t*)user_data;
  state->applied |= loom_condition_integer_relation_apply_to_value_facts(
      relation, state->fact_table, state->value_id, state->facts);
  return true;
}

bool loom_condition_fact_scope_apply_to_value_facts(
    const loom_condition_fact_scope_t* scope,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  loom_condition_fact_scope_apply_state_t state = {
      .fact_table = fact_table,
      .value_id = value_id,
      .facts = inout_facts,
  };
  const loom_condition_integer_operand_t anchor = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
  };
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    if (current->local_derivation != NULL) {
      state.applied |= loom_condition_fact_set_apply_to_value_facts(
          &current->local_derivation->integer_facts, fact_table, value_id,
          inout_facts);
    } else if (current->edge_projection != NULL) {
      state.applied |= loom_condition_edge_projection_apply_to_value_facts(
          current->edge_projection, fact_table, value_id, inout_facts);
    } else if (current->relation_view != NULL) {
      (void)loom_cfg_condition_relation_view_for_each_while(
          current->relation_table, current->relation_view, fact_table, anchor,
          loom_condition_fact_scope_apply_relation, &state);
    }
  }
  return state.applied;
}

static loom_integer_order_outcome_bits_t
loom_condition_fact_scope_flat_exclusions(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried) {
  const loom_integer_order_outcome_bits_t outcomes =
      loom_condition_fact_set_query_integer_order(
          facts, fact_table, queried->left, queried->right);
  // An inconsistent local derivation cannot establish a proof.
  return outcomes ? LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &
                        (loom_integer_order_outcome_bits_t)~outcomes
                  : 0;
}

static loom_integer_order_outcome_bits_t
loom_condition_fact_scope_projection_exclusions(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried) {
  const loom_integer_order_outcome_bits_t outcomes =
      loom_condition_edge_projection_query_integer_order(
          projection, fact_table, queried->left, queried->right);
  return outcomes ? LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &
                        (loom_integer_order_outcome_bits_t)~outcomes
                  : 0;
}

loom_integer_order_outcome_bits_t loom_condition_fact_scope_query_integer_order(
    const loom_condition_fact_scope_t* scope,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  const loom_condition_integer_relation_t queried = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      .left = left,
      .right = right,
  };
  loom_integer_order_outcome_bits_t exclusions = 0;
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    if (current->local_derivation != NULL) {
      exclusions |= loom_condition_fact_scope_flat_exclusions(
          &current->local_derivation->integer_facts, fact_table, &queried);
    } else if (current->edge_projection != NULL) {
      exclusions |= loom_condition_fact_scope_projection_exclusions(
          current->edge_projection, fact_table, &queried);
    } else if (current->relation_view != NULL) {
      exclusions |= loom_cfg_condition_relation_view_query_excluded_outcomes(
          current->relation_table, current->relation_view, fact_table, left,
          right);
    }
  }
  return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &
         (loom_integer_order_outcome_bits_t)~exclusions;
}

bool loom_condition_fact_scope_proves_integer_relation(
    const loom_condition_fact_scope_t* scope,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  const loom_integer_order_outcome_bits_t outcomes =
      loom_condition_fact_scope_query_integer_order(
          scope, fact_table, queried->left, queried->right);
  return loom_symbolic_integer_relation_resolve(outcomes, queried->relation,
                                                out_result);
}

static bool loom_condition_fact_scope_query_local_boolean(
    const loom_condition_derivation_t* derivation, loom_value_id_t value_id,
    bool* out_value) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = derivation->boolean_fact_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (derivation->boolean_facts[middle].value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  bool known[2] = {false, false};
  while (begin < derivation->boolean_fact_count &&
         derivation->boolean_facts[begin].value_id == value_id) {
    known[derivation->boolean_facts[begin].value ? 1 : 0] = true;
    ++begin;
  }
  if (known[0] == known[1]) {
    return false;
  }
  *out_value = known[1];
  return true;
}

static bool loom_condition_fact_scope_query_boolean(
    const loom_condition_fact_scope_t* scope, loom_value_id_t value_id,
    bool* out_value) {
  bool known[2] = {false, false};
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    bool value = false;
    bool found = false;
    if (current->local_derivation != NULL) {
      found = loom_condition_fact_scope_query_local_boolean(
          current->local_derivation, value_id, &value);
    } else if (current->edge_projection != NULL) {
      found = loom_condition_edge_projection_query_boolean(
          current->edge_projection, value_id, &value);
    } else if (current->relation_view != NULL) {
      found = loom_cfg_condition_relation_view_query_boolean(
          current->relation_table, current->relation_view, value_id, &value);
    }
    if (found) {
      known[value ? 1 : 0] = true;
    }
  }
  if (known[0] == known[1]) {
    return false;
  }
  *out_value = known[1];
  return true;
}

static bool loom_condition_fact_scope_resolver_query_boolean(
    const void* user_data, loom_value_id_t value_id, bool* out_value) {
  return loom_condition_fact_scope_query_boolean(
      (const loom_condition_fact_scope_t*)user_data, value_id, out_value);
}

static bool loom_condition_fact_scope_resolver_apply_to_value_facts(
    const void* user_data, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* inout_facts) {
  return loom_condition_fact_scope_apply_to_value_facts(
      (const loom_condition_fact_scope_t*)user_data, fact_table, value_id,
      inout_facts);
}

static bool loom_condition_fact_scope_resolver_proves_integer_relation(
    const void* user_data, const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  return loom_condition_fact_scope_proves_integer_relation(
      (const loom_condition_fact_scope_t*)user_data, fact_table, queried,
      out_result);
}

iree_status_t loom_condition_fact_scope_proves_condition(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_scope_t* scope, loom_value_id_t condition_value,
    bool* out_condition, bool* out_proven) {
  const loom_condition_fact_resolver_t resolver = {
      .user_data = scope,
      .query_boolean = loom_condition_fact_scope_resolver_query_boolean,
      .apply_to_value_facts =
          loom_condition_fact_scope_resolver_apply_to_value_facts,
      .proves_integer_relation =
          loom_condition_fact_scope_resolver_proves_integer_relation,
  };
  return loom_condition_fact_resolver_proves_condition(
      query, fact_table, scope != NULL ? &resolver : NULL, condition_value,
      out_condition, out_proven);
}

typedef struct loom_condition_fact_scope_anchor_list_t {
  // Explicit relation operands, when supplied by symbolic proof.
  const loom_condition_integer_operand_t* operands;

  // Number of entries in operands.
  iree_host_size_t count;
} loom_condition_fact_scope_anchor_list_t;

static bool loom_condition_fact_scope_visit_projected_relations(
    const loom_condition_edge_projection_t* projection,
    loom_condition_fact_scope_anchor_list_t anchors,
    loom_cfg_condition_relation_visit_fn_t visit, void* user_data) {
  const loom_condition_fact_set_t* facts =
      &projection->source_derivation.integer_facts;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* source =
        &facts->integer_relations[i];
    loom_condition_integer_relation_t projected = *source;
    if (!loom_condition_edge_projection_target_operand(projection, source->left,
                                                       &projected.left) ||
        !loom_condition_edge_projection_target_operand(
            projection, source->right, &projected.right)) {
      continue;
    }
    bool selected_left_anchor = false;
    bool selected_right_anchor = false;
    for (iree_host_size_t j = 0; j < anchors.count; ++j) {
      const loom_condition_integer_operand_t anchor = anchors.operands[j];
      const loom_condition_integer_operand_t source_anchor =
          loom_condition_edge_projection_source_operand(projection, anchor);
      if (!selected_left_anchor &&
          loom_condition_integer_operands_equal(source_anchor, source->left)) {
        projected.left = anchor;
        selected_left_anchor = true;
      }
      if (!selected_right_anchor &&
          loom_condition_integer_operands_equal(source_anchor, source->right)) {
        projected.right = anchor;
        selected_right_anchor = true;
      }
    }
    if (!visit(user_data, &projected)) {
      return false;
    }
  }
  return true;
}

bool loom_condition_fact_scope_for_each_anchored_while(
    const loom_condition_fact_scope_t* scope,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_operand_t* anchors,
    iree_host_size_t anchor_count, loom_cfg_condition_relation_visit_fn_t visit,
    void* user_data) {
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    if (current->local_derivation != NULL) {
      const loom_condition_fact_set_t* facts =
          &current->local_derivation->integer_facts;
      for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
        if (!visit(user_data, &facts->integer_relations[i])) {
          return false;
        }
      }
    } else if (current->edge_projection != NULL) {
      if (!loom_condition_fact_scope_visit_projected_relations(
              current->edge_projection,
              (loom_condition_fact_scope_anchor_list_t){
                  .operands = anchors,
                  .count = anchor_count,
              },
              visit, user_data)) {
        return false;
      }
    } else if (current->relation_view != NULL) {
      for (iree_host_size_t i = 0; i < anchor_count; ++i) {
        if (!loom_cfg_condition_relation_view_for_each_while(
                current->relation_table, current->relation_view, fact_table,
                anchors[i], visit, user_data)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool loom_condition_fact_scope_for_each_value_anchored_while(
    const loom_condition_fact_scope_t* scope,
    const loom_value_fact_table_t* fact_table, const loom_value_id_t* value_ids,
    iree_host_size_t value_count, loom_cfg_condition_relation_visit_fn_t visit,
    void* user_data) {
  for (const loom_condition_fact_scope_t* current = scope; current != NULL;
       current = current->parent) {
    if (current->local_derivation != NULL) {
      const loom_condition_fact_set_t* facts =
          &current->local_derivation->integer_facts;
      for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
        if (!visit(user_data, &facts->integer_relations[i])) {
          return false;
        }
      }
    } else if (current->edge_projection != NULL) {
      if (!loom_condition_fact_scope_visit_projected_relations(
              current->edge_projection,
              (loom_condition_fact_scope_anchor_list_t){0}, visit, user_data)) {
        return false;
      }
    } else if (current->relation_view != NULL) {
      for (iree_host_size_t i = 0; i < value_count; ++i) {
        const loom_condition_integer_operand_t anchor = {
            .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
            .value_id = value_ids[i],
        };
        if (!loom_cfg_condition_relation_view_for_each_while(
                current->relation_table, current->relation_view, fact_table,
                anchor, visit, user_data)) {
          return false;
        }
      }
    }
  }
  return true;
}
