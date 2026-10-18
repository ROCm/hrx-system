// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/conditioned_value_facts.h"

#include <string.h>

#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_conditioned_value_facts_region_t {
  // Immutable region described by this record.
  const loom_region_t* region;
  // Borrowed fact-owner structure, or NULL for a structured region.
  const loom_value_fact_cfg_region_t* structure;
  // Structured-edge projections inherited by every block in the region.
  const loom_condition_fact_scope_t* projected_scope;
  // Local condition scope per block, indexed by region ordinal.
  struct loom_conditioned_value_facts_local_scope_t* local_scopes;
  // Next record in the region-address hash bucket.
  struct loom_conditioned_value_facts_region_t* next_bucket;
  // Next record in the complete scope list.
  struct loom_conditioned_value_facts_region_t* next;
} loom_conditioned_value_facts_region_t;

typedef struct loom_conditioned_value_facts_active_update_t {
  // Relation applied while entering the owning local scope.
  const loom_condition_integer_relation_t* relation;
  // Dense ordinal of the relation operand refined by this update.
  loom_value_ordinal_t value_ordinal;
  // Previous active refinement for the same value.
  struct loom_conditioned_value_facts_active_update_t* previous;
  // Cumulative facts after applying relation.
  loom_value_facts_t facts;
} loom_conditioned_value_facts_active_update_t;

typedef struct loom_conditioned_value_facts_local_scope_t {
  // Retained integer relations, or NULL for a structural placeholder.
  loom_condition_fact_set_t* facts;
  // Outer local condition scope, or NULL at the function root.
  struct loom_conditioned_value_facts_local_scope_t* parent;
  // Mutable traversal updates, one for each relation value operand.
  loom_conditioned_value_facts_active_update_t* updates;
  // Number of entries in updates.
  uint32_t update_count;
  // One-based lexical depth used to transition between scopes.
  uint32_t depth;
  // Next local scope in the complete solve list.
  struct loom_conditioned_value_facts_local_scope_t* next;
} loom_conditioned_value_facts_local_scope_t;

static_assert(sizeof(loom_conditioned_value_facts_local_scope_t) ==
                  (IREE_PTR_SIZE == 8 ? 40 : 24),
              "local scopes must stay compact because every block owns one");

typedef struct loom_conditioned_value_facts_t {
  // Function whose IR stays immutable throughout the solve.
  loom_module_t* module;
  // Numeric scope being refined in place.
  loom_value_fact_table_t* table;
  // Construction and retained relation storage for this solve.
  iree_arena_allocator_t arena;
  // Temporary function-local value identity domain.
  loom_local_value_domain_t domain;
  // Exact identity representatives. CFG identities are discovered first and
  // then composed in place with retained fact identities for O(1) lookup.
  loom_cfg_value_identity_table_t identities;
  // Reusable recursive condition query over this immutable function.
  loom_condition_query_t query;
  // Dominance wrapper borrowing the fact owner's existing trees.
  loom_dominance_info_t dominance;
  // Region-address hash buckets for per-operation scope lookup.
  loom_conditioned_value_facts_region_t** buckets;
  // Power-of-two number of buckets.
  iree_host_size_t bucket_count;
  // All region records, with arena lifetime.
  loom_conditioned_value_facts_region_t* regions;
  // Local Boolean-edge scopes retained during the solve.
  loom_conditioned_value_facts_local_scope_t* local_scopes;
  // Number of entries in local_scopes.
  iree_host_size_t local_scope_count;
  // Number of local scopes carrying integer relations.
  iree_host_size_t local_condition_scope_count;
  // Innermost active update indexed by local value ordinal.
  loom_conditioned_value_facts_active_update_t** active_updates;
  // Scratch path used when entering several nested local scopes.
  loom_conditioned_value_facts_local_scope_t** transition_path;
  // Local scope active at the previous operand-refinement callback.
  loom_conditioned_value_facts_local_scope_t* active_scope;
} loom_conditioned_value_facts_t;

static void loom_conditioned_value_facts_initialize_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* parent,
    loom_condition_fact_set_t* facts,
    loom_conditioned_value_facts_local_scope_t* scope) {
  *scope = (loom_conditioned_value_facts_local_scope_t){
      .facts = facts,
      .parent = parent,
      .depth = parent ? parent->depth + 1 : 1,
      .next = state->local_scopes,
  };
  state->local_scopes = scope;
  ++state->local_scope_count;
  state->local_condition_scope_count += facts != NULL ? 1 : 0;
}

static iree_status_t loom_conditioned_value_facts_allocate_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* parent,
    loom_condition_fact_set_t* facts,
    loom_conditioned_value_facts_local_scope_t** out_scope) {
  loom_conditioned_value_facts_local_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&state->arena, sizeof(*scope), (void**)&scope));
  loom_conditioned_value_facts_initialize_local_scope(state, parent, facts,
                                                      scope);
  *out_scope = scope;
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_retain_integer_facts(
    loom_conditioned_value_facts_t* state,
    const loom_condition_fact_set_t* source,
    loom_condition_fact_set_t** out_facts) {
  *out_facts = NULL;
  if (!source->integer_relation_count) {
    return iree_ok_status();
  }
  const iree_host_size_t relation_bytes =
      source->integer_relation_count * sizeof(*source->integer_relations);
  const iree_host_size_t relation_offset =
      iree_host_align(sizeof(loom_condition_fact_set_t),
                      iree_alignof(loom_condition_integer_relation_t));
  loom_condition_fact_set_t* retained = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &state->arena, relation_offset + relation_bytes, (void**)&retained));
  retained->integer_relations =
      (loom_condition_integer_relation_t*)((uint8_t*)retained +
                                           relation_offset);
  retained->integer_relation_count = source->integer_relation_count;
  retained->integer_relation_capacity = source->integer_relation_count;
  memcpy(retained->integer_relations, source->integer_relations,
         relation_bytes);
  *out_facts = retained;
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_query_integer_facts(
    loom_conditioned_value_facts_t* state, loom_value_id_t condition,
    bool assumed_truth, loom_condition_fact_set_t** out_facts) {
  *out_facts = NULL;
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(state->table->transient_arena);
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(state->table->transient_arena,
                                       &derivation);
  iree_status_t status = loom_condition_facts_query_complete(
      &state->query, state->table, condition, assumed_truth, &derivation);
  if (iree_status_is_ok(status)) {
    status = loom_conditioned_value_facts_retain_integer_facts(
        state, &derivation.integer_facts, out_facts);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

static iree_host_size_t loom_conditioned_value_facts_bucket(
    const loom_conditioned_value_facts_t* state, const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  bits ^= bits >> 11;
  return (iree_host_size_t)bits & (state->bucket_count - 1);
}

static iree_status_t loom_conditioned_value_facts_extend_region(
    loom_conditioned_value_facts_t* state, loom_op_t* parent_op,
    loom_region_t* region, const loom_condition_fact_scope_t* parent,
    loom_conditioned_value_facts_local_scope_t* local_parent,
    const loom_condition_fact_scope_t** out_scope,
    loom_conditioned_value_facts_local_scope_t** out_local_scope) {
  *out_local_scope = local_parent;
  IREE_RETURN_IF_ERROR(loom_condition_fact_scope_extend_region(
      state->table, region, parent, &state->arena, out_scope));
  const loom_region_branch_truth_t truth =
      loom_value_fact_table_lookup_region_branch_truth(state->table, region);
  if (truth == LOOM_REGION_BRANCH_TRUTH_UNKNOWN) {
    return iree_ok_status();
  }

  loom_region_branch_t branch =
      loom_region_branch_cast(state->module, parent_op);
  IREE_ASSERT(loom_region_branch_isa(branch));
  loom_condition_fact_set_t* retained_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_query_integer_facts(
      state, loom_region_branch_selector(branch),
      truth == LOOM_REGION_BRANCH_TRUTH_TRUE, &retained_facts));
  if (!retained_facts) {
    return iree_ok_status();
  }
  loom_conditioned_value_facts_local_scope_t* local_scope = NULL;
  IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_allocate_local_scope(
      state, local_parent, retained_facts, &local_scope));
  *out_local_scope = local_scope;
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_collect(
    loom_conditioned_value_facts_t* state, loom_region_t* region,
    const loom_condition_fact_scope_t* projected_scope,
    loom_conditioned_value_facts_local_scope_t* local_scope) {
  loom_conditioned_value_facts_region_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&state->arena, sizeof(*entry), (void**)&entry));
  const iree_host_size_t bucket =
      loom_conditioned_value_facts_bucket(state, region);
  *entry = (loom_conditioned_value_facts_region_t){
      .region = region,
      .structure =
          loom_value_fact_table_lookup_cfg_region(state->table, region),
      .projected_scope = projected_scope,
      .next_bucket = state->buckets[bucket],
      .next = state->regions,
  };
  state->buckets[bucket] = entry;
  state->regions = entry;
  if (entry->structure) {
    IREE_RETURN_IF_ERROR(loom_dominance_info_add_cfg_graph(
        &state->dominance, &entry->structure->graph,
        &entry->structure->dominance));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, region->block_count, sizeof(*entry->local_scopes),
      (void**)&entry->local_scopes));
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_conditioned_value_facts_initialize_local_scope(
        state, local_scope, NULL, &entry->local_scopes[block->region_index]);
  }
  if (entry->structure) {
    const loom_cfg_dominance_t* dominance = &entry->structure->dominance;
    for (iree_host_size_t i = 1; i < dominance->preorder.count; ++i) {
      const uint16_t block_index = dominance->preorder.values[i];
      const uint16_t parent_index =
          dominance->immediate_dominators[block_index];
      loom_conditioned_value_facts_local_scope_t* block_scope =
          &entry->local_scopes[block_index];
      block_scope->parent = &entry->local_scopes[parent_index];
      block_scope->depth = block_scope->parent->depth + 1;
    }
  }
  loom_region_for_each_block(region, block) {
    loom_conditioned_value_facts_local_scope_t* block_scope =
        &entry->local_scopes[block->region_index];
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i]) {
          const loom_condition_fact_scope_t* child_projected_scope = NULL;
          loom_conditioned_value_facts_local_scope_t* child_local_scope = NULL;
          IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_extend_region(
              state, op, regions[i], projected_scope, block_scope,
              &child_projected_scope, &child_local_scope));
          IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_collect(
              state, regions[i], child_projected_scope, child_local_scope));
        }
      }
    }
  }
  return iree_ok_status();
}

static void loom_conditioned_value_facts_exit_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* scope) {
  for (uint32_t i = scope->update_count; i > 0; --i) {
    loom_conditioned_value_facts_active_update_t* update =
        &scope->updates[i - 1];
    state->active_updates[update->value_ordinal] = update->previous;
  }
}

static void loom_conditioned_value_facts_enter_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* scope,
    const loom_value_fact_table_t* table) {
  // Each relation operand dominates the edge that established the scope. A
  // cyclic solve therefore revisits its producer in an ancestor scope before
  // re-entering this scope with changed facts. Rebuilding on entry keeps the
  // cumulative summaries current without rescanning the lexical chain at each
  // use.
  for (uint32_t i = 0; i < scope->update_count; ++i) {
    loom_conditioned_value_facts_active_update_t* update = &scope->updates[i];
    update->previous = state->active_updates[update->value_ordinal];
    const loom_value_id_t value_id =
        state->domain.value_ids[update->value_ordinal];
    update->facts = update->previous
                        ? update->previous->facts
                        : loom_value_fact_table_lookup(table, value_id);
    (void)loom_condition_integer_relation_apply_to_value_facts(
        update->relation, table, value_id, &update->facts);
    state->active_updates[update->value_ordinal] = update;
  }
}

static void loom_conditioned_value_facts_transition_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* target,
    const loom_value_fact_table_t* table) {
  loom_conditioned_value_facts_local_scope_t* current = state->active_scope;
  iree_host_size_t path_count = 0;
  while (current && (!target || current->depth > target->depth)) {
    loom_conditioned_value_facts_exit_local_scope(state, current);
    current = current->parent;
  }
  while (target && (!current || target->depth > current->depth)) {
    state->transition_path[path_count++] = target;
    target = target->parent;
  }
  while (current != target) {
    IREE_ASSERT(current && target);
    loom_conditioned_value_facts_exit_local_scope(state, current);
    current = current->parent;
    state->transition_path[path_count++] = target;
    target = target->parent;
  }
  while (path_count) {
    loom_conditioned_value_facts_local_scope_t* entered =
        state->transition_path[--path_count];
    loom_conditioned_value_facts_enter_local_scope(state, entered, table);
    current = entered;
  }
  state->active_scope = current;
}

static void loom_conditioned_value_facts_apply_active_refinement(
    const loom_value_facts_t* active, loom_value_facts_t* inout_facts) {
  inout_facts->range_lo = iree_max(inout_facts->range_lo, active->range_lo);
  inout_facts->range_hi = iree_min(inout_facts->range_hi, active->range_hi);
  // Comparison relations do not introduce divisibility. An exact relation can
  // only strengthen a consistent existing divisor to one of its multiples.
  if (active->known_divisor % inout_facts->known_divisor == 0) {
    inout_facts->known_divisor = active->known_divisor;
  }
  const uint32_t retained_flags =
      active->flags & (LOOM_VALUE_FACT_NON_ZERO | LOOM_VALUE_FACT_POWER_OF_TWO);
  loom_value_facts_recompute_flags(inout_facts);
  inout_facts->flags |= retained_flags;
}

static void loom_conditioned_value_facts_refine_operands(
    void* user_data, const loom_value_fact_table_t* table, const loom_op_t* op,
    loom_value_facts_t* operands) {
  loom_conditioned_value_facts_t* state = user_data;
  const loom_region_t* region = op->parent_block->parent_region;
  const iree_host_size_t bucket =
      loom_conditioned_value_facts_bucket(state, region);
  const loom_conditioned_value_facts_region_t* entry = state->buckets[bucket];
  while (entry && entry->region != region) {
    entry = entry->next_bucket;
  }
  // Projected configuration regions lie outside the function body domain.
  if (!entry) {
    return;
  }
  if (state->local_condition_scope_count) {
    loom_conditioned_value_facts_transition_local_scope(
        state, &entry->local_scopes[op->parent_block->region_index], table);
  }
  if (!op->operand_count) {
    return;
  }
  const loom_value_id_t* values = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t canonical =
        loom_cfg_value_identity_table_lookup(&state->identities, values[i]);
    if (state->active_updates) {
      const loom_value_ordinal_t ordinal =
          loom_local_value_domain_try_ordinal(&state->domain, canonical);
      if (ordinal != LOOM_VALUE_ORDINAL_INVALID &&
          state->active_updates[ordinal]) {
        loom_conditioned_value_facts_apply_active_refinement(
            &state->active_updates[ordinal]->facts, &operands[i]);
      }
    }
    loom_condition_fact_scope_apply_to_value_facts(
        entry->projected_scope, table, canonical, &operands[i]);
  }
}

static void loom_conditioned_value_facts_canonicalize_facts(
    const loom_cfg_value_identity_table_t* identities,
    loom_condition_fact_set_t* facts) {
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    loom_condition_integer_relation_t* relation = &facts->integer_relations[i];
    if (relation->left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
      relation->left.value_id = loom_cfg_value_identity_table_lookup(
          identities, relation->left.value_id);
    }
    if (relation->right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
      relation->right.value_id = loom_cfg_value_identity_table_lookup(
          identities, relation->right.value_id);
    }
  }
}

// Composes the two exact-identity domains once before indexing conditions.
// Both mappings select an available predecessor identity, so alternating them
// reaches a fixed point. Reusing the CFG table storage keeps operand lookup
// O(1) without allocating another value-sized index.
static void loom_conditioned_value_facts_compose_identities(
    loom_conditioned_value_facts_t* state) {
  for (loom_value_ordinal_t ordinal = 0; ordinal < state->domain.value_count;
       ++ordinal) {
    const loom_value_id_t original = state->domain.value_ids[ordinal];
    loom_value_id_t canonical = original;
    for (iree_host_size_t depth = 0;; ++depth) {
      IREE_ASSERT_LT(depth, state->domain.value_count);
      const loom_value_id_t next = loom_cfg_value_identity_table_lookup(
          &state->identities,
          loom_value_fact_table_query_identity(state->table, canonical));
      if (next == canonical) {
        break;
      }
      canonical = next;
    }
    state->identities.representatives[ordinal] =
        canonical == original ? LOOM_VALUE_ID_INVALID : canonical;
  }
}

static bool loom_conditioned_value_facts_operand_needs_update(
    const loom_conditioned_value_facts_t* state,
    loom_condition_integer_operand_t operand) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
         !loom_value_facts_is_exact(
             loom_value_fact_table_lookup(state->table, operand.value_id));
}

static iree_status_t loom_conditioned_value_facts_prepare_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* scope) {
  if (!scope->facts) {
    return iree_ok_status();
  }
  const loom_condition_fact_set_t* facts = scope->facts;
  iree_host_size_t update_count = 0;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* relation =
        &facts->integer_relations[i];
    const bool update_left = loom_conditioned_value_facts_operand_needs_update(
        state, relation->left);
    update_count += update_left ? 1 : 0;
    update_count += loom_conditioned_value_facts_operand_needs_update(
                        state, relation->right) &&
                            (!update_left || relation->right.value_id !=
                                                 relation->left.value_id)
                        ? 1
                        : 0;
  }
  if (!update_count) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(update_count, UINT32_MAX);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&state->arena, update_count,
                                                 sizeof(*scope->updates),
                                                 (void**)&scope->updates));
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* relation =
        &facts->integer_relations[i];
    const bool update_left = loom_conditioned_value_facts_operand_needs_update(
        state, relation->left);
    if (update_left) {
      scope->updates[scope->update_count++] =
          (loom_conditioned_value_facts_active_update_t){
              .relation = relation,
              .value_ordinal = loom_local_value_domain_ordinal(
                  &state->domain, relation->left.value_id),
          };
    }
    if (loom_conditioned_value_facts_operand_needs_update(state,
                                                          relation->right) &&
        (!update_left || relation->right.value_id != relation->left.value_id)) {
      scope->updates[scope->update_count++] =
          (loom_conditioned_value_facts_active_update_t){
              .relation = relation,
              .value_ordinal = loom_local_value_domain_ordinal(
                  &state->domain, relation->right.value_id),
          };
    }
  }
  IREE_ASSERT_EQ(scope->update_count, update_count);
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_prepare_local_scopes(
    loom_conditioned_value_facts_t* state) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, state->domain.value_count, sizeof(*state->active_updates),
      (void**)&state->active_updates));
  memset(state->active_updates, 0,
         state->domain.value_count * sizeof(*state->active_updates));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, state->local_scope_count, sizeof(*state->transition_path),
      (void**)&state->transition_path));
  for (loom_conditioned_value_facts_local_scope_t* scope = state->local_scopes;
       scope; scope = scope->next) {
    IREE_RETURN_IF_ERROR(
        loom_conditioned_value_facts_prepare_local_scope(state, scope));
  }
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_solve(
    loom_conditioned_value_facts_t* state, loom_func_like_t function) {
  state->dominance = (loom_dominance_info_t){
      .module = state->module,
      .arena = &state->arena,
  };
  state->bucket_count = state->table->regions.bucket_count;
  loom_condition_query_initialize(state->module, &state->domain, &state->arena,
                                  &state->query);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, state->bucket_count, sizeof(*state->buckets),
      (void**)&state->buckets));
  memset(state->buckets, 0, state->bucket_count * sizeof(*state->buckets));
  IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_collect(
      state, loom_func_like_body(function), NULL, NULL));
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_table_initialize(
      &state->domain, &state->arena, &state->identities));
  for (loom_conditioned_value_facts_region_t* entry = state->regions; entry;
       entry = entry->next) {
    if (entry->structure) {
      IREE_RETURN_IF_ERROR(loom_cfg_value_identity_table_update(
          &state->identities, entry->structure, &state->dominance,
          &state->arena));
    }
  }
  bool has_conditions = state->table->condition_integer_projection_count != 0 ||
                        state->local_condition_scope_count != 0;
  for (loom_conditioned_value_facts_region_t* entry = state->regions; entry;
       entry = entry->next) {
    if (!entry->structure) {
      continue;
    }
    const loom_cfg_graph_t* graph = &entry->structure->graph;
    const loom_cfg_dominance_t* dominance = &entry->structure->dominance;
    for (iree_host_size_t i = 1; i < dominance->preorder.count; ++i) {
      const uint16_t block = dominance->preorder.values[i];
      const uint16_t predecessor = dominance->entry_predecessors[block];
      if (predecessor == LOOM_CFG_DOMINATOR_INVALID) {
        continue;
      }
      // The dominance owner has already proved that this predecessor's direct
      // alternative enters the block and dominates its entire subtree.
      const loom_cfg_edge_index_span_t edges =
          loom_cfg_graph_successor_edges(graph, predecessor);
      const loom_cfg_edge_info_t* first = &graph->edges[edges.values[0]];
      if (!loom_cfg_cond_br_isa(first->terminator)) {
        continue;
      }
      const loom_cfg_edge_info_t* second = &graph->edges[edges.values[1]];
      // Parallel outcomes establish no truth value for their common target.
      if (first->target_block_index == second->target_block_index) {
        continue;
      }
      const loom_cfg_edge_info_t* edge =
          first->target_block_index == block ? first : second;
      loom_condition_fact_set_t* facts = NULL;
      IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_query_integer_facts(
          state, loom_cfg_cond_br_condition(edge->terminator),
          edge->successor_index == 0, &facts));
      if (facts) {
        entry->local_scopes[block].facts = facts;
        ++state->local_condition_scope_count;
        has_conditions = true;
      }
    }
  }
  if (!has_conditions) {
    return iree_ok_status();
  }
  loom_conditioned_value_facts_compose_identities(state);
  for (loom_conditioned_value_facts_local_scope_t* local_scope =
           state->local_scopes;
       local_scope; local_scope = local_scope->next) {
    if (local_scope->facts) {
      loom_conditioned_value_facts_canonicalize_facts(&state->identities,
                                                      local_scope->facts);
    }
  }
  if (state->local_condition_scope_count) {
    IREE_RETURN_IF_ERROR(
        loom_conditioned_value_facts_prepare_local_scopes(state));
  }
  state->table->has_conditioned_results = true;
  state->table->context.refine_operands.user_data = state;
  state->table->context.refine_operands.fn =
      loom_conditioned_value_facts_refine_operands;
  iree_status_t status =
      loom_value_fact_table_compute(state->table, state->module, function);
  state->table->context.refine_operands.fn = NULL;
  state->table->context.refine_operands.user_data = NULL;
  return status;
}

iree_status_t loom_conditioned_value_facts_compute(
    loom_value_fact_table_t* table, loom_module_t* module,
    loom_func_like_t function) {
  if ((!table->regions.cfg_count &&
       !table->condition_integer_projection_count &&
       !table->has_boolean_branch_regions) ||
      !loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_conditioned_value_facts_t state = {.module = module, .table = table};
  iree_arena_initialize(table->arena->block_pool, &state.arena);
  iree_status_t status = loom_local_value_domain_acquire_for_region_tree(
      module, loom_func_like_body(function), &state.arena, &state.domain);
  if (iree_status_is_ok(status)) {
    status = loom_conditioned_value_facts_solve(&state, function);
  }
  if (loom_local_value_domain_is_acquired(&state.domain)) {
    loom_local_value_domain_release(&state.domain);
  }
  iree_arena_deinitialize(&state.arena);
  return status;
}
