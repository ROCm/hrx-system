// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/conditioned_value_facts.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/cast.h"
#include "loom/ops/op_defs.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_cast.h"

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

// The low two incidence bits retain which original endpoints need translation
// to this update's canonical value. A zero mask keeps the direct relation path.
enum {
  LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_LEFT = 1u << 0,
  LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_RIGHT = 1u << 1,
  LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_MASK =
      LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_LEFT |
      LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_RIGHT,
};

typedef struct loom_conditioned_value_facts_active_update_t {
  // Encoded relation/translation bits for a singleton, or packed incidence
  // offset for an update with several relations.
  iree_host_size_t relation_index;
  // Number of relations applied to this value, in source order.
  uint32_t relation_count;
  // Dense ordinal of the value refined by this update.
  loom_value_ordinal_t value_ordinal;
  // Previous active refinement for the same value.
  struct loom_conditioned_value_facts_active_update_t* previous;
  // Cumulative facts after applying this scope's relations for the value.
  loom_value_facts_t facts;
} loom_conditioned_value_facts_active_update_t;

typedef struct loom_conditioned_value_facts_local_scope_t {
  // Immutable relations with original provider IDs, or NULL for a placeholder.
  const loom_condition_fact_set_t* facts;
  // Outer local condition scope, or NULL at the function root.
  struct loom_conditioned_value_facts_local_scope_t* parent;
  // One update per value, followed by packed indices for non-singleton updates.
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
  // Immutable projection of producer interpretations into CFG identities.
  struct loom_conditioned_value_facts_cast_t* casts;
  // Allocated conversion row capacity.
  iree_host_size_t cast_capacity;
  // Sparse result-to-conversion index.
  loom_intern_table_t cast_index;
  // Reusable numeric representatives borrowed by one result publication.
  loom_value_id_t* numeric_operands;
  // Largest operand count in the immutable function.
  uint16_t max_operand_count;
} loom_conditioned_value_facts_t;

typedef struct loom_conditioned_value_facts_cast_t {
  // Result whose conversion semantics are retained.
  loom_value_id_t value;
  // Exact interpretation of the source payload.
  loom_integer_cast_t cast;
  // Transitive interpretation composed once during immutable construction.
  loom_integer_cast_t composed;
} loom_conditioned_value_facts_cast_t;

typedef struct loom_conditioned_value_facts_cast_key_t {
  // Immutable conversion rows.
  const loom_conditioned_value_facts_t* state;
  // Conversion result being queried.
  loom_value_id_t value;
} loom_conditioned_value_facts_cast_key_t;

static bool loom_conditioned_value_facts_cast_equal(const void* user_data,
                                                    uint32_t ordinal) {
  const loom_conditioned_value_facts_cast_key_t* key = user_data;
  return key->state->casts[ordinal].value == key->value;
}

static uint32_t loom_conditioned_value_facts_cast_hash(loom_value_id_t value) {
  return loom_structural_hash_finalize(
      loom_structural_hash_mix_u32(loom_structural_hash_initialize(), value));
}

static uint32_t loom_conditioned_value_facts_lookup_cast(
    const loom_conditioned_value_facts_t* state, loom_value_id_t value) {
  const loom_conditioned_value_facts_cast_key_t key = {state, value};
  return loom_intern_table_probe(&state->cast_index,
                                 loom_conditioned_value_facts_cast_hash(value),
                                 loom_conditioned_value_facts_cast_equal, &key)
      .index;
}

static iree_status_t loom_conditioned_value_facts_retain_cast(
    void* user_data, loom_value_id_t value,
    loom_value_fact_integer_cast_t cast) {
  loom_conditioned_value_facts_t* state = user_data;
  // Globally value-preserving conversions already have an indexed numeric
  // representative. Only interpretations that can change value need a row.
  if (loom_local_value_domain_try_ordinal(&state->domain, value) ==
          LOOM_VALUE_ORDINAL_INVALID ||
      loom_value_fact_table_query_numeric_identity(state->table, value) !=
          value) {
    return iree_ok_status();
  }
  const uint32_t hash = loom_conditioned_value_facts_cast_hash(value);
  const loom_conditioned_value_facts_cast_key_t key = {state, value};
  loom_intern_probe_t probe = loom_intern_table_probe(
      &state->cast_index, hash, loom_conditioned_value_facts_cast_equal, &key);
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      &state->arena, &state->cast_index, hash, 1, &probe.slot));
  if (state->cast_index.count == state->cast_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &state->arena, state->cast_index.count, state->cast_index.count + 1,
        sizeof(*state->casts), &state->cast_capacity, (void**)&state->casts));
  }
  const uint32_t ordinal = (uint32_t)state->cast_index.count;
  state->casts[ordinal] = (loom_conditioned_value_facts_cast_t){
      .value = value,
      .cast = cast.immediate,
      .composed = cast.composed,
  };
  loom_intern_table_insert(&state->cast_index, probe.slot, hash, ordinal);
  return iree_ok_status();
}

static iree_status_t loom_conditioned_value_facts_compose_casts(
    loom_conditioned_value_facts_t* state) {
  if (!state->cast_index.count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, state->max_operand_count, sizeof(*state->numeric_operands),
      (void**)&state->numeric_operands));
  bool needs_composition = false;
  for (uint32_t i = 0; i < state->cast_index.count; ++i) {
    loom_conditioned_value_facts_cast_t* row = &state->casts[i];
    const loom_value_id_t immediate = loom_cfg_value_identity_table_lookup(
        &state->identities, row->cast.source_value_id);
    const loom_value_id_t composed = loom_cfg_value_identity_table_lookup(
        &state->identities, row->composed.source_value_id);
    // Ordinary inference already composed the interpretation. Only new CFG
    // equalities can expose additional payloads in this identity domain.
    if (immediate != row->cast.source_value_id ||
        composed != row->composed.source_value_id) {
      row->cast.source_value_id = immediate;
      row->composed = (loom_integer_cast_t){0};
      needs_composition = true;
    }
  }
  if (!needs_composition) {
    return iree_ok_status();
  }
  uint32_t* path = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, state->cast_index.count, sizeof(*path), (void**)&path));
  for (uint32_t i = 0; i < state->cast_index.count; ++i) {
    uint32_t ordinal = i;
    iree_host_size_t path_count = 0;
    while (ordinal != UINT32_MAX && !state->casts[ordinal].composed.bit_count) {
      loom_conditioned_value_facts_cast_t* row = &state->casts[ordinal];
      row->composed = row->cast;
      path[path_count++] = ordinal;
      ordinal = loom_conditioned_value_facts_lookup_cast(
          state, row->cast.source_value_id);
    }
    while (path_count) {
      ordinal = path[--path_count];
      loom_conditioned_value_facts_cast_t* row = &state->casts[ordinal];
      const uint32_t parent = loom_conditioned_value_facts_lookup_cast(
          state, row->cast.source_value_id);
      if (parent != UINT32_MAX && parent != ordinal) {
        row->composed = loom_integer_cast_compose(
            state->casts[parent].composed,
            (loom_integer_cast_kind_t)row->cast.kind, row->cast.bit_count);
      }
    }
  }
  return iree_ok_status();
}

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
      state->max_operand_count =
          iree_max(state->max_operand_count, op->operand_count);
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

// Both summaries constrain the same integer value. Only numeric information
// transfers; execution distribution, topology, and context-local extensions
// remain owned by the destination. This lattice has no unreachable element,
// so an empty intersection retains a conservative input.
static void loom_conditioned_value_facts_apply_active_refinement(
    const loom_value_facts_t* constraint, loom_value_facts_t* destination) {
  loom_value_facts_t result = *destination;
  result.range_lo = iree_max(result.range_lo, constraint->range_lo);
  result.range_hi = iree_min(result.range_hi, constraint->range_hi);
  if (result.range_lo > result.range_hi) {
    return;
  }
  if (result.known_divisor == 1) {
    result.known_divisor = constraint->known_divisor;
  } else if (constraint->known_divisor != 1 &&
             constraint->known_divisor != result.known_divisor) {
    int64_t divisor = 0;
    if (!iree_math_checked_lcm_i64(result.known_divisor,
                                   constraint->known_divisor, &divisor)) {
      // Either individual divisor remains valid if their joint divisor does
      // not fit the compact numeric summary.
      divisor = iree_max(result.known_divisor, constraint->known_divisor);
    }
    result.known_divisor = divisor;
  }
  const uint32_t predicates =
      (result.flags | constraint->flags) &
      (LOOM_VALUE_FACT_NON_ZERO | LOOM_VALUE_FACT_POWER_OF_TWO);
  loom_value_facts_refine_divisible_range(&result);
  if (result.range_lo == result.range_hi &&
      (((predicates & LOOM_VALUE_FACT_NON_ZERO) && result.range_lo == 0) ||
       ((predicates & LOOM_VALUE_FACT_POWER_OF_TWO) &&
        (result.range_lo <= 0 ||
         ((uint64_t)result.range_lo & ((uint64_t)result.range_lo - 1)))))) {
    return;
  }
  loom_value_facts_recompute_flags(&result);
  result.flags |= predicates;
  *destination = result;
}

// Only original providers with a different numeric representative need
// translation. Their current facts are consumed when entering this scope.
IREE_ATTRIBUTE_NOINLINE static void
loom_conditioned_value_facts_apply_provider_relation(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    iree_host_size_t incidence,
    const loom_condition_integer_relation_t* original,
    loom_value_facts_t* facts) {
  loom_condition_integer_relation_t relation = *original;
  if (incidence & LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_LEFT) {
    const loom_value_facts_t provider =
        loom_value_fact_table_lookup(table, relation.left.value_id);
    loom_conditioned_value_facts_apply_active_refinement(&provider, facts);
    relation.left.value_id = value;
  }
  if (incidence & LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_RIGHT) {
    const loom_value_facts_t provider =
        loom_value_fact_table_lookup(table, relation.right.value_id);
    loom_conditioned_value_facts_apply_active_refinement(&provider, facts);
    relation.right.value_id = value;
  }
  (void)loom_condition_integer_relation_apply_to_value_facts(&relation, table,
                                                             value, facts);
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
    const iree_host_size_t* relation_indices =
        (const iree_host_size_t*)(scope->updates + scope->update_count);
    loom_conditioned_value_facts_active_update_t* update = &scope->updates[i];
    update->previous = state->active_updates[update->value_ordinal];
    const loom_value_id_t value_id =
        state->domain.value_ids[update->value_ordinal];
    update->facts = update->previous
                        ? update->previous->facts
                        : loom_value_fact_table_lookup(table, value_id);
    for (uint32_t j = 0; j < update->relation_count; ++j) {
      const iree_host_size_t incidence =
          update->relation_count == 1
              ? update->relation_index
              : relation_indices[update->relation_index + j];
      const loom_condition_integer_relation_t* relation =
          &scope->facts->integer_relations[incidence >> 2];
      if (incidence & LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_MASK) {
        loom_conditioned_value_facts_apply_provider_relation(
            table, value_id, incidence, relation, &update->facts);
      } else {
        (void)loom_condition_integer_relation_apply_to_value_facts(
            relation, table, value_id, &update->facts);
      }
    }
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

static void loom_conditioned_value_facts_refine_canonical_value(
    const loom_conditioned_value_facts_t* state,
    const loom_conditioned_value_facts_region_t* entry,
    const loom_value_fact_table_t* table, loom_value_id_t canonical,
    loom_value_facts_t* inout_facts) {
  if (state->active_updates) {
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_try_ordinal(&state->domain, canonical);
    if (ordinal != LOOM_VALUE_ORDINAL_INVALID &&
        state->active_updates[ordinal]) {
      loom_conditioned_value_facts_apply_active_refinement(
          &state->active_updates[ordinal]->facts, inout_facts);
    }
  }
  loom_condition_fact_scope_apply_to_value_facts(entry->projected_scope, table,
                                                 canonical, inout_facts);
}

static loom_value_id_t loom_conditioned_value_facts_refine_cast(
    const loom_conditioned_value_facts_t* state,
    const loom_conditioned_value_facts_region_t* entry,
    const loom_value_fact_table_t* table, const loom_integer_cast_t* cast,
    loom_value_facts_t* inout_facts) {
  loom_value_facts_t source_facts =
      loom_value_fact_table_lookup(table, cast->source_value_id);
  loom_conditioned_value_facts_refine_canonical_value(
      state, entry, table, cast->source_value_id, &source_facts);
  const loom_value_facts_t transferred =
      loom_integer_cast_apply(cast, source_facts);
  loom_conditioned_value_facts_apply_active_refinement(&transferred,
                                                       inout_facts);
  return loom_integer_cast_is_identity(cast, source_facts)
             ? cast->source_value_id
             : LOOM_VALUE_ID_INVALID;
}

static const loom_conditioned_value_facts_region_t*
loom_conditioned_value_facts_enter_operation(
    loom_conditioned_value_facts_t* state, const loom_value_fact_table_t* table,
    const loom_op_t* op) {
  const loom_region_t* region = op->parent_block->parent_region;
  const iree_host_size_t bucket =
      loom_conditioned_value_facts_bucket(state, region);
  const loom_conditioned_value_facts_region_t* entry = state->buckets[bucket];
  while (entry && entry->region != region) {
    entry = entry->next_bucket;
  }
  if (entry && state->local_condition_scope_count) {
    loom_conditioned_value_facts_transition_local_scope(
        state, &entry->local_scopes[op->parent_block->region_index], table);
  }
  return entry;
}

static loom_value_id_t loom_conditioned_value_facts_refine_reference(
    loom_conditioned_value_facts_t* state,
    const loom_conditioned_value_facts_region_t* entry,
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_facts_t* facts) {
  const loom_value_id_t canonical =
      loom_cfg_value_identity_table_lookup(&state->identities, value);
  loom_conditioned_value_facts_refine_canonical_value(state, entry, table,
                                                      canonical, facts);
  loom_value_id_t representative = canonical;
  const uint32_t ordinal =
      loom_conditioned_value_facts_lookup_cast(state, canonical);
  if (ordinal != UINT32_MAX) {
    const loom_conditioned_value_facts_cast_t* row = &state->casts[ordinal];
    loom_value_id_t numeric = loom_conditioned_value_facts_refine_cast(
        state, entry, table, &row->cast, facts);
    if (numeric != LOOM_VALUE_ID_INVALID) {
      representative = numeric;
    }
    if (row->composed.source_value_id != row->cast.source_value_id) {
      numeric = loom_conditioned_value_facts_refine_cast(state, entry, table,
                                                         &row->composed, facts);
      if (numeric != LOOM_VALUE_ID_INVALID) {
        representative = numeric;
      }
    }
  }
  if (!table->exact_scalar_origins.entries) {
    return representative;
  }
  // The producer retains the transitive origin. Every SSA reference observes
  // that origin in its use scope, including references inside predicates.
  loom_value_fact_exact_scalar_origin_t origin;
  if (!loom_value_fact_table_query_exact_scalar_origin(table, state->module,
                                                       canonical, &origin)) {
    return representative;
  }
  const loom_value_id_t source = loom_cfg_value_identity_table_lookup(
      &state->identities, origin.source_value_id);
  loom_value_facts_t source_facts = loom_value_fact_table_lookup(table, source);
  loom_conditioned_value_facts_refine_canonical_value(state, entry, table,
                                                      source, &source_facts);
  const loom_scalar_type_t source_type =
      loom_type_element_type(loom_module_value_type(state->module, source));
  const loom_scalar_type_t result_type =
      loom_type_element_type(loom_module_value_type(state->module, canonical));
  const loom_value_facts_t transferred =
      loom_index_cast_transfer_facts(source_type, result_type, source_facts);
  loom_conditioned_value_facts_apply_active_refinement(&transferred, facts);
  return representative;
}

static loom_value_id_t loom_conditioned_value_facts_refine_value(
    void* user_data, const loom_value_fact_table_t* table, const loom_op_t* op,
    loom_value_id_t value, loom_value_facts_t* facts) {
  loom_conditioned_value_facts_t* state = user_data;
  const loom_conditioned_value_facts_region_t* entry =
      loom_conditioned_value_facts_enter_operation(state, table, op);
  return entry ? loom_conditioned_value_facts_refine_reference(
                     state, entry, table, value, facts)
               : value;
}

static const loom_value_id_t* loom_conditioned_value_facts_refine_operands(
    void* user_data, const loom_value_fact_table_t* table, const loom_op_t* op,
    loom_value_facts_t* operands) {
  loom_conditioned_value_facts_t* state = user_data;
  const loom_conditioned_value_facts_region_t* entry =
      loom_conditioned_value_facts_enter_operation(state, table, op);
  const loom_value_id_t* values = loom_op_const_operands(op);
  // Projected configuration regions lie outside the function body domain.
  if (!entry) {
    return values;
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t numeric =
        loom_conditioned_value_facts_refine_reference(state, entry, table,
                                                      values[i], &operands[i]);
    if (state->numeric_operands) {
      state->numeric_operands[i] = numeric;
    }
  }
  return state->numeric_operands ? state->numeric_operands : values;
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
          &state->identities, loom_value_fact_table_query_numeric_identity(
                                  state->table, canonical));
      if (next == canonical) {
        break;
      }
      canonical = next;
    }
    state->identities.representatives[ordinal] =
        canonical == original ? LOOM_VALUE_ID_INVALID : canonical;
  }
}

static loom_value_ordinal_t loom_conditioned_value_facts_update_ordinal(
    const loom_conditioned_value_facts_t* state,
    loom_condition_integer_operand_t operand) {
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    operand.value_id = loom_cfg_value_identity_table_lookup(&state->identities,
                                                            operand.value_id);
  }
  if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      loom_value_facts_is_exact(
          loom_value_fact_table_lookup(state->table, operand.value_id))) {
    return LOOM_VALUE_ORDINAL_INVALID;
  }
  return loom_local_value_domain_ordinal(&state->domain, operand.value_id);
}

static uint32_t loom_conditioned_value_facts_relation_update_ordinals(
    const loom_conditioned_value_facts_t* state,
    const loom_condition_integer_relation_t* relation,
    loom_value_ordinal_t ordinals[2], uint32_t translations[2]) {
  const loom_value_ordinal_t left =
      loom_conditioned_value_facts_update_ordinal(state, relation->left);
  const loom_value_ordinal_t right =
      loom_conditioned_value_facts_update_ordinal(state, relation->right);
  uint32_t count = 0;
  if (left != LOOM_VALUE_ORDINAL_INVALID) {
    ordinals[count] = left;
    translations[count++] =
        relation->left.value_id == state->domain.value_ids[left]
            ? 0
            : LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_LEFT;
  }
  if (right != LOOM_VALUE_ORDINAL_INVALID) {
    const uint32_t translation =
        relation->right.value_id == state->domain.value_ids[right]
            ? 0
            : LOOM_CONDITIONED_VALUE_FACTS_TRANSLATE_RIGHT;
    if (right == left) {
      translations[0] |= translation;
    } else {
      ordinals[count] = right;
      translations[count++] = translation;
    }
  }
  return count;
}

static iree_status_t loom_conditioned_value_facts_prepare_single_relation(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* scope) {
  loom_value_ordinal_t ordinals[2];
  uint32_t translations[2];
  const uint32_t count = loom_conditioned_value_facts_relation_update_ordinals(
      state, scope->facts->integer_relations, ordinals, translations);
  if (!count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &state->arena, count, sizeof(*scope->updates), (void**)&scope->updates));
  scope->update_count = count;
  for (uint32_t i = 0; i < count; ++i) {
    scope->updates[i] = (loom_conditioned_value_facts_active_update_t){
        .relation_index = translations[i],
        .relation_count = 1,
        .value_ordinal = ordinals[i],
    };
  }
  return iree_ok_status();
}

// Construction associates each non-exact endpoint with its source relation.
// Sorting by value and then relation keeps each value's refinement order while
// bounding temporary storage by the largest condition, not the whole function.
typedef struct loom_conditioned_value_facts_incidence_t {
  // Dense ordinal of the value refined by the relation.
  loom_value_ordinal_t value_ordinal;
  // Relation index shifted by two, followed by left/right translation bits.
  iree_host_size_t relation_index;
} loom_conditioned_value_facts_incidence_t;

static bool loom_conditioned_value_facts_incidence_less(
    const loom_conditioned_value_facts_incidence_t* left,
    const loom_conditioned_value_facts_incidence_t* right) {
  return left->value_ordinal < right->value_ordinal ||
         (left->value_ordinal == right->value_ordinal &&
          left->relation_index < right->relation_index);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_conditioned_value_facts_sort_incidences,
                          loom_conditioned_value_facts_incidence_t,
                          loom_conditioned_value_facts_incidence_less)

static iree_status_t loom_conditioned_value_facts_prepare_local_scope(
    loom_conditioned_value_facts_t* state,
    loom_conditioned_value_facts_local_scope_t* scope,
    iree_arena_allocator_t* scratch, iree_host_size_t* incidence_capacity,
    loom_conditioned_value_facts_incidence_t** incidences) {
  const loom_condition_fact_set_t* facts = scope->facts;
  iree_host_size_t maximum_incidence_count = 0;
  if (!iree_host_size_checked_mul(facts->integer_relation_count, 2,
                                  &maximum_incidence_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition incidence storage size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      scratch, 0, maximum_incidence_count, sizeof(**incidences),
      incidence_capacity, (void**)incidences));
  iree_host_size_t incidence_count = 0;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    loom_value_ordinal_t ordinals[2];
    uint32_t translations[2];
    const uint32_t count =
        loom_conditioned_value_facts_relation_update_ordinals(
            state, &facts->integer_relations[i], ordinals, translations);
    for (uint32_t j = 0; j < count; ++j) {
      (*incidences)[incidence_count++] =
          (loom_conditioned_value_facts_incidence_t){
              .value_ordinal = ordinals[j],
              .relation_index = (i << 2) | translations[j],
          };
    }
  }
  if (!incidence_count) {
    return iree_ok_status();
  }
  loom_conditioned_value_facts_sort_incidences(*incidences, incidence_count);
  iree_host_size_t update_count = 0;
  iree_host_size_t index_count = 0;
  for (iree_host_size_t begin = 0; begin < incidence_count;) {
    iree_host_size_t end = begin + 1;
    while (end < incidence_count && (*incidences)[end].value_ordinal ==
                                        (*incidences)[begin].value_ordinal) {
      ++end;
    }
    ++update_count;
    if (end - begin > 1) {
      index_count += end - begin;
    }
    begin = end;
  }
  iree_host_size_t update_bytes = 0;
  iree_host_size_t index_bytes = 0;
  iree_host_size_t allocation_bytes = 0;
  if (!iree_host_size_checked_mul(update_count, sizeof(*scope->updates),
                                  &update_bytes) ||
      !iree_host_size_checked_mul(index_count, sizeof(iree_host_size_t),
                                  &index_bytes) ||
      !iree_host_size_checked_add(update_bytes, index_bytes,
                                  &allocation_bytes)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition refinement storage size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&state->arena, allocation_bytes,
                                           (void**)&scope->updates));
  iree_host_size_t* relation_indices =
      (iree_host_size_t*)(scope->updates + update_count);
  iree_host_size_t next_index = 0;
  for (iree_host_size_t begin = 0; begin < incidence_count;) {
    iree_host_size_t end = begin + 1;
    while (end < incidence_count && (*incidences)[end].value_ordinal ==
                                        (*incidences)[begin].value_ordinal) {
      ++end;
    }
    const iree_host_size_t degree = end - begin;
    IREE_ASSERT_LE(degree, UINT32_MAX);
    scope->updates[scope->update_count++] =
        (loom_conditioned_value_facts_active_update_t){
            .relation_index =
                degree == 1 ? (*incidences)[begin].relation_index : next_index,
            .relation_count = (uint32_t)degree,
            .value_ordinal = (*incidences)[begin].value_ordinal,
        };
    if (degree > 1) {
      for (iree_host_size_t i = begin; i < end; ++i) {
        relation_indices[next_index++] = (*incidences)[i].relation_index;
      }
    }
    begin = end;
  }
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
  iree_arena_allocator_t scratch;
  iree_arena_initialize(state->arena.block_pool, &scratch);
  iree_host_size_t incidence_capacity = 0;
  loom_conditioned_value_facts_incidence_t* incidences = NULL;
  iree_status_t status = iree_ok_status();
  for (loom_conditioned_value_facts_local_scope_t* scope = state->local_scopes;
       scope && iree_status_is_ok(status); scope = scope->next) {
    if (!scope->facts || !scope->facts->integer_relation_count) {
      continue;
    }
    if (scope->facts->integer_relation_count == 1) {
      status =
          loom_conditioned_value_facts_prepare_single_relation(state, scope);
      continue;
    }
    status = loom_conditioned_value_facts_prepare_local_scope(
        state, scope, &scratch, &incidence_capacity, &incidences);
  }
  iree_arena_deinitialize(&scratch);
  return status;
}

static iree_status_t loom_conditioned_value_facts_solve(
    loom_conditioned_value_facts_t* state, loom_func_like_t function) {
  state->dominance = (loom_dominance_info_t){
      .module = state->module,
      .arena = &state->arena,
  };
  state->bucket_count = state->table->regions.bucket_count;
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(&state->arena, 0, &state->cast_index));
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
  IREE_RETURN_IF_ERROR(loom_value_fact_table_for_each_integer_cast(
      state->table, loom_conditioned_value_facts_retain_cast, state));
  IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_compose_casts(state));
  if (state->local_condition_scope_count) {
    IREE_RETURN_IF_ERROR(
        loom_conditioned_value_facts_prepare_local_scopes(state));
  }
  state->table->has_conditioned_results = true;
  state->table->context.refine_operands.user_data = state;
  state->table->context.refine_operands.fn =
      loom_conditioned_value_facts_refine_operands;
  state->table->context.refine_operands.value_fn =
      loom_conditioned_value_facts_refine_value;
  iree_status_t status =
      loom_value_fact_table_compute(state->table, state->module, function);
  state->table->context.refine_operands.fn = NULL;
  state->table->context.refine_operands.value_fn = NULL;
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
