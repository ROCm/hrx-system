// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/canonicalizer.h"

#include <string.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/greedy.h"
#include "loom/rewrite/rewriter.h"
#include "loom/rewrite/type_propagation.h"
#include "loom/target/facts.h"
#include "loom/transforms/cleanup/fact_refinement.h"
#include "loom/transforms/cleanup/patterns.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

loom_canonicalizer_pattern_registries_t
loom_canonicalizer_pattern_registries_from_cleanup_registry(
    const loom_cleanup_pattern_registry_t* registry) {
  if (registry == NULL) {
    return (loom_canonicalizer_pattern_registries_t){0};
  }
  return (loom_canonicalizer_pattern_registries_t){
      .region_initialization = registry->region_initialization,
      .pre_fold = registry->universal_pre_fold,
      .post_type = registry->universal_post_type,
  };
}

static bool loom_canonicalize_value_has_empty_materializer(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  return policy->type_has_empty_materializer(
      loom_module_value_type(module, value_id));
}

static bool loom_canonicalize_op_has_poison_operand(const loom_module_t* module,
                                                    const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    if (!loom_value_is_block_arg(value) &&
        loom_traits_are_poison(loom_value_def_op(value)->traits)) {
      return true;
    }
  }
  return false;
}

static bool loom_canonicalize_can_replace_results_with_poison(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, const loom_op_t* op) {
  if (op->result_count == 0) {
    return false;
  }
  if (op->region_count != 0) {
    return false;
  }
  if (op->tied_result_count != 0) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      return false;
    }
    loom_type_t type = loom_module_value_type(module, results[i]);
    if (!policy->type_has_poison_materializer(type)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_canonicalize_try_propagate_poison(
    const loom_cleanup_special_value_policy_t* policy,
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_propagated) {
  *out_propagated = false;
  if (policy == NULL || !loom_module_has_poison(rewriter->module)) {
    return iree_ok_status();
  }
  loom_trait_flags_t traits = loom_op_effective_traits(rewriter->module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    return iree_ok_status();
  }
  if (loom_traits_are_convergent(traits)) {
    return iree_ok_status();
  }
  if (!loom_canonicalize_op_has_poison_operand(rewriter->module, op)) {
    return iree_ok_status();
  }
  if (!loom_canonicalize_can_replace_results_with_poison(
          policy, rewriter->module, op)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_results_with_materialized_values_and_erase(
          rewriter, op, policy->materialize_poison));
  *out_propagated = true;
  return iree_ok_status();
}

static bool loom_canonicalize_can_replace_results_with_empty(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, const loom_op_t* op) {
  if (policy->op_is_empty(op) || loom_traits_are_poison(op->traits)) {
    return false;
  }
  if (op->result_count == 0) {
    return false;
  }
  if (op->region_count != 0) {
    return false;
  }
  if (op->tied_result_count != 0) {
    return false;
  }

  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    return false;
  }
  if (loom_traits_are_convergent(traits)) {
    return false;
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      return false;
    }
    loom_type_t type = loom_module_value_type(module, results[i]);
    if (!policy->type_has_empty_materializer(type)) {
      return false;
    }
  }
  return true;
}

static bool loom_canonicalize_optional_empty_value(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, loom_value_id_t value_id) {
  return value_id == LOOM_VALUE_ID_INVALID ||
         loom_canonicalize_value_has_empty_materializer(policy, module,
                                                        value_id);
}

static bool loom_canonicalize_required_empty_value(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, loom_value_id_t value_id) {
  return value_id != LOOM_VALUE_ID_INVALID &&
         loom_canonicalize_value_has_empty_materializer(policy, module,
                                                        value_id);
}

static bool loom_canonicalize_memory_access_results_are_empty(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, const loom_op_t* op) {
  if (op->result_count == 0) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!loom_canonicalize_required_empty_value(policy, module, results[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_canonicalize_memory_access_optional_roles_are_empty(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, loom_memory_access_t access) {
  return loom_canonicalize_optional_empty_value(
             policy, module, loom_memory_access_mask(access)) &&
         loom_canonicalize_optional_empty_value(
             policy, module, loom_memory_access_passthrough(access)) &&
         loom_canonicalize_optional_empty_value(
             policy, module, loom_memory_access_offsets(access));
}

static bool loom_canonicalize_memory_access_has_empty_footprint(
    const loom_cleanup_special_value_policy_t* policy,
    const loom_module_t* module, const loom_op_t* op,
    loom_memory_access_t access) {
  if (!loom_canonicalize_memory_access_optional_roles_are_empty(policy, module,
                                                                access)) {
    return false;
  }

  switch (loom_memory_access_operation_kind(access)) {
    case LOOM_MEMORY_ACCESS_OPERATION_LOAD:
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_LOAD:
      return loom_canonicalize_memory_access_results_are_empty(policy, module,
                                                               op);
    case LOOM_MEMORY_ACCESS_OPERATION_STORE:
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_STORE:
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_REDUCE:
      return loom_canonicalize_required_empty_value(
          policy, module, loom_memory_access_value(access));
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_RMW:
      return loom_canonicalize_memory_access_results_are_empty(policy, module,
                                                               op) &&
             loom_canonicalize_required_empty_value(
                 policy, module, loom_memory_access_value(access));
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG:
      return loom_canonicalize_memory_access_results_are_empty(policy, module,
                                                               op) &&
             loom_canonicalize_required_empty_value(
                 policy, module, loom_memory_access_expected(access)) &&
             loom_canonicalize_required_empty_value(
                 policy, module, loom_memory_access_replacement(access));
    case LOOM_MEMORY_ACCESS_OPERATION_PREFETCH:
    case LOOM_MEMORY_ACCESS_OPERATION_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_canonicalize_try_elide_empty_memory_effect(
    const loom_cleanup_special_value_policy_t* policy,
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_elided) {
  *out_elided = false;

  loom_memory_access_t access = loom_memory_access_cast(rewriter->module, op);
  if (!loom_memory_access_isa(access)) {
    return iree_ok_status();
  }
  if (!loom_canonicalize_memory_access_has_empty_footprint(
          policy, rewriter->module, op, access)) {
    return iree_ok_status();
  }

  if (op->result_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_results_with_materialized_values_and_erase(
            rewriter, op, policy->materialize_empty));
    *out_elided = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_elided = true;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_try_elide_empty_op(
    const loom_cleanup_special_value_policy_t* policy,
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_elided) {
  *out_elided = false;
  if (policy == NULL) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_elide_empty_memory_effect(
      policy, rewriter, op, out_elided));
  if (*out_elided) {
    return iree_ok_status();
  }

  if (!loom_canonicalize_can_replace_results_with_empty(policy,
                                                        rewriter->module, op)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_results_with_materialized_values_and_erase(
          rewriter, op, policy->materialize_empty));
  *out_elided = true;
  return iree_ok_status();
}

struct loom_canonicalizer_state_t {
  // Shared greedy rewrite session used by canonicalize region runs.
  loom_greedy_rewrite_driver_t rewrite_driver;
};

static void loom_canonicalizer_reset_run_state(
    loom_canonicalizer_t* canonicalizer) {
  if (canonicalizer->state) {
    loom_greedy_rewrite_driver_reset(&canonicalizer->state->rewrite_driver);
    loom_greedy_rewrite_driver_set_fact_table(
        &canonicalizer->state->rewrite_driver, NULL);
  } else if (canonicalizer->scratch_arena_initialized) {
    iree_arena_reset(&canonicalizer->scratch_arena);
  }
}

static void loom_canonicalizer_merge_result(
    loom_canonicalizer_result_t* target,
    const loom_canonicalizer_result_t* source) {
  if (!target || !source) {
    return;
  }
  target->changed |= source->changed;
  target->facts_changed |= source->facts_changed;
  target->types_changed |= source->types_changed;
  target->boundary_maybe_changed |= source->boundary_maybe_changed;
  target->ops_modified += source->ops_modified;
  target->type_propagation_conflicts += source->type_propagation_conflicts;
  target->type_propagation_rejection_cache_hits +=
      source->type_propagation_rejection_cache_hits;
}

iree_status_t loom_canonicalizer_initialize(
    loom_module_t* module, iree_arena_allocator_t* parent_arena,
    loom_pass_value_fact_owner_t* value_facts,
    const loom_cleanup_special_value_policy_t* special_value_policy,
    const loom_fact_refinement_policy_t* fact_refinement_policy,
    loom_canonicalizer_t* out_canonicalizer) {
  memset(out_canonicalizer, 0, sizeof(*out_canonicalizer));
  loom_canonicalizer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(parent_arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  out_canonicalizer->module = module;
  out_canonicalizer->value_facts = value_facts;
  out_canonicalizer->special_value_policy = special_value_policy;
  out_canonicalizer->fact_refinement_policy = fact_refinement_policy;
  out_canonicalizer->parent_arena = parent_arena;
  out_canonicalizer->state = state;
  iree_arena_initialize(parent_arena->block_pool,
                        &out_canonicalizer->scratch_arena);
  out_canonicalizer->scratch_arena_initialized = true;
  loom_greedy_rewrite_driver_initialize(
      module, &out_canonicalizer->scratch_arena,
      /*fact_table=*/NULL, &state->rewrite_driver);
  return iree_ok_status();
}

void loom_canonicalizer_deinitialize(loom_canonicalizer_t* canonicalizer) {
  if (!canonicalizer) {
    return;
  }
  loom_canonicalizer_reset_run_state(canonicalizer);
  if (canonicalizer->scratch_arena_initialized) {
    iree_arena_deinitialize(&canonicalizer->scratch_arena);
  }
  memset(canonicalizer, 0, sizeof(*canonicalizer));
}

const loom_value_fact_table_t* loom_canonicalizer_fact_table(
    const loom_canonicalizer_t* canonicalizer) {
  if (!canonicalizer || !canonicalizer->state) {
    return NULL;
  }
  return loom_greedy_rewrite_driver_fact_table(
      &canonicalizer->state->rewrite_driver);
}

typedef struct loom_canonicalize_rewrite_state_t {
  // Compiler-selected special-value policy, or NULL.
  const loom_cleanup_special_value_policy_t* special_value_policy;

  // Dialect-composed relation preservation selected once for this region run.
  const loom_fact_refinement_policy_t* fact_refinement_policy;

  // Symbolic expression context for exact address/integer cleanup.
  loom_symbolic_expr_context_t expression_context;

  // Table-driven type propagator for this region run.
  loom_type_propagator_t* type_propagator;

  // Cumulative type propagation activity captured before region cleanup.
  loom_type_propagator_statistics_t type_propagator_statistics;

  // Borrowed whole-module owner permitting callable boundary type changes.
  loom_type_propagator_boundary_callback_t refine_boundary;

  // Optional phase-specific pattern registries sharing this rewrite session.
  loom_canonicalizer_pattern_registries_t patterns;

  // Invocation-local context shared by indexed cleanup patterns.
  loom_cleanup_pattern_context_t pattern_context;

  // True after expression_context has been initialized.
  bool expression_context_initialized;

  // True after ordered region-initialization patterns have run.
  bool region_initialization_complete;
} loom_canonicalize_rewrite_state_t;

static iree_status_t loom_canonicalize_preserve_pending_exact_relations(
    void* user_data, loom_rewriter_t* rewriter) {
  return loom_fact_refinement_preserve_pending(
      rewriter, (const loom_fact_refinement_policy_t*)user_data);
}

static iree_status_t loom_canonicalize_prepare_region(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  loom_canonicalize_rewrite_state_t* state =
      (loom_canonicalize_rewrite_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_type_propagator_allocate(
      driver->module, state->refine_boundary, driver->scratch_arena,
      &state->type_propagator));
  IREE_RETURN_IF_ERROR(loom_type_propagator_prepare_region(
      state->type_propagator, region, parent_op));
  loom_symbolic_expr_context_initialize(
      driver->module, loom_type_propagator_value_domain(state->type_propagator),
      driver->rewriter.fact_table, driver->scratch_arena,
      &state->expression_context);
  state->pattern_context.symbolic_expression_context =
      &state->expression_context;
  state->expression_context_initialized = true;
  return iree_ok_status();
}

static void loom_canonicalize_cleanup_region(
    void* user_data, loom_greedy_rewrite_driver_t* driver) {
  loom_canonicalize_rewrite_state_t* state =
      (loom_canonicalize_rewrite_state_t*)user_data;
  state->type_propagator_statistics =
      loom_type_propagator_statistics(state->type_propagator);
  loom_type_propagator_deinitialize(state->type_propagator);
  state->type_propagator = NULL;
  state->pattern_context.symbolic_expression_context = NULL;
  state->expression_context_initialized = false;
}

static iree_status_t loom_canonicalize_apply_patterns(
    loom_canonicalize_rewrite_state_t* state,
    const loom_rewrite_pattern_registry_t* registry, loom_op_t* op,
    loom_rewriter_t* rewriter, loom_greedy_rewrite_result_t* result,
    bool* out_changed) {
  *out_changed = false;
  if (registry == NULL) {
    return iree_ok_status();
  }
  rewriter->flags = 0;
  IREE_RETURN_IF_ERROR(loom_rewrite_pattern_registry_apply(
      registry, &state->pattern_context, op, rewriter, out_changed));
  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);
  if (*out_changed) {
    loom_greedy_rewrite_result_record_change(
        result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
  }
  return iree_ok_status();
}

static void loom_canonicalize_reset_symbolic_context(
    void* user_data, loom_greedy_rewrite_driver_t* driver) {
  loom_canonicalize_rewrite_state_t* state =
      (loom_canonicalize_rewrite_state_t*)user_data;
  if (state->expression_context_initialized) {
    loom_symbolic_expr_context_reset(&state->expression_context);
  }
}

typedef struct loom_canonicalize_region_initialization_t {
  // Canonicalizer state supplying the configured pattern registry and context.
  loom_canonicalize_rewrite_state_t* state;
  // Active rewriter receiving initialization mutations.
  loom_rewriter_t* rewriter;
  // Greedy result receiving mutation and fact accounting.
  loom_greedy_rewrite_result_t* result;
  // True when at least one initialization pattern changed the region.
  bool changed;
} loom_canonicalize_region_initialization_t;

static iree_status_t loom_canonicalize_initialize_region_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_canonicalize_region_initialization_t* initialization =
      (loom_canonicalize_region_initialization_t*)user_data;
  bool changed = false;
  IREE_RETURN_IF_ERROR(loom_canonicalize_apply_patterns(
      initialization->state,
      initialization->state->patterns.region_initialization, op,
      initialization->rewriter, initialization->result, &changed));
  initialization->changed |= changed;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_begin_iteration(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_region_t* region, loom_greedy_rewrite_result_t* result,
    bool* out_changed) {
  loom_canonicalize_rewrite_state_t* state =
      (loom_canonicalize_rewrite_state_t*)user_data;
  loom_type_propagator_begin_iteration(state->type_propagator);
  *out_changed = false;
  const loom_rewrite_pattern_registry_t* initialization_registry =
      state->patterns.region_initialization;
  if (!state->region_initialization_complete && initialization_registry &&
      initialization_registry->pattern_count > 0) {
    loom_canonicalize_region_initialization_t initialization = {
        .state = state,
        .rewriter = &driver->rewriter,
        .result = result,
    };
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_RETURN_IF_ERROR(loom_walk_region(
        driver->module, region, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_canonicalize_initialize_region_op,
                               &initialization},
        &walk_result));
    *out_changed = initialization.changed;
  }
  state->region_initialization_complete = true;

  if (state->fact_refinement_policy &&
      loom_value_fact_table_has_pending_exact_relations(
          driver->rewriter.fact_table)) {
    driver->rewriter.flags = 0;
    IREE_RETURN_IF_ERROR(loom_fact_refinement_preserve_pending(
        &driver->rewriter, state->fact_refinement_policy));
    if (iree_any_bit_set(driver->rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
      loom_greedy_rewrite_result_record_change(
          result, &driver->rewriter,
          LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_rewrite_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_canonicalize_rewrite_state_t* state =
      (loom_canonicalize_rewrite_state_t*)user_data;
  loom_rewriter_t* rewriter = &driver->rewriter;

  // Mini-DCE: erase trivially dead ops before fold/canonicalize.
  bool erased = false;
  rewriter->flags = 0;
  IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, op, &erased));
  if (erased) {
    loom_greedy_rewrite_result_record_change(
        result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_NONE);
    *out_changed = true;
    return iree_ok_status();
  }

  bool pattern_changed = false;
  IREE_RETURN_IF_ERROR(loom_canonicalize_apply_patterns(
      state, state->patterns.pre_fold, op, rewriter, result, &pattern_changed));
  if (pattern_changed) {
    *out_changed = true;
    return iree_ok_status();
  }

  // Static empty aggregates have valid empty values, not poison. Handle them
  // before poison propagation so zero-footprint effects disappear without
  // observing operands. Dialect-specific empty rules run in the pre-fold
  // registry immediately above.
  bool empty_elided = false;
  rewriter->flags = 0;
  IREE_RETURN_IF_ERROR(loom_canonicalize_try_elide_empty_op(
      state->special_value_policy, rewriter, op, &empty_elided));
  if (empty_elided) {
    loom_greedy_rewrite_result_record_change(
        result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
    return iree_ok_status();
  }

  // Poison propagation: pure scalar/vector-result ops with any poison
  // operand become typed poison values. Boundary diagnostics decide later
  // whether the remaining poison is observable.
  bool poison_propagated = false;
  rewriter->flags = 0;
  IREE_RETURN_IF_ERROR(loom_canonicalize_try_propagate_poison(
      state->special_value_policy, rewriter, op, &poison_propagated));
  if (poison_propagated) {
    loom_greedy_rewrite_result_record_change(
        result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
    return iree_ok_status();
  }

  // Try fold: constant fold via facts and replace with constants.
  bool folded = false;
  rewriter->flags = 0;
  IREE_RETURN_IF_ERROR(loom_rewriter_try_fold(rewriter, op, &folded));
  if (folded) {
    loom_greedy_rewrite_result_record_change(
        result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
    return iree_ok_status();
  }
  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);

  const loom_op_vtable_t* vtable = loom_op_vtable(driver->module, op);

  // Table-driven type propagation: generated equality constraints and
  // value facts can narrow dynamic shapes, encoding roles, and static
  // attachments without hand-writing one pattern per op family.
  bool types_propagated = false;
  if (loom_type_propagator_may_apply_op(state->type_propagator, rewriter, op,
                                        vtable)) {
    rewriter->flags = 0;
    IREE_RETURN_IF_ERROR(loom_type_propagator_apply_op(
        state->type_propagator, rewriter, op, &types_propagated));
    if (types_propagated) {
      loom_greedy_rewrite_result_record_change(
          result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
      *out_changed = true;
      return iree_ok_status();
    }
  }

  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);
  IREE_RETURN_IF_ERROR(
      loom_canonicalize_apply_patterns(state, state->patterns.post_type, op,
                                       rewriter, result, &pattern_changed));
  if (pattern_changed) {
    *out_changed = true;
    return iree_ok_status();
  }

  // Structural canonicalization patterns.
  if (vtable && vtable->canonicalize) {
    rewriter->flags = 0;
    IREE_RETURN_IF_ERROR(vtable->canonicalize(op, rewriter));
    if (iree_any_bit_set(rewriter->flags, LOOM_REWRITER_FLAG_CHANGED)) {
      loom_greedy_rewrite_result_record_change(
          result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
      *out_changed = true;
      return iree_ok_status();
    }
  }
  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);
  IREE_RETURN_IF_ERROR(loom_canonicalize_apply_patterns(
      state, state->patterns.post_canonicalization, op, rewriter, result,
      &pattern_changed));
  if (pattern_changed) {
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static void loom_canonicalizer_import_greedy_result(
    const loom_greedy_rewrite_result_t* source,
    loom_canonicalizer_result_t* target) {
  if (!source || !target) {
    return;
  }
  *target = (loom_canonicalizer_result_t){
      .changed = source->changed,
      .facts_changed = source->facts_changed,
      .types_changed = source->types_changed,
      .boundary_maybe_changed = source->boundary_maybe_changed,
      .ops_modified = source->ops_modified,
  };
}

static iree_status_t loom_canonicalizer_run_precomputed_region(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result) {
  if (out_result) {
    memset(out_result, 0, sizeof(*out_result));
  }
  loom_canonicalize_rewrite_state_t state = {
      .special_value_policy = canonicalizer->special_value_policy,
      .fact_refinement_policy = canonicalizer->fact_refinement_policy,
      .patterns = options ? options->patterns
                          : (loom_canonicalizer_pattern_registries_t){0},
      .refine_boundary = options
                             ? options->refine_boundary
                             : (loom_type_propagator_boundary_callback_t){0},
  };
  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_CANONICALIZER_DEFAULT_MAX_ITERATIONS;
  loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = max_iterations,
      .materialize_constant =
          canonicalizer->special_value_policy
              ? canonicalizer->special_value_policy->materialize_constant
              : NULL,
      .pending_exact_relations_callback =
          {
              .user_data = (void*)canonicalizer->fact_refinement_policy,
              .fn = canonicalizer->fact_refinement_policy
                        ? loom_canonicalize_preserve_pending_exact_relations
                        : NULL,
          },
      .math_policy = options ? options->math_policy : NULL,
      .math_target_bundle =
          options ? loom_target_facts_bundle(options->target_facts) : NULL,
  };
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .prepare_region = loom_canonicalize_prepare_region,
      .cleanup_region = loom_canonicalize_cleanup_region,
      .before_worklist = loom_canonicalize_begin_iteration,
      .rewrite_op = loom_canonicalize_rewrite_op,
      .changed = loom_canonicalize_reset_symbolic_context,
  };
  loom_greedy_rewrite_result_t rewrite_result = {0};
  iree_status_t status = loom_greedy_rewrite_run_region(
      &canonicalizer->state->rewrite_driver, function, region, parent_op,
      &rewrite_options, &callbacks, &rewrite_result);
  if (iree_status_is_ok(status)) {
    loom_canonicalizer_import_greedy_result(&rewrite_result, out_result);
    if (out_result) {
      out_result->type_propagation_conflicts =
          (int64_t)state.type_propagator_statistics.conflict_count;
      out_result->type_propagation_rejection_cache_hits =
          (int64_t)state.type_propagator_statistics.rejection_cache_hit_count;
    }
  }
  return status;
}

static iree_status_t loom_canonicalizer_prepare_region_facts(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_canonicalizer_options_t* options) {
  loom_value_fact_table_t* facts = NULL;
  loom_pass_value_fact_scope_t scope =
      loom_pass_value_fact_scope_region_for_target(
          function, region, parent_op, options ? options->target_facts : NULL);
  if (options) {
    scope.seed_facts = options->seed_facts;
  }
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare(
      canonicalizer->value_facts, canonicalizer->module, scope, &facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region(
      facts, canonicalizer->module, function, region, parent_op));
  loom_greedy_rewrite_driver_set_fact_table(
      &canonicalizer->state->rewrite_driver, facts);
  return iree_ok_status();
}

static iree_status_t loom_canonicalizer_prepare_function_facts(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    const loom_canonicalizer_options_t* options) {
  loom_pass_value_fact_scope_t scope =
      loom_pass_value_fact_scope_function_for_target(
          function, options ? options->target_facts : NULL);
  loom_value_fact_table_t* facts = NULL;
  if (options && options->seed_facts.table) {
    scope.seed_facts = options->seed_facts;
    IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare(
        canonicalizer->value_facts, canonicalizer->module, scope, &facts));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_compute(facts, canonicalizer->module, function));
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_acquire(
        canonicalizer->value_facts, canonicalizer->module, scope, &facts));
  }
  loom_greedy_rewrite_driver_set_fact_table(
      &canonicalizer->state->rewrite_driver, facts);
  return iree_ok_status();
}

iree_status_t loom_canonicalizer_run_region(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result) {
  if (out_result) {
    memset(out_result, 0, sizeof(*out_result));
  }
  loom_canonicalizer_reset_run_state(canonicalizer);
  if (!region) {
    return iree_ok_status();
  }
  iree_status_t status = loom_canonicalizer_prepare_region_facts(
      canonicalizer, function, region, parent_op, options);
  if (iree_status_is_ok(status)) {
    status = loom_canonicalizer_run_precomputed_region(
        canonicalizer, function, region, parent_op, options, out_result);
  }
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(canonicalizer->value_facts);
    loom_greedy_rewrite_driver_set_fact_table(
        &canonicalizer->state->rewrite_driver, NULL);
  }
  return status;
}

iree_status_t loom_canonicalizer_run_function(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result) {
  if (out_result) {
    memset(out_result, 0, sizeof(*out_result));
  }
  loom_canonicalizer_reset_run_state(canonicalizer);
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  iree_status_t status = loom_canonicalizer_prepare_function_facts(
      canonicalizer, function, options);
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(canonicalizer->value_facts);
    loom_greedy_rewrite_driver_set_fact_table(
        &canonicalizer->state->rewrite_driver, NULL);
    return status;
  }

  const uint8_t body_region_index = loom_func_like_body_region_index(function);
  loom_canonicalizer_result_t aggregate_result = {0};
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (i == body_region_index) {
      continue;
    }
    loom_region_t* region = loom_func_like_region(function, i);
    if (!region) {
      continue;
    }
    loom_canonicalizer_result_t region_result = {0};
    status = loom_canonicalizer_run_precomputed_region(
        canonicalizer, function, region, function.op, options, &region_result);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_canonicalizer_merge_result(&aggregate_result, &region_result);
  }

  if (iree_status_is_ok(status)) {
    loom_canonicalizer_result_t body_result = {0};
    status = loom_canonicalizer_run_precomputed_region(
        canonicalizer, function, body, function.op, options, &body_result);
    if (iree_status_is_ok(status)) {
      loom_canonicalizer_merge_result(&aggregate_result, &body_result);
    }
  }

  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(canonicalizer->value_facts);
    loom_greedy_rewrite_driver_set_fact_table(
        &canonicalizer->state->rewrite_driver, NULL);
  }
  if (out_result) {
    *out_result = aggregate_result;
  }
  return status;
}
