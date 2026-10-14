// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll_tile.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/movement.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/transforms/scf/scf_body.h"

//===----------------------------------------------------------------------===//
// Scheduled body copies and cross-iteration dependencies
//===----------------------------------------------------------------------===//

typedef struct loom_scf_unroll_tile_context_t {
  // Pass owning diagnostics for a requested schedule.
  loom_pass_t* pass;
  // Module containing the source loop and emitted body copies.
  loom_module_t* module;
  // Rewriter positioned at the tile's insertion point.
  loom_rewriter_t* rewriter;
  // Source value facts used to refine memory dependencies.
  const loom_value_fact_table_t* fact_table;
} loom_scf_unroll_tile_context_t;

static iree_status_t loom_scf_unroll_emit_policy_error(
    const loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    iree_string_view_t field_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission);
}

#define LOOM_SCF_UNROLL_EFFECT_INDEX_INVALID UINT32_MAX
#define LOOM_SCF_UNROLL_SCHEDULED_EFFECT_OP_LIMIT 64

typedef struct loom_scf_unroll_effect_dependency_plan_t {
  // Dense cross-iteration effect conflict matrix.
  const bool* conflicts;
  // Source body index for each effectful operation.
  const uint32_t* body_op_indices;
  // Effect row for each body operation, or INVALID for pure operations.
  const uint32_t* body_to_effect_indices;
  // Completed body copies, indexed by effect row then iteration ordinal.
  bool* cloned_ordinals;
  // First incomplete iteration ordinal for each effect row.
  uint32_t* completed_ordinals;
  // Number of effect rows.
  uint32_t effect_count;
  // Number of iteration copies represented by the tile.
  uint32_t unroll_count;
} loom_scf_unroll_effect_dependency_plan_t;

static bool loom_scf_unroll_effects_conflict(
    loom_scf_body_effect_flags_t prior_flags,
    loom_scf_body_effect_flags_t candidate_flags) {
  if ((prior_flags | candidate_flags) == 0) {
    return false;
  }
  if (iree_any_bit_set(
          prior_flags | candidate_flags,
          LOOM_SCF_BODY_EFFECT_ORDERED | LOOM_SCF_BODY_EFFECT_CONVERGENT)) {
    return true;
  }
  if (!iree_any_bit_set(prior_flags | candidate_flags,
                        LOOM_SCF_BODY_EFFECT_WRITE)) {
    return false;
  }
  return iree_any_bit_set(prior_flags, LOOM_SCF_BODY_EFFECT_READ |
                                           LOOM_SCF_BODY_EFFECT_WRITE) &&
         iree_any_bit_set(candidate_flags, LOOM_SCF_BODY_EFFECT_READ |
                                               LOOM_SCF_BODY_EFFECT_WRITE);
}

// Cross-execution value correspondence consumes the body owner's retained
// dependencies. Source-local definitions begin varying; pure scheduling units
// refine their results when every dependency is invariant.
typedef struct loom_scf_unroll_value_states_t {
  // Existing movement domain, borrowed for this plan's lifetime.
  const loom_local_value_domain_t* domain;
  // One state byte per source-local definition; captures are invariant.
  bool* invariant;
} loom_scf_unroll_value_states_t;

static bool loom_scf_unroll_value_is_invariant(
    const loom_scf_unroll_value_states_t* facts, loom_value_id_t value) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(facts->domain, value);
  return ordinal >= facts->domain->definition_count ||
         facts->invariant[ordinal];
}

static bool loom_scf_unroll_expression_is_invariant(
    const loom_scf_unroll_value_states_t* facts,
    const loom_symbolic_expr_t* expression) {
  if (!loom_symbolic_expr_is_linear(expression)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t term = expression->terms[i];
    if (!loom_scf_unroll_value_is_invariant(facts, term.value_id) ||
        !loom_scf_unroll_value_is_invariant(facts, term.relation_value_id)) {
      return false;
    }
  }
  return true;
}

// Each movement description byte packs the source scope in bits 0..1, the
// destination scope in bits 2..3, and description success in bit 4. Endpoint
// construction classifies symbolic terms once, before conflict-pair queries.
enum {
  LOOM_SCF_UNROLL_SCOPE_INVARIANT_ROOT = 1u << 0,
  LOOM_SCF_UNROLL_SCOPE_INVARIANT_RANGE = 1u << 1,
  LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT = 2,
  LOOM_SCF_UNROLL_MOVEMENT_DESCRIBED = 1u << 4,
};

static uint8_t loom_scf_unroll_endpoint_scope(
    const loom_scf_unroll_value_states_t* facts,
    const loom_movement_endpoint_t* endpoint) {
  if (endpoint->kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
    return 0;
  }
  uint8_t scope =
      loom_scf_unroll_value_is_invariant(facts, endpoint->root_value_id)
          ? LOOM_SCF_UNROLL_SCOPE_INVARIANT_ROOT
          : 0;
  if (loom_scf_unroll_expression_is_invariant(facts,
                                              &endpoint->begin_byte_offset) &&
      loom_scf_unroll_expression_is_invariant(facts, &endpoint->byte_length) &&
      loom_scf_unroll_expression_is_invariant(facts,
                                              &endpoint->end_byte_offset)) {
    scope |= LOOM_SCF_UNROLL_SCOPE_INVARIANT_RANGE;
  }
  return scope;
}

static iree_status_t loom_scf_unroll_build_movement_scopes(
    const loom_scf_body_t* body, const loom_local_value_domain_t* domain,
    iree_arena_allocator_t* arena,
    const loom_scf_unroll_effect_dependency_plan_t* plan,
    loom_movement_analysis_t* movement, const loom_movement_request_t* requests,
    uint8_t* scopes) {
  loom_scf_unroll_value_states_t states = {.domain = domain};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain->definition_count, sizeof(*states.invariant),
      (void**)&states.invariant));
  memset(states.invariant, 0,
         domain->definition_count * sizeof(*states.invariant));
  for (uint32_t i = 0; i < body->count; ++i) {
    const loom_scf_body_operation_t* operation = &body->operations[i];
    const loom_scf_body_effect_flags_t effects =
        operation->effects & ~LOOM_SCF_BODY_EFFECT_SOURCE_ORDER;
    const uint32_t effect = plan->body_to_effect_indices[i];
    if (effect != LOOM_SCF_UNROLL_EFFECT_INDEX_INVALID && scopes[effect]) {
      scopes[effect] |=
          loom_scf_unroll_endpoint_scope(&states, &requests[effect].source) |
          (loom_scf_unroll_endpoint_scope(&states, &requests[effect].dest)
           << LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT);
    }
    bool varying = effects != 0;
    if (effects == LOOM_SCF_BODY_EFFECT_READ) {
      const loom_movement_endpoint_t* source = &requests[effect].source;
      // A fixed address into storage unchanged by this body yields the same
      // value. The movement owner has already retained its interference facts.
      if (iree_all_bits_set(scopes[effect],
                            LOOM_SCF_UNROLL_SCOPE_INVARIANT_ROOT |
                                LOOM_SCF_UNROLL_SCOPE_INVARIANT_RANGE) &&
          loom_view_region_table_root_is_stable(
              &movement->view_regions, source->root_value_id,
              source->alias_scope_id, source->memory_space)) {
        varying = false;
      }
    }
    for (iree_host_size_t j = 0; j < operation->reference_count; ++j) {
      const loom_scf_body_reference_t* reference =
          &body->references[operation->reference_begin + j];
      varying |=
          !loom_scf_unroll_value_is_invariant(&states, reference->value_id);
    }
    for (uint16_t j = 0; j < operation->op->result_count; ++j) {
      const loom_value_id_t result = loom_op_const_results(operation->op)[j];
      states.invariant[loom_local_value_domain_ordinal(domain, result)] =
          !varying;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_endpoints_no_overlap(
    loom_movement_analysis_t* movement_analysis,
    const loom_movement_endpoint_t* left, const loom_movement_endpoint_t* right,
    uint8_t left_scope, uint8_t right_scope, bool* out_no_overlap) {
  *out_no_overlap = false;
  if (left->kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
      right->kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
    return iree_ok_status();
  }
  if (loom_value_fact_reference_origins_are_disjoint(left->origin,
                                                     right->origin) ||
      loom_view_memory_spaces_are_disjoint(left->memory_space,
                                           right->memory_space)) {
    *out_no_overlap = true;
    return iree_ok_status();
  }
  if (left->root_value_id == LOOM_VALUE_ID_INVALID ||
      right->root_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (left->root_value_id != right->root_value_id) {
    // A same-execution promise also covers an arbitrary execution on one side
    // when the other side is invariant.
    if (!iree_any_bit_set(left_scope | right_scope,
                          LOOM_SCF_UNROLL_SCOPE_INVARIANT_ROOT)) {
      return iree_ok_status();
    }
  } else {
    if (!iree_any_bit_set(left_scope, LOOM_SCF_UNROLL_SCOPE_INVARIANT_ROOT)) {
      return iree_ok_status();
    }
    if (!iree_any_bit_set(left_scope & right_scope,
                          LOOM_SCF_UNROLL_SCOPE_INVARIANT_RANGE)) {
      // These marginal ranges quantify both expressions independently. Shared
      // varying SSA symbols cannot cancel across two different iterations.
      *out_no_overlap = left->end_byte_offset.facts.range_hi <=
                            right->begin_byte_offset.facts.range_lo ||
                        right->end_byte_offset.facts.range_hi <=
                            left->begin_byte_offset.facts.range_lo;
      return iree_ok_status();
    }
  }
  loom_view_region_t left_region = {0}, right_region = {0};
  if (!loom_movement_endpoint_as_view_region(left, &left_region) ||
      !loom_movement_endpoint_as_view_region(right, &right_region)) {
    return iree_ok_status();
  }
  return loom_view_regions_prove_no_overlap(&movement_analysis->view_regions,
                                            &left_region, &right_region,
                                            out_no_overlap);
}

static bool loom_scf_unroll_request_access_endpoint(
    const loom_movement_request_t* request, loom_scf_body_effect_flags_t flags,
    loom_scf_body_effect_flags_t access_kind,
    const loom_movement_endpoint_t** out_endpoint) {
  *out_endpoint = NULL;
  if (!iree_any_bit_set(flags, access_kind)) {
    return false;
  }
  if (access_kind == LOOM_SCF_BODY_EFFECT_WRITE &&
      request->dest.kind == LOOM_MOVEMENT_ENDPOINT_VIEW) {
    *out_endpoint = &request->dest;
    return true;
  }
  if (access_kind == LOOM_SCF_BODY_EFFECT_READ &&
      request->source.kind == LOOM_MOVEMENT_ENDPOINT_VIEW) {
    *out_endpoint = &request->source;
    return true;
  }
  return false;
}

static iree_status_t loom_scf_unroll_movement_requests_conflict(
    loom_movement_analysis_t* movement_analysis,
    const loom_movement_request_t* prior_request,
    loom_scf_body_effect_flags_t prior_flags,
    const loom_movement_request_t* candidate_request,
    loom_scf_body_effect_flags_t candidate_flags, uint8_t prior_scopes,
    uint8_t candidate_scopes, bool* out_conflict) {
  *out_conflict = true;
  const loom_movement_endpoint_t* prior_write = NULL;
  const loom_movement_endpoint_t* prior_read = NULL;
  const loom_movement_endpoint_t* candidate_write = NULL;
  const loom_movement_endpoint_t* candidate_read = NULL;
  (void)loom_scf_unroll_request_access_endpoint(
      prior_request, prior_flags, LOOM_SCF_BODY_EFFECT_WRITE, &prior_write);
  (void)loom_scf_unroll_request_access_endpoint(
      prior_request, prior_flags, LOOM_SCF_BODY_EFFECT_READ, &prior_read);
  (void)loom_scf_unroll_request_access_endpoint(
      candidate_request, candidate_flags, LOOM_SCF_BODY_EFFECT_WRITE,
      &candidate_write);
  (void)loom_scf_unroll_request_access_endpoint(
      candidate_request, candidate_flags, LOOM_SCF_BODY_EFFECT_READ,
      &candidate_read);

  const loom_movement_endpoint_t* left = NULL;
  const loom_movement_endpoint_t* right = NULL;
  uint8_t left_scope = 0, right_scope = 0;
  if (prior_write && candidate_write) {
    left = prior_write;
    right = candidate_write;
    left_scope = prior_scopes >> LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT;
    right_scope = candidate_scopes >> LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT;
  } else if (prior_write && candidate_read) {
    left = prior_write;
    right = candidate_read;
    left_scope = prior_scopes >> LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT;
    right_scope = candidate_scopes;
  } else if (prior_read && candidate_write) {
    left = prior_read;
    right = candidate_write;
    left_scope = prior_scopes;
    right_scope = candidate_scopes >> LOOM_SCF_UNROLL_DEST_SCOPE_SHIFT;
  } else {
    *out_conflict = false;
    return iree_ok_status();
  }

  bool no_overlap = false;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_endpoints_no_overlap(
      movement_analysis, left, right, left_scope, right_scope, &no_overlap));
  *out_conflict = !no_overlap;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_describe_movement_requests(
    loom_movement_analysis_t* movement_analysis,
    const loom_scf_body_t* body_ops,
    const loom_scf_unroll_effect_dependency_plan_t* plan,
    loom_movement_request_t* requests, uint8_t* scopes) {
  for (uint32_t i = 0; i < plan->effect_count; ++i) {
    const uint32_t body_op_index = plan->body_op_indices[i];
    loom_movement_diagnostic_t diagnostic = {0};
    bool described = false;
    IREE_RETURN_IF_ERROR(loom_movement_request_describe_op(
        movement_analysis, body_ops->operations[body_op_index].op, &requests[i],
        &diagnostic, &described));
    scopes[i] = described ? LOOM_SCF_UNROLL_MOVEMENT_DESCRIBED : 0;
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_effects_conflict_with_movement(
    loom_movement_analysis_t* movement_analysis,
    const loom_movement_request_t* movement_requests,
    const uint8_t* movement_scopes,
    const loom_scf_unroll_effect_dependency_plan_t* plan,
    const loom_scf_body_t* body_ops, uint32_t prior_effect_index,
    uint32_t candidate_effect_index, bool* out_conflict) {
  const uint32_t prior_op_index = plan->body_op_indices[prior_effect_index];
  const uint32_t candidate_op_index =
      plan->body_op_indices[candidate_effect_index];
  const loom_scf_body_effect_flags_t prior_flags =
      body_ops->operations[prior_op_index].effects;
  const loom_scf_body_effect_flags_t candidate_flags =
      body_ops->operations[candidate_op_index].effects;
  *out_conflict =
      loom_scf_unroll_effects_conflict(prior_flags, candidate_flags);
  if (!*out_conflict) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          prior_flags | candidate_flags,
          LOOM_SCF_BODY_EFFECT_ORDERED | LOOM_SCF_BODY_EFFECT_CONVERGENT)) {
    return iree_ok_status();
  }
  if (!movement_scopes[prior_effect_index] ||
      !movement_scopes[candidate_effect_index]) {
    return iree_ok_status();
  }
  return loom_scf_unroll_movement_requests_conflict(
      movement_analysis, &movement_requests[prior_effect_index], prior_flags,
      &movement_requests[candidate_effect_index], candidate_flags,
      movement_scopes[prior_effect_index],
      movement_scopes[candidate_effect_index], out_conflict);
}

static bool loom_scf_unroll_effects_conflict_is_refinable(
    loom_scf_body_effect_flags_t prior_flags,
    loom_scf_body_effect_flags_t candidate_flags) {
  return loom_scf_unroll_effects_conflict(prior_flags, candidate_flags) &&
         !iree_any_bit_set(
             prior_flags | candidate_flags,
             LOOM_SCF_BODY_EFFECT_ORDERED | LOOM_SCF_BODY_EFFECT_CONVERGENT);
}

static iree_status_t loom_scf_unroll_build_effect_dependency_plan(
    const loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, const loom_scf_body_t* body_ops,
    uint32_t unroll_count, loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena,
    loom_scf_unroll_effect_dependency_plan_t* out_plan) {
  *out_plan = (loom_scf_unroll_effect_dependency_plan_t){0};
  if (body_ops->count == 0) {
    return iree_ok_status();
  }

  // Every conflicting pair contains an ordered effect or a write, which also
  // conflicts with itself. The same holds for refinable writes. Summarize those
  // rows once so read-only bodies never build or probe a quadratic pair table.
  uint32_t effect_count = 0;
  bool has_conflicts = false;
  bool has_refinable_conflicts = false;
  for (uint32_t i = 0; i < body_ops->count; ++i) {
    const loom_scf_body_effect_flags_t flags = body_ops->operations[i].effects;
    if (!(flags & ~LOOM_SCF_BODY_EFFECT_SOURCE_ORDER)) {
      continue;
    }
    ++effect_count;
    has_conflicts =
        has_conflicts || loom_scf_unroll_effects_conflict(flags, flags);
    has_refinable_conflicts =
        has_refinable_conflicts ||
        loom_scf_unroll_effects_conflict_is_refinable(flags, flags);
  }
  if (!has_conflicts) {
    return iree_ok_status();
  }
  if (effect_count > LOOM_SCF_UNROLL_SCHEDULED_EFFECT_OP_LIMIT) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), effect_count,
        IREE_SV("effectful body operation count within scheduled tile "
                "limit"));
  }

  uint32_t* body_to_effect_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, body_ops->count, sizeof(*body_to_effect_indices),
      (void**)&body_to_effect_indices));
  for (uint32_t i = 0; i < body_ops->count; ++i) {
    body_to_effect_indices[i] = LOOM_SCF_UNROLL_EFFECT_INDEX_INVALID;
  }

  uint32_t* body_op_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, effect_count,
                                                 sizeof(*body_op_indices),
                                                 (void**)&body_op_indices));
  uint32_t effect_index = 0;
  for (uint32_t i = 0; i < body_ops->count; ++i) {
    if (!(body_ops->operations[i].effects &
          ~LOOM_SCF_BODY_EFFECT_SOURCE_ORDER)) {
      continue;
    }
    body_to_effect_indices[i] = effect_index;
    body_op_indices[effect_index++] = i;
  }

  const iree_host_size_t matrix_count =
      (iree_host_size_t)effect_count * effect_count;
  bool* conflicts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, matrix_count, sizeof(*conflicts), (void**)&conflicts));
  memset(conflicts, 0, matrix_count * sizeof(*conflicts));
  for (uint32_t prior_effect_index = 0; prior_effect_index < effect_count;
       ++prior_effect_index) {
    const uint32_t prior_op_index = body_op_indices[prior_effect_index];
    for (uint32_t candidate_effect_index = 0;
         candidate_effect_index < effect_count; ++candidate_effect_index) {
      const uint32_t candidate_op_index =
          body_op_indices[candidate_effect_index];
      const iree_host_size_t matrix_index =
          (iree_host_size_t)prior_effect_index * effect_count +
          candidate_effect_index;
      conflicts[matrix_index] = loom_scf_unroll_effects_conflict(
          body_ops->operations[prior_op_index].effects,
          body_ops->operations[candidate_op_index].effects);
    }
  }

  iree_status_t status = iree_ok_status();
  loom_scf_unroll_effect_dependency_plan_t plan = {
      .conflicts = conflicts,
      .body_op_indices = body_op_indices,
      .body_to_effect_indices = body_to_effect_indices,
      .effect_count = effect_count,
      .unroll_count = unroll_count,
  };
  if (has_refinable_conflicts) {
    // Only the conflict matrix survives refinement. Reuse the analysis storage
    // for clone planning after publishing that matrix and releasing ordinals.
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(scratch_arena);
    loom_local_value_domain_t value_domain = {0};
    loom_movement_analysis_t movement_analysis = {0};
    loom_movement_request_t* movement_requests = NULL;
    uint8_t* movement_scopes = NULL;
    status = loom_local_value_domain_acquire_for_region_tree(
        context->module, body_block->parent_region, scratch_arena,
        &value_domain);
    if (iree_status_is_ok(status)) {
      status =
          loom_movement_analysis_initialize(context->fact_table, &value_domain,
                                            scratch_arena, &movement_analysis);
    }
    if (iree_status_is_ok(status)) {
      status = loom_movement_analysis_analyze(&movement_analysis);
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(scratch_arena, effect_count,
                                         sizeof(*movement_requests),
                                         (void**)&movement_requests);
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(scratch_arena, effect_count,
                                         sizeof(*movement_scopes),
                                         (void**)&movement_scopes);
    }
    if (iree_status_is_ok(status)) {
      status = loom_scf_unroll_describe_movement_requests(
          &movement_analysis, body_ops, &plan, movement_requests,
          movement_scopes);
    }
    if (iree_status_is_ok(status)) {
      status = loom_scf_unroll_build_movement_scopes(
          body_ops, &value_domain, scratch_arena, &plan, &movement_analysis,
          movement_requests, movement_scopes);
    }
    for (uint32_t prior_effect_index = 0;
         iree_status_is_ok(status) && prior_effect_index < effect_count;
         ++prior_effect_index) {
      const uint32_t prior_op_index = body_op_indices[prior_effect_index];
      for (uint32_t candidate_effect_index = 0;
           candidate_effect_index < effect_count; ++candidate_effect_index) {
        const uint32_t candidate_op_index =
            body_op_indices[candidate_effect_index];
        if (!loom_scf_unroll_effects_conflict_is_refinable(
                body_ops->operations[prior_op_index].effects,
                body_ops->operations[candidate_op_index].effects)) {
          continue;
        }
        const iree_host_size_t matrix_index =
            (iree_host_size_t)prior_effect_index * effect_count +
            candidate_effect_index;
        status = loom_scf_unroll_effects_conflict_with_movement(
            &movement_analysis, movement_requests, movement_scopes, &plan,
            body_ops, prior_effect_index, candidate_effect_index,
            &conflicts[matrix_index]);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
    }
    loom_local_value_domain_release(&value_domain);
    iree_arena_checkpoint_restore(&checkpoint);
  }
  IREE_RETURN_IF_ERROR(status);

  bool has_remaining_conflicts = false;
  for (iree_host_size_t i = 0; i < matrix_count; ++i) {
    has_remaining_conflicts = has_remaining_conflicts || conflicts[i];
  }
  if (!has_remaining_conflicts) {
    return iree_ok_status();
  }

  iree_host_size_t cloned_ordinal_count = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)effect_count, unroll_count,
                                  &cloned_ordinal_count)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), schedule,
        IREE_SV("effect dependency state representable"));
  }
  bool* cloned_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, cloned_ordinal_count, sizeof(*cloned_ordinals),
      (void**)&cloned_ordinals));
  memset(cloned_ordinals, 0, cloned_ordinal_count * sizeof(*cloned_ordinals));
  uint32_t* completed_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, effect_count,
                                                 sizeof(*completed_ordinals),
                                                 (void**)&completed_ordinals));
  memset(completed_ordinals, 0,
         (iree_host_size_t)effect_count * sizeof(*completed_ordinals));

  plan.cloned_ordinals = cloned_ordinals;
  plan.completed_ordinals = completed_ordinals;
  *out_plan = plan;
  return iree_ok_status();
}

static bool loom_scf_unroll_effect_dependencies_are_ready(
    const loom_scf_unroll_effect_dependency_plan_t* plan, uint32_t op_index,
    uint32_t ordinal) {
  if (!plan->conflicts) {
    return true;
  }
  const uint32_t effect_index = plan->body_to_effect_indices[op_index];
  if (effect_index == LOOM_SCF_UNROLL_EFFECT_INDEX_INVALID) {
    return true;
  }
  for (uint32_t prior_effect_index = 0; prior_effect_index < plan->effect_count;
       ++prior_effect_index) {
    const iree_host_size_t matrix_index =
        (iree_host_size_t)prior_effect_index * plan->effect_count +
        effect_index;
    if (!plan->conflicts[matrix_index]) {
      continue;
    }
    const uint32_t prior_op_index = plan->body_op_indices[prior_effect_index];
    const uint32_t required_completed_ordinal =
        ordinal + (prior_op_index < op_index ? 1u : 0u);
    if (plan->completed_ordinals[prior_effect_index] <
        required_completed_ordinal) {
      return false;
    }
  }
  return true;
}

static void loom_scf_unroll_release_effect_dependencies(
    loom_scf_unroll_effect_dependency_plan_t* plan, uint32_t op_index,
    uint32_t ordinal) {
  if (!plan->conflicts) {
    return;
  }
  const uint32_t effect_index = plan->body_to_effect_indices[op_index];
  if (effect_index == LOOM_SCF_UNROLL_EFFECT_INDEX_INVALID) {
    return;
  }
  const iree_host_size_t ordinal_index =
      (iree_host_size_t)effect_index * plan->unroll_count + ordinal;
  plan->cloned_ordinals[ordinal_index] = true;
  uint32_t completed_ordinal = plan->completed_ordinals[effect_index];
  while (completed_ordinal < plan->unroll_count &&
         plan->cloned_ordinals[(iree_host_size_t)effect_index *
                                   plan->unroll_count +
                               completed_ordinal]) {
    ++completed_ordinal;
  }
  plan->completed_ordinals[effect_index] = completed_ordinal;
}

static bool loom_scf_unroll_payload_is_ready(
    const loom_scf_body_t* body, const loom_scf_body_operation_t* operation,
    const loom_ir_remap_t* remap) {
  for (iree_host_size_t i = 0; i < operation->reference_count; ++i) {
    const loom_scf_body_reference_t* reference =
        &body->references[operation->reference_begin + i];
    loom_value_id_t mapped_value = LOOM_VALUE_ID_INVALID;
    if (!loom_ir_remap_try_lookup_value(remap, reference->value_id,
                                        &mapped_value) ||
        (!reference->allow_identity_mapping &&
         mapped_value == reference->value_id)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_scf_unroll_rename_cloned_op_results(
    loom_scf_unroll_tile_context_t* context, const loom_op_t* source_op,
    loom_op_t* cloned_op, uint32_t ordinal) {
  if (ordinal == 0 ||
      !iree_any_bit_set(context->rewriter->name_policy,
                        LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES)) {
    return iree_ok_status();
  }

  char suffix[32] = {0};
  int suffix_length =
      iree_snprintf(suffix, sizeof(suffix), "%" PRIu32, ordinal);
  if (suffix_length <= 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  iree_string_view_t suffix_view =
      iree_make_string_view(suffix, (iree_host_size_t)suffix_length);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* cloned_results = loom_op_const_results(cloned_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_clear_value_name(context->rewriter, cloned_results[i]));
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        context->rewriter, source_results[i], cloned_results[i], suffix_view));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_mark_iteration_complete(
    const loom_block_t* body_block, const loom_op_t* yield, uint32_t ordinal,
    uint32_t trip_count, uint16_t carried_count, loom_ir_remap_t* remaps,
    iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values) {
  if (carried_count == 0) {
    return iree_ok_status();
  }

  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  loom_value_id_t* resolved_values = final_carried_values;
  if (ordinal + 1 < trip_count) {
    resolved_values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, carried_count,
                                                   sizeof(*resolved_values),
                                                   (void**)&resolved_values));
  }
  for (uint16_t i = 0; i < carried_count; ++i) {
    // Payload readiness has already established every local mapping. Values
    // captured from outside the body remain unchanged when no mapping exists.
    if (!loom_ir_remap_try_lookup_value(
            &remaps[ordinal], yielded_values.values[i], &resolved_values[i])) {
      resolved_values[i] = yielded_values.values[i];
    }
  }
  if (ordinal + 1 < trip_count) {
    for (uint16_t i = 0; i < carried_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&remaps[ordinal + 1],
                                                   body_block->arg_ids[1 + i],
                                                   resolved_values[i]));
    }
  }
  return iree_ok_status();
}

typedef struct loom_scf_unroll_scheduled_tile_t {
  // Ordered source body operations excluding the yield terminator.
  loom_scf_body_t body_ops;
  // Cross-iteration effect dependencies governing legal clone order.
  loom_scf_unroll_effect_dependency_plan_t effect_dependency_plan;
  // Per-iteration source-to-clone value maps.
  loom_ir_remap_t* remaps;
  // Per-iteration completion flags.
  bool* completed_iterations;
  // Per-iteration numbers of cloned body operations.
  uint32_t* cloned_counts;
  // Flattened per-iteration body operation clone flags.
  bool* cloned;
  // Position within the lexically expanded body's source-order ranges.
  struct {
    // First source slot whose clone has not been emitted.
    iree_host_size_t frontier;
    // Next boundary's source slot, or the total slot count after the last one.
    iree_host_size_t boundary_slot;
    // Index of the next boundary within one source iteration.
    uint32_t boundary_index;
  } source_order;
  // Number of iterations materialized in the tile.
  uint32_t unroll_count;
  // Number of body operation clone slots not yet materialized.
  iree_host_size_t remaining_clone_count;
} loom_scf_unroll_scheduled_tile_t;

static iree_status_t loom_scf_unroll_initialize_scheduled_tile(
    loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, const loom_value_id_t* iteration_indices,
    uint32_t iteration_count, const loom_value_id_t* initial_carried_values,
    uint16_t carried_count, loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena,
    loom_scf_unroll_scheduled_tile_t* out_tile) {
  *out_tile = (loom_scf_unroll_scheduled_tile_t){
      .unroll_count = iteration_count,
  };
  const loom_op_t* unstructured_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_body_build(
      context->module, body_block, /*capture_block=*/NULL, /*spaces=*/NULL,
      LOOM_SCF_BODY_MODE_SCHEDULE, scratch_arena, &out_tile->body_ops,
      &unstructured_op));
  if (unstructured_op != NULL) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), schedule,
        IREE_SV("body operations with only scf.if/scf.for regions "
                "and no successors"));
  }
  if (out_tile->unroll_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, out_tile->unroll_count, sizeof(*out_tile->remaps),
      (void**)&out_tile->remaps));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, out_tile->unroll_count,
                                sizeof(*out_tile->completed_iterations),
                                (void**)&out_tile->completed_iterations));
  memset(out_tile->completed_iterations, 0,
         (iree_host_size_t)out_tile->unroll_count *
             sizeof(*out_tile->completed_iterations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, out_tile->unroll_count, sizeof(*out_tile->cloned_counts),
      (void**)&out_tile->cloned_counts));
  memset(out_tile->cloned_counts, 0,
         (iree_host_size_t)out_tile->unroll_count *
             sizeof(*out_tile->cloned_counts));

  if (!iree_host_size_checked_mul((iree_host_size_t)out_tile->unroll_count,
                                  out_tile->body_ops.count,
                                  &out_tile->remaining_clone_count)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), schedule,
        IREE_SV("unroll count * body operation count representable"));
  }
  if (out_tile->remaining_clone_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, out_tile->remaining_clone_count,
        sizeof(*out_tile->cloned), (void**)&out_tile->cloned));
    memset(out_tile->cloned, 0,
           out_tile->remaining_clone_count * sizeof(*out_tile->cloned));
  }
  if (out_tile->body_ops.source_order_boundary_count != 0) {
    out_tile->source_order.boundary_slot =
        out_tile->body_ops.source_order_boundaries[0];
  }
  IREE_RETURN_IF_ERROR(loom_scf_unroll_build_effect_dependency_plan(
      context, op, body_block, &out_tile->body_ops, out_tile->unroll_count,
      schedule, scratch_arena, &out_tile->effect_dependency_plan));

  for (uint32_t ordinal = 0; ordinal < out_tile->unroll_count; ++ordinal) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        context->module, context->module, scratch_arena,
        &(loom_ir_remap_options_t){
            .allow_unmapped_values = true,
            .remap_symbol = loom_ir_remap_symbol_callback_empty(),
        },
        &out_tile->remaps[ordinal]));
    if (iteration_indices != NULL) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&out_tile->remaps[ordinal],
                                                   body_block->arg_ids[0],
                                                   iteration_indices[ordinal]));
    }
  }
  for (uint16_t i = 0; i < carried_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&out_tile->remaps[0],
                                                 body_block->arg_ids[1 + i],
                                                 initial_carried_values[i]));
  }
  return iree_ok_status();
}

// Advances over each source slot once. Boundaries partition the lexically
// expanded body, so a range may contain the end of one iteration and the
// beginning of the next without imposing an extra iteration boundary.
static void loom_scf_unroll_release_source_order(
    loom_scf_unroll_scheduled_tile_t* tile, iree_host_size_t slot,
    uint32_t ordinal) {
  if (tile->body_ops.source_order_boundary_count == 0) {
    return;
  }
  const iree_host_size_t slot_count =
      (iree_host_size_t)tile->unroll_count * tile->body_ops.count;
  while (tile->source_order.frontier < slot_count &&
         tile->cloned[tile->source_order.frontier]) {
    ++tile->source_order.frontier;
  }
  if (slot != tile->source_order.boundary_slot) {
    return;
  }
  if (++tile->source_order.boundary_index ==
      tile->body_ops.source_order_boundary_count) {
    tile->source_order.boundary_index = 0;
    ++ordinal;
  }
  tile->source_order.boundary_slot =
      ordinal < tile->unroll_count
          ? (iree_host_size_t)ordinal * tile->body_ops.count +
                tile->body_ops
                    .source_order_boundaries[tile->source_order.boundary_index]
          : slot_count;
}

static iree_status_t loom_scf_unroll_try_clone_scheduled_body_op(
    loom_scf_unroll_tile_context_t* context, uint32_t op_index,
    uint32_t ordinal, loom_scf_unroll_scheduled_tile_t* tile,
    bool* out_cloned) {
  *out_cloned = false;
  const iree_host_size_t slot =
      (iree_host_size_t)ordinal * tile->body_ops.count + op_index;
  if (tile->cloned[slot]) {
    return iree_ok_status();
  }
  if (tile->body_ops.source_order_boundary_count != 0 &&
      (slot > tile->source_order.boundary_slot ||
       (slot == tile->source_order.boundary_slot &&
        slot != tile->source_order.frontier))) {
    return iree_ok_status();
  }
  if (!loom_scf_unroll_effect_dependencies_are_ready(
          &tile->effect_dependency_plan, op_index, ordinal)) {
    return iree_ok_status();
  }
  const loom_op_t* source_op = tile->body_ops.operations[op_index].op;
  if (!loom_scf_unroll_payload_is_ready(&tile->body_ops,
                                        &tile->body_ops.operations[op_index],
                                        &tile->remaps[ordinal])) {
    return iree_ok_status();
  }

  loom_op_t* cloned_op = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_clone_op(&context->rewriter->builder, source_op,
                                        &tile->remaps[ordinal], &cloned_op));
  IREE_RETURN_IF_ERROR(loom_scf_unroll_rename_cloned_op_results(
      context, source_op, cloned_op, ordinal));
  tile->cloned[slot] = true;
  loom_scf_unroll_release_source_order(tile, slot, ordinal);
  loom_scf_unroll_release_effect_dependencies(&tile->effect_dependency_plan,
                                              op_index, ordinal);
  ++tile->cloned_counts[ordinal];
  --tile->remaining_clone_count;
  *out_cloned = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_try_complete_scheduled_iteration(
    const loom_block_t* body_block, const loom_op_t* yield, uint32_t ordinal,
    uint16_t carried_count, iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values,
    loom_scf_unroll_scheduled_tile_t* tile, bool* out_completed) {
  *out_completed = false;
  if (tile->completed_iterations[ordinal] ||
      tile->cloned_counts[ordinal] != tile->body_ops.count ||
      !loom_scf_unroll_payload_is_ready(&tile->body_ops,
                                        &tile->body_ops.terminator,
                                        &tile->remaps[ordinal])) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scf_unroll_mark_iteration_complete(
      body_block, yield, ordinal, tile->unroll_count, carried_count,
      tile->remaps, scratch_arena, final_carried_values));
  tile->completed_iterations[ordinal] = true;
  *out_completed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_emit_interleaved_tile(
    loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, loom_op_t* yield, uint16_t carried_count,
    iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values,
    loom_scf_unroll_scheduled_tile_t* tile) {
  // Each source operation's clones form an ordinal prefix: local producers,
  // carried values, effects and source boundaries all become ready in ordinal
  // order. A blocked copy therefore blocks the rest of that row. Remembering
  // its frontier avoids revisiting completed copies or scanning blocked
  // suffixes on every round of a carried recurrence.
  uint32_t* next_ordinals = NULL;
  if (tile->body_ops.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, tile->body_ops.count, sizeof(*next_ordinals),
        (void**)&next_ordinals));
    memset(next_ordinals, 0, tile->body_ops.count * sizeof(*next_ordinals));
  }

  for (uint32_t ordinal = 0; ordinal < tile->unroll_count; ++ordinal) {
    bool completed = false;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_try_complete_scheduled_iteration(
        body_block, yield, ordinal, carried_count, scratch_arena,
        final_carried_values, tile, &completed));
  }

  while (tile->remaining_clone_count > 0) {
    bool made_progress = false;
    for (uint32_t op_index = 0; op_index < tile->body_ops.count; ++op_index) {
      while (next_ordinals[op_index] < tile->unroll_count) {
        const uint32_t ordinal = next_ordinals[op_index];
        bool cloned = false;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_try_clone_scheduled_body_op(
            context, op_index, ordinal, tile, &cloned));
        if (!cloned) {
          break;
        }
        ++next_ordinals[op_index];
        made_progress = true;
        bool completed = false;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_try_complete_scheduled_iteration(
            body_block, yield, ordinal, carried_count, scratch_arena,
            final_carried_values, tile, &completed));
      }
    }
    if (!made_progress) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("schedule"),
          LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED,
          IREE_SV("acyclic body-local SSA dependencies"));
    }
  }
  for (uint32_t ordinal = 0; ordinal < tile->unroll_count; ++ordinal) {
    if (tile->completed_iterations[ordinal]) {
      continue;
    }
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"),
        LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED,
        IREE_SV("acyclic loop-carried dependencies"));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_classify_recurrence_body_ops(
    const loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, const loom_scf_body_t* body_ops,
    iree_arena_allocator_t* scratch_arena, bool** out_independent_ops,
    uint32_t* out_independent_count, uint32_t* out_dependent_count) {
  *out_independent_ops = NULL;
  *out_independent_count = 0;
  *out_dependent_count = 0;
  if (body_ops->count == 0) {
    return iree_ok_status();
  }

  bool* independent_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, body_ops->count,
                                                 sizeof(*independent_ops),
                                                 (void**)&independent_ops));
  memset(independent_ops, 0, body_ops->count * sizeof(*independent_ops));

  loom_ir_remap_t available_values = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, scratch_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &available_values));
  const loom_value_id_t availability_marker = loom_scf_for_lower_bound(op);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
      &available_values, body_block->arg_ids[0], availability_marker));

  for (uint32_t op_index = 0; op_index < body_ops->count; ++op_index) {
    const loom_op_t* source_op = body_ops->operations[op_index].op;
    if (!loom_scf_unroll_payload_is_ready(
            body_ops, &body_ops->operations[op_index], &available_values)) {
      continue;
    }
    independent_ops[op_index] = true;
    ++*out_independent_count;
    const loom_value_id_t* results = loom_op_const_results(source_op);
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
          &available_values, results[i], availability_marker));
    }
  }

  *out_independent_ops = independent_ops;
  *out_dependent_count = body_ops->count - *out_independent_count;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_emit_schedule_fence(
    loom_scf_unroll_tile_context_t* context, loom_location_id_t location) {
  loom_op_t* fence_op = NULL;
  return loom_scf_schedule_fence_build(&context->rewriter->builder, location,
                                       &fence_op);
}

static iree_status_t loom_scf_unroll_emit_recurrence_tile(
    loom_scf_unroll_tile_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, loom_op_t* yield, uint16_t carried_count,
    iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values,
    loom_scf_unroll_scheduled_tile_t* tile) {
  if (carried_count == 0) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"),
        LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE,
        IREE_SV("one or more loop-carried values"));
  }

  bool* independent_ops = NULL;
  uint32_t independent_count = 0;
  uint32_t dependent_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_classify_recurrence_body_ops(
      context, op, body_block, &tile->body_ops, scratch_arena, &independent_ops,
      &independent_count, &dependent_count));
  if (independent_count == 0) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"),
        LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE,
        IREE_SV("one or more loop-carried-independent producer operations"));
  }
  if (dependent_count == 0) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"),
        LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE,
        IREE_SV("one or more loop-carried-dependent consumer operations"));
  }

  if (tile->unroll_count > 1) {
    const uint32_t prologue_count = iree_min(tile->unroll_count, 2u);
    for (uint32_t ordinal = 0; ordinal < prologue_count; ++ordinal) {
      for (uint32_t op_index = 0; op_index < tile->body_ops.count; ++op_index) {
        if (!independent_ops[op_index]) {
          continue;
        }
        bool cloned = false;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_try_clone_scheduled_body_op(
            context, op_index, ordinal, tile, &cloned));
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_scf_unroll_emit_schedule_fence(context, op->location));
  }

  for (uint32_t ordinal = 0; ordinal < tile->unroll_count; ++ordinal) {
    while (!tile->completed_iterations[ordinal]) {
      bool made_progress = false;
      for (uint32_t op_index = 0; op_index < tile->body_ops.count; ++op_index) {
        bool cloned = false;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_try_clone_scheduled_body_op(
            context, op_index, ordinal, tile, &cloned));
        made_progress = made_progress || cloned;
      }
      bool completed = false;
      IREE_RETURN_IF_ERROR(loom_scf_unroll_try_complete_scheduled_iteration(
          body_block, yield, ordinal, carried_count, scratch_arena,
          final_carried_values, tile, &completed));
      made_progress = made_progress || completed;
      if (!made_progress) {
        return loom_scf_unroll_emit_policy_error(
            context, op, IREE_SV("schedule"),
            LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE,
            IREE_SV("acyclic body-local and loop-carried dependencies"));
      }
    }

    const uint32_t lookahead_ordinal = ordinal + 2;
    if (lookahead_ordinal < tile->unroll_count) {
      for (uint32_t op_index = 0; op_index < tile->body_ops.count; ++op_index) {
        if (!independent_ops[op_index]) {
          continue;
        }
        bool cloned = false;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_try_clone_scheduled_body_op(
            context, op_index, lookahead_ordinal, tile, &cloned));
      }
    }
    if (ordinal + 1 < tile->unroll_count) {
      IREE_RETURN_IF_ERROR(
          loom_scf_unroll_emit_schedule_fence(context, op->location));
    }
  }

  if (tile->remaining_clone_count != 0) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"),
        LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE,
        IREE_SV("all scheduled body operations materialized"));
  }
  return iree_ok_status();
}

iree_status_t loom_scf_unroll_tile_emit(
    loom_pass_t* pass, loom_rewriter_t* rewriter,
    const loom_value_fact_table_t* fact_table, loom_op_t* op,
    const loom_value_id_t* iteration_indices, uint32_t iteration_count,
    const loom_value_id_t* initial_carried_values,
    loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values) {
  loom_scf_unroll_tile_context_t context_storage = {
      .pass = pass,
      .module = rewriter->module,
      .rewriter = rewriter,
      .fact_table = fact_table,
  };
  loom_scf_unroll_tile_context_t* context = &context_storage;
  const loom_block_t* body_block =
      loom_region_entry_block(loom_scf_for_body(op));
  loom_op_t* yield = body_block->last_op;
  const uint16_t carried_count = op->result_count;
  if (iteration_count == 0) {
    if (carried_count > 0) {
      memcpy(final_carried_values, initial_carried_values,
             (iree_host_size_t)carried_count * sizeof(*final_carried_values));
    }
    return iree_ok_status();
  }

  loom_scf_unroll_scheduled_tile_t tile = {0};
  IREE_RETURN_IF_ERROR(loom_scf_unroll_initialize_scheduled_tile(
      context, op, body_block, iteration_indices, iteration_count,
      initial_carried_values, carried_count, schedule, scratch_arena, &tile));
  if (loom_pass_has_error_diagnostics(context->pass)) {
    return iree_ok_status();
  }

  switch (schedule) {
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED:
      return loom_scf_unroll_emit_interleaved_tile(
          context, op, body_block, yield, carried_count, scratch_arena,
          final_carried_values, &tile);
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE:
      return loom_scf_unroll_emit_recurrence_tile(
          context, op, body_block, yield, carried_count, scratch_arena,
          final_carried_values, &tile);
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_LINEAR:
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_COUNT_:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("schedule"), schedule,
          IREE_SV("interleaved or recurrence scheduled tile"));
  }
  return iree_ok_status();
}
