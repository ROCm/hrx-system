// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/place_loop_invariants.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/motion.h"
#include "loom/codegen/low/lower/source_pressure.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/scf/residency.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/reporting/report.h"
#include "loom/target/selection.h"

#define LOOM_PLACE_LOOP_INVARIANTS_STATISTICS(V, statistics_type)             \
  V(statistics_type, functions, "functions",                                  \
    "Number of target-bound source functions inspected.")                     \
  V(statistics_type, loops, "loops", "Number of structured loops inspected.") \
  V(statistics_type, candidates, "candidates",                                \
    "Number of legal invariant placement candidates evaluated.")              \
  V(statistics_type, ops_hoisted, "ops-hoisted",                              \
    "Number of invariant operations moved outside a loop.")                   \
  V(statistics_type, residency_rejections, "residency-rejections",            \
    "Number of legal moves retained to preserve a residency tier.")           \
  V(statistics_type, pressure_evaluations, "pressure-evaluations",            \
    "Number of cumulative placement frontiers evaluated.")                    \
  V(statistics_type, refinement_limits, "refinement-limits",                  \
    "Number of bounded cliff refinements that exhausted their budget.")       \
  V(statistics_type, policies_consumed, "policies-consumed",                  \
    "Number of authored residency policies consumed.")                        \
  V(statistics_type, functions_without_model, "functions-without-model",      \
    "Number of functions placed without a target residency model.")

LOOM_PASS_STATISTICS_DEFINE(loom_place_loop_invariants_statistics,
                            loom_place_loop_invariants_statistics_t,
                            LOOM_PLACE_LOOP_INVARIANTS_STATISTICS)

static const loom_pass_info_t loom_place_loop_invariants_pass_info_storage = {
    .name = IREE_SVL("place-loop-invariants"),
    .description = IREE_SVL(
        "Place loop invariants using target residency pressure cliffs."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_place_loop_invariants_statistics_layout,
};

const loom_pass_info_t* loom_place_loop_invariants_pass_info(void) {
  return &loom_place_loop_invariants_pass_info_storage;
}

typedef struct loom_place_loop_record_t {
  // Structured loop operation whose body defines this pressure scope.
  loom_op_t* op;
  // Loop body retained across operation movement and policy consumption.
  loom_region_t* body;
  // Structured nesting depth used to process inner loops first.
  uint32_t depth;
  // Stable discovery ordinal used to break equal-depth ordering ties.
  uint32_t ordinal;
  // Minimum target tier accepted for the final loop body.
  uint32_t required_tier;
  // Target tier of the authored loop body before placement.
  uint32_t baseline_tier;
  // Number of legal invariant operations considered for this loop.
  uint32_t candidate_count;
  // Number of legal invariant operations selected outside this loop.
  uint32_t selected_count;
  // Stable policy-resolution failure key, or empty when resolution succeeded.
  iree_string_view_t policy_failure_reason;
  // True when the author supplied preserve or a minimum tier.
  bool has_authored_policy;
  // True when the authored policy carries an SSA minimum tier.
  bool has_minimum_policy;
} loom_place_loop_record_t;

typedef struct loom_place_region_stack_entry_t {
  // Region waiting to be traversed.
  loom_region_t* region;
  // Structured depth assigned to operations directly in |region|.
  uint32_t depth;
} loom_place_region_stack_entry_t;

typedef struct loom_place_region_stack_t {
  // Arena-owned traversal entries.
  loom_place_region_stack_entry_t* entries;
  // Number of populated entries.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
} loom_place_region_stack_t;

typedef struct loom_place_candidate_t {
  // Invariant operation considered for movement before its loop.
  loom_op_t* op;
  // Original next operation used to restore speculative movement exactly.
  loom_op_t* restore_before_op;
  // Fixed-point wave in which the operation became legally movable.
  uint32_t discovery_wave;
  // Union-find parent for candidate dependency closures.
  uint32_t dependency_parent;
  // True when the final selected frontier contains this candidate.
  bool accepted;
} loom_place_candidate_t;

typedef struct loom_place_candidate_list_t {
  // Arena-owned candidates in deterministic legal movement order.
  loom_place_candidate_t* values;
  // Number of populated entries in |values|.
  iree_host_size_t count;
  // Allocated candidate capacity.
  iree_host_size_t capacity;
} loom_place_candidate_list_t;

typedef struct loom_place_selected_boundary_t {
  // Accepted invariant operation whose result may be rematerialized.
  loom_op_t* op;
  // Loop defining the original materialization boundary.
  loom_op_t* loop_op;
  // Result index identifying the retained materialized value.
  uint16_t result_index;
  // Stable identity shared by source and exact report rows.
  uint32_t candidate_id;
  // Projected dynamic rematerialization count.
  uint32_t recompute_cost;
  // True when this movement protects its source placement baseline.
  bool preserves_baseline;
} loom_place_selected_boundary_t;

// A cliff is uncommon, but refinement must remain finite when a generated loop
// contains a very large invariant closure. The complete closure gets one trial;
// this budget applies only after that common fast path fails.
#define LOOM_PLACE_MAX_REFINEMENT_EVALUATIONS 32

typedef struct loom_place_function_context_t {
  // Pass instance owning statistics, diagnostics, and scratch storage.
  loom_pass_t* pass;
  // Module containing the selected source function.
  loom_module_t* module;
  // Concrete target and lowering policy selected for the function.
  const loom_low_source_selection_t* selection;
  // Descriptor set named by the selected target contract.
  const loom_low_descriptor_set_t* descriptor_set;
  // Rewriter used for operation movement and policy consumption.
  loom_rewriter_t rewriter;
  // Function-local facts used by motion and SSA minimum resolution.
  loom_value_fact_table_t* fact_table;
  // Active whole-function value domain shared by motion and pressure.
  loom_local_value_domain_t value_domain;
  // Shared movement legality analysis.
  loom_motion_analysis_t motion;
  // Loops ordered from deepest to shallowest after collection.
  loom_place_loop_record_t* loops;
  // Number of records in |loops|.
  iree_host_size_t loop_count;
  // Allocated capacity of |loops|.
  iree_host_size_t loop_capacity;
  // Accepted invariant boundaries retained for exact allocation repair.
  loom_place_selected_boundary_t* selected_boundaries;
  // Number of entries in |selected_boundaries|.
  iree_host_size_t selected_boundary_count;
  // Allocated capacity of |selected_boundaries|.
  iree_host_size_t selected_boundary_capacity;
  // Pressure scopes: function body followed by loop bodies in |loops| order.
  const loom_region_t** pressure_regions;
  // Required tier for each entry in |pressure_regions|.
  uint32_t* required_tiers;
  // Tier of the last accepted placement for each pressure scope.
  uint32_t* current_tiers;
  // Number of function and loop pressure scopes.
  iree_host_size_t pressure_region_count;
  // Number of loop pressure contracts initialized for structured reporting.
  iree_host_size_t reportable_loop_count;
  // Optional compile report receiving structured placement decisions.
  loom_target_compile_report_t* compile_report;
  // Number of named non-source contributors applied to source pressure.
  uint32_t pressure_reserve_count;
  // Reusable stack for loop and candidate traversal.
  loom_place_region_stack_t region_stack;
  // True when the target selected a residency model.
  bool has_residency_model;
  // True when source values and reserves bound all target pressure.
  bool pressure_projection_complete;
  // Strongest authored numeric tier required after exact allocation.
  uint32_t minimum_required_tier;
  // Projected whole-function tier protected by baseline-preserving moves.
  uint32_t projected_baseline_tier;
  // Next stable candidate identity emitted into the exact contract.
  uint32_t next_candidate_id;
  // True when at least one loop supplied an authored numeric minimum.
  bool has_minimum_requirement;
  // True when at least one selected boundary preserves its baseline tier.
  bool preserves_baseline;
  // True after any operation movement or policy consumption.
  bool changed;
} loom_place_function_context_t;

static iree_status_t loom_place_region_stack_push(
    iree_arena_allocator_t* arena, loom_place_region_stack_t* stack,
    loom_region_t* region, uint32_t depth) {
  if (region == NULL || region->block_count == 0) return iree_ok_status();
  if (stack->count == stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(*stack->entries),
        &stack->capacity, (void**)&stack->entries));
  }
  stack->entries[stack->count++] = (loom_place_region_stack_entry_t){
      .region = region,
      .depth = depth,
  };
  return iree_ok_status();
}

static bool loom_place_region_stack_pop(
    loom_place_region_stack_t* stack,
    loom_place_region_stack_entry_t* out_entry) {
  if (stack->count == 0) return false;
  *out_entry = stack->entries[--stack->count];
  return true;
}

static iree_status_t loom_place_append_loop(
    loom_place_function_context_t* context, loom_op_t* op, uint32_t depth) {
  if (context->loop_count == context->loop_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        context->pass->arena, context->loop_count, context->loop_count + 1,
        sizeof(*context->loops), &context->loop_capacity,
        (void**)&context->loops));
  }
  context->loops[context->loop_count] = (loom_place_loop_record_t){
      .op = op,
      .body = loom_loop_like_body(loom_loop_like_cast(context->module, op)),
      .depth = depth,
      .ordinal = (uint32_t)context->loop_count,
  };
  ++context->loop_count;
  return iree_ok_status();
}

static iree_status_t loom_place_append_selected_boundary(
    loom_place_function_context_t* context, loom_op_t* op, loom_op_t* loop_op,
    uint16_t result_index, uint32_t candidate_id, uint32_t recompute_cost,
    bool preserves_baseline) {
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "residency boundary result is out of range");
  }
  if (context->selected_boundary_count == context->selected_boundary_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        context->pass->arena, context->selected_boundary_count,
        context->selected_boundary_count + 1,
        sizeof(*context->selected_boundaries),
        &context->selected_boundary_capacity,
        (void**)&context->selected_boundaries));
  }
  context->selected_boundaries[context->selected_boundary_count++] =
      (loom_place_selected_boundary_t){
          .op = op,
          .loop_op = loop_op,
          .result_index = result_index,
          .candidate_id = candidate_id,
          .recompute_cost = recompute_cost,
          .preserves_baseline = preserves_baseline,
      };
  context->preserves_baseline =
      context->preserves_baseline || preserves_baseline;
  return iree_ok_status();
}

static int loom_place_compare_loops(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_place_loop_record_t* lhs =
      (const loom_place_loop_record_t*)lhs_ptr;
  const loom_place_loop_record_t* rhs =
      (const loom_place_loop_record_t*)rhs_ptr;
  if (lhs->depth != rhs->depth) return lhs->depth > rhs->depth ? -1 : 1;
  if (lhs->ordinal != rhs->ordinal) return lhs->ordinal < rhs->ordinal ? -1 : 1;
  return 0;
}

static iree_status_t loom_place_collect_loops(
    loom_place_function_context_t* context) {
  context->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_place_region_stack_push(
      context->pass->arena, &context->region_stack,
      loom_func_like_body(context->selection->func), 0));
  loom_place_region_stack_entry_t entry;
  while (loom_place_region_stack_pop(&context->region_stack, &entry)) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(entry.region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
        const loom_loop_like_t loop = loom_loop_like_cast(context->module, op);
        if (loom_loop_like_isa(loop)) {
          IREE_RETURN_IF_ERROR(
              loom_place_append_loop(context, op, entry.depth));
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_place_region_stack_push(
              context->pass->arena, &context->region_stack, regions[i],
              entry.depth + 1));
        }
      }
    }
  }
  if (context->loop_count > 1) {
    qsort(context->loops, context->loop_count, sizeof(*context->loops),
          loom_place_compare_loops);
  }
  return iree_ok_status();
}

static iree_status_t loom_place_emit_policy_error(
    loom_place_function_context_t* context, loom_op_t* op,
    iree_string_view_t reason) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(context->pass->info->name),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_049,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission);
}

static bool loom_place_loop_has_residency_policy(const loom_op_t* op) {
  return loom_scf_for_isa(op) &&
         (loom_scf_for_residency_minimum_is_present(op) ||
          !loom_attr_is_absent(
              loom_op_attrs(op)[loom_scf_for_residency_policy_ATTR_INDEX]));
}

static iree_status_t loom_place_resolve_loop_policy(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop,
    const loom_low_source_pressure_t* baseline, bool* out_emitted_error) {
  *out_emitted_error = false;
  loop->baseline_tier = baseline->minimum_tier;
  loop->required_tier = baseline->minimum_tier;
  loop->has_authored_policy = loom_place_loop_has_residency_policy(loop->op);
  loop->has_minimum_policy =
      loom_scf_for_isa(loop->op) &&
      loom_scf_for_residency_minimum_is_present(loop->op);
  if (!loop->has_authored_policy) return iree_ok_status();
  if (!context->has_residency_model) {
    loop->policy_failure_reason = IREE_SV("missing_residency_model");
    *out_emitted_error = true;
    return loom_place_emit_policy_error(
        context, loop->op,
        IREE_SV("the selected target has no residency model"));
  }
  if (!loop->has_minimum_policy) {
    return iree_ok_status();
  }

  int64_t minimum_tier = 0;
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      context->fact_table, loom_scf_for_residency_minimum(loop->op));
  if (!loom_value_facts_as_exact_i64(facts, &minimum_tier)) {
    loop->policy_failure_reason = IREE_SV("nonconstant_minimum");
    *out_emitted_error = true;
    return loom_place_emit_policy_error(
        context, loop->op,
        IREE_SV("the minimum tier is not an exact compile-time integer"));
  }
  if (minimum_tier < 0 || minimum_tier > UINT32_MAX) {
    loop->policy_failure_reason = IREE_SV("minimum_out_of_range");
    *out_emitted_error = true;
    return loom_place_emit_policy_error(
        context, loop->op,
        IREE_SV("the minimum tier is outside the nonnegative uint32 range"));
  }
  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .function = context->selection->func,
      .bundle = context->selection->target_bundle,
      .target_data = context->selection->target_data,
      .target_ref = context->selection->target_ref,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->fact_table,
      .value_domain = &context->value_domain,
      .arena = context->pass->arena,
  };
  const loom_target_residency_model_t* model =
      context->selection->policy->residency_model.fn(
          context->selection->policy->residency_model.user_data, &environment);
  if ((uint64_t)minimum_tier > model->best_tier) {
    loop->policy_failure_reason = IREE_SV("minimum_exceeds_target");
    *out_emitted_error = true;
    return loom_place_emit_policy_error(
        context, loop->op,
        IREE_SV("the minimum tier exceeds the selected target's best tier"));
  }
  loop->required_tier = (uint32_t)minimum_tier;
  context->required_tiers[0] =
      iree_max(context->required_tiers[0], loop->required_tier);
  context->has_minimum_requirement = true;
  context->minimum_required_tier =
      iree_max(context->minimum_required_tier, loop->required_tier);
  return iree_ok_status();
}

static loom_target_contract_query_environment_t
loom_place_make_query_environment(loom_place_function_context_t* context,
                                  iree_arena_allocator_t* arena) {
  return (loom_target_contract_query_environment_t){
      .module = context->module,
      .function = context->selection->func,
      .bundle = context->selection->target_bundle,
      .target_data = context->selection->target_data,
      .target_ref = context->selection->target_ref,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->fact_table,
      .value_domain = &context->value_domain,
      .arena = arena,
  };
}

static iree_status_t loom_place_analyze_pressure(
    loom_place_function_context_t* context, iree_arena_allocator_t* arena,
    loom_low_source_pressure_t* out_pressures) {
  const loom_target_contract_query_environment_t environment =
      loom_place_make_query_environment(context, arena);
  const loom_low_source_pressure_options_t options =
      loom_low_source_pressure_options_empty();
  return loom_low_source_pressure_analyze_regions(
      &environment, context->selection->policy, &options,
      context->pressure_regions, context->pressure_region_count, arena,
      out_pressures);
}

static iree_status_t loom_place_report_pressure_resources(
    loom_place_function_context_t* context,
    const loom_place_loop_record_t* loop, iree_string_view_t phase,
    const loom_low_source_pressure_t* pressure, iree_arena_allocator_t* arena) {
  if (!loom_target_compile_report_wants_details(
          context->compile_report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS) ||
      !loom_low_source_pressure_model_available(pressure)) {
    return iree_ok_status();
  }
  loom_target_residency_query_t query = {0};
  IREE_RETURN_IF_ERROR(loom_target_residency_query(
      pressure->residency_model, pressure->minimum_tier_direct_resource_units,
      pressure->direct_resource_count, arena, &query));
  if (query.tier != pressure->minimum_tier) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source pressure limiting vector selected tier %u but envelope "
        "recorded tier %u",
        query.tier, pressure->minimum_tier);
  }
  for (iree_host_size_t i = 0; i < query.resource_count; ++i) {
    const loom_target_residency_resource_evaluation_t* resource =
        &query.resources[i];
    const bool has_next_worse = iree_any_bit_set(
        resource->flags,
        LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_HAS_NEXT_WORSE_TIER);
    const loom_target_compile_report_source_low_residency_resource_row_t row = {
        .function_name = context->selection->function_name,
        .source_op_name = loom_op_name(context->module, loop->op),
        .source_op_kind = loop->op->kind,
        .phase = phase,
        .program_point = pressure->minimum_tier_point,
        .resource_name = resource->name,
        .resource_kind =
            resource->kind == LOOM_TARGET_RESIDENCY_RESOURCE_KIND_DIRECT
                ? IREE_SV("direct")
                : IREE_SV("derived"),
        .resource_id = resource->resource_id,
        .units = resource->units,
        .tier = resource->tier,
        .next_better_tier =
            query.has_next_better_tier ? query.next_better_tier : 0,
        .next_worse_tier = has_next_worse ? resource->next_worse_tier : 0,
        .next_worse_cliff_units =
            has_next_worse ? resource->next_worse_cliff_units : 0,
        .additional_units_to_next_worse_tier =
            has_next_worse ? resource->additional_units_to_next_worse_tier : 0,
        .reduction_units_to_next_better_tier =
            resource->reduction_units_to_next_better_tier,
        .limiting = iree_any_bit_set(
            resource->flags,
            LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_LIMITING),
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_source_low_residency_resource_row(
            context->compile_report, &row));
  }
  return iree_ok_status();
}

static iree_status_t loom_place_report_pressure_reserves(
    loom_place_function_context_t* context,
    const loom_low_source_pressure_t* pressure) {
  if (!loom_target_compile_report_wants_details(
          context->compile_report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS) ||
      !loom_low_source_pressure_model_available(pressure)) {
    return iree_ok_status();
  }
  for (iree_host_size_t reserve_index = 0;
       reserve_index < pressure->reserve_count; ++reserve_index) {
    const loom_low_lower_pressure_reserve_t* reserve =
        &pressure->reserves[reserve_index];
    for (uint16_t resource_id = 0;
         resource_id < pressure->direct_resource_count; ++resource_id) {
      const uint64_t units = reserve->direct_resource_units[resource_id];
      if (units == 0) continue;
      const loom_target_compile_report_source_low_residency_reserve_row_t row =
          {
              .function_name = context->selection->function_name,
              .reserve_name = reserve->name,
              .resource_name = pressure->residency_model->direct_resources
                                   .names[resource_id],
              .resource_id = resource_id,
              .units = units,
          };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_residency_reserve_row(
              context->compile_report, &row));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_place_prepare_pressure_contracts(
    loom_place_function_context_t* context, bool* out_emitted_error) {
  *out_emitted_error = false;
  context->pressure_region_count = context->loop_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, context->pressure_region_count,
      sizeof(*context->pressure_regions), (void**)&context->pressure_regions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, context->pressure_region_count,
      sizeof(*context->required_tiers), (void**)&context->required_tiers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, context->pressure_region_count,
      sizeof(*context->current_tiers), (void**)&context->current_tiers));
  context->pressure_regions[0] = loom_func_like_body(context->selection->func);
  for (iree_host_size_t i = 0; i < context->loop_count; ++i) {
    context->pressure_regions[i + 1] = context->loops[i].body;
  }

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  loom_low_source_pressure_t* baseline = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&analysis_arena, context->pressure_region_count,
                                sizeof(*baseline), (void**)&baseline);
  if (iree_status_is_ok(status)) {
    status = loom_place_analyze_pressure(context, &analysis_arena, baseline);
  }
  if (iree_status_is_ok(status)) {
    context->has_residency_model =
        loom_low_source_pressure_model_available(&baseline[0]);
    context->pressure_projection_complete =
        loom_low_source_pressure_projection_complete(&baseline[0]);
    if (baseline[0].reserve_count > UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "source pressure reserve count exceeds report representation");
    } else {
      context->pressure_reserve_count = (uint32_t)baseline[0].reserve_count;
    }
    context->required_tiers[0] = 0;
    context->current_tiers[0] = baseline[0].minimum_tier;
    for (iree_host_size_t i = 0;
         i < context->loop_count && iree_status_is_ok(status) &&
         !*out_emitted_error;
         ++i) {
      status = loom_place_resolve_loop_policy(
          context, &context->loops[i], &baseline[i + 1], out_emitted_error);
      context->required_tiers[i + 1] = context->loops[i].required_tier;
      context->current_tiers[i + 1] = baseline[i + 1].minimum_tier;
      context->reportable_loop_count = i + 1;
    }
  }
  if (iree_status_is_ok(status) &&
      loom_target_compile_report_wants_details(
          context->compile_report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    for (iree_host_size_t i = 0;
         i < context->reportable_loop_count && iree_status_is_ok(status); ++i) {
      status = loom_place_report_pressure_resources(
          context, &context->loops[i], IREE_SV("baseline"), &baseline[i + 1],
          &analysis_arena);
    }
    if (iree_status_is_ok(status)) {
      status = loom_place_report_pressure_reserves(context, &baseline[0]);
    }
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static iree_status_t loom_place_pressure_allows_current_ir(
    loom_place_function_context_t* context, bool* out_allowed) {
  *out_allowed = true;
  if (!context->has_residency_model) return iree_ok_status();

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  loom_low_source_pressure_t* pressure = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&analysis_arena, context->pressure_region_count,
                                sizeof(*pressure), (void**)&pressure);
  if (iree_status_is_ok(status)) {
    status = loom_place_analyze_pressure(context, &analysis_arena, pressure);
  }
  if (iree_status_is_ok(status)) {
    bool has_unsatisfied_scope = false;
    bool made_progress = false;
    for (iree_host_size_t i = 0; i < context->pressure_region_count; ++i) {
      const uint32_t current_tier = context->current_tiers[i];
      const uint32_t required_tier = context->required_tiers[i];
      const uint32_t accepted_floor = iree_min(current_tier, required_tier);
      if (pressure[i].minimum_tier < accepted_floor) {
        *out_allowed = false;
        break;
      }
      if (current_tier < required_tier) {
        has_unsatisfied_scope = true;
        made_progress =
            made_progress || pressure[i].minimum_tier > current_tier;
      }
    }
    if (*out_allowed && has_unsatisfied_scope && !made_progress) {
      *out_allowed = false;
    }
    if (*out_allowed) {
      for (iree_host_size_t i = 0; i < context->pressure_region_count; ++i) {
        context->current_tiers[i] = pressure[i].minimum_tier;
      }
    }
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static iree_status_t loom_place_validate_final_pressure_contracts(
    loom_place_function_context_t* context, bool* out_emitted_error) {
  *out_emitted_error = false;
  for (iree_host_size_t i = 0; i < context->loop_count; ++i) {
    const loom_place_loop_record_t* loop = &context->loops[i];
    if (!loop->has_authored_policy) continue;
    if (context->current_tiers[i + 1] < loop->required_tier ||
        (loop->has_minimum_policy &&
         context->current_tiers[0] < loop->required_tier)) {
      *out_emitted_error = true;
      return loom_place_emit_policy_error(
          context, loop->op,
          IREE_SV("no legal invariant placement satisfies the residency "
                  "contract"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_place_report_decisions(
    loom_place_function_context_t* context) {
  if (!loom_target_compile_report_wants_details(
          context->compile_report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < context->reportable_loop_count; ++i) {
    const loom_place_loop_record_t* loop = &context->loops[i];
    const uint32_t selected_tier = context->current_tiers[i + 1];
    const bool policy_resolution_failed =
        !iree_string_view_is_empty(loop->policy_failure_reason);
    const bool contract_unsatisfied =
        policy_resolution_failed ||
        (loop->has_authored_policy &&
         (selected_tier < loop->required_tier ||
          (loop->has_minimum_policy &&
           context->current_tiers[0] < loop->required_tier)));
    const iree_string_view_t policy =
        loop->has_minimum_policy    ? IREE_SV("minimum")
        : loop->has_authored_policy ? IREE_SV("preserve")
                                    : IREE_SV("automatic");
    iree_string_view_t outcome = IREE_SV("unchanged");
    iree_string_view_t reason = IREE_SV("no_legal_candidates");
    if (policy_resolution_failed) {
      outcome = IREE_SV("rejected");
      reason = loop->policy_failure_reason;
    } else if (contract_unsatisfied) {
      outcome = IREE_SV("unsatisfied");
      reason = IREE_SV("contract_unsatisfied");
    } else if (loop->selected_count != 0) {
      outcome = IREE_SV("selected");
      reason = loop->selected_count == loop->candidate_count
                   ? IREE_SV("dynamic_work_removed")
                   : IREE_SV("residency_cliff");
    } else if (loop->candidate_count != 0) {
      outcome = IREE_SV("retained");
      reason = IREE_SV("residency_cliff");
    }

    const loom_target_compile_report_source_low_residency_row_t row = {
        .function_name = context->selection->function_name,
        .source_op_name = loom_op_name(context->module, loop->op),
        .source_op_kind = loop->op->kind,
        .policy = policy,
        .outcome = outcome,
        .reason = reason,
        .baseline_tier = loop->baseline_tier,
        .selected_tier = selected_tier,
        .required_tier = loop->required_tier,
        .candidate_count = loop->candidate_count,
        .selected_count = loop->selected_count,
        .rejected_count = loop->candidate_count - loop->selected_count,
        .reserve_count = context->pressure_reserve_count,
        .projection_complete = context->pressure_projection_complete,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_source_low_residency_row(
            context->compile_report, &row));
  }

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  loom_low_source_pressure_t* selected = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&analysis_arena, context->pressure_region_count,
                                sizeof(*selected), (void**)&selected);
  if (iree_status_is_ok(status)) {
    status = loom_place_analyze_pressure(context, &analysis_arena, selected);
  }
  for (iree_host_size_t i = 0;
       i < context->reportable_loop_count && iree_status_is_ok(status); ++i) {
    if (selected[i + 1].minimum_tier != context->current_tiers[i + 1]) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "reported selected source pressure tier changed from %u to %u",
          context->current_tiers[i + 1], selected[i + 1].minimum_tier);
      break;
    }
    status = loom_place_report_pressure_resources(
        context, &context->loops[i], IREE_SV("selected"), &selected[i + 1],
        &analysis_arena);
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static iree_status_t loom_place_append_candidate(
    loom_place_function_context_t* context,
    loom_place_candidate_list_t* candidates, loom_op_t* op,
    uint32_t discovery_wave) {
  if (op->next_op == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "legal invariant placement candidate has no restoration anchor");
  }
  if (candidates->count == candidates->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        context->pass->arena, candidates->count, candidates->count + 1,
        sizeof(*candidates->values), &candidates->capacity,
        (void**)&candidates->values));
  }
  if (candidates->count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "loop invariant candidate count exceeds uint32");
  }
  const uint32_t candidate_index = (uint32_t)candidates->count;
  candidates->values[candidates->count++] = (loom_place_candidate_t){
      .op = op,
      .restore_before_op = op->next_op,
      .discovery_wave = discovery_wave,
      .dependency_parent = candidate_index,
  };
  ++loom_place_loop_invariants_statistics(context->pass)->candidates;
  return iree_ok_status();
}

static bool loom_place_op_has_nonoperand_result_uses(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (loom_module_value_has_predicate_attribute_uses(module, results[i]) ||
        loom_module_value_has_type_uses(module, results[i])) {
      return true;
    }
  }
  return false;
}

// Collects one simultaneous legality wave. Candidates are not moved while the
// region tree is inspected, so a later operation cannot become available only
// because an earlier lexical operation happened to be visited first.
static iree_status_t loom_place_collect_candidate_wave(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop,
    uint32_t discovery_wave, loom_place_candidate_list_t* candidates,
    iree_host_size_t* out_wave_start) {
  *out_wave_start = candidates->count;
  context->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_place_region_stack_push(
      context->pass->arena, &context->region_stack, loop->body, 0));

  loom_place_region_stack_entry_t entry;
  while (loom_place_region_stack_pop(&context->region_stack, &entry)) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(entry.region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
        loom_motion_loop_hoist_result_t legality = {0};
        IREE_RETURN_IF_ERROR(loom_motion_subtree_evaluate_hoist_before_loop(
            &context->motion, loom_loop_like_cast(context->module, loop->op),
            op, &legality));
        if (loom_motion_loop_hoist_result_is_legal(&legality)) {
          // Exact repair reconstructs finite expression recipes and redirects
          // ordinary operand uses through their retained boundary. Structured,
          // successor-bearing, and tied operations, plus results referenced by
          // predicates or types, remain authored until the marker can represent
          // them exactly.
          if (op->region_count != 0 || op->successor_count != 0 ||
              op->tied_result_count != 0 ||
              loom_place_op_has_nonoperand_result_uses(context->module, op)) {
            continue;
          }
          IREE_RETURN_IF_ERROR(loom_place_append_candidate(context, candidates,
                                                           op, discovery_wave));
          continue;
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_place_region_stack_push(
              context->pass->arena, &context->region_stack, regions[i],
              entry.depth + 1));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_place_move_candidate_range(
    loom_place_function_context_t* context, loom_op_t* loop_op,
    loom_place_candidate_list_t* candidates, iree_host_size_t start,
    iree_host_size_t end) {
  iree_host_size_t moved_end = start;
  iree_status_t status = iree_ok_status();
  while (moved_end < end && iree_status_is_ok(status)) {
    status = loom_rewriter_move_before(
        &context->rewriter, candidates->values[moved_end].op, loop_op);
    if (iree_status_is_ok(status)) ++moved_end;
  }
  for (iree_host_size_t i = moved_end; i > start && !iree_status_is_ok(status);
       --i) {
    const iree_status_t restore_status = loom_rewriter_move_before(
        &context->rewriter, candidates->values[i - 1].op,
        candidates->values[i - 1].restore_before_op);
    status = iree_status_join(status, restore_status);
  }
  return status;
}

static iree_status_t loom_place_restore_candidate_indices(
    loom_place_function_context_t* context,
    loom_place_candidate_list_t* candidates, const uint32_t* indices,
    iree_host_size_t index_count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = index_count; i > 0; --i) {
    loom_place_candidate_t* candidate = &candidates->values[indices[i - 1]];
    status = iree_status_join(
        status, loom_rewriter_move_before(&context->rewriter, candidate->op,
                                          candidate->restore_before_op));
  }
  return status;
}

static iree_status_t loom_place_restore_all_candidates(
    loom_place_function_context_t* context,
    loom_place_candidate_list_t* candidates) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = candidates->count; i > 0; --i) {
    loom_place_candidate_t* candidate = &candidates->values[i - 1];
    status = iree_status_join(
        status, loom_rewriter_move_before(&context->rewriter, candidate->op,
                                          candidate->restore_before_op));
  }
  return status;
}

static uint32_t loom_place_candidate_dependency_root(
    loom_place_candidate_list_t* candidates, uint32_t candidate_index) {
  loom_place_candidate_t* candidate = &candidates->values[candidate_index];
  if (candidate->dependency_parent == candidate_index) return candidate_index;
  candidate->dependency_parent = loom_place_candidate_dependency_root(
      candidates, candidate->dependency_parent);
  return candidate->dependency_parent;
}

static void loom_place_candidate_dependency_union(
    loom_place_candidate_list_t* candidates, uint32_t lhs_index,
    uint32_t rhs_index) {
  const uint32_t lhs_root =
      loom_place_candidate_dependency_root(candidates, lhs_index);
  const uint32_t rhs_root =
      loom_place_candidate_dependency_root(candidates, rhs_index);
  if (lhs_root == rhs_root) return;
  // The earliest candidate remains the stable component root.
  if (lhs_root < rhs_root) {
    candidates->values[rhs_root].dependency_parent = lhs_root;
  } else {
    candidates->values[lhs_root].dependency_parent = rhs_root;
  }
}

static iree_status_t loom_place_build_candidate_dependencies(
    loom_place_function_context_t* context,
    loom_place_candidate_list_t* candidates) {
  uint32_t* candidate_by_value = NULL;
  if (context->value_domain.value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->value_domain.value_count,
        sizeof(*candidate_by_value), (void**)&candidate_by_value));
    for (loom_value_ordinal_t i = 0; i < context->value_domain.value_count;
         ++i) {
      candidate_by_value[i] = UINT32_MAX;
    }
  }
  for (uint32_t i = 0; i < (uint32_t)candidates->count; ++i) {
    const loom_op_t* op = candidates->values[i].op;
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t j = 0; j < op->result_count; ++j) {
      const loom_value_ordinal_t value_ordinal =
          loom_local_value_domain_try_ordinal(&context->value_domain,
                                              results[j]);
      if (value_ordinal != LOOM_VALUE_ORDINAL_INVALID) {
        candidate_by_value[value_ordinal] = i;
      }
    }
  }
  for (uint32_t i = 0; i < (uint32_t)candidates->count; ++i) {
    const loom_op_t* op = candidates->values[i].op;
    const loom_value_id_t* operands = loom_op_const_operands(op);
    for (uint16_t j = 0; j < op->operand_count; ++j) {
      const loom_value_ordinal_t value_ordinal =
          loom_local_value_domain_try_ordinal(&context->value_domain,
                                              operands[j]);
      if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) continue;
      const uint32_t dependency_index = candidate_by_value[value_ordinal];
      if (dependency_index != UINT32_MAX) {
        loom_place_candidate_dependency_union(candidates, i, dependency_index);
      }
    }

    // A later fixed-point wave has at least one capture that became available
    // after prior candidates moved. Ordinary operands above identify the usual
    // dependency exactly. Types, predicates, and encoding attributes may also
    // capture SSA, so conservatively keep the wave with all earlier candidates
    // rather than splitting a dependency closure we cannot enumerate here.
    if (candidates->values[i].discovery_wave != 0) {
      for (uint32_t j = 0; j < i; ++j) {
        if (candidates->values[j].discovery_wave <
            candidates->values[i].discovery_wave) {
          loom_place_candidate_dependency_union(candidates, i, j);
        }
      }
    }
  }
  for (uint32_t i = 0; i < (uint32_t)candidates->count; ++i) {
    (void)loom_place_candidate_dependency_root(candidates, i);
  }
  return iree_ok_status();
}

// Applies one cumulative candidate frontier, validates every protected scope,
// and either commits the frontier or restores the exact prior placement.
static iree_status_t loom_place_trial_candidate_indices(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop,
    loom_place_candidate_list_t* candidates, const uint32_t* indices,
    iree_host_size_t index_count, uint32_t* moved_indices,
    iree_host_size_t* out_moved_count, bool* out_accepted) {
  *out_moved_count = 0;
  *out_accepted = false;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < index_count && iree_status_is_ok(status);
       ++i) {
    const uint32_t candidate_index = indices[i];
    loom_place_candidate_t* candidate = &candidates->values[candidate_index];
    if (candidate->accepted) continue;
    loom_motion_loop_hoist_result_t legality = {0};
    status = loom_motion_subtree_evaluate_hoist_before_loop(
        &context->motion, loom_loop_like_cast(context->module, loop->op),
        candidate->op, &legality);
    if (!iree_status_is_ok(status)) break;
    if (!loom_motion_loop_hoist_result_is_legal(&legality)) continue;
    status =
        loom_rewriter_move_before(&context->rewriter, candidate->op, loop->op);
    if (!iree_status_is_ok(status)) break;
    moved_indices[(*out_moved_count)++] = candidate_index;
  }
  if (!iree_status_is_ok(status)) {
    const iree_status_t restore_status = loom_place_restore_candidate_indices(
        context, candidates, moved_indices, *out_moved_count);
    return iree_status_join(status, restore_status);
  }
  if (*out_moved_count == 0) return iree_ok_status();

  ++loom_place_loop_invariants_statistics(context->pass)->pressure_evaluations;
  bool pressure_allowed = true;
  status = loom_place_pressure_allows_current_ir(context, &pressure_allowed);
  if (!iree_status_is_ok(status) || !pressure_allowed) {
    const iree_status_t restore_status = loom_place_restore_candidate_indices(
        context, candidates, moved_indices, *out_moved_count);
    return iree_status_join(status, restore_status);
  }
  for (iree_host_size_t i = 0; i < *out_moved_count; ++i) {
    candidates->values[moved_indices[i]].accepted = true;
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_place_refine_candidate_component(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop,
    loom_place_candidate_list_t* candidates, const uint32_t* component_indices,
    iree_host_size_t component_count, uint32_t* moved_indices,
    uint32_t* refinement_evaluations) {
  if (component_count == 0 ||
      *refinement_evaluations >= LOOM_PLACE_MAX_REFINEMENT_EVALUATIONS) {
    return iree_ok_status();
  }

  // Prefer the largest dependency-ordered prefix. Trying the complete closure
  // first is important: a provider may temporarily increase pressure while a
  // dependent consumer moved in the same frontier shortens that live range.
  iree_host_size_t selected_prefix_count = 0;
  for (iree_host_size_t prefix_count = component_count; prefix_count > 0;
       --prefix_count) {
    if (*refinement_evaluations >= LOOM_PLACE_MAX_REFINEMENT_EVALUATIONS) {
      break;
    }
    ++*refinement_evaluations;
    iree_host_size_t moved_count = 0;
    bool accepted = false;
    IREE_RETURN_IF_ERROR(loom_place_trial_candidate_indices(
        context, loop, candidates, component_indices, prefix_count,
        moved_indices, &moved_count, &accepted));
    if (accepted) {
      selected_prefix_count = prefix_count;
      break;
    }
  }

  // A connected component can contain multiple fan-out branches. Once the
  // strongest prefix is fixed, retain any later branch that remains legal and
  // same-tier in the resulting cumulative context.
  for (iree_host_size_t i = selected_prefix_count; i < component_count; ++i) {
    if (*refinement_evaluations >= LOOM_PLACE_MAX_REFINEMENT_EVALUATIONS) {
      break;
    }
    ++*refinement_evaluations;
    iree_host_size_t moved_count = 0;
    bool accepted = false;
    IREE_RETURN_IF_ERROR(loom_place_trial_candidate_indices(
        context, loop, candidates, &component_indices[i], 1, moved_indices,
        &moved_count, &accepted));
  }
  return iree_ok_status();
}

static bool loom_place_use_is_nested_in_loop(const loom_op_t* user_op,
                                             void* user_data) {
  const loom_op_t* loop_op = (const loom_op_t*)user_data;
  for (const loom_op_t* ancestor = user_op; ancestor != NULL;
       ancestor = ancestor->parent_op) {
    if (ancestor == loop_op) return true;
  }
  return false;
}

static bool loom_place_value_has_use_nested_in_loop(const loom_module_t* module,
                                                    loom_value_id_t value_id,
                                                    const loom_op_t* loop_op) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_place_use_is_nested_in_loop(loom_use_user_op(uses[i]),
                                         (void*)loop_op)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_place_record_candidate_frontier(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop,
    const loom_place_candidate_list_t* candidates) {
  const bool preserves_baseline = !loop->has_minimum_policy;
  for (iree_host_size_t i = 0; i < candidates->count; ++i) {
    const loom_place_candidate_t* candidate = &candidates->values[i];
    const loom_value_id_t* results = loom_op_const_results(candidate->op);
    for (uint16_t result_index = 0; result_index < candidate->op->result_count;
         ++result_index) {
      const loom_value_id_t source = results[result_index];
      const loom_value_t* source_value =
          loom_module_value(context->module, source);
      if (loom_value_is_consumed(source_value) ||
          !loom_place_value_has_use_nested_in_loop(context->module, source,
                                                   loop->op)) {
        continue;
      }

      uint32_t recompute_cost = 0;
      const loom_use_t* uses = loom_value_uses(source_value);
      for (uint32_t use_index = 0; use_index < source_value->use_count;
           ++use_index) {
        if (loom_place_use_is_nested_in_loop(loom_use_user_op(uses[use_index]),
                                             loop->op)) {
          if (recompute_cost == UINT32_MAX) {
            return iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "residency candidate recompute cost exceeds uint32");
          }
          ++recompute_cost;
        }
      }
      if (context->next_candidate_id == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "residency candidate ID exceeds uint32");
      }
      const uint32_t candidate_id = context->next_candidate_id++;
      if (loom_target_compile_report_wants_details(
              context->compile_report,
              LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
        const loom_target_compile_report_residency_candidate_row_t report_row =
            {
                .function_name = context->selection->function_name,
                .source_op_name = loom_op_name(context->module, candidate->op),
                .source_op_kind = candidate->op->kind,
                .candidate_id = candidate_id,
                .stage = IREE_SV("source"),
                .outcome = candidate->accepted ? IREE_SV("selected")
                                               : IREE_SV("retained"),
                .projected_recompute_cost = recompute_cost,
                .preserves_baseline = preserves_baseline,
            };
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_residency_candidate_row(
                context->compile_report, &report_row));
      }
      if (candidate->accepted) {
        IREE_RETURN_IF_ERROR(loom_place_append_selected_boundary(
            context, candidate->op, loop->op, result_index, candidate_id,
            recompute_cost, preserves_baseline));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_place_invariants_for_loop(
    loom_place_function_context_t* context, loom_place_loop_record_t* loop) {
  // Automatic placement and preserve both protect the current cumulative
  // function tier. An explicit numeric minimum intentionally replaces that
  // default objective with the authored floor for moves across this loop.
  context->required_tiers[0] = loop->has_minimum_policy
                                   ? context->minimum_required_tier
                                   : context->current_tiers[0];
  loom_place_candidate_list_t candidates = {0};
  for (uint32_t discovery_wave = 0;; ++discovery_wave) {
    iree_host_size_t wave_start = 0;
    IREE_RETURN_IF_ERROR(loom_place_collect_candidate_wave(
        context, loop, discovery_wave, &candidates, &wave_start));
    if (wave_start == candidates.count) break;
    IREE_RETURN_IF_ERROR(loom_place_move_candidate_range(
        context, loop->op, &candidates, wave_start, candidates.count));
    if (discovery_wave == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "loop invariant discovery wave overflow");
    }
  }
  loop->candidate_count = (uint32_t)candidates.count;
  if (candidates.count == 0) return iree_ok_status();

  // Fast path: validate the complete legal dependency closure in one liveness
  // and residency evaluation. This is the expected production path.
  ++loom_place_loop_invariants_statistics(context->pass)->pressure_evaluations;
  bool complete_frontier_allowed = true;
  iree_status_t status = loom_place_pressure_allows_current_ir(
      context, &complete_frontier_allowed);
  if (!iree_status_is_ok(status)) {
    const iree_status_t restore_status =
        loom_place_restore_all_candidates(context, &candidates);
    return iree_status_join(status, restore_status);
  }
  if (complete_frontier_allowed) {
    for (iree_host_size_t i = 0; i < candidates.count; ++i) {
      candidates.values[i].accepted = true;
    }
    loom_place_loop_invariants_statistics(context->pass)->ops_hoisted +=
        (int64_t)candidates.count;
    loop->selected_count = (uint32_t)candidates.count;
    context->changed = true;
    return context->has_residency_model ? loom_place_record_candidate_frontier(
                                              context, loop, &candidates)
                                        : iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_place_restore_all_candidates(context, &candidates));
  IREE_RETURN_IF_ERROR(
      loom_place_build_candidate_dependencies(context, &candidates));

  uint32_t* component_indices = NULL;
  uint32_t* moved_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, candidates.count, sizeof(*component_indices),
      (void**)&component_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, candidates.count, sizeof(*moved_indices),
      (void**)&moved_indices));
  uint32_t refinement_evaluations = 0;
  for (uint32_t root_index = 0; root_index < (uint32_t)candidates.count;
       ++root_index) {
    if (loom_place_candidate_dependency_root(&candidates, root_index) !=
        root_index) {
      continue;
    }
    iree_host_size_t component_count = 0;
    for (uint32_t i = 0; i < (uint32_t)candidates.count; ++i) {
      if (loom_place_candidate_dependency_root(&candidates, i) == root_index) {
        component_indices[component_count++] = i;
      }
    }
    IREE_RETURN_IF_ERROR(loom_place_refine_candidate_component(
        context, loop, &candidates, component_indices, component_count,
        moved_indices, &refinement_evaluations));
  }

  iree_host_size_t accepted_count = 0;
  for (iree_host_size_t i = 0; i < candidates.count; ++i) {
    if (!candidates.values[i].accepted) continue;
    ++accepted_count;
  }
  loom_place_loop_invariants_statistics(context->pass)->ops_hoisted +=
      (int64_t)accepted_count;
  loop->selected_count = (uint32_t)accepted_count;
  loom_place_loop_invariants_statistics(context->pass)->residency_rejections +=
      (int64_t)(candidates.count - accepted_count);
  if (refinement_evaluations >= LOOM_PLACE_MAX_REFINEMENT_EVALUATIONS &&
      accepted_count != candidates.count) {
    ++loom_place_loop_invariants_statistics(context->pass)->refinement_limits;
  }
  context->changed = context->changed || accepted_count != 0;
  return context->has_residency_model
             ? loom_place_record_candidate_frontier(context, loop, &candidates)
             : iree_ok_status();
}

static iree_status_t loom_place_copy_loop_result_types(
    loom_place_function_context_t* context, loom_op_t* op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (op->result_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, op->result_count, sizeof(**out_result_types),
      (void**)out_result_types));
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    (*out_result_types)[i] =
        loom_module_value_type(context->module, results[i]);
  }
  return iree_ok_status();
}

static iree_status_t loom_place_materialize_selected_boundaries(
    loom_place_function_context_t* context) {
  const bool preserve_baseline = context->preserves_baseline;
  if (!context->has_minimum_requirement && !preserve_baseline) {
    return iree_ok_status();
  }

  loom_block_t* entry_block =
      loom_region_entry_block(loom_func_like_body(context->selection->func));
  if (entry_block->first_op == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source function with residency contract has no entry operation");
  }
  loom_builder_set_before(&context->rewriter.builder, entry_block->first_op);
  loom_op_t* requirement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_residency_require_build(
      &context->rewriter.builder,
      context->has_minimum_requirement ? (int64_t)context->minimum_required_tier
                                       : -1,
      preserve_baseline,
      preserve_baseline ? (int64_t)context->projected_baseline_tier : 0,
      context->selection->func.op->location, &requirement_op));

  for (iree_host_size_t i = 0; i < context->selected_boundary_count; ++i) {
    const loom_place_selected_boundary_t* boundary =
        &context->selected_boundaries[i];
    const loom_value_id_t* results = loom_op_const_results(boundary->op);
    const loom_value_id_t source = results[boundary->result_index];
    const loom_value_t* source_value =
        loom_module_value(context->module, source);
    if (loom_value_is_consumed(source_value) ||
        !loom_place_value_has_use_nested_in_loop(context->module, source,
                                                 boundary->loop_op)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "recorded residency boundary was invalidated before materialization");
    }

    loom_builder_set_before(&context->rewriter.builder, boundary->loop_op);
    loom_op_t* marker_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_residency_candidate_build_for_proven_source(
        &context->rewriter.builder, boundary->candidate_id,
        boundary->recompute_cost, boundary->op, boundary->result_index,
        boundary->preserves_baseline, boundary->loop_op->location, &marker_op));
    const loom_value_id_t marker_result =
        loom_scf_residency_candidate_result(marker_op);
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        &context->rewriter, source, marker_result, IREE_SV("residency")));
    IREE_RETURN_IF_ERROR(loom_value_replace_uses_if(
        context->module, source, marker_result,
        loom_place_use_is_nested_in_loop, boundary->loop_op));
    IREE_ASSERT_GT(loom_module_value(context->module, marker_result)->use_count,
                   0u);
  }
  context->changed = true;
  return iree_ok_status();
}

static iree_status_t loom_place_copy_loop_tied_results(
    loom_place_function_context_t* context, loom_op_t* op,
    loom_tied_result_t** out_tied_results) {
  *out_tied_results = NULL;
  if (op->tied_result_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, op->tied_result_count, sizeof(**out_tied_results),
      (void**)out_tied_results));
  memcpy(*out_tied_results, loom_op_tied_results(op),
         op->tied_result_count * sizeof(**out_tied_results));
  if (loom_scf_for_residency_minimum_is_present(op)) {
    const uint16_t minimum_operand_index = op->operand_count - 1;
    for (uint16_t i = 0; i < op->tied_result_count; ++i) {
      if ((*out_tied_results)[i].operand_index == minimum_operand_index) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "scf.for residency minimum cannot be tied to a loop result");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_place_rebuild_loop_without_residency(
    loom_place_function_context_t* context, loom_op_t* op) {
  loom_scf_for_build_flags_t build_flags = 0;
  loom_value_id_t unroll_factor = LOOM_VALUE_ID_INVALID;
  if (loom_scf_for_unroll_factor_is_present(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    unroll_factor = loom_scf_for_unroll_factor(op);
  }
  loom_scf_for_unroll_policy_t unroll_policy = 0;
  if (!loom_attr_is_absent(
          loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX])) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = loom_scf_for_unroll_policy(op);
  }
  loom_scf_for_unroll_schedule_t unroll_schedule = 0;
  if (!loom_attr_is_absent(
          loom_op_attrs(op)[loom_scf_for_unroll_schedule_ATTR_INDEX])) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    unroll_schedule = loom_scf_for_unroll_schedule(op);
  }

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_place_copy_loop_result_types(context, op, &result_types));
  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      loom_place_copy_loop_tied_results(context, op, &tied_results));
  const loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_op_t* old_yield =
      loom_region_entry_block(loom_scf_for_body(op))->last_op;
  const loom_value_slice_t yielded_values = loom_scf_yield_values(old_yield);

  loom_builder_set_before(&context->rewriter.builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(&context->rewriter);
  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter.builder, build_flags, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), loom_scf_for_step(op), iter_args.values,
      iter_args.count, result_types, op->result_count, tied_results,
      op->tied_result_count, unroll_factor, unroll_policy, unroll_schedule,
      LOOM_VALUE_ID_INVALID, /*residency_policy=*/0, op->location, &new_loop));

  loom_region_t* old_body = loom_scf_for_body(op);
  loom_block_t* old_block = loom_region_entry_block(old_body);
  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&context->rewriter.builder, new_loop, new_body);
  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(&context->rewriter.builder, yielded_values.values,
                           yielded_values.count, op->location, &new_yield));
  loom_builder_restore(&context->rewriter.builder, saved_ip);

  loom_block_t* new_block = loom_region_entry_block(new_body);
  for (uint16_t i = 0; i < old_block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        &context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(new_block, i)));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        &context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(new_block, i)));
  }
  loom_op_t* child_op = old_block->first_op;
  while (child_op != NULL && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(&context->rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  const loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      &context->rewriter, op, new_results.values, new_results.count,
      value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      &context->rewriter, op, new_results.values, new_results.count);
}

static iree_status_t loom_place_consume_policies(
    loom_place_function_context_t* context) {
  for (iree_host_size_t i = 0; i < context->loop_count; ++i) {
    loom_place_loop_record_t* loop = &context->loops[i];
    if (!loop->has_authored_policy) continue;
    if (loop->has_minimum_policy) {
      IREE_RETURN_IF_ERROR(
          loom_place_rebuild_loop_without_residency(context, loop->op));
    } else {
      IREE_RETURN_IF_ERROR(loom_rewriter_set_attr(
          &context->rewriter, loop->op,
          loom_scf_for_residency_policy_ATTR_INDEX, loom_attr_absent()));
    }
    ++loom_place_loop_invariants_statistics(context->pass)->policies_consumed;
    context->changed = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_place_function(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_source_selection_t* selection, bool* out_emitted_error) {
  *out_emitted_error = false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(
          descriptor_registry,
          selection->target_bundle->config->contract_set_key);
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected target descriptor set is unavailable");
  }

  loom_place_function_context_t context = {
      .pass = pass,
      .module = module,
      .selection = selection,
      .descriptor_set = descriptor_set,
      .compile_report = loom_low_pass_capability_compile_report(
          loom_low_pass_capability_from_pass(pass)),
  };
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&context.rewriter, module, pass->arena));
  iree_status_t status = loom_place_collect_loops(&context);
  if (iree_status_is_ok(status) && context.loop_count != 0) {
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            selection->func, selection->target_bundle, selection->target_data),
        &context.fact_table);
  }
  if (iree_status_is_ok(status) && context.loop_count != 0) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module, loom_func_like_body(selection->func), pass->arena,
        &context.value_domain);
  }
  if (iree_status_is_ok(status) && context.loop_count != 0) {
    status = loom_motion_analysis_initialize(module, context.fact_table,
                                             &context.value_domain, pass->arena,
                                             &context.motion);
  }
  if (iree_status_is_ok(status) && context.loop_count != 0) {
    status = loom_place_prepare_pressure_contracts(&context, out_emitted_error);
  }
  if (iree_status_is_ok(status) && !*out_emitted_error &&
      context.loop_count != 0) {
    if (!context.has_residency_model) {
      ++loom_place_loop_invariants_statistics(pass)->functions_without_model;
    }
    for (iree_host_size_t i = 0;
         i < context.loop_count && iree_status_is_ok(status); ++i) {
      status = loom_place_invariants_for_loop(&context, &context.loops[i]);
    }
  }
  if (iree_status_is_ok(status) && context.preserves_baseline) {
    context.projected_baseline_tier = context.current_tiers[0];
  }
  if (iree_status_is_ok(status)) {
    status = loom_place_report_decisions(&context);
  }
  if (iree_status_is_ok(status) && !*out_emitted_error) {
    status = loom_place_validate_final_pressure_contracts(&context,
                                                          out_emitted_error);
  }
  if (iree_status_is_ok(status) && !*out_emitted_error) {
    status = loom_place_materialize_selected_boundaries(&context);
  }
  loom_local_value_domain_release(&context.value_domain);
  if (iree_status_is_ok(status) && !*out_emitted_error) {
    status = loom_place_consume_policies(&context);
  }
  if (iree_status_is_ok(status) && context.changed) {
    loom_pass_mark_changed(pass);
    if (pass->value_facts != NULL) {
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }
  loom_place_loop_invariants_statistics(pass)->loops +=
      (int64_t)context.loop_count;
  loom_rewriter_deinitialize(&context.rewriter);
  return status;
}

iree_status_t loom_place_loop_invariants_run(loom_pass_t* pass,
                                             loom_module_t* module) {
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  if (descriptor_registry == NULL || policy_registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "place-loop-invariants requires target-low descriptor and policy "
        "registries");
  }
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);

  iree_arena_allocator_t run_arena;
  iree_arena_initialize(module->arena.block_pool, &run_arena);
  const loom_low_source_selection_options_t selection_options = {
      .policy_registry = policy_registry,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .lowering_kind = IREE_SV("place-loop-invariants"),
      .target_selection =
          loom_target_pass_capability_target_selection(target_capability),
      .target_ref = loom_target_pass_capability_target_ref(target_capability),
  };
  loom_low_source_selection_list_t selection_list = {0};
  iree_status_t status = loom_low_select_source_funcs(
      module, &selection_options, &run_arena, &selection_list);
  bool emitted_error = false;
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) && !emitted_error;
       ++i) {
    status = loom_place_function(pass, module, descriptor_registry,
                                 &selection_list.values[i], &emitted_error);
    if (iree_status_is_ok(status) && !emitted_error) ++function_count;
  }
  iree_arena_deinitialize(&run_arena);
  if (iree_status_is_ok(status)) {
    loom_place_loop_invariants_statistics(pass)->functions +=
        (int64_t)function_count;
  }
  return status;
}
