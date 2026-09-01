// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/contract_vector.h"
#include "loom/analysis/kernel_async_legality.h"
#include "loom/analysis/vector_memory_footprint.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/function.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/codegen/low/lower/report.h"
#include "loom/codegen/low/lower/storage.h"
#include "loom/codegen/low/lower/structural.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"

static iree_status_t loom_low_lowering_frame_initialize_value_ordinals(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  // Target-legalization query scopes can inspect source ops before CFG
  // conversion, so nested source-region values must be ordinal-addressable even
  // when the final source-to-low boundary expects CFG.
  return loom_local_value_domain_acquire_for_region_tree(
      context->module, source_body, &context->function_arena,
      &context->lowering.value_domain);
}

static void loom_low_lowering_frame_deinitialize(
    loom_low_lower_context_t* context) {
  loom_local_value_domain_release(&context->lowering.value_domain);
}

typedef struct loom_low_lower_contract_query_state_t {
  // Mutable lowering context whose scoped arena and target policy back the
  // read-only contract query.
  loom_low_lower_context_t* context;
  // Target contract environment provided by target-low legality.
  const loom_target_contract_query_environment_t* environment;
} loom_low_lower_contract_query_state_t;

static iree_status_t loom_low_lower_contract_query_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_map_contract_value_callback_t map_contract_value =
      state->context->policy->map_contract_value;
  if (map_contract_value.fn == NULL) {
    return iree_ok_status();
  }
  return map_contract_value.fn(map_contract_value.user_data, state->environment,
                               source_op, source_value_id, out_mapped_value);
}

static iree_status_t loom_low_lower_contract_query_can_materialize(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id,
    bool* out_can_materialize) {
  (void)match_context;
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  const loom_low_lower_value_materializer_t* materializer =
      &rule_set->materializers[materializer_index];
  return materializer->can_materialize(state->context, source_op,
                                       source_value_id, out_can_materialize);
}

struct loom_low_lower_source_query_scope_t {
  // Read-only lowering context backing source-to-low contract queries.
  loom_low_lower_context_t context;
  // Diagnostic/result scratch required by the lowering context contract.
  loom_low_lower_result_t result;
  // True after context.lowering.value_domain has acquired module scratch.
  bool value_domain_initialized;
};

static iree_status_t loom_low_lower_query_target_contract_from_context(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_descriptor_set_t* saved_descriptor_set =
      context->descriptor_set;
  loom_value_fact_table_t* saved_fact_table = context->lowering.fact_table;
  const bool fact_table_changed =
      saved_fact_table != (loom_value_fact_table_t*)environment->fact_table;
  context->descriptor_set = environment->descriptor_set;
  context->lowering.fact_table =
      (loom_value_fact_table_t*)environment->fact_table;
  if (fact_table_changed) {
    context->lowering.function_analysis =
        (loom_low_lower_function_analysis_t){0};
  }
  iree_status_t status = iree_ok_status();
  loom_target_contract_query_environment_t query_environment = *environment;
  if (query_environment.value_domain == NULL) {
    query_environment.value_domain =
        loom_low_lower_context_value_domain(context);
  }
  if (query_environment.arena == NULL) {
    query_environment.arena = &context->function_arena;
  }
  if (query_environment.target_state_allocator.fn == NULL) {
    query_environment.target_state_allocator =
        (loom_target_contract_query_state_allocator_t){
            .fn = loom_low_lower_contract_query_get_or_allocate_target_state,
            .user_data = context,
        };
  }
  if (query_environment.view_regions == NULL) {
    const loom_view_region_table_t* view_regions = NULL;
    status = loom_low_lower_context_view_regions(context, &view_regions);
    query_environment.view_regions = view_regions;
  }
  loom_low_lower_contract_query_state_t state = {
      .context = context,
      .environment = &query_environment,
  };
  const loom_low_lower_contract_query_options_t query_options = {
      .contract_index = context->contract_set->index,
      .rule_sets = context->contract_set->rule_sets,
      .map_value =
          {
              .fn = loom_low_lower_contract_query_map_value,
              .user_data = &state,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_contract_query_can_materialize,
              .user_data = &state,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .descriptor_matrix = context->policy->descriptor_matrix,
  };

  if (iree_status_is_ok(status)) {
    status = loom_low_lower_query_target_contract(
        &query_environment, &query_options, source_op, out_result);
  }
  context->descriptor_set = saved_descriptor_set;
  context->lowering.fact_table = saved_fact_table;
  if (fact_table_changed) {
    context->lowering.function_analysis =
        (loom_low_lower_function_analysis_t){0};
  }
  return status;
}

iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_scope);
  *out_scope = NULL;
  loom_low_lower_assert_options(module, source_function, options);

  loom_low_lower_source_query_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    scope->result.report_allocator = options->report_allocator;
    scope->result.memory_report_row_allocator = module->allocator;
  }
  scope->context = (loom_low_lower_context_t){
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .contract_set = options->policy->contract_set,
      .result = &scope->result,
  };
  scope->context.lowering.fact_table = options->fact_table;
  iree_arena_initialize(module->arena.block_pool,
                        &scope->context.function_arena);
  loom_condition_query_initialize(module, &scope->context.function_arena,
                                  &scope->context.lowering.condition_query);

  iree_status_t status =
      loom_target_low_descriptor_set_select_for_source_lowering(
          options->descriptor_registry,
          loom_target_facts_bundle(options->target_facts),
          &scope->context.descriptor_set);
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (iree_status_is_ok(status) && source_body != NULL) {
    status = loom_low_lowering_frame_initialize_value_ordinals(&scope->context,
                                                               source_body);
    scope->value_domain_initialized = iree_status_is_ok(status);
  }
  if (!iree_status_is_ok(status)) {
    loom_low_lower_source_query_scope_destroy(scope);
    return status;
  }

  *out_scope = scope;
  return iree_ok_status();
}

void loom_low_lower_source_query_scope_destroy(
    loom_low_lower_source_query_scope_t* scope) {
  if (scope == NULL) {
    return;
  }
  if (scope->value_domain_initialized) {
    loom_low_lowering_frame_deinitialize(&scope->context);
    scope->value_domain_initialized = false;
  }
  loom_low_lower_result_deinitialize(&scope->result);
  iree_arena_deinitialize(&scope->context.function_arena);
  memset(scope, 0, sizeof(*scope));
}

loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope) {
  IREE_ASSERT_ARGUMENT(scope);
  return (loom_target_contract_query_callback_t){
      .fn = loom_low_lower_query_target_contract_from_context,
      .user_data = &scope->context,
  };
}

const loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    const loom_low_lower_source_query_scope_t* scope) {
  IREE_ASSERT_ARGUMENT(scope);
  return scope->value_domain_initialized ? &scope->context.lowering.value_domain
                                         : NULL;
}

iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions) {
  IREE_ASSERT_ARGUMENT(scope);
  return loom_low_lower_context_view_regions(&scope->context, out_view_regions);
}

static bool loom_low_lower_cluster_size_product_fits_u32(
    loom_target_workgroup_cluster_size_t cluster_size) {
  const uint64_t cluster_size_xy =
      (uint64_t)cluster_size.x * (uint64_t)cluster_size.y;
  return cluster_size_xy <= UINT32_MAX &&
         cluster_size.z <= UINT32_MAX / cluster_size_xy;
}

static bool loom_low_lower_cluster_size_is_trivial(
    loom_target_workgroup_cluster_size_t cluster_size) {
  return cluster_size.x == 1 && cluster_size.y == 1 && cluster_size.z == 1;
}

static iree_status_t loom_low_lower_verify_static_cluster_divisibility(
    loom_low_lower_context_t* context, const loom_op_t* launch_config,
    loom_target_dispatch_workgroup_count_t workgroup_count,
    loom_target_workgroup_cluster_size_t cluster_size) {
  const uint32_t workgroup_counts[] = {
      workgroup_count.x,
      workgroup_count.y,
      workgroup_count.z,
  };
  const uint32_t cluster_sizes[] = {
      cluster_size.x,
      cluster_size.y,
      cluster_size.z,
  };
  const iree_string_view_t axes[] = {
      IREE_SV("x"),
      IREE_SV("y"),
      IREE_SV("z"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workgroup_counts); ++i) {
    if (workgroup_counts[i] % cluster_sizes[i] == 0) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(axes[i]),
        loom_param_u32(workgroup_counts[i]),
        loom_param_u32(cluster_sizes[i]),
    };
    return loom_low_lower_emit_target_context_error(context, launch_config,
                                                    LOOM_ERR_TARGET_064, params,
                                                    IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_static_launch_config(
    loom_low_lower_context_t* context) {
  const loom_module_t* module = context->module;
  const loom_func_like_t source_function = context->source_function;
  const loom_value_fact_table_t* fact_table = context->options->fact_table;
  loom_low_lower_result_t* result = context->result;
  if (!loom_kernel_def_isa(source_function.op)) {
    return iree_ok_status();
  }
  if (loom_kernel_def_static_workgroup_size_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_size)) {
    result->static_launch_config_flags |=
        LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_SIZE;
  }
  if (loom_kernel_def_static_workgroup_count_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_count)) {
    result->static_launch_config_flags |=
        LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT;
  }

  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(source_function.op);
  if (!loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    return iree_ok_status();
  }
  if (!loom_kernel_def_static_workgroup_cluster_size_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_cluster_size)) {
    return loom_low_lower_emit_target_context_error(
        context, launch_config, LOOM_ERR_TARGET_062, /*extra_params=*/NULL,
        /*extra_param_count=*/0);
  }
  if (!loom_low_lower_cluster_size_product_fits_u32(
          result->static_workgroup_cluster_size)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(result->static_workgroup_cluster_size.x),
        loom_param_u32(result->static_workgroup_cluster_size.y),
        loom_param_u32(result->static_workgroup_cluster_size.z),
    };
    return loom_low_lower_emit_target_context_error(context, launch_config,
                                                    LOOM_ERR_TARGET_063, params,
                                                    IREE_ARRAYSIZE(params));
  }
  if (iree_any_bit_set(result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_verify_static_cluster_divisibility(
        context, launch_config, result->static_workgroup_count,
        result->static_workgroup_cluster_size));
    if (result->error_count != 0) {
      return iree_ok_status();
    }
  }
  if (loom_low_lower_cluster_size_is_trivial(
          result->static_workgroup_cluster_size)) {
    result->static_workgroup_cluster_size =
        (loom_target_workgroup_cluster_size_t){0};
    return iree_ok_status();
  }
  result->static_launch_config_flags |=
      LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_CLUSTER_SIZE;
  return iree_ok_status();
}

iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result) {
  IREE_ASSERT(out_result != NULL);
  loom_low_lower_assert_options(module, source_function, options);
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    out_result->report_allocator = options->report_allocator;
    out_result->memory_report_row_allocator = module->allocator;
  }
  loom_region_t* source_body = loom_func_like_body(source_function);
  IREE_ASSERT(source_body != NULL);

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .contract_set = options->policy->contract_set,
      .result = out_result,
      .module_state = options->module_state,
  };
  context.lowering.fact_table = options->fact_table;
  iree_arena_initialize(module->arena.block_pool, &context.function_arena);
  loom_condition_query_initialize(module, &context.function_arena,
                                  &context.lowering.condition_query);

  iree_status_t status =
      loom_low_lowering_frame_initialize_value_ordinals(&context, source_body);
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_record_static_launch_config(&context);
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.function_arena);
    return iree_ok_status();
  }
  loom_vector_memory_footprint_result_t footprint_result = {0};
  if (iree_status_is_ok(status)) {
    const loom_vector_memory_footprint_options_t footprint_options = {
        .fact_table = context.lowering.fact_table,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_vector_memory_footprint_verify_function(
        module, source_function, &footprint_options, &footprint_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += footprint_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.function_arena);
    return iree_ok_status();
  }

  loom_kernel_async_legality_result_t async_legality_result = {0};
  if (iree_status_is_ok(status)) {
    loom_kernel_async_legality_options_t async_legality_options = {
        .fact_table = context.lowering.fact_table,
        .value_domain = &context.lowering.value_domain,
        .emitter = options->emitter,
        .phase_name = IREE_SV("source-low"),
    };
    status = loom_kernel_async_legality_verify_function(module, source_function,
                                                        &async_legality_options,
                                                        &async_legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += async_legality_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.function_arena);
    return iree_ok_status();
  }

  const loom_view_region_table_t* legality_view_regions = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_low_lower_context_view_regions(&context, &legality_view_regions);
  }
  loom_target_low_legality_result_t legality_result = {};
  if (iree_status_is_ok(status)) {
    loom_target_low_legality_options_t legality_options = {
        .target_facts = options->target_facts,
        .descriptor_registry = options->descriptor_registry,
        .error_catalog = options->policy->error_catalog,
        .provider_list = options->legality_provider_list,
        .contract_query =
            {
                .fn = loom_low_lower_query_target_contract_from_context,
                .user_data = &context,
            },
        .type_supported = context.policy->source_type_supported,
        .view_regions = legality_view_regions,
        .structural_legality_flags =
            loom_low_lower_structured_low_enabled(&context)
                ? LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF
                : 0,
        .diagnostic_flags = options->legality_diagnostic_flags,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_target_low_verify_function_legality(
        module, source_function, &legality_options, &legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count = legality_result.error_count;
    out_result->remark_count = legality_result.remark_count;
    out_result->descriptor_set = legality_result.descriptor_set;
    context.descriptor_set = legality_result.descriptor_set;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.function_arena);
    return iree_ok_status();
  }

  iree_arena_initialize(module->arena.block_pool, &context.emission_arena);

  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.function_arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_map),
        (void**)&context.lowering.value_map);
  }
  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.function_arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_storage_flags),
        (void**)&context.lowering.value_storage_flags);
  }
  if (iree_status_is_ok(status)) {
    for (loom_value_ordinal_t i = 0;
         i < context.lowering.value_domain.value_count; ++i) {
      context.lowering.value_map[i] = LOOM_VALUE_ID_INVALID;
      context.lowering.value_storage_flags[i] = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(module->arena.block_pool, &context.planning_arena);
    status = loom_low_lower_check_function_signature(&context, source_body);
    if (iree_status_is_ok(status) &&
        !loom_low_lower_context_should_stop(&context)) {
      status = loom_low_lower_select_plans(&context, source_body);
    }
    if (iree_status_is_ok(status)) {
      loom_low_lower_analyze_storage_demands(&context, source_body);
    }
    iree_arena_deinitialize(&context.planning_arena);
  }
  if (iree_status_is_ok(status) && context.result->error_count == 0) {
    loom_symbol_ref_t low_func_ref = loom_func_like_callee(source_function);
    loom_low_lower_context_emission_scope_begin(&context);
    status =
        loom_low_lower_create_function_op(&context, source_body, low_func_ref);
    loom_low_lower_context_emission_scope_end(&context);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_map_blocks(&context, source_body);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_remap_function_predicates(&context);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_prepare_branches(&context, source_body);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_low_lower_context_emission_scope_begin(&context);
      status = loom_low_lower_emit_preamble(&context);
      loom_low_lower_context_emission_scope_end(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_low_lower_context_emission_scope_begin(&context);
      status = loom_low_lower_emit_argument_resource_imports(&context);
      loom_low_lower_context_emission_scope_end(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_low_lower_context_emission_scope_begin(&context);
      status = loom_low_lower_emit_direct_argument_transfers(&context);
      loom_low_lower_context_emission_scope_end(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_low_lower_context_emission_scope_begin(&context);
      status = loom_low_lower_emit_entry_setup(&context);
      loom_low_lower_context_emission_scope_end(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_emit_body(&context, source_body);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_low_lower_context_emission_scope_begin(&context);
      status = loom_low_lower_finalize_function(&context);
      loom_low_lower_context_emission_scope_end(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count != 0 &&
        context.low_func_op != NULL) {
      status = loom_op_erase(module, context.low_func_op);
      context.low_func_op = NULL;
      out_result->low_func_op = NULL;
      out_result->low_func_ref = loom_symbol_ref_null();
    }
    // The replacement low op carries the source symbol while the source op
    // still owns the symbol table entry. Erase clears that entry; relink it to
    // the replacement so callers keep the same symbol identity.
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_copy_function_source_presentation(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_op_erase(module, source_function.op);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_module_link_symbol_defining_op(
          module, context.low_func_op,
          loom_op_vtable(module, context.low_func_op));
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0 &&
        context.lowering.memory_access_record_count != 0) {
      out_result->memory_access_table = (loom_low_memory_access_table_t){
          .function_op = context.low_func_op,
          .values = context.lowering.memory_access_records,
          .count = context.lowering.memory_access_record_count,
      };
    }
  }

  loom_low_lowering_frame_deinitialize(&context);
  iree_arena_deinitialize(&context.emission_arena);
  iree_arena_deinitialize(&context.function_arena);
  return status;
}
