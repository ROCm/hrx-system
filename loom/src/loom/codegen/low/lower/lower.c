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

static iree_status_t loom_low_lower_intern_type_id(
    loom_low_lower_context_t* context, loom_type_t type,
    loom_type_id_t* out_type_id) {
  return loom_module_intern_type_id(context->module, type, out_type_id);
}

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

static iree_status_t loom_low_lower_map_blocks(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.block_map),
      (void**)&context->lowering.block_map));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.successor_interpositions),
      (void**)&context->lowering.successor_interpositions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.branch_plans),
      (void**)&context->lowering.branch_plans));
  memset(context->lowering.block_map, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.block_map));
  memset(context->lowering.successor_interpositions, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.successor_interpositions));
  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    context->lowering.branch_plans[i] = loom_low_lower_plan_empty();
  }

  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    loom_block_t* source_block = loom_region_block(source_body, i);
    loom_block_t* low_block = NULL;
    if (i == 0) {
      low_block = loom_region_entry_block(low_body);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(context->module, low_body, &low_block));
    }
    low_block->label_id = source_block->label_id;
    context->lowering.block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->lowering.block_map[block_index];
    if (block_index == 0) {
      const uint16_t direct_argument_count =
          loom_low_lower_direct_argument_count(context);
      IREE_ASSERT_EQ(low_block->arg_count, direct_argument_count);
      uint16_t direct_argument_index = 0;
      for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
           ++arg_index) {
        if (context->lowering.argument_map[arg_index].kind !=
            LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_block->arg_ids[arg_index],
            low_block->arg_ids[direct_argument_index]));
        ++direct_argument_index;
      }
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, context->source_function.op,
          source_block->arg_ids[arg_index], &low_type));
      IREE_ASSERT_NE(loom_type_kind(low_type), LOOM_TYPE_NONE);
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_argument_resource_import(
    loom_low_lower_context_t* context, const loom_value_id_t* source_arguments,
    uint16_t argument_index) {
  const loom_low_lower_abi_argument_t* argument =
      &context->lowering.argument_map[argument_index];
  loom_type_t source_type = argument->resource_source_type;
  if (loom_type_kind(source_type) == LOOM_TYPE_NONE) {
    source_type = loom_module_value_type(context->module,
                                         source_arguments[argument_index]);
  }
  loom_type_id_t source_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_intern_type_id(context, source_type, &source_type_id));
  loom_op_t* resource_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resource_build(
      &context->builder, argument->resource_build_flags,
      (uint8_t)argument->resource_import_kind, LOOM_VALUE_ID_INVALID,
      argument->resource_index, source_type_id, argument->resource_extent,
      argument->resource_cache_swizzle_stride, argument->abi_type,
      context->source_function.op->location, &resource_op));
  return loom_low_lower_bind_value(context, source_arguments[argument_index],
                                   loom_low_resource_result(resource_op));
}

static iree_status_t loom_low_lower_emit_argument_resource_imports(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0 ||
      argument_count == loom_low_lower_direct_argument_count(context)) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      continue;
    }
    status = loom_low_lower_emit_argument_resource_import(context,
                                                          source_arguments, i);
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

// Separates the target ABI live-in from the semantic function argument. The
// transfer normally coalesces to the same storage and emits nothing. When an
// internal call needs the ABI location while the argument remains live,
// allocation can place the semantic identity elsewhere and materialize the
// exact-state move at entry.
static iree_status_t loom_low_lower_emit_direct_argument_transfers(
    loom_low_lower_context_t* context) {
  if (!iree_any_bit_set(context->policy->flags,
                        LOOM_LOW_LOWER_POLICY_FLAG_EXPLICIT_ABI_TRANSFERS)) {
    return iree_ok_status();
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0) return iree_ok_status();

  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      continue;
    }
    loom_value_id_t abi_value = LOOM_VALUE_ID_INVALID;
    status =
        loom_low_lower_lookup_value(context, source_arguments[i], &abi_value);
    if (!iree_status_is_ok(status)) break;
    loom_op_t* move_op = NULL;
    status =
        loom_low_move_build(&context->builder, abi_value, /*detached=*/false,
                            loom_module_value_type(context->module, abi_value),
                            context->source_function.op->location, &move_op);
    if (!iree_status_is_ok(status)) break;

    const loom_value_ordinal_t source_ordinal =
        loom_low_lowering_frame_value_ordinal(&context->lowering,
                                              source_arguments[i]);
    const loom_value_id_t semantic_value = loom_low_move_result(move_op);
    context->lowering.value_map[source_ordinal] = semantic_value;
    status = loom_low_lower_copy_value_name(context, source_arguments[i],
                                            semantic_value);
  }
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_preamble(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_preamble.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_preamble.fn(
      context->policy->emit_preamble.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_entry_setup(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_entry_setup.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_entry_setup.fn(
      context->policy->emit_entry_setup.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_prepare_branches(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  if (context->policy->prepare_branch.fn == NULL) {
    return iree_ok_status();
  }

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_body->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_block(source_body, block_index);
    const loom_op_t* source_terminator = loom_block_const_last_op(source_block);
    if (source_terminator == NULL || source_terminator->successor_count == 0) {
      continue;
    }
    if (loom_low_lower_cfg_cond_br_exact_bool(context, source_terminator,
                                              NULL)) {
      continue;
    }
    status = context->policy->prepare_branch.fn(
        context->policy->prepare_branch.user_data, context, source_terminator,
        &analysis_arena);
    iree_arena_reset(&analysis_arena);
    if (loom_low_lower_context_should_stop(context)) {
      break;
    }
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static iree_status_t loom_low_lower_remap_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* source_values, iree_host_size_t value_count,
    loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_values[i], &low_values[i]));
    loom_type_t required_low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
        context, source_op, source_values[i], &required_low_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_structural_operand(
        context, source_op, i, source_values[i], required_low_type,
        &low_values[i]));
  }
  *out_low_values = low_values;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_or_elide_alias(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  if (loom_low_lower_source_value_has_low_mapping(context, source_value_id)) {
    return loom_low_lower_bind_value_alias(context, source_value_id,
                                           result_value_id);
  }
  if (!loom_low_lower_result_storage_required(context, result_value_id)) {
    return loom_low_lower_elide_value(context, result_value_id);
  }
  return loom_low_lower_bind_value_alias(context, source_value_id,
                                         result_value_id);
}

static iree_status_t loom_low_lower_bind_identity_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->operand_count, source_op->result_count);
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_or_elide_alias(
        context, source_operands[i], source_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_func_ref_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t source_id = loom_func_ref_cast_source(source_op);
  const loom_value_id_t result_id = loom_func_ref_cast_result(source_op);
  loom_value_id_t low_source_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_id, &low_source_id));
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, result_id,
                                                &low_result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_ref_cast_build(
      &context->builder, low_source_id, low_result_type, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(context, result_id,
                                   loom_low_func_ref_cast_result(low_op));
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks);

static iree_status_t loom_low_lower_map_op_result_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (source_op->result_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_op->result_count, sizeof(*result_types),
      (void**)&result_types));
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
        context, source_op, source_results[i], &result_types[i]));
    if (loom_type_kind(result_types[i]) == LOOM_TYPE_NONE) {
      return iree_ok_status();
    }
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_op_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_op) {
  IREE_ASSERT_EQ(source_op->result_count, low_op->result_count);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* low_results = loom_op_const_results(low_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_func_null(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_null_build(
      &context->builder, result_types[0], source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_compare_null(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t* operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, loom_op_const_operands(source_op),
      source_op->operand_count, &operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_compare_null_build(
      &context->builder, operands[0], result_types[0], source_op->location,
      &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_address_build(
      &context->builder, loom_func_address_callee(source_op), result_types[0],
      source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_import_resolved(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_import_resolved_build(
      &context->builder, loom_func_import_resolved_callee(source_op),
      result_types[0], source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_emit_scf_yield(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_slice_t values = loom_scf_yield_values(source_op);
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, values.values, values.count, &low_values));
  loom_op_t* low_yield_op = NULL;
  return loom_low_scf_yield_build(&context->builder, low_values, values.count,
                                  source_op->location, &low_yield_op);
}

static iree_status_t loom_low_lower_emit_scf_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_condition_condition(source_op), &low_condition));
  const loom_value_slice_t forwarded = loom_scf_condition_forwarded(source_op);
  loom_value_id_t* low_forwarded = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, forwarded.values, forwarded.count, &low_forwarded));
  loom_op_t* low_condition_op = NULL;
  return loom_low_scf_condition_build(&context->builder, low_condition,
                                      low_forwarded, forwarded.count,
                                      source_op->location, &low_condition_op);
}

static iree_status_t loom_low_lower_emit_scf_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_if_condition(source_op), &low_condition));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (source_op->result_count != 0 && result_types == NULL) {
    return iree_ok_status();
  }

  loom_low_scf_if_build_flags_t build_flags = 0;
  if (loom_scf_if_else_region(source_op) != NULL) {
    build_flags |= LOOM_LOW_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION;
  }
  loom_op_t* low_if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_if_build(
      &context->builder, build_flags, low_condition, result_types,
      source_op->result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_if_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_if_op, loom_low_scf_if_then_region(low_if_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_if_then_region(source_op),
      /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_region_t* source_else_region = loom_scf_if_else_region(source_op);
  if (source_else_region == NULL) {
    return iree_ok_status();
  }
  saved_ip = loom_builder_enter_region(&context->builder, low_if_op,
                                       loom_low_scf_if_else_region(low_if_op));
  status = loom_low_lower_emit_region_ops(context, source_else_region,
                                          /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_bind_region_entry_args(
    loom_low_lower_context_t* context, const loom_region_t* source_region,
    const loom_region_t* low_region) {
  const loom_block_t* source_entry =
      loom_region_const_entry_block(source_region);
  const loom_block_t* low_entry = loom_region_const_entry_block(low_region);
  IREE_ASSERT_EQ(source_entry->arg_count, low_entry->arg_count);
  for (uint16_t i = 0; i < source_entry->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_entry->arg_ids[i], low_entry->arg_ids[i]));
  }
  return iree_ok_status();
}

static bool loom_low_lower_op_attr_present(const loom_op_t* op,
                                           uint8_t attr_index) {
  return !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static iree_status_t loom_low_lower_emit_scf_for(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_lower_bound(source_op), &low_lower_bound));
  loom_value_id_t low_upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_upper_bound(source_op), &low_upper_bound));
  loom_value_id_t low_step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_step(source_op), &low_step));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count, &low_iter_args));

  loom_value_id_t low_unroll_factor = LOOM_VALUE_ID_INVALID;
  loom_low_scf_for_build_flags_t build_flags = 0;
  if (loom_scf_for_unroll_factor_is_present(source_op)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_scf_for_unroll_factor(source_op), &low_unroll_factor));
  }

  uint8_t unroll_policy = 0;
  if (loom_low_lower_op_attr_present(source_op,
                                     loom_scf_for_unroll_policy_ATTR_INDEX)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = (uint8_t)loom_scf_for_unroll_policy(source_op);
  }

  loom_op_t* low_for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_for_build(
      &context->builder, build_flags, low_lower_bound, low_upper_bound,
      low_step, low_iter_args, iter_args.count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, low_unroll_factor, unroll_policy,
      source_op->location, &low_for_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_for_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_for_body(source_op),
      loom_low_scf_for_body(low_for_op)));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_for_op, loom_low_scf_for_body(low_for_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_for_body(source_op), /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_scf_while(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_slice_t iter_args = loom_scf_while_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count, &low_iter_args));

  loom_op_t* low_while_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_while_build(
      &context->builder, low_iter_args, iter_args.count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_while_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_while_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_while_before(source_op),
      loom_low_scf_while_before(low_while_op)));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_while_after(source_op),
      loom_low_scf_while_after(low_while_op)));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_while_op, loom_low_scf_while_before(low_while_op));
  iree_status_t status =
      loom_low_lower_emit_region_ops(context, loom_scf_while_before(source_op),
                                     /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&context->builder, low_while_op,
                                       loom_low_scf_while_after(low_while_op));
  status =
      loom_low_lower_emit_region_ops(context, loom_scf_while_after(source_op),
                                     /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

// Moves each call result out of its ABI transfer identity before binding it to
// the source result. This leaves the call result available for target-fixed
// placement while preserving exact reference ownership in the semantic value.
static iree_status_t loom_low_lower_bind_call_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_call_op) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* low_results = loom_op_const_results(low_call_op);
  if (!iree_any_bit_set(context->policy->flags,
                        LOOM_LOW_LOWER_POLICY_FLAG_EXPLICIT_ABI_TRANSFERS)) {
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_results[i],
                                                     low_results[i]));
    }
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    loom_op_t* move_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_move_build(
        &context->builder, low_results[i], /*detached=*/false,
        loom_module_value_type(context->module, low_results[i]),
        source_op->location, &move_op));
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_results[i], loom_low_move_result(move_op)));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = true;
  if (loom_func_ref_cast_isa(source_op)) {
    return loom_low_lower_func_ref_cast(context, source_op);
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    return loom_low_lower_bind_identity_results(context, source_op);
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    IREE_ASSERT(source_op->result_count == 1);
    return loom_low_lower_bind_or_elide_alias(
        context, loom_op_const_operands(source_op)[0],
        loom_op_const_results(source_op)[0]);
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT: {
      return loom_low_lower_bind_or_elide_alias(
          context, loom_buffer_assume_same_root_buffer(source_op),
          loom_buffer_assume_same_root_result(source_op));
    }
    case LOOM_OP_FUNC_NULL:
      return loom_low_lower_func_null(context, source_op);
    case LOOM_OP_FUNC_COMPARE_NULL:
      return loom_low_lower_func_compare_null(context, source_op);
    case LOOM_OP_FUNC_ADDRESS:
      return loom_low_lower_func_address(context, source_op);
    case LOOM_OP_FUNC_IMPORT_RESOLVED:
      return loom_low_lower_func_import_resolved(context, source_op);
    case LOOM_OP_FUNC_RETURN: {
      loom_value_slice_t values = loom_func_return_operands(source_op);
      loom_value_id_t* low_values = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, values.values, values.count, &low_values));
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, low_values, values.count,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_FUNC_CALL: {
      loom_value_slice_t operands = loom_func_call_operands(source_op);
      loom_value_id_t* low_operands = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, operands.values, operands.count, &low_operands));

      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      loom_type_t* result_types = NULL;
      bool has_unmapped_result = false;
      if (source_op->result_count != 0) {
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
            context, source_op->result_count, sizeof(*result_types),
            (void**)&result_types));
        for (uint16_t i = 0; i < source_op->result_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
              context, source_op, source_results[i], &result_types[i]));
          has_unmapped_result |=
              loom_type_kind(result_types[i]) == LOOM_TYPE_NONE;
        }
      }
      if (has_unmapped_result) {
        return iree_ok_status();
      }

      loom_low_func_call_build_flags_t build_flags = 0;
      uint8_t purity = loom_func_call_purity(source_op);
      if (purity != 0) {
        build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
      }
      loom_op_t* low_call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_build(
          &context->builder, build_flags, purity,
          loom_func_call_callee(source_op), low_operands, operands.count,
          result_types, source_op->result_count,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_call_op));
      return loom_low_lower_bind_call_results(context, source_op, low_call_op);
    }
    case LOOM_OP_FUNC_CALL_INDIRECT: {
      loom_value_id_t* low_operands = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, loom_op_const_operands(source_op),
          source_op->operand_count, &low_operands));

      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      loom_type_t* result_types = NULL;
      bool has_unmapped_result = false;
      if (source_op->result_count != 0) {
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
            context, source_op->result_count, sizeof(*result_types),
            (void**)&result_types));
        for (uint16_t i = 0; i < source_op->result_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
              context, source_op, source_results[i], &result_types[i]));
          has_unmapped_result |=
              loom_type_kind(result_types[i]) == LOOM_TYPE_NONE;
        }
      }
      if (has_unmapped_result) {
        return iree_ok_status();
      }

      loom_op_t* low_call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_indirect_build(
          &context->builder, low_operands[0], low_operands + 1,
          source_op->operand_count - 1, result_types, source_op->result_count,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_call_op));
      return loom_low_lower_bind_call_results(context, source_op, low_call_op);
    }
    case LOOM_OP_KERNEL_RETURN: {
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, NULL, 0,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_SCF_SCHEDULE_FENCE: {
      loom_op_t* low_fence_op = NULL;
      return loom_low_schedule_fence_build(&context->builder,
                                           source_op->location, &low_fence_op);
    }
    case LOOM_OP_CFG_BR: {
      loom_block_t* low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_dest));
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_id_t* low_args = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_successor_args(
          context, source_op, 0, low_dest, args.values, args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder, low_dest, low_args,
                               args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      loom_block_t* low_true_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_true_dest));
      loom_block_t* low_false_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 1, &low_false_dest));
      bool condition = false;
      if (loom_low_lower_cfg_cond_br_exact_bool(context, source_op,
                                                &condition)) {
        loom_block_t* low_dest = condition ? low_true_dest : low_false_dest;
        loom_op_t* low_br_op = NULL;
        return loom_low_br_build(&context->builder, low_dest, NULL, 0,
                                 source_op->location, &low_br_op);
      }
      loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op), &low_condition));
      if (context->policy->emit_cond_branch.fn != NULL) {
        return context->policy->emit_cond_branch.fn(
            context->policy->emit_cond_branch.user_data, context, source_op,
            low_condition, low_true_dest, low_false_dest);
      }
      loom_op_t* low_cond_br_op = NULL;
      return loom_low_cond_br_build(&context->builder, low_condition,
                                    low_true_dest, low_false_dest,
                                    source_op->location, &low_cond_br_op);
    }
    case LOOM_OP_CFG_SWITCH: {
      if (context->policy->switch_lowering.emit == NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "cfg.switch reached a target lowering policy without direct "
            "switch emission");
      }
      loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_switch_selector(source_op), &low_selector));
      loom_block_t* low_default_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_default_dest));
      const loom_successor_slice_t case_dests =
          loom_cfg_switch_case_dests(source_op);
      loom_block_t** low_case_dests = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
          context, case_dests.count, sizeof(*low_case_dests),
          (void**)&low_case_dests));
      for (uint16_t i = 0; i < case_dests.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
            context, source_op, (uint16_t)(i + 1), &low_case_dests[i]));
      }
      return context->policy->switch_lowering.emit(
          context->policy->switch_lowering.user_data, context, source_op,
          low_selector, low_default_dest, low_case_dests, case_dests.count);
    }
    case LOOM_OP_SCF_YIELD:
      return loom_low_lower_emit_scf_yield(context, source_op);
    case LOOM_OP_SCF_CONDITION:
      return loom_low_lower_emit_scf_condition(context, source_op);
    case LOOM_OP_SCF_IF:
      return loom_low_lower_emit_scf_if(context, source_op);
    case LOOM_OP_SCF_FOR:
      return loom_low_lower_emit_scf_for(context, source_op);
    case LOOM_OP_SCF_WHILE:
      return loom_low_lower_emit_scf_while(context, source_op);
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_emit_elided_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_elide_value(context, source_results[i]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
      context, selected_plan, /*emitted_low_op_count=*/0));
  return iree_ok_status();
}

static loom_value_id_t loom_low_lower_descriptor_matrix_sparse_source_value(
    const loom_contract_request_t* request) {
  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  const loom_contract_operand_t* operands[] = {
      &request->lhs,
      &request->rhs,
      &request->accumulator,
      &request->result,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    const loom_contract_value_ref_t ref =
        operands[i]->encoded.auxiliary_value_refs
            [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA];
    if (!loom_contract_value_ref_is_present(ref)) {
      continue;
    }
    const loom_value_id_t operand_source_value =
        loom_contract_value_ref_value_id(ref);
    if (source_value == LOOM_VALUE_ID_INVALID) {
      source_value = operand_source_value;
      continue;
    }
    IREE_ASSERT_EQ(source_value, operand_source_value,
                   "descriptor-matrix selected sparse source is ambiguous");
  }
  IREE_ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID,
                 "descriptor-matrix selected sparse source is unavailable");
  return source_value;
}

static loom_value_id_t loom_low_lower_descriptor_matrix_auxiliary_source_value(
    const loom_contract_operand_t* operand,
    loom_contract_auxiliary_operand_key_t key) {
  const loom_contract_value_ref_t ref =
      operand->encoded.auxiliary_value_refs[key];
  IREE_ASSERT(loom_contract_value_ref_is_present(ref),
              "descriptor-matrix selected auxiliary operand is unavailable");
  return loom_contract_value_ref_value_id(ref);
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_low_operand_source_binding_t source_binding, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t* out_low_value) {
  switch (source_binding) {
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS:
      *out_low_value = low_lhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS:
      *out_low_value = low_rhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_ACCUMULATOR:
      *out_low_value = low_init;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_SPARSE_METADATA: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_sparse_source_value(
              &plan->contract_request);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.lhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.rhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_NONE:
    default:
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected packet operand has no source binding");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t** out_operands, iree_host_size_t* out_operand_count) {
  *out_operands = NULL;
  *out_operand_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;

  iree_host_size_t operand_count = 0;
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const uint32_t row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[row];
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++operand_count;
    }
  }

  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, operand_count, sizeof(*operands), (void**)&operands));
  }

  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_value(
        context, plan, operand->source_binding, low_lhs, low_rhs, low_init,
        &operands[operand->source_value_index]));
  }

  *out_operands = operands;
  *out_operand_count = operand_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  iree_host_size_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t row = descriptor->constraint_start + i;
    if (descriptor_set->constraints[row].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    if (constraint->lhs_operand_index >= descriptor->result_count ||
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE ||
        constraint->rhs_operand_index >= descriptor->operand_count) {
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected tied result constraint is invalid");
      IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_operand_t* tied_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->rhs_operand_index];
    IREE_ASSERT_NE(tied_operand->source_value_index, LOOM_LOW_ID_NONE);
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index = tied_operand->source_value_index,
        .has_type_change = false,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static bool loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t constraint_index,
    uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < constraint_index; ++i) {
    const loom_low_constraint_t* previous =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (previous->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(previous->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(previous->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* previous_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  previous->rhs_operand_index];
    if (previous_operand->source_value_index == packet_operand_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_descriptor_matrix_copy_destructive_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t* operands) {
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(constraint->lhs_operand_index < descriptor->result_count);
    IREE_ASSERT(constraint->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(constraint->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* destructive_operand =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->rhs_operand_index];
    const uint16_t packet_operand_index =
        destructive_operand->source_value_index;
    IREE_ASSERT_NE(packet_operand_index, LOOM_LOW_ID_NONE);
    if (loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
            descriptor_set, descriptor, i, packet_operand_index)) {
      continue;
    }
    const loom_value_id_t source_value = operands[packet_operand_index];
    const loom_type_t copy_type =
        loom_module_value_type(context->module, source_value);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), source_value, false, copy_type,
        source_op->location, &copy_op));
    operands[packet_operand_index] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_vector_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  if (descriptor->result_count != 1) {
    IREE_ASSERT_UNREACHABLE(
        "descriptor-matrix vector.mma descriptor result count is invalid");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_lhs(source_op), &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_rhs(source_op), &low_rhs));
  loom_value_id_t low_init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_init(source_op), &low_init));

  const loom_value_id_t result = loom_vector_mma_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, result, &result_low_type));
  IREE_ASSERT(loom_low_type_is_register(result_low_type));

  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_operands(
      context, plan, low_lhs, low_rhs, low_init, &operands, &operand_count));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_descriptor_matrix_copy_destructive_operands(
          context, source_op, plan, operands));
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_tied_results(
      context, plan, &tied_results, &tied_result_count));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, plan->attrs,
      &result_low_type, 1, tied_results, tied_result_count, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, result, loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  switch (plan->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return loom_low_lower_emit_descriptor_matrix_vector_mma(context,
                                                              source_op, plan);
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown descriptor-matrix source");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static uint64_t loom_low_lower_count_low_body_ops(
    const loom_low_lower_context_t* context) {
  uint64_t op_count = 0;
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_ASSERT(low_body != NULL);
  if (low_body == NULL) {
    return 0;
  }
  for (uint16_t block_index = 0; block_index < low_body->block_count;
       ++block_index) {
    op_count += loom_region_block(low_body, block_index)->op_count;
  }
  return op_count;
}

static iree_status_t loom_low_lower_emit_selected_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_LT(context->lowering.selected_plan_emit_index,
                 context->lowering.selected_plan_count);
  const loom_low_lower_selected_plan_t selected_plan =
      context->lowering
          .selected_plans[context->lowering.selected_plan_emit_index++];
  IREE_ASSERT_EQ(selected_plan.source_op, source_op);
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
    return loom_low_lower_emit_elided_selected_plan(context, &selected_plan);
  }
  const bool report_allocator_provided =
      !iree_allocator_is_null(context->options->report_allocator);
  uint64_t before_op_count = 0;
  if (report_allocator_provided) {
    before_op_count = loom_low_lower_count_low_body_ops(context);
  }
  if (selected_plan.kind == LOOM_LOW_LOWER_SELECTED_PLAN_RULE) {
    IREE_ASSERT(selected_plan.rule_set != NULL);
    IREE_ASSERT(selected_plan.rule != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_emit_rule(
        context, selected_plan.rule_set, source_op, selected_plan.rule,
        selected_plan.resolved_emits, selected_plan.source_memory_access));
  } else if (selected_plan.kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_matrix_plan(
        context, source_op,
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan.plan.target_data));
  } else {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_ASSERT(context->policy->emit_op.fn != NULL);
    IREE_RETURN_IF_ERROR(
        context->policy->emit_op.fn(context->policy->emit_op.user_data, context,
                                    source_op, selected_plan.plan));
  }
  if (report_allocator_provided) {
    const uint64_t after_op_count = loom_low_lower_count_low_body_ops(context);
    IREE_ASSERT_GE(after_op_count, before_op_count);
    const uint64_t emitted_op_count = after_op_count - before_op_count;
    IREE_ASSERT_LE(emitted_op_count, UINT32_MAX);
    IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
        context, &selected_plan, (uint32_t)emitted_op_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_source_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (context->lowering.selected_plan_emit_index <
          context->lowering.selected_plan_count &&
      context->lowering
              .selected_plans[context->lowering.selected_plan_emit_index]
              .source_op == source_op) {
    return loom_low_lower_emit_selected_plan(context, source_op);
  }
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_structural_op(context, source_op, &handled));
  if (handled || loom_low_lower_op_is_source_metadata(source_op->kind)) {
    return iree_ok_status();
  }
  IREE_ASSERT_UNREACHABLE(
      "source op has neither a structural lowering nor a selected plan");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks) {
  // A rejected plan leaves its low results unbound, so emission cannot resume
  // elsewhere in the region after any diagnostic error.
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_region->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_region, block_index);
    if (map_source_blocks) {
      loom_builder_set_block(&context->builder,
                             context->lowering.block_map[block_index]);
    } else if (block_index != 0) {
      IREE_ASSERT_UNREACHABLE(
          "structured source region with multiple blocks reached target-low "
          "emission");
      IREE_BUILTIN_UNREACHABLE();
    }
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      const uint32_t before_error_count = context->result->error_count;
      loom_low_lower_context_emission_scope_begin(context);
      status = loom_low_lower_emit_source_op(context, source_op);
      // Builders copy all caller-provided arrays and attribute payloads into
      // module-owned storage. Nested structured emission may reset this arena
      // while its parent is active because the parent builder has already
      // consumed every temporary before entering a child region.
      loom_low_lower_context_emission_scope_end(context);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (context->result->error_count != before_error_count) {
        return iree_ok_status();
      }
    }
  }
  return status;
}

static iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, source_body, /*map_source_blocks=*/true);
  loom_builder_restore(&context->builder, saved_ip);
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    IREE_ASSERT_EQ(context->lowering.selected_plan_emit_index,
                   context->lowering.selected_plan_count);
  }
  return status;
}

static iree_status_t loom_low_lower_finalize_function(
    loom_low_lower_context_t* context) {
  if (context->policy->finalize_function.fn == NULL) {
    return iree_ok_status();
  }
  return context->policy->finalize_function.fn(
      context->policy->finalize_function.user_data, context);
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
