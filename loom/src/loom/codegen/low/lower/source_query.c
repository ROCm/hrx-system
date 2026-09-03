// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_query.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/target/low_descriptor_registry.h"

typedef struct loom_low_lower_contract_query_state_t {
  // Mutable lowering context backing the read-only contract query.
  loom_low_lower_context_t* context;
  // Query environment provided by the target-low legality caller.
  const loom_target_contract_query_environment_t* environment;
} loom_low_lower_contract_query_state_t;

static iree_status_t loom_low_lower_source_query_allocate_target_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data) {
  return loom_low_lower_get_or_allocate_target_state(
      (loom_low_lower_context_t*)user_data, key, data_length, out_data);
}

iree_status_t loom_low_lower_source_query_environment_initialize(
    loom_low_lower_context_t* context,
    loom_target_contract_query_environment_t* out_environment) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_environment = (loom_target_contract_query_environment_t){
      .module = context->module,
      .function = context->source_function,
      .target_facts = context->options->target_facts,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->lowering.fact_table,
      .value_domain = &context->lowering.value_domain,
      .source_program = &context->lowering.source_program,
      .source_dataflow = loom_low_lower_context_source_dataflow(context),
      .source_representation_plan =
          loom_low_lower_context_source_representation_plan(context),
      .view_regions = view_regions,
      .arena = &context->function_arena,
      .target_state_allocator =
          {
              .fn = loom_low_lower_source_query_allocate_target_state,
              .user_data = context,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_map_value(
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

static iree_status_t loom_low_lower_source_query_can_materialize(
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

static iree_status_t loom_low_lower_source_query_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  const loom_source_program_t* source_program =
      &context->lowering.source_program;
  const loom_source_dataflow_result_t* source_dataflow =
      loom_low_lower_context_source_dataflow(context);
  if ((environment->module != NULL && environment->module != context->module) ||
      (loom_func_like_isa(environment->function) &&
       environment->function.op != context->source_function.op) ||
      (environment->target_facts != NULL &&
       environment->target_facts != context->options->target_facts) ||
      (environment->descriptor_set != NULL &&
       environment->descriptor_set != context->descriptor_set) ||
      (environment->fact_table != NULL &&
       environment->fact_table != context->lowering.fact_table) ||
      (environment->value_domain != NULL &&
       environment->value_domain != value_domain) ||
      (environment->source_program != NULL &&
       environment->source_program != source_program) ||
      (environment->source_dataflow != NULL &&
       environment->source_dataflow != source_dataflow) ||
      (environment->source_representation_plan != NULL &&
       environment->source_representation_plan !=
           loom_low_lower_context_source_representation_plan(context))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source contract query inputs differ from the retained query scope");
  }

  iree_status_t status = iree_ok_status();
  loom_target_contract_query_environment_t query_environment = *environment;
  query_environment.module = context->module;
  query_environment.function = context->source_function;
  query_environment.target_facts = context->options->target_facts;
  query_environment.descriptor_set = context->descriptor_set;
  query_environment.fact_table = context->lowering.fact_table;
  if (query_environment.value_domain == NULL) {
    query_environment.value_domain = value_domain;
  }
  if (query_environment.source_program == NULL) {
    query_environment.source_program = source_program;
  }
  if (query_environment.source_dataflow == NULL) {
    query_environment.source_dataflow = source_dataflow;
  }
  if (query_environment.source_representation_plan == NULL) {
    query_environment.source_representation_plan =
        loom_low_lower_context_source_representation_plan(context);
  }
  if (query_environment.arena == NULL) {
    query_environment.arena = &context->function_arena;
  }
  if (query_environment.target_state_allocator.fn == NULL) {
    query_environment.target_state_allocator =
        (loom_target_contract_query_state_allocator_t){
            .fn = loom_low_lower_source_query_allocate_target_state,
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
      .contract_index = &context->contract_index,
      .rule_sets = context->policy->rule_sets,
      .map_value =
          {
              .fn = loom_low_lower_source_query_map_value,
              .user_data = &state,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_source_query_can_materialize,
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
  return status;
}

loom_target_contract_query_callback_t loom_low_lower_source_query_callback(
    loom_low_lower_context_t* context) {
  return (loom_target_contract_query_callback_t){
      .fn = loom_low_lower_source_query_contract,
      .user_data = context,
  };
}

struct loom_low_lower_source_query_scope_t {
  // Read-only lowering context backing source-to-Low contract queries.
  loom_low_lower_context_t context;
  // Diagnostic and result scratch required by the lowering context.
  loom_low_lower_result_t result;
  // True after context.lowering.value_domain acquires module storage.
  bool value_domain_initialized;
};

iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope) {
  *out_scope = NULL;
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
      .descriptor_set = descriptor_set,
      .result = &scope->result,
  };
  scope->context.lowering.fact_table = options->fact_table;
  iree_arena_initialize(module->arena.block_pool,
                        &scope->context.function_arena);
  loom_condition_query_initialize(module, &scope->context.function_arena,
                                  &scope->context.lowering.condition_query);

  iree_status_t status =
      descriptor_set != NULL
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "source query scope requires a descriptor set");
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (iree_status_is_ok(status) && source_body != NULL) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module, source_body, &scope->context.function_arena,
        &scope->context.lowering.value_domain);
    scope->value_domain_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) && source_body != NULL) {
    status = loom_source_program_build(module, source_function.op, source_body,
                                       &scope->context.lowering.value_domain,
                                       &scope->context.function_arena,
                                       &scope->context.lowering.source_program);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_contract_index_compose(
        scope->context.policy->contract_bindings,
        scope->context.policy->contract_binding_count,
        &scope->context.contract_index, &scope->context.function_arena);
  }
  const loom_view_region_table_t* view_regions = NULL;
  if (iree_status_is_ok(status) &&
      scope->context.policy->source_dataflow_provider != NULL) {
    status =
        loom_low_lower_context_view_regions(&scope->context, &view_regions);
  }
  if (iree_status_is_ok(status) &&
      scope->context.policy->source_dataflow_provider != NULL) {
    status = loom_low_lower_context_initialize_source_dataflow(&scope->context,
                                                               view_regions);
  }
  if (!iree_status_is_ok(status)) {
    loom_low_lower_source_query_scope_deinitialize(scope);
    return status;
  }

  *out_scope = scope;
  return iree_ok_status();
}

void loom_low_lower_source_query_scope_deinitialize(
    loom_low_lower_source_query_scope_t* scope) {
  if (scope == NULL) {
    return;
  }
  if (scope->value_domain_initialized) {
    loom_local_value_domain_release(&scope->context.lowering.value_domain);
    scope->value_domain_initialized = false;
  }
  loom_low_lower_result_deinitialize(&scope->result);
  iree_arena_deinitialize(&scope->context.function_arena);
  memset(scope, 0, sizeof(*scope));
}

loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope) {
  return loom_low_lower_source_query_callback(&scope->context);
}

const loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    const loom_low_lower_source_query_scope_t* scope) {
  return scope->value_domain_initialized ? &scope->context.lowering.value_domain
                                         : NULL;
}

const loom_source_program_t* loom_low_lower_source_query_scope_program(
    const loom_low_lower_source_query_scope_t* scope) {
  return scope->value_domain_initialized
             ? &scope->context.lowering.source_program
             : NULL;
}

const loom_low_descriptor_set_t*
loom_low_lower_source_query_scope_descriptor_set(
    const loom_low_lower_source_query_scope_t* scope) {
  return scope->context.descriptor_set;
}

const loom_source_dataflow_result_t* loom_low_lower_source_query_scope_dataflow(
    const loom_low_lower_source_query_scope_t* scope) {
  return loom_low_lower_context_source_dataflow(&scope->context);
}

iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions) {
  return loom_low_lower_context_view_regions(&scope->context, out_view_regions);
}
