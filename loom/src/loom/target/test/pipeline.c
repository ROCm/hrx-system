// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/test/provider.h"
#include "loom/transforms/pipeline/control.h"

// A two-engine synthetic device uses ordinary calls for phase entry and join.
// The real realization lifecycle owns construction, worker cloning, captured
// facts, pruning, and mutation. Only physical selection/emission is synthetic.
typedef struct loom_test_pipeline_t {
  // Compiler-owned invocation and allocation lifetime.
  loom_pass_t* pass;
  // Engine selected by each worker's authored one-dimensional coordinate.
  iree_host_size_t* engines;
  // Production overlap and phase-participation plan.
  loom_pipeline_execution_t execution;
  // Synthetic device operations visible in the projected program.
  loom_symbol_ref_t boundaries[2];
  // Unique declaration suffix for synthetic collective transport operations.
  uint32_t next_collective;
} loom_test_pipeline_t;

static iree_status_t loom_test_pipeline_select(
    void* user_data, loom_module_t* module,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  (void)module;
  loom_test_pipeline_t* state = user_data;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, realization->invocations.worker_count,
      sizeof(*state->engines), (void**)&state->engines));
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &realization->workers[i];
    if (worker->rank != 1 || worker->axes[0].count != 1 ||
        worker->axes[0].origin >= 2) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test device requires one worker at [0] or [1]");
    }
    state->engines[i] = worker->axes[0].origin;
  }
  loom_pipeline_execution_conflict_t conflict;
  IREE_RETURN_IF_ERROR(loom_pipeline_execution_build(
      &realization->invocations, state->engines, 2, state->pass->arena,
      &state->execution, &conflict));
  if (conflict.second_worker != IREE_HOST_SIZE_MAX) {
    *out_valid = false;
    const loom_diagnostic_param_t parameters[] = {
        loom_param_string(IREE_SV("independent concurrent engines"))};
    return iree_diagnostic_emit(
        state->pass->diagnostic_emitter,
        &(loom_diagnostic_emission_t){
            .op = realization->invocations.workers[conflict.second_worker]
                      .strand->declaration,
            .error = LOOM_ERR_LOWERING_066,
            .params = parameters,
            .param_count = IREE_ARRAYSIZE(parameters)});
  }
  *out_valid = true;
  return iree_ok_status();
}

static bool loom_test_pipeline_has_worker_body(const void* user_data,
                                               iree_host_size_t worker_index) {
  (void)user_data;
  (void)worker_index;
  return true;
}

static iree_status_t loom_test_pipeline_worker(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker) {
  (void)user_data;
  (void)rewriter;
  (void)realization;
  (void)worker;
  return iree_ok_status();
}

static iree_status_t loom_test_pipeline_boundary(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_invocation_t* child, bool before,
    const loom_value_id_t* state_arguments) {
  (void)child;
  (void)state_arguments;
  if (!scope) {
    return iree_ok_status();
  }
  loom_test_pipeline_t* state = user_data;
  loom_op_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(engine), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &value));
  const loom_value_id_t argument = loom_index_constant_result(value);
  loom_op_t* call = NULL;
  return loom_func_call_build(builder, 0, 0, 0, 0, state->boundaries[before],
                              &argument, 1, NULL, 0, NULL, 0,
                              LOOM_LOCATION_UNKNOWN, &call);
}

static iree_status_t loom_test_pipeline_collective_call(
    loom_test_pipeline_t* state, loom_builder_t* builder, const char* role,
    const loom_value_id_t* arguments, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_call) {
  loom_module_t* module = builder->module;
  char text[64];
  iree_snprintf(text, sizeof(text), "__test_collective_%s_%u", role,
                state->next_collective++);
  loom_string_id_t name;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, iree_make_cstring_view(text), &name));
  loom_symbol_ref_t symbol = {0};
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol.symbol_id));
  loom_type_t* argument_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, argument_count, sizeof(*argument_types),
      (void**)&argument_types));
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    argument_types[i] = loom_module_value_type(module, arguments[i]);
  }
  loom_builder_t declaration_builder = *builder;
  loom_builder_enter_region(&declaration_builder, NULL, module->body);
  loom_op_t* declaration = NULL;
  IREE_RETURN_IF_ERROR(loom_func_decl_build(
      &declaration_builder, 0, 0, 0, LOOM_STRING_ID_INVALID,
      LOOM_STRING_ID_INVALID, 0, 0, 0, 0, loom_symbol_ref_null(),
      loom_attr_absent(), 0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
      argument_types, argument_count, result_types, result_count, NULL, 0, NULL,
      0, LOOM_LOCATION_UNKNOWN, &declaration));
  return loom_func_call_build(builder, 0, 0, 0, 0, symbol, arguments,
                              argument_count, result_types, result_count, NULL,
                              0, LOOM_LOCATION_UNKNOWN, out_call);
}

static iree_host_size_t loom_test_pipeline_collective_reader(
    loom_test_pipeline_t* state,
    const loom_pipeline_invocation_schedule_t* scope) {
  const loom_pipeline_execution_schedule_t* participants =
      loom_pipeline_execution_schedule(&state->execution, scope);
  return participants->workers[0] != IREE_HOST_SIZE_MAX ? 0 : 1;
}

static iree_status_t loom_test_pipeline_collective_begin(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* collective,
    const loom_value_id_t* state_arguments, loom_type_t payload_type,
    loom_value_id_t* out_payload) {
  (void)collective;
  (void)state_arguments;
  loom_test_pipeline_t* state = user_data;
  *out_payload = LOOM_VALUE_ID_INVALID;
  if (engine == loom_test_pipeline_collective_reader(state, scope)) {
    loom_op_t* call = NULL;
    IREE_RETURN_IF_ERROR(loom_test_pipeline_collective_call(
        state, builder, "acquire", NULL, 0, &payload_type, 1, &call));
    *out_payload = loom_op_results(call)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_test_pipeline_collective_end(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* collective, loom_value_id_t* values) {
  loom_test_pipeline_t* state = user_data;
  const bool reader =
      engine == loom_test_pipeline_collective_reader(state, scope);
  const iree_host_size_t count = collective->result_count;
  loom_op_t* call = NULL;
  if (reader) {
    IREE_RETURN_IF_ERROR(loom_test_pipeline_collective_call(
        state, builder, "send", values, count, NULL, 0, &call));
  } else {
    loom_type_t* types = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, count, sizeof(*types), (void**)&types));
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_pipeline_collective_result_t* result = &collective->results[i];
      types[i] = loom_module_value_type(
          builder->module,
          loom_op_results(
              collective->operations[result->operation])[result->result]);
    }
    IREE_RETURN_IF_ERROR(loom_test_pipeline_collective_call(
        state, builder, "receive", NULL, 0, types, count, &call));
    memcpy(values, loom_op_results(call), count * sizeof(*values));
  }
  IREE_RETURN_IF_ERROR(loom_test_pipeline_boundary(user_data, builder, engine,
                                                   scope, NULL, false, NULL));
  if (reader) {
    IREE_RETURN_IF_ERROR(loom_test_pipeline_collective_call(
        state, builder, "release", NULL, 0, NULL, 0, &call));
  }
  return iree_ok_status();
}

static iree_status_t loom_test_pipeline_entry(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization) {
  loom_test_pipeline_t* state = user_data;
  loom_builder_enter_region(&rewriter->builder, NULL, rewriter->module->body);
  const char* names[] = {"__test_phase_join", "__test_phase_enter"};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        rewriter->module, iree_make_cstring_view(names[i]), &name));
    state->boundaries[i].module_id = 0;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(
        rewriter->module, name, &state->boundaries[i].symbol_id));
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* declaration = NULL;
    IREE_RETURN_IF_ERROR(loom_func_decl_build(
        &rewriter->builder, 0, 0, 0, LOOM_STRING_ID_INVALID,
        LOOM_STRING_ID_INVALID, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
        state->boundaries[i], &type, 1, NULL, 0, NULL, 0, NULL, 0,
        LOOM_LOCATION_UNKNOWN, &declaration));
  }
  const loom_pipeline_control_options_t options = {
      .worker_engines = state->engines,
      .emit = {.boundary = loom_test_pipeline_boundary,
               .collective_begin = loom_test_pipeline_collective_begin,
               .collective_end = loom_test_pipeline_collective_end,
               .user_data = state}};
  loom_pipeline_control_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_control_materialize(
      rewriter, realization, &state->execution, &options, &functions));
  for (iree_host_size_t i = 0; i < state->execution.engine_count; ++i) {
    if (loom_func_like_isa(functions[i].function)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_attr(rewriter, functions[i].function.op, 1,
                                 loom_attr_enum(LOOM_FUNC_VISIBILITY_PUBLIC)));
    }
  }
  return loom_rewriter_erase(rewriter, realization->function.op);
}

static iree_status_t loom_test_pipeline_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  if (!loom_pipeline_def_isa(function.op) ||
      loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
    return iree_ok_status();
  }
  loom_test_pipeline_t state = {.pass = pass};
  const loom_pipeline_realization_inventory_t inventory = {0};
  const loom_pipeline_realization_callback_t callbacks = {
      .select = loom_test_pipeline_select,
      .has_worker_body = loom_test_pipeline_has_worker_body,
      .worker = loom_test_pipeline_worker,
      .entry = loom_test_pipeline_entry,
      .user_data = &state};
  return loom_pipeline_realize(pass, module, function, &inventory, &callbacks);
}

static const loom_pass_info_t* loom_test_pipeline_info(void) {
  static const loom_pass_info_t info = {
      .name = IREE_SVL("test.realize-pipeline"),
      .description =
          IREE_SVL("Realize resident control on two synthetic engines."),
      .kind = LOOM_PASS_FUNCTION};
  return &info;
}

static const loom_pass_descriptor_t kPipelinePasses[] = {{
    .key = IREE_SVL("test.realize-pipeline"),
    .info = loom_test_pipeline_info,
    .function_run = loom_test_pipeline_run,
}};

static const loom_pass_registry_t kPipelineRegistry = {
    .descriptors = kPipelinePasses,
    .descriptor_count = IREE_ARRAYSIZE(kPipelinePasses)};

const loom_target_provider_t loom_test_pipeline_provider = {
    .pass_registry = &kPipelineRegistry};
