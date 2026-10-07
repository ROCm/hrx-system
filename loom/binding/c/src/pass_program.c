// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pass_program.h"

#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/atomics.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/environment.h"
#include "loom/pass/pipeline_snapshot.h"
#include "loom/pass/tooling.h"
#include "loom/target/pipeline.h"
#include "loom/target/predicate.h"
#include "loomc/iree.h"
#include "loomc/target.h"
#include "module.h"
#include "option_chain.h"
#include "result.h"
#include "target.h"

enum {
  LOOMC_PASS_PROGRAM_DEFAULT_BLOCK_SIZE = 32 * 1024,
};

typedef enum loomc_pass_program_trace_stage_e {
  LOOMC_PASS_PROGRAM_TRACE_STAGE_PIPELINE_TEXT = 0,
  LOOMC_PASS_PROGRAM_TRACE_STAGE_PIPELINE_SYMBOL = 1,
  LOOMC_PASS_PROGRAM_TRACE_STAGE_SOURCE_LOW = 2,
  LOOMC_PASS_PROGRAM_TRACE_STAGE_PREPARED_LOW = 3,
} loomc_pass_program_trace_stage_t;

struct loomc_pass_program_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for pass-program-owned storage.
  loomc_allocator_t allocator;

  // Context retained by the pass program.
  loomc_context_t* context;

  // Stable arena block pool backing the pipeline module and compiled program.
  iree_arena_block_pool_t block_pool;

  // Owned scratch module containing the selected pass.pipeline IR.
  loom_module_t* pipeline_module;

  // Compiled immutable pass instruction program.
  loom_pass_program_t program;

  // Composed pass registry storage owned for descriptor pointer stability.
  loom_pass_registry_storage_t registry_storage;

  // Immutable registry view over registry_storage.
  const loom_pass_registry_t* registry;

  // Compiler-defined stage reported by pass-boundary traces.
  loomc_pass_program_trace_stage_t trace_stage;

  // True when program has been initialized and must be deinitialized.
  bool program_initialized;
};

typedef struct loomc_pass_program_compile_state_t {
  // Borrowed pass environment storage over context target tables.
  loom_codegen_pass_environment_storage_t codegen_environment_storage;

  // Target-aware pass.where predicate provider storage.
  loom_target_pass_predicate_provider_storage_t predicate_storage;

  // Compile options passed to the Loom pass program compiler.
  loom_pass_program_compile_options_t compile_options;
} loomc_pass_program_compile_state_t;

static loomc_status_t loomc_pass_program_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_pass_program_validate_options(
    const loomc_pass_program_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass program options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass program options structure_size is too small");
  }
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options->next, /*allowed_options=*/0, &option_chain));
  return loomc_pass_program_validate_string_view(options->identifier);
}

static loomc_status_t loomc_pass_program_validate_target_pipeline_options(
    const loomc_target_pipeline_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target pipeline options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target pipeline options structure_size is too small");
  }
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options->next, LOOMC_OPTION_CHAIN_ALLOW_SANITIZER, &option_chain));
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_program_validate_string_view(options->identifier));
  switch (options->kind) {
    case LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW:
    case LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW:
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target pipeline kind is invalid");
  }
  switch (options->control_flow_lowering) {
    case LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG:
    case LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW:
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target control-flow lowering mode is invalid");
  }
  return loomc_ok_status();
}

static iree_string_view_t loomc_pass_program_identifier(
    const loomc_pass_program_options_t* options) {
  if (options == NULL || loomc_string_view_is_empty(options->identifier)) {
    return IREE_SV("__loomc_pass_program");
  }
  return iree_string_view_from_loomc(options->identifier);
}

static iree_string_view_t loomc_pass_program_target_pipeline_identifier(
    const loomc_target_pipeline_options_t* options) {
  if (options == NULL || loomc_string_view_is_empty(options->identifier)) {
    return IREE_SV("__loomc_target_pipeline");
  }
  return iree_string_view_from_loomc(options->identifier);
}

static loomc_status_t loomc_pass_program_allocate_storage(
    loomc_context_t* context, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  loomc_pass_program_t* pass_program = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(allocator, sizeof(*pass_program),
                                               (void**)&pass_program));
  memset(pass_program, 0, sizeof(*pass_program));
  iree_atomic_ref_count_init(&pass_program->ref_count);
  pass_program->allocator = allocator;
  pass_program->context = context;
  loomc_context_retain(context);
  iree_arena_block_pool_initialize(LOOMC_PASS_PROGRAM_DEFAULT_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &pass_program->block_pool);
  loomc_status_t status = loomc_target_pass_registry_initialize(
      loomc_context_target_environment(context),
      &pass_program->registry_storage, &pass_program->registry);
  if (!loomc_status_is_ok(status)) {
    iree_arena_block_pool_deinitialize(&pass_program->block_pool);
    loomc_context_release(context);
    loomc_allocator_free(allocator, pass_program);
    return status;
  }

  *out_pass_program = pass_program;
  return loomc_ok_status();
}

static loomc_status_t loomc_pass_program_compile_state_initialize(
    loomc_pass_program_t* pass_program,
    loomc_pass_program_compile_state_t* out_state) {
  *out_state = (loomc_pass_program_compile_state_t){
      .compile_options =
          {
              .registry = pass_program->registry,
          },
  };

  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(pass_program->context);
  const loomc_target_pass_environment_t* target_pass_environment =
      target_environment != NULL
          ? loomc_target_environment_pass_environment(target_environment)
          : NULL;
  out_state->compile_options.environment =
      loomc_codegen_pass_environment_storage_initialize(
          target_pass_environment,
          loomc_context_cleanup_pattern_registry(pass_program->context),
          /*function_version_owner=*/NULL,
          /*launch_config_capability=*/NULL, /*compile_report=*/NULL,
          &out_state->codegen_environment_storage);
  if (target_environment == NULL) {
    return loomc_ok_status();
  }
  loom_target_pass_predicate_provider_storage_initialize(
      &pass_program->block_pool, &out_state->predicate_storage);
  out_state->compile_options.predicate_provider =
      loom_target_pass_predicate_provider(&out_state->predicate_storage);
  return loomc_ok_status();
}

static void loomc_pass_program_compile_state_deinitialize(
    loomc_pass_program_compile_state_t* state) {
  (void)state;
}

static loomc_status_t loomc_pass_program_allocate_pipeline_module(
    loomc_pass_program_t* pass_program,
    const loomc_pass_program_options_t* options) {
  iree_status_t status = loom_module_allocate(
      loomc_context_loom_context(pass_program->context),
      loomc_pass_program_identifier(options), &pass_program->block_pool, NULL,
      iree_allocator_from_loomc(pass_program->allocator),
      &pass_program->pipeline_module);
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_compile_pipeline_op(
    loomc_pass_program_t* pass_program, const loom_op_t* pipeline_op) {
  loomc_pass_program_compile_state_t state = {0};
  loomc_status_t status =
      loomc_pass_program_compile_state_initialize(pass_program, &state);
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  iree_status_t compile_status = loom_pass_program_compile_pipeline(
      pass_program->pipeline_module, pipeline_op, &state.compile_options,
      &pass_program->block_pool, &pass_program->program);
  loomc_pass_program_compile_state_deinitialize(&state);
  pass_program->program_initialized = iree_status_is_ok(compile_status);
  return loomc_status_from_iree(compile_status);
}

static loomc_status_t loomc_pass_program_compile_pipeline_op_with_state(
    loomc_pass_program_t* pass_program, const loom_op_t* pipeline_op,
    const loomc_pass_program_compile_state_t* state) {
  iree_status_t status = loom_pass_program_compile_pipeline(
      pass_program->pipeline_module, pipeline_op, &state->compile_options,
      &pass_program->block_pool, &pass_program->program);
  pass_program->program_initialized = iree_status_is_ok(status);
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_compile_flat_pipeline(
    loomc_pass_program_t* pass_program, loomc_string_view_t pipeline_text) {
  const loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_tool_build_flat_pipeline(
      pass_program->pipeline_module, iree_string_view_from_loomc(pipeline_text),
      pass_program->registry, &pipeline_op);
  if (iree_status_is_ok(status)) {
    return loomc_pass_program_compile_pipeline_op(pass_program, pipeline_op);
  }
  return loomc_status_from_iree(status);
}

static loom_target_control_flow_lowering_t
loomc_pass_program_target_control_flow_lowering(
    loomc_target_control_flow_lowering_t lowering) {
  switch (lowering) {
    case LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW:
      return LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
    case LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG:
    default:
      return LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
  }
}

static loom_target_pipeline_options_t
loomc_pass_program_target_pipeline_options(
    const loomc_target_pipeline_options_t* options,
    const loomc_option_chain_t* option_chain) {
  if (options == NULL) {
    return (loom_target_pipeline_options_t){
        .sanitizer = option_chain ? option_chain->sanitizer
                                  : (loom_sanitizer_options_t){0},
    };
  }
  return (loom_target_pipeline_options_t){
      .source_to_low_max_errors = options->source_to_low_max_errors,
      .control_flow_lowering = loomc_pass_program_target_control_flow_lowering(
          options->control_flow_lowering),
      .sanitizer = option_chain ? option_chain->sanitizer
                                : (loom_sanitizer_options_t){0},
  };
}

static loomc_status_t loomc_pass_program_build_target_pipeline(
    loomc_pass_program_t* pass_program, loomc_target_pipeline_kind_t kind,
    iree_string_view_t identifier,
    const loom_target_pipeline_options_t* pipeline_options,
    const loomc_pass_program_compile_state_t* state,
    const loom_op_t** out_pipeline_op) {
  *out_pipeline_op = NULL;
  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(pass_program->context);
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "context has no target environment");
  }

  loom_op_t* pipeline_op = NULL;
  iree_status_t status = iree_ok_status();
  switch (kind) {
    case LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW:
      status = loom_target_pipeline_build_to_prepared_low(
          pass_program->pipeline_module, identifier, pipeline_options,
          loomc_target_environment_loom_target_environment(target_environment),
          state->compile_options.environment, &pipeline_op);
      break;
    case LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW:
      status = loom_target_pipeline_build_to_source_low(
          pass_program->pipeline_module, identifier, pipeline_options,
          loomc_target_environment_loom_target_environment(target_environment),
          state->compile_options.environment, &pipeline_op);
      break;
  }
  if (iree_status_is_ok(status)) {
    *out_pipeline_op = pipeline_op;
  }
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_fail_result_from_status(
    loomc_result_t* result, loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("PASS_PROGRAM/INVALID"), status);
}

static loomc_status_t loomc_pass_program_create_target_pipeline_into_result(
    loomc_context_t* context, loomc_target_pipeline_kind_t kind,
    iree_string_view_t identifier,
    const loom_target_pipeline_options_t* pipeline_options,
    loomc_allocator_t allocator, loomc_result_t* result,
    loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(pipeline_options);
  IREE_ASSERT_ARGUMENT(result);
  IREE_ASSERT(loomc_result_succeeded(result));

  const loomc_pass_program_options_t pass_options = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      .structure_size = sizeof(pass_options),
      .identifier = loomc_string_view_from_iree(identifier),
  };
  loomc_pass_program_t* pass_program = NULL;
  const loom_op_t* pipeline_op = NULL;
  loomc_pass_program_compile_state_t compile_state = {0};
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    pass_program->trace_stage =
        kind == LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW
            ? LOOMC_PASS_PROGRAM_TRACE_STAGE_SOURCE_LOW
            : LOOMC_PASS_PROGRAM_TRACE_STAGE_PREPARED_LOW;
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_allocate_pipeline_module(pass_program,
                                                         &pass_options);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_state_initialize(pass_program,
                                                         &compile_state);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_build_target_pipeline(
        pass_program, kind, identifier, pipeline_options, &compile_state,
        &pipeline_op);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_pipeline_op_with_state(
        pass_program, pipeline_op, &compile_state);
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_pass_program_fail_result_from_status(result, status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    *out_pass_program = pass_program;
    pass_program = NULL;
  }
  loomc_pass_program_compile_state_deinitialize(&compile_state);
  loomc_pass_program_release(pass_program);
  return status;
}

loomc_status_t loomc_pass_program_create_from_internal_target_pipeline(
    loomc_context_t* context,
    const loom_target_pipeline_options_t* pipeline_options,
    loomc_allocator_t allocator, loomc_result_t* result,
    loomc_pass_program_t** out_pass_program) {
  if (context == NULL || pipeline_options == NULL || result == NULL ||
      out_pass_program == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, pipeline_options, result, and out_pass_program must not be "
        "NULL");
  }
  if (loomc_context_target_environment(context) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "context has no target environment");
  }
  return loomc_pass_program_create_target_pipeline_into_result(
      context, LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      IREE_SV("__loomc_target_pipeline"), pipeline_options, allocator, result,
      out_pass_program);
}

loomc_status_t loomc_pass_program_create_empty(
    loomc_context_t* context, const loomc_pass_program_options_t* options,
    loomc_allocator_t allocator, loomc_pass_program_t** out_pass_program) {
  if (out_pass_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_pass_program must not be NULL");
  }
  *out_pass_program = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));

  loomc_pass_program_t* pass_program = NULL;
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_allocate_pipeline_module(pass_program, options);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_flat_pipeline(
        pass_program, loomc_string_view_empty());
  }
  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    pass_program = NULL;
  }
  loomc_pass_program_release(pass_program);
  return status;
}

loomc_status_t loomc_pass_program_create_from_pipeline_text(
    loomc_context_t* context, loomc_string_view_t pipeline_text,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result) {
  if (out_pass_program == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_pass_program and out_result must not be NULL");
  }
  *out_pass_program = NULL;
  *out_result = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_string_view(pipeline_text));
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_RESULT_STATE_SUCCEEDED, loomc_context_source_retention(context),
      allocator, &result));

  loomc_pass_program_t* pass_program = NULL;
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_allocate_pipeline_module(pass_program, options);
  }
  if (loomc_status_is_ok(status)) {
    status =
        loomc_pass_program_compile_flat_pipeline(pass_program, pipeline_text);
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_pass_program_fail_result_from_status(result, status);
    loomc_pass_program_release(pass_program);
    pass_program = NULL;
  }

  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    *out_result = result;
    pass_program = NULL;
    result = NULL;
  }
  loomc_pass_program_release(pass_program);
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_pass_program_create_from_module_symbol(
    const loomc_module_t* module, loomc_string_view_t pipeline_symbol,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result) {
  if (out_pass_program == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_pass_program and out_result must not be NULL");
  }
  *out_pass_program = NULL;
  *out_result = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_program_validate_string_view(pipeline_symbol));
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));
  loomc_context_t* context = loomc_module_context(module);

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_RESULT_STATE_SUCCEEDED, loomc_context_source_retention(context),
      allocator, &result));

  loomc_pass_program_t* pass_program = NULL;
  const loom_module_t* source_module = loomc_module_const_loom_module(module);
  const loom_op_t* pipeline_op = NULL;
  loom_pass_pipeline_snapshot_t snapshot = {0};
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    pass_program->trace_stage = LOOMC_PASS_PROGRAM_TRACE_STAGE_PIPELINE_SYMBOL;
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_pass_pipeline_snapshot_initialize(
        source_module, iree_string_view_from_loomc(pipeline_symbol),
        loomc_pass_program_identifier(options), &pass_program->block_pool,
        iree_allocator_from_loomc(pass_program->allocator), &snapshot));
  }
  if (loomc_status_is_ok(status)) {
    pass_program->pipeline_module = snapshot.module;
    pipeline_op = snapshot.pipeline_op;
    snapshot = (loom_pass_pipeline_snapshot_t){0};
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_pipeline_op(pass_program, pipeline_op);
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_pass_program_fail_result_from_status(result, status);
    loomc_pass_program_release(pass_program);
    pass_program = NULL;
  }

  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    *out_result = result;
    pass_program = NULL;
    result = NULL;
  }
  loomc_pass_program_release(pass_program);
  loomc_result_release(result);
  loom_pass_pipeline_snapshot_deinitialize(&snapshot);
  return status;
}

loomc_status_t loomc_pass_program_create_from_target_pipeline(
    loomc_context_t* context, const loomc_target_pipeline_options_t* options,
    loomc_allocator_t allocator, loomc_pass_program_t** out_pass_program,
    loomc_result_t** out_result) {
  if (out_pass_program == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_pass_program and out_result must not be NULL");
  }
  *out_pass_program = NULL;
  *out_result = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  if (loomc_context_target_environment(context) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "context has no target environment");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_program_validate_target_pipeline_options(options));
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options ? options->next : NULL, LOOMC_OPTION_CHAIN_ALLOW_SANITIZER,
      &option_chain));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_RESULT_STATE_SUCCEEDED, loomc_context_source_retention(context),
      allocator, &result));

  const loom_target_pipeline_options_t internal_options =
      loomc_pass_program_target_pipeline_options(options, &option_chain);
  loomc_pass_program_t* pass_program = NULL;
  loomc_status_t status = loomc_pass_program_create_target_pipeline_into_result(
      context,
      options ? options->kind : LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      loomc_pass_program_target_pipeline_identifier(options), &internal_options,
      allocator, result, &pass_program);

  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    *out_result = result;
    pass_program = NULL;
    result = NULL;
  }
  loomc_pass_program_release(pass_program);
  loomc_result_release(result);
  return status;
}

void loomc_pass_program_retain(loomc_pass_program_t* pass_program) {
  if (pass_program == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&pass_program->ref_count);
}

void loomc_pass_program_release(loomc_pass_program_t* pass_program) {
  if (pass_program == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&pass_program->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = pass_program->allocator;
  if (pass_program->program_initialized) {
    loom_pass_program_deinitialize(&pass_program->program);
  }
  loom_module_free(pass_program->pipeline_module);
  iree_arena_block_pool_deinitialize(&pass_program->block_pool);
  loomc_context_release(pass_program->context);
  loomc_allocator_free(allocator, pass_program);
}

loomc_context_t* loomc_pass_program_context(
    const loomc_pass_program_t* pass_program) {
  return pass_program ? pass_program->context : NULL;
}

const loom_pass_program_t* loomc_pass_program_loom_pass_program(
    const loomc_pass_program_t* pass_program) {
  return pass_program ? &pass_program->program : NULL;
}

iree_string_view_t loomc_pass_program_trace_stage(
    const loomc_pass_program_t* pass_program) {
  if (pass_program == NULL) {
    return iree_string_view_empty();
  }
  switch (pass_program->trace_stage) {
    case LOOMC_PASS_PROGRAM_TRACE_STAGE_PIPELINE_TEXT:
      return IREE_SV("pipeline-text");
    case LOOMC_PASS_PROGRAM_TRACE_STAGE_PIPELINE_SYMBOL:
      return IREE_SV("pipeline-symbol");
    case LOOMC_PASS_PROGRAM_TRACE_STAGE_SOURCE_LOW:
      return IREE_SV("source-low");
    case LOOMC_PASS_PROGRAM_TRACE_STAGE_PREPARED_LOW:
      return IREE_SV("prepared-low");
    default:
      return IREE_SV("unknown");
  }
}
