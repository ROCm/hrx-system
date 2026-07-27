// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/pipeline.h"

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/error/diagnostic.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/registry.h"
#include "loom/pass/tooling.h"
#include "loom/target/entry_selection.h"
#include "loom/target/pipeline.h"
#include "loom/target/predicate.h"
#include "loom/target/provider.h"

enum {
  LOOM_COMPILE_DEFAULT_MAX_PIPELINE_ERRORS = 20,
};

void loom_compile_pipeline_options_initialize(
    loom_compile_pipeline_options_t* out_options) {
  *out_options = (loom_compile_pipeline_options_t){
      .pipeline = IREE_SVL("default"),
      .default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW,
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_COMPILE_DEFAULT_MAX_PIPELINE_ERRORS,
  };
}

bool loom_compile_pipeline_is_disabled(iree_string_view_t pipeline) {
  pipeline = iree_string_view_trim(pipeline);
  return iree_string_view_equal(pipeline, IREE_SV("none"));
}

bool loom_compile_pipeline_is_default(iree_string_view_t pipeline) {
  pipeline = iree_string_view_trim(pipeline);
  return iree_string_view_is_empty(pipeline) ||
         iree_string_view_equal(pipeline, IREE_SV("default"));
}

static iree_status_t loom_compile_pipeline_registry_initialize(
    const loom_target_environment_t* target_environment,
    loom_pass_registry_storage_t* out_storage,
    const loom_pass_registry_t** out_registry) {
  const loom_pass_registry_t* registries[] = {
      loom_pass_builtin_registry(),
      loom_target_environment_pass_registry(target_environment),
  };
  IREE_RETURN_IF_ERROR(loom_pass_registry_storage_initialize_from_registries(
      registries, IREE_ARRAYSIZE(registries), out_storage));
  *out_registry = loom_pass_registry_storage_registry(out_storage);
  return iree_ok_status();
}

static iree_status_t loom_compile_build_default_pipeline(
    loom_module_t* pipeline_module,
    const loom_compile_pipeline_options_t* options,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  switch (options->default_pipeline) {
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW:
      return loom_target_pipeline_build_to_source_low(
          pipeline_module, IREE_SV("__loom_compile_default"),
          &options->target_pipeline_options, options->target_environment,
          pass_environment, out_pipeline_op);
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_ARTIFACTS:
      return loom_target_pipeline_build_to_source_low_artifacts(
          pipeline_module, IREE_SV("__loom_compile_default"),
          &options->target_pipeline_options, options->target_environment,
          pass_environment, out_pipeline_op);
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_DIAGNOSTIC_ARTIFACTS:
      return loom_target_pipeline_build_to_source_low_diagnostic_artifacts(
          pipeline_module, IREE_SV("__loom_compile_default"),
          &options->target_pipeline_options, options->target_environment,
          pass_environment, out_pipeline_op);
    case LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW:
      return loom_target_pipeline_build_to_prepared_low(
          pipeline_module, IREE_SV("__loom_compile_default"),
          &options->target_pipeline_options, options->target_environment,
          pass_environment, out_pipeline_op);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown Loom default compile pipeline %d",
                              (int)options->default_pipeline);
  }
}

static iree_string_view_t loom_compile_default_pipeline_stage_name(
    loom_compile_default_pipeline_t default_pipeline) {
  switch (default_pipeline) {
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW:
      return IREE_SV("source-low");
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_ARTIFACTS:
      return IREE_SV("source-low-artifacts");
    case LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW_DIAGNOSTIC_ARTIFACTS:
      return IREE_SV("source-low-diagnostic-artifacts");
    case LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW:
      return IREE_SV("prepared-low");
    default:
      return IREE_SV("default");
  }
}

static iree_string_view_t loom_compile_pipeline_stage_name(
    const loom_compile_pipeline_options_t* options,
    iree_string_view_t pipeline) {
  if (loom_compile_pipeline_is_default(pipeline)) {
    return loom_compile_default_pipeline_stage_name(options->default_pipeline);
  }
  if (iree_string_view_starts_with_char(pipeline, '@')) {
    return IREE_SV("module-pipeline");
  }
  return IREE_SV("command-line");
}

static iree_status_t loom_compile_run_default_pipeline(
    loom_module_t* module, const loom_compile_pipeline_options_t* options,
    const loom_pass_tool_run_options_t* run_options,
    loom_pass_run_result_t* out_result) {
  *out_result = (loom_pass_run_result_t){0};

  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      module->context, IREE_SV("__loom_compile_default_pipeline"),
      run_options->block_pool, NULL, module->allocator, &pipeline_module);
  loom_op_t* pipeline_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_compile_build_default_pipeline(
        pipeline_module, options, run_options->environment, &pipeline_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_pass_tool_run_pipeline_module_op(
        module, pipeline_module, pipeline_op, run_options, out_result);
  }
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  return status;
}

iree_status_t loom_compile_run_pipeline(
    loom_module_t* module, const loom_compile_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool, loom_pass_run_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_pass_run_result_t){0};

  iree_string_view_t pipeline = iree_string_view_trim(options->pipeline);
  if (loom_compile_pipeline_is_disabled(pipeline)) {
    return iree_ok_status();
  }
  if (options->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Loom compile pass pipelines require a target "
                            "environment");
  }
  if (options->low_descriptor_registry == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Loom compile pass pipelines require a target-low "
                            "descriptor registry");
  }

  loom_pass_registry_storage_t pass_registry_storage = {0};
  const loom_pass_registry_t* pass_registry = NULL;
  IREE_RETURN_IF_ERROR(loom_compile_pipeline_registry_initialize(
      options->target_environment, &pass_registry_storage, &pass_registry));

  loom_low_lower_policy_registry_t low_lower_policy_registry = {0};
  IREE_RETURN_IF_ERROR(
      loom_target_environment_initialize_low_lower_policy_registry(
          options->target_environment, &low_lower_policy_registry));
  loom_target_math_policy_registry_t math_policy_registry = {0};
  IREE_RETURN_IF_ERROR(loom_target_environment_initialize_math_policy_registry(
      options->target_environment, &math_policy_registry));
  const loom_target_low_legality_provider_list_t low_legality_provider_list =
      loom_target_environment_low_legality_provider_list(
          options->target_environment);
  const loom_target_legalizer_provider_list_t legalizer_provider_list =
      loom_target_environment_legalizer_provider_list(
          options->target_environment);

  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
  };
  loom_target_entry_diagnostic_emitter_t pass_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &entry_options, LOOM_EMITTER_PASS, &pass_emitter);
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  if (!loom_target_selection_is_empty(options->target_selection)) {
    IREE_RETURN_IF_ERROR(loom_target_environment_materialize_selection(
        options->target_environment, module, options->target_selection,
        &target_ref));
  }

  loom_low_pass_environment_storage_t low_pass_environment_storage = {0};
  loom_target_pass_predicate_provider_storage_t predicate_storage = {0};
  loom_target_pass_predicate_provider_storage_initialize(block_pool,
                                                         &predicate_storage);
  if (options->report != NULL) {
    loom_target_compile_report_initialize_if_empty(options->report,
                                                   block_pool->block_allocator);
  }

  loom_pass_trace_options_t trace_options = {0};
  loom_pass_trace_t trace = {0};
  loom_pass_trace_t* trace_ptr = NULL;
  if (loom_pass_trace_options_is_enabled(options->trace_options)) {
    trace_options = *options->trace_options;
    trace_options.stage = loom_compile_pipeline_stage_name(options, pipeline);
    loom_pass_trace_initialize(&trace_options, &trace);
    trace_ptr = &trace;
  }
  loom_pass_tool_run_options_t run_options = {
      .registry = pass_registry,
      .environment = loom_low_pass_environment_storage_initialize(
          &options->low_descriptor_registry->registry,
          &low_lower_policy_registry, &low_legality_provider_list,
          &legalizer_provider_list, &math_policy_registry, options->report,
          options->target_selection, target_ref, &low_pass_environment_storage),
      .predicate_provider =
          loom_target_pass_predicate_provider(&predicate_storage),
      .block_pool = block_pool,
      .diagnostic_emitter = loom_target_entry_emitter(&pass_emitter),
      .trace = trace_ptr,
  };

  iree_status_t status = iree_ok_status();
  if (loom_compile_pipeline_is_default(pipeline)) {
    status = loom_compile_run_default_pipeline(module, options, &run_options,
                                               out_result);
  } else if (iree_string_view_starts_with_char(pipeline, '@')) {
    status = loom_pass_tool_run_pipeline_symbol(module, pipeline, &run_options,
                                                out_result);
  } else {
    status = loom_pass_tool_run_flat_pipeline(module, pipeline, &run_options,
                                              out_result);
  }
  return status;
}
