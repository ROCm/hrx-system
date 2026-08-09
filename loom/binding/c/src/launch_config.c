// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <stdint.h>

#include "config.h"
#include "context.h"
#include "diagnostic.h"
#include "iree/base/api.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loom/ops/op_defs.h"
#include "loom/tooling/compile/pipeline.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

typedef struct loomc_launch_config_target_capture_t {
  // Result receiving target contract diagnostics.
  loomc_result_t* result;
} loomc_launch_config_target_capture_t;

typedef struct loomc_launch_config_resolved_options_t {
  // Kernel function symbol to evaluate.
  iree_string_view_t function_symbol;

  // Requested config bindings.
  const loomc_config_options_t* config;

  // Workload argument values.
  const int64_t* workload_arguments;

  // Number of workload argument values.
  loomc_host_size_t workload_argument_count;

  // Fields that must be proven.
  loomc_launch_config_field_flags_t required_fields;

  // Optional exact per-function target specializations.
  const loomc_target_specialization_options_t* target_specialization;
} loomc_launch_config_resolved_options_t;

typedef struct loomc_launch_config_target_preparation_t {
  // Expanded-source pipeline products owning exact target facts.
  loom_compile_pipeline_result_t pipeline_result;

  // Exact target facts selected for the requested launch function.
  const loom_target_facts_t* effective_target_facts;

  // True when pipeline_result requires deinitialization.
  bool pipeline_initialized;
} loomc_launch_config_target_preparation_t;

static iree_status_t loomc_launch_config_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_launch_config_target_capture_t* capture =
      (loomc_launch_config_target_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static iree_status_t loomc_launch_config_capture_pipeline_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_launch_config_target_capture_t* capture =
      (loomc_launch_config_target_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, /*source=*/NULL, diagnostic));
}

static bool loomc_launch_config_string_view_is_valid(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static bool loomc_launch_config_fields_are_valid(
    loomc_launch_config_field_flags_t fields) {
  const loomc_launch_config_field_flags_t known_fields =
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE |
      LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  return (fields & ~known_fields) == 0;
}

static loom_kernel_launch_config_field_flags_t
loomc_launch_config_field_flags_to_kernel(
    loomc_launch_config_field_flags_t fields) {
  loom_kernel_launch_config_field_flags_t kernel_fields = 0;
  if (iree_any_bit_set(fields,
                       LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT)) {
    kernel_fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT;
  }
  if (iree_any_bit_set(fields, LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE)) {
    kernel_fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE;
  }
  if (iree_any_bit_set(fields, LOOMC_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE)) {
    kernel_fields |= LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE;
  }
  if (iree_any_bit_set(
          fields, LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES)) {
    kernel_fields |=
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  }
  return kernel_fields;
}

static loomc_launch_config_field_flags_t
loomc_launch_config_field_flags_from_kernel(
    loom_kernel_launch_config_field_flags_t fields) {
  loomc_launch_config_field_flags_t public_fields = 0;
  if (iree_any_bit_set(fields,
                       LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT)) {
    public_fields |= LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT;
  }
  if (iree_any_bit_set(fields,
                       LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE)) {
    public_fields |= LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE;
  }
  if (iree_any_bit_set(fields,
                       LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE)) {
    public_fields |= LOOMC_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE;
  }
  if (iree_any_bit_set(
          fields,
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES)) {
    public_fields |= LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES;
  }
  return public_fields;
}

static bool loomc_launch_config_options_have_target_specialization(
    const loomc_launch_config_resolved_options_t* options) {
  return options->target_specialization != NULL &&
         options->target_specialization->specialization_count != 0;
}

// TODO(benvanik): Remove source-module cloning when compilation emits a
// compact launch configuration artifact. Configuration and target
// specialization must then be applied only by the owning compilation.
static bool loomc_launch_config_options_require_module_clone(
    const loomc_launch_config_resolved_options_t* options) {
  const loomc_config_options_t* config = options->config;
  return loomc_launch_config_options_have_target_specialization(options) ||
         (config != NULL && (config->binding_count != 0 ||
                             !loomc_string_view_is_empty(config->json_object) ||
                             config->flags != 0));
}

static loomc_status_t loomc_launch_config_validate_result(
    const loomc_launch_config_t* config) {
  if (config == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_config must not be NULL");
  }
  if (config->type != LOOMC_STRUCTURE_TYPE_NONE &&
      config->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config has an unknown structure type");
  }
  if (config->structure_size != 0 && config->structure_size < sizeof(*config)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config structure_size is too small");
  }
  if (config->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config result extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_resolve_options(
    const loomc_launch_config_eval_options_t* options,
    loomc_launch_config_resolved_options_t* out_options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config eval options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_EVAL_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config eval options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config eval options structure_size is too small");
  }
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, &target_specialization));
  if (!loomc_launch_config_string_view_is_valid(options->function_symbol)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "function symbol has length but no data");
  }
  if (loomc_string_view_is_empty(options->function_symbol)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "function symbol must not be empty");
  }
  if (options->workload_argument_count != 0 &&
      options->workload_arguments == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workload_argument_count is non-zero but workload_arguments is NULL");
  }
  if (!loomc_launch_config_fields_are_valid(options->required_fields)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config required_fields contains unknown bits");
  }
  LOOMC_RETURN_IF_ERROR(loomc_config_validate_options(&options->config));

  *out_options = (loomc_launch_config_resolved_options_t){
      .function_symbol = iree_string_view_from_loomc(options->function_symbol),
      .config = &options->config,
      .workload_arguments = options->workload_arguments,
      .workload_argument_count = options->workload_argument_count,
      .required_fields = options->required_fields,
      .target_specialization = target_specialization,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_fail_status(loomc_result_t* result,
                                                      loomc_string_view_t code,
                                                      loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status);
}

static loomc_status_t loomc_launch_config_fail_missing_required_field(
    loomc_result_t* result, loomc_string_view_t code) {
  return loomc_launch_config_fail_status(
      result, code,
      loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "required launch config field could not be resolved to a concrete "
          "value"));
}

static loomc_status_t loomc_launch_config_report_failure(
    loomc_result_t* result, loom_kernel_launch_config_failure_t failure) {
  switch (failure) {
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE:
      return loomc_ok_status();
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_FUNCTION_NOT_FOUND:
      return loomc_launch_config_fail_status(
          result, loomc_make_cstring_view("LAUNCH_CONFIG/NOT_FOUND"),
          loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                            "kernel function symbol was not found"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NOT_KERNEL:
      return loomc_launch_config_fail_status(
          result, loomc_make_cstring_view("LAUNCH_CONFIG/NOT_KERNEL"),
          loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "function is not a source kernel with launch config"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_COUNT:
      return loomc_launch_config_fail_status(
          result,
          loomc_make_cstring_view("LAUNCH_CONFIG/WORKLOAD_ARGUMENT_COUNT"),
          loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                            "workload argument count does not match launch "
                            "config signature"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_WORKLOAD_ARGUMENT_TYPE:
      return loomc_launch_config_fail_status(
          result,
          loomc_make_cstring_view("LAUNCH_CONFIG/WORKLOAD_ARGUMENT_TYPE"),
          loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "workload argument type cannot be seeded from i64"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_TARGET_CONTRACT:
      if (loomc_result_diagnostic_count(result) != 0) {
        return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
      }
      return loomc_launch_config_fail_status(
          result, loomc_make_cstring_view("LAUNCH_CONFIG/TARGET_CONTRACT"),
          loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                            "target contract could not be resolved"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_COUNT:
      return loomc_launch_config_fail_missing_required_field(
          result,
          loomc_make_cstring_view("LAUNCH_CONFIG/MISSING_WORKGROUP_COUNT"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_SIZE:
      return loomc_launch_config_fail_missing_required_field(
          result,
          loomc_make_cstring_view("LAUNCH_CONFIG/MISSING_WORKGROUP_SIZE"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_SUBGROUP_SIZE:
      return loomc_launch_config_fail_missing_required_field(
          result,
          loomc_make_cstring_view("LAUNCH_CONFIG/MISSING_SUBGROUP_SIZE"));
    case LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_MISSING_WORKGROUP_STORAGE_BYTES:
      return loomc_launch_config_fail_missing_required_field(
          result, loomc_make_cstring_view(
                      "LAUNCH_CONFIG/MISSING_WORKGROUP_STORAGE_BYTES"));
  }
  return loomc_launch_config_fail_status(
      result, loomc_make_cstring_view("LAUNCH_CONFIG/UNKNOWN_FAILURE"),
      loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "launch config evaluation returned an unknown failure"));
}

static void loomc_launch_config_copy_from_kernel(
    const loom_kernel_launch_config_t* source, loomc_launch_config_t* target) {
  target->fields = loomc_launch_config_field_flags_from_kernel(source->fields);
  target->workgroup_count = (loomc_dimension3_t){
      .x = source->workgroup_count.x,
      .y = source->workgroup_count.y,
      .z = source->workgroup_count.z,
  };
  target->workgroup_size = (loomc_dimension3_t){
      .x = source->workgroup_size.x,
      .y = source->workgroup_size.y,
      .z = source->workgroup_size.z,
  };
  target->subgroup_size = source->subgroup_size;
  target->workgroup_storage_bytes = source->workgroup_storage_bytes;
}

static loom_kernel_launch_config_options_t
loomc_launch_config_make_kernel_options(
    const loomc_launch_config_resolved_options_t* options,
    const loom_target_facts_t* effective_target_facts,
    iree_diagnostic_emitter_t emitter) {
  return (loom_kernel_launch_config_options_t){
      .function_symbol = options->function_symbol,
      .workload_arguments = options->workload_arguments,
      .workload_argument_count = options->workload_argument_count,
      .required_fields =
          loomc_launch_config_field_flags_to_kernel(options->required_fields),
      .effective_target_facts = effective_target_facts,
      .diagnostic_emitter = emitter,
  };
}

static loom_func_like_t loomc_launch_config_find_function(
    loom_module_t* module, iree_string_view_t function_symbol) {
  function_symbol = iree_string_view_trim(function_symbol);
  (void)iree_string_view_consume_prefix_char(&function_symbol, '@');
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, function_symbol);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return (loom_func_like_t){0};
  }
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return (loom_func_like_t){0};
  }
  return loom_func_like_cast(module,
                             module->symbols.entries[symbol_id].defining_op);
}

static void loomc_launch_config_target_preparation_deinitialize(
    loomc_launch_config_target_preparation_t* preparation) {
  if (preparation->pipeline_initialized) {
    loom_compile_pipeline_result_deinitialize(&preparation->pipeline_result);
  }
  *preparation = (loomc_launch_config_target_preparation_t){0};
}

// TODO(benvanik): Delete this parallel expanded-source compilation when the
// device compilation emits launch functions from its prepared function
// versions.
static loomc_status_t loomc_launch_config_prepare_target(
    loomc_context_t* context, loom_module_t* module,
    loomc_workspace_t* workspace,
    const loomc_launch_config_resolved_options_t* options,
    loomc_result_t* result,
    loomc_launch_config_target_preparation_t* out_preparation) {
  *out_preparation = (loomc_launch_config_target_preparation_t){0};
  const loomc_target_specialization_options_t* target_specialization =
      options->target_specialization;
  if (target_specialization == NULL ||
      target_specialization->specialization_count == 0) {
    return loomc_ok_status();
  }

  loomc_target_environment_t* public_target_environment =
      loomc_context_target_environment(context);
  const loomc_target_pass_environment_t* pass_environment =
      loomc_context_target_pass_environment(context);

  iree_arena_allocator_t request_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &request_arena);
  loom_target_specialization_request_list_t requests = {0};
  loomc_status_t status = loomc_target_specialization_options_make_request_list(
      target_specialization, &request_arena, &requests);
  if (loomc_status_is_ok(status)) {
    loomc_launch_config_target_capture_t capture = {
        .result = result,
    };
    loom_compile_pipeline_options_t pipeline_options = {0};
    loom_compile_pipeline_options_initialize(&pipeline_options);
    pipeline_options.default_pipeline =
        LOOM_COMPILE_DEFAULT_PIPELINE_EXPANDED_SOURCE;
    pipeline_options.target_environment =
        loomc_target_environment_loom_target_environment(
            public_target_environment);
    pipeline_options.target_specializations = requests;
    pipeline_options.low_descriptor_registry =
        &pass_environment->low_descriptor_registry;
    pipeline_options.diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = loomc_launch_config_capture_pipeline_diagnostic,
        .user_data = &capture,
    };
    out_preparation->pipeline_initialized = true;
    status = loomc_status_from_iree(loom_compile_run_pipeline(
        module, &pipeline_options, loomc_workspace_block_pool(workspace),
        &out_preparation->pipeline_result));
  }
  iree_arena_deinitialize(&request_arena);

  if (loomc_status_is_ok(status) &&
      out_preparation->pipeline_result.pass.error_count != 0) {
    status = loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loom_func_like_t function =
        loomc_launch_config_find_function(module, options->function_symbol);
    if (loom_func_like_isa(function)) {
      const loom_target_function_version_t* function_version =
          loom_target_function_version_list_find(
              &out_preparation->pipeline_result.function_versions.list,
              function);
      if (function_version != NULL) {
        out_preparation->effective_target_facts =
            function_version->effective_target_facts;
      }
    }
  }
  return status;
}

loomc_status_t loomc_module_evaluate_launch_config(
    const loomc_module_t* module, loomc_workspace_t* workspace,
    const loomc_launch_config_eval_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_t* out_config,
    loomc_result_t** out_result) {
  if (module == NULL || workspace == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module, workspace, and out_result must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_result(out_config));
  *out_config = (loomc_launch_config_t){
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(*out_config),
  };
  *out_result = NULL;

  loomc_launch_config_resolved_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_launch_config_resolve_options(options, &resolved_options));
  loomc_context_t* context = loomc_module_context(module);
  LOOMC_RETURN_IF_ERROR(
      loomc_target_specialization_options_validate_environment(
          resolved_options.target_specialization,
          loomc_context_target_environment(context)));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  const loom_module_t* source_internal_module =
      loomc_module_const_loom_module(module);
  const bool requires_module_clone =
      loomc_launch_config_options_require_module_clone(&resolved_options);
  if (source_internal_module != NULL && !requires_module_clone &&
      resolved_options.workload_argument_count == 0) {
    bool direct_evaluated = false;
    loom_kernel_launch_config_t kernel_config = {0};
    const loom_kernel_launch_config_options_t kernel_options =
        loomc_launch_config_make_kernel_options(&resolved_options, NULL,
                                                (iree_diagnostic_emitter_t){0});
    loomc_status_t direct_status =
        loomc_status_from_iree(loom_kernel_launch_config_try_evaluate_direct(
            source_internal_module, loomc_workspace_block_pool(workspace),
            &kernel_options, &kernel_config, &direct_evaluated));
    if (!loomc_status_is_ok(direct_status)) {
      loomc_result_release(result);
      return direct_status;
    }
    if (direct_evaluated) {
      loomc_launch_config_copy_from_kernel(&kernel_config, out_config);
      *out_result = result;
      return loomc_ok_status();
    }
  }

  loomc_module_t* scratch_module = NULL;
  loomc_launch_config_target_preparation_t target_preparation = {0};
  loomc_status_t status = loomc_ok_status();
  const loom_module_t* internal_module = source_internal_module;
  if (internal_module == NULL || requires_module_clone) {
    status = loomc_module_clone(module, workspace, allocator, &scratch_module);
    internal_module = loomc_status_is_ok(status)
                          ? loomc_module_const_loom_module(scratch_module)
                          : NULL;
  }
  if (loomc_status_is_ok(status) && scratch_module != NULL) {
    const loomc_config_apply_to_module_options_t config_options = {
        .config = resolved_options.config,
        .module = loomc_module_loom_module(scratch_module),
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = allocator,
    };
    status = loomc_config_apply_to_module(&config_options);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      loomc_launch_config_options_have_target_specialization(
          &resolved_options)) {
    status = loomc_launch_config_prepare_target(
        context, loomc_module_loom_module(scratch_module), workspace,
        &resolved_options, result, &target_preparation);
  }

  loom_kernel_launch_config_t kernel_config = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_launch_config_target_capture_t capture = {
        .result = result,
    };
    const iree_diagnostic_emitter_t emitter = {
        .fn = loomc_launch_config_capture_diagnostic,
        .user_data = &capture,
    };
    const loom_kernel_launch_config_options_t kernel_options =
        loomc_launch_config_make_kernel_options(
            &resolved_options, target_preparation.effective_target_facts,
            emitter);
    status = loomc_status_from_iree(loom_kernel_launch_config_evaluate(
        internal_module, loomc_workspace_block_pool(workspace), &kernel_options,
        &kernel_config));
    if (loomc_status_is_ok(status)) {
      loomc_launch_config_copy_from_kernel(&kernel_config, out_config);
    }
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_launch_config_report_failure(result, kernel_config.failure);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }

  loomc_result_release(result);
  loomc_launch_config_target_preparation_deinitialize(&target_preparation);
  loomc_module_release(scratch_module);
  return status;
}
