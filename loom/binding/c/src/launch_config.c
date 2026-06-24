// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <stdint.h>

#include "config.h"
#include "diagnostic.h"
#include "iree/base/api.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
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
} loomc_launch_config_resolved_options_t;

static iree_status_t loomc_launch_config_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_launch_config_target_capture_t* capture =
      (loomc_launch_config_target_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
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

static bool loomc_launch_config_options_have_specialization(
    const loomc_launch_config_resolved_options_t* options) {
  const loomc_config_options_t* config = options->config;
  return options->workload_argument_count != 0 ||
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
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config eval option extensions are not supported");
  }
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
    iree_diagnostic_emitter_t emitter) {
  return (loom_kernel_launch_config_options_t){
      .function_symbol = options->function_symbol,
      .workload_arguments = options->workload_arguments,
      .workload_argument_count = options->workload_argument_count,
      .required_fields =
          loomc_launch_config_field_flags_to_kernel(options->required_fields),
      .diagnostic_emitter = emitter,
  };
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

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  const loom_module_t* source_internal_module =
      loomc_module_const_loom_module(module);
  if (source_internal_module != NULL &&
      !loomc_launch_config_options_have_specialization(&resolved_options)) {
    bool direct_evaluated = false;
    loom_kernel_launch_config_t kernel_config = {0};
    const loom_kernel_launch_config_options_t kernel_options =
        loomc_launch_config_make_kernel_options(&resolved_options,
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
  loomc_status_t status =
      loomc_module_clone(module, workspace, allocator, &scratch_module);
  loom_module_t* internal_module =
      loomc_status_is_ok(status) ? loomc_module_loom_module(scratch_module)
                                 : NULL;
  if (loomc_status_is_ok(status)) {
    const loomc_config_apply_to_module_options_t config_options = {
        .config = resolved_options.config,
        .module = internal_module,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = allocator,
    };
    status = loomc_config_apply_to_module(&config_options);
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
        loomc_launch_config_make_kernel_options(&resolved_options, emitter);
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
  loomc_module_release(scratch_module);
  return status;
}
