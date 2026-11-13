// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/cli/loomc_options.h"

#include "loomc/iree.h"

IREE_STATIC_ASSERT_ENUM_EQ(LOOM_SANITIZER_REPORTING_MODE_DEFAULT,
                           LOOMC_SANITIZER_REPORTING_MODE_DEFAULT,
                           "default sanitizer reporting modes must match");
IREE_STATIC_ASSERT_ENUM_EQ(LOOM_SANITIZER_REPORTING_MODE_TRAP,
                           LOOMC_SANITIZER_REPORTING_MODE_TRAP,
                           "trap sanitizer reporting modes must match");
IREE_STATIC_ASSERT_ENUM_EQ(LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY,
                           LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY,
                           "report-only sanitizer reporting modes must match");

iree_status_t loom_tooling_cli_make_loomc_config_options(
    const loom_config_text_binding_set_t* text_set, iree_allocator_t allocator,
    loomc_config_binding_t** out_bindings,
    loomc_config_options_t* out_options) {
  *out_bindings = NULL;
  *out_options = (loomc_config_options_t){
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  if (text_set->binding_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, text_set->binding_count, sizeof(**out_bindings),
      (void**)out_bindings));
  for (iree_host_size_t i = 0; i < text_set->binding_count; ++i) {
    (*out_bindings)[i] = (loomc_config_binding_t){
        .key = loomc_string_view_from_iree(text_set->bindings[i].key),
        .value = loomc_string_view_from_iree(text_set->bindings[i].value),
    };
  }
  out_options->bindings = *out_bindings;
  out_options->binding_count = text_set->binding_count;
  return iree_ok_status();
}

void loom_tooling_cli_make_loomc_sanitizer_options(
    const loom_sanitizer_options_t* options,
    loomc_sanitizer_options_t* out_options) {
  *out_options = (loomc_sanitizer_options_t){
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(*out_options),
      .checks = options->checks,
      .flags = options->flags,
      .reporting_mode =
          (loomc_sanitizer_reporting_mode_t)options->reporting_mode,
  };
}

bool loom_tooling_cli_pipeline_uses_default(iree_string_view_t pipeline) {
  pipeline = iree_string_view_trim(pipeline);
  return iree_string_view_is_empty(pipeline) ||
         iree_string_view_equal(pipeline, IREE_SV("default"));
}

iree_status_t loom_tooling_cli_prepare_loomc_pass_program(
    loomc_context_t* context, const loomc_module_t* module,
    iree_string_view_t pipeline, iree_string_view_t identifier,
    iree_allocator_t host_allocator, loomc_pass_program_t** out_pass_program,
    loomc_result_t** out_result) {
  *out_pass_program = NULL;
  *out_result = NULL;
  pipeline = iree_string_view_trim(pipeline);
  if (loom_tooling_cli_pipeline_uses_default(pipeline)) {
    return iree_ok_status();
  }
  const loomc_pass_program_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      .structure_size = sizeof(options),
      .identifier = loomc_string_view_from_iree(identifier),
  };
  if (iree_string_view_equal(pipeline, IREE_SV("none"))) {
    return iree_status_from_loomc(loomc_pass_program_create_empty(
        context, &options, loomc_allocator_from_iree(host_allocator),
        out_pass_program));
  }
  if (pipeline.data[0] == '@') {
    return iree_status_from_loomc(loomc_pass_program_create_from_module_symbol(
        module, loomc_string_view_from_iree(pipeline), &options,
        loomc_allocator_from_iree(host_allocator), out_pass_program,
        out_result));
  }
  return iree_status_from_loomc(loomc_pass_program_create_from_pipeline_text(
      context, loomc_string_view_from_iree(pipeline), &options,
      loomc_allocator_from_iree(host_allocator), out_pass_program, out_result));
}
