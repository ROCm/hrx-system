// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "config.h"

#include "diagnostic.h"
#include "loom/tooling/config/config.h"
#include "loomc/iree.h"

loomc_status_t loomc_config_validate_policy_flags(
    loomc_config_policy_flags_t flags) {
  const loomc_config_policy_flags_t known_flags =
      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  if ((flags & ~known_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "config policy contains unknown flag bits");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_config_apply_module(
    const loomc_config_apply_module_options_t* options,
    loomc_config_application_result_t* out_application_result) {
  *out_application_result = (loomc_config_application_result_t){0};

  loom_tooling_config_materialize_result_t materialize_result = {0};
  loomc_status_t status = loomc_ok_status();
  if (options->config_module != NULL) {
    status = loomc_result_verify_loom_module(options->config_module,
                                             /*source=*/NULL, options->result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      options->config_module != NULL) {
    status = loomc_status_from_iree(loom_tooling_config_overlay_module(
        options->target_module, options->config_module, options->block_pool,
        &materialize_result));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      materialize_result.materialized_count != 0) {
    status = loomc_result_verify_loom_module(options->target_module,
                                             /*source=*/NULL, options->result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      iree_any_bit_set(options->policy_flags,
                       LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED)) {
    status = loomc_status_from_iree(loom_tooling_config_require_resolved_module(
        options->target_module, /*out_result=*/NULL));
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    loomc_string_view_t diagnostic_code = options->diagnostic_code;
    if (loomc_string_view_is_empty(diagnostic_code)) {
      diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID");
    }
    status = loomc_result_fail_status_diagnostic_consume(
        options->result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        diagnostic_code, status);
  }
  if (loomc_status_is_ok(status)) {
    *out_application_result = (loomc_config_application_result_t){
        .materialized_count = materialize_result.materialized_count,
        .ignored_count = materialize_result.ignored_count,
    };
  }
  return status;
}
