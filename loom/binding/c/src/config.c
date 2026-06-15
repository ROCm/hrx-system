// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "config.h"

#include "diagnostic.h"
#include "loom/tooling/config/config.h"
#include "loomc/iree.h"

static loomc_status_t loomc_config_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static bool loomc_config_options_empty(const loomc_config_options_t* options) {
  return options == NULL || (options->binding_count == 0 &&
                             loomc_string_view_is_empty(options->json_object) &&
                             options->flags == 0);
}

loomc_status_t loomc_config_validate_options(
    const loomc_config_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->binding_count != 0 && options->bindings == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "config binding_count is non-zero but bindings is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->binding_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_config_validate_string_view(options->bindings[i].key));
    LOOMC_RETURN_IF_ERROR(
        loomc_config_validate_string_view(options->bindings[i].value));
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_config_validate_string_view(options->json_object));
  const loomc_config_policy_flags_t known_flags =
      LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  if ((options->flags & ~known_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "config options contain unknown flag bits");
  }
  return loomc_ok_status();
}

static iree_string_view_t loomc_config_normalize_key(loomc_string_view_t key) {
  iree_string_view_t normalized_key = iree_string_view_from_loomc(key);
  normalized_key = iree_string_view_trim(normalized_key);
  (void)iree_string_view_consume_prefix_char(&normalized_key, '@');
  return iree_string_view_trim(normalized_key);
}

static bool loomc_config_binding_overrides_json(
    const loomc_config_options_t* options,
    const loom_tooling_config_binding_t* json_binding) {
  for (loomc_host_size_t i = 0; i < options->binding_count; ++i) {
    iree_string_view_t binding_key =
        loomc_config_normalize_key(options->bindings[i].key);
    if (iree_string_view_equal(binding_key, json_binding->key)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loomc_config_populate_set(
    const loomc_config_options_t* options, iree_allocator_t host_allocator,
    loom_tooling_config_set_t* config_set) {
  loom_tooling_config_set_t json_config_set;
  loom_tooling_config_set_initialize(host_allocator, &json_config_set);

  iree_status_t status = iree_ok_status();
  if (!loomc_string_view_is_empty(options->json_object)) {
    status = loom_tooling_config_set_append_json_object(
        &json_config_set, iree_string_view_from_loomc(options->json_object));
  }
  for (iree_host_size_t i = 0;
       i < json_config_set.binding_count && iree_status_is_ok(status); ++i) {
    const loom_tooling_config_binding_t* binding = &json_config_set.bindings[i];
    if (loomc_config_binding_overrides_json(options, binding)) {
      continue;
    }
    status = loom_tooling_config_set_append(config_set, binding->key,
                                            binding->value);
  }
  for (loomc_host_size_t i = 0;
       i < options->binding_count && iree_status_is_ok(status); ++i) {
    status = loom_tooling_config_set_append(
        config_set, iree_string_view_from_loomc(options->bindings[i].key),
        iree_string_view_from_loomc(options->bindings[i].value));
  }

  loom_tooling_config_set_deinitialize(&json_config_set);
  return status;
}

static loom_tooling_config_materialize_flags_t loomc_config_materialize_flags(
    loomc_config_policy_flags_t flags) {
  loom_tooling_config_materialize_flags_t result = 0;
  if (iree_any_bit_set(flags, LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN)) {
    result |= LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES;
  }
  return result;
}

loomc_status_t loomc_config_apply_to_module(
    const loomc_config_apply_to_module_options_t* options) {
  if (options == NULL || options->module == NULL || options->result == NULL ||
      options->block_pool == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "config application requires options, module, result, and block_pool");
  }
  if (loomc_config_options_empty(options->config)) {
    return loomc_ok_status();
  }

  iree_allocator_t host_allocator =
      iree_allocator_from_loomc(options->allocator);
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(host_allocator, &config_set);

  loomc_status_t status = loomc_status_from_iree(
      loomc_config_populate_set(options->config, host_allocator, &config_set));
  loom_tooling_config_materialize_result_t materialize_result = {0};
  if (loomc_status_is_ok(status)) {
    loom_tooling_config_materialize_options_t materialize_options;
    loom_tooling_config_materialize_options_initialize(&materialize_options);
    materialize_options.flags =
        loomc_config_materialize_flags(options->config->flags);
    materialize_options.config_set = &config_set;
    status = loomc_status_from_iree(loom_tooling_config_materialize_module(
        options->module, &materialize_options, options->block_pool,
        &materialize_result));
  }
  if (loomc_status_is_ok(status) &&
      materialize_result.materialized_count != 0) {
    status = loomc_result_verify_loom_module(options->module, /*source=*/NULL,
                                             options->result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      iree_any_bit_set(options->config->flags,
                       LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED)) {
    status = loomc_status_from_iree(
        loom_tooling_config_require_resolved_module(options->module, NULL));
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    loomc_string_view_t diagnostic_code = options->diagnostic_code;
    if (loomc_string_view_is_empty(diagnostic_code)) {
      diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID");
    }
    status = loomc_result_fail_status_diagnostic_consume(
        options->result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, diagnostic_code,
        status);
  }

  loom_tooling_config_set_deinitialize(&config_set);
  return status;
}
