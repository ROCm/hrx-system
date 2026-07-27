// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/requirements.h"

static bool loom_check_builtin_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement,
                                IREE_SV("loom-check-test-unavailable"));
}

static iree_status_t loom_check_builtin_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  (void)provider;
  (void)environment;
  (void)allocator;
  if (iree_string_view_equal(requirement,
                             IREE_SV("loom-check-test-unavailable"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "deterministic unavailable test requirement");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_builtin_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "loom-check-test-unavailable");
}

static const loom_check_requirement_provider_t*
loom_check_builtin_requirement_provider(void) {
  static const loom_check_requirement_provider_t provider = {
      .name = IREE_SVL("loom-check.builtin"),
      .match = loom_check_builtin_requirement_provider_matches,
      .query = loom_check_builtin_requirement_provider_query,
      .append_names = loom_check_builtin_requirement_provider_append_names,
  };
  return &provider;
}

static const loom_check_requirement_provider_t*
loom_check_lookup_requirement_provider(
    const loom_check_environment_t* environment,
    iree_string_view_t requirement) {
  const loom_check_requirement_provider_t* builtin_provider =
      loom_check_builtin_requirement_provider();
  if (builtin_provider->match(builtin_provider, requirement)) {
    return builtin_provider;
  }
  if (environment == NULL) {
    return NULL;
  }
  if (environment->requirement_providers.providers == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0;
       i < environment->requirement_providers.provider_count; ++i) {
    const loom_check_requirement_provider_t* provider =
        environment->requirement_providers.providers[i];
    if (provider == NULL || provider->match == NULL) {
      continue;
    }
    if (provider->match(provider, requirement)) {
      return provider;
    }
  }
  return NULL;
}

static bool loom_check_requirement_name_is_known(
    const loom_check_environment_t* environment,
    iree_string_view_t requirement) {
  return loom_check_lookup_requirement_provider(environment, requirement) !=
         NULL;
}

static iree_status_t loom_check_append_supported_requirement_names(
    const loom_check_environment_t* environment, loom_check_result_t* result) {
  const loom_check_requirement_provider_t* builtin_provider =
      loom_check_builtin_requirement_provider();
  IREE_RETURN_IF_ERROR(
      builtin_provider->append_names(builtin_provider, &result->detail));
  if (environment != NULL &&
      environment->requirement_providers.providers != NULL) {
    for (iree_host_size_t i = 0;
         i < environment->requirement_providers.provider_count; ++i) {
      const loom_check_requirement_provider_t* provider =
          environment->requirement_providers.providers[i];
      if (provider == NULL || provider->append_names == NULL) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(&result->detail, ", "));
      IREE_RETURN_IF_ERROR(provider->append_names(provider, &result->detail));
    }
  }
  return iree_string_builder_append_cstring(&result->detail, "\n");
}

static iree_status_t loom_check_fail_unknown_requirement(
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &result->detail,
      "unknown REQUIRES requirement '%.*s'; supported external requirements "
      "are ",
      (int)requirement.size, requirement.data));
  return loom_check_append_supported_requirement_names(environment, result);
}

static iree_status_t loom_check_skip_unavailable_requirement(
    iree_string_view_t requirement, iree_status_t availability_status,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_SKIP;
  result->final_outcome = LOOM_CHECK_SKIP;
  const char* status_name =
      iree_status_code_string(iree_status_code(availability_status));
  iree_status_t status = iree_string_builder_append_format(
      &result->detail, "skipped: requirement '%.*s' unavailable: %s",
      (int)requirement.size, requirement.data, status_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, ": ");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_status(&result->detail, availability_status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, "\n");
  }
  iree_status_free(availability_status);
  return status;
}

static iree_status_t loom_check_require_emit_tool_declarations(
    const loom_check_environment_t* environment,
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution) {
  if (test_case->mode != LOOM_CHECK_MODE_EMIT) {
    return iree_ok_status();
  }

  iree_string_view_t emit_target =
      iree_string_view_trim(test_case->emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, NULL);
  target_name = iree_string_view_trim(target_name);
  const loom_check_emit_provider_t* provider =
      loom_check_environment_lookup_emit_provider(environment, target_name);
  if (provider != NULL && provider->check_requirements != NULL) {
    return provider->check_requirements(provider, test_case, result,
                                        out_continue_execution);
  }

  return iree_ok_status();
}

iree_status_t loom_check_preflight_requirements(
    const loom_check_case_t* test_case,
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_continue_execution) {
  *out_continue_execution = true;
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    iree_string_view_t requirement = test_case->requirements[i];
    if (!loom_check_requirement_name_is_known(environment, requirement)) {
      *out_continue_execution = false;
      return loom_check_fail_unknown_requirement(environment, requirement,
                                                 result);
    }
  }

  IREE_RETURN_IF_ERROR(loom_check_require_emit_tool_declarations(
      environment, test_case, result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    iree_string_view_t requirement = test_case->requirements[i];
    const loom_check_requirement_provider_t* provider =
        loom_check_lookup_requirement_provider(environment, requirement);
    if (provider == NULL || provider->query == NULL) {
      *out_continue_execution = false;
      return loom_check_fail_unknown_requirement(environment, requirement,
                                                 result);
    }
    iree_status_t status =
        provider->query(provider, environment, requirement, allocator);
    if (!iree_status_is_ok(status)) {
      *out_continue_execution = false;
      return loom_check_skip_unavailable_requirement(requirement, status,
                                                     result);
    }
  }

  return iree_ok_status();
}
