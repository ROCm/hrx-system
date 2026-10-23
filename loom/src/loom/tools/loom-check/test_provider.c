// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only provider composition for checked-in .loom-test files.

#include "loom/tools/loom-check/test_provider.h"

#include "iree/base/api.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/test/config/providers.h"
#include "loom/tools/loom-check/test/low/providers.h"

static const loom_check_emit_provider_t* const kLoomCheckTestEmitProviders[] = {
    &loom_check_test_config_materialize_provider,
    &loom_check_test_config_schema_provider,
    &loom_check_test_low_allocation_provider,
    &loom_check_test_low_schedule_provider,
    &loom_check_test_low_synthetic_hazard_provider,
};

static bool loom_check_test_unavailable_requirement_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement,
                                IREE_SV("loom-check-test-unavailable"));
}

static iree_status_t loom_check_test_unavailable_requirement_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  (void)provider;
  (void)environment;
  (void)requirement;
  (void)allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "deterministic unavailable test requirement");
}

static iree_status_t loom_check_test_unavailable_requirement_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "loom-check-test-unavailable");
}

static const loom_check_requirement_provider_t
    kLoomCheckTestUnavailableRequirementProvider = {
        .name = IREE_SVL("test-unavailable"),
        .match = loom_check_test_unavailable_requirement_matches,
        .query = loom_check_test_unavailable_requirement_query,
        .append_names = loom_check_test_unavailable_requirement_append_names,
};

static const loom_check_requirement_provider_t* const
    kLoomCheckTestRequirementProviders[] = {
        &kLoomCheckTestUnavailableRequirementProvider,
};

const loom_check_provider_t loom_check_test_provider = {
    .name = IREE_SVL("test"),
    .emit_providers = kLoomCheckTestEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomCheckTestEmitProviders),
    .requirement_providers = kLoomCheckTestRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomCheckTestRequirementProviders),
};
