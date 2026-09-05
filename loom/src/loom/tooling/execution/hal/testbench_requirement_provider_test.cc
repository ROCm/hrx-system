// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_requirement_provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_status_t QuerySyntheticRequirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  (void)user_data;
  (void)module;
  (void)attrs;
  *out_result = (loom_testbench_requirement_provider_result_t){
      .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED,
  };
  return iree_ok_status();
}

static void InitializeSyntheticRequirementAlpha(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SVL("synthetic.alpha"),
      .user_data = context,
      .query = QuerySyntheticRequirement,
  };
}

static void InitializeSyntheticRequirementBeta(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider) {
  *out_provider = (loom_testbench_requirement_provider_t){
      .name = IREE_SVL("synthetic.beta"),
      .user_data = context,
      .query = QuerySyntheticRequirement,
  };
}

TEST(TestbenchRequirementProviderTest, PopulatesInInitializerOrder) {
  const loom_run_hal_testbench_requirement_provider_initializer_t
      initializers[] = {
          InitializeSyntheticRequirementAlpha,
          InitializeSyntheticRequirementBeta,
      };
  const loom_run_hal_testbench_requirement_initializer_set_t initializer_set = {
      .initializers = initializers,
      .initializer_count = IREE_ARRAYSIZE(initializers),
  };
  int synthetic_context_storage = 0;
  auto* context = reinterpret_cast<loom_run_hal_testbench_context_t*>(
      &synthetic_context_storage);
  loom_testbench_requirement_provider_t providers[2] = {};
  iree_host_size_t provider_count = 0;

  IREE_ASSERT_OK(loom_run_hal_testbench_requirement_providers_populate(
      &initializer_set, context, IREE_ARRAYSIZE(providers), providers,
      &provider_count));

  ASSERT_EQ(provider_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(providers[0].name, IREE_SV("synthetic.alpha")));
  EXPECT_TRUE(
      iree_string_view_equal(providers[1].name, IREE_SV("synthetic.beta")));
  EXPECT_EQ(providers[0].user_data, context);
  EXPECT_EQ(providers[1].user_data, context);
  EXPECT_EQ(providers[0].query, QuerySyntheticRequirement);
  EXPECT_EQ(providers[1].query, QuerySyntheticRequirement);
}

TEST(TestbenchRequirementProviderTest, RejectsInsufficientCapacityAtomically) {
  const loom_run_hal_testbench_requirement_provider_initializer_t
      initializers[] = {
          InitializeSyntheticRequirementAlpha,
          InitializeSyntheticRequirementBeta,
      };
  const loom_run_hal_testbench_requirement_initializer_set_t initializer_set = {
      .initializers = initializers,
      .initializer_count = IREE_ARRAYSIZE(initializers),
  };
  int synthetic_context_storage = 0;
  auto* context = reinterpret_cast<loom_run_hal_testbench_context_t*>(
      &synthetic_context_storage);
  loom_testbench_requirement_provider_t provider = {
      .name = IREE_SVL("synthetic.sentinel"),
  };
  iree_host_size_t provider_count = 99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_run_hal_testbench_requirement_providers_populate(
                            &initializer_set, context, /*provider_capacity=*/1,
                            &provider, &provider_count));
  EXPECT_EQ(provider_count, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(provider.name, IREE_SV("synthetic.sentinel")));
}

}  // namespace
}  // namespace loom
