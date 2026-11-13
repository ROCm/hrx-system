// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string_view>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/profile.h"
#include "loomc/target.h"
#include "loomc/target/configured_target.h"
#include "target.h"
#include "test/util.h"

namespace {

using TargetEnvironmentPtr =
    loomc::testing::HandlePtr<loomc_target_environment_t,
                              loomc_target_environment_release>;
using TargetProfilePtr =
    loomc::testing::HandlePtr<loomc_target_profile_t,
                              loomc_target_profile_release>;

class TargetProfileRefinementTest
    : public ::testing::TestWithParam<const char*> {};

TEST_P(TargetProfileRefinementTest, RetainsFamilyAndSeparatesDeploymentFacts) {
  loomc_target_environment_t* raw_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_configured_target(
      loomc_allocator_system(), &raw_environment));
  TargetEnvironmentPtr environment(raw_environment);
  const std::string_view specification = GetParam();
  const auto family = specification.substr(0, specification.find(':'));
  bool configured = false;
  for (loomc_host_size_t i = 0;
       i < loomc_target_environment_target_family_count(environment.get());
       ++i) {
    const auto name =
        loomc_target_environment_target_family_at(environment.get(), i);
    configured |= std::string_view(name.data, name.size) == family;
  }
  if (!configured) {
    GTEST_SKIP() << "Target family is not configured";
  }

  loomc_target_profile_t* raw_base = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      environment.get(), loomc_make_cstring_view(GetParam()),
      loomc_allocator_system(), &raw_base));
  TargetProfilePtr base(raw_base);
  loomc_target_profile_t* raw_refined = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_refine_execution_unit_count(
      base.get(), 13, loomc_allocator_system(), &raw_refined));
  TargetProfilePtr refined(raw_refined);
  EXPECT_FALSE(loomc_target_profile_equal(base.get(), refined.get()));
  const auto* original = loomc_target_profile_loom_target_profile(base.get());
  const auto* updated = loomc_target_profile_loom_target_profile(refined.get());
  EXPECT_EQ(original->target_bundle->snapshot->execution_unit_count, 0u);
  EXPECT_EQ(updated->target_bundle->snapshot->execution_unit_count, 13u);
  EXPECT_EQ(original->type, updated->type);

  loomc_target_profile_t* raw_same = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_refine_execution_unit_count(
      base.get(), 13, loomc_allocator_system(), &raw_same));
  TargetProfilePtr same(raw_same);
  EXPECT_TRUE(loomc_target_profile_equal(refined.get(), same.get()));
  loomc_target_profile_t* raw_other = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_refine_execution_unit_count(
      base.get(), 7, loomc_allocator_system(), &raw_other));
  TargetProfilePtr other(raw_other);
  EXPECT_FALSE(loomc_target_profile_equal(refined.get(), other.get()));

  loomc_target_profile_t* repeated = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_refine_execution_unit_count(
      refined.get(), 13, loomc_allocator_system(), &repeated));
  EXPECT_EQ(repeated, refined.get());
  loomc_target_profile_release(repeated);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_refine_execution_unit_count(
          refined.get(), 7, loomc_allocator_system(), &repeated));
  EXPECT_EQ(repeated, nullptr);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_refine_execution_unit_count(
          base.get(), 0, loomc_allocator_system(), &repeated));

  base.reset();
  environment.reset();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_facts_t* facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(updated, &arena, &facts));
  EXPECT_EQ(facts->storage.snapshot.execution_unit_count, 13u);
  EXPECT_TRUE(loom_target_fact_field_set_contains(
      facts->explicit_fields, LOOM_TARGET_FACT_FIELD_EXECUTION_UNIT_COUNT));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

INSTANTIATE_TEST_SUITE_P(ConfiguredFamilies, TargetProfileRefinementTest,
                         ::testing::Values("vm:core", "wasm:simd128",
                                           "amdgpu:gfx1100",
                                           "spirv:vulkan1.3+bda", "x86:avx2",
                                           "amd.xdna.aie2p:core"));

}  // namespace
