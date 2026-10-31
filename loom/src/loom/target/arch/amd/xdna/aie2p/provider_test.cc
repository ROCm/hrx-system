// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"

namespace loom {
namespace {

TEST(Aie2pProviderTest, ExactProfilesPreserveDeploymentAndExecutionKind) {
  const loom_aie2p_target_profile_t* halo = nullptr;
  const loom_aie2p_target_profile_t* strix = nullptr;
  const loom_aie2p_target_profile_t* core = nullptr;
  IREE_ASSERT_OK(loom_aie2p_target_profile_select(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), &halo));
  IREE_ASSERT_OK(loom_aie2p_target_profile_select(
      IREE_SV("amd.xdna.strix.17f0_10"), &strix));
  IREE_ASSERT_OK(loom_aie2p_target_profile_select(IREE_SV("core"), &core));
  auto copy = *halo;
  EXPECT_TRUE(loom_target_profile_equal(&halo->base, &copy.base));
  EXPECT_FALSE(loom_target_profile_equal(&halo->base, &strix->base));
  EXPECT_FALSE(loom_target_profile_equal(&halo->base, &core->base));
  copy.kind = core->kind;
  EXPECT_FALSE(loom_target_profile_equal(&halo->base, &copy.base));
}

TEST(Aie2pProviderTest, RejectsMissingCallee) {
  ASSERT_NE(loom_aie2p_target_provider.select_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_aie2p_target_provider.select_call_policy(
                &resolved_target, nullptr, LOOM_CALL_LIKE_KIND_LOW_INTERNAL,
                loom_call_like_t{}, loom_func_like_t{}),
            LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE);
}

}  // namespace
}  // namespace loom
