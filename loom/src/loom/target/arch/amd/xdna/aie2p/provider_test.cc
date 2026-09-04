// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(Aie2pProviderTest, RequiresLowCallsInline) {
  ASSERT_NE(loom_aie2p_target_provider.select_low_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_aie2p_target_provider.select_low_call_policy(&resolved_target),
            LOOM_TARGET_LOW_CALL_POLICY_REQUIRE_INLINE);
}

}  // namespace
}  // namespace loom
