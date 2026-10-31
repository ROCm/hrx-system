// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/profile.h"

namespace loom {
namespace {

TEST(WasmProviderTest, ComparesCompleteBaseProfile) {
  const loom_target_profile_t* selected = nullptr;
  IREE_ASSERT_OK(
      loom_wasm_target_provider.select_profile(IREE_SV("simd128"), &selected));
  ASSERT_NE(selected, nullptr);
  auto copy = *selected;
  EXPECT_TRUE(loom_target_profile_equal(selected, &copy));
  copy.explicit_fields ^= UINT64_C(1) << LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH;
  EXPECT_FALSE(loom_target_profile_equal(selected, &copy));
}

}  // namespace
}  // namespace loom
