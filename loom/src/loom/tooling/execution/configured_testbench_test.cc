// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/configured_testbench.h"

#include "iree/testing/gtest.h"

#ifndef LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#define LOOM_CONFIG_EXECUTION_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#ifndef LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#define LOOM_CONFIG_EXECUTION_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

namespace loom {
namespace {

TEST(ConfiguredTestbenchTest, ExposesOnlyEnabledInitializers) {
  const loom_run_hal_testbench_requirement_initializer_set_t* initializer_set =
      loom_tooling_configured_testbench_requirement_initializers();
  ASSERT_NE(initializer_set, nullptr);
  EXPECT_EQ(initializer_set,
            loom_tooling_configured_testbench_requirement_initializers());
  const iree_host_size_t expected_initializer_count =
      LOOM_CONFIG_EXECUTION_HAVE_AMDGPU + 2 * LOOM_CONFIG_EXECUTION_HAVE_SPIRV;
  EXPECT_EQ(initializer_set->initializer_count, expected_initializer_count);
  EXPECT_EQ(initializer_set->initializers == nullptr,
            expected_initializer_count == 0);
  for (iree_host_size_t i = 0; i < initializer_set->initializer_count; ++i) {
    EXPECT_NE(initializer_set->initializers[i], nullptr);
  }
}

}  // namespace
}  // namespace loom
