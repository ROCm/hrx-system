// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/abi/task/state_layout.h"

#include "iree/hal/driver/task/executable/library/abi.h"
#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(HalBuiltinTest, StateFieldsMatchTheVersionedSchema) {
  const size_t offsets[] = {
      offsetof(iree_hal_executable_workgroup_state_v0_t, workgroup_id_x),
      offsetof(iree_hal_executable_workgroup_state_v0_t, workgroup_id_y),
      offsetof(iree_hal_executable_workgroup_state_v0_t, workgroup_id_z),
      offsetof(iree_hal_executable_dispatch_state_v0_t, workgroup_count_x),
      offsetof(iree_hal_executable_dispatch_state_v0_t, workgroup_count_y),
      offsetof(iree_hal_executable_dispatch_state_v0_t, workgroup_count_z),
  };
  ASSERT_EQ(IREE_ARRAYSIZE(offsets), LOOM_TASK_BUILTIN_COUNT_);
  for (size_t i = 0; i < IREE_ARRAYSIZE(offsets); ++i) {
    EXPECT_EQ(loom_task_builtins[i].offset, offsets[i]);
    EXPECT_EQ(loom_task_builtins[i].size, i % 3 == 2 ? 2u : 4u);
    EXPECT_EQ(loom_task_builtins[i].state_argument, i < 3 ? 2u : 1u);
  }
}

}  // namespace
}  // namespace loom
