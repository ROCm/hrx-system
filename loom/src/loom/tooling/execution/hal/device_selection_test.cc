// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tooling/execution/hal/testbench_actual.h"

namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

TEST(HalDeviceSelectionTest, RejectsDeviceWithoutLinkedTargetAdapter) {
  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(
      /*target_environment=*/nullptr, /*target_routes=*/nullptr,
      /*target_route_count=*/0, iree_allocator_system(), &context);

  iree::Status status =
      loom_run_hal_testbench_context_validate_explicit_device(&context);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("--device=loom-unavailable"));
  EXPECT_THAT(status.ToString(), HasSubstr("no linked LoomC target adapter"));

  loom_run_hal_testbench_context_deinitialize(&context);
}

}  // namespace
