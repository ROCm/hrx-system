// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/kernel/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;

TEST(KernelLaunchConfigTest, AcceptsAbsentOrCompleteClusterSize) {
  for (const uint16_t operand_count : {6, 9}) {
    loom_op_t op = {};
    op.operand_count = operand_count;
    DiagnosticEmissionCapture capture;
    IREE_EXPECT_OK(loom_kernel_launch_config_verify(
        /*module=*/nullptr, &op, capture.emitter()));
    EXPECT_TRUE(capture.emissions.empty());
  }
}

TEST(KernelLaunchConfigTest, RejectsPartialClusterSize) {
  for (const uint16_t operand_count : {7, 8}) {
    loom_op_t op = {};
    op.operand_count = operand_count;
    DiagnosticEmissionCapture capture;
    IREE_EXPECT_OK(loom_kernel_launch_config_verify(
        /*module=*/nullptr, &op, capture.emitter()));
    ASSERT_EQ(capture.emissions.size(), 1u);
    const auto& emission = capture.emissions.front();
    EXPECT_EQ(emission.error, LOOM_ERR_STRUCTURE_001);
    ASSERT_EQ(emission.u32_params.size(), 2u);
    EXPECT_EQ(emission.u32_params[0], operand_count);
    EXPECT_EQ(emission.u32_params[1], 9u);
  }
}

}  // namespace
}  // namespace loom
