// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/error_state.h"
#include "binding/hip/status_conversion.h"
#include "iree/hal/drivers/amdgpu/feedback_state_test_util.h"
#include "iree/testing/gtest.h"

namespace {

class StatusConversionAmdgpuTest : public testing::Test {
 protected:
  void SetUp() override { iree_hip_error_state_reset(); }
  void TearDown() override { iree_hip_error_state_reset(); }
};

TEST_F(StatusConversionAmdgpuTest,
       TsanFailDeviceProducerConvertsAndPublishesAsNonfatal) {
  iree_status_t status =
      iree_hal_amdgpu_feedback_state_test_make_tsan_fail_device_status();
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, iree_status_code(status));

  const hipError_t converted_result = iree_status_to_hip_result(status);
  EXPECT_EQ(hipErrorLaunchFailure, converted_result);
  const iree_hip_error_state_token_t token = iree_hip_error_state_begin();
  EXPECT_EQ(hipErrorLaunchFailure,
            iree_hip_error_state_publish(token, converted_result));
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_begin().fatal_result);
  EXPECT_EQ(hipErrorLaunchFailure,
            iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipErrorLaunchFailure,
            iree_hip_error_state_get_and_clear_last_error());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
}

}  // namespace
