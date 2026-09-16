// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/status_conversion.h"

#include "binding/hip/error_state.h"
#include "iree/testing/gtest.h"

namespace {

TEST(StatusConversionTest, DistinguishesDeviceFailureClasses) {
  EXPECT_EQ(hipErrorLaunchFailure,
            iree_status_to_hip_result(iree_make_status(
                IREE_STATUS_DATA_LOSS, "device execution failed")));
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_status_to_hip_result(iree_make_status(
                IREE_STATUS_ABORTED, "device memory access fault")));
  EXPECT_EQ(hipErrorUnknown, iree_status_to_hip_result(iree_make_status(
                                 IREE_STATUS_UNKNOWN, "unknown failure")));
}

TEST(StatusConversionTest, OnlyMemoryFaultCarrierLatchesFatalState) {
  iree_hip_error_state_reset();

  iree_hip_error_state_token_t token = iree_hip_error_state_begin();
  EXPECT_EQ(
      hipErrorIllegalAddress,
      iree_hip_error_state_publish(
          token, iree_status_to_hip_result(iree_make_status(
                     IREE_STATUS_ABORTED, "device memory access fault"))));
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_fatal_result());

  iree_hip_error_state_reset();
  token = iree_hip_error_state_begin();
  EXPECT_EQ(hipErrorLaunchFailure,
            iree_hip_error_state_publish(
                token, iree_status_to_hip_result(iree_make_status(
                           IREE_STATUS_DATA_LOSS, "hardware exception"))));
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_begin().fatal_result);

  iree_hip_error_state_reset();
  token = iree_hip_error_state_begin();
  EXPECT_EQ(hipErrorUnknown,
            iree_hip_error_state_publish(
                token, iree_status_to_hip_result(iree_make_status(
                           IREE_STATUS_UNKNOWN, "unknown failure"))));
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());

  iree_hip_error_state_reset();
}

TEST(StatusConversionTest, MapsValidationAndAvailability) {
  EXPECT_EQ(hipErrorInvalidValue,
            iree_status_to_hip_result(
                iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_status_to_hip_result(
                iree_status_from_code(IREE_STATUS_OUT_OF_RANGE)));
  EXPECT_EQ(hipErrorNotReady, iree_status_to_hip_result(iree_status_from_code(
                                  IREE_STATUS_UNAVAILABLE)));
  EXPECT_EQ(hipErrorNotSupported,
            iree_status_to_hip_result(
                iree_status_from_code(IREE_STATUS_UNIMPLEMENTED)));
}

}  // namespace
