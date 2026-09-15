// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/amdf_status.h"

#include <cerrno>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using testing::HasSubstr;

TEST(AmdfStatusTest, PreservesSuccess) {
  Status status(iree_hal_amd_status_from_amdf_status(
      __FILE__, __LINE__, AMDF_STATUS_OK, "operation"));
  EXPECT_TRUE(status.ok());
}

TEST(AmdfStatusTest, MapsPortableApiCodes) {
  struct TestCase {
    // Portable libamdf API status code.
    amdf_status_code_t amdf_code;
    // Canonical IREE status code.
    StatusCode iree_code;
  };
  constexpr TestCase kTestCases[] = {
      {AMDF_STATUS_CODE_INVALID_ARGUMENT, StatusCode::kInvalidArgument},
      {AMDF_STATUS_CODE_OUT_OF_RANGE, StatusCode::kOutOfRange},
      {AMDF_STATUS_CODE_BUFFER_TOO_SMALL, StatusCode::kOutOfRange},
      {AMDF_STATUS_CODE_UNSUPPORTED, StatusCode::kUnimplemented},
      {AMDF_STATUS_CODE_NOT_FOUND, StatusCode::kNotFound},
      {AMDF_STATUS_CODE_RESOURCE_EXHAUSTED, StatusCode::kResourceExhausted},
      {AMDF_STATUS_CODE_BUSY, StatusCode::kAborted},
      {AMDF_STATUS_CODE_DEADLINE_EXCEEDED, StatusCode::kDeadlineExceeded},
      {AMDF_STATUS_CODE_PERMISSION_DENIED, StatusCode::kPermissionDenied},
      {AMDF_STATUS_CODE_DEVICE_LOST, StatusCode::kDataLoss},
      {AMDF_STATUS_CODE_VERSION_MISMATCH, StatusCode::kIncompatible},
      {AMDF_STATUS_CODE_FAILED_PRECONDITION, StatusCode::kFailedPrecondition},
      {AMDF_STATUS_CODE_INTERNAL, StatusCode::kInternal},
  };
  for (const TestCase& test_case : kTestCases) {
    SCOPED_TRACE(test_case.amdf_code);
    Status status(iree_hal_amd_status_from_amdf_status(
        __FILE__, __LINE__, amdf_make_api_status(test_case.amdf_code),
        "operation"));
    EXPECT_EQ(status.code(), test_case.iree_code);
  }
}

TEST(AmdfStatusTest, MapsErrnoCodes) {
  Status status(iree_hal_amd_status_from_amdf_status(
      __FILE__, __LINE__, amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EACCES),
      "memory_map"));
  EXPECT_EQ(status.code(), StatusCode::kPermissionDenied);
}

TEST(AmdfStatusTest, PreservesRawDomainAndCodeInDiagnostics) {
  Status status(iree_hal_amd_status_from_amdf_status(
      __FILE__, __LINE__,
      amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, UINT32_C(0xA5)),
      "kernel_queue_submit"));
  EXPECT_EQ(status.code(), StatusCode::kUnknown);
  EXPECT_THAT(status.ToString(), HasSubstr("[kernel_queue_submit]"));
  EXPECT_THAT(status.ToString(), HasSubstr("FIRMWARE status 0x000000A5"));
}

}  // namespace
