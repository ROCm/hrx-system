// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <string>

#include "api.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

TEST(HipInitializationApiTest, FailedInitializationPreservesCauseAndCanRetry) {
  hrx::hip::testing::HipDso dso;
  ASSERT_TRUE(dso.Open()) << dso.error();
  const auto initialize = dso.Resolve<decltype(&hipInit)>("hipInit");
  const auto deinitialize =
      dso.Resolve<decltype(&hipHALDeinit)>("hipHALDeinit");
  const auto get_device_count =
      dso.Resolve<decltype(&hipGetDeviceCount)>("hipGetDeviceCount");
  const auto get_last_error =
      dso.Resolve<decltype(&hipGetLastError)>("hipGetLastError");
  const auto peek_at_last_error =
      dso.Resolve<decltype(&hipPeekAtLastError)>("hipPeekAtLastError");
  const auto get_error_string =
      dso.Resolve<decltype(&hipGetErrorString)>("hipGetErrorString");
  const auto get_error_name =
      dso.Resolve<decltype(&hipGetErrorName)>("hipGetErrorName");
  const auto set_event_sink = dso.Resolve<decltype(&hipHRXSetDeviceEventSink)>(
      "hipHRXSetDeviceEventSink");
  ASSERT_NE(nullptr, set_event_sink);
  ASSERT_NE(nullptr, initialize);
  ASSERT_NE(nullptr, deinitialize);
  ASSERT_NE(nullptr, get_device_count);
  ASSERT_NE(nullptr, get_last_error);
  ASSERT_NE(nullptr, peek_at_last_error);
  ASSERT_NE(nullptr, get_error_string);
  ASSERT_NE(nullptr, get_error_name);

  EXPECT_EQ(hipErrorInvalidValue, set_event_sink({}));
  const hrx_device_event_sink_t sink = {
      .fn = [](void*, const hrx_device_event_t*) {},
  };
  ASSERT_EQ(hipSuccess, set_event_sink(sink));

  // An invalid backend reaches real runtime initialization without requiring a
  // GPU or a failure-injection hook. Diagnostics must not depend on debug logs.
  ASSERT_STREQ("0", std::getenv("HRX_GPU_DEBUG"));
  ASSERT_STREQ("invalid-initial-driver", std::getenv("HRX_GPU_DRIVER"));
  int device_count = -1;
  testing::internal::CaptureStderr();
  const hipError_t count_result = get_device_count(&device_count);
  const std::string count_diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(hipErrorNotInitialized, count_result);
  EXPECT_EQ(0, device_count);
  EXPECT_NE(
      std::string::npos,
      count_diagnostic.find("unknown HRX_GPU_DRIVER 'invalid-initial-driver'"));
  EXPECT_NE(std::string::npos,
            count_diagnostic.find("initializing HIP runtime"));
  EXPECT_EQ(hipErrorNotInitialized, peek_at_last_error());
  EXPECT_EQ(hipErrorNotInitialized, get_last_error());
  EXPECT_EQ(hipSuccess, peek_at_last_error());
  EXPECT_STREQ("initialization error", get_error_string(count_result));
  EXPECT_STREQ("hipErrorNotInitialized", get_error_name(count_result));

  // A failed construction releases its runtime and leaves sink selection open.
  EXPECT_EQ(hipSuccess, set_event_sink(sink));

  // This failure precedes native driver creation. Explicit initialization can
  // retry and reports the new attempt's cause, not stale diagnostic text.
  ASSERT_EQ(0, setenv("HRX_GPU_DRIVER", "invalid-retry-driver", 1));
  testing::internal::CaptureStderr();
  const hipError_t initialize_result = initialize(0);
  const std::string retry_diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(hipErrorNotInitialized, initialize_result);
  EXPECT_NE(
      std::string::npos,
      retry_diagnostic.find("unknown HRX_GPU_DRIVER 'invalid-retry-driver'"));
  EXPECT_NE(std::string::npos,
            retry_diagnostic.find("initializing HIP runtime"));
  EXPECT_EQ(std::string::npos, retry_diagnostic.find("invalid-initial-driver"));

  // Deliberately hiding all devices is a metadata-only query, not a failed
  // runtime construction. It remains quiet even after a failed attempt.
  ASSERT_EQ(0, setenv("ROCR_VISIBLE_DEVICES", "-1", 1));
  device_count = -1;
  testing::internal::CaptureStderr();
  const hipError_t hidden_result = get_device_count(&device_count);
  const std::string hidden_diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(hipErrorNoDevice, hidden_result);
  EXPECT_EQ(0, device_count);
  EXPECT_TRUE(hidden_diagnostic.empty()) << hidden_diagnostic;

  ASSERT_EQ(hipSuccess, deinitialize());
  EXPECT_TRUE(dso.Close()) << dso.error();
}

}  // namespace
