// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/xdna/device_profile.h"
#include "libamdf/src/xdna/umd/mcdm/adapter_info.h"

namespace {

class WindowsXdnaAdapterInfoSystemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(
        amdf_platform_instance_create(amdf_allocator_system(), &instance_),
        AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (endpoint_) {
      ASSERT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    if (instance_) {
      EXPECT_EQ(amdf_platform_instance_destroy(instance_), AMDF_STATUS_OK);
    }
  }

  // Query-only instance; no logical device is created by this test.
  amdf_platform_instance_t* instance_ = nullptr;
  // Adapter retained through early assertion exits and closed before instance_.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
};

TEST_F(WindowsXdnaAdapterInfoSystemTest,
       QueriesNativeInterfaceBeforeDeviceCreation) {
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(instance_, 0, nullptr, &count),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries(count);
  ASSERT_EQ(amdf_platform_endpoint_enumerate(instance_, count, summaries.data(),
                                             &count),
            AMDF_STATUS_OK);
  bool found = false;
  for (const auto& summary : summaries) {
    if (summary.engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
    SCOPED_TRACE(summary.name);
    amdf_endpoint_info_t info = {};
    ASSERT_EQ(
        amdf_platform_endpoint_open(instance_, &summary.id, &endpoint_, &info),
        AMDF_STATUS_OK);
    amdf_xdna_device_info_t device_info = {};
    amdf_xdna_device_profile_t profile = {};
    if (amdf_xdna_device_profile_initialize(&info, &device_info, &profile) &&
        (profile.execution_capabilities &
         AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1) != 0) {
      found = true;
      // Interface discovery receives only a query procedure; it cannot depend
      // on device, context, paging or allocation creation.
      amdf_kmt_api_t query_api = {};
      query_api.query_adapter_info = instance_->kmt.query_adapter_info;
      amdf_windows_xdna_adapter_info_t adapter_info = {};
      ASSERT_EQ(amdf_windows_xdna_adapter_info_query(
                    &query_api, endpoint_->adapter, &adapter_info),
                AMDF_STATUS_OK);
      RecordProperty("shared_kernel_buffers",
                     adapter_info.shared_kernel_buffers ? 1 : 0);
      RecordProperty("native_protocol",
                     adapter_info.protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT
                         ? "direct"
                         : "metadata");
    }
    ASSERT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    endpoint_ = nullptr;
  }
  if (!found) {
    GTEST_SKIP() << "no XDNA endpoint with interpreter execution support";
  }
}

}  // namespace
