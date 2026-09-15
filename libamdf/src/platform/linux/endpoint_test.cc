// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/endpoint.h"

#include <drm/drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

class LinuxEndpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_platform_instance_create(amdf_allocator_system(), &instance),
              AMDF_STATUS_OK);
  }

  void TearDown() override {
    EXPECT_EQ(amdf_linux_file_close(&first), AMDF_STATUS_OK);
    EXPECT_EQ(amdf_linux_file_close(&second), AMDF_STATUS_OK);
    if (endpoint) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint), AMDF_STATUS_OK);
    }
    if (instance) {
      EXPECT_EQ(amdf_platform_instance_destroy(instance), AMDF_STATUS_OK);
    }
  }

  // Instance owned until the test has released all native files.
  amdf_platform_instance_t* instance = nullptr;
  // Query endpoint retained across assertions that may stop the test.
  amdf_platform_endpoint_t* endpoint = nullptr;
  // First independent native file.
  int first = -1;
  // Second independent native file.
  int second = -1;
};

TEST_F(LinuxEndpointTest, NativeIdentityAndIndependentFiles) {
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(instance, 0, nullptr, &count),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries(count);
  ASSERT_EQ(amdf_platform_endpoint_enumerate(instance, count, summaries.data(),
                                             &count),
            AMDF_STATUS_OK);
  for (const auto& summary : summaries) {
    SCOPED_TRACE(summary.name);
    std::cout << "Endpoint: " << summary.name << std::endl;
    amdf_endpoint_info_t info;
    ASSERT_EQ(
        amdf_platform_endpoint_open(instance, &summary.id, &endpoint, &info),
        AMDF_STATUS_OK);
    EXPECT_TRUE(amdf_endpoint_id_is_equal(&summary.id, &info.id));
    EXPECT_EQ(summary.engine_kind, info.engine_kind);
    EXPECT_STREQ(summary.name, info.name);
    amdf_linux_drm_version_t first_version = {};
    amdf_linux_drm_version_t second_version = {};
    ASSERT_EQ(amdf_linux_endpoint_open_file(endpoint, &first, &first_version),
              AMDF_STATUS_OK);
    ASSERT_EQ(amdf_linux_endpoint_open_file(endpoint, &second, &second_version),
              AMDF_STATUS_OK);
    EXPECT_EQ(first_version.major, second_version.major);
    EXPECT_EQ(first_version.minor, second_version.minor);
    struct drm_version native_version = {};
    ASSERT_EQ(ioctl(first, DRM_IOCTL_VERSION, &native_version), 0);
    EXPECT_EQ(first_version.major,
              static_cast<uint32_t>(native_version.version_major));
    EXPECT_EQ(first_version.minor,
              static_cast<uint32_t>(native_version.version_minor));
    EXPECT_NE(first, second);
    EXPECT_NE(fcntl(first, F_GETFD) & FD_CLOEXEC, 0);
    EXPECT_NE(fcntl(second, F_GETFD) & FD_CLOEXEC, 0);
    // File status flags belong to an open file description. Changing one must
    // not affect the other, as it would if device creation used dup().
    int second_flags = fcntl(second, F_GETFL);
    ASSERT_GE(second_flags, 0);
    ASSERT_EQ(fcntl(first, F_SETFL, second_flags ^ O_NONBLOCK), 0);
    EXPECT_EQ(fcntl(second, F_GETFL), second_flags);
    EXPECT_EQ(amdf_platform_endpoint_close(endpoint), AMDF_STATUS_OK);
    endpoint = nullptr;
    EXPECT_GE(fcntl(first, F_GETFD), 0);
    EXPECT_EQ(amdf_linux_file_close(&first), AMDF_STATUS_OK);
    EXPECT_EQ(amdf_linux_file_close(&second), AMDF_STATUS_OK);

    amdf_endpoint_id_t stale = summary.id;
    stale.words[1] ^= UINT64_C(1) << 63;
    amdf_platform_endpoint_t* failed_endpoint =
        reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
    EXPECT_EQ(
        amdf_platform_endpoint_open(instance, &stale, &failed_endpoint, &info),
        amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(failed_endpoint), uintptr_t{1});
  }
}

}  // namespace
