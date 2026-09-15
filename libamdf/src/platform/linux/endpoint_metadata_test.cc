// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <stdlib.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

// Sysfs is the dependency fixture; endpoint enumeration/open are the production
// implementation. There is deliberately no corresponding execution device
// node, so accidental native activation cannot satisfy these tests.
class LinuxEndpointMetadataTest
    : public ::testing::TestWithParam<amdf_engine_kind_t> {
 protected:
  void SetUp() override {
    std::string directory = ::testing::TempDir() + "/amdf-sysfs-XXXXXX";
    ASSERT_NE(mkdtemp(directory.data()), nullptr);
    directory_ = directory;
    const bool xdna = GetParam() == AMDF_ENGINE_KIND_XDNA;
    const std::string node_path =
        xdna ? "class/accel/accel987654321" : "class/drm/renderD987654321";
    const auto node = directory_ / node_path;
    const auto device = directory_ / "devices/0000:00:00.0";
    std::filesystem::create_directories(node);
    std::filesystem::create_directories(device);
    std::filesystem::create_directories(directory_ / "dev/char");
    std::filesystem::create_directory_symlink("../../../devices/0000:00:00.0",
                                              node / "device");
    std::filesystem::create_directory_symlink(
        xdna ? "../../bus/pci/drivers/amdxdna" : "../../bus/pci/drivers/amdgpu",
        device / "driver");
    std::filesystem::create_directory_symlink("../../" + node_path,
                                              directory_ / "dev/char/511:63");
    WriteAttribute(node / "dev", "511:63\n");
    WriteAttribute(device / "vendor", xdna ? "0x1022\n" : "0x1002\n");
    WriteAttribute(device / "device", "0xffff\n");
    WriteAttribute(device / "subsystem_vendor", "0x1022\n");
    WriteAttribute(device / "subsystem_device", "0xffff\n");
    WriteAttribute(device / "revision", "0x01\n");
    instance_.host_allocator = amdf_allocator_system();
    instance_.sysfs_descriptor =
        open(directory_.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
    ASSERT_GE(instance_.sysfs_descriptor, 0);
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_linux_file_close(&instance_.sysfs_descriptor),
              AMDF_STATUS_OK);
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }

  void WriteAttribute(const std::filesystem::path& path, const char* value) {
    std::ofstream stream(path);
    stream << value;
    stream.close();
    ASSERT_TRUE(stream.good()) << path;
  }

  // Temporary sysfs fixture owned through endpoint teardown.
  std::filesystem::path directory_;
  // Native instance state with the fixture as its sysfs dependency.
  amdf_platform_instance_t instance_ = {.sysfs_descriptor = -1};
  // Query endpoint, if a test has successfully opened it.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
};

TEST_P(LinuxEndpointMetadataTest, DiscoversWithoutAnExecutionNode) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(summary.engine_kind, GetParam());
  amdf_endpoint_info_t info;
  ASSERT_EQ(
      amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
      AMDF_STATUS_OK);
  EXPECT_TRUE(amdf_endpoint_id_is_equal(&summary.id, &info.id));
  EXPECT_STREQ(summary.name, info.name);
  EXPECT_EQ(amdf_platform_endpoint_query_queue_publication_modes(
                endpoint_, AMDF_QUEUE_COMMAND_TYPE_XDNA),
            GetParam() == AMDF_ENGINE_KIND_XDNA
                ? AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                : 0u);
  EXPECT_EQ(amdf_platform_endpoint_query_queue_publication_modes(
                endpoint_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4),
            0u);

  int descriptor = -1;
  amdf_linux_drm_version_t version = {UINT32_MAX, UINT32_MAX};
  EXPECT_EQ(amdf_linux_endpoint_open_file(endpoint_, &descriptor, &version),
            amdf_linux_error(ENOENT));
  EXPECT_EQ(descriptor, -1);
  EXPECT_EQ(version.major, UINT32_MAX);
  EXPECT_EQ(version.minor, UINT32_MAX);
}

TEST_P(LinuxEndpointMetadataTest, StaleIdentityDoesNotPublishMetadata) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  summary.id.words[1] ^= UINT64_C(1) << 63;
  amdf_endpoint_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_endpoint_info_t original = info;
  EXPECT_EQ(
      amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
      amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(endpoint_, nullptr);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST_P(LinuxEndpointMetadataTest, ActivationRevalidatesTheEndpointIdentity) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  amdf_endpoint_info_t info;
  ASSERT_EQ(
      amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
      AMDF_STATUS_OK);
  WriteAttribute(directory_ / "devices/0000:00:00.0/revision", "0x02\n");
  int descriptor = -1;
  amdf_linux_drm_version_t version = {UINT32_MAX, UINT32_MAX};
  EXPECT_EQ(amdf_linux_endpoint_open_file(endpoint_, &descriptor, &version),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(descriptor, -1);
  EXPECT_EQ(version.major, UINT32_MAX);
  EXPECT_EQ(version.minor, UINT32_MAX);
}

INSTANTIATE_TEST_SUITE_P(EngineKinds, LinuxEndpointMetadataTest,
                         ::testing::Values(AMDF_ENGINE_KIND_GPU,
                                           AMDF_ENGINE_KIND_XDNA));

}  // namespace
