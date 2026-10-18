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
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
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
  EXPECT_EQ(info.native_identity.type,
            AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE);
  EXPECT_EQ(info.native_identity.value.linux_device.major, 511u);
  EXPECT_EQ(info.native_identity.value.linux_device.minor, 63u);
  EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), UINT32_MAX);
  EXPECT_EQ(amdf_platform_endpoint_query_queue_publication_modes(
                endpoint_, AMDF_QUEUE_COMMAND_TYPE_XDNA),
            GetParam() == AMDF_ENGINE_KIND_XDNA
                ? AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                : 0u);
  EXPECT_EQ(amdf_platform_endpoint_query_queue_publication_modes(
                endpoint_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4),
            0u);

  int descriptor = -1;
  EXPECT_EQ(amdf_linux_endpoint_open_file(endpoint_, &descriptor),
            amdf_linux_error(ENOENT));
  EXPECT_EQ(descriptor, -1);
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
  EXPECT_EQ(amdf_linux_endpoint_open_file(endpoint_, &descriptor),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(descriptor, -1);
}

TEST_P(LinuxEndpointMetadataTest,
       CapturesCpuNodeZeroAndNonzeroWithoutActivation) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  for (const uint32_t node : {0u, 7u}) {
    SCOPED_TRACE(node);
    const auto cpu_node =
        directory_ / "devices/system/node" / ("node" + std::to_string(node));
    std::filesystem::create_directories(cpu_node);
    WriteAttribute(cpu_node / "cpulist", "96-191\n");
    WriteAttribute(directory_ / "devices/0000:00:00.0/numa_node",
                   (std::to_string(node) + "\n").c_str());
    amdf_endpoint_info_t info;
    ASSERT_EQ(
        amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
        AMDF_STATUS_OK);
    EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), node);
    WriteAttribute(directory_ / "devices/0000:00:00.0/numa_node", "-1\n");
    EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), node);
    ASSERT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    endpoint_ = nullptr;
  }
}

TEST_P(LinuxEndpointMetadataTest, ExcludesUnknownMissingAndMemoryOnlyNodes) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  const auto cpu_node = directory_ / "devices/system/node/node4";
  std::filesystem::create_directories(cpu_node);
  WriteAttribute(cpu_node / "cpulist", "\n");
  for (const char* node : {"-1\n", "4\n", "5\n"}) {
    SCOPED_TRACE(node);
    WriteAttribute(directory_ / "devices/0000:00:00.0/numa_node", node);
    amdf_endpoint_info_t info;
    ASSERT_EQ(
        amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
        AMDF_STATUS_OK);
    EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), UINT32_MAX);
    ASSERT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    endpoint_ = nullptr;
  }
}

TEST_P(LinuxEndpointMetadataTest, ReadsTheKernelCpuListWithoutLosingItsPrefix) {
  // Real sysfs attributes have different short-read behavior from regular
  // fixture files. Keep endpoint identity synthetic, but exercise the kernel's
  // CPU-list formatter through the production reader.
  const std::filesystem::path nodes = "/sys/devices/system/node";
  if (!std::filesystem::exists(nodes)) {
    GTEST_SKIP() << "kernel does not expose NUMA node attributes";
  }
  std::filesystem::path online_cpu_list;
  for (const auto& entry : std::filesystem::directory_iterator(nodes)) {
    const auto candidate = entry.path() / "cpulist";
    std::ifstream stream(candidate);
    std::string cpus;
    if (std::getline(stream, cpus) && !cpus.empty() && cpus[0] >= '0' &&
        cpus[0] <= '9') {
      online_cpu_list = candidate;
      break;
    }
  }
  ASSERT_FALSE(online_cpu_list.empty());
  const auto fixture_node = directory_ / "devices/system/node/node7";
  std::filesystem::create_directories(fixture_node);
  std::filesystem::create_symlink(online_cpu_list, fixture_node / "cpulist");
  WriteAttribute(directory_ / "devices/0000:00:00.0/numa_node", "7\n");
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  amdf_endpoint_info_t info;
  ASSERT_EQ(
      amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
      AMDF_STATUS_OK);
  EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), 7u);
}

TEST_P(LinuxEndpointMetadataTest, RejectsMalformedLocalityWithoutPublishing) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  const auto cpu_node = directory_ / "devices/system/node/node0";
  std::filesystem::create_directories(cpu_node);
  WriteAttribute(cpu_node / "cpulist", "invalid\n");
  for (const char* node :
       {"", "-2\n", "4294967295\n", "0junk\n", "0\nignored", "0\n"}) {
    SCOPED_TRACE(node);
    WriteAttribute(directory_ / "devices/0000:00:00.0/numa_node", node);
    amdf_endpoint_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const auto original = info;
    EXPECT_EQ(
        amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
        amdf_linux_error(EPROTO));
    EXPECT_EQ(endpoint_, nullptr);
    EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(EngineKinds, LinuxEndpointMetadataTest,
                         ::testing::Values(AMDF_ENGINE_KIND_GPU,
                                           AMDF_ENGINE_KIND_XDNA));

}  // namespace
