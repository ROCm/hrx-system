// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <stdlib.h>
#include <sys/syscall.h>

#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

struct DeviceOpenState {
  // Synthetic device node intercepted by the open dependency.
  const char* path = nullptr;
  // Interrupted attempts remaining before the terminal dependency result.
  uint32_t interruptions = 0;
  // Number of attempts at opening the synthetic device node.
  uint32_t attempts = 0;
  // Terminal error, or zero to use the real nonexistent-node result.
  int error = 0;
};

// Borrowed only during the synchronous endpoint activation under test.
DeviceOpenState* device_open = nullptr;

struct DeviceAccessState {
  // Synthetic device node whose native permission query is intercepted.
  const char* path = nullptr;
  // Interrupted attempts remaining before the terminal dependency result.
  uint32_t interruptions = 0;
  // Number of permission queries for this node.
  uint32_t attempts = 0;
  // Native permission-query error, or zero for allowed access.
  int error = 0;
};

// Borrowed only during synchronous enumeration under test.
DeviceAccessState* device_access = nullptr;

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
    AddEndpoint(987654321, 63, "0000:00:00.0");
    instance_.host_allocator = amdf_allocator_system();
    instance_.kind = AMDF_LINUX_PLATFORM_KIND_NATIVE_DEVICE;
    ASSERT_EQ(pthread_mutex_init(&instance_.native_mutex, nullptr), 0);
    instance_.sysfs_descriptor =
        open(directory_.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
    ASSERT_GE(instance_.sysfs_descriptor, 0);
  }

  void AddEndpoint(uint32_t ordinal, uint32_t minor, const char* pci_address) {
    const bool xdna = GetParam() == AMDF_ENGINE_KIND_XDNA;
    const std::string node_path =
        std::string(xdna ? "class/accel/accel" : "class/drm/renderD") +
        std::to_string(ordinal);
    const auto node = directory_ / node_path;
    const auto device = directory_ / "devices" / pci_address;
    std::filesystem::create_directories(node);
    std::filesystem::create_directories(device);
    std::filesystem::create_directories(directory_ / "dev/char");
    std::filesystem::create_directory_symlink(
        std::string("../../../devices/") + pci_address, node / "device");
    std::filesystem::create_directory_symlink(
        xdna ? "../../bus/pci/drivers/amdxdna" : "../../bus/pci/drivers/amdgpu",
        device / "driver");
    std::filesystem::create_directory_symlink(
        "../../" + node_path,
        directory_ / "dev/char" / ("511:" + std::to_string(minor)));
    WriteAttribute(node / "dev", "511:" + std::to_string(minor) + "\n");
    WriteAttribute(device / "vendor", xdna ? "0x1022\n" : "0x1002\n");
    WriteAttribute(device / "device", "0xffff\n");
    WriteAttribute(device / "subsystem_vendor", "0x1022\n");
    WriteAttribute(device / "subsystem_device", "0xffff\n");
    WriteAttribute(device / "revision", "0x01\n");
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_linux_file_close(&instance_.sysfs_descriptor),
              AMDF_STATUS_OK);
    EXPECT_EQ(pthread_mutex_destroy(&instance_.native_mutex), 0);
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  void WriteAttribute(const std::filesystem::path& path,
                      const std::string& value) {
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
  EXPECT_STREQ(info.name, GetParam() == AMDF_ENGINE_KIND_XDNA
                              ? "AMD XDNA 1022:ffff:01 (0000:00:00.0)"
                              : "AMD GPU 1002:ffff:01 (0000:00:00.0)");
  EXPECT_EQ(info.native_identity.type,
            AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE);
  EXPECT_EQ(info.native_identity.value.linux_device.major, 511u);
  EXPECT_EQ(info.native_identity.value.linux_device.minor, 63u);
  EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), UINT32_MAX);
  const auto* address = amdf_platform_endpoint_pci_address(endpoint_);
  ASSERT_NE(address, nullptr);
  EXPECT_EQ(address->available, 1u);
  EXPECT_EQ(address->domain, 0u);
  EXPECT_EQ(address->bus, 0u);
  EXPECT_EQ(address->device, 0u);
  EXPECT_EQ(address->function, 0u);
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

TEST_P(LinuxEndpointMetadataTest,
       ProductNamesPreserveDistinctNativeIdentities) {
  AddEndpoint(987654322, 64, "0000:00:00.1");
  for (const char* address : {"0000:00:00.0", "0000:00:00.1"}) {
    const auto device = directory_ / "devices" / address;
    WriteAttribute(device / "device", "0x1586\n");
    WriteAttribute(device / "revision", "0xc1\n");
  }
  amdf_endpoint_summary_t summaries[2] = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 2, summaries, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 2u);
  EXPECT_FALSE(amdf_endpoint_id_is_equal(&summaries[0].id, &summaries[1].id));
  for (const auto& summary : summaries) {
    amdf_endpoint_info_t info;
    ASSERT_EQ(
        amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
        AMDF_STATUS_OK);
    EXPECT_STREQ(summary.name, info.name);
    EXPECT_TRUE(amdf_endpoint_id_is_equal(&summary.id, &info.id));
    EXPECT_EQ(info.pci.device_id, 0x1586u);
    EXPECT_EQ(info.pci.revision_id, 0xC1u);
    if (GetParam() == AMDF_ENGINE_KIND_GPU) {
      EXPECT_STREQ(info.name, "AMD Radeon 8060S Graphics");
    } else {
      EXPECT_EQ(std::string(info.name).find("AMD XDNA 1022:1586:c1 ("), 0u);
    }
    EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    endpoint_ = nullptr;
  }
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

TEST_P(LinuxEndpointMetadataTest, CountsOnlyAccessibleEndpointsWithoutOpening) {
  AddEndpoint(987654322, 64, "0000:00:00.1");
  AddEndpoint(987654323, 65, "0000:00:00.2");
  const char* path = GetParam() == AMDF_ENGINE_KIND_XDNA
                         ? "/dev/accel/accel987654321"
                         : "/dev/dri/renderD987654321";
  for (const int error : {EPERM, EACCES}) {
    SCOPED_TRACE(error);
    DeviceAccessState access = {.path = path, .error = error};
    DeviceOpenState opened = {.path = path, .error = error};
    device_access = &access;
    device_open = &opened;
    uint32_t count = 0;
    const auto count_status =
        amdf_platform_endpoint_enumerate(&instance_, 0, nullptr, &count);
    amdf_endpoint_summary_t summaries[2] = {};
    uint32_t short_count = 0;
    const auto short_status = amdf_platform_endpoint_enumerate(
        &instance_, 1, summaries, &short_count);
    uint32_t full_count = 0;
    const auto full_status =
        amdf_platform_endpoint_enumerate(&instance_, 2, summaries, &full_count);
    device_access = nullptr;
    device_open = nullptr;
    EXPECT_EQ(count_status, AMDF_STATUS_OK);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(short_status,
              amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    EXPECT_EQ(short_count, 2u);
    ASSERT_EQ(full_status, AMDF_STATUS_OK);
    ASSERT_EQ(full_count, 2u);
    EXPECT_NE(summaries[0].id.words[0], summaries[1].id.words[0]);
    for (const auto& summary : summaries) {
      EXPECT_NE(static_cast<uint32_t>(summary.id.words[0]), 63u);
      EXPECT_EQ(summary.engine_kind, GetParam());
    }
    EXPECT_GT(access.attempts, 0u);
    EXPECT_EQ(opened.attempts, 0u);
  }
}

TEST_P(LinuxEndpointMetadataTest,
       PermissionQueriesDistinguishDenialFromErrors) {
  const char* path = GetParam() == AMDF_ENGINE_KIND_XDNA
                         ? "/dev/accel/accel987654321"
                         : "/dev/dri/renderD987654321";
  for (const int error : {0, EPERM, EACCES, ENOENT, EIO}) {
    SCOPED_TRACE(error);
    DeviceAccessState access = {
        .path = path,
        .interruptions = 1,
        .error = error,
    };
    device_access = &access;
    amdf_endpoint_summary_t summary;
    std::memset(&summary, 0xA5, sizeof(summary));
    const auto original = summary;
    uint32_t count = UINT32_MAX;
    const auto status =
        amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count);
    device_access = nullptr;
    ASSERT_EQ(status, AMDF_STATUS_OK);
    const bool denied = error == EPERM || error == EACCES;
    EXPECT_EQ(count, denied ? 0u : 1u);
    if (denied) {
      EXPECT_EQ(std::memcmp(&summary, &original, sizeof(summary)), 0);
    }
    EXPECT_EQ(access.interruptions, 0u);
    EXPECT_GT(access.attempts, 1u);
  }
}

TEST_P(LinuxEndpointMetadataTest, InterruptedOpenPreservesTerminalFailure) {
  amdf_endpoint_summary_t summary = {};
  uint32_t count = 0;
  ASSERT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            AMDF_STATUS_OK);
  ASSERT_EQ(count, 1u);
  amdf_endpoint_info_t info;
  ASSERT_EQ(
      amdf_platform_endpoint_open(&instance_, &summary.id, &endpoint_, &info),
      AMDF_STATUS_OK);
  for (const int terminal_error : {0, EACCES}) {
    DeviceOpenState state;
    state.path = GetParam() == AMDF_ENGINE_KIND_XDNA
                     ? "/dev/accel/accel987654321"
                     : "/dev/dri/renderD987654321";
    state.interruptions = 2;
    state.error = terminal_error;
    device_open = &state;
    int descriptor = 73;
    const amdf_status_t status =
        amdf_linux_endpoint_open_file(endpoint_, &descriptor);
    device_open = nullptr;
    EXPECT_EQ(status,
              amdf_linux_error(terminal_error ? terminal_error : ENOENT));
    EXPECT_EQ(state.attempts, 3u);
    EXPECT_EQ(state.interruptions, 0u);
    EXPECT_EQ(descriptor, 73);
  }
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

extern "C" int __real_open(const char* path, int flags, ...);
extern "C" long __real_syscall(long number, ...);

extern "C" long __wrap_syscall(long number, ...) {
  // Only this native syscall is used by endpoint enumeration. Do not forward
  // an unknown variadic signature by guessing its argument count or types.
  if (number != SYS_faccessat2) {
    ADD_FAILURE() << "unexpected syscall " << number;
    errno = ENOSYS;
    return -1;
  }
  va_list arguments;
  va_start(arguments, number);
  const int directory = va_arg(arguments, int);
  const char* path = va_arg(arguments, const char*);
  const int mode = va_arg(arguments, int);
  const int flags = va_arg(arguments, int);
  va_end(arguments);
  if (device_access && std::strcmp(path, device_access->path) == 0) {
    EXPECT_EQ(directory, AT_FDCWD);
    EXPECT_EQ(mode, R_OK | W_OK);
    EXPECT_EQ(flags, AT_EACCESS);
    ++device_access->attempts;
    if (device_access->interruptions != 0) {
      --device_access->interruptions;
      errno = EINTR;
      return -1;
    }
    errno = device_access->error;
    return device_access->error ? -1 : 0;
  }
  return __real_syscall(number, directory, path, mode, flags);
}

extern "C" int __wrap_open(const char* path, int flags, ...) {
  if (device_open && std::strcmp(path, device_open->path) == 0) {
    EXPECT_EQ(flags, O_RDWR | O_CLOEXEC);
    ++device_open->attempts;
    if (device_open->interruptions) {
      --device_open->interruptions;
      errno = EINTR;
      return -1;
    }
    if (device_open->error) {
      errno = device_open->error;
      return -1;
    }
  }
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list arguments;
    va_start(arguments, flags);
    mode = va_arg(arguments, mode_t);
    va_end(arguments);
  }
  return __real_open(path, flags, mode);
}
