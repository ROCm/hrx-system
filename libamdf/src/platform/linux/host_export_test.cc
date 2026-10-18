// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_export.h"

#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdarg>
#include <cstring>
#include <string>

#include "gtest/gtest.h"

namespace {

// Only optional exporter discovery and its native create ioctl are replaced.
// Backing creation, immutable seals, descriptor ownership, and identity queries
// use the production implementation and real operating-system resources.
struct ExportDependency {
  // Exact failure opening the optional provider, or zero for availability.
  int open_error = 0;
  // Exact failure reading the configured limit, or zero for availability.
  int limit_error = 0;
  // Configured module-parameter contents.
  std::string limit = "64\n";
  // Exact failure creating an export, before native progress.
  int create_error = 0;
  // Borrowed service descriptor owned by the production instance service.
  int service = -1;
  // Export identities created by the dependency.
  uint32_t create_count = 0;
};

thread_local ExportDependency* dependency = nullptr;

}  // namespace

extern "C" int __real_open(const char* path, int flags, ...);
extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);

extern "C" int __wrap_open(const char* path, int flags, ...) {
  if (dependency != nullptr && std::strcmp(path, "/dev/udmabuf") == 0) {
    EXPECT_EQ(flags, O_RDWR | O_CLOEXEC);
    if (dependency->open_error != 0) {
      errno = dependency->open_error;
      return -1;
    }
    dependency->service = memfd_create("amdf-export-service", MFD_CLOEXEC);
    return dependency->service;
  }
  if (dependency != nullptr &&
      std::strcmp(path, "/sys/module/udmabuf/parameters/size_limit_mb") == 0) {
    EXPECT_EQ(flags, O_RDONLY | O_CLOEXEC);
    if (dependency->limit_error != 0) {
      errno = dependency->limit_error;
      return -1;
    }
    const int descriptor = memfd_create("amdf-export-limit", MFD_CLOEXEC);
    if (descriptor >= 0) {
      EXPECT_EQ(pwrite(descriptor, dependency->limit.data(),
                       dependency->limit.size(), 0),
                static_cast<ssize_t>(dependency->limit.size()));
    }
    return descriptor;
  }
  va_list arguments;
  va_start(arguments, flags);
  const mode_t mode = (flags & O_CREAT) != 0 ? va_arg(arguments, mode_t) : 0;
  va_end(arguments);
  return __real_open(path, flags, mode);
}

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (dependency == nullptr || descriptor != dependency->service) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(request, UDMABUF_CREATE);
  const auto& create = *static_cast<const udmabuf_create*>(argument);
  EXPECT_EQ(create.flags, UDMABUF_FLAGS_CLOEXEC);
  EXPECT_EQ(create.offset, 0u);
  EXPECT_EQ(fcntl(create.memfd, F_GET_SEALS),
            F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL);
  ++dependency->create_count;
  if (dependency->create_error != 0) {
    errno = dependency->create_error;
    return -1;
  }
  const int exported = memfd_create("amdf-export-identity", MFD_CLOEXEC);
  if (exported >= 0) {
    EXPECT_EQ(ftruncate(exported, create.size), 0);
  }
  return exported;
}

namespace {

class HostExportTest : public ::testing::Test {
 protected:
  void SetUp() override { dependency = &native_; }

  void TearDown() override {
    if (initialized_) {
      EXPECT_EQ(amdf_platform_host_export_deinitialize(&service_),
                AMDF_STATUS_OK);
      if (native_.service >= 0) {
        EXPECT_EQ(fcntl(native_.service, F_GETFD), -1);
        EXPECT_EQ(errno, EBADF);
      }
    }
    dependency = nullptr;
  }

  void Initialize() {
    amdf_platform_host_export_initialize(&service_);
    initialized_ = true;
    EXPECT_EQ(service_.type, AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
  }

  // Per-test native dependencies; there is no live GPU or device mutation.
  ExportDependency native_;
  // Production optional provider owner.
  amdf_platform_host_export_t service_ = {};
  // Whether service cleanup is required, including failed discovery.
  bool initialized_ = false;
};

TEST_F(HostExportTest, DiscoversCachedLimitAndFreshExportOwnership) {
  Initialize();
  ASSERT_EQ(service_.availability, AMDF_STATUS_OK);
  EXPECT_EQ(service_.maximum_byte_length, UINT64_C(64) << 20);
  EXPECT_EQ(fcntl(service_.native_handle, F_GETFD), FD_CLOEXEC);
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_create(4096, &backing), AMDF_STATUS_OK);
  EXPECT_EQ(ftruncate(backing.handle, 8192), -1);
  EXPECT_EQ(errno, EPERM);
  EXPECT_EQ(ftruncate(backing.handle, 0), -1);
  EXPECT_EQ(errno, EPERM);
  std::memset(backing.pointer, 0xB7, 4096);
  amdf_external_memory_t first = {};
  amdf_external_memory_t second = {};
  ASSERT_EQ(amdf_platform_host_export_memory(&service_, &backing, 4096, &first),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      amdf_platform_host_export_memory(&service_, &backing, 4096, &second),
      AMDF_STATUS_OK);
  EXPECT_EQ(native_.create_count, 2u);
  EXPECT_NE(first.payload.file_descriptor, second.payload.file_descriptor);
  EXPECT_EQ(first.byte_length, 4096u);
  EXPECT_EQ(second.byte_length, 4096u);
  EXPECT_FALSE(amdf_physical_memory_id_is_equal(&first.physical_backing_id,
                                                &second.physical_backing_id));
  const int first_descriptor = first.payload.file_descriptor;
  first.release(first.release_user_data, first.type, first.payload);
  EXPECT_EQ(fcntl(first_descriptor, F_GETFD), -1);
  EXPECT_EQ(fcntl(second.payload.file_descriptor, F_GETFD), FD_CLOEXEC);
  second.release(second.release_user_data, second.type, second.payload);
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096), AMDF_STATUS_OK);
}

TEST_F(HostExportTest, AbsenceAndDenialRemainNativeAvailability) {
  for (int error : {ENOENT, EACCES, ENODEV}) {
    native_.open_error = error;
    Initialize();
    EXPECT_EQ(amdf_status_domain(service_.availability),
              AMDF_STATUS_DOMAIN_ERRNO);
    EXPECT_EQ(amdf_status_code(service_.availability), error);
    EXPECT_EQ(service_.native_handle, -1);
    EXPECT_EQ(service_.maximum_byte_length, 0u);
    EXPECT_EQ(amdf_platform_host_export_deinitialize(&service_),
              AMDF_STATUS_OK);
  }
}

TEST_F(HostExportTest, FailedLimitDiscoveryRetiresOnlyTheService) {
  native_.limit_error = EACCES;
  Initialize();
  EXPECT_EQ(amdf_status_domain(service_.availability),
            AMDF_STATUS_DOMAIN_ERRNO);
  EXPECT_EQ(amdf_status_code(service_.availability), EACCES);
  EXPECT_EQ(service_.native_handle, -1);
}

TEST_F(HostExportTest, RejectsUnusableOrMalformedNativeLimits) {
  for (const char* value : {"0\n", "", "\n", "-1\n", "2147483648\n",
                            "9999999999999999999999999", "64 extra\n"}) {
    SCOPED_TRACE(value);
    native_.limit = value;
    Initialize();
    EXPECT_EQ(amdf_status_domain(service_.availability),
              AMDF_STATUS_DOMAIN_ERRNO);
    EXPECT_EQ(amdf_status_code(service_.availability),
              value[0] == '0' ? ENOSPC : EPROTO);
    EXPECT_EQ(service_.native_handle, -1);
    EXPECT_EQ(fcntl(native_.service, F_GETFD), -1);
    EXPECT_EQ(amdf_platform_host_export_deinitialize(&service_),
              AMDF_STATUS_OK);
  }
}

TEST_F(HostExportTest, NativeCreateFailurePreservesCallerOutputAndBacking) {
  Initialize();
  ASSERT_EQ(service_.availability, AMDF_STATUS_OK);
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_create(4096, &backing), AMDF_STATUS_OK);
  std::memset(backing.pointer, 0xC3, 4096);
  for (int error : {ENOMEM, ENOSPC, EINVAL}) {
    native_.create_error = error;
    amdf_external_memory_t output;
    std::memset(&output, 0xA5, sizeof(output));
    const auto original = output;
    const auto status =
        amdf_platform_host_export_memory(&service_, &backing, 4096, &output);
    EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_ERRNO);
    EXPECT_EQ(amdf_status_code(status), error);
    EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
    EXPECT_EQ(static_cast<uint8_t*>(backing.pointer)[4095], 0xC3);
  }
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096), AMDF_STATUS_OK);
}

}  // namespace
