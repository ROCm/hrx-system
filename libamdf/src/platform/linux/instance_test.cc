// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/instance.h"

#include <fcntl.h>

#include <cerrno>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

struct NativeState {
  int dxg_result = 10000;
  int dxg_error = 0;
  int sysfs_result = 10001;
  int dxg_open_count = 0;
  int sysfs_open_count = 0;
  int dxg_close_count = 0;
  int sysfs_close_count = 0;
};

NativeState* native_state = nullptr;

class LinuxInstanceTest : public ::testing::Test {
 protected:
  void SetUp() override { native_state = &native_; }
  void TearDown() override { native_state = nullptr; }

  NativeState native_;
};

TEST_F(LinuxInstanceTest, SelectsDxgWhenTheServiceIsAvailable) {
  amdf_platform_instance_t* instance = nullptr;
  ASSERT_EQ(amdf_platform_instance_create(amdf_allocator_system(), &instance),
            AMDF_STATUS_OK);
  ASSERT_NE(instance, nullptr);
  EXPECT_EQ(instance->kind, AMDF_LINUX_PLATFORM_KIND_DXG);
  EXPECT_EQ(instance->connection.descriptor, native_.dxg_result);
  EXPECT_EQ(instance->sysfs_descriptor, -1);
  EXPECT_EQ(native_.dxg_open_count, 1);
  EXPECT_EQ(native_.sysfs_open_count, 0);

  EXPECT_EQ(amdf_platform_instance_destroy(instance), AMDF_STATUS_OK);
  EXPECT_EQ(native_.dxg_close_count, 1);
  EXPECT_EQ(native_.sysfs_close_count, 0);
}

TEST_F(LinuxInstanceTest, AbsenceSelectsNativeLinux) {
  native_.dxg_result = -1;
  native_.dxg_error = ENOENT;
  amdf_platform_instance_t* instance = nullptr;
  ASSERT_EQ(amdf_platform_instance_create(amdf_allocator_system(), &instance),
            AMDF_STATUS_OK);
  ASSERT_NE(instance, nullptr);
  EXPECT_EQ(instance->kind, AMDF_LINUX_PLATFORM_KIND_NATIVE_DEVICE);
  EXPECT_EQ(instance->connection.descriptor, -1);
  EXPECT_EQ(instance->sysfs_descriptor, native_.sysfs_result);
  EXPECT_EQ(native_.dxg_open_count, 1);
  EXPECT_EQ(native_.sysfs_open_count, 1);

  EXPECT_EQ(amdf_platform_instance_destroy(instance), AMDF_STATUS_OK);
  EXPECT_EQ(native_.dxg_close_count, 0);
  EXPECT_EQ(native_.sysfs_close_count, 1);
}

TEST_F(LinuxInstanceTest, DxgFailureDoesNotSilentlyChangeProviders) {
  native_.dxg_result = -1;
  native_.dxg_error = EACCES;
  auto* const original = reinterpret_cast<amdf_platform_instance_t*>(0x1234);
  amdf_platform_instance_t* instance = original;
  EXPECT_EQ(amdf_platform_instance_create(amdf_allocator_system(), &instance),
            amdf_linux_error(EACCES));
  EXPECT_EQ(instance, original);
  EXPECT_EQ(native_.dxg_open_count, 1);
  EXPECT_EQ(native_.sysfs_open_count, 0);
  EXPECT_EQ(native_.dxg_close_count, 0);
  EXPECT_EQ(native_.sysfs_close_count, 0);
}

}  // namespace

extern "C" {
int __real_open(const char* path, int flags, ...);
int __real_close(int descriptor);

int __wrap_open(const char* path, int flags, ...) {
  if (native_state == nullptr) {
    return __real_open(path, flags);
  }
  if (std::strcmp(path, "/dev/dxg") == 0) {
    EXPECT_EQ(flags, O_RDONLY | O_CLOEXEC);
    ++native_state->dxg_open_count;
    if (native_state->dxg_result < 0) {
      errno = native_state->dxg_error;
    }
    return native_state->dxg_result;
  }
  if (std::strcmp(path, "/sys") == 0) {
    EXPECT_EQ(flags, O_PATH | O_DIRECTORY | O_CLOEXEC);
    ++native_state->sysfs_open_count;
    return native_state->sysfs_result;
  }
  return __real_open(path, flags);
}

int __wrap_close(int descriptor) {
  if (native_state != nullptr && descriptor == native_state->dxg_result) {
    ++native_state->dxg_close_count;
    return 0;
  }
  if (native_state != nullptr && descriptor == native_state->sysfs_result) {
    ++native_state->sysfs_close_count;
    return 0;
  }
  return __real_close(descriptor);
}
}
