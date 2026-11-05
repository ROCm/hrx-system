// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/connection.h"

#include <fcntl.h>
#include <sys/ioctl.h>

#include <cerrno>
#include <cstdarg>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

struct NativeState {
  int open_result = 10000;
  int open_error = 0;
  int ioctl_result = 0;
  int ioctl_error = 0;
  int close_result = 0;
  int close_error = 0;
  int open_count = 0;
  int ioctl_count = 0;
  int close_count = 0;
  unsigned long request = 0;
  void* arguments = nullptr;
};

NativeState* native_state = nullptr;

class DxgConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override { native_state = &native_; }
  void TearDown() override { native_state = nullptr; }

  NativeState native_;
};

TEST_F(DxgConnectionTest, OwnsOneExplicitDescriptor) {
  amdf_linux_dxg_connection_t connection = {.descriptor = -1};
  ASSERT_EQ(amdf_linux_dxg_connection_initialize(&connection), AMDF_STATUS_OK);
  EXPECT_EQ(connection.descriptor, native_.open_result);
  EXPECT_EQ(native_.open_count, 1);

  int arguments = 42;
  constexpr unsigned long request = _IOWR('G', 0x3a, int);
  native_.ioctl_result = 7;
  EXPECT_EQ(amdf_linux_dxg_connection_invoke(&connection, request, &arguments),
            7u);
  EXPECT_EQ(native_.ioctl_count, 1);
  EXPECT_EQ(native_.request, request);
  EXPECT_EQ(native_.arguments, &arguments);
  EXPECT_EQ(native_.open_count, 1);

  EXPECT_EQ(amdf_linux_dxg_connection_deinitialize(&connection),
            AMDF_STATUS_OK);
  EXPECT_EQ(connection.descriptor, -1);
  EXPECT_EQ(native_.close_count, 1);
}

TEST_F(DxgConnectionTest, OpenFailurePreservesOutput) {
  native_.open_result = -1;
  native_.open_error = ENODEV;
  amdf_linux_dxg_connection_t connection = {.descriptor = 77};
  EXPECT_EQ(amdf_linux_dxg_connection_initialize(&connection),
            amdf_linux_error(ENODEV));
  EXPECT_EQ(connection.descriptor, 77);
  EXPECT_EQ(native_.open_count, 1);
  EXPECT_EQ(native_.close_count, 0);
}

TEST_F(DxgConnectionTest, TranslatesPublishedLibdxgErrorsExactly) {
  amdf_linux_dxg_connection_t connection = {.descriptor = 10000};
  struct ErrorCase {
    int error;
    uint32_t status;
  };
  const ErrorCase cases[] = {
      {EEXIST, 0xC0000035u},       // STATUS_OBJECT_NAME_COLLISION
      {ENOMEM, 0xC0000017u},       // STATUS_NO_MEMORY
      {EINVAL, 0xC000000Du},       // STATUS_INVALID_PARAMETER
      {ENOENT, 0xC0000033u},       // STATUS_OBJECT_NAME_INVALID
      {EAGAIN, 0x00000102u},       // STATUS_TIMEOUT
      {EOVERFLOW, 0xC0000023u},    // STATUS_BUFFER_TOO_SMALL
      {ENODEV, 0xC00002B6u},       // STATUS_DEVICE_REMOVED
      {EACCES, 0xC0000022u},       // STATUS_ACCESS_DENIED
      {ENOSYS, 0xC00000BBu},       // STATUS_NOT_SUPPORTED
      {EPERM, 0xC00000BBu},        // STATUS_NOT_SUPPORTED
      {EOPNOTSUPP, 0xC000001Du},   // STATUS_ILLEGAL_INSTRUCTION
      {EBADF, 0xC0000008u},        // STATUS_INVALID_HANDLE
      {EINPROGRESS, 0xC01E0102u},  // STATUS_GRAPHICS_ALLOCATION_BUSY
      {EPROTOTYPE, 0xC0000024u},   // STATUS_OBJECT_TYPE_MISMATCH
      {EIO, 0xC0000001u},          // STATUS_UNSUCCESSFUL
  };
  for (const ErrorCase& test_case : cases) {
    native_.ioctl_result = -1;
    native_.ioctl_error = test_case.error;
    EXPECT_EQ(amdf_linux_dxg_connection_invoke(&connection, 1, nullptr),
              test_case.status)
        << test_case.error;
  }
  EXPECT_EQ(native_.ioctl_count,
            static_cast<int>(sizeof(cases) / sizeof(cases[0])));
  EXPECT_EQ(native_.open_count, 0);
}

TEST_F(DxgConnectionTest, CloseFailureConsumesDescriptor) {
  native_.close_result = -1;
  native_.close_error = EIO;
  amdf_linux_dxg_connection_t connection = {.descriptor = 10000};
  EXPECT_EQ(amdf_linux_dxg_connection_deinitialize(&connection),
            amdf_linux_error(EIO));
  EXPECT_EQ(connection.descriptor, -1);
  EXPECT_EQ(native_.close_count, 1);
  EXPECT_EQ(amdf_linux_dxg_connection_deinitialize(&connection),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.close_count, 1);
}

}  // namespace

extern "C" {
int __real_open(const char* path, int flags, ...);
int __real_close(int descriptor);
int __real_ioctl(int descriptor, unsigned long request, ...);

int __wrap_open(const char* path, int flags, ...) {
  if (native_state != nullptr && std::strcmp(path, "/dev/dxg") == 0) {
    EXPECT_EQ(flags, O_RDONLY | O_CLOEXEC);
    ++native_state->open_count;
    if (native_state->open_result < 0) {
      errno = native_state->open_error;
    }
    return native_state->open_result;
  }
  return __real_open(path, flags);
}

int __wrap_close(int descriptor) {
  if (native_state == nullptr || descriptor < 10000) {
    return __real_close(descriptor);
  }
  ++native_state->close_count;
  if (native_state->close_result < 0) {
    errno = native_state->close_error;
  }
  return native_state->close_result;
}

int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_state == nullptr || descriptor < 10000) {
    return __real_ioctl(descriptor, request, argument);
  }
  ++native_state->ioctl_count;
  native_state->request = request;
  native_state->arguments = argument;
  if (native_state->ioctl_result < 0) {
    errno = native_state->ioctl_error;
  }
  return native_state->ioctl_result;
}
}
