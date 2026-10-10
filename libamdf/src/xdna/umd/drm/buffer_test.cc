// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/drm/buffer.h"

#include <drm/amdxdna_accel.h>
#include <drm/drm.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>

#include "gtest/gtest.h"

namespace {

// Only GEM ioctls are replaced. CPU mapping, sharing, and unmapping use a real
// file-backed allocation through the production buffer implementation.
struct NativeBufferState {
  // File whose contents stand in for the native BO's host backing.
  int descriptor = -1;
  // BO size in bytes, established from the host's mapping granularity.
  size_t byte_length = 0;
  // Number of native address queries, including the required mmap offset.
  uint32_t query_count = 0;
  // Error returned instead of the initial mapping-offset query.
  int query_error = 0;
  // Number of native handle releases.
  uint32_t release_count = 0;
};

// Scoped to each single-threaded fixture invocation.
NativeBufferState* native_buffer = nullptr;

class LinuxXdnaBufferMappingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    native_.byte_length = static_cast<size_t>(page_size);
    native_.descriptor = memfd_create("amdf-command-mapping-test", MFD_CLOEXEC);
    ASSERT_GE(native_.descriptor, 0);
    ASSERT_EQ(ftruncate(native_.descriptor, native_.byte_length), 0);
    native_buffer = &native_;
  }

  void TearDown() override {
    EXPECT_EQ(amdf_linux_xdna_buffer_deinitialize(native_.descriptor, &buffer_),
              AMDF_STATUS_OK);
    native_buffer = nullptr;
    if (native_.descriptor >= 0) {
      EXPECT_EQ(close(native_.descriptor), 0);
    }
  }

  amdf_status_t CreateAndAttach(uint32_t type) {
    const amdf_status_t status = amdf_linux_xdna_buffer_create(
        native_.descriptor, type, native_.byte_length, &buffer_);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    return amdf_linux_xdna_buffer_attach(
        native_.descriptor, native_.byte_length, native_.byte_length, nullptr,
        &buffer_);
  }

  void ReleaseMapping() {
    ASSERT_NE(buffer_.mapping.base, nullptr);
    ASSERT_EQ(amdf_linux_xdna_buffer_deinitialize(native_.descriptor, &buffer_),
              AMDF_STATUS_OK);
    EXPECT_EQ(native_.release_count, 1u);
    EXPECT_EQ(buffer_.mapping.base, nullptr);
    EXPECT_EQ(buffer_.host_pointer, nullptr);
    EXPECT_EQ(buffer_.handle, 0u);
  }

  // Native dependency owner, live through production buffer release.
  NativeBufferState native_;
  // Actual buffer owner under test, including partial attachment state.
  amdf_linux_xdna_buffer_t buffer_ = {};
};

TEST_F(LinuxXdnaBufferMappingTest,
       CommandsShareHostBackingWithoutDeviceAddress) {
  ASSERT_EQ(CreateAndAttach(AMDXDNA_BO_CMD), AMDF_STATUS_OK);
  ASSERT_NE(buffer_.host_pointer, nullptr);
  EXPECT_EQ(buffer_.device_address, 0u);
  EXPECT_EQ(native_.query_count, 1u);
  // The kernel reads a packet and writes its result through the same backing.
  // Neither direction requires a device address or device cache maintenance.
  *static_cast<uint32_t*>(buffer_.host_pointer) = 0x12345671;
  uint32_t packet = 0;
  ASSERT_EQ(pread(native_.descriptor, &packet, sizeof(packet), 0),
            sizeof(packet));
  EXPECT_EQ(packet, 0x12345671u);
  packet = 0x12345674;
  ASSERT_EQ(pwrite(native_.descriptor, &packet, sizeof(packet), 0),
            sizeof(packet));
  EXPECT_EQ(*static_cast<uint32_t*>(buffer_.host_pointer), packet);
  ASSERT_NO_FATAL_FAILURE(ReleaseMapping());
}

TEST_F(LinuxXdnaBufferMappingTest,
       DeviceMemoryStillRequiresPostMappingAddress) {
  EXPECT_EQ(CreateAndAttach(AMDXDNA_BO_SHARE),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENXIO));
  EXPECT_EQ(native_.query_count, 2u);
  EXPECT_EQ(buffer_.host_pointer, nullptr);
  // The failed attachment retains its mapping for local rollback.
  ASSERT_NO_FATAL_FAILURE(ReleaseMapping());
}

TEST_F(LinuxXdnaBufferMappingTest,
       OffsetQueryFailureLeavesOnlyHandleToRelease) {
  native_.query_error = EIO;
  EXPECT_EQ(CreateAndAttach(AMDXDNA_BO_CMD),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(buffer_.host_pointer, nullptr);
  EXPECT_EQ(buffer_.mapping.base, nullptr);
  ASSERT_EQ(amdf_linux_xdna_buffer_deinitialize(native_.descriptor, &buffer_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.release_count, 1u);
}

TEST(LinuxXdnaBufferTest, CreateFailureLeavesOutputUnchanged) {
  amdf_linux_xdna_buffer_t buffer = {
      .handle = 0x10,
      .type = AMDXDNA_BO_DEV,
      .byte_length = 0x2000,
      .device_address = UINT64_C(0x12340000),
      .host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000}),
  };
  buffer.mapping.base = reinterpret_cast<void*>(uintptr_t{0x9ABC0000});
  buffer.mapping.byte_length = 0x3000;

  EXPECT_EQ(amdf_linux_xdna_buffer_create(-1, AMDXDNA_BO_SHARE, 4096, &buffer),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(buffer.handle, 0x10u);
  EXPECT_EQ(buffer.type, AMDXDNA_BO_DEV);
  EXPECT_EQ(buffer.byte_length, 0x2000u);
  EXPECT_EQ(buffer.device_address, UINT64_C(0x12340000));
  EXPECT_EQ(buffer.host_pointer,
            reinterpret_cast<void*>(uintptr_t{0x56780000}));
  EXPECT_EQ(buffer.mapping.base,
            reinterpret_cast<void*>(uintptr_t{0x9ABC0000}));
  EXPECT_EQ(buffer.mapping.byte_length, 0x3000u);
}

TEST(LinuxXdnaBufferTest, ImportFailureLeavesOutputUnchanged) {
  amdf_linux_xdna_buffer_t buffer = {
      .handle = 0x10,
      .type = AMDXDNA_BO_DEV,
      .byte_length = 0x2000,
      .device_address = UINT64_C(0x12340000),
      .host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000}),
  };
  buffer.mapping.base = reinterpret_cast<void*>(uintptr_t{0x9ABC0000});
  buffer.mapping.byte_length = 0x3000;

  EXPECT_EQ(amdf_linux_xdna_buffer_import_dma_buf(-1, 7, 4096, &buffer),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(buffer.handle, 0x10u);
  EXPECT_EQ(buffer.type, AMDXDNA_BO_DEV);
  EXPECT_EQ(buffer.byte_length, 0x2000u);
  EXPECT_EQ(buffer.device_address, UINT64_C(0x12340000));
  EXPECT_EQ(buffer.host_pointer,
            reinterpret_cast<void*>(uintptr_t{0x56780000}));
  EXPECT_EQ(buffer.mapping.base,
            reinterpret_cast<void*>(uintptr_t{0x9ABC0000}));
  EXPECT_EQ(buffer.mapping.byte_length, 0x3000u);
}

TEST(LinuxXdnaBufferTest, RegistrationFailureLeavesOutputUnchanged) {
  amdf_linux_xdna_buffer_t buffer = {
      .handle = 0x10,
      .type = AMDXDNA_BO_DEV,
      .byte_length = 0x2000,
      .device_address = UINT64_C(0x12340000),
      .host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000}),
  };
  buffer.mapping.base = reinterpret_cast<void*>(uintptr_t{0x9ABC0000});
  buffer.mapping.byte_length = 0x3000;

  EXPECT_EQ(
      amdf_linux_xdna_buffer_register_host_pages(
          -1, reinterpret_cast<void*>(uintptr_t{0x100000}), 4096, &buffer),
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(buffer.handle, 0x10u);
  EXPECT_EQ(buffer.type, AMDXDNA_BO_DEV);
  EXPECT_EQ(buffer.byte_length, 0x2000u);
  EXPECT_EQ(buffer.device_address, UINT64_C(0x12340000));
  EXPECT_EQ(buffer.host_pointer,
            reinterpret_cast<void*>(uintptr_t{0x56780000}));
  EXPECT_EQ(buffer.mapping.base,
            reinterpret_cast<void*>(uintptr_t{0x9ABC0000}));
  EXPECT_EQ(buffer.mapping.byte_length, 0x3000u);
}

TEST(LinuxXdnaBufferTest, ExportFailureLeavesOutputUnchanged) {
  const amdf_linux_xdna_buffer_t buffer = {
      .handle = 0x10,
      .type = AMDXDNA_BO_SHARE,
      .byte_length = 0x2000,
  };
  int descriptor = 73;

  EXPECT_EQ(amdf_linux_xdna_buffer_export_dma_buf(-1, &buffer, &descriptor),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(descriptor, 73);
}

}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_buffer == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(descriptor, native_buffer->descriptor);
  if (request == DRM_IOCTL_AMDXDNA_CREATE_BO) {
    auto* create = static_cast<amdxdna_drm_create_bo*>(argument);
    EXPECT_EQ(create->size, native_buffer->byte_length);
    create->handle = 17;
    return 0;
  }
  if (request == DRM_IOCTL_AMDXDNA_GET_BO_INFO) {
    auto* info = static_cast<amdxdna_drm_get_bo_info*>(argument);
    EXPECT_EQ(info->handle, 17u);
    ++native_buffer->query_count;
    if (native_buffer->query_error != 0 || native_buffer->query_count > 1) {
      errno =
          native_buffer->query_error != 0 ? native_buffer->query_error : ENXIO;
      return -1;
    }
    // Host mapping is supported; device addressing is deliberately unavailable.
    info->map_offset = 0;
    info->vaddr = 0;
    info->xdna_addr = AMDXDNA_INVALID_ADDR;
    return 0;
  }
  if (request == DRM_IOCTL_GEM_CLOSE) {
    EXPECT_EQ(static_cast<drm_gem_close*>(argument)->handle, 17u);
    ++native_buffer->release_count;
    return 0;
  }
  ADD_FAILURE() << "Unexpected native buffer request: " << request;
  errno = ENOTTY;
  return -1;
}
