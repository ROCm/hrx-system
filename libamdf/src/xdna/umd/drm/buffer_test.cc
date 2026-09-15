// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/drm/buffer.h"

#include <drm/amdxdna_accel.h>
#include <errno.h>

#include <cstdint>

#include "gtest/gtest.h"

namespace {

TEST(LinuxXdnaBufferTest, CreateFailureLeavesOutputUnchanged) {
  amdf_linux_xdna_buffer_t buffer = {};
  buffer.handle = 0x10;
  buffer.type = AMDXDNA_BO_DEV;
  buffer.byte_length = 0x2000;
  buffer.device_address = UINT64_C(0x12340000);
  buffer.host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000});
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
  amdf_linux_xdna_buffer_t buffer = {};
  buffer.handle = 0x10;
  buffer.type = AMDXDNA_BO_DEV;
  buffer.byte_length = 0x2000;
  buffer.device_address = UINT64_C(0x12340000);
  buffer.host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000});
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
  amdf_linux_xdna_buffer_t buffer = {};
  buffer.handle = 0x10;
  buffer.type = AMDXDNA_BO_DEV;
  buffer.byte_length = 0x2000;
  buffer.device_address = UINT64_C(0x12340000);
  buffer.host_pointer = reinterpret_cast<void*>(uintptr_t{0x56780000});
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
