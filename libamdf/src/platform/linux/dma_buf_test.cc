// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dma_buf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(DmaBufTest, QueriesStableFileIdentityAndExtent) {
  const int descriptor = memfd_create("amdf-dma-buf-test", MFD_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  ASSERT_EQ(ftruncate(descriptor, 8192), 0);
  const int duplicate = dup(descriptor);
  ASSERT_GE(duplicate, 0);

  amdf_linux_dma_buf_info_t first = {};
  amdf_linux_dma_buf_info_t second = {};
  ASSERT_EQ(amdf_linux_dma_buf_query(descriptor, &first), AMDF_STATUS_OK);
  ASSERT_EQ(amdf_linux_dma_buf_query(duplicate, &second), AMDF_STATUS_OK);
  EXPECT_EQ(first.byte_length, 8192u);
  EXPECT_TRUE(amdf_physical_memory_id_is_valid(&first.physical_backing_id));
  EXPECT_TRUE(amdf_physical_memory_id_is_equal(&first.physical_backing_id,
                                               &second.physical_backing_id));

  ASSERT_EQ(close(duplicate), 0);
  ASSERT_EQ(close(descriptor), 0);
}

TEST(DmaBufTest, FailurePreservesCallerStorage) {
  const int descriptor = memfd_create("amdf-dma-buf-test", MFD_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  ASSERT_EQ(ftruncate(descriptor, 4096), 0);
  ASSERT_EQ(close(descriptor), 0);

  amdf_linux_dma_buf_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_linux_dma_buf_info_t original = info;
  const amdf_status_t status = amdf_linux_dma_buf_query(descriptor, &info);
  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_ERRNO);
  EXPECT_EQ(amdf_status_code(status), EBADF);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(DmaBufTest, ReleaseCallbackClosesOwnedDescriptor) {
  const int descriptor = memfd_create("amdf-dma-buf-test", MFD_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  amdf_external_memory_payload_t payload = {};
  payload.file_descriptor = descriptor;
  amdf_linux_dma_buf_release(nullptr, AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
                             payload);
  EXPECT_EQ(fcntl(descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

}  // namespace
