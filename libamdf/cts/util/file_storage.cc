// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/util/file_storage.h"

#include <fcntl.h>
#include <linux/btrfs.h>
#include <linux/fs.h>
#include <linux/magic.h>
#include <linux/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Query residency without touching file bytes or leaving a live mapping across
// workload execution. The private inode is owned by this user, so mincore can
// report its actual residency instead of the restricted non-owner result.
void ExpectFileCacheEmpty(int file, size_t byte_length) {
  if (byte_length == 0) {
    return;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const size_t page_byte_length = static_cast<size_t>(page_size);
  std::vector<unsigned char> residency((byte_length + page_byte_length - 1) /
                                       page_byte_length);
  void* mapping = mmap(nullptr, byte_length, PROT_READ, MAP_SHARED, file, 0);
  ASSERT_NE(mapping, MAP_FAILED) << std::strerror(errno);
  const int query_result = mincore(mapping, byte_length, residency.data());
  const int query_error = errno;
  const int unmap_result = munmap(mapping, byte_length);
  const int unmap_error = errno;
  ASSERT_EQ(query_result, 0) << std::strerror(query_error);
  ASSERT_EQ(unmap_result, 0) << std::strerror(unmap_error);
  ASSERT_EQ(std::count_if(residency.begin(), residency.end(),
                          [](unsigned char value) { return value & 1; }),
            0)
      << "direct file I/O populated the page cache";
}

void PrepareFileStorage(int file, FileMode mode, size_t buffer_alignment) {
  struct statfs filesystem = {};
  ASSERT_EQ(fstatfs(file, &filesystem), 0) << std::strerror(errno);
  ::testing::Test::RecordProperty("io_filesystem_type",
                                  std::to_string(filesystem.f_type));
  ::testing::Test::RecordProperty(
      "io_mode", mode == FileMode::kDirect ? "direct" : "buffered");
  if (mode == FileMode::kBuffered) {
    return;
  }
  if (filesystem.f_type == TMPFS_MAGIC) {
    GTEST_SKIP() << "direct-storage witness requires a disk-backed file";
  }
  if (filesystem.f_type == BTRFS_SUPER_MAGIC) {
    // Btrfs rejects changing between compression policy and NOCOW in one
    // operation. Clear inherited compression policy before enabling NOCOW.
    unsigned int attributes = 0;
    ASSERT_EQ(ioctl(file, FS_IOC_GETFLAGS, &attributes), 0)
        << std::strerror(errno);
    if (attributes & (FS_COMPR_FL | FS_NOCOMP_FL)) {
      attributes &= ~(FS_COMPR_FL | FS_NOCOMP_FL);
      ASSERT_EQ(ioctl(file, FS_IOC_SETFLAGS, &attributes), 0)
          << std::strerror(errno);
    }
    // Enabling NOCOW on the empty inode also establishes NODATASUM. Checksummed
    // writes can select buffered I/O despite O_DIRECT on current Btrfs kernels.
    attributes |= FS_NOCOW_FL;
    ASSERT_EQ(ioctl(file, FS_IOC_SETFLAGS, &attributes), 0)
        << std::strerror(errno);
    ASSERT_EQ(ioctl(file, FS_IOC_GETFLAGS, &attributes), 0)
        << std::strerror(errno);
    ASSERT_NE(attributes & FS_NOCOW_FL, 0u);
    ::testing::Test::RecordProperty("io_btrfs_file_flags", attributes);
  }
  struct statx geometry = {};
  const int query_result = static_cast<int>(
      syscall(__NR_statx, file, "", AT_EMPTY_PATH, STATX_DIOALIGN, &geometry));
  if (query_result < 0) {
    ASSERT_TRUE(errno == ENOSYS || errno == EOPNOTSUPP || errno == EINVAL)
        << "statx direct-I/O alignment: " << std::strerror(errno);
    geometry = {};
  }
  const char* alignment_source = "statx";
  if (!(geometry.stx_mask & STATX_DIOALIGN) &&
      filesystem.f_type == BTRFS_SUPER_MAGIC) {
    // Btrfs checks user buffers and file extents against its sector size,
    // including kernels whose getattr does not expose STATX_DIOALIGN.
    btrfs_ioctl_fs_info_args information = {};
    ASSERT_EQ(ioctl(file, BTRFS_IOC_FS_INFO, &information), 0)
        << std::strerror(errno);
    geometry.stx_mask |= STATX_DIOALIGN;
    geometry.stx_dio_mem_align = information.sectorsize;
    geometry.stx_dio_offset_align = information.sectorsize;
    alignment_source = "BTRFS_IOC_FS_INFO";
  }
  if (!(geometry.stx_mask & STATX_DIOALIGN) ||
      geometry.stx_dio_mem_align == 0 || geometry.stx_dio_offset_align == 0) {
    GTEST_SKIP() << "filesystem does not report a direct-I/O contract";
  }
  ASSERT_EQ(buffer_alignment % geometry.stx_dio_mem_align, 0u);
  ASSERT_EQ(buffer_alignment % geometry.stx_dio_offset_align, 0u);
  ::testing::Test::RecordProperty("io_direct_memory_alignment",
                                  geometry.stx_dio_mem_align);
  ::testing::Test::RecordProperty("io_direct_offset_alignment",
                                  geometry.stx_dio_offset_align);
  ::testing::Test::RecordProperty("io_direct_alignment_source",
                                  alignment_source);
  const int flags = fcntl(file, F_GETFL);
  ASSERT_GE(flags, 0) << std::strerror(errno);
  ASSERT_EQ(fcntl(file, F_SETFL, flags | O_DIRECT), 0) << std::strerror(errno);
}

}  // namespace

void InitializeFileStorage(int file, FileMode mode, size_t buffer_alignment,
                           const void* seed, size_t byte_length) {
  ASSERT_NO_FATAL_FAILURE(PrepareFileStorage(file, mode, buffer_alignment));
  if (::testing::Test::IsSkipped()) {
    return;
  }

  std::unique_ptr<void, decltype(&std::free)> aligned_seed(nullptr, &std::free);
  if (mode == FileMode::kDirect && byte_length != 0) {
    void* allocation = nullptr;
    const int allocation_result =
        posix_memalign(&allocation, buffer_alignment, byte_length);
    ASSERT_EQ(allocation_result, 0) << std::strerror(allocation_result);
    aligned_seed.reset(allocation);
    std::memcpy(allocation, seed, byte_length);
    seed = allocation;
  }
  const auto* bytes = static_cast<const uint8_t*>(seed);
  size_t written = 0;
  while (written < byte_length) {
    const ssize_t result =
        pwrite(file, bytes + written, byte_length - written, written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    ASSERT_GT(result, 0) << std::strerror(errno);
    written += static_cast<size_t>(result);
  }
  if (mode == FileMode::kDirect) {
    SCOPED_TRACE("after direct file initialization");
    ASSERT_NO_FATAL_FAILURE(ExpectFileCacheEmpty(file, byte_length));
  }
}

void FinishDirectFileIo(int file, size_t byte_length) {
  SCOPED_TRACE("after direct file I/O");
  ASSERT_NO_FATAL_FAILURE(ExpectFileCacheEmpty(file, byte_length));
  const int flags = fcntl(file, F_GETFL);
  ASSERT_GE(flags, 0) << std::strerror(errno);
  ASSERT_EQ(fcntl(file, F_SETFL, flags & ~O_DIRECT), 0) << std::strerror(errno);
}
