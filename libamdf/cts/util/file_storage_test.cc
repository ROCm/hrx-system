// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/util/file_storage.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

class FileStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_byte_length_ = static_cast<size_t>(page_size);
    const char* temporary_directory = std::getenv("TEST_TMPDIR");
    std::string path =
        std::string(temporary_directory ? temporary_directory : "/tmp") +
        "/amdf-file-storage-XXXXXX";
    file_ = mkstemp(path.data());
    ASSERT_GE(file_, 0) << std::strerror(errno);
    ASSERT_EQ(unlink(path.c_str()), 0) << std::strerror(errno);
  }

  void TearDown() override {
    if (file_ >= 0) {
      EXPECT_EQ(close(file_), 0) << std::strerror(errno);
    }
  }

  void ExpectSeed(const void* seed, size_t byte_length) {
    struct stat information = {};
    ASSERT_EQ(fstat(file_, &information), 0) << std::strerror(errno);
    ASSERT_EQ(information.st_size, static_cast<off_t>(byte_length));
    std::vector<uint8_t> observed(byte_length);
    size_t read = 0;
    while (read < byte_length) {
      const ssize_t result =
          pread(file_, observed.data() + read, byte_length - read, read);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      ASSERT_GT(result, 0) << std::strerror(errno);
      read += static_cast<size_t>(result);
    }
    EXPECT_EQ(std::memcmp(observed.data(), seed, byte_length), 0);
  }

  // Unlinked private file owned until teardown, including skipped cases.
  int file_ = -1;
  // Native page length used for direct-I/O alignment and seed extents.
  size_t page_byte_length_ = 0;
};

TEST_F(FileStorageTest, BufferedSeedAllowsUnalignedExtents) {
  const std::vector<uint8_t> seed(3 * page_byte_length_ + 7, 0xC3);
  ASSERT_NO_FATAL_FAILURE(InitializeFileStorage(
      file_, FileMode::kBuffered, page_byte_length_, seed.data(), seed.size()));
  const int flags = fcntl(file_, F_GETFL);
  ASSERT_GE(flags, 0);
  EXPECT_EQ(flags & O_DIRECT, 0);
  ASSERT_NO_FATAL_FAILURE(ExpectSeed(seed.data(), seed.size()));
}

TEST_F(FileStorageTest, DirectSeedAcceptsUnalignedMemoryWithoutCaching) {
  std::vector<uint8_t> storage(3 * page_byte_length_ + 1);
  for (size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<uint8_t>(i * 37 + 19);
  }
  const uint8_t* seed = storage.data() + 1;
  const size_t byte_length = storage.size() - 1;
  ASSERT_NO_FATAL_FAILURE(InitializeFileStorage(
      file_, FileMode::kDirect, page_byte_length_, seed, byte_length));
  if (IsSkipped()) {
    return;
  }
  const int flags = fcntl(file_, F_GETFL);
  ASSERT_GE(flags, 0);
  EXPECT_NE(flags & O_DIRECT, 0);
  ASSERT_NO_FATAL_FAILURE(FinishDirectFileIo(file_, byte_length));
  ASSERT_NO_FATAL_FAILURE(ExpectSeed(seed, byte_length));
}

TEST_F(FileStorageTest, EmptyDirectFileRemainsEmpty) {
  ASSERT_NO_FATAL_FAILURE(InitializeFileStorage(file_, FileMode::kDirect,
                                                page_byte_length_, nullptr, 0));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(FinishDirectFileIo(file_, 0));
  struct stat information = {};
  ASSERT_EQ(fstat(file_, &information), 0) << std::strerror(errno);
  EXPECT_EQ(information.st_size, 0);
}

}  // namespace
