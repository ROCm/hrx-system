// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/sysfs.h"

#include <array>
#include <cstring>
#include <fstream>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)

using namespace iree::testing::status;

TEST(SysfsTest, ReadSmallFileCapacity) {
  iree::testing::TempFilePath path("sysfs-small-file");
  for (const char* contents : {"", "0-3\n"}) {
    SCOPED_TRACE(contents);
    std::ofstream output(path.path(), std::ios::binary);
    output << contents;
    output.close();
    ASSERT_TRUE(output);
    for (size_t capacity : {1u, 4u, 5u, 8u}) {
      SCOPED_TRACE(capacity);
      std::array<char, 16> buffer;
      buffer.fill('?');
      iree_host_size_t length = 123;
      iree_status_t status = iree_sysfs_read_small_file(
          path.path().c_str(), buffer.data(), capacity, &length);
      const bool fits = strlen(contents) < capacity;
      if (fits) {
        IREE_ASSERT_OK(status);
        EXPECT_EQ(strlen(contents), length);
        EXPECT_STREQ(contents, buffer.data());
      } else {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
        EXPECT_EQ(0u, length);
      }
      EXPECT_EQ('?', buffer[capacity]);

      buffer.fill('?');
      length = 123;
      ASSERT_EQ(fits,
                iree_sysfs_try_read_small_file(
                    path.path().c_str(), buffer.data(), capacity, &length));
      if (fits) {
        EXPECT_EQ(strlen(contents), length);
        EXPECT_STREQ(contents, buffer.data());
      } else {
        EXPECT_EQ(0u, length);
      }
      EXPECT_EQ('?', buffer[capacity]);
    }
  }
  EXPECT_TRUE(path.Remove());
}

TEST(SysfsTest, ReadSmallFileErrorsClearLength) {
  iree::testing::TempFilePath missing("sysfs-missing-file");
  char buffer[16];
  iree_host_size_t length = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_sysfs_read_small_file(missing.path().c_str(), buffer, sizeof(buffer),
                                 &length));
  EXPECT_EQ(0u, length);
  length = 123;
  EXPECT_FALSE(iree_sysfs_try_read_small_file(missing.path().c_str(), buffer,
                                              sizeof(buffer), &length));
  EXPECT_EQ(0u, length);

  // The helper created the temporary directory. Opening it succeeds, but the
  // native read fails, exercising descriptor cleanup after a successful open.
  const std::string directory = ::testing::TempDir();
  length = 123;
  IREE_EXPECT_NOT_OK(iree_sysfs_read_small_file(directory.c_str(), buffer,
                                                sizeof(buffer), &length));
  EXPECT_EQ(0u, length);
  length = 123;
  EXPECT_FALSE(iree_sysfs_try_read_small_file(directory.c_str(), buffer,
                                              sizeof(buffer), &length));
  EXPECT_EQ(0u, length);
}

struct CpuRanges {
  // Inclusive start of each parsed range.
  uint32_t start_cpus[4];
  // Exclusive end of each parsed range.
  uint32_t end_cpus[4];
  // Number of populated ranges.
  iree_host_size_t count;
};

static bool AppendCpuRange(uint32_t start_cpu, uint32_t end_cpu,
                           void* user_data) {
  CpuRanges* ranges = (CpuRanges*)user_data;
  if (ranges->count >= IREE_ARRAYSIZE(ranges->start_cpus)) {
    return false;
  }
  ranges->start_cpus[ranges->count] = start_cpu;
  ranges->end_cpus[ranges->count] = end_cpu;
  ++ranges->count;
  return true;
}

TEST(SysfsTest, ParseCpuList) {
  CpuRanges ranges = {};
  IREE_ASSERT_OK(
      iree_sysfs_parse_cpu_list(IREE_SV("0,2-3, 7"), AppendCpuRange, &ranges));

  EXPECT_EQ(3, ranges.count);
  EXPECT_EQ(0, ranges.start_cpus[0]);
  EXPECT_EQ(1, ranges.end_cpus[0]);
  EXPECT_EQ(2, ranges.start_cpus[1]);
  EXPECT_EQ(4, ranges.end_cpus[1]);
  EXPECT_EQ(7, ranges.start_cpus[2]);
  EXPECT_EQ(8, ranges.end_cpus[2]);
}

TEST(SysfsTest, TryParseCpuListRejectsMalformedInput) {
  CpuRanges ranges = {};
  EXPECT_FALSE(iree_sysfs_try_parse_cpu_list(IREE_SV("not-a-cpu-list"),
                                             AppendCpuRange, &ranges));
  EXPECT_FALSE(
      iree_sysfs_try_parse_cpu_list(IREE_SV("3-2"), AppendCpuRange, &ranges));
  EXPECT_FALSE(iree_sysfs_try_parse_cpu_list(IREE_SV("4294967295"),
                                             AppendCpuRange, &ranges));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_sysfs_parse_cpu_list(IREE_SV("3-2"), AppendCpuRange, &ranges));
}

TEST(SysfsTest, ParseSizeString) {
  uint64_t size = 0;
  IREE_ASSERT_OK(iree_sysfs_parse_size_string(IREE_SV("32K"), &size));
  EXPECT_EQ(32768u, size);
  IREE_ASSERT_OK(iree_sysfs_parse_size_string(IREE_SV("512"), &size));
  EXPECT_EQ(512u, size);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_sysfs_parse_size_string(IREE_SV(""), &size));
}

#else

TEST(SysfsTest, PlatformDisabled) {}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM

}  // namespace
