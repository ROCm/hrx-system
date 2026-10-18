// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_memory.h"

#include <fcntl.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

class HostBackingTest : public ::testing::TestWithParam<uint64_t> {
 protected:
  void SetUp() override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_size_ = static_cast<size_t>(page_size);
    ASSERT_EQ(amdf_platform_host_backing_create(GetParam(), &backing_),
              AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (backing_.pointer != nullptr) {
      EXPECT_EQ(amdf_platform_host_backing_destroy(&backing_, GetParam()),
                AMDF_STATUS_OK);
    }
  }

  void ExpectPresent() {
    // Presence bits are readable without PFN privileges. Inspecting them
    // neither touches the payload nor installs missing CPU translations.
    const int descriptor = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(descriptor, 0);
    std::vector<uint64_t> entries(GetParam() / page_size_);
    const auto length = entries.size() * sizeof(entries[0]);
    const off_t offset =
        (reinterpret_cast<uintptr_t>(backing_.pointer) / page_size_) *
        sizeof(entries[0]);
    const ssize_t result = pread(descriptor, entries.data(), length, offset);
    EXPECT_EQ(close(descriptor), 0);
    ASSERT_EQ(result, static_cast<ssize_t>(length));
    for (size_t i = 0; i < entries.size(); ++i) {
      ASSERT_NE(entries[i] & (UINT64_C(1) << 63), 0u) << "page " << i;
    }
  }

  // Real immutable file source and its owned, persistent ordinary host view.
  amdf_platform_host_backing_t backing_ = {};
  // Native base-page size for residency and placement queries.
  size_t page_size_ = 0;
};

TEST_P(HostBackingTest, PreparesEveryTranslationAndPreservesFileAliases) {
  ASSERT_EQ(amdf_platform_host_backing_prepare(&backing_, GetParam(), {},
                                               amdf_allocator_system()),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ExpectPresent());
  auto* bytes = static_cast<uint8_t*>(backing_.pointer);
  std::vector<uint8_t> expected(GetParam());
  ASSERT_EQ(std::memcmp(bytes, expected.data(), expected.size()), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    bytes[i] = expected[i] = (i * 17 + i / page_size_ + 0xB7) & 0xFF;
  }
  void* alias = mmap(nullptr, GetParam(), PROT_READ | PROT_WRITE, MAP_SHARED,
                     static_cast<int>(backing_.handle), 0);
  ASSERT_NE(alias, MAP_FAILED);
  EXPECT_EQ(std::memcmp(alias, expected.data(), expected.size()), 0);
  std::memset(alias, 0xA9, GetParam());
  EXPECT_EQ(munmap(alias, GetParam()), 0);
  std::fill(expected.begin(), expected.end(), 0xA9);
  EXPECT_EQ(std::memcmp(bytes, expected.data(), expected.size()), 0);
}

TEST_P(HostBackingTest, BindsPlacementBeforePreparingFoliosAndTail) {
  constexpr size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
  std::vector<unsigned long> allowed(page_size_ / sizeof(unsigned long));
  const size_t node_count = allowed.size() * kWordBits;
  ASSERT_EQ(syscall(SYS_get_mempolicy, nullptr, allowed.data(), node_count + 1,
                    nullptr, MPOL_F_MEMS_ALLOWED),
            0);
  uint32_t tested_nodes = 0;
  for (uint32_t node = 0; node < node_count; ++node) {
    if ((allowed[node / kWordBits] & (1UL << (node % kWordBits))) == 0) {
      continue;
    }
    SCOPED_TRACE(node);
    // Preparation binds exclusively owned, fresh pages. Recreate the backing
    // between nodes so this cannot pass by querying an earlier allocation.
    ASSERT_EQ(amdf_platform_host_backing_destroy(&backing_, GetParam()),
              AMDF_STATUS_OK);
    backing_ = {};
    ASSERT_EQ(amdf_platform_host_backing_create(GetParam(), &backing_),
              AMDF_STATUS_OK);
    ASSERT_EQ(
        amdf_platform_host_backing_prepare(
            &backing_, GetParam(), {AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, node},
            amdf_allocator_system()),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ExpectPresent());
    for (size_t offset = 0; offset < GetParam(); offset += page_size_) {
      int actual_node = -1;
      ASSERT_EQ(syscall(SYS_get_mempolicy, &actual_node, nullptr, 0,
                        static_cast<uint8_t*>(backing_.pointer) + offset,
                        MPOL_F_NODE | MPOL_F_ADDR),
                0);
      ASSERT_EQ(actual_node, static_cast<int>(node)) << "offset " << offset;
    }
    ++tested_nodes;
  }
  EXPECT_GT(tested_nodes, 0u);
  RecordProperty("numa_nodes", tested_nodes);
}

INSTANTIATE_TEST_SUITE_P(Geometry, HostBackingTest,
                         ::testing::Values<uint64_t>(4096, 2u << 20,
                                                     (2u << 20) + 4096,
                                                     (4u << 20) + 4096));

}  // namespace
