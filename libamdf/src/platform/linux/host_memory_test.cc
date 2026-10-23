// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_memory.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

// Inject only native outcomes; backing ownership and rollback use production
// code and real file descriptors/mappings.
struct NativeState {
  // Failure of the initial view map, or zero for a real mapping.
  int map_error = 0;
  // Failure before unmapping the persistent view, or zero for real unmapping.
  int unmap_error = 0;
  // Error reported after close has consumed the descriptor.
  int close_error = 0;
  // Real descriptor acquired by the production backing preparation.
  int descriptor = -1;
  // Real persistent mapping, if the production map succeeded.
  void* pointer = nullptr;
  // Number of source release attempts through the production implementation.
  uint32_t close_count = 0;
};

thread_local NativeState* native_state = nullptr;
// Overrides only kernel identity; page population uses real native mappings.
thread_local const char* kernel_release = nullptr;

}  // namespace

extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" int __real_munmap(void*, size_t);
extern "C" int __real_close(int);
extern "C" int __real_uname(struct utsname*);

extern "C" int __wrap_uname(struct utsname* kernel) {
  const int result = __real_uname(kernel);
  if (result == 0 && kernel_release != nullptr) {
    std::snprintf(kernel->release, sizeof(kernel->release), "%s",
                  kernel_release);
  }
  return result;
}

extern "C" void* __wrap_mmap(void* address, size_t length, int protection,
                             int flags, int descriptor, off_t offset) {
  if (native_state == nullptr || (flags & MAP_SHARED) == 0 || descriptor < 0) {
    return __real_mmap(address, length, protection, flags, descriptor, offset);
  }
  native_state->descriptor = descriptor;
  if (native_state->map_error != 0) {
    errno = native_state->map_error;
    return MAP_FAILED;
  }
  void* pointer =
      __real_mmap(address, length, protection, flags, descriptor, offset);
  if (pointer != MAP_FAILED) {
    native_state->pointer = pointer;
  }
  return pointer;
}

extern "C" int __wrap_munmap(void* address, size_t length) {
  if (native_state != nullptr && address == native_state->pointer &&
      native_state->unmap_error != 0) {
    errno = native_state->unmap_error;
    return -1;
  }
  const int result = __real_munmap(address, length);
  if (native_state != nullptr && address == native_state->pointer &&
      result == 0) {
    native_state->pointer = nullptr;
  }
  return result;
}

extern "C" int __wrap_close(int descriptor) {
  if (native_state == nullptr || descriptor != native_state->descriptor) {
    return __real_close(descriptor);
  }
  ++native_state->close_count;
  const int result = __real_close(descriptor);
  if (result != 0) {
    return result;
  }
  native_state->descriptor = -1;
  if (native_state->close_error != 0) {
    errno = native_state->close_error;
    return -1;
  }
  return 0;
}

namespace {

class HostMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override { native_state = &native_; }

  void TearDown() override {
    // Test-owned native residue is independent of consumed library metadata.
    // Retire it directly, without retrying the production operation.
    native_state = nullptr;
    if (native_.pointer != nullptr) {
      EXPECT_EQ(__real_munmap(native_.pointer, 4096), 0);
    }
    if (native_.descriptor >= 0) {
      EXPECT_EQ(__real_close(native_.descriptor), 0);
    }
  }

  // Native dependency state retained through final assertions.
  NativeState native_;
};

TEST_F(HostMemoryTest, FailedPreparationRetainsItsSourceForOwnerCleanup) {
  native_.map_error = ENOMEM;
  amdf_platform_host_backing_t backing = {};
  EXPECT_EQ(amdf_platform_host_backing_prepare(4096, &backing),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  ASSERT_GE(native_.descriptor, 0);
  EXPECT_EQ(backing.handle, static_cast<uintptr_t>(native_.descriptor));
  EXPECT_EQ(backing.pointer, nullptr);
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), FD_CLOEXEC);
  EXPECT_EQ(native_.close_count, 0u);
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096), AMDF_STATUS_OK);
  EXPECT_EQ(backing.handle, UINTPTR_MAX);
  EXPECT_EQ(native_.close_count, 1u);
}

TEST_F(HostMemoryTest, FailedViewRemovalPreservesBackingAndReleaseProgress) {
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_prepare(4096, &backing), AMDF_STATUS_OK);
  auto* bytes = static_cast<uint8_t*>(backing.pointer);
  bytes[17] = 0xA5;
  native_.unmap_error = EIO;
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(backing.pointer, bytes);
  EXPECT_EQ(bytes[17], 0xA5);
  EXPECT_EQ(backing.handle, static_cast<uintptr_t>(native_.descriptor));
  EXPECT_EQ(fcntl(native_.descriptor, F_GETFD), FD_CLOEXEC);
  EXPECT_EQ(native_.close_count, 0u);
}

TEST_F(HostMemoryTest, CloseErrorPreservesTheFactThatTheSourceWasConsumed) {
  amdf_platform_host_backing_t backing = {};
  ASSERT_EQ(amdf_platform_host_backing_prepare(4096, &backing), AMDF_STATUS_OK);
  const int descriptor = native_.descriptor;
  native_.close_error = EIO;
  EXPECT_EQ(amdf_platform_host_backing_destroy(&backing, 4096),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(backing.pointer, nullptr);
  EXPECT_EQ(backing.handle, UINTPTR_MAX);
  EXPECT_EQ(fcntl(descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(native_.close_count, 1u);
}

struct KernelPagePolicy {
  // Native release identifying the exporter's scatterlist implementation.
  const char* release;
  // Whether export backing must prohibit huge-page allocation/promotion.
  bool requires_base_pages;
};

class HostBackingTest : public ::testing::TestWithParam<
                            std::tuple<uint64_t, bool, KernelPagePolicy>> {
 protected:
  void SetUp() override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_size_ = static_cast<size_t>(page_size);
    byte_length_ = std::get<0>(GetParam());
    kernel_release = std::get<2>(GetParam()).release;
    ASSERT_NO_FATAL_FAILURE(CreateBacking());
  }

  void CreateBacking() {
    ASSERT_EQ(amdf_platform_host_backing_prepare(byte_length_, &backing_),
              AMDF_STATUS_OK);
    if (std::get<1>(GetParam())) {
      // Exercise native refusal of huge-page promotion on an otherwise valid
      // writable file mapping, without exhausting machine-wide resources.
      ASSERT_EQ(madvise(backing_.pointer, byte_length_, MADV_NOHUGEPAGE), 0);
    }
  }

  void TearDown() override {
    kernel_release = nullptr;
    if (backing_.pointer != nullptr) {
      EXPECT_EQ(amdf_platform_host_backing_destroy(&backing_, byte_length_),
                AMDF_STATUS_OK);
    }
  }

  void ExpectPageSizePolicy() {
    SCOPED_TRACE(kernel_release);
    // VmFlags exposes the actual native VMA policy without reading PFNs or
    // assuming huge-page promotion succeeds under machine-wide pressure.
    std::ifstream mappings("/proc/self/smaps");
    ASSERT_TRUE(mappings.is_open());
    const auto address = reinterpret_cast<uintptr_t>(backing_.pointer);
    bool contains_backing = false;
    std::string line;
    while (std::getline(mappings, line)) {
      uintptr_t begin = 0, end = 0;
      if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2) {
        contains_backing = address >= begin && address < end;
      } else if (contains_backing && line.starts_with("VmFlags:")) {
        const bool prohibits_huge_pages = line.find(" nh") != std::string::npos;
        EXPECT_EQ(prohibits_huge_pages,
                  std::get<1>(GetParam()) ||
                      std::get<2>(GetParam()).requires_base_pages);
        return;
      }
    }
    FAIL() << "native backing missing from smaps";
  }

  void ExpectPresent() {
    // Presence bits are readable without PFN privileges. Inspecting them
    // neither touches the payload nor installs missing CPU translations.
    const int descriptor = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(descriptor, 0);
    std::vector<uint64_t> entries(byte_length_ / page_size_);
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
  // Logical backing length shared by both page-size policies.
  uint64_t byte_length_ = 0;
  // Native base-page size for residency and placement queries.
  size_t page_size_ = 0;
};

TEST_P(HostBackingTest, PreparesEveryTranslationAndPreservesFileAliases) {
  ASSERT_EQ(amdf_platform_host_backing_populate(&backing_, byte_length_, {},
                                                amdf_allocator_system()),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ExpectPresent());
  ASSERT_NO_FATAL_FAILURE(ExpectPageSizePolicy());
  auto* bytes = static_cast<uint8_t*>(backing_.pointer);
  std::vector<uint8_t> expected(byte_length_);
  ASSERT_EQ(std::memcmp(bytes, expected.data(), expected.size()), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    bytes[i] = expected[i] = (i * 17 + i / page_size_ + 0xB7) & 0xFF;
  }
  void* alias = mmap(nullptr, byte_length_, PROT_READ | PROT_WRITE, MAP_SHARED,
                     static_cast<int>(backing_.handle), 0);
  ASSERT_NE(alias, MAP_FAILED);
  EXPECT_EQ(std::memcmp(alias, expected.data(), expected.size()), 0);
  std::memset(alias, 0xA9, byte_length_);
  EXPECT_EQ(munmap(alias, byte_length_), 0);
  std::fill(expected.begin(), expected.end(), 0xA9);
  EXPECT_EQ(std::memcmp(bytes, expected.data(), expected.size()), 0);
}

TEST_P(HostBackingTest, BindsPlacementBeforePreparingFoliosAndTail) {
  constexpr size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
  std::vector<unsigned long> allowed(page_size_ / sizeof(unsigned long));
  const size_t node_count = allowed.size() * kWordBits;
  const long result = syscall(SYS_get_mempolicy, nullptr, allowed.data(),
                              node_count + 1, nullptr, MPOL_F_MEMS_ALLOWED);
  if (result == -1 && (errno == EPERM || errno == ENOSYS)) {
    GTEST_SKIP() << "NUMA policy queries unavailable: " << std::strerror(errno);
  }
  ASSERT_EQ(result, 0) << std::strerror(errno);
  uint32_t tested_nodes = 0;
  for (uint32_t node = 0; node < node_count; ++node) {
    if ((allowed[node / kWordBits] & (1UL << (node % kWordBits))) == 0) {
      continue;
    }
    SCOPED_TRACE(node);
    // Preparation binds exclusively owned, fresh pages. Recreate the backing
    // between nodes so this cannot pass by querying an earlier allocation.
    ASSERT_EQ(amdf_platform_host_backing_destroy(&backing_, byte_length_),
              AMDF_STATUS_OK);
    backing_ = {};
    ASSERT_NO_FATAL_FAILURE(CreateBacking());
    ASSERT_EQ(amdf_platform_host_backing_populate(
                  &backing_, byte_length_,
                  {AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, node},
                  amdf_allocator_system()),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ExpectPresent());
    for (size_t offset = 0; offset < byte_length_; offset += page_size_) {
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

INSTANTIATE_TEST_SUITE_P(
    Geometry, HostBackingTest,
    ::testing::Combine(
        ::testing::Values<uint64_t>(4096, 2u << 20, (2u << 20) + 4096,
                                    (4u << 20) + 4096),
        ::testing::Bool(),
        ::testing::Values(KernelPagePolicy{"6.19.5-arch1-1", true},
                          KernelPagePolicy{"7.0.0-38-generic", true},
                          KernelPagePolicy{"7.1.12-200.fc44.x86_64", true},
                          KernelPagePolicy{"7.2.0-rc1", false},
                          KernelPagePolicy{"7.10.0", false},
                          KernelPagePolicy{"8.0.0", false},
                          KernelPagePolicy{"custom", true})));

}  // namespace
