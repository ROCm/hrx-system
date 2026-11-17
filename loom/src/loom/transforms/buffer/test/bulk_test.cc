// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"

// Linked from the same source compiled for each native width policy. Guard
// pages observe machine-code reads as well as writes, including code that is
// outside the host compiler's sanitizer instrumentation.
extern "C" void bulk_copy(const void* source, uint64_t source_offset,
                          void* target, uint64_t target_offset,
                          uint64_t length);

extern "C" void bulk_fill_i8(void* target, uint64_t target_offset,
                             uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_i16(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_i32(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_i64(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_f8E4M3(void* target, uint64_t target_offset,
                                 uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_f8E5M2(void* target, uint64_t target_offset,
                                 uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_f16(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_bf16(void* target, uint64_t target_offset,
                               uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_f32(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);
extern "C" void bulk_fill_f64(void* target, uint64_t target_offset,
                              uint64_t length, uint64_t pattern);

extern "C" int32_t bulk_compare(const void* left, uint64_t left_offset,
                                const void* right, uint64_t right_offset,
                                uint64_t length);

extern "C" int32_t bulk_compare3(const void* left, uint64_t left_offset,
                                 const void* right, uint64_t right_offset);
extern "C" int32_t bulk_compare5(const void* left, uint64_t left_offset,
                                 const void* right, uint64_t right_offset);

extern "C" void signed_copy(const void* source, int64_t source_position,
                            void* target, int64_t target_position,
                            uint64_t length);
extern "C" void signed_fill(void* target, int64_t position, uint64_t length,
                            uint32_t pattern);
extern "C" int32_t signed_compare(const void* left, int64_t left_position,
                                  const void* right, int64_t right_position,
                                  uint64_t length);
extern "C" int32_t signed_byte(const void* source, int64_t position,
                               void* target, uint64_t target_position);
extern "C" bool signed_address_equal(const void* left, int64_t left_position,
                                     const void* right,
                                     uint64_t right_position);

namespace {

class GuardedPage {
 public:
  GuardedPage() {
    mapping_ = mmap(nullptr, 3 * page_size_, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ != MAP_FAILED) {
      ready_ = mprotect(data(), page_size_, PROT_READ | PROT_WRITE) == 0;
    }
  }
  ~GuardedPage() {
    if (mapping_ != MAP_FAILED) {
      munmap(mapping_, 3 * page_size_);
    }
  }
  bool ready() const { return ready_; }
  uint8_t* data() const { return static_cast<uint8_t*>(mapping_) + page_size_; }
  size_t size() const { return page_size_; }

 private:
  // Host page size used by both accessible payload and inaccessible guards.
  size_t page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  // Owned mapping: one accessible page between two inaccessible pages.
  void* mapping_ = MAP_FAILED;
  // Whether allocation and protection setup both succeeded.
  bool ready_ = false;
};

class BulkMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if LOOM_TEST_VECTOR_BITS == 512
    if (!__builtin_cpu_supports("avx512f") ||
        !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") ||
        !__builtin_cpu_supports("avx512vl")) {
      GTEST_SKIP() << "AVX512 execution requires supporting hardware";
    }
#elif LOOM_TEST_VECTOR_BITS == 256
    if (!__builtin_cpu_supports("avx2")) {
      GTEST_SKIP() << "AVX2 execution requires supporting hardware";
    }
#endif
  }
};

TEST_F(BulkMemoryTest, SignedInteriorOrigins) {
  alignas(64) std::array<uint8_t, 256> source;
  alignas(64) std::array<uint8_t, 256> target;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (i * 29 + 17) & 255;
  }
  for (int64_t position : {-64, -31, -1, 0, 1, 31, 64}) {
    for (uint64_t length : {0, 1, 3, 4, 15, 16, 31, 64}) {
      target.fill(0xA5);
      auto expected = target;
      const auto start = static_cast<size_t>(64 + position);
      std::memcpy(expected.data() + start, source.data() + start, length);
      signed_copy(source.data() + 64, position, target.data() + 64, position,
                  length);
      ASSERT_EQ(target, expected);
      ASSERT_EQ(signed_compare(source.data() + 64, position,
                               target.data() + start, 0, length),
                0);
      const uint32_t pattern = 0x1234ABCD;
      for (size_t i = 0; i < length; ++i) {
        expected[start + i] = pattern >> (8 * (i % 4));
      }
      signed_fill(target.data() + 64, position, length, pattern);
      ASSERT_EQ(target, expected);
    }
    const auto start = static_cast<size_t>(64 + position);
    ASSERT_EQ(signed_byte(source.data() + 64, position, target.data(), 7),
              source[start]);
    ASSERT_EQ(target[7], source[start]);
    ASSERT_TRUE(signed_address_equal(source.data() + 64, position,
                                     source.data(), start));
    ASSERT_FALSE(signed_address_equal(source.data() + 64, position,
                                      source.data(), start + 1));
  }
}

TEST_F(BulkMemoryTest, CopyExactRange) {
  alignas(64) std::array<uint8_t, 256> source;
  alignas(64) std::array<uint8_t, 256> target;
  std::array<uint8_t, 256> expected;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (i * 29 + 17) & 255;
  }
  for (size_t length = 0; length <= 129; ++length) {
    for (size_t source_offset = 0; source_offset < 64; ++source_offset) {
      const size_t target_offset = (3 * source_offset + length + 7) % 64;
      target.fill(0xA5);
      expected.fill(0xA5);
      std::memcpy(expected.data() + target_offset,
                  source.data() + source_offset, length);
      bulk_copy(source.data(), source_offset, target.data(), target_offset,
                length);
      ASSERT_EQ(target, expected)
          << "length=" << length << "source offset=" << source_offset
          << "target offset=" << target_offset;
    }
  }
}

TEST_F(BulkMemoryTest, CopyDoesNotAccessOutsideRange) {
  GuardedPage source;
  GuardedPage target;
  ASSERT_TRUE(source.ready());
  ASSERT_TRUE(target.ready());
  for (size_t i = 0; i < source.size(); ++i) {
    source.data()[i] = i & 255;
  }
  bulk_copy(nullptr, 0, nullptr, 0, 0);
  for (size_t length = 0; length <= 129; ++length) {
    const size_t source_offset = source.size() - length;
    const size_t target_offset = target.size() - length;
    std::memset(target.data(), 0xA5, target.size());
    bulk_copy(source.data(), source_offset, target.data(), target_offset,
              length);
    EXPECT_EQ(std::memcmp(source.data() + source_offset,
                          target.data() + target_offset, length),
              0);
    EXPECT_EQ(target.data()[target_offset - 1], 0xA5);
  }
}

struct FillCase {
  // Authored pattern type, included in failing expectations.
  const char* name;
  // Period of the pattern's raw little-endian bytes.
  size_t byte_count;
  // Native function compiled from the corresponding authored fill operation.
  void (*fill)(void*, uint64_t, uint64_t, uint64_t);
};

const FillCase kFillCases[] = {
    {"i8", 1, bulk_fill_i8},         {"i16", 2, bulk_fill_i16},
    {"i32", 4, bulk_fill_i32},       {"i64", 8, bulk_fill_i64},
    {"f8E4M3", 1, bulk_fill_f8E4M3}, {"f8E5M2", 1, bulk_fill_f8E5M2},
    {"f16", 2, bulk_fill_f16},       {"bf16", 2, bulk_fill_bf16},
    {"f32", 4, bulk_fill_f32},       {"f64", 8, bulk_fill_f64},
};

TEST_F(BulkMemoryTest, FillExactPatternAndRange) {
  const uint64_t patterns[] = {0, UINT64_MAX, 0xFEDCBA9876543210ull,
                               0x7FF000007FC0FF80ull};
  alignas(64) std::array<uint8_t, 256> target;
  std::array<uint8_t, 256> expected;
  for (const FillCase& test : kFillCases) {
    SCOPED_TRACE(test.name);
    for (uint64_t pattern : patterns) {
      for (size_t length = 0; length <= 129; ++length) {
        for (size_t offset = 0; offset < 16; ++offset) {
          target.fill(0xA5);
          expected.fill(0xA5);
          for (size_t i = 0; i < length; ++i) {
            expected[offset + i] = pattern >> (8 * (i % test.byte_count));
          }
          test.fill(target.data(), offset, length, pattern);
          ASSERT_EQ(target, expected)
              << "length=" << length << " offset=" << offset;
        }
      }
    }
  }
}

TEST_F(BulkMemoryTest, FillDoesNotAccessOutsideRange) {
  GuardedPage target;
  ASSERT_TRUE(target.ready());
  constexpr uint64_t pattern = 0xFEDCBA9876543210ull;
  for (const FillCase& test : kFillCases) {
    SCOPED_TRACE(test.name);
    test.fill(nullptr, 0, 0, pattern);
    for (size_t length = 0; length <= 129; ++length) {
      const size_t offset = target.size() - length;
      std::memset(target.data(), 0xA5, target.size());
      test.fill(target.data(), offset, length, pattern);
      for (size_t i = 0; i < length; ++i) {
        EXPECT_EQ(target.data()[offset + i],
                  static_cast<uint8_t>(pattern >> (8 * (i % test.byte_count))));
      }
      EXPECT_EQ(target.data()[offset - 1], 0xA5);
    }
  }
}

TEST_F(BulkMemoryTest, CompareFirstDifferentUnsignedByte) {
  alignas(64) std::array<uint8_t, 256> left;
  alignas(64) std::array<uint8_t, 256> right;
  for (size_t length = 0; length <= 129; ++length) {
    for (size_t left_offset = 0; left_offset < 16; ++left_offset) {
      const size_t right_offset = (3 * left_offset + length + 7) % 64;
      left.fill(0xA5);
      right.fill(0xA5);
      ASSERT_EQ(bulk_compare(left.data(), left_offset, right.data(),
                             right_offset, length),
                0);
      for (size_t first = 0; first < length; ++first) {
        left[left_offset + first] = 0x80;
        right[right_offset + first] = 0x7F;
        // A later byte orders the entire little-endian word in the opposite
        // direction. The first byte still determines lexicographic order.
        if (first + 1 < length) {
          left[left_offset + first + 1] = 0;
          right[right_offset + first + 1] = 255;
        }
        ASSERT_EQ(bulk_compare(left.data(), left_offset, right.data(),
                               right_offset, length),
                  1)
            << "length=" << length << " first=" << first;
        ASSERT_EQ(bulk_compare(right.data(), right_offset, left.data(),
                               left_offset, length),
                  -1);
        left[left_offset + first] = right[right_offset + first] = 0xA5;
        if (first + 1 < length) {
          left[left_offset + first + 1] = right[right_offset + first + 1] =
              0xA5;
        }
      }
    }
  }
}

TEST_F(BulkMemoryTest, CompareFixedShortWords) {
  GuardedPage left;
  GuardedPage right;
  ASSERT_TRUE(left.ready());
  ASSERT_TRUE(right.ready());
  for (size_t length : {3u, 5u}) {
    auto compare = length == 3 ? bulk_compare3 : bulk_compare5;
    const size_t left_offset = left.size() - length;
    const size_t right_offset = right.size() - length;
    for (size_t first = 0; first <= length; ++first) {
      std::memset(left.data(), 0xA5, left.size());
      std::memset(right.data(), 0xA5, right.size());
      if (first < length) {
        left.data()[left_offset + first] = 0x80;
        right.data()[right_offset + first] = 0x7F;
      }
      if (first + 1 < length) {
        left.data()[left_offset + first + 1] = 0;
        right.data()[right_offset + first + 1] = 255;
      }
      EXPECT_EQ(compare(left.data(), left_offset, right.data(), right_offset),
                first == length ? 0 : 1);
      EXPECT_EQ(compare(right.data(), right_offset, left.data(), left_offset),
                first == length ? 0 : -1);
    }
  }
}

TEST_F(BulkMemoryTest, CompareDoesNotAccessOutsideRange) {
  GuardedPage left;
  GuardedPage right;
  ASSERT_TRUE(left.ready());
  ASSERT_TRUE(right.ready());
  EXPECT_EQ(bulk_compare(nullptr, 0, nullptr, 0, 0), 0);
  for (size_t length = 0; length <= 129; ++length) {
    const size_t left_offset = left.size() - length;
    const size_t right_offset = right.size() - length;
    std::memset(left.data(), 0xA5, left.size());
    std::memset(right.data(), 0xA5, right.size());
    EXPECT_EQ(bulk_compare(left.data(), left_offset, right.data(), right_offset,
                           length),
              0);
    if (length) {
      left.data()[left.size() - 1] = 0x80;
      right.data()[right.size() - 1] = 0x7F;
      EXPECT_EQ(bulk_compare(left.data(), left_offset, right.data(),
                             right_offset, length),
                1);
    }
  }
}

}  // namespace
