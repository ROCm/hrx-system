// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory_copy.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

std::vector<size_t> CopyLengths() {
  std::vector<size_t> lengths;
  for (size_t length = 0; length <= 272; ++length) {
    lengths.push_back(length);
  }
  for (size_t length : {511, 512, 513, 4095, 4096, 4097, 8191, 8192, 8193,
                        16383, 16384, 16385}) {
    lengths.push_back(length);
  }
  return lengths;
}

TEST(WriteOnlyCopy, Empty) { iree_memcpy_write_only(nullptr, nullptr, 0); }

TEST(WriteOnlyCopy, ExactExtentAtEveryAlignment) {
  for (size_t length : CopyLengths()) {
    for (size_t source_offset = 0; source_offset < 16; ++source_offset) {
      // The allocation ends at the copy extent so ASAN catches wide tail loads.
      std::vector<uint8_t> source(length + source_offset);
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i * 37 + length);
      }
      const auto original = source;
      for (size_t target_offset = 0; target_offset < 16; ++target_offset) {
        std::vector<uint8_t> target(length + target_offset + 16, 0x5a);
        const uint8_t* input =
            source.empty() ? nullptr : source.data() + source_offset;
        iree_memcpy_write_only(target.data() + target_offset, input, length);
        for (size_t i = 0; i < target.size(); ++i) {
          const uint8_t expected =
              i >= target_offset && i - target_offset < length
                  ? source[source_offset + i - target_offset]
                  : 0x5a;
          ASSERT_EQ(target[i], expected)
              << "length=" << length << " source=" << source_offset
              << " target=" << target_offset << " index=" << i;
        }
        ASSERT_EQ(source, original);
      }
    }
  }
}

TEST(WriteOnlyFill, Empty) { iree_memset_write_only(nullptr, 0xff, 0); }

TEST(WriteOnlyFill, ExactExtentAtEveryAlignment) {
  for (size_t length : CopyLengths()) {
    for (size_t target_offset = 0; target_offset < 16; ++target_offset) {
      for (uint8_t value : {0x00, 0x01, 0xa5, 0xff}) {
        std::vector<uint8_t> target(length + target_offset + 16, 0x5a);
        iree_memset_write_only(target.data() + target_offset, value, length);
        for (size_t i = 0; i < target.size(); ++i) {
          EXPECT_EQ(target[i], i >= target_offset && i - target_offset < length
                                   ? value
                                   : 0x5a)
              << "length=" << length << " target=" << target_offset
              << " value=" << static_cast<int>(value) << " index=" << i;
        }
      }
    }
  }
}

}  // namespace
