// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/string_pool.h"

#include <string>
#include <string_view>

#include "iree/testing/gtest.h"

namespace {

TEST(StringPool, OverlappingSlicesRetainStorage) {
  static const char data[] = "amdgpu.flat_atomic_and_b64";
  const loom_string_pool_t pool = {data, sizeof(data) - 1};
  const auto qualified =
      loom_string_pool_get(&pool, LOOM_STRING_REF(0, sizeof(data) - 1));
  const auto mnemonic = loom_string_pool_get(
      &pool, LOOM_STRING_REF(7, sizeof("flat_atomic_and_b64") - 1));
  EXPECT_EQ(std::string_view(qualified.data, qualified.size), data);
  EXPECT_EQ(std::string_view(mnemonic.data, mnemonic.size),
            "flat_atomic_and_b64");
  EXPECT_EQ(mnemonic.data, qualified.data + 7);
  EXPECT_EQ(sizeof(loom_string_ref_t), 4);
}

TEST(StringPool, EmptyAndAbsentAreDistinct) {
  const loom_string_pool_t pool = {"", 0};
  EXPECT_TRUE(loom_string_pool_contains(&pool, LOOM_STRING_REF(0, 0)));
  EXPECT_EQ(loom_string_pool_get(&pool, LOOM_STRING_REF(0, 0)).size, 0);
  EXPECT_FALSE(loom_string_pool_contains(&pool, LOOM_STRING_REF_NONE));
  EXPECT_FALSE(loom_string_pool_contains(&pool, LOOM_STRING_REF(0, 1)));
}

TEST(StringPool, ValidatesCompleteSlices) {
  const loom_string_pool_t pool = {"abc", 3};
  EXPECT_TRUE(loom_string_pool_contains(&pool, LOOM_STRING_REF(1, 2)));
  EXPECT_TRUE(loom_string_pool_contains(&pool, LOOM_STRING_REF(3, 0)));
  EXPECT_FALSE(loom_string_pool_contains(&pool, LOOM_STRING_REF(1, 3)));
  EXPECT_FALSE(loom_string_pool_contains(&pool, LOOM_STRING_REF(4, 0)));
  EXPECT_FALSE(loom_string_pool_contains(&pool, LOOM_STRING_REF(0xFFFFFF, 1)));
}

TEST(StringPool, MaximumLengthAndOffsetsAbove64KiB) {
  std::string data(65536, 'x');
  data.append(255, 'y');
  const loom_string_pool_t pool = {data.data(),
                                   static_cast<uint32_t>(data.size())};
  const auto view = loom_string_pool_get(&pool, LOOM_STRING_REF(65536, 255));
  EXPECT_EQ(view.data, data.data() + 65536);
  EXPECT_EQ(std::string_view(view.data, view.size), std::string(255, 'y'));
}

TEST(StringPool, ByteLengthsPreserveEmbeddedZeroAndUtf8) {
  static const char data[] = "\xC3\xA9\0x";
  const loom_string_pool_t pool = {data, sizeof(data) - 1};
  const auto view = loom_string_pool_get(&pool, LOOM_STRING_REF(0, 4));
  EXPECT_EQ(std::string_view(view.data, view.size), std::string_view(data, 4));
}

}  // namespace
