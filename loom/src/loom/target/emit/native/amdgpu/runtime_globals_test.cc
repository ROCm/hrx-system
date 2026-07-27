// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/runtime_globals.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(AmdgpuRuntimeGlobalsTest, ValidatesRuntimeGlobalBits) {
  IREE_EXPECT_OK(loom_amdgpu_runtime_global_flags_validate(
      LOOM_AMDGPU_RUNTIME_GLOBAL_NONE));
  IREE_EXPECT_OK(loom_amdgpu_runtime_global_flags_validate(
      LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
      LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG |
      LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_runtime_global_flags_validate(
                            (loom_amdgpu_runtime_global_flags_t)0x80000000u));
}

TEST(AmdgpuRuntimeGlobalsTest, ResolvesRequestedSymbols) {
  loom_amdgpu_hsaco_data_symbol_t
      symbols[LOOM_AMDGPU_RUNTIME_GLOBAL_SYMBOL_CAPACITY];
  iree_host_size_t symbol_count = 0;
  loom_amdgpu_runtime_global_symbols(
      LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
          LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG |
          LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG,
      symbols, &symbol_count);

  ASSERT_EQ(symbol_count, 3u);
  EXPECT_EQ(loom_amdgpu_runtime_global_count(
                LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
                LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG |
                LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG),
            symbol_count);

  EXPECT_EQ(StringViewToString(symbols[0].name),
            StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME));
  EXPECT_EQ(symbols[0].byte_length,
            LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH);
  EXPECT_EQ(symbols[0].alignment, LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT);
  EXPECT_EQ(symbols[0].flags, LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE);

  EXPECT_EQ(StringViewToString(symbols[1].name),
            StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_NAME));
  EXPECT_EQ(symbols[1].byte_length,
            LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_BYTE_LENGTH);
  EXPECT_EQ(symbols[1].alignment, LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT);
  EXPECT_EQ(symbols[1].flags, LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE);

  EXPECT_EQ(
      StringViewToString(symbols[2].name),
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME));
  EXPECT_EQ(symbols[2].byte_length,
            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);
  EXPECT_EQ(symbols[2].alignment, LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT);
  EXPECT_EQ(symbols[2].flags, LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE);
}

}  // namespace
}  // namespace loom
