// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

#include <array>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

class CommandsTest : public ::testing::Test {
 protected:
  void SetUp() override { storage_.fill(kGuard); }

  uint32_t* words() { return storage_.data() + 1; }

  void Expect(uint32_t* end, std::initializer_list<uint32_t> expected) {
    ASSERT_EQ(end, words() + expected.size());
    EXPECT_EQ(std::vector<uint32_t>(words(), end),
              std::vector<uint32_t>(expected));
    EXPECT_EQ(storage_.front(), kGuard);
    for (auto* word = end; word != storage_.data() + storage_.size(); ++word) {
      EXPECT_EQ(*word, kGuard);
    }
  }

  // Sentinel outside the exact emitted extent.
  static constexpr uint32_t kGuard = 0xd8c37f91;
  // Complete packet sequences with guards before and after them.
  std::array<uint32_t, 64> storage_;
};

TEST_F(CommandsTest, CopyPreservesByteAddressesAndCountWithoutNpd) {
  auto* end = iree_hal_amd_gpu_sdma_emit_copy(
      words(), 0, UINT64_C(0x1234ffffffff), UINT64_C(0x567800000003), 1);
  end = iree_hal_amd_gpu_sdma_emit_copy(
      end, AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE, UINT64_C(0x123500000001),
      UINT64_C(0x567800004002), IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
  Expect(end, {1, 0, 0, 0xffffffff, 0x1234, 3, 0x5678, 1, 0x3ffeff, 0x0c0c0000,
               1, 0x1235, 0x4002, 0x5678});
}

TEST_F(CommandsTest, FenceUsesOnlyAdmittedNativeFields) {
  struct Encoding {
    // Native capability combination, independent of physical shader ISA.
    amdf_queue_format_features_t features;
    // Complete expected fence header for that combination.
    uint32_t header;
  };
  const Encoding encodings[] = {
      {0, 5},
      {AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE, 0x00030005},
      {AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM, 0x00130005},
      {AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE, 0x03000005},
      {AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
           AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
       0x03030005},
      {AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
           AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
       0x03130005},
  };
  for (const auto& encoding : encodings) {
    SCOPED_TRACE(encoding.features);
    storage_.fill(kGuard);
    Expect(
        iree_hal_amd_gpu_sdma_emit_fence(words(), encoding.features,
                                         UINT64_C(0x1234fffffffc), 0xa5c31f27),
        {encoding.header, 0xfffffffc, 0x1234, 0xa5c31f27});
  }
}

TEST_F(CommandsTest, AcquireAndReleasePreserveDistinctCacheControls) {
  auto* end = iree_hal_amd_gpu_sdma_emit_cache(words(), 0xc3c0);
  end = iree_hal_amd_gpu_sdma_emit_cache(end, 0x8040);
  Expect(end, {0x111, 0, 0xc3c00000, 0, 0, 0x111, 0, 0x80400000, 0, 0});
}

TEST_F(CommandsTest, WaitUsesExactBinaryValueAndAdmittedScope) {
  auto* end =
      iree_hal_amd_gpu_sdma_emit_wait(words(), 0, UINT64_C(0x1234fffffffc), 1);
  end = iree_hal_amd_gpu_sdma_emit_wait(
      end, AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE, UINT64_C(0x567800000008),
      0);
  Expect(end, {0xb0000008, 0xfffffffc, 0x1234, 1, 0xffffffff, 0x0fff0004,
               0xb0000008, 8, 0x5678, 0, 0xffffffff, 0x3fff0004});
}

TEST_F(CommandsTest, FillKeepsDwordSizeCountAndScopeIndependent) {
  auto* end = iree_hal_amd_gpu_sdma_emit_fill(
      words(), 0, UINT64_C(0x1234fffffffc), 4, 0x59e2c17a);
  end = iree_hal_amd_gpu_sdma_emit_fill(
      end, AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE, UINT64_C(0x123500000000),
      IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH, 0x7ac1e259);
  Expect(end, {0x8000000b, 0xfffffffc, 0x1234, 0x59e2c17a, 3, 0x8300000b, 0,
               0x1235, 0x7ac1e259, 0x3ffeff});
}

TEST_F(CommandsTest, TimestampUsesGlobalClockAndFullAddress) {
  auto* end = iree_hal_amd_gpu_sdma_emit_timestamp(words(), 0,
                                                   UINT64_C(0x1234ffffffe0));
  end = iree_hal_amd_gpu_sdma_emit_timestamp(
      end, AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE, UINT64_C(0x123500000020));
  Expect(end, {0x20d, 0xffffffe0, 0x1234, 0x0300020d, 0x20, 0x1235});
}

}  // namespace
