// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"

namespace iree::hal::amd::gpu {
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

  // Sentinel outside each emitted packet's exact extent.
  static constexpr uint32_t kGuard = 0xd8c37f91;
  // Command storage with space before and after the returned output range.
  std::array<uint32_t, 128> storage_;
};

TEST_F(CommandsTest, ProgramAddressAndResourceRegisters) {
  const uint32_t address[] = {0x12345678, 0x0000009a};
  auto* end = iree_hal_amd_gpu_pm4_emit_registers(words(), 0x2e0c, 2, address);
  const uint32_t resources[] = {0xe0af0000, 0x0000009e};
  end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e12, 2, resources);
  Expect(end, {0xc0027602, 0x0000020c, 0x12345678, 0x0000009a, 0xc0027602,
               0x00000212, 0xe0af0000, 0x0000009e});
}

TEST_F(CommandsTest, Resource3AddressIsSelectedByLaunchPreparation) {
  const uint32_t value = 0x20;
  // GFX11 and GFX12.0 use 0x2e28; physical GC12.1 (gfx1250) uses 0x2e23.
  auto* end = iree_hal_amd_gpu_pm4_emit_registers(words(), 0x2e28, 1, &value);
  end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e23, 1, &value);
  Expect(end, {0xc0017602, 0x00000228, 0x00000020, 0xc0017602, 0x00000223,
               0x00000020});
}

TEST_F(CommandsTest, CompleteUserRegisterInterval) {
  const uint32_t values[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  };
  Expect(iree_hal_amd_gpu_pm4_emit_registers(words(), 0x2e40, 16, values),
         {0xc0107602, 0x00000240, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
          14, 15});
}

TEST_F(CommandsTest, DirectCapturesWorkitemAndWorkgroupModes) {
  const uint32_t workitems[] = {0xfffffffe, 0x12345, 1};
  auto* end =
      iree_hal_amd_gpu_pm4_emit_dispatch_direct(words(), workitems, 0x8025);
  const uint32_t groups[] = {9, 7, 3};
  end = iree_hal_amd_gpu_pm4_emit_dispatch_direct(end, groups, 0x5);
  Expect(end, {0xc0031502, 0xfffffffe, 0x00012345, 1, 0x00008025, 0xc0031502, 9,
               7, 3, 0x00000005});
}

TEST_F(CommandsTest, DirectPreservesPreparedPartialGroupControl) {
  const uint32_t groups[] = {3, 5, 2};
  Expect(iree_hal_amd_gpu_pm4_emit_dispatch_direct(words(), groups, 0x8007),
         {0xc0031502, 3, 5, 2, 0x00008007});
}

TEST_F(CommandsTest, IndirectUsesAbsoluteAddressAndCompiledWaveMode) {
  auto* end = iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
      words(), UINT64_C(0x00001234fffffff4), 0x8005);
  end = iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
      end, UINT64_C(0x0000567800000000), 0x5);
  Expect(end, {0xc0021602, 0xfffffff4, 0x00001234, 0x00008005, 0xc0021602,
               0x00000000, 0x00005678, 0x00000005});
}

TEST_F(CommandsTest, ExecutionAndVisibilityRemainSeparate) {
  auto* end = iree_hal_amd_gpu_pm4_emit_compute_flush(words());
  end = iree_hal_amd_gpu_pm4_emit_acquire(end, 0xc3a1);
  Expect(end, {0xc0004600, 0x00000407, 0xc0065800, 0, 0xffffffff, 0xff, 0, 0,
               0x0a, 0x0000c3a1});
}

TEST_F(CommandsTest, AcquirePreservesTargetSpecificControls) {
  auto* end = iree_hal_amd_gpu_pm4_emit_acquire(words(), 0xc181);
  end = iree_hal_amd_gpu_pm4_emit_acquire(end, 0x1c1e1);
  Expect(end, {0xc0065800, 0, 0xffffffff, 0xff, 0, 0, 0x0a, 0x0000c181,
               0xc0065800, 0, 0xffffffff, 0xff, 0, 0, 0x0a, 0x0001c1e1});
}

TEST_F(CommandsTest, DmaCompletionDoesNotTouchMemoryOrPfpOnlyFields) {
  Expect(iree_hal_amd_gpu_pm4_emit_dma_flush(words()),
         {0xc0055000, 0, 0, 0, 0, 0, 0});
}

TEST_F(CommandsTest, DmaCopyPreservesByteRangesAndFullCount) {
  auto* end = iree_hal_amd_gpu_pm4_emit_dma_copy(
      words(), UINT64_C(0x1234fffffffd), UINT64_C(0x567800000003), 1);
  end = iree_hal_amd_gpu_pm4_emit_dma_copy(
      end, UINT64_C(0x123400000000), UINT64_C(0x567801000000), 0x03ffffff);
  Expect(end,
         {0xc0055000, 0x60300000, 0xfffffffd, 0x1234, 3, 0x5678, 1, 0xc0055000,
          0x60300000, 0, 0x1234, 0x01000000, 0x5678, 0x03ffffff});
}

TEST_F(CommandsTest, DmaFillCarriesImmediatePatternAndConfirmedWrites) {
  auto* end = iree_hal_amd_gpu_pm4_emit_dma_fill(
      words(), UINT64_C(0x1234fffffffc), 4, 0xdeadbeef);
  end = iree_hal_amd_gpu_pm4_emit_dma_fill(end, UINT64_C(0x567800000000),
                                           0x03fffffc, 0);
  Expect(end, {0xc0055000, 0x40300000, 0xdeadbeef, 0, 0xfffffffc, 0x1234, 4,
               0xc0055000, 0x40300000, 0, 0, 0, 0x5678, 0x03fffffc});
}

TEST_F(CommandsTest, WaitsPreserveEveryComparisonWidthAndMask) {
  struct Comparison {
    // Native semantic selected by admission.
    iree_hal_amd_gpu_pm4_comparison_t value;
    // Complete expected memory-space control word.
    uint32_t control;
  };
  const Comparison comparisons[] = {
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_ALWAYS, 0x10},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_LESS, 0x11},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_LESS_OR_EQUAL, 0x12},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL, 0x13},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_NOT_EQUAL, 0x14},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_GREATER_OR_EQUAL, 0x15},
      {IREE_HAL_AMD_GPU_PM4_COMPARISON_GREATER, 0x16},
  };
  for (const auto& comparison : comparisons) {
    for (uint64_t mask :
         {UINT64_C(0), UINT64_MAX, UINT64_C(0xff00aa5555aa00ff)}) {
      SCOPED_TRACE(comparison.control);
      SCOPED_TRACE(mask);
      storage_.fill(kGuard);
      const uint64_t reference = UINT64_C(0xfedcba9876543210) & mask;
      auto* end = iree_hal_amd_gpu_pm4_emit_wait32(
          words(), UINT64_C(0x00001234fffffffc),
          static_cast<uint32_t>(reference), static_cast<uint32_t>(mask),
          comparison.value);
      end = iree_hal_amd_gpu_pm4_emit_wait64(end, UINT64_C(0x0000567800000008),
                                             reference, mask, comparison.value);
      Expect(
          end,
          {0xc0053c00, comparison.control, 0xfffffffc, 0x1234,
           static_cast<uint32_t>(reference), static_cast<uint32_t>(mask), 4,
           0xc0079300, comparison.control, 8, 0x5678,
           static_cast<uint32_t>(reference),
           static_cast<uint32_t>(reference >> 32), static_cast<uint32_t>(mask),
           static_cast<uint32_t>(mask >> 32), 4});
    }
  }
}

TEST_F(CommandsTest, ReleaseConfirmsGenerationWithoutInterrupt) {
  Expect(iree_hal_amd_gpu_pm4_emit_release32(
             words(), 0x30e, UINT64_C(0x00001234fffffffc), UINT32_MAX),
         {0xc0064900, 0x0030e528, 0x23010000, 0xfffffffc, 0x00001234,
          0xffffffff, 0, 0});
}

TEST_F(CommandsTest, ReleaseCacheControlIsIndependentOfGeneration) {
  auto* end = iree_hal_amd_gpu_pm4_emit_release32(words(), 0, 0, 0);
  end = iree_hal_amd_gpu_pm4_emit_release32(end, 0x304, 0x1000, 1);
  end = iree_hal_amd_gpu_pm4_emit_release32(end, 0x1706, 0x2000, 2);
  Expect(end, {0xc0064900, 0x00000528, 0x23010000, 0,      0, 0, 0, 0,
               0xc0064900, 0x00304528, 0x23010000, 0x1000, 0, 1, 0, 0,
               0xc0064900, 0x01706528, 0x23010000, 0x2000, 0, 2, 0, 0});
}

TEST_F(CommandsTest, Release64PreservesBothValueWords) {
  Expect(iree_hal_amd_gpu_pm4_emit_release64(words(), 0x30e,
                                             UINT64_C(0x00001234fffffff8),
                                             UINT64_C(0xfedcba9876543210)),
         {0xc0064900, 0x0030e528, 0x43010000, 0xfffffff8, 0x00001234,
          0x76543210, 0xfedcba98, 0});
}

TEST_F(CommandsTest, NotifyPreservesMailboxAndInterruptContext) {
  auto* end = iree_hal_amd_gpu_pm4_emit_notify32(
      words(), 0x30e, UINT64_C(0x0000abcd01234568), 0xfedcba98, 0x76543210);
  end = iree_hal_amd_gpu_pm4_emit_notify32(end, 0x304, 0x1000, 0, 1);
  end = iree_hal_amd_gpu_pm4_emit_notify32(end, 0x1706, 0x2000, 1, 0);
  Expect(end, {0xc0064900, 0x0030e528, 0x22010000, 0x01234568, 0x0000abcd,
               0xfedcba98, 0,          0x76543210, 0xc0064900, 0x00304528,
               0x22010000, 0x1000,     0,          0,          0,
               1,          0xc0064900, 0x01706528, 0x22010000, 0x2000,
               0,          1,          0,          0});
}

TEST_F(CommandsTest, Copy64HasIndependentFullAddressesAndConfirmation) {
  Expect(iree_hal_amd_gpu_pm4_emit_copy64(words(), UINT64_C(0x00001234fffffff8),
                                          UINT64_C(0x0000567800000008)),
         {0xc0044000, 0x00110202, 0xfffffff8, 0x00001234, 8, 0x00005678});
}

TEST_F(CommandsTest, Copy32SelectsOnlyOneDestinationWord) {
  Expect(iree_hal_amd_gpu_pm4_emit_copy32(words(), UINT64_C(0x00001234fffffff4),
                                          UINT64_C(0x0000567800000004)),
         {0xc0044000, 0x00100202, 0xfffffff4, 0x00001234, 4, 0x00005678});
}

TEST_F(CommandsTest, WriteCapturesOneOrSeveralWords) {
  uint32_t values[] = {0x12345678, 0x9abcdef0, 0, 0xffffffff};
  auto* end = iree_hal_amd_gpu_pm4_emit_write(
      words(), UINT64_C(0x0000123400000004), 1, values);
  end = iree_hal_amd_gpu_pm4_emit_write(end, UINT64_C(0x0000567800000010), 4,
                                        values);
  values[0] = 0;
  Expect(end,
         {0xc0033700, 0x00100200, 4, 0x00001234, 0x12345678, 0xc0063700,
          0x00100200, 0x10, 0x00005678, 0x12345678, 0x9abcdef0, 0, 0xffffffff});
}

TEST_F(CommandsTest, NopInitializesExactPaddingExtent) {
  auto* end = iree_hal_amd_gpu_pm4_emit_nop(words(), 2);
  end = iree_hal_amd_gpu_pm4_emit_nop(end, 8);
  end = iree_hal_amd_gpu_pm4_emit_nop(end, 9);
  Expect(end, {0xc0001000, 0, 0xc0061000, 0, 0, 0, 0, 0, 0, 0, 0xc0071000, 0, 0,
               0, 0, 0, 0, 0, 0});
}

TEST(CommandsExtentTest, MaximumType3Payload) {
  constexpr size_t kPacketWords = 16385;
  std::vector<uint32_t> words(kPacketWords + 2, 0x12345678);
  std::vector<uint32_t> values(kPacketWords - 4);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = i;
  }
  auto* end = iree_hal_amd_gpu_pm4_emit_write(words.data() + 1, 0x1000,
                                              values.size(), values.data());
  EXPECT_EQ(end, words.data() + 1 + kPacketWords);
  EXPECT_EQ(words[1], 0xffff3700u);
  EXPECT_EQ(words[2], 0x00100200u);
  EXPECT_EQ(words[3], 0x1000u);
  EXPECT_EQ(words[4], 0u);
  EXPECT_EQ(std::vector<uint32_t>(words.begin() + 5, words.end() - 1), values);
  EXPECT_EQ(words.front(), 0x12345678u);
  EXPECT_EQ(words.back(), 0x12345678u);

  end = iree_hal_amd_gpu_pm4_emit_nop(words.data() + 1, kPacketWords);
  EXPECT_EQ(end, words.data() + 1 + kPacketWords);
  EXPECT_EQ(words[1], 0xffff1000u);
  EXPECT_EQ(std::vector<uint32_t>(words.begin() + 2, words.end() - 1),
            std::vector<uint32_t>(kPacketWords - 1, 0));
  EXPECT_EQ(words.front(), 0x12345678u);
  EXPECT_EQ(words.back(), 0x12345678u);
}

}  // namespace
}  // namespace iree::hal::amd::gpu
