// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/thread_trace.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

constexpr uint32_t kGuard = 0xd8c37f91;

class ThreadTraceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
        {11, 5, 1}, &profile_));
    config_ = {};
    config_.profile = &profile_;
    config_.engine_count = 2;
    config_.engine_mask = 3;
    config_.buffer_length = 1024 * 1024;
    config_.buffer_addresses[0] = UINT64_C(0x0000123456789000);
    config_.buffer_addresses[1] = UINT64_C(0x0000123456889000);
    config_.info_address = UINT64_C(0x0000009a12345670);
    storage_.fill(kGuard);
  }

  uint32_t* words() { return storage_.data() + 1; }

  std::vector<uint32_t> Emitted(uint32_t* end) {
    EXPECT_EQ(storage_.front(), kGuard);
    for (auto* word = end; word != storage_.data() + storage_.size(); ++word) {
      EXPECT_EQ(*word, kGuard);
    }
    return std::vector<uint32_t>(words(), end);
  }

  // GFX11.5 register table under test.
  iree_hal_amd_gpu_pm4_thread_trace_profile_t profile_;
  // Two-engine bracket with distinct buffers.
  iree_hal_amd_gpu_pm4_thread_trace_config_t config_;
  // Command storage with a guard on both sides of the emitted range.
  std::array<uint32_t, 256> storage_;
};

TEST_F(ThreadTraceTest, OnlyTheQualifiedGenerationHasRegisters) {
  iree_hal_amd_gpu_pm4_thread_trace_profile_t profile;
  for (const iree_hal_amd_gpu_gfxip_version_t version :
       {iree_hal_amd_gpu_gfxip_version_t{11, 0, 0},
        iree_hal_amd_gpu_gfxip_version_t{12, 0, 0},
        iree_hal_amd_gpu_gfxip_version_t{10, 3, 0}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
                              version, &profile));
    EXPECT_EQ(profile.control, 0u);
  }
  // The write pointer counts 32-byte units from zero on this generation.
  EXPECT_FALSE(profile_.write_pointer_includes_base);
}

TEST_F(ThreadTraceTest, StartProgramsSizeBeforeBaseAndControlLast) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_thread_trace_config_validate(&config_));
  config_.workgroup_processor = 4;
  config_.simd = 1;
  auto* end = iree_hal_amd_gpu_pm4_thread_trace_emit_start(words(), &config_);
  ASSERT_EQ(end - words(),
            iree_hal_amd_gpu_pm4_thread_trace_start_word_count(&config_));
  EXPECT_EQ(Emitted(end), (std::vector<uint32_t>{
                              // Join earlier compute work.
                              0xc0004600,
                              0x80000407,
                              // SPI_CONFIG_CNTL with the SQG event bits.
                              0xc0017900,
                              0x00000440,
                              0xc362c688,
                              // Engine 0: select, size|base_hi, base, mask,
                              // token mask, control.
                              0xc0017900,
                              0x00000200,
                              0x40000000,
                              0xc0017900,
                              0x000019e9,
                              0x00010001,
                              0xc0017900,
                              0x000019e8,
                              0x23456789,
                              0xc0017900,
                              0x000019ed,
                              0x00015441,
                              0xc0017900,
                              0x000019ee,
                              0x003f0683,
                              0xc0017900,
                              0x000019ec,
                              0x80023d41,
                              // Engine 1.
                              0xc0017900,
                              0x00000200,
                              0x40010000,
                              0xc0017900,
                              0x000019e9,
                              0x00010001,
                              0xc0017900,
                              0x000019e8,
                              0x23456889,
                              0xc0017900,
                              0x000019ed,
                              0x00015441,
                              0xc0017900,
                              0x000019ee,
                              0x003f0683,
                              0xc0017900,
                              0x000019ec,
                              0x80023d41,
                              // Broadcast, then COMPUTE_THREAD_TRACE_ENABLE.
                              0xc0017900,
                              0x00000200,
                              0xe0000000,
                              0xc0017602,
                              0x0000021e,
                              0x00000001,
                          }));
}

TEST_F(ThreadTraceTest, StopWaitsInTheRingBeforeReadingEachEngine) {
  auto* end = iree_hal_amd_gpu_pm4_thread_trace_emit_stop(words(), &config_);
  ASSERT_EQ(end - words(),
            iree_hal_amd_gpu_pm4_thread_trace_stop_word_count(&config_));
  const std::vector<uint32_t> emitted = Emitted(end);
  const std::vector<uint32_t> head(emitted.begin(), emitted.begin() + 7);
  EXPECT_EQ(head, (std::vector<uint32_t>{
                      // Join the traced work, clear the enable, finish.
                      0xc0004600, 0x80000407, 0xc0017602, 0x0000021e,
                      0x00000000, 0xc0004600, 0x00000037}));
  const std::vector<uint32_t> engine0(emitted.begin() + 7,
                                      emitted.begin() + 45);
  EXPECT_EQ(engine0,
            (std::vector<uint32_t>{
                0xc0017900, 0x00000200, 0x40000000,
                // STATUS.FINISH_DONE != 0.
                0xc0053c00, 0x00000004, 0x0000d9f4, 0x00000000, 0x00000000,
                0x00fff000, 0x00000004,
                // Mode off.
                0xc0017900, 0x000019ec, 0x80023d40,
                // STATUS.BUSY == 0.
                0xc0053c00, 0x00000003, 0x0000d9f4, 0x00000000, 0x00000000,
                0x02000000, 0x00000004,
                // WPTR, STATUS, DROPPED_CNTR.
                0xc0044000, 0x00100204, 0x0000d9ef, 0x00000000, 0x12345670,
                0x0000009a, 0xc0044000, 0x00100204, 0x0000d9f4, 0x00000000,
                0x12345674, 0x0000009a, 0xc0044000, 0x00100204, 0x0000d9fa,
                0x00000000, 0x12345678, 0x0000009a}));
  // Engine 1 copies into its own info words.
  EXPECT_EQ(emitted[45 + 2], 0x40010000u);
  EXPECT_EQ(emitted[45 + 24], 0x12345680u);
  const std::vector<uint32_t> tail(emitted.end() - 6, emitted.end());
  EXPECT_EQ(tail, (std::vector<uint32_t>{
                      // Broadcast, then SPI_CONFIG_CNTL back to its default.
                      0xc0017900, 0x00000200, 0xe0000000, 0xc0017900,
                      0x00000440, 0xc062c688}));
}

TEST_F(ThreadTraceTest, UnselectedEnginesGetNoPacketAndNoWords) {
  config_.engine_mask = 2;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_thread_trace_config_validate(&config_));
  EXPECT_EQ(iree_hal_amd_gpu_pm4_thread_trace_start_word_count(&config_), 29u);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_thread_trace_stop_word_count(&config_), 51u);
  auto* end = iree_hal_amd_gpu_pm4_thread_trace_emit_start(words(), &config_);
  const std::vector<uint32_t> emitted = Emitted(end);
  ASSERT_EQ(emitted.size(), 29u);
  // The only select names engine 1.
  EXPECT_EQ(emitted[7], 0x40010000u);
  // Mask bits above the device's engines select nothing.
  config_.engine_mask = 0xfc;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_pm4_thread_trace_config_validate(&config_));
}

TEST_F(ThreadTraceTest, ConfigRejectsUnprogrammableBuffersAndSelectors) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_thread_trace_config_validate(&config_));
  auto expect = [&](iree_status_code_t code) {
    IREE_EXPECT_STATUS_IS(
        code, iree_hal_amd_gpu_pm4_thread_trace_config_validate(&config_));
  };
  const auto valid = config_;
  config_.buffer_length = 64 * 1024;
  expect(IREE_STATUS_OUT_OF_RANGE);
  config_ = valid;
  config_.buffer_length = 512u * 1024 * 1024;
  expect(IREE_STATUS_OUT_OF_RANGE);
  config_ = valid;
  config_.buffer_length += 512;
  expect(IREE_STATUS_OUT_OF_RANGE);
  config_ = valid;
  config_.buffer_addresses[1] += 256;
  expect(IREE_STATUS_INVALID_ARGUMENT);
  config_ = valid;
  config_.buffer_addresses[0] = UINT64_C(1) << 48;
  expect(IREE_STATUS_INVALID_ARGUMENT);
  config_ = valid;
  config_.info_address += 2;
  expect(IREE_STATUS_INVALID_ARGUMENT);
  config_ = valid;
  config_.workgroup_processor = 16;
  expect(IREE_STATUS_OUT_OF_RANGE);
  config_ = valid;
  config_.simd = 4;
  expect(IREE_STATUS_OUT_OF_RANGE);
  config_ = valid;
  config_.engine_count = 9;
  expect(IREE_STATUS_INVALID_ARGUMENT);
}

TEST_F(ThreadTraceTest, InfoWordsDecodeExtentAndLoss) {
  iree_hal_amd_gpu_pm4_thread_trace_result_t result;
  // 0x76 units of 32 bytes, finish done, owner VMID 9, nothing dropped.
  const uint32_t clean[] = {0x00000076, 0x90008000, 0, 0xffffffff};
  iree_hal_amd_gpu_pm4_thread_trace_read_info(&config_, 0, clean, &result);
  EXPECT_TRUE(result.collected);
  EXPECT_TRUE(result.finish_done);
  EXPECT_FALSE(result.write_error);
  EXPECT_FALSE(result.buffer_full);
  EXPECT_EQ(result.written_length, 0x76u * 32);
  EXPECT_EQ(result.dropped_count, 0u);

  // The pointer stops one unit short of a full buffer.
  const uint32_t full[] = {0x00007fff, 0x91008000, 41268, 0xffffffff};
  iree_hal_amd_gpu_pm4_thread_trace_read_info(&config_, 1, full, &result);
  EXPECT_TRUE(result.buffer_full);
  EXPECT_TRUE(result.write_error);
  EXPECT_EQ(result.written_length, UINT64_C(0x7fff) * 32);
  EXPECT_EQ(result.dropped_count, 41268u);

  // A pointer past the buffer never yields bytes outside it.
  const uint32_t wild[] = {0x1fffffff, 0x00000000, 0, 0xffffffff};
  iree_hal_amd_gpu_pm4_thread_trace_read_info(&config_, 0, wild, &result);
  EXPECT_EQ(result.written_length, config_.buffer_length);
  EXPECT_FALSE(result.finish_done);

  // Words the stop sequence never replaced are not a result.
  const uint32_t unwritten[] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
  iree_hal_amd_gpu_pm4_thread_trace_read_info(&config_, 0, unwritten, &result);
  EXPECT_FALSE(result.collected);
  EXPECT_EQ(result.written_length, 0u);
}

}  // namespace
}  // namespace iree::hal::amd::gpu
