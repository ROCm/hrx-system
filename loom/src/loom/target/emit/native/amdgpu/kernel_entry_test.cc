// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_entry.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transformed entry text and fixups.
  iree_arena_allocator_t arena_ = {0};
};

TEST(AmdgpuKernelEntryTest, SelectsInitialVmemReplayEnvelope) {
  static const uint8_t kExpectedText[] = {
      0x00, 0x40, 0x17, 0xee, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x7e, 0x41, 0x06, 0x80, 0xb9, 0x01, 0x00, 0x00, 0x00,
  };
  static constexpr char kExpectedAssembly[] =
      "  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE\n"
      "  v_nop\n"
      "  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1\n";
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile =
      LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY;

  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  EXPECT_EQ(std::string(envelope->assembly.data, envelope->assembly.size),
            kExpectedAssembly);
  ASSERT_EQ(envelope->text.data_length, sizeof(kExpectedText));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(envelope->text.data),
                        envelope->text.data_length),
            std::string(reinterpret_cast<const char*>(kExpectedText),
                        sizeof(kExpectedText)));
  EXPECT_EQ(envelope->instruction_count, 3u);
  EXPECT_EQ(envelope->minimum_sgpr_count, 2u);
  EXPECT_EQ(envelope->minimum_vgpr_count, 1u);
}

TEST(AmdgpuKernelEntryTest, PrependsTextAndDisplacesBodyFixups) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile =
      LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY;
  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd, 0x00, 0x00, 0x00, 0x00};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
      /*.literal_byte_offset=*/4,
      /*.base_pc_byte_offset=*/0,
      /*.target_symbol=*/IREE_SV("target_data"),
      /*.target_symbol_byte_offset=*/12,
  };

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1,
      &text, &fixups, arena.arena()));

  ASSERT_EQ(text.data_length, envelope->text.data_length + sizeof(body));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(text.data),
                        envelope->text.data_length),
            std::string(reinterpret_cast<const char*>(envelope->text.data),
                        envelope->text.data_length));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(
                            text.data + envelope->text.data_length),
                        sizeof(body)),
            std::string(reinterpret_cast<const char*>(body), sizeof(body)));
  ASSERT_NE(fixups, nullptr);
  EXPECT_EQ(fixups[0].literal_byte_offset, 28u);
  EXPECT_EQ(fixups[0].base_pc_byte_offset, 24u);
  EXPECT_TRUE(
      iree_string_view_equal(fixups[0].target_symbol, IREE_SV("target_data")));
  EXPECT_EQ(fixups[0].target_symbol_byte_offset, 12u);
}

TEST(AmdgpuKernelEntryTest, EmptyEnvelopePreservesBodyStorage) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;
  const loom_amdgpu_kernel_entry_envelope_t* envelope =
      loom_amdgpu_kernel_entry_envelope_for_properties(&properties);
  const uint8_t body[] = {0x00, 0x00, 0xb0, 0xbf};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      /*.kind=*/LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
  };

  TestArena arena;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1,
      &text, &fixups, arena.arena()));

  EXPECT_EQ(text.data, body);
  EXPECT_EQ(text.data_length, sizeof(body));
  EXPECT_EQ(fixups, &body_fixup);
}

}  // namespace
}  // namespace loom
