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

  TestArena arena;
  const loom_amdgpu_hal_kernel_abi_verify_result_t abi = {};
  const loom_amdgpu_kernel_entry_envelope_t* envelope = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_envelope_build(
      &properties, &abi, 0, arena.arena(), &envelope));
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

TEST(AmdgpuKernelEntryTest, InitializesAbsoluteFlatScratchFromRetainedLaunch) {
  TestArena arena;
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_descriptor.flags =
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_USER_FLAT_SCRATCH_INIT;
  const loom_amdgpu_hal_kernel_abi_verify_result_t abi = {
      .user_sgpr_count = 6,
      .flat_scratch_sgprs = {.init = 4, .wavefront_offset = 9},
  };
  const loom_amdgpu_kernel_entry_envelope_t* envelope = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_envelope_build(
      &properties, &abi, 32, arena.arena(), &envelope));
  // Independently assembled by LLVM for gfx90a in both feature modes.
  const uint8_t expected[] = {0x04, 0x09, 0x66, 0x80, 0x05, 0x80, 0x67, 0x82};
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(envelope->text.data),
                  envelope->text.data_length),
      std::string(reinterpret_cast<const char*>(expected), sizeof(expected)));
  EXPECT_EQ(std::string(envelope->assembly.data, envelope->assembly.size),
            "  s_add_u32 flat_scratch_lo, s4, s9\n"
            "  s_addc_u32 flat_scratch_hi, s5, 0\n");
  EXPECT_EQ(envelope->instruction_count, 2u);
  EXPECT_EQ(envelope->minimum_sgpr_count, 10u);
  EXPECT_EQ(envelope->minimum_vgpr_count, 0u);

  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_envelope_build(
      &properties, &abi, 0, arena.arena(), &envelope));
  EXPECT_EQ(envelope->instruction_count, 0u);
  EXPECT_EQ(envelope->text.data_length, 0u);
  EXPECT_EQ(envelope->assembly.size, 0u);
  EXPECT_EQ(envelope->minimum_sgpr_count, 0u);
}

TEST(AmdgpuKernelEntryTest, PrependsTextAndDisplacesBodyFixups) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile =
      LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY;
  TestArena arena;
  const loom_amdgpu_hal_kernel_abi_verify_result_t abi = {};
  const loom_amdgpu_kernel_entry_envelope_t* envelope = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_envelope_build(
      &properties, &abi, 0, arena.arena(), &envelope));
  const uint8_t body[] = {0xaa, 0xbb, 0xcc, 0xdd, 0x00, 0x00, 0x00, 0x00};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      .kind = LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
      .target_data_symbol_ordinal = 7,
      .literal_byte_offset = 4,
      .base_pc_byte_offset = 0,
      .target_symbol_byte_offset = 12,
  };

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1,
      arena.arena(), &text, &fixups));

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
  EXPECT_EQ(fixups[0].target_data_symbol_ordinal, 7u);
  EXPECT_EQ(fixups[0].target_symbol_byte_offset, 12u);
}

TEST(AmdgpuKernelEntryTest, EmptyEnvelopePreservesBodyStorage) {
  loom_amdgpu_processor_properties_t properties = {};
  properties.kernel_entry.profile = LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE;
  TestArena arena;
  const loom_amdgpu_hal_kernel_abi_verify_result_t abi = {};
  const loom_amdgpu_kernel_entry_envelope_t* envelope = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_envelope_build(
      &properties, &abi, 0, arena.arena(), &envelope));
  const uint8_t body[] = {0x00, 0x00, 0xb0, 0xbf};
  const loom_amdgpu_hsaco_text_fixup_t body_fixup = {
      .kind = LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
  };

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_hsaco_text_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_kernel_entry_prepend_text(
      envelope, iree_make_const_byte_span(body, sizeof(body)), &body_fixup, 1,
      arena.arena(), &text, &fixups));

  EXPECT_EQ(text.data, body);
  EXPECT_EQ(text.data_length, sizeof(body));
  EXPECT_EQ(fixups, &body_fixup);
}

}  // namespace
}  // namespace loom
