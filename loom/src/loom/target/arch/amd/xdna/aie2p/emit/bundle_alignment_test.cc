// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_alignment.h"

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

class Aie2pBundleAlignmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Backing pool for alignment scratch allocations.
  iree_arena_block_pool_t block_pool_;
  // Scratch arena presented to the alignment component.
  iree_arena_allocator_t arena_;
};

TEST_F(Aie2pBundleAlignmentTest, SelectsLatestOptimalExactWidening) {
  // A two-byte prefix and 106 four-byte store packets leave the endpoint six
  // bytes short of its next 16-byte boundary. Every store can supply that
  // exact widening at the same minimum slot cost.
  constexpr iree_host_size_t kRegionPacketCount = 106;
  constexpr iree_host_size_t kBundleCount = kRegionPacketCount + 1;
  constexpr iree_host_size_t kSlotCapacity = kBundleCount + 1;

  const loom_aie2p_slot_t nop_slot = LOOM_AIE2P_SLOT_NOP;
  const loom_aie2p_bundle_format_id_t nop_format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&nop_slot, 1);
  ASSERT_NE(nop_format, LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
  loom_aie2p_bundle_format_info_t nop_format_info;
  ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(nop_format,
                                                           &nop_format_info));
  ASSERT_EQ(nop_format_info.bit_count, 16u);

  const loom_aie2p_slot_t store_slot = LOOM_AIE2P_SLOT_ST;
  const loom_aie2p_bundle_format_id_t store_format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&store_slot, 1);
  ASSERT_NE(store_format, LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
  loom_aie2p_bundle_format_info_t store_format_info;
  ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(store_format,
                                                           &store_format_info));
  ASSERT_EQ(store_format_info.bit_count, 32u);

  std::array<loom_aie2p_planned_bundle_t, kBundleCount> bundles = {};
  std::array<loom_aie2p_planned_slot_t, kSlotCapacity> slots = {};
  std::array<loom_aie2p_slot_realization_t, kSlotCapacity> realizations = {};
  bundles[0] = {
      .issue_cycle = 0,
      .block_index = 0,
      .logical_issue_cycle = 0,
      .byte_offset = 0,
      .slot_start = 0,
      .format = nop_format,
      .slot_count = 1,
  };
  slots[0] = {
      .encoded_slot = {LOOM_AIE2P_SLOT_NOP, 0},
      .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
      .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
  };
  for (iree_host_size_t packet_index = 0; packet_index < kRegionPacketCount;
       ++packet_index) {
    const iree_host_size_t bundle_index = packet_index + 1;
    bundles[bundle_index] = {
        .issue_cycle = (uint32_t)bundle_index,
        .block_index = 0,
        .logical_issue_cycle = (uint32_t)bundle_index,
        .byte_offset = (uint32_t)(2 + packet_index * 4),
        .slot_start = (uint32_t)bundle_index,
        .format = store_format,
        .slot_count = 1,
    };
    slots[bundle_index] = {
        .encoded_slot = {LOOM_AIE2P_SLOT_ST, 0},
        .scheduled_packet_index = (uint32_t)packet_index,
        .flags = 0,
    };
  }

  loom_aie2p_bundle_alignment_storage_t storage = {
      .scratch_arena = &arena_,
      .bundles = bundles.data(),
      .bundle_count = bundles.size(),
      .slots = slots.data(),
      .slot_capacity = slots.size(),
      .slot_count = kBundleCount,
      .slot_realizations = realizations.data(),
      .encoded_byte_length = 2 + kRegionPacketCount * 4,
      .trailing_nop_byte_length = 0,
  };
  bool aligned = false;
  IREE_ASSERT_OK(loom_aie2p_bundle_alignment_try_loop_boundary(
      &storage, /*region_bundle_start=*/1,
      /*boundary_bundle_index=*/storage.bundle_count,
      /*minimum_added_byte_count=*/0, &aligned));

  ASSERT_TRUE(aligned);
  EXPECT_EQ(storage.encoded_byte_length, 432u);
  EXPECT_EQ(storage.slot_count, kSlotCapacity);
  EXPECT_EQ(storage.trailing_nop_byte_length, 0u);
  for (iree_host_size_t packet_index = 0; packet_index + 1 < kRegionPacketCount;
       ++packet_index) {
    const loom_aie2p_planned_bundle_t& bundle = bundles[packet_index + 1];
    EXPECT_EQ(bundle.format, store_format);
    EXPECT_EQ(bundle.slot_count, 1u);
    EXPECT_EQ(bundle.byte_offset, 2u + packet_index * 4u);
  }

  const loom_aie2p_planned_bundle_t& widened_bundle = bundles.back();
  loom_aie2p_bundle_format_info_t widened_format_info;
  ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(
      widened_bundle.format, &widened_format_info));
  EXPECT_EQ(widened_bundle.byte_offset, 422u);
  EXPECT_EQ(widened_format_info.bit_count, 80u);
  ASSERT_EQ(widened_bundle.slot_count, 2u);
  uint32_t store_slot_count = 0;
  uint32_t synthetic_slot_count = 0;
  for (uint8_t slot_offset = 0; slot_offset < widened_bundle.slot_count;
       ++slot_offset) {
    const loom_aie2p_planned_slot_t& slot =
        slots[widened_bundle.slot_start + slot_offset];
    store_slot_count += slot.encoded_slot.slot == LOOM_AIE2P_SLOT_ST;
    synthetic_slot_count += iree_any_bit_set(
        slot.flags, LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
    if (slot.encoded_slot.slot == LOOM_AIE2P_SLOT_ST) {
      EXPECT_EQ(slot.scheduled_packet_index, kRegionPacketCount - 1u);
    }
  }
  EXPECT_EQ(store_slot_count, 1u);
  EXPECT_EQ(synthetic_slot_count, 1u);
}

TEST_F(Aie2pBundleAlignmentTest,
       WidensToMinimumExtentWithoutExtraIssuePositions) {
  const loom_aie2p_slot_t store_slot = LOOM_AIE2P_SLOT_ST;
  const auto format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&store_slot, 1);
  for (uint32_t prefix : {0u, 2u, 6u, 14u}) {
    for (uint8_t minimum : {0, 1, 6, 16, 31, 64, 96, 112}) {
      SCOPED_TRACE(::testing::Message()
                   << "prefix=" << prefix << " minimum=" << unsigned(minimum));
      constexpr uint32_t kRegionPackets = 16;
      constexpr uint32_t kBundleCount = kRegionPackets + 1;
      constexpr uint32_t kSlotCapacity =
          kBundleCount + LOOM_AIE2P_BUNDLE_ALIGNMENT_LOOP_MAX_ADDED_SLOTS;
      std::array<loom_aie2p_planned_bundle_t, kBundleCount> bundles = {};
      std::array<loom_aie2p_planned_slot_t, kSlotCapacity> slots = {};
      std::array<loom_aie2p_slot_realization_t, kSlotCapacity> realizations =
          {};
      for (uint32_t i = 0; i < kBundleCount; ++i) {
        bundles[i] = {
            .issue_cycle = i,
            .logical_issue_cycle = i,
            .byte_offset = prefix + 4 * i,
            .slot_start = i,
            .format = format,
            .slot_count = 1,
        };
        slots[i] = {
            .encoded_slot = {store_slot, 0},
            .scheduled_packet_index = i,
        };
        realizations[i].descriptor_ordinal = 17 + i;
      }
      const uint32_t boundary = prefix + 4 * kRegionPackets;
      loom_aie2p_bundle_alignment_storage_t storage = {
          .scratch_arena = &arena_,
          .bundles = bundles.data(),
          .bundle_count = bundles.size(),
          .slots = slots.data(),
          .slot_capacity = slots.size(),
          .slot_count = kBundleCount,
          .slot_realizations = realizations.data(),
          .encoded_byte_length = boundary + 4,
      };
      bool aligned = false;
      IREE_ASSERT_OK(loom_aie2p_bundle_alignment_try_loop_boundary(
          &storage, 0, kRegionPackets, minimum, &aligned));
      ASSERT_TRUE(aligned);
      EXPECT_EQ(storage.bundle_count, kBundleCount);
      EXPECT_EQ(bundles.back().byte_offset,
                iree_host_align(boundary + minimum, 16));
      EXPECT_EQ(storage.encoded_byte_length, bundles.back().byte_offset + 4);
      for (uint32_t i = 0; i < kBundleCount; ++i) {
        EXPECT_EQ(bundles[i].issue_cycle, i);
        EXPECT_EQ(bundles[i].logical_issue_cycle, i);
        uint32_t useful = 0;
        for (uint8_t j = 0; j < bundles[i].slot_count; ++j) {
          const uint32_t index = bundles[i].slot_start + j;
          if (slots[index].flags & LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP) {
            continue;
          }
          ++useful;
          EXPECT_EQ(slots[index].encoded_slot.slot, store_slot);
          EXPECT_EQ(slots[index].scheduled_packet_index, i);
          EXPECT_EQ(realizations[index].descriptor_ordinal, 17 + i);
        }
        EXPECT_EQ(useful, 1u);
      }
    }
  }
}

TEST_F(Aie2pBundleAlignmentTest, KeepsStreamWhenRequiredExtentCannotBeEncoded) {
  const loom_aie2p_slot_t store_slot = LOOM_AIE2P_SLOT_ST;
  const auto format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&store_slot, 1);
  loom_aie2p_planned_bundle_t bundle = {
      .issue_cycle = 7,
      .byte_offset = 0,
      .format = format,
      .slot_count = 1,
  };
  std::array<loom_aie2p_planned_slot_t, 32> slots = {};
  slots[0] = {.encoded_slot = {store_slot, 0}, .scheduled_packet_index = 3};
  std::array<loom_aie2p_slot_realization_t, 32> realizations = {};
  realizations[0].descriptor_ordinal = 17;
  loom_aie2p_bundle_alignment_storage_t storage = {
      .scratch_arena = &arena_,
      .bundles = &bundle,
      .bundle_count = 1,
      .slots = slots.data(),
      .slot_capacity = slots.size(),
      .slot_count = 1,
      .slot_realizations = realizations.data(),
      .encoded_byte_length = 4,
  };
  bool aligned = true;
  IREE_ASSERT_OK(loom_aie2p_bundle_alignment_try_loop_boundary(&storage, 0, 1,
                                                               32, &aligned));
  EXPECT_FALSE(aligned);
  EXPECT_EQ(storage.encoded_byte_length, 4u);
  EXPECT_EQ(storage.slot_count, 1u);
  EXPECT_EQ(storage.bundle_count, 1u);
  EXPECT_EQ(bundle.format, format);
  EXPECT_EQ(bundle.issue_cycle, 7u);
  EXPECT_EQ(bundle.byte_offset, 0u);
  EXPECT_EQ(slots[0].scheduled_packet_index, 3u);
  EXPECT_EQ(realizations[0].descriptor_ordinal, 17u);
}

TEST_F(Aie2pBundleAlignmentTest,
       InsertsSparseGapWhenDenseCallWindowCannotWidenEnough) {
  // This format-valid closed call window can add every even byte count through
  // twelve by widening its packets, but not fourteen. A seven-NOP stream prefix
  // makes fourteen the required residue. Alignment must add one more implicit
  // NOP before the call, then use the reachable twelve-byte widening.
  constexpr std::array<const char*, 6> kFormatNames = {
      "I96_LDA_LNG_VEC",
      "I96_LDA_ST_ALU_MV",
      "I96_LDA_ST_ALU_MV",
      "I128_LDA_LDB_ST_ALU_MV_VEC",
      "I128_LDA_LDB_ST_ALU_MV_VEC",
      "I128_LDA_LDB_ST_ALU_MV_VEC",
  };
  constexpr uint32_t kCallIssueCycle = 7;
  constexpr uint32_t kContinuationCycle = kCallIssueCycle + kFormatNames.size();
  constexpr iree_host_size_t kSlotCapacity =
      kFormatNames.size() * LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT;

  std::array<loom_aie2p_planned_bundle_t, kFormatNames.size()> bundles = {};
  std::array<loom_aie2p_planned_slot_t, kSlotCapacity> slots = {};
  std::array<loom_aie2p_slot_realization_t, kSlotCapacity> realizations = {};
  uint32_t byte_offset = kCallIssueCycle * 2;
  iree_host_size_t slot_count = 0;
  for (iree_host_size_t packet_index = 0; packet_index < kFormatNames.size();
       ++packet_index) {
    const loom_aie2p_bundle_format_id_t format =
        loom_aie2p_encoding_find_bundle_format(
            iree_make_cstring_view(kFormatNames[packet_index]));
    ASSERT_NE(format, LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
    loom_aie2p_bundle_format_info_t format_info;
    ASSERT_TRUE(
        loom_aie2p_encoding_query_bundle_format_info(format, &format_info));
    loom_aie2p_slot_t format_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    const iree_host_size_t format_slot_count =
        loom_aie2p_encoding_query_bundle_format_slots(
            format, IREE_ARRAYSIZE(format_slots), format_slots);
    ASSERT_EQ(format_slot_count, format_info.slot_count);
    bundles[packet_index] = {
        .issue_cycle = (uint32_t)(kCallIssueCycle + packet_index),
        .block_index = 0,
        .logical_issue_cycle = (uint32_t)packet_index,
        .byte_offset = byte_offset,
        .slot_start = (uint32_t)slot_count,
        .format = format,
        .slot_count = (uint8_t)format_slot_count,
    };
    for (iree_host_size_t slot_index = 0; slot_index < format_slot_count;
         ++slot_index) {
      ASSERT_LT(slot_count, slots.size());
      const loom_aie2p_slot_t physical_slot = format_slots[slot_index];
      slots[slot_count++] = {
          .encoded_slot = {physical_slot, 0},
          .scheduled_packet_index = (uint32_t)packet_index,
          .flags = static_cast<loom_aie2p_planned_slot_flags_t>(
              physical_slot == LOOM_AIE2P_SLOT_LNG
                  ? LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL
                  : 0),
      };
    }
    byte_offset += format_info.bit_count / 8;
  }
  const uint32_t unaligned_byte_length = byte_offset;
  ASSERT_EQ(unaligned_byte_length % 16u, 2u);
  const iree_host_size_t original_slot_count = slot_count;

  loom_aie2p_bundle_alignment_storage_t storage = {
      .scratch_arena = &arena_,
      .bundles = bundles.data(),
      .bundle_count = bundles.size(),
      .slots = slots.data(),
      .slot_capacity = slots.size(),
      .slot_count = slot_count,
      .slot_realizations = realizations.data(),
      .encoded_byte_length = unaligned_byte_length,
      .trailing_nop_byte_length = 0,
  };
  uint8_t inserted_nop_count = 0;
  IREE_ASSERT_OK(loom_aie2p_bundle_alignment_align_call_continuation(
      &storage, /*call_bundle_index=*/0, kContinuationCycle,
      &inserted_nop_count));

  EXPECT_EQ(inserted_nop_count, 1u);
  EXPECT_EQ(storage.encoded_byte_length, unaligned_byte_length + 14u);
  EXPECT_EQ(storage.encoded_byte_length % 16u, 0u);
  EXPECT_EQ(storage.bundle_count, bundles.size());
  EXPECT_EQ(storage.trailing_nop_byte_length, 0u);
  EXPECT_EQ(bundles.front().byte_offset, (kCallIssueCycle + 1u) * 2u);
  iree_host_size_t useful_slot_count = 0;
  for (iree_host_size_t packet_index = 0; packet_index < bundles.size();
       ++packet_index) {
    EXPECT_EQ(bundles[packet_index].issue_cycle,
              kCallIssueCycle + packet_index + 1u);
    for (uint8_t slot_index = 0; slot_index < bundles[packet_index].slot_count;
         ++slot_index) {
      useful_slot_count += !iree_any_bit_set(
          slots[bundles[packet_index].slot_start + slot_index].flags,
          LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
    }
  }
  EXPECT_EQ(useful_slot_count, original_slot_count);
}

}  // namespace
}  // namespace loom
