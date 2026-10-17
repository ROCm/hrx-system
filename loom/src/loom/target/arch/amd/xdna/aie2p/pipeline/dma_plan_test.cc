// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

const loom_storage_geometry_t kStridedRecord = {
    .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
    .element_bit_count = 32,
    .rank = 2,
    .axes = {{2, 4}, {2, 1}},
};
const loom_storage_geometry_t kThreeDimensionRecord = {
    .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
    .element_bit_count = 32,
    .rank = 3,
    .axes = {{2, 8}, {1, 4}, {2, 1}},
};

void ExpectIngress(const loom_storage_geometry_t& geometry,
                   uint32_t record_bytes, uint32_t byte_stride,
                   uint32_t record_count, uint32_t task_repetitions,
                   std::array<uint32_t, 8> expected) {
  const auto* family = loom_xdna_npu2_array_family();
  const auto* facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  std::array<uint32_t, 8> words = {};
  uint32_t repetitions = 0;
  ASSERT_TRUE(loom_aie2p_native_encode_ingress(
      family, &geometry, &facts->dma, record_bytes, byte_stride, record_count,
      words.data(), &repetitions));
  EXPECT_EQ(repetitions, task_repetitions);
  EXPECT_EQ(words, expected);
}

TEST(Aie2pDmaPlanTest, DenseSequenceUsesOneTransfer) {
  const loom_storage_geometry_t geometry = {
      .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      .element_bit_count = 8,
      .rank = 1,
      .axes = {{18496, 1}},
  };
  ExpectIngress(geometry, 18496, 18496, 3360, 1,
                {15536640, 0, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000});
}

TEST(Aie2pDmaPlanTest, StridedSequenceAddsAnAddressDimension) {
  ExpectIngress(
      kStridedRecord, 16, 32, 3361, 1,
      {13444, 0, 0, 0x00200000, 0xc0200003, 0x02000007, 0, 0x02000000});
}

TEST(Aie2pDmaPlanTest, FixedSourceKeepsRecordGeometryAndTaskRepetition) {
  ExpectIngress(kStridedRecord, 16, 0, 3361, 3361,
                {4, 0, 0, 0x00200000, 0xc0000003, 0x02000000, 0, 0x02000000});
}

TEST(Aie2pDmaPlanTest, ThreeDimensionsRetainTheIndependentIterator) {
  ExpectIngress(
      kThreeDimensionRecord, 16, 64, 64, 64,
      {4, 0, 0, 0x00200000, 0xc0100003, 0x02000007, 0x03f0000f, 0x02000000});
}

TEST(Aie2pDmaPlanTest, UnrepresentableSequenceCannotTruncateItsIterator) {
  const auto* family = loom_xdna_npu2_array_family();
  const auto* facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  for (auto count : {65u, 3361u}) {
    uint32_t words[8] = {};
    uint32_t repetitions = 0;
    EXPECT_FALSE(loom_aie2p_native_encode_ingress(
        family, &kThreeDimensionRecord, &facts->dma, 16, 64, count, words,
        &repetitions));
  }
}

uint32_t Decode(uint32_t word, loom_xdna_register_field_id_t field) {
  loom_xdna_register_field_info_t info = {};
  IREE_CHECK_OK(loom_xdna_register_field_info(field, &info));
  return (word >> info.least_significant_bit) &
         (UINT32_MAX >> (32 - info.bit_width));
}

TEST(Aie2pDmaPlanTest, EveryAddressWrapFieldRejectsTruncation) {
  const auto* family = loom_xdna_npu2_array_family();
  const struct {
    // Tile module whose descriptor owns the address generator.
    loom_xdna_tile_kind_t kind;
    // Physical wrap field, counted from the innermost address dimension.
    uint8_t dimension;
    // Largest nonzero count representable by this hardware field.
    uint16_t maximum;
    // Descriptor word containing the wrap field.
    uint8_t word;
    // Register field decoded independently of descriptor planning.
    loom_xdna_register_field_id_t field;
  } cases[] = {
      {LOOM_XDNA_TILE_KIND_COMPUTE, 0, 255, 3,
       LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD3_D0_WRAP},
      {LOOM_XDNA_TILE_KIND_COMPUTE, 1, 255, 3,
       LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD3_D1_WRAP},
      {LOOM_XDNA_TILE_KIND_MEMORY, 0, 1023, 2,
       LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD2_D0_WRAP},
      {LOOM_XDNA_TILE_KIND_MEMORY, 1, 1023, 3,
       LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD3_D1_WRAP},
      {LOOM_XDNA_TILE_KIND_MEMORY, 2, 1023, 4,
       LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD4_D2_WRAP},
      {LOOM_XDNA_TILE_KIND_SHIM_NOC, 0, 1023, 3,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD3_D0_WRAP},
      {LOOM_XDNA_TILE_KIND_SHIM_NOC, 1, 1023, 4,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD4_D1_WRAP},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "tile=" << entry.kind << " dimension=" << +entry.dimension);
    const auto* dma = &loom_xdna_array_tile_kind_facts(family, entry.kind)->dma;
    loom_storage_geometry_t geometry = {
        .layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
        .element_bit_count = 32,
        .rank = static_cast<uint8_t>(entry.dimension + 2),
        .axes = {{2, 2064},
                 {entry.maximum, entry.dimension ? 2u : 1u},
                 {1, 1},
                 {1, 1}},
    };
    uint32_t words[8] = {};
    iree_string_view_t reason;
    ASSERT_TRUE(loom_aie2p_native_encode_descriptor(
        family, &geometry, dma, 2 * entry.maximum * sizeof(uint32_t), 1,
        entry.kind, words, &reason));
    EXPECT_EQ(Decode(words[entry.word], entry.field), entry.maximum);

    ++geometry.axes[1].extent;
    uint32_t rejected_words[8] = {};
    EXPECT_FALSE(loom_aie2p_native_encode_descriptor(
        family, &geometry, dma, 2 * (entry.maximum + 1) * sizeof(uint32_t), 1,
        entry.kind, rejected_words, &reason));
    EXPECT_TRUE(iree_string_view_equal(
        reason,
        IREE_SV("an address dimension exceeds the nonzero DMA wrap field")));
  }
}

class Aie2pDmaSequenceTest
    : public ::testing::TestWithParam<loom_xdna_tile_kind_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  const loom_xdna_dma_facts_t* dma() {
    return &loom_xdna_array_tile_kind_facts(loom_xdna_npu2_array_family(),
                                            GetParam())
                ->dma;
  }
  const loom_aie2p_native_memory_layout_t* layout() {
    switch (GetParam()) {
      case LOOM_XDNA_TILE_KIND_COMPUTE:
        return &loom_aie2p_compute_memory;
      case LOOM_XDNA_TILE_KIND_MEMORY:
        return &loom_aie2p_sram_memory;
      default:
        return nullptr;
    }
  }
  uint16_t bank_size() {
    return dma()->buffer_descriptor_count /
           (GetParam() == LOOM_XDNA_TILE_KIND_MEMORY ? 2 : 1);
  }
  uint16_t first_descriptor() {
    // Use a nonzero ordinal, including the SRAM odd-channel bank.
    return GetParam() == LOOM_XDNA_TILE_KIND_MEMORY ? 26 : 2;
  }
  loom_xdna_register_field_id_t start_field(
      loom_xdna_dma_direction_t direction) {
    if (layout()) {
      return layout()->directions[direction - 1].start;
    }
    return direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
               ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
               : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID;
  }
  loom_xdna_register_field_id_t repeat_field(
      loom_xdna_dma_direction_t direction) {
    if (layout()) {
      return layout()->directions[direction - 1].repeat;
    }
    return direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
               ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT
               : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_REPEAT_COUNT;
  }
  std::array<uint32_t, 8> descriptor_template() {
    std::array<uint32_t, 8> words = {};
    words[0] = 16 - dma()->transfer_length_offset;
    if (layout()) {
      words[layout()->lock_word] =
          loom_xdna_register_field_encode_admitted(layout()->locks.valid, 1) |
          loom_xdna_register_field_encode_admitted(
              layout()->locks.acquire_id, layout()->local_lock_base + 2) |
          loom_xdna_register_field_encode_admitted(
              layout()->locks.acquire_enable, 1) |
          loom_xdna_register_field_encode_admitted(
              layout()->locks.acquire_value, -1) |
          loom_xdna_register_field_encode_admitted(
              layout()->locks.release_id, layout()->local_lock_base + 3) |
          loom_xdna_register_field_encode_admitted(
              layout()->locks.release_value, 1);
    } else {
      words[4] = 0xc0000000;
      words[5] = 0x02000000;
      words[7] = 0x02000000;
    }
    return words;
  }

  void ExpectExecution(const loom_aie2p_native_dma_sequence_t& sequence,
                       loom_xdna_dma_direction_t direction,
                       uint32_t repetitions, uint32_t capacity) {
    const auto* memory = layout();
    const uint8_t chain_word = memory ? memory->chain.word : 7;
    const auto next =
        memory ? memory->chain.next
               : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
    const auto enable =
        memory ? memory->chain.enable
               : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
    ASSERT_LE(sequence.task_count, dma()->task_queue_depth);
    ASSERT_LE(sequence.descriptor_count, bank_size() - 2);
    std::vector<uint32_t> iterations(sequence.descriptor_count, 0);
    uint32_t records = 0;
    // Execute the encoded queue/links. This oracle does not choose a chain or
    // derive the builder's factoring; it observes the resulting record order.
    for (uint8_t task = 0; task < sequence.task_count; ++task) {
      const uint32_t cycles =
          Decode(sequence.tasks[task], repeat_field(direction)) + 1;
      ASSERT_LE(cycles, dma()->maximum_task_repeat_count);
      for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
        uint32_t descriptor =
            Decode(sequence.tasks[task], start_field(direction));
        for (uint16_t traversed = 0;; ++traversed) {
          ASSERT_LT(traversed, sequence.descriptor_count);
          ASSERT_GE(descriptor, sequence.descriptor);
          const uint32_t index = descriptor - sequence.descriptor;
          ASSERT_LT(index, sequence.descriptor_count);
          ASSERT_LT(records, repetitions);
          const uint32_t* words = sequence.words[index];
          if (memory) {
            const uint32_t step =
                Decode(words[memory->iteration_word], memory->iteration.step) +
                1;
            const uint32_t wrap =
                Decode(words[memory->iteration_word], memory->iteration.wrap) +
                1;
            const uint32_t address =
                (Decode(words[memory->address_word], memory->address)
                 << dma()->address_encoding_shift) +
                iterations[index] * step * dma()->transfer_length_granularity;
            EXPECT_EQ(address, memory->local_address_base + 128 +
                                   (records % capacity) * 64);
            iterations[index] = (iterations[index] + 1) % wrap;
          }
          ++records;
          if (!Decode(words[chain_word], enable)) {
            break;
          }
          descriptor = Decode(words[chain_word], next);
        }
      }
    }
    EXPECT_EQ(records, repetitions);
    EXPECT_EQ(sequence.rewind_iteration, memory && iterations[0] != 0);
  }

  // Allocation owner for descriptor and task storage.
  iree_arena_block_pool_t pool_;
  // Per-test planner storage, retired after all sequence assertions.
  iree_arena_allocator_t arena_;
};

TEST_P(Aie2pDmaSequenceTest, QueueAndRingBoundariesPreserveEveryRecord) {
  for (auto direction : {LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                         LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY}) {
    for (uint32_t capacity : {1u, 2u, 3u, 4u}) {
      if (!layout() && capacity != 1) {
        continue;
      }
      for (uint32_t count : {1u, 63u, 64u, 65u, 255u, 256u, 257u, 1023u, 1024u,
                             1025u, 3360u, 3361u, 4294u, 5190u}) {
        SCOPED_TRACE(::testing::Message()
                     << "direction=" << direction << " capacity=" << capacity
                     << " count=" << count);
        loom_aie2p_native_dma_sequence_t sequence = {};
        iree_string_view_t reason;
        bool valid = false;
        const auto words = descriptor_template();
        IREE_ASSERT_OK(loom_aie2p_native_build_sequence(
            dma(), layout(), first_descriptor(), bank_size() - 2, direction,
            count, capacity, 128, 64, words.data(), &sequence, &reason, &valid,
            &arena_));
        ASSERT_TRUE(valid);
        ExpectExecution(sequence, direction, count, capacity);
      }
    }
  }
}

TEST_P(Aie2pDmaSequenceTest, FiniteChainRetainsExactTailAndTaskEncodings) {
  const uint32_t capacity = layout() ? 3 : 1;
  // 3361 = 6 * (256 + 256 + 48) + 1 for local rings, and
  //        5 * (256 + 256 + 160) + 1 for a fixed external address.
  const uint32_t chain_length = layout() ? 6 : 5;
  const auto words = descriptor_template();
  const auto direction = LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM;
  loom_aie2p_native_dma_sequence_t sequence = {};
  iree_string_view_t reason;
  bool valid = false;
  IREE_ASSERT_OK(loom_aie2p_native_build_sequence(
      dma(), layout(), first_descriptor(), bank_size() - 2, direction, 3361,
      capacity, 128, 64, words.data(), &sequence, &reason, &valid, &arena_));
  ASSERT_TRUE(valid);
  ASSERT_EQ(sequence.descriptor_count, chain_length + (layout() ? 1 : 0));
  ASSERT_EQ(sequence.task_count, 4);
  EXPECT_FALSE(sequence.rewind_iteration);
  const uint32_t cycles[] = {256, 256, layout() ? 48u : 160u, 1};
  for (uint32_t task = 0; task < 4; ++task) {
    const uint32_t tail = layout() ? chain_length : chain_length - 1;
    const uint32_t head = sequence.descriptor + (task == 3 ? tail : 0);
    EXPECT_EQ(
        sequence.tasks[task],
        loom_xdna_register_field_encode_admitted(start_field(direction), head) |
            loom_xdna_register_field_encode_admitted(repeat_field(direction),
                                                     cycles[task] - 1));
  }
  for (uint32_t index = 0; index < sequence.descriptor_count; ++index) {
    auto expected = words;
    if (layout()) {
      const uint32_t slot = index == chain_length ? 0 : index % 3;
      expected[layout()->address_word] |=
          loom_xdna_register_field_encode_admitted(
              layout()->address,
              (layout()->local_address_base + 128 + slot * 64) >>
                  dma()->address_encoding_shift);
    }
    if (index + 1 < chain_length) {
      const auto next =
          layout() ? layout()->chain.next
                   : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
      const auto enable =
          layout() ? layout()->chain.enable
                   : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
      expected[layout() ? layout()->chain.word : 7] |=
          loom_xdna_register_field_encode_admitted(
              next, sequence.descriptor + index + 1) |
          loom_xdna_register_field_encode_admitted(enable, 1);
    }
    for (uint32_t word = 0; word < expected.size(); ++word) {
      EXPECT_EQ(sequence.words[index][word], expected[word])
          << "descriptor=" << index << " word=" << word;
    }
  }
}

TEST_P(Aie2pDmaSequenceTest, AlignedTailFitsWithoutDuplicatingRingSlots) {
  for (auto direction : {LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                         LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY}) {
    for (uint32_t capacity : {1u, 2u, 3u, 4u}) {
      if (!layout() && capacity != 1) {
        continue;
      }
      SCOPED_TRACE(::testing::Message()
                   << "direction=" << direction << " capacity=" << capacity);
      const uint32_t count = capacity * 1025;
      const auto words = descriptor_template();
      loom_aie2p_native_dma_sequence_t sequence = {};
      iree_string_view_t reason;
      bool valid = false;
      IREE_ASSERT_OK(loom_aie2p_native_build_sequence(
          dma(), layout(), first_descriptor(), 2 * capacity, direction, count,
          capacity, 128, 64, words.data(), &sequence, &reason, &valid,
          &arena_));
      ASSERT_TRUE(valid);
      ASSERT_EQ(sequence.descriptor_count, 2 * capacity);
      ASSERT_EQ(sequence.task_count, 3);
      EXPECT_EQ(sequence.tasks[0],
                loom_xdna_register_field_encode_admitted(start_field(direction),
                                                         sequence.descriptor) |
                    loom_xdna_register_field_encode_admitted(
                        repeat_field(direction), 255));
      EXPECT_EQ(sequence.tasks[1], sequence.tasks[0]);
      EXPECT_EQ(sequence.tasks[2],
                loom_xdna_register_field_encode_admitted(
                    start_field(direction), sequence.descriptor + capacity));
      ExpectExecution(sequence, direction, count, capacity);
    }
  }
}

TEST_P(Aie2pDmaSequenceTest, DescriptorBudgetRejectsWithoutPartialProgram) {
  const auto words = descriptor_template();
  for (uint16_t available : {uint16_t{0}, uint16_t{1}}) {
    loom_aie2p_native_dma_sequence_t sequence = {};
    iree_string_view_t reason;
    bool valid = true;
    IREE_ASSERT_OK(loom_aie2p_native_build_sequence(
        dma(), layout(), first_descriptor(), available,
        LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, 1025, 1, 128, 64,
        words.data(), &sequence, &reason, &valid, &arena_));
    EXPECT_FALSE(valid);
    EXPECT_EQ(sequence.words, nullptr);
    EXPECT_EQ(sequence.tasks, nullptr);
    EXPECT_EQ(sequence.descriptor_count, 0);
    EXPECT_TRUE(iree_string_view_equal(
        reason,
        available
            ? IREE_SV("a finite record stream within DMA descriptor and "
                      "task-queue capacity")
            : IREE_SV("an available descriptor in the DMA channel's bank")));
  }
}

INSTANTIATE_TEST_SUITE_P(TileKinds, Aie2pDmaSequenceTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_COMPUTE,
                                           LOOM_XDNA_TILE_KIND_MEMORY,
                                           LOOM_XDNA_TILE_KIND_SHIM_NOC));

}  // namespace
