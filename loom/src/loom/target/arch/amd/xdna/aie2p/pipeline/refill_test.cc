// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"

namespace {

uint32_t Decode(uint32_t word, loom_xdna_register_field_id_t field) {
  const auto info = loom_xdna_register_field_info(field);
  return (word >> info.least_significant_bit) &
         (UINT32_MAX >> (32 - info.bit_width));
}

class Aie2pRefillTest : public ::testing::TestWithParam<loom_xdna_tile_kind_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    context_.workers = &worker_;
    context_.pool_tiles = pool_tiles_;
    tile_.facts = loom_xdna_array_tile_kind_facts(context_.family, GetParam());
    tile_.coordinate = {1, tile_.facts->first_row};
    tile_.memory_layout = GetParam() == LOOM_XDNA_TILE_KIND_MEMORY
                              ? &loom_aie2p_sram_memory
                              : &loom_aie2p_compute_memory;
    source_.progress = &reader_;
    source_.source = &binding_;
    // The existing reader is the continuation owner, not a new service core.
    reader_.cursor.reader.worker = 0;
    reader_.cursor.reader.maximum_admissions = 65539;
    words_[0] = 16 - tile_.facts->dma.transfer_length_offset;
    const auto& layout = *tile_.memory_layout;
    words_[layout.lock_word] =
        loom_xdna_register_field_encode(layout.locks.valid, 1) |
        loom_xdna_register_field_encode(layout.locks.acquire_enable, 1) |
        loom_xdna_register_field_encode(layout.locks.acquire_value, -1) |
        loom_xdna_register_field_encode(layout.locks.release_value, 1);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  void Select(uint32_t records, uint32_t capacity = 3,
              loom_xdna_dma_direction_t direction =
                  LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
              uint32_t credit_interval = 1) {
    valid_ = false;
    IREE_ASSERT_OK(loom_aie2p_native_select_sequence(
        &context_, &tile_, 0, direction, &source_, records, capacity, 128,
        capacity == 1 ? 0 : 64, credit_interval, 0, words_, &sequence_,
        &reason_, &valid_));
  }
  void ExpectWindow(uint32_t records, uint32_t final_records, bool rewind) {
    reader_.cursor.reader.maximum_admissions = records;
    ASSERT_NO_FATAL_FAILURE(Select(records));
    ASSERT_TRUE(valid_);
    ASSERT_NE(worker_.refills, nullptr);
    const auto& refill = *worker_.refills;
    EXPECT_EQ(refill.progress, &reader_);
    EXPECT_EQ(refill.window, 255u);
    EXPECT_EQ(refill.next, nullptr);
    ASSERT_NE(refill.queues, nullptr);
    const auto& queue = *refill.queues;
    EXPECT_EQ(queue.next, nullptr);
    EXPECT_EQ(queue.tile, &tile_);
    EXPECT_EQ(sequence_.descriptor_count, 1u);
    EXPECT_EQ(sequence_.task_count, 1u);
    EXPECT_EQ(sequence_.rewind_iteration, rewind);
    const auto& fields = tile_.memory_layout->directions[0];
    EXPECT_EQ(Decode(queue.task, fields.repeat) + 1, 255u);
    EXPECT_EQ(Decode(queue.final_task, fields.repeat) + 1, final_records);
    EXPECT_EQ(Decode(queue.task, fields.start), sequence_.descriptor);
    EXPECT_EQ(Decode(queue.final_task, fields.start), sequence_.descriptor);
    // Count the actual queued windows, including startup and the exact tail.
    EXPECT_EQ(255u * refill.remaining_tasks + final_records, records);
  }
  uint32_t PhasedWindow(uint32_t records, uint32_t capacity = 2,
                        uint32_t cycles = 2, size_t prefixes = 2) {
    binding_.capacity = capacity;
    reader_.cursor.reader.maximum_admissions = uint64_t{cycles} * records;
    return loom_aie2p_native_phased_window(&context_, &source_, prefixes,
                                           cycles, records);
  }
  // Lifetime of the planner's retained descriptor and continuation results.
  iree_arena_block_pool_t pool_ = {};
  // Shared allocation arena for one native plan.
  iree_arena_allocator_t arena_ = {};
  // Allocation service, without any compiler module or source IR.
  loom_pass_t pass_ = {};
  // Native planner's physical resource owner.
  loom_aie2p_native_context_t context_ = {};
  // Real module facts and current descriptor-bank consumption.
  loom_aie2p_native_tile_t tile_ = {};
  // Physical endpoint selected for the source's storage pool.
  loom_aie2p_native_tile_t* pool_tiles_[1] = {&tile_};
  // Authored bulk-ring capacity, retained by resource analysis.
  loom_pipeline_resource_channel_t binding_ = {};
  // Existing arithmetic worker receiving refill continuations.
  loom_aie2p_native_worker_t worker_ = {};
  // Source protocol with a previously selected equal-record observer.
  loom_aie2p_native_channel_t source_ = {};
  // Reader identity and exact admissions proved by common analysis.
  loom_aie2p_native_channel_t reader_ = {};
  // One record's length and credit actions, before address selection.
  uint32_t words_[8] = {};
  // Last selected descriptor and startup task sequence.
  loom_aie2p_native_dma_sequence_t sequence_ = {};
  // Last selection's admission result.
  bool valid_ = false;
  // Static requirement retained for an unsupported sequence.
  iree_string_view_t reason_ = {};
};

TEST_P(Aie2pRefillTest, ShortSequenceKeepsStartupOnlyAdmission) {
  Select(12);
  ASSERT_TRUE(valid_);
  EXPECT_EQ(worker_.refills, nullptr);
}

TEST_P(Aie2pRefillTest, ExactWindowsNeedNoPartialTaskOrIteratorRewind) {
  ExpectWindow(65535, 255, false);
}

TEST_P(Aie2pRefillTest, PartialWindowTransfersOnlyTheRemainingRecords) {
  ExpectWindow(65539, 4, true);
}

TEST_P(Aie2pRefillTest, CompleteRingTailDoesNotNeedIteratorRewind) {
  ExpectWindow(65538, 3, false);
}

TEST_P(Aie2pRefillTest, EquivalentEndpointWindowsShareOneContinuation) {
  Select(65539);
  ASSERT_TRUE(valid_);
  auto* first = worker_.refills;
  ASSERT_NE(first, nullptr);
  Select(65539, 3, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY);
  ASSERT_TRUE(valid_);
  EXPECT_EQ(worker_.refills, first);
  ASSERT_NE(first->queues->next, nullptr);
  EXPECT_EQ(first->queues->direction, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY);
  EXPECT_EQ(first->queues->next->direction,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM);
  EXPECT_EQ(first->queues->next->next, nullptr);
}

TEST_P(Aie2pRefillTest, DifferentRingWindowsNeedIndependentContinuations) {
  Select(65539);
  ASSERT_TRUE(valid_);
  auto* first = worker_.refills;
  ASSERT_NE(first, nullptr);
  loom_aie2p_native_tile_t second_tile = {
      .coordinate = {2, tile_.coordinate.row},
      .facts = tile_.facts,
      .memory_layout = tile_.memory_layout,
  };
  loom_aie2p_native_channel_t second_source = {.progress = &reader_};
  IREE_ASSERT_OK(loom_aie2p_native_select_sequence(
      &context_, &second_tile, 0, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
      &second_source, 65539, 2, 128, 64, 1, 0, words_, &sequence_, &reason_,
      &valid_));
  ASSERT_TRUE(valid_);
  ASSERT_NE(worker_.refills, first);
  EXPECT_EQ(worker_.refills->next, first);
  EXPECT_EQ(worker_.refills->window, 256u);
  EXPECT_EQ(worker_.refills->queues->tile, &second_tile);
}

TEST_P(Aie2pRefillTest, UnprovedOrUnequalProgressCannotReplenishTheQueue) {
  source_.progress = nullptr;
  Select(65539);
  EXPECT_FALSE(valid_);
  source_.progress = &reader_;
  reader_.cursor.reader.maximum_admissions = 65538;
  Select(65539);
  EXPECT_FALSE(valid_);
  EXPECT_EQ(worker_.refills, nullptr);
  EXPECT_EQ(tile_.descriptor_counts[0], 0u);
}

TEST_P(Aie2pRefillTest, BatchCreditsCannotStandInForRecordProgress) {
  reader_.cursor.reader.maximum_admissions = 65538;
  Select(65538, 3, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM, 3);
  EXPECT_FALSE(valid_);
  EXPECT_EQ(worker_.refills, nullptr);
}

TEST_P(Aie2pRefillTest, PhasedWindowDividesTheCycleAndCompleteRingTurns) {
  EXPECT_EQ(PhasedWindow(2880), 240u);
  EXPECT_EQ(PhasedWindow(512), 256u);
  EXPECT_EQ(PhasedWindow(514), 2u);
  EXPECT_EQ(PhasedWindow(2880, 3), 240u);
  EXPECT_EQ(PhasedWindow(510, 3), 255u);
  EXPECT_EQ(PhasedWindow(6, 3), 6u);
  // Selection itself neither allocates descriptors nor adds a continuation.
  EXPECT_EQ(tile_.descriptor_counts[0], 0u);
  EXPECT_EQ(worker_.refills, nullptr);
}

TEST_P(Aie2pRefillTest, SingleCycleAllowsTheOrdinaryPartialWindowTail) {
  EXPECT_EQ(PhasedWindow(513, 2, 1), 256u);
  EXPECT_EQ(PhasedWindow(513, 2, 2), 0u);
  EXPECT_EQ(PhasedWindow(1, 2, 1), 1u);
  EXPECT_EQ(PhasedWindow(1, 2, 2), 0u);
}

TEST_P(Aie2pRefillTest, PhasedQueueBudgetIncludesActiveAndNextBulkTasks) {
  const size_t depth = tile_.facts->dma.task_queue_depth;
  EXPECT_NE(PhasedWindow(2880, 2, 2, depth - 2), 0u);
  EXPECT_EQ(PhasedWindow(2880, 2, 2, depth - 1), 0u);
}

TEST_P(Aie2pRefillTest, CompleteRingMustFitOneImmutableTask) {
  const uint32_t limit = tile_.facts->dma.maximum_task_repeat_count;
  EXPECT_EQ(PhasedWindow(2 * limit, limit), limit);
  EXPECT_EQ(PhasedWindow(2 * (limit + 1), limit + 1), 0u);
}

TEST_P(Aie2pRefillTest, PhasedQueueNeedsAnExactBulkProgressObserver) {
  binding_.capacity = 2;
  source_.progress = nullptr;
  EXPECT_EQ(loom_aie2p_native_phased_window(&context_, &source_, 2, 2, 2880),
            0u);
  source_.progress = &reader_;
  for (uint64_t admissions : {5759u, 5760u, 5761u}) {
    reader_.cursor.reader.maximum_admissions = admissions;
    EXPECT_EQ(loom_aie2p_native_phased_window(&context_, &source_, 2, 2, 2880),
              admissions == 5760 ? 240u : 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(OnChipMemory, Aie2pRefillTest,
                         ::testing::Values(LOOM_XDNA_TILE_KIND_COMPUTE,
                                           LOOM_XDNA_TILE_KIND_MEMORY));

}  // namespace
