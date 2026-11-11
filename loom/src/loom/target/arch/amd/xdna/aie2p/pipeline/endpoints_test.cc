// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/distribution.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

// The common execution producer has admitted one core for distinct serialized
// children. Native endpoint selection consumes its engine and construction
// correspondence; it does not derive ordering from addresses or queue IDs.
class EndpointStorageTest
    : public ::testing::TestWithParam<
          std::tuple<uint8_t, uint8_t, loom_xdna_dma_direction_t>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    context_.tiles = tiles_.data();
    tile_ = &tiles_[std::get<0>(GetParam())];
    tile_->coordinate = {0, std::get<0>(GetParam())};
    tile_->facts =
        loom_xdna_array_tile_facts(context_.family, tile_->coordinate);
    tile_->memory_layout = tile_->coordinate.row == 1
                               ? &loom_aie2p_sram_memory
                               : &loom_aie2p_compute_memory;
    pools_[0] = tile_;
    context_.pool_tiles = pools_.data();
    context_.channels = channels_.data();
    context_.workers = workers_.data();
    context_.core_engines = engines_.data();
    invocations_.entries = entries_.data();
    invocations_.count = entries_.size();
    invocations_.channels = ownership_.data();
    invocations_.channel_count = ownership_.size();
    invocations_.workers = controller_owners_.data();
    invocations_.worker_count = controller_owners_.size();
    invocations_.lifetimes.has_serial = true;
    for (size_t i = 0; i < children_.size(); ++i) {
      children_[i].ordinal = i;
      entries_[i] = &children_[i];
      controller_owners_[i].invocation = &children_[i];
      engines_[i] = 2;
    }
    engines_[0] = IREE_HOST_SIZE_MAX;
    workers_[0].execution = LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
    for (size_t i = 0; i < channels_.size(); ++i) {
      sources_[i].capacity = 2;
      channels_[i].source = &sources_[i];
      channels_[i].record_byte_length = 64;
      channels_[i].byte_stride = 64;
      channels_[i].record_dense = true;
      channels_[i].ready_lock = 1;
    }
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  bool Select(size_t endpoint, size_t child, size_t channel_owner,
              uint32_t repetitions, bool share_queue = true,
              uint32_t reserved_tasks = 0) {
    auto& channel = channels_[endpoint];
    channel.cursor.reader.worker = channel.cursor.writer.worker = child;
    ownership_[endpoint].invocation = &children_[channel_owner];
    iree_string_view_t reason;
    bool valid = false;
    IREE_EXPECT_OK(loom_aie2p_native_select_channel_endpoint(
        &context_, &invocations_, &children_[child], &channel,
        std::get<2>(GetParam()), repetitions, 1, reserved_tasks,
        static_cast<uint8_t>(std::get<1>(GetParam()) ^ (share_queue ? 0 : 1)),
        &channel, &endpoints_[endpoint], &reason, &valid));
    EXPECT_TRUE(valid) << std::string(reason.data, reason.size);
    return valid;
  }

  void ExpectReentryRestoresGeometry(uint32_t repetitions) {
    channels_[1].record_byte_length = 32;
    channels_[1].byte_stride = 32;
    channels_[1].byte_offset = 128;
    channels_[1].free_lock = 4;
    channels_[1].ready_lock = 5;
    ASSERT_TRUE(Select(0, 1, 1, repetitions));
    ASSERT_TRUE(Select(1, 2, 2, repetitions));
    ASSERT_TRUE(Select(2, 3, 3, repetitions));
    const auto* storage = endpoints_[0].storage;
    ASSERT_NE(storage, nullptr);
    EXPECT_EQ(endpoints_[1].storage, storage);
    EXPECT_EQ(endpoints_[2].storage, storage);
    IREE_ASSERT_OK(loom_aie2p_native_finalize_endpoints(&context_));
    ASSERT_NE(storage->changed_words, nullptr);
    const auto& first = endpoints_[0].sequence;
    std::vector<std::array<uint32_t, 8>> registers(first.descriptor_count);
    for (size_t i = 0; i < registers.size(); ++i) {
      std::copy_n(first.words[i], 8, registers[i].begin());
    }
    const auto* layout = tile_->memory_layout;
    for (size_t visit : {0, 1, 2, 0, 2, 1}) {
      const auto& sequence = endpoints_[visit].sequence;
      EXPECT_EQ(sequence.descriptor, first.descriptor);
      for (size_t i = 0; i < registers.size(); ++i) {
        if (registers[i][layout->iteration_word] &
            loom_xdna_register_field_encode(layout->iteration.wrap, -1)) {
          registers[i][layout->iteration_word] |=
              loom_xdna_register_field_encode(layout->iteration.current, 1);
          EXPECT_NE(storage->changed_words[i] & (1u << layout->iteration_word),
                    0u);
        }
        for (size_t word = 0; word < layout->descriptor_word_count; ++word) {
          if (storage->changed_words[i] & (1u << word)) {
            registers[i][word] = sequence.words[i][word];
          }
          EXPECT_EQ(registers[i][word], sequence.words[i][word])
              << "visit " << visit << " descriptor " << i << " word " << word;
        }
      }
    }
    EXPECT_EQ(tile_->descriptor_counts[tile_->coordinate.row == 1
                                           ? std::get<1>(GetParam()) & 1
                                           : 0],
              first.descriptor_count);
  }

  // Selection-owned temporary memory.
  iree_arena_block_pool_t blocks_ = {};
  // Arena holding ranges, sequence images, and activation masks.
  iree_arena_allocator_t arena_ = {};
  // Compiler ownership of the selection arena.
  loom_pass_t pass_ = {};
  // Complete physical tile index domain.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Memory module under test: compute or SRAM.
  loom_aie2p_native_tile_t* tile_ = nullptr;
  // Selected backing pool's physical location.
  std::array<loom_aie2p_native_tile_t*, 1> pools_ = {};
  // Root and three serialized child construction identities.
  std::array<loom_pipeline_invocation_t, 4> children_ = {};
  // Canonical construction order retained by common analysis.
  std::array<const loom_pipeline_invocation_t*, 4> entries_ = {};
  // Owning construction for each core controller.
  std::array<loom_pipeline_invocation_worker_t, 4> controller_owners_ = {};
  // Core engine assignment whose nonoverlap has already been admitted.
  std::array<iree_host_size_t, 4> engines_ = {};
  // Native instruction workers referenced by channel cursors.
  std::array<loom_aie2p_native_worker_t, 4> workers_ = {};
  // Channel geometry for distinct endpoint occurrences.
  std::array<loom_pipeline_resource_channel_t, 4> sources_ = {};
  // Physical storage and protocol facts of those channels.
  std::array<loom_aie2p_native_channel_t, 4> channels_ = {};
  // Channel construction ownership, independent of storage addresses.
  std::array<loom_pipeline_invocation_channel_t, 4> ownership_ = {};
  // Complete retained ownership domain consumed by selection.
  loom_pipeline_invocations_t invocations_ = {};
  // Stable storage for endpoint images referenced by shared ranges.
  std::array<loom_aie2p_native_distribution_endpoint_t, 4> endpoints_ = {};
  // Native physical resource selection state.
  loom_aie2p_native_context_t context_ = {};
};

TEST_P(EndpointStorageTest, IterationAndChangedGeometryRestoreOnEveryVisit) {
  ExpectReentryRestoresGeometry(4);
}

TEST_P(EndpointStorageTest, FiniteChainsReuseTheirPhysicalDescriptorIds) {
  ExpectReentryRestoresGeometry(tile_->facts->dma.maximum_task_repeat_count *
                                tile_->facts->dma.task_queue_depth * 2);
}

TEST_P(EndpointStorageTest, ReusedGeometryRespectsReservedPhaseTasks) {
  const uint32_t queue_depth = tile_->facts->dma.task_queue_depth;
  const uint32_t repetitions =
      tile_->facts->dma.maximum_task_repeat_count * queue_depth;
  ASSERT_TRUE(Select(0, 1, 1, repetitions));
  ASSERT_EQ(endpoints_[0].sequence.task_count, queue_depth);
  ASSERT_EQ(endpoints_[0].sequence.descriptor_count, 1u);

  // Reserving one slot for a prefix forces a longer chain. Reusing the first
  // child's single descriptor would consume the prefix's task slot.
  ASSERT_TRUE(Select(1, 2, 2, repetitions, true, 1));
  EXPECT_LE(endpoints_[1].sequence.task_count, queue_depth - 1);
  EXPECT_GT(endpoints_[1].sequence.descriptor_count, 1u);
  EXPECT_NE(endpoints_[1].storage, endpoints_[0].storage);

  // The larger geometry can serve another child with the same queue budget.
  ASSERT_TRUE(Select(2, 3, 3, repetitions, true, 1));
  EXPECT_EQ(endpoints_[2].storage, endpoints_[1].storage);
  EXPECT_LE(endpoints_[2].sequence.task_count, queue_depth - 1);
}

TEST_P(EndpointStorageTest, SameChildAndEarlierSelectionSweepsCannotAlias) {
  ASSERT_TRUE(Select(0, 1, 1, 4));
  ASSERT_TRUE(Select(1, 1, 1, 4));
  EXPECT_NE(endpoints_[0].storage, endpoints_[1].storage);
  ASSERT_TRUE(Select(2, 2, 2, 4));
  // A later transfer-selection sweep revisits the first construction. Its
  // endpoint cannot steal a range that also serves that construction already.
  ASSERT_TRUE(Select(3, 1, 1, 4));
  EXPECT_NE(endpoints_[3].storage, endpoints_[0].storage);
  EXPECT_NE(endpoints_[3].storage, endpoints_[1].storage);
}

TEST_P(EndpointStorageTest, SharedQueueRetainsEachChildsRefillOwnership) {
  const auto& dma = tile_->facts->dma;
  // Neither child fits in an autonomous sequence, even with the whole bank.
  // Each reader must replenish the shared queue using its own record progress.
  const uint32_t full_sequence_bound = dma.maximum_task_repeat_count *
                                       dma.task_queue_depth *
                                       dma.buffer_descriptor_count;
  for (size_t i = 0; i < 2; ++i) {
    auto& channel = channels_[i];
    const uint32_t repetitions = full_sequence_bound + i + 1;
    channel.progress = &channel;
    channel.cursor.reader.maximum_admissions = repetitions;
    channel.cursor.reader.exact_admissions = true;
    ASSERT_TRUE(Select(i, i + 1, i + 1, repetitions));
    const auto* refill = workers_[i + 1].refills;
    ASSERT_NE(refill, nullptr);
    EXPECT_EQ(refill->next, nullptr);
    EXPECT_EQ(refill->progress, &channel);
    EXPECT_EQ(refill->window, dma.maximum_task_repeat_count);
    EXPECT_EQ(refill->remaining_tasks, (repetitions - 1) / refill->window);
    const auto* queue = refill->queues;
    ASSERT_NE(queue, nullptr);
    EXPECT_EQ(queue->next, nullptr);
    EXPECT_EQ(queue->tile, tile_);
    EXPECT_EQ(queue->engine, endpoints_[0].engine);
    EXPECT_EQ(queue->direction, std::get<2>(GetParam()));
    EXPECT_EQ(queue->task, endpoints_[i].sequence.tasks[0]);
    EXPECT_EQ(endpoints_[i].sequence.task_count, 1u);
  }
  EXPECT_NE(workers_[1].refills, workers_[2].refills);
  EXPECT_NE(workers_[1].refills->queues->final_task,
            workers_[2].refills->queues->final_task);
}

TEST_P(EndpointStorageTest, ImportedAndDifferentControllerStateRemainResident) {
  ASSERT_TRUE(Select(0, 1, 1, 4));
  ASSERT_TRUE(Select(1, 2, 0, 4));
  EXPECT_EQ(endpoints_[1].storage, nullptr);
  EXPECT_NE(endpoints_[1].sequence.descriptor,
            endpoints_[0].sequence.descriptor);
  engines_[3] = 3;
  ASSERT_TRUE(Select(2, 3, 3, 4));
  EXPECT_NE(endpoints_[2].storage, endpoints_[0].storage);
}

TEST_P(EndpointStorageTest, QueueIdentityDoesNotDefineDescriptorOwnership) {
  ASSERT_TRUE(Select(0, 1, 1, 4));
  // Distinct compute queues share one bank; distinct SRAM parities do not.
  ASSERT_TRUE(Select(1, 2, 2, 4, false));
  EXPECT_NE(endpoints_[0].engine, endpoints_[1].engine);
  if (tile_->coordinate.row == 1) {
    EXPECT_NE(endpoints_[0].storage, endpoints_[1].storage);
  } else {
    EXPECT_EQ(endpoints_[0].storage, endpoints_[1].storage);
  }
}

TEST_P(EndpointStorageTest, UnchangedFixedGeometryRequiresNoActivationWrites) {
  sources_[0].capacity = sources_[1].capacity = 1;
  ASSERT_TRUE(Select(0, 1, 1, 4));
  ASSERT_TRUE(Select(1, 2, 2, 4));
  ASSERT_EQ(endpoints_[0].storage, endpoints_[1].storage);
  IREE_ASSERT_OK(loom_aie2p_native_finalize_endpoints(&context_));
  EXPECT_EQ(endpoints_[0].storage->changed_words, nullptr);
}

TEST_P(EndpointStorageTest,
       FinalFramingCannotLeaveLaterDescriptorsUninitialized) {
  sources_[0].capacity = sources_[1].capacity = 1;
  const uint32_t repetitions = tile_->facts->dma.maximum_task_repeat_count *
                               tile_->facts->dma.task_queue_depth * 2;
  ASSERT_TRUE(Select(0, 1, 1, repetitions));
  ASSERT_TRUE(Select(1, 2, 2, repetitions));
  ASSERT_EQ(endpoints_[0].storage, endpoints_[1].storage);
  ASSERT_GT(endpoints_[0].sequence.descriptor_count, 1u);
  // Receiver-granted framing replaces an autonomous chain with one task per
  // granted record. Configuration will install only this shortened sequence.
  auto& initial = endpoints_[0].sequence;
  const auto* layout = tile_->memory_layout;
  initial.descriptor_count = 1;
  initial.words[0][layout->chain.word] &=
      ~loom_xdna_register_field_encode(layout->chain.enable, 1);
  initial.task_count = 1;
  initial.tasks[0] = loom_xdna_register_field_encode(
      layout->directions[std::get<2>(GetParam()) - 1].start,
      initial.descriptor);
  IREE_ASSERT_OK(loom_aie2p_native_finalize_endpoints(&context_));
  const auto* storage = endpoints_[0].storage;
  ASSERT_NE(storage->changed_words, nullptr);
  for (size_t i = 1; i < endpoints_[1].sequence.descriptor_count; ++i) {
    EXPECT_EQ(storage->changed_words[i],
              (1u << layout->descriptor_word_count) - 1);
  }
  EXPECT_NE(storage->changed_words[0] & (1u << layout->chain.word), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MemoryAndDirections, EndpointStorageTest,
    ::testing::Combine(
        ::testing::Values(uint8_t{1}, uint8_t{2}),
        ::testing::Values(uint8_t{0}, uint8_t{1}),
        ::testing::Values(LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                          LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY)));
