// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"

#include <deque>
#include <tuple>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"

namespace {

class DmaResourcesTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    local_.memory_layout =
        GetParam() ? &loom_aie2p_sram_memory : &loom_aie2p_compute_memory;
    path_.local = &local_;
    path_.shim = &shim_;
    issue_.parent_block = completion_op_.parent_block = &block_;
    issue_.block_ordinal = 0;
    completion_op_.block_ordinal = 2;
    source_.request.op = &issue_;
    previous_.source = next_.source = &source_;
    previous_.completion = next_.completion = &completion_op_;
    previous_.external_view = next_.external_view = &external_;
    // Two ordered occurrences of precisely the same source operations.
    previous_.worker_index = 0;
    next_.worker_index = 1;
    next_.task = &task_;
    task_.first_transfer = task_.last_transfer = &next_;
    task_.transfer_count = 1;
    task_.path = &path_;
    descriptor_.path = &path_;
    descriptor_.last_transfer = &previous_;
    descriptor_.sequence.descriptor_count = 1;
    descriptor_.sequence.words = words_;
    descriptor_.completion = &completion_;
    completion_.last_transfer = &previous_;
    completion_.tile = &local_;
    owner_.execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
    owner_.descriptors = &descriptor_;
    owner_.dma_completions = &completion_;
    // These are descriptor templates before completion-release insertion.
    words_[0][0] = next_.local_words[0] = 16;
    words_[0][local_.memory_layout->lock_word] =
        next_.local_words[local_.memory_layout->lock_word] =
            loom_xdna_register_field_encode(local_.memory_layout->locks.valid,
                                            1);
  }

  auto* Descriptor(bool shim = false) {
    return loom_aie2p_native_reusable_descriptor(&owner_, &task_, shim);
  }

  auto* Completion() {
    return loom_aie2p_native_reusable_completion(&owner_, &local_, &next_);
  }

  // Common straight-line source scope for both ordered occurrences.
  loom_block_t block_ = {};
  // The transfer issue precedes its explicit completion in this source block.
  loom_op_t issue_ = {};
  // Producer-retained wait operation, shared across occurrences.
  loom_op_t completion_op_ = {};
  // Canonical movement site, independent of physical occurrence identity.
  loom_kernel_async_transfer_t source_ = {};
  // External endpoint with an exact fixed offset.
  loom_view_region_t external_ = {};
  // On-chip memory owner and its architecture-specific descriptor fields.
  loom_aie2p_native_tile_t local_ = {};
  // External descriptor bank owner.
  loom_aie2p_native_tile_t shim_ = {};
  // One selected engine pair and its routes.
  loom_aie2p_native_dma_path_t path_ = {};
  // Last transfer assigned to the retained resources.
  loom_aie2p_native_transfer_t previous_ = {};
  // Next occurrence requesting resources for the same source copy.
  loom_aie2p_native_transfer_t next_ = {};
  // Complete one-segment task whose resources are being selected.
  loom_aie2p_native_transfer_task_t task_ = {};
  // Immutable descriptor images excluding completion release actions.
  uint32_t words_[1][8] = {};
  // Previously allocated descriptor sequence.
  loom_aie2p_native_dma_descriptor_t descriptor_ = {};
  // Previously allocated arrival credit.
  loom_aie2p_native_dma_completion_t completion_ = {};
  // Physical execution owner containing the reusable resource inventories.
  loom_aie2p_native_worker_t owner_ = {};
};

TEST_P(DmaResourcesTest, RepeatedSourceOccurrencesReuseDescriptorsAndCredits) {
  EXPECT_EQ(Descriptor(), &descriptor_);
  EXPECT_EQ(Completion(), &completion_);
  EXPECT_FALSE(descriptor_.address_dynamic);
  EXPECT_TRUE(loom_aie2p_native_transfer_retired(&previous_, next_.worker_index,
                                                 &issue_));
}

TEST_P(DmaResourcesTest, OneOccurrenceCannotReuseBeforeItsCompletion) {
  next_.worker_index = previous_.worker_index;
  EXPECT_EQ(Descriptor(), nullptr);
  EXPECT_EQ(Completion(), nullptr);
  EXPECT_FALSE(loom_aie2p_native_transfer_retired(&previous_,
                                                  next_.worker_index, &issue_));
  auto later_issue = issue_;
  later_issue.block_ordinal = 3;
  auto later_source = source_;
  later_source.request.op = &later_issue;
  next_.source = &later_source;
  EXPECT_EQ(Descriptor(), &descriptor_);
  EXPECT_EQ(Completion(), &completion_);
}

TEST_P(DmaResourcesTest, BlockExitRetiresEveryAdmittedGroup) {
  loom_block_t following_block = {};
  auto later_issue = issue_;
  later_issue.parent_block = &following_block;
  EXPECT_TRUE(loom_aie2p_native_transfer_retired(
      &previous_, previous_.worker_index, &later_issue));
}

TEST_P(DmaResourcesTest, SharedCreditMustRetireItsMostRecentUser) {
  // The descriptor's old use has completed, but a different descriptor has
  // since borrowed the same credit in the current occurrence.
  auto outstanding = previous_;
  outstanding.worker_index = next_.worker_index;
  completion_.last_transfer = &outstanding;
  EXPECT_EQ(Descriptor(), nullptr);
  EXPECT_EQ(Completion(), nullptr);
}

TEST_P(DmaResourcesTest, GeometryChangesReuseTheCreditButNotTheDescriptor) {
  next_.local_words[0] = 32;
  EXPECT_EQ(Descriptor(), nullptr);
  EXPECT_EQ(Completion(), &completion_);
}

TEST_P(DmaResourcesTest, AutonomousIterationStateLivesUntilInvocationRetires) {
  owner_.execution = LOOM_AIE2P_NATIVE_EXECUTION_DMA;
  EXPECT_EQ(Descriptor(), nullptr);
  EXPECT_EQ(Completion(), nullptr);
  owner_.execution = LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
  EXPECT_EQ(Descriptor(), &descriptor_);
  EXPECT_EQ(Completion(), &completion_);
}

TEST_P(DmaResourcesTest, LocalAddressChangesPromoteTheSharedTemplate) {
  const auto& layout = *local_.memory_layout;
  words_[0][layout.address_word] |=
      loom_xdna_register_field_encode(layout.address, 32);
  next_.local_words[layout.address_word] |=
      loom_xdna_register_field_encode(layout.address, 64);
  EXPECT_EQ(Descriptor(), &descriptor_);
  EXPECT_TRUE(descriptor_.address_dynamic);
  const auto field = loom_xdna_register_field_info(layout.address);
  const auto mask = (UINT32_MAX >> (32 - field.bit_width))
                    << field.least_significant_bit;
  EXPECT_EQ(words_[0][layout.address_word] & mask, 0u);
  EXPECT_EQ(words_[0][0], 16u);
  EXPECT_EQ(words_[0][layout.lock_word],
            loom_xdna_register_field_encode(layout.locks.valid, 1));
}

TEST_P(DmaResourcesTest, LocalResourcesCannotCrossPathsOrTiles) {
  auto other_path = path_;
  task_.path = &other_path;
  EXPECT_EQ(Descriptor(), nullptr);
  loom_aie2p_native_tile_t other_tile = local_;
  EXPECT_EQ(loom_aie2p_native_reusable_completion(&owner_, &other_tile, &next_),
            nullptr);
}

TEST_P(DmaResourcesTest, ShimReuseSharesTheBankAndTracksExternalAddresses) {
  descriptor_.shim = true;
  descriptor_.completion = nullptr;
  words_[0][local_.memory_layout->lock_word] = 0;
  next_.shim_words[0] = words_[0][0];
  auto other_path = path_;
  task_.path = &other_path;
  EXPECT_EQ(Descriptor(true), &descriptor_);
  EXPECT_FALSE(descriptor_.address_dynamic);
  next_.binding = 1;
  EXPECT_EQ(Descriptor(true), &descriptor_);
  EXPECT_TRUE(descriptor_.address_dynamic);
  auto other_shim = shim_;
  other_path.shim = &other_shim;
  EXPECT_EQ(Descriptor(true), nullptr);
}

TEST_P(DmaResourcesTest, LinkedSequencesCannotBeReusedAsSingletons) {
  // A valid two-descriptor chain remains indivisible even after retirement.
  uint32_t chain_words[2][8] = {};
  descriptor_.sequence.descriptor = 4;
  descriptor_.sequence.descriptor_count = 2;
  descriptor_.sequence.words = chain_words;
  auto second = next_;
  second.descriptor_ordinal = 1;
  next_.task_next = &second;
  task_.last_transfer = &second;
  task_.transfer_count = 2;
  loom_aie2p_native_task_descriptor_words(&task_, &descriptor_, &next_,
                                          chain_words[0]);
  loom_aie2p_native_task_descriptor_words(&task_, &descriptor_, &second,
                                          chain_words[1]);
  EXPECT_EQ(Descriptor(), &descriptor_);
  const auto& chain = local_.memory_layout->chain;
  EXPECT_EQ(chain_words[0][chain.word],
            next_.local_words[chain.word] |
                loom_xdna_register_field_encode(chain.next, 5) |
                loom_xdna_register_field_encode(chain.enable, 1));
  next_.task_next = nullptr;
  task_.last_transfer = &next_;
  task_.transfer_count = 1;
  EXPECT_EQ(Descriptor(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(MemoryLayouts, DmaResourcesTest,
                         ::testing::Values(false, true));

// Required geometry admission and optional address refinement use the actual
// resource owner on compute and SRAM descriptor banks, in both directions.
class DmaAddressVariantsTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  struct Task {
    // Issue operations ordered before this task's completion.
    loom_op_t issues[5] = {};
    // Retained source movements for this task's descriptor chain.
    loom_kernel_async_transfer_t sources[5] = {};
    // Explicit completion before the next task may reuse its resources.
    loom_op_t completion = {};
    // Caller-relative external offsets, one per chain segment.
    loom_view_region_t external[5] = {};
    // Physical transfers carrying immutable descriptor templates.
    loom_aie2p_native_transfer_t transfers[5] = {};
    // Complete chain selected by the production admission owner.
    loom_aie2p_native_transfer_task_t task = {};
  };

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    const bool memory = std::get<0>(GetParam());
    local_.coordinate = {0, static_cast<uint8_t>(memory ? 1 : 2)};
    local_.facts =
        loom_xdna_array_tile_facts(context_.family, local_.coordinate);
    local_.memory_layout =
        memory ? &loom_aie2p_sram_memory : &loom_aie2p_compute_memory;
    shim_.coordinate = {0, 0};
    shim_.facts = loom_xdna_array_tile_facts(context_.family, shim_.coordinate);
    path_.local = &local_;
    path_.shim = &shim_;
    path_.local_engine = memory ? 1 : 0;
    path_.ingress = std::get<1>(GetParam());
    path_.completion_tokens = memory ? 2 : path_.ingress ? 0 : 1;
    worker_.execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
    worker_.channels.indices = &channel_index_;
    worker_.channels.accesses = &access_;
    worker_.channels.count = 1;
    access_.channel = &channel_;
    access_.window = &window_;
    context_.channels = &channel_;
    context_.workers = &worker_;
    context_.core_engines = &engine_;
    context_.execution.workers = &owner_;
    descriptor_tail_ = &worker_.descriptors;
    completion_tail_ = &worker_.dma_completions;
    task_tail_ = &worker_.transfer_tasks;
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void AddTask(uint32_t address, uint32_t count = 1, bool local_dynamic = false,
               bool external_dynamic = false, uint32_t binding = 0) {
    tasks_.emplace_back();
    auto& record = tasks_.back();
    const uint32_t position = static_cast<uint32_t>(tasks_.size() - 1) * 8;
    record.completion.parent_block = &block_;
    record.completion.block_ordinal = position + count;
    record.task.path = &path_;
    record.task.first_transfer = record.transfers;
    record.task.last_transfer = &record.transfers[count - 1];
    record.task.transfer_count = count;
    record.task.local_address_dynamic = local_dynamic;
    record.task.shim_address_dynamic = external_dynamic;
    for (uint32_t i = 0; i < count; ++i) {
      record.issues[i].parent_block = &block_;
      record.issues[i].block_ordinal = position + i;
      record.sources[i].request.op = &record.issues[i];
      record.external[i].begin_byte_offset.constant = address + i * 4;
      auto& transfer = record.transfers[i];
      transfer.source = &record.sources[i];
      transfer.completion = &record.completion;
      transfer.external_view = &record.external[i];
      transfer.binding = binding;
      transfer.local_channel = &channel_;
      transfer.task = &record.task;
      transfer.descriptor_ordinal = i;
      transfer.task_next = i + 1 == count ? nullptr : &record.transfers[i + 1];
      transfer.local_words[0] = transfer.shim_words[0] = 1;
      transfer.local_words[local_.memory_layout->address_word] |=
          loom_xdna_register_field_encode(local_.memory_layout->address,
                                          address + i * 4);
      transfer.local_words[local_.memory_layout->lock_word] |=
          loom_xdna_register_field_encode(local_.memory_layout->locks.valid, 1);
      transfer.shim_words[7] = loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD, 1);
    }
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_select_task_descriptors(
        &context_, &worker_, &worker_, &record.task, false, &descriptor_tail_,
        &completion_tail_, &valid));
    ASSERT_TRUE(valid);
    *task_tail_ = &record.task;
    task_tail_ = &record.task.next;
  }

  void Refine() {
    IREE_ASSERT_OK(loom_aie2p_native_refine_transfer_addresses(&context_, 1));
    IREE_ASSERT_OK(loom_aie2p_native_group_submissions(&context_, 1));
  }

  uint16_t Used(bool shim) {
    uint16_t base, count;
    return *loom_aie2p_native_descriptor_bank(
        shim ? &shim_ : &local_, shim ? path_.shim_engine : path_.local_engine,
        &base, &count);
  }

  // Arena retains the admitted and refined resource inventories.
  iree_arena_block_pool_t blocks_;
  // Compiler scratch for descriptor images and shared submission signatures.
  iree_arena_allocator_t arena_;
  // Production allocation and diagnostic context.
  loom_pass_t pass_ = {};
  // Selected physical execution and resource inventories.
  loom_aie2p_native_context_t context_ = {};
  // Common straight-line control scope with explicit retirement edges.
  loom_block_t block_ = {};
  // On-chip descriptor bank, selected from the actual family facts.
  loom_aie2p_native_tile_t local_ = {};
  // External descriptor bank, shared by both directions.
  loom_aie2p_native_tile_t shim_ = {};
  // One admitted route and engine pair shared by the ordered copies.
  loom_aie2p_native_dma_path_t path_ = {};
  // Resident core owning descriptors, completion credits, and submissions.
  loom_aie2p_native_worker_t worker_ = {};
  // Canonical physical execution owner for this worker.
  iree_host_size_t owner_ = 0;
  // Physical engine selected for the worker.
  iree_host_size_t engine_ = 0;
  // Channel inventory index used for core-local arrival selectors.
  uint32_t channel_index_ = 0;
  // Local channel identity whose completion credit is reused.
  loom_aie2p_native_channel_t channel_ = {};
  // Core-visible channel access with the admitted lock aperture.
  loom_aie2p_native_channel_access_t access_ = {};
  // Existing aperture for completion selectors.
  loom_xdna_address_window_t window_ = {};
  // Next append location in the required descriptor inventory.
  loom_aie2p_native_dma_descriptor_t** descriptor_tail_ = nullptr;
  // Next append location in the required completion-credit inventory.
  loom_aie2p_native_dma_completion_t** completion_tail_ = nullptr;
  // Next append location in the complete ordered task list.
  loom_aie2p_native_transfer_task_t** task_tail_ = nullptr;
  // Stable task/transfer addresses retained by the production planner.
  std::deque<Task> tasks_;
};

TEST_P(DmaAddressVariantsTest, RequiredChainsWinBeforeOptionalFixedAddresses) {
  for (uint32_t i = 0; i < 32; ++i) {
    AddTask(i * 4);
  }
  AddTask(128, 2);
  AddTask(4);
  EXPECT_EQ(Used(false), 3u);
  EXPECT_EQ(Used(true), 3u);
  const uint16_t locks = local_.next_lock;
  const auto* chain_local = tasks_[32].task.local_descriptor;
  const auto* chain_shim = tasks_[32].task.shim_descriptor;
  Refine();
  EXPECT_EQ(Used(false), local_.facts->dma.buffer_descriptor_count /
                             (std::get<0>(GetParam()) ? 2 : 1));
  EXPECT_EQ(Used(true), shim_.facts->dma.buffer_descriptor_count);
  EXPECT_EQ(local_.next_lock, locks);
  EXPECT_EQ(tasks_[32].task.local_descriptor, chain_local);
  EXPECT_EQ(tasks_[32].task.shim_descriptor, chain_shim);
  EXPECT_EQ(chain_local->sequence.descriptor_count, 2u);
  EXPECT_EQ(chain_shim->sequence.descriptor_count, 2u);
  for (bool shim : {false, true}) {
    const auto* first =
        shim ? tasks_[0].task.shim_descriptor : tasks_[0].task.local_descriptor;
    const auto* repeated = shim ? tasks_[33].task.shim_descriptor
                                : tasks_[33].task.local_descriptor;
    EXPECT_TRUE(first->address_dynamic);
    EXPECT_TRUE(first->address_required);
    EXPECT_FALSE(repeated->address_dynamic);
    EXPECT_EQ(repeated, shim ? tasks_[1].task.shim_descriptor
                             : tasks_[1].task.local_descriptor);
    EXPECT_EQ(repeated->address_origin, first);
    EXPECT_EQ(repeated->completion, first->completion);
    EXPECT_EQ(repeated->path, first->path);
  }
}

TEST_P(DmaAddressVariantsTest, FixedChainsRestoreTheirImmutableAddressWords) {
  for (uint32_t i = 0; i < 4; ++i) {
    AddTask(i * 8, 2);
  }
  EXPECT_TRUE(tasks_[0].task.local_descriptor->address_dynamic);
  EXPECT_TRUE(tasks_[0].task.shim_descriptor->address_dynamic);
  Refine();
  EXPECT_EQ(Used(false), 8u);
  EXPECT_EQ(Used(true), 8u);
  for (auto& record : tasks_) {
    for (bool shim : {false, true}) {
      const auto* descriptor =
          shim ? record.task.shim_descriptor : record.task.local_descriptor;
      EXPECT_FALSE(descriptor->address_dynamic);
      for (uint32_t i = 0; i < 2; ++i) {
        const auto& transfer = record.transfers[i];
        EXPECT_EQ(shim ? transfer.shim_descriptor : transfer.local_descriptor,
                  descriptor);
        const auto& layout = *local_.memory_layout;
        const uint8_t chain_word = shim ? 7 : layout.chain.word;
        for (uint8_t word = 0; word < 8; ++word) {
          uint32_t expected =
              shim ? transfer.shim_words[word] : transfer.local_words[word];
          if (word == chain_word && i == 0) {
            expected |= loom_xdna_register_field_encode(
                shim ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD
                     : layout.chain.next,
                descriptor->sequence.descriptor + 1);
            expected |= loom_xdna_register_field_encode(
                shim
                    ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD
                    : layout.chain.enable,
                1);
          }
          EXPECT_EQ(descriptor->sequence.words[i][word], expected);
        }
      }
    }
  }
}

TEST_P(DmaAddressVariantsTest, FixedVariantsShareTheExternalPatchBody) {
  for (uint32_t i = 0; i < 4; ++i) {
    AddTask(i * 8, 2, false, true);
  }
  Refine();
  auto* group = tasks_[0].task.submit_group;
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->use_count, 4u);
  EXPECT_TRUE(group->local_queue_argument);
  EXPECT_EQ(Used(false), 8u);
  EXPECT_EQ(Used(true), 2u);
  for (auto& record : tasks_) {
    EXPECT_EQ(record.task.submit_group, group);
    EXPECT_FALSE(record.task.local_descriptor->address_dynamic);
    EXPECT_TRUE(record.task.shim_descriptor->address_required);
  }
}

TEST_P(DmaAddressVariantsTest, MixedLocalAddressesKeepOneSubmissionBody) {
  // One segment and the largest shareable dynamic chain each retain one body.
  // Required admission sees the later dynamic sites before refinement starts.
  for (uint32_t count : {1u, 4u}) {
    for (uint32_t i = 0; i < 4; ++i) {
      AddTask(i * count * 4, count, false, true);
    }
    for (uint32_t i = 0; i < 3; ++i) {
      AddTask(0, count, true, true);
    }
  }
  const uint16_t locks = local_.next_lock;
  Refine();
  EXPECT_EQ(Used(false), 5u);
  EXPECT_EQ(Used(true), 5u);
  EXPECT_EQ(local_.next_lock, locks);
  EXPECT_NE(tasks_[0].task.submit_group, tasks_[7].task.submit_group);
  for (size_t first : {0u, 7u}) {
    const auto* descriptor = tasks_[first].task.local_descriptor;
    const auto* group = tasks_[first].task.submit_group;
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->use_count, 7u);
    EXPECT_FALSE(group->local_queue_argument);
    EXPECT_TRUE(descriptor->address_dynamic);
    EXPECT_TRUE(descriptor->address_required);
    EXPECT_EQ(descriptor->address_variants, nullptr);
    for (size_t i = first; i < first + 7; ++i) {
      const auto& task = tasks_[i].task;
      EXPECT_EQ(task.submit_group, group);
      EXPECT_EQ(task.local_descriptor, descriptor);
      for (const auto* transfer = task.first_transfer; transfer;
           transfer = transfer->task_next) {
        EXPECT_EQ(transfer->local_descriptor, descriptor);
      }
    }
  }
}

TEST_P(DmaAddressVariantsTest, UnsharedSitesKeepFixedLocalAddresses) {
  AddTask(4, 1, false, true);
  AddTask(0, 1, true, true);
  Refine();
  EXPECT_EQ(Used(false), 2u);
  EXPECT_EQ(Used(true), 1u);
  EXPECT_FALSE(tasks_[0].task.local_descriptor->address_dynamic);
  EXPECT_TRUE(tasks_[1].task.local_descriptor->address_dynamic);
  EXPECT_NE(tasks_[0].task.submit_group, tasks_[1].task.submit_group);
}

TEST_P(DmaAddressVariantsTest, UnshareableChainsKeepFixedLocalAddresses) {
  AddTask(20, 5, false, true);
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(0, 5, true, true);
  }
  Refine();
  // Five dynamic segments need 16 scalar arguments and cannot share a body.
  // The fixed site's smaller signature must still retain its specialization.
  EXPECT_EQ(Used(false), 10u);
  EXPECT_EQ(Used(true), 5u);
  EXPECT_FALSE(tasks_[0].task.local_descriptor->address_dynamic);
  ASSERT_NE(tasks_[0].task.submit_group, nullptr);
  EXPECT_EQ(tasks_[0].task.submit_group->use_count, 1u);
  for (size_t i = 1; i < tasks_.size(); ++i) {
    EXPECT_TRUE(tasks_[i].task.local_descriptor->address_dynamic);
    EXPECT_EQ(tasks_[i].task.submit_group, nullptr);
  }
}

TEST_P(DmaAddressVariantsTest, FixedExternalSitesStillRemoveBothPatches) {
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(i * 4);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(0, 1, true, true);
  }
  Refine();
  EXPECT_EQ(Used(false), 4u);
  EXPECT_EQ(Used(true), 4u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(tasks_[i].task.local_descriptor->address_dynamic);
    EXPECT_FALSE(tasks_[i].task.shim_descriptor->address_dynamic);
    EXPECT_EQ(tasks_[i].task.submit_group, nullptr);
  }
  const auto* group = tasks_[3].task.submit_group;
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->use_count, 3u);
  for (size_t i = 3; i < tasks_.size(); ++i) {
    EXPECT_EQ(tasks_[i].task.submit_group, group);
    EXPECT_TRUE(tasks_[i].task.local_descriptor->address_dynamic);
    EXPECT_TRUE(tasks_[i].task.shim_descriptor->address_dynamic);
  }
}

TEST_P(DmaAddressVariantsTest, ExistingFixedVariantRemainsReusable) {
  AddTask(4);
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(4, 1, false, true);
    AddTask(0, 1, true, true);
  }
  Refine();
  EXPECT_EQ(Used(false), 2u);
  EXPECT_EQ(Used(true), 2u);
  const auto* variant = tasks_[0].task.local_descriptor;
  const auto* fixed = tasks_[1].task.submit_group;
  const auto* dynamic = tasks_[2].task.submit_group;
  ASSERT_NE(fixed, nullptr);
  ASSERT_NE(dynamic, nullptr);
  EXPECT_NE(fixed, dynamic);
  EXPECT_EQ(fixed->use_count, 3u);
  EXPECT_EQ(dynamic->use_count, 3u);
  EXPECT_TRUE(fixed->local_queue_argument);
  EXPECT_FALSE(dynamic->local_queue_argument);
  for (size_t i = 1; i < tasks_.size(); i += 2) {
    EXPECT_EQ(tasks_[i].task.local_descriptor, variant);
    EXPECT_EQ(tasks_[i].task.submit_group, fixed);
    EXPECT_EQ(tasks_[i + 1].task.submit_group, dynamic);
    EXPECT_EQ(tasks_[i + 1].task.local_descriptor, variant->address_origin);
  }
}

TEST_P(DmaAddressVariantsTest, SubmissionGroupsRespectTheNativeRegisterABI) {
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(0, 4, true, true);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    AddTask(0, 5, true, true);
  }
  Refine();
  // Four dynamic segments use 13 scalar arguments; five require 16, while
  // the native ABI admits only 15. The larger tasks still have descriptors.
  ASSERT_NE(tasks_[0].task.submit_group, nullptr);
  EXPECT_EQ(tasks_[0].task.submit_group->use_count, 3u);
  for (size_t i = 3; i < tasks_.size(); ++i) {
    EXPECT_EQ(tasks_[i].task.submit_group, nullptr);
    EXPECT_EQ(tasks_[i].task.local_descriptor->sequence.descriptor_count, 5u);
    EXPECT_TRUE(tasks_[i].task.local_descriptor->address_dynamic);
  }
  EXPECT_EQ(Used(false), 9u);
  EXPECT_EQ(Used(true), 9u);
}

TEST_P(DmaAddressVariantsTest,
       ExternalBindingIdentityDistinguishesFixedAddresses) {
  AddTask(0, 1, false, false, 0);
  AddTask(0, 1, false, false, 1);
  Refine();
  EXPECT_EQ(tasks_[0].task.local_descriptor, tasks_[1].task.local_descriptor);
  EXPECT_NE(tasks_[0].task.shim_descriptor, tasks_[1].task.shim_descriptor);
  EXPECT_FALSE(tasks_[0].task.shim_descriptor->address_dynamic);
  EXPECT_FALSE(tasks_[1].task.shim_descriptor->address_dynamic);
  EXPECT_EQ(Used(false), 1u);
  EXPECT_EQ(Used(true), 2u);
}

INSTANTIATE_TEST_SUITE_P(Endpoints, DmaAddressVariantsTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
