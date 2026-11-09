// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"

#include "iree/testing/gtest.h"

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

}  // namespace
