// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/credits.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"

namespace {

class NativeCreditsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.channels = &channel_;
    context_.workers = workers_;
    context_.core_engines = engines_;
    context_.execution.engine_count = 2;
    context_.execution.schedule_count = 1;
    for (size_t i = 0; i < 3; ++i) {
      invocations_[i].ordinal = i;
      invocations_[i].parent = i ? &invocations_[i - 1] : nullptr;
      entries_[i] = &invocations_[i];
    }
    invocations_[1].schedule = &schedule_;
    for (size_t i = 0; i < 2; ++i) {
      occurrences_[i].invocation = &invocations_[1];
      workers_[i].execution = LOOM_AIE2P_NATIVE_EXECUTION_CORE;
      workers_[i].channels.accesses = &access_;
      workers_[i].channels.indices = &access_index_;
    }
    source_.capacity = 2;
    protocol_.invocation = &invocations_[1];
    protocol_.source = &source_;
    channel_.source = &source_;
    channel_.cursor.writer = {
        .worker = 0,
        .maximum_admissions = 2,
        .exact_admissions = true,
    };
    channel_.cursor.reader = {
        .worker = 1,
        .maximum_admissions = 1,
        .exact_admissions = true,
    };
    realization_.invocations = {
        .entries = entries_,
        .count = 3,
        .workers = occurrences_,
        .worker_count = 2,
        .channels = &protocol_,
        .channel_count = 1,
    };
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  void Select(bool restores, bool quiescent) {
    iree_host_size_t control_count = 0;
    IREE_ASSERT_OK(loom_aie2p_native_select_credits(&context_, &realization_,
                                                    &control_count));
    EXPECT_EQ(context_.credit_scopes != nullptr, restores);
    EXPECT_EQ(loom_aie2p_native_credits_require_quiescence(&context_,
                                                           &invocations_[1]),
              quiescent);
    EXPECT_FALSE(loom_aie2p_native_credits_require_quiescence(
        &context_, &invocations_[0]));
    EXPECT_EQ(control_count, quiescent ? 2u : 0u);
  }

  // Root, serial child, and a concurrent descendant of that child.
  loom_pipeline_invocation_t invocations_[3] = {};
  // Canonical parent-before-child inventory.
  const loom_pipeline_invocation_t* entries_[3] = {};
  // The child's enclosing serial scope supplies its completion join.
  loom_pipeline_schedule_t schedule_ = {};
  // Worker ownership establishes the multiplicity of each endpoint.
  loom_pipeline_invocation_worker_t occurrences_[2] = {};
  // Constructed logical channel and record capacity.
  loom_pipeline_resource_channel_t source_ = {};
  // Lifetime owner of that channel.
  loom_pipeline_invocation_channel_t protocol_ = {};
  // Selected protocol and exact admission counts.
  loom_aie2p_native_channel_t channel_ = {};
  // Existing arithmetic participants eligible to retire the residual.
  loom_aie2p_native_worker_t workers_[2] = {};
  // Selected aperture, retained by a residual action without new allocation.
  loom_aie2p_native_channel_access_t access_ = {};
  // Channel-to-access index for both participants.
  uint32_t access_index_ = 0;
  // Physical engines assigned to the two logical workers.
  const iree_host_size_t engines_[2] = {0, 1};
  // Canonical invocation graph consumed by restoration selection.
  loom_pipeline_realization_t realization_ = {};
  // Native protocol selection owner.
  loom_aie2p_native_context_t context_ = {};
  // Pass allocation context; these legal protocols emit no diagnostics.
  loom_pass_t pass_ = {};
  // Backing for selection results.
  iree_arena_block_pool_t blocks_;
  // Lifetime of one selection result.
  iree_arena_allocator_t arena_;
};

TEST_F(NativeCreditsTest, ParentChannelSurvivesChildCompletion) {
  protocol_.invocation = &invocations_[0];
  Select(false, false);
}

TEST_F(NativeCreditsTest, ExactBalancedEndpointsNeedNoRestoration) {
  channel_.cursor.reader.maximum_admissions = 2;
  channel_.balance.balanced = true;
  Select(false, false);
}

TEST_F(NativeCreditsTest, ExactUnreadTailRetiresWithoutControlRoutes) {
  Select(true, false);
}

TEST_F(NativeCreditsTest, NoReaderRetiresAllPublishedRecords) {
  channel_.cursor.reader.worker = UINT32_MAX;
  Select(true, false);
}

TEST_F(NativeCreditsTest, UnknownReadCountRequiresQuiescence) {
  channel_.cursor.reader.exact_admissions = false;
  Select(true, true);
}

TEST_F(NativeCreditsTest, DifferentInvocationCountsCannotProveBalance) {
  channel_.cursor.reader.maximum_admissions = 2;
  occurrences_[1].invocation = &invocations_[2];
  Select(true, true);
}

TEST_F(NativeCreditsTest, ConcurrentOwnerUsesItsEnclosingSerialBoundary) {
  protocol_.invocation = &invocations_[2];
  occurrences_[0].invocation = &invocations_[2];
  occurrences_[1].invocation = &invocations_[2];
  Select(true, false);
}

TEST_F(NativeCreditsTest, BalancedSharedWorkersNeedNoRestoration) {
  channel_.cursor.reader = channel_.cursor.writer;
  channel_.cursor.reader.shared = channel_.cursor.writer.shared = true;
  channel_.balance.balanced = true;
  Select(false, false);
}

TEST_F(NativeCreditsTest, EarlierUnbalancedWorkerRequiresQuiescence) {
  channel_.cursor.reader = channel_.cursor.writer;
  channel_.cursor.reader.shared = channel_.cursor.writer.shared = true;
  channel_.balance.balanced = false;
  Select(true, true);
}

TEST_F(NativeCreditsTest, RetainedBatchesUsePhysicalCredits) {
  channel_.replay.record_count = 96;
  channel_.replay.credits = 1;
  EXPECT_EQ(loom_aie2p_native_initial_credits(&context_, &channel_), 1u);
  Select(true, true);
}

TEST_F(NativeCreditsTest, ConfigurationReservationIsDebitedOnlyOnce) {
  EXPECT_EQ(loom_aie2p_native_initial_credits(&context_, &channel_), 2u);
  workers_[0].execution = LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
  channel_.cursor.writer.maximum_admissions = 1;
  EXPECT_EQ(loom_aie2p_native_initial_credits(&context_, &channel_), 1u);
  channel_.replay.record_count = 96;
  channel_.replay.credits = 1;
  EXPECT_EQ(loom_aie2p_native_initial_credits(&context_, &channel_), 1u);
}

}  // namespace
