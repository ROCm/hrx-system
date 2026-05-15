// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/sequence_window.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SequenceWindowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0, /*initial_capacity=*/4,
        iree_allocator_system(), &window_));
  }

  void TearDown() override { iree_net_sequence_window_deinitialize(&window_); }

  static std::vector<uint64_t> DrainSequences(iree_net_sequence_node_t* head) {
    std::vector<uint64_t> sequences;
    while (head) {
      sequences.push_back(head->sequence);
      head = head->next;
    }
    return sequences;
  }

  iree_net_sequence_window_t window_;
};

TEST_F(SequenceWindowTest, InOrderObservationsAdvancePrefix) {
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 1);
  EXPECT_EQ(ready_list, nullptr);

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 2, &ready_list));
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 2);
  EXPECT_EQ(ready_list, nullptr);
}

TEST_F(SequenceWindowTest, OutOfOrderObservationClosesHole) {
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 2, &ready_list));
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 0);
  EXPECT_TRUE(iree_net_sequence_window_has_observed(&window_, 2));

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 2);
  EXPECT_TRUE(iree_net_sequence_window_has_observed(&window_, 1));
  EXPECT_TRUE(iree_net_sequence_window_has_observed(&window_, 2));
}

TEST_F(SequenceWindowTest, PendingNodesBecomeReadyOnContiguousAdvance) {
  iree_net_sequence_node_t nodes[3] = {};
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(iree_net_sequence_window_defer_until(&window_, 3, &nodes[0],
                                                      &ready_list));
  EXPECT_EQ(ready_list, nullptr);
  IREE_EXPECT_OK(iree_net_sequence_window_defer_until(&window_, 3, &nodes[1],
                                                      &ready_list));
  EXPECT_EQ(ready_list, nullptr);
  IREE_EXPECT_OK(iree_net_sequence_window_defer_until(&window_, 2, &nodes[2],
                                                      &ready_list));
  EXPECT_EQ(ready_list, nullptr);

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 3, &ready_list));
  EXPECT_EQ(ready_list, nullptr);
  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  EXPECT_EQ(ready_list, nullptr);
  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 2, &ready_list));

  std::vector<uint64_t> ready_sequences = DrainSequences(ready_list);
  EXPECT_EQ(ready_sequences, (std::vector<uint64_t>{2, 3, 3}));
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 3);
}

TEST_F(SequenceWindowTest, AlreadyCoveredDeferIsReadyImmediately) {
  iree_net_sequence_node_t* ready_list = nullptr;
  iree_net_sequence_node_t node = {};

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  IREE_EXPECT_OK(
      iree_net_sequence_window_defer_until(&window_, 1, &node, &ready_list));

  ASSERT_EQ(ready_list, &node);
  EXPECT_EQ(ready_list->sequence, 1);
  EXPECT_EQ(ready_list->next, nullptr);
}

TEST_F(SequenceWindowTest, DuplicateObservationFails) {
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_net_sequence_window_observe(&window_, 1, &ready_list));
}

TEST_F(SequenceWindowTest, GrowsForLargeOutOfOrderGap) {
  iree_net_sequence_node_t node = {};
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(
      iree_net_sequence_window_defer_until(&window_, 9, &node, &ready_list));
  EXPECT_EQ(ready_list, nullptr);
  EXPECT_GE(window_.capacity, 16);

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 9, &ready_list));
  EXPECT_TRUE(iree_net_sequence_window_has_observed(&window_, 9));

  for (uint64_t sequence = 1; sequence < 9; ++sequence) {
    IREE_EXPECT_OK(
        iree_net_sequence_window_observe(&window_, sequence, &ready_list));
  }

  ASSERT_EQ(ready_list, &node);
  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 9);
}

TEST_F(SequenceWindowTest, ReusesRingAfterWrap) {
  iree_net_sequence_node_t* ready_list = nullptr;

  for (uint64_t sequence = 1; sequence <= 12; ++sequence) {
    IREE_EXPECT_OK(
        iree_net_sequence_window_observe(&window_, sequence, &ready_list));
    EXPECT_EQ(ready_list, nullptr);
  }

  EXPECT_EQ(iree_net_sequence_window_observed(&window_), 12);
  EXPECT_FALSE(iree_net_sequence_window_has_observed(&window_, 13));
}

TEST_F(SequenceWindowTest, TakesPendingNodesForOwnerCleanup) {
  iree_net_sequence_node_t nodes[2] = {};
  iree_net_sequence_node_t* ready_list = nullptr;

  IREE_EXPECT_OK(iree_net_sequence_window_defer_until(&window_, 3, &nodes[0],
                                                      &ready_list));
  IREE_EXPECT_OK(iree_net_sequence_window_defer_until(&window_, 7, &nodes[1],
                                                      &ready_list));

  iree_net_sequence_node_t* pending_list = nullptr;
  iree_net_sequence_window_take_pending(&window_, &pending_list);
  std::vector<uint64_t> pending_sequences = DrainSequences(pending_list);
  EXPECT_EQ(pending_sequences.size(), 2);

  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 1, &ready_list));
  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 2, &ready_list));
  IREE_EXPECT_OK(iree_net_sequence_window_observe(&window_, 3, &ready_list));
  EXPECT_EQ(ready_list, nullptr);
}

}  // namespace
