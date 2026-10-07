// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_producer_index.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

using Node = iree_hal_amd_xdna_queue_producer_node_t;
using Index = iree_hal_amd_xdna_queue_producer_index_t;

struct TreeFacts {
  const Node* minimum = nullptr;
  const Node* maximum = nullptr;
  size_t count = 0;
  uint8_t height = 0;
};

static bool NodeLess(const Node* lhs, const Node* rhs) {
  const uintptr_t lhs_semaphore = reinterpret_cast<uintptr_t>(lhs->semaphore);
  const uintptr_t rhs_semaphore = reinterpret_cast<uintptr_t>(rhs->semaphore);
  if (lhs_semaphore != rhs_semaphore) {
    return lhs_semaphore < rhs_semaphore;
  }
  if (lhs->value != rhs->value) {
    return lhs->value < rhs->value;
  }
  return reinterpret_cast<uintptr_t>(lhs) < reinterpret_cast<uintptr_t>(rhs);
}

static TreeFacts VerifyTree(const Node* node, const Node* expected_parent) {
  if (!node) {
    return {};
  }
  EXPECT_EQ(node->parent, expected_parent);
  TreeFacts left = VerifyTree(node->left, node);
  TreeFacts right = VerifyTree(node->right, node);
  if (node->left) {
    EXPECT_TRUE(NodeLess(left.maximum, node));
  }
  if (node->right) {
    EXPECT_TRUE(NodeLess(node, right.minimum));
  }
  EXPECT_LE(
      std::abs(static_cast<int>(left.height) - static_cast<int>(right.height)),
      1);
  const uint8_t height = 1 + std::max(left.height, right.height);
  EXPECT_EQ(node->height, height);
  return {
      /*.minimum=*/node->left ? left.minimum : node,
      /*.maximum=*/node->right ? right.maximum : node,
      /*.count=*/left.count + right.count + 1,
      /*.height=*/height,
  };
}

TEST(QueueProducerIndexTest, FindsLowerBoundAcrossDeepBacklog) {
  Index index;
  iree_hal_amd_xdna_queue_producer_index_initialize(&index);
  uintptr_t semaphore_storage[4] = {};
  constexpr size_t kNodeCount = 1024;
  std::vector<Node> nodes(kNodeCount);
  std::vector<size_t> order(kNodeCount);
  for (size_t i = 0; i < kNodeCount; ++i) {
    order[i] = i;
    nodes[i].semaphore = reinterpret_cast<iree_hal_semaphore_t*>(
        &semaphore_storage[i % std::size(semaphore_storage)]);
    nodes[i].value = 2 * (i / std::size(semaphore_storage)) + 1;
    nodes[i].producer = &nodes[i];
  }
  std::mt19937 generator(0x58444E41u);
  std::shuffle(order.begin(), order.end(), generator);
  for (size_t i : order) {
    iree_hal_amd_xdna_queue_producer_index_insert(&index, &nodes[i]);
  }

  TreeFacts facts = VerifyTree(index.root, nullptr);
  EXPECT_EQ(facts.count, kNodeCount);
  EXPECT_LE(facts.height, 16);
  for (size_t semaphore_index = 0;
       semaphore_index < std::size(semaphore_storage); ++semaphore_index) {
    iree_hal_semaphore_t* semaphore = reinterpret_cast<iree_hal_semaphore_t*>(
        &semaphore_storage[semaphore_index]);
    for (uint64_t value = 0; value < kNodeCount / 2; ++value) {
      Node* found = iree_hal_amd_xdna_queue_producer_index_find(
          &index, semaphore, value, /*excluded_producer=*/nullptr);
      ASSERT_NE(found, nullptr);
      EXPECT_EQ(found->semaphore, semaphore);
      EXPECT_EQ(found->value, value % 2 ? value : value + 1);
    }
  }

  std::shuffle(order.begin(), order.end(), generator);
  for (size_t position = 0; position < order.size(); ++position) {
    iree_hal_amd_xdna_queue_producer_index_erase(&index,
                                                 &nodes[order[position]]);
    if ((position & 31) == 0) {
      VerifyTree(index.root, nullptr);
    }
  }
  EXPECT_EQ(index.root, nullptr);
}

TEST(QueueProducerIndexTest, ExcludesConsumerOwnedSignal) {
  Index index;
  iree_hal_amd_xdna_queue_producer_index_initialize(&index);
  uintptr_t semaphore_storage = 0;
  iree_hal_semaphore_t* semaphore =
      reinterpret_cast<iree_hal_semaphore_t*>(&semaphore_storage);
  std::array<Node, 3> nodes = {};
  uintptr_t first_producer = 0;
  uintptr_t second_producer = 0;
  nodes[0].semaphore = semaphore;
  nodes[0].value = 3;
  nodes[0].producer = &first_producer;
  nodes[1].semaphore = semaphore;
  nodes[1].value = 3;
  nodes[1].producer = &second_producer;
  nodes[2].semaphore = semaphore;
  nodes[2].value = 5;
  nodes[2].producer = &first_producer;
  for (Node& node : nodes) {
    iree_hal_amd_xdna_queue_producer_index_insert(&index, &node);
  }

  Node* found = iree_hal_amd_xdna_queue_producer_index_find(&index, semaphore,
                                                            3, &first_producer);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->value, 3u);
  EXPECT_EQ(found->producer, &second_producer);
  EXPECT_EQ(iree_hal_amd_xdna_queue_producer_index_find(&index, semaphore, 4,
                                                        &first_producer),
            nullptr);
}

}  // namespace
