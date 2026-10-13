// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/value_tree.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

TEST(ValueTree, DistinctEmptyForms) {
  for (int64_t token : {1, 2, 3}) {
    EXPECT_TRUE(loom_value_tree_validate(&token, 1, 0));
    EXPECT_FALSE(loom_value_tree_validate(&token, 1, 1));
  }
  const int64_t shape[] = {18, 1, 2, 3, 7, 10, 0, 0};
  EXPECT_TRUE(loom_value_tree_validate(shape, IREE_ARRAYSIZE(shape), 2));
}

TEST(ValueTree, InvalidGeometryAndCoverage) {
  const std::vector<std::vector<int64_t>> shapes = {
      {}, {0, 0}, {6}, {4}, {5}, {-1}, {INT64_MAX}, {0, 1}, {10, 7, 0}};
  for (const auto& shape : shapes) {
    EXPECT_FALSE(loom_value_tree_validate(shape.data(), shape.size(), 0));
    EXPECT_FALSE(loom_value_tree_validate(shape.data(), shape.size(), 1));
  }
  EXPECT_FALSE(loom_value_tree_validate(nullptr, 1, 0));
}

TEST(ValueTree, FullCountAndDepthDomain) {
  std::vector<int64_t> shape(UINT16_MAX, 0);
  shape[0] = ((UINT16_MAX - 1) << 2) | LOOM_VALUE_TREE_RANGE;
  EXPECT_TRUE(
      loom_value_tree_validate(shape.data(), shape.size(), UINT16_MAX - 1));
  EXPECT_FALSE(
      loom_value_tree_validate(shape.data(), shape.size(), UINT16_MAX));
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    shape[i] = 7;
  }
  EXPECT_TRUE(loom_value_tree_validate(shape.data(), shape.size(), 1));
}

}  // namespace
