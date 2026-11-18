// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/weights_test_base.h"

namespace loom_runtime::testing {

class WeightLayoutTest : public WeightsTest {};

TEST_P(WeightLayoutTest, ContainedRootBeforeWholeRootSharesPreparedStorage) {
  IREE_ASSERT_OK(Compile(0, "layout_parts"));
  IREE_ASSERT_OK(Compile(1, "layout_whole"));
  // The first stage consumes a small view of the later stage's root and an
  // independent tensor. Both parameter sets must remain distinct and complete.
  IREE_ASSERT_OK(Load(0, 2));
  EXPECT_EQ(loom_runtime_weights_parameter_count(plans_[0]), 4u);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, 4 * 32u);
  EXPECT_EQ(compilation_.commands, 4u);
  Run(0, 93);
  Run(1, 70);
}

TEST_P(WeightLayoutTest, SharedContainedRootsRetainOneOffsetView) {
  IREE_ASSERT_OK(Compile(0, "layout_tail"));
  IREE_ASSERT_OK(Compile(1, "layout_tail"));
  IREE_ASSERT_OK(Compile(2, "layout_whole"));
  // The initial identical group is a tail view of the later complete root.
  // Equal padded root extents must not hide the containment relationship.
  IREE_ASSERT_OK(Load(2, 3));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  EXPECT_EQ(loom_runtime_weights_parameter_count(plans_[0]), 3u);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, 3 * 32u);
  Run(0, 90);
  Run(1, 90);
  Run(2, 70);
}

TEST_P(WeightLayoutTest, PartialOverlapRejectsBeforeStorageOrFileWork) {
  IREE_ASSERT_OK(Compile(0, "layout_whole"));
  IREE_ASSERT_OK(Compile(1, "layout_overlap"));
  iree::Status status(Load(0, 2));
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("parameter 'd'"));
  EXPECT_EQ(buffers_[0][0], nullptr);
  EXPECT_EQ(buffers_[1][0], nullptr);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, 0u);
  EXPECT_EQ(loom_runtime_weights_statistics(plans_[0]).committed_bytes, 0u);
}

TEST_P(WeightLayoutTest, ContainedRootMustPreserveRequiredAlignment) {
  IREE_ASSERT_OK(Compile(0, "layout_whole"));
  IREE_ASSERT_OK(Compile(1, "layout_tail"));
  loom_runtime_weight_root_t roots[] = {Root(0, 0), Root(1, 0)};
  // The tail starts at byte 512; strengthening its alignment cannot be met
  // by changing the base allocation's alignment alone.
  roots[1].root.minimum_alignment = 1024;
  iree::Status status(
      LoadRoots(0, IREE_ARRAYSIZE(roots), roots, {path_.path()}));
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("cannot satisfy alignment 1024"));
  EXPECT_EQ(buffers_[0][0], nullptr);
  EXPECT_EQ(buffers_[1][0], nullptr);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes, 0u);
}

INSTANTIATE_TEST_SUITE_P(Backing, WeightLayoutTest,
                         ::testing::Values(false, true));

}  // namespace loom_runtime::testing
