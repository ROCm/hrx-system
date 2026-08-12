// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/input_ring.h"

#include "iree/testing/gtest.h"

namespace {

constexpr uint64_t kDefaultTargetBytes = 32 * 1024 * 1024;

TEST(InputRingTest, LargeBindingSetIsNotDuplicated) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*binding_set_bytes=*/110729216,
                /*dispatches_per_batch=*/64),
            1u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*binding_set_bytes=*/UINT64_C(7) * 1024 * 1024 * 1024,
                /*dispatches_per_batch=*/64),
            1u);
}

TEST(InputRingTest, AutomaticCountApproachesByteTargetWithinBatch) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*binding_set_bytes=*/1024 * 1024,
                /*dispatches_per_batch=*/64),
            32u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*binding_set_bytes=*/1,
                /*dispatches_per_batch=*/64),
            64u);
}

TEST(InputRingTest, DisabledOrEmptyAutomaticRingUsesOneSet) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, /*requested_min_bytes=*/0,
                /*binding_set_bytes=*/1024,
                /*dispatches_per_batch=*/64),
            1u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*binding_set_bytes=*/0,
                /*dispatches_per_batch=*/64),
            1u);
}

TEST(InputRingTest, ExplicitCountIsExact) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/96, kDefaultTargetBytes,
                /*binding_set_bytes=*/110729216,
                /*dispatches_per_batch=*/64),
            96u);
}

}  // namespace
