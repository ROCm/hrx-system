// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/transfer_plan.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TransferPlanTest : public ::testing::Test {
 protected:
  void Append(std::array<uint64_t, 5> fields) {
    size_t begin = bytes_.size();
    bytes_.resize(begin + LOOM_SERVE_TRANSFER_RANGE_BYTES);
    for (size_t i = 0; i < fields.size(); ++i) {
      iree_unaligned_store_le_u64(bytes_.data() + begin + i * sizeof(uint64_t),
                                  fields[i]);
    }
  }

  iree_status_t Initialize() {
    return loom_serve_transfer_plan_initialize(
        iree_make_const_byte_span(bytes_.data(), bytes_.size()),
        sources_.size(), sources_.data(), targets_.size(), targets_.data(),
        &plan_);
  }

  // Untrusted source-authored records, borrowed by the checked view.
  std::vector<uint8_t> bytes_;
  // Native binding extents are independent of the text service's packet count.
  std::array<uint64_t, 3> sources_ = {64, 256, UINT64_MAX};
  // Target bindings need not mirror source counts or capacities.
  std::array<uint64_t, 4> targets_ = {1024, 2048, 4096, UINT64_MAX};
  // Checked descriptor view, never owning payload or transport state.
  loom_serve_transfer_plan_t plan_ = {};
};

TEST_F(TransferPlanTest, IndependentBindingsOffsetsAndRepeatedSources) {
  Append({1, 200, 0, 64, 56});
  Append({1, 200, 0, 512, 56});
  Append({0, 0, 2, 4000, 64});
  Append({2, UINT64_MAX - 4, 3, UINT64_MAX - 8, 4});
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(plan_.count, 4u);
  EXPECT_EQ(plan_.records, bytes_.data());
  const auto first = loom_serve_transfer_plan_range(plan_, 0);
  EXPECT_EQ(first.source_binding, 1u);
  EXPECT_EQ(first.source_offset, 200u);
  EXPECT_EQ(first.target_binding, 0u);
  EXPECT_EQ(first.target_offset, 64u);
  EXPECT_EQ(first.length, 56u);
  const auto last = loom_serve_transfer_plan_range(plan_, 3);
  EXPECT_EQ(last.source_binding, 2u);
  EXPECT_EQ(last.source_offset, UINT64_MAX - 4);
  EXPECT_EQ(last.target_binding, 3u);
  EXPECT_EQ(last.target_offset, UINT64_MAX - 8);
  EXPECT_EQ(last.length, 4u);
}

TEST_F(TransferPlanTest, EmptyAndInertRanges) {
  IREE_ASSERT_OK(loom_serve_transfer_plan_initialize(
      iree_const_byte_span_empty(), 0, nullptr, 0, nullptr, &plan_));
  EXPECT_EQ(plan_.count, 0u);
  Append({0, 0, 2, 4000, 64});
  // An inert lower-numbered target does not impose an execution dependency.
  Append({0, 64, 0, 1024, 0});
  Append({2, UINT64_MAX, 3, UINT64_MAX, 0});
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(plan_.count, 3u);
}

TEST_F(TransferPlanTest, ValidatesEveryExternalFieldAndClearsFailedView) {
  const std::array<std::array<uint64_t, 5>, 8> invalid = {{
      {3, 0, 3, 0, 1},
      {0, 0, 4, 0, 1},
      {0, 65, 3, 0, 0},
      {0, 63, 3, 0, 2},
      {0, 0, 3, UINT64_MAX, 1},
      {2, UINT64_MAX, 3, 0, 1},
      {2, 1, 3, 0, UINT64_MAX},
      {2, 0, 3, 1, UINT64_MAX},
  }};
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    bytes_.clear();
    Append({0, 0, 0, 0, 1});
    IREE_ASSERT_OK(Initialize());
    Append(invalid[i]);
    IREE_EXPECT_STATUS_IS(
        i < 2 ? IREE_STATUS_INVALID_ARGUMENT : IREE_STATUS_OUT_OF_RANGE,
        Initialize());
    EXPECT_EQ(plan_.records, nullptr);
    EXPECT_EQ(plan_.count, 0u);
  }
  bytes_.assign(LOOM_SERVE_TRANSFER_RANGE_BYTES - 1, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Initialize());
  EXPECT_EQ(plan_.count, 0u);
}

TEST_F(TransferPlanTest, RejectsOverlappingOrUnorderedTargets) {
  const std::array<std::array<uint64_t, 5>, 3> invalid = {{
      {0, 0, 1, 12, 1},
      {0, 0, 1, 4, 4},
      {0, 0, 0, 512, 1},
  }};
  for (const auto& next : invalid) {
    bytes_.clear();
    Append({0, 0, 1, 8, 16});
    Append(next);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Initialize());
  }
  bytes_.clear();
  Append({0, 0, 1, 8, 16});
  Append({0, 0, 1, 24, 16});
  IREE_ASSERT_OK(Initialize());
}

}  // namespace
