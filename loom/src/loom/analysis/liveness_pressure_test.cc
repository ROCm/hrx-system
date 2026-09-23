// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_pressure.h"

#include <algorithm>
#include <array>
#include <random>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace {

class LivenessPressureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &scratch_);
    iree_arena_initialize(&pool_, &result_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&result_);
    iree_arena_deinitialize(&scratch_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_liveness_value_class_t RegisterClass() {
    loom_liveness_value_class_t result = {};
    result.type_kind = LOOM_TYPE_REGISTER;
    result.register_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_I32;
    result.register_descriptor_set_stable_id =
        loom_test_low_core_descriptor_set()->stable_id;
    return result;
  }

  static loom_liveness_value_class_t ScalarClass(loom_scalar_type_t type) {
    loom_liveness_value_class_t result = {};
    result.type_kind = LOOM_TYPE_SCALAR;
    result.element_type = type;
    result.register_class_id = LOOM_LOW_REGISTER_CLASS_ID_INVALID;
    return result;
  }

  // Canonical interval indices deliberately differ from value ordinals when
  // unused values are present. Each supplied segment is already block-local.
  void AddValue(loom_value_id_t id, loom_liveness_value_class_t value_class,
                uint32_t units,
                const std::vector<loom_liveness_segment_t>& segments) {
    value_ids_.push_back(id);
    ranges_.push_back({static_cast<uint32_t>(segments_.size()),
                       static_cast<uint32_t>(segments.size())});
    indices_.push_back(segments.empty()
                           ? UINT32_MAX
                           : static_cast<uint32_t>(intervals_.size()));
    if (!segments.empty()) {
      loom_liveness_interval_t interval = {};
      interval.value_id = id;
      interval.start_point = segments.front().start_point;
      interval.end_point = segments.back().end_point;
      interval.value_class = value_class;
      interval.unit_count = units;
      intervals_.push_back(interval);
      segments_.insert(segments_.end(), segments.begin(), segments.end());
    }
  }

  loom_liveness_analysis_t Analysis() const {
    loom_liveness_analysis_t analysis = {};
    analysis.blocks = blocks_.data();
    analysis.block_count = blocks_.size();
    analysis.intervals = intervals_.data();
    analysis.interval_count = intervals_.size();
    analysis.value_ids = value_ids_.data();
    analysis.value_count = value_ids_.size();
    analysis.value_interval_indices = indices_.data();
    analysis.segments = segments_.data();
    analysis.segment_count = segments_.size();
    analysis.value_segment_ranges = ranges_.data();
    return analysis;
  }

  // Brute-force live membership at each distinct endpoint, independent of
  // endpoint deltas, dense tables, and the production class index.
  std::vector<loom_liveness_pressure_summary_t> Reference() const {
    std::vector<std::pair<uint32_t, loom_value_id_t>> starts;
    std::vector<uint32_t> points;
    for (size_t i = 0; i < value_ids_.size(); ++i) {
      const auto range = ranges_[i];
      if (range.count != 0) {
        starts.emplace_back(segments_[range.start].start_point, value_ids_[i]);
      }
      for (uint32_t j = 0; j < range.count; ++j) {
        points.push_back(segments_[range.start + j].start_point);
        points.push_back(segments_[range.start + j].end_point);
      }
    }
    std::sort(starts.begin(), starts.end());
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    std::vector<loom_liveness_pressure_summary_t> summaries;
    for (const auto& start : starts) {
      const auto interval = std::find_if(
          intervals_.begin(), intervals_.end(),
          [&](const auto& entry) { return entry.value_id == start.second; });
      if (std::none_of(summaries.begin(), summaries.end(),
                       [&](const auto& summary) {
                         return loom_liveness_value_class_equal(
                             summary.value_class, interval->value_class);
                       })) {
        loom_liveness_pressure_summary_t summary = {};
        summary.value_class = interval->value_class;
        summaries.push_back(summary);
      }
    }
    for (uint32_t point : points) {
      for (auto& summary : summaries) {
        uint32_t units = 0;
        uint32_t values = 0;
        for (size_t i = 0; i < value_ids_.size(); ++i) {
          const auto range = ranges_[i];
          if (range.count == 0 ||
              !loom_liveness_value_class_equal(
                  intervals_[indices_[i]].value_class, summary.value_class)) {
            continue;
          }
          for (uint32_t j = 0; j < range.count; ++j) {
            const auto& segment = segments_[range.start + j];
            if (segment.start_point <= point && point < segment.end_point) {
              units += intervals_[indices_[i]].unit_count;
              ++values;
            }
          }
        }
        if (std::pair(units, values) >
            std::pair(summary.peak_live_units, summary.peak_live_values)) {
          summary.peak_live_units = units;
          summary.peak_live_values = values;
          summary.peak_point = point;
          summary.peak_block = nullptr;
          for (const auto& block : blocks_) {
            if (block.start_point <= point && point <= block.end_point) {
              summary.peak_block = block.block;
            }
          }
        }
      }
    }
    return summaries;
  }

  void CheckReference() {
    iree_arena_reset(&result_);
    iree_arena_reset(&scratch_);
    const auto expected = Reference();
    const auto analysis = Analysis();
    const loom_liveness_pressure_summary_t* actual = nullptr;
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(loom_liveness_compute_segment_pressure(
        &analysis, &scratch_, &result_, &actual, &count));
    ASSERT_EQ(count, expected.size());
    for (size_t i = 0; i < count; ++i) {
      SCOPED_TRACE(i);
      EXPECT_TRUE(loom_liveness_value_class_equal(actual[i].value_class,
                                                  expected[i].value_class));
      EXPECT_EQ(actual[i].peak_live_units, expected[i].peak_live_units);
      EXPECT_EQ(actual[i].peak_live_values, expected[i].peak_live_values);
      EXPECT_EQ(actual[i].peak_point, expected[i].peak_point);
      EXPECT_EQ(actual[i].peak_block, expected[i].peak_block);
      EXPECT_EQ(actual[i].peak_op, nullptr);
    }
  }

  // Storage shared by scratch and result arenas.
  iree_arena_block_pool_t pool_;
  // Transient pressure computation storage.
  iree_arena_allocator_t scratch_;
  // Published pressure summary storage.
  iree_arena_allocator_t result_;
  // Stable block identities used by the canonical point ranges.
  std::array<loom_block_t, 2> block_storage_ = {};
  // Ordered block point ranges, including exit points.
  std::array<loom_liveness_block_info_t, 2> blocks_ = {};
  // Non-contiguous SSA identities in local ordinal order.
  std::vector<loom_value_id_t> value_ids_;
  // Local ordinal to interval index map, including unused values.
  std::vector<uint32_t> indices_;
  // Canonical intervals for values with live ranges.
  std::vector<loom_liveness_interval_t> intervals_;
  // Per-value ranges into segments_.
  std::vector<loom_liveness_segment_range_t> ranges_;
  // Increasing, disjoint segments for each value.
  std::vector<loom_liveness_segment_t> segments_;
};

TEST_F(LivenessPressureTest, EmptyRangesNeedNoStorage) {
  CheckReference();
  EXPECT_EQ(scratch_.used_allocation_size, 0u);
  EXPECT_EQ(result_.used_allocation_size, 0u);
}

TEST_F(LivenessPressureTest, MatchesMembershipAcrossDenseAndSparseDomains) {
  std::mt19937 random(0xC1A55);
  const std::array classes = {
      RegisterClass(),
      ScalarClass(LOOM_SCALAR_TYPE_I32),
      ScalarClass(LOOM_SCALAR_TYPE_F32),
      ScalarClass(LOOM_SCALAR_TYPE_I1),
      ScalarClass(LOOM_SCALAR_TYPE_I8),
      ScalarClass(LOOM_SCALAR_TYPE_I16),
      ScalarClass(LOOM_SCALAR_TYPE_I64),
      ScalarClass(LOOM_SCALAR_TYPE_F16),
      ScalarClass(LOOM_SCALAR_TYPE_BF16),
      ScalarClass(LOOM_SCALAR_TYPE_F64),
  };
  for (uint32_t scale : {1u, 65536u}) {
    for (uint32_t value_count : {3u, 64u, 512u}) {
      SCOPED_TRACE(scale);
      SCOPED_TRACE(value_count);
      value_ids_.clear();
      indices_.clear();
      intervals_.clear();
      ranges_.clear();
      segments_.clear();
      for (uint32_t block = 0; block < 2; ++block) {
        blocks_[block].block = &block_storage_[block];
        blocks_[block].start_point = block * 32 * scale;
        blocks_[block].end_point = (block + 1) * 32 * scale - 1;
      }
      AddValue(7, classes[0], 1, {});
      for (uint32_t value = 0; value < value_count; ++value) {
        std::vector<loom_liveness_segment_t> segments;
        for (uint32_t block = 0; block < 2; ++block) {
          const uint32_t start = random() % 16;
          const uint32_t end = 17 + random() % 16;
          segments.push_back(
              {(block * 32 + start) * scale, (block * 32 + end) * scale});
        }
        const uint32_t class_count = value_count == 512 ? classes.size() : 3;
        const uint32_t class_index = value % class_count;
        AddValue((value_count - value) * 11, classes[class_index],
                 class_index == 0 ? 1u + random() % 4 : 1u, segments);
      }
      CheckReference();
    }
  }
}

TEST_F(LivenessPressureTest, PreservesBoundaryTiesAndFirstClassOrder) {
  blocks_[0].block = &block_storage_[0];
  blocks_[0].start_point = 0;
  blocks_[0].end_point = 5;
  blocks_[1].block = &block_storage_[1];
  blocks_[1].start_point = 6;
  blocks_[1].end_point = 8;
  const auto registers = RegisterClass();
  AddValue(40, registers, 4, {{0, 2}, {6, 9}});
  AddValue(3, ScalarClass(LOOM_SCALAR_TYPE_F32), 1, {{0, 6}});
  for (uint32_t id = 50; id < 54; ++id) {
    AddValue(id, registers, 1, {{2, 6}});
  }
  // Equal unit pressure at the half-open boundary favors four live values;
  // the later tuple is a repeated unit peak. Live-out ends one past the last
  // block exit, and class order is independent of local value ordinal order.
  const auto expected = Reference();
  ASSERT_EQ(expected.size(), 2u);
  EXPECT_EQ(expected[0].value_class.element_type, LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(expected[1].peak_live_units, 4u);
  EXPECT_EQ(expected[1].peak_live_values, 4u);
  EXPECT_EQ(expected[1].peak_point, 2u);
  CheckReference();
}

TEST_F(LivenessPressureTest, SparseSegmentsAtPointDomainLimit) {
  blocks_[0].block = &block_storage_[0];
  blocks_[0].start_point = 0;
  blocks_[0].end_point = UINT32_MAX - 3;
  blocks_[1].block = &block_storage_[1];
  blocks_[1].start_point = UINT32_MAX - 2;
  blocks_[1].end_point = UINT32_MAX;
  AddValue(12, RegisterClass(), 4, {{UINT32_MAX - 2, UINT32_MAX}});
  AddValue(2, ScalarClass(LOOM_SCALAR_TYPE_F32), 1,
           {{UINT32_MAX - 1, UINT32_MAX}});
  CheckReference();
}

TEST_F(LivenessPressureTest, ChecksLiveWidthAfterSimultaneousEndpoints) {
  blocks_[0].block = &block_storage_[0];
  blocks_[0].start_point = 0;
  blocks_[0].end_point = 2;
  blocks_[1].block = &block_storage_[1];
  blocks_[1].start_point = 3;
  blocks_[1].end_point = 3;
  AddValue(4, RegisterClass(), UINT32_MAX, {{0, 1}});
  AddValue(5, RegisterClass(), UINT32_MAX, {{1, 2}});
  // A wide value can replace another at the same point without overflowing
  // pressure. Adding an overlapping unit must report the unrepresentable peak.
  CheckReference();
  AddValue(6, RegisterClass(), 1, {{0, 1}});
  const auto analysis = Analysis();
  const loom_liveness_pressure_summary_t* summaries = nullptr;
  iree_host_size_t count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_liveness_compute_segment_pressure(&analysis, &scratch_, &result_,
                                             &summaries, &count));
}

}  // namespace
