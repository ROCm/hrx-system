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
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<LivenessPressureTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected pressure allocation failure");
    }
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      const auto* allocation =
          static_cast<const iree_allocator_alloc_params_t*>(parameters);
      if (allocation->byte_length > test->maximum_allocation_size_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "pressure allocation exceeds the test bound");
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage(iree_host_size_t block_size = 4096) {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &scratch_);
    iree_arena_initialize(&pool_, &result_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&result_);
    iree_arena_deinitialize(&scratch_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void SetUp() override { InitializeStorage(); }

  void TearDown() override { DeinitializeStorage(); }

  static loom_liveness_value_class_t RegisterClass() {
    loom_liveness_value_class_t result = {
        .type_kind = LOOM_TYPE_REGISTER,
        .register_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
        .register_descriptor_set_stable_id =
            loom_test_low_core_descriptor_set()->stable_id};
    return result;
  }

  static loom_liveness_value_class_t ScalarClass(loom_scalar_type_t type) {
    loom_liveness_value_class_t result = {
        .type_kind = LOOM_TYPE_SCALAR,
        .element_type = type,
        .register_class_id = LOOM_LOW_REGISTER_CLASS_ID_INVALID};
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
      loom_liveness_interval_t interval = {
          .value_id = id,
          .start_point = segments.front().start_point,
          .end_point = segments.back().end_point,
          .value_class = value_class,
          .unit_count = units};
      intervals_.push_back(interval);
      segments_.insert(segments_.end(), segments.begin(), segments.end());
    }
  }

  loom_liveness_analysis_t Analysis() const {
    loom_liveness_analysis_t analysis = {
        .blocks = blocks_.data(),
        .block_count = blocks_.size(),
        .intervals = intervals_.data(),
        .interval_count = intervals_.size(),
        .value_ids = value_ids_.data(),
        .value_count = value_ids_.size(),
        .value_interval_indices = indices_.data(),
        .segments = segments_.data(),
        .segment_count = segments_.size(),
        .value_segment_ranges = ranges_.data()};
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
        loom_liveness_pressure_summary_t summary = {.value_class =
                                                        interval->value_class};
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

  void AddDensePressureValues() {
    blocks_[0].block = &block_storage_[0];
    blocks_[0].start_point = 0;
    blocks_[0].end_point = 3998;
    blocks_[1].block = &block_storage_[1];
    blocks_[1].start_point = 3999;
    blocks_[1].end_point = 3999;
    // Four thousand point/class cells equal the four thousand endpoints,
    // selecting the dense sweep exactly at its admission boundary.
    for (uint32_t i = 0; i < 2000; ++i) {
      AddValue(3 * i, RegisterClass(), 1 + i % 4, {{0, 3999}});
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
  // Attempted backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Allocation selected for injected failure, or SIZE_MAX.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Request limit for class-scaling checks, independent of host memory size.
  iree_host_size_t maximum_allocation_size_ = SIZE_MAX;
};

TEST_F(LivenessPressureTest, ClassStorageGrowsWithCardinality) {
  maximum_allocation_size_ = 1024 * 1024;
  for (uint32_t count : {8u, 16u, 64u, 257u, 1025u}) {
    for (uint32_t extent : {1u, 65536u}) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(extent);
      value_ids_.clear();
      indices_.clear();
      intervals_.clear();
      ranges_.clear();
      segments_.clear();
      blocks_[0].block = &block_storage_[0];
      blocks_[0].start_point = 0;
      blocks_[0].end_point = extent;
      blocks_[1].block = &block_storage_[1];
      blocks_[1].start_point = extent + 1;
      blocks_[1].end_point = extent + 1;
      for (uint32_t i = 0; i < count; ++i) {
        auto value_class = RegisterClass();
        value_class.register_descriptor_set_stable_id =
            UINT64_C(0x12345678) | (uint64_t(i) << 32);
        AddValue(i, value_class, 1, {{0, extent}});
      }
      const auto analysis = Analysis();
      const loom_liveness_pressure_summary_t* summaries = nullptr;
      iree_host_size_t summary_count = 0;
      IREE_ASSERT_OK(loom_liveness_compute_segment_pressure(
          &analysis, &scratch_, &result_, &summaries, &summary_count));
      ASSERT_EQ(summary_count, count);
      for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(summaries[i].value_class.register_descriptor_set_stable_id,
                  UINT64_C(0x12345678) | (uint64_t(i) << 32));
        EXPECT_EQ(summaries[i].peak_live_units, 1u);
        EXPECT_EQ(summaries[i].peak_live_values, 1u);
        EXPECT_EQ(summaries[i].peak_point, 0u);
      }
      // Geometric class-array growth, endpoint storage, and index maps fit
      // this linear envelope, including arena alignment and small fixed state.
      EXPECT_LE(scratch_.used_allocation_size, size_t(count) * 512 + 4096);
      iree_arena_reset(&scratch_);
      iree_arena_reset(&result_);
    }
  }
}

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

TEST_F(LivenessPressureTest, DenseColumnsReusePoolBlocks) {
  AddDensePressureValues();
  DeinitializeStorage();
  InitializeStorage(32 * 1024);
  ASSERT_NO_FATAL_FAILURE(CheckReference());
  const auto allocation_count = allocation_count_;
  for (int epoch = 0; epoch < 3; ++epoch) {
    SCOPED_TRACE(epoch);
    ASSERT_NO_FATAL_FAILURE(CheckReference());
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&pool_, &statistics);
    EXPECT_EQ(statistics.oversized_allocation_count, 0u);
    EXPECT_EQ(allocation_count_, allocation_count);
  }
}

TEST_F(LivenessPressureTest, BackingFailureLeavesNoPublishedPressure) {
  AddDensePressureValues();
  DeinitializeStorage();
  InitializeStorage(32 * 1024);
  ASSERT_NO_FATAL_FAILURE(CheckReference());
  const auto allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 1u);
  const auto analysis = Analysis();
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DeinitializeStorage();
    InitializeStorage(32 * 1024);
    failure_index_ = i;
    const loom_liveness_pressure_summary_t unpublished = {};
    const loom_liveness_pressure_summary_t* summaries = &unpublished;
    iree_host_size_t count = 1;
    IREE_ASSERT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_liveness_compute_segment_pressure(&analysis, &scratch_, &result_,
                                               &summaries, &count));
    EXPECT_EQ(allocation_count_, i + 1);
    EXPECT_EQ(summaries, nullptr);
    EXPECT_EQ(count, 0u);
    failure_index_ = SIZE_MAX;
    ASSERT_NO_FATAL_FAILURE(CheckReference());
  }
}

}  // namespace
