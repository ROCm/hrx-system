// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <deque>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/report.h"

namespace loom {
namespace {

using Selection = loom_target_compile_report_source_low_row_t;
using Summary = loom_target_compile_report_source_low_selection_summary_t;

static Selection MakeSelection() {
  return Selection{
      .function_name = IREE_SVL("kernel"),
      .source_op_name = IREE_SVL("scalar.addi"),
      .source_op_kind = 42,
      .selection_kind = LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE,
      .plan_key = IREE_SVL("native_add"),
      .descriptor_key = IREE_SVL("test.add.i32"),
      .descriptor_semantic_tag = IREE_SVL("integer.add.i32"),
      .emitted_low_op_count = 2,
      .execution_count_plus_one = 2,
  };
}

static std::vector<const Summary*> Summaries(
    const loom_target_compile_report_t& report) {
  std::vector<const Summary*> result;
  for (auto* block = report.source_low_selection_summaries.head; block;
       block = block->next) {
    const auto* rows = static_cast<const Summary*>(
        loom_target_compile_report_vec_const_rows(block));
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      result.push_back(&rows[i]);
    }
  }
  return result;
}

struct LimitedAllocator {
  // Number of successful allocations permitted before injecting a failure.
  size_t remaining = 0;
  // Number of live row blocks requiring teardown.
  size_t live = 0;
  // Largest request, bounding storage growth independently of row count.
  size_t maximum_request = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto& allocator = *static_cast<LimitedAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE && allocator.remaining == 0) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    auto backing = iree_allocator_system();
    IREE_RETURN_IF_ERROR(backing.ctl(backing.self, command, params, pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      --allocator.live;
    } else {
      --allocator.remaining;
      ++allocator.live;
      allocator.maximum_request =
          std::max(allocator.maximum_request,
                   static_cast<const iree_allocator_alloc_params_t*>(params)
                       ->byte_length);
    }
    return iree_ok_status();
  }
};

class SelectionSummaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_target_compile_report_initialize(&report_, iree_allocator_system());
  }
  void TearDown() override {
    loom_target_compile_report_deinitialize(&report_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Tests using this helper vary only the numeric operation key.
  static uint32_t CheckTree(const Summary* node, const Summary* lower,
                            const Summary* upper, size_t& count) {
    if (node == nullptr) {
      return 0;
    }
    if (lower != nullptr) {
      EXPECT_GT(node->source_op_kind, lower->source_op_kind);
    }
    if (upper != nullptr) {
      EXPECT_LT(node->source_op_kind, upper->source_op_kind);
    }
    ++count;
    const auto left = CheckTree(node->index.children[0], lower, node, count);
    const auto right = CheckTree(node->index.children[1], node, upper, count);
    EXPECT_LE(left, right + 1);
    EXPECT_LE(right, left + 1);
    EXPECT_EQ(node->index.height, 1 + std::max(left, right));
    return 1 + std::max(left, right);
  }

  static void CheckIndex(const loom_target_compile_report_t& report) {
    size_t count = 0;
    CheckTree(report.source_low_selection_root, nullptr, nullptr, count);
    EXPECT_EQ(count, report.source_low_selection_summaries.count);
  }

  // Owns summary and detail blocks independently of borrowed key strings.
  loom_target_compile_report_t report_ = {};
  // Allocator state outlives report teardown, including assertion failures.
  LimitedAllocator allocator_;
  // Lazily allocated blocks recycled by arena-mode reports.
  iree_arena_block_pool_t pool_ = {};
  // Report storage for the arena lifecycle test.
  iree_arena_allocator_t arena_ = {};
};

TEST_F(SelectionSummaryTest, GroupsEveryKeyFieldByContents) {
  iree_string_view_t Selection::* const fields[] = {
      &Selection::function_name, &Selection::source_op_name,
      &Selection::plan_key, &Selection::descriptor_key,
      &Selection::descriptor_semantic_tag};
  std::vector<Selection> rows = {MakeSelection()};
  for (auto field : fields) {
    auto row = MakeSelection();
    row.*field = IREE_SV("different");
    rows.push_back(row);
    row.*field = iree_string_view_empty();
    rows.push_back(row);
  }
  auto row = MakeSelection();
  ++row.source_op_kind;
  rows.push_back(row);
  row = MakeSelection();
  row.selection_kind = LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN;
  rows.push_back(row);
  for (const auto& selection : rows) {
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_row(&report_, &selection));
  }
  // Equal strings in independent allocations must find the original groups.
  std::deque<std::string> strings;
  for (auto iterator = rows.rbegin(); iterator != rows.rend(); ++iterator) {
    auto copy = *iterator;
    for (auto field : fields) {
      const auto value = copy.*field;
      strings.emplace_back(value.size ? std::string(value.data, value.size)
                                      : "");
      copy.*field =
          iree_make_string_view(strings.back().data(), strings.back().size());
      ASSERT_NE((copy.*field).data, value.data);
    }
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_row(&report_, &copy));
  }
  const auto summaries = Summaries(report_);
  ASSERT_EQ(summaries.size(), rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(summaries[i]->source_op_kind, rows[i].source_op_kind);
    EXPECT_EQ(summaries[i]->selection_kind, rows[i].selection_kind);
    EXPECT_EQ(summaries[i]->selected_op_count, 2u);
    EXPECT_EQ(summaries[i]->emitted_low_op_count, 4u);
    const iree_string_view_t values[] = {
        summaries[i]->function_name, summaries[i]->source_op_name,
        summaries[i]->plan_key, summaries[i]->descriptor_key,
        summaries[i]->descriptor_semantic_tag};
    for (size_t j = 0; j < IREE_ARRAYSIZE(fields); ++j) {
      EXPECT_TRUE(iree_string_view_equal(values[j], rows[i].*fields[j]));
    }
  }
}

TEST_F(SelectionSummaryTest, PreservesCountsDynamicEvidenceAndPeaks) {
  auto row = MakeSelection();
  const uint64_t executions[] = {0, 1, 2, 7, UINT64_MAX};
  const uint32_t emitted[] = {9, 3, 4, 5, 3};
  for (size_t i = 0; i < IREE_ARRAYSIZE(executions); ++i) {
    row.execution_count_plus_one = executions[i];
    row.emitted_low_op_count = emitted[i];
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_row(&report_, &row));
  }
  // A zero-emission selection has no summary even when its key is distinct.
  row.plan_key = IREE_SV("elided");
  row.emitted_low_op_count = 0;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_row(&report_, &row));
  row.plan_key = row.descriptor_key = row.descriptor_semantic_tag =
      iree_string_view_empty();
  row.emitted_low_op_count = 17;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_row(&report_, &row));
  EXPECT_EQ(report_.source_low_unkeyed_maximum_emitted_op_count, 17u);
  const auto summaries = Summaries(report_);
  ASSERT_EQ(summaries.size(), 1u);
  EXPECT_EQ(summaries[0]->selected_op_count, 5u);
  EXPECT_EQ(summaries[0]->emitted_low_op_count, 24u);
  EXPECT_EQ(summaries[0]->maximum_emitted_low_op_count, 9u);
  EXPECT_EQ(summaries[0]->exact_dynamic_op_count, 3u);
  EXPECT_EQ(summaries[0]->unknown_dynamic_op_count, 2u);
  EXPECT_EQ(summaries[0]->dynamic_selected_op_count, 7u);
  EXPECT_EQ(summaries[0]->dynamic_emitted_low_op_count, 34u);
}

TEST_F(SelectionSummaryTest, BalancesOrderedAndPermutedKeysAcrossBlocks) {
  constexpr uint32_t kCount = 2048;
  // Ascending, descending and permuted insertion exercise both rotations.
  for (uint32_t multiplier : {1u, kCount - 1, 129u}) {
    SCOPED_TRACE(multiplier);
    std::vector<uint32_t> order;
    auto row = MakeSelection();
    for (uint32_t i = 0; i < kCount; ++i) {
      row.source_op_kind = (i * multiplier) % kCount;
      order.push_back(row.source_op_kind);
      IREE_ASSERT_OK(
          loom_target_compile_report_record_source_low_row(&report_, &row));
      // Check every insertion while small, then the final multi-block index.
      if (i < 16) {
        CheckIndex(report_);
      }
    }
    for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
      row.source_op_kind = *iterator;
      IREE_ASSERT_OK(
          loom_target_compile_report_record_source_low_row(&report_, &row));
    }
    CheckIndex(report_);
    const auto summaries = Summaries(report_);
    ASSERT_EQ(summaries.size(), kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
      EXPECT_EQ(summaries[i]->source_op_kind, order[i]);
      EXPECT_EQ(summaries[i]->selected_op_count, 2u);
    }
    loom_target_compile_report_deinitialize(&report_);
    loom_target_compile_report_initialize(&report_, iree_allocator_system());
  }
}

TEST_F(SelectionSummaryTest, BalancesEveryThreeKeyInsertionOrder) {
  std::array<uint32_t, 3> order = {1, 2, 3};
  do {
    auto row = MakeSelection();
    for (auto key : order) {
      row.source_op_kind = key;
      IREE_ASSERT_OK(
          loom_target_compile_report_record_source_low_row(&report_, &row));
      CheckIndex(report_);
    }
    EXPECT_EQ(report_.source_low_selection_root->source_op_kind, 2u);
    loom_target_compile_report_deinitialize(&report_);
    loom_target_compile_report_initialize(&report_, iree_allocator_system());
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_F(SelectionSummaryTest, MergingOwnsIndexAfterEntryStorageIsReleased) {
  auto row = MakeSelection();
  for (uint32_t entry_index = 0; entry_index < 3; ++entry_index) {
    loom_target_compile_report_t entry;
    loom_target_compile_report_initialize(&entry, iree_allocator_system());
    for (uint32_t i = 0; i < 256; ++i) {
      row.source_op_kind = entry_index * 128 + i;
      IREE_ASSERT_OK(
          loom_target_compile_report_record_source_low_row(&entry, &row));
    }
    IREE_ASSERT_OK(
        loom_target_compile_report_record_entry_report(&report_, &entry));
    loom_target_compile_report_deinitialize(&entry);
    CheckIndex(report_);
  }
  for (uint32_t i = 0; i < 512; ++i) {
    row.source_op_kind = 511 - i;
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_row(&report_, &row));
  }
  const auto summaries = Summaries(report_);
  ASSERT_EQ(summaries.size(), 512u);
  for (uint32_t i = 0; i < 512; ++i) {
    EXPECT_EQ(summaries[i]->source_op_kind, i);
    EXPECT_EQ(summaries[i]->selected_op_count, i >= 128 && i < 384 ? 3u : 2u);
  }
}

TEST_F(SelectionSummaryTest, ArenaResetAndDisabledStorage) {
  auto row = MakeSelection();
  for (int repetition = 0; repetition < 2; ++repetition) {
    loom_target_compile_report_initialize(&report_,
                                          iree_arena_allocator(&arena_));
    for (uint32_t i = 0; i < 100; ++i) {
      row.source_op_kind = i;
      IREE_ASSERT_OK(
          loom_target_compile_report_record_source_low_row(&report_, &row));
    }
    CheckIndex(report_);
    loom_target_compile_report_deinitialize(&report_);
    EXPECT_EQ(report_.source_low_selection_root, nullptr);
    iree_arena_reset(&arena_);
  }
  loom_target_compile_report_initialize(&report_, iree_allocator_null());
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_row(&report_, &row));
  EXPECT_EQ(report_.source_low_rows.count, 0u);
  EXPECT_EQ(report_.source_low_selection_summaries.count, 0u);
  EXPECT_EQ(report_.source_low_selection_root, nullptr);
}

TEST_F(SelectionSummaryTest, FailedAppendLeavesIndexUsableAndReleasesBlocks) {
  for (size_t allowance = 0; allowance < 16; ++allowance) {
    SCOPED_TRACE(allowance);
    allocator_ = LimitedAllocator{allowance};
    loom_target_compile_report_initialize(
        &report_, {&allocator_, LimitedAllocator::Control});
    auto row = MakeSelection();
    iree_status_t status = iree_ok_status();
    uint32_t recorded = 0;
    while (recorded < 1024 && iree_status_is_ok(status)) {
      row.source_op_kind = recorded;
      status = loom_target_compile_report_record_source_low_row(&report_, &row);
      if (iree_status_is_ok(status)) {
        ++recorded;
      }
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(report_.source_low_selection_summaries.count, recorded);
    CheckIndex(report_);
    allocator_.remaining = 1024;
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_row(&report_, &row));
    EXPECT_EQ(report_.source_low_selection_summaries.count, recorded + 1);
    CheckIndex(report_);
    loom_target_compile_report_deinitialize(&report_);
    EXPECT_EQ(allocator_.live, 0u);
    EXPECT_LE(allocator_.maximum_request, 4096u);
  }
}

}  // namespace
}  // namespace loom
