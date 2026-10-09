// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleLocationTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleLocationTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      const auto* allocation =
          static_cast<const iree_allocator_alloc_params_t*>(parameters);
      test->largest_allocation_ =
          iree_max(test->largest_allocation_, allocation->byte_length);
      if (test->allocation_count_++ == test->failure_index_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected location allocation failure");
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void CreateModule() {
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("locations"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    IREE_ASSERT_OK(loom_module_register_source(module_, IREE_SV("kernel.loom"),
                                               &source_id_));
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(GetParam(), {this, Allocate}, &pool_);
    ASSERT_NO_FATAL_FAILURE(CreateModule());
  }

  void ResetPool(iree_host_size_t block_size) {
    loom_module_free(module_);
    module_ = nullptr;
    iree_arena_block_pool_deinitialize(&pool_);
    allocation_count_ = 0;
    largest_allocation_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate}, &pool_);
    ASSERT_NO_FATAL_FAILURE(CreateModule());
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_location_id_t AppendLocation(uint16_t line) {
    loom_location_id_t id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module_, loom_location_file_range(source_id_, line, 1, line, 12), &id));
    return id;
  }

  // Number of backing allocation attempts since reset.
  iree_host_size_t allocation_count_ = 0;
  // Largest backing request since reset, in bytes.
  iree_host_size_t largest_allocation_ = 0;
  // Backing allocation ordinal to fail, or SIZE_MAX to allow allocations.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Minimal context; location APIs need no registered operations.
  loom_context_t context_ = {};
  // Shared backing pool retained across module destruction.
  iree_arena_block_pool_t pool_ = {};
  // Owner of the location rows and copied field spans.
  loom_module_t* module_ = nullptr;
  // Registered source identity used by file locations.
  loom_source_id_t source_id_ = LOOM_SOURCE_ID_INVALID;
};

TEST_P(ModuleLocationTest, EmptyAndFirstLocationAllocation) {
  EXPECT_EQ(module_->locations.count, 0u);
  EXPECT_EQ(module_->locations.segments.segment_count, 0u);
  const auto used_before = module_->arena.used_allocation_size;
  const auto id = AppendLocation(7);
  EXPECT_EQ(id, 1u);
  EXPECT_EQ(module_->locations.count, 2u);
  EXPECT_EQ(module_->locations.segments.segment_count, 1u);
  EXPECT_EQ(module_->arena.used_allocation_size - used_before,
            sizeof(loom_location_segment_t));
  EXPECT_EQ(loom_location_table_const_entry(&module_->locations, 0)->kind,
            LOOM_LOCATION_NONE);
  const auto* entry = loom_location_table_const_entry(&module_->locations, id);
  EXPECT_EQ(entry->kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(entry->file.source_id, source_id_);
  EXPECT_EQ(entry->file.start_line, 7u);
}

TEST_P(ModuleLocationTest, GrowthPreservesRowsWithoutAbandonedArrays) {
  constexpr uint32_t kLocationCount =
      LOOM_LOCATION_SEGMENT_CAPACITY *
          LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT +
      3;
  const uint32_t retained_ids[] = {1, 255, 256, 4095, 4096};
  std::array<const loom_location_entry_t*, IREE_ARRAYSIZE(retained_ids)>
      retained = {};
  size_t retained_count = 0;
  const auto used_before = module_->arena.used_allocation_size;
  for (uint32_t i = 1; i <= kLocationCount; ++i) {
    const auto id = AppendLocation(static_cast<uint16_t>(i));
    ASSERT_EQ(id, i);
    if (retained_count < retained.size() &&
        id == retained_ids[retained_count]) {
      retained[retained_count++] =
          loom_location_table_const_entry(&module_->locations, id);
    }
  }
  ASSERT_EQ(retained_count, retained.size());
  for (size_t i = 0; i < retained.size(); ++i) {
    EXPECT_EQ(retained[i], loom_location_table_const_entry(&module_->locations,
                                                           retained_ids[i]));
    EXPECT_EQ(retained[i]->file.start_line, retained_ids[i]);
  }
  const auto segment_count =
      (kLocationCount + 1 + LOOM_LOCATION_SEGMENT_CAPACITY - 1) /
      LOOM_LOCATION_SEGMENT_CAPACITY;
  EXPECT_EQ(module_->locations.count, kLocationCount + 1);
  EXPECT_EQ(module_->locations.segments.segment_count, segment_count);
  EXPECT_EQ(module_->arena.used_allocation_size - used_before,
            segment_count * sizeof(loom_location_segment_t) +
                sizeof(loom_segmented_storage_page_t));
  EXPECT_LE(largest_allocation_, GetParam());

  const auto cold_count = allocation_count_;
  ASSERT_GT(cold_count, 0u);
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    loom_module_free(module_);
    module_ = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateModule());
    for (uint32_t i = 0; i < kLocationCount; ++i) {
      AppendLocation(1);
    }
    EXPECT_EQ(allocation_count_, cold_count);
  }
}

TEST_P(ModuleLocationTest, FieldSpansRemainAttachedToTheStableRow) {
  const auto id = AppendLocation(1);
  const auto* entry = loom_location_table_const_entry(&module_->locations, id);
  for (uint32_t i = 0; i < 2 * LOOM_LOCATION_SEGMENT_CAPACITY; ++i) {
    AppendLocation(2);
  }
  loom_location_field_span_t span = {
      .kind = LOOM_LOCATION_FIELD_OPERAND,
      .index = 0,
      .start_line = 1,
      .start_col = 3,
      .end_line = 1,
      .end_col = 8,
  };
  IREE_ASSERT_OK(
      loom_module_attach_location_field_spans(module_, id, &span, 1));
  EXPECT_EQ(entry, loom_location_table_const_entry(&module_->locations, id));
  ASSERT_EQ(entry->file.field_span_count, 1u);
  ASSERT_NE(entry->file.field_spans, nullptr);
  EXPECT_NE(entry->file.field_spans, &span);
  span.start_col = 10;

  iree_arena_allocator_t scratch;
  iree_arena_initialize(&pool_, &scratch);
  void* scratch_data = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&scratch, GetParam() / 2, &scratch_data));
  iree_arena_deinitialize(&scratch);
  iree_arena_block_pool_trim(&pool_);

  EXPECT_EQ(entry, loom_location_table_const_entry(&module_->locations, id));
  ASSERT_EQ(entry->file.field_span_count, 1u);
  EXPECT_EQ(entry->file.field_spans[0].start_col, 3u);
  EXPECT_EQ(entry->file.field_spans[0].end_col, 8u);
}

TEST_P(ModuleLocationTest, FailedAppendCanRewindAndRetryWithoutChangingIds) {
  const uint32_t boundaries[] = {
      0,
      LOOM_LOCATION_SEGMENT_CAPACITY,
      LOOM_LOCATION_SEGMENT_CAPACITY *
          LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT,
  };
  for (const auto boundary : boundaries) {
    SCOPED_TRACE(boundary);
    const uint32_t expected_id = boundary == 0 ? 1 : boundary;
    for (iree_host_size_t failure_index = 0;; ++failure_index) {
      SCOPED_TRACE(failure_index);
      ASSERT_NO_FATAL_FAILURE(ResetPool(128));
      while (module_->locations.count < boundary) {
        AppendLocation(1);
      }
      const auto before = module_->locations;
      const auto checkpoint = iree_arena_checkpoint_save(&module_->arena);
      allocation_count_ = 0;
      failure_index_ = failure_index;
      loom_location_id_t id = LOOM_LOCATION_UNKNOWN;
      iree_status_t status = loom_module_add_location(
          module_, loom_location_file_range(source_id_, 2, 1, 2, 12), &id);
      failure_index_ = SIZE_MAX;
      if (iree_status_is_ok(status)) {
        EXPECT_LE(allocation_count_, failure_index);
        EXPECT_EQ(id, expected_id);
        break;
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(id, LOOM_LOCATION_UNKNOWN);
      EXPECT_EQ(module_->locations.count, before.count);
      EXPECT_EQ(module_->locations.segments.segment_count,
                before.segments.segment_count);
      EXPECT_EQ(module_->locations.segments.primary_page,
                before.segments.primary_page);
      iree_arena_checkpoint_restore(&checkpoint);
      IREE_ASSERT_OK(loom_module_add_location(
          module_, loom_location_file_range(source_id_, 2, 1, 2, 12), &id));
      EXPECT_EQ(id, expected_id);
      for (uint32_t i = 1; i < before.count; ++i) {
        EXPECT_EQ(loom_location_table_const_entry(&module_->locations, i)
                      ->file.start_line,
                  1u);
      }
      EXPECT_EQ(loom_location_table_const_entry(&module_->locations, id)
                    ->file.start_line,
                2u);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PoolSizes, ModuleLocationTest,
                         ::testing::Values(32 * 1024, 128 * 1024));

}  // namespace
}  // namespace loom
