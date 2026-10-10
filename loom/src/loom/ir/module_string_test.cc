// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleStringTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleStringTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      const auto* allocation =
          static_cast<const iree_allocator_alloc_params_t*>(parameters);
      test->largest_allocation_ =
          iree_max(test->largest_allocation_, allocation->byte_length);
      if (test->allocation_count_++ == test->failure_index_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected string allocation failure");
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(32768, {this, Allocate}, &block_pool_);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&block_pool_);
    loom_context_deinitialize(&context_);
  }

  void ResetPool(iree_host_size_t block_size) {
    loom_module_free(module_);
    module_ = nullptr;
    iree_arena_block_pool_deinitialize(&block_pool_);
    allocation_count_ = 0;
    largest_allocation_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate},
                                     &block_pool_);
  }

  void CreateModule(const loom_module_size_hints_t* hints = nullptr) {
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("strings"),
                                        &block_pool_, hints,
                                        iree_allocator_system(), &module_));
  }

  loom_string_id_t Intern(iree_string_view_t value) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, value, &id));
    return id;
  }

  loom_string_id_t AppendNumberedString(uint32_t number) {
    char buffer[32];
    const int length =
        iree_snprintf(buffer, sizeof(buffer), "string_%u", number);
    return Intern(iree_make_string_view(buffer, length));
  }

  // Number of backing allocation attempts since the last pool reset.
  iree_host_size_t allocation_count_ = 0;
  // Largest backing request since the last pool reset, in bytes.
  iree_host_size_t largest_allocation_ = 0;
  // Backing allocation ordinal to fail, or SIZE_MAX to allow all allocations.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Minimal context; string interning has no operation or parser dependency.
  loom_context_t context_ = {};
  // Shared storage retained across module destruction for reuse tests.
  iree_arena_block_pool_t block_pool_ = {};
  // Optional module whose string rows and bytes are under test.
  loom_module_t* module_ = nullptr;
};

TEST_F(ModuleStringTest, InternString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);

  iree_string_view_t stored = loom_string_table_get(&module->strings, id);
  EXPECT_TRUE(iree_string_view_equal(stored, IREE_SV("hello")));
  loom_module_free(module);
}

TEST_F(ModuleStringTest, InternStringDedup) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id2));
  EXPECT_EQ(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleStringTest, InternDifferentStrings) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id1 = LOOM_STRING_ID_INVALID;
  loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("hello"), &id1));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("world"), &id2));
  EXPECT_NE(id1, id2);
  loom_module_free(module);
}

TEST_F(ModuleStringTest, InternEmptyString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_string_id_t id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, iree_string_view_empty(), &id));
  EXPECT_NE(id, LOOM_STRING_ID_INVALID);
  iree_string_view_t stored = loom_string_table_get(&module->strings, id);
  EXPECT_EQ(stored.size, 0u);
  loom_module_free(module);
}

TEST_F(ModuleStringTest, InternStringStress) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  // Intern 1000 unique strings and verify dedup.
  char buffer[32];
  for (int i = 0; i < 1000; ++i) {
    int length = iree_snprintf(buffer, sizeof(buffer), "string_%d", i);
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id));
    // Intern again, expect same ID.
    loom_string_id_t id2 = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_string_view(buffer, length), &id2));
    EXPECT_EQ(id, id2);
  }
  loom_module_free(module);
}

TEST_F(ModuleStringTest, LookupString) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t hello_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &hello_id));

  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("hello")), hello_id);
  EXPECT_EQ(loom_module_lookup_string(module, IREE_SV("missing")),
            LOOM_STRING_ID_INVALID);
  EXPECT_EQ(module->strings.count, 2u);  // "test" module name + "hello".

  loom_module_free(module);
}

TEST_F(ModuleStringTest,
       InternStringRejectsInvalidSentinelIdButKeepsDedupWorking) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_string_id_t existing_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &existing_id));

  module->strings.count = LOOM_STRING_ID_INVALID;

  loom_string_id_t duplicate_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("hello"), &duplicate_id));
  EXPECT_EQ(duplicate_id, existing_id);

  loom_string_id_t new_id = LOOM_STRING_ID_INVALID;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_intern_string(module, IREE_SV("world"), &new_id));

  loom_module_free(module);
}

TEST_F(ModuleStringTest, CopiesCompleteBytesBeforeReturning) {
  ASSERT_NO_FATAL_FAILURE(CreateModule());
  char bytes[] = {'a', '\0', 'b'};
  const auto id = Intern(iree_make_string_view(bytes, sizeof(bytes)));
  const auto stored = loom_string_table_get(&module_->strings, id);
  EXPECT_NE(stored.data, bytes);
  EXPECT_EQ(stored.size, sizeof(bytes));
  bytes[0] = 'z';
  const auto expected = iree_make_string_view("a\0b", sizeof(bytes));
  EXPECT_TRUE(iree_string_view_equal(stored, expected));
  const auto used = module_->arena.used_allocation_size;
  EXPECT_EQ(Intern(expected), id);
  EXPECT_EQ(module_->arena.used_allocation_size, used);
}

TEST_F(ModuleStringTest, HintsSizeBucketsWithoutReservingStringViews) {
  loom_module_size_hints_t hints = {.string_count = 4096};
  ASSERT_NO_FATAL_FAILURE(CreateModule(&hints));
  EXPECT_EQ(module_->strings.count, 1u);
  EXPECT_EQ(module_->strings.segments.segment_count, 1u);
  EXPECT_GE(module_->string_intern.capacity, hints.string_count);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&module_->strings, module_->name_id),
      IREE_SV("strings")));
}

TEST_F(ModuleStringTest, ViewsSurviveGrowthScratchReuseAndTrim) {
  ASSERT_NO_FATAL_FAILURE(CreateModule());
  constexpr uint32_t kSegmentCount =
      LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT + 2;
  std::array<iree_string_view_t, kSegmentCount> retained = {};
  std::array<const void*, kSegmentCount> segments = {};
  retained[0] = loom_string_table_get(&module_->strings, 0);
  segments[0] =
      loom_segmented_storage_const_segment(&module_->strings.segments, 0);
  for (uint32_t id = 1; id < kSegmentCount * LOOM_STRING_SEGMENT_CAPACITY;
       ++id) {
    ASSERT_EQ(AppendNumberedString(id), id);
    if (id % LOOM_STRING_SEGMENT_CAPACITY == 0) {
      const uint32_t segment = id / LOOM_STRING_SEGMENT_CAPACITY;
      retained[segment] = loom_string_table_get(&module_->strings, id);
      segments[segment] = loom_segmented_storage_const_segment(
          &module_->strings.segments, segment);
    }
  }
  ASSERT_EQ(module_->strings.segments.segment_count, kSegmentCount);
  ASSERT_NE(module_->strings.segments.primary_page, nullptr);
  iree_arena_allocator_t scratch;
  iree_arena_initialize(&block_pool_, &scratch);
  void* temporary = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&scratch, 16384, &temporary));
  iree_arena_deinitialize(&scratch);
  iree_arena_block_pool_trim(&block_pool_);
  const auto used = module_->arena.used_allocation_size;
  for (uint32_t segment = 0; segment < kSegmentCount; ++segment) {
    const auto id = segment * LOOM_STRING_SEGMENT_CAPACITY;
    const auto view = loom_string_table_get(&module_->strings, id);
    EXPECT_EQ(view.data, retained[segment].data);
    EXPECT_EQ(view.size, retained[segment].size);
    EXPECT_EQ(loom_segmented_storage_const_segment(&module_->strings.segments,
                                                   segment),
              segments[segment]);
    EXPECT_EQ(Intern(retained[segment]), id);
  }
  EXPECT_EQ(module_->arena.used_allocation_size, used);
}

TEST_F(ModuleStringTest, WarmConstructionReusesOnlyFixedPoolBlocks) {
  for (const iree_host_size_t block_size : {32768u, 131072u}) {
    SCOPED_TRACE(block_size);
    ResetPool(block_size);
    iree_host_size_t cold_calls = 0;
    for (uint32_t iteration = 0; iteration < 3; ++iteration) {
      ASSERT_NO_FATAL_FAILURE(CreateModule());
      for (uint32_t id = 1; id < 8193; ++id) {
        ASSERT_EQ(AppendNumberedString(id), id);
      }
      EXPECT_LE(largest_allocation_, block_size);
      if (iteration == 0) {
        cold_calls = allocation_count_;
        ASSERT_GT(cold_calls, 0u);
      } else {
        EXPECT_EQ(allocation_count_, cold_calls);
      }
      loom_module_free(module_);
      module_ = nullptr;
    }
  }
}

TEST_F(ModuleStringTest, AllocationFailureKeepsPublishedStringsAndRetries) {
  // Cross row, bucket, and both primary directory growth boundaries. Tiny
  // blocks make each allocation observable without changing the module API.
  for (const uint32_t boundary : {96u, 128u, 1536u, 2048u}) {
    SCOPED_TRACE(boundary);
    for (iree_host_size_t failure_index = 0;; ++failure_index) {
      SCOPED_TRACE(failure_index);
      ResetPool(128);
      const loom_module_size_hints_t hints = {};
      ASSERT_NO_FATAL_FAILURE(CreateModule(&hints));
      for (uint32_t id = 1; id < boundary; ++id) {
        ASSERT_EQ(AppendNumberedString(id), id);
      }
      const auto name = loom_string_table_get(&module_->strings, 0);
      const auto interner_capacity = module_->string_intern.capacity;
      const std::string bytes(256, 'x');
      const auto value = iree_make_string_view(bytes.data(), bytes.size());
      allocation_count_ = 0;
      failure_index_ = failure_index;
      loom_string_id_t id = LOOM_STRING_ID_INVALID;
      iree_status_t status = loom_module_intern_string(module_, value, &id);
      failure_index_ = SIZE_MAX;
      if (iree_status_is_ok(status)) {
        EXPECT_LE(allocation_count_, failure_index);
        EXPECT_EQ(id, boundary);
        EXPECT_EQ(module_->strings.count, boundary + 1);
        break;
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(allocation_count_, failure_index + 1);
      EXPECT_EQ(id, LOOM_STRING_ID_INVALID);
      EXPECT_EQ(module_->strings.count, boundary);
      EXPECT_EQ(module_->string_intern.count, boundary);
      EXPECT_EQ(module_->string_intern.capacity, interner_capacity);
      EXPECT_EQ(loom_string_table_get(&module_->strings, 0).data, name.data);
      EXPECT_EQ(loom_module_lookup_string(module_, value),
                LOOM_STRING_ID_INVALID);
      for (uint32_t prior = 1; prior < boundary; ++prior) {
        EXPECT_EQ(AppendNumberedString(prior), prior);
      }
      // Row capacity and copied bytes may have been reserved before failure;
      // retry consumes that capacity without publishing a gap or another ID.
      EXPECT_EQ(Intern(value), boundary);
      EXPECT_EQ(Intern(value), boundary);
      EXPECT_EQ(module_->strings.count, boundary + 1);
      EXPECT_TRUE(iree_string_view_equal(
          loom_string_table_get(&module_->strings, boundary), value));
    }
  }
}

}  // namespace
}  // namespace loom
