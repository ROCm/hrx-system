// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_parts.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"

namespace loom {
namespace {

class RegisterPartsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("parts"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  void Acquire(size_t count) {
    if (count > module_->values.count) {
      loom_value_id_t base;
      IREE_ASSERT_OK(loom_module_define_untyped_values(
          module_, count - module_->values.count, &base));
    }
    loom_value_u32_scratch_acquire_zeroed(&module_->scratch.values,
                                          module_->values.count);
    parts_ = {};
    parts_.masks = &module_->scratch.values;
    parts_.arena = &arena_;
  }
  void Release() {
    loom_value_u32_scratch_release_zeroed(&module_->scratch.values);
    iree_arena_reset(&arena_);
  }

  // Pool shared by the value table and invocation-owned analysis storage.
  iree_arena_block_pool_t pool_;
  // Function-local analysis allocations.
  iree_arena_allocator_t arena_;
  // Context owning the value table's type infrastructure.
  loom_context_t context_;
  // Module provides real value-indexed scratch without any authored IR.
  loom_module_t* module_ = nullptr;
  // Register facts under test.
  loom_low_register_parts_t parts_ = {};
};

TEST_F(RegisterPartsTest, FullMasksNeedNoDependencyStorage) {
  Acquire(3);
  loom_low_register_parts_define(&parts_, 0, 1);
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 1, 0, 2, 3));
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 2, 1, 1, 3));
  EXPECT_EQ(parts_.continuations.count, 0u);
  IREE_ASSERT_OK(loom_low_register_parts_resolve(&parts_));
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 0), 1u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 1), 3u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 2), 3u);
  loom_low_register_part_requirement_t requirement = {.value = 2, .mask = 3};
  IREE_ASSERT_OK(loom_low_register_parts_require(&parts_, &requirement));
  EXPECT_EQ(parts_.requirements.count, 0u);
  EXPECT_EQ(arena_.used_allocation_size, 0u);
  Release();
}

TEST_F(RegisterPartsTest, DeferredRequirementsKeepTheirOriginalFacts) {
  constexpr uint32_t count = 256;
  Acquire(count);
  for (uint32_t i = 0; i < count; ++i) {
    loom_low_register_part_requirement_t requirement = {};
    requirement.op_name = IREE_SV("packet");
    requirement.field_name = IREE_SV("source");
    requirement.field_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
    requirement.value = i;
    requirement.mask = 3;
    IREE_ASSERT_OK(loom_low_register_parts_require(&parts_, &requirement));
  }
  for (uint32_t i = 0; i < count; ++i) {
    loom_low_register_parts_define(&parts_, i, i % 2 ? 3 : 1);
  }
  IREE_ASSERT_OK(loom_low_register_parts_resolve(&parts_));
  ASSERT_EQ(parts_.requirements.count, count);
  for (uint32_t i = 0; i < count; ++i) {
    const auto& requirement = parts_.requirements.values[i];
    EXPECT_EQ(requirement.value, i);
    EXPECT_EQ(requirement.mask, 3u);
    EXPECT_EQ(requirement.field_ref.kind, LOOM_DIAGNOSTIC_FIELD_OPERAND);
    EXPECT_EQ(requirement.field_ref.index, i);
    EXPECT_EQ(requirement.field_ref.occurrence, 0u);
    EXPECT_TRUE(iree_string_view_equal(requirement.op_name, IREE_SV("packet")));
    EXPECT_TRUE(
        iree_string_view_equal(requirement.field_name, IREE_SV("source")));
    EXPECT_EQ(loom_low_register_parts_mask(&parts_, i), i % 2 ? 3u : 1u);
  }
  Release();
}

TEST_F(RegisterPartsTest, ReverseChainWithSparseValueIds) {
  constexpr uint32_t count = 32768;
  Acquire(count * 2);
  for (uint32_t i = 0; i + 1 < count; ++i) {
    IREE_ASSERT_OK(
        loom_low_register_parts_continue(&parts_, i * 2, (i + 1) * 2, 1, 3));
  }
  loom_low_register_parts_define(&parts_, (count - 1) * 2, 2);
  IREE_ASSERT_OK(loom_low_register_parts_resolve(&parts_));
  for (uint32_t i = 0; i + 1 < count; ++i) {
    EXPECT_EQ(loom_low_register_parts_mask(&parts_, i * 2), 3u);
    EXPECT_EQ(loom_low_register_parts_mask(&parts_, i * 2 + 1), 0u);
  }
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, (count - 1) * 2), 2u);
  Release();
}

TEST_F(RegisterPartsTest, CycleDoesNotInheritIncomingPathMasks) {
  Acquire(5);
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 0, 1, 4, 15));
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 1, 2, 1, 15));
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 2, 1, 2, 15));
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 3, 2, 8, 15));
  IREE_ASSERT_OK(loom_low_register_parts_continue(&parts_, 4, 4, 16, 31));
  IREE_ASSERT_OK(loom_low_register_parts_resolve(&parts_));
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 0), 7u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 1), 3u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 2), 3u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 3), 11u);
  EXPECT_EQ(loom_low_register_parts_mask(&parts_, 4), 16u);
  Release();
}

TEST_F(RegisterPartsTest, ArbitraryGraphsAgainstFixedPoint) {
  uint32_t seed = 1977;
  auto random = [&]() { return seed = seed * 1664525u + 1013904223u; };
  for (int iteration = 0; iteration < 2000; ++iteration) {
    SCOPED_TRACE(iteration);
    const size_t count = 1 + random() % 48;
    std::vector<uint32_t> initial(count);
    std::vector<int> sources(count);
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < count; ++i) {
      initial[i] = 1u << (random() % 32);
      sources[i] = static_cast<int>(random() % (count + 1)) - 1;
      std::swap(order[i], order[random() % (i + 1)]);
    }
    auto expected = initial;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i < count; ++i) {
        if (sources[i] < 0) {
          continue;
        }
        uint32_t mask = expected[i] | expected[sources[i]];
        changed |= mask != expected[i];
        expected[i] = mask;
      }
    }
    Acquire(count);
    for (uint32_t value : order) {
      if (sources[value] < 0) {
        loom_low_register_parts_define(&parts_, value, initial[value]);
      } else {
        IREE_ASSERT_OK(loom_low_register_parts_continue(
            &parts_, value, sources[value], initial[value], UINT32_MAX));
      }
    }
    IREE_ASSERT_OK(loom_low_register_parts_resolve(&parts_));
    for (size_t i = 0; i < count; ++i) {
      EXPECT_EQ(loom_low_register_parts_mask(&parts_, i), expected[i]) << i;
    }
    Release();
  }
}

}  // namespace
}  // namespace loom
