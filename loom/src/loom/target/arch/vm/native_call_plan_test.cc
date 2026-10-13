// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/native_call_plan.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace {

class VMNativeCallPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("calls"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    const char* names[] = {"arguments", "record", "results", "version"};
    for (size_t i = 0; i < attrs_.size(); ++i) {
      IREE_ASSERT_OK(loom_module_intern_string(
          module_, iree_make_cstring_view(names[i]), &attrs_[i].name_id));
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  iree_status_t Build(uint16_t arguments = 1, uint16_t results = 1) {
    attrs_[0].value = loom_attr_i64_array(arguments_.data(), arguments_.size());
    attrs_[1].value = loom_attr_i64_array(words_.data(), words_.size());
    attrs_[2].value = loom_attr_i64_array(results_.data(), results_.size());
    attrs_[3].value = loom_attr_i64(version_);
    return loom_vm_native_call_template_build(
        module_, {attrs_.data(), attrs_.size()}, arguments, results, &arena_,
        &call_template_);
  }

  // Public ABI dictionary storage passed to the constructor.
  std::array<loom_named_attr_t, 4> attrs_ = {};
  // Argument placeholder offsets.
  std::vector<int64_t> arguments_ = {2};
  // Root range followed by an input/output record.
  std::vector<int64_t> words_ = {1, 2, 0, 0};
  // Result placeholder offsets.
  std::vector<int64_t> results_ = {3};
  // Required provider version.
  int64_t version_ = 7;
  // Owned compiler environment.
  loom_context_t context_ = {};
  // Shared module and plan block pool.
  iree_arena_block_pool_t pool_ = {};
  // Plan storage independent of module allocations.
  iree_arena_allocator_t arena_ = {};
  // Module supplying dictionary key spellings.
  loom_module_t* module_ = nullptr;
  // Last successfully constructed template.
  loom_vm_native_call_template_t* call_template_ = nullptr;
};

TEST_F(VMNativeCallPlanTest, RetainsIndexedLayoutAndPartitionsCallsites) {
  IREE_ASSERT_OK(Build());
  call_template_->module_name = 3;
  call_template_->export_name = 4;
  loom_vm_native_call_plan_t plan = {
      .arena = &arena_,
  };
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t ordinal = UINT32_MAX;
    uint32_t* words = nullptr;
    IREE_ASSERT_OK(loom_vm_native_call_plan_append(&plan, call_template_, i,
                                                   &ordinal, &words));
    EXPECT_EQ(ordinal, i);
    words[call_template_->operand_offsets[0]] = 10 + i;
    words[call_template_->operand_offsets[1]] = 20 + i;
    EXPECT_EQ(plan.tail->row.function_ordinal_u16, i);
    EXPECT_EQ(plan.tail->row.module_name_u16, 3);
    EXPECT_EQ(plan.tail->row.export_name_u16, 4);
    EXPECT_EQ(plan.tail->row.word_offset_u32, i * 4);
    EXPECT_EQ(plan.tail->row.version_u32, 7);
  }
  EXPECT_EQ(plan.word_count, 12);
  EXPECT_EQ(plan.head->words[2], 10);
  EXPECT_EQ(plan.head->next->words[2], 11);
  EXPECT_EQ(call_template_->words[2], 0);
}

TEST_F(VMNativeCallPlanTest, AcceptsEmptyRecord) {
  arguments_.clear();
  results_.clear();
  words_.clear();
  IREE_ASSERT_OK(Build(0, 0));
  EXPECT_EQ(call_template_->word_count, 0);
}

TEST_F(VMNativeCallPlanTest, RejectsMissingAndDuplicateOperands) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build(2, 1));
  results_[0] = arguments_[0];
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
  EXPECT_EQ(call_template_, nullptr);
}

TEST_F(VMNativeCallPlanTest, RejectsOverwritingLiteralAndOutOfRangeOffsets) {
  for (int64_t offset : {-1, 0, 1, 4}) {
    arguments_[0] = offset;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
  }
}

TEST_F(VMNativeCallPlanTest, RejectsNonU32WordsAndVersion) {
  words_[0] = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Build());
  words_[0] = INT64_C(0x100000000);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Build());
  words_[0] = 1;
  version_ = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
  version_ = INT64_C(0x100000000);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
}

TEST_F(VMNativeCallPlanTest, RejectsUnknownLayoutField) {
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("unknown"),
                                           &attrs_[0].name_id));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
}

}  // namespace
