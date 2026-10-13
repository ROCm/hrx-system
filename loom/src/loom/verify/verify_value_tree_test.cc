// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// API callers may construct shapes that the text parser cannot produce. The
// module verifier owns their admission, independently of the printing format.
class VerifyValueTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("trees"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("test"), &name));
    uint16_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, {0, symbol}, &type,
                                        1, nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &function));
    auto* body = loom_test_func_body(function);
    (void)loom_builder_enter_region(&builder_, function, body);
    value_ = loom_region_entry_arg_id(body, 0);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* Build(const std::vector<int64_t>& shape, uint16_t value_count) {
    std::vector<loom_value_id_t> values(value_count, value_);
    const int64_t empty = 3;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_operand_tree_build(
        &builder_, shape.data(), shape.size(), values.data(), values.size(),
        &empty, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    loom_op_t* terminator = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &terminator));
    return op;
  }

  void Verify(bool expected_valid) {
    testing::DiagnosticCapture capture;
    loom_verify_options_t options = {.sink = capture.sink()};
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, expected_valid ? 0u : 1u);
    if (!expected_valid) {
      ASSERT_EQ(capture.diagnostics.size(), 1u);
      EXPECT_EQ(capture.diagnostics[0].error,
                loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
    }
  }

  // Shared persistent and verifier scratch blocks.
  iree_arena_block_pool_t pool_ = {};
  // Canonical test dialect metadata.
  loom_context_t context_ = {};
  // Module owning the function and shape attributes.
  loom_module_t* module_ = nullptr;
  // Builder inserting inside the function body.
  loom_builder_t builder_ = {};
  // Dominating function argument, reusable as multiple tree leaves.
  loom_value_id_t value_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(VerifyValueTreeTest, RepeatedValuesAndDistinctEmptyGroups) {
  Build({19, 10, 0, 0, 1, 2, 3}, 2);
  Verify(true);
}

TEST_F(VerifyValueTreeTest, RejectsWrongLeafCoverage) {
  Build({11, 0, 0}, 1);
  Verify(false);
}

TEST_F(VerifyValueTreeTest, RejectsMissingAndTrailingNodes) {
  auto* op = Build({11, 0}, 1);
  Verify(false);
  int64_t trailing[] = {0, 1};
  loom_op_attrs(op)[0] =
      loom_attr_i64_array(trailing, IREE_ARRAYSIZE(trailing));
  Verify(false);
}

TEST_F(VerifyValueTreeTest, RejectsChildrenOnLeafOrAbsence) {
  auto* op = Build({4, 0}, 1);
  Verify(false);
  int64_t absent[] = {5, 0};
  loom_op_attrs(op)[0] = loom_attr_i64_array(absent, IREE_ARRAYSIZE(absent));
  Verify(false);
}

}  // namespace
}  // namespace loom
