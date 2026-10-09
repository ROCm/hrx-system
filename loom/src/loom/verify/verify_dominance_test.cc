// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// Forward references cannot be authored in text. Exercise the verifier's API
// boundary with two unreachable blocks whose values retain ordinary use lists.
class UnreachableDominanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("dominance"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("dead"), &name));
    uint16_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder_, 0, 0, 0, {0, symbol}, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    region_ = loom_test_func_body(function);
    seed_ = Constant(loom_region_entry_block(region_), LOOM_SCALAR_TYPE_I32);
    AppendYield();
    IREE_ASSERT_OK(loom_region_append_block(module_, region_, &first_));
    IREE_ASSERT_OK(loom_region_append_block(module_, region_, &second_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Constant(loom_block_t* block,
                           loom_scalar_type_t type = LOOM_SCALAR_TYPE_INDEX) {
    loom_builder_set_block(&builder_, block);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(&builder_, loom_attr_i64(4),
                                           loom_type_scalar(type),
                                           LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  }

  void AppendYield() {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
  }

  void Verify(uint16_t expected_diagnostic) {
    for (loom_block_t* block : {first_, second_}) {
      loom_builder_set_block(&builder_, block);
      AppendYield();
    }
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    testing::DiagnosticCapture capture;
    loom_verify_options_t options = {.sink = capture.sink()};
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 1u);
    EXPECT_NE(testing::FindDiagnostic(
                  capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE,
                                                 expected_diagnostic)),
              nullptr);
  }

  // Storage shared by the module and verification scratch.
  iree_arena_block_pool_t pool_;
  // Registered operations used by the API fixtures.
  loom_context_t context_;
  // Owns all values, blocks, and use records.
  loom_module_t* module_ = nullptr;
  // Insertion point for each small fixture.
  loom_builder_t builder_;
  // Function body with one live entry and two dead blocks.
  loom_region_t* region_ = nullptr;
  // Earlier dead block in serialization order.
  loom_block_t* first_ = nullptr;
  // Later dead block in serialization order.
  loom_block_t* second_ = nullptr;
  // Entry value available to both unreachable blocks.
  loom_value_id_t seed_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(UnreachableDominanceTest, RejectsForwardOperand) {
  loom_value_id_t later = Constant(second_);
  loom_builder_set_block(&builder_, first_);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &later, 1, LOOM_LOCATION_UNKNOWN, &use));
  Verify(1);
}

TEST_F(UnreachableDominanceTest, RejectsCyclicOperands) {
  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_builder_set_block(&builder_, first_);
  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, seed_, seed_, type,
                                      LOOM_LOCATION_UNKNOWN, &first));
  loom_builder_set_block(&builder_, second_);
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, loom_test_addi_result(first),
                                      seed_, type, LOOM_LOCATION_UNKNOWN,
                                      &second));
  loom_op_operands(first)[0] = loom_test_addi_result(second);
  Verify(1);
}

TEST_F(UnreachableDominanceTest, RejectsSignatureOnlyResultOperand) {
  loom_builder_set_block(&builder_, first_);
  const loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* signature = nullptr;
  IREE_ASSERT_OK(loom_test_signature_sink_build(
      &builder_, nullptr, 0, &result_type, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN,
      &signature));
  const loom_value_id_t result =
      loom_test_signature_sink_results(signature).values[0];
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));
  Verify(1);
}

TEST_F(UnreachableDominanceTest, RejectsForwardResultTypeReference) {
  loom_value_id_t later = Constant(second_);
  loom_builder_set_block(&builder_, first_);
  loom_op_t* op = nullptr;
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_dynamic(later), 0);
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, seed_, type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  Verify(16);
}

TEST_F(UnreachableDominanceTest, RejectsForwardBlockArgumentTypeReference) {
  loom_value_id_t later = Constant(second_);
  loom_value_id_t argument;
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_dynamic(later), 0);
  IREE_ASSERT_OK(loom_module_define_value(module_, type, &argument));
  IREE_ASSERT_OK(loom_block_add_arg(module_, first_, argument));
  Verify(16);
}

}  // namespace
}  // namespace loom
