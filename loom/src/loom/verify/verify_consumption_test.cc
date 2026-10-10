// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// API-only boundaries: selected function lists and CFG definition order cannot
// be expressed by a verify-mode text fixture. Program semantics live in the
// companion Low consumption fixtures.
class ConsumptionVerificationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    type_ =
        loom_low_register_type(loom_test_low_core_descriptor_set()->stable_id,
                               TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
    ResetModule();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void ResetModule() {
    loom_module_free(module_);
    module_ = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("consumption"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  loom_op_t* Function(const char* name) {
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    uint16_t symbol;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol));
    loom_op_t* function = nullptr;
    IREE_CHECK_OK(loom_test_func_build(&builder_, 0, 0, 0, {0, symbol}, &type_,
                                       1, nullptr, 0, nullptr, 0, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function));
    loom_builder_enter_region(&builder_, function,
                              loom_test_func_body(function));
    return function;
  }

  loom_value_id_t Alias(loom_value_id_t source) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_assume_build(&builder_, &source, 1, nullptr, 0,
                                        &type_, 1, LOOM_LOCATION_UNKNOWN, &op));
    return loom_low_assume_results(op).values[0];
  }

  loom_value_id_t Move(loom_value_id_t source) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_low_move_build(&builder_, source, false, type_,
                                      LOOM_LOCATION_UNKNOWN, &op));
    return loom_low_move_result(op);
  }

  void Use(loom_value_id_t source) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(
        loom_test_use_build(&builder_, &source, 1, LOOM_LOCATION_UNKNOWN, &op));
  }

  void Yield() {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
  }

  void VerifyFunctions(std::initializer_list<loom_op_t*> functions,
                       uint16_t expected_diagnostic = 0) {
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    std::vector<loom_func_like_t> handles;
    for (loom_op_t* function : functions) {
      handles.push_back(loom_func_like_cast(module_, function));
    }
    testing::DiagnosticCapture capture;
    loom_verify_options_t options = {.sink = capture.sink()};
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_functions(module_, handles.data(),
                                         handles.size(), &options, &result));
    EXPECT_EQ(result.error_count, expected_diagnostic ? 1u : 0u);
    if (expected_diagnostic) {
      EXPECT_NE(testing::FindDiagnostic(
                    capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE,
                                                   expected_diagnostic)),
                nullptr);
    }
  }

  // Shared storage for module ownership and verifier scratch.
  iree_arena_block_pool_t pool_;
  // Minimal registered operation set used by these API fixtures.
  loom_context_t context_;
  // Owns functions, values, and their ordinary use lists.
  loom_module_t* module_ = nullptr;
  // Current operation insertion point.
  loom_builder_t builder_;
  // A real test descriptor's one-unit register carrier.
  loom_type_t type_;
};

TEST_F(ConsumptionVerificationTest, RepeatedFunctionSelection) {
  loom_op_t* function = Function("selected");
  loom_value_id_t seed = loom_block_arg_id(builder_.ip.block, 0);
  Use(Move(Alias(seed)));
  Yield();
  VerifyFunctions({function, function});
}

TEST_F(ConsumptionVerificationTest, SelectedFunctionsDoNotShareOwnership) {
  loom_op_t* first = Function("first");
  Use(Move(Alias(loom_block_arg_id(builder_.ip.block, 0))));
  Yield();
  loom_op_t* second = Function("second");
  Use(Move(Alias(loom_block_arg_id(builder_.ip.block, 0))));
  Yield();
  VerifyFunctions({second, first, second});
}

TEST_F(ConsumptionVerificationTest, SkipsUnselectedFunction) {
  loom_op_t* valid = Function("valid");
  Use(Move(Alias(loom_block_arg_id(builder_.ip.block, 0))));
  Yield();
  loom_op_t* invalid = Function("invalid");
  loom_value_id_t seed = loom_block_arg_id(builder_.ip.block, 0);
  Move(Alias(seed));
  Use(seed);
  Yield();
  VerifyFunctions({valid});
  VerifyFunctions({invalid}, 2);
  VerifyFunctions({invalid, invalid}, 2);
  VerifyFunctions({});
}

TEST_F(ConsumptionVerificationTest, PublishesAliasBeforeDominatedEarlierBlock) {
  loom_op_t* function = Function("reordered");
  loom_region_t* region = loom_test_func_body(function);
  loom_block_t* entry = builder_.ip.block;
  loom_value_id_t seed = loom_block_arg_id(entry, 0);
  loom_block_t* use_block = nullptr;
  loom_block_t* alias_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &use_block));
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &alias_block));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&builder_, alias_block,
                                    LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, alias_block);
  loom_value_id_t alias = Alias(seed);
  IREE_ASSERT_OK(
      loom_test_br_build(&builder_, use_block, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, use_block);
  Move(alias);
  Use(seed);
  Yield();
  VerifyFunctions({function}, 2);
}

TEST_F(ConsumptionVerificationTest, MatchesIndependentPathOracle) {
  // Three binary branch nodes and one exit have 4^6 possible topologies.
  // Nodes zero and one consume the same owner, node two observes it. This
  // exercises reachable/dead consumers, exclusive paths, cycles and joins
  // through the real verifier rather than manufactured analysis tables.
  for (uint32_t topology = 0; topology < 4096; ++topology) {
    SCOPED_TRACE(topology);
    ResetModule();
    loom_op_t* function = Function("graph");
    loom_region_t* region = loom_test_func_body(function);
    loom_value_id_t seed = loom_block_arg_id(builder_.ip.block, 0);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_low_copy_build(&builder_, seed, false, type_,
                                       LOOM_LOCATION_UNKNOWN, &op));
    const loom_value_id_t condition = loom_low_copy_result(op);
    const loom_value_id_t alias = Alias(seed);
    loom_block_t* blocks[4];
    for (loom_block_t*& block : blocks) {
      IREE_ASSERT_OK(loom_region_append_block(module_, region, &block));
    }
    IREE_ASSERT_OK(
        loom_test_br_build(&builder_, blocks[0], LOOM_LOCATION_UNKNOWN, &op));
    bool reaches[4][4] = {};
    for (uint16_t b = 0; b < 4; ++b) {
      loom_builder_set_block(&builder_, blocks[b]);
      if (b < 2) {
        Move(alias);
      } else if (b == 2) {
        Use(seed);
      }
      if (b == 3) {
        Yield();
      } else {
        const uint16_t first = (topology >> (b * 4)) & 3u;
        const uint16_t second = (topology >> (b * 4 + 2)) & 3u;
        reaches[b][first] = reaches[b][second] = true;
        IREE_ASSERT_OK(loom_low_cond_br_build(&builder_, condition,
                                              blocks[first], blocks[second],
                                              LOOM_LOCATION_UNKNOWN, &op));
      }
    }
    for (uint16_t k = 0; k < 4; ++k) {
      for (uint16_t i = 0; i < 4; ++i) {
        for (uint16_t j = 0; j < 4; ++j) {
          reaches[i][j] |= reaches[i][k] && reaches[k][j];
        }
      }
    }
    const bool invalid = reaches[0][0] || reaches[0][1] || reaches[0][2];
    VerifyFunctions({function}, invalid ? 2 : 0);
  }
}

}  // namespace
}  // namespace loom
