// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class OwnershipTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseAndVerify(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("ownership_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
    return ModulePtr(module);
  }

  static loom_op_t* FindOpInRegion(loom_region_t* region, loom_op_kind_t kind,
                                   iree_host_size_t* ordinal) {
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (op->kind == kind) {
          if (*ordinal == 0) return op;
          --*ordinal;
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          loom_op_t* nested = FindOpInRegion(regions[i], kind, ordinal);
          if (nested != nullptr) return nested;
        }
      }
    }
    return nullptr;
  }

  static loom_op_t* FindOp(loom_module_t* module, loom_op_kind_t kind,
                           iree_host_size_t ordinal = 0) {
    return FindOpInRegion(module->body, kind, &ordinal);
  }

  static bool PoolTypeMatches(loom_type_t type, void* user_data) {
    (void)user_data;
    return loom_type_is_pool(type);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(OwnershipTest, DescriptorSizesStayInExistingClass) {
  static_assert(sizeof(loom_op_t) == 64);
  static_assert(sizeof(loom_value_t) == 64);
  static_assert(sizeof(loom_op_vtable_t) == 192);
  static_assert(sizeof(loom_operand_descriptor_t) == 16);
  static_assert(sizeof(loom_result_descriptor_t) == 16);
}

TEST_F(OwnershipTest, TestResourceOpsExposeOwnershipEffects) {
  ModulePtr module = ParseAndVerify(R"(
func.def @ownership(%size: index) {
  %fresh = test.resource.alloc %size : index -> pool<16>
  test.resource.borrow %fresh : pool<16>
  test.resource.borrow_ref %fresh : pool<16>
  %retained = test.resource.retain %fresh : pool<16> -> pool<16>
  %alias = test.resource.alias %retained : pool<16> -> pool<16>
  %borrowed = test.resource.borrowed %alias : pool<16> -> pool<16>
  test.resource.release %retained : pool<16>
  test.resource.discard %alias : pool<16>
  test.resource.escape %borrowed : pool<16>
  test.resource.consume %fresh : pool<16>
  func.return
}
)");

  loom_ownership_resource_family_t pool_family = {
      /*.name=*/IREE_SV("pool"),
      /*.type_matches=*/PoolTypeMatches,
  };

  loom_ownership_result_effect_t result_effect = {};
  loom_op_t* alloc = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_ALLOC);
  ASSERT_NE(alloc, nullptr);
  EXPECT_TRUE(
      loom_ownership_result_effect_at(module.get(), alloc, 0, &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_FRESH);
  EXPECT_TRUE(loom_ownership_value_matches(module.get(), &pool_family,
                                           result_effect.value_id));

  loom_ownership_operand_effect_t operand_effect = {};
  loom_op_t* borrow = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_BORROW);
  ASSERT_NE(borrow, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), borrow, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_BORROW);
  EXPECT_EQ(operand_effect.carrier, LOOM_OWNERSHIP_CARRIER_BY_VALUE);

  loom_op_t* borrow_ref =
      FindOp(module.get(), LOOM_OP_TEST_RESOURCE_BORROW_REF);
  ASSERT_NE(borrow_ref, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), borrow_ref, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_BORROW);
  EXPECT_EQ(operand_effect.carrier, LOOM_OWNERSHIP_CARRIER_BY_REFERENCE);

  loom_op_t* retain = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_RETAIN);
  ASSERT_NE(retain, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), retain, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_RETAIN);
  EXPECT_TRUE(
      loom_ownership_result_effect_at(module.get(), retain, 0, &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_RETAINED);

  loom_op_t* alias = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_ALIAS);
  ASSERT_NE(alias, nullptr);
  EXPECT_TRUE(
      loom_ownership_result_effect_at(module.get(), alias, 0, &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_ALIAS);
  EXPECT_EQ(result_effect.source_operand_index, 0);

  loom_op_t* borrowed = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_BORROWED);
  ASSERT_NE(borrowed, nullptr);
  EXPECT_TRUE(loom_ownership_result_effect_at(module.get(), borrowed, 0,
                                              &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_BORROWED);

  loom_op_t* release = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_RELEASE);
  ASSERT_NE(release, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), release, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_RELEASE);

  loom_op_t* discard = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_DISCARD);
  ASSERT_NE(discard, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), discard, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_DISCARD);

  loom_op_t* escape = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_ESCAPE);
  ASSERT_NE(escape, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), escape, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_ESCAPE);

  loom_op_t* consume = FindOp(module.get(), LOOM_OP_TEST_RESOURCE_CONSUME);
  ASSERT_NE(consume, nullptr);
  EXPECT_TRUE(loom_ownership_operand_effect_at(module.get(), consume, 0,
                                               &operand_effect));
  EXPECT_EQ(operand_effect.effect, LOOM_OPERAND_OWNERSHIP_CONSUME);
}

TEST_F(OwnershipTest, AllocatingResultsDeriveFreshOwnership) {
  ModulePtr module = ParseAndVerify(R"(
func.def @alloc(%size: index) {
  %resource = test.alloc %size : index -> pool<16>
  func.return
}
)");

  loom_op_t* alloc = FindOp(module.get(), LOOM_OP_TEST_ALLOC);
  ASSERT_NE(alloc, nullptr);
  loom_ownership_result_effect_t result_effect = {};
  EXPECT_TRUE(
      loom_ownership_result_effect_at(module.get(), alloc, 0, &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_FRESH);
}

TEST_F(OwnershipTest, TiedResultsDeriveOwnership) {
  ModulePtr module = ParseAndVerify(R"(
func.def @tied(%tile: tile<4xf32>, %tensor: tensor<4xf32>, %off: index) -> (tensor<4xf32>) {
  %result = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)
  func.return %result : tensor<4xf32>
}
)");

  loom_op_t* update = FindOp(module.get(), LOOM_OP_TEST_UPDATE);
  ASSERT_NE(update, nullptr);
  loom_ownership_result_effect_t result_effect = {};
  EXPECT_TRUE(
      loom_ownership_result_effect_at(module.get(), update, 0, &result_effect));
  EXPECT_EQ(result_effect.effect, LOOM_RESULT_OWNERSHIP_TIED);
  EXPECT_EQ(result_effect.source_operand_index, 1);
}

}  // namespace
}  // namespace loom
