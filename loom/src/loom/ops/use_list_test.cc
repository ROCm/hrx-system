// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/use_list.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class UseListTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCALAR,
                                                 vtables, (uint16_t)count));
    vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("uses"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_op_t* size = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(16), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &size));
    size_ = loom_test_constant_result(size);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Allocate() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_resource_alloc_build(
        &builder_, size_, ResourceType(), LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_resource_alloc_result(op);
  }

  loom_op_t* Borrow(loom_value_id_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_resource_borrow_build(&builder_, value,
                                                  LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* Move(loom_value_id_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_resource_move_build(
        &builder_, value, ResourceType(), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void ExpectUses(loom_value_id_t value_id, uint32_t count,
                  uint32_t ownership_count) {
    const loom_value_t* value = loom_module_value(module_, value_id);
    EXPECT_EQ(value->use_count, count);
    EXPECT_EQ(loom_value_ownership_use_count(value), ownership_count);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t i = 0; i < count; ++i) {
      loom_op_t* op = loom_use_user_op(uses[i]);
      const uint16_t operand = loom_use_operand_index(uses[i]);
      EXPECT_EQ(loom_op_const_operands(op)[operand], value_id);
      EXPECT_EQ(loom_op_operand_use_indices(op)[operand], i);
    }
  }

  static loom_type_t ResourceType() { return loom_type_pool(); }

  // Module arena backing store retained for the fixture lifetime.
  iree_arena_block_pool_t pool_ = {};
  // Registered production dialect metadata used by the normal builders.
  loom_context_t context_ = {};
  // Owned IR module under mutation.
  loom_module_t* module_ = nullptr;
  // Builder inserting operations into the module's entry block.
  loom_builder_t builder_ = {};
  // Constant size used by the resource-allocation builders.
  loom_value_id_t size_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(UseListTest, InlineOverflowAndRemovalPreserveOwnership) {
  static_assert(sizeof(loom_use_t) == 8);
  static_assert(sizeof(loom_value_t) == 64);
  const loom_value_id_t source = Allocate();
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_test_resource_alias_build(
      &builder_, source, ResourceType(), LOOM_LOCATION_UNKNOWN, &alias));
  ExpectUses(source, 1, 1);
  EXPECT_EQ(
      loom_use_flags(loom_value_uses(loom_module_value(module_, source))[0]),
      LOOM_USE_FLAG_IDENTITY);
  loom_op_t* reads[8];
  for (int i = 0; i < 8; ++i) {
    reads[i] = Borrow(source);
    ExpectUses(source, i + 2, 1);
  }
  loom_op_t* moved = Move(source);
  ExpectUses(source, 10, 2);
  IREE_ASSERT_OK(loom_op_erase(module_, reads[1]));
  ExpectUses(source, 9, 2);
  IREE_ASSERT_OK(loom_op_erase(module_, alias));
  ExpectUses(source, 8, 1);
  IREE_ASSERT_OK(loom_op_erase(module_, moved));
  ExpectUses(source, 7, 0);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 7, 0);
}

TEST_F(UseListTest, OperandReplacementCarriesClassification) {
  const loom_value_id_t source = Allocate();
  const loom_value_id_t replacement = Allocate();
  for (int i = 0; i < 4; ++i) {
    Borrow(source);
  }
  loom_op_t* moved = Move(source);
  ExpectUses(source, 5, 1);
  IREE_ASSERT_OK(loom_op_set_operand(module_, moved, 0, replacement));
  ExpectUses(source, 4, 0);
  ExpectUses(replacement, 1, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 4, 0);
  ExpectUses(replacement, 1, 1);
  IREE_ASSERT_OK(loom_op_erase(module_, moved));
  ExpectUses(replacement, 0, 0);
}

TEST_F(UseListTest, FullReplacementTransfersInlineOwnershipIntoOverflow) {
  const loom_value_id_t source = Allocate();
  const loom_value_id_t replacement = Allocate();
  for (int i = 0; i < 3; ++i) {
    Borrow(replacement);
  }
  Borrow(source);
  Move(source);
  ExpectUses(source, 2, 1);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_with(module_, source, replacement));
  ExpectUses(source, 0, 0);
  ExpectUses(replacement, 5, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(replacement, 5, 1);
}

TEST_F(UseListTest, FilteredReplacementLeavesConsumptionWithItsValue) {
  const loom_value_id_t source = Allocate();
  const loom_value_id_t replacement = Allocate();
  for (int i = 0; i < 4; ++i) {
    Borrow(source);
  }
  loom_op_t* moved = Move(source);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_except(module_, source, replacement, moved));
  ExpectUses(source, 1, 1);
  ExpectUses(replacement, 4, 0);
  IREE_ASSERT_OK(loom_value_replace_uses_if(
      module_, source, replacement,
      [](const loom_op_t* op, void* user_data) {
        return loom_test_resource_move_isa(op);
      },
      nullptr));
  ExpectUses(source, 0, 0);
  ExpectUses(replacement, 5, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 0, 0);
  ExpectUses(replacement, 5, 1);
}

TEST_F(UseListTest, SourceFactIdentityDoesNotConstrainStorageOwnership) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_value_id_t sources[] = {size_, size_};
  const loom_type_t types[] = {type, type};
  loom_op_t* assumed = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, sources, 2, nullptr, 0,
                                          types, 2, LOOM_LOCATION_UNKNOWN,
                                          &assumed));
  ExpectUses(size_, 2, 0);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(size_, 2, 0);
  IREE_ASSERT_OK(loom_op_erase(module_, assumed));
  ExpectUses(size_, 0, 0);
}

TEST_F(UseListTest, StorageFactIdentityClassifiesEveryOperandOccurrence) {
  const loom_type_t type = loom_low_register_type(1, 0, 1);
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), type, &source));
  const loom_value_id_t sources[] = {source, source};
  const loom_type_t types[] = {type, type};
  loom_op_t* assumed = nullptr;
  IREE_ASSERT_OK(loom_low_assume_build(&builder_, sources, 2, nullptr, 0, types,
                                       2, LOOM_LOCATION_UNKNOWN, &assumed));
  ExpectUses(source, 2, 2);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 2, 2);
  IREE_ASSERT_OK(loom_op_erase(module_, assumed));
  ExpectUses(source, 0, 0);
}

TEST_F(UseListTest, DirectConsumptionHasNoResultDependency) {
  const loom_value_id_t source = Allocate();
  loom_op_t* consumer = nullptr;
  IREE_ASSERT_OK(loom_test_resource_consume_build(
      &builder_, source, LOOM_LOCATION_UNKNOWN, &consumer));
  ExpectUses(source, 1, 1);
  EXPECT_EQ(
      loom_use_flags(loom_value_uses(loom_module_value(module_, source))[0]),
      LOOM_USE_FLAG_CONSUMES);
  IREE_ASSERT_OK(loom_op_erase(module_, consumer));
  ExpectUses(source, 0, 0);
}

TEST_F(UseListTest, EveryTerminalOwnershipEffectIsClassified) {
  using BuildConsumer = iree_status_t (*)(loom_builder_t*, loom_value_id_t,
                                          loom_location_id_t, loom_op_t**);
  const BuildConsumer builders[] = {
      loom_test_resource_consume_build, loom_test_resource_release_build,
      loom_test_resource_discard_build, loom_test_resource_escape_build};
  for (BuildConsumer build : builders) {
    const loom_value_id_t source = Allocate();
    loom_op_t* consumer = nullptr;
    IREE_ASSERT_OK(build(&builder_, source, LOOM_LOCATION_UNKNOWN, &consumer));
    ExpectUses(source, 1, 1);
    IREE_ASSERT_OK(loom_op_erase(module_, consumer));
    ExpectUses(source, 0, 0);
  }
}

TEST_F(UseListTest, PopulatingClearedOperandReclassifiesItsUse) {
  const loom_value_id_t source = Allocate();
  loom_op_t* moved = Move(source);
  IREE_ASSERT_OK(loom_op_set_operand(module_, moved, 0, LOOM_VALUE_ID_INVALID));
  ExpectUses(source, 0, 0);
  IREE_ASSERT_OK(loom_op_set_operand(module_, moved, 0, source));
  ExpectUses(source, 1, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 1, 1);
}

TEST_F(UseListTest, FullReplacementTransfersOverflowOwnership) {
  const loom_value_id_t source = Allocate();
  const loom_value_id_t replacement = Allocate();
  for (int i = 0; i < 9; ++i) {
    Borrow(source);
  }
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_test_resource_alias_build(
      &builder_, source, ResourceType(), LOOM_LOCATION_UNKNOWN, &alias));
  Move(source);
  ExpectUses(source, 11, 2);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_with(module_, source, replacement));
  ExpectUses(source, 0, 0);
  ExpectUses(replacement, 11, 2);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(replacement, 11, 2);
}

TEST_F(UseListTest, ResultCompactionPreservesExplicitTieClassification) {
  const loom_value_id_t source = Allocate();
  for (int i = 0; i < 4; ++i) {
    Borrow(source);
  }
  const loom_type_t types[] = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                               ResourceType()};
  loom_tied_result_t tie = {.result_index = 1, .operand_index = 0};
  loom_op_t* sink = nullptr;
  IREE_ASSERT_OK(loom_test_signature_sink_build(
      &builder_, &source, 1, types, 2, &tie, 1, LOOM_LOCATION_UNKNOWN, &sink));
  ExpectUses(source, 5, 1);
  const bool remove[] = {true, false};
  uint16_t removed_count = 0;
  iree_arena_allocator_t scratch;
  iree_arena_initialize(&pool_, &scratch);
  IREE_ASSERT_OK(
      loom_op_remove_results(module_, sink, remove, &scratch, &removed_count));
  iree_arena_deinitialize(&scratch);
  EXPECT_EQ(removed_count, 1);
  EXPECT_EQ(loom_op_tied_results(sink)[0].result_index, 0);
  ExpectUses(source, 5, 1);
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  ExpectUses(source, 5, 1);
  IREE_ASSERT_OK(loom_op_erase(module_, sink));
  ExpectUses(source, 4, 0);
}

}  // namespace
}  // namespace loom
