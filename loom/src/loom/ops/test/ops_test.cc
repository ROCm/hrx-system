// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/test/ops.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

static const loom_op_vtable_t* TestVtable(loom_op_kind_t kind) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  uint8_t index = loom_op_dialect_index(kind);
  if (loom_op_dialect_id(kind) != LOOM_DIALECT_TEST || index >= count) {
    return nullptr;
  }
  return vtables[index];
}

//===----------------------------------------------------------------------===//
// Struct sizes
//===----------------------------------------------------------------------===//

TEST(OpDefs, AttributeSize) { static_assert(sizeof(loom_attribute_t) == 16); }

TEST(OpDefs, OperationHeaderSize) { static_assert(sizeof(loom_op_t) == 64); }

TEST(OpDefs, FormatElementSize) {
  static_assert(sizeof(loom_format_element_t) == 4);
}

//===----------------------------------------------------------------------===//
// Op kind encoding
//===----------------------------------------------------------------------===//

TEST(OpKind, TestAddiValue) {
  EXPECT_EQ(LOOM_OP_TEST_ADDI, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0));
  EXPECT_EQ(LOOM_OP_TEST_ADDI, 0x0100);
}

TEST(OpKind, AllDistinct) {
  int kinds[] = {
      LOOM_OP_TEST_ADDI,         LOOM_OP_TEST_NEG,     LOOM_OP_TEST_CAST,
      LOOM_OP_TEST_CONSTANT,     LOOM_OP_TEST_CMP,     LOOM_OP_TEST_MAP,
      LOOM_OP_TEST_UPDATE,       LOOM_OP_TEST_INVOKE,  LOOM_OP_TEST_SLICE,
      LOOM_OP_TEST_LOOP,         LOOM_OP_TEST_BRANCH,  LOOM_OP_TEST_YIELD,
      LOOM_OP_TEST_FUNC,         LOOM_OP_TEST_DECL,    LOOM_OP_TEST_ATTRS,
      LOOM_OP_TEST_OPERAND_DICT, LOOM_OP_TEST_DEFLATE,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kinds); ++i) {
    for (iree_host_size_t j = i + 1; j < IREE_ARRAYSIZE(kinds); ++j) {
      EXPECT_NE(kinds[i], kinds[j])
          << "kinds[" << i << "] == kinds[" << j << "]";
    }
  }
}

TEST(OpKind, DialectExtraction) {
  EXPECT_EQ(loom_op_dialect_id(LOOM_OP_TEST_ADDI), LOOM_DIALECT_TEST);
  EXPECT_EQ(loom_op_dialect_index(LOOM_OP_TEST_ADDI), 0);
}

TEST(OpKind, Count) { EXPECT_GT(LOOM_OP_TEST_COUNT_, 0); }

//===----------------------------------------------------------------------===//
// Vtable structure
//===----------------------------------------------------------------------===//

TEST(Vtable, AddiName) {
  iree_string_view_t name = loom_op_vtable_name(TestVtable(LOOM_OP_TEST_ADDI));
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("test.addi")));
}

TEST(Vtable, AddiShortName) {
  iree_string_view_t name =
      loom_op_vtable_short_name(TestVtable(LOOM_OP_TEST_ADDI));
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("addi")));
}

TEST(Vtable, AddiNamespace) {
  iree_string_view_t ns =
      loom_op_vtable_namespace(TestVtable(LOOM_OP_TEST_ADDI));
  EXPECT_TRUE(iree_string_view_equal(ns, IREE_SV("test")));
}

TEST(Vtable, AddiCounts) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_ADDI);
  EXPECT_EQ(vtable->fixed_operand_count, 2);
  EXPECT_EQ(vtable->fixed_result_count, 1);
  EXPECT_EQ(vtable->attribute_count, 0);
  EXPECT_EQ(vtable->region_count, 0);
  EXPECT_FALSE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS);
  EXPECT_FALSE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS);
}

TEST(Vtable, TypePropagationCandidateFlags) {
  EXPECT_FALSE(TestVtable(LOOM_OP_TEST_ADDI)->vtable_flags &
               LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE);
  EXPECT_TRUE(TestVtable(LOOM_OP_TEST_ATTRS)->vtable_flags &
              LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE);
}

TEST(Vtable, AddiTraits) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_ADDI);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_PURE, 0u);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_COMMUTATIVE, 0u);
  EXPECT_EQ(vtable->traits & LOOM_TRAIT_TERMINATOR, 0u);
}

TEST(Vtable, InvokeCounts) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_INVOKE);
  EXPECT_EQ(vtable->fixed_operand_count, 0);
  EXPECT_EQ(vtable->fixed_result_count, 0);
  EXPECT_EQ(vtable->attribute_count, 1);
  EXPECT_EQ(vtable->region_count, 0);
  EXPECT_TRUE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS);
  EXPECT_TRUE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS);
}

TEST(Vtable, LoopCounts) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_LOOP);
  EXPECT_EQ(vtable->fixed_operand_count, 3);
  EXPECT_EQ(vtable->region_count, 1);
  EXPECT_TRUE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS);
  EXPECT_TRUE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS);
}

TEST(Vtable, SegmentedOperandCounts) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_SEGMENTED);
  EXPECT_EQ(vtable->fixed_operand_count, 0);
  EXPECT_EQ(vtable->operand_descriptor_count, 4);
  EXPECT_FALSE(vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS);
  EXPECT_TRUE(vtable->vtable_flags & LOOM_OP_VTABLE_SEGMENTED_OPERANDS);
  ASSERT_NE(vtable->operand_descriptors, nullptr);
  EXPECT_EQ(vtable->operand_descriptors[0].flags, 0u);
  EXPECT_TRUE(vtable->operand_descriptors[1].flags & LOOM_OPERAND_OPTIONAL);
  EXPECT_TRUE(vtable->operand_descriptors[2].flags & LOOM_OPERAND_VARIADIC);
  EXPECT_TRUE(vtable->operand_descriptors[3].flags & LOOM_OPERAND_VARIADIC);
}

TEST(Vtable, BranchRegions) {
  EXPECT_EQ(TestVtable(LOOM_OP_TEST_BRANCH)->region_count, 2);
}

TEST(Vtable, YieldTraits) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_YIELD);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_TERMINATOR, 0u);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_POISON_BOUNDARY, 0u);
}

TEST(Vtable, MapTraits) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_MAP);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_PURE, 0u);
  EXPECT_NE(vtable->traits & LOOM_TRAIT_ELEMENTWISE, 0u);
}

TEST(Vtable, FormatElementCount) {
  // test.addi: Ref, COMMA, Ref, COLON, TypeOf = 5 elements.
  EXPECT_EQ(TestVtable(LOOM_OP_TEST_ADDI)->format_element_count, 5);
  // test.yield: OptionalGroup(Refs, COLON, TypesOf) = 4 elements.
  EXPECT_EQ(TestVtable(LOOM_OP_TEST_YIELD)->format_element_count, 4);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

TEST(Registration, VtableCount) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  EXPECT_GT(count, 0u);
  EXPECT_NE(vtables, nullptr);
  EXPECT_TRUE(iree_string_view_equal(loom_op_vtable_name(vtables[0]),
                                     IREE_SV("test.addi")));
}

//===----------------------------------------------------------------------===//
// Op allocation and accessors
//===----------------------------------------------------------------------===//

class BuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    // Create a minimal context for module creation.
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    // Create a module with default capacities. The module has a body
    // region with a block that the builder inserts into.
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(BuilderTest, AllocBinaryOp) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI,
                                          /*operand_count=*/2,
                                          /*result_count=*/1,
                                          /*region_count=*/0,
                                          /*tied_result_count=*/0,
                                          /*attribute_count=*/0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_ADDI);
  EXPECT_EQ(op->operand_count, 2);
  EXPECT_EQ(op->result_count, 1);
  EXPECT_EQ(op->region_count, 0);
  EXPECT_EQ(op->tied_result_count, 0);
  EXPECT_EQ(op->attribute_count, 0);
}

TEST_F(BuilderTest, AllocOpWithSuccessorsKeepsTrailingFieldsDistinct) {
  loom_block_t* then_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module_, module_->body, &then_block));
  loom_block_t* else_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module_, module_->body, &else_block));

  loom_region_t* child_region = NULL;
  IREE_ASSERT_OK(loom_module_allocate_region(module_, 1, &child_region));

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op_with_successors(
      &builder_, LOOM_OP_TEST_MAP,
      /*operand_count=*/1,
      /*result_count=*/1,
      /*successor_count=*/2,
      /*region_count=*/1,
      /*tied_result_count=*/0,
      /*attribute_count=*/1, LOOM_LOCATION_UNKNOWN, &op));

  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->successor_count, 2u);
  EXPECT_EQ(op->region_count, 1u);

  loom_op_successors(op)[0] = then_block;
  loom_op_successors(op)[1] = else_block;
  loom_op_regions(op)[0] = child_region;
  loom_op_operands(op)[0] = 42;
  loom_op_results(op)[0] = 7;
  loom_op_attrs(op)[0] = loom_attr_i64(99);

  EXPECT_EQ(loom_op_const_successors(op)[0], then_block);
  EXPECT_EQ(loom_op_const_successors(op)[1], else_block);
  EXPECT_EQ(loom_op_regions(op)[0], child_region);
  EXPECT_EQ(loom_op_operands(op)[0], 42u);
  EXPECT_EQ(loom_op_results(op)[0], 7u);
  EXPECT_EQ(loom_attr_as_i64(loom_op_attrs(op)[0]), 99);
}

TEST_F(BuilderTest, OperandAccessors) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = 42;
  loom_op_operands(op)[1] = 99;
  EXPECT_EQ(loom_test_addi_lhs(op), 42u);
  EXPECT_EQ(loom_test_addi_rhs(op), 99u);
}

TEST_F(BuilderTest, ResultAccessors) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_results(op)[0] = 7;
  EXPECT_EQ(loom_test_addi_result(op), 7u);
}

TEST_F(BuilderTest, IsaCheck) {
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &addi));
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_NEG, 1, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &neg));
  EXPECT_TRUE(loom_test_addi_isa(addi));
  EXPECT_FALSE(loom_test_addi_isa(neg));
  EXPECT_TRUE(loom_test_neg_isa(neg));
  EXPECT_FALSE(loom_test_neg_isa(addi));
}

//===----------------------------------------------------------------------===//
// Effect summary maintenance
//===----------------------------------------------------------------------===//

TEST_F(BuilderTest, DirectWriteEffectSummaryUpdatesOnErase) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), pool_type, &pool_id));

  loom_op_t* data_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &data_op));
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, loom_test_constant_result(data_op),
      LOOM_LOCATION_UNKNOWN, &write_op));

  EXPECT_EQ(module_->body->read_effect_count, 0u);
  EXPECT_EQ(module_->body->write_effect_count, 1u);

  IREE_ASSERT_OK(loom_op_erase(module_, write_op));
  EXPECT_EQ(module_->body->read_effect_count, 0u);
  EXPECT_EQ(module_->body->write_effect_count, 0u);
}

TEST_F(BuilderTest, UnknownOpEffectsAreConservative) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(
      &builder_, LOOM_OP_KIND(LOOM_DIALECT_RESERVED, 0), /*operand_count=*/0,
      /*result_count=*/0, /*region_count=*/0, /*tied_result_count=*/0,
      /*attribute_count=*/0, LOOM_LOCATION_UNKNOWN, &op));
  IREE_ASSERT_OK(loom_builder_finalize_op(&builder_, op));

  EXPECT_EQ(loom_op_effective_traits(module_, op), LOOM_TRAIT_UNKNOWN_EFFECTS);
  EXPECT_TRUE(loom_op_may_write(module_, op));
  EXPECT_EQ(module_->body->read_effect_count, 1u);
  EXPECT_EQ(module_->body->write_effect_count, 1u);
}

TEST_F(BuilderTest, NestedWriteEffectSummaryPropagatesToAncestors) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), pool_type, &pool_id));
  loom_value_id_t input_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), i32, &input_id));

  loom_op_t* data_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &data_op));
  loom_value_id_t data_id = loom_test_constant_result(data_op);

  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input_id, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&builder_, map_op, map_body);
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, data_id, LOOM_LOCATION_UNKNOWN, &write_op));
  loom_builder_restore(&builder_, saved_ip);

  EXPECT_EQ(map_body->write_effect_count, 1u);
  EXPECT_EQ(module_->body->write_effect_count, 1u);

  IREE_ASSERT_OK(loom_op_erase(module_, map_op));
  EXPECT_EQ(map_body->write_effect_count, 0u);
  EXPECT_EQ(module_->body->write_effect_count, 0u);
  EXPECT_EQ(write_op->flags & LOOM_OP_FLAG_EFFECTS_COUNTED, 0u);
}

TEST_F(BuilderTest, VariadicOperands) {
  // test.yield has variadic operands.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_YIELD, 3, 0,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = 10;
  loom_op_operands(op)[1] = 20;
  loom_op_operands(op)[2] = 30;

  loom_value_slice_t values = loom_test_yield_values(op);
  EXPECT_EQ(values.count, 3);
  EXPECT_EQ(values.values[0], 10u);
  EXPECT_EQ(values.values[1], 20u);
  EXPECT_EQ(values.values[2], 30u);
}

TEST_F(BuilderTest, SegmentedOperandAccessorsPresentOptional) {
  loom_value_id_t root = LOOM_VALUE_ID_INVALID;
  loom_value_id_t guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lhs0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lhs1 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &root));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &guard));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &lhs0));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &lhs1));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &rhs));

  loom_value_id_t lhs[] = {lhs0, lhs1};
  loom_value_id_t rhs_values[] = {rhs};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, LOOM_TEST_SEGMENTED_BUILD_FLAG_HAS_GUARD, root, guard, lhs,
      IREE_ARRAYSIZE(lhs), rhs_values, IREE_ARRAYSIZE(rhs_values), i32,
      LOOM_LOCATION_UNKNOWN, &op));

  EXPECT_EQ(op->operand_count, 5u);
  const uint16_t* counts = loom_op_const_operand_segment_counts(op);
  EXPECT_EQ(counts[0], 1u);
  EXPECT_EQ(counts[1], 1u);
  EXPECT_EQ(counts[2], 2u);
  EXPECT_EQ(counts[3], 1u);
  EXPECT_EQ(loom_test_segmented_root(op), root);
  EXPECT_TRUE(loom_test_segmented_guard_is_present(op));
  EXPECT_EQ(loom_test_segmented_guard(op), guard);
  loom_value_slice_t lhs_slice = loom_test_segmented_lhs(op);
  ASSERT_EQ(lhs_slice.count, 2u);
  EXPECT_EQ(lhs_slice.values[0], lhs0);
  EXPECT_EQ(lhs_slice.values[1], lhs1);
  loom_value_slice_t rhs_slice = loom_test_segmented_rhs(op);
  ASSERT_EQ(rhs_slice.count, 1u);
  EXPECT_EQ(rhs_slice.values[0], rhs);

  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_SEGMENTED);
  loom_value_slice_t field_span = loom_op_operand_field_span(vtable, op, 2);
  ASSERT_EQ(field_span.count, 2u);
  EXPECT_EQ(field_span.values[0], lhs0);
  EXPECT_EQ(field_span.values[1], lhs1);

  const loom_operand_descriptor_t* descriptor = NULL;
  uint8_t field_index = 0;
  uint16_t element_index = 0;
  EXPECT_TRUE(loom_op_operand_descriptor_at(vtable, op, /*operand_index=*/3,
                                            &descriptor, &field_index,
                                            &element_index));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(field_index, 2u);
  EXPECT_EQ(element_index, 1u);
}

TEST_F(BuilderTest, SegmentedOperandAccessorsAbsentOptionalAndEmptySpans) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t root = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs1 = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &root));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &rhs0));
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &rhs1));

  loom_value_id_t rhs[] = {rhs0, rhs1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, root, LOOM_VALUE_ID_INVALID, NULL, 0, rhs,
      IREE_ARRAYSIZE(rhs), i32, LOOM_LOCATION_UNKNOWN, &op));

  EXPECT_EQ(op->operand_count, 3u);
  const uint16_t* counts = loom_op_const_operand_segment_counts(op);
  EXPECT_EQ(counts[0], 1u);
  EXPECT_EQ(counts[1], 0u);
  EXPECT_EQ(counts[2], 0u);
  EXPECT_EQ(counts[3], 2u);
  EXPECT_EQ(loom_test_segmented_root(op), root);
  EXPECT_FALSE(loom_test_segmented_guard_is_present(op));
  EXPECT_EQ(loom_test_segmented_guard(op), LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(loom_test_segmented_lhs(op).count, 0u);
  loom_value_slice_t rhs_slice = loom_test_segmented_rhs(op);
  ASSERT_EQ(rhs_slice.count, 2u);
  EXPECT_EQ(rhs_slice.values[0], rhs0);
  EXPECT_EQ(rhs_slice.values[1], rhs1);
}

TEST_F(BuilderTest, OperandDictBuilderCanonicalizesNames) {
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  loom_value_id_t beta = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alpha = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &input));
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &beta));
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &alpha));

  loom_string_id_t alpha_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t beta_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("alpha"), &alpha_name));
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("beta"), &beta_name));

  loom_named_value_t params[] = {
      {/*.name_id=*/beta_name, /*.reserved=*/0, /*.value_id=*/beta},
      {/*.name_id=*/alpha_name, /*.reserved=*/0, /*.value_id=*/alpha},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_operand_dict_build(
      &builder_, input, params, IREE_ARRAYSIZE(params),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &op));

  ASSERT_NE(op, nullptr);
  ASSERT_TRUE(loom_test_operand_dict_isa(op));
  EXPECT_EQ(loom_test_operand_dict_input(op), input);
  loom_value_slice_t sorted_params = loom_test_operand_dict_params(op);
  ASSERT_EQ(sorted_params.count, 2u);
  EXPECT_EQ(sorted_params.values[0], alpha);
  EXPECT_EQ(sorted_params.values[1], beta);

  loom_named_attr_slice_t names = loom_test_operand_dict_param_names(op);
  ASSERT_EQ(names.count, 2u);
  ASSERT_NE(names.entries, nullptr);
  EXPECT_EQ(names.entries[0].name_id, alpha_name);
  EXPECT_EQ(names.entries[0].value.i64, 0);
  EXPECT_EQ(names.entries[1].name_id, beta_name);
  EXPECT_EQ(names.entries[1].value.i64, 1);
}

TEST_F(BuilderTest, GeneratedBuildersRejectCountsBeforeNarrowing) {
  constexpr iree_host_size_t kTooMany = (iree_host_size_t)UINT16_MAX + 1;
  loom_op_t* op = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_test_assume_build(&builder_, NULL, 0, NULL, 0, NULL, kTooMany,
                             LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(op, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_test_assume_build(&builder_, NULL, 0, NULL, kTooMany, NULL, 0,
                             LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(op, nullptr);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &input));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_test_operand_dict_build(&builder_, input, NULL, UINT16_MAX,
                                   loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                                   LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(op, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_test_slice_build(&builder_, input, NULL, 0, NULL, kTooMany,
                            loom_type_scalar(LOOM_SCALAR_TYPE_F32), NULL, 0,
                            LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(op, nullptr);
}

TEST_F(BuilderTest, ReplaceAllUsesWithUpdatesPredicateAttrs) {
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t old_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &old_id));
  loom_value_id_t new_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &new_id));
  loom_value_id_t bound_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &bound_id));

  loom_value_id_t values[] = {old_id, bound_id};
  loom_predicate_t predicates[] = {{
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)old_id, (int64_t)bound_id, 0},
  }};
  loom_type_t result_types[] = {index, index};

  loom_op_t* assume_op = NULL;
  IREE_ASSERT_OK(loom_test_assume_build(
      &builder_, values, IREE_ARRAYSIZE(values), predicates,
      IREE_ARRAYSIZE(predicates), result_types, IREE_ARRAYSIZE(result_types),
      LOOM_LOCATION_UNKNOWN, &assume_op));

  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, old_id)));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, bound_id)));
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_id)));

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_id, new_id));

  ASSERT_TRUE(loom_test_assume_isa(assume_op));
  EXPECT_EQ(loom_test_assume_values(assume_op).values[0], new_id);
  loom_attribute_t predicate_attr = loom_op_attrs(assume_op)[0];
  ASSERT_EQ(predicate_attr.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_EQ(predicate_attr.count, 1u);
  ASSERT_NE(predicate_attr.predicate_list, nullptr);
  EXPECT_EQ(predicate_attr.predicate_list[0].args[0], (int64_t)new_id);
  EXPECT_EQ(predicate_attr.predicate_list[0].args[1], (int64_t)bound_id);
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_id)));
}

TEST_F(BuilderTest, ReplaceAllUsesWithUpdatesTypeAttrs) {
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* old_dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), index,
                                          LOOM_LOCATION_UNKNOWN, &old_dim_op));
  loom_value_id_t old_dim = loom_test_constant_result(old_dim_op);
  loom_op_t* new_dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), index,
                                          LOOM_LOCATION_UNKNOWN, &new_dim_op));
  loom_value_id_t new_dim = loom_test_constant_result(new_dim_op);

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_op_t* input_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), f32,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim), 0);
  loom_type_id_t vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module_, vector_type, &vector_type_id));

  loom_string_id_t type_attr_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("shape"), &type_attr_name));
  loom_named_attr_t entries[] = {
      {/*.name_id=*/type_attr_name, /*.reserved=*/{},
       /*.value=*/loom_attr_type(vector_type_id)},
  };
  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &attrs_op));

  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, old_dim)));
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_dim)));

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_dim, new_dim));

  loom_named_attr_slice_t dict = loom_test_attrs_dict(attrs_op);
  ASSERT_EQ(dict.count, 1u);
  ASSERT_EQ(dict.entries[0].value.kind, LOOM_ATTR_TYPE);
  loom_type_t replaced_type =
      module_->types.entries[dict.entries[0].value.type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_type, 0), new_dim);
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_dim)));
}

TEST_F(BuilderTest, EnumAttribute) {
  // test.cmp: 2 operands, 1 result, 1 enum attr.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_CMP, 2, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_attrs(op)[0] = loom_attr_enum(3);  // "le" predicate.
  EXPECT_EQ(loom_test_cmp_predicate(op), 3);
}

TEST_F(BuilderTest, RegionAccessor) {
  // test.map: variadic operands, 1 result, 1 region.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_MAP, 2, 1, 1,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  // Set a sentinel region pointer.
  loom_region_t sentinel_region = {0};
  loom_op_regions(op)[0] = &sentinel_region;
  EXPECT_EQ(loom_test_map_body(op), &sentinel_region);
}

TEST_F(BuilderTest, TrailingStorageAlignment) {
  // Allocate an op with an odd tied-result count. This exercises the boundary
  // between the 4-byte use-index array and 2-byte tied-result records.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_CMP, 2, 1, 0,
                                          1, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_use_index_t* operand_use_indices = loom_op_operand_use_indices(op);
  EXPECT_EQ((uintptr_t)operand_use_indices % iree_alignof(loom_use_index_t), 0u)
      << "Operand use-index array must be 4-byte aligned";
  loom_tied_result_t* tied_results = loom_op_tied_results(op);
  EXPECT_EQ((uintptr_t)tied_results % iree_alignof(loom_tied_result_t), 0u)
      << "Tied-result array must be 2-byte aligned";
  loom_attribute_t* attrs = loom_op_attrs(op);
  EXPECT_EQ((uintptr_t)attrs % iree_alignof(loom_attribute_t), 0u)
      << "Attribute array must be 8-byte aligned";
}

//===----------------------------------------------------------------------===//
// Builders
//===----------------------------------------------------------------------===//

TEST_F(BuilderTest, BinaryBuilder) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, 42, 99, i32, LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_ADDI);
  EXPECT_EQ(op->operand_count, 2);
  EXPECT_EQ(op->result_count, 1);
  EXPECT_EQ(loom_test_addi_lhs(op), 42u);
  EXPECT_EQ(loom_test_addi_rhs(op), 99u);
  // Result value was defined in the module's value table.
  loom_value_id_t result_id = loom_test_addi_result(op);
  EXPECT_NE(result_id, LOOM_VALUE_ID_INVALID);
  loom_value_t* result_value = loom_module_value(module_, result_id);
  EXPECT_EQ(loom_type_kind(result_value->type), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(result_value->type), LOOM_SCALAR_TYPE_I32);
}

TEST_F(BuilderTest, UnaryBuilder) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, 7, (loom_type_t){0},
                                     LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_NEG);
  EXPECT_EQ(op->operand_count, 1);
  EXPECT_EQ(loom_test_neg_input(op), 7u);
}

TEST_F(BuilderTest, ComparisonBuilder) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_cmp_build(&builder_, 2, 10, 20, i32, i32,
                                     LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_CMP);
  EXPECT_EQ(op->attribute_count, 1);
  EXPECT_EQ(loom_test_cmp_predicate(op), 2);
  EXPECT_EQ(loom_test_cmp_lhs(op), 10u);
  EXPECT_EQ(loom_test_cmp_rhs(op), 20u);
  EXPECT_NE(loom_test_cmp_result(op), LOOM_VALUE_ID_INVALID);
}

TEST_F(BuilderTest, CastBuilder) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, 5, (loom_type_t){0},
                                      (loom_type_t){0}, LOOM_LOCATION_UNKNOWN,
                                      &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_CAST);
  EXPECT_EQ(loom_test_cast_input(op), 5u);
}

TEST_F(BuilderTest, ConstantBuilder) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_CONSTANT);
  EXPECT_EQ(op->result_count, 1);
  EXPECT_EQ(op->attribute_count, 1);
  EXPECT_EQ(loom_attr_as_i64(loom_op_attrs(op)[0]), 42);
  EXPECT_NE(loom_op_results(op)[0], LOOM_VALUE_ID_INVALID);
}

TEST_F(BuilderTest, YieldBuilder) {
  loom_value_id_t values[] = {10, 20, 30};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_yield_build(&builder_, values, 3, LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_YIELD);
  EXPECT_EQ(op->operand_count, 3);
  EXPECT_EQ(op->result_count, 0);
  loom_value_slice_t slice = loom_test_yield_values(op);
  EXPECT_EQ(slice.count, 3);
  EXPECT_EQ(slice.values[0], 10u);
  EXPECT_EQ(slice.values[2], 30u);
}

TEST_F(BuilderTest, MapBuilder) {
  // Create real tile-typed values so the builder can extract element types.
  loom_type_t tile_f32 = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_value_id_t input0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input1 = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, tile_f32, &input0));
  IREE_ASSERT_OK(loom_module_define_value(module_, tile_f32, &input1));

  loom_value_id_t inputs[] = {input0, input1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile_f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_MAP);
  EXPECT_EQ(op->operand_count, 2);
  EXPECT_EQ(op->region_count, 1);

  // Region was auto-created with an entry block.
  loom_region_t* body = loom_test_map_body(op);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->block_count, 1);

  // Entry block has 2 block args (one per input) with element types.
  loom_block_t* entry = loom_region_entry_block(body);
  EXPECT_EQ(entry->arg_count, 2);
  loom_type_t arg0_type = loom_block_arg_type(module_, entry, 0);
  EXPECT_EQ(loom_type_kind(arg0_type), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(arg0_type), LOOM_SCALAR_TYPE_F32);

  loom_value_slice_t ins = loom_test_map_inputs(op);
  EXPECT_EQ(ins.count, 2);
  EXPECT_EQ(ins.values[0], input0);
}

TEST_F(BuilderTest, InvokeBuilder) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t result_types[] = {f32, f32};
  loom_value_id_t operands[] = {5, 6, 7};
  loom_symbol_ref_t callee = {0, 1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, operands, 3,
                                        result_types, 2, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_INVOKE);
  EXPECT_EQ(op->operand_count, 3);
  EXPECT_EQ(op->result_count, 2);
  EXPECT_EQ(op->attribute_count, 1);
  loom_symbol_ref_t read_callee = loom_test_invoke_callee(op);
  EXPECT_EQ(read_callee.symbol_id, 1);
}

TEST_F(BuilderTest, LoopBuilder) {
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Create real values for bounds, step, and iter_args.
  loom_value_id_t lo = LOOM_VALUE_ID_INVALID;
  loom_value_id_t hi = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t init = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &lo));
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &hi));
  IREE_ASSERT_OK(loom_module_define_value(module_, index, &step));
  IREE_ASSERT_OK(loom_module_define_value(module_, f32, &init));

  loom_value_id_t iter_args[] = {init};
  loom_type_t loop_result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lo, hi, step, iter_args, 1,
                                      loop_result_types, 1, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_LOOP);
  EXPECT_EQ(op->operand_count, 4);
  EXPECT_EQ(op->result_count, 1);
  EXPECT_EQ(op->region_count, 1);
  EXPECT_EQ(loom_test_loop_lower_bound(op), lo);
  EXPECT_EQ(loom_test_loop_upper_bound(op), hi);
  EXPECT_EQ(loom_test_loop_step(op), step);

  // Region was auto-created. Entry block has IV (index) + 1 iter_arg (f32).
  loom_region_t* body = loom_test_loop_body(op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry = loom_region_entry_block(body);
  EXPECT_EQ(entry->arg_count, 2);

  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, entry, 0)),
            LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, entry, 1)),
            LOOM_SCALAR_TYPE_F32);

  loom_value_slice_t iargs = loom_test_loop_iter_args(op);
  EXPECT_EQ(iargs.count, 1);
  EXPECT_EQ(iargs.values[0], init);
}

TEST_F(BuilderTest, BranchBuilder) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t cond = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &cond));

  loom_type_t branch_result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, cond, branch_result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_BRANCH);
  EXPECT_EQ(op->operand_count, 1);
  EXPECT_EQ(op->region_count, 2);
  EXPECT_EQ(loom_test_branch_condition(op), cond);

  // Both regions auto-created with 1 empty block each.
  loom_region_t* then_region = loom_test_branch_then_region(op);
  loom_region_t* else_region = loom_test_branch_else_region(op);
  ASSERT_NE(then_region, nullptr);
  ASSERT_NE(else_region, nullptr);
  EXPECT_EQ(then_region->block_count, 1);
  EXPECT_EQ(else_region->block_count, 1);
  EXPECT_EQ(loom_region_entry_arg_count(then_region), 0);
  EXPECT_EQ(loom_region_entry_arg_count(else_region), 0);
}

TEST_F(BuilderTest, SliceBuilder) {
  loom_value_id_t offsets[] = {50};
  int64_t static_offsets[] = {0};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, 1, offsets, 1, static_offsets,
                                       1, (loom_type_t){0}, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_SLICE);
  EXPECT_EQ(op->operand_count, 2);
  EXPECT_EQ(op->attribute_count, 1);
  EXPECT_EQ(loom_test_slice_source(op), 1u);
  loom_value_slice_t offs = loom_test_slice_offsets(op);
  EXPECT_EQ(offs.count, 1);
  EXPECT_EQ(offs.values[0], 50u);
  // Verify static_offsets attr was stored.
  loom_attribute_t attr = loom_op_attrs(op)[0];
  EXPECT_EQ(attr.kind, LOOM_ATTR_I64_ARRAY);
  EXPECT_EQ(attr.count, 1);
  EXPECT_EQ(attr.i64_array[0], 0);
}

TEST_F(BuilderTest, UpdateBuilder) {
  loom_value_id_t offsets[] = {50, 60};
  int64_t static_offsets[] = {-1, 0};  // -1 = dynamic sentinel, 0 = static.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_update_build(&builder_, 1, 2, offsets, 2,
                                        static_offsets, 2, (loom_type_t){0},
                                        LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_UPDATE);
  EXPECT_EQ(op->operand_count, 4);
  EXPECT_EQ(op->attribute_count, 1);
  EXPECT_EQ(loom_test_update_source(op), 1u);
  EXPECT_EQ(loom_test_update_target(op), 2u);
  loom_value_slice_t update_offs = loom_test_update_offsets(op);
  EXPECT_EQ(update_offs.count, 2);
  EXPECT_EQ(update_offs.values[0], 50u);
  EXPECT_EQ(update_offs.values[1], 60u);
  // Verify static_offsets attr.
  loom_attribute_t attr = loom_op_attrs(op)[0];
  EXPECT_EQ(attr.kind, LOOM_ATTR_I64_ARRAY);
  EXPECT_EQ(attr.count, 2);
  EXPECT_EQ(attr.i64_array[0], -1);
  EXPECT_EQ(attr.i64_array[1], 0);
}

TEST_F(BuilderTest, FuncBuilder) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t name = {0, 5};
  loom_type_t arg_types[] = {f32, i32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, name, arg_types, 2,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_FUNC);
  EXPECT_EQ(op->attribute_count, 4);  // callee, visibility, cc, predicates
  EXPECT_EQ(op->region_count, 1);
  loom_symbol_ref_t callee = loom_test_func_callee(op);
  EXPECT_EQ(callee.symbol_id, 5);

  // Body region auto-created with entry block args from arg_types.
  loom_region_t* body = loom_test_func_body(op);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->block_count, 1);
  loom_block_t* entry = loom_region_entry_block(body);
  EXPECT_EQ(entry->arg_count, 2);
  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, entry, 0)),
            LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, entry, 1)),
            LOOM_SCALAR_TYPE_I32);
}

TEST_F(BuilderTest, SplitFuncBuilderProjectsSignatureArgs) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t name = {0, 7};
  loom_type_t arg_types[] = {f32, i32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_split_func_build(&builder_, 0, 0, 0, name, arg_types,
                                            2, LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_SPLIT_FUNC);
  EXPECT_EQ(op->region_count, 2);

  loom_region_t* body = loom_test_split_func_body(op);
  loom_region_t* config = loom_test_split_func_config(op);
  ASSERT_NE(body, nullptr);
  ASSERT_NE(config, nullptr);
  loom_block_t* body_entry = loom_region_entry_block(body);
  loom_block_t* config_entry = loom_region_entry_block(config);
  ASSERT_NE(body_entry, nullptr);
  ASSERT_NE(config_entry, nullptr);
  ASSERT_EQ(body_entry->arg_count, 2);
  ASSERT_EQ(config_entry->arg_count, 2);

  EXPECT_NE(loom_block_arg_id(body_entry, 0),
            loom_block_arg_id(config_entry, 0));
  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, body_entry, 0)),
            LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(
      loom_type_element_type(loom_block_arg_type(module_, config_entry, 0)),
      LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(loom_type_element_type(loom_block_arg_type(module_, body_entry, 1)),
            LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(
      loom_type_element_type(loom_block_arg_type(module_, config_entry, 1)),
      LOOM_SCALAR_TYPE_I32);
}

TEST_F(BuilderTest, TargetLikeInterfaceReadsTargetRecordFields) {
  loom_symbol_ref_t symbol = {0, 7};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_target_build(
      &builder_, /*build_flags=*/0, LOOM_TEST_TARGET_KIND_LOW_CORE, symbol,
      /*codegen_format=*/0, /*artifact_format=*/0,
      /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
      /*offset_bitwidth=*/0, /*max_workgroup_size_x=*/0,
      /*max_workgroup_size_y=*/0, /*max_workgroup_size_z=*/0,
      /*max_flat_workgroup_size=*/0, /*subgroup_size=*/0,
      /*max_grid_size_x=*/0, /*max_grid_size_y=*/0, /*max_grid_size_z=*/0,
      /*max_flat_grid_size=*/0,
      /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
      /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
      /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
      /*memory_space_constant=*/0, /*memory_space_private=*/0,
      /*memory_space_host=*/0, /*memory_space_descriptor=*/0, /*abi=*/0,
      /*export_symbol=*/LOOM_STRING_ID_INVALID, /*linkage=*/0,
      /*hal_buffer_resource_flags=*/0,
      /*contract_set_key=*/LOOM_STRING_ID_INVALID, /*contract_feature_bits=*/0,
      LOOM_LOCATION_UNKNOWN, &op));

  loom_target_like_t target = loom_target_like_cast(module_, op);
  ASSERT_TRUE(loom_target_like_isa(target));
  EXPECT_TRUE(loom_symbol_definition_implements(
      loom_op_vtable(module_, op)->symbol_def, LOOM_SYMBOL_INTERFACE_TARGET));
  EXPECT_EQ(loom_target_like_symbol(target).symbol_id, 7u);
  EXPECT_EQ(loom_attr_as_enum(loom_target_like_selector(target)),
            LOOM_TEST_TARGET_KIND_LOW_CORE);
  EXPECT_EQ(loom_target_like_extension_attrs(target).count, 0u);
  EXPECT_NE(loom_target_like_descriptor(target), nullptr);
}

TEST_F(BuilderTest, AttrsBuilder) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 42,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_ATTRS);
  EXPECT_EQ(op->operand_count, 1);
  EXPECT_EQ(op->result_count, 1);
  EXPECT_EQ(loom_test_attrs_input(op), 42u);
  loom_named_attr_slice_t dict = loom_test_attrs_dict(op);
  EXPECT_EQ(dict.entries, nullptr);
  EXPECT_EQ(dict.count, 0u);
}

TEST_F(BuilderTest, AttrsBuilderCanonicalizesDict) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, 42,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &op));

  loom_named_attr_slice_t dict = loom_test_attrs_dict(op);
  loom_attribute_t dict_attr =
      loom_make_canonical_attr_dict(dict.entries, dict.count);
  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module_, dict_attr));
  ASSERT_EQ(dict.count, 2u);
  ASSERT_NE(dict.entries, nullptr);
  EXPECT_EQ(dict.entries[0].name_id, alpha_id);
  EXPECT_EQ(loom_attr_as_i64(dict.entries[0].value), 1);
  EXPECT_EQ(dict.entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(dict.entries[1].value), 2);
}

TEST_F(BuilderTest, AttrsBuilderRejectsDuplicateDictKeys) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_op_t* op = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_test_attrs_build(
          &builder_, 42,
          loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
          LOOM_LOCATION_UNKNOWN, &op));
}

TEST_F(BuilderTest, DeflateBuilder) {
  loom_type_t tensor_dyn = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(0), 0);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t result_types[] = {tensor_dyn, index};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_deflate_build(&builder_, 42, result_types, 2, NULL,
                                         0, LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, LOOM_OP_TEST_DEFLATE);
  EXPECT_EQ(op->operand_count, 1);
  EXPECT_EQ(op->result_count, 2);
  EXPECT_EQ(loom_test_deflate_input(op), 42u);
  loom_value_slice_t results = loom_test_deflate_results(op);
  EXPECT_EQ(results.count, 2);
}

TEST_F(BuilderTest, OpErase) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, 1, 2, (loom_type_t){0},
                                      LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(op->flags & LOOM_OP_FLAG_DEAD, 0u);
  loom_block_t* block = loom_module_block(module_);
  EXPECT_EQ(block->op_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(module_, op));
  EXPECT_NE(op->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_EQ(block->op_count, 0u);
  EXPECT_EQ(block->first_op, nullptr);
  EXPECT_EQ(block->last_op, nullptr);
  EXPECT_EQ(op->prev_op, nullptr);
  EXPECT_EQ(op->next_op, nullptr);
  EXPECT_EQ(op->parent_block, block);
}

TEST_F(BuilderTest, SetBeforePreservesConsecutiveInsertionOrder) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* first = NULL;
  loom_op_t* anchor = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(4), i32,
                                          LOOM_LOCATION_UNKNOWN, &anchor));

  loom_builder_set_before(&builder_, anchor);
  loom_op_t* inserted_a = NULL;
  loom_op_t* inserted_b = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inserted_a));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &inserted_b));

  loom_block_t* block = loom_module_block(module_);
  ASSERT_EQ(block->op_count, 4u);
  EXPECT_EQ(loom_block_op(block, 0), first);
  EXPECT_EQ(loom_block_op(block, 1), inserted_a);
  EXPECT_EQ(loom_block_op(block, 2), inserted_b);
  EXPECT_EQ(loom_block_op(block, 3), anchor);
  EXPECT_LT(first->block_ordinal, inserted_a->block_ordinal);
  EXPECT_LT(inserted_a->block_ordinal, inserted_b->block_ordinal);
  EXPECT_LT(inserted_b->block_ordinal, anchor->block_ordinal);
}

TEST_F(BuilderTest, SetBeforeInheritsNestedParentOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* input_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&builder_, map_op, body);
  loom_op_t* anchor = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &anchor));
  loom_builder_restore(&builder_, saved_ip);

  loom_builder_set_before(&builder_, anchor);
  loom_op_t* inserted = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inserted));

  EXPECT_EQ(inserted->parent_op, map_op);
  EXPECT_EQ(inserted->parent_block, loom_region_entry_block(body));
  EXPECT_EQ(loom_block_op(loom_region_entry_block(body), 0), inserted);
  EXPECT_EQ(loom_block_op(loom_region_entry_block(body), 1), anchor);
}

TEST_F(BuilderTest, SaveRestore) {
  loom_builder_ip_t saved = loom_builder_save(&builder_);
  EXPECT_EQ(saved.block, loom_module_block(module_));
  EXPECT_EQ(saved.before_op, nullptr);

  // Create a second block and switch to it.
  loom_block_t* other_block = NULL;
  IREE_ASSERT_OK(loom_module_allocate_block(module_, &other_block));
  loom_builder_set_block(&builder_, other_block);
  EXPECT_EQ(builder_.ip.block, other_block);

  // Restore back to the original block.
  loom_builder_restore(&builder_, saved);
  EXPECT_EQ(builder_.ip.block, loom_module_block(module_));
}

//===----------------------------------------------------------------------===//
// Use-def chain maintenance
//===----------------------------------------------------------------------===//

// Helper: build a test.constant op returning a value of the given type.
static loom_value_id_t build_constant(loom_builder_t* builder,
                                      loom_module_t* module, loom_type_t type,
                                      loom_op_t** out_op = nullptr) {
  loom_op_t* op = NULL;
  IREE_CHECK_OK(loom_test_constant_build(builder, loom_attr_i64(0), type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  if (out_op) *out_op = op;
  return loom_test_constant_result(op);
}

TEST_F(BuilderTest, UseCountAfterBuild) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t b = loom_test_addi_result(addi);

  // %a is used twice (lhs and rhs of addi).
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 2);
  // %b (addi result) has no uses.
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 0);
}

TEST_F(BuilderTest, OpRemoveResultsCompactsDefinitionsAndTiedResults) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t operand = build_constant(&builder_, module_, f32);
  loom_value_id_t operands[] = {operand};
  loom_type_t result_types[] = {f32, f32, f32};
  loom_tied_result_t tied[] = {
      {/*.result_index=*/2, /*.operand_index=*/0, /*.has_type_change=*/false}};
  loom_symbol_ref_t callee = {0, 1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, callee, operands, IREE_ARRAYSIZE(operands), result_types,
      IREE_ARRAYSIZE(result_types), tied, IREE_ARRAYSIZE(tied),
      LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t old_result0 = loom_op_results(op)[0];
  const loom_value_id_t old_result1 = loom_op_results(op)[1];
  const loom_value_id_t old_result2 = loom_op_results(op)[2];

  bool remove_results[] = {false, true, false};
  uint16_t removed_count = 0;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  iree_status_t status = loom_op_remove_results(module_, op, remove_results,
                                                &scratch_arena, &removed_count);
  iree_arena_deinitialize(&scratch_arena);
  IREE_ASSERT_OK(status);

  EXPECT_EQ(removed_count, 1u);
  EXPECT_EQ(op->result_count, 2u);
  EXPECT_EQ(loom_op_results(op)[0], old_result0);
  EXPECT_EQ(loom_op_results(op)[1], old_result2);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, old_result0)), op);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module_, old_result0)), 0u);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, old_result2)), op);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module_, old_result2)), 1u);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, old_result1)),
            nullptr);
  ASSERT_EQ(op->tied_result_count, 1u);
  EXPECT_EQ(loom_op_tied_results(op)[0].result_index, 1u);
  EXPECT_EQ(loom_op_tied_results(op)[0].operand_index, 0u);
  EXPECT_EQ(loom_test_invoke_callee(op).symbol_id, 1u);

  EXPECT_EQ(loom_module_value(module_, operand)->use_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(module_, op));
  EXPECT_EQ(loom_module_value(module_, operand)->use_count, 0u);
}

TEST_F(BuilderTest, InlineUsesPopulated) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));

  loom_value_t* value_a = loom_module_value(module_, a);
  ASSERT_EQ(value_a->use_count, 2);
  EXPECT_FALSE(loom_value_has_overflow_uses(value_a));

  // Both uses point to the addi op with different operand indices.
  const loom_use_t* uses = loom_value_uses(value_a);
  EXPECT_EQ(loom_use_user_op(uses[0]), addi);
  EXPECT_EQ(loom_use_user_op(uses[1]), addi);
  // Operand indices should be 0 and 1 (order may vary due to no ordering
  // guarantee, but since they were added sequentially they'll be in order).
  uint16_t idx0 = loom_use_operand_index(uses[0]);
  uint16_t idx1 = loom_use_operand_index(uses[1]);
  EXPECT_NE(idx0, idx1);
  EXPECT_TRUE((idx0 == 0 && idx1 == 1) || (idx0 == 1 && idx1 == 0));
}

TEST_F(BuilderTest, OverflowTransition) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t v = build_constant(&builder_, module_, i32);

  // Build 4 neg ops that all use %v. The 4th triggers overflow.
  loom_op_t* neg_ops[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(loom_test_neg_build(&builder_, v, i32, LOOM_LOCATION_UNKNOWN,
                                       &neg_ops[i]));
  }

  loom_value_t* value_v = loom_module_value(module_, v);
  EXPECT_EQ(value_v->use_count, 4);
  EXPECT_TRUE(loom_value_has_overflow_uses(value_v));

  // All 4 uses should be accessible and point to the correct ops.
  const loom_use_t* uses = loom_value_uses(value_v);
  for (int i = 0; i < 4; ++i) {
    loom_op_t* user = loom_use_user_op(uses[i]);
    // Each use should be one of our neg ops.
    bool found = false;
    for (int j = 0; j < 4; ++j) {
      if (user == neg_ops[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "use " << i << " points to unknown op";
    EXPECT_EQ(loom_use_operand_index(uses[i]), 0);
  }
}

TEST_F(BuilderTest, ZeroOperandOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  loom_value_id_t c = build_constant(&builder_, module_, i32, &const_op);

  // Constant has 0 operands, so no uses are added.
  // Its result has 0 uses (nothing consumes it yet).
  EXPECT_EQ(loom_module_value(module_, c)->use_count, 0);
}

TEST_F(BuilderTest, SetOperand) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t b = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 1);

  // Replace operand 0 (was %a) with %b.
  IREE_ASSERT_OK(loom_op_set_operand(module_, addi, 0, b));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 2);
  EXPECT_EQ(loom_op_operands(addi)[0], b);
  EXPECT_EQ(loom_op_operands(addi)[1], b);
}

TEST_F(BuilderTest, ReplaceAllUsesWith) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t d = build_constant(&builder_, module_, i32);

  // %b = addi %a, %a (two uses of %a)
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));
  // %c = neg %a (one more use of %a)
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 3);
  EXPECT_EQ(loom_module_value(module_, d)->use_count, 0);

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, a, d));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, d)->use_count, 3);

  // Operand slots should all point to %d now.
  EXPECT_EQ(loom_op_operands(addi)[0], d);
  EXPECT_EQ(loom_op_operands(addi)[1], d);
  EXPECT_EQ(loom_op_operands(neg)[0], d);
}

TEST_F(BuilderTest, ReplaceAllUsesWithUpdatesTypeUses) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t old_dim = build_constant(&builder_, module_, index_type);
  loom_value_id_t new_dim = build_constant(&builder_, module_, index_type);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim), 0);
  loom_value_id_t vector = build_constant(&builder_, module_, vector_type);

  EXPECT_EQ(loom_module_value(module_, old_dim)->use_count, 0);
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, old_dim));
  EXPECT_FALSE(loom_module_value_has_type_uses(module_, new_dim));

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_dim, new_dim));

  loom_type_t replaced_type = loom_module_value_type(module_, vector);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_type, 0), new_dim);
  EXPECT_FALSE(loom_module_value_has_type_uses(module_, old_dim));
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, new_dim));
}

TEST_F(BuilderTest, ReplaceAllUsesExcept) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t replacement = build_constant(&builder_, module_, i32);

  loom_op_t* neg_b = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg_b));
  loom_op_t* neg_c = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg_c));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 2);

  // Replace all uses of %a with %replacement, except in neg_b.
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_except(module_, a, replacement, neg_b));

  // neg_b still uses %a, neg_c now uses %replacement.
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
  EXPECT_EQ(loom_module_value(module_, replacement)->use_count, 1);
  EXPECT_EQ(loom_op_operands(neg_b)[0], a);
  EXPECT_EQ(loom_op_operands(neg_c)[0], replacement);
}

static bool is_specific_op(const loom_op_t* user_op, void* user_data) {
  return user_op == (const loom_op_t*)user_data;
}

TEST_F(BuilderTest, ReplaceUsesIf) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t replacement = build_constant(&builder_, module_, i32);

  loom_op_t* neg_b = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg_b));
  loom_op_t* neg_c = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg_c));

  // Replace uses only in neg_c.
  IREE_ASSERT_OK(loom_value_replace_uses_if(module_, a, replacement,
                                            is_specific_op, neg_c));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
  EXPECT_EQ(loom_module_value(module_, replacement)->use_count, 1);
  EXPECT_EQ(loom_op_operands(neg_b)[0], a);
  EXPECT_EQ(loom_op_operands(neg_c)[0], replacement);
}

TEST_F(BuilderTest, ReplaceAllUsesWithSelf) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // RAUW(a, a) should be a no-op.
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, a, a));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
  EXPECT_EQ(loom_op_operands(neg)[0], a);
}

TEST_F(BuilderTest, EraseWithUses) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 2);

  // Erase the addi (its result has no uses, so this should succeed).
  IREE_ASSERT_OK(loom_op_erase(module_, addi));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_NE(addi->flags & LOOM_OP_FLAG_DEAD, 0u);
}

TEST_F(BuilderTest, EraseWithUsedResultFails) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  loom_value_id_t a = build_constant(&builder_, module_, i32, &const_op);

  // Use %a so the constant's result has uses.
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // Erasing the constant should fail because its result still has uses.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_op_erase(module_, const_op));

  // The op should NOT be dead (erase failed).
  EXPECT_EQ(const_op->flags & LOOM_OP_FLAG_DEAD, 0u);
}

TEST_F(BuilderTest, EraseWithTypeUsedResultFails) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  loom_value_id_t dim = build_constant(&builder_, module_, index_type, &dim_op);
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim), 0);
  loom_value_id_t vector = build_constant(&builder_, module_, vector_type);

  EXPECT_EQ(loom_module_value(module_, dim)->use_count, 0);
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, dim));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_op_erase(module_, dim_op));

  EXPECT_EQ(dim_op->flags & LOOM_OP_FLAG_DEAD, 0u);
  EXPECT_TRUE(
      loom_type_references_value(loom_module_value_type(module_, vector), dim));
}

TEST_F(BuilderTest, ComputeUses) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // Verify initial state.
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 3);

  // Manually zero all use counts.
  for (iree_host_size_t i = 0; i < module_->values.count; ++i) {
    loom_module_value(module_, i)->use_count = 0;
    loom_module_value(module_, i)->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);

  // Rebuild from scratch.
  IREE_ASSERT_OK(loom_module_compute_uses(module_));

  // Should match the original state.
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 3);
}

TEST_F(BuilderTest, SameValueTwoOperands) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t x = build_constant(&builder_, module_, i32);

  // %r = addi %x, %x — same value for both operands.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, x, x, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(loom_module_value(module_, x)->use_count, 2);

  // RAUW should replace both operand slots.
  loom_value_id_t y = build_constant(&builder_, module_, i32);
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, x, y));
  EXPECT_EQ(loom_module_value(module_, x)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, y)->use_count, 2);
  EXPECT_EQ(loom_op_operands(addi)[0], y);
  EXPECT_EQ(loom_op_operands(addi)[1], y);

  // Erasing should remove both uses.
  IREE_ASSERT_OK(loom_op_erase(module_, addi));
  EXPECT_EQ(loom_module_value(module_, y)->use_count, 0);
}

TEST_F(BuilderTest, VariadicOperandUses) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t b = build_constant(&builder_, module_, i32);
  loom_value_id_t c = build_constant(&builder_, module_, i32);

  loom_value_id_t values[] = {a, b, c};
  loom_op_t* yield = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, values, 3,
                                       LOOM_LOCATION_UNKNOWN, &yield));

  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 1);
  EXPECT_EQ(loom_module_value(module_, c)->use_count, 1);

  // Erasing removes all three uses.
  IREE_ASSERT_OK(loom_op_erase(module_, yield));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, c)->use_count, 0);
}

TEST_F(BuilderTest, OverflowGrowth) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t v = build_constant(&builder_, module_, i32);

  // Build 20 ops that all use %v. This tests multiple overflow growths.
  loom_op_t* ops[20];
  for (int i = 0; i < 20; ++i) {
    IREE_ASSERT_OK(
        loom_test_neg_build(&builder_, v, i32, LOOM_LOCATION_UNKNOWN, &ops[i]));
  }

  loom_value_t* value_v = loom_module_value(module_, v);
  EXPECT_EQ(value_v->use_count, 20);
  EXPECT_TRUE(loom_value_has_overflow_uses(value_v));

  // All 20 entries should be accessible.
  const loom_use_t* uses = loom_value_uses(value_v);
  for (int i = 0; i < 20; ++i) {
    EXPECT_NE(loom_use_user_op(uses[i]), nullptr);
    EXPECT_EQ(loom_use_operand_index(uses[i]), 0);
  }
}

TEST_F(BuilderTest, HighFanoutUsesExceedUint16AndRemoveInConstantTime) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t v = build_constant(&builder_, module_, i32);

  constexpr uint32_t kUseCount = 70000;
  loom_op_t* first = NULL;
  loom_op_t* middle = NULL;
  loom_op_t* last = NULL;
  for (uint32_t i = 0; i < kUseCount; ++i) {
    loom_op_t* op = NULL;
    IREE_ASSERT_OK(
        loom_test_neg_build(&builder_, v, i32, LOOM_LOCATION_UNKNOWN, &op));
    if (i == 0) first = op;
    if (i == kUseCount / 2) middle = op;
    if (i == kUseCount - 1) last = op;
  }

  loom_value_t* value_v = loom_module_value(module_, v);
  EXPECT_EQ(value_v->use_count, kUseCount);
  EXPECT_TRUE(loom_value_has_overflow_uses(value_v));

  // Removing from the middle swaps another use into the removed slot and must
  // update that moved user's operand backpointer.
  IREE_ASSERT_OK(loom_op_erase(module_, middle));
  EXPECT_EQ(value_v->use_count, kUseCount - 1);
  IREE_ASSERT_OK(loom_op_erase(module_, first));
  EXPECT_EQ(value_v->use_count, kUseCount - 2);
  IREE_ASSERT_OK(loom_op_erase(module_, last));
  EXPECT_EQ(value_v->use_count, kUseCount - 3);
}

TEST_F(BuilderTest, HasNoUsesAndSingleUse) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  EXPECT_TRUE(loom_value_has_no_uses(loom_module_value(module_, a)));
  EXPECT_FALSE(loom_value_has_single_use(loom_module_value(module_, a)));

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  EXPECT_FALSE(loom_value_has_no_uses(loom_module_value(module_, a)));
  EXPECT_TRUE(loom_value_has_single_use(loom_module_value(module_, a)));
}

TEST_F(BuilderTest, ReplaceAllUsesWithCrossesOverflowBoundary) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t old_v = build_constant(&builder_, module_, i32);
  loom_value_id_t new_v = build_constant(&builder_, module_, i32);

  // Give old 3 uses (fills inline), give new 2 uses (inline).
  loom_op_t* old_users[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(loom_test_neg_build(&builder_, old_v, i32,
                                       LOOM_LOCATION_UNKNOWN, &old_users[i]));
  }
  loom_op_t* new_users[2];
  for (int i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_test_neg_build(&builder_, new_v, i32,
                                       LOOM_LOCATION_UNKNOWN, &new_users[i]));
  }

  EXPECT_EQ(loom_module_value(module_, old_v)->use_count, 3);
  EXPECT_EQ(loom_module_value(module_, new_v)->use_count, 2);
  EXPECT_FALSE(loom_value_has_overflow_uses(loom_module_value(module_, new_v)));

  // RAUW: new goes from 2 to 5 uses (must transition to overflow).
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_v, new_v));

  EXPECT_EQ(loom_module_value(module_, old_v)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, new_v)->use_count, 5);
  EXPECT_TRUE(loom_value_has_overflow_uses(loom_module_value(module_, new_v)));

  // Verify all operand slots were patched.
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(loom_op_operands(old_users[i])[0], new_v);
  }
}

TEST_F(BuilderTest, ChainedReplaceAllUsesWith) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);
  loom_value_id_t b = build_constant(&builder_, module_, i32);
  loom_value_id_t c = build_constant(&builder_, module_, i32);

  // Two ops use %a.
  loom_op_t* neg1 = NULL;
  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg1));
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg2));

  // RAUW a -> b, then b -> c.
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, a, b));
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 2);

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, b, c));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, b)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, c)->use_count, 2);
  EXPECT_EQ(loom_op_operands(neg1)[0], c);
  EXPECT_EQ(loom_op_operands(neg2)[0], c);
}

TEST_F(BuilderTest, SetOperandToInvalid) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);

  // Disconnect the operand by setting to INVALID.
  IREE_ASSERT_OK(loom_op_set_operand(module_, neg, 0, LOOM_VALUE_ID_INVALID));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 0);
  EXPECT_EQ(loom_op_operands(neg)[0], LOOM_VALUE_ID_INVALID);
}

TEST_F(BuilderTest, ComputeUsesSkipsDeadOps) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* neg1 = NULL;
  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg1));
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg2));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 2);

  // Erase neg1 (RAUW its result first — it has no uses, so erase succeeds).
  IREE_ASSERT_OK(loom_op_erase(module_, neg1));
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);

  // Zero everything and rebuild. Dead ops should be skipped.
  for (iree_host_size_t i = 0; i < module_->values.count; ++i) {
    loom_module_value(module_, i)->use_count = 0;
    loom_module_value(module_, i)->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  IREE_ASSERT_OK(loom_module_compute_uses(module_));

  // Only neg2 should be counted (neg1 is dead).
  EXPECT_EQ(loom_module_value(module_, a)->use_count, 1);
}

TEST_F(BuilderTest, ForEachUseMacro) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // Walk uses with the macro. Count how many point to addi vs neg.
  loom_value_t* value_a = loom_module_value(module_, a);
  int addi_count = 0;
  int neg_count = 0;
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value_a, use) {
    if (loom_use_user_op(*use) == addi) ++addi_count;
    if (loom_use_user_op(*use) == neg) ++neg_count;
  }
  EXPECT_EQ(addi_count, 2);  // lhs and rhs
  EXPECT_EQ(neg_count, 1);
}

TEST_F(BuilderTest, ReplaceAllUsesWithBothOverflow) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t old_v = build_constant(&builder_, module_, i32);
  loom_value_id_t new_v = build_constant(&builder_, module_, i32);

  // Give both values >3 uses (both in overflow mode).
  for (int i = 0; i < 5; ++i) {
    loom_op_t* unused = NULL;
    IREE_ASSERT_OK(loom_test_neg_build(&builder_, old_v, i32,
                                       LOOM_LOCATION_UNKNOWN, &unused));
    IREE_ASSERT_OK(loom_test_neg_build(&builder_, new_v, i32,
                                       LOOM_LOCATION_UNKNOWN, &unused));
  }
  EXPECT_TRUE(loom_value_has_overflow_uses(loom_module_value(module_, old_v)));
  EXPECT_TRUE(loom_value_has_overflow_uses(loom_module_value(module_, new_v)));

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_v, new_v));

  EXPECT_EQ(loom_module_value(module_, old_v)->use_count, 0);
  EXPECT_EQ(loom_module_value(module_, new_v)->use_count, 10);
}

//===----------------------------------------------------------------------===//
// Value def pointers
//===----------------------------------------------------------------------===//

TEST_F(BuilderTest, DefOpAfterBuild) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));
  loom_value_id_t result_id = loom_test_addi_result(addi);

  loom_value_t* result = loom_module_value(module_, result_id);
  EXPECT_FALSE(loom_value_is_block_arg(result));
  EXPECT_EQ(loom_value_def_op(result), addi);
  EXPECT_EQ(loom_value_def_index(result), 0);
}

TEST_F(BuilderTest, DefOpConstant) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  loom_value_id_t c = build_constant(&builder_, module_, i32, &const_op);

  loom_value_t* value = loom_module_value(module_, c);
  EXPECT_EQ(loom_value_def_op(value), const_op);
  EXPECT_EQ(loom_value_def_index(value), 0);
}

TEST_F(BuilderTest, DefOpMultiResult) {
  loom_type_t tensor_dyn = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(0), 0);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t result_types[] = {tensor_dyn, index};
  loom_op_t* deflate = NULL;
  IREE_ASSERT_OK(loom_test_deflate_build(&builder_, 42, result_types, 2, NULL,
                                         0, LOOM_LOCATION_UNKNOWN, &deflate));

  loom_value_slice_t results = loom_test_deflate_results(deflate);
  ASSERT_EQ(results.count, 2);

  // Both results point to the same op but with different indices.
  loom_value_t* r0 = loom_module_value(module_, results.values[0]);
  loom_value_t* r1 = loom_module_value(module_, results.values[1]);
  EXPECT_EQ(loom_value_def_op(r0), deflate);
  EXPECT_EQ(loom_value_def_op(r1), deflate);
  EXPECT_EQ(loom_value_def_index(r0), 0);
  EXPECT_EQ(loom_value_def_index(r1), 1);
}

TEST_F(BuilderTest, DefBlockArg) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_block_t* block = NULL;
  IREE_ASSERT_OK(loom_module_allocate_block(module_, &block));

  // Define a value and add it as a block argument.
  loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &arg_id));
  IREE_ASSERT_OK(loom_block_add_arg(module_, block, arg_id));

  loom_value_t* value = loom_module_value(module_, arg_id);
  EXPECT_TRUE(loom_value_is_block_arg(value));
  EXPECT_EQ(loom_value_def_block(value), block);
  EXPECT_EQ(loom_value_def_index(value), 0);

  // Add a second block arg and verify its index.
  loom_value_id_t arg2_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, i32, &arg2_id));
  IREE_ASSERT_OK(loom_block_add_arg(module_, block, arg2_id));

  loom_value_t* value2 = loom_module_value(module_, arg2_id);
  EXPECT_EQ(loom_value_def_block(value2), block);
  EXPECT_EQ(loom_value_def_index(value2), 1);
}

TEST_F(BuilderTest, DefOpClearedOnErase) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  loom_value_id_t c = build_constant(&builder_, module_, i32, &const_op);

  loom_value_t* value = loom_module_value(module_, c);
  EXPECT_EQ(loom_value_def_op(value), const_op);

  // Erase the constant (no uses, so erase succeeds).
  IREE_ASSERT_OK(loom_op_erase(module_, const_op));

  // Def pointer should be cleared to NULL.
  EXPECT_EQ(loom_value_def_op(value), nullptr);
}

TEST_F(BuilderTest, DefOpRestoredByComputeUses) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  loom_value_id_t c = build_constant(&builder_, module_, i32, &const_op);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, c, i32, LOOM_LOCATION_UNKNOWN, &neg));
  loom_value_id_t neg_result = loom_test_neg_result(neg);

  // Verify initial state.
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, c)), const_op);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, neg_result)), neg);

  // Rebuild from scratch (compute_uses clears and restores).
  IREE_ASSERT_OK(loom_module_compute_uses(module_));

  // Def pointers should be restored.
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, c)), const_op);
  EXPECT_EQ(loom_value_def_op(loom_module_value(module_, neg_result)), neg);
  EXPECT_EQ(loom_value_def_index(loom_module_value(module_, neg_result)), 0);
}

TEST_F(BuilderTest, DefOpPatternMatch) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // %a = constant 42
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  // %b = neg %a
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // Pattern match: walk from neg's operand to %a's def op.
  loom_value_id_t neg_input = loom_test_neg_input(neg);
  EXPECT_EQ(neg_input, a);

  loom_value_t* input_value = loom_module_value(module_, neg_input);
  loom_op_t* def_op = loom_value_def_op(input_value);
  ASSERT_NE(def_op, nullptr);
  EXPECT_TRUE(loom_test_constant_isa(def_op));
  EXPECT_EQ(loom_attr_as_i64(loom_op_attrs(def_op)[0]), 42);
}

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

TEST_F(BuilderTest, OpHasTrait) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, 0, 0, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_TRUE(loom_op_has_trait(module_, addi, LOOM_TRAIT_PURE));
  EXPECT_TRUE(loom_op_has_trait(module_, addi, LOOM_TRAIT_COMMUTATIVE));
  EXPECT_FALSE(loom_op_has_trait(module_, addi, LOOM_TRAIT_TERMINATOR));
}

TEST_F(BuilderTest, OpOperandValue) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  // loom_op_operand_value resolves the value ID to the value struct.
  loom_value_t* operand_val = loom_op_operand_value(module_, neg, 0);
  EXPECT_EQ(operand_val, loom_module_value(module_, a));
  EXPECT_EQ(loom_type_element_type(operand_val->type), LOOM_SCALAR_TYPE_I32);
}

TEST_F(BuilderTest, OpResultValue) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  loom_value_id_t result_id = loom_test_neg_result(neg);
  loom_value_t* result_val = loom_op_result_value(module_, neg, 0);
  EXPECT_EQ(result_val, loom_module_value(module_, result_id));
}

TEST_F(BuilderTest, ValueSingleUse) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t a = build_constant(&builder_, module_, i32);

  // No uses: single_use returns NULL.
  EXPECT_EQ(loom_value_single_use(loom_module_value(module_, a)), nullptr);

  // One use: single_use returns it.
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  const loom_use_t* use = loom_value_single_use(loom_module_value(module_, a));
  ASSERT_NE(use, nullptr);
  EXPECT_EQ(loom_use_user_op(*use), neg);
  EXPECT_EQ(loom_use_operand_index(*use), 0);

  // Two uses: single_use returns NULL again.
  loom_op_t* neg2 = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg2));
  EXPECT_EQ(loom_value_single_use(loom_module_value(module_, a)), nullptr);
}

TEST_F(BuilderTest, OpVtable) {
  loom_op_t* addi = NULL;
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, 0, 0, i32, LOOM_LOCATION_UNKNOWN, &addi));
  const loom_op_vtable_t* vtable = loom_op_vtable(module_, addi);
  EXPECT_NE(vtable, nullptr);

  iree_string_view_t name = loom_op_name(module_, addi);
  EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("test.addi")));
}

//===----------------------------------------------------------------------===//
// Attribute equality and hashing
//===----------------------------------------------------------------------===//

TEST(AttributeEqual, InlineI64) {
  loom_attribute_t a = loom_attr_i64(42);
  loom_attribute_t b = loom_attr_i64(42);
  loom_attribute_t c = loom_attr_i64(99);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, InlineF64) {
  loom_attribute_t a = loom_attr_f64(3.14);
  loom_attribute_t b = loom_attr_f64(3.14);
  loom_attribute_t c = loom_attr_f64(2.71);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, InlineBool) {
  loom_attribute_t t = loom_attr_bool(true);
  loom_attribute_t t2 = loom_attr_bool(true);
  loom_attribute_t f = loom_attr_bool(false);
  EXPECT_TRUE(loom_attribute_equal(&t, &t2));
  EXPECT_FALSE(loom_attribute_equal(&t, &f));
}

TEST(AttributeEqual, InlineEnum) {
  loom_attribute_t a = loom_attr_enum(3);
  loom_attribute_t b = loom_attr_enum(3);
  loom_attribute_t c = loom_attr_enum(5);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, DifferentKinds) {
  loom_attribute_t i = loom_attr_i64(1);
  loom_attribute_t f = loom_attr_f64(1.0);
  EXPECT_FALSE(loom_attribute_equal(&i, &f));
}

TEST(AttributeEqual, I64ArraySameContent) {
  int64_t values_a[] = {1, 2, 3, 4};
  int64_t values_b[] = {1, 2, 3, 4};
  loom_attribute_t a = loom_attr_i64_array(values_a, 4);
  loom_attribute_t b = loom_attr_i64_array(values_b, 4);
  // Different pointers, same content — must be equal.
  EXPECT_NE(a.i64_array, b.i64_array);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArrayDifferentContent) {
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 4};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 3);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArrayDifferentCount) {
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 2);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArraySamePointer) {
  int64_t values[] = {10, 20};
  loom_attribute_t a = loom_attr_i64_array(values, 2);
  loom_attribute_t b = loom_attr_i64_array(values, 2);
  // Same pointer — fast path.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, PredicateListSameContent) {
  loom_predicate_t preds_a[2];
  memset(preds_a, 0, sizeof(preds_a));
  preds_a[0].kind = LOOM_PREDICATE_MUL;
  preds_a[0].arg_count = 2;
  preds_a[0].args[0] = 100;
  preds_a[0].args[1] = 16;
  preds_a[1].kind = LOOM_PREDICATE_MIN;
  preds_a[1].arg_count = 2;
  preds_a[1].args[0] = 100;
  preds_a[1].args[1] = 1;

  loom_predicate_t preds_b[2];
  memcpy(preds_b, preds_a, sizeof(preds_a));

  loom_attribute_t a = loom_attr_predicate_list(preds_a, 2);
  loom_attribute_t b = loom_attr_predicate_list(preds_b, 2);
  EXPECT_NE(a.predicate_list, b.predicate_list);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, PredicateListDifferentContent) {
  loom_predicate_t preds_a[1];
  memset(preds_a, 0, sizeof(preds_a));
  preds_a[0].kind = LOOM_PREDICATE_MUL;
  preds_a[0].arg_count = 2;
  preds_a[0].args[0] = 100;
  preds_a[0].args[1] = 16;

  loom_predicate_t preds_b[1];
  memcpy(preds_b, preds_a, sizeof(preds_a));
  preds_b[0].args[1] = 32;  // Different multiplier.

  loom_attribute_t a = loom_attr_predicate_list(preds_a, 1);
  loom_attribute_t b = loom_attr_predicate_list(preds_b, 1);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictSameContent) {
  loom_named_attr_t entries_a[2];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64(42);
  entries_a[1].name_id = 2;
  entries_a[1].value = loom_attr_bool(true);

  loom_named_attr_t entries_b[2];
  memcpy(entries_b, entries_a, sizeof(entries_a));

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 2);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 2);
  EXPECT_NE(a.dict_entries, b.dict_entries);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictDifferentName) {
  loom_named_attr_t entries_a[1];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64(42);

  loom_named_attr_t entries_b[1];
  memcpy(entries_b, entries_a, sizeof(entries_a));
  entries_b[0].name_id = 99;

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 1);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 1);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictWithNestedArray) {
  int64_t arr_a[] = {5, 10, 15};
  int64_t arr_b[] = {5, 10, 15};

  loom_named_attr_t entries_a[1];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64_array(arr_a, 3);

  loom_named_attr_t entries_b[1];
  memset(entries_b, 0, sizeof(entries_b));
  entries_b[0].name_id = 1;
  entries_b[0].value = loom_attr_i64_array(arr_b, 3);

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 1);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 1);
  // Nested array attribute with different pointers, same content.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeHash, EqualAttributesEqualHash) {
  // Inline.
  loom_attribute_t i1 = loom_attr_i64(42);
  loom_attribute_t i2 = loom_attr_i64(42);
  EXPECT_EQ(loom_attribute_hash(&i1), loom_attribute_hash(&i2));

  // I64 array with different pointers.
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 3};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 3);
  EXPECT_EQ(loom_attribute_hash(&a), loom_attribute_hash(&b));
}

TEST(AttributeHash, DifferentAttributesDifferentHash) {
  loom_attribute_t a = loom_attr_i64(42);
  loom_attribute_t b = loom_attr_i64(43);
  // Not guaranteed by spec, but extremely likely with FNV-1a.
  EXPECT_NE(loom_attribute_hash(&a), loom_attribute_hash(&b));

  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 4};
  loom_attribute_t arr_a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t arr_b = loom_attr_i64_array(values_b, 3);
  EXPECT_NE(loom_attribute_hash(&arr_a), loom_attribute_hash(&arr_b));
}

TEST(AttributeHash, DifferentKindsDifferentHash) {
  loom_attribute_t i = loom_attr_i64(0);
  loom_attribute_t b = loom_attr_bool(false);
  // Both have zero payload but different kinds.
  EXPECT_NE(loom_attribute_hash(&i), loom_attribute_hash(&b));
}

//===----------------------------------------------------------------------===//
// Attribute DICT depth limiting
//===----------------------------------------------------------------------===//

// Builds a chain of DICT attributes nested to |depth| levels.
// The innermost value is an i64(42). Each level wraps the previous
// in a single-entry DICT. Returns the outermost attribute.
// All loom_named_attr_t storage is allocated from |buffer|, which
// must have room for |depth| entries.
static loom_attribute_t make_nested_dict(loom_named_attr_t* buffer, int depth) {
  loom_attribute_t inner = loom_attr_i64(42);
  for (int i = 0; i < depth; ++i) {
    memset(&buffer[i], 0, sizeof(buffer[i]));
    buffer[i].name_id = (loom_string_id_t)(i + 1);
    buffer[i].value = inner;
    inner = loom_make_canonical_attr_dict(&buffer[i], 1);
  }
  return inner;
}

TEST(AttributeEqual, DeeplyNestedDictDoesNotCrash) {
  // Build two identical DICT chains deeper than the recursion limit.
  // The function must not stack-overflow; it should return false
  // (conservative) for the portion beyond the depth limit.
  static const int kDepth = 200;
  loom_named_attr_t buffer_a[kDepth];
  loom_named_attr_t buffer_b[kDepth];
  loom_attribute_t a = make_nested_dict(buffer_a, kDepth);
  loom_attribute_t b = make_nested_dict(buffer_b, kDepth);

  // Should not crash. The result is false because the comparison
  // bails at the depth limit rather than confirming equality.
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, ShallowNestedDictStillWorks) {
  // Nesting within the depth limit should compare correctly.
  loom_named_attr_t buffer_a[3];
  loom_named_attr_t buffer_b[3];
  loom_attribute_t a = make_nested_dict(buffer_a, 3);
  loom_attribute_t b = make_nested_dict(buffer_b, 3);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeHash, DeeplyNestedDictDoesNotCrash) {
  static const int kDepth = 200;
  loom_named_attr_t buffer[kDepth];
  loom_attribute_t attr = make_nested_dict(buffer, kDepth);

  // Should not crash. Returns some deterministic hash.
  uint32_t hash = loom_attribute_hash(&attr);
  EXPECT_EQ(hash, loom_attribute_hash(&attr));
}

TEST(AttributeHash, ShallowNestedDictConsistent) {
  loom_named_attr_t buffer_a[3];
  loom_named_attr_t buffer_b[3];
  loom_attribute_t a = make_nested_dict(buffer_a, 3);
  loom_attribute_t b = make_nested_dict(buffer_b, 3);

  // Within depth limit: equal attributes must have equal hashes.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_EQ(loom_attribute_hash(&a), loom_attribute_hash(&b));
}

TEST_F(BuilderTest, UpdateBuilderHasTiedResult) {
  loom_type_t tensor_f32 = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(256), 0);
  int64_t static_offsets[] = {0};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_update_build(&builder_, 1, 2, NULL, 0,
                                        static_offsets, 1, tensor_f32,
                                        LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->tied_result_count, 1);
  loom_tied_result_t* tied = loom_op_tied_results(op);
  EXPECT_EQ(tied[0].result_index, 0);
  EXPECT_EQ(tied[0].operand_index, 1);
  EXPECT_TRUE(tied[0].has_type_change);
}

TEST_F(BuilderTest, InvokeBuilderWithDynamicTied) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t operands[] = {5, 6};
  loom_type_t result_types[] = {f32, index_type};
  loom_tied_result_t tied[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false}};
  loom_symbol_ref_t callee = {0, 1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, operands, 2,
                                        result_types, 2, tied, 1,
                                        LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->tied_result_count, 1);
  loom_tied_result_t* read_tied = loom_op_tied_results(op);
  EXPECT_EQ(read_tied[0].result_index, 0);
  EXPECT_EQ(read_tied[0].operand_index, 0);
  EXPECT_FALSE(read_tied[0].has_type_change);
}

TEST_F(BuilderTest, InvokeBuilderNoTied) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t operands[] = {5};
  loom_type_t result_types[] = {f32};
  loom_symbol_ref_t callee = {0, 1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, operands, 1,
                                        result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->tied_result_count, 0);
}

}  // namespace
}  // namespace loom
