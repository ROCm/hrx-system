// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_vector_partition.h"

namespace loom {
namespace {

class ErasureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_VECTOR,
                                                 vtables, (uint16_t)count));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCALAR,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("erasure"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    loom_builder_set_block(&rewriter_.builder, loom_module_block(module_));
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* BuildIdentity(loom_value_id_t source) {
    const loom_type_t type = loom_module_value_type(module_, source);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_assume_build(&rewriter_.builder, &source, 1,
                                           nullptr, 0, &type, 1,
                                           LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  // Shared backing for source and rewrite allocations.
  iree_arena_block_pool_t pool_;
  // Rewrite worklist and temporary storage.
  iree_arena_allocator_t arena_;
  // Registered vocabulary of fixture operations.
  loom_context_t context_;
  // Source module owned by this fixture.
  loom_module_t* module_ = nullptr;
  // Consumer that maintains uses, effects, and the rewrite worklist.
  loom_rewriter_t rewriter_ = {};
};

TEST_F(ErasureTest, ClosedSetRetiresReferencesInProducerOrder) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* shared = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          type, LOOM_LOCATION_UNKNOWN,
                                          &shared));
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(
      loom_test_effectful_constant_build(&rewriter_.builder, loom_attr_i64(3),
                                         type, LOOM_LOCATION_UNKNOWN, &source));
  const loom_value_id_t shared_value = loom_test_constant_result(shared);
  const loom_value_id_t source_value =
      loom_test_effectful_constant_result(source);
  loom_op_t* consumer = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&rewriter_.builder, source_value,
                                      shared_value, type, LOOM_LOCATION_UNKNOWN,
                                      &consumer));
  loom_op_t* survivor = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&rewriter_.builder, shared_value,
                                      shared_value, type, LOOM_LOCATION_UNKNOWN,
                                      &survivor));
  EXPECT_TRUE(loom_region_has_observable_effects(module_->body));
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
  loom_op_t* consumed[] = {source, consumer};
  IREE_ASSERT_OK(loom_rewriter_erase_closed_set(&rewriter_, consumed,
                                                IREE_ARRAYSIZE(consumed)));
  EXPECT_TRUE(iree_any_bit_set(source->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(consumer->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(survivor->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(loom_module_value(module_, source_value)->use_count, 0u);
  EXPECT_EQ(loom_module_value(module_, shared_value)->use_count, 2u);
  EXPECT_FALSE(loom_region_has_observable_effects(module_->body));
  EXPECT_EQ(loom_rewriter_pop(&rewriter_), shared);
  EXPECT_EQ(loom_rewriter_pop(&rewriter_), nullptr);
  EXPECT_EQ(loom_test_addi_lhs(survivor), shared_value);
  EXPECT_EQ(loom_test_addi_rhs(survivor), shared_value);
}

TEST_F(ErasureTest, VectorMaterializationsFollowReplacementAndDeadProducers) {
  loom_value_fact_table_t facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &arena_, 0));
  loom_rewriter_attach_value_facts(&rewriter_, &facts);
  const loom_type_t scalar_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t pair_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(2), 0);
  const loom_type_t full_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_op_t* original = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          scalar_type, LOOM_LOCATION_UNKNOWN,
                                          &original));
  const loom_value_id_t original_value = loom_test_constant_result(original);
  const loom_value_id_t elements[] = {original_value, original_value};
  loom_op_t* pair = nullptr;
  IREE_ASSERT_OK(loom_vector_from_elements_build(&rewriter_.builder, elements,
                                                 2, pair_type,
                                                 LOOM_LOCATION_UNKNOWN, &pair));
  const loom_value_id_t pair_value = loom_vector_from_elements_result(pair);
  const loom_value_id_t parts[] = {pair_value, pair_value};
  loom_op_t* full = nullptr;
  IREE_ASSERT_OK(loom_vector_concat_build(&rewriter_.builder, 0, parts, 2,
                                          full_type, LOOM_LOCATION_UNKNOWN,
                                          &full));
  const loom_value_id_t full_value = loom_vector_concat_result(full);
  const auto before =
      loom_value_fact_table_query_vector_partition(&facts, full_value);
  ASSERT_EQ(before.count, 4u);
  ASSERT_EQ(before.snapshot_count, 2u);

  loom_builder_set_before(&rewriter_.builder, original);
  loom_op_t* replacement = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          scalar_type, LOOM_LOCATION_UNKNOWN,
                                          &replacement));
  const loom_value_id_t replacement_value =
      loom_test_constant_result(replacement);
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_and_erase(
      &rewriter_, original, &replacement_value, 1));
  // The outer concat is not an IR user of the replaced scalar. Its flattened
  // materialization still changes immediately, without draining a worklist.
  const auto after =
      loom_value_fact_table_query_vector_partition(&facts, full_value);
  ASSERT_EQ(after.count, 4u);
  for (uint32_t i = 0; i < after.count; ++i) {
    EXPECT_EQ(after.components[i].value, replacement_value);
  }

  loom_builder_set_before(&rewriter_.builder, pair);
  loom_op_t* replacement_pair = nullptr;
  IREE_ASSERT_OK(loom_vector_constant_build(
      &rewriter_.builder, loom_attr_i64(7), pair_type, LOOM_LOCATION_UNKNOWN,
      &replacement_pair));
  const loom_value_id_t replacement_pair_value =
      loom_vector_constant_result(replacement_pair);
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_and_erase(
      &rewriter_, pair, &replacement_pair_value, 1));
  bool erased = false;
  IREE_ASSERT_OK(loom_rewriter_erase_if_dead(&rewriter_, replacement, &erased));
  ASSERT_TRUE(erased);
  // DCE removes the old exact leaves, but the independently retained packet
  // snapshot names the live vector replacement and remains usable.
  const auto snapshot =
      loom_value_fact_table_query_vector_partition(&facts, full_value);
  EXPECT_EQ(snapshot.count, 0u);
  ASSERT_EQ(snapshot.snapshot_count, 2u);
  EXPECT_EQ(snapshot.snapshot_components[0].value, replacement_pair_value);
  EXPECT_EQ(snapshot.snapshot_components[1].value, replacement_pair_value);

  loom_op_t* consumed[] = {replacement_pair, full};
  IREE_ASSERT_OK(loom_rewriter_erase_closed_set(&rewriter_, consumed, 2));
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&facts, full_value)
                .snapshot_count,
            0u);
  loom_rewriter_attach_value_facts(&rewriter_, nullptr);
}

TEST_F(ErasureTest, TypeChangesWithdrawMaterializations) {
  loom_value_fact_table_t facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &arena_, 0));
  loom_rewriter_attach_value_facts(&rewriter_, &facts);
  loom_op_t* scalar = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &rewriter_.builder, loom_attr_i64(7),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &scalar));
  const loom_value_id_t scalar_value = loom_test_constant_result(scalar);
  const loom_value_id_t elements[] = {scalar_value, scalar_value};
  loom_op_t* vector = nullptr;
  IREE_ASSERT_OK(loom_vector_from_elements_build(
      &rewriter_.builder, elements, 2,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(2), 0),
      LOOM_LOCATION_UNKNOWN, &vector));
  const loom_value_id_t vector_value = loom_vector_from_elements_result(vector);
  ASSERT_EQ(
      loom_value_fact_table_query_vector_partition(&facts, vector_value).count,
      2u);

  IREE_ASSERT_OK(loom_rewriter_set_value_type(
      &rewriter_, scalar_value, loom_type_scalar(LOOM_SCALAR_TYPE_I64)));
  EXPECT_EQ(
      loom_value_fact_table_query_vector_partition(&facts, vector_value).count,
      0u);
  IREE_ASSERT_OK(loom_rewriter_set_value_type(
      &rewriter_, vector_value,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64,
                          loom_dim_pack_static(2), 0)));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&facts, module_,
                                                             vector, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(
      loom_value_fact_table_query_vector_partition(&facts, vector_value).count,
      2u);
  loom_rewriter_attach_value_facts(&rewriter_, nullptr);
}

TEST_F(ErasureTest, RetainedIdentitiesFollowReplacementBeforeInference) {
  loom_value_fact_table_t facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &arena_, 0));
  loom_rewriter_attach_value_facts(&rewriter_, &facts);
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* original = nullptr;
  loom_op_t* replacement = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          type, LOOM_LOCATION_UNKNOWN,
                                          &original));
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          type, LOOM_LOCATION_UNKNOWN,
                                          &replacement));
  const loom_value_id_t original_value = loom_test_constant_result(original);
  const loom_value_id_t replacement_value =
      loom_test_constant_result(replacement);
  loom_op_t* direct = BuildIdentity(original_value);
  loom_op_t* indirect = BuildIdentity(loom_op_const_results(direct)[0]);
  const loom_value_id_t indirect_value = loom_op_const_results(indirect)[0];
  ASSERT_EQ(loom_rewriter_value_identity(&rewriter_, indirect_value),
            original_value);

  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_with(&rewriter_, original_value,
                                                     replacement_value));
  EXPECT_EQ(loom_rewriter_value_identity(&rewriter_, indirect_value),
            replacement_value);
  // Replacing uses does not erase or redefine the original SSA value. An
  // identity subsequently created from it belongs to a separate publication.
  loom_op_t* subsequent = BuildIdentity(original_value);
  const loom_value_id_t subsequent_value = loom_op_const_results(subsequent)[0];
  EXPECT_EQ(loom_rewriter_value_identity(&rewriter_, subsequent_value),
            original_value);
  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, subsequent));
  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, original));
  EXPECT_EQ(loom_rewriter_value_identity(&rewriter_, indirect_value),
            replacement_value);
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&facts, module_,
                                                             direct, &changed));
  EXPECT_TRUE(changed);
  changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
      &facts, module_, indirect, &changed));
  EXPECT_TRUE(changed);
  loom_rewriter_attach_value_facts(&rewriter_, nullptr);
}

TEST_F(ErasureTest, PartialMutationWithdrawsErasedIdentityBeforeInference) {
  loom_value_fact_table_t facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &arena_, 0));
  loom_rewriter_attach_value_facts(&rewriter_, &facts);
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* original = nullptr;
  loom_op_t* replacement = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          type, LOOM_LOCATION_UNKNOWN,
                                          &original));
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter_.builder, loom_attr_i64(7),
                                          type, LOOM_LOCATION_UNKNOWN,
                                          &replacement));
  const loom_value_id_t original_value = loom_test_constant_result(original);
  const loom_value_id_t replacement_value =
      loom_test_constant_result(replacement);
  loom_op_t* direct = BuildIdentity(original_value);
  loom_op_t* indirect = BuildIdentity(loom_op_const_results(direct)[0]);
  const loom_value_id_t indirect_value = loom_op_const_results(indirect)[0];
  IREE_ASSERT_OK(
      loom_rewriter_set_operand(&rewriter_, direct, 0, replacement_value));
  // The downstream equation is pending when DCE removes its former source.
  // Withdrawing that source makes the downstream value its own identity until
  // normal inference publishes the replacement, without scanning its users.
  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter_, original));
  EXPECT_EQ(loom_rewriter_value_identity(&rewriter_, indirect_value),
            indirect_value);
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
      &facts, module_, indirect, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_rewriter_value_identity(&rewriter_, indirect_value),
            replacement_value);
  loom_rewriter_attach_value_facts(&rewriter_, nullptr);
}

}  // namespace
}  // namespace loom
