// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/reference_transfer.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ReferenceTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("references"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 16));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Argument(loom_type_t type) {
    loom_value_id_t value;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &value));
    IREE_CHECK_OK(
        loom_block_add_arg(module_, loom_module_block(module_), value));
    return value;
  }

  loom_value_facts_t Buffer(loom_value_id_t value,
                            loom_value_fact_reference_origin_t origin) {
    loom_value_fact_buffer_reference_t reference = {};
    reference.maximum_byte_extent = loom_value_facts_exact_i64(256);
    reference.minimum_alignment = 64;
    reference.root_value_id = value;
    reference.alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    reference.origin = origin;
    loom_value_facts_t facts;
    IREE_CHECK_OK(loom_value_facts_make_buffer_reference(&facts_.context,
                                                         reference, &facts));
    IREE_CHECK_OK(loom_value_fact_table_define(&facts_, value, facts));
    return facts;
  }

  // Pool and arena owning the module and reference extensions.
  iree_arena_block_pool_t pool_;
  // Arena retaining fact tables throughout each test.
  iree_arena_allocator_t arena_;
  // Type context for the module's argument values.
  loom_context_t context_;
  // Owned module carrying stable SSA argument identities.
  loom_module_t* module_ = nullptr;
  // Caller fact table used for actual-argument substitution.
  loom_value_fact_table_t facts_ = {};
};

TEST_F(ReferenceTransferTest, RecursiveArgumentsUseCallerProvenance) {
  const auto input = Argument(loom_type_buffer());
  const auto scratch = Argument(loom_type_buffer());
  const loom_value_fact_reference_origin_t entry = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, input};
  const loom_value_fact_reference_origin_t allocation = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION, LOOM_VALUE_ID_INVALID};
  Buffer(input, entry);
  Buffer(scratch, allocation);
  // A recursive invocation receives local scratch as its first argument.
  loom_value_id_t arguments[] = {scratch, input};
  loom_reference_call_t call;
  loom_reference_call_initialize(module_, &facts_, {arguments, 2}, entry,
                                 &call);
  const auto returned = loom_reference_call_result_origin(&call, entry);
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(returned, allocation));
  EXPECT_FALSE(
      loom_value_fact_reference_origins_are_disjoint(returned, allocation));
  auto second = entry;
  second.entry_value_id = scratch;
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(
      loom_reference_call_result_origin(&call, second), entry));
  auto selected = entry;
  selected.entry_value_id = LOOM_VALUE_ID_INVALID;
  EXPECT_EQ(loom_reference_call_result_origin(&call, selected).kind,
            LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN);
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(
      loom_reference_call_result_origin(&call, allocation), allocation));
}

TEST_F(ReferenceTransferTest, StableArgumentIdentitySurvivesPruning) {
  Argument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const auto input = Argument(loom_type_buffer());
  const loom_value_fact_reference_origin_t entry = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, input};
  Buffer(input, entry);
  IREE_ASSERT_OK(loom_block_remove_arg(module_, loom_module_block(module_), 0));
  loom_value_id_t arguments[] = {input};
  loom_reference_call_t call;
  loom_reference_call_initialize(module_, &facts_, {arguments, 1}, entry,
                                 &call);
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(
      loom_reference_call_result_origin(&call, entry), entry));
}

TEST_F(ReferenceTransferTest, JoinedInputsRetainCommonOrigin) {
  const auto first = Argument(loom_type_buffer());
  const auto second = Argument(loom_type_buffer());
  const auto condition = Argument(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_fact_reference_origin_t entry = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, first};
  Buffer(first, entry);
  auto second_entry = entry;
  second_entry.entry_value_id = second;
  Buffer(second, second_entry);
  loom_value_id_t arguments[] = {first, second, condition};
  loom_reference_call_t call;
  loom_reference_call_initialize(module_, &facts_, {arguments, 3}, entry,
                                 &call);
  auto selected = entry;
  selected.entry_value_id = LOOM_VALUE_ID_INVALID;
  const auto returned = loom_reference_call_result_origin(&call, selected);
  EXPECT_EQ(returned.kind, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY);
  EXPECT_EQ(returned.entry_value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(loom_value_fact_reference_origins_are_disjoint(
      returned, call.allocation_origin));
  Buffer(second, {});
  loom_reference_call_initialize(module_, &facts_, {arguments, 3}, entry,
                                 &call);
  EXPECT_EQ(loom_reference_call_result_origin(&call, selected).kind,
            LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN);
}

TEST_F(ReferenceTransferTest, RebindingPreservesOtherFactsAndInternsOrigins) {
  const auto value = Argument(loom_type_buffer());
  const loom_value_fact_reference_origin_t entry = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, value};
  auto source = Buffer(value, entry);
  loom_value_facts_mark_workgroup_uniform(&source);
  loom_value_fact_table_t target;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 16));
  auto allocation = entry;
  allocation.kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION;
  allocation.entry_value_id = LOOM_VALUE_ID_INVALID;
  auto rebound = source;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&target, &facts_, source, &rebound));
  IREE_ASSERT_OK(loom_value_facts_rebind_reference_origin(
      &target.context, allocation, &rebound));
  loom_value_fact_buffer_reference_t reference;
  ASSERT_TRUE(loom_value_facts_query_buffer_reference(&target.context, rebound,
                                                      &reference));
  EXPECT_EQ(reference.root_value_id, value);
  EXPECT_EQ(reference.maximum_byte_extent.range_lo, 256);
  EXPECT_EQ(reference.minimum_alignment, 64);
  EXPECT_EQ(reference.origin.kind, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION);
  auto scalar_source = source;
  auto scalar_rebound = rebound;
  scalar_source.extension_id = scalar_rebound.extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  EXPECT_TRUE(loom_value_facts_equal(scalar_source, scalar_rebound));
  EXPECT_FALSE(
      loom_value_fact_table_facts_equal(&facts_, source, &target, rebound));
  auto cloned = source;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&target, &facts_, source, &cloned));
  EXPECT_TRUE(
      loom_value_fact_table_facts_equal(&facts_, source, &target, cloned));
  EXPECT_NE(cloned.extension_id, rebound.extension_id);
  IREE_ASSERT_OK(loom_value_facts_rebind_reference_origin(&target.context,
                                                          entry, &rebound));
  EXPECT_EQ(rebound.extension_id, cloned.extension_id);
}

TEST_F(ReferenceTransferTest, ViewOriginsPreserveCoordinatesAcrossTables) {
  const auto buffer = Argument(loom_type_buffer());
  loom_value_fact_view_reference_t source_reference = {};
  source_reference.base_byte_offset = loom_value_facts_exact_i64(16);
  source_reference.footprint_byte_length = loom_value_facts_exact_i64(32);
  source_reference.minimum_alignment = 16;
  source_reference.root_minimum_alignment = 64;
  source_reference.static_element_byte_count = 4;
  source_reference.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  source_reference.root_value_id = buffer;
  source_reference.buffer_value_id = buffer;
  source_reference.alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
  source_reference.address_bitwidth = 32;
  source_reference.origin = {1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
                             buffer};
  loom_value_facts_t source;
  IREE_ASSERT_OK(loom_value_facts_make_view_reference(
      &facts_.context, source_reference, &source));
  loom_value_fact_table_t target;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 16));
  const loom_value_fact_reference_origin_t allocation = {
      2, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION, LOOM_VALUE_ID_INVALID};
  auto rebound = source;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&target, &facts_, source, &rebound));
  IREE_ASSERT_OK(loom_value_facts_rebind_reference_origin(
      &target.context, allocation, &rebound));
  loom_value_fact_view_reference_t reference;
  ASSERT_TRUE(loom_value_facts_query_view_reference(&target.context, rebound,
                                                    &reference));
  EXPECT_EQ(reference.root_value_id, buffer);
  EXPECT_EQ(reference.buffer_value_id, buffer);
  EXPECT_EQ(reference.base_byte_offset.range_lo, 16);
  EXPECT_EQ(reference.footprint_byte_length.range_hi, 32);
  EXPECT_EQ(reference.minimum_alignment, 16);
  EXPECT_EQ(reference.root_minimum_alignment, 64);
  EXPECT_EQ(reference.static_element_byte_count, 4);
  EXPECT_EQ(reference.memory_space, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(reference.address_bitwidth, 32);
  EXPECT_TRUE(
      loom_value_fact_reference_origin_equal(reference.origin, allocation));
  EXPECT_FALSE(
      loom_value_fact_table_facts_equal(&facts_, source, &target, rebound));
  loom_value_facts_t cloned;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&target, &facts_, source, &cloned));
  EXPECT_TRUE(
      loom_value_fact_table_facts_equal(&facts_, source, &target, cloned));
  EXPECT_NE(cloned.extension_id, rebound.extension_id);
  IREE_ASSERT_OK(loom_value_facts_rebind_reference_origin(
      &target.context, source_reference.origin, &rebound));
  EXPECT_EQ(rebound.extension_id, cloned.extension_id);
}

}  // namespace
}  // namespace loom
