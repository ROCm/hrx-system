// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

class EncodingFactDomainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &lhs_arena_);
    iree_arena_initialize(&block_pool_, &rhs_arena_);
    iree_arena_initialize(&block_pool_, &target_arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&lhs_, &lhs_arena_, 0));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&rhs_, &rhs_arena_, 0));
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&target_, &target_arena_, 0));
    loom_type_registry_configure_fact_context(&lhs_.context);
    loom_type_registry_configure_fact_context(&rhs_.context);
    loom_type_registry_configure_fact_context(&target_.context);
  }

  void TearDown() override {
    iree_arena_deinitialize(&target_arena_);
    iree_arena_deinitialize(&rhs_arena_);
    iree_arena_deinitialize(&lhs_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t lhs_arena_;
  iree_arena_allocator_t rhs_arena_;
  iree_arena_allocator_t target_arena_;
  loom_value_fact_table_t lhs_ = {};
  loom_value_fact_table_t rhs_ = {};
  loom_value_fact_table_t target_ = {};
};

static iree_status_t MakeEncodingSummary(
    loom_value_fact_table_t* table, loom_encoding_role_t role,
    uint16_t static_spec_encoding_id,
    loom_value_fact_address_layout_kind_t layout_kind,
    const loom_value_facts_t* strides, uint8_t rank,
    loom_value_fact_storage_schema_t storage_schema,
    loom_value_facts_t* out_facts) {
  loom_value_fact_encoding_summary_t summary = {
      .role = role,
      .static_spec_encoding_id = static_spec_encoding_id,
      .address_layout =
          {
              .kind = layout_kind,
              .rank = rank,
              .strides = strides,
          },
      .storage_schema = storage_schema,
  };
  return loom_value_facts_make_encoding_summary(&table->context, summary,
                                                out_facts);
}

TEST_F(EncodingFactDomainTest, RegistryOwnsEncodingFacts) {
  const loom_type_t type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(loom_type_registry_resolve_fact_domain(nullptr, &target_.context,
                                                   nullptr, type),
            &loom_encoding_fact_domain);
}

TEST_F(EncodingFactDomainTest, MeetsCompatibleStridedLayoutsPerAxis) {
  const loom_value_facts_t lhs_strides[] = {loom_value_facts_exact_i64(4),
                                            loom_value_facts_exact_i64(1)};
  const loom_value_facts_t rhs_strides[] = {loom_value_facts_exact_i64(1),
                                            loom_value_facts_exact_i64(64)};
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     7, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     lhs_strides, IREE_ARRAYSIZE(lhs_strides),
                                     {}, &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     9, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     rhs_strides, IREE_ARRAYSIZE(rhs_strides),
                                     {}, &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  EXPECT_EQ(result.role, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(result.static_spec_encoding_id, 0);
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED);
  ASSERT_EQ(result.address_layout.rank, 2);
  EXPECT_EQ(result.address_layout.strides[0].range_lo, 1);
  EXPECT_EQ(result.address_layout.strides[0].range_hi, 4);
  EXPECT_EQ(result.address_layout.strides[1].range_lo, 1);
  EXPECT_EQ(result.address_layout.strides[1].range_hi, 64);
}

TEST_F(EncodingFactDomainTest, MeetsDynamicAndExactStrideFacts) {
  const loom_value_facts_t lhs_strides[] = {loom_value_facts_make(8, 16, 8),
                                            loom_value_facts_exact_i64(1)};
  const loom_value_facts_t rhs_strides[] = {loom_value_facts_exact_i64(32),
                                            loom_value_facts_exact_i64(1)};
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     lhs_strides, IREE_ARRAYSIZE(lhs_strides),
                                     {}, &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     rhs_strides, IREE_ARRAYSIZE(rhs_strides),
                                     {}, &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  ASSERT_EQ(result.address_layout.rank, 2);
  EXPECT_EQ(result.address_layout.strides[0].range_lo, 8);
  EXPECT_EQ(result.address_layout.strides[0].range_hi, 32);
  EXPECT_EQ(result.address_layout.strides[0].known_divisor, 8);
  int64_t unit_stride = 0;
  EXPECT_TRUE(loom_value_facts_as_exact_i64(result.address_layout.strides[1],
                                            &unit_stride));
  EXPECT_EQ(unit_stride, 1);
}

TEST_F(EncodingFactDomainTest, RetainsDenseKindWithoutExactIdentity) {
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     3, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
                                     nullptr, 0, {}, &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     4, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
                                     nullptr, 0, {}, &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  EXPECT_EQ(result.static_spec_encoding_id, 0);
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);
}

TEST_F(EncodingFactDomainTest, DropsIncompatibleLayoutShape) {
  const loom_value_facts_t rank_one[] = {loom_value_facts_exact_i64(1)};
  const loom_value_facts_t rank_two[] = {loom_value_facts_exact_i64(4),
                                         loom_value_facts_exact_i64(1)};
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     rank_one, IREE_ARRAYSIZE(rank_one), {},
                                     &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     rank_two, IREE_ARRAYSIZE(rank_two), {},
                                     &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  EXPECT_EQ(result.role, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN);
}

TEST_F(EncodingFactDomainTest, DropsSummariesWithDifferentUntypedRoles) {
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
                                     nullptr, 0, {}, &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN,
                                     nullptr, 0, {}, &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_UNKNOWN), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));
  EXPECT_EQ(result_facts.extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
}

TEST_F(EncodingFactDomainTest, RetainsCommonStorageSchemaFacts) {
  loom_value_fact_storage_schema_t schema = {};
  schema.static_spec_encoding_id = 5;
  schema.encoded_operand.payload_element_count = 32;
  loom_value_facts_t lhs_facts;
  loom_value_facts_t rhs_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(&lhs_, LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
                                     7, LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN,
                                     nullptr, 0, schema, &lhs_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
                                     9, LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN,
                                     nullptr, 0, schema, &rhs_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA), &lhs_,
      lhs_facts, &rhs_, rhs_facts, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  EXPECT_EQ(result.static_spec_encoding_id, 0);
  EXPECT_EQ(result.storage_schema.static_spec_encoding_id, 5);
  EXPECT_EQ(result.storage_schema.encoded_operand.payload_element_count, 32);
}

TEST_F(EncodingFactDomainTest, WideningRetainsStridedDescriptorShape) {
  const loom_value_facts_t previous_strides[] = {loom_value_facts_exact_i64(4),
                                                 loom_value_facts_exact_i64(1)};
  const loom_value_facts_t next_strides[] = {loom_value_facts_exact_i64(8),
                                             loom_value_facts_exact_i64(1)};
  loom_value_facts_t previous_facts;
  loom_value_facts_t next_facts;
  IREE_ASSERT_OK(MakeEncodingSummary(
      &lhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, 0,
      LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED, previous_strides,
      IREE_ARRAYSIZE(previous_strides), {}, &previous_facts));
  IREE_ASSERT_OK(MakeEncodingSummary(&rhs_, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
                                     0, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                                     next_strides, IREE_ARRAYSIZE(next_strides),
                                     {}, &next_facts));

  loom_value_facts_t result_facts;
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &target_, nullptr,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &lhs_,
      previous_facts, &rhs_, next_facts, 2, &result_facts));

  loom_value_fact_encoding_summary_t result = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target_.context,
                                                      result_facts, &result));
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED);
  ASSERT_EQ(result.address_layout.rank, 2);
  EXPECT_EQ(result.address_layout.strides[0].range_lo, INT64_MIN);
  EXPECT_EQ(result.address_layout.strides[0].range_hi, INT64_MAX);
  EXPECT_EQ(result.address_layout.strides[0].known_divisor, 4);
  int64_t unit_stride = 0;
  EXPECT_TRUE(loom_value_facts_as_exact_i64(result.address_layout.strides[1],
                                            &unit_stride));
  EXPECT_EQ(unit_stride, 1);
}

}  // namespace
}  // namespace loom
