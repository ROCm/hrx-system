// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <cstdint>
#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class FactTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

static constexpr uint8_t kTestRawPayloadTag = 42;

static const loom_value_fact_domain_t* FactTableTestResolveDomain(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type) {
  (void)context;
  (void)module;
  (void)type;
  return static_cast<const loom_value_fact_domain_t*>(user_data);
}

static bool FactTableTestRawExtensionsEqual(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs) {
  (void)domain;
  (void)module;
  (void)type;
  const void* lhs_payload = nullptr;
  const void* rhs_payload = nullptr;
  iree_host_size_t lhs_length = 0;
  iree_host_size_t rhs_length = 0;
  bool lhs_has = loom_value_facts_query_extension_payload(
      lhs_table ? &lhs_table->context : nullptr, lhs, kTestRawPayloadTag,
      &lhs_payload, &lhs_length);
  bool rhs_has = loom_value_facts_query_extension_payload(
      rhs_table ? &rhs_table->context : nullptr, rhs, kTestRawPayloadTag,
      &rhs_payload, &rhs_length);
  if (!lhs_has || !rhs_has) {
    return lhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE &&
           rhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  }
  return lhs_length == rhs_length &&
         std::memcmp(lhs_payload, rhs_payload, lhs_length) == 0;
}

static iree_status_t FactTableTestRawCloneExtension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* source, loom_value_facts_t facts,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  const void* payload = nullptr;
  iree_host_size_t payload_length = 0;
  if (!loom_value_facts_query_extension_payload(&source->context, facts,
                                                kTestRawPayloadTag, &payload,
                                                &payload_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source fact has no test raw payload");
  }
  loom_value_facts_t cloned_extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_extension_payload(
      &target->context, kTestRawPayloadTag, payload, payload_length,
      &cloned_extension));
  inout_facts->extension_id = cloned_extension.extension_id;
  return iree_ok_status();
}

static iree_status_t FactTableTestRawMeetExtension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs,
    loom_value_facts_t* inout_facts) {
  if (!FactTableTestRawExtensionsEqual(domain, module, type, lhs_table, lhs,
                                       rhs_table, rhs)) {
    return iree_ok_status();
  }
  if (lhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return iree_ok_status();
  }
  loom_value_facts_t cloned_extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(FactTableTestRawCloneExtension(
      domain, module, type, target, lhs_table, lhs, &cloned_extension));
  inout_facts->extension_id = cloned_extension.extension_id;
  return iree_ok_status();
}

static iree_status_t FactTableTestRawWidenExtension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    uint32_t iteration, loom_value_facts_t* inout_facts) {
  (void)iteration;
  return FactTableTestRawMeetExtension(domain, module, type, target,
                                       previous_table, previous, next_table,
                                       next, inout_facts);
}

static const loom_value_fact_domain_t kTestRawFactDomain = {
    /*.extensions_equal=*/FactTableTestRawExtensionsEqual,
    /*.clone_extension=*/FactTableTestRawCloneExtension,
    /*.meet_extension=*/FactTableTestRawMeetExtension,
    /*.widen_extension=*/FactTableTestRawWidenExtension,
};

//===----------------------------------------------------------------------===//
// Initialize and lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, ZeroInitIsValid) {
  loom_value_fact_table_t table = {0};
  // Lookup on empty table returns unknown.
  loom_value_facts_t facts = loom_value_fact_table_lookup(&table, 0);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  facts = loom_value_fact_table_lookup(&table, 42);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
}

TEST_F(FactTableTest, InitializePreallocates) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));
  EXPECT_EQ(table.capacity, (iree_host_size_t)100);
  // All entries are undefined (unknown).
  loom_value_facts_t facts = loom_value_fact_table_lookup(&table, 0);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  facts = loom_value_fact_table_lookup(&table, 99);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
}

//===----------------------------------------------------------------------===//
// Define and lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, DefineAndLookup) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  loom_value_facts_t exact_42 = loom_value_facts_exact_i64(42);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 5, exact_42));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 5);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 42);
  EXPECT_EQ(result.range_hi, 42);
}

TEST_F(FactTableTest, UndefinedEntriesAreUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  // Define entry 5, check that entry 3 is still unknown.
  loom_value_facts_t exact_42 = loom_value_facts_exact_i64(42);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 5, exact_42));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 3);
  EXPECT_TRUE(loom_value_facts_is_unknown(result));
}

TEST_F(FactTableTest, DefineUpdatesExisting) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(42)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(99)));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 5);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 99);
}

TEST_F(FactTableTest, ClearScopeClearsOnlyTouchedEntriesAndKeepsStorage) {
  iree_arena_allocator_t transient_arena = {};
  iree_arena_initialize(&block_pool_, &transient_arena);

  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
      &table, &arena_, &transient_arena, 100));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(42)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 8, loom_value_facts_exact_i64(99)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 8, loom_value_facts_exact_i64(7)));

  loom_value_facts_t* const entries = table.entries;
  loom_value_id_t* const touched_values = table.touched_values;
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table, 5)));
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table, 8)));

  loom_value_fact_table_clear_scope(&table);
  iree_arena_reset(&transient_arena);

  EXPECT_EQ(table.entries, entries);
  EXPECT_EQ(table.touched_values, touched_values);
  EXPECT_EQ(table.touched_count, 0u);
  EXPECT_EQ(table.count, 0u);
  EXPECT_TRUE(
      loom_value_facts_is_unknown(loom_value_fact_table_lookup(&table, 5)));
  EXPECT_TRUE(
      loom_value_facts_is_unknown(loom_value_fact_table_lookup(&table, 8)));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(123)));
  EXPECT_EQ(table.touched_count, 1u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 5).range_lo, 123);

  iree_arena_deinitialize(&transient_arena);
}

//===----------------------------------------------------------------------===//
// Capacity
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, DefineWithinCapacityPreservesExistingEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 101));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 0, loom_value_facts_exact_i64(10)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_exact_i64(20)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 100,
                                              loom_value_facts_exact_i64(30)));

  // Original entries are preserved.
  loom_value_facts_t r0 = loom_value_fact_table_lookup(&table, 0);
  EXPECT_EQ(r0.range_lo, 10);
  loom_value_facts_t r1 = loom_value_fact_table_lookup(&table, 1);
  EXPECT_EQ(r1.range_lo, 20);
  loom_value_facts_t r100 = loom_value_fact_table_lookup(&table, 100);
  EXPECT_EQ(r100.range_lo, 30);
}

TEST_F(FactTableTest, DefineBeyondCapacityGrowsAndPreservesEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 0, loom_value_facts_exact_i64(10)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 7, loom_value_facts_exact_i64(70)));

  EXPECT_GE(table.capacity, (iree_host_size_t)8);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 0).range_lo, 10);
  EXPECT_TRUE(
      loom_value_facts_is_unknown(loom_value_fact_table_lookup(&table, 6)));
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 7).range_lo, 70);
}

//===----------------------------------------------------------------------===//
// Range facts
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, RangeFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  loom_value_facts_t range = loom_value_facts_make(0, 1024, 64);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 7, range));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 7);
  EXPECT_FALSE(loom_value_facts_is_exact(result));
  EXPECT_TRUE(loom_value_facts_is_non_negative(result));
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 1024);
  EXPECT_EQ(result.known_divisor, 64);
  EXPECT_TRUE(loom_value_facts_divisible_by(result, 16));
}

//===----------------------------------------------------------------------===//
// Out-of-range lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, OutOfRangeIsUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 10));

  // Value ID beyond count is unknown.
  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 999);
  EXPECT_TRUE(loom_value_facts_is_unknown(result));
}

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, UniformElementExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(42), &facts));

  EXPECT_NE(facts.extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
  EXPECT_FALSE(loom_value_facts_is_unknown(facts));

  loom_value_fact_uniform_element_t uniform = {0};
  EXPECT_TRUE(
      loom_value_facts_query_uniform_element(&table.context, facts, &uniform));
  EXPECT_TRUE(loom_value_facts_is_exact(uniform.element));
  EXPECT_EQ(uniform.element.range_lo, 42);
}

TEST_F(FactTableTest, IdenticalExtensionsInternToSameId) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_t rhs = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &lhs));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &rhs));

  EXPECT_EQ(lhs.extension_id, rhs.extension_id);
  EXPECT_TRUE(loom_value_facts_equal(lhs, rhs));
}

TEST_F(FactTableTest, DifferentExtensionsDoNotAlias) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_t rhs = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &lhs));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(8), &rhs));

  EXPECT_NE(lhs.extension_id, rhs.extension_id);
  EXPECT_FALSE(loom_value_facts_equal(lhs, rhs));
}

TEST_F(FactTableTest, SmallStaticLanesExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[] = {
      loom_value_facts_exact_i64(1),
      loom_value_facts_make(2, 4, 2),
      loom_value_facts_unknown(),
  };
  loom_value_fact_small_static_lanes_t lane_slice = {
      /*.lanes=*/lanes,
      /*.count=*/IREE_ARRAYSIZE(lanes),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(&table.context,
                                                          lane_slice, &facts));

  loom_value_fact_small_static_lanes_t result = {};
  EXPECT_TRUE(loom_value_facts_query_small_static_lanes(&table.context, facts,
                                                        &result));
  EXPECT_EQ(result.count, IREE_ARRAYSIZE(lanes));
  EXPECT_NE(result.lanes, lanes);
  EXPECT_EQ(result.lanes[0].range_lo, 1);
  EXPECT_EQ(result.lanes[1].range_hi, 4);
  EXPECT_TRUE(loom_value_facts_is_unknown(result.lanes[2]));
}

TEST_F(FactTableTest, OversizedSmallStaticLanesDegradesToUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT + 1] = {};
  loom_value_fact_small_static_lanes_t lane_slice = {
      /*.lanes=*/lanes,
      /*.count=*/IREE_ARRAYSIZE(lanes),
  };
  loom_value_facts_t facts = loom_value_facts_exact_i64(123);
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(&table.context,
                                                          lane_slice, &facts));

  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  EXPECT_FALSE(
      loom_value_facts_query_small_static_lanes(&table.context, facts, NULL));
}

TEST_F(FactTableTest, VectorIotaExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_vector_iota_t iota = {
      /*.base=*/loom_value_facts_exact_i64(2),
      /*.step=*/loom_value_facts_exact_i64(3),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_iota(&table.context, iota, &facts));

  EXPECT_FALSE(
      loom_value_facts_query_uniform_element(&table.context, facts, NULL));
  loom_value_fact_vector_iota_t result = {};
  EXPECT_TRUE(
      loom_value_facts_query_vector_iota(&table.context, facts, &result));
  EXPECT_EQ(result.base.range_lo, 2);
  EXPECT_EQ(result.step.range_lo, 3);
}

TEST_F(FactTableTest, VectorPrefixMaskExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_vector_prefix_mask_t mask = {
      /*.lower_bound=*/loom_value_facts_exact_i64(0),
      /*.upper_bound=*/loom_value_facts_make(0, 16, 1),
      /*.step=*/loom_value_facts_exact_i64(1),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_prefix_mask(&table.context, mask, &facts));

  loom_value_fact_vector_prefix_mask_t result = {};
  EXPECT_TRUE(loom_value_facts_query_vector_prefix_mask(&table.context, facts,
                                                        &result));
  EXPECT_EQ(result.lower_bound.range_lo, 0);
  EXPECT_EQ(result.upper_bound.range_hi, 16);
  EXPECT_EQ(result.step.range_lo, 1);
}

TEST_F(FactTableTest, EncodingSummaryDenseLayoutRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_encoding_summary_t summary = {
      /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      /*.static_spec_encoding_id=*/0,
      /*.address_layout=*/
      {
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
          /*.rank=*/0,
          /*.strides=*/nullptr,
      },
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_encoding_summary(&table.context, summary, &facts));

  loom_value_fact_encoding_summary_t result = {};
  EXPECT_TRUE(
      loom_value_facts_query_encoding_summary(&table.context, facts, &result));
  EXPECT_EQ(result.role, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);
}

TEST_F(FactTableTest, EncodingSummaryStridedLayoutInternsStrideFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t strides[] = {
      loom_value_facts_make(8, 16, 8),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_encoding_summary_t summary = {
      /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      /*.static_spec_encoding_id=*/0,
      /*.address_layout=*/
      {
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
          /*.rank=*/IREE_ARRAYSIZE(strides),
          /*.strides=*/strides,
      },
  };
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_t rhs = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_encoding_summary(&table.context, summary, &lhs));
  IREE_ASSERT_OK(
      loom_value_facts_make_encoding_summary(&table.context, summary, &rhs));

  EXPECT_EQ(lhs.extension_id, rhs.extension_id);
  loom_value_fact_encoding_summary_t result = {};
  EXPECT_TRUE(
      loom_value_facts_query_encoding_summary(&table.context, lhs, &result));
  EXPECT_EQ(result.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED);
  EXPECT_EQ(result.address_layout.rank, IREE_ARRAYSIZE(strides));
  EXPECT_NE(result.address_layout.strides, strides);
  EXPECT_EQ(result.address_layout.strides[0].range_lo, 8);
  EXPECT_EQ(result.address_layout.strides[0].range_hi, 16);
  EXPECT_EQ(result.address_layout.strides[1].range_lo, 1);
}

TEST_F(FactTableTest, CloneDefinedFactsReinternsExtensions) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  loom_value_facts_t strides[] = {
      loom_value_facts_make(16, 64, 16),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_encoding_summary_t summary = {
      /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      /*.static_spec_encoding_id=*/0,
      /*.address_layout=*/
      {
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
          /*.rank=*/IREE_ARRAYSIZE(strides),
          /*.strides=*/strides,
      },
  };
  loom_value_facts_t source_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(
      &source.context, summary, &source_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, 7, source_facts));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_defined_facts(&target, &source, nullptr));

  loom_value_facts_t cloned_facts = loom_value_fact_table_lookup(&target, 7);
  EXPECT_NE(cloned_facts.extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
  EXPECT_TRUE(loom_value_facts_query_encoding_summary(&target.context,
                                                      cloned_facts, &summary));
  EXPECT_EQ(summary.address_layout.kind,
            LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED);
  EXPECT_EQ(summary.address_layout.rank, IREE_ARRAYSIZE(strides));
  EXPECT_NE(summary.address_layout.strides, strides);
  EXPECT_EQ(summary.address_layout.strides[0].range_lo, 16);
  EXPECT_EQ(summary.address_layout.strides[0].range_hi, 64);
  EXPECT_EQ(summary.address_layout.strides[1].range_lo, 1);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactTableTest, CrossTableFactsEqualComparesExtensionPayloads) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));

  loom_value_facts_t strides[] = {
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_encoding_summary_t summary = {
      /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      /*.static_spec_encoding_id=*/0,
      /*.address_layout=*/
      {
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
          /*.rank=*/IREE_ARRAYSIZE(strides),
          /*.strides=*/strides,
      },
  };
  loom_value_facts_t source_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(
      &source.context, summary, &source_facts));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 0));

  loom_value_facts_t padding = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &target.context, loom_value_facts_exact_i64(99), &padding));
  loom_value_facts_t target_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_clone_fact(&target, &source,
                                                  source_facts, &target_facts));

  EXPECT_FALSE(loom_value_facts_equal(source_facts, target_facts));
  EXPECT_TRUE(loom_value_fact_table_facts_equal(&source, source_facts, &target,
                                                target_facts));
  EXPECT_TRUE(loom_value_fact_table_extensions_equal(&source, source_facts,
                                                     &target, target_facts));

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactTableTest, CrossTableFactsDifferentExtensionsDoNotCompareEqual) {
  loom_value_fact_table_t lhs = {0};
  loom_value_fact_table_t rhs = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&lhs, &arena_, 0));

  iree_arena_allocator_t rhs_arena;
  iree_arena_initialize(&block_pool_, &rhs_arena);
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&rhs, &rhs_arena, 0));

  loom_value_facts_t lhs_facts = loom_value_facts_unknown();
  loom_value_facts_t rhs_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &lhs.context, loom_value_facts_exact_i64(7), &lhs_facts));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &rhs.context, loom_value_facts_exact_i64(8), &rhs_facts));

  EXPECT_FALSE(
      loom_value_fact_table_facts_equal(&lhs, lhs_facts, &rhs, rhs_facts));
  EXPECT_FALSE(
      loom_value_fact_table_extensions_equal(&lhs, lhs_facts, &rhs, rhs_facts));

  iree_arena_deinitialize(&rhs_arena);
}

TEST_F(FactTableTest, TypeOwnedRawPayloadClonesAndMeetsThroughDomain) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));
  source.context.resolve_type_domain =
      loom_value_fact_type_domain_resolver_callback_make(
          FactTableTestResolveDomain,
          const_cast<loom_value_fact_domain_t*>(&kTestRawFactDomain));

  const uint8_t payload[] = {1, 3, 5, 7};
  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &source.context, kTestRawPayloadTag, payload, sizeof(payload),
      &extension));
  loom_value_facts_t same_extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &source.context, kTestRawPayloadTag, payload, sizeof(payload),
      &same_extension));
  EXPECT_EQ(extension.extension_id, same_extension.extension_id);
  loom_value_facts_t source_facts = loom_value_facts_exact_i64(11);
  source_facts.extension_id = extension.extension_id;

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 0));
  target.context.resolve_type_domain =
      loom_value_fact_type_domain_resolver_callback_make(
          FactTableTestResolveDomain,
          const_cast<loom_value_fact_domain_t*>(&kTestRawFactDomain));

  loom_type_t type = loom_type_none();
  loom_value_facts_t cloned = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_clone_fact_for_type(
      &target, &source, nullptr, type, source_facts, &cloned));

  EXPECT_EQ(cloned.range_lo, 11);
  EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
      nullptr, type, &source, source_facts, &target, cloned));

  const void* cloned_payload = nullptr;
  iree_host_size_t cloned_payload_length = 0;
  ASSERT_TRUE(loom_value_facts_query_extension_payload(
      &target.context, cloned, kTestRawPayloadTag, &cloned_payload,
      &cloned_payload_length));
  EXPECT_EQ(cloned_payload_length, sizeof(payload));
  EXPECT_NE(cloned_payload, payload);
  EXPECT_EQ(std::memcmp(cloned_payload, payload, sizeof(payload)), 0);

  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &target, nullptr, type, &source, source_facts, &target, cloned, &joined));
  EXPECT_EQ(joined.range_lo, 11);
  ASSERT_TRUE(loom_value_facts_query_extension_payload(
      &target.context, joined, kTestRawPayloadTag, &cloned_payload,
      &cloned_payload_length));
  EXPECT_EQ(cloned_payload_length, sizeof(payload));
  EXPECT_EQ(std::memcmp(cloned_payload, payload, sizeof(payload)), 0);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactTableTest, TypeOwnedRawPayloadRejectsTagMismatch) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  table.context.resolve_type_domain =
      loom_value_fact_type_domain_resolver_callback_make(
          FactTableTestResolveDomain,
          const_cast<loom_value_fact_domain_t*>(&kTestRawFactDomain));

  const uint8_t payload[] = {2, 4, 6, 8};
  loom_value_facts_t expected = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &table.context, kTestRawPayloadTag, payload, sizeof(payload), &expected));
  loom_value_facts_t other = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &table.context, kTestRawPayloadTag + 1, payload, sizeof(payload),
      &other));

  EXPECT_FALSE(loom_value_fact_table_facts_equal_for_type(
      nullptr, loom_type_none(), &table, expected, &table, other));
  EXPECT_FALSE(loom_value_facts_query_extension_payload(
      &table.context, other, kTestRawPayloadTag, nullptr, nullptr));
}

TEST_F(FactTableTest, TypeOwnedRawPayloadDegradesWhenTooLarge) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  uint8_t payload[LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT + 1] = {};
  loom_value_facts_t facts = loom_value_facts_exact_i64(5);
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &table.context, kTestRawPayloadTag, payload, sizeof(payload), &facts));

  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  EXPECT_EQ(facts.extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
}

TEST_F(FactTableTest, TypedMeetPreservesStableExtension) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &extension));
  loom_value_facts_t lhs = loom_value_facts_exact_i64(4);
  lhs.extension_id = extension.extension_id;
  loom_value_facts_t rhs = loom_value_facts_exact_i64(12);
  rhs.extension_id = extension.extension_id;

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64, loom_dim_pack_static(4), 0);
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, vector_type, &table, lhs, &table, rhs, &joined));

  EXPECT_EQ(joined.range_lo, 4);
  EXPECT_EQ(joined.range_hi, 12);
  loom_value_fact_uniform_element_t uniform = {};
  EXPECT_TRUE(
      loom_value_facts_query_uniform_element(&table.context, joined, &uniform));
  EXPECT_EQ(uniform.element.range_lo, 7);
}

TEST_F(FactTableTest, TypedWidenDropsChangingScalarButKeepsStableExtension) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(3), &extension));
  loom_value_facts_t previous = loom_value_facts_exact_i64(4);
  previous.extension_id = extension.extension_id;
  loom_value_facts_t next = loom_value_facts_exact_i64(8);
  next.extension_id = extension.extension_id;

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64, loom_dim_pack_static(4), 0);
  loom_value_facts_t widened = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &table, nullptr, vector_type, &table, previous, &table, next,
      /*iteration=*/2, &widened));

  EXPECT_EQ(widened.range_lo, INT64_MIN);
  EXPECT_EQ(widened.range_hi, INT64_MAX);
  loom_value_fact_uniform_element_t uniform = {};
  EXPECT_TRUE(loom_value_facts_query_uniform_element(&table.context, widened,
                                                     &uniform));
  EXPECT_EQ(uniform.element.range_lo, 3);
}

}  // namespace
}  // namespace loom
