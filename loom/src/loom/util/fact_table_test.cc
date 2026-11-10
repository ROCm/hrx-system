// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/float_facts.h"

namespace loom {
namespace {

static std::vector<loom_value_id_t> CollectScopeValues(
    const loom_value_fact_table_t* table) {
  std::vector<loom_value_id_t> values;
  loom_value_fact_scope_iterator_t iterator;
  loom_value_fact_table_scope_begin(table, &iterator);
  for (loom_value_id_t value_id = loom_value_fact_scope_next(&iterator);
       value_id != LOOM_VALUE_ID_INVALID;
       value_id = loom_value_fact_scope_next(&iterator)) {
    values.push_back(value_id);
  }
  std::sort(values.begin(), values.end());
  return values;
}

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

  // Backing pool shared by independently owned fact tables.
  iree_arena_block_pool_t block_pool_;
  // Persistent storage used for table entries and standalone clones.
  iree_arena_allocator_t arena_;
};

TEST_F(FactTableTest, ValueEqualityUsesSsaOrExactIntegers) {
  EXPECT_TRUE(loom_value_fact_table_values_equal(nullptr, 1, 1));
  EXPECT_FALSE(loom_value_fact_table_values_equal(nullptr, 1, 2));
  EXPECT_FALSE(loom_value_fact_table_values_equal(
      nullptr, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID));
  loom_value_fact_table_t table;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 8));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_exact_i64(4)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 2, loom_value_facts_exact_i64(4)));
  EXPECT_TRUE(loom_value_fact_table_values_equal(&table, 1, 2));
  for (const auto facts :
       {loom_value_facts_exact_i64(5), loom_value_facts_make(0, 8, 1),
        loom_value_facts_unknown()}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&table, 2, facts));
    EXPECT_FALSE(loom_value_fact_table_values_equal(&table, 1, 2));
  }
  const auto range = loom_value_facts_make(0, 8, 1);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 1, range));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 2, range));
  EXPECT_FALSE(loom_value_fact_table_values_equal(&table, 1, 2));
  const auto floating = loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 4.0);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 1, floating));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 2, floating));
  EXPECT_FALSE(loom_value_fact_table_values_equal(&table, 1, 2));
}

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
    .extensions_equal = FactTableTestRawExtensionsEqual,
    .clone_extension = FactTableTestRawCloneExtension,
    .meet_extension = FactTableTestRawMeetExtension,
    .widen_extension = FactTableTestRawWidenExtension,
};

//===----------------------------------------------------------------------===//
// Initialize and lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, UndefinedEntriesRetainScopeMembershipAcrossGrowth) {
  loom_value_fact_table_t table;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 4));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 2, loom_value_facts_exact_i64(7)));
  loom_value_fact_table_undefine(&table, 2);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 2));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 129, loom_value_facts_exact_i64(9)));

  auto scope_values = CollectScopeValues(&table);
  ASSERT_EQ(scope_values.size(), 2u);
  EXPECT_EQ(scope_values[0], 2u);
  EXPECT_EQ(scope_values[1], 129u);

  loom_value_fact_table_t clone;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 4));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table, scope_values.data(), scope_values.size()}, nullptr));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&clone, 2));
  EXPECT_EQ(loom_value_fact_table_lookup(&clone, 129).range_lo, 9);
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 2, loom_value_facts_exact_i64(11)));
  EXPECT_EQ(table.touched_count, 2u);

  loom_value_fact_table_clear_scope(&table);
  EXPECT_TRUE(CollectScopeValues(&table).empty());
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 2));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 129));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 129,
                                              loom_value_facts_exact_i64(13)));
  EXPECT_EQ(table.touched_count, 1u);
}

TEST_F(FactTableTest, ZeroInitIsValid) {
  loom_value_fact_table_t table = {0};
  EXPECT_TRUE(CollectScopeValues(nullptr).empty());
  EXPECT_TRUE(CollectScopeValues(&table).empty());
  // Lookup on empty table returns unknown.
  loom_value_facts_t facts = loom_value_fact_table_lookup(&table, 0);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  facts = loom_value_fact_table_lookup(&table, 42);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
}

TEST_F(FactTableTest, InitializePreallocates) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));
  EXPECT_EQ(table.capacity, (iree_host_size_t)128);
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

TEST_F(FactTableTest, DefineAndLookupPreservesIndependentFlagBits) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 5));

  constexpr loom_value_fact_flags_t kFlags[] = {
      LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NAN,
      LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_INF,
      LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_SUBNORMAL,
      LOOM_VALUE_FACT_SUBGROUP_UNIFORM | LOOM_VALUE_FACT_WORKGROUP_UNIFORM,
      LOOM_VALUE_FACT_SUBGROUP_UNIFORM | LOOM_VALUE_FACT_WORKGROUP_UNIFORM |
          LOOM_VALUE_FACT_CLUSTER_UNIFORM,
  };
  for (loom_value_id_t value_id = 0; value_id < IREE_ARRAYSIZE(kFlags);
       ++value_id) {
    loom_value_facts_t facts = loom_value_facts_unknown();
    facts.flags = kFlags[value_id];
    IREE_ASSERT_OK(loom_value_fact_table_define(&table, value_id, facts));
  }

  for (loom_value_id_t value_id = 0; value_id < IREE_ARRAYSIZE(kFlags);
       ++value_id) {
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(&table, value_id);
    EXPECT_EQ(facts.flags, kFlags[value_id]);
  }
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

  loom_value_fact_segment_t* const first_segment = table.segments[0];
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table, 5)));
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table, 8)));

  loom_value_fact_table_clear_scope(&table);
  iree_arena_reset(&transient_arena);

  EXPECT_EQ(table.segments[0], first_segment);
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

TEST_F(FactTableTest, LazyScopeOmitsUnrelatedValuePrefix) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  constexpr loom_value_id_t kFirst = (1u << 20) + 63;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kFirst,
                                              loom_value_facts_exact_i64(17)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kFirst + 1,
                                              loom_value_facts_exact_i64(19)));

  EXPECT_LE(table.capacity, 128u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kFirst).range_lo, 17);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kFirst + 1).range_lo, 19);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 0));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, kFirst - 1));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, kFirst + 2));
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_EQ(table.count, kFirst + 2);

  loom_value_fact_table_t cloned = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&cloned, &arena_, 0));
  const loom_value_id_t values[] = {kFirst, kFirst + 2};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &cloned, {&table, values, IREE_ARRAYSIZE(values)}, nullptr));
  EXPECT_EQ(loom_value_fact_table_lookup(&cloned, kFirst).range_lo, 17);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&cloned, kFirst + 1));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&cloned, kFirst + 2));
  EXPECT_LE(cloned.capacity, 64u);
}

TEST_F(FactTableTest, RangeGrowthRetainsUndefinedMembershipAndScopeReuse) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 257, loom_value_facts_exact_i64(7)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 320,
                                              loom_value_facts_exact_i64(11)));
  loom_value_fact_table_undefine(&table, 320);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 191,
                                              loom_value_facts_exact_i64(13)));
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 257).range_lo, 7);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 191).range_lo, 13);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 320));
  EXPECT_EQ(table.touched_count, 3u);

  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 1024));
  EXPECT_EQ(table.first_value_id, 0u);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 320,
                                              loom_value_facts_exact_i64(23)));
  EXPECT_EQ(table.touched_count, 3u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 257).range_lo, 7);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 320).range_lo, 23);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 191).range_lo, 13);

  loom_value_fact_table_clear_scope(&table);
  for (loom_value_id_t value : {191u, 257u, 320u}) {
    EXPECT_FALSE(loom_value_fact_table_has_entry(&table, value));
  }
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 320,
                                              loom_value_facts_exact_i64(29)));
  EXPECT_EQ(table.touched_count, 1u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 320).range_lo, 29);
}

TEST_F(FactTableTest, ReserveRangeOmitsPrefixAndPreservesFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_reserve_range(&table, (loom_value_fact_range_t){
                                                      .first_value_id = 1027,
                                                      .capacity = 97,
                                                  }));
  EXPECT_EQ(table.first_value_id, 1024u);
  EXPECT_EQ(table.capacity, 128u);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 1027));

  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 1050,
                                              loom_value_facts_exact_i64(17)));
  loom_value_fact_segment_t* const retained_segment =
      table.segments[(1050 - table.first_value_id) >>
                     LOOM_VALUE_FACT_SEGMENT_SHIFT];
  IREE_ASSERT_OK(
      loom_value_fact_table_reserve_range(&table, (loom_value_fact_range_t){
                                                      .first_value_id = 1030,
                                                      .capacity = 32,
                                                  }));
  EXPECT_EQ(table.segments[(1050 - table.first_value_id) >>
                           LOOM_VALUE_FACT_SEGMENT_SHIFT],
            retained_segment);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 1050).range_lo, 17);

  IREE_ASSERT_OK(
      loom_value_fact_table_reserve_range(&table, (loom_value_fact_range_t){
                                                      .first_value_id = 960,
                                                      .capacity = 300,
                                                  }));
  EXPECT_EQ(table.first_value_id, 960u);
  EXPECT_GE(table.capacity, 300u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 1050).range_lo, 17);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 959));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 1259));
}

TEST_F(FactTableTest, ReserveEmptyRangeDoesNotAllocate) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_reserve_range(
      &table, (loom_value_fact_range_t){0}));
  EXPECT_EQ(table.segments, nullptr);
  EXPECT_EQ(table.capacity, 0u);
}

TEST_F(FactTableTest, DescendingGrowthBoundsStorageAndPreservesMembership) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  constexpr loom_value_id_t kLast = (1u << 20) + 7;
  constexpr uint32_t kCount = 16;
  loom_value_fact_segment_t** previous_directory = nullptr;
  iree_host_size_t directory_allocation_count = 0;
  loom_value_fact_segment_t* last_segment = nullptr;
  for (uint32_t i = 0; i < kCount; ++i) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table, kLast - 64 * i, loom_value_facts_exact_i64(i + 1)));
    if (table.segment_directory != previous_directory) {
      previous_directory = table.segment_directory;
      ++directory_allocation_count;
    }
    if (i == 0) {
      last_segment = table.segments[0];
    }
    // Stop on excessive growth before a broken implementation can exhaust
    // memory. Each definition adds one touched-membership word to the span.
    ASSERT_LE(table.capacity, 2u * (i + 1) * 64);
    if (i == 1) {
      loom_value_fact_table_undefine(&table, kLast - 64);
    }
  }
  // The balanced pointer directory grows geometrically even when every new
  // segment is prepended. Payload segments never move while it grows.
  EXPECT_LE(directory_allocation_count, 6u);
  EXPECT_EQ(table.segments[kCount - 1], last_segment);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, kLast - 64));
  for (uint32_t i = 0; i < kCount; ++i) {
    if (i == 1) {
      continue;
    }
    EXPECT_EQ(loom_value_fact_table_lookup(&table, kLast - 64 * i).range_lo,
              i + 1);
  }
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kLast - 64,
                                              loom_value_facts_exact_i64(42)));
  EXPECT_EQ(table.touched_count, kCount);

  loom_value_fact_segment_t* const retained_segment =
      table.segments[(kLast - 64 - table.first_value_id) >>
                     LOOM_VALUE_FACT_SEGMENT_SHIFT];
  loom_value_fact_table_clear_scope(&table);
  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_FALSE(loom_value_fact_table_has_entry(&table, kLast - 64 * i));
  }
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kLast - 64,
                                              loom_value_facts_exact_i64(7)));
  EXPECT_EQ(table.segments[(kLast - 64 - table.first_value_id) >>
                           LOOM_VALUE_FACT_SEGMENT_SHIFT],
            retained_segment);
  EXPECT_EQ(table.touched_count, 1u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kLast - 64).range_lo, 7);
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

TEST_F(FactTableTest, ReserveAllocatesPooledSegmentsWithoutOversizedStorage) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2048));
  iree_arena_block_pool_statistics_t before;
  iree_arena_block_pool_query_statistics(&block_pool_, &before);

  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 9548));
  EXPECT_EQ(table.capacity, 16384u);
  EXPECT_EQ(table.count, 0u);
  EXPECT_EQ(table.touched_count, 0u);
  iree_arena_block_pool_statistics_t after;
  iree_arena_block_pool_query_statistics(&block_pool_, &after);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(
      after.oversized_allocation_count - before.oversized_allocation_count, 0u);
  EXPECT_EQ(
      after.oversized_allocation_bytes - before.oversized_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  // Subsequent definitions use the reserved pooled payloads directly.
  loom_value_fact_segment_t* const first_segment = table.segments[0];
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 9755,
                                              loom_value_facts_exact_i64(23)));
  EXPECT_EQ(table.segments[0], first_segment);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 9755).range_lo, 23);
}

TEST_F(FactTableTest, ReservePreservesFactsAndUndefinedMembership) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 4));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_exact_i64(7)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 2, loom_value_facts_exact_i64(9)));
  loom_value_fact_table_undefine(&table, 2);
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 130));
  EXPECT_EQ(table.capacity, 256u);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, 1).range_lo, 7);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 2));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 129));
  EXPECT_EQ(table.touched_count, 2u);

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 2, loom_value_facts_exact_i64(11)));
  EXPECT_EQ(table.touched_count, 2u);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 129,
                                              loom_value_facts_exact_i64(13)));
  EXPECT_EQ(table.touched_count, 3u);
  loom_value_fact_table_clear_scope(&table);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 1));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 2));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table, 129));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 129,
                                              loom_value_facts_exact_i64(17)));
  EXPECT_EQ(table.touched_count, 1u);
}

TEST_F(FactTableTest, ReserveUsesExistingCapacityAndPreservesSatisfiedStorage) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 0));
  EXPECT_EQ(table.segments, nullptr);
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 3));
  EXPECT_EQ(table.capacity, 64u);
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 130));
  EXPECT_EQ(table.capacity, 256u);
  loom_value_fact_segment_t* const first_segment = table.segments[0];
  loom_value_fact_segment_t** const segments = table.segments;
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 256));
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table, 0));
  EXPECT_EQ(table.segments, segments);
  EXPECT_EQ(table.segments[0], first_segment);
  EXPECT_EQ(table.capacity, 256u);
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

TEST_F(FactTableTest, UniformElementPreservesFloatPredicateFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t element = loom_value_facts_unknown();
  element.flags |= LOOM_VALUE_FACT_FLOAT_PREDICATE_MASK;

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_uniform_element(&table.context, element, &facts));

  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_TRUE(loom_value_facts_is_not_nan(facts));
  EXPECT_TRUE(loom_value_facts_is_not_inf(facts));
  EXPECT_TRUE(loom_value_facts_is_finite(facts));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(facts));
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

TEST_F(FactTableTest, OversizedExtensionEntriesPersistAcrossScopes) {
  iree_arena_allocator_t transient_arena = {};
  iree_arena_initialize(&block_pool_, &transient_arena);
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
      &table, &arena_, &transient_arena, 0));

  const uint8_t first_payload[] = {1, 2, 3, 4};
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &table.context, kTestRawPayloadTag, first_payload,
      IREE_ARRAYSIZE(first_payload), &facts));
  const auto* entries = table.extensions.entries;
  const auto capacity = table.extensions.capacity;
  const auto persistent_bytes = arena_.used_allocation_size;
  EXPECT_TRUE(table.has_persistent_extension_entries);

  loom_value_fact_table_clear_scope(&table);
  iree_arena_reset(&transient_arena);
  EXPECT_EQ(table.extensions.entries, entries);
  EXPECT_EQ(table.extensions.capacity, capacity);
  EXPECT_EQ(table.extensions.count, 0u);

  const uint8_t second_payload[] = {5, 6, 7};
  facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
      &table.context, kTestRawPayloadTag, second_payload,
      IREE_ARRAYSIZE(second_payload), &facts));
  EXPECT_EQ(table.extensions.entries, entries);
  EXPECT_EQ(table.extensions.capacity, capacity);
  EXPECT_EQ(arena_.used_allocation_size, persistent_bytes);
  const void* stored_payload = nullptr;
  iree_host_size_t stored_length = 0;
  ASSERT_TRUE(loom_value_facts_query_extension_payload(
      &table.context, facts, kTestRawPayloadTag, &stored_payload,
      &stored_length));
  EXPECT_EQ(stored_length, IREE_ARRAYSIZE(second_payload));
  EXPECT_EQ(std::memcmp(stored_payload, second_payload, stored_length), 0);

  iree_arena_deinitialize(&transient_arena);
}

TEST(FactTableStorageTest, PooledExtensionEntriesClearWithScope) {
  iree_arena_block_pool_t block_pool = {};
  iree_arena_block_pool_initialize(16384, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena = {};
  iree_arena_initialize(&block_pool, &arena);
  iree_arena_allocator_t transient_arena = {};
  iree_arena_initialize(&block_pool, &transient_arena);
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
      &table, &arena, &transient_arena, 0));

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(42), &facts));
  EXPECT_FALSE(table.has_persistent_extension_entries);

  loom_value_fact_table_clear_scope(&table);
  EXPECT_EQ(table.extensions.entries, nullptr);
  EXPECT_EQ(table.extensions.capacity, 0u);
  EXPECT_EQ(table.extensions.count, 0u);

  iree_arena_deinitialize(&transient_arena);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(FactTableStorageTest,
     PooledExtensionEntriesReach1024WithoutOversizedStorage) {
  iree_arena_block_pool_t block_pool = {};
  iree_arena_block_pool_initialize(131072, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena = {};
  iree_arena_initialize(&block_pool, &arena);
  iree_arena_allocator_t transient_arena = {};
  iree_arena_initialize(&block_pool, &transient_arena);
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
      &table, &arena, &transient_arena, 0));

  const auto populate = [](loom_value_fact_table_t* table) -> iree_status_t {
    for (uint32_t i = 0; i < 1024; ++i) {
      loom_value_facts_t facts;
      IREE_RETURN_IF_ERROR(loom_value_facts_make_uniform_element(
          &table->context, loom_value_facts_exact_i64(i), &facts));
      if (facts.extension_id != i + 1) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "extension ID mismatch at %u", i);
      }
    }
    return iree_ok_status();
  };

  IREE_ASSERT_OK(populate(&table));
  EXPECT_EQ(table.extensions.capacity, 1024u);
  EXPECT_FALSE(table.has_persistent_extension_entries);
  for (const uint32_t i : {0u, 511u, 512u, 1023u}) {
    loom_value_facts_t facts = loom_value_facts_unknown();
    facts.extension_id = i + 1;
    loom_value_fact_uniform_element_t uniform = {};
    ASSERT_TRUE(loom_value_facts_query_uniform_element(&table.context, facts,
                                                       &uniform));
    EXPECT_EQ(uniform.element.range_lo, i);
    EXPECT_EQ(uniform.element.range_hi, i);
  }
  iree_arena_block_pool_statistics_t warm_statistics;
  iree_arena_block_pool_query_statistics(&block_pool, &warm_statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(warm_statistics.oversized_allocation_count, 0u);
  EXPECT_EQ(warm_statistics.oversized_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  loom_value_fact_table_clear_scope(&table);
  iree_arena_reset(&transient_arena);
  EXPECT_EQ(table.extensions.entries, nullptr);
  EXPECT_EQ(table.extensions.capacity, 0u);
  IREE_ASSERT_OK(populate(&table));
  iree_arena_block_pool_statistics_t reused_statistics;
  iree_arena_block_pool_query_statistics(&block_pool, &reused_statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(reused_statistics.block_system_allocation_count,
            warm_statistics.block_system_allocation_count);
  EXPECT_EQ(reused_statistics.oversized_allocation_count,
            warm_statistics.oversized_allocation_count);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_deinitialize(&transient_arena);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
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
      .lanes = lanes,
      .count = IREE_ARRAYSIZE(lanes),
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

TEST_F(FactTableTest, SmallStaticLanesPreserveCommonFloatPredicates) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[] = {
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0),
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, INFINITY),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
      &table.context,
      loom_value_fact_small_static_lanes_t{
          .lanes = lanes,
          .count = IREE_ARRAYSIZE(lanes),
      },
      &facts));

  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_TRUE(loom_value_facts_is_not_nan(facts));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(facts));
  EXPECT_FALSE(loom_value_facts_is_not_inf(facts));
  EXPECT_FALSE(loom_value_facts_is_finite(facts));
}

TEST_F(FactTableTest, OversizedSmallStaticLanesDegradesToUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT + 1] = {};
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = IREE_ARRAYSIZE(lanes),
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
      .base = loom_value_facts_exact_i64(2),
      .step = loom_value_facts_exact_i64(3),
      .bit_count = 8,
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
  EXPECT_EQ(result.bit_count, 8);

  loom_value_facts_t repeated;
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_iota(&table.context, iota, &repeated));
  EXPECT_EQ(facts.extension_id, repeated.extension_id);
  iota.bit_count = 16;
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_iota(&table.context, iota, &repeated));
  EXPECT_NE(facts.extension_id, repeated.extension_id);

  loom_value_fact_table_t imported;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&imported, &arena_, 0));
  loom_value_facts_t cloned;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&imported, &table, facts, &cloned));
  EXPECT_TRUE(
      loom_value_fact_table_facts_equal(&table, facts, &imported, cloned));
  EXPECT_FALSE(
      loom_value_fact_table_facts_equal(&table, repeated, &imported, cloned));
  ASSERT_TRUE(
      loom_value_facts_query_vector_iota(&imported.context, cloned, &result));
  EXPECT_EQ(result.bit_count, 8);
}

TEST_F(FactTableTest, IotaElementsWrapInEveryFixedWidthCarrier) {
  for (const uint8_t bit_count : {8, 16, 32, 64}) {
    SCOPED_TRACE(bit_count);
    const int64_t maximum = INT64_MAX >> (64 - bit_count);
    const int64_t minimum = -maximum - 1;
    loom_value_fact_vector_iota_t iota = {
        .base = loom_value_facts_exact_i64(maximum),
        .step = loom_value_facts_exact_i64(1),
        .bit_count = bit_count,
    };
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 0).range_lo, maximum);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 1).range_lo, minimum);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 2).range_lo,
              minimum + 1);
    iota.base = loom_value_facts_exact_i64(minimum);
    iota.step = loom_value_facts_exact_i64(-1);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 1).range_lo, maximum);
    iota.base = loom_value_facts_exact_i64(0);
    iota.step = loom_value_facts_exact_i64(minimum);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 2).range_lo, 0);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, 3).range_lo, minimum);
    iota.step = loom_value_facts_exact_i64(0);
    EXPECT_EQ(loom_value_fact_vector_iota_element(iota, UINT64_MAX).range_lo,
              0);
  }
}

TEST_F(FactTableTest, IotaMathematicalIndicesRemainUnknownOnOverflow) {
  loom_value_fact_vector_iota_t iota = {
      .base = loom_value_facts_exact_i64(INT64_MAX),
      .step = loom_value_facts_exact_i64(1),
      .bit_count = 0,
  };
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_vector_iota_element(iota, 1)));
  iota.base = loom_value_facts_unknown();
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_vector_iota_element(iota, 0)));
}

TEST_F(FactTableTest, IotaBoundsRequireBothEndpointsToFit) {
  for (const uint8_t bit_count : {8, 16, 32, 64}) {
    SCOPED_TRACE(bit_count);
    const int64_t maximum = INT64_MAX >> (64 - bit_count);
    const int64_t minimum = -maximum - 1;
    loom_value_fact_vector_iota_t iota = {
        .base = loom_value_facts_make(maximum - 3, maximum - 2, 1),
        .step = loom_value_facts_make(1, 2, 1),
        .bit_count = bit_count,
    };
    int64_t lower = 0;
    int64_t upper = 0;
    ASSERT_TRUE(loom_value_fact_vector_iota_bounds(iota, 2, &lower, &upper));
    EXPECT_EQ(lower, maximum - 3);
    EXPECT_EQ(upper, maximum);
    EXPECT_FALSE(loom_value_fact_vector_iota_bounds(iota, 3, &lower, &upper));
    iota.base = loom_value_facts_make(minimum + 2, minimum + 3, 1);
    iota.step = loom_value_facts_make(-2, -1, 1);
    ASSERT_TRUE(loom_value_fact_vector_iota_bounds(iota, 2, &lower, &upper));
    EXPECT_EQ(lower, minimum);
    EXPECT_EQ(upper, minimum + 3);
    EXPECT_FALSE(loom_value_fact_vector_iota_bounds(iota, 3, &lower, &upper));
    // Equal first/last values do not make a wrapping sequence monotonic.
    iota.base = loom_value_facts_exact_i64(0);
    iota.step = loom_value_facts_exact_i64(minimum);
    EXPECT_FALSE(loom_value_fact_vector_iota_bounds(iota, 3, &lower, &upper));
    iota.step = loom_value_facts_exact_i64(0);
    ASSERT_TRUE(
        loom_value_fact_vector_iota_bounds(iota, UINT64_MAX, &lower, &upper));
    EXPECT_EQ(lower, 0);
    EXPECT_EQ(upper, 0);
  }
}

TEST_F(FactTableTest, MaximumElementCountUsesExtentFacts) {
  loom_value_fact_table_t table;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2));
  const auto type = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(4),
      loom_dim_pack_dynamic(1), /*encoding_id=*/0);
  EXPECT_EQ(loom_value_fact_table_maximum_element_count(&table, type),
            UINT64_MAX);
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_make(1, 8, 1)));
  EXPECT_EQ(loom_value_fact_table_maximum_element_count(&table, type), 32);
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_exact_i64(0)));
  EXPECT_EQ(loom_value_fact_table_maximum_element_count(&table, type), 0);
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table, 1, loom_value_facts_make(0, INT64_MAX - 1, 1)));
  EXPECT_EQ(loom_value_fact_table_maximum_element_count(&table, type),
            UINT64_MAX);
}

TEST_F(FactTableTest, VectorIntegerBoundsSummarizeExplicitLanes) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_make(-4, 11, 1), &facts));
  int64_t lower = 0;
  int64_t upper = 0;
  EXPECT_TRUE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, std::numeric_limits<uint64_t>::max(), &lower,
      &upper));
  EXPECT_EQ(lower, -4);
  EXPECT_EQ(upper, 11);

  loom_value_facts_t lanes[] = {
      loom_value_facts_make(-7, -3, 1),
      loom_value_facts_exact_i64(5),
      loom_value_facts_make(9, 15, 1),
  };
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
      &table.context, {.lanes = lanes, .count = IREE_ARRAYSIZE(lanes)},
      &facts));
  EXPECT_TRUE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, std::numeric_limits<uint64_t>::max(), &lower,
      &upper));
  EXPECT_EQ(lower, -7);
  EXPECT_EQ(upper, 15);
}

TEST_F(FactTableTest, VectorIntegerBoundsSummarizeIotas) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_vector_iota(
      &table.context,
      {
          .base = loom_value_facts_make(2, 4, 1),
          .step = loom_value_facts_exact_i64(3),
          .bit_count = 0,
      },
      &facts));
  int64_t lower = 0;
  int64_t upper = 0;
  EXPECT_TRUE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, /*maximum_lane_count=*/5, &lower, &upper));
  EXPECT_EQ(lower, 2);
  EXPECT_EQ(upper, 16);

  IREE_ASSERT_OK(loom_value_facts_make_vector_iota(
      &table.context,
      {
          .base = loom_value_facts_make(20, 22, 1),
          .step = loom_value_facts_exact_i64(-4),
          .bit_count = 0,
      },
      &facts));
  EXPECT_TRUE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, /*maximum_lane_count=*/4, &lower, &upper));
  EXPECT_EQ(lower, 8);
  EXPECT_EQ(upper, 22);
  EXPECT_TRUE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, /*maximum_lane_count=*/0, &lower, &upper));
  EXPECT_EQ(lower, 0);
  EXPECT_EQ(upper, -1);
  EXPECT_FALSE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, std::numeric_limits<uint64_t>::max(), &lower,
      &upper));

  IREE_ASSERT_OK(loom_value_facts_make_vector_iota(
      &table.context,
      {
          .base = loom_value_facts_exact_i64(INT64_MAX),
          .step = loom_value_facts_exact_i64(1),
          .bit_count = 0,
      },
      &facts));
  EXPECT_FALSE(loom_value_facts_query_vector_integer_bounds(
      &table.context, facts, /*maximum_lane_count=*/2, &lower, &upper));
}

TEST_F(FactTableTest, VectorPrefixMaskExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_vector_prefix_mask_t mask = {
      .lower_bound = loom_value_facts_exact_i64(0),
      .upper_bound = loom_value_facts_make(0, 16, 1),
      .step = loom_value_facts_exact_i64(1),
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
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .static_spec_encoding_id = 0,
      .address_layout =
          {
              .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
              .rank = 0,
              .strides = nullptr,
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

TEST(EncodedOperandSchemaTest, SemanticAbsenceDoesNotImplyScale) {
  loom_value_fact_encoded_operand_schema_t schema = {
      .flags = LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ELEMENT_FORMAT_NONE |
               LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_AFFINE_NONE,
  };
  EXPECT_FALSE(loom_value_fact_encoded_operand_schema_has_scale(schema));

  schema.flags |= LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  EXPECT_TRUE(loom_value_fact_encoded_operand_schema_has_scale(schema));
}

TEST_F(FactTableTest, EncodingSummaryStridedLayoutInternsStrideFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t strides[] = {
      loom_value_facts_make(8, 16, 8),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .static_spec_encoding_id = 0,
      .address_layout =
          {
              .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
              .rank = IREE_ARRAYSIZE(strides),
              .strides = strides,
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

TEST_F(FactTableTest, CloneValuesCopiesOnlySelectedFacts) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, 100,
                                              loom_value_facts_exact_i64(100)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 2, loom_value_facts_exact_i64(2)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, 100,
                                              loom_value_facts_exact_i64(101)));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 0));
  const loom_value_id_t values[] = {100, 50};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, values, IREE_ARRAYSIZE(values)}, nullptr));

  EXPECT_EQ(loom_value_fact_table_lookup(&target, 100).range_lo, 101);
  EXPECT_FALSE(loom_value_fact_table_has_entry(&target, 2));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&target, 50));
  EXPECT_EQ(target.touched_count, 1u);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactTableTest, CloneValuesReinternsExtensions) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  loom_value_facts_t strides[] = {
      loom_value_facts_make(16, 64, 16),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .static_spec_encoding_id = 0,
      .address_layout =
          {
              .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
              .rank = IREE_ARRAYSIZE(strides),
              .strides = strides,
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
  const auto scope_values = CollectScopeValues(&source);
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, scope_values.data(), scope_values.size()}, nullptr));

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
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .static_spec_encoding_id = 0,
      .address_layout =
          {
              .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
              .rank = IREE_ARRAYSIZE(strides),
              .strides = strides,
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

TEST_F(FactTableTest, ConvergedFactsReuseOwnedPayloads) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const loom_value_facts_t lanes[] = {
      loom_value_facts_exact_i64(8),
      loom_value_facts_make(16, 32, 16),
  };
  const loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = IREE_ARRAYSIZE(lanes),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(&table.context,
                                                          lane_slice, &facts));
  const iree_host_size_t used_bytes = arena_.used_allocation_size;

  // A fixed point may visit the same aggregate repeatedly. Retaining its
  // immutable extension must not allocate a discarded copy on each visit.
  for (uint32_t iteration = 0; iteration < 16; ++iteration) {
    loom_value_facts_t result;
    IREE_ASSERT_OK(
        loom_value_fact_table_clone_fact(&table, &table, facts, &result));
    EXPECT_TRUE(loom_value_facts_equal(result, facts));
    IREE_ASSERT_OK(loom_value_fact_table_clone_fact_for_type(
        &table, &table, nullptr, loom_type_none(), facts, &result));
    EXPECT_TRUE(loom_value_facts_equal(result, facts));
    IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
        &table, nullptr, loom_type_none(), &table, facts, &table, facts,
        &result));
    EXPECT_TRUE(loom_value_facts_equal(result, facts));
    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, loom_type_none(), &table, facts, &table, facts,
        iteration, &result));
    EXPECT_TRUE(loom_value_facts_equal(result, facts));
  }
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
}

TEST_F(FactTableTest, ClonedPayloadOutlivesSourceArena) {
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 0));
  loom_value_facts_t cloned;
  {
    iree_arena_allocator_t source_arena;
    iree_arena_initialize(&block_pool_, &source_arena);
    loom_value_fact_table_t source = {0};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &source_arena, 0));
    const loom_value_facts_t lanes[] = {
        loom_value_facts_exact_i64(8),
        loom_value_facts_make(16, 32, 16),
    };
    const loom_value_fact_small_static_lanes_t lane_slice = {
        .lanes = lanes,
        .count = IREE_ARRAYSIZE(lanes),
    };
    loom_value_facts_t facts = loom_value_facts_unknown();
    IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
        &source.context, lane_slice, &facts));
    IREE_ASSERT_OK(
        loom_value_fact_table_clone_fact(&target, &source, facts, &cloned));
    iree_arena_deinitialize(&source_arena);
    iree_arena_block_pool_trim(&block_pool_);
  }
  loom_value_fact_small_static_lanes_t result;
  ASSERT_TRUE(loom_value_facts_query_small_static_lanes(&target.context, cloned,
                                                        &result));
  ASSERT_EQ(result.count, 2u);
  EXPECT_TRUE(loom_value_facts_is_exact(result.lanes[0]));
  EXPECT_EQ(result.lanes[0].range_lo, 8);
  EXPECT_EQ(result.lanes[1].range_lo, 16);
  EXPECT_EQ(result.lanes[1].range_hi, 32);
  EXPECT_EQ(result.lanes[1].known_divisor, 16);
}

TEST_F(FactTableTest, CrossTableClonesReuseOwnedArrayAndRawPayloads) {
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 0));
  // Destination IDs differ from source IDs, even for identical payloads.
  loom_value_facts_t padding;
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &target.context, loom_value_facts_exact_i64(99), &padding));
  loom_value_facts_t copied[3];
  uint8_t payload[LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT];
  for (size_t i = 0; i < IREE_ARRAYSIZE(payload); ++i) {
    payload[i] = static_cast<uint8_t>(i);
  }
  {
    iree_arena_allocator_t source_arena;
    iree_arena_initialize(&block_pool_, &source_arena);
    loom_value_fact_table_t source = {0};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &source_arena, 0));
    loom_value_facts_t elements[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT];
    for (size_t i = 0; i < IREE_ARRAYSIZE(elements); ++i) {
      elements[i] = loom_value_facts_make(i, i + 16, 1);
    }
    loom_value_facts_t original[3];
    IREE_ASSERT_OK(loom_value_facts_make_extension_payload(
        &source.context, kTestRawPayloadTag, payload, sizeof(payload),
        &original[2]));
    // Type-owned lane extensions must be imported before interning the array.
    elements[0].extension_id = original[2].extension_id;
    IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
        &source.context, {elements, IREE_ARRAYSIZE(elements)}, &original[0]));
    elements[0].extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    loom_value_fact_encoding_summary_t summary = {
        .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        .address_layout =
            {
                .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
                .rank = LOOM_TYPE_MAX_RANK,
                .strides = elements,
            },
    };
    IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(
        &source.context, summary, &original[1]));
    for (size_t i = 0; i < IREE_ARRAYSIZE(original); ++i) {
      IREE_ASSERT_OK(loom_value_fact_table_clone_fact(&target, &source,
                                                      original[i], &copied[i]));
      EXPECT_NE(copied[i].extension_id, original[i].extension_id);
      EXPECT_TRUE(loom_value_fact_table_facts_equal(&source, original[i],
                                                    &target, copied[i]));
    }
    const auto used_bytes = arena_.used_allocation_size;
    const auto extension_count = target.extensions.count;
    for (int iteration = 0; iteration < 16; ++iteration) {
      for (size_t i = 0; i < IREE_ARRAYSIZE(original); ++i) {
        loom_value_facts_t repeated;
        IREE_ASSERT_OK(loom_value_fact_table_clone_fact(
            &target, &source, original[i], &repeated));
        EXPECT_TRUE(loom_value_facts_equal(repeated, copied[i]));
        EXPECT_EQ(arena_.used_allocation_size, used_bytes);
        EXPECT_EQ(target.extensions.count, extension_count);
      }
    }
    iree_arena_deinitialize(&source_arena);
    iree_arena_block_pool_trim(&block_pool_);
  }
  loom_value_fact_small_static_lanes_t lanes;
  ASSERT_TRUE(loom_value_facts_query_small_static_lanes(&target.context,
                                                        copied[0], &lanes));
  ASSERT_EQ(lanes.count, LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT);
  for (size_t i = 0; i < lanes.count; ++i) {
    EXPECT_EQ(lanes.lanes[i].range_lo, i);
    EXPECT_EQ(lanes.lanes[i].range_hi, i + 16);
  }
  EXPECT_EQ(lanes.lanes[0].extension_id, copied[2].extension_id);
  loom_value_fact_encoding_summary_t summary;
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target.context,
                                                      copied[1], &summary));
  ASSERT_EQ(summary.address_layout.rank, LOOM_TYPE_MAX_RANK);
  for (size_t i = 0; i < summary.address_layout.rank; ++i) {
    EXPECT_EQ(summary.address_layout.strides[i].range_lo, i);
    EXPECT_EQ(summary.address_layout.strides[i].range_hi, i + 16);
  }
  const void* copied_payload = nullptr;
  iree_host_size_t copied_length = 0;
  ASSERT_TRUE(loom_value_facts_query_extension_payload(
      &target.context, copied[2], kTestRawPayloadTag, &copied_payload,
      &copied_length));
  ASSERT_EQ(copied_length, sizeof(payload));
  EXPECT_EQ(std::memcmp(copied_payload, payload, sizeof(payload)), 0);
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

}  // namespace
}  // namespace loom
