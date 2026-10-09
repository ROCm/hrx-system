// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/physical_domains.h"

#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

loom_liveness_interval_t Interval(uint16_t class_id, uint32_t start,
                                  uint32_t end) {
  loom_liveness_interval_t interval = {
      .start_point = start, .end_point = end, .unit_count = 1};
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.value_class.register_class_id = class_id;
  return interval;
}

loom_low_placement_relation_t Relation(loom_value_ordinal_t source,
                                       loom_value_ordinal_t result,
                                       loom_low_placement_cause_t cause) {
  loom_low_placement_relation_t relation =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  relation.source_ordinal = source;
  relation.result_ordinal = result;
  relation.unit_count = 1;
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
  relation.cause = cause;
  relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
                   (cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT
                        ? LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD
                        : LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
  return relation;
}

class LowAllocationPhysicalDomainsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptors_, IREE_SV("test.explicit32"), &broad_, nullptr));
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptors_, IREE_SV("test.packed.narrow"), &narrow_, nullptr));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // These are final producer facts: origin intervals include their tied
  // members' storage lifetime. The domain planner must consume that identity
  // without charging another reservation for each semantic name.
  iree_status_t Build(
      std::vector<loom_liveness_interval_t> intervals,
      std::vector<loom_value_ordinal_t> origins,
      std::vector<loom_low_placement_relation_t> relations = {}) {
    intervals_ = std::move(intervals);
    const size_t count = intervals_.size();
    std::vector<loom_value_id_t> value_ids(count);
    std::vector<uint32_t> interval_indices(count);
    std::vector<loom_low_allocation_unit_liveness_value_t> values(count);
    std::vector<uint32_t> starts(count);
    std::vector<uint32_t> ends(count);
    for (size_t i = 0; i < count; ++i) {
      const uint32_t ordinal = static_cast<uint32_t>(i);
      intervals_[i].value_id = ordinal;
      value_ids[i] = ordinal;
      interval_indices[i] = ordinal;
      values[i] = {ordinal, intervals_[i].start_point};
      starts[i] = intervals_[i].start_point;
      ends[i] = intervals_[i].end_point;
    }
    loom_liveness_analysis_t liveness = {
        .intervals = intervals_.data(),
        .interval_count = count,
        .value_ids = value_ids.data(),
        .value_count = count,
        .value_interval_indices = interval_indices.data()};
    loom_low_allocation_unit_liveness_t unit_liveness = {
        .values = values.data(),
        .start_points = starts.data(),
        .end_points = ends.data(),
        .point_count = count};
    loom_low_placement_table_t placement =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment conversion
             // differs from list initialization.
    placement.value_ids = value_ids.data();
    placement.value_count = count;
    placement.relations = relations.data();
    placement.relation_count = relations.size();
    placement.tied_storage_origins_by_value_ordinal =
        origins.empty() ? nullptr : origins.data();
    return loom_low_allocation_physical_domains_build(
        descriptors_, &liveness, &unit_liveness, &placement, &arena_,
        &domains_);
  }

  loom_low_allocation_physical_domain_row_t Row(size_t index) const {
    loom_liveness_analysis_t liveness = {.intervals = intervals_.data(),
                                         .interval_count = intervals_.size()};
    return loom_low_allocation_physical_domains_for_interval(
        &domains_, &liveness, &intervals_[index]);
  }

  // Canonical test target with broad r0-r3 and narrow r0-r1 domains.
  const loom_low_descriptor_set_t* descriptors_ =
      loom_test_low_core_descriptor_set();
  // Broad class in the canonical test descriptor set.
  uint16_t broad_ = LOOM_LOW_REG_CLASS_NONE;
  // Restricted class in the canonical test descriptor set.
  uint16_t narrow_ = LOOM_LOW_REG_CLASS_NONE;
  // Pool backing retained preference rows.
  iree_arena_block_pool_t pool_;
  // Preference storage retained through each test.
  iree_arena_allocator_t arena_;
  // Interval domain used by the retained preference lookup.
  std::vector<loom_liveness_interval_t> intervals_;
  // Preference plan under test.
  loom_low_allocation_physical_domains_t domains_ = {};
};

TEST_F(LowAllocationPhysicalDomainsTest, TiedNamesReserveOneNarrowCandidate) {
  IREE_ASSERT_OK(Build({Interval(broad_, 0, 20), Interval(narrow_, 2, 18),
                        Interval(narrow_, 4, 18), Interval(narrow_, 6, 18)},
                       {0, 1, 1, 1},
                       {Relation(1, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
                        Relation(2, 3, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT)}));
  const auto broad = Row(0);
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 0));
  EXPECT_FALSE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 1));
}

TEST_F(LowAllocationPhysicalDomainsTest,
       IndependentOriginsReserveBothCandidates) {
  IREE_ASSERT_OK(Build({Interval(broad_, 0, 20), Interval(narrow_, 2, 18),
                        Interval(narrow_, 4, 18), Interval(narrow_, 6, 18)},
                       {0, 1, 2, 1},
                       {Relation(1, 3, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT)}));
  const auto broad = Row(0);
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 0));
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 1));
}

TEST_F(LowAllocationPhysicalDomainsTest,
       OptionalAffinityRetainsSeparateDemand) {
  IREE_ASSERT_OK(Build({Interval(broad_, 0, 20), Interval(narrow_, 2, 18),
                        Interval(narrow_, 4, 18)},
                       {},
                       {Relation(1, 2, LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY)}));
  const auto broad = Row(0);
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 0));
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 1));
}

TEST_F(LowAllocationPhysicalDomainsTest, DisjointOriginsReuseReservedCapacity) {
  IREE_ASSERT_OK(Build({Interval(broad_, 0, 20), Interval(narrow_, 2, 8),
                        Interval(narrow_, 4, 8), Interval(narrow_, 8, 18),
                        Interval(narrow_, 10, 18)},
                       {0, 1, 1, 3, 3},
                       {Relation(1, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
                        Relation(3, 4, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT)}));
  const auto broad = Row(0);
  EXPECT_TRUE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 0));
  EXPECT_FALSE(
      loom_low_allocation_physical_domain_row_candidate_is_reserved(broad, 1));
}

TEST_F(LowAllocationPhysicalDomainsTest,
       TiedBroadNamesDoNotCreateCapacityDebt) {
  IREE_ASSERT_OK(Build({Interval(broad_, 0, 20), Interval(broad_, 0, 20),
                        Interval(broad_, 4, 20), Interval(broad_, 6, 20),
                        Interval(narrow_, 8, 18)},
                       {0, 1, 0, 0, 4},
                       {Relation(0, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
                        Relation(2, 3, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT)}));
  const auto broad = Row(1);
  // Two broad origins fit in r2-r3. Aliases must not make either location
  // less attractive than the idle narrow r1 candidate.
  EXPECT_EQ(loom_low_allocation_physical_domain_row_candidate_rank(broad, 2),
            loom_low_allocation_physical_domain_row_candidate_rank(broad, 1));
  EXPECT_EQ(loom_low_allocation_physical_domain_row_candidate_rank(broad, 3),
            loom_low_allocation_physical_domain_row_candidate_rank(broad, 1));
}

}  // namespace
}  // namespace loom
