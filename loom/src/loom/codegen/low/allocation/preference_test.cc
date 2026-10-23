// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/preference.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowAllocationPreferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    classes_[0].flags = classes_[1].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
    classes_[0].alloc_unit_bits = classes_[1].alloc_unit_bits = 32;
    classes_[0].allocatable_count = classes_[1].allocatable_count = 7;
    classes_[1].target_bank_id = 1;
    descriptors_.reg_classes = classes_;
    descriptors_.reg_class_count = 2;
    target_.descriptor_set = &descriptors_;
    constraints_.target = &target_;
    for (uint32_t i = 0; i < 4; ++i) {
      intervals_[i].value_class.type_kind = LOOM_TYPE_REGISTER;
      intervals_[i].unit_count = 2;
      assignments_[i].location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
      assignments_[i].unit_count = assignments_[i].location_count = 2;
    }
    liveness_.value_ids = value_ids_;
    liveness_.value_count = 4;
    liveness_.intervals = intervals_;
    liveness_.interval_count = 4;
    liveness_.value_interval_indices = interval_indices_;
    assignment_map_.liveness = &liveness_;
    assignment_map_.assignments = assignments_;
    assignment_map_.assignment_count = 4;
    assignment_map_.assignment_indices_by_value_ordinal = assignment_indices_;
    placement_.value_ids = value_ids_;
    placement_.value_count = 4;
    placement_.tied_storage_origins_by_value_ordinal = origins_;
    placement_.ranges_by_result_ordinal = ranges_;
    placement_.ranges_by_source_ordinal = source_ranges_;
    preference_ = {values_, predicates_, clauses_, 3, 2};
    use_ = {&preference_, 0, 3, {2, 3}};
    index_.uses = &use_;
    index_.use_count = 1;
    index_.bindings = bindings_;
    index_.binding_count = 3;
    index_.use_indices = use_indices_;
    index_.offsets_by_origin = offsets_;
    index_.max_incident_use_count = 1;
    index_.max_incident_binding_count = 3;
    index_.max_memo_entry_count = 4;
    IREE_ASSERT_OK(loom_low_allocation_preference_workspace_initialize(
        &index_, &arena_, &workspace_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_low_allocation_preference_query_t Prepare(
      uint32_t secondary = LOOM_VALUE_ORDINAL_INVALID) {
    return loom_low_allocation_preference_prepare(
        &index_, &placement_, &assignment_map_, &constraints_, 0, secondary,
        &workspace_);
  }

  uint32_t Penalty(const loom_low_allocation_preference_query_t& query,
                   uint32_t primary, uint32_t secondary = 0) {
    auto first = assignments_[0];
    auto second = assignments_[0];
    first.location_base = primary;
    second.location_base = secondary;
    return loom_low_allocation_preference_penalty(&descriptors_, &query, &first,
                                                  &second);
  }

  // One joint preference over three actual values, plus a possible copy source.
  const loom_value_id_t value_ids_[4] = {0, 1, 2, 3};
  // Every value has one liveness interval in ordinal order.
  const uint32_t interval_indices_[4] = {0, 1, 2, 3};
  // Published assignments; absent values remain available for prediction.
  uint32_t assignment_indices_[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                     UINT32_MAX};
  // Canonical mandatory storage identity of each value.
  uint32_t origins_[4] = {0, 1, 2, 3};
  // Incoming structural relations, including defining transfers.
  loom_low_placement_relation_range_t ranges_[4] = {};
  // Outgoing structural relation incidence.
  loom_low_placement_relation_range_t source_ranges_[4] = {};
  // Register footprints of the four values.
  loom_liveness_interval_t intervals_[4] = {};
  // Concrete locations indexed by assignment_indices_.
  loom_low_allocation_assignment_t assignments_[4] = {};
  // Two incompatible linear register banks.
  loom_low_reg_class_t classes_[2] = {};
  // Descriptor storage semantics used by the production predicate interpreter.
  loom_low_descriptor_set_t descriptors_ = {};
  // Resolved target borrowing descriptors_.
  loom_low_resolved_target_t target_ = {};
  // No fixed values unless a test explicitly supplies them.
  loom_low_allocation_target_constraints_t constraints_ = {};
  // Dense value and interval domain borrowed by assignment lookup.
  loom_liveness_analysis_t liveness_ = {};
  // Mutable published assignment state for each preparation.
  loom_low_allocation_assignment_map_t assignment_map_ = {};
  // Structural analysis facts, independent of soft preferences.
  loom_low_placement_table_t placement_ = {};
  // Real instruction input coordinates of the joint preference.
  const loom_low_placement_value_ref_t values_[3] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 1},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 2}};
  // Two violated masked-location relations with distinct peers.
  loom_low_placement_predicate_t predicates_[2] = {
      {0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 3},
      {0, 2, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION,
       3}};
  // Independent ANY and ALL weights over the same two predicates.
  const loom_low_placement_clause_t clauses_[2] = {
      {0, 2, 2, LOOM_LOW_PLACEMENT_CLAUSE_ANY},
      {0, 2, 5, LOOM_LOW_PLACEMENT_CLAUSE_ALL}};
  // Immutable target recipe borrowing the declared spans.
  loom_low_placement_preference_t preference_ = {};
  // One bound instruction with a scheduling priority multiplier.
  loom_low_placement_preference_use_t use_ = {};
  // Real ordinals and first-slot representatives from placement analysis.
  loom_low_placement_preference_binding_t bindings_[3] = {
      {0, 0}, {1, 1}, {2, 2}};
  // The single use occurs once in each distinct origin list.
  const uint32_t use_indices_[3] = {0, 0, 0};
  // Origin incidence prefixes, adjusted when a test joins mandatory origins.
  uint32_t offsets_[5] = {0, 1, 2, 3, 3};
  // Allocation-local bound recipe index.
  loom_low_placement_preference_index_t index_ = {};
  // Reused preparation storage; candidates allocate nothing.
  loom_low_allocation_preference_workspace_t workspace_ = {};
  // Blocks backing the allocation attempt scratch.
  iree_arena_block_pool_t pool_;
  // Allocation-attempt arena.
  iree_arena_allocator_t arena_;
};

TEST_F(LowAllocationPreferenceTest, WeightedClausesAndMergedIncidence) {
  assignment_indices_[2] = 2;
  const auto ordinary = Prepare();
  EXPECT_EQ(Penalty(ordinary, 0),
            6u);  // Unknown second predicate cannot prove ALL.
  EXPECT_EQ(Penalty(ordinary, 1), 0u);
  const auto joint = Prepare(1);
  ASSERT_EQ(joint.use_count, 1u);  // Both candidate origins reference this use.
  EXPECT_EQ(Penalty(joint, 0, 0), 21u);
  EXPECT_EQ(Penalty(joint, 0, 1), 6u);
  EXPECT_EQ(Penalty(joint, 1, 0), 0u);
  const loom_low_placement_clause_t large_weights[] = {
      {0, 2, UINT16_MAX, LOOM_LOW_PLACEMENT_CLAUSE_ANY},
      {0, 2, UINT16_MAX, LOOM_LOW_PLACEMENT_CLAUSE_ALL}};
  preference_.clauses = large_weights;
  use_.priority = UINT16_MAX;
  EXPECT_EQ(Penalty(Prepare(1), 0, 0), UINT32_MAX);
}

TEST_F(LowAllocationPreferenceTest, PublishedPrimaryRemainsTentative) {
  assignment_indices_[0] = 0;
  assignment_indices_[2] = 2;
  assignments_[0].location_base = 0;
  assignments_[2].location_base = 0;
  const auto query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 6u);
  EXPECT_EQ(Penalty(query, 1), 0u);
}

TEST_F(LowAllocationPreferenceTest, InstructionUsesDoNotAffectStorageSearch) {
  index_.instruction_use_count = index_.use_count;
  const auto arena_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_allocation_preference_workspace_initialize(
      &index_, &arena_, &workspace_));
  EXPECT_EQ(arena_.used_allocation_size, arena_bytes);
  EXPECT_EQ(workspace_.use_indices, nullptr);
  EXPECT_FALSE(
      loom_low_allocation_preference_has_uses(&index_, &placement_, 0));
  EXPECT_EQ(Prepare().use_count, 0u);
}

TEST_F(LowAllocationPreferenceTest,
       WholeCopyPredictionYieldsToPublishedMembers) {
  loom_low_placement_relation_t copy = {
      .result_ordinal = 1,
      .source_ordinal = 3,
      .unit_count = 2,
      .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
  };
  placement_.relations = &copy;
  placement_.relation_count = 1;
  ranges_[1] = {0, 1};
  source_ranges_[3] = {0, 1};
  const uint32_t source_index = 0;
  placement_.relation_indices_by_source_ordinal = &source_index;
  assignment_indices_[3] = 3;
  EXPECT_EQ(Penalty(Prepare(), 0), 6u);
  copy.flags = 0;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
  copy.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  copy.unit_count = 1;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
  copy.unit_count = 2;

  // A later bound member has the same mandatory origin; its assignment wins
  // before the first member's permitted-copy prediction is considered.
  origins_[2] = 1;
  bindings_[2].representative = 1;
  offsets_[3] = offsets_[4] = 2;
  assignment_indices_[2] = 2;
  assignments_[2].location_base = 1;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
  EXPECT_EQ(Penalty(Prepare(), 1), 21u);
  assignments_[2].location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
  assignments_[2].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignments_[2].descriptor_reg_class_id = 1;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
}

TEST_F(LowAllocationPreferenceTest, MandatoryIdentityProvesOnlyInvariantMasks) {
  intervals_[1].unit_count = intervals_[2].unit_count = 8;
  origins_[2] = 1;
  bindings_[2].representative = 1;
  offsets_[3] = offsets_[4] = 2;
  preference_.clause_count = 1;
  predicates_[0].result = predicates_[1].result = 1;
  predicates_[0].source = predicates_[1].source = 2;
  predicates_[0].result_unit_offset = predicates_[1].result_unit_offset = 1;
  predicates_[0].source_unit_offset = predicates_[1].source_unit_offset = 5;
  EXPECT_EQ(Penalty(Prepare(), 0), 6u);
  predicates_[0].location_mask = predicates_[1].location_mask = 5;
  use_.memo.location_bit_count = 3;
  EXPECT_EQ(Penalty(Prepare(), 0), 0u);
}

TEST_F(LowAllocationPreferenceTest, MemoKeysBothLocationsAndPresence) {
  assignment_indices_[2] = 2;
  const auto query = Prepare(1);
  ASSERT_NE(query.memo.entries, nullptr);
  EXPECT_EQ(query.memo.index_mask, 3u);
  EXPECT_EQ(Penalty(query, 0, 0), 21u);
  EXPECT_EQ(Penalty(query, 0, 1), 6u);
  EXPECT_EQ(Penalty(query, 0, 0), 21u);
  EXPECT_EQ(Penalty(query, 4, 4), 21u);
  // Absent and present-at-zero have different partial-information costs.
  EXPECT_EQ(loom_low_allocation_preference_penalty(&descriptors_, &query,
                                                   &assignments_[0], nullptr),
            6u);
  EXPECT_EQ(Penalty(query, 0, 0), 21u);
  EXPECT_EQ(loom_low_allocation_preference_penalty(&descriptors_, &query,
                                                   nullptr, &assignments_[0]),
            0u);
  EXPECT_EQ(Penalty(query, 0, 0), 21u);
}

TEST_F(LowAllocationPreferenceTest, MemoCollisionsKeepOffsetCarryBits) {
  // The retained carry-closed mask is wider than the domain-capped memo.
  // Base 0 and base 4 select the same entry but have different exact scores.
  predicates_[0].location_mask = predicates_[1].location_mask = 4;
  predicates_[0].result_unit_offset = predicates_[1].result_unit_offset = 1;
  use_.memo.location_bit_count = 3;
  assignment_indices_[1] = 1;
  assignment_indices_[2] = 2;
  const auto query = Prepare();
  ASSERT_NE(query.memo.entries, nullptr);
  EXPECT_EQ(Penalty(query, 0), 21u);
  EXPECT_EQ(Penalty(query, 3), 0u);
  EXPECT_EQ(Penalty(query, 4), 0u);
  EXPECT_EQ(Penalty(query, 0), 21u);
  // The index remains bounded above the nominal register domain as well.
  EXPECT_EQ(Penalty(query, 32), 21u);
  EXPECT_EQ(Penalty(query, 36), 0u);
}

TEST_F(LowAllocationPreferenceTest, PreparingAgainInvalidatesScores) {
  assignment_indices_[1] = 1;
  assignment_indices_[2] = 2;
  EXPECT_EQ(Penalty(Prepare(), 0), 21u);
  assignments_[2].location_base = 1;
  EXPECT_EQ(Penalty(Prepare(), 0), 6u);
  assignments_[1].location_base = 1;
  const auto used = arena_.used_allocation_size;
  const auto query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 4), 0u);
  EXPECT_EQ(arena_.used_allocation_size, used);
}

TEST_F(LowAllocationPreferenceTest, NonperiodicCostsRemainDirect) {
  predicates_[0].kind = predicates_[1].kind =
      LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE;
  predicates_[0].location_mask = predicates_[1].location_mask = 0;
  use_.memo = {};
  assignment_indices_[1] = 1;
  assignment_indices_[2] = 2;
  const auto query = Prepare();
  EXPECT_EQ(query.memo.entries, nullptr);
  EXPECT_EQ(Penalty(query, 0), 21u);
  EXPECT_EQ(Penalty(query, 4), 0u);
}

TEST_F(LowAllocationPreferenceTest,
       PreferredRegisterOrdinalScoresDisjointClasses) {
  const uint16_t candidate_ids[] = {0, 1, 2, 3};
  const uint16_t candidate_ordinals[] = {0, 1, 0, 1};
  descriptors_.physical_register_candidate_ids = candidate_ids;
  descriptors_.physical_register_candidate_count =
      IREE_ARRAYSIZE(candidate_ids);
  descriptors_.physical_register_candidate_ordinals = candidate_ordinals;
  for (uint16_t i = 0; i < 2; ++i) {
    classes_[i].flags |= LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    classes_[i].allocatable_count = 2;
    classes_[i].physical_register_candidate_start = 2 * i;
    classes_[i].candidate_lookup = {(uint32_t)(2 * i), (uint16_t)(2 * i), 2};
    intervals_[i].value_class.register_class_id = i;
    intervals_[i].unit_count = 1;
    assignments_[i].descriptor_reg_class_id = i;
    assignments_[i].unit_count = assignments_[i].location_count = 1;
  }
  loom_low_placement_relation_t relation = {
      .result_ordinal = 0,
      .source_ordinal = 1,
      .unit_count = 1,
      .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_EDGE_PAIR_AFFINITY,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
      .priority = 3,
  };
  placement_.relations = &relation;
  placement_.relation_count = placement_.location_relation_count = 1;
  ranges_[0] = {0, 1};
  source_ranges_[1] = {0, 1};
  const uint32_t source_index = 0;
  placement_.relation_indices_by_source_ordinal = &source_index;
  assignment_indices_[1] = 1;
  assignments_[1].location_base = 3;
  index_.instruction_use_count = index_.use_count;

  const auto query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 3u);
  EXPECT_EQ(Penalty(query, 1), 0u);
}

class HardOrdinalCopyPreferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    descriptor_set_ = loom_test_low_core_descriptor_set();
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptor_set_, IREE_SV("test.coindexed.partner"), &tuple_class_,
        nullptr));
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptor_set_, IREE_SV("test.packed.narrow"), &copy_class_, nullptr));
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptor_set_, IREE_SV("test.explicit_range32"), &source_class_,
        nullptr));

    intervals_[0].value_class = {
        .type_kind = LOOM_TYPE_REGISTER,
        .register_class_id = tuple_class_,
    };
    intervals_[1].value_class = {
        .type_kind = LOOM_TYPE_REGISTER,
        .register_class_id = copy_class_,
    };
    intervals_[2].value_class = {
        .type_kind = LOOM_TYPE_REGISTER,
        .register_class_id = source_class_,
    };
    intervals_[3].value_class = {
        .type_kind = LOOM_TYPE_REGISTER,
        .register_class_id = copy_class_,
    };
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals_); ++i) {
      intervals_[i].value_id = value_ids_[i];
      intervals_[i].unit_count = 1;
      assignments_[i] = {
          .value_id = value_ids_[i],
          .descriptor_reg_class_id =
              intervals_[i].value_class.register_class_id,
          .unit_count = 1,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_count = 1,
      };
    }
    // The copy source is published. The hard-ordinal peer and its copy result
    // remain unassigned when the tuple candidate is ranked.
    assignment_indices_[2] = 2;
    assignments_[2].location_base = 1;

    liveness_.value_ids = value_ids_;
    liveness_.value_count = IREE_ARRAYSIZE(value_ids_);
    liveness_.intervals = intervals_;
    liveness_.interval_count = IREE_ARRAYSIZE(intervals_);
    liveness_.value_interval_indices = interval_indices_;
    assignment_map_.liveness = &liveness_;
    assignment_map_.assignments = assignments_;
    assignment_map_.assignment_count = IREE_ARRAYSIZE(assignments_);
    assignment_map_.assignment_indices_by_value_ordinal = assignment_indices_;

    relations_[0] = {
        .result_ordinal = 1,
        .source_ordinal = 2,
        .unit_count = 1,
        .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
    };
    relations_[1] = {
        .result_ordinal = 0,
        .source_ordinal = 1,
        .unit_count = 1,
        .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_DESCRIPTOR_CONSTRAINT,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
        .priority = 7,
    };
    // Defining transfers occupy the producer-reserved first result relation.
    result_ranges_[0] = {1, 1};
    result_ranges_[1] = {0, 1};
    // The hard relation is indexed under copy-result origin 1. Identity value
    // 3 has the same origin and can replace its authored endpoint below.
    source_ranges_[1] = {0, 1};
    source_ranges_[2] = {1, 1};
    placement_.relations = relations_;
    placement_.relation_count = IREE_ARRAYSIZE(relations_);
    placement_.location_relation_count = 1;
    placement_.hard_location_relation_count = 1;
    placement_.value_ids = value_ids_;
    placement_.value_count = IREE_ARRAYSIZE(value_ids_);
    placement_.tied_storage_origins_by_value_ordinal = origins_;
    placement_.ranges_by_result_ordinal = result_ranges_;
    placement_.ranges_by_source_ordinal = source_ranges_;
    placement_.relation_indices_by_source_ordinal = source_relation_indices_;

    target_.descriptor_set = descriptor_set_;
    constraints_.target = &target_;
  }

  loom_low_allocation_preference_query_t Prepare() {
    return loom_low_allocation_preference_prepare(
        &preference_index_, &placement_, &assignment_map_, &constraints_, 0,
        LOOM_VALUE_ORDINAL_INVALID, &workspace_);
  }

  uint32_t Penalty(const loom_low_allocation_preference_query_t& query,
                   uint16_t tuple_ordinal) {
    auto candidate = assignments_[0];
    candidate.location_base =
        loom_low_descriptor_set_physical_register_candidate(
            descriptor_set_, tuple_class_, tuple_ordinal);
    return loom_low_allocation_preference_penalty(descriptor_set_, &query,
                                                  &candidate, nullptr);
  }

  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  uint16_t tuple_class_ = LOOM_LOW_REG_CLASS_NONE;
  uint16_t copy_class_ = LOOM_LOW_REG_CLASS_NONE;
  uint16_t source_class_ = LOOM_LOW_REG_CLASS_NONE;
  const loom_value_id_t value_ids_[4] = {0, 1, 2, 3};
  const uint32_t interval_indices_[4] = {0, 1, 2, 3};
  uint32_t assignment_indices_[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                     UINT32_MAX};
  const uint32_t origins_[4] = {0, 1, 2, 1};
  loom_liveness_interval_t intervals_[4] = {};
  loom_low_allocation_assignment_t assignments_[4] = {};
  loom_liveness_analysis_t liveness_ = {};
  loom_low_allocation_assignment_map_t assignment_map_ = {};
  loom_low_placement_relation_t relations_[2] = {};
  loom_low_placement_relation_range_t result_ranges_[4] = {};
  loom_low_placement_relation_range_t source_ranges_[4] = {};
  const uint32_t source_relation_indices_[2] = {1, 0};
  loom_low_placement_table_t placement_ = {};
  loom_low_placement_preference_index_t preference_index_ = {};
  loom_low_allocation_preference_workspace_t workspace_ = {};
  loom_low_resolved_target_t target_ = {};
  loom_low_allocation_target_constraints_t constraints_ = {};
};

TEST_F(HardOrdinalCopyPreferenceTest,
       ProjectsCopyStorageThroughDestinationCandidateOrder) {
  // The source class encodes r0 as candidate 1, while the copy-result class
  // encodes the same physical register as candidate 0.
  assignments_[2].location_base = 1;
  auto query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 7u);

  // Reversing the source to r1 reverses the preferred tuple without assuming
  // that the two classes share candidate ordinals or location encodings.
  assignments_[2].location_base = 0;
  query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 7u);
  EXPECT_EQ(Penalty(query, 1), 0u);
}

TEST_F(HardOrdinalCopyPreferenceTest,
       UsesIdentityForwardedPeerInEitherRelationDirection) {
  relations_[1].source_ordinal = 3;
  auto query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 7u);

  relations_[1].result_ordinal = 3;
  relations_[1].source_ordinal = 0;
  result_ranges_[0] = {};
  result_ranges_[1] = {0, 2};
  source_ranges_[0] = {0, 1};
  source_ranges_[1] = {};
  query = Prepare();
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 7u);
}

TEST_F(HardOrdinalCopyPreferenceTest,
       IgnoresUnavailableAndNonAliasCopyPredictions) {
  // Source candidate 3 is r2, which the two-register copy class cannot name.
  assignments_[2].location_base = 3;
  auto query = Prepare();
  EXPECT_EQ(query.structural.placement, nullptr);
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 0u);

  assignments_[2].location_base = 1;
  relations_[0].flags = 0;
  query = Prepare();
  EXPECT_EQ(query.structural.placement, nullptr);
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 0u);
}

TEST_F(HardOrdinalCopyPreferenceTest, LeavesPublishedPeerToHardLegality) {
  assignment_indices_[1] = 1;
  assignments_[1].location_base =
      loom_low_descriptor_set_physical_register_candidate(descriptor_set_,
                                                          copy_class_, 1);
  const auto query = Prepare();
  EXPECT_EQ(query.structural.placement, nullptr);
  EXPECT_EQ(Penalty(query, 0), 0u);
  EXPECT_EQ(Penalty(query, 1), 0u);
}

TEST_F(LowAllocationPreferenceTest, EmptyIndexAllocatesNothing) {
  loom_low_allocation_preference_workspace_t empty = {};
  loom_low_placement_preference_index_t index = {};
  const auto used = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_allocation_preference_workspace_initialize(
      &index, &arena_, &empty));
  EXPECT_EQ(empty.use_indices, nullptr);
  EXPECT_EQ(empty.locations, nullptr);
  EXPECT_EQ(empty.memo_entries, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, used);
}

}  // namespace
}  // namespace loom
