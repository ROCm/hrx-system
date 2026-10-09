// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/numbering.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/storage_lease_index.h"

namespace loom {
namespace {

class LowAllocationNumberingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    for (auto& reg_class : classes_) {
      reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
      reg_class.alloc_unit_bits = 32;
      reg_class.allocatable_count = 16;
    }
    descriptors_.reg_classes = classes_;
    descriptors_.reg_class_count = IREE_ARRAYSIZE(classes_);
    target_.descriptor_set = &descriptors_;
    constraints_.target = &target_;
    constraints_.max_assigned_location_end_by_reg_class = extents_;
    liveness_.value_count = IREE_ARRAYSIZE(assignments_);
    liveness_.interval_count = IREE_ARRAYSIZE(intervals_);
    liveness_.intervals = intervals_;
    liveness_.value_interval_indices = indices_;
    allocation_.assignments = assignments_;
    allocation_.assignment_count = IREE_ARRAYSIZE(assignments_);
    allocation_.assignment_indices_by_value_ordinal = indices_;
    allocation_.assignment_map.liveness = &liveness_;
    allocation_.assignment_map.assignments = assignments_;
    allocation_.assignment_map.assignment_count = IREE_ARRAYSIZE(assignments_);
    allocation_.assignment_map.assignment_indices_by_value_ordinal = indices_;
    placement_.operand_constraints_by_interval = operands_;
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(assignments_); ++i) {
      auto& assignment = assignments_[i];
      assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
      assignment.unit_count = assignment.location_count = i < 2 ? 4 : 1;
      assignment.location_base = i < 2 ? i * 4 : i + 6;
      intervals_[i].unit_count = assignment.unit_count;
      intervals_[i].value_class.type_kind = LOOM_TYPE_REGISTER;
    }
    preference_ = {values_, &predicate_, &clause_, 2, 1};
    use_ = {&preference_, 0, 1, {2, 3}};
    preferences_.uses = &use_;
    preferences_.use_count = preferences_.instruction_use_count = 1;
    preferences_.bindings = bindings_;
    preferences_.binding_count = 2;
    context_.placement = &placement_;
    context_.preferences = &preferences_;
    context_.target_constraints = &constraints_;
    context_.unit_liveness = &unit_liveness_;
    context_.interval_assignment = &allocation_;
    context_.storage_leases = &leases_;
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Number() {
    IREE_ASSERT_OK(loom_low_allocation_number_registers(&context_, &arena_));
  }

  void ExpectImproved() {
    EXPECT_NE(assignments_[0].location_base & 3,
              assignments_[1].location_base & 3);
    ExpectFootprint();
  }

  void ExpectIdentity() {
    const uint32_t expected[] = {0, 4, 8, 9};
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(assignments_); ++i) {
      EXPECT_EQ(assignments_[i].location_base, expected[i]);
    }
    ExpectFootprint();
  }

  void ExpectFootprint() {
    std::array<bool, 10> occupied = {};
    for (const auto& assignment : assignments_) {
      ASSERT_LE(assignment.location_base + assignment.location_count,
                extents_[assignment.descriptor_reg_class_id]);
      for (uint32_t i = 0; i < assignment.location_count; ++i) {
        const auto unit = assignment.location_base + i;
        ASSERT_LT(unit, occupied.size());
        EXPECT_FALSE(occupied[unit]);
        occupied[unit] = true;
      }
    }
    for (bool live : occupied) {
      EXPECT_TRUE(live);
    }
    EXPECT_EQ(extents_[0], 10u);
  }

  // Two independently addressable physical classes, aliased by specific tests.
  loom_low_reg_class_t classes_[2] = {};
  // Target storage descriptors borrowed throughout numbering.
  loom_low_descriptor_set_t descriptors_ = {};
  // Resolved descriptor owner.
  loom_low_resolved_target_t target_ = {};
  // Original published class footprints; numbering cannot grow either.
  uint32_t extents_[2] = {10, 10};
  // Fixed and reserved locations, supplied by individual tests.
  loom_low_allocation_target_constraints_t constraints_ = {};
  // Two four-unit tuples followed by two independent scalar allocations.
  loom_low_allocation_assignment_t assignments_[4] = {};
  // One interval per assignment, with identical dense value ordinals.
  loom_liveness_interval_t intervals_[4] = {};
  // Shared identity ordinal indexes for intervals and assignments.
  uint32_t indices_[4] = {0, 1, 2, 3};
  // Retained interval domain used by the real assignment result.
  loom_liveness_analysis_t liveness_ = {};
  // Mutable completed storage assignment.
  loom_low_allocation_interval_assignment_result_t allocation_ = {};
  // Per-interval operand legality retained by placement.
  loom_low_placement_operand_constraints_t operands_[4] = {};
  // Hard relations and indexed operand facts.
  loom_low_placement_table_t placement_ = {};
  // Producer-owned implicit physical coordinates.
  loom_low_allocation_unit_liveness_t unit_liveness_ = {};
  // Optional asynchronous storage leases.
  loom_low_allocation_storage_lease_state_t leases_ = {};
  // Instruction coordinates corresponding to the two tuple assignments.
  const loom_low_placement_value_ref_t values_[2] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 1}};
  // The tuples initially share their two low location bits.
  loom_low_placement_predicate_t predicate_ = {
      0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 3};
  // One unit penalty when both tuples share a masked location.
  const loom_low_placement_clause_t clause_ = {0, 1, 1,
                                               LOOM_LOW_PLACEMENT_CLAUSE_ANY};
  // Descriptor recipe consumed directly by production numbering.
  loom_low_placement_preference_t preference_ = {};
  // One deferred instruction use.
  loom_low_placement_preference_use_t use_ = {};
  // Actual value ordinals bound by placement.
  loom_low_placement_preference_binding_t bindings_[2] = {{0, 0}, {1, 1}};
  // Bound instruction and optional scheduled-pair preferences.
  loom_low_placement_preference_index_t preferences_ = {};
  // Finalization boundary borrowing all producer state above.
  loom_low_allocation_numbering_context_t context_ = {};
  // Scratch block owner for numbering and optional lease indexing.
  iree_arena_block_pool_t pool_;
  // Numbering lifetime; no candidate allocates separately.
  iree_arena_allocator_t arena_;
};

TEST_F(LowAllocationNumberingTest, ImprovesCostWithoutGrowingStorage) {
  Number();
  ExpectImproved();
}

TEST_F(LowAllocationNumberingTest, EntryIdentitiesKeepExternalCoordinates) {
  const loom_low_allocation_abi_location_t entry[] = {
      {.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       .descriptor_reg_class_id = 0,
       .location_base = 0},
      {.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       .descriptor_reg_class_id = 0,
       .location_base = 4},
  };
  context_.entry_locations = entry;
  context_.entry_location_count = IREE_ARRAYSIZE(entry);
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, EntryMoveSourceKeepsExternalCoordinates) {
  const loom_low_allocation_abi_location_t entry[] = {
      {},
      {},
      {.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       .descriptor_reg_class_id = 0,
       .location_base = 4}};
  loom_low_move_t move = {};
  move.source.location_kind = move.destination.location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  move.source.location = 4;
  move.destination.location = 8;
  context_.entry_locations = entry;
  context_.entry_location_count = IREE_ARRAYSIZE(entry);
  context_.moves = &move;
  context_.move_count = 1;
  Number();
  EXPECT_EQ(move.source.location, 4u);
  EXPECT_EQ(move.destination.location, assignments_[2].location_base);
  ExpectFootprint();
}

TEST_F(LowAllocationNumberingTest, UpdatesLeasesAndCycleScratchTogether) {
  loom_low_allocation_storage_lease_t instances[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    instances[i].location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    instances[i].location_base = 4 + i;
    instances[i].location_count = 3 - i;
    instances[i].start_point = 1 + i;
    instances[i].end_point = 12;
  }
  loom_low_allocation_storage_lease_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
      &index, instances, 2, 5, 3, &arena_));
  for (uint32_t i = 0; i < 2; ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i,
                                                        /*lease_flags=*/0);
  }
  const auto node_count = index.node_count;
  const auto* nodes = index.nodes;
  leases_.instances = instances;
  leases_.instance_count = 2;
  leases_.unit_index = &index;
  loom_low_move_t moves[2] = {};
  moves[0].source.location_kind = moves[0].destination.location_kind =
      moves[1].source.location_kind = moves[1].destination.location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  moves[0].source.location = 5;
  moves[0].destination.location = moves[1].source.location = 8;
  moves[1].destination.location = 9;
  context_.moves = moves;
  context_.move_count = 2;
  Number();
  ExpectImproved();
  EXPECT_EQ(instances[0].location_base, assignments_[1].location_base);
  EXPECT_EQ(instances[1].location_base, assignments_[1].location_base + 1);
  EXPECT_EQ(moves[0].source.location, assignments_[1].location_base + 1);
  EXPECT_EQ(moves[0].destination.location, assignments_[2].location_base);
  EXPECT_EQ(moves[1].source.location, assignments_[2].location_base);
  EXPECT_EQ(moves[1].destination.location, assignments_[3].location_base);
  EXPECT_EQ(index.nodes, nodes);
  EXPECT_EQ(index.node_count, node_count);
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      &index, &descriptors_, 0, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      instances[1].location_base, 1, 0, 20, nullptr, &query);
  uint32_t ordinal;
  uint32_t membership = 0;
  while (loom_low_allocation_storage_lease_unit_query_next(&query, &ordinal)) {
    membership |= 1u << ordinal;
  }
  EXPECT_EQ(membership, 3u);
}

TEST_F(LowAllocationNumberingTest, PreservesSatisfiedScheduledPair) {
  const loom_low_placement_predicate_t pair_predicates[] = {
      {0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 1},
      {0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE, 0}};
  const loom_low_placement_clause_t pair_clause = {
      0, 2, 1, LOOM_LOW_PLACEMENT_CLAUSE_ANY};
  const loom_low_placement_preference_t pair = {values_, pair_predicates,
                                                &pair_clause, 2, 1};
  const loom_low_placement_preference_use_t uses[] = {use_,
                                                      {&pair, 2, 1, {1, 0}}};
  const loom_low_placement_preference_binding_t bindings[] = {
      {0, 0}, {1, 1}, {2, 0}, {3, 1}};
  preferences_.uses = uses;
  preferences_.use_count = 2;
  preferences_.bindings = bindings;
  preferences_.binding_count = 4;
  Number();
  ExpectImproved();
  EXPECT_NE(assignments_[2].location_base & 1,
            assignments_[3].location_base & 1);
}

TEST_F(LowAllocationNumberingTest, HonorsOperandAlignmentAndAddressability) {
  operands_[1].addressable_unit_count = 8;
  Number();
  ExpectIdentity();
  operands_[1].addressable_unit_count = 0;
  operands_[0].unit_alignment_log2 = operands_[1].unit_alignment_log2 = 2;
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, HonorsEachAliasedClassExtent) {
  classes_[0].alias_set_id = classes_[1].alias_set_id = 1;
  assignments_[1].descriptor_reg_class_id = 1;
  extents_[1] = 8;
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, ImprovesCostsAcrossAliasedClasses) {
  classes_[0].alias_set_id = classes_[1].alias_set_id = 1;
  assignments_[1].descriptor_reg_class_id = 1;
  Number();
  ExpectImproved();
}

TEST_F(LowAllocationNumberingTest, LeavesIndependentStorageUnchanged) {
  assignments_[3].descriptor_reg_class_id = 1;
  assignments_[3].location_base = 0;
  extents_[0] = 9;
  extents_[1] = 1;
  Number();
  EXPECT_NE(assignments_[0].location_base & 3,
            assignments_[1].location_base & 3);
  EXPECT_EQ(assignments_[3].location_base, 0u);
  EXPECT_EQ(extents_[0], 9u);
  EXPECT_EQ(extents_[1], 1u);
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_LE(assignments_[i].location_base + assignments_[i].location_count,
              extents_[0]);
  }
}

TEST_F(LowAllocationNumberingTest, IndependentClassesDoNotConflict) {
  assignments_[1].descriptor_reg_class_id = 1;
  extents_[1] = 8;
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, OverlappingLeaseJoinsRigidBlocks) {
  loom_low_allocation_storage_lease_t instance = {
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 3,
      .location_count = 2};
  leases_.instances = &instance;
  leases_.instance_count = 1;
  Number();
  // A lease crossing the tuple boundary prevents an insertion between them.
  EXPECT_EQ(assignments_[1].location_base, assignments_[0].location_base + 4);
  EXPECT_EQ(instance.location_base, assignments_[0].location_base + 3);
  ExpectFootprint();
}

TEST_F(LowAllocationNumberingTest, ReservedRangeKeepsItsOriginalCoordinates) {
  loom_low_allocation_resolved_reserved_range_t reserved = {
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 8,
      .location_count = 1};
  constraints_.reserved_ranges = &reserved;
  constraints_.reserved_range_count = 1;
  // The scalar at the reserved coordinate represents its ABI-fixed occupant.
  loom_low_allocation_resolved_fixed_value_t fixed = {.assignment =
                                                          assignments_[2]};
  constraints_.fixed_values = &fixed;
  constraints_.fixed_value_count = 1;
  Number();
  EXPECT_EQ(assignments_[2].location_base, 8u);
  ExpectFootprint();
}

TEST_F(LowAllocationNumberingTest, KeepsFixedAndImplicitLocationsAnchored) {
  loom_low_allocation_resolved_fixed_value_t fixed = {.assignment =
                                                          assignments_[0]};
  constraints_.fixed_values = &fixed;
  constraints_.fixed_value_count = 1;
  uint16_t implicit_counts[] = {1, 0};
  unit_liveness_.implicit_location_counts_by_reg_class = implicit_counts;
  Number();
  ExpectImproved();
  EXPECT_EQ(assignments_[0].location_base, 0u);
}

TEST_F(LowAllocationNumberingTest, KeepsTargetAddressStateUnchanged) {
  operands_[0].has_target_address_state = true;
  operands_[1].has_target_address_state = true;
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, AnchorsHardMaskedRelations) {
  loom_low_placement_relation_t relation = {};
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION;
  relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD;
  relation.result_ordinal = 0;
  relation.source_ordinal = 1;
  relation.source_unit_offset = 1;
  relation.location_mask = 3;
  relation.unit_count = 1;
  placement_.relations = &relation;
  placement_.relation_count = 1;
  Number();
  ExpectIdentity();
}

TEST_F(LowAllocationNumberingTest, DoesNothingWithoutInstructionUses) {
  preferences_.instruction_use_count = 0;
  const auto allocated = arena_.total_allocation_size;
  Number();
  ExpectIdentity();
  EXPECT_EQ(arena_.total_allocation_size, allocated);
}

TEST_F(LowAllocationNumberingTest, KeepsAlreadySatisfiedCosts) {
  predicate_.source_unit_offset = 1;
  Number();
  ExpectIdentity();
}

}  // namespace
}  // namespace loom
