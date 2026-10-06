// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/read_retention.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"
#include "loom/util/cfg_graph.h"

namespace loom {
namespace {

enum class WriteKind { None, Copy, Instruction };
enum class BindingKind { Free, AliasedDestination, ForcedWrite };

class WriteInterferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &module_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    iree_arena_block_pool_initialize(4096, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &decision_);
    iree_arena_initialize(&pool_, &scratch_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&decision_);
    iree_arena_block_pool_deinitialize(&pool_);
    EXPECT_EQ(live_allocations_, 0u);
    if (loom_local_value_domain_is_acquired(&domain_)) {
      loom_local_value_domain_release(&domain_);
    }
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&module_pool_);
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<WriteInterferenceTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected backing allocation failure");
    }
    const bool freed = command == IREE_ALLOCATOR_COMMAND_FREE && *pointer;
    const auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      if (freed) {
        --test->live_allocations_;
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE) {
        ++test->live_allocations_;
      }
    }
    return status;
  }

  // A retained read, memory result and wide copy produce one conditional row
  // per copied unit. The source is defined after the retained input dies, so
  // zero-copy placements also satisfy ordinary liveness. Liveness/placement
  // come from their production producers instead of hand-authored tables.
  void Initialize(uint32_t width, WriteKind kind) {
    kind_ = kind;
    register_class_.name_string_ref = LOOM_STRING_REF(0, 1);
    register_class_.alloc_unit_bits = 32;
    register_class_.allocatable_count = 32768;
    register_class_.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
    operands_[0].role = LOOM_LOW_OPERAND_ROLE_PREDICATE;
    operands_[1].role = LOOM_LOW_OPERAND_ROLE_RESULT;
    for (auto& operand : operands_) {
      operand.source_value_index = 0;
      operand.reg_class_alt_count = 1;
      operand.unit_count = width;
    }
    alternative_.reg_class_id = 0;
    alternative_.register_part_id = LOOM_LOW_REGISTER_PART_NONE;
    for (uint16_t i = 0; i < 3; ++i) {
      descriptors_[i].operand_start = i == 0 ? 0 : 1;
      descriptors_[i].operand_count = 1;
      descriptors_[i].result_count = i == 0 ? 0 : 1;
    }
    views_[0].instruction_class_flags =
        LOOM_LOW_INSTRUCTION_CLASS_FLAG_VECTOR_ALU;
    views_[1].instruction_class_flags =
        LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_ALU;
    views_[2].instruction_class_flags =
        LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_MEMORY;
    descriptor_set_.stable_id = 1;
    descriptor_set_.string_pool = {"r", 1};
    descriptor_set_.reg_classes = &register_class_;
    descriptor_set_.reg_class_count = 1;
    descriptor_set_.reg_class_alts = &alternative_;
    descriptor_set_.reg_class_alt_count = 1;
    descriptor_set_.operands = operands_;
    descriptor_set_.operand_count = 2;
    descriptor_set_.descriptors = descriptors_;
    descriptor_set_.descriptor_views = views_;
    descriptor_set_.descriptor_count = 3;
    rule_.register_class = IREE_SV("r");
    rule_.subgroup_size = 64;
    rule_.reader_classes = views_[0].instruction_class_flags;
    rule_.writer_classes = views_[1].instruction_class_flags;
    rule_.retained_operand_role = operands_[0].role;
    facts_.storage.snapshot.subgroup_size = rule_.subgroup_size;
    facts_.read_retention = &rule_;
    target_.target_facts = &facts_;
    target_.descriptor_set = &descriptor_set_;

    loom_type_t type = loom_low_register_type(1, 0, width);
    IREE_ASSERT_OK(loom_module_intern_type(module_, type, &type));
    IREE_ASSERT_OK(loom_module_define_value(module_, type, &values_[0]));
    IREE_ASSERT_OK(
        loom_block_add_arg(module_, loom_module_block(module_), values_[0]));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder, &descriptor_set_, &descriptors_[0], 0, values_, 1, {},
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &reader_));
    loom_op_t* writer = nullptr;
    if (kind == WriteKind::Copy) {
      loom_op_t* source = nullptr;
      IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
          &builder, &descriptor_set_, &descriptors_[2], 0, nullptr, 0, {},
          &type, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &source));
      values_[1] = loom_op_results(source)[0];
      IREE_ASSERT_OK(loom_low_copy_build(&builder, values_[1], false, type,
                                         LOOM_LOCATION_UNKNOWN, &writer));
    } else if (kind == WriteKind::Instruction) {
      IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
          &builder, &descriptor_set_, &descriptors_[1], 0, nullptr, 0, {},
          &type, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &writer));
    }
    const uint32_t value_count =
        kind == WriteKind::Copy ? 3 : 1 + (writer != nullptr);
    if (writer) {
      values_[value_count - 1] = loom_op_results(writer)[0];
    }
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, module_->body, &module_->arena, &domain_));
    IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
        &domain_, loom_liveness_order_empty(), &module_->arena, &liveness_));
    IREE_ASSERT_OK(
        loom_cfg_graph_build(module_, module_->body, &module_->arena, &graph_));
    loom_low_placement_preference_index_t preferences = {};
    IREE_ASSERT_OK(loom_low_placement_analyze_region(
        module_, module_->body, &descriptor_set_, &domain_, &liveness_, {}, {},
        &module_->arena, &module_->arena, &placement_, &preferences));
    if (kind == WriteKind::Copy) {
      ASSERT_EQ(placement_.relation_count, 1u);
      ASSERT_EQ(placement_.relations[0].unit_count, width);
    }
    for (uint32_t i = 0; i < value_count; ++i) {
      ASSERT_EQ(loom_local_value_domain_ordinal(&domain_, values_[i]), i);
      assignment_indices_[i] = i;
      const auto* interval =
          loom_liveness_interval_for_value_ordinal(&liveness_, i);
      ASSERT_NE(interval, nullptr);
      auto& assignment = assignments_[i];
      assignment.value_id = values_[i];
      assignment.descriptor_reg_class_id =
          interval->value_class.register_class_id;
      assignment.start_point = interval->start_point;
      assignment.end_point = interval->end_point + 1;
      assignment.unit_count = assignment.location_count = width;
      assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    }
    assignments_[0].location_base = width + 10;
    map_.module = module_;
    map_.liveness = &liveness_;
    map_.assignments = assignments_;
    map_.assignment_count = value_count;
    map_.assignment_indices_by_value_ordinal = assignment_indices_;
  }

  void ResetArenas(iree_host_size_t failure_index = SIZE_MAX) {
    failure_index_ = SIZE_MAX;
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&decision_);
    iree_arena_block_pool_deinitialize(&pool_);
    ASSERT_EQ(live_allocations_, 0u);
    iree_arena_block_pool_initialize(4096, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &decision_);
    iree_arena_initialize(&pool_, &scratch_);
    IREE_ASSERT_OK(iree_arena_allocate(&scratch_, 16, (void**)&sentinel_));
    memset(sentinel_, 0x5A, 16);
    scratch_checkpoint_ = iree_arena_checkpoint_save(&scratch_);
    allocation_count_ = 0;
    failure_index_ = failure_index;
    table_ = nullptr;
  }

  iree_status_t BuildTable(BindingKind binding = BindingKind::Free) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_write_interference_create(
        &target_, &placement_, &liveness_, &decision_, &table_));
    IREE_RETURN_IF_ERROR(loom_low_allocation_write_interference_note_operand(
        table_, &domain_, &descriptor_set_, &descriptors_[0], reader_, 0,
        loom_liveness_operation_at(&liveness_, 0)->start_point + 1,
        &decision_));
    if (kind_ != WriteKind::None) {
      const auto* descriptor = &descriptors_[kind_ == WriteKind::Copy ? 2 : 1];
      IREE_RETURN_IF_ERROR(loom_low_allocation_write_interference_note_operand(
          table_, &domain_, &descriptor_set_, descriptor,
          loom_liveness_operation_at(&liveness_, 1)->op, 0,
          loom_liveness_operation_at(&liveness_, 1)->start_point + 1,
          &decision_));
    }
    if (binding != BindingKind::Free) {
      assignments_[2].location_base = assignments_[0].location_base;
      for (uint32_t ordinal : {0u, 2u}) {
        loom_low_allocation_write_interference_note_fixed(
            table_, ordinal, &assignments_[ordinal]);
      }
      if (binding == BindingKind::ForcedWrite) {
        loom_low_allocation_write_interference_note_fixed(table_, 1,
                                                          &assignments_[1]);
      }
    }
    return loom_low_allocation_write_interference_finalize(
        table_, &liveness_, &graph_, &placement_, &decision_, &scratch_);
  }

  void ExpectScratchPreserved() {
    EXPECT_EQ(scratch_.block_head, scratch_checkpoint_.block_head);
    EXPECT_EQ(scratch_.allocation_head, scratch_checkpoint_.allocation_head);
    EXPECT_EQ(scratch_.used_allocation_size,
              scratch_checkpoint_.used_allocation_size);
    EXPECT_EQ(scratch_.total_allocation_size,
              scratch_checkpoint_.total_allocation_size);
    EXPECT_EQ(scratch_.block_bytes_remaining,
              scratch_checkpoint_.block_bytes_remaining);
    for (uint32_t i = 0; i < 16; ++i) {
      EXPECT_EQ(sentinel_[i], 0x5A);
    }
  }

  loom_value_ordinal_t Query(uint32_t ordinal, uint32_t base) {
    auto candidate = assignments_[ordinal];
    candidate.location_base = base;
    return loom_low_allocation_write_interference_conflicting_read(
        table_, &map_, &candidate);
  }

  void Publish(uint32_t ordinal, uint32_t base) {
    assignment_indices_[ordinal] = ordinal;
    assignments_[ordinal].location_base = base;
    loom_low_allocation_write_interference_note_assignment(
        table_, ordinal, &assignments_[ordinal]);
  }

  // Source module and value-domain lifetime, independent of injected failures.
  iree_arena_block_pool_t module_pool_ = {};
  // Registered Low operations used by the small API fixture.
  loom_context_t context_ = {};
  // Module owning actual register-typed values and operations.
  loom_module_t* module_ = nullptr;
  // Acquired module ordinal map through all candidate queries.
  loom_local_value_domain_t domain_ = {};
  // Shared pool for construction scratch and decision facts.
  iree_arena_block_pool_t pool_ = {};
  // Persistent retained-row and query-index owner.
  iree_arena_allocator_t decision_ = {};
  // Temporary construction owner restored after finalization.
  iree_arena_allocator_t scratch_ = {};
  // Pre-existing caller state in the construction arena.
  uint8_t* sentinel_ = nullptr;
  // Exact construction-arena state before finalization.
  iree_arena_checkpoint_t scratch_checkpoint_ = {};
  // Backing allocation attempts since the latest reset.
  iree_host_size_t allocation_count_ = 0;
  // Request ordinal to fail, or SIZE_MAX for no injected failure.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Outstanding successful backing allocations in the tested pool.
  iree_host_size_t live_allocations_ = 0;
  // Selected transfer semantics.
  WriteKind kind_ = WriteKind::None;
  // Linear physical bank with room for all tested placements.
  loom_low_reg_class_t register_class_ = {};
  // Shared register alternative for the read and write descriptors.
  loom_low_reg_class_alt_t alternative_ = {};
  // Retained input and unconditional result operand contracts.
  loom_low_operand_t operands_[2] = {};
  // Reader, interfering ALU result and asynchronously completed memory result.
  loom_low_descriptor_t descriptors_[3] = {};
  // Per-descriptor read and write instruction class facts.
  loom_low_descriptor_view_t views_[3] = {};
  // Descriptor ownership and dense row domains.
  loom_low_descriptor_set_t descriptor_set_ = {};
  // Retained-read rule for the fixture's linear bank.
  loom_low_read_retention_t rule_ = {};
  // Matching subgroup facts activating the rule.
  loom_target_facts_t facts_ = {};
  // Bound target and descriptor set used by the production constructor.
  loom_low_resolved_target_t target_ = {};
  // Actual descriptor-backed retained read.
  loom_op_t* reader_ = nullptr;
  // Retained input, optional memory result and optional destination.
  loom_value_id_t values_[3] = {};
  // Real liveness facts consumed by retained-write construction.
  loom_liveness_analysis_t liveness_ = {};
  // Real control-flow graph for the fixture's straight-line block.
  loom_cfg_graph_t graph_ = {};
  // Structural facts in the same local ordinal domain.
  loom_low_placement_table_t placement_ = {};
  // Current physical assignments, mutable between independent queries.
  loom_low_allocation_assignment_t assignments_[3] = {};
  // Published assignment indices, or UINT32_MAX for unassigned endpoints.
  uint32_t assignment_indices_[3] = {};
  // Assignment consumer view with module ordinal ownership intact.
  loom_low_allocation_assignment_map_t map_ = {};
  // Published constructor result; usable only after successful finalization.
  loom_low_allocation_write_interference_t* table_ = nullptr;
};

class WriteInterferenceBoundaryTest
    : public WriteInterferenceTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(WriteInterferenceBoundaryTest,
       StableQueriesAcrossSegmentsAndScratchReuse) {
  Initialize(GetParam(), WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  ExpectScratchPreserved();
  for (uint32_t i = 0; i < 32; ++i) {
    void* overwritten = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&scratch_, 1024, &overwritten));
    memset(overwritten, 0xCC, 1024);
  }
  // Only the last copied unit overlaps. Every preceding row must remain
  // addressable, and dropping the partial final segment would miss the hazard.
  EXPECT_EQ(Query(2, 11), 0u);
  EXPECT_EQ(Query(2, 10), LOOM_VALUE_ORDINAL_INVALID);
  // Equal locations remove the write even where the last unit overlaps the
  // retained range. This observes the source offset in that final row.
  Publish(1, 11);
  EXPECT_EQ(Query(2, 11), LOOM_VALUE_ORDINAL_INVALID);
}

INSTANTIATE_TEST_SUITE_P(SegmentEdges, WriteInterferenceBoundaryTest,
                         ::testing::Values(1u, 127u, 128u, 129u, 2048u, 2049u));

TEST_F(WriteInterferenceTest, ReadWithoutSubsequentWrites) {
  Initialize(2, WriteKind::None);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  ExpectScratchPreserved();
  EXPECT_EQ(Query(0, 0), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest, UnconditionalWriteKeepsItsFullExtent) {
  Initialize(4, WriteKind::Instruction);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  EXPECT_EQ(Query(1, 11), 0u);
  EXPECT_EQ(Query(1, 10), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_P(WriteInterferenceBoundaryTest,
       ForcedAliasRetainsBidirectionalCopyEquation) {
  Initialize(GetParam(), WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable(BindingKind::AliasedDestination));
  ExpectScratchPreserved();
  EXPECT_EQ(Query(1, assignments_[0].location_base),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(Query(1, assignments_[0].location_base + 1), 0u);
  loom_low_allocation_write_interference_reset_inference(table_);
  assignment_indices_[1] = UINT32_MAX;
  EXPECT_EQ(Query(2, assignments_[0].location_base),
            LOOM_VALUE_ORDINAL_INVALID);
}

TEST_P(WriteInterferenceBoundaryTest, CompletedCopiesDoNotEnterTheFinalIndex) {
  Initialize(GetParam(), WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable(BindingKind::ForcedWrite));
  ExpectScratchPreserved();
  EXPECT_EQ(Query(2, assignments_[0].location_base),
            LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest, SimultaneousProposalUsesTheCompleteLocationSet) {
  Initialize(129, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  loom_low_allocation_write_proposal_t proposal = {};
  IREE_ASSERT_OK(loom_low_allocation_write_proposal_initialize(
      table_, &decision_, &proposal));
  for (uint32_t i = 0; i < 3; ++i) {
    loom_low_allocation_write_proposal_add(table_, &map_, &assignments_[i], 10,
                                           &proposal);
  }
  EXPECT_FALSE(
      loom_low_allocation_write_proposal_conflicts(table_, &map_, &proposal));
  loom_low_allocation_write_proposal_add(table_, &map_, &assignments_[1], 11,
                                         &proposal);
  EXPECT_TRUE(
      loom_low_allocation_write_proposal_conflicts(table_, &map_, &proposal));
  loom_low_allocation_write_proposal_reset(&proposal);
  EXPECT_EQ(proposal.count, 0u);
  EXPECT_EQ(Query(2, 11), 0u);
}

TEST_F(WriteInterferenceTest, PublicationInvalidatesAnIncompatibleWitness) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  assignment_indices_[1] = UINT32_MAX;
  assignment_indices_[2] = UINT32_MAX;
  const uint32_t retained_base = assignments_[0].location_base;
  EXPECT_EQ(Query(2, retained_base), LOOM_VALUE_ORDINAL_INVALID);
  // The unknown source was inferred at the destination. Publishing a distinct
  // source invalidates that witness even when the candidate itself repeats.
  Publish(1, retained_base + 1);
  EXPECT_EQ(Query(2, retained_base), 0u);
}

TEST_F(WriteInterferenceTest, RecoloringProofDoesNotReplaceCommittedLocation) {
  Initialize(16, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  assignment_indices_[1] = UINT32_MAX;
  Publish(0, 0);
  Publish(2, 2);
  // Both destination locations overlap the retained range. The first query
  // proves source=4 only under destination=4, without committing that move.
  EXPECT_EQ(Query(2, 4), LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_EQ(Query(1, 4), 0u);
  EXPECT_EQ(Query(1, 2), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest,
       RecoloringProofDoesNotReplaceFutureFixedLocation) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable(BindingKind::AliasedDestination));
  assignment_indices_[1] = UINT32_MAX;
  assignment_indices_[2] = UINT32_MAX;
  const uint32_t retained_base = assignments_[0].location_base;
  EXPECT_EQ(Query(2, retained_base + 2), LOOM_VALUE_ORDINAL_INVALID);
  // The destination is not yet assigned, but its fixed binding still governs
  // a later query that is no longer hypothetically moving that destination.
  EXPECT_EQ(Query(1, retained_base + 2), 0u);
  EXPECT_EQ(Query(1, retained_base), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest, ProposalDiscardsASeparateCandidateWitness) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  Publish(1, 1);
  EXPECT_EQ(Query(2, 10), LOOM_VALUE_ORDINAL_INVALID);
  // Moving only the retained input must inspect the actual destination at 0,
  // not the speculative destination at 10 in the preceding candidate query.
  loom_low_allocation_write_proposal_t proposal = {};
  IREE_ASSERT_OK(loom_low_allocation_write_proposal_initialize(
      table_, &decision_, &proposal));
  loom_low_allocation_write_proposal_add(table_, &map_, &assignments_[0], 0,
                                         &proposal);
  EXPECT_TRUE(
      loom_low_allocation_write_proposal_conflicts(table_, &map_, &proposal));
  EXPECT_EQ(Query(2, 0), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest, MatchingPublicationAndSpillsPreserveFeasibility) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  assignment_indices_[1] = UINT32_MAX;
  assignment_indices_[2] = UINT32_MAX;
  const uint32_t retained_base = assignments_[0].location_base;
  EXPECT_EQ(Query(2, retained_base), LOOM_VALUE_ORDINAL_INVALID);
  Publish(1, retained_base);
  EXPECT_EQ(Query(1, retained_base), LOOM_VALUE_ORDINAL_INVALID);
  assignments_[1].location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  Publish(1, 0);
  EXPECT_EQ(Query(2, retained_base), LOOM_VALUE_ORDINAL_INVALID);
}

TEST_F(WriteInterferenceTest, RetainedPublicationActivatesAConditionalWrite) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  assignment_indices_[0] = UINT32_MAX;
  EXPECT_EQ(Query(2, 10), LOOM_VALUE_ORDINAL_INVALID);
  // An endpoint outside the prior inference can activate an incident row.
  Publish(0, 10);
  EXPECT_EQ(Query(2, 10), 0u);
}

TEST_F(WriteInterferenceTest, NewAssignmentAttemptDoesNotInheritInference) {
  Initialize(2, WriteKind::Copy);
  ResetArenas();
  IREE_ASSERT_OK(BuildTable());
  assignment_indices_[1] = UINT32_MAX;
  assignment_indices_[2] = UINT32_MAX;
  const uint32_t retained_base = assignments_[0].location_base;
  EXPECT_EQ(Query(2, retained_base), LOOM_VALUE_ORDINAL_INVALID);
  loom_low_allocation_write_interference_reset_inference(table_);
  assignment_indices_[1] = 1;
  EXPECT_EQ(Query(2, retained_base), 0u);
}

TEST_F(WriteInterferenceTest, EveryBackingFailureRestoresScratchAndReclaims) {
  Initialize(2049, WriteKind::Copy);
  for (iree_host_size_t failure_index = 0;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    ResetArenas(failure_index);
    iree_status_t status = BuildTable(BindingKind::AliasedDestination);
    failure_index_ = SIZE_MAX;
    ExpectScratchPreserved();
    if (iree_status_is_ok(status)) {
      EXPECT_LE(allocation_count_, failure_index);
      EXPECT_EQ(Query(1, assignments_[0].location_base + 1), 0u);
      RecordProperty("backing_allocation_failures", (int)failure_index);
      break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(allocation_count_, failure_index + 1);
  }
}

}  // namespace
}  // namespace loom
