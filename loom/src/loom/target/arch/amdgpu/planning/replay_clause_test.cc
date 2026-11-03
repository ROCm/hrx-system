// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/replay_clause.h"

#include <initializer_list>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

class AmdgpuReplayClauseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    for (uint32_t i = 0; i < kValueCount; ++i) {
      value_ids_[i] = i + 1;
      assignment_indices_[i] = i;
      assignments_[i].value_id = value_ids_[i];
      assignments_[i].unit_count = 1;
      assignments_[i].location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
      assignments_[i].location_count = 1;
    }
    allocation_.liveness.value_ids = value_ids_;
    allocation_.liveness.value_count = kValueCount;
    allocation_.assignments = assignments_;
    allocation_.assignment_count = kValueCount;
    allocation_.assignment_indices_by_value_ordinal = assignment_indices_;
    allocation_.physical_extents.ends_by_reg_class = physical_ends_;
    allocation_.physical_extents.fixed_ends_by_reg_class = fixed_ends_;
    allocation_.physical_extents.count = kRegisterClassCount;
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_amdgpu_replay_clause_t* Allocate(
      iree_string_view_t target_name, loom_amdgpu_target_feature_state_t xnack,
      loom_amdgpu_descriptor_traits_t function_traits) {
    iree_arena_reset(&arena_);
    facts_ = {};
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_EXPECT_OK(
        loom_amdgpu_target_info_lookup_processor(target_name, &processor));
    if (processor == nullptr) {
      return nullptr;
    }
    facts_.base.fact_type = &loom_amdgpu_target_fact_type;
    facts_.properties.processor = &processor->properties;
    facts_.properties.amdhsa_features.xnack = xnack;
    schedule_ = {};
    schedule_.target.target_facts = &facts_.base;
    loom_amdgpu_replay_clause_t* state = nullptr;
    IREE_EXPECT_OK(loom_amdgpu_replay_clause_allocate(
        &schedule_, &allocation_, function_traits, &arena_, &state));
    return state;
  }

  void Assign(uint32_t ordinal, uint16_t register_class, uint32_t base,
              uint32_t count = 1) {
    assignments_[ordinal].descriptor_reg_class_id = register_class;
    assignments_[ordinal].unit_count = count;
    assignments_[ordinal].location_base = base;
    assignments_[ordinal].location_count = count;
  }

  uint32_t Apply(loom_amdgpu_replay_clause_t* state, uint32_t node_index,
                 loom_amdgpu_descriptor_traits_t traits,
                 std::initializer_list<loom_value_ordinal_t> operands,
                 std::initializer_list<loom_value_ordinal_t> results,
                 loom_trait_flags_t node_traits = 0,
                 uint64_t instruction_count = 1) {
    loom_low_schedule_node_t node = {
        .traits = node_traits,
        .operand_count = static_cast<uint16_t>(operands.size()),
        .result_count = static_cast<uint16_t>(results.size()),
    };
    EXPECT_LE(operands.size() + results.size(),
              LOOM_LOW_SCHEDULE_NODE_INLINE_VALUE_ORDINAL_CAPACITY);
    loom_value_ordinal_t* ordinals =
        loom_low_schedule_node_value_ordinals(&node);
    for (loom_value_ordinal_t ordinal : operands) {
      *ordinals++ = ordinal;
    }
    for (loom_value_ordinal_t ordinal : results) {
      *ordinals++ = ordinal;
    }
    const loom_low_packet_view_t packet = {
        .node_index = node_index,
        .node = &node,
    };
    return loom_amdgpu_replay_clause_apply(state, &packet, traits,
                                           instruction_count);
  }

  static constexpr uint32_t kValueCount = 12;
  static constexpr uint32_t kRegisterClassCount = 3;
  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  loom_amdgpu_target_facts_t facts_ = {};
  loom_low_schedule_table_t schedule_ = {};
  loom_low_allocation_table_t allocation_ = {};
  loom_value_id_t value_ids_[kValueCount] = {};
  uint32_t assignment_indices_[kValueCount] = {};
  loom_low_allocation_assignment_t assignments_[kValueCount] = {};
  uint32_t physical_ends_[kRegisterClassCount] = {128, 128, 128};
  uint32_t fixed_ends_[kRegisterClassCount] = {};
};

TEST_F(AmdgpuReplayClauseTest, SelectsTargetReplayModes) {
  constexpr auto kBoth = LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY |
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY;
  for (iree_string_view_t target :
       {IREE_SV("gfx90a"), IREE_SV("gfx942"), IREE_SV("gfx950")}) {
    SCOPED_TRACE(std::string(target.data, target.size));
    EXPECT_NE(Allocate(target, LOOM_AMDGPU_TARGET_FEATURE_ANY, kBoth), nullptr);
    EXPECT_NE(Allocate(target, LOOM_AMDGPU_TARGET_FEATURE_ON, kBoth), nullptr);
    EXPECT_EQ(Allocate(target, LOOM_AMDGPU_TARGET_FEATURE_OFF, kBoth), nullptr);
  }
  EXPECT_NE(Allocate(IREE_SV("gfx1250"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                     LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY),
            nullptr);
  EXPECT_EQ(Allocate(IREE_SV("gfx1250"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                     LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY),
            nullptr);
  EXPECT_EQ(Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                     LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU),
            nullptr);
}

TEST_F(AmdgpuReplayClauseTest, RetainsEveryDefinitionAndSourceInClause) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 8);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 9);
  Assign(4, 0, 4, 2);
  Assign(5, 0, 0);
  EXPECT_EQ(
      Apply(state, 3, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 5, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 8, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY, {4}, {5}),
      5u);
}

TEST_F(AmdgpuReplayClauseTest, HandlesWideRangesAcrossBitsetWords) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 62, 2);
  Assign(1, 0, 68, 4);
  Assign(2, 0, 66, 2);
  Assign(3, 0, 60, 4);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      0u);
}

TEST_F(AmdgpuReplayClauseTest, SeparatesPhysicalRegisterClasses) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 1, 0);
  Assign(2, 0, 1, 2);
  Assign(3, 2, 0);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
}

TEST_F(AmdgpuReplayClauseTest, LaterSelfAliasBreaksExistingClause) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 4);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 2);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      0u);
}

TEST_F(AmdgpuReplayClauseTest, ConcreteNonMemoryInstructionEndsClause) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 4);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 0);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(Apply(state, 1, 0, {}, {}, 0, 0), LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 2, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      0u);

  state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                   LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(Apply(state, 1, 0, {}, {}), LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 2, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
}

TEST_F(AmdgpuReplayClauseTest, ExplicitBreakAndMemoryClassSwitchEndClause) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY |
                             LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 4);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 0);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 2, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
  loom_amdgpu_replay_clause_break(state);
  EXPECT_EQ(
      Apply(state, 3, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
}

TEST_F(AmdgpuReplayClauseTest, StoresBreakOnlyClausesWithDefinitions) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 4);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 5);
  EXPECT_EQ(
      Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0}, {1}),
      LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2, 3},
                  {}, LOOM_TRAIT_WRITES_MEMORY),
            0u);

  state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                   LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0, 1},
                  {}, LOOM_TRAIT_WRITES_MEMORY),
            LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(
      Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2}, {3}),
      LOOM_LOW_PACKET_INDEX_NONE);
}

TEST_F(AmdgpuReplayClauseTest, ReturningStoreStartsDefinitionClause) {
  auto* state = Allocate(IREE_SV("gfx90a"), LOOM_AMDGPU_TARGET_FEATURE_ANY,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
  ASSERT_NE(state, nullptr);
  Assign(0, 0, 0, 2);
  Assign(1, 0, 4);
  Assign(2, 0, 2, 2);
  Assign(3, 0, 5);
  Assign(4, 0, 6);
  EXPECT_EQ(Apply(state, 0, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {0, 1},
                  {4}, LOOM_TRAIT_WRITES_MEMORY),
            LOOM_LOW_PACKET_INDEX_NONE);
  EXPECT_EQ(Apply(state, 1, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY, {2, 3},
                  {}, LOOM_TRAIT_WRITES_MEMORY),
            0u);
}

}  // namespace
}  // namespace loom
