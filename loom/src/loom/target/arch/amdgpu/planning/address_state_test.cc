// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/address_state.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace loom {
namespace {

class AmdgpuAddressStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry, IREE_SV("amdgpu.rdna4.gfx125x.core"));
    ASSERT_NE(descriptor_set_, nullptr);
    descriptor_ = loom_amdgpu_descriptor_ref_descriptor(
        descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32);
    ASSERT_NE(descriptor_, nullptr);
    ASSERT_EQ(descriptor_->result_count, 1u);
    ASSERT_EQ(descriptor_->operand_count, 5u);
    ASSERT_EQ(
        descriptor_set_->operands[descriptor_->operand_start].address_map_kind,
        LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE);
    ASSERT_EQ(descriptor_set_->operands[descriptor_->operand_start + 1]
                  .address_map_kind,
              LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE);
    ASSERT_EQ(descriptor_set_->operands[descriptor_->operand_start + 2]
                  .address_map_kind,
              LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE);
    InitializeTables();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void InitializeAssignment(loom_value_ordinal_t ordinal, uint16_t unit_count,
                            uint32_t location_base) {
    value_ids_[ordinal] = ordinal;
    assignment_indices_by_value_ordinal_[ordinal] = ordinal;
    assignments_[ordinal] = {};
    assignments_[ordinal].value_id = ordinal;
    assignments_[ordinal].descriptor_reg_class_id =
        LOOM_AMDGPU_REG_CLASS_ID_VGPR;
    assignments_[ordinal].unit_count = unit_count;
    assignments_[ordinal].location_kind =
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    assignments_[ordinal].location_base = location_base;
    assignments_[ordinal].location_count = unit_count;
  }

  void InitializeDescriptorNode(uint32_t node_index,
                                loom_value_ordinal_t result_ordinal,
                                loom_value_ordinal_t lhs_ordinal,
                                loom_value_ordinal_t rhs_ordinal) {
    loom_low_schedule_node_t& node = nodes_[node_index];
    node = {};
    node.descriptor = descriptor_;
    node.schedule_class =
        &descriptor_set_->schedule_classes[descriptor_->schedule_class_id];
    node.block_index = 0;
    node.source_ordinal = node_index;
    node.scheduled_ordinal = node_index;
    node.kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    node.operand_count = 2;
    node.result_count = 1;
    loom_low_schedule_node_operand_ordinals(&node)[0] = lhs_ordinal;
    loom_low_schedule_node_operand_ordinals(&node)[1] = rhs_ordinal;
    loom_low_schedule_node_result_ordinals(&node)[0] = result_ordinal;
  }

  void InitializeTables() {
    const loom_low_operand_t& result_operand =
        descriptor_set_->operands[descriptor_->operand_start];
    const loom_low_operand_t& source_operand =
        descriptor_set_->operands[descriptor_->operand_start + 1];
    const loom_low_operand_t& rhs_operand =
        descriptor_set_->operands[descriptor_->operand_start + 2];

    InitializeDescriptorNode(/*node_index=*/0, /*result_ordinal=*/0,
                             /*lhs_ordinal=*/1, /*rhs_ordinal=*/2);
    InitializeDescriptorNode(/*node_index=*/1, /*result_ordinal=*/3,
                             /*lhs_ordinal=*/4, /*rhs_ordinal=*/5);
    nodes_[2] = {};
    nodes_[2].block_index = 0;
    nodes_[2].source_ordinal = 2;
    nodes_[2].scheduled_ordinal = 2;
    nodes_[2].traits = LOOM_TRAIT_TERMINATOR;
    nodes_[2].kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;

    region_blocks_[0] = &source_block_;
    region_.block_count = IREE_ARRAYSIZE(region_blocks_);
    region_.block_capacity = IREE_ARRAYSIZE(region_blocks_);
    region_.blocks = region_blocks_;
    source_block_.parent_region = &region_;
    source_block_.region_index = 0;
    for (loom_low_schedule_node_t& node : nodes_) {
      node.block = &source_block_;
    }
    scheduled_node_indices_[0] = 0;
    scheduled_node_indices_[1] = 1;
    scheduled_node_indices_[2] = 2;
    block_ = {};
    block_.block = &source_block_;
    block_.node_start = 0;
    block_.node_count = 3;
    block_.scheduled_node_start = 0;
    block_.scheduled_node_count = 3;
    schedule_ = {};
    schedule_.module = &module_;
    schedule_.function_op = &function_op_;
    schedule_.target.descriptor_set = descriptor_set_;
    schedule_.blocks = &block_;
    schedule_.block_count = 1;
    schedule_.nodes = nodes_;
    schedule_.node_count = 3;
    schedule_.scheduled_node_indices = scheduled_node_indices_;
    schedule_.scheduled_node_count = 3;

    InitializeAssignment(/*ordinal=*/0, result_operand.unit_count,
                         /*location_base=*/300);
    InitializeAssignment(/*ordinal=*/1, source_operand.unit_count,
                         /*location_base=*/10);
    InitializeAssignment(/*ordinal=*/2, rhs_operand.unit_count,
                         /*location_base=*/11);
    InitializeAssignment(/*ordinal=*/3, result_operand.unit_count,
                         /*location_base=*/20);
    InitializeAssignment(/*ordinal=*/4, source_operand.unit_count,
                         /*location_base=*/310);
    InitializeAssignment(/*ordinal=*/5, rhs_operand.unit_count,
                         /*location_base=*/21);
    allocation_ = {};
    allocation_.module = &module_;
    allocation_.function_op = &function_op_;
    allocation_.target.descriptor_set = descriptor_set_;
    allocation_.liveness.value_ids = value_ids_;
    allocation_.liveness.value_count = IREE_ARRAYSIZE(value_ids_);
    allocation_.assignments = assignments_;
    allocation_.assignment_count = IREE_ARRAYSIZE(assignments_);
    allocation_.assignment_indices_by_value_ordinal =
        assignment_indices_by_value_ordinal_;
  }

  loom_amdgpu_address_state_requirement_t Requirement(uint32_t packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&schedule_, packet_index);
    return loom_amdgpu_address_state_requirement_for_packet(&allocation_,
                                                            &packet);
  }

  iree_status_t BuildPlan(loom_amdgpu_address_state_plan_t* out_plan) {
    return loom_amdgpu_address_state_plan_build(&schedule_, &allocation_,
                                                &arena_, out_plan);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  const loom_low_descriptor_t* descriptor_ = nullptr;
  loom_module_t module_ = {};
  loom_op_t function_op_ = {};
  loom_region_t region_ = {};
  loom_block_t source_block_ = {};
  loom_block_t* region_blocks_[1] = {};
  loom_low_schedule_block_t block_ = {};
  loom_low_schedule_node_t nodes_[3] = {};
  uint32_t scheduled_node_indices_[3] = {};
  loom_low_schedule_table_t schedule_ = {};
  loom_low_allocation_assignment_t assignments_[6] = {};
  loom_value_id_t value_ids_[6] = {};
  uint32_t assignment_indices_by_value_ordinal_[6] = {};
  loom_low_allocation_table_t allocation_ = {};
};

TEST_F(AmdgpuAddressStateTest, BuildsDeterministicScheduledTransitions) {
  const loom_amdgpu_address_state_requirement_t first_requirement =
      Requirement(0);
  const loom_amdgpu_address_state_requirement_t second_requirement =
      Requirement(1);
  ASSERT_NE(first_requirement.value, 0u);
  ASSERT_NE(second_requirement.value, 0u);
  ASSERT_NE(first_requirement.value, second_requirement.value);

  loom_amdgpu_address_state_plan_t first_plan = {};
  IREE_ASSERT_OK(BuildPlan(&first_plan));
  ASSERT_EQ(first_plan.transition_count, 3u);
  EXPECT_EQ(first_plan.transitions[0].node_index, 0u);
  EXPECT_EQ(first_plan.transitions[0].mode_immediate, first_requirement.value);
  EXPECT_EQ(first_plan.transitions[1].node_index, 1u);
  EXPECT_EQ(first_plan.transitions[1].mode_immediate,
            static_cast<uint16_t>(
                (static_cast<uint16_t>(first_requirement.value) << 8) |
                second_requirement.value));
  EXPECT_EQ(first_plan.transitions[2].node_index, 2u);
  EXPECT_EQ(first_plan.transitions[2].mode_immediate,
            static_cast<uint16_t>(
                static_cast<uint16_t>(second_requirement.value) << 8));

  loom_amdgpu_address_state_plan_t second_plan = {};
  IREE_ASSERT_OK(BuildPlan(&second_plan));
  ASSERT_EQ(second_plan.transition_count, first_plan.transition_count);
  EXPECT_EQ(std::memcmp(
                second_plan.transitions, first_plan.transitions,
                first_plan.transition_count * sizeof(*first_plan.transitions)),
            0);
}

TEST_F(AmdgpuAddressStateTest, IgnoresSelectedScalarRegisterAlternative) {
  const loom_low_operand_t& source_operand =
      descriptor_set_->operands[descriptor_->operand_start + 1];
  const uint8_t source_slot_mask =
      static_cast<uint8_t>(0x3u << loom_amdgpu_vgpr_msb_slot_shift(
                               static_cast<loom_amdgpu_vgpr_msb_slot_t>(
                                   source_operand.address_state_slot)));
  assignments_[1].descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  assignments_[1].location_base = 300;

  const loom_amdgpu_address_state_requirement_t scalar_requirement =
      Requirement(0);
  EXPECT_EQ(scalar_requirement.mask & source_slot_mask, 0u);

  assignments_[1].descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
  const loom_amdgpu_address_state_requirement_t vector_requirement =
      Requirement(0);
  EXPECT_EQ(vector_requirement.mask & source_slot_mask, source_slot_mask);
  EXPECT_NE(vector_requirement.value & source_slot_mask, 0u);
}

TEST_F(AmdgpuAddressStateTest, ProducesNoTransitionsForLowVgprWindow) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(assignments_); ++i) {
    assignments_[i].location_base = static_cast<uint32_t>(i);
  }
  loom_amdgpu_address_state_plan_t plan = {};
  IREE_ASSERT_OK(BuildPlan(&plan));
  EXPECT_EQ(plan.transition_count, 0u);
}

TEST_F(AmdgpuAddressStateTest, StructuralPacketsHaveNoDescriptorRequirement) {
  const loom_amdgpu_address_state_requirement_t requirement = Requirement(2);
  EXPECT_EQ(requirement.mask, 0u);
  EXPECT_EQ(requirement.value, 0u);
}

}  // namespace
}  // namespace loom
