// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/immediates.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/wait_frontier.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

// The planner consumes scheduled descriptor facts and dependency edges, not
// source IR or assembly. These compact fixtures keep the independent
// issue-limit and memory-completion inputs explicit while allowing CFG tests.
class AmdgpuWaitPlanTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<AmdgpuWaitPlanTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected wait-plan allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializePlanStorage() {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(32 * 1024, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &transient_arena_);
  }

  void DeinitializePlanStorage() {
    iree_arena_deinitialize(&transient_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void SetUp() override {
    InitializePlanStorage();
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &module_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("wait_plan"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&registry_);
    descriptors_ = loom_low_descriptor_registry_lookup(
        &registry_.registry, IREE_SV("amdgpu.rdna4.gfx125x.core"));
    ASSERT_NE(descriptors_, nullptr);
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1250"),
                                                            &processor));
    facts_.base.fact_type = &loom_amdgpu_target_fact_type;
    facts_.properties.processor = &processor->properties;

    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    loom_string_id_t descriptor_set = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("transfers"), &name));
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder_, IREE_SV("amdgpu.rdna4.gfx125x.core"), &descriptor_set));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_type_t argument_types[4];
    const uint32_t widths[] = {4, 8, 4, 4};
    for (uint32_t i = 0; i < 4; ++i) {
      IREE_ASSERT_OK(loom_low_build_register_type(
          descriptors_, LOOM_AMDGPU_REG_CLASS_ID_SGPR, widths[i],
          &argument_types[i]));
    }
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, 0, 0, 0, 0, 0, 0, 0, 0, descriptor_set, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, {0, symbol}, argument_types, 4, nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    loom_block_t* block =
        loom_region_entry_block(loom_low_func_def_body(function));
    block_.block = block;
    loom_builder_initialize(module_, &module_->arena, block, &builder_);
    schedule_.module = module_;
    schedule_.function_op = function;
    schedule_.target.descriptor_set = descriptors_;
    schedule_.target.target_facts = &facts_.base;
    schedule_.blocks = &block_;
    schedule_.block_count = 1;
    schedule_.value_count = 4;
    schedule_.value_ids = block->arg_ids;
    schedule_.value_producer_nodes = value_producer_nodes_;
    loom_low_schedule_dependency_graph_initialize(&schedule_.dependencies);
    const uint32_t locations[] = {0, 8, 4, 16};
    for (uint32_t i = 0; i < 4; ++i) {
      value_producer_nodes_[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
      assignments_[i].value_id = block->arg_ids[i];
      assignments_[i].descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
      assignments_[i].unit_count = widths[i];
      assignments_[i].location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
      assignments_[i].location_base = locations[i];
      assignments_[i].location_count = widths[i];
      assignment_indices_[i] = i;
    }
    allocation_.module = module_;
    allocation_.function_op = function;
    allocation_.target = schedule_.target;
    allocation_.liveness.value_ids = block->arg_ids;
    allocation_.liveness.value_count = 4;
    allocation_.assignments = assignments_;
    allocation_.assignment_count = 4;
    allocation_.assignment_indices_by_value_ordinal = assignment_indices_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&module_pool_);
    DeinitializePlanStorage();
  }

  void Tensor(loom_amdgpu_descriptor_ref_t descriptor_ref =
                  LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D2) {
    Append(descriptor_ref, loom_named_attr_slice_empty());
  }

  void Wait(uint16_t bound) {
    AppendImmediate(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_TENSORCNT,
                    IREE_SV("tensorcnt"), bound);
  }

  void Delay() {
    AppendImmediate(LOOM_AMDGPU_DESCRIPTOR_REF_S_DELAY_ALU, IREE_SV("delay"),
                    0);
  }

  void AppendImmediate(loom_amdgpu_descriptor_ref_t descriptor_ref,
                       iree_string_view_t field_name, uint16_t value) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&builder_, field_name, &name));
    loom_named_attr_t attr = {};
    attr.name_id = name;
    attr.value = loom_attr_i64(value);
    Append(descriptor_ref, loom_make_named_attr_slice(&attr, 1));
  }

  void MemoryDependency(uint32_t producer, uint32_t consumer) {
    if (schedule_.effect_dependencies.count == 0) {
      schedule_.effect_dependencies.start =
          static_cast<uint32_t>(schedule_.dependencies.count);
    } else {
      IREE_ASSERT_EQ(schedule_.dependencies.count,
                     schedule_.effect_dependencies.start +
                         schedule_.effect_dependencies.count);
    }
    loom_low_schedule_dependency_t dependency = {};
    dependency.producer_node = producer;
    dependency.consumer_node = consumer;
    dependency.kind = LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT;
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &schedule_.dependencies, dependency, &module_->arena));
    ++schedule_.effect_dependencies.count;
  }

  void FinalizeSchedule() {
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    loom_low_schedule_node_t node = {};
    node.op = return_op;
    node.kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
    node.traits = return_op->traits;
    node.source_ordinal = nodes_.size();
    node.scheduled_ordinal = nodes_.size();
    nodes_.push_back(node);
    order_.resize(nodes_.size());
    for (uint32_t i = 0; i < order_.size(); ++i) {
      order_[i] = i;
    }
    block_.node_count = nodes_.size();
    block_.scheduled_node_count = nodes_.size();
    schedule_.nodes = nodes_.data();
    schedule_.node_count = nodes_.size();
    schedule_.scheduled_node_indices = order_.data();
    schedule_.scheduled_node_count = order_.size();
    schedule_.effect_uses = effects_.data();
    schedule_.effect_use_count = effects_.size();
    schedule_.hazard_uses = hazards_.data();
    schedule_.hazard_use_count = hazards_.size();
  }

  iree_status_t BuildPlan() {
    const loom_amdgpu_address_state_plan_t address_state = {};
    return loom_amdgpu_wait_plan_build(&schedule_, &allocation_, &address_state,
                                       &arena_, &transient_arena_, &plan_);
  }

  void Build() {
    FinalizeSchedule();
    IREE_ASSERT_OK(BuildPlan());
  }

  void RecordColdStorage() {
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&pool_, &statistics);
    RecordProperty("cold_pool_blocks",
                   std::to_string(statistics.block_system_allocation_count));
    RecordProperty("cold_oversized_count",
                   std::to_string(statistics.oversized_allocation_count));
    RecordProperty("cold_oversized_bytes",
                   std::to_string(statistics.oversized_allocation_bytes));
    RecordProperty("cold_backing_calls", std::to_string(allocation_count_));
  }

  void ExpectReusablePlan() {
    FinalizeSchedule();
    iree_host_size_t first_allocation_count = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
      SCOPED_TRACE(epoch);
      iree_arena_reset(&arena_);
      IREE_ASSERT_OK(BuildPlan());
      iree_arena_reset(&transient_arena_);
      iree_arena_block_pool_statistics_t statistics = {};
      iree_arena_block_pool_query_statistics(&pool_, &statistics);
      EXPECT_EQ(statistics.oversized_allocation_count, 0u);
      if (epoch == 0) {
        first_allocation_count = allocation_count_;
        RecordColdStorage();
      }
      EXPECT_EQ(allocation_count_, first_allocation_count);
    }
  }

  void ExpectAction(iree_host_size_t index, uint32_t node, uint16_t bound,
                    loom_amdgpu_wait_plan_reason_t reason) {
    ASSERT_LT(index, plan_.action_count);
    const auto& action = plan_.actions[index];
    EXPECT_EQ(action.node_index, node);
    EXPECT_EQ(action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
    EXPECT_EQ(action.target_count, bound);
    EXPECT_EQ(action.reason, reason);
  }

  void Append(loom_amdgpu_descriptor_ref_t descriptor_ref,
              loom_named_attr_slice_t attrs) {
    const auto* descriptor =
        loom_amdgpu_descriptor_ref_descriptor(descriptors_, descriptor_ref);
    const auto* view =
        loom_low_descriptor_set_descriptor_view(descriptors_, descriptor);
    loom_low_schedule_node_t node = {};
    node.source_ordinal = nodes_.size();
    node.scheduled_ordinal = nodes_.size();
    node.kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    node.descriptor = descriptor;
    node.source_descriptor_ordinal =
        loom_low_descriptor_set_descriptor_ordinal(descriptors_, descriptor);
    node.schedule_class_id = view->schedule_class_id;
    node.schedule_class =
        &descriptors_->schedule_classes[view->schedule_class_id];
    node.operand_count = descriptor->minimum_packet_operand_count;
    for (uint16_t i = 0; i < node.operand_count; ++i) {
      loom_low_schedule_node_operand_ordinals(&node)[i] = i;
    }
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder_, descriptors_, descriptor, 0, block_.block->arg_ids,
        node.operand_count, attrs, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &op));
    node.op = op;
    node.traits = op->traits;
    node.immediate_presence = loom_low_bind_immediate_presence(
        module_, descriptors_, descriptor, loom_low_op_attrs(op));
    nodes_.push_back(node);
    for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
      const auto& effect = descriptors_->effects[descriptor->effect_start + i];
      loom_low_schedule_effect_use_t use = {};
      use.node_index = node.source_ordinal;
      use.scheduled_ordinal = node.scheduled_ordinal;
      use.effect_ordinal = i;
      use.kind = effect.kind;
      use.memory_space = effect.memory_space;
      use.scope_id = effect.scope_id;
      use.effect_flags = effect.flags;
      use.counter_id = effect.counter_id;
      use.width_bits = effect.width_bits;
      effects_.push_back(use);
    }
    for (uint16_t i = 0; i < node.schedule_class->hazard_count; ++i) {
      const auto& hazard =
          descriptors_->hazards[node.schedule_class->hazard_start + i];
      loom_low_schedule_hazard_use_t use = {};
      use.node_index = node.source_ordinal;
      use.scheduled_ordinal = node.scheduled_ordinal;
      use.hazard_ordinal = i;
      use.kind = hazard.kind;
      use.reference_kind = hazard.reference_kind;
      use.reference_id = hazard.reference_id;
      use.producer_stage = hazard.producer_stage;
      use.consumer_stage = hazard.consumer_stage;
      use.distance = hazard.distance;
      use.hazard_flags = hazard.flags;
      hazards_.push_back(use);
    }
  }

  // Reusable planner blocks, independent of immutable input storage.
  iree_arena_block_pool_t pool_;
  // Published planner result lifetime.
  iree_arena_allocator_t arena_;
  // Classification and action-building scratch, released before result use.
  iree_arena_allocator_t transient_arena_;
  // Module and schedule dependency storage surviving planner resets.
  iree_arena_block_pool_t module_pool_;
  // Attempted planner backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation selected for injected failure, or SIZE_MAX.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // IR ownership for descriptor immediates and operand identities.
  loom_context_t context_;
  // Small Low function owning the scheduled operations.
  loom_module_t* module_ = nullptr;
  // Appends descriptor operations to the fixture block.
  loom_builder_t builder_;
  // Canonical native descriptor registry.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Descriptor contract used by every scheduled node.
  const loom_low_descriptor_set_t* descriptors_ = nullptr;
  // Canonical processor properties selecting the issue constraint.
  loom_amdgpu_target_facts_t facts_ = {};
  // Single straight-line block under test.
  loom_low_schedule_block_t block_ = {};
  // Descriptor nodes in scheduled order.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Node indices retained for the result's borrowed schedule.
  std::vector<uint32_t> order_;
  // Descriptor memory and counter effects.
  std::vector<loom_low_schedule_effect_use_t> effects_;
  // Descriptor counter hazards.
  std::vector<loom_low_schedule_hazard_use_t> hazards_;
  // Immutable input assembled before the planner call.
  loom_low_schedule_table_t schedule_ = {};
  // Nonoverlapping physical SGPR tuples backing the function arguments.
  loom_low_allocation_assignment_t assignments_[4] = {};
  // Direct value-ordinal to assignment mapping.
  uint32_t assignment_indices_[4] = {};
  // Schedule-owned producer identity for the four live-in arguments.
  uint32_t value_producer_nodes_[4] = {};
  // Successful allocation consumed by the production planner.
  loom_low_allocation_table_t allocation_ = {};
  // Actual production planner result inspected by each test.
  loom_amdgpu_wait_plan_t plan_ = {};
};

TEST_F(AmdgpuWaitPlanTest, IndependentTransfersKeepTheHardwareIssueBound) {
  Tensor();
  Tensor(LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D4);
  Build();
  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN);
  EXPECT_EQ(plan_.actions[0].outstanding_before, 1u);
}

TEST_F(AmdgpuWaitPlanTest, AuthoredPartialWaitSatisfiesTheIssueBound) {
  Tensor();
  Wait(10);
  Tensor();
  Build();
  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
}

TEST_F(AmdgpuWaitPlanTest, LargerAuthoredBoundDoesNotSatisfyTheIssueBound) {
  for (uint32_t i = 0; i < 12; ++i) {
    Tensor();
  }
  Wait(11);
  Tensor();
  Build();
  ASSERT_EQ(plan_.action_count, 13u);
  ExpectAction(11, 12, 11, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  ExpectAction(12, 13, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN);
}

TEST_F(AmdgpuWaitPlanTest, DependencyCompletionAlsoSatisfiesTheIssueBound) {
  Tensor();
  Tensor();
  MemoryDependency(0, 1);
  Build();
  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
}

TEST_F(AmdgpuWaitPlanTest, PartialWaitCompletesAnOlderTransfer) {
  Tensor();
  Tensor();
  Wait(1);
  Tensor();
  MemoryDependency(0, 3);
  Build();
  ASSERT_EQ(plan_.action_count, 2u);
  ExpectAction(0, 1, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN);
  ExpectAction(1, 2, 1, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
}

TEST_F(AmdgpuWaitPlanTest, PartialWaitDoesNotCompleteTheNewestTransfer) {
  Tensor();
  Wait(10);
  Tensor();
  MemoryDependency(0, 2);
  Build();
  ASSERT_EQ(plan_.action_count, 2u);
  ExpectAction(0, 1, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  ExpectAction(1, 2, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
}

TEST_F(AmdgpuWaitPlanTest, NodesWithoutBoundsReusePoolBlocks) {
  for (int i = 0; i < 1500; ++i) {
    Delay();
  }
  ExpectReusablePlan();
  EXPECT_EQ(plan_.action_count, 0u);
  ASSERT_EQ(plan_.progress.record_count, 1u);
  EXPECT_EQ(plan_.progress.records[0].packet_index, 1500u);
  EXPECT_EQ(plan_.progress.records[0].progress_class_id,
            LOOM_AMDGPU_WAIT_COUNTER_X);
  EXPECT_EQ(plan_.progress.records[0].action,
            LOOM_LOW_PACKET_PROGRESS_ACTION_RESET);
}

TEST_F(AmdgpuWaitPlanTest, UnconstrainedWaitsReusePoolBlocks) {
  for (int i = 0; i < 1500; ++i) {
    Wait(UINT16_MAX);
  }
  ExpectReusablePlan();
  EXPECT_EQ(plan_.action_count, 0u);
  ASSERT_EQ(plan_.progress.record_count, 1u);
  EXPECT_EQ(plan_.progress.records[0].packet_index, 1500u);
  EXPECT_EQ(plan_.progress.records[0].progress_class_id,
            LOOM_AMDGPU_WAIT_COUNTER_X);
  EXPECT_EQ(plan_.progress.records[0].action,
            LOOM_LOW_PACKET_PROGRESS_ACTION_RESET);
}

TEST_F(AmdgpuWaitPlanTest, SparseFullWidthBoundsSurviveTransientReset) {
  Tensor();
  Wait(UINT16_MAX - 1);
  for (int i = 0; i < 1496; ++i) {
    Delay();
  }
  Tensor();
  Wait(257);
  Wait(UINT16_MAX);
  ExpectReusablePlan();
  ASSERT_EQ(plan_.action_count, 3u);
  ExpectAction(0, 1, UINT16_MAX - 1,
               LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  ExpectAction(1, 1498, 10, LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN);
  ExpectAction(2, 1499, 257, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  ASSERT_EQ(plan_.progress.record_count, 5u);
  const uint32_t expected_packets[] = {0, 1, 1498, 1499, 1501};
  const uint32_t expected_units[] = {1, UINT16_MAX - 1, 1, 257, 0};
  const loom_low_packet_progress_action_t expected_actions[] = {
      LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
      LOOM_LOW_PACKET_PROGRESS_ACTION_BOUND,
      LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
      LOOM_LOW_PACKET_PROGRESS_ACTION_BOUND,
      LOOM_LOW_PACKET_PROGRESS_ACTION_RESET,
  };
  for (iree_host_size_t i = 0; i < plan_.progress.record_count; ++i) {
    const auto& record = plan_.progress.records[i];
    EXPECT_EQ(record.packet_index, expected_packets[i]);
    EXPECT_EQ(record.units, expected_units[i]);
    EXPECT_EQ(record.action, expected_actions[i]);
    EXPECT_EQ(record.progress_class_id, i == 4
                                            ? LOOM_AMDGPU_WAIT_COUNTER_X
                                            : LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
  }
}

TEST_F(AmdgpuWaitPlanTest, BackingFailureDoesNotPublishPartialPlan) {
  // Each table fits a pool block, but the bounds array needs a fresh block.
  constexpr uint32_t kWaitCount = 1750;
  for (uint32_t i = 0; i < kWaitCount; ++i) {
    Wait(i == kWaitCount / 2 ? 257 : UINT16_MAX);
  }
  Build();
  RecordColdStorage();
  const iree_host_size_t allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 0u);
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DeinitializePlanStorage();
    InitializePlanStorage();
    failure_index_ = i;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, BuildPlan());
    EXPECT_EQ(plan_.schedule, nullptr);
    EXPECT_EQ(plan_.allocation, nullptr);
    EXPECT_EQ(plan_.actions, nullptr);
    EXPECT_EQ(plan_.action_count, 0u);
    EXPECT_EQ(plan_.progress.records, nullptr);
    EXPECT_EQ(plan_.progress.record_count, 0u);
    EXPECT_EQ(plan_.hazard_plan.progress, nullptr);
    EXPECT_EQ(plan_.elided_wait_nodes, nullptr);
    iree_arena_reset(&transient_arena_);
    iree_arena_reset(&arena_);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(BuildPlan());
    iree_arena_reset(&transient_arena_);
    ASSERT_EQ(plan_.action_count, 1u);
    ExpectAction(0, kWaitCount / 2, 257,
                 LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  }
}

class AmdgpuWaitFrontierTest : public ::testing::Test {
 protected:
  static constexpr iree_host_size_t kMaxNodeCount = 132;

  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &comparison_arena_);
    IREE_ASSERT_OK(
        loom_low_memory_access_map_create(&arena_, &memory_accesses_));
    runtime_bounds_.arithmetic_valid = true;
  }

  void TearDown() override {
    iree_arena_deinitialize(&comparison_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  uint32_t AddNode(uint16_t block_index) {
    const uint32_t node_index = static_cast<uint32_t>(nodes_.size());
    IREE_ASSERT_LT(node_index, kMaxNodeCount);
    loom_low_schedule_node_t node = {};
    node.op = &ops_[node_index];
    node.descriptor = &descriptors_[node_index];
    node.block_index = block_index;
    node.source_ordinal = node_index;
    node.kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    nodes_.push_back(node);
    frontier_nodes_.push_back({});
    completion_nodes_.push_back({});
    return node_index;
  }

  uint32_t AddAccess(uint16_t block_index, loom_low_effect_kind_t kind,
                     uint16_t counter_id,
                     const loom_low_memory_access_summary_t* summary) {
    const uint32_t node_index = AddNode(block_index);
    const loom_low_memory_space_t memory_space =
        static_cast<loom_low_memory_space_t>(
            summary == nullptr ? LOOM_LOW_MEMORY_SPACE_WORKGROUP
                               : summary->memory_space);
    loom_low_schedule_effect_use_t effect = {};
    effect.node_index = node_index;
    effect.block_index = block_index;
    effect.effect_ordinal = 0;
    effect.kind = kind;
    effect.memory_space = memory_space;
    effect.effect_flags = LOOM_LOW_EFFECT_FLAG_DEPENDENCY;
    effect.counter_id = counter_id;
    effects_.push_back(effect);
    const auto space_flags = loom_amdgpu_wait_memory_space_flag(memory_space);
    const auto counter_mask = loom_amdgpu_wait_counter_mask(counter_id);
    if (kind == LOOM_LOW_EFFECT_KIND_READ) {
      frontier_nodes_[node_index].read_counter_mask = counter_mask;
      frontier_nodes_[node_index].read_space_flags = space_flags;
    } else {
      IREE_ASSERT_EQ(kind, LOOM_LOW_EFFECT_KIND_WRITE);
      frontier_nodes_[node_index].write_counter_mask = counter_mask;
      frontier_nodes_[node_index].write_space_flags = space_flags;
    }
    if (summary != nullptr) {
      IREE_CHECK_OK(loom_low_memory_access_map_insert(
          memory_accesses_, &ops_[node_index], /*effect_ordinal=*/0, summary));
    }
    return node_index;
  }

  void AddExternalRead(uint32_t node_index, uint16_t counter_id) {
    IREE_ASSERT_LT(node_index, nodes_.size());
    loom_low_schedule_effect_use_t effect = {};
    effect.node_index = node_index;
    effect.block_index = nodes_[node_index].block_index;
    effect.effect_ordinal = 1;
    effect.kind = LOOM_LOW_EFFECT_KIND_READ;
    effect.memory_space = LOOM_LOW_MEMORY_SPACE_NONE;
    effect.counter_id = counter_id;
    effects_.push_back(effect);
    const auto counter_mask = loom_amdgpu_wait_counter_mask(counter_id);
    frontier_nodes_[node_index].read_counter_mask |= counter_mask;
    frontier_nodes_[node_index].external_counter_mask |= counter_mask;
  }

  void FinalizeSchedule(const loom_cfg_graph_t* graph) {
    blocks_.resize(graph->block_count);
    order_.clear();
    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      auto& block = blocks_[block_index];
      block.scheduled_node_start = static_cast<uint32_t>(order_.size());
      block.node_start = UINT32_MAX;
      for (uint32_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        if (nodes_[node_index].block_index != block_index) {
          continue;
        }
        block.node_start = iree_min(block.node_start, node_index);
        nodes_[node_index].scheduled_ordinal = block.scheduled_node_count;
        for (auto& effect : effects_) {
          if (effect.node_index == node_index) {
            effect.scheduled_ordinal = block.scheduled_node_count;
          }
        }
        order_.push_back(node_index);
        ++block.node_count;
        ++block.scheduled_node_count;
      }
      if (block.node_start == UINT32_MAX) {
        block.node_start = block.scheduled_node_start;
      }
    }
    schedule_ = {};
    schedule_.memory_accesses = memory_accesses_;
    schedule_.blocks = blocks_.data();
    schedule_.block_count = blocks_.size();
    schedule_.cfg_graph = *graph;
    schedule_.nodes = nodes_.data();
    schedule_.node_count = nodes_.size();
    schedule_.scheduled_node_indices = order_.data();
    schedule_.scheduled_node_count = order_.size();
    schedule_.effect_uses = effects_.data();
    schedule_.effect_use_count = effects_.size();
  }

  void Initialize(const loom_cfg_graph_t* graph,
                  const loom_low_allocation_table_t* allocation = nullptr) {
    FinalizeSchedule(graph);
    IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
        &schedule_, allocation, frontier_nodes_.data(),
        completion_nodes_.data(), /*dependencies=*/nullptr,
        /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
        &runtime_bounds_, &arena_, &frontier_));
  }

  void ProcessBlock(uint16_t block_index) {
    loom_amdgpu_wait_frontier_begin_block(&frontier_, block_index);
    loom_amdgpu_wait_frontier_end_block(&frontier_);
  }

  void ExpectCoarseOnly() {
    EXPECT_EQ(frontier_.memory.coarse_nodes, frontier_nodes_.data());
    EXPECT_EQ(frontier_.memory.precise_accesses, nullptr);
    EXPECT_EQ(frontier_.memory.precise_access_indices_by_node, nullptr);
    EXPECT_EQ(frontier_.memory.precise_access_count, 0u);
    EXPECT_EQ(frontier_.memory.precise_word_count, 0u);
    EXPECT_EQ(frontier_.memory.precise_static_outgoing_words, nullptr);
    EXPECT_EQ(frontier_.memory.precise_resolved_outgoing_words, nullptr);
    EXPECT_EQ(frontier_.memory.precise_active_words, nullptr);
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t comparison_arena_;
  loom_low_memory_access_map_t* memory_accesses_ = nullptr;
  std::array<loom_op_t, kMaxNodeCount> ops_ = {};
  std::array<loom_low_descriptor_t, kMaxNodeCount> descriptors_ = {};
  std::vector<loom_low_schedule_node_t> nodes_;
  std::vector<loom_low_schedule_effect_use_t> effects_;
  std::vector<loom_amdgpu_wait_frontier_node_t> frontier_nodes_;
  std::vector<loom_amdgpu_wait_completion_node_t> completion_nodes_;
  std::vector<loom_low_schedule_block_t> blocks_;
  std::vector<uint32_t> order_;
  loom_low_schedule_table_t schedule_ = {};
  loom_amdgpu_wait_frontier_precise_runtime_bounds_t runtime_bounds_ = {};
  loom_amdgpu_wait_frontier_t frontier_ = {};
};

struct PingPongAccesses {
  PingPongAccesses() {
    constexpr int64_t kStageBytes = 34048;
    constexpr int64_t kFootprintBytes = 32240;
    terms[0] = {kStageBytes, 17, 17};
    terms[1] = {-kStageBytes, 17, 17};
    for (iree_host_size_t i = 0; i < 2; ++i) {
      intervals[i].scope = &scope;
      intervals[i].storage_id = 4;
      intervals[i].origin.constant = i == 0 ? 0 : kStageBytes;
      intervals[i].origin.terms = &terms[i];
      intervals[i].origin.term_count = 1;
      intervals[i].origin.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
      intervals[i].upper = kFootprintBytes;
      accesses[i].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
      accesses[i].relative_interval = &intervals[i];
    }
  }

  int scope = 0;
  loom_symbolic_term_t terms[2] = {};
  loom_low_memory_relative_interval_t intervals[2] = {};
  loom_low_memory_access_summary_t accesses[2] = {};
};

TEST_F(AmdgpuWaitPlanTest,
       CrossBlockPlanUsesPreciseDisjointMemorySummaries) {
  const iree_host_size_t producer_effect_start = effects_.size();
  AppendImmediate(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32,
                  IREE_SV("scope"), LOOM_CACHE_SCOPE_DEVICE);
  const uint32_t producer = static_cast<uint32_t>(nodes_.size() - 1);
  ASSERT_EQ(effects_.size(), producer_effect_start + 1);
  EXPECT_EQ(effects_[producer_effect_start].kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(effects_[producer_effect_start].memory_space,
            LOOM_LOW_MEMORY_SPACE_GLOBAL);

  loom_region_t* body = loom_low_func_def_body(schedule_.function_op);
  loom_block_t* successor = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &successor));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&builder_, successor, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_low_schedule_node_t branch_node = {};
  branch_node.op = branch;
  branch_node.block_index = 0;
  branch_node.source_ordinal = nodes_.size();
  branch_node.scheduled_ordinal = 1;
  branch_node.kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
  branch_node.traits = branch->traits;
  nodes_.push_back(branch_node);

  loom_builder_set_block(&builder_, successor);
  const iree_host_size_t consumer_effect_start = effects_.size();
  const iree_host_size_t consumer_hazard_start = hazards_.size();
  Tensor();
  const uint32_t consumer = static_cast<uint32_t>(nodes_.size() - 1);
  nodes_[consumer].block_index = 1;
  nodes_[consumer].scheduled_ordinal = 0;
  for (iree_host_size_t i = consumer_effect_start; i < effects_.size(); ++i) {
    effects_[i].block_index = 1;
    effects_[i].scheduled_ordinal = 0;
  }
  for (iree_host_size_t i = consumer_hazard_start; i < hazards_.size(); ++i) {
    hazards_[i].block_index = 1;
    hazards_[i].scheduled_ordinal = 0;
  }
  ASSERT_EQ(effects_.size(), consumer_effect_start + 2);
  EXPECT_EQ(effects_[consumer_effect_start].kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(effects_[consumer_effect_start].memory_space,
            LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(effects_[consumer_effect_start].counter_id,
            LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
  EXPECT_EQ(effects_[consumer_effect_start + 1].kind,
            LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(effects_[consumer_effect_start + 1].memory_space,
            LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effects_[consumer_effect_start + 1].counter_id,
            LOOM_AMDGPU_WAIT_COUNTER_TENSOR);

  FinalizeSchedule();
  const uint32_t return_node = static_cast<uint32_t>(nodes_.size() - 1);
  nodes_[return_node].block_index = 1;
  nodes_[return_node].scheduled_ordinal = 1;

  std::array<loom_low_schedule_block_t, 2> blocks = {};
  blocks[0].block = block_.block;
  blocks[0].node_start = producer;
  blocks[0].node_count = 2;
  blocks[0].scheduled_node_start = 0;
  blocks[0].scheduled_node_count = 2;
  blocks[1].block = successor;
  blocks[1].node_start = consumer;
  blocks[1].node_count = 2;
  blocks[1].scheduled_node_start = 2;
  blocks[1].scheduled_node_count = 2;
  testing::CfgGraph graph({{1}, {}});
  schedule_.blocks = blocks.data();
  schedule_.block_count = blocks.size();
  schedule_.cfg_graph = *graph.get();

  loom_low_memory_access_map_t* memory_accesses = nullptr;
  IREE_ASSERT_OK(
      loom_low_memory_access_map_create(&module_->arena, &memory_accesses));
  PingPongAccesses ping_pong;
  loom_low_memory_access_summary_t producer_global = ping_pong.accesses[0];
  producer_global.memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  loom_low_memory_access_summary_t consumer_global = ping_pong.accesses[1];
  consumer_global.memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  IREE_ASSERT_OK(loom_low_memory_access_map_insert(
      memory_accesses, nodes_[producer].op, /*effect_ordinal=*/0,
      &producer_global));
  IREE_ASSERT_OK(loom_low_memory_access_map_insert(
      memory_accesses, nodes_[consumer].op, /*effect_ordinal=*/0,
      &consumer_global));
  IREE_ASSERT_OK(loom_low_memory_access_map_insert(
      memory_accesses, nodes_[consumer].op, /*effect_ordinal=*/1,
      &ping_pong.accesses[1]));
  schedule_.memory_accesses = memory_accesses;

  const loom_amdgpu_address_state_plan_t address_state = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(
      &schedule_, &allocation_, &address_state, &arena_, &transient_arena_,
      &plan_));
  for (iree_host_size_t i = 0; i < plan_.action_count; ++i) {
    EXPECT_NE(plan_.actions[i].reason,
              LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
  }

  iree_arena_reset(&arena_);
  iree_arena_reset(&transient_arena_);
  schedule_.memory_accesses = nullptr;
  plan_ = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(
      &schedule_, &allocation_, &address_state, &arena_, &transient_arena_,
      &plan_));
  iree_host_size_t memory_effect_action_count = 0;
  for (iree_host_size_t i = 0; i < plan_.action_count; ++i) {
    const auto& action = plan_.actions[i];
    if (action.reason != LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT) {
      continue;
    }
    ++memory_effect_action_count;
    EXPECT_EQ(action.node_index, consumer);
    EXPECT_EQ(action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
    EXPECT_EQ(action.target_count, 0);
  }
  EXPECT_EQ(memory_effect_action_count, 1u);
}

struct DisconnectedCfgGraph {
  explicit DisconnectedCfgGraph(iree_host_size_t block_count)
      : reachable_graph({{1}, {}}), blocks(block_count) {
    IREE_ASSERT_GE(block_count, 2u);
    graph = *reachable_graph.get();
    blocks[0] = graph.blocks[0];
    blocks[1] = graph.blocks[1];
    for (iree_host_size_t i = 2; i < block_count; ++i) {
      blocks[i].preorder = UINT16_MAX;
      blocks[i].parent = UINT16_MAX;
    }
    graph.blocks = blocks.data();
    graph.block_count = block_count;
  }

  const loom_cfg_graph_t* get() const { return &graph; }

  testing::CfgGraph reachable_graph;
  std::vector<loom_cfg_block_info_t> blocks;
  loom_cfg_graph_t graph = {};
};

struct ManyTermAccess {
  explicit ManyTermAccess(iree_host_size_t term_count) {
    IREE_ASSERT_LE(term_count, terms.size());
    for (iree_host_size_t i = 0; i < terms.size(); ++i) {
      terms[i] = {1, static_cast<loom_value_id_t>(i + 1),
                  static_cast<loom_value_id_t>(i + 1)};
    }
    interval.scope = &scope;
    interval.storage_id = 1;
    interval.origin.terms = terms.data();
    interval.origin.term_count = term_count;
    interval.origin.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    interval.upper = 4;
    access.memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    access.relative_interval = &interval;
  }

  int scope = 0;
  std::array<loom_symbolic_term_t, 65> terms = {};
  loom_low_memory_relative_interval_t interval = {};
  loom_low_memory_access_summary_t access = {};
};

TEST_F(AmdgpuWaitFrontierTest, AcyclicDiamondElidesDisjointPingPongWait) {
  testing::CfgGraph graph({{1, 2}, {3}, {3}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(3, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(1);
  ProcessBlock(2);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 3);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      0u);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_,
          loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC),
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, AcyclicDiamondRetainsOverlappingWait) {
  testing::CfgGraph graph({{1, 2}, {3}, {3}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(3, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[0]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(1);
  ProcessBlock(2);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 3);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, ParallelEdgesPreservePreciseDisjointElision) {
  testing::CfgGraph graph({{1, 1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(graph.get());
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      0u);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       FilteredForwardEdgePreservesPreciseDisjointElision) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  completion_nodes_[consumer].completed_before_block_exit_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR;
  Initialize(graph.get());
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      0u);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       UnreachableCompletionStillActivatesCollectorHeavyPremium) {
  constexpr iree_host_size_t kBlockCount = 801;
  DisconnectedCfgGraph graph(kBlockCount);
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
            LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  const uint32_t unreachable_node = AddNode(kBlockCount - 1);
  FinalizeSchedule(graph.get());

  loom_amdgpu_wait_frontier_t light_frontier = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &comparison_arena_, &light_frontier));
  ASSERT_NE(light_frontier.memory.precise_active_words, nullptr);

  ASSERT_FALSE(graph.blocks.back().reachable);
  completion_nodes_[unreachable_node]
      .completed_before_block_exit_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR;
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &arena_, &frontier_));
  ExpectCoarseOnly();
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, StorageLeaseOnlyChangeRequeuesCycleHeader) {
  // Block 1 first propagates lease 0 to block 3. Block 2 then contributes
  // lease 1 through the backedge, so block 1 must be requeued to carry that
  // second lease to block 3. The memory producer in block 0 is drained on
  // entry to block 1 and cannot be responsible for either requeue.
  testing::CfgGraph graph({{1}, {2, 3}, {1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t first_lease_node = AddNode(1);
  const uint32_t second_lease_node = AddNode(2);
  completion_nodes_[first_lease_node].reset_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;

  const loom_low_storage_lease_record_t records[] = {
      {
          .packet_index = 1,
          .node_index = first_lease_node,
          .block_index = 1,
          .scheduled_ordinal = 0,
          .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
          .release_class_id = LOOM_AMDGPU_WAIT_COUNTER_LDS,
      },
      {
          .packet_index = 2,
          .node_index = second_lease_node,
          .block_index = 2,
          .scheduled_ordinal = 0,
          .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
          .release_class_id = LOOM_AMDGPU_WAIT_COUNTER_LDS,
      },
  };
  const loom_low_allocation_storage_lease_t instances[] = {
      {.lease_record_index = 0},
      {.lease_record_index = 1},
  };
  loom_low_allocation_table_t allocation = {};
  allocation.storage_leases = {
      .schedule = &schedule_,
      .records = records,
      .record_count = IREE_ARRAYSIZE(records),
  };
  allocation.storage_lease_instances = instances;
  allocation.storage_lease_instance_count = IREE_ARRAYSIZE(instances);

  Initialize(graph.get(), &allocation);
  ASSERT_NE(frontier_.storage_leases.static_outgoing_words, nullptr);
  ASSERT_EQ(frontier_.storage_leases.word_count, 1u);
  EXPECT_EQ(frontier_.storage_leases.static_outgoing_words[3], UINT64_C(3));
  ASSERT_NE(frontier_.memory.static_outgoing_states, nullptr);
  for (const uint16_t counter_masks :
       frontier_.memory.static_outgoing_states[3].access_counter_masks) {
    EXPECT_EQ(counter_masks, 0u);
  }
  ASSERT_NE(frontier_.memory.precise_static_outgoing_words, nullptr);
  EXPECT_EQ(frontier_.memory.precise_static_outgoing_words[3], 0u);
}

TEST_F(AmdgpuWaitFrontierTest, CoarseOnlyBackedgeUsesResolvedState) {
  testing::CfgGraph graph({{1}, {1}});
  const uint32_t producer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, /*summary=*/nullptr);
  frontier_nodes_[producer].xcnt_group_flags =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;

  const loom_low_storage_lease_record_t records[] = {{
      .packet_index = 0,
      .node_index = producer,
      .block_index = 1,
      .scheduled_ordinal = 0,
      .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
      .release_class_id = LOOM_AMDGPU_WAIT_COUNTER_LDS,
  }};
  const loom_low_allocation_storage_lease_t instances[] = {{
      .lease_record_index = 0,
      .location_count = 1,
  }};
  loom_low_allocation_storage_lease_unit_index_t unit_index = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
      &unit_index, instances, IREE_ARRAYSIZE(instances),
      /*lease_unit_capacity=*/1, /*distinct_unit_capacity=*/1, &arena_));
  loom_low_allocation_table_t allocation = {};
  allocation.storage_leases = {
      .schedule = &schedule_,
      .records = records,
      .record_count = IREE_ARRAYSIZE(records),
  };
  allocation.storage_lease_instances = instances;
  allocation.storage_lease_instance_count = IREE_ARRAYSIZE(instances);
  allocation.storage_lease_unit_index = &unit_index;

  Initialize(graph.get(), &allocation);
  ASSERT_EQ(frontier_.memory.precise_active_words, nullptr);
  ASSERT_NE(frontier_.memory.static_outgoing_states, nullptr);
  ASSERT_NE(frontier_.memory.resolved_outgoing_states, nullptr);
  ASSERT_NE(frontier_.storage_leases.active_words, nullptr);
  ASSERT_NE(frontier_.xcnt.static_outgoing_flags, nullptr);

  ProcessBlock(1);
  memset(frontier_.memory.static_outgoing_states, 0,
         2 * sizeof(*frontier_.memory.static_outgoing_states));
  memset(frontier_.memory.resolved_outgoing_states, 0,
         2 * sizeof(*frontier_.memory.resolved_outgoing_states));
  frontier_.memory.static_outgoing_states[1].access_counter_masks[0] =
      (uint16_t)(LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE << 8);
  frontier_.memory.resolved_outgoing_states[1].access_counter_masks[0] =
      (uint16_t)(LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR << 8);
  memset(frontier_.storage_leases.static_outgoing_words, 0,
         2 * frontier_.storage_leases.word_count * sizeof(uint64_t));
  memset(frontier_.storage_leases.resolved_outgoing_words, 0,
         2 * frontier_.storage_leases.word_count * sizeof(uint64_t));
  frontier_.storage_leases.resolved_outgoing_words[1] = UINT64_C(1);
  frontier_.xcnt.static_outgoing_flags[1] =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;
  frontier_.xcnt.resolved_outgoing_flags[1] =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM;

  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_,
          loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC),
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
  EXPECT_EQ(frontier_.storage_leases.active_words[0], UINT64_C(1));
  EXPECT_EQ(loom_amdgpu_wait_frontier_active_xcnt_groups(&frontier_),
            LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       PreciseBackedgeUsesStaticStateAcrossAllFrontiers) {
  testing::CfgGraph graph({{1}, {1}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  frontier_nodes_[producer].xcnt_group_flags =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;

  const loom_low_storage_lease_record_t records[] = {{
      .packet_index = 0,
      .node_index = producer,
      .block_index = 1,
      .scheduled_ordinal = 0,
      .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
      .release_class_id = LOOM_AMDGPU_WAIT_COUNTER_LDS,
  }};
  const loom_low_allocation_storage_lease_t instances[] = {{
      .lease_record_index = 0,
      .location_count = 1,
  }};
  loom_low_allocation_storage_lease_unit_index_t unit_index = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
      &unit_index, instances, IREE_ARRAYSIZE(instances),
      /*lease_unit_capacity=*/1, /*distinct_unit_capacity=*/1, &arena_));
  loom_low_allocation_table_t allocation = {};
  allocation.storage_leases = {
      .schedule = &schedule_,
      .records = records,
      .record_count = IREE_ARRAYSIZE(records),
  };
  allocation.storage_lease_instances = instances;
  allocation.storage_lease_instance_count = IREE_ARRAYSIZE(instances);
  allocation.storage_lease_unit_index = &unit_index;

  Initialize(graph.get(), &allocation);
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  ASSERT_NE(frontier_.storage_leases.active_words, nullptr);
  ASSERT_NE(frontier_.xcnt.static_outgoing_flags, nullptr);
  ProcessBlock(1);

  memset(frontier_.memory.static_outgoing_states, 0,
         2 * sizeof(*frontier_.memory.static_outgoing_states));
  memset(frontier_.memory.resolved_outgoing_states, 0,
         2 * sizeof(*frontier_.memory.resolved_outgoing_states));
  frontier_.memory.static_outgoing_states[1].access_counter_masks[0] =
      (uint16_t)(LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE << 8);
  frontier_.memory.resolved_outgoing_states[1].access_counter_masks[0] =
      (uint16_t)(LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR << 8);
  memset(frontier_.memory.precise_static_outgoing_words, 0,
         2 * frontier_.memory.precise_word_count * sizeof(uint64_t));
  memset(frontier_.memory.precise_resolved_outgoing_words, 0,
         2 * frontier_.memory.precise_word_count * sizeof(uint64_t));
  memset(frontier_.storage_leases.static_outgoing_words, 0,
         2 * frontier_.storage_leases.word_count * sizeof(uint64_t));
  memset(frontier_.storage_leases.resolved_outgoing_words, 0,
         2 * frontier_.storage_leases.word_count * sizeof(uint64_t));
  frontier_.storage_leases.static_outgoing_words[1] = UINT64_C(1);
  frontier_.xcnt.static_outgoing_flags[1] =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;
  frontier_.xcnt.resolved_outgoing_flags[1] =
      LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM;

  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_,
          loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC),
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  EXPECT_EQ(frontier_.memory.precise_active_words[0], 0u);
  EXPECT_EQ(frontier_.storage_leases.active_words[0], UINT64_C(1));
  EXPECT_EQ(loom_amdgpu_wait_frontier_active_xcnt_groups(&frontier_),
            LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, ResolvedBackedgeCollapsesExactState) {
  testing::CfgGraph graph({{1}, {2}, {1}});
  PingPongAccesses ping_pong;
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  AddAccess(2, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(1);
  ProcessBlock(2);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       ResolvedBackedgeCollapsesExactStateWithNonmonotonicBlockOrder) {
  // The latch precedes the header in region order. Backedge classification must
  // use DFS metadata instead of dense block indices.
  testing::CfgGraph graph({{2}, {2}, {1}});
  PingPongAccesses ping_pong;
  AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(2, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(2);
  ProcessBlock(1);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 2);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, UnknownMergeRetainsCoarseWait) {
  testing::CfgGraph graph({{1, 2}, {3}, {3}, {}});
  PingPongAccesses ping_pong;
  AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  AddAccess(2, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            /*summary=*/nullptr);
  const uint32_t consumer =
      AddAccess(3, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(1);
  ProcessBlock(2);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 3);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, PartialDrainKeepsOtherPreciseCounter) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_LDS,
            &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  const auto generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_, generic_space, LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
          LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS);
  loom_amdgpu_wait_frontier_drain(&frontier_,
                                  LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_, generic_space, LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS);
  loom_amdgpu_wait_frontier_drain(&frontier_,
                                  LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_, generic_space, LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      0u);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, AcyclicAndCyclicMergeKeepExactAndCoarse) {
  testing::CfgGraph graph({{1, 2}, {4}, {3, 4}, {2}, {}});
  PingPongAccesses ping_pong;
  AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  AddAccess(3, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_LDS,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(4, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  ProcessBlock(1);
  ProcessBlock(2);
  ProcessBlock(3);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 4);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_query(
          &frontier_,
          loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC),
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
          LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, MalformedCfgFallsBackToCoarseState) {
  testing::CfgGraph graph_fixture({{1}, {}});
  loom_cfg_graph_t graph = *graph_fixture.get();
  graph.malformed = true;
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(&graph);
  ExpectCoarseOnly();
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       InvalidInRangeDfsMetadataFallsBackToCoarseState) {
  testing::CfgGraph graph_fixture({{1}, {0}});
  loom_cfg_graph_t graph = *graph_fixture.get();
  std::vector<loom_cfg_block_info_t> blocks(graph.blocks,
                                            graph.blocks + graph.block_count);
  graph.blocks = blocks.data();
  ASSERT_EQ(blocks[0].preorder, 0u);
  ASSERT_EQ(blocks[0].preorder_end, 2u);
  blocks[0].preorder_end = 1;

  PingPongAccesses ping_pong;
  const uint32_t consumer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  Initialize(&graph);
  ExpectCoarseOnly();
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 0);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, HiddenPredecessorCycleFallsBackToCoarseState) {
  testing::CfgGraph graph_fixture({{1, 2}, {}, {1}});
  loom_cfg_graph_t graph = *graph_fixture.get();
  std::vector<loom_cfg_block_info_t> blocks(graph.blocks,
                                            graph.blocks + graph.block_count);
  // Declared edges are 0->1, 0->2, and 2->1. The extra predecessor-only 1->2
  // edge closes a hidden cycle and runs backward in the declared graph's RPO.
  std::array<uint16_t, 4> successor_indices = {1, 2, 1, 0};
  std::array<uint16_t, 4> predecessor_indices = {0, 2, 0, 1};
  std::array<loom_cfg_edge_index_t, 4> successor_edges = {0, 1, 2, 3};
  std::array<loom_cfg_edge_index_t, 4> predecessor_edges = {0, 2, 1, 3};
  std::array<loom_cfg_edge_info_t, 4> edges = {};
  for (iree_host_size_t i = 0; i < graph.edge_count; ++i) {
    edges[i] = graph.edges[i];
  }
  edges[3].source_block_index = 1;
  edges[3].target_block_index = 2;
  graph.blocks = blocks.data();
  graph.edges = edges.data();
  graph.successor_indices = successor_indices.data();
  graph.successor_edge_indices = successor_edges.data();
  graph.predecessor_indices = predecessor_indices.data();
  graph.predecessor_edge_indices = predecessor_edges.data();
  graph.edge_count = successor_indices.size();
  blocks[2].predecessor_count = 2;

  PingPongAccesses ping_pong;
  AddAccess(1, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(2, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  Initialize(&graph);
  ExpectCoarseOnly();
  ProcessBlock(1);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 2);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, OversizedPreciseStateFallsBackToCoarseState) {
  constexpr iree_host_size_t kBlockCount = UINT16_MAX;
  constexpr uint16_t kPreciseEffectCount = 1024;
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const loom_low_schedule_effect_use_t prototype = effects_[0];
  for (uint16_t effect_ordinal = 1; effect_ordinal < kPreciseEffectCount;
       ++effect_ordinal) {
    loom_low_schedule_effect_use_t effect = prototype;
    effect.effect_ordinal = effect_ordinal;
    effects_.push_back(effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[producer], effect_ordinal,
        &ping_pong.accesses[0]));
  }
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);

  std::vector<loom_cfg_block_info_t> graph_blocks(kBlockCount);
  std::array<uint16_t, 1> successor_indices = {1};
  std::array<uint16_t, 1> predecessor_indices = {0};
  const std::array<uint16_t, 2> reverse_postorder = {0, 1};
  graph_blocks[0].successor_count = 1;
  graph_blocks[0].reachable = true;
  graph_blocks[0].preorder = 0;
  graph_blocks[0].preorder_end = 2;
  graph_blocks[1].predecessor_count = 1;
  graph_blocks[1].reachable = true;
  graph_blocks[1].preorder = 1;
  graph_blocks[1].preorder_end = 2;
  graph_blocks[1].parent = 0;
  loom_cfg_graph_t graph = {};
  graph.blocks = graph_blocks.data();
  graph.successor_indices = successor_indices.data();
  graph.predecessor_indices = predecessor_indices.data();
  graph.block_count = kBlockCount;
  graph.edge_count = 1;
  graph.reverse_postorder = {reverse_postorder.data(),
                             reverse_postorder.size()};

  Initialize(&graph);
  ExpectCoarseOnly();
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, PreciseAccessCount256IsAdmitted) {
  constexpr uint16_t kEffectsPerNode = 128;
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  const loom_low_schedule_effect_use_t producer_prototype = effects_[0];
  const loom_low_schedule_effect_use_t consumer_prototype = effects_[1];
  for (uint16_t effect_ordinal = 1; effect_ordinal < kEffectsPerNode;
       ++effect_ordinal) {
    loom_low_schedule_effect_use_t producer_effect = producer_prototype;
    producer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(producer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[producer], effect_ordinal,
        &ping_pong.accesses[0]));
    loom_low_schedule_effect_use_t consumer_effect = consumer_prototype;
    consumer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(consumer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[consumer], effect_ordinal,
        &ping_pong.accesses[1]));
  }

  Initialize(graph.get());
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  EXPECT_EQ(frontier_.memory.precise_access_count, 256u);
  EXPECT_EQ(frontier_.memory.precise_word_count, 32u);
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      0u);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       AdditiveScoreOnlyCapFallsBackWithoutRetainedScratch) {
  constexpr uint16_t kEffectsPerDirection = 172;
  constexpr uint64_t kPreciseAccessCount = 2 * kEffectsPerDirection;
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  const loom_low_schedule_effect_use_t producer_prototype = effects_[0];
  const loom_low_schedule_effect_use_t consumer_prototype = effects_[1];
  for (uint16_t effect_ordinal = 1;
       effect_ordinal < kEffectsPerDirection; ++effect_ordinal) {
    loom_low_schedule_effect_use_t producer_effect = producer_prototype;
    producer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(producer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[producer], effect_ordinal,
        &ping_pong.accesses[0]));
    loom_low_schedule_effect_use_t consumer_effect = consumer_prototype;
    consumer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(consumer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[consumer], effect_ordinal,
        &ping_pong.accesses[1]));
  }
  loom_low_memory_access_summary_t generic_access = ping_pong.accesses[0];
  generic_access.memory_space = LOOM_LOW_MEMORY_SPACE_GENERIC;
  AddAccess(1, LOOM_LOW_EFFECT_KIND_READ, LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD,
            &generic_access);
  while (nodes_.size() < kMaxNodeCount) {
    AddNode(1);
  }
  FinalizeSchedule(graph.get());

  loom_amdgpu_wait_frontier_precise_budget_input_t budget_input = {};
  budget_input.precise_access_count = kPreciseAccessCount;
  budget_input.producer_counter_bit_count = kPreciseAccessCount;
  budget_input.read_producer_counter_bit_count = kEffectsPerDirection;
  budget_input.write_producer_counter_bit_count = kEffectsPerDirection;
  budget_input.precise_read_count = kEffectsPerDirection;
  budget_input.precise_write_count = kEffectsPerDirection;
  budget_input.read_effective_term_sum = kEffectsPerDirection;
  budget_input.write_effective_term_sum = kEffectsPerDirection;
  budget_input.precise_word_count = 43;
  budget_input.block_count = 2;
  budget_input.reachable_block_count = 2;
  budget_input.unfiltered_forward_edge_count = 1;
  budget_input.memory_space_count =
      LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MEMORY_SPACE_COUNT;
  budget_input.maximum_effective_term_count = 1;
  budget_input.allocation_byte_count = 9881;
  budget_input.effect_use_count = effects_.size();
  budget_input.cfg_edge_count = 1;
  budget_input.scheduled_node_count = nodes_.size();
  budget_input.node_count = nodes_.size();
  budget_input.dependency_read_query_count = 1;
  budget_input.collector_heavy_path_possible = true;
  budget_input.barrier_query_count = 131;
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_calculate(&budget_input,
                                                                 &usage));
  EXPECT_EQ(usage.structural_score_scaled, UINT64_C(2685061287321));
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));

  runtime_bounds_.barrier_query_count = 131;
  loom_amdgpu_wait_frontier_t admitted_frontier = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &comparison_arena_, &admitted_frontier));
  ASSERT_NE(admitted_frontier.memory.precise_active_words, nullptr);
  EXPECT_EQ(admitted_frontier.memory.precise_access_count, 344u);
  EXPECT_EQ(admitted_frontier.memory.precise_word_count, 43u);
  iree_arena_reset(&comparison_arena_);

  runtime_bounds_.barrier_query_count = 132;
  budget_input.barrier_query_count = 132;
  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_calculate(&budget_input,
                                                                 &usage));
  EXPECT_EQ(usage.structural_score_scaled, UINT64_C(2690005425867));
  auto hard_guard_usage = usage;
  hard_guard_usage.structural_score_scaled =
      LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_SCORE_SCALED;
  EXPECT_TRUE(loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(
      &hard_guard_usage));
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));

  const iree_host_size_t initial_used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &arena_, &frontier_));
  ExpectCoarseOnly();

  loom_low_schedule_table_t control_schedule = schedule_;
  control_schedule.memory_accesses = nullptr;
  loom_amdgpu_wait_frontier_t control_frontier = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &control_schedule, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &comparison_arena_, &control_frontier));
  EXPECT_EQ(arena_.used_allocation_size - initial_used_bytes,
            comparison_arena_.used_allocation_size);

  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       FinalPreciseRetainedAllocationFailureRollsBackAndRetries) {
  constexpr uint16_t kEffectsPerNode = 128;
  constexpr iree_host_size_t kPoolBlockSize = 512;
  constexpr iree_host_size_t kStateByteLength = 512;
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  const loom_low_schedule_effect_use_t producer_prototype = effects_[0];
  const loom_low_schedule_effect_use_t consumer_prototype = effects_[1];
  for (uint16_t effect_ordinal = 1; effect_ordinal < kEffectsPerNode;
       ++effect_ordinal) {
    loom_low_schedule_effect_use_t producer_effect = producer_prototype;
    producer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(producer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[producer], effect_ordinal,
        &ping_pong.accesses[0]));
    loom_low_schedule_effect_use_t consumer_effect = consumer_prototype;
    consumer_effect.effect_ordinal = effect_ordinal;
    effects_.push_back(consumer_effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[consumer], effect_ordinal,
        &ping_pong.accesses[1]));
  }
  FinalizeSchedule(graph.get());

  struct StateAllocationFailureState {
    bool failure_enabled = true;
    iree_host_size_t matching_request_count = 0;
    iree_host_size_t failed_byte_length = 0;
    iree_host_size_t failure_count = 0;
  } failure_state;
  const iree_allocator_t failing_allocator = {
      &failure_state,
      +[](void* self, iree_allocator_command_t command, const void* parameters,
          void** pointer) -> iree_status_t {
        auto* state = static_cast<StateAllocationFailureState*>(self);
        if (command == IREE_ALLOCATOR_COMMAND_MALLOC) {
          const auto* allocation =
              static_cast<const iree_allocator_alloc_params_t*>(parameters);
          const iree_host_size_t state_allocation_byte_length =
              sizeof(iree_arena_oversized_allocation_t) + kStateByteLength;
          // Matching request #2 is precise_resolved_outgoing_words, the
          // seventh and final retained allocation after six successes.
          if (allocation->byte_length == state_allocation_byte_length &&
              ++state->matching_request_count == 2 &&
              state->failure_enabled) {
            state->failed_byte_length = allocation->byte_length;
            ++state->failure_count;
            *pointer = nullptr;
            return iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "injected precise retained-allocation failure");
          }
        }
        const iree_allocator_t system_allocator = iree_allocator_system();
        return system_allocator.ctl(system_allocator.self, command, parameters,
                                    pointer);
      }};
  iree_arena_block_pool_t failing_pool;
  iree_arena_allocator_t failing_arena;
  iree_arena_block_pool_initialize(kPoolBlockSize, failing_allocator,
                                   &failing_pool);
  iree_arena_initialize(&failing_pool, &failing_arena);

  iree_status_t status = loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &failing_arena, &frontier_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(failure_state.matching_request_count, 2u);
  EXPECT_EQ(failure_state.failure_count, 1u);
  EXPECT_EQ(failure_state.failed_byte_length,
            sizeof(iree_arena_oversized_allocation_t) + kStateByteLength);
  EXPECT_EQ(failing_arena.total_allocation_size, 0u);
  EXPECT_EQ(failing_arena.used_allocation_size, 0u);
  EXPECT_EQ(failing_arena.allocation_head, nullptr);
  EXPECT_EQ(failing_arena.block_head, nullptr);
  EXPECT_EQ(failing_arena.block_tail, nullptr);
  EXPECT_EQ(failing_arena.block_bytes_remaining, 0u);
  ExpectCoarseOnly();

  failure_state.failure_enabled = false;
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &failing_arena, &frontier_));
  EXPECT_EQ(failure_state.matching_request_count, 4u);
  EXPECT_EQ(failure_state.failure_count, 1u);
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  EXPECT_EQ(frontier_.memory.precise_access_count, 256u);
  EXPECT_EQ(frontier_.memory.precise_word_count, 32u);
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      0u);
  loom_amdgpu_wait_frontier_end_block(&frontier_);

  iree_arena_deinitialize(&failing_arena);
  iree_arena_block_pool_deinitialize(&failing_pool);
}

TEST_F(AmdgpuWaitFrontierTest,
       ProducerCounterBitCountAbovePreciseAccessCountFallsBack) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  effects_[0].counter_id = LOOM_AMDGPU_WAIT_COUNTER_NONE;
  frontier_nodes_[producer].write_counter_mask |=
      LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR;
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);

  Initialize(graph.get());
  ExpectCoarseOnly();
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
          LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, PreciseAccessCapFallsBackWithoutPublication) {
  constexpr uint16_t kPreciseEffectCount = 345;
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  const loom_low_schedule_effect_use_t prototype = effects_[0];
  for (uint16_t effect_ordinal = 1; effect_ordinal < kPreciseEffectCount - 1;
       ++effect_ordinal) {
    loom_low_schedule_effect_use_t effect = prototype;
    effect.effect_ordinal = effect_ordinal;
    effects_.push_back(effect);
    IREE_ASSERT_OK(loom_low_memory_access_map_insert(
        memory_accesses_, &ops_[producer], effect_ordinal,
        &ping_pong.accesses[0]));
  }
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);

  FinalizeSchedule(graph.get());
  const iree_host_size_t initial_used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &arena_, &frontier_));
  ExpectCoarseOnly();

  loom_low_schedule_table_t control_schedule = schedule_;
  control_schedule.memory_accesses = nullptr;
  loom_amdgpu_wait_frontier_t control_frontier = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &control_schedule, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &comparison_arena_, &control_frontier));
  EXPECT_EQ(arena_.used_allocation_size - initial_used_bytes,
            comparison_arena_.used_allocation_size);

  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, EffectiveTermCapAdmits64) {
  testing::CfgGraph graph({{1}, {}});
  ManyTermAccess access(64);
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &access.access);
  AddAccess(1, LOOM_LOW_EFFECT_KIND_READ, LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD,
            &access.access);
  Initialize(graph.get());
  EXPECT_NE(frontier_.memory.precise_active_words, nullptr);
}

TEST_F(AmdgpuWaitFrontierTest, EffectiveTermCapRejects65Conservatively) {
  testing::CfgGraph graph({{1}, {}});
  ManyTermAccess access(65);
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &access.access);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &access.access);
  FinalizeSchedule(graph.get());
  const iree_host_size_t initial_used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &schedule_, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &arena_, &frontier_));
  ExpectCoarseOnly();

  loom_low_schedule_table_t control_schedule = schedule_;
  control_schedule.memory_accesses = nullptr;
  loom_amdgpu_wait_frontier_t control_frontier = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
      &control_schedule, /*allocation=*/nullptr, frontier_nodes_.data(),
      completion_nodes_.data(), /*dependencies=*/nullptr,
      /*dependency_count=*/0, /*planned_block_drain_counter_masks=*/nullptr,
      &runtime_bounds_, &comparison_arena_, &control_frontier));
  EXPECT_EQ(arena_.used_allocation_size - initial_used_bytes,
            comparison_arena_.used_allocation_size);

  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, InvalidXcntDomainRejectsPreciseState) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  const uint32_t consumer =
      AddAccess(1, LOOM_LOW_EFFECT_KIND_READ,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD, &ping_pong.accesses[1]);
  frontier_nodes_[0].xcnt_group_flags = 1u << 2;
  Initialize(graph.get());
  ExpectCoarseOnly();
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_EQ(
      loom_amdgpu_wait_frontier_memory_dependency_mask(&frontier_, consumer),
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, ExternalResultCannotBeProvenByMemoryState) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  AddExternalRead(producer, LOOM_AMDGPU_WAIT_COUNTER_ASYNC);
  Initialize(graph.get());
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  EXPECT_FALSE(loom_amdgpu_wait_frontier_producer_is_complete(
      &frontier_, producer, LOOM_AMDGPU_WAIT_COUNTER_MASK_ASYNC));
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest,
       SameCounterExternalResultCannotBeProvenByEmptyMemoryState) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  const uint32_t producer =
      AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE,
                LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE, &ping_pong.accesses[0]);
  AddExternalRead(producer, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  Initialize(graph.get());
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  ASSERT_NE(frontier_.memory.precise_active_words, nullptr);
  for (iree_host_size_t i = 0; i < frontier_.memory.precise_word_count; ++i) {
    frontier_.memory.precise_active_words[i] = 0;
  }
  EXPECT_FALSE(loom_amdgpu_wait_frontier_producer_is_complete(
      &frontier_, producer, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE));
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

TEST_F(AmdgpuWaitFrontierTest, PlanBarrierExitAndTensorQueriesSeePreciseState) {
  testing::CfgGraph graph({{1}, {}});
  PingPongAccesses ping_pong;
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
            &ping_pong.accesses[0]);
  AddAccess(0, LOOM_LOW_EFFECT_KIND_WRITE, LOOM_AMDGPU_WAIT_COUNTER_TENSOR,
            &ping_pong.accesses[1]);
  Initialize(graph.get());
  ProcessBlock(0);
  loom_amdgpu_wait_frontier_begin_block(&frontier_, 1);
  const auto generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  // Mirror the generic queries used by the plan's barrier, program-exit, and
  // tensor-issue handlers so precise-only incoming state cannot bypass them.
  const uint32_t all_pending = loom_amdgpu_wait_frontier_memory_query(
      &frontier_, generic_space,
      LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ |
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
  EXPECT_EQ(all_pending & (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
                           LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR),
            LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
                LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
  EXPECT_EQ(loom_amdgpu_wait_frontier_memory_query(
                &frontier_, generic_space,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
                LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
            LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  EXPECT_TRUE(
      iree_any_bit_set(all_pending, LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR));
  loom_amdgpu_wait_frontier_end_block(&frontier_);
}

}  // namespace
}  // namespace loom
