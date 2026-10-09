// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/immediates.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

// The planner consumes scheduled descriptor facts and dependency edges, not
// source IR or assembly. These compact fixtures keep the independent
// issue-limit and memory-completion inputs explicit.
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
    loom_named_attr_t attr = {.name_id = name, .value = loom_attr_i64(value)};
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
    loom_low_schedule_dependency_t dependency = {
        .producer_node = producer,
        .consumer_node = consumer,
        .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT};
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &schedule_.dependencies, dependency, &module_->arena));
    ++schedule_.effect_dependencies.count;
  }

  void MemoryCompletion(uint32_t producer_effect, uint32_t consumer_effect) {
    memory_completions_.push_back({producer_effect, consumer_effect});
    schedule_.memory_completion_edges = memory_completions_.data();
    schedule_.memory_completion_edge_count = memory_completions_.size();
  }

  struct CrossBlockMemoryFixture {
    uint32_t producer_node;
    uint32_t consumer_node;
    uint32_t producer_effect;
    uint32_t consumer_effect;
  };

  void ConfigureRefinedAcyclicMemory(const testing::CfgGraph& graph,
                                     CrossBlockMemoryFixture* out_fixture) {
    const uint32_t producer_effect = static_cast<uint32_t>(effects_.size());
    AppendImmediate(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32,
                    IREE_SV("scope"), LOOM_CACHE_SCOPE_DEVICE);
    const uint32_t producer_node = static_cast<uint32_t>(nodes_.size() - 1);
    IREE_ASSERT_EQ(effects_.size(), producer_effect + 1);
    effects_[producer_effect].flags |=
        LOOM_LOW_SCHEDULE_EFFECT_USE_FLAG_REFINED_MEMORY;

    loom_region_t* body = loom_low_func_def_body(schedule_.function_op);
    loom_block_t* successor = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module_, body, &successor));
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_low_br_build(&builder_, successor, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    loom_low_schedule_node_t branch_node =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment conversion
             // differs from list initialization.
    branch_node.op = branch;
    branch_node.block_index = 0;
    branch_node.source_ordinal = nodes_.size();
    branch_node.scheduled_ordinal = 1;
    branch_node.kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
    branch_node.traits = branch->traits;
    nodes_.push_back(branch_node);

    loom_builder_set_block(&builder_, successor);
    const uint32_t consumer_effect = static_cast<uint32_t>(effects_.size());
    const iree_host_size_t consumer_hazard = hazards_.size();
    Tensor();
    const uint32_t consumer_node = static_cast<uint32_t>(nodes_.size() - 1);
    nodes_[consumer_node].block_index = 1;
    nodes_[consumer_node].scheduled_ordinal = 0;
    for (iree_host_size_t i = consumer_effect; i < effects_.size(); ++i) {
      effects_[i].block_index = 1;
      effects_[i].scheduled_ordinal = 0;
      effects_[i].flags |= LOOM_LOW_SCHEDULE_EFFECT_USE_FLAG_REFINED_MEMORY;
    }
    for (iree_host_size_t i = consumer_hazard; i < hazards_.size(); ++i) {
      hazards_[i].block_index = 1;
      hazards_[i].scheduled_ordinal = 0;
    }

    FinalizeSchedule();
    nodes_.back().block_index = 1;
    nodes_.back().scheduled_ordinal = 1;
    blocks_.resize(2);
    blocks_[0] = {
        .block = block_.block,
        .node_start = producer_node,
        .node_count = 2,
        .scheduled_node_start = 0,
        .scheduled_node_count = 2,
    };
    blocks_[1] = {
        .block = successor,
        .node_start = consumer_node,
        .node_count = 2,
        .scheduled_node_start = 2,
        .scheduled_node_count = 2,
    };
    schedule_.blocks = blocks_.data();
    schedule_.block_count = blocks_.size();
    schedule_.cfg_graph = *graph.get();
    schedule_.nodes = nodes_.data();
    schedule_.effect_uses = effects_.data();
    schedule_.hazard_uses = hazards_.data();
    *out_fixture = {producer_node, consumer_node, producer_effect,
                    consumer_effect};
  }

  void FinalizeSchedule() {
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    loom_low_schedule_node_t node =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
             // differs from declaration order.
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
    schedule_.memory_completion_edges = memory_completions_.data();
    schedule_.memory_completion_edge_count = memory_completions_.size();
    schedule_.hazard_uses = hazards_.data();
    schedule_.hazard_use_count = hazards_.size();
  }

  iree_status_t BuildPlan() {
    if (allocation_.storage_leases.schedule == nullptr) {
      loom_low_storage_lease_provider_t provider = {};
      loom_amdgpu_storage_lease_provider(&provider);
      IREE_RETURN_IF_ERROR(loom_low_storage_lease_build(
          &schedule_, &provider, &module_->arena, &allocation_.storage_leases));
    }
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
    loom_low_schedule_node_t node =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment conversion
             // differs from list initialization.
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
      loom_low_schedule_effect_use_t use = {
          .node_index = node.source_ordinal,
          .scheduled_ordinal = node.scheduled_ordinal,
          .effect_ordinal = i,
          .kind = effect.kind,
          .memory_space = effect.memory_space,
          .scope_id = effect.scope_id,
          .effect_flags = effect.flags,
          .counter_id = effect.counter_id,
          .width_bits = effect.width_bits};
      effects_.push_back(use);
    }
    for (uint16_t i = 0; i < node.schedule_class->hazard_count; ++i) {
      const auto& hazard =
          descriptors_->hazards[node.schedule_class->hazard_start + i];
      loom_low_schedule_hazard_use_t use = {
          .node_index = node.source_ordinal,
          .scheduled_ordinal = node.scheduled_ordinal,
          .hazard_ordinal = i,
          .kind = hazard.kind,
          .reference_kind = hazard.reference_kind,
          .reference_id = hazard.reference_id,
          .producer_stage = hazard.producer_stage,
          .consumer_stage = hazard.consumer_stage,
          .distance = hazard.distance,
          .hazard_flags = hazard.flags};
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
  // Multi-block schedule storage used by control-flow fixtures.
  std::vector<loom_low_schedule_block_t> blocks_;
  // Descriptor nodes in scheduled order.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Node indices retained for the result's borrowed schedule.
  std::vector<uint32_t> order_;
  // Descriptor memory and counter effects.
  std::vector<loom_low_schedule_effect_use_t> effects_;
  // Exact cross-block memory completions retained by the generic scheduler.
  std::vector<loom_low_schedule_memory_completion_edge_t> memory_completions_;
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

TEST_F(AmdgpuWaitPlanTest, RefinedCrossBlockMemoryOmitsDisjointWait) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  IREE_ASSERT_OK(BuildPlan());
  EXPECT_EQ(plan_.action_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, RefinedCrossBlockMemoryWaitsForExactCompletion) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  MemoryCompletion(fixture.producer_effect, fixture.consumer_effect);
  IREE_ASSERT_OK(BuildPlan());
  ASSERT_EQ(plan_.action_count, 1u);
  const loom_amdgpu_wait_plan_action_t& action = plan_.actions[0];
  EXPECT_EQ(action.kind, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED);
  EXPECT_EQ(action.block_index, 1u);
  EXPECT_EQ(action.node_index, fixture.consumer_node);
  EXPECT_EQ(action.scheduled_ordinal, 0u);
  EXPECT_EQ(action.producer_node, fixture.producer_node);
  EXPECT_EQ(action.consumer_node, fixture.consumer_node);
  EXPECT_EQ(action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(action.target_count, 0);
  EXPECT_EQ(action.reason, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
}

TEST_F(AmdgpuWaitPlanTest, MixedConsumerEffectsRemainConservative) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  loom_low_schedule_effect_use_t unrefined_read =
      effects_[fixture.consumer_effect];
  unrefined_read.flags = 0;
  effects_.push_back(unrefined_read);
  schedule_.effect_uses = effects_.data();
  schedule_.effect_use_count = effects_.size();

  IREE_ASSERT_OK(BuildPlan());
  ASSERT_EQ(plan_.action_count, 1u);
  const loom_amdgpu_wait_plan_action_t& action = plan_.actions[0];
  EXPECT_EQ(action.block_index, 1u);
  EXPECT_EQ(action.node_index, fixture.consumer_node);
  EXPECT_EQ(action.producer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(action.consumer_node, fixture.consumer_node);
  EXPECT_EQ(action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(action.target_count, 0);
  EXPECT_EQ(action.reason, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
}

TEST_F(AmdgpuWaitPlanTest, BackedgeMemoryRemainsConservative) {
  testing::CfgGraph graph({{1}, {0}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  IREE_ASSERT_OK(BuildPlan());
  ASSERT_EQ(plan_.action_count, 2u);
  const loom_amdgpu_wait_plan_action_t& header_action = plan_.actions[0];
  EXPECT_EQ(header_action.block_index, 0u);
  EXPECT_EQ(header_action.node_index, fixture.producer_node);
  EXPECT_EQ(header_action.producer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(header_action.consumer_node, fixture.producer_node);
  EXPECT_EQ(header_action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
  EXPECT_EQ(header_action.target_count, 0);
  EXPECT_EQ(header_action.reason, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
  const loom_amdgpu_wait_plan_action_t& body_action = plan_.actions[1];
  EXPECT_EQ(body_action.block_index, 1u);
  EXPECT_EQ(body_action.node_index, fixture.consumer_node);
  EXPECT_EQ(body_action.producer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(body_action.consumer_node, fixture.consumer_node);
  EXPECT_EQ(body_action.counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(body_action.target_count, 0);
  EXPECT_EQ(body_action.reason, LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
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

}  // namespace
}  // namespace loom
