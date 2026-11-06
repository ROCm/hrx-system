// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#include <algorithm>
#include <initializer_list>
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
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"
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
    schedule_model_ = loom_low_descriptor_set_lookup_schedule_model_by_id(
        descriptors_, UINT64_C(0x2a42ccc9fb04b34e));
    ASSERT_NE(schedule_model_, nullptr);
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
    loom_string_id_t helper_name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("helper"), &helper_name));
    uint16_t helper_symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_add_symbol(module_, helper_name, &helper_symbol));
    loom_op_t* helper = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY, 0, 0, 0, 0,
        0, LOOM_INLINE_POLICY_NOINLINE, 0, 0, descriptor_set, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, {0, helper_symbol}, nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &helper));
    call_target_ = loom_low_func_def_callee(helper);
    loom_builder_t helper_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_low_func_def_body(helper)),
        &helper_builder);
    loom_op_t* helper_return = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(
        &helper_builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &helper_return));
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
        &builder_, 0, 0, 0, 0, 0, 0, 0, 0, 0, descriptor_set, {}, 0, {}, {},
        LOOM_STRING_ID_INVALID, {}, {0, symbol}, argument_types, 4, nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    loom_block_t* block =
        loom_region_entry_block(loom_low_func_def_body(function));
    block_.block = block;
    loom_builder_initialize(module_, &module_->arena, block, &builder_);
    schedule_.module = module_;
    schedule_.function_op = function;
    schedule_.target.descriptor_set = descriptors_;
    schedule_.target.schedule_model = schedule_model_;
    schedule_.target.target_facts = &facts_.base;
    schedule_.blocks = &block_;
    schedule_.block_count = 1;
    schedule_.value_count = 4;
    schedule_.value_ids = block->arg_ids;
    schedule_.value_producer_nodes = value_producer_nodes_;
    loom_low_schedule_dependency_graph_initialize(&effect_dependencies_);
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
    physical_extents_.assign(descriptors_->reg_class_count, 0);
    fixed_physical_extents_.assign(descriptors_->reg_class_count, 0);
    for (const loom_low_allocation_assignment_t& assignment : assignments_) {
      physical_extents_[assignment.descriptor_reg_class_id] =
          std::max(physical_extents_[assignment.descriptor_reg_class_id],
                   assignment.location_base + assignment.location_count);
    }
    allocation_.physical_extents.ends_by_reg_class = physical_extents_.data();
    allocation_.physical_extents.fixed_ends_by_reg_class =
        fixed_physical_extents_.data();
    allocation_.physical_extents.count = physical_extents_.size();
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

  static loom_low_move_location_t Physical(uint16_t register_class,
                                           uint32_t location) {
    return loom_low_move_location_t{
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .descriptor_reg_class_id = register_class,
        .location = location,
    };
  }

  static loom_low_move_t Move(uint16_t register_class, uint32_t destination,
                              uint32_t source) {
    return loom_low_move_t{
        .destination = Physical(register_class, destination),
        .source = Physical(register_class, source),
    };
  }

  static const loom_amdgpu_call_state_t* QueryCallState(
      void* user_data, loom_symbol_ref_t callee) {
    auto* test = static_cast<AmdgpuWaitPlanTest*>(user_data);
    IREE_ASSERT_EQ(callee.module_id, test->call_target_.module_id);
    IREE_ASSERT_EQ(callee.symbol_id, test->call_target_.symbol_id);
    return &test->call_state_;
  }

  loom_amdgpu_wait_plan_options_t CallOptions(uint32_t counter_mask) {
    call_state_.counter_mask =
        static_cast<loom_amdgpu_wait_counter_mask_t>(counter_mask);
    return loom_amdgpu_wait_plan_options_t{
        .call_states =
            {
                .query = QueryCallState,
                .user_data = this,
            },
    };
  }

  void Call(std::initializer_list<loom_low_move_t> arguments = {},
            std::initializer_list<loom_low_move_t> results = {}) {
    const uint32_t node_index = static_cast<uint32_t>(nodes_.size());
    loom_op_t* call = nullptr;
    IREE_ASSERT_OK(loom_low_func_call_build(&builder_, 0, 0, 0, 0, call_target_,
                                            nullptr, 0, nullptr, 0, nullptr, 0,
                                            LOOM_LOCATION_UNKNOWN, &call));
    loom_low_schedule_node_t node = {
        .op = call,
        .traits = call->traits,
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
    };
    node.scheduled_ordinal = node_index;
    nodes_.push_back(node);
    const iree_host_size_t argument_start = moves_.size();
    moves_.insert(moves_.end(), arguments);
    const iree_host_size_t result_start = moves_.size();
    moves_.insert(moves_.end(), results);
    call_moves_.push_back({
        .source_ordinal = node_index,
        .argument_count = static_cast<uint16_t>(arguments.size()),
        .result_count = static_cast<uint16_t>(results.size()),
        .arguments =
            {
                .moves =
                    {
                        .start = argument_start,
                        .count = arguments.size(),
                    },
            },
        .results =
            {
                .moves =
                    {
                        .start = result_start,
                        .count = results.size(),
                    },
            },
    });
    for (const loom_low_move_t& move : moves_) {
      for (const loom_low_move_location_t* location :
           {&move.source, &move.destination}) {
        if (location->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
          continue;
        }
        physical_extents_[location->descriptor_reg_class_id] =
            std::max(physical_extents_[location->descriptor_reg_class_id],
                     location->location + 1);
      }
    }
    allocation_.call_moves = call_moves_.data();
    allocation_.call_move_count = call_moves_.size();
    allocation_.moves = moves_.data();
    allocation_.move_count = moves_.size();
  }

  void AppendImmediate(loom_amdgpu_descriptor_ref_t descriptor_ref,
                       iree_string_view_t field_name, uint16_t value) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&builder_, field_name, &name));
    loom_named_attr_t attr = {.name_id = name, .value = loom_attr_i64(value)};
    Append(descriptor_ref, loom_make_named_attr_slice(&attr, 1));
  }

  void MemoryDependency(uint32_t producer, uint32_t consumer) {
    const loom_low_schedule_dependency_t dependency = {
        .producer_node = producer,
        .consumer_node = consumer,
        .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &effect_dependencies_, dependency, &module_->arena));
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_retain_range(
        &effect_dependencies_,
        {0, static_cast<uint32_t>(effect_dependencies_.count)}, &module_->arena,
        &schedule_.effect_dependencies));
  }

  void MemoryCompletion(uint32_t producer_effect, uint32_t consumer_effect) {
    memory_completions_.push_back({producer_effect, consumer_effect});
    schedule_.memory_completion_edges = memory_completions_.data();
    schedule_.memory_completion_edge_count = memory_completions_.size();
  }

  struct CrossBlockMemoryFixture {
    // Ordinary global store in the predecessor block.
    uint32_t producer_node;
    // Tensor transfer in the successor block.
    uint32_t consumer_node;
    // Producer's global write effect.
    uint32_t producer_effect;
    // Consumer's first memory effect.
    uint32_t consumer_effect;
  };

  void ConfigureRefinedAcyclicMemory(const testing::CfgGraph& graph,
                                     loom_memory_access_flags_t access_flags,
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
    loom_low_schedule_node_t branch_node = {
        .op = branch,
        .block_index = 0,
    };
    branch_node.scheduled_ordinal = 1;
    branch_node.kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
    branch_node.traits = branch->traits;
    nodes_.push_back(branch_node);

    loom_builder_set_block(&builder_, successor);
    const uint32_t consumer_effect = static_cast<uint32_t>(effects_.size());
    Append(LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D2,
           loom_named_attr_slice_empty(), access_flags);
    const uint32_t consumer_node = static_cast<uint32_t>(nodes_.size() - 1);
    nodes_[consumer_node].block_index = 1;
    nodes_[consumer_node].scheduled_ordinal = 0;
    for (iree_host_size_t i = consumer_effect; i < effects_.size(); ++i) {
      effects_[i].block_index = 1;
      effects_[i].scheduled_ordinal = 0;
      if (!iree_any_bit_set(access_flags, LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
        effects_[i].flags |= LOOM_LOW_SCHEDULE_EFFECT_USE_FLAG_REFINED_MEMORY;
      }
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

  void ConfigureRefinedAcyclicMemory(const testing::CfgGraph& graph,
                                     CrossBlockMemoryFixture* out_fixture) {
    ConfigureRefinedAcyclicMemory(graph, /*access_flags=*/0, out_fixture);
  }

  void FinalizeSchedule() {
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_low_return_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    loom_low_schedule_node_t node = {
        .op = return_op,
        .traits = return_op->traits,
        .kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR,
    };
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

  iree_status_t BuildPlan(
      const loom_amdgpu_wait_plan_options_t* options = nullptr) {
    if (allocation_.storage_leases.schedule == nullptr) {
      loom_low_storage_lease_provider_t provider = {};
      loom_amdgpu_storage_lease_provider(&provider);
      IREE_RETURN_IF_ERROR(loom_low_storage_lease_build(
          &schedule_, &provider, &module_->arena, &allocation_.storage_leases));
    }
    const loom_amdgpu_address_state_plan_t address_state = {};
    return loom_amdgpu_wait_plan_build(&schedule_, &allocation_, options,
                                       &address_state, /*matrix_arena=*/nullptr,
                                       &arena_, &transient_arena_,
                                       /*out_matrix_results=*/nullptr, &plan_);
  }

  void Build(const loom_amdgpu_wait_plan_options_t* options = nullptr) {
    FinalizeSchedule();
    IREE_ASSERT_OK(BuildPlan(options));
  }

  void BuildWaitStates(bool callable = false,
                       loom_amdgpu_call_fixed_state_mask_t fixed_state_mask =
                           LOOM_AMDGPU_CALL_FIXED_STATE_ALL) {
    call_state_.fixed_state_mask = fixed_state_mask;
    const loom_amdgpu_wait_plan_options_t wait_options = {
        .call_states =
            {
                .query = QueryCallState,
                .user_data = this,
            },
        .callable = callable,
    };
    Build(&wait_options);
    loom_amdgpu_wait_packet_plan_t wait_packets = {};
    IREE_ASSERT_OK(
        loom_amdgpu_wait_packet_plan_build(&plan_, &arena_, &wait_packets));
    const loom_amdgpu_address_state_plan_t address_state = {
        .schedule = &schedule_,
        .allocation = &allocation_,
    };
    const loom_amdgpu_wait_state_plan_options_t wait_state_options = {
        .call_states =
            {
                .query = QueryCallState,
                .user_data = this,
            },
        .callable = callable,
    };
    IREE_ASSERT_OK(loom_amdgpu_wait_state_plan_build(
        &schedule_, &allocation_, &wait_state_options,
        facts_.properties.processor, &address_state, &plan_, &wait_packets,
        /*vopd_plan=*/nullptr,
        /*matrix_coexecution=*/nullptr, /*matrix_results=*/nullptr, &arena_,
        &transient_arena_, &wait_state_plan_));
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
              loom_named_attr_slice_t attrs,
              loom_memory_access_flags_t access_flags = 0) {
    const uint32_t node_index = (uint32_t)nodes_.size();
    const auto* descriptor =
        loom_amdgpu_descriptor_ref_descriptor(descriptors_, descriptor_ref);
    const auto* layout = loom_low_descriptor_layout(descriptors_, descriptor);
    const auto* view =
        loom_low_descriptor_set_descriptor_view(descriptors_, descriptor);
    loom_low_schedule_node_t node = {};
    node.scheduled_ordinal = nodes_.size();
    node.kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    node.descriptor = descriptor;
    node.source_descriptor_ordinal =
        loom_low_descriptor_set_descriptor_ordinal(descriptors_, descriptor);
    node.schedule_class_id = view->schedule_class_id;
    node.schedule_class =
        loom_low_schedule_model_schedule_class(schedule_model_, view);
    node.operand_count = layout->minimum_packet_operand_count;
    for (uint16_t i = 0; i < node.operand_count; ++i) {
      loom_low_schedule_node_operand_ordinals(&node)[i] = i;
    }
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_low_build_resolved_descriptor_op(
        &builder_, descriptors_, descriptor, access_flags,
        block_.block->arg_ids, node.operand_count, attrs, nullptr, 0, nullptr,
        0, LOOM_LOCATION_UNKNOWN, &op));
    node.op = op;
    node.traits = op->traits;
    node.immediate_presence = loom_low_bind_immediate_presence(
        module_, descriptors_, descriptor, loom_low_op_attrs(op));
    nodes_.push_back(node);
    for (uint16_t i = 0; i < layout->effect_count; ++i) {
      const auto& effect = descriptors_->effects[layout->effect_start + i];
      loom_low_schedule_effect_use_t use = {
          .node_index = node_index,
          .scheduled_ordinal = node.scheduled_ordinal,
          .effect_ordinal = i,
          .kind = effect.kind,
          .memory_space = effect.memory_space,
          .scope_id = effect.scope_id,
          .effect_flags = effect.flags,
          .counter_id = effect.counter_id,
          .width_bits = effect.width_bits,
      };
      effects_.push_back(use);
    }
    for (uint16_t i = 0; i < node.schedule_class->hazard_count; ++i) {
      loom_low_schedule_hazard_use_t use = {
          .node_index = node_index,
          .hazard_index =
              static_cast<uint32_t>(node.schedule_class->hazard_start + i),
      };
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
  // Explicit baseline schedule model selected for this target fixture.
  const loom_low_schedule_model_t* schedule_model_ = nullptr;
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
  // Memory ordering edges retained independently of the scheduler graph.
  loom_low_schedule_dependency_graph_t effect_dependencies_;
  // Exact cross-block memory completions retained by the generic scheduler.
  std::vector<loom_low_schedule_memory_completion_edge_t> memory_completions_;
  // Descriptor counter hazards.
  std::vector<loom_low_schedule_hazard_use_t> hazards_;
  // ABI transports for retained direct-call boundary fixtures.
  std::vector<loom_low_allocation_call_moves_t> call_moves_;
  // Sequential physical transport rows owned by retained calls.
  std::vector<loom_low_move_t> moves_;
  // Maximum assigned or moved location by descriptor register class.
  std::vector<uint32_t> physical_extents_;
  // Maximum used ABI-fixed location by descriptor register class.
  std::vector<uint32_t> fixed_physical_extents_;
  // Distinct zero-argument leaf helper reached by direct-call fixtures.
  loom_symbol_ref_t call_target_ = loom_symbol_ref_null();
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
  loom_amdgpu_call_state_t call_state_ = {};
  // Fixed-delay plan produced after counter waits are materialized.
  loom_amdgpu_wait_state_plan_t wait_state_plan_ = {};
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

TEST_F(AmdgpuWaitPlanTest, DirectCallDrainsOutstandingCounterWork) {
  Tensor();
  Call();
  Build();
  const loom_amdgpu_structural_packet_info_t structural =
      loom_amdgpu_structural_packet_analyze(
          &schedule_, &allocation_, /*node_index=*/1,
          LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION);
  EXPECT_TRUE(iree_any_bit_set(structural.flags,
                               LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_CALL));
  EXPECT_EQ(structural.instruction_count, 4u);
  EXPECT_EQ(structural.scalar_alu_instruction_count, 4u);
  EXPECT_EQ(loom_amdgpu_structural_packet_call_boundary_offset(&structural),
            0u);
  EXPECT_EQ(loom_amdgpu_structural_packet_call_arguments(&structural), nullptr);
  EXPECT_EQ(loom_amdgpu_structural_packet_call_results(&structural), nullptr);
  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_CALLABLE_BOUNDARY);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallPreservesCountersUnusedByCallee) {
  Tensor();
  Call();
  const loom_amdgpu_wait_plan_options_t options =
      CallOptions(LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  Build(&options);

  EXPECT_EQ(plan_.action_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallDrainsCountersUsedByCallee) {
  Tensor();
  Call();
  const loom_amdgpu_wait_plan_options_t options =
      CallOptions(LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
  Build(&options);

  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_CALLABLE_BOUNDARY);
}

TEST_F(AmdgpuWaitPlanTest, CallableReturnsWithProducedCountersEmpty) {
  Tensor();
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  Build(&options);

  EXPECT_EQ(plan_.produced_counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
  ASSERT_EQ(plan_.action_count, 1u);
  ExpectAction(0, 1, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_CALLABLE_BOUNDARY);
}

TEST_F(AmdgpuWaitPlanTest, OrdinaryPlanSkipsFixedStateSummary) {
  Build();

  EXPECT_EQ(plan_.fixed_state_mask, 0u);
}

TEST_F(AmdgpuWaitPlanTest, PublishesAllocatedFixedStateFamilies) {
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  Build(&options);

  EXPECT_EQ(plan_.fixed_state_mask,
            LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
                LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU |
                LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE |
                LOOM_AMDGPU_CALL_FIXED_STATE_SGPR_READ);
}

TEST_F(AmdgpuWaitPlanTest, PublishesTransportFixedStateFamilies) {
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0)});
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  Build(&options);

  EXPECT_EQ(plan_.fixed_state_mask, LOOM_AMDGPU_CALL_FIXED_STATE_ALL);
}

TEST_F(AmdgpuWaitPlanTest, PublishesImplicitFixedStateFamilies) {
  Append(LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, loom_named_attr_slice_empty());
  std::fill(physical_extents_.begin(), physical_extents_.end(), 0);
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  Build(&options);

  EXPECT_EQ(plan_.fixed_state_mask, LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU);
}

TEST_F(AmdgpuWaitPlanTest, CallableReusesAuthoredCounterDrains) {
  Tensor();
  Wait(0);
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  Build(&options);

  ASSERT_EQ(plan_.action_count, 2u);
  ExpectAction(0, 1, 0, LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
  EXPECT_EQ(plan_.actions[1].node_index, 1u);
  EXPECT_EQ(plan_.actions[1].counter_id, LOOM_AMDGPU_WAIT_COUNTER_X);
  EXPECT_EQ(plan_.actions[1].reason,
            LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallInvalidatesArgumentDelayHistory) {
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0)},
       {Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 3, 2)});
  BuildWaitStates(/*callable=*/false, LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU);

  const loom_amdgpu_structural_packet_info_t structural =
      loom_amdgpu_structural_packet_analyze(
          &schedule_, &allocation_, /*node_index=*/0,
          LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION);
  EXPECT_EQ(loom_amdgpu_structural_packet_call_boundary_offset(&structural),
            1u);
  ASSERT_NE(loom_amdgpu_structural_packet_call_arguments(&structural), nullptr);
  EXPECT_EQ(
      loom_amdgpu_structural_packet_call_arguments(&structural)->moves.count,
      1u);
  ASSERT_NE(loom_amdgpu_structural_packet_call_results(&structural), nullptr);
  EXPECT_EQ(
      loom_amdgpu_structural_packet_call_results(&structural)->moves.count, 1u);

  EXPECT_EQ(wait_state_plan_.state_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallTracksResultMovesInFreshHintEpoch) {
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0)},
       {Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 3, 2)});
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4, 3)});
  BuildWaitStates(/*callable=*/false, LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU);

  ASSERT_EQ(wait_state_plan_.state_count, 1u);
  const loom_amdgpu_wait_state_t& state = wait_state_plan_.states[0];
  EXPECT_EQ(state.reason, LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY);
  EXPECT_EQ(state.action, LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU);
  EXPECT_EQ(state.node_index, 1u);
  EXPECT_EQ(state.instruction_offset, 0u);
  EXPECT_EQ(state.required_cycle_count, 4u);
  EXPECT_EQ(state.observed_cycle_count, 1u);
  EXPECT_EQ(state.cycle_count, 3u);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallScalarResultMoveSatisfiesItsHintWindow) {
  Call({}, {Move(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 3, 2)});
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4, 3)});
  BuildWaitStates(/*callable=*/false, LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU);

  EXPECT_EQ(wait_state_plan_.state_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallCreditsControlToInstructionWindows) {
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0)});
  BuildWaitStates(/*callable=*/false, LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT);

  EXPECT_EQ(wait_state_plan_.state_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, DirectCallUsesPublishedFixedStateMask) {
  Call({Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0)});
  BuildWaitStates(/*callable=*/false, /*fixed_state_mask=*/0);

  EXPECT_EQ(wait_state_plan_.state_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, CallableReturnDoesNotSerializeOptionalResultHints) {
  Call({}, {Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 0, 1)});
  BuildWaitStates(/*callable=*/true);

  EXPECT_EQ(wait_state_plan_.state_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, ReturnMovesPrecedeControlTransfer) {
  FinalizeSchedule();
  moves_.push_back(Move(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, 0));
  const loom_low_allocation_exit_moves_t exit_moves = {
      .source_ordinal = static_cast<uint32_t>(nodes_.size() - 1),
      .result_count = 1,
      .results = {.moves = {.start = 0, .count = 1}},
  };
  allocation_.exit_moves = &exit_moves;
  allocation_.exit_move_count = 1;
  allocation_.moves = moves_.data();
  allocation_.move_count = moves_.size();

  const loom_amdgpu_structural_packet_info_t structural =
      loom_amdgpu_structural_packet_analyze(
          &schedule_, &allocation_, exit_moves.source_ordinal,
          LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION);
  EXPECT_EQ(structural.instruction_count, 2u);
  EXPECT_EQ(structural.scalar_alu_instruction_count, 1u);
  ASSERT_EQ(loom_amdgpu_structural_packet_move_segment_count(&structural), 1u);
  const loom_amdgpu_structural_move_segment_t segment =
      loom_amdgpu_structural_packet_move_segment_at(&structural, 0);
  EXPECT_EQ(segment.group, &exit_moves.results);
  EXPECT_EQ(segment.instruction_offset, 0u);
}

TEST_F(AmdgpuWaitPlanTest, RefinedCrossBlockMemoryOmitsDisjointWait) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  IREE_ASSERT_OK(BuildPlan());
  EXPECT_EQ(plan_.action_count, 0u);
}

TEST_F(AmdgpuWaitPlanTest, CallableDrainsProducedCountersAcrossBlocks) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  ConfigureRefinedAcyclicMemory(graph, &fixture);
  const loom_amdgpu_wait_plan_options_t options = {.callable = true};
  IREE_ASSERT_OK(BuildPlan(&options));

  EXPECT_EQ(plan_.produced_counter_mask,
            LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE |
                LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR |
                LOOM_AMDGPU_WAIT_COUNTER_MASK_X);
  ASSERT_EQ(plan_.action_count, 2u);
  const uint32_t return_node = static_cast<uint32_t>(nodes_.size() - 1);
  EXPECT_EQ(plan_.actions[0].node_index, return_node);
  EXPECT_EQ(plan_.actions[0].counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE);
  EXPECT_EQ(plan_.actions[0].reason,
            LOOM_AMDGPU_WAIT_PLAN_REASON_CALLABLE_BOUNDARY);
  EXPECT_EQ(plan_.actions[1].node_index, return_node);
  EXPECT_EQ(plan_.actions[1].counter_id, LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
  EXPECT_EQ(plan_.actions[1].reason,
            LOOM_AMDGPU_WAIT_PLAN_REASON_CALLABLE_BOUNDARY);
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

TEST_F(AmdgpuWaitPlanTest, VolatileConsumerOrdersRefinedIncomingMemory) {
  testing::CfgGraph graph({{1}, {}});
  CrossBlockMemoryFixture fixture = {};
  // Scheduling excludes volatile accesses from source alias refinement. The
  // ordinary producer still occupies the separate refined incoming frontier.
  ConfigureRefinedAcyclicMemory(graph, LOOM_MEMORY_ACCESS_FLAG_VOLATILE,
                                &fixture);

  IREE_ASSERT_OK(BuildPlan());
  ASSERT_EQ(plan_.action_count, 1u);
  const loom_amdgpu_wait_plan_action_t& action = plan_.actions[0];
  EXPECT_EQ(action.node_index, fixture.consumer_node);
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
