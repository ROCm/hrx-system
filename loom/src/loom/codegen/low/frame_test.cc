// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/guarded_motion.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/facts_builder.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LowEmissionFrameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    loom_target_facts_builder_initialize(
        &loom_test_target_fact_type,
        loom_test_target_bundles.values[LOOM_TEST_TARGET_KIND_LOW_CORE],
        &target_facts_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("frame_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr ParseModule() {
    return ParseModule(R"(
low.func.def target<test.low.core> @structural_model() -> (reg<test.i32 x4>) asm {
  %storage = storage {byte_alignment = 16, byte_length = 64} : low.storage<workgroup>
  %address = storage_address %storage : low.storage<workgroup> -> reg<test.ptr>
  %value = test.load.v4i32 %address
  return %value
}
)");
  }

  iree_status_t BuildFrame(
      loom_module_t* module,
      loom_low_schedule_structural_model_list_t structural_models,
      loom_low_emission_frame_t* out_frame,
      loom_low_schedule_strategy_t schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      loom_low_schedule_flags_t schedule_flags = 0) {
    loom_block_t* module_block = loom_module_block(module);
    IREE_ASSERT_EQ(module_block->op_count, 1);
    loom_low_emission_frame_options_t options = {
        .descriptor_registry = &registry_.registry,
        .function_target_facts = &target_facts_,
        .schedule_structural_models = structural_models,
        .schedule_strategy = schedule_strategy,
        .schedule_flags = schedule_flags,
    };
    bool accepted = false;
    iree_status_t status =
        loom_low_emission_frame_build(module, loom_block_op(module_block, 0),
                                      &options, &arena_, out_frame, &accepted);
    if (iree_status_is_ok(status)) {
      EXPECT_TRUE(accepted);
      // Returned frames own their rows after scheduler and trial scratch dies.
      iree_arena_block_pool_trim(&block_pool_);
    }
    return status;
  }

  const loom_low_schedule_node_t* FindNode(
      const loom_low_emission_frame_t& frame, loom_op_kind_t op_kind) {
    for (iree_host_size_t i = 0; i < frame.schedule.node_count; ++i) {
      if (frame.schedule.nodes[i].op->kind == op_kind) {
        return &frame.schedule.nodes[i];
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_target_facts_t target_facts_ = {};
  iree_arena_allocator_t arena_ = {};
};

TEST_F(LowEmissionFrameTest, RejectsTargetlessFunctionBeforeScheduling) {
  ModulePtr module = ParseModule();
  uint32_t diagnostic_count = 0;
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    EXPECT_EQ(emission->error, LOOM_ERR_TARGET_026);
    ++*static_cast<uint32_t*>(user_data);
    return iree_ok_status();
  };
  options.emitter.user_data = &diagnostic_count;

  loom_low_emission_frame_t frame = {};
  bool accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));

  EXPECT_FALSE(accepted);
  EXPECT_NE(frame.target.descriptor_set, nullptr);
  EXPECT_EQ(frame.target.schedule_model, nullptr);
  EXPECT_EQ(frame.schedule.node_count, 0u);
  EXPECT_EQ(diagnostic_count, 1u);
}

TEST_F(LowEmissionFrameTest, ResolvesSparseEntryValuesIntoLocalDomain) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @entry_values(%argument: reg<test.phys>) -> (reg<test.phys>, reg<test.phys>) asm {
  %live = live_in<test.arg0> : reg<test.phys>
  %copied = copy %live : reg<test.phys> -> reg<test.phys>
  return %argument, %copied
}
)");
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  loom_block_t* entry =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t argument = loom_block_arg_id(entry, 0);
  const loom_value_id_t live = loom_low_live_in_result(loom_block_op(entry, 0));
  const loom_value_id_t copied = loom_low_copy_result(loom_block_op(entry, 1));
  const loom_low_allocation_abi_location_t argument_location = {
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_PHYS,
      .location_base = 6,
  };
  const loom_low_allocation_entry_value_t entry_value = {
      .value_id = live,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 5,
  };
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_entry_locations = &argument_location,
      .allocation_entry_location_count = 1,
      .allocation_entry_values = &entry_value,
      .allocation_entry_value_count = 1,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(module.get(), function, &options,
                                               &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  const auto find_assignment = [&](loom_value_id_t value_id) {
    for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
      if (frame.allocation.assignments[i].value_id == value_id) {
        return &frame.allocation.assignments[i];
      }
    }
    return static_cast<const loom_low_allocation_assignment_t*>(nullptr);
  };
  const loom_low_allocation_assignment_t* argument_assignment =
      find_assignment(argument);
  const loom_low_allocation_assignment_t* live_assignment =
      find_assignment(live);
  const loom_low_allocation_assignment_t* copied_assignment =
      find_assignment(copied);
  ASSERT_NE(argument_assignment, nullptr);
  ASSERT_NE(live_assignment, nullptr);
  ASSERT_NE(copied_assignment, nullptr);
  EXPECT_EQ(argument_assignment->descriptor_reg_class_id,
            TEST_LOW_CORE_REG_CLASS_ID_TEST_PHYS);
  EXPECT_EQ(argument_assignment->location_base, 6u);
  EXPECT_EQ(live_assignment->descriptor_reg_class_id,
            TEST_LOW_CORE_REG_CLASS_ID_TEST_PHYS);
  EXPECT_EQ(live_assignment->location_base, 5u);
  EXPECT_EQ(copied_assignment->descriptor_reg_class_id,
            TEST_LOW_CORE_REG_CLASS_ID_TEST_PHYS);
  EXPECT_EQ(copied_assignment->location_base, 5u);
  EXPECT_EQ(frame.allocation.entry_moves.moves.count, 0u);
}

constexpr const char kLoopPlacementSource[] = R"(
low.func.def target<test.low.core> @loop_placement(%source: reg<test.i32>) asm {
  %temporary = test.total.add.i32 %source, %source
  %private = copy %source : reg<test.i32> -> reg<test.i32>
  %candidate = test.total.tied.i32 %private
  %dead = test.total.tied.i32 %temporary
  %keep_source_live = test.total.add.i32 %source, %source
  low.br ^loop
^loop:
  low.cond_br %candidate, ^loop, ^exit : reg<test.i32>
^exit:
  return
}
)";

struct LoopPlacementCost {
  uint64_t maximum_execution_multiplier;
  uint32_t query_count;
};

bool QueryLoopPlacementCost(void* user_data,
                            const loom_low_descriptor_t* descriptor,
                            loom_low_loop_placement_cost_t* out_cost) {
  auto* cost = static_cast<LoopPlacementCost*>(user_data);
  EXPECT_EQ(descriptor, loom_low_descriptor_set_descriptor_at(
                            loom_test_low_core_descriptor_set(),
                            TEST_LOW_CORE_DESCRIPTOR_REF_TEST_TOTAL_TIED_I32));
  ++cost->query_count;
  *out_cost = {
      .maximum_execution_multiplier = cost->maximum_execution_multiplier,
      .maximum_schedule_cycle_count = 2,
  };
  return true;
}

TEST_F(LowEmissionFrameTest, LoopPlacementConsumesRetainedHeaderCount) {
  ModulePtr module = ParseModule(kLoopPlacementSource);
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t* entry = loom_region_block(body, 0);
  loom_block_t* loop = loom_region_block(body, 1);
  loom_op_t* copy = loom_block_op(entry, 1);
  loom_op_t* candidate = loom_block_op(entry, 2);
  loom_op_t* original_anchor = loom_block_op(entry, 3);

  loom_low_loop_execution_map_t* loop_execution = nullptr;
  IREE_ASSERT_OK(
      loom_low_loop_execution_map_allocate(&arena_, &loop_execution));
  IREE_ASSERT_OK(loom_low_loop_execution_map_insert(
      loop_execution, loop->last_op,
      {.trip_count = 7, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED}));
  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.i32"),
      .max_units = 2,
  };
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .loop_execution = loop_execution,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_budgets = &budget,
      .allocation_budget_count = 1,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(module.get(), function, &options,
                                               &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);
  ASSERT_EQ(frame.allocation.spill_count, 2u);
  const loom_low_loop_execution_facts_t* retained_facts =
      loom_low_loop_execution_index_lookup(&frame.schedule.loop_execution,
                                           loop);
  ASSERT_NE(retained_facts, nullptr);
  EXPECT_EQ(retained_facts->trip_count, 7u);
  EXPECT_EQ(retained_facts->count_kind, LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED);

  LoopPlacementCost cost = {.maximum_execution_multiplier = 7};
  loom_low_loop_placement_plan_t rejected_plan = {};
  IREE_ASSERT_OK(loom_low_loop_placement_plan(
      &frame.schedule, &frame.allocation, QueryLoopPlacementCost, &cost,
      &arena_, &rejected_plan));
  EXPECT_EQ(rejected_plan.candidate_count, 0u);

  cost.maximum_execution_multiplier = 8;
  loom_low_loop_placement_plan_t accepted_plan = {};
  IREE_ASSERT_OK(loom_low_loop_placement_plan(
      &frame.schedule, &frame.allocation, QueryLoopPlacementCost, &cost,
      &arena_, &accepted_plan));
  ASSERT_EQ(accepted_plan.candidate_count, 1u);
  EXPECT_EQ(cost.query_count, 2u);

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, module.get(), &arena_);
  IREE_ASSERT_OK(loom_low_loop_placement_apply(&rewriter, &accepted_plan));
  EXPECT_EQ(copy->parent_block, loop);
  EXPECT_EQ(candidate->parent_block, loop);
  EXPECT_EQ(copy->next_op, candidate);
  EXPECT_EQ(candidate->next_op, loop->last_op);
  IREE_ASSERT_OK(loom_low_loop_placement_rollback(&rewriter, &accepted_plan));
  loom_rewriter_deinitialize(&rewriter);
  EXPECT_EQ(copy->parent_block, entry);
  EXPECT_EQ(candidate->parent_block, entry);
  EXPECT_EQ(copy->next_op, candidate);
  EXPECT_EQ(candidate->next_op, original_anchor);
  EXPECT_EQ(original_anchor->parent_block, entry);
}

TEST_F(LowEmissionFrameTest, ScheduleRetainsCrossingNaturalLoops) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @crossing_loops(%condition: reg<test.i32>) asm {
  low.br ^first
^first:
  low.cond_br %condition, ^first_latch, ^second : reg<test.i32>
^second:
  low.cond_br %condition, ^second_latch, ^exit : reg<test.i32>
^first_latch:
  low.br ^first
^second_latch:
  low.br ^second
^exit:
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  // The source-order ranges [first, first_latch] and
  // [second, second_latch] cross, so they cannot form layout intervals.
  EXPECT_EQ(frame.schedule.loop_forest.interval_count, 0u);
  EXPECT_EQ(frame.schedule.loop_forest.reachable_backward_edge_count, 2u);

  const loom_cfg_loop_nest_t& loops = frame.schedule.loop_nest;
  EXPECT_EQ(loops.graph, &frame.schedule.cfg_graph);
  EXPECT_TRUE(loops.reducible);
  ASSERT_EQ(loops.loop_count, 2u);
  const uint16_t first = loom_cfg_loop_nest_innermost(&loops, 1);
  const uint16_t second = loom_cfg_loop_nest_innermost(&loops, 2);
  ASSERT_LT(first, loops.loop_count);
  ASSERT_LT(second, loops.loop_count);
  EXPECT_NE(first, second);
  EXPECT_TRUE(loom_cfg_loop_nest_contains(&loops, first, 3));
  EXPECT_FALSE(loom_cfg_loop_nest_contains(&loops, first, 2));
  EXPECT_TRUE(loom_cfg_loop_nest_contains(&loops, second, 4));
  EXPECT_FALSE(loom_cfg_loop_nest_contains(&loops, second, 3));
  EXPECT_EQ(loops.loops[first].backedges.count, 1u);
  EXPECT_EQ(loops.loops[second].backedges.count, 1u);
}

TEST_F(LowEmissionFrameTest, LoopPlacementRestoresRejectedTrial) {
  ModulePtr module = ParseModule(kLoopPlacementSource);
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t* entry = loom_region_block(body, 0);
  loom_op_t* copy = loom_block_op(entry, 1);
  loom_op_t* candidate = loom_block_op(entry, 2);
  loom_op_t* original_anchor = loom_block_op(entry, 3);

  loom_low_loop_execution_map_t* loop_execution = nullptr;
  IREE_ASSERT_OK(
      loom_low_loop_execution_map_allocate(&arena_, &loop_execution));
  IREE_ASSERT_OK(loom_low_loop_execution_map_insert(
      loop_execution, loom_region_block(body, 1)->last_op,
      {.trip_count = 7, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED}));
  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.i32"),
      .max_units = 2,
  };
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .loop_execution = loop_execution,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_budgets = &budget,
      .allocation_budget_count = 1,
  };
  loom_low_planning_statistics_t statistics = {};
  options.statistics = &statistics;
  LoopPlacementCost cost = {.maximum_execution_multiplier = 8};
  loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_NONE,
          },
      .loop_placement_cost_query = QueryLoopPlacementCost,
      .loop_placement_cost_query_user_data = &cost,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), function, &options, &spill_free_options, &arena_, &frame,
      &accepted));

  EXPECT_FALSE(accepted);
  EXPECT_EQ(cost.query_count, 1u);
  EXPECT_EQ(statistics.frame_build_count, 2u);
  EXPECT_EQ(statistics.allocation_run_count, 2u);
  EXPECT_EQ(copy->parent_block, entry);
  EXPECT_EQ(candidate->parent_block, entry);
  EXPECT_EQ(copy->next_op, candidate);
  EXPECT_EQ(candidate->next_op, original_anchor);
}

TEST_F(LowEmissionFrameTest, ResidencyQueryConsumesRetainedFunctionFacts) {
  ModulePtr module = ParseModule();
  static const loom_target_residency_model_t model = {.best_tier = 4};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .residency_query =
          [](const loom_low_resolved_target_t* target,
             const loom_low_storage_layout_space_sizes_t* storage_sizes) {
            EXPECT_NE(target->target_facts, nullptr);
            EXPECT_EQ(target->descriptor_set,
                      loom_test_low_core_descriptor_set());
            EXPECT_EQ(storage_sizes->workgroup_bytes, 64u);
            return loom_target_residency_view(&model, 2);
          },
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);
  // The function model has been released; the frame retains only the borrowed
  // immutable policy and its value ceiling, not analysis-owned storage.
  iree_arena_block_pool_trim(&block_pool_);
  EXPECT_EQ(frame.residency.model, &model);
  EXPECT_EQ(frame.residency.tier_limit, 2u);
}

TEST_F(LowEmissionFrameTest,
       RepacksAggregateTopologyBlockingInstructionPreferences) {
  // First-fit places the odd-width connector across the boundary between the
  // later equal-width operands. Physical numbering must translate that entire
  // overlap-connected range rigidly and cannot satisfy the low-bit preference.
  // Packing the scalar lifetimes above dense aggregates aligns the connector
  // with one operand, leaving the other independently numberable.
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fragmented_preferences(%source: reg<test.phys>, %seed: reg<test.phys x8>, %persistent0: reg<test.phys>, %persistent1: reg<test.phys>, %persistent2: reg<test.phys>, %persistent3: reg<test.phys>, %persistent4: reg<test.phys>, %persistent5: reg<test.phys>, %persistent6: reg<test.phys>) -> (reg<test.phys>, reg<test.phys x8>) asm {
  %temporary0 = test.event.write.fast.phys %source
  %temporary1 = test.event.write.fast.phys %source
  %connector0 = test.event.write.fast.phys %source
  %connector1 = test.event.write.fast.phys %source
  %connector2 = test.event.write.fast.phys %source
  %connector = concat(%connector0, %connector1, %connector2) : (reg<test.phys>, reg<test.phys>, reg<test.phys>) -> reg<test.phys x3>
  %connector_first = slice %connector[0] : reg<test.phys x3> -> reg<test.phys>
  test.event.read.late.phys %connector_first
  test.event.read.late.phys %temporary0
  test.event.read.late.phys %temporary1
  %lhs0 = test.event.write.fast.phys %source
  %lhs1 = test.event.write.fast.phys %source
  %lhs2 = test.event.write.fast.phys %source
  %lhs3 = test.event.write.fast.phys %source
  %lhs = concat(%lhs0, %lhs1, %lhs2, %lhs3) : (reg<test.phys>, reg<test.phys>, reg<test.phys>, reg<test.phys>) -> reg<test.phys x4>
  %rhs0 = test.event.write.fast.phys %source
  %rhs1 = test.event.write.fast.phys %source
  %rhs2 = test.event.write.fast.phys %source
  %rhs3 = test.event.write.fast.phys %source
  %rhs = concat(%rhs0, %rhs1, %rhs2, %rhs3) : (reg<test.phys>, reg<test.phys>, reg<test.phys>, reg<test.phys>) -> reg<test.phys x4>
  %lhs_first = slice %lhs[0] : reg<test.phys x4> -> reg<test.phys>
  %rhs_first = slice %rhs[0] : reg<test.phys x4> -> reg<test.phys>
  %result = test.add.phys %lhs_first, %rhs_first
  test.event.read.late.phys %persistent0
  test.event.read.late.phys %persistent1
  test.event.read.late.phys %persistent2
  test.event.read.late.phys %persistent3
  test.event.read.late.phys %persistent4
  test.event.read.late.phys %persistent5
  test.event.read.late.phys %persistent6
  return %result, %seed
}
)");
  static const loom_low_placement_value_ref_t values[] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 1},
  };
  static const loom_low_placement_predicate_t predicates[] = {
      {0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 3},
  };
  static const loom_low_placement_clause_t clauses[] = {
      {0, 1, 1, LOOM_LOW_PLACEMENT_CLAUSE_ANY},
  };
  static const loom_low_placement_preference_t preferences[] = {{
      values,
      predicates,
      clauses,
      IREE_ARRAYSIZE(values),
      IREE_ARRAYSIZE(clauses),
  }};
  uint16_t indices[TEST_LOW_CORE_DESCRIPTOR_REF_TEST_PACKING_EARLY_EXPAND + 1] =
      {};
  indices[TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_PHYS] = 1;
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .instruction_preferences = {indices, preferences},
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
  EXPECT_EQ(frame.allocation.physical_extents
                .ends_by_reg_class[TEST_LOW_CORE_REG_CLASS_ID_TEST_PHYS],
            23u);

  uint32_t preferred_instruction_count = 0;
  for (iree_host_size_t i = 0; i < loom_low_packet_count(&frame.schedule);
       ++i) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&frame.schedule, i);
    if (packet.descriptor_ordinal !=
        TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_PHYS) {
      continue;
    }
    const loom_low_allocation_assignment_t* lhs =
        loom_low_packet_operand_assignment(&frame.allocation, &packet, 0);
    const loom_low_allocation_assignment_t* rhs =
        loom_low_packet_operand_assignment(&frame.allocation, &packet, 1);
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);
    EXPECT_NE(lhs->location_base & 3, rhs->location_base & 3);
    ++preferred_instruction_count;
  }
  EXPECT_EQ(preferred_instruction_count, 1u);
}

TEST_F(LowEmissionFrameTest, StorageLeaseFrameRetainsValueProducers) {
  ModulePtr module = ParseModule();
  const loom_low_storage_lease_provider_t storage_lease_provider = {
      .user_data = nullptr,
      .query =
          [](void* user_data, const loom_low_schedule_table_t* schedule,
             const loom_low_schedule_node_t* node,
             const loom_low_storage_lease_query_sink_t* sink) {
            (void)user_data;
            (void)schedule;
            (void)node;
            (void)sink;
            return iree_ok_status();
          },
  };
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .storage_lease_provider = &storage_lease_provider,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);

  // Allocation consumes this relation after scheduler scratch is released.
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_NE(frame.schedule.value_producer_nodes, nullptr);
  const loom_low_schedule_node_t* load_node = FindNode(frame, LOOM_OP_LOW_OP);
  ASSERT_NE(load_node, nullptr);
  ASSERT_EQ(load_node->result_count, 1u);
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(load_node)[0];
  EXPECT_EQ(
      loom_low_schedule_value_producer_node(&frame.schedule, result_ordinal),
      static_cast<uint32_t>(load_node - frame.schedule.nodes));
}

TEST_F(LowEmissionFrameTest, MaterializationPreservesReadyDescriptorPair) {
  constexpr loom_low_schedule_strategy_t kScheduleStrategies[] = {
      LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
      LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
      LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  for (const loom_low_schedule_strategy_t strategy : kScheduleStrategies) {
    SCOPED_TRACE(static_cast<int>(strategy));
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @constant_pair(%input: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.const.issued.i32 1
  %second = test.const.issued.i32 2
  %first_result = test.add.i32 %input, %first
  %second_result = test.add.i32 %input, %second
  return %first_result, %second_result
}
)");
    const loom_low_descriptor_set_t* descriptor_set =
        loom_test_low_core_descriptor_set();
    const loom_low_descriptor_t* constant =
        loom_low_descriptor_set_descriptor_at(
            descriptor_set, TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_ISSUED_I32);
    loom_low_schedule_pair_affinity_t affinity = {
        .first_descriptor = constant,
        .second_descriptor = constant,
        .priority = 1,
    };
    loom_low_emission_frame_options_t options = {
        .descriptor_registry = &registry_.registry,
        .function_target_facts = &target_facts_,
        .schedule_pair_affinities = {&affinity, 1},
        .schedule_strategy = strategy,
    };
    loom_low_emission_frame_t frame = {};
    bool frame_accepted = false;
    IREE_ASSERT_OK(loom_low_emission_frame_build(
        module.get(), loom_block_op(loom_module_block(module.get()), 0),
        &options, &arena_, &frame, &frame_accepted));
    ASSERT_TRUE(frame_accepted);
    ASSERT_EQ(frame.schedule.error_count, 0u);
    EXPECT_EQ(frame.schedule.nodes[1].scheduled_ordinal,
              frame.schedule.nodes[0].scheduled_ordinal + 1);
    EXPECT_LT(frame.schedule.nodes[1].scheduled_ordinal,
              frame.schedule.nodes[2].scheduled_ordinal);
  }
}

TEST_F(LowEmissionFrameTest, MaterializationPairCannotExceedRegisterBudget) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @bounded_pair(%address: reg<test.ptr>, %value: reg<test.i32 x4>) asm {
  %first = test.const.issued.i32 0
  %second = test.const.issued.i32 4
  test.store.index.v4i32 %address, %first, %value
  test.store.index.v4i32 %address, %second, %value
  return
}
)");
  const loom_low_descriptor_t* constant = loom_low_descriptor_set_descriptor_at(
      loom_test_low_core_descriptor_set(),
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_ISSUED_I32);
  loom_low_schedule_pair_affinity_t affinity = {
      .first_descriptor = constant,
      .second_descriptor = constant,
      .priority = 1,
  };
  loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.i32"),
      .max_units = 5,
  };
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_pair_affinities = {&affinity, 1},
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .allocation_budgets = &budget,
      .allocation_budget_count = 1,
  };
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.schedule.error_count, 0u);
  EXPECT_LT(frame.schedule.nodes[2].scheduled_ordinal,
            frame.schedule.nodes[1].scheduled_ordinal);
  EXPECT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
}

TEST_F(LowEmissionFrameTest, FeedbackConsumesTheAcceptedFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @feedback(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %first = test.resource.serial.i32 %lhs, %rhs
  %second = test.resource.serial.i32 %rhs, %lhs
  %copy = copy %first : reg<test.i32> -> reg<test.i32>
  %sum = test.add.i32 %copy, %second
  return %sum
}
)");
  struct CapturedDiagnostics {
    // Total feedback records delivered by the frame.
    uint32_t count = 0;
    // Pressure budget retained independently of the caller's option storage.
    uint32_t pressure_budget = 0;
    // Number of allocation copy decisions observed.
    uint32_t copy_count = 0;
  } captured;
  iree_diagnostic_emitter_t emitter = {
      .fn =
          [](void* user_data, const loom_diagnostic_emission_t* emission) {
            auto* captured = static_cast<CapturedDiagnostics*>(user_data);
            ++captured->count;
            if (emission->error == LOOM_ERR_BACKEND_003) {
              captured->pressure_budget = emission->params[5].u32;
            } else if (emission->error == LOOM_ERR_BACKEND_006) {
              ++captured->copy_count;
            }
            return iree_ok_status();
          },
      .user_data = &captured,
  };
  loom_low_allocation_budget_t budget = {IREE_SV("test.i32"), 8};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .schedule_diagnostic_flags =
          LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS |
          LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS |
          LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS |
          LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS |
          LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY,
      .allocation_budgets = &budget,
      .allocation_budget_count = 1,
      .allocation_diagnostic_flags =
          LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS,
      .emitter = emitter,
      .statistics = &statistics,
  };
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.schedule.loop_nest.graph, &frame.schedule.cfg_graph);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);
  EXPECT_GT(captured.count, 1u);
  EXPECT_EQ(captured.pressure_budget, 8u);
  EXPECT_EQ(captured.copy_count, 1u);

  loom_low_emission_frame_options_t quiet_options = options;
  quiet_options.schedule_diagnostic_flags = 0;
  quiet_options.allocation_diagnostic_flags = 0;
  quiet_options.statistics = nullptr;
  loom_low_emission_frame_t quiet_frame = {};
  bool quiet_frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0),
      &quiet_options, &spill_free_options, &arena_, &quiet_frame,
      &quiet_frame_accepted));
  ASSERT_TRUE(quiet_frame_accepted);
  // Release pooled scratch before comparing schedules and formatting retained
  // diagnostics. ASAN catches any accidental result borrowing from that state.
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_EQ(quiet_frame.schedule.node_count, frame.schedule.node_count);
  for (iree_host_size_t i = 0; i < frame.schedule.node_count; ++i) {
    EXPECT_EQ(quiet_frame.schedule.scheduled_node_indices[i],
              frame.schedule.scheduled_node_indices[i]);
    EXPECT_EQ(quiet_frame.schedule.nodes[i].issue_cycle,
              frame.schedule.nodes[i].issue_cycle);
  }
  ASSERT_EQ(quiet_frame.allocation.assignment_count,
            frame.allocation.assignment_count);
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& quiet = quiet_frame.allocation.assignments[i];
    const auto& verbose = frame.allocation.assignments[i];
    EXPECT_EQ(quiet.value_id, verbose.value_id);
    EXPECT_EQ(quiet.location_kind, verbose.location_kind);
    EXPECT_EQ(quiet.location_base, verbose.location_base);
    EXPECT_EQ(quiet.location_count, verbose.location_count);
  }

  // Formatting is a read-only consumer after the model and option scratch
  // have gone away. Neither another plan nor another arena allocation occurs.
  budget.max_units = 1;
  captured = {};
  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_schedule_diagnostics_emit(
      &frame.schedule, options.schedule_diagnostic_flags, emitter));
  EXPECT_GT(captured.count, 0u);
  EXPECT_EQ(captured.pressure_budget, 8u);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(statistics.frame_build_count, 1u);
}

TEST_F(LowEmissionFrameTest, FeedbackConsumesTheRejectedFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @too_wide(%wide: reg<test.special x2>) -> (reg<test.special x2>) asm {
  return %wide
}
)");
  struct CapturedFailure {
    // Number of terminal allocation diagnostics.
    uint32_t count = 0;
    // Original operation anchoring the diagnostic.
    const loom_op_t* op = nullptr;
  } captured;
  loom_low_allocation_budget_t budget = {IREE_SV("test.special"), 1};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_budgets = &budget,
      .allocation_budget_count = 1,
  };
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    auto* captured = static_cast<CapturedFailure*>(user_data);
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_005);
    EXPECT_EQ(emission->params[5].u32, 1u);
    EXPECT_EQ(emission->params[6].u32, 2u);
    ++captured->count;
    captured->op = emission->op;
    return iree_ok_status();
  };
  options.emitter.user_data = &captured;
  options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_FALSE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 1u);
  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.op, frame.allocation.failure.op);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);

  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_allocation_diagnostics_emit(
      &frame.allocation, /*flags=*/0, options.emitter));
  EXPECT_EQ(captured.count, 2u);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
}

TEST_F(LowEmissionFrameTest, FinalValidatorRejectsWithoutDiagnosticSink) {
  ModulePtr module = ParseModule();
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  bool validator_invoked = false;
  spill_free_options.validate_frame =
      [](void* user_data, const loom_low_emission_frame_t* frame,
         iree_arena_allocator_t* arena, bool* out_accepted) {
        bool* invoked = static_cast<bool*>(user_data);
        EXPECT_FALSE(*invoked);
        *invoked = true;
        EXPECT_EQ(frame->schedule.error_count, 0u);
        EXPECT_EQ(frame->allocation.error_count, 0u);
        (void)arena;
        *out_accepted = false;
        return iree_ok_status();
      };
  spill_free_options.validate_frame_user_data = &validator_invoked;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_TRUE(validator_invoked);
  EXPECT_FALSE(frame_accepted);
}

TEST_F(LowEmissionFrameTest, InputErrorsAreNotDeferredOrReplayed) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @invalid_budget(%value: reg<test.i32>) -> (reg<test.i32>) asm {
  return %value
}
)");
  uint32_t diagnostic_count = 0;
  const loom_low_allocation_budget_t budgets[] = {
      {IREE_SV("test.i32"), 1},
      {IREE_SV("test.i32"), 2},
  };
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_budgets = budgets,
      .allocation_budget_count = IREE_ARRAYSIZE(budgets),
  };
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_024);
    ++*static_cast<uint32_t*>(user_data);
    return iree_ok_status();
  };
  options.emitter.user_data = &diagnostic_count;
  options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_FALSE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 1u);
  EXPECT_FALSE(
      loom_low_allocation_failure_is_present(&frame.allocation.failure));
  EXPECT_EQ(diagnostic_count, 1u);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);
}

TEST_F(LowEmissionFrameTest, SingletonReservationsPreserveTiedStateUpdates) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fixed_updates(%state: reg<test.fixed.r0>, %delta: reg<test.i32>, %independent: reg<test.explicit32>) -> (reg<test.fixed.r0>, reg<test.explicit32>) asm {
  %first = test.fixed.update.i32 %state, %delta
  %second = test.fixed.update.i32 %first, %delta
  return %second, %independent
}
)");
  ASSERT_NE(module, nullptr);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
  EXPECT_EQ(frame.allocation.materialized_copy_count, 0u);
  uint32_t fixed_count = 0;
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& assignment = frame.allocation.assignments[i];
    if (assignment.descriptor_reg_class_id ==
        TEST_LOW_CORE_REG_CLASS_ID_TEST_EXPLICIT32) {
      EXPECT_NE(assignment.location_base, 0u);
    }
    if (assignment.descriptor_reg_class_id !=
        TEST_LOW_CORE_REG_CLASS_ID_TEST_FIXED_R0) {
      continue;
    }
    EXPECT_EQ(assignment.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(assignment.location_base, 0u);
    ++fixed_count;
  }
  EXPECT_EQ(fixed_count, 3u);
}

TEST_F(LowEmissionFrameTest, SingletonTiedStateCrossesDynamicLoopBackedge) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fixed_loop(%initial: reg<test.fixed.r0>, %count: reg<test.i32>) -> (reg<test.fixed.r0>) asm {
  %decrement = test.const.i32 -1
  low.br ^loop(%initial: reg<test.fixed.r0>, %count: reg<test.i32>)
^loop(%state: reg<test.fixed.r0>, %remaining: reg<test.i32>):
  low.cond_br %remaining, ^body, ^exit : reg<test.i32>
^body:
  %next_state = test.fixed.update.i32 %state, %decrement
  %next_remaining = test.add.i32 %remaining, %decrement
  low.br ^loop(%next_state: reg<test.fixed.r0>, %next_remaining: reg<test.i32>)
^exit:
  return %state
}
)");
  ASSERT_NE(module, nullptr);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
  EXPECT_EQ(frame.allocation.materialized_copy_count, 0u);
  EXPECT_EQ(frame.allocation.first_coalesced_incoming_copy_by_value_ordinal,
            nullptr);
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& assignment = frame.allocation.assignments[i];
    if (assignment.descriptor_reg_class_id ==
        TEST_LOW_CORE_REG_CLASS_ID_TEST_FIXED_R0) {
      EXPECT_EQ(assignment.location_base, 0u);
    }
  }

  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .allocation_flags =
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_COALESCED_INCOMING_INDEX,
  };
  loom_low_emission_frame_t indexed_frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &indexed_frame, &accepted));
  ASSERT_TRUE(accepted);
  ASSERT_EQ(indexed_frame.allocation.error_count, 0u);
  ASSERT_NE(
      indexed_frame.allocation.first_coalesced_incoming_copy_by_value_ordinal,
      nullptr);

  const loom_region_t* body =
      loom_low_func_def_body(loom_block_op(loom_module_block(module.get()), 0));
  const loom_block_t* entry = loom_region_const_block(body, 0);
  const loom_block_t* loop = loom_region_const_block(body, 1);
  const loom_block_t* loop_body = loom_region_const_block(body, 2);
  const loom_value_id_t destination_value_id = loom_block_arg_id(loop, 0);
  const loom_value_id_t expected_source_value_ids[] = {
      loom_low_br_args(loop_body->last_op).values[0],
      loom_low_br_args(entry->last_op).values[0],
  };
  loom_value_ordinal_t destination_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  for (loom_value_ordinal_t i = 0;
       i < indexed_frame.allocation.liveness.value_count; ++i) {
    if (indexed_frame.allocation.liveness.value_ids[i] ==
        destination_value_id) {
      destination_ordinal = i;
      break;
    }
  }
  ASSERT_NE(destination_ordinal, LOOM_VALUE_ORDINAL_INVALID);

  uint32_t copy_index =
      indexed_frame.allocation
          .first_coalesced_incoming_copy_by_value_ordinal[destination_ordinal];
  for (const loom_value_id_t expected_source_value_id :
       expected_source_value_ids) {
    ASSERT_NE(copy_index, LOOM_LOW_ALLOCATION_EDGE_COPY_INDEX_NONE);
    ASSERT_LT(copy_index, indexed_frame.allocation.edge_copy_count);
    const auto& copy = indexed_frame.allocation.edge_copies[copy_index];
    EXPECT_EQ(copy.kind, LOOM_LOW_ALLOCATION_COPY_COALESCED);
    EXPECT_EQ(copy.destination_ordinal, destination_ordinal);
    EXPECT_EQ(indexed_frame.allocation.liveness.value_ids[copy.source_ordinal],
              expected_source_value_id);
    copy_index = copy.next_coalesced_incoming_copy_index;
  }
  EXPECT_EQ(copy_index, LOOM_LOW_ALLOCATION_EDGE_COPY_INDEX_NONE);
}

TEST_F(LowEmissionFrameTest, StateOnlyScheduleLivenessIsTransient) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @state_live_out(%state: reg<test.schedule_state>, %rhs: reg<test.schedule_state>) asm {
  low.br ^use
^use:
  %read = test.explicit.state.add.schedule_state %rhs, %state
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  iree_arena_block_pool_trim(&block_pool_);

  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.schedule.liveness.region, nullptr);
}

TEST_F(LowEmissionFrameTest, FailedScheduleDoesNotPublishSourceSuffixBounds) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @dependency_cycle(%lhs: reg<test.schedule_state>, %rhs: reg<test.schedule_state>) asm {
  low.br ^cycle
^cycle:
  %old = test.add.schedule_state %lhs, %rhs
  %writer = test.state.add.schedule_state %lhs, %rhs
  %cycle = test.explicit.state.add.schedule_state %writer, %old
  return
}
)");
  const loom_region_t* body =
      loom_low_func_def_body(loom_block_op(loom_module_block(module.get()), 0));
  const loom_value_id_t expected_state_value_id = loom_op_const_results(
      loom_block_const_op(loom_region_const_block(body, 1), 0))[0];
  uint32_t diagnostic_count = 0;
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_044);
    ++*static_cast<uint32_t*>(user_data);
    return iree_ok_status();
  };
  options.emitter.user_data = &diagnostic_count;
  loom_low_emission_frame_t frame = {};
  bool accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));

  EXPECT_FALSE(accepted);
  EXPECT_EQ(frame.schedule.error_count, 1u);
  EXPECT_EQ(frame.schedule.failure.state_value_id, expected_state_value_id);
  EXPECT_EQ(frame.schedule.source_suffix_issue_cycle_lower_bounds, nullptr);
  EXPECT_EQ(diagnostic_count, 1u);
}

TEST_F(LowEmissionFrameTest, GuardedMotionReplaysRetainedLocations) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @guarded_store(%base: reg<test.ptr>, %origin: reg<test.i32>, %pixel: reg<test.i32>, %color: reg<test.i32 x4>) asm {
  test.store.index.v4i32 %base, %origin, %color
  %loaded = test.counter_load.v4i32 %base
  %condition = test.event.fast.i32 %origin, %pixel
  low.cond_br %condition, ^store, ^done : reg<test.i32>
^store:
  %index = test.total.add.i32 %origin, %pixel
  test.store.index.v4i32 %base, %index, %color
  low.br ^done
^done:
  return
}
)");
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t* entry_block = loom_region_block(body, 0);
  loom_block_t* guarded_block = loom_region_block(body, 1);
  loom_op_t* initial_store = entry_block->first_op;
  loom_op_t* load = loom_block_op(entry_block, 1);
  loom_op_t* branch = entry_block->last_op;
  loom_op_t* index = loom_block_op(guarded_block, 0);

  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .statistics = &statistics,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(module.get(), function, &options,
                                               &arena_, &frame, &accepted));

  EXPECT_TRUE(accepted);
  EXPECT_EQ(index->parent_block, entry_block);
  EXPECT_EQ(index->next_op, branch);
  EXPECT_EQ(statistics.frame_build_count, 2u);
  EXPECT_EQ(statistics.allocation_run_count, 2u);
  EXPECT_EQ(statistics.retained_location_replay_count, 1u);

  // Accepted motion retires the original frame. Its replacement owns the
  // effect and hazard rows consumed after trial and scheduling scratch expire.
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_EQ(frame.schedule.effect_dependencies.count, 2u);
  const auto* dependency = loom_low_schedule_dependency_slice_at(
      &frame.schedule.effect_dependencies, 0);
  EXPECT_EQ(frame.schedule.nodes[dependency->producer_node].op, initial_store);
  EXPECT_EQ(frame.schedule.nodes[dependency->consumer_node].op, load);
  dependency = loom_low_schedule_dependency_slice_at(
      &frame.schedule.effect_dependencies, 1);
  EXPECT_EQ(frame.schedule.nodes[dependency->producer_node].op, initial_store);
  // Cross-block effect timing is enforced at the source terminator.
  EXPECT_EQ(frame.schedule.nodes[dependency->consumer_node].op, branch);
  ASSERT_EQ(frame.schedule.hazard_use_count, 1u);
  const auto& hazard_use = frame.schedule.hazard_uses[0];
  EXPECT_EQ(frame.schedule.nodes[hazard_use.node_index].op, load);
  const auto& hazard =
      frame.schedule.target.schedule_model->hazards[hazard_use.hazard_index];
  EXPECT_EQ(hazard.kind, LOOM_LOW_HAZARD_KIND_WAIT_COUNTER);
  EXPECT_EQ(hazard.reference_id, 0u);
}

TEST_F(LowEmissionFrameTest, ProvenNoOpGuardedMotionSkipsTrialFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @guarded_tail(%base: reg<test.ptr>, %origin: reg<test.i32>, %pixel: reg<test.i32>, %color: reg<test.i32 x4>) -> (reg<test.i32>) asm {
  %condition = test.event.fast.i32 %origin, %pixel
  low.cond_br %condition, ^store, ^done : reg<test.i32>
^store:
  %index = test.total.add.i32 %origin, %pixel
  test.store.index.v4i32 %base, %index, %color
  %tail = test.event.fast.i32 %origin, %pixel
  %result = test.add.i32 %tail, %pixel
  return %result
^done:
  return %condition
}
)");
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &registry_.registry,
      .function_target_facts = &target_facts_,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .statistics = &statistics,
  };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));

  EXPECT_TRUE(accepted);
  EXPECT_NE(frame.schedule.source_suffix_issue_cycle_lower_bounds, nullptr);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);

  loom_low_schedule_table_t unproven_schedule = frame.schedule;
  unproven_schedule.source_suffix_issue_cycle_lower_bounds = nullptr;
  loom_low_guarded_motion_plan_t unproven_plan = {};
  IREE_ASSERT_OK(loom_low_guarded_motion_plan(&unproven_schedule, &arena_,
                                              &unproven_plan));
  ASSERT_EQ(unproven_plan.region_count, 1u);
  const loom_low_guarded_motion_region_t& region = unproven_plan.regions[0];
  const uint32_t source_suffix = region.node_start + region.node_count;
  const uint32_t source_block =
      frame.schedule.nodes[region.node_start].block_index;
  const loom_low_schedule_block_t& block = frame.schedule.blocks[source_block];
  ASSERT_NE(block.issue_group_count, 0u);
  const uint32_t source_extent =
      frame.schedule
          .issue_groups[block.issue_group_start + block.issue_group_count - 1]
          .issue_cycle;
  EXPECT_EQ(
      frame.schedule.source_suffix_issue_cycle_lower_bounds[source_suffix],
      source_extent);
}

TEST_F(LowEmissionFrameTest, OrderedEffectUsesDirectionalTimingEndpoints) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @directional_effect(%address: reg<test.ptr>, %payload: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %payload
  test.barrier
  %loaded = test.load.v4i32 %address
  return %loaded
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));

  ASSERT_GE(frame.schedule.node_count, 3u);
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 2u);
  EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 4u);

  const loom_low_schedule_dependency_t* store_to_barrier = nullptr;
  const loom_low_schedule_dependency_t* barrier_to_load = nullptr;
  for (uint32_t i = 0; i < frame.schedule.effect_dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_slice_at(
            &frame.schedule.effect_dependencies, i);
    EXPECT_EQ(dependency->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
    if (dependency->separation_source !=
        LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR) {
      continue;
    }
    if (dependency->producer_node == 0 && dependency->consumer_node == 1) {
      store_to_barrier = dependency;
    } else if (dependency->producer_node == 1 &&
               dependency->consumer_node == 2) {
      barrier_to_load = dependency;
    }
  }
  ASSERT_NE(store_to_barrier, nullptr);
  ASSERT_NE(barrier_to_load, nullptr);
  EXPECT_EQ(store_to_barrier->minimum_issue_separation_cycles, 2);
  EXPECT_EQ(barrier_to_load->minimum_issue_separation_cycles, 2);
  EXPECT_NE(store_to_barrier->consumer_event_id,
            barrier_to_load->producer_event_id);
  EXPECT_EQ(frame.schedule.dependency_graph, nullptr);
  EXPECT_GT(frame.schedule.dependency_count,
            frame.schedule.effect_dependencies.count);

  const auto dependency_count = frame.schedule.dependency_count;
  const auto effect_dependency_count = frame.schedule.effect_dependencies.count;
  loom_low_emission_frame_t inspected_frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &inspected_frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
                            LOOM_LOW_SCHEDULE_FLAG_RETAIN_DEPENDENCIES));
  ASSERT_NE(inspected_frame.schedule.dependency_graph, nullptr);
  EXPECT_EQ(inspected_frame.schedule.dependency_count, dependency_count);
  EXPECT_EQ(inspected_frame.schedule.dependency_graph->count, dependency_count);
  EXPECT_EQ(inspected_frame.schedule.effect_dependencies.count,
            effect_dependency_count);
}

TEST_F(LowEmissionFrameTest, StructuralModelCarriesNativePacketTiming) {
  ModulePtr module = ParseModule();
  const loom_low_schedule_structural_model_t models[] = {
      {
          .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
          .result_reg_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          .schedule_descriptor_ordinal =
              TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));

  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_low_descriptor_view_t* schedule_descriptor_view =
      loom_low_descriptor_set_descriptor_view_at(
          descriptor_set, TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32);
  const uint16_t expected_schedule_class_id =
      schedule_descriptor_view->schedule_class_id;
  const loom_low_schedule_model_t* schedule_model =
      loom_low_descriptor_set_lookup_schedule_model_by_id(
          descriptor_set, UINT64_C(0x235c515ddcc8fd92));
  ASSERT_NE(schedule_model, nullptr);
  const loom_low_schedule_class_t* expected_schedule_class =
      loom_low_schedule_model_schedule_class(schedule_model,
                                             schedule_descriptor_view);
  ASSERT_EQ(expected_schedule_class->latency_cycles, 1u);
  ASSERT_EQ(expected_schedule_class->minimum_issue_separation_cycles, 1);
  ASSERT_EQ(expected_schedule_class->issue_use_count, 1u);

  const loom_low_schedule_node_t* address_node =
      FindNode(frame, LOOM_OP_LOW_STORAGE_ADDRESS);
  const loom_low_schedule_node_t* load_node = FindNode(frame, LOOM_OP_LOW_OP);
  ASSERT_NE(address_node, nullptr);
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(address_node->kind, LOOM_LOW_SCHEDULE_NODE_STRUCTURAL);
  EXPECT_EQ(address_node->descriptor, nullptr);
  EXPECT_EQ(address_node->source_descriptor_ordinal,
            LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  EXPECT_EQ(address_node->schedule_class_id, expected_schedule_class_id);
  EXPECT_EQ(address_node->schedule_class, expected_schedule_class);
  EXPECT_GE(load_node->issue_cycle,
            address_node->issue_cycle +
                expected_schedule_class->minimum_issue_separation_cycles);

  const loom_low_schedule_model_summary_t* model_summary = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.model_summary_count; ++i) {
    if (frame.schedule.model_summaries[i].schedule_class_id ==
        expected_schedule_class_id) {
      model_summary = &frame.schedule.model_summaries[i];
      break;
    }
  }
  ASSERT_NE(model_summary, nullptr);
  EXPECT_EQ(model_summary->use_count, 1u);

  const loom_low_issue_use_t* issue_use =
      &frame.schedule.target.schedule_model
           ->issue_uses[expected_schedule_class->issue_use_start];
  const loom_low_schedule_resource_summary_t* resource_summary = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.resource_summary_count; ++i) {
    if (frame.schedule.resource_summaries[i].resource_id ==
        issue_use->resource_id) {
      resource_summary = &frame.schedule.resource_summaries[i];
      break;
    }
  }
  ASSERT_NE(resource_summary, nullptr);
  EXPECT_EQ(resource_summary->use_count, 1u);
}

TEST_F(LowEmissionFrameTest, RejectsInvalidStructuralModelDescriptor) {
  ModulePtr module = ParseModule();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_low_schedule_structural_model_t models[] = {
      {
          .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
          .result_reg_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          .schedule_descriptor_ordinal =
              descriptor_set->descriptor_ordinal_count,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));
}

TEST_F(LowEmissionFrameTest, RejectsOverlappingStructuralModels) {
  ModulePtr module = ParseModule();
  const loom_low_schedule_structural_model_t models[] = {
      {
          .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
          .result_reg_class_id = LOOM_LOW_REG_CLASS_NONE,
          .schedule_descriptor_ordinal =
              TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32,
      },
      {
          .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
          .result_reg_class_id = TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          .schedule_descriptor_ordinal =
              TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_I32,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));
}

}  // namespace
}  // namespace loom
