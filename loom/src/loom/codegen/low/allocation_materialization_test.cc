// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/cfg_graph.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

static iree_status_t CountDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  (void)emission;
  ++*static_cast<iree_host_size_t*>(user_data);
  return iree_ok_status();
}

class LowAllocationMaterializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    descriptor_registry_.descriptor_set_providers = kDescriptorSetProviders;
    descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kDescriptorSetProviders);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("allocation_materialization_test.loom"),
                        &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindLowFunction(loom_module_t* module,
                             iree_string_view_t function_name) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_region_entry_block(module->body), op) {
      if (!loom_low_func_def_isa(op)) {
        continue;
      }
      if (iree_string_view_equal(loom_low_diagnostic_function_name(module, op),
                                 function_name)) {
        return op;
      }
    }
    IREE_ASSERT(false, "low function not found");
    return nullptr;
  }

  loom_value_id_t FindValueByName(loom_module_t* module,
                                  iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < module->values.count; ++i) {
      if (iree_string_view_equal(
              loom_low_diagnostic_value_name(module, (loom_value_id_t)i),
              name)) {
        return (loom_value_id_t)i;
      }
    }
    IREE_ASSERT(false, "value name not found");
    return LOOM_VALUE_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_low_descriptor_registry_t descriptor_registry_ = {};
};

TEST_F(LowAllocationMaterializationTest,
       ExcludesStalePlansFromTrafficAndStorageColoring) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.func.def target<test.low.core>(@test_target) @stale_slice_plan(%wide: reg<test.i32 x4>) -> (reg<test.i32>) asm {
  %early = test.const.i32 1
  %early_rhs = test.const.i32 2
  %early_use = test.add.i32 %early, %early_rhs
  low.br ^middle
^middle:
  %lane = slice %wide[1] : reg<test.i32 x4> -> reg<test.i32>
  low.br ^late
^late:
  %late = test.const.i32 3
  %late_rhs = test.const.i32 4
  %late_use = test.add.i32 %late, %late_rhs
  %result = test.add.i32 %lane, %late_use
  return %result
}
)");
  ASSERT_NE(module.get(), nullptr);
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("stale_slice_plan"));
  const loom_value_id_t wide = FindValueByName(module.get(), IREE_SV("wide"));
  const loom_value_id_t early = FindValueByName(module.get(), IREE_SV("early"));
  const loom_value_id_t lane = FindValueByName(module.get(), IREE_SV("lane"));
  const loom_value_id_t late = FindValueByName(module.get(), IREE_SV("late"));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_allocation_table_t table = {
      .module = module.get(),
      .function_op = function_op,
      .target = {.descriptor_set = loom_test_low_core_descriptor_set()},
  };
  const loom_region_t* body = loom_low_func_def_body(function_op);
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module.get(), body, &arena, &table.cfg_graph));
  IREE_ASSERT_OK(loom_liveness_analyze_region(module.get(), body, &arena,
                                              &table.liveness));
  IREE_ASSERT_OK(loom_liveness_retain_operation_index(&table.liveness, &arena));
  const loom_liveness_interval_t* wide_interval =
      loom_liveness_interval_for_value(&table.liveness, wide);
  const loom_liveness_interval_t* early_interval =
      loom_liveness_interval_for_value(&table.liveness, early);
  const loom_liveness_interval_t* lane_interval =
      loom_liveness_interval_for_value(&table.liveness, lane);
  const loom_liveness_interval_t* late_interval =
      loom_liveness_interval_for_value(&table.liveness, late);
  ASSERT_NE(wide_interval, nullptr);
  ASSERT_NE(early_interval, nullptr);
  ASSERT_NE(lane_interval, nullptr);
  ASSERT_NE(late_interval, nullptr);
  std::vector<loom_value_ordinal_t> storage_value_order;
  storage_value_order.reserve(table.liveness.value_count);
  loom_value_ordinal_t wide_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  for (loom_value_ordinal_t i = 0; i < table.liveness.value_count; ++i) {
    if (table.liveness.value_ids[i] == wide) {
      wide_ordinal = i;
    } else {
      storage_value_order.push_back(i);
    }
  }
  ASSERT_NE(wide_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  storage_value_order.push_back(wide_ordinal);
  table.placement.value_count = table.liveness.value_count;
  table.placement.storage_value_order = storage_value_order.data();
  table.placement.storage_value_order_count = storage_value_order.size();

  const auto make_assignment = [](loom_value_id_t value_id,
                                  const loom_liveness_interval_t* interval,
                                  uint32_t unit_count, uint32_t location_base) {
    return loom_low_allocation_assignment_t{
        .value_id = value_id,
        .descriptor_reg_class_id = 0,
        .flags = {},
        .start_point = interval->start_point,
        .end_point = interval->end_point,
        .unit_count = unit_count,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .location_base = location_base,
        .location_count = unit_count,
        .unit_point_start = 0,
    };
  };
  loom_low_allocation_assignment_t assignments[] = {
      make_assignment(wide, wide_interval, 4, 0),
      make_assignment(early, early_interval, 1, 1),
      make_assignment(lane, lane_interval, 1, 2),
      make_assignment(late, late_interval, 1, 3),
  };
  const auto make_spill_plan = [](loom_value_id_t value_id,
                                  uint32_t assignment_index, uint32_t byte_size,
                                  uint32_t byte_alignment) {
    return loom_low_allocation_spill_plan_t{
        .value_id = value_id,
        .assignment_index = assignment_index,
        .slot_index = assignment_index,
        .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
        .byte_size = byte_size,
        .byte_alignment = byte_alignment,
        .store_count = 1,
        .reload_count = 1,
    };
  };
  loom_low_allocation_spill_plan_t spill_plans[] = {
      make_spill_plan(lane, 2, 4, 4),
      make_spill_plan(early, 1, 4, 4),
      make_spill_plan(wide, 0, 16, 16),
      make_spill_plan(late, 3, 4, 4),
  };
  // The retained source-before-users traversal makes the lane plan stale even
  // though spill-decision order presents it first. Its unsupported slot space
  // must not diagnose or reserve storage once exact traffic is known.
  spill_plans[0].slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK;
  table.assignments = assignments;
  table.assignment_count = IREE_ARRAYSIZE(assignments);
  table.spill_plans = spill_plans;
  table.spill_plan_count = IREE_ARRAYSIZE(spill_plans);
  loom_low_allocation_materialization_result_t result = {};
  loom_low_allocation_materialization_options_t options = {};
  iree_host_size_t diagnostic_count = 0;
  options.has_supported_storage_spaces = true;
  options.supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_PRIVATE;
  options.record_materialized_spills = true;
  options.emitter = {
      .fn = CountDiagnostic,
      .user_data = &diagnostic_count,
  };
  IREE_ASSERT_OK(loom_low_allocation_materialize_spills(
      &table, table.spill_plans, table.spill_plan_count, &options, &arena,
      &result));

  EXPECT_EQ(diagnostic_count, 0u);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.storage_count, 2u);
  EXPECT_EQ(result.storage_bytes, 20u);
  EXPECT_EQ(result.storage_minimum_alignment, 16u);
  EXPECT_EQ(result.spill_count, 3u);
  EXPECT_EQ(result.spill_bytes, 24u);
  EXPECT_EQ(result.reload_count, 3u);
  EXPECT_EQ(result.reload_bytes, 12u);
  ASSERT_EQ(result.materialized_spill_count, 4u);
  ASSERT_NE(result.materialized_spills, nullptr);
  EXPECT_EQ(result.materialized_spills[2].value_id, wide);
  EXPECT_TRUE(iree_all_bits_set(
      result.materialized_spills[2].flags,
      LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT));
  EXPECT_EQ(result.materialized_spills[0].value_id, lane);
  EXPECT_EQ(result.materialized_spills[0].store_count, 0u);
  EXPECT_EQ(result.materialized_spills[0].reload_count, 0u);

  ASSERT_EQ(result.required_register_values.count, 6u);
  iree_host_size_t spill_source_count = 0;
  iree_host_size_t reload_result_count = 0;
  for (iree_host_size_t i = 0; i < result.required_register_values.count; ++i) {
    const loom_value_id_t value_id = result.required_register_values.values[i];
    if (value_id == wide || value_id == early || value_id == late) {
      ++spill_source_count;
      continue;
    }
    const loom_value_t* value = loom_module_value(module.get(), value_id);
    ASSERT_NE(value, nullptr);
    EXPECT_TRUE(loom_low_reload_isa(loom_value_def_op(value)));
    ++reload_result_count;
  }
  EXPECT_EQ(spill_source_count, 3u);
  EXPECT_EQ(reload_result_count, 3u);
  ASSERT_EQ(result.per_user_placement_values.count, 3u);
  for (iree_host_size_t i = 0; i < result.per_user_placement_values.count;
       ++i) {
    const loom_value_t* value = loom_module_value(
        module.get(), result.per_user_placement_values.values[i]);
    ASSERT_NE(value, nullptr);
    EXPECT_TRUE(loom_low_reload_isa(loom_value_def_op(value)));
  }

  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
