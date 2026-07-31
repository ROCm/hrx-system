// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness_json.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LivenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    const loom_low_descriptor_set_provider_t descriptor_set_providers[] = {
        loom_test_low_core_descriptor_set,
    };
    loom_low_descriptor_registry_t descriptor_registry = {
        /*.descriptor_sets=*/{},
        /*.descriptor_set_count=*/{},
        /*.descriptor_set_providers=*/descriptor_set_providers,
        /*.descriptor_set_provider_count=*/
        IREE_ARRAYSIZE(descriptor_set_providers),
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("liveness_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
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

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    loom_func_like_t func = loom_func_like_cast(module, op);
    IREE_ASSERT(func.op != NULL);
    return func;
  }

  loom_liveness_analysis_t AnalyzeBody(loom_module_t* module,
                                       loom_func_like_t func) {
    loom_liveness_analysis_t analysis;
    IREE_CHECK_OK(loom_liveness_analyze_region(
        module, loom_func_like_body(func), &analysis_arena_, &analysis));
    return analysis;
  }

  loom_liveness_analysis_t AnalyzeBodyRegionTree(loom_module_t* module,
                                                 loom_func_like_t func) {
    loom_local_value_domain_t value_domain = {};
    IREE_CHECK_OK(loom_local_value_domain_acquire_for_region_tree(
        module, loom_func_like_body(func), &analysis_arena_, &value_domain));
    loom_liveness_analysis_t analysis;
    IREE_CHECK_OK(loom_liveness_analyze_local_value_domain(
        &value_domain, loom_liveness_order_empty(), &analysis_arena_,
        &analysis));
    loom_local_value_domain_release(&value_domain);
    return analysis;
  }

  static bool ContainsValue(const loom_value_id_t* values,
                            iree_host_size_t count, loom_value_id_t value) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (values[i] == value) return true;
    }
    return false;
  }

  static loom_value_ordinal_t FindValueOrdinal(
      const loom_liveness_analysis_t& analysis, loom_value_id_t value_id) {
    for (iree_host_size_t i = 0; i < analysis.value_count; ++i) {
      if (analysis.value_ids[i] == value_id) {
        return static_cast<loom_value_ordinal_t>(i);
      }
    }
    return LOOM_VALUE_ORDINAL_INVALID;
  }

  static const loom_liveness_pressure_summary_t* FindRegisterPressure(
      const loom_liveness_analysis_t& analysis,
      uint64_t descriptor_set_stable_id, uint16_t class_id) {
    for (iree_host_size_t i = 0; i < analysis.pressure_summary_count; ++i) {
      const auto* summary = &analysis.pressure_summaries[i];
      if (summary->value_class.type_kind == LOOM_TYPE_REGISTER &&
          summary->value_class.register_descriptor_set_stable_id ==
              descriptor_set_stable_id &&
          summary->value_class.register_class_id == class_id) {
        return summary;
      }
    }
    return nullptr;
  }

  static const loom_liveness_pressure_summary_t* FindScalarPressure(
      const loom_liveness_analysis_t& analysis,
      loom_scalar_type_t scalar_type) {
    for (iree_host_size_t i = 0; i < analysis.pressure_summary_count; ++i) {
      const auto* summary = &analysis.pressure_summaries[i];
      if (summary->value_class.type_kind == LOOM_TYPE_SCALAR &&
          summary->value_class.element_type == scalar_type) {
        return summary;
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(LivenessTest, SingleBlockIntervalsAndDeadDefs) {
  ModulePtr module = ParseModule(R"(
func.def @linear(%a: i32, %b: i32) -> (i32) {
  %sum = scalar.addi %a, %b : i32
  %dead = scalar.addi %a, %b : i32
  func.return %sum : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("linear"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_EQ(analysis.block_count, 1u);
  EXPECT_EQ(analysis.blocks[0].live_in_count, 0u);
  EXPECT_EQ(analysis.blocks[0].live_out_count, 0u);

  const loom_block_t* entry =
      loom_region_const_entry_block(loom_func_like_body(func));
  const loom_op_t* add = loom_block_const_op(entry, 0);
  const loom_op_t* dead_add = loom_block_const_op(entry, 1);
  loom_value_id_t sum = loom_op_const_results(add)[0];
  loom_value_id_t dead = loom_op_const_results(dead_add)[0];

  const loom_liveness_interval_t* a_interval =
      loom_liveness_interval_for_value(&analysis, args[0]);
  const loom_liveness_interval_t* sum_interval =
      loom_liveness_interval_for_value(&analysis, sum);
  const loom_liveness_interval_t* dead_interval =
      loom_liveness_interval_for_value(&analysis, dead);
  ASSERT_NE(a_interval, nullptr);
  ASSERT_NE(sum_interval, nullptr);
  ASSERT_NE(dead_interval, nullptr);
  EXPECT_EQ(a_interval->start_point, 0u);
  EXPECT_EQ(a_interval->end_point, 2u);
  EXPECT_EQ(sum_interval->start_point, 1u);
  EXPECT_EQ(sum_interval->end_point, 3u);
  EXPECT_EQ(dead_interval->start_point, 2u);
  EXPECT_EQ(dead_interval->end_point, 2u);

  ASSERT_EQ(analysis.operation_count, 3u);
  EXPECT_EQ(analysis.operation_points[0].op, add);
  EXPECT_EQ(analysis.operation_points[0].parent_operation_index, UINT32_MAX);
  EXPECT_EQ(analysis.operation_points[0].start_point, 0u);
  EXPECT_EQ(analysis.operation_points[0].end_point, 1u);
  EXPECT_EQ(analysis.operation_points[0].direct_use_count, 2u);
  EXPECT_EQ(analysis.operation_points[0].use_count, 2u);
  EXPECT_EQ(analysis.operation_points[1].op, dead_add);
  EXPECT_EQ(analysis.operation_points[2].op, loom_block_const_op(entry, 2));
  EXPECT_EQ(analysis.operation_use_count, 5u);
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis, 0),
            FindValueOrdinal(analysis, args[0]));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis, 1),
            FindValueOrdinal(analysis, args[1]));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis, 2),
            FindValueOrdinal(analysis, args[0]));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis, 3),
            FindValueOrdinal(analysis, args[1]));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis, 4),
            FindValueOrdinal(analysis, sum));
}

TEST_F(LivenessTest, OperationRowsFollowAcceptedOrder) {
  ModulePtr module = ParseModule(R"(
func.def @ordered(%a: i32, %b: i32) -> (i32) {
  %first = scalar.addi %a, %b : i32
  %second = scalar.addi %a, %b : i32
  func.return %a : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("ordered"));
  const loom_region_t* body = loom_func_like_body(func);
  const loom_block_t* entry = loom_region_const_entry_block(body);
  const loom_op_t* const ordered_ops[] = {
      loom_block_const_op(entry, 1),
      loom_block_const_op(entry, 0),
      loom_block_const_op(entry, 2),
  };
  const loom_liveness_block_order_t block_order = {
      /*.block=*/entry,
      /*.ops=*/ordered_ops,
      /*.op_count=*/IREE_ARRAYSIZE(ordered_ops),
  };
  const loom_liveness_order_t order = {
      /*.blocks=*/&block_order,
      /*.block_count=*/1,
  };
  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module.get(), body, &analysis_arena_, &value_domain));
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
      &value_domain, order, &analysis_arena_, &analysis));
  loom_local_value_domain_release(&value_domain);

  ASSERT_EQ(analysis.operation_count, IREE_ARRAYSIZE(ordered_ops));
  ASSERT_EQ(analysis.blocks[0].operation_start, 0u);
  ASSERT_EQ(analysis.blocks[0].operation_count, IREE_ARRAYSIZE(ordered_ops));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ordered_ops); ++i) {
    EXPECT_EQ(analysis.operation_points[i].op, ordered_ops[i]);
    EXPECT_EQ(analysis.operation_points[i].start_point, i);
    EXPECT_EQ(analysis.operation_points[i].end_point, i + 1u);
  }
}

TEST_F(LivenessTest, OperationUseRangesContainDistinctValues) {
  ModulePtr module = ParseModule(R"(
func.def @distinct_uses(%a: i32) -> (i32) {
  %sum = scalar.addi %a, %a : i32
  func.return %sum : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("distinct_uses"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  ASSERT_EQ(analysis.operation_count, 2u);
  EXPECT_EQ(analysis.operation_points[0].direct_use_count, 1u);
  EXPECT_EQ(analysis.operation_points[0].use_count, 1u);
  EXPECT_EQ(analysis.operation_use_count, 2u);
}

TEST_F(LivenessTest, RegionLocalOrdinalsHandleSparseModuleValueIds) {
  ModulePtr module = ParseModule(R"(
func.def @padding(%p0: i32, %p1: i32, %p2: i32, %p3: i32) -> (i32) {
  %r0 = scalar.addi %p0, %p1 : i32
  %r1 = scalar.addi %p2, %p3 : i32
  %r2 = scalar.addi %r0, %r1 : i32
  func.return %r2 : i32
}
func.def @sparse(%a: i32, %b: i32) -> (i32) {
  %sum = scalar.addi %a, %b : i32
  func.return %sum : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("sparse"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  EXPECT_LT(analysis.value_count, module.get()->values.count);
  EXPECT_GE(args[0], analysis.value_count);
  ASSERT_NE(loom_liveness_interval_for_value(&analysis, args[0]), nullptr);
  ASSERT_NE(loom_liveness_interval_for_value(&analysis, args[1]), nullptr);
}

TEST_F(LivenessTest, CfgLiveInOutUsesSuccessorEdgesAndBranchOperands) {
  ModulePtr module = ParseModule(R"(
func.def @cfg_select(%cond: i1, %a: i32, %b: i32) -> (i32) {
  cfg.cond_br %cond, ^then, ^else
^then:
  cfg.br ^join(%a: i32)
^else:
  cfg.br ^join(%b: i32)
^join(%result: i32):
  func.return %result : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("cfg_select"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_TRUE(analysis.is_cfg);
  ASSERT_EQ(analysis.block_count, 4u);

  const loom_liveness_block_info_t& entry = analysis.blocks[0];
  EXPECT_FALSE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[0]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[1]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[2]));

  const loom_liveness_block_info_t& then_block = analysis.blocks[1];
  const loom_liveness_block_info_t& else_block = analysis.blocks[2];
  const loom_liveness_block_info_t& join_block = analysis.blocks[3];
  EXPECT_TRUE(ContainsValue(then_block.live_in_values, then_block.live_in_count,
                            args[1]));
  EXPECT_TRUE(ContainsValue(else_block.live_in_values, else_block.live_in_count,
                            args[2]));
  EXPECT_EQ(join_block.live_in_count, 0u);
}

TEST_F(LivenessTest, CfgLoopPropagatesFixedPointLiveness) {
  ModulePtr module = ParseModule(R"(
func.def @cfg_loop(%cond: i1, %x: i32) -> (i32) {
  cfg.br ^loop(%x: i32)
^loop(%iter: i32):
  cfg.cond_br %cond, ^body, ^exit
^body:
  %next = scalar.addi %iter, %x : i32
  cfg.br ^loop(%next: i32)
^exit:
  func.return %iter : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("cfg_loop"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_TRUE(analysis.is_cfg);
  ASSERT_EQ(analysis.block_count, 4u);

  const loom_liveness_block_info_t& entry = analysis.blocks[0];
  const loom_liveness_block_info_t& loop = analysis.blocks[1];
  const loom_liveness_block_info_t& body = analysis.blocks[2];
  const loom_liveness_block_info_t& exit = analysis.blocks[3];
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[0]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[1]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[0]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[1]));
  EXPECT_TRUE(ContainsValue(body.live_in_values, body.live_in_count, args[1]));
  EXPECT_EQ(exit.live_in_count, 1u);

  const loom_value_id_t iter = loom_block_arg_id(loop.block, 0);
  const loom_op_t* add = loom_block_const_op(body.block, 0);
  const loom_value_id_t next = loom_op_const_results(add)[0];
  const loom_liveness_segment_range_t iter_segments =
      loom_liveness_segment_range_for_value_ordinal(
          &analysis, FindValueOrdinal(analysis, iter));
  const loom_liveness_segment_range_t next_segments =
      loom_liveness_segment_range_for_value_ordinal(
          &analysis, FindValueOrdinal(analysis, next));
  const loom_liveness_segment_range_t invariant_segments =
      loom_liveness_segment_range_for_value_ordinal(
          &analysis, FindValueOrdinal(analysis, args[1]));
  EXPECT_EQ(iter_segments.count, 3u);
  EXPECT_EQ(next_segments.count, 1u);
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(&analysis, iter_segments,
                                                    next_segments));
  EXPECT_TRUE(loom_liveness_segment_ranges_overlap(
      &analysis, invariant_segments, next_segments));

  const loom_liveness_pressure_summary_t* pressure =
      FindScalarPressure(analysis, LOOM_SCALAR_TYPE_I32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 2u);
}

TEST_F(LivenessTest, OperandTypeReferencesKeepDynamicDimsLive) {
  ModulePtr module = ParseModule(R"(
func.def @type_ref(%N: index, %v: vector<[%N]xi32>) -> (vector<[%N]xi32>) {
  func.return %v : vector<[%N]xi32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("type_ref"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  const loom_liveness_interval_t* dim_interval =
      loom_liveness_interval_for_value(&analysis, args[0]);
  const loom_liveness_interval_t* vector_interval =
      loom_liveness_interval_for_value(&analysis, args[1]);
  ASSERT_NE(dim_interval, nullptr);
  ASSERT_NE(vector_interval, nullptr);
  EXPECT_EQ(dim_interval->end_point, vector_interval->end_point);

  ASSERT_EQ(analysis.operation_count, 1u);
  const loom_liveness_operation_point_t& return_point =
      analysis.operation_points[0];
  ASSERT_EQ(return_point.direct_use_count, 2u);
  ASSERT_EQ(return_point.use_count, 2u);
  EXPECT_EQ(
      loom_liveness_operation_use_ordinal(&analysis, return_point.use_start),
      FindValueOrdinal(analysis, args[1]));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis,
                                                return_point.use_start + 1u),
            FindValueOrdinal(analysis, args[0]));
}

TEST_F(LivenessTest, TiedResultOperandIsLiveThroughConsumingOp) {
  ModulePtr module = ParseModule(R"(
func.def @tied_update(%tile: tile<4xf32>, %tensor: tensor<4xf32>, %off: index) -> (tensor<4xf32>) {
  %updated = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)
  func.return %updated : tensor<4xf32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("tied_update"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  const loom_liveness_interval_t* tensor_interval =
      loom_liveness_interval_for_value(&analysis, args[1]);
  ASSERT_NE(tensor_interval, nullptr);
  EXPECT_LT(tensor_interval->start_point, tensor_interval->end_point);
}

TEST_F(LivenessTest, RegisterPressureGroupsByRegisterClassUnits) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target
low.func.def target<test.low.core>(@test_target) @register_pressure(%a: reg<test.i32>, %b: reg<test.i32>, %c: reg<test.i32>) -> (reg<test.i32>) {
  %ab = low.copy %a : reg<test.i32> -> reg<test.i32>
  %bc = low.copy %b : reg<test.i32> -> reg<test.i32>
  %cc = low.copy %c : reg<test.i32> -> reg<test.i32>
  low.return %ab : reg<test.i32>
}
)");
  loom_func_like_t func =
      FindFunction(module.get(), IREE_SV("register_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_liveness_pressure_summary_t* pressure = FindRegisterPressure(
      analysis, descriptor_set->stable_id, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 3u);
  EXPECT_EQ(pressure->peak_live_values, 3u);
  const loom_block_t* entry =
      loom_region_const_entry_block(loom_func_like_body(func));
  EXPECT_EQ(pressure->peak_op, loom_block_const_op(entry, 1));
  EXPECT_EQ(pressure->peak_point, 1u);
}

TEST_F(LivenessTest, RegionTreePressureIncludesNestedOperations) {
  ModulePtr module = ParseModule(R"(
func.def @region_tree_pressure(%input: tile<4xf32>, %bias: f32) -> (tile<4xf32>) {
  %mapped = test.map(%element = %input : tile<4xf32>) {
    %biased = scalar.addf %element, %bias : f32
    %biased_again = scalar.addf %biased, %bias : f32
    %neg0 = test.neg %biased_again : f32
    %neg1 = test.neg %neg0 : f32
    test.yield %neg1 : f32
  } -> (tile<4xf32>)
  func.return %mapped : tile<4xf32>
}
)");
  loom_func_like_t func =
      FindFunction(module.get(), IREE_SV("region_tree_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBodyRegionTree(module.get(), func);

  EXPECT_TRUE(loom_liveness_analysis_includes_region_tree(&analysis));
  const loom_liveness_pressure_summary_t* pressure =
      FindScalarPressure(analysis, LOOM_SCALAR_TYPE_F32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_GE(pressure->peak_live_units, 1u);
  EXPECT_GE(pressure->peak_live_values, 1u);

  ASSERT_EQ(analysis.operation_count, 7u);
  const loom_liveness_operation_point_t& map_point =
      analysis.operation_points[0];
  EXPECT_EQ(map_point.parent_operation_index, UINT32_MAX);
  ASSERT_GT(map_point.use_count, map_point.direct_use_count);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);
  const loom_value_ordinal_t bias_ordinal = FindValueOrdinal(analysis, args[1]);
  uint32_t bias_capture_count = 0;
  for (uint32_t i = map_point.direct_use_count; i < map_point.use_count; ++i) {
    bias_capture_count +=
        loom_liveness_operation_use_ordinal(
            &analysis, map_point.use_start + i) == bias_ordinal;
  }
  EXPECT_EQ(bias_capture_count, 1u);
  for (uint32_t i = 1; i < 6; ++i) {
    EXPECT_EQ(analysis.operation_points[i].parent_operation_index, 0u);
  }
  EXPECT_EQ(analysis.operation_points[6].parent_operation_index, UINT32_MAX);
}

TEST_F(LivenessTest, PressureBudgetReportsHighUnrolledRegisterUse) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target
low.func.def target<test.low.core>(@test_target) @high_pressure(%a0: reg<test.i32>, %a1: reg<test.i32>, %a2: reg<test.i32>, %a3: reg<test.i32>, %a4: reg<test.i32>, %a5: reg<test.i32>) -> (reg<test.i32>) {
  %r0 = low.copy %a0 : reg<test.i32> -> reg<test.i32>
  %r1 = low.copy %a1 : reg<test.i32> -> reg<test.i32>
  %r2 = low.copy %a2 : reg<test.i32> -> reg<test.i32>
  %r3 = low.copy %a3 : reg<test.i32> -> reg<test.i32>
  %r4 = low.copy %a4 : reg<test.i32> -> reg<test.i32>
  %r5 = low.copy %a5 : reg<test.i32> -> reg<test.i32>
  low.return %r0 : reg<test.i32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("high_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_liveness_pressure_summary_t* pressure = FindRegisterPressure(
      analysis, descriptor_set->stable_id, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 6u);

  loom_liveness_pressure_budget_t budget = {
      /*.value_class=*/pressure->value_class,
      /*.max_live_units=*/4,
      /*.max_live_values=*/4,
  };
  const loom_liveness_pressure_budget_violation_t* violations = nullptr;
  iree_host_size_t violation_count = 0;
  IREE_ASSERT_OK(loom_liveness_collect_pressure_budget_violations(
      &analysis, &budget, 1, &analysis_arena_, &violations, &violation_count));
  ASSERT_EQ(violation_count, 1u);
  EXPECT_EQ(violations[0].budget_index, 0u);
  EXPECT_EQ(violations[0].summary, pressure);
  EXPECT_EQ(violations[0].violation_bits,
            LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_UNITS |
                LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_VALUES);
}

TEST_F(LivenessTest, FormatsMachineReadableJsonSummary) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target
low.func.def target<test.low.core>(@test_target) @json_pressure(%a: reg<test.i32>, %b: reg<test.i32>) -> (reg<test.i32>) {
  %r = low.copy %a : reg<test.i32> -> reg<test.i32>
  %dead = low.copy %b : reg<test.i32> -> reg<test.i32>
  low.return %r : reg<test.i32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("json_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_liveness_format_json(&analysis, NULL, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  EXPECT_NE(json.find("\"format\":\"loom.liveness.v0\""), std::string::npos);
  EXPECT_NE(json.find("\"blocks\""), std::string::npos);
  EXPECT_NE(json.find("\"intervals\""), std::string::npos);
  EXPECT_NE(json.find("\"pressure_summaries\""), std::string::npos);
  EXPECT_NE(json.find("\"register_descriptor_set\":"), std::string::npos);
  EXPECT_NE(json.find("\"register_class_id\":"), std::string::npos);
  EXPECT_NE(json.find("\"peak_live_units\":2"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
