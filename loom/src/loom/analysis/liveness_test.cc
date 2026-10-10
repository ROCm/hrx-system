// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

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
        .descriptor_sets = {},
        .descriptor_set_count = {},
        .descriptor_set_providers = descriptor_set_providers,
        .descriptor_set_provider_count =
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
      if (values[i] == value) {
        return true;
      }
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
  EXPECT_EQ(a_interval->definition_point, 0u);
  EXPECT_EQ(a_interval->start_point, 0u);
  EXPECT_EQ(a_interval->end_point, 2u);
  EXPECT_EQ(sum_interval->definition_point, 1u);
  EXPECT_EQ(sum_interval->start_point, 1u);
  EXPECT_EQ(sum_interval->end_point, 3u);
  EXPECT_EQ(dead_interval->definition_point, 2u);
  EXPECT_EQ(dead_interval->start_point, 2u);
  EXPECT_EQ(dead_interval->end_point, 2u);

  ASSERT_EQ(analysis.operation_count, 3u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->op, add);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->parent_operation_index,
            UINT32_MAX);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->start_point, 0u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->end_point, 1u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->direct_use_count, 2u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->use_count, 2u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 1)->op, dead_add);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 2)->op,
            loom_block_const_op(entry, 2));
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
  for (uint32_t use_index = 0; use_index < analysis.operation_use_count;
       ++use_index) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_ordinal(&analysis, use_index);
    const loom_liveness_segment_range_t range =
        loom_liveness_segment_range_for_value_ordinal(&analysis, value_ordinal);
    ASSERT_EQ(range.count, 1u);
    EXPECT_EQ(loom_liveness_operation_use_segment_index(&analysis, use_index),
              range.start);
  }
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
      .block = entry,
      .ops = ordered_ops,
      .op_count = IREE_ARRAYSIZE(ordered_ops),
  };
  const loom_liveness_order_t order = {
      .blocks = &block_order,
      .block_count = 1,
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
    EXPECT_EQ(loom_liveness_operation_at(&analysis, i)->op, ordered_ops[i]);
    EXPECT_EQ(loom_liveness_operation_at(&analysis, i)->start_point, i);
    EXPECT_EQ(loom_liveness_operation_at(&analysis, i)->end_point, i + 1u);
    for (uint16_t result_index = 0; result_index < ordered_ops[i]->result_count;
         ++result_index) {
      const auto* interval = loom_liveness_interval_for_value(
          &analysis, loom_op_const_results(ordered_ops[i])[result_index]);
      ASSERT_NE(interval, nullptr);
      EXPECT_EQ(interval->definition_point, i + 1u);
    }
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
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->direct_use_count, 1u);
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 0)->use_count, 1u);
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

  // The event sweep indexes intervals by region-local ordinals even when all
  // module value IDs exceed the interval table size.
  loom_liveness_analysis_t tree_analysis =
      AnalyzeBodyRegionTree(module.get(), func);
  ASSERT_GT(args[0], tree_analysis.value_count);
  const loom_liveness_pressure_summary_t* pressure =
      FindScalarPressure(tree_analysis, LOOM_SCALAR_TYPE_I32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 2u);
  EXPECT_EQ(pressure->peak_live_values, 2u);
  EXPECT_EQ(pressure->peak_point, 0u);
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

  const auto expect_branch_use_segment = [&](const auto& block,
                                             loom_value_id_t value) {
    const loom_liveness_operation_point_t* branch =
        loom_liveness_operation_at(&analysis, block.operation_start);
    ASSERT_EQ(branch->direct_use_count, 1u);
    const uint32_t use_index = branch->use_start;
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_ordinal(&analysis, use_index);
    ASSERT_EQ(analysis.value_ids[value_ordinal], value);
    const loom_liveness_segment_range_t range =
        loom_liveness_segment_range_for_value_ordinal(&analysis, value_ordinal);
    ASSERT_EQ(range.count, 2u);
    const uint32_t segment_index =
        loom_liveness_operation_use_segment_index(&analysis, use_index);
    EXPECT_EQ(segment_index, range.start + 1u);
    EXPECT_LE(analysis.segments[segment_index].start_point,
              branch->start_point);
    EXPECT_GT(analysis.segments[segment_index].end_point, branch->start_point);
  };
  expect_branch_use_segment(then_block, args[1]);
  expect_branch_use_segment(else_block, args[2]);
}

class LivenessStorageTest : public LivenessTest {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<LivenessStorageTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected liveness allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage(iree_host_size_t block_size = 128 * 1024) {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate},
                                     &result_pool_);
    iree_arena_initialize(&result_pool_, &result_arena_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&result_arena_);
    iree_arena_block_pool_deinitialize(&result_pool_);
  }

  void SetUp() override {
    LivenessTest::SetUp();
    InitializeStorage();
  }

  void TearDown() override {
    DeinitializeStorage();
    module_.reset();
    LivenessTest::TearDown();
  }

  void BuildChain(uint16_t block_count, uint16_t argument_count = 8,
                  uint16_t result_count = 8) {
    ASSERT_LE(result_count, argument_count);
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module));
    module_.reset(module);
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("chain"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    std::vector<loom_type_t> types(argument_count,
                                   loom_type_scalar(LOOM_SCALAR_TYPE_I32));
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), {0, symbol_id}, types.data(),
        argument_count, types.data(), result_count, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function));
    body_ = loom_func_like_body(loom_func_like_cast(module, function));
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_block_t* entry = loom_region_entry_block(body_);
    loom_builder_set_block(&builder, entry);
    builder.ip.parent_op = function;
    for (uint16_t i = 1; i < block_count; ++i) {
      loom_block_t* successor = nullptr;
      IREE_ASSERT_OK(loom_region_append_block(module, body_, &successor));
      loom_op_t* branch = nullptr;
      IREE_ASSERT_OK(loom_cfg_br_build(&builder, successor, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &branch));
      loom_builder_set_block(&builder, successor);
    }
    loom_op_t* return_op = nullptr;
    // The live suffix follows any unused arguments in the local value domain.
    const auto* results = result_count == 0
                              ? nullptr
                              : entry->arg_ids + argument_count - result_count;
    IREE_ASSERT_OK(loom_func_return_build(&builder, results, result_count,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void ExpectSegments(const loom_liveness_analysis_t& analysis) {
    const auto* entry = loom_region_const_entry_block(body_);
    ASSERT_EQ(analysis.block_count, body_->block_count);
    ASSERT_EQ(analysis.operation_count, body_->block_count);
    ASSERT_EQ(analysis.value_count, entry->arg_count);
    ASSERT_EQ(analysis.interval_count, entry->arg_count);
    ASSERT_EQ(analysis.segment_count,
              static_cast<size_t>(body_->block_count) * entry->arg_count);
    for (uint16_t i = 0; i < entry->arg_count; ++i) {
      const auto ordinal = FindValueOrdinal(analysis, entry->arg_ids[i]);
      ASSERT_NE(ordinal, LOOM_VALUE_ORDINAL_INVALID);
      const auto range = analysis.value_segment_ranges[ordinal];
      EXPECT_EQ(range.start, ordinal * body_->block_count);
      ASSERT_EQ(range.count, body_->block_count);
      const auto* interval =
          loom_liveness_interval_for_value_ordinal(&analysis, ordinal);
      ASSERT_NE(interval, nullptr);
      EXPECT_EQ(interval->start_point, 0u);
      EXPECT_EQ(interval->end_point, 2u * body_->block_count - 1u);
      for (uint32_t block_index = 0; block_index < range.count; ++block_index) {
        const auto segment = analysis.segments[range.start + block_index];
        EXPECT_EQ(segment.start_point, 2u * block_index);
        EXPECT_EQ(
            segment.end_point,
            2u * block_index + (block_index + 1u == range.count ? 1u : 2u));
      }
    }
    if (entry->arg_count != 0) {
      const auto* pressure = FindScalarPressure(analysis, LOOM_SCALAR_TYPE_I32);
      ASSERT_NE(pressure, nullptr);
      EXPECT_EQ(pressure->peak_live_units, entry->arg_count);
      EXPECT_EQ(pressure->peak_live_values, entry->arg_count);
    }
  }

  // Module construction uses the fixture pool, separate from analyzed storage.
  ModulePtr module_;
  // CFG whose entry arguments stay live through the last block's return.
  loom_region_t* body_ = nullptr;
  // Pool shared by analysis results and internal scratch.
  iree_arena_block_pool_t result_pool_ = {};
  // Result lifetime ends between repeated analyses, retaining reusable blocks.
  iree_arena_allocator_t result_arena_ = {};
  // Attempted backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Allocation selected for injected failure, or SIZE_MAX.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

class LivenessStorageBoundaryTest
    : public LivenessStorageTest,
      public ::testing::WithParamInterface<uint16_t> {};

TEST_P(LivenessStorageBoundaryTest, PreservesSegmentsAndReusesPoolBlocks) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(GetParam()));
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  ASSERT_NO_FATAL_FAILURE(ExpectSegments(analysis));
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&result_pool_, &statistics);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  const auto allocation_count = allocation_count_;
  iree_arena_reset(&result_arena_);
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  ASSERT_NO_FATAL_FAILURE(ExpectSegments(analysis));
  EXPECT_EQ(allocation_count_, allocation_count);
}

INSTANTIATE_TEST_SUITE_P(SegmentChunks, LivenessStorageBoundaryTest,
                         ::testing::Values(1, 31, 32, 33, 127, 128, 129, 1023,
                                           1024, 1025));

TEST_F(LivenessStorageTest, EmptySegmentStream) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(33, 0, 0));
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  ASSERT_NO_FATAL_FAILURE(ExpectSegments(analysis));
  EXPECT_EQ(analysis.segments, nullptr);
}

TEST_F(LivenessStorageTest, UnusedValuesHaveEmptySegmentRanges) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(33, 8, 0));
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  ASSERT_EQ(analysis.value_count, 8u);
  // Unused arguments require neither a defining write nor a live interval.
  ASSERT_EQ(analysis.interval_count, 0u);
  EXPECT_EQ(analysis.segment_count, 0u);
  EXPECT_EQ(analysis.segments, nullptr);
  for (loom_value_ordinal_t ordinal = 0; ordinal < analysis.value_count;
       ++ordinal) {
    EXPECT_EQ(analysis.value_segment_ranges[ordinal].start, 0u);
    EXPECT_EQ(analysis.value_segment_ranges[ordinal].count, 0u);
  }
}

class LivenessStorageFailureTest
    : public LivenessStorageTest,
      public ::testing::WithParamInterface<
          std::tuple<uint16_t, uint16_t, iree_host_size_t>> {};

TEST_P(LivenessStorageFailureTest, BackingFailureLeavesNoPublishedAnalysis) {
  const auto [block_count, value_count, block_size] = GetParam();
  ASSERT_NO_FATAL_FAILURE(BuildChain(block_count, value_count, value_count));
  DeinitializeStorage();
  InitializeStorage(block_size);
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  ASSERT_NO_FATAL_FAILURE(ExpectSegments(analysis));
  const auto allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 1u);
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DeinitializeStorage();
    InitializeStorage(block_size);
    failure_index_ = i;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_liveness_analyze_region(
                              module_.get(), body_, &result_arena_, &analysis));
    EXPECT_EQ(allocation_count_, i + 1);
    EXPECT_EQ(analysis.module, nullptr);
    EXPECT_EQ(analysis.segments, nullptr);
    EXPECT_EQ(analysis.segment_count, 0u);
    EXPECT_EQ(analysis.intervals, nullptr);
    EXPECT_EQ(analysis.operations, nullptr);
    failure_index_ = SIZE_MAX;
    iree_arena_reset(&result_arena_);
    IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                                &result_arena_, &analysis));
    ASSERT_NO_FATAL_FAILURE(ExpectSegments(analysis));
  }
}

INSTANTIATE_TEST_SUITE_P(Construction, LivenessStorageFailureTest,
                         ::testing::Values(std::make_tuple(1025, 8, 128 * 1024),
                                           std::make_tuple(1, 800, 32 * 1024)));

class LivenessOperationStorageTest
    : public LivenessStorageTest,
      public ::testing::WithParamInterface<uint32_t> {
 protected:
  void BuildOperationUses(uint32_t operation_count) {
    ASSERT_NO_FATAL_FAILURE(BuildChain(1, 1, 1));
    auto* entry = loom_region_entry_block(body_);
    loom_builder_t builder;
    loom_builder_initialize(module_.get(), &module_->arena, entry, &builder);
    loom_builder_set_before(&builder, entry->last_op);
    for (uint32_t i = 1; i < operation_count; ++i) {
      loom_op_t* use = nullptr;
      IREE_ASSERT_OK(loom_test_use_build(&builder, entry->arg_ids, 1,
                                         LOOM_LOCATION_UNKNOWN, &use));
    }
  }
};

TEST_P(LivenessOperationStorageTest, StableAcceptedOrderAndPoolReuse) {
  ASSERT_NO_FATAL_FAILURE(BuildOperationUses(GetParam()));
  const auto* entry = loom_region_const_entry_block(body_);
  std::vector<const loom_op_t*> operations;
  const loom_op_t* op = nullptr;
  loom_block_for_each_op(entry, op) { operations.push_back(op); }
  // Independent sinks can appear in any accepted order. Keep the terminator
  // last while forcing row identities to differ from source order.
  std::reverse(operations.begin(), operations.end() - 1);
  const loom_liveness_block_order_t block_order = {entry, operations.data(),
                                                   operations.size()};
  const loom_liveness_order_t order = {&block_order, 1};
  iree_host_size_t allocation_count = 0;
  for (uint32_t pass = 0; pass < 2; ++pass) {
    loom_liveness_analysis_t analysis = {};
    IREE_ASSERT_OK(loom_liveness_analyze_region_with_order(
        module_.get(), body_, order, &result_arena_, &analysis));
    ASSERT_EQ(analysis.operation_count, operations.size());
    ASSERT_EQ(analysis.operation_use_count, operations.size());
    std::vector<const loom_liveness_operation_point_t*> rows;
    for (uint32_t i = 0; i < analysis.operation_count; ++i) {
      const auto* row = loom_liveness_operation_at(&analysis, i);
      rows.push_back(row);
      EXPECT_EQ(row->op, operations[i]);
      EXPECT_EQ(row->parent_operation_index, UINT32_MAX);
      EXPECT_EQ(row->start_point, i);
      EXPECT_EQ(row->end_point, i + 1u);
      EXPECT_EQ(row->direct_use_count, 1u);
      EXPECT_EQ(row->use_count, 1u);
      EXPECT_EQ(analysis.value_ids[loom_liveness_operation_use_ordinal(
                    &analysis, row->use_start)],
                entry->arg_ids[0]);
    }
    for (uint32_t start : {0u, GetParam() / 2, GetParam() - 1}) {
      for (uint32_t end : {start + 1, GetParam()}) {
        for (uint32_t index = start; index < end;) {
          const auto span = loom_liveness_operation_span(&analysis, index, end);
          ASSERT_GT(span.count, 0u);
          ASSERT_LE(span.count, end - index);
          for (uint32_t i = 0; i < span.count; ++i) {
            EXPECT_EQ(&span.rows[i], rows[index + i]);
            EXPECT_EQ(span.rows[i].op, operations[index + i]);
          }
          index += span.count;
        }
      }
    }
    // Borrow returned construction blocks while all published row pointers
    // remain live. No address depends on the scratch arena's former contents.
    iree_arena_allocator_t scratch;
    iree_arena_initialize(&result_pool_, &scratch);
    void* payload = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&scratch, 4096, &payload));
    memset(payload, 0xA5, 4096);
    for (uint32_t i = 0; i < rows.size(); ++i) {
      EXPECT_EQ(rows[i], loom_liveness_operation_at(&analysis, i));
      EXPECT_EQ(rows[i]->op, operations[i]);
    }
    iree_arena_deinitialize(&scratch);
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&result_pool_, &statistics);
    EXPECT_EQ(statistics.oversized_allocation_count, 0u);
    if (pass == 0) {
      allocation_count = allocation_count_;
    } else {
      EXPECT_EQ(allocation_count_, allocation_count);
    }
    iree_arena_reset(&result_arena_);
  }
}

INSTANTIATE_TEST_SUITE_P(OperationSegments, LivenessOperationStorageTest,
                         ::testing::Values(1, 127, 128, 129, 2048, 2049, 4097,
                                           65537));

TEST_F(LivenessOperationStorageTest,
       EveryBackingFailureKeepsResultUnpublished) {
  ASSERT_NO_FATAL_FAILURE(BuildOperationUses(2049));
  DeinitializeStorage();
  InitializeStorage(8 * 1024);
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                              &result_arena_, &analysis));
  const auto allocation_count = allocation_count_;
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    SCOPED_TRACE(i);
    DeinitializeStorage();
    InitializeStorage(8 * 1024);
    failure_index_ = i;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_liveness_analyze_region(
                              module_.get(), body_, &result_arena_, &analysis));
    EXPECT_EQ(allocation_count_, i + 1);
    EXPECT_EQ(analysis.operations, nullptr);
    EXPECT_EQ(analysis.operation_count, 0u);
    EXPECT_EQ(analysis.operation_use_count, 0u);
    failure_index_ = SIZE_MAX;
    iree_arena_reset(&result_arena_);
    IREE_ASSERT_OK(loom_liveness_analyze_region(module_.get(), body_,
                                                &result_arena_, &analysis));
    EXPECT_EQ(loom_liveness_operation_at(&analysis, 2048)->op,
              loom_region_const_entry_block(body_)->last_op);
  }
}

TEST_F(LivenessOperationStorageTest, ParentCompletionCrossesRowSegments) {
  ASSERT_NO_FATAL_FAILURE(BuildChain(1, 1, 1));
  auto* entry = loom_region_entry_block(body_);
  loom_builder_t builder;
  loom_builder_initialize(module_.get(), &module_->arena, entry, &builder);
  loom_builder_set_before(&builder, entry->last_op);
  loom_op_t* parent = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(&builder, entry->arg_ids, 1,
                                            LOOM_LOCATION_UNKNOWN, &parent));
  auto* nested_region = loom_test_block_args_body(parent);
  auto saved = loom_builder_enter_region(&builder, parent, nested_region);
  const auto nested_value = loom_block_arg_id(builder.ip.block, 0);
  const loom_value_id_t inputs[] = {nested_value, entry->arg_ids[0]};
  for (uint32_t i = 0; i < 2048; ++i) {
    loom_op_t* use = nullptr;
    IREE_ASSERT_OK(loom_test_use_build(&builder, inputs, IREE_ARRAYSIZE(inputs),
                                       LOOM_LOCATION_UNKNOWN, &use));
  }
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder, saved);
  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module_.get(), body_, &result_arena_, &domain));
  loom_liveness_analysis_t analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
      &domain, loom_liveness_order_empty(), &result_arena_, &analysis));
  loom_local_value_domain_release(&domain);
  ASSERT_EQ(analysis.operation_count, 2051u);
  const auto* parent_row = loom_liveness_operation_at(&analysis, 0);
  EXPECT_EQ(parent_row->op, parent);
  EXPECT_EQ(parent_row->start_point, 0u);
  EXPECT_EQ(parent_row->end_point, 2051u);
  EXPECT_EQ(parent_row->direct_use_count, 1u);
  EXPECT_EQ(parent_row->use_count, 2u);
  for (uint32_t i = 1; i < 2050; ++i) {
    const auto* row = loom_liveness_operation_at(&analysis, i);
    EXPECT_EQ(row->parent_operation_index, 0u);
    EXPECT_EQ(row->start_point, i);
    EXPECT_EQ(row->end_point, i + 1u);
  }
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 2049)->op, yield);
  const auto* return_row = loom_liveness_operation_at(&analysis, 2050);
  EXPECT_EQ(return_row->op, entry->last_op);
  EXPECT_EQ(return_row->parent_operation_index, UINT32_MAX);
  EXPECT_EQ(return_row->start_point, parent_row->end_point);
}

class LivenessIntervalStorageTest
    : public LivenessStorageTest,
      public ::testing::WithParamInterface<
          std::tuple<uint16_t, loom_local_value_domain_flags_t>> {};

TEST_P(LivenessIntervalStorageTest, CompactIntervalsRetainOnlyLiveMetadata) {
  const auto [result_count, flags] = GetParam();
  ASSERT_NO_FATAL_FAILURE(BuildChain(1, 800, result_count));
  auto* entry = loom_region_entry_block(body_);
  DeinitializeStorage();
  InitializeStorage(32 * 1024);
  iree_host_size_t first_allocation_count = 0;
  for (int epoch = 0; epoch < 3; ++epoch) {
    SCOPED_TRACE(epoch);
    loom_local_value_domain_t domain = {};
    const auto acquire =
        iree_any_bit_set(flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)
            ? loom_local_value_domain_acquire_for_region_tree
            : loom_local_value_domain_acquire_for_region;
    IREE_ASSERT_OK(acquire(module_.get(), body_, &result_arena_, &domain));
    loom_liveness_analysis_t analysis = {};
    IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
        &domain, loom_liveness_order_empty(), &result_arena_, &analysis));
    loom_local_value_domain_release(&domain);
    EXPECT_EQ(analysis.value_count, 800u);
    ASSERT_EQ(analysis.interval_count, result_count);
    EXPECT_EQ(analysis.segment_count, result_count);
    for (uint16_t i = 0; i < entry->arg_count; ++i) {
      const auto ordinal = FindValueOrdinal(analysis, entry->arg_ids[i]);
      const auto* interval =
          loom_liveness_interval_for_value_ordinal(&analysis, ordinal);
      if (i < 800 - result_count) {
        EXPECT_EQ(interval, nullptr);
        continue;
      }
      ASSERT_NE(interval, nullptr);
      EXPECT_EQ(interval->value_id, entry->arg_ids[i]);
      EXPECT_EQ(interval->start_point, 0u);
      EXPECT_EQ(interval->end_point, 1u);
      EXPECT_EQ(interval->definition_point, 0u);
      EXPECT_EQ(interval->unit_count, 1u);
      EXPECT_EQ(interval->value_class.type_kind, LOOM_TYPE_SCALAR);
      EXPECT_EQ(interval->value_class.element_type, LOOM_SCALAR_TYPE_I32);
      EXPECT_EQ(interval->value_class.register_class_id,
                LOOM_LOW_REGISTER_CLASS_ID_INVALID);
      EXPECT_EQ(interval->value_class.register_descriptor_set_stable_id, 0u);
    }
    const auto* pressure = FindScalarPressure(analysis, LOOM_SCALAR_TYPE_I32);
    if (result_count == 0) {
      EXPECT_EQ(pressure, nullptr);
    } else {
      ASSERT_NE(pressure, nullptr);
      EXPECT_EQ(pressure->peak_live_values, result_count);
      EXPECT_EQ(pressure->peak_live_units, result_count);
      EXPECT_EQ(pressure->peak_point, 0u);
    }
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&result_pool_, &statistics);
    EXPECT_EQ(statistics.oversized_allocation_count, 0u);
    if (epoch == 0) {
      first_allocation_count = allocation_count_;
    } else {
      EXPECT_EQ(allocation_count_, first_allocation_count);
    }
    iree_arena_reset(&result_arena_);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BoundCollection, LivenessIntervalStorageTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 400, 800),
        ::testing::Values(loom_local_value_domain_flags_t{0},
                          loom_local_value_domain_flags_t{
                              LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE})));

TEST_F(LivenessTest, CfgLoopPropagatesFixedPointLiveness) {
  ModulePtr module = ParseModule(R"(
func.def @cfg_loop(%cond: i1, %x: i32) -> (i32) {
  cfg.br ^loop(%x: i32)
^dispatch:
  cfg.br ^exit
^loop(%iter: i32):
  cfg.cond_br %cond, ^body, ^dispatch
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
  ASSERT_EQ(analysis.block_count, 5u);

  const loom_liveness_block_info_t& entry = analysis.blocks[0];
  const loom_liveness_block_info_t& dispatch = analysis.blocks[1];
  const loom_liveness_block_info_t& loop = analysis.blocks[2];
  const loom_liveness_block_info_t& body = analysis.blocks[3];
  const loom_liveness_block_info_t& exit = analysis.blocks[4];
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[0]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[1]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[0]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[1]));
  EXPECT_TRUE(ContainsValue(body.live_in_values, body.live_in_count, args[1]));
  EXPECT_EQ(exit.live_in_count, 1u);

  const loom_value_id_t iter = loom_block_arg_id(loop.block, 0);
  const auto* iter_interval = loom_liveness_interval_for_value(&analysis, iter);
  ASSERT_NE(iter_interval, nullptr);
  // Live-through blocks can precede the defining block in numeric layout.
  EXPECT_TRUE(
      ContainsValue(dispatch.live_in_values, dispatch.live_in_count, iter));
  EXPECT_EQ(iter_interval->start_point, dispatch.start_point);
  EXPECT_EQ(iter_interval->definition_point, loop.start_point);
  EXPECT_LT(iter_interval->start_point, iter_interval->definition_point);
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
  EXPECT_EQ(iter_segments.count, 4u);
  EXPECT_EQ(next_segments.count, 1u);
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(
      analysis.segments, iter_segments, next_segments));
  EXPECT_TRUE(loom_liveness_segment_ranges_overlap(
      analysis.segments, invariant_segments, next_segments));

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
      *loom_liveness_operation_at(&analysis, 0);
  ASSERT_EQ(return_point.direct_use_count, 2u);
  ASSERT_EQ(return_point.use_count, 2u);
  EXPECT_EQ(
      loom_liveness_operation_use_ordinal(&analysis, return_point.use_start),
      FindValueOrdinal(analysis, args[1]));
  EXPECT_FALSE(loom_liveness_operation_use_has_type_reference(
      &analysis, return_point.use_start));
  EXPECT_EQ(loom_liveness_operation_use_ordinal(&analysis,
                                                return_point.use_start + 1u),
            FindValueOrdinal(analysis, args[0]));
  EXPECT_TRUE(loom_liveness_operation_use_has_type_reference(
      &analysis, return_point.use_start + 1u));
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
  // The captured bias overlaps one nested temporary at a time. Successive
  // temporary intervals meet at half-open boundaries without extra pressure.
  EXPECT_EQ(pressure->peak_live_units, 2u);
  EXPECT_EQ(pressure->peak_live_values, 2u);
  EXPECT_EQ(pressure->peak_point, 1u);

  ASSERT_EQ(analysis.operation_count, 7u);
  const loom_liveness_operation_point_t& map_point =
      *loom_liveness_operation_at(&analysis, 0);
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
    EXPECT_EQ(loom_liveness_operation_at(&analysis, i)->parent_operation_index,
              0u);
  }
  EXPECT_EQ(loom_liveness_operation_at(&analysis, 6)->parent_operation_index,
            UINT32_MAX);

  const loom_region_t* nested_region = loom_op_regions(map_point.op)[0];
  const loom_block_t* nested_block =
      loom_region_const_entry_block(nested_region);
  const loom_value_id_t element = loom_block_arg_id(nested_block, 0);
  const auto* element_interval =
      loom_liveness_interval_for_value(&analysis, element);
  const auto* mapped_interval = loom_liveness_interval_for_value(
      &analysis, loom_op_const_results(map_point.op)[0]);
  const auto* bias_interval =
      loom_liveness_interval_for_value(&analysis, args[1]);
  ASSERT_NE(element_interval, nullptr);
  ASSERT_NE(mapped_interval, nullptr);
  ASSERT_NE(bias_interval, nullptr);
  EXPECT_EQ(element_interval->definition_point,
            loom_liveness_operation_at(&analysis, 1)->start_point);
  EXPECT_EQ(mapped_interval->definition_point, map_point.end_point);
  EXPECT_EQ(bias_interval->definition_point, 0u);

  // An analysis rooted at the nested region owns its argument definition but
  // sees the enclosing function's bias only as a capture.
  loom_liveness_analysis_t nested_analysis = {};
  IREE_ASSERT_OK(loom_liveness_analyze_region(
      module.get(), nested_region, &analysis_arena_, &nested_analysis));
  const auto* nested_element_interval =
      loom_liveness_interval_for_value(&nested_analysis, element);
  const auto* captured_bias_interval =
      loom_liveness_interval_for_value(&nested_analysis, args[1]);
  ASSERT_NE(nested_element_interval, nullptr);
  ASSERT_NE(captured_bias_interval, nullptr);
  EXPECT_EQ(nested_element_interval->definition_point, 0u);
  EXPECT_EQ(captured_bias_interval->definition_point, UINT32_MAX);
  EXPECT_EQ(captured_bias_interval->start_point, 0u);
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
      .value_class = pressure->value_class,
      .max_live_units = 4,
      .max_live_values = 4,
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

TEST(LivenessSegmentsTest, SparseOverlapAndHalfOpenBoundaries) {
  const loom_liveness_segment_t segments[] = {
      {1, 3}, {8, 10},            // First value, with a gap.
      {0, 1}, {3, 8},  {10, 12},  // Second value touches but never overlaps.
      {3, 8}, {9, 12},            // Third value overlaps only at the tail.
  };
  const loom_liveness_segment_range_t first = {0, 2};
  const loom_liveness_segment_range_t second = {2, 3};
  const loom_liveness_segment_range_t third = {5, 2};
  const loom_liveness_segment_range_t empty = {7, 0};
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(segments, first, second));
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(segments, second, first));
  EXPECT_TRUE(loom_liveness_segment_ranges_overlap(segments, first, third));
  EXPECT_TRUE(loom_liveness_segment_ranges_overlap(segments, third, first));
  EXPECT_TRUE(loom_liveness_segment_ranges_overlap(segments, first, first));
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(segments, first, empty));
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(segments, empty, first));
  EXPECT_FALSE(loom_liveness_segment_ranges_overlap(segments, empty, empty));
}

TEST(LivenessSegmentsTest, ExhaustiveSmallSparseSets) {
  // Enumerate every live/dead pattern over six program points. Coalesce
  // adjacent points into the sorted nonempty segments produced by liveness.
  const auto append = [](uint32_t bits, loom_liveness_segment_t* segments,
                         uint32_t& count) {
    const uint32_t start = count;
    for (uint32_t point = 0; point < 6;) {
      if ((bits & (1u << point)) == 0) {
        ++point;
        continue;
      }
      const uint32_t begin = point;
      do {
        ++point;
      } while (point < 6 && (bits & (1u << point)) != 0);
      segments[count++] = {begin, point};
    }
    return loom_liveness_segment_range_t{start, count - start};
  };
  for (uint32_t lhs_bits = 0; lhs_bits < 64; ++lhs_bits) {
    for (uint32_t rhs_bits = 0; rhs_bits < 64; ++rhs_bits) {
      loom_liveness_segment_t segments[6] = {};
      uint32_t count = 0;
      const auto lhs = append(lhs_bits, segments, count);
      const auto rhs = append(rhs_bits, segments, count);
      for (uint32_t point = 0; point <= 6; ++point) {
        EXPECT_EQ(loom_liveness_segment_range_contains(segments, lhs, point),
                  (lhs_bits & (1u << point)) != 0)
            << "bits=" << lhs_bits << ", point=" << point;
        EXPECT_EQ(loom_liveness_segment_range_contains(segments, rhs, point),
                  (rhs_bits & (1u << point)) != 0)
            << "bits=" << rhs_bits << ", point=" << point;
      }
      EXPECT_EQ(loom_liveness_segment_ranges_overlap(segments, lhs, rhs),
                (lhs_bits & rhs_bits) != 0)
          << "lhs=" << lhs_bits << ", rhs=" << rhs_bits;
    }
  }
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
